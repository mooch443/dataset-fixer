"""Content-addressed training dataset preparation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml
from PIL import Image, ImageOps
from tqdm.auto import tqdm

from ..errors import DatasetValidationError, ValidationIssue
from ..geometry import Geometry, normalize_size
from ..sources import cache_root, fingerprint_files
from ..utils import slugify, to_jsonable


class Kind(str, Enum):
    """Supported preparation targets."""

    YOLO_SEM = "yolo-sem"
    YOLO_SEG = "yolo-seg"
    NNUNET = "nnunet"


@dataclass(frozen=True)
class Prepared:
    """Typed identity and paths for one reusable prepared dataset."""

    kind: Kind
    location: Path
    geometry: Geometry
    content_sha256: str
    paths: Mapping[str, Path]
    hashes: Mapping[str, str]
    split_statistics: Mapping[str, Mapping[str, int]]
    manifests: tuple[Path, ...]
    backend: Mapping[str, Any] = field(default_factory=dict)
    reused: bool = False

    @property
    def data_yaml(self) -> Path | None:
        value = self.paths.get("data_yaml")
        return Path(value) if value is not None else None

    @property
    def manifest(self) -> Path:
        return self.manifests[0]


def _prepared_from_manifest(path: Path, *, reused: bool) -> Prepared:
    value = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent

    def resolve(raw: str) -> Path:
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else (root / candidate).resolve()

    geometry = Geometry.create(**dict(value["geometry"]))
    backend = dict(value.get("backend") or {})
    if isinstance(backend.get("environment"), Mapping):
        backend["environment"] = {
            str(key): str(resolve(str(raw)))
            for key, raw in dict(backend["environment"]).items()
        }
    return Prepared(
        kind=Kind(value["preparation_kind"]),
        location=root,
        geometry=geometry,
        content_sha256=str(value["content_sha256"]),
        paths={key: resolve(raw) for key, raw in dict(value.get("paths") or {}).items()},
        hashes={str(key): str(raw) for key, raw in dict(value.get("hashes") or {}).items()},
        split_statistics={
            str(split): {str(key): int(raw) for key, raw in dict(statistics).items()}
            for split, statistics in dict(value.get("split_statistics") or {}).items()
        },
        manifests=tuple(resolve(raw) for raw in value.get("manifests") or [path.name]),
        backend=backend,
        reused=reused,
    )


def _dataset_files(dataset: Any) -> tuple[list[Path], dict[Path, Path]]:
    samples = list(dataset._samples)
    masks = dict(getattr(dataset, "_mask_paths", {}) or {})
    files = [sample.image_path for sample in samples]
    if dataset.format == "semantic_masks":
        missing: list[str] = []
        for sample in samples:
            mask = masks.get(sample.image_path.resolve())
            if mask is None:
                missing.append(str(sample.image_path))
            else:
                files.append(mask)
        if missing:
            raise DatasetValidationError(
                ValidationIssue(
                    "Semantic dataset has images without resolved masks",
                    value=missing[:20],
                )
            )
    return files, masks


def _identity(dataset: Any, kind: Kind, settings: Mapping[str, Any], *, progress: bool) -> str:
    files, _ = _dataset_files(dataset)
    source_digest = fingerprint_files(
        files,
        progress=progress,
        description="Hashing dataset preparation inputs",
    )
    digest = hashlib.sha256()
    digest.update(source_digest.encode("ascii"))
    digest.update(
        json.dumps(
            {"kind": kind.value, **to_jsonable(dict(settings))},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _mask_values(mask_path: Path, *, threshold: int) -> tuple[np.ndarray, dict[str, Any]]:
    with Image.open(mask_path) as opened:
        values = np.asarray(opened.convert("L"), dtype=np.uint8)
    unique = sorted(int(value) for value in np.unique(values))
    is_jpeg = mask_path.suffix.lower() in {".jpg", ".jpeg"}
    if is_jpeg:
        binary = (values >= threshold).astype(np.uint8)
        mapping = {
            "source_format": "jpeg",
            "source_values": unique,
            "threshold": threshold,
            "threshold_rule": f"value >= {threshold}",
            "lossless_intermediate_values": [0, 255],
        }
    else:
        if not set(unique).issubset({0, 1, 255}):
            raise DatasetValidationError(
                ValidationIssue(
                    "Binary semantic mask contains unsupported label values",
                    source=str(mask_path),
                    value=unique[:32],
                    expected=[0, 1, 255],
                    suggestion=(
                        "Use JPEG only when threshold-based recovery is intentional, "
                        "or convert labels without changing their values."
                    ),
                )
            )
        binary = (values > 0).astype(np.uint8)
        mapping = {
            "source_format": mask_path.suffix.lower().lstrip("."),
            "source_values": unique,
            "threshold": None,
            "lossless_intermediate_values": [0, 255],
        }
    return binary, mapping


def _convert_semantic_case(
    item: tuple[Any, Path, Path],
    *,
    root: Path,
    geometry: Geometry,
    threshold: int,
    foreground_value: int,
) -> dict[str, Any]:
    sample, image_path, mask_path = item
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    mask, mapping = _mask_values(mask_path, threshold=threshold)
    if image.size != (mask.shape[1], mask.shape[0]):
        raise DatasetValidationError(
            ValidationIssue(
                "Image/mask dimensions differ during preparation",
                source=str(image_path),
                value={"image": image.size, "mask": (mask.shape[1], mask.shape[0])},
            )
        )
    native = geometry.native_tile_size
    if native is not None and image.size != (native[1], native[0]):
        raise DatasetValidationError(
            ValidationIssue(
                "Source image dimensions do not match native_tile_size",
                source=str(image_path),
                value=(image.height, image.width),
                expected=native,
            )
        )
    target = geometry.input_size or (image.height, image.width)
    if image.size != (target[1], target[0]):
        image = image.resize((target[1], target[0]), Image.Resampling.BICUBIC)
        mask_image = Image.fromarray(mask).resize(
            (target[1], target[0]), Image.Resampling.NEAREST
        )
        mask = np.asarray(mask_image, dtype=np.uint8)
    case_id = f"{sample.split}_{slugify(item[0].relative_path.stem)}_{hashlib.sha1(str(item[0].relative_path).encode()).hexdigest()[:8]}"
    image_output = root / "images" / sample.split / f"{case_id}.png"
    mask_output = root / "masks" / sample.split / f"{case_id}.png"
    image_output.parent.mkdir(parents=True, exist_ok=True)
    mask_output.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_output, format="PNG", compress_level=1)
    output_mask = (mask > 0).astype(np.uint8) * foreground_value
    Image.fromarray(output_mask).save(mask_output, format="PNG", compress_level=1)
    return {
        "case_id": case_id,
        "split": sample.split,
        "source_image": str(image_path),
        "source_mask": str(mask_path),
        "prepared_image": str(image_output.relative_to(root)),
        "prepared_mask": str(mask_output.relative_to(root)),
        "foreground_pixels": int(np.count_nonzero(output_mask)),
        "pixels": int(output_mask.size),
        "mask_source": mapping,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), indent=2, sort_keys=True), encoding="utf-8")


def _replace_root(value: Any, old: Path, new: str = ".") -> Any:
    """Replace atomic staging paths before serializing reusable metadata."""

    if isinstance(value, str):
        prefix = str(old)
        if value == prefix:
            return new
        if value.startswith(prefix + os.sep):
            return str(Path(new) / Path(value).relative_to(old))
        return value
    if isinstance(value, Mapping):
        return {key: _replace_root(item, old, new) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_replace_root(item, old, new) for item in value]
    return value


def _prepare_yolo_sem(
    dataset: Any,
    root: Path,
    *,
    geometry: Geometry,
    threshold: int,
    workers: int,
    progress: bool,
) -> tuple[dict[str, Path], dict[str, dict[str, int]], dict[str, Any]]:
    if dataset.format != "semantic_masks":
        raise DatasetValidationError("YOLO-SEM preparation requires semantic image/mask pairs")
    masks = dict(dataset._mask_paths)
    items = [
        (sample, sample.image_path.resolve(), masks[sample.image_path.resolve()])
        for sample in dataset._samples
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(
            tqdm(
                pool.map(
                    lambda item: _convert_semantic_case(
                        item,
                        root=root,
                        geometry=geometry,
                        threshold=threshold,
                        foreground_value=1,
                    ),
                    items,
                ),
                total=len(items),
                desc="Preparing YOLO semantic data",
                disable=not progress,
            )
        )
    statistics: dict[str, dict[str, int]] = {}
    for split in dataset.splits:
        selected = [record for record in records if record["split"] == split]
        statistics[split] = {
            "images": len(selected),
            "foreground_pixels": sum(record["foreground_pixels"] for record in selected),
            "pixels": sum(record["pixels"] for record in selected),
        }
    data = {
        "path": str(root),
        **{split: f"images/{split}" for split in dataset.splits},
        "masks_dir": "masks",
        "names": {int(key): value for key, value in dataset.classes.items()},
    }
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    cases = root / "cases.json"
    _write_json(cases, records)
    backend = {
        "task": "semantic",
        "image_resize": "bicubic",
        "mask_resize": "nearest",
        "label_mapping": {"background": 0, "foreground": 1, "ignore": 255},
        "source_binary_jpeg_threshold": threshold,
    }
    return {"data_yaml": data_yaml, "dataset_root": root, "cases": cases}, statistics, backend


def _prepare_yolo_seg(
    dataset: Any,
    root: Path,
) -> tuple[dict[str, Path], dict[str, dict[str, int]], dict[str, Any]]:
    if dataset.format == "semantic_masks":
        raise DatasetValidationError(
            ValidationIssue(
                "YOLO-SEG preparation accepts polygon instance annotations only",
                expected="existing polygons",
                suggestion="Semantic masks are never converted to polygons.",
            )
        )
    if dataset.task.value != "segment":
        raise DatasetValidationError("YOLO-SEG preparation requires task='segment'")
    failures: list[str] = []
    statistics: dict[str, dict[str, int]] = {}
    for split in dataset.splits:
        samples = [sample for sample in dataset._samples if sample.split == split]
        instances = 0
        for sample in samples:
            for annotation in sample.annotations:
                if not annotation.polygon or len(annotation.polygon) < 3 or annotation.rle:
                    failures.append(str(sample.relative_path))
                else:
                    instances += 1
        statistics[split] = {"images": len(samples), "instances": instances}
    if failures:
        raise DatasetValidationError(
            ValidationIssue(
                "YOLO-SEG source contains non-polygon instance annotations",
                value=failures[:20],
                expected="auditable polygon instances",
            )
        )
    audit = root / "polygon-audit.json"
    _write_json(audit, {"status": "passed", "splits": statistics})
    paths = {"dataset_root": dataset.location, "audit": audit}
    if dataset.data_yaml is not None:
        paths["data_yaml"] = dataset.data_yaml
    return paths, statistics, {"task": "segment", "conversion": "none", "polygon_audit": "passed"}


def _prepare_nnunet(
    dataset: Any,
    root: Path,
    *,
    geometry: Geometry,
    threshold: int,
    workers: int,
    progress: bool,
    preprocess: bool,
    dataset_id: int,
    planner: str | None,
) -> tuple[dict[str, Path], dict[str, dict[str, int]], dict[str, Any]]:
    if dataset.format != "semantic_masks":
        raise DatasetValidationError("nnU-Net preparation requires semantic image/mask pairs")
    raw_root = root / "nnUNet_raw"
    preprocessed_root = root / "nnUNet_preprocessed"
    results_root = root / "nnUNet_results"
    dataset_name = f"Dataset{dataset_id:03d}_{slugify(dataset.name).replace('-', '_')}"
    raw_dataset = raw_root / dataset_name
    images_tr = raw_dataset / "imagesTr"
    labels_tr = raw_dataset / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    masks = dict(dataset._mask_paths)
    items = [
        (sample, sample.image_path.resolve(), masks[sample.image_path.resolve()])
        for sample in dataset._samples
        if sample.split in {"train", "val"}
    ]
    intermediate = root / ".semantic"
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(
            tqdm(
                pool.map(
                    lambda item: _convert_semantic_case(
                        item,
                        root=intermediate,
                        geometry=geometry,
                        threshold=threshold,
                        foreground_value=1,
                    ),
                    items,
                ),
                total=len(items),
                desc="Preparing nnU-Net raw data",
                disable=not progress,
            )
        )
    statistics: dict[str, dict[str, int]] = {}
    split_cases: dict[str, list[str]] = {"train": [], "val": []}
    for record in records:
        case_id = record["case_id"]
        split_cases[record["split"]].append(case_id)
        shutil.move(intermediate / record["prepared_image"], images_tr / f"{case_id}_0000.png")
        shutil.move(intermediate / record["prepared_mask"], labels_tr / f"{case_id}.png")
    shutil.rmtree(intermediate, ignore_errors=True)
    for split, cases in split_cases.items():
        statistics[split] = {"images": len(cases), "cases": len(cases)}
    dataset_json = raw_dataset / "dataset.json"
    _write_json(
        dataset_json,
        {
            "channel_names": {"0": "RGB"},
            "labels": {"background": 0, "foreground": 1},
            "numTraining": len(records),
            "file_ending": ".png",
        },
    )
    splits = [{"train": split_cases["train"], "val": split_cases["val"]}]
    raw_splits = raw_dataset / "splits_final.json"
    _write_json(raw_splits, splits)
    environment = {
        "nnUNet_raw": str(raw_root),
        "nnUNet_preprocessed": str(preprocessed_root),
        "nnUNet_results": str(results_root),
    }
    preprocessed_dataset = preprocessed_root / dataset_name
    if preprocess:
        from ..nnunet_engine import require_nnunet

        require_nnunet()
        command = [
            "nnUNetv2_plan_and_preprocess",
            "-d",
            str(dataset_id),
            "--verify_dataset_integrity",
            "-np",
            str(workers),
        ]
        if planner:
            command.extend(["-pl", planner])
        if progress:
            print("Running nnU-Net planning/preprocessing:", " ".join(command))
        process = subprocess.run(
            command,
            env={**os.environ, **environment},
            text=True,
            check=False,
        )
        if process.returncode:
            raise RuntimeError(
                f"nnU-Net planning/preprocessing failed with exit code {process.returncode}"
            )
        preprocessed_dataset.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw_splits, preprocessed_dataset / "splits_final.json")
    paths = {
        "raw_root": raw_root,
        "raw_dataset": raw_dataset,
        "dataset_json": dataset_json,
        "splits": raw_splits,
        "preprocessed_root": preprocessed_root,
        "results_root": results_root,
    }
    return paths, statistics, {
        "task": "semantic",
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "environment": environment,
        "image_resize": "bicubic",
        "mask_resize": "nearest",
        "label_mapping": {"background": 0, "foreground": 1},
        "source_binary_jpeg_threshold": threshold,
        "preprocessed": preprocess,
        "planner": planner,
    }


def prepare(
    dataset: Any,
    kind: Kind,
    *,
    native_tile_size: int | tuple[int, int] | None = None,
    upscale_factor: int = 1,
    destination: str | Path | None = None,
    workers: int = 4,
    mask_threshold: int = 128,
    preprocess: bool = True,
    dataset_id: int | None = None,
    planner: str | None = None,
    progress: bool = True,
) -> Prepared:
    """Prepare one dataset for a selected backend using a reusable identity.

    JPEG binary masks use the explicitly recorded ``mask_threshold``.  Lossless
    masks preserve class values until the target backend's required 0/1 label
    encoding is applied.  Semantic masks are never converted to instances.
    """

    from ..dataset import Dataset

    if not isinstance(dataset, Dataset):
        dataset = Dataset.open(dataset, progress=progress)
    if dataset._plan:
        raise DatasetValidationError(
            "Dataset preparation requires a fixed on-disk dataset; export it first"
        )
    try:
        target_kind = kind if isinstance(kind, Kind) else Kind(kind)
    except ValueError as exc:
        raise ValueError(f"Unsupported preparation kind: {kind!r}") from exc
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if isinstance(mask_threshold, bool) or not 0 <= int(mask_threshold) <= 255:
        raise ValueError("mask_threshold must be an integer in [0, 255]")
    native = normalize_size(native_tile_size, field="native_tile_size")
    if native is None:
        sizes = {(sample.height, sample.width) for sample in dataset._samples}
        if len(sizes) == 1:
            native = next(iter(sizes))
    geometry = Geometry.create(
        native_tile_size=native,
        upscale_factor=upscale_factor,
        source=dataset.name,
    )
    settings = {
        "native_tile_size": geometry.native_tile_size,
        "upscale_factor": geometry.upscale_factor,
        "input_size": geometry.input_size,
        "workers": workers,
        "mask_threshold": mask_threshold,
        "preprocess": preprocess if target_kind == Kind.NNUNET else None,
        "dataset_id": dataset_id if target_kind == Kind.NNUNET else None,
        "planner": planner if target_kind == Kind.NNUNET else None,
    }
    content_digest = _identity(dataset, target_kind, settings, progress=progress)
    resolved_dataset_id = dataset_id or (500 + int(content_digest[:8], 16) % 500)
    root = (
        Path(destination).expanduser().resolve()
        if destination is not None
        else cache_root() / "prepared" / target_kind.value / content_digest
    )
    manifest_path = root / "preparation.json"
    if manifest_path.is_file():
        existing = _prepared_from_manifest(manifest_path, reused=True)
        if existing.content_sha256 != content_digest or existing.kind != target_kind:
            raise DatasetValidationError(
                f"Preparation destination contains a different identity: {root}"
            )
        if progress:
            print(f"Cache hit: prepared {target_kind.value} dataset at {root}")
        return existing
    if root.exists():
        raise FileExistsError(f"Preparation destination is non-empty or incomplete: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{content_digest[:12]}-", dir=root.parent))
    try:
        if target_kind == Kind.YOLO_SEM:
            paths, statistics, backend = _prepare_yolo_sem(
                dataset,
                temporary,
                geometry=geometry,
                threshold=int(mask_threshold),
                workers=workers,
                progress=progress,
            )
        elif target_kind == Kind.YOLO_SEG:
            paths, statistics, backend = _prepare_yolo_seg(dataset, temporary)
        else:
            paths, statistics, backend = _prepare_nnunet(
                dataset,
                temporary,
                geometry=geometry,
                threshold=int(mask_threshold),
                workers=workers,
                progress=progress,
                preprocess=preprocess,
                dataset_id=resolved_dataset_id,
                planner=planner,
            )
        # Paths created below the temporary root become relative so the atomic
        # rename does not leave stale temporary paths in the result manifest.
        serialized_paths = {}
        for key, value in paths.items():
            try:
                serialized_paths[key] = str(value.relative_to(temporary))
            except ValueError:
                serialized_paths[key] = str(value)
        manifest_names = ["preparation.json"]
        for key in ("data_yaml", "cases", "dataset_json", "splits", "audit"):
            if key in serialized_paths:
                raw = Path(serialized_paths[key])
                if not raw.is_absolute():
                    manifest_names.append(str(raw))
        manifest = {
            "schema": 1,
            "format": "prepared-dataset",
            "preparation_kind": target_kind.value,
            "content_sha256": content_digest,
            "source": {"name": dataset.name, "location": str(dataset.location)},
            "geometry": geometry.as_dict(),
            "paths": serialized_paths,
            "hashes": {"source_content": content_digest},
            "split_statistics": statistics,
            "backend": _replace_root(backend, temporary),
            "manifests": list(dict.fromkeys(manifest_names)),
        }
        _write_json(temporary / "preparation.json", manifest)
        temporary.replace(root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return _prepared_from_manifest(root / "preparation.json", reused=False)


__all__ = ["Kind", "Prepared", "prepare"]
