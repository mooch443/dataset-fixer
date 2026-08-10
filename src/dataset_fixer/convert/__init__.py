"""Content-addressed training dataset preparation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping

import numpy as np
import yaml
from PIL import Image, ImageOps
from tqdm.auto import tqdm

from ..errors import DatasetValidationError, ValidationIssue
from ..geometry import Geometry, normalize_size
from ..sources import cache_root, fingerprint_files
from ..utils import slugify, to_jsonable

if TYPE_CHECKING:
    from ..bundle import Config


class Kind(str, Enum):
    """Supported preparation targets."""

    YOLO_SEM = "yolo-sem"
    YOLO_SEG = "yolo-seg"
    NNUNET = "nnunet"


@dataclass(frozen=True)
class Prepared:
    """Typed identity and paths for one reusable prepared dataset.

    Parameters:
        kind: Backend preparation target.
        location: Content-addressed preparation root.
        geometry: Source-tile, scale, and training-input geometry.
        content_sha256: Preparation identity derived from inputs and settings.
        source_name: Portable basename of the original training folder or ZIP.
        paths: Named backend-specific output paths.
        hashes: Digests for generated configuration and manifest files.
        split_statistics: Per-split image and annotation counts.
        manifests: Generated preparation manifests.
        backend: Backend-specific configuration and label mapping.
        config: Complete model-bundle configuration when ``prepare()`` was
            given a model name and training choices.
        reused: Whether an existing content-addressed preparation was reused.
    """

    kind: Kind
    location: Path
    geometry: Geometry
    content_sha256: str
    paths: Mapping[str, Path]
    hashes: Mapping[str, str]
    split_statistics: Mapping[str, Mapping[str, int]]
    manifests: tuple[Path, ...]
    source_name: str | None = None
    backend: Mapping[str, Any] = field(default_factory=dict)
    config: Config | None = field(default=None, compare=False)
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
    source = dict(value.get("source") or {})
    return Prepared(
        kind=Kind(value["preparation_kind"]),
        location=root,
        geometry=geometry,
        content_sha256=str(value["content_sha256"]),
        source_name=(str(source["basename"]) if source.get("basename") else None),
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


def _nnunet_plans(prepared: Prepared, explicit: str | None) -> str | None:
    """Resolve the generated nnU-Net plans identifier without importing nnU-Net."""

    if explicit:
        return explicit
    recorded = prepared.backend.get("plans")
    if recorded:
        return str(recorded)
    preprocessed_root = prepared.paths.get("preprocessed_root")
    dataset_name = prepared.backend.get("dataset_name")
    if preprocessed_root is None or not dataset_name:
        return None
    candidates: set[str] = set()
    for path in (Path(preprocessed_root) / str(dataset_name)).glob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(value, Mapping) and value.get("plans_name"):
            candidates.add(str(value["plans_name"]))
    if len(candidates) == 1:
        return candidates.pop()
    return None


def _attach_config(
    prepared: Prepared,
    *,
    name: str | None,
    base_model: str | Path | None,
    trainer: str | None,
    plans: str | None,
    configuration: str | None,
    fold: int | None,
    epochs: int | None,
    checkpoint_name: str | None,
    workers: int,
    device: str | None,
) -> Prepared:
    """Attach run-specific bundle metadata without changing dataset identity."""

    requested = (base_model, trainer, plans, configuration, fold, epochs, checkpoint_name, device)
    if name is None:
        if any(value is not None for value in requested):
            raise ValueError("prepare() requires name when training configuration is supplied")
        return prepared
    if not str(name).strip():
        raise ValueError("name must not be empty")
    if epochs is not None and (
        isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0
    ):
        raise ValueError("epochs must be a positive integer")
    if fold is not None and (
        isinstance(fold, bool) or not isinstance(fold, int) or fold < 0
    ):
        raise ValueError("fold must be a non-negative integer")

    from ..bundle import Config

    input_size = prepared.geometry.input_size
    serialized_input_size: int | list[int] | None = None
    if input_size is not None:
        serialized_input_size = (
            input_size[0] if input_size[0] == input_size[1] else list(input_size)
        )
    model: dict[str, Any] = {"name": str(name)}
    training: dict[str, Any] = {"workers": workers}
    if epochs is not None:
        training["epochs"] = int(epochs)
    if device is not None:
        training["device"] = str(device)

    if prepared.kind in {Kind.YOLO_SEM, Kind.YOLO_SEG}:
        unsupported = {
            "plans": plans,
            "configuration": configuration,
            "fold": fold,
        }
        supplied = [key for key, value in unsupported.items() if value is not None]
        if supplied:
            raise ValueError(
                f"YOLO preparation does not accept nnU-Net options: {', '.join(supplied)}"
            )
        resolved_trainer = trainer or "ultralytics.YOLO"
        model.update({"base_model": str(base_model)} if base_model is not None else {})
        if checkpoint_name is not None:
            model["checkpoint"] = checkpoint_name
        training.update({"trainer": resolved_trainer})
        if base_model is not None:
            training["base_model"] = str(base_model)
        if serialized_input_size is not None:
            training["imgsz"] = serialized_input_size
        framework = "ultralytics"
        task = "semantic" if prepared.kind == Kind.YOLO_SEM else "segment"
    else:
        if base_model is not None:
            raise ValueError("nnU-Net preparation does not accept base_model")
        resolved_plans = _nnunet_plans(prepared, plans)
        if resolved_plans is None:
            raise ValueError(
                "Could not infer nnU-Net plans; run preprocessing or supply plans explicitly"
            )
        resolved_trainer = trainer or "nnUNetTrainer"
        resolved_configuration = configuration or "2d"
        resolved_fold = 0 if fold is None else int(fold)
        resolved_checkpoint = checkpoint_name or "checkpoint_final.pth"
        planner = prepared.backend.get("planner")
        model.update(
            {
                "planner": planner,
                "plans": resolved_plans,
                "trainer": resolved_trainer,
                "configuration": resolved_configuration,
                "fold": resolved_fold,
                "checkpoint": resolved_checkpoint,
            }
        )
        training.update(
            {
                "trainer": resolved_trainer,
                "planner": planner,
                "plans": resolved_plans,
                "configuration": resolved_configuration,
                "fold": resolved_fold,
            }
        )
        framework = "nnunetv2"
        task = "semantic"

    config = Config(
        name=str(name),
        framework=framework,
        task=task,
        geometry=prepared.geometry,
        dataset=prepared,
        model=model,
        training=training,
    )
    return replace(prepared, config=config)


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
    errors: Literal["raise", "skip"],
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
    source_size = (image.height, image.width)
    smaller_than_native = native is not None and source_size != native
    if native is not None and (image.height > native[0] or image.width > native[1]):
        if errors == "raise":
            raise DatasetValidationError(
                ValidationIssue(
                    "Source image dimensions exceed native_tile_size",
                    source=str(image_path),
                    value=source_size,
                    expected=native,
                    suggestion="Pass errors='skip' to omit oversized source images.",
                )
            )
        return {
            "status": "skipped",
            "split": sample.split,
            "source_image": str(image_path),
            "source_mask": str(mask_path),
            "actual_size": list(source_size),
            "maximum_size": list(native),
            "reason": "image exceeds native_tile_size",
        }
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
        "source_size": list(source_size),
        "native_size_validation": "smaller" if smaller_than_native else "matched",
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


def _partition_semantic_records(
    records: list[dict[str, Any]],
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Path | None]:
    retained = [record for record in records if record.get("status") != "skipped"]
    skipped = [record for record in records if record.get("status") == "skipped"]
    if not retained:
        raise DatasetValidationError(
            ValidationIssue(
                "Preparation contains no usable images after skipping oversized sources",
                value=skipped[:20],
            )
        )
    if not skipped:
        return retained, skipped, None
    report = root / "preparation-skips.json"
    _write_json(
        report,
        {
            "errors": "skip",
            "skipped_images": len(skipped),
            "records": skipped,
        },
    )
    return retained, skipped, report


def _prepare_yolo_sem(
    dataset: Any,
    root: Path,
    *,
    geometry: Geometry,
    threshold: int,
    workers: int,
    progress: bool,
    errors: Literal["raise", "skip"],
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
                        errors=errors,
                    ),
                    items,
                ),
                total=len(items),
                desc="Preparing YOLO semantic data",
                disable=not progress,
            )
        )
    records, skipped, skip_report = _partition_semantic_records(records, root)
    statistics: dict[str, dict[str, int]] = {}
    for split in dataset.splits:
        selected = [record for record in records if record["split"] == split]
        skipped_split = [record for record in skipped if record["split"] == split]
        statistics[split] = {
            "images": len(selected),
            "foreground_pixels": sum(record["foreground_pixels"] for record in selected),
            "pixels": sum(record["pixels"] for record in selected),
            "resized_smaller_images": sum(
                record["native_size_validation"] == "smaller" for record in selected
            ),
            "skipped_oversized_images": len(skipped_split),
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
        "source_size_policy": {
            "errors": errors,
            "smaller_or_equal": "retain-and-resize",
            "oversized": "skip" if errors == "skip" else "raise",
        },
    }
    paths = {"data_yaml": data_yaml, "dataset_root": root, "cases": cases}
    if skip_report is not None:
        paths["skips"] = skip_report
    return paths, statistics, backend


def _prepare_yolo_seg(
    dataset: Any,
    root: Path,
    *,
    geometry: Geometry,
    errors: Literal["raise", "skip"],
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
    skipped: list[dict[str, Any]] = []
    retained_by_split: dict[str, list[Any]] = {}
    statistics: dict[str, dict[str, int]] = {}
    for split in dataset.splits:
        samples = [sample for sample in dataset._samples if sample.split == split]
        retained: list[Any] = []
        for sample in samples:
            actual = (int(sample.height), int(sample.width))
            native = geometry.native_tile_size
            if native is not None and (
                actual[0] > native[0] or actual[1] > native[1]
            ):
                skipped.append(
                    {
                        "status": "skipped",
                        "split": split,
                        "source_image": str(sample.image_path.resolve()),
                        "actual_size": list(actual),
                        "maximum_size": list(native),
                        "reason": "image exceeds native_tile_size",
                    }
                )
            else:
                retained.append(sample)
        if skipped and errors == "raise":
            raise DatasetValidationError(
                ValidationIssue(
                    "Source image dimensions exceed native_tile_size",
                    value=skipped[:20],
                    suggestion="Pass errors='skip' to omit oversized source images.",
                )
            )
        retained_by_split[split] = retained
        instances = 0
        for sample in retained:
            for annotation in sample.annotations:
                if not annotation.polygon or len(annotation.polygon) < 3 or annotation.rle:
                    failures.append(str(sample.relative_path))
                else:
                    instances += 1
        statistics[split] = {
            "images": len(retained),
            "instances": instances,
            "skipped_oversized_images": len(samples) - len(retained),
        }
    if not any(retained_by_split.values()):
        raise DatasetValidationError(
            ValidationIssue(
                "Preparation contains no usable images after skipping oversized sources",
                value=skipped[:20],
            )
        )
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
    if skipped:
        for split, samples in retained_by_split.items():
            (root / f"{split}.txt").write_text(
                "".join(f"{sample.image_path.resolve()}\n" for sample in samples),
                encoding="utf-8",
            )
        data_yaml = root / "data.yaml"
        data_yaml.write_text(
            yaml.safe_dump(
                {
                    **{split: f"{split}.txt" for split in retained_by_split},
                    "names": {int(key): value for key, value in dataset.classes.items()},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        skip_report = root / "preparation-skips.json"
        _write_json(
            skip_report,
            {"errors": "skip", "skipped_images": len(skipped), "records": skipped},
        )
        paths.update({"data_yaml": data_yaml, "skips": skip_report})
    elif dataset.data_yaml is not None:
        paths["data_yaml"] = dataset.data_yaml
    return paths, statistics, {
        "task": "segment",
        "conversion": "none",
        "polygon_audit": "passed",
        "source_size_policy": {
            "errors": errors,
            "smaller_or_equal": "retain",
            "oversized": "skip" if errors == "skip" else "raise",
        },
    }


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
    errors: Literal["raise", "skip"],
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
                        errors=errors,
                    ),
                    items,
                ),
                total=len(items),
                desc="Preparing nnU-Net raw data",
                disable=not progress,
            )
        )
    records, skipped, skip_report = _partition_semantic_records(records, root)
    statistics: dict[str, dict[str, int]] = {}
    split_cases: dict[str, list[str]] = {"train": [], "val": []}
    for record in records:
        case_id = record["case_id"]
        split_cases[record["split"]].append(case_id)
        shutil.move(intermediate / record["prepared_image"], images_tr / f"{case_id}_0000.png")
        shutil.move(intermediate / record["prepared_mask"], labels_tr / f"{case_id}.png")
    shutil.rmtree(intermediate, ignore_errors=True)
    for split, cases in split_cases.items():
        selected = [record for record in records if record["split"] == split]
        skipped_split = [record for record in skipped if record["split"] == split]
        statistics[split] = {
            "images": len(cases),
            "cases": len(cases),
            "resized_smaller_images": sum(
                record["native_size_validation"] == "smaller" for record in selected
            ),
            "skipped_oversized_images": len(skipped_split),
        }
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
    if skip_report is not None:
        paths["skips"] = skip_report
    return paths, statistics, {
        "task": "semantic",
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "environment": environment,
        "image_resize": "bicubic",
        "mask_resize": "nearest",
        "label_mapping": {"background": 0, "foreground": 1},
        "source_binary_jpeg_threshold": threshold,
        "source_size_policy": {
            "errors": errors,
            "smaller_or_equal": "retain-and-resize",
            "oversized": "skip" if errors == "skip" else "raise",
        },
        "preprocessed": preprocess,
        "planner": planner,
    }


def prepare(
    dataset: Any,
    kind: Kind,
    *,
    name: str | None = None,
    native_tile_size: int | tuple[int, int] | None = None,
    upscale_factor: int = 1,
    destination: str | Path | None = None,
    workers: int = 4,
    mask_threshold: int = 128,
    preprocess: bool = True,
    dataset_id: int | None = None,
    planner: str | None = None,
    base_model: str | Path | None = None,
    trainer: str | None = None,
    plans: str | None = None,
    configuration: str | None = None,
    fold: int | None = None,
    epochs: int | None = None,
    checkpoint_name: str | None = None,
    device: str | None = None,
    errors: Literal["raise", "skip"] = "raise",
    progress: bool = True,
) -> Prepared:
    """Prepare one dataset for a selected backend using a reusable identity.

    JPEG binary masks use the explicitly recorded ``mask_threshold``.  Lossless
    masks preserve class values until the target backend's required 0/1 label
    encoding is applied.  Semantic masks are never converted to instances.

    Parameters:
        dataset: Open dataset or any source accepted by :meth:`Dataset.open`.
        kind: Backend preparation target.
        name: Model and run name. When supplied, the result includes a complete
            :class:`dataset_fixer.bundle.Config` as ``prepared.config``.
        native_tile_size: Source tile edge or two-item size before upscaling.
        upscale_factor: Positive integer source-to-training scale.
        destination: Explicit preparation root. The automatic content cache is
            used when omitted.
        workers: Parallel image conversion worker count.
        mask_threshold: Inclusive threshold used when converting lossy binary
            JPEG masks to exact labels.
        preprocess: Run nnU-Net planning and preprocessing for ``Kind.NNUNET``.
        dataset_id: Optional nnU-Net dataset number; a deterministic number is
            derived when omitted.
        planner: Optional nnU-Net experiment planner class name.
        base_model: YOLO base checkpoint or architecture identifier.
        trainer: Training implementation. YOLO defaults to
            ``ultralytics.YOLO`` and nnU-Net defaults to ``nnUNetTrainer``.
        plans: nnU-Net plans identifier. It is inferred from generated plans
            when omitted.
        configuration: nnU-Net configuration such as ``"2d"``.
        fold: nnU-Net validation fold, defaulting to zero for configured runs.
        epochs: Training epoch count recorded in bundle and W&B metadata.
        checkpoint_name: Expected checkpoint name. nnU-Net defaults to
            ``checkpoint_final.pth``.
        device: Training device recorded in bundle and W&B metadata.
        errors: Oversized-source policy. Images whose height and width are at
            most ``native_tile_size`` are always retained and resized with
            their masks to the training input size. ``"raise"`` rejects an
            image exceeding either native dimension; ``"skip"`` omits it and
            writes an audit report.
        progress: Show hashing, conversion, and preprocessing progress.

    Returns:
        Typed paths, identity, statistics, backend configuration, and—when a
        model name is supplied—the complete run-specific bundle configuration.
    """

    from ..dataset import Dataset

    errors = errors.lower()
    if errors not in {"raise", "skip"}:
        raise ValueError("errors must be 'raise' or 'skip'")
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
    if target_kind != Kind.NNUNET and planner is not None:
        raise ValueError("planner is supported only for nnU-Net preparation")
    run_specific = (
        base_model,
        trainer,
        plans,
        configuration,
        fold,
        epochs,
        checkpoint_name,
        device,
    )
    if name is None and any(value is not None for value in run_specific):
        raise ValueError("prepare() requires name when training configuration is supplied")
    if epochs is not None and (
        isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0
    ):
        raise ValueError("epochs must be a positive integer")
    if fold is not None and (
        isinstance(fold, bool) or not isinstance(fold, int) or fold < 0
    ):
        raise ValueError("fold must be a non-negative integer")
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
        "errors": errors,
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
        existing = replace(existing, source_name=dataset.source_name)
        return _attach_config(
            existing,
            name=name,
            base_model=base_model,
            trainer=trainer,
            plans=plans,
            configuration=configuration,
            fold=fold,
            epochs=epochs,
            checkpoint_name=checkpoint_name,
            workers=workers,
            device=device,
        )
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
                errors=errors,
            )
        elif target_kind == Kind.YOLO_SEG:
            paths, statistics, backend = _prepare_yolo_seg(
                dataset,
                temporary,
                geometry=geometry,
                errors=errors,
            )
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
                errors=errors,
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
        for key in ("data_yaml", "cases", "dataset_json", "splits", "audit", "skips"):
            if key in serialized_paths:
                raw = Path(serialized_paths[key])
                if not raw.is_absolute():
                    manifest_names.append(str(raw))
        manifest = {
            "schema": 1,
            "format": "prepared-dataset",
            "preparation_kind": target_kind.value,
            "content_sha256": content_digest,
            "source": {
                "name": dataset.name,
                "basename": dataset.source_name,
                "location": str(dataset.location),
            },
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
    prepared = _prepared_from_manifest(root / "preparation.json", reused=False)
    return _attach_config(
        prepared,
        name=name,
        base_model=base_model,
        trainer=trainer,
        plans=plans,
        configuration=configuration,
        fold=fold,
        epochs=epochs,
        checkpoint_name=checkpoint_name,
        workers=workers,
        device=device,
    )


__all__ = ["Kind", "Prepared", "prepare"]
