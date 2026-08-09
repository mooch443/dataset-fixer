"""Internal source inference for :meth:`dataset_fixer.Model.load_many`."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from tqdm.auto import tqdm

from .errors import DatasetValidationError, ValidationIssue
from .geometry import Geometry, first_value, from_metadata
from .sources import cache_root, extract_archive, local_source, sha256_progress


SUPPORTED_BUNDLE_FORMATS = {
    "model-bundle",
    "dataset-fixer-nnunet-model-folder-v1",
    "ultralytics-yolo26-sem-reproducibility-bundle-v1",
    "ultralytics-yolo26-instance-seg-reproducibility-bundle-v1",
}
_DOWNLOAD_CHUNK = 8 * 1024 * 1024


@dataclass(frozen=True)
class ResolvedModelSource:
    path: Path
    name: str
    options: dict[str, Any] = field(default_factory=dict)
    geometry: Geometry = field(default_factory=Geometry)
    manifest: dict[str, Any] = field(default_factory=dict)
    source: str | None = None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(
            ValidationIssue("Could not read model metadata", value=str(path))
        ) from exc
    if not isinstance(value, dict):
        raise DatasetValidationError(
            ValidationIssue("Model metadata must be a JSON object", value=str(path))
        )
    return value


def _one(paths: list[Path], label: str) -> Path:
    values = sorted({path.resolve() for path in paths})
    if len(values) != 1:
        raise DatasetValidationError(
            ValidationIssue(
                f"Model bundle must contain exactly one {label}",
                value=[str(path) for path in values],
            )
        )
    return values[0]


def _bundle_manifest(root: Path) -> Path:
    return _one(
        [
            path
            for name in ("bundle_manifest.json", "model-bundle.json")
            for path in root.rglob(name)
        ],
        "bundle manifest",
    )


def _task(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("_", "-")
    aliases = {
        "semantic": "semantic",
        "semantic-segment": "semantic",
        "semantic-segmentation": "semantic",
        "yolo-sem": "semantic",
        "yolo-seg": "segment",
        "instance-segmentation": "segment",
        "seg": "segment",
        "nnunet": "semantic",
    }
    return aliases.get(normalized, normalized)


def _model_type(value: Any, *, kind: str, task: str | None = None) -> str:
    """Normalize bundle/checkpoint architecture metadata for report labels."""

    raw = str(value or "").strip().lower().replace("_", "-")
    if kind == "nnunet":
        aliases = {
            "resenc-m": "nnunet-m",
            "resenc-l": "nnunet-l",
            "resenc-xl": "nnunet-xl",
            "resenc-capped": "nnunet-capped",
            "plain": "nnunet-plain",
        }
        if raw in aliases:
            return aliases[raw]
        compact = re.sub(r"[^a-z0-9]+", "", raw)
        if "resenccapped" in compact:
            return "nnunet-capped"
        for size in ("xl", "l", "m"):
            if re.search(rf"resenc(?:unet)?{size}(?:plans|$)", compact):
                return f"nnunet-{size}"
        if "resenc" in compact:
            return "nnunet-resenc"
        if "plainconv" in compact or compact == "plain":
            return "nnunet-plain"
        if raw.startswith("nnunet-"):
            return raw
        return f"nnunet-{raw}" if raw else "nnunet"

    stem = Path(raw.removeprefix("ultralytics:")).stem
    matches = re.findall(
        r"(?:yolox|yolo(?:\d+)?[nslmx]?)(?:-(?:sem|seg))?",
        stem,
    )
    normalized = matches[-1] if matches else stem
    if normalized in {"best", "last", "model", "weights", "checkpoint"}:
        normalized = ""
    if normalized and not normalized.endswith(("-sem", "-seg")):
        if task in {"semantic", "semantic_segment"}:
            normalized += "-sem"
        elif task == "segment":
            normalized += "-seg"
    if normalized:
        return normalized
    return "ultralytics-sem" if task in {"semantic", "semantic_segment"} else "ultralytics-seg"


def _source_dataset_zip(*values: Mapping[str, Any]) -> str | None:
    """Return a portable training-dataset archive name from model metadata."""

    candidates: list[Any] = []
    for value in values:
        dataset = dict(value.get("dataset") or {})
        signature = dict(dataset.get("dataset_signature") or {})
        source_dataset = dict(dataset.get("source_dataset") or {})
        run = dict(value.get("run") or {})
        stable_state = dict(run.get("stable_run_state") or {})
        top_level_stable_state = dict(value.get("stable_run_state") or {})
        candidates.extend(
            (
                dataset.get("source_dataset_zip"),
                signature.get("source"),
                dataset.get("source_archive"),
                source_dataset.get("path"),
                stable_state.get("source_dataset_zip"),
                top_level_stable_state.get("source_dataset_zip"),
                value.get("source_dataset_zip"),
                value.get("source_archive"),
                dataset.get("name"),
            )
        )
    selected = first_value(*candidates)
    if selected is None:
        return None
    return Path(str(selected)).name


def _manifest_geometry(manifest: Mapping[str, Any], source: str) -> Geometry:
    geometry = dict(manifest.get("geometry") or {})
    dataset = dict(manifest.get("dataset") or {})
    model = dict(manifest.get("model") or {})
    comparison = dict(manifest.get("compare_models") or {})
    training = dict(manifest.get("training") or {})
    train_args = dict(manifest.get("train_args") or training.get("train_args") or {})
    explicit = {
        "native_tile_size": first_value(
            geometry.get("native_tile_size"),
            manifest.get("native_tile_size"),
            dataset.get("native_tile_size"),
        ),
        "upscale_factor": first_value(
            manifest.get("upscale_factor"),
            geometry.get("upscale_factor"),
            dataset.get("upscale_factor"),
            comparison.get("upscale_factor"),
        ),
        "input_size": first_value(
            manifest.get("input_size"),
            geometry.get("input_size"),
            manifest.get("adapter_output_size"),
            dataset.get("input_size"),
            dataset.get("adapter_output_size"),
            model.get("input_size"),
            model.get("imgsz"),
            train_args.get("imgsz"),
        ),
    }
    return from_metadata(explicit, source=source)


def _validate_checkpoint(path: Path, manifest: Mapping[str, Any], *, progress: bool) -> str:
    model = dict(manifest.get("model") or {})
    outcome = dict(manifest.get("outcome") or manifest.get("training_outcome") or {})
    expected = first_value(
        model.get("checkpoint_sha256"),
        outcome.get("checkpoint_sha256"),
        manifest.get("checkpoint_sha256"),
    )
    actual = sha256_progress(path, progress=progress)
    if expected is not None and str(expected).lower() != actual:
        raise DatasetValidationError(
            ValidationIssue(
                "Model checkpoint SHA-256 does not match its bundle manifest",
                value=actual,
                expected=str(expected),
                source=str(path),
            )
        )
    return actual


def _resolve_bundle(path: Path, *, name: str | None, progress: bool) -> ResolvedModelSource:
    extracted = extract_archive(path, category="models", progress=progress)
    manifest_path = _bundle_manifest(extracted.root)
    manifest = _read_json(manifest_path)
    bundle_format = str(manifest.get("format") or manifest.get("kind") or "")
    if bundle_format not in SUPPORTED_BUNDLE_FORMATS:
        raise DatasetValidationError(
            ValidationIssue(
                "Unsupported model bundle format",
                value=bundle_format,
                expected=sorted(SUPPORTED_BUNDLE_FORMATS),
                source=str(manifest_path),
            )
        )
    manifest_root = manifest_path.parent
    geometry = _manifest_geometry(manifest, str(path))
    model_metadata = dict(manifest.get("model") or {})
    run = dict(manifest.get("run") or {})
    configured_name = first_value(
        name,
        model_metadata.get("name"),
        run.get("name"),
        run.get("wandb_run_name"),
        path.stem,
    )
    options: dict[str, Any] = {}
    source_dataset_zip = _source_dataset_zip(manifest)
    if source_dataset_zip is not None:
        options["source_dataset_zip"] = source_dataset_zip

    is_nnunet = bundle_format == "dataset-fixer-nnunet-model-folder-v1" or str(
        first_value(model_metadata.get("framework"), manifest.get("framework"), "")
    ).lower() in {"nnunet", "nnunetv2", "nnunet-v2"}
    if is_nnunet:
        candidates = [
            value.parent
            for value in extracted.root.rglob("plans.json")
            if (value.parent / "dataset.json").is_file()
        ]
        model_root = _one(candidates, "nnU-Net model folder")
        compare = dict(manifest.get("compare_models") or {})
        folds_value = first_value(
            model_metadata.get("folds"),
            compare.get("folds"),
            manifest.get("folds"),
            [model_metadata.get("fold")] if model_metadata.get("fold") is not None else None,
        )
        folds = tuple(str(value) for value in (folds_value or (0,)))
        checkpoint = str(
            first_value(
                model_metadata.get("checkpoint"),
                compare.get("checkpoint"),
                manifest.get("checkpoint"),
                "checkpoint_final.pth",
            )
        )
        checkpoint_files = [model_root / f"fold_{fold}" / checkpoint for fold in folds]
        missing = [value for value in checkpoint_files if not value.is_file()]
        if missing:
            raise DatasetValidationError(
                ValidationIssue(
                    "Incomplete nnU-Net model bundle",
                    value=[str(value) for value in missing],
                    expected=f"{checkpoint} in every selected fold",
                )
            )
        for checkpoint_path in checkpoint_files:
            _validate_checkpoint(checkpoint_path, manifest, progress=progress)
        options.update(
            kind="nnunet",
            task="semantic",
            model_type=_model_type(
                first_value(
                    model_metadata.get("model_preset"),
                    model_metadata.get("plans"),
                    model_metadata.get("planner"),
                    manifest.get("model_preset"),
                    model_root.name,
                ),
                kind="nnunet",
                task="semantic",
            ),
            folds=folds,
            checkpoint=checkpoint,
        )
        source_path = model_root
    else:
        checkpoint_name = str(
            first_value(model_metadata.get("checkpoint"), manifest.get("checkpoint"), "best.pt")
        )
        candidates = [
            value
            for value in manifest_root.rglob(Path(checkpoint_name).name)
            if value.is_file()
        ]
        if not candidates and Path(checkpoint_name).name != "best.pt":
            candidates = [value for value in manifest_root.rglob("best.pt") if value.is_file()]
        source_path = _one(candidates, "Ultralytics checkpoint")
        _validate_checkpoint(source_path, manifest, progress=progress)
        resolved_task = _task(model_metadata.get("task"))
        options.update(
            kind="ultralytics",
            task=resolved_task,
            model_type=_model_type(
                first_value(
                    model_metadata.get("model_weights"),
                    model_metadata.get("initialization_source"),
                    model_metadata.get("base_model"),
                    dict(manifest.get("training") or {}).get("base_model"),
                    source_path.stem,
                ),
                kind="ultralytics",
                task=resolved_task,
            ),
        )
        if geometry.input_size is not None:
            if geometry.input_size[0] == geometry.input_size[1]:
                options["resolution"] = geometry.input_size[0]
    return ResolvedModelSource(
        source_path,
        str(configured_name),
        options=options,
        geometry=geometry,
        manifest=manifest,
        source=str(path),
    )


def normalize_wandb_run(value: str) -> str:
    raw = value.strip()
    if raw.startswith("wandb:"):
        raw = raw.removeprefix("wandb:").strip("/")
    if "wandb.ai/" in raw:
        parsed = urlparse(raw if "://" in raw else f"https://{raw}")
        parts = [part for part in parsed.path.split("/") if part]
    else:
        parts = [part for part in raw.split("/") if part]
    if len(parts) == 4 and parts[2] == "runs":
        parts = [parts[0], parts[1], parts[3]]
    if len(parts) != 3:
        raise ValueError(
            "W&B source must be wandb:entity/project/run-id or a full run URL"
        )
    return "/".join(parts)


def _wandb_file(run: Any, requested: str | None) -> Any:
    files = {str(value.name): value for value in run.files()}

    def match(name: str) -> list[str]:
        return [
            candidate
            for candidate in files
            if candidate == name or Path(candidate).name == Path(name).name
        ]

    summary = dict(getattr(run, "summary", {}) or {})
    preferred = requested or summary.get("evaluation_bundle")
    if preferred:
        matches = match(str(preferred))
        if len(matches) == 1:
            return files[matches[0]]
        if requested:
            raise FileNotFoundError(f"Requested W&B file is missing or ambiguous: {requested}")
    zipped = sorted(name for name in files if name.lower().endswith(".zip"))
    if len(zipped) == 1:
        return files[zipped[0]]
    best = sorted(name for name in files if Path(name).name.lower() == "best.pt")
    if not zipped and len(best) == 1:
        return files[best[0]]
    raise DatasetValidationError(
        ValidationIssue(
            "Could not select one model file from the W&B run",
            value={"zip_files": zipped, "best_pt_files": best},
            suggestion="Specify run_file in the source mapping.",
        )
    )


def _remote_identity(run: Any, remote: Any) -> dict[str, Any]:
    return {
        "run": str(getattr(run, "path", None) or getattr(run, "id", "")),
        "run_updated_at": str(getattr(run, "updated_at", "") or ""),
        "name": str(remote.name),
        "size": getattr(remote, "size", None),
        "md5": getattr(remote, "md5", None),
        "updated_at": str(getattr(remote, "updated_at", "") or ""),
    }


def _download_wandb(
    reference: str,
    *,
    requested: str | None,
    progress: bool,
) -> tuple[Path, Any]:
    try:
        sdk = importlib.import_module("wandb")
    except ImportError as exc:
        raise ImportError(
            "Loading wandb: model sources requires W&B; reinstall dataset-fixer"
        ) from exc
    run_path = normalize_wandb_run(reference)
    run = sdk.Api().run(run_path)
    remote = _wandb_file(run, requested)
    identity = _remote_identity(run, remote)
    identity_hash = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    root = cache_root() / "models" / "wandb" / Path(run_path) / identity_hash
    destination = root / Path(str(remote.name)).name
    metadata = destination.with_suffix(destination.suffix + ".remote.json")
    if destination.is_file() and metadata.is_file():
        try:
            stored = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stored = {}
        expected_size = identity.get("size")
        if stored == identity and (
            expected_size is None or destination.stat().st_size == int(expected_size)
        ):
            if progress:
                print(f"Cache hit: W&B file {remote.name}")
            return destination, run

    root.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".download-", dir=root))
    try:
        downloaded_path = _download_wandb_file(
            remote,
            temporary_root,
            expected_size=identity.get("size"),
            progress=progress,
        )
        if not downloaded_path.is_file():
            raise RuntimeError(f"W&B download did not create a file: {downloaded_path}")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".partial", dir=root
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        temporary.unlink()
        downloaded_path.replace(temporary)
        temporary.replace(destination)
        metadata.write_text(json.dumps(identity, indent=2, sort_keys=True), encoding="utf-8")
    finally:
        import shutil

        shutil.rmtree(temporary_root, ignore_errors=True)
    return destination, run


def _download_wandb_file(
    remote: Any,
    root: Path,
    *,
    expected_size: int | None,
    progress: bool,
) -> Path:
    """Stream a signed W&B file URL with byte progress when available."""

    direct_url = str(getattr(remote, "url", "") or "").strip()
    target = root / Path(str(remote.name)).name
    if direct_url:
        try:
            request = Request(direct_url, headers={"User-Agent": "dataset-fixer"})
            with urlopen(request, timeout=60) as response, target.open("wb") as handle, tqdm(
                total=(
                    int(expected_size)
                    if expected_size is not None
                    else int(response.headers.get("Content-Length") or 0) or None
                ),
                desc=f"Downloading W&B file {remote.name}",
                unit="B",
                unit_scale=True,
                disable=not progress,
            ) as bar:
                for chunk in iter(lambda: response.read(_DOWNLOAD_CHUNK), b""):
                    handle.write(chunk)
                    bar.update(len(chunk))
                handle.flush()
                os.fsync(handle.fileno())
            if expected_size is not None and target.stat().st_size != int(expected_size):
                raise OSError(
                    f"W&B file size changed during download: expected {expected_size}, "
                    f"received {target.stat().st_size}"
                )
            return target.resolve()
        except Exception:
            target.unlink(missing_ok=True)
            if progress:
                print(
                    "Direct W&B byte-progress download was unavailable; "
                    "using the authenticated SDK downloader."
                )

    if progress and not direct_url:
        print(
            f"Downloading W&B file {remote.name} with the authenticated SDK "
            "(byte progress is unavailable for this file)."
        )
    downloaded = remote.download(root=str(root), exist_ok=True)
    try:
        return Path(downloaded.name).resolve()
    finally:
        close = getattr(downloaded, "close", None)
        if callable(close):
            close()


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for candidate in (
        path.with_suffix(".json"),
        path.parent / "model_metadata.json",
        path.parent / "metadata.json",
        path.parent / "bundle_manifest.json",
    ):
        if candidate.is_file():
            values.update(_read_json(candidate))
    try:
        torch = importlib.import_module("torch")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        checkpoint = None
    if isinstance(checkpoint, Mapping):
        train_args = dict(checkpoint.get("train_args") or {})
        serialized = checkpoint.get("model") or checkpoint.get("ema")
        model_args = dict(getattr(serialized, "args", {}) or {})
        values.setdefault("train_args", train_args)
        values.setdefault("model_args", model_args)
    return values


def _resolve_checkpoint(
    path: Path,
    *,
    name: str | None,
    run: Any = None,
    progress: bool,
) -> ResolvedModelSource:
    metadata = _checkpoint_metadata(path)
    train_args = dict(metadata.get("train_args") or {})
    model_args = dict(metadata.get("model_args") or {})
    run_config = dict(getattr(run, "config", {}) or {})
    run_summary = dict(getattr(run, "summary", {}) or {})
    geometry = from_metadata(
        dict(metadata.get("geometry") or {}),
        dict(metadata.get("dataset") or {}),
        metadata,
        dict(run_config.get("geometry") or {}),
        dict(run_config.get("dataset") or {}),
        run_config,
        train_args,
        model_args,
        source=str(path),
    )
    task = _task(
        first_value(metadata.get("task"), train_args.get("task"), model_args.get("task"), run_config.get("task"))
    )
    digest = sha256_progress(path, progress=progress)
    expected = first_value(
        metadata.get("checkpoint_sha256"),
        run_summary.get("selected_checkpoint_sha256"),
        run_summary.get("checkpoint_sha256"),
    )
    if expected is not None and str(expected).lower() != digest:
        raise DatasetValidationError(
            ValidationIssue(
                "Standalone checkpoint SHA-256 does not match metadata",
                source=str(path),
                value=digest,
                expected=str(expected),
            )
        )
    options: dict[str, Any] = {"kind": "ultralytics"}
    source_dataset_zip = _source_dataset_zip(metadata, run_config)
    if source_dataset_zip is not None:
        options["source_dataset_zip"] = source_dataset_zip
    if task:
        options["task"] = task
    model_metadata = dict(metadata.get("model") or {})
    run_train_args = dict(run_config.get("train_args") or {})
    options["model_type"] = _model_type(
        first_value(
            model_metadata.get("model_weights"),
            model_metadata.get("initialization_source"),
            model_metadata.get("base_model"),
            train_args.get("model"),
            model_args.get("model"),
            run_train_args.get("model"),
            run_config.get("model_weights"),
            run_config.get("base_model"),
            path.stem,
        ),
        kind="ultralytics",
        task=task,
    )
    if geometry.input_size and geometry.input_size[0] == geometry.input_size[1]:
        options["resolution"] = geometry.input_size[0]
    display = first_value(name, getattr(run, "display_name", None), path.stem)
    return ResolvedModelSource(
        path,
        str(display),
        options=options,
        geometry=geometry,
        manifest=metadata,
        source=str(path),
    )


def resolve_model_source(
    source: str | Path,
    *,
    name: str | None = None,
    run_file: str | None = None,
    progress: bool = True,
) -> ResolvedModelSource:
    """Infer and locally resolve one supported model source."""

    raw = str(source)
    run = None
    if raw.startswith("wandb:") or "wandb.ai/" in raw:
        source_key = f"wandb:{normalize_wandb_run(raw)}"
        path, run = _download_wandb(source_key, requested=run_file, progress=progress)
        resolved_name = name or source_key
    else:
        source_key = raw
        path = local_source(Path(source), progress=progress)
        resolved_name = name
    if path.is_dir():
        if path.name.startswith("fold_") and not (path / "plans.json").is_file():
            parent = path.parent
            if (parent / "dataset.json").is_file() and (parent / "plans.json").is_file():
                fold = path.name.removeprefix("fold_")
                metadata = (
                    _read_json(parent / "model_metadata.json")
                    if (parent / "model_metadata.json").is_file()
                    else {}
                )
                return ResolvedModelSource(
                    parent,
                    str(resolved_name or parent.name),
                    options={
                        "kind": "nnunet",
                        "task": "semantic",
                        "model_type": _model_type(
                            first_value(
                                metadata.get("model_preset"),
                                dict(metadata.get("model") or {}).get("model_preset"),
                                dict(metadata.get("model") or {}).get("plans"),
                                parent.name,
                            ),
                            kind="nnunet",
                            task="semantic",
                        ),
                        **(
                            {"source_dataset_zip": source_dataset_zip}
                            if (
                                source_dataset_zip := _source_dataset_zip(metadata)
                            ) is not None
                            else {}
                        ),
                        "folds": (fold,),
                    },
                    geometry=from_metadata(metadata, source=str(parent)),
                    source=source_key,
                )
        if (path / "dataset.json").is_file() and (path / "plans.json").is_file():
            metadata = (
                _read_json(path / "model_metadata.json")
                if (path / "model_metadata.json").is_file()
                else {}
            )
            return ResolvedModelSource(
                path,
                str(resolved_name or path.name),
                options={
                    "kind": "nnunet",
                    "task": "semantic",
                    "model_type": _model_type(
                        first_value(
                            metadata.get("model_preset"),
                            dict(metadata.get("model") or {}).get("model_preset"),
                            dict(metadata.get("model") or {}).get("plans"),
                            path.name,
                        ),
                        kind="nnunet",
                        task="semantic",
                    ),
                    **(
                        {"source_dataset_zip": source_dataset_zip}
                        if (
                            source_dataset_zip := _source_dataset_zip(metadata)
                        ) is not None
                        else {}
                    ),
                },
                geometry=from_metadata(metadata, source=str(path)),
                source=source_key,
            )
        raise DatasetValidationError(
            ValidationIssue(
                "Model folder is not an official nnU-Net model folder",
                value=str(path),
                expected="dataset.json and plans.json",
            )
        )
    if path.suffix.lower() == ".zip":
        resolved = _resolve_bundle(path, name=resolved_name, progress=progress)
        return replace(resolved, source=source_key)
    if path.suffix.lower() == ".pt":
        resolved = _resolve_checkpoint(
            path,
            name=resolved_name,
            run=run,
            progress=progress,
        )
        return replace(resolved, source=source_key)
    raise DatasetValidationError(
        ValidationIssue(
            "Unsupported model source",
            value=str(path),
            expected="a .pt file, nnU-Net folder, model bundle ZIP, or W&B run",
        )
    )


def is_wandb_reference(value: str) -> bool:
    return value.startswith("wandb:") or bool(
        re.match(r"https?://(?:www\.)?wandb\.ai/", value)
    )
