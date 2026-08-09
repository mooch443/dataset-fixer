"""Portable, framework-neutral model bundle creation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from tqdm.auto import tqdm

from ..convert import Prepared
from ..geometry import Geometry
from ..sources import cache_root, sha256_progress
from ..utils import slugify, to_jsonable


@dataclass(frozen=True)
class Config:
    """Framework-neutral inputs recorded in a model bundle.

    Parameters:
        name: Stable model display name used for the bundle file.
        framework: Training/inference framework identifier.
        task: Normalized model task.
        geometry: Source-tile, scale, and model-input geometry.
        dataset: Prepared dataset identity or equivalent manifest mapping.
        model: Framework-specific model configuration.
        training: Searchable training hyperparameters.
        run: Optional external training-run identity.
        files: Additional files mapped from archive names to local paths.
    """

    name: str
    framework: str
    task: str
    geometry: Geometry | Mapping[str, Any]
    dataset: Prepared | Mapping[str, Any]
    model: Mapping[str, Any] = field(default_factory=dict)
    training: Mapping[str, Any] = field(default_factory=dict)
    run: Mapping[str, Any] = field(default_factory=dict)
    files: Mapping[str, str | Path] = field(default_factory=dict)


@dataclass(frozen=True)
class Outcome:
    """Selected checkpoint and optional training/evaluation results.

    Parameters:
        checkpoint: Selected local checkpoint file or model directory.
        metrics: Final training or evaluation metrics.
        selected_epoch: Epoch that produced the selected checkpoint.
        selection_metric: Metric used to select the checkpoint.
        selection_value: Value of the checkpoint-selection metric.
        files: Additional result files mapped from archive names to paths.
    """

    checkpoint: str | Path | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    selected_epoch: int | None = None
    selection_metric: str | None = None
    selection_value: float | None = None
    files: Mapping[str, str | Path] = field(default_factory=dict)


@dataclass(frozen=True)
class Bundle:
    """Local bundle identity, contents, and optional remote upload outcome.

    Parameters:
        path: Local ZIP path, which remains available after upload attempts.
        size: ZIP size in bytes.
        sha256: SHA-256 digest of the ZIP.
        manifest: Parsed model-bundle manifest.
        files: Archive member names included in the ZIP.
        uploaded: Whether a remote upload completed successfully.
        remote_url: Remote file URL when supplied by the service.
        warnings: Non-fatal upload or publication messages.
    """

    path: Path
    size: int
    sha256: str
    manifest: Mapping[str, Any]
    files: tuple[str, ...]
    uploaded: bool = False
    remote_url: str | None = None
    warnings: tuple[str, ...] = ()


def _geometry(value: Geometry | Mapping[str, Any]) -> Geometry:
    return value if isinstance(value, Geometry) else Geometry.create(**dict(value))


def _dataset(value: Prepared | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Prepared):
        return {
            "name": value.location.name,
            "content_sha256": value.content_sha256,
            "preparation_kind": value.kind.value,
            "geometry": value.geometry.as_dict(),
            "split_statistics": to_jsonable(value.split_statistics),
            "manifest": str(value.manifest),
        }
    return to_jsonable(dict(value))


def _archive_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Bundle file name must be a safe relative path: {value!r}")
    return path


def _collect_files(
    config: Config,
    outcome: Outcome | None,
) -> list[tuple[Path, PurePosixPath, str]]:
    declared: list[tuple[str, str | Path, str]] = [
        (name, path, "model") for name, path in config.files.items()
    ]
    if outcome is not None:
        declared.extend((name, path, "outcome") for name, path in outcome.files.items())
        if outcome.checkpoint is not None:
            checkpoint = Path(outcome.checkpoint).expanduser().resolve()
            declared.append((f"weights/{checkpoint.name}", checkpoint, "checkpoint"))
    collected: list[tuple[Path, PurePosixPath, str]] = []
    names: set[str] = set()
    for archive_name, raw_path, role in declared:
        source = Path(raw_path).expanduser().resolve()
        archive = _archive_path(str(archive_name))
        if not source.exists():
            raise FileNotFoundError(source)
        if source.is_symlink():
            raise ValueError(f"Bundle inputs may not be symbolic links: {source}")
        values = [source] if source.is_file() else sorted(
            path for path in source.rglob("*") if path.is_file()
        )
        for value in values:
            if value.is_symlink():
                raise ValueError(f"Bundle inputs may not be symbolic links: {value}")
            target = archive if source.is_file() else archive / value.relative_to(source).as_posix()
            key = target.as_posix()
            if key in names:
                raise ValueError(f"Duplicate bundle member: {key}")
            names.add(key)
            collected.append((value, target, role))
    if not collected:
        raise ValueError("bundle.create() requires a checkpoint or at least one declared file")
    return collected


def create(
    config: Config,
    outcome: Outcome | None = None,
    *,
    destination: str | Path | None = None,
    progress: bool = True,
) -> Bundle:
    """Create a local, checksummed ZIP using the shared model-bundle manifest.

    Parameters:
        config: Framework-neutral model, dataset, and training configuration.
        outcome: Optional selected checkpoint and result metadata.
        destination: Output ZIP path or directory. The automatic cache is used
            when omitted.
        progress: Show hashing and ZIP creation progress.

    Returns:
        The immutable local bundle identity and manifest.
    """

    if not isinstance(config, Config):
        raise TypeError("config must be bundle.Config")
    if outcome is not None and not isinstance(outcome, Outcome):
        raise TypeError("outcome must be bundle.Outcome or None")
    geometry = _geometry(config.geometry)
    files = _collect_files(config, outcome)
    entries: list[dict[str, Any]] = []
    content = hashlib.sha256()
    for source, archive, role in tqdm(
        files,
        desc="Hashing model bundle files",
        disable=not progress,
    ):
        digest = sha256_progress(source, progress=False)
        entry = {
            "path": archive.as_posix(),
            "role": role,
            "size": source.stat().st_size,
            "sha256": digest,
        }
        entries.append(entry)
        content.update(json.dumps(entry, sort_keys=True).encode("utf-8"))
    dataset = _dataset(config.dataset)
    model = {**to_jsonable(dict(config.model)), "framework": config.framework, "task": config.task}
    checkpoint_entry = next((entry for entry in entries if entry["role"] == "checkpoint"), None)
    if checkpoint_entry is not None:
        model.setdefault("checkpoint", Path(checkpoint_entry["path"]).name)
        model["checkpoint_sha256"] = checkpoint_entry["sha256"]
    manifest = {
        "schema_version": 1,
        "format": "model-bundle",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "name": str(config.name),
        "framework": str(config.framework),
        "task": str(config.task),
        "geometry": geometry.as_dict(),
        "dataset": dataset,
        "model": model,
        "training": to_jsonable(dict(config.training)),
        "run": to_jsonable(dict(config.run)),
        "outcome": (
            {
                "metrics": to_jsonable(dict(outcome.metrics)),
                "selected_epoch": outcome.selected_epoch,
                "selection_metric": outcome.selection_metric,
                "selection_value": outcome.selection_value,
                "checkpoint_sha256": checkpoint_entry["sha256"] if checkpoint_entry else None,
            }
            if outcome is not None
            else None
        ),
        "files": entries,
    }
    identity_value = {
        key: value for key, value in manifest.items() if key != "created_at"
    }
    identity = hashlib.sha256(
        json.dumps(identity_value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    filename = f"{slugify(config.name)}-{slugify(config.task)}-{identity[:12]}.zip"
    if destination is None:
        output = cache_root() / "bundles" / filename
    else:
        requested = Path(destination).expanduser().resolve()
        output = requested / filename if requested.suffix.lower() != ".zip" else requested
    if Path("/content/drive") in output.parents:
        raise ValueError(
            "Model bundles are always created in local storage; choose /content or "
            "omit destination, then download or move the returned ZIP yourself."
        )
    if output.is_file():
        try:
            with zipfile.ZipFile(output) as zipped:
                cached_manifest = json.loads(
                    zipped.read("bundle_manifest.json").decode("utf-8")
                )
                cached_files = tuple(zipped.namelist())
        except (OSError, KeyError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
            raise FileExistsError(
                f"Bundle destination already contains an unrelated/incomplete file: {output}"
            ) from exc
        cached_identity = hashlib.sha256(
            json.dumps(
                {key: value for key, value in cached_manifest.items() if key != "created_at"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if cached_identity != identity:
            raise FileExistsError(
                f"Bundle destination already contains a different bundle: {output}"
            )
        digest = sha256_progress(output, progress=False)
        if progress:
            print(f"Cache hit: model bundle {output}")
        return Bundle(
            output,
            output.stat().st_size,
            digest,
            cached_manifest,
            cached_files,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    total = sum(entry["size"] for entry in entries)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as zipped, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc="Creating model bundle ZIP",
            disable=not progress,
        ) as bar:
            zipped.writestr(
                "bundle_manifest.json",
                json.dumps(manifest, indent=2, sort_keys=True),
            )
            for source, archive, _role in files:
                with source.open("rb") as input_handle, zipped.open(archive.as_posix(), "w") as output_handle:
                    for chunk in iter(lambda: input_handle.read(8 * 1024 * 1024), b""):
                        output_handle.write(chunk)
                        bar.update(len(chunk))
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    digest = sha256_progress(output, progress=progress, description="Hashing model bundle ZIP")
    return Bundle(
        output,
        output.stat().st_size,
        digest,
        manifest,
        tuple(["bundle_manifest.json", *(entry["path"] for entry in entries)]),
    )


__all__ = ["Bundle", "Config", "Outcome", "create"]
