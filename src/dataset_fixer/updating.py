from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from PIL import Image
from tqdm.auto import tqdm

from .artifacts import (
    DATASET_INFO_SCHEMA,
    LINEAGE_SCHEMA,
    collect_report_payloads,
    dataset_info_path,
    lineage_path,
    prune_report_directory,
    source_info_path,
    split_image_summary,
    stable_dataset_id,
    write_json,
    write_lineage,
)
from .dataset_report import render_dataset_report
from .errors import DatasetValidationError, ValidationIssue
from .utils import environment_snapshot, ensure_safe_destination, sha256_file

if TYPE_CHECKING:
    from .dataset import Dataset
    from .models import Sample


def update_dataset(
    dataset: "Dataset",
    *,
    dest: str | Path | None,
    progress: bool = True,
) -> "Dataset":
    """Upgrade generated dataset metadata in place or in a new physical copy."""

    source_root = dataset.location.resolve()
    if dataset.manifest_path is None:
        raise DatasetValidationError(
            ValidationIssue(
                "Dataset.update requires a dataset-fixer manifest",
                source=str(source_root),
                expected="reports/dataset-info.json or legacy dataset-fixer.json",
                suggestion="export the source dataset once before updating it",
            )
        )
    target = source_root if dest is None else Path(dest).expanduser().resolve()
    in_place = target == source_root
    if not in_place:
        target.parent.mkdir(parents=True, exist_ok=True)
        ensure_safe_destination(source_root, target)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{target.name}.update-",
                dir=target.parent,
            )
        )
        try:
            _copy_dataset_tree(source_root, staging, progress=progress)
            _update_root(
                dataset,
                staging,
                published_location=target,
                progress=progress,
            )
            _validate_updated_root(dataset, staging, progress=progress)
            os.replace(staging, target)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    else:
        _update_root(
            dataset,
            source_root,
            published_location=source_root,
            progress=progress,
        )
        _validate_updated_root(dataset, source_root, progress=progress)

    from .dataset import Dataset

    return Dataset.open(target, task=dataset.task, progress=False)


def _copy_dataset_tree(source_root: Path, staging: Path, *, progress: bool) -> None:
    paths = [path for path in sorted(source_root.rglob("*")) if path.is_file()]
    iterator = tqdm(
        paths,
        desc="Copying dataset files",
        unit="file",
        disable=not progress,
    )
    for path in iterator:
        destination = staging / path.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def _update_root(
    dataset: "Dataset",
    root: Path,
    *,
    published_location: Path,
    progress: bool,
) -> None:
    manifest = dict(dataset.manifest)
    previous_schema = int(manifest.get("schema_version") or 1)
    records = _current_lineage_records(dataset, progress=progress)
    dataset_id = str(
        manifest.get("dataset_id")
        or stable_dataset_id(
            name=dataset.name,
            task=dataset.task.value,
            format_name=dataset.format,
            classes=dataset.classes,
            records=records,
        )
    )
    for record in records:
        record["dataset_id"] = dataset_id

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{root.name}.reports-update-",
            dir=root.parent,
        )
    )
    staged_reports = temporary_root / "reports"
    existing_reports = root / "reports"
    if existing_reports.is_dir():
        shutil.copytree(existing_reports, staged_reports)
    else:
        staged_reports.mkdir(parents=True)

    audits = dict(manifest.get("audits") or {})
    audits.update(collect_report_payloads(staged_reports))
    coverage = root / "coverage_summary"
    if coverage.is_dir():
        staged_coverage = temporary_root / "coverage_summary"
        shutil.copytree(coverage, staged_coverage)
        audits.update(
            {
                f"coverage.{key}": value
                for key, value in collect_report_payloads(staged_coverage).items()
            }
        )
    prune_report_directory(staged_reports)
    if progress:
        print("Rendering dataset report from the physically present dataset...")
    render_dataset_report(
        root,
        records,
        name=dataset.name,
        task=dataset.task.value,
        format_name=dataset.format,
        classes=dataset.classes,
        output=staged_reports / "plots.png",
        metadata=dataset._metadata,
        coverage=audits.get("coverage.source_coverage"),
    )

    environment = environment_snapshot()
    try:
        from . import __version__
    except ImportError:
        __version__ = "unknown"
    manifest.update(
        {
            "schema": "dataset-fixer-dataset-info",
            "schema_version": DATASET_INFO_SCHEMA,
            "dataset_id": dataset_id,
            "name": dataset.name,
            "location": str(published_location),
            "task": dataset.task.value,
            "format": dataset.format,
            "splits": list(dataset.splits),
            "split_summary": split_image_summary(records),
            "classes": dataset.classes,
            "dataset_fixer": {
                "version": __version__,
                "commit": environment["dataset_fixer_git"]["commit"],
                "dirty": environment["dataset_fixer_git"]["dirty"],
            },
            "environment": environment,
            "lineage": {
                "path": "reports/lineage.json.gz",
                "schema_version": LINEAGE_SCHEMA,
                "records": len(records),
            },
            "audits": audits,
            "artifact_update": {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "from_schema_version": previous_schema,
                "to_schema_version": DATASET_INFO_SCHEMA,
                "data_files_changed": False,
            },
        }
    )
    source_payload = _current_source_payload(dataset, staged_reports)
    if source_payload is not None:
        source_payload["schema"] = "dataset-fixer-source"
        source_payload["schema_version"] = DATASET_INFO_SCHEMA
        write_json(source_info_path(temporary_root), source_payload)
    write_lineage(lineage_path(temporary_root), records)
    write_json(dataset_info_path(temporary_root), manifest)

    previous_reports = temporary_root / "previous-reports"
    try:
        if existing_reports.exists():
            os.replace(existing_reports, previous_reports)
        os.replace(staged_reports, existing_reports)
    except Exception:
        if not existing_reports.exists() and previous_reports.exists():
            os.replace(previous_reports, existing_reports)
        raise
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)

    _make_yaml_portable(dataset, root)
    (root / "dataset-fixer.json").unlink(missing_ok=True)
    (root / "provenance.jsonl").unlink(missing_ok=True)
    if coverage.is_dir():
        shutil.rmtree(coverage)


def _current_lineage_records(
    dataset: "Dataset",
    *,
    progress: bool = False,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    iterator = tqdm(
        dataset._samples,
        desc="Hashing and indexing images",
        unit="image",
        disable=not progress,
    )
    for sample in iterator:
        relative_image = _relative_to_root(sample.image_path, dataset.location)
        record = dict(sample.provenance or dataset._provenance.get(relative_image, {}))
        image_sha = sha256_file(sample.image_path)
        record.update(
            {
                "output_image": relative_image,
                "output_split": sample.split,
                "output_annotation_count": len(sample.annotations),
                "output_has_labels": _sample_has_labels(dataset, sample),
            }
        )
        if dataset.format == "semantic_masks":
            mask_path = dataset._mask_paths.get(sample.image_path.resolve())
            if mask_path is None:
                raise DatasetValidationError(
                    f"Semantic dataset has no mask for {sample.image_path}"
                )
            record.update(
                {
                    "output_image_sha256": image_sha,
                    "output_mask": _relative_to_root(mask_path, dataset.location),
                    "output_mask_sha256": sha256_file(mask_path),
                }
            )
        else:
            record["output_sha256"] = image_sha
        records.append(record)
    return records


def _sample_has_labels(dataset: "Dataset", sample: "Sample") -> bool:
    if dataset.format != "semantic_masks":
        return bool(sample.annotations)
    mask_path = dataset._mask_paths.get(sample.image_path.resolve())
    if mask_path is None:
        return False
    with Image.open(mask_path) as opened:
        return opened.convert("L").getbbox() is not None


def _current_source_payload(
    dataset: "Dataset",
    reports: Path,
) -> dict[str, Any] | None:
    current = reports / "source.json"
    if current.is_file():
        try:
            value = json.loads(current.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            value = None
        if isinstance(value, dict):
            return value
    source = dataset.manifest.get("source_dataset")
    if not isinstance(source, dict) or not source:
        return None
    return {
        "id": source.get("id"),
        "path": source.get("path") or source.get("location"),
        "name": source.get("name"),
        "format": source.get("format"),
        "task": source.get("task"),
        "fingerprint": source.get("fingerprint"),
        "classes": source.get("classes"),
        "splits": source.get("splits"),
    }


def _make_yaml_portable(dataset: "Dataset", root: Path) -> None:
    """Drop the legacy absolute or relative ``path`` key from a training YAML.

    Split entries are already relative to the YAML file, so removing ``path``
    makes the dataset resolve correctly wherever it is moved or mounted.
    """

    if dataset.data_yaml is None:
        return
    relative = dataset.data_yaml.resolve().relative_to(dataset.location.resolve())
    yaml_path = root / relative
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if "path" not in data:
        return
    data.pop("path")
    temporary = yaml_path.with_suffix(yaml_path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    temporary.replace(yaml_path)


def _validate_updated_root(dataset: "Dataset", root: Path, *, progress: bool) -> None:
    from .dataset import Dataset

    if progress:
        print("Validating the upgraded dataset before publication...")
    reopened = Dataset.open(root, task=dataset.task, progress=False)
    before = _payload_hashes(dataset.location)
    after = _payload_hashes(root)
    if before != after:
        raise DatasetValidationError("Dataset.update changed physical image, label, or mask files")
    before_images = {
        _relative_to_root(sample.image_path, dataset.location)
        for sample in dataset._samples
    }
    after_images = {
        _relative_to_root(sample.image_path, root)
        for sample in reopened._samples
    }
    if before_images != after_images:
        raise DatasetValidationError("Dataset.update changed the physical image cohort")
    info = reopened.manifest
    if int(info.get("schema_version") or 0) != DATASET_INFO_SCHEMA:
        raise DatasetValidationError("Dataset.update did not write the latest schema")
    if not isinstance(info.get("split_summary"), dict):
        raise DatasetValidationError("Dataset.update did not write split_summary")


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise DatasetValidationError(
            f"Dataset file lies outside its physical root: {path}"
        ) from exc


def _payload_hashes(root: Path) -> dict[str, str]:
    payload: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for directory in ("images", "labels", "masks"):
            base = root / split / directory
            if not base.is_dir():
                continue
            for path in sorted(base.rglob("*")):
                if path.is_file():
                    payload[path.relative_to(root).as_posix()] = sha256_file(path)
    return payload
