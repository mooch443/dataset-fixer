from __future__ import annotations

import csv
import gzip
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from .utils import settings_fingerprint, to_jsonable


REPORTS_DIR = "reports"
DATASET_INFO_NAME = "dataset-info.json"
SOURCE_INFO_NAME = "source.json"
LINEAGE_NAME = "lineage.json.gz"
DATASET_INFO_SCHEMA = 4
LINEAGE_SCHEMA = 2

# Files a generated dataset's reports/ directory is allowed to contain. Anything
# else is a transient operation artifact that belongs in dataset-info.json.
CANONICAL_REPORT_FILES = (
    DATASET_INFO_NAME,
    SOURCE_INFO_NAME,
    LINEAGE_NAME,
    "plots.png",
)


def dataset_info_path(root: Path) -> Path:
    return root / REPORTS_DIR / DATASET_INFO_NAME


def source_info_path(root: Path) -> Path:
    return root / REPORTS_DIR / SOURCE_INFO_NAME


def lineage_path(root: Path) -> Path:
    return root / REPORTS_DIR / LINEAGE_NAME


def stable_dataset_id(
    *,
    name: str,
    task: str,
    format_name: str,
    classes: Any,
    records: Iterable[dict[str, Any]],
) -> str:
    samples = [
        {
            "output": str(record.get("output_image")),
            "image_sha256": record.get("output_sha256")
            or record.get("output_image_sha256"),
            "mask_sha256": record.get("output_mask_sha256"),
            "annotations": record.get("output_annotation_count"),
        }
        for record in records
    ]
    return settings_fingerprint(
        {
            "schema": DATASET_INFO_SCHEMA,
            "name": name,
            "task": task,
            "format": format_name,
            "classes": classes,
            "samples": samples,
        }
    )


def source_dataset_id(
    *,
    source_manifest: dict[str, Any],
    source_name: str,
    source_path: Path,
    records: Iterable[dict[str, Any]],
) -> str:
    existing = source_manifest.get("dataset_id") or source_manifest.get("id")
    if existing:
        return str(existing)
    # Paths are deliberately excluded from the identity. A dataset may be moved
    # or mounted elsewhere without becoming a different physical dataset.
    samples = sorted(
        (
            str(record.get("parent_sha256")),
            str(record.get("original_sha256") or ""),
        )
        for record in records
    )
    return settings_fingerprint(
        {
            "schema": DATASET_INFO_SCHEMA,
            "name": source_manifest.get("name") or source_name,
            "samples": samples,
        }
    )


def split_image_summary(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Count labeled and background output images for every dataset split."""

    counts: dict[str, dict[str, int]] = {}
    for record in records:
        split = str(record.get("output_split") or "unknown")
        details = counts.setdefault(
            split,
            {
                "total_images": 0,
                "labeled_images": 0,
                "background_images": 0,
            },
        )
        has_labels = record.get("output_has_labels")
        if has_labels is None:
            has_labels = int(record.get("output_annotation_count") or 0) > 0
        details["total_images"] += 1
        details["labeled_images" if bool(has_labels) else "background_images"] += 1
    return {split: counts[split] for split in sorted(counts)}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_lineage(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "dataset-fixer-lineage",
        "schema_version": LINEAGE_SCHEMA,
        "records": [to_jsonable(record) for record in records],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(path)


def read_lineage(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "dataset-fixer-lineage":
        raise ValueError(f"Unsupported lineage document: {path}")
    if int(payload.get("schema_version", -1)) != LINEAGE_SCHEMA:
        raise ValueError(
            f"Unsupported lineage schema {payload.get('schema_version')!r}; "
            f"expected {LINEAGE_SCHEMA}"
        )
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Lineage records must be a list: {path}")
    return [dict(record) for record in records if isinstance(record, dict)]


def collect_report_payloads(reports_dir: Path) -> dict[str, Any]:
    """Merge operation report tables into one manifest section and remove them."""

    payloads: dict[str, Any] = {}
    for path in sorted(reports_dir.rglob("*.json")):
        if path.name in {DATASET_INFO_NAME, SOURCE_INFO_NAME}:
            continue
        key = path.relative_to(reports_dir).with_suffix("").as_posix().replace("/", ".")
        try:
            payloads[key] = _normalize_audit_payload(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError):
            payloads[key] = {"status": "unreadable"}
        path.unlink(missing_ok=True)
    for path in sorted(reports_dir.rglob("*.csv")):
        key = path.relative_to(reports_dir).with_suffix("").as_posix().replace("/", ".")
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except OSError:
            rows = []
        if key in payloads:
            payloads[key] = {"summary": payloads[key], "rows": rows}
        else:
            payloads[key] = rows
        path.unlink(missing_ok=True)
    return payloads


def _normalize_audit_payload(value: Any) -> Any:
    """Remove legacy before/after pairs from newly written dataset reports."""

    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name == "before" or name.endswith("_before"):
                continue
            if name == "after":
                continue
            if name.endswith("_after"):
                name = name.removesuffix("_after")
            output[name] = _normalize_audit_payload(item)
        if "after" in value:
            output["result"] = _normalize_audit_payload(value["after"])
        return output
    if isinstance(value, list):
        return [_normalize_audit_payload(item) for item in value]
    return value


def prune_report_directory(reports_dir: Path, *, extra_roots: Iterable[Path] = ()) -> None:
    """Reduce a generated ``reports/`` tree to the canonical artifact files.

    Operation audits already live inside ``dataset-info.json``; the transient
    preview images and nested legacy report directories that produced them are
    removed so a derived dataset never inherits another dataset's history.
    """

    for root in extra_roots:
        if root.is_dir():
            shutil.rmtree(root, ignore_errors=True)
    if not reports_dir.is_dir():
        return
    for path in sorted(reports_dir.iterdir(), reverse=True):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.name not in CANONICAL_REPORT_FILES:
            path.unlink(missing_ok=True)
