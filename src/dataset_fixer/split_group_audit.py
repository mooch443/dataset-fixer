from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import pandas as pd

from .errors import DatasetValidationError, ValidationIssue
from .tabular import frame
from .utils import STANDARD_SPLITS, to_jsonable

if TYPE_CHECKING:
    from .dataset import Dataset


REPORT_NAME = "split_group_audit.json"
REPORT_PATH = "reports/dataset-info.json#audits.split_group_audit"


def audit_split_groups(
    dataset: "Dataset",
    *,
    exported_splits: Iterable[str],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Audit the latest group-aware split across the complete current dataset."""

    latest_split = _latest_split_record(dataset._manifest.get("history") or [])
    exported = _ordered_splits(exported_splits)
    if latest_split is None:
        return _not_applicable("dataset history contains no split operation", exported), None

    settings = latest_split.get("settings")
    if not isinstance(settings, dict) or settings.get("group_by") is None:
        return _not_applicable("latest split operation did not use group_by", exported), None

    samples = frame(
        {
            "split": sample.split,
            "group": str((sample.provenance or {}).get("split_group")),
            "has_group": "split_group" in (sample.provenance or {}),
        }
        for sample in dataset._samples
    )
    missing = samples[~samples["has_group"]]
    if not missing.empty:
        raise DatasetValidationError(
            ValidationIssue(
                "Physical split-group isolation is unverifiable because samples lack group identity",
                value={
                    "missing_count": len(missing),
                    "missing_by_split": missing["split"].value_counts().sort_index().to_dict(),
                },
                expected="split_group provenance for every current dataset image",
                suggestion="recreate the dataset from the latest group-aware split before exporting",
            )
        )

    counts = samples.groupby(["group", "split"], sort=True).size().rename("images")
    overlaps = [
        {"group": group, "splits": values.droplevel("group").to_dict()}
        for group, values in counts.groupby(level="group", sort=True)
        if len(values) > 1
    ]
    if overlaps:
        raise DatasetValidationError(
            ValidationIssue(
                "Physical split group appears in multiple dataset splits",
                value={
                    "overlap_count": len(overlaps),
                    "groups": overlaps[:10],
                },
                expected="each split_group to occur in exactly one of train, val, or test",
                suggestion="recreate the split with the intended group_by callback before exporting",
            )
        )

    audited = _ordered_splits(samples["split"])
    per_split = {
        split: {
            "images": int(len(rows)),
            "distinct_groups": int(rows["group"].nunique()),
            "group_size": _size_distribution(rows.groupby("group").size()),
        }
        for split in audited
        for rows in [samples[samples["split"].eq(split)]]
    }
    report = {
        "schema_version": 1,
        "status": "passed",
        "verified": True,
        "scope": "all_current_source_splits",
        "group_by": to_jsonable(settings["group_by"]),
        "audited_splits": audited,
        "exported_splits": exported,
        "total": {
            "images": len(dataset._samples),
            "distinct_groups": int(samples["group"].nunique()),
        },
        "splits": per_split,
        "overlap_count": 0,
    }
    validation = {
        "status": "passed",
        "verified": True,
        "scope": report["scope"],
        "report": REPORT_PATH,
        "audited_splits": audited,
        "exported_splits": exported,
        "images": report["total"]["images"],
        "distinct_groups": report["total"]["distinct_groups"],
        "overlap_count": 0,
    }
    return validation, report


def write_split_group_audit(reports_dir: Path, report: dict[str, Any] | None) -> Path | None:
    """Write the current audit or remove a stale inherited report."""

    path = reports_dir / REPORT_NAME
    if report is None:
        path.unlink(missing_ok=True)
        return None
    path.write_text(
        json.dumps(to_jsonable(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def print_split_group_audit(report: dict[str, Any] | None) -> None:
    if report is None:
        return
    print("Split-group audit: passed (all current source splits)")
    for split in report["audited_splits"]:
        summary = report["splits"][split]
        sizes = summary["group_size"]
        print(
            f"  {split}: {summary['images']} images / {summary['distinct_groups']} groups "
            f"| group size min={sizes['min']}, median={sizes['median']:g}, "
            f"mean={sizes['mean']:g}, max={sizes['max']}"
        )


def _latest_split_record(history: Iterable[Any]) -> dict[str, Any] | None:
    for record in reversed(list(history)):
        if isinstance(record, dict) and record.get("operation") in {
            "split",
            "move-images-with-classes",
            "move-n-groups",
        }:
            return record
    return None


def _not_applicable(reason: str, exported_splits: list[str]) -> dict[str, Any]:
    return {
        "status": "not_applicable",
        "verified": False,
        "scope": "all_current_source_splits",
        "report": None,
        "reason": reason,
        "exported_splits": exported_splits,
    }


def _ordered_splits(values: Iterable[str]) -> list[str]:
    available = {str(value) for value in values}
    return [
        *[split for split in STANDARD_SPLITS if split in available],
        *sorted(available - set(STANDARD_SPLITS)),
    ]


def _size_distribution(values: Iterable[int]) -> dict[str, Any]:
    sizes = pd.Series(values, dtype="int64").sort_values()
    return {
        "min": int(sizes.min()),
        "max": int(sizes.max()),
        "mean": statistics.mean(sizes.tolist()),
        "median": statistics.median(sizes.tolist()),
        "histogram": {
            str(size): int(count)
            for size, count in sizes.value_counts().sort_index().items()
        },
    }
