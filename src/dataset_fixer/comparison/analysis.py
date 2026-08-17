"""Shared report analyses for native and semantic comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .grouping import annotate_group_splits
from .metrics import grouped_binary_metric_breakdown, grouped_presence_metric_breakdown
from .object_sizes import render_grouped_metric_breakdown, render_grouped_presence_metric_breakdown


def analyze_grouped_metrics(
    rows_by_model: Mapping[str, list[dict[str, Any]]],
    groups: Mapping[str, str],
    ranking: list[dict[str, Any]],
    reports: Path,
    *,
    group_settings: Mapping[str, Any] | None,
    aggregation_unit: str,
) -> tuple[dict[str, Any], str]:
    """Compute and render the common grouped pixel and presence analysis."""

    group_splits = dict((group_settings or {}).get("group_splits") or {})
    binary: dict[str, dict[str, Any]] = {}
    presence: dict[str, dict[str, Any]] = {}
    for row in ranking:
        name = str(row["model"])
        result = grouped_binary_metric_breakdown(rows_by_model[name], groups)
        annotate_group_splits(result, group_splits)
        binary[name] = result
        row.update({key: value for key, value in result.items() if key != "per_group"})

    presence_available = all(
        "component_filtered_predicted_presence" in row
        for rows in rows_by_model.values()
        for row in rows
    )
    if presence_available:
        for row in ranking:
            name = str(row["model"])
            decisions = {
                str(value.get("case_id", value.get("image_id"))): bool(
                    value["component_filtered_predicted_presence"]
                )
                for value in rows_by_model[name]
            }
            result = grouped_presence_metric_breakdown(rows_by_model[name], groups, decisions)
            annotate_group_splits(result, group_splits)
            presence[name] = result
            row.update({key: value for key, value in result.items() if key != "per_group"})

    path = render_grouped_metric_breakdown(
        reports, ranking, binary, group_splits=group_splits
    )
    report_paths: dict[str, str | None] = {metric: None for metric in ("precision", "recall", "f1")}
    if presence_available:
        for metric in report_paths:
            rendered = render_grouped_presence_metric_breakdown(
                reports, ranking, presence, metric=metric, group_splits=group_splits
            )
            report_paths[metric] = str(rendered.relative_to(reports.parent))
    return {
        "status": "complete",
        "aggregation": "pool TP/FP/FN within group, then macro-average group scores",
        "primary_ranking_unchanged": True,
        "grouping": dict(group_settings or {}),
        "models": binary,
        "presence": {
            "status": "complete" if presence_available else "skipped",
            "prediction_definition": (
                "at least one predicted 8-connected foreground component "
                "at or above the resolved area threshold"
            ),
            "aggregation": (
                f"pool {aggregation_unit}-level TP/FP/FN/TN within group, then "
                "macro-average defined group scores"
            ),
            "models": presence,
        },
        "reports": report_paths,
    }, str(path.relative_to(reports.parent))
