from __future__ import annotations

import math
import re
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


SIZE_GROUPS = ("small", "medium", "large")


def object_size_report_artifacts_exist(root: Path, manifest: Mapping[str, Any]) -> bool:
    """Check the schema-added report paths before reusing a comparison."""

    reports = manifest.get("reports")
    if not isinstance(reports, Mapping):
        return False
    if "object_size_breakdown" not in reports or "large_object_examples" not in reports:
        return False
    analysis = manifest.get("object_size_analysis")
    if not isinstance(analysis, Mapping):
        return False
    # Schema 13/8 reports created before tiled cohorts were enabled recorded
    # this skip. Treat them as incomplete so cached predictions are rescored.
    if analysis.get("reason") == (
        "object-size analysis is unavailable for tiled evaluation datasets"
    ):
        return False
    dataset = manifest.get("dataset")
    dataset_task = dataset.get("task") if isinstance(dataset, Mapping) else None
    is_segmentation = (
        manifest.get("kind") == "semantic-mask-model-comparison"
        or dataset_task == "segment"
    )
    if is_segmentation and not isinstance(reports.get("metric_breakdown"), str):
        return False
    if (
        analysis.get("status") == "complete"
        and not isinstance(reports.get("object_size_breakdown"), str)
    ):
        return False
    paths: list[str] = []
    for key in (
        "metric_breakdown",
        "grouped_metric_breakdown",
        "object_size_breakdown",
    ):
        value = reports.get(key)
        if value is not None:
            if not isinstance(value, str):
                return False
            paths.append(value)
    examples = reports.get("large_object_examples", [])
    if not isinstance(examples, list):
        return False
    for example in examples:
        if isinstance(example, str):
            paths.append(example)
            continue
        if not isinstance(example, Mapping) or not isinstance(example.get("path"), str):
            return False
        paths.append(str(example["path"]))
    return all((root / path).is_file() for path in paths)


@dataclass(frozen=True)
class ObjectComponent:
    """One foreground object stored as a compact mask in source coordinates."""

    component_id: str
    image_id: str
    relative_path: str
    image_path: Path
    class_id: int
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    area: int


@dataclass(frozen=True)
class ObjectSizeReference:
    status: str
    reason: str | None
    components: dict[str, tuple[ObjectComponent, ...]]
    p10_area: float | None
    p90_area: float | None
    reference_extraction: str | None
    prediction_extraction: str | None
    connectivity: int | None
    matching_class_policy: str | None

    def group(self, area: int) -> str:
        if self.p10_area is None or self.p90_area is None:
            raise ValueError("Object-size thresholds are unavailable")
        # Small has priority when a degenerate cohort makes p10 == p90.  This
        # keeps the groups mutually exclusive while preserving the inclusive
        # lower-tail rule.
        if area <= self.p10_area:
            return "small"
        if area >= self.p90_area:
            return "large"
        return "medium"

    @property
    def all_components(self) -> tuple[ObjectComponent, ...]:
        return tuple(
            component
            for image_id in sorted(self.components)
            for component in self.components[image_id]
        )

    def metadata(self) -> dict[str, Any]:
        common = {
            "coordinate_space": "native-source-pixels",
            "object_measure": "foreground-pixel-area",
            "reference_source": "active-held-out-cohort",
            "reference_object_extraction": self.reference_extraction,
            "prediction_object_extraction": self.prediction_extraction,
            "prediction_stage": "final-reconstructed-source-image",
            "evaluation_frequency": "once-per-source-image",
            "connectivity": self.connectivity,
            "percentiles": {"lower": 10, "upper": 90},
            "percentile_method": "linear",
            "thresholds": {
                "p10_area_px": self.p10_area,
                "p90_area_px": self.p90_area,
            },
            "groups": {
                "small": "area <= p10",
                "medium": "p10 < area < p90",
                "large": "area >= p90",
            },
            "percentile_tie_policy": "small-precedence-when-p10-equals-p90",
            "matching": "hungarian-one-to-one-maximum-dice-positive-overlap",
            "matching_class_policy": self.matching_class_policy,
            "unmatched_score": 0.0,
        }
        if self.status != "complete":
            return {
                "status": self.status,
                "reason": self.reason,
                **common,
                "reference_support": {group: 0 for group in SIZE_GROUPS},
            }
        support = {group: 0 for group in SIZE_GROUPS}
        for component in self.all_components:
            support[self.group(component.area)] += 1
        return {
            "status": "complete",
            **common,
            "reference_support": support,
        }


@dataclass(frozen=True)
class ObjectSizeModelResult:
    summary: dict[str, Any]
    reference_scores: dict[str, float]
    matched_prediction_ids: dict[str, str]


def skipped_object_size_reference(reason: str) -> ObjectSizeReference:
    return ObjectSizeReference(
        status="skipped",
        reason=reason,
        components={},
        p10_area=None,
        p90_area=None,
        reference_extraction=None,
        prediction_extraction=None,
        connectivity=None,
        matching_class_policy=None,
    )


def unavailable_object_size_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for group in SIZE_GROUPS:
        summary.update(
            {
                f"{group}_object_dice": math.nan,
                f"{group}_object_reference_count": 0,
                f"{group}_object_prediction_count": 0,
                f"{group}_object_match_count": 0,
                f"{group}_object_scoring_count": 0,
            }
        )
    return summary


def prepare_object_size_reference(
    components: dict[str, tuple[ObjectComponent, ...]],
    *,
    reference_extraction: str = "semantic-8-connected-foreground-components",
    prediction_extraction: str = (
        "8-connected-components-from-final-full-image-binary-mask"
    ),
    connectivity: int | None = 8,
    matching_class_policy: str = "binary-foreground",
) -> ObjectSizeReference:
    areas = np.asarray(
        [component.area for values in components.values() for component in values],
        dtype=float,
    )
    if not len(areas):
        return ObjectSizeReference(
            status="skipped",
            reason="held-out cohort contains no reference foreground objects",
            components=components,
            p10_area=None,
            p90_area=None,
            reference_extraction=reference_extraction,
            prediction_extraction=prediction_extraction,
            connectivity=connectivity,
            matching_class_policy=matching_class_policy,
        )
    p10, p90 = np.percentile(areas, [10, 90])
    return ObjectSizeReference(
        status="complete",
        reason=None,
        components=components,
        p10_area=float(p10),
        p90_area=float(p90),
        reference_extraction=reference_extraction,
        prediction_extraction=prediction_extraction,
        connectivity=connectivity,
        matching_class_policy=matching_class_policy,
    )


def binary_mask_components(
    mask: np.ndarray,
    *,
    image_id: str,
    relative_path: str,
    image_path: Path,
    prefix: str,
    class_id: int = 1,
) -> tuple[ObjectComponent, ...]:
    foreground = np.asarray(mask, dtype=bool)
    if foreground.ndim != 2:
        raise ValueError("Object component masks must be two-dimensional")
    labels, count = ndimage.label(
        foreground,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    objects = ndimage.find_objects(labels, max_label=count)
    components: list[ObjectComponent] = []
    for label_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        y_slice, x_slice = slices
        cropped = labels[y_slice, x_slice] == label_id
        area = int(np.sum(cropped))
        if area <= 0:
            continue
        bbox = (
            int(x_slice.start),
            int(y_slice.start),
            int(x_slice.stop),
            int(y_slice.stop),
        )
        components.append(
            ObjectComponent(
                component_id=f"{image_id}:{prefix}:{label_id - 1}",
                image_id=image_id,
                relative_path=relative_path,
                image_path=Path(image_path),
                class_id=int(class_id),
                bbox=bbox,
                mask=np.ascontiguousarray(cropped),
                area=area,
            )
        )
    return tuple(components)


def polygon_components(
    width: int,
    height: int,
    objects: Any,
    *,
    image_id: str,
    relative_path: str,
    image_path: Path,
    prefix: str,
    strict: bool,
) -> tuple[ObjectComponent, ...]:
    components: list[ObjectComponent] = []
    for index, item in enumerate(objects):
        polygon = _object_value(item, "polygon")
        polygons = _object_value(item, "polygons") or (
            [polygon] if polygon else []
        )
        valid_polygons: list[list[tuple[float, float]]] = []
        for points in polygons:
            valid = len(points) >= 3 and all(
                math.isfinite(float(x)) and math.isfinite(float(y))
                for x, y in points
            )
            if not valid:
                if strict:
                    raise ValueError(
                        f"Invalid segmentation polygon in {relative_path}: object {index}"
                    )
                continue
            valid_polygons.append(
                [(float(x), float(y)) for x, y in points]
            )
        if not valid_polygons:
            continue
        xs = [x for points in valid_polygons for x, _ in points]
        ys = [y for points in valid_polygons for _, y in points]
        left = max(0, int(math.floor(min(xs))))
        top = max(0, int(math.floor(min(ys))))
        right = min(int(width), int(math.ceil(max(xs))) + 1)
        bottom = min(int(height), int(math.ceil(max(ys))) + 1)
        if right <= left or bottom <= top:
            continue
        canvas = Image.new("L", (right - left, bottom - top), 0)
        draw = ImageDraw.Draw(canvas)
        for points in valid_polygons:
            draw.polygon([(x - left, y - top) for x, y in points], fill=1)
        mask = np.asarray(canvas, dtype=np.uint8) > 0
        area = int(np.sum(mask))
        if area <= 0:
            continue
        components.append(
            ObjectComponent(
                component_id=f"{image_id}:{prefix}:{index}",
                image_id=image_id,
                relative_path=relative_path,
                image_path=Path(image_path),
                class_id=int(_object_value(item, "class_id", 0)),
                bbox=(left, top, right, bottom),
                mask=np.ascontiguousarray(mask),
                area=area,
            )
        )
    return tuple(components)


def _object_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def semantic_components_for_cases(
    cases: Any,
    *,
    prediction_directory: Path | None = None,
    prefix: str,
) -> dict[str, tuple[ObjectComponent, ...]]:
    """Extract binary 8-connected objects from final full-image masks."""

    components: dict[str, tuple[ObjectComponent, ...]] = {}
    for case in cases:
        mask_path = (
            Path(case.mask_path)
            if prediction_directory is None
            else Path(prediction_directory) / f"{case.case_id}.png"
        )
        with Image.open(mask_path) as opened:
            mask = np.asarray(opened.convert("L")) > 0
        components[str(case.case_id)] = binary_mask_components(
            mask,
            image_id=str(case.case_id),
            relative_path=Path(case.relative_path).as_posix(),
            image_path=Path(case.image_path),
            prefix=prefix,
            class_id=1,
        )
    return components


def component_dice(left: ObjectComponent, right: ObjectComponent) -> float:
    if left.class_id != right.class_id:
        return 0.0
    lx1, ly1, lx2, ly2 = left.bbox
    rx1, ry1, rx2, ry2 = right.bbox
    x1, y1 = max(lx1, rx1), max(ly1, ry1)
    x2, y2 = min(lx2, rx2), min(ly2, ry2)
    intersection = 0
    if x2 > x1 and y2 > y1:
        left_crop = left.mask[y1 - ly1 : y2 - ly1, x1 - lx1 : x2 - lx1]
        right_crop = right.mask[y1 - ry1 : y2 - ry1, x1 - rx1 : x2 - rx1]
        intersection = int(np.sum(left_crop & right_crop))
    denominator = left.area + right.area
    return 2 * intersection / denominator if denominator else 0.0


def match_components(
    reference: tuple[ObjectComponent, ...],
    prediction: tuple[ObjectComponent, ...],
) -> tuple[
    tuple[tuple[int, int, float], ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    if not reference or not prediction:
        return (), tuple(range(len(reference))), tuple(range(len(prediction)))
    scores = np.zeros((len(reference), len(prediction)), dtype=float)
    for ref_index, ref in enumerate(reference):
        for pred_index, pred in enumerate(prediction):
            scores[ref_index, pred_index] = component_dice(ref, pred)
    rows, columns = linear_sum_assignment(-scores)
    matches = tuple(
        (int(ref_index), int(pred_index), float(scores[ref_index, pred_index]))
        for ref_index, pred_index in zip(rows, columns)
        if scores[ref_index, pred_index] > 0
    )
    matched_reference = {row[0] for row in matches}
    matched_prediction = {row[1] for row in matches}
    return (
        matches,
        tuple(index for index in range(len(reference)) if index not in matched_reference),
        tuple(index for index in range(len(prediction)) if index not in matched_prediction),
    )


def evaluate_object_size_model(
    reference: ObjectSizeReference,
    predictions: dict[str, tuple[ObjectComponent, ...]],
) -> ObjectSizeModelResult:
    if reference.status != "complete":
        return ObjectSizeModelResult(
            summary=unavailable_object_size_summary(),
            reference_scores={},
            matched_prediction_ids={},
        )
    scores = {group: [] for group in SIZE_GROUPS}
    reference_counts = {group: 0 for group in SIZE_GROUPS}
    prediction_counts = {group: 0 for group in SIZE_GROUPS}
    match_counts = {group: 0 for group in SIZE_GROUPS}
    reference_scores: dict[str, float] = {}
    matched_prediction_ids: dict[str, str] = {}
    image_ids = sorted(set(reference.components) | set(predictions))
    for image_id in image_ids:
        truth = reference.components.get(image_id, ())
        predicted = predictions.get(image_id, ())
        for component in truth:
            reference_counts[reference.group(component.area)] += 1
        for component in predicted:
            prediction_counts[reference.group(component.area)] += 1
        matches, unmatched_truth, unmatched_prediction = match_components(
            truth, predicted
        )
        for truth_index, prediction_index, dice in matches:
            component = truth[truth_index]
            group = reference.group(component.area)
            scores[group].append(dice)
            match_counts[group] += 1
            reference_scores[component.component_id] = dice
            matched_prediction_ids[component.component_id] = predicted[
                prediction_index
            ].component_id
        for truth_index in unmatched_truth:
            component = truth[truth_index]
            scores[reference.group(component.area)].append(0.0)
            reference_scores[component.component_id] = 0.0
        for prediction_index in unmatched_prediction:
            component = predicted[prediction_index]
            scores[reference.group(component.area)].append(0.0)
    summary: dict[str, Any] = {}
    for group in SIZE_GROUPS:
        values = scores[group]
        summary.update(
            {
                f"{group}_object_dice": (
                    float(np.mean(values))
                    if reference_counts[group] and values
                    else math.nan
                ),
                f"{group}_object_reference_count": reference_counts[group],
                f"{group}_object_prediction_count": prediction_counts[group],
                f"{group}_object_match_count": match_counts[group],
                f"{group}_object_scoring_count": len(values),
            }
        )
    return ObjectSizeModelResult(
        summary=summary,
        reference_scores=reference_scores,
        matched_prediction_ids=matched_prediction_ids,
    )


def select_large_examples(
    reference: ObjectSizeReference,
    results: Mapping[str, ObjectSizeModelResult],
    *,
    limit: int = 4,
) -> list[dict[str, Any]]:
    if reference.status != "complete":
        return []
    large = [
        component
        for component in reference.all_components
        if reference.group(component.area) == "large"
    ]
    if not large:
        return []

    def mean_score(component: ObjectComponent) -> float:
        values = [
            result.reference_scores.get(component.component_id, 0.0)
            for result in results.values()
        ]
        return float(np.mean(values)) if values else math.nan

    largest = sorted(
        large,
        key=lambda component: (-component.area, component.component_id),
    )
    worst = sorted(
        large,
        key=lambda component: (mean_score(component), component.component_id),
    )
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add(component: ObjectComponent, reason: str) -> None:
        if component.component_id in selected_ids or len(selected) >= limit:
            return
        selected_ids.add(component.component_id)
        selected.append(
            {
                "component": component,
                "selection_reason": reason,
                "mean_matched_dice": mean_score(component),
            }
        )

    for component in largest[:2]:
        add(component, "largest-reference-area")
    for component in worst:
        if len([row for row in selected if row["selection_reason"] == "worst-mean-matched-dice"]) >= 2:
            break
        add(component, "worst-mean-matched-dice")
    for component in [*largest, *worst]:
        add(component, "large-reference-fill")
        if len(selected) >= limit:
            break
    return selected


def render_object_size_breakdown(
    reports: Path,
    ranking: list[dict[str, Any]],
    analysis: ObjectSizeReference,
    *,
    labels: Mapping[str, str] | None = None,
) -> Path | None:
    if analysis.status != "complete":
        return None
    import matplotlib.pyplot as plt

    values = np.asarray(
        [
            [float(row.get(f"{group}_object_dice", math.nan)) for group in SIZE_GROUPS]
            for row in ranking
        ],
        dtype=float,
    )
    masked = np.ma.masked_invalid(values)
    figure, axis = plt.subplots(
        figsize=(9.2, max(3.8, 0.75 * len(ranking) + 2.2))
    )
    colormap = plt.get_cmap("viridis").with_extremes(bad="#D9D9D9")
    image = axis.imshow(masked, vmin=0, vmax=1, cmap=colormap, aspect="auto")
    support = analysis.metadata()["reference_support"]
    column_labels = [
        f"{group.title()}-object Dice\nreference n={support[group]}"
        for group in SIZE_GROUPS
    ]
    axis.set_xticks(np.arange(len(SIZE_GROUPS)), column_labels, fontsize=9)
    axis.set_yticks(
        np.arange(len(ranking)),
        [
            labels.get(str(row["model"]), str(row["model"]))
            if labels is not None
            else str(row["model"])
            for row in ranking
        ],
        fontsize=8.5,
    )
    axis.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            label = f"{value:.3f}" if math.isfinite(value) else "n/a"
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color="white" if math.isfinite(value) and value < 0.65 else "black",
                fontsize=8.5,
            )
    axis.set_title(
        "Object Dice by reference foreground area\n"
        f"small ≤ p10 ({analysis.p10_area:.1f} px²), "
        f"medium p10–p90, large ≥ p90 ({analysis.p90_area:.1f} px²)",
        pad=48,
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.025)
    colorbar.set_label("Macro object Dice (unmatched objects = 0)")
    figure.tight_layout()
    path = reports / "object-size-breakdown.png"
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def render_segmentation_metric_breakdown(
    reports: Path,
    ranking: list[dict[str, Any]],
    *,
    title: str = "Segmentation metric breakdown — final reconstructed source images",
    labels: Mapping[str, str] | None = None,
    minimum_component_area: float | None = None,
) -> Path:
    """Render pixel metrics and raw/area-filtered image-presence metrics."""

    import matplotlib.pyplot as plt

    columns = (
        ("dice", "Mean Dice"),
        ("micro_dice", "Pooled foreground\nDice"),
        ("foreground_precision", "Foreground\nprecision"),
        ("foreground_recall", "Foreground\nrecall"),
        ("raw_presence_precision", "Presence precision\nraw"),
        (
            "component_filtered_presence_precision",
            "Presence precision\narea-filtered",
        ),
        ("raw_positive_image_recall", "Positive recall\nraw"),
        (
            "component_filtered_positive_image_recall",
            "Positive recall\narea-filtered",
        ),
        ("raw_empty_image_specificity", "Empty specificity\nraw"),
        (
            "component_filtered_empty_image_specificity",
            "Empty specificity\narea-filtered",
        ),
    )
    values = np.asarray(
        [
            [float(row.get(key, math.nan)) for key, _ in columns]
            for row in ranking
        ],
        dtype=float,
    )
    masked = np.ma.masked_invalid(values)
    figure, axis = plt.subplots(
        figsize=(20.5, max(4.2, 0.85 * len(ranking) + 2.3))
    )
    colormap = plt.get_cmap("viridis").with_extremes(bad="#D9D9D9")
    image = axis.imshow(masked, vmin=0, vmax=1, cmap=colormap, aspect="auto")
    axis.set_xticks(
        np.arange(len(columns)),
        [label for _, label in columns],
        fontsize=9,
    )
    axis.set_yticks(
        np.arange(len(ranking)),
        [
            labels.get(str(row["model"]), str(row["model"]))
            if labels is not None
            else str(row["model"])
            for row in ranking
        ],
        fontsize=8.5,
    )
    axis.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            label = f"{value:.3f}" if math.isfinite(value) else "n/a"
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color="white" if math.isfinite(value) and value < 0.65 else "black",
                fontsize=8.5,
            )
    threshold_note = (
        f"\nArea-filtered presence requires an 8-connected component ≥ "
        f"{minimum_component_area:.1f} px²; raw presence requires any foreground pixel"
        if minimum_component_area is not None
        else "\nArea-filtered presence threshold unavailable; raw presence requires any foreground pixel"
    )
    axis.set_title(title + threshold_note, pad=58)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.03, pad=0.025)
    colorbar.set_label("Higher is better")
    figure.tight_layout()
    path = reports / "metric-breakdown.png"
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def render_grouped_metric_breakdown(
    reports: Path,
    ranking: list[dict[str, Any]],
    grouped_by_model: Mapping[str, Mapping[str, Any]],
    *,
    labels: Mapping[str, str] | None = None,
) -> Path:
    """Render per-group pooled Dice without changing the primary ranking."""

    import matplotlib.pyplot as plt

    groups = sorted(
        {
            str(group["group"])
            for result in grouped_by_model.values()
            for group in result.get("per_group", [])
        }
    )
    columns = ["Macro", *groups]
    values: list[list[float]] = []
    for row in ranking:
        result = grouped_by_model[str(row["model"])]
        lookup = {
            str(group["group"]): float(group["dice"])
            for group in result.get("per_group", [])
        }
        values.append(
            [float(result.get("group_macro_dice", math.nan))]
            + [lookup.get(group, math.nan) for group in groups]
        )
    array = np.asarray(values, dtype=float)
    masked = np.ma.masked_invalid(array)
    figure, axis = plt.subplots(
        figsize=(max(10.5, 4.5 + 0.58 * len(columns)), max(4.2, 0.85 * len(ranking) + 2.6))
    )
    colormap = plt.get_cmap("viridis").with_extremes(bad="#D9D9D9")
    image = axis.imshow(masked, vmin=0, vmax=1, cmap=colormap, aspect="auto")
    axis.set_xticks(np.arange(len(columns)), columns, fontsize=8.5, rotation=60, ha="left")
    axis.set_yticks(
        np.arange(len(ranking)),
        [
            labels.get(str(row["model"]), str(row["model"]))
            if labels is not None
            else str(row["model"])
            for row in ranking
        ],
        fontsize=8.5,
    )
    axis.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
    for row_index in range(array.shape[0]):
        for column_index in range(array.shape[1]):
            value = array[row_index, column_index]
            label = f"{value:.3f}" if math.isfinite(value) else "n/a"
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color="white" if math.isfinite(value) and value < 0.65 else "black",
                fontsize=7.5,
            )
    axis.set_title(
        "Grouped foreground Dice — TP/FP/FN pooled within each group\n"
        "Macro gives every defined group equal weight; primary ranking is unchanged",
        pad=72,
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("Foreground Dice")
    figure.tight_layout()
    path = reports / "grouped-metric-breakdown.png"
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def render_large_object_examples(
    reports: Path,
    selections: list[dict[str, Any]],
    predictions: Mapping[str, dict[str, tuple[ObjectComponent, ...]]],
    results: Mapping[str, ObjectSizeModelResult],
    backends: Mapping[str, str],
) -> list[dict[str, Any]]:
    if not selections:
        return []
    import matplotlib.pyplot as plt

    output_root = reports / "large-object-examples"
    output_root.mkdir(parents=True, exist_ok=True)
    model_names = list(predictions)
    rendered: list[dict[str, Any]] = []
    for selection_index, selection in enumerate(selections, start=1):
        component: ObjectComponent = selection["component"]
        left, top, right, bottom = component.bbox
        edge = max(right - left, bottom - top)
        padding = min(128, max(32, int(round(edge * 0.25))))
        with Image.open(component.image_path) as opened:
            image = opened.convert("RGB")
            crop_box = (
                max(0, left - padding),
                max(0, top - padding),
                min(image.width, right + padding),
                min(image.height, bottom + padding),
            )
            crop = np.asarray(image.crop(crop_box))
        rows = 1 + math.ceil(len(model_names) / 2)
        figure, axes = plt.subplots(
            rows,
            2,
            figsize=(10, 4.6 * rows),
            squeeze=False,
        )
        _show_component_panel(axes[0, 0], crop, (), crop_box, "Source crop")
        _show_component_panel(
            axes[0, 1], crop, (component,), crop_box, "Selected ground truth"
        )
        for model_index, model_name in enumerate(model_names):
            row = 1 + model_index // 2
            column = model_index % 2
            score = results[model_name].reference_scores.get(
                component.component_id, 0.0
            )
            values = predictions[model_name].get(component.image_id, ())
            display_name = textwrap.fill(str(model_name), width=48)
            _show_component_panel(
                axes[row, column],
                crop,
                values,
                crop_box,
                f"{display_name}\n"
                f"{backends.get(model_name, 'unknown')} · matched Dice {score:.3f}",
            )
        for model_index in range(len(model_names), (rows - 1) * 2):
            row = 1 + model_index // 2
            column = model_index % 2
            axes[row, column].axis("off")
        figure.suptitle(
            f"Large reference object · {component.area:,} px² · "
            f"{selection['selection_reason']}",
            fontsize=13,
        )
        figure.tight_layout()
        stem = _safe_stem(Path(component.relative_path).stem)
        path = output_root / f"{selection_index:02d}-{stem}.png"
        figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        rendered.append(
            {
                "path": str(path.relative_to(reports.parent)),
                "component_id": component.component_id,
                "image_id": component.image_id,
                "relative_path": component.relative_path,
                "bbox": list(component.bbox),
                "area_px": component.area,
                "selection_reason": selection["selection_reason"],
                "mean_matched_dice": selection["mean_matched_dice"],
            }
        )
    return rendered


def _show_component_panel(
    axis: Any,
    image: np.ndarray,
    components: Any,
    crop_box: tuple[int, int, int, int],
    title: str,
) -> None:
    mask = np.zeros(image.shape[:2], dtype=bool)
    crop_left, crop_top, crop_right, crop_bottom = crop_box
    for component in components:
        left, top, right, bottom = component.bbox
        x1, y1 = max(left, crop_left), max(top, crop_top)
        x2, y2 = min(right, crop_right), min(bottom, crop_bottom)
        if x2 <= x1 or y2 <= y1:
            continue
        source = component.mask[
            y1 - top : y2 - top,
            x1 - left : x2 - left,
        ]
        destination = mask[
            y1 - crop_top : y2 - crop_top,
            x1 - crop_left : x2 - crop_left,
        ]
        destination |= source
    axis.imshow(image)
    if np.any(mask):
        overlay = np.zeros((*mask.shape, 4), dtype=float)
        overlay[mask] = (1.0, 0.25, 0.1, 0.42)
        axis.imshow(overlay)
        axis.contour(mask.astype(float), levels=[0.5], colors=["white"], linewidths=1)
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def _safe_stem(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return normalized[:96] or "object"
