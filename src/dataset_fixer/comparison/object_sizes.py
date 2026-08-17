from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from ..static_rendering import save_chart
from ..tabular import chart_data
from ..utils import bounded_slug
from ..visualization import (
    VisualizationItem,
    VisualizationOptions,
    VisualizationPanel,
    draw_mask_outline,
    visualize_records,
)
from .plot_labels import model_identity_card, model_identity_chart, model_identity_row_height
from .metrics import component_filtered_presence_breakdown, component_filtered_presence_decisions


SIZE_GROUPS = ("small", "medium", "large")

def _group_split_values(
    group: str,
    group_splits: Mapping[str, Iterable[str]] | None,
) -> tuple[str, ...]:
    if not group_splits or group not in group_splits:
        return ()
    raw = group_splits[group]
    values = (raw,) if isinstance(raw, str) else tuple(raw)
    aliases = {"valid": "val", "validation": "val"}
    normalized = {aliases.get(str(value).lower(), str(value).lower()) for value in values}
    return tuple(
        sorted(
            normalized,
            key=lambda value: (
                {"train": 0, "val": 1, "test": 2}.get(value, 3),
                value,
            ),
        )
    )


def _heatmap_chart(
    rows: list[dict[str, Any]],
    *,
    x_order: list[str],
    models: list[dict[str, Any]],
    title: str,
    legend_title: str,
) -> Any:
    """Render every report matrix through one declarative heatmap builder."""

    import altair as alt

    data = chart_data(rows)
    row_height = model_identity_row_height(models)
    y_order = [str(index) for index in range(len(models))]
    base = alt.Chart(data).encode(
        x=alt.X("column:N", sort=x_order, title=None, axis=alt.Axis(labelAngle=-38, labelLimit=220)),
        y=alt.Y("model_key:N", sort=y_order, title=None, axis=None),
    )
    rectangles = base.mark_rect().encode(
        color=alt.condition(
            "datum.display_value == null",
            alt.value("#D9D9D9"),
            alt.Color("display_value:Q", scale=alt.Scale(domain=[0, 1], scheme="viridis"), title=legend_title),
        ),
        tooltip=["model:N", "column:N", "label:N"],
    )
    labels = base.mark_text(fontSize=11).encode(
        text="label:N",
        color=alt.condition("datum.display_value < 0.65", alt.value("white"), alt.value("black")),
    )
    heatmap = (rectangles + labels).properties(
        width=max(420, 94 * len(x_order)),
        height=alt.Step(row_height),
        title=alt.TitleParams(text=title.splitlines()[0], subtitle=title.splitlines()[1:]),
    )
    return alt.hconcat(
        model_identity_chart(models, row_height=row_height),
        heatmap,
        spacing=14,
    )


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
    grouped_analysis = manifest.get("grouped_analysis")
    if isinstance(grouped_analysis, Mapping):
        grouped_presence = grouped_analysis.get("presence")
        if (
            isinstance(grouped_presence, Mapping)
            and grouped_presence.get("status") == "complete"
            and not all(
                isinstance(reports.get(key), str)
                for key in (
                    "grouped_presence_precision",
                    "grouped_presence_recall",
                    "grouped_presence_f1",
                )
            )
        ):
            return False
    paths: list[str] = []
    for key in (
        "metric_breakdown",
        "grouped_metric_breakdown",
        "grouped_presence_precision",
        "grouped_presence_recall",
        "grouped_presence_f1",
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


def apply_component_presence(
    reference: ObjectSizeReference,
    ranking: list[dict[str, Any]],
    rows_by_model: Mapping[str, list[dict[str, Any]]],
    component_areas: Mapping[str, Mapping[str, list[float]]],
    requested_area: float | None,
) -> dict[str, Any]:
    """Attach one canonical connected-component presence analysis."""

    resolved = requested_area if requested_area is not None else reference.p10_area
    analysis: dict[str, Any] = {
        "raw_definition": "any predicted foreground pixel",
        "component_filtered_definition": (
            "at least one predicted 8-connected foreground component with "
            "area greater than or equal to the resolved threshold"
        ),
        "connectivity": 8,
        "requested_min_connected_component_area_px": requested_area,
        "resolved_min_connected_component_area_px": resolved,
        "threshold_source": "explicit" if requested_area is not None else "held-out-reference-object-p10",
    }
    if resolved is None:
        analysis.update(
            status="skipped",
            reason=(
                "minimum connected-component area is unavailable because the "
                "held-out cohort has no reference foreground objects"
            ),
        )
        return analysis
    analysis["status"] = "complete"
    for row in ranking:
        name = str(row["model"])
        rows, areas = rows_by_model[name], component_areas[name]
        row.update(component_filtered_presence_breakdown(rows, areas, resolved))
        decisions = component_filtered_presence_decisions(rows, areas, resolved)
        for metric in rows:
            case_id = str(metric.get("case_id", metric.get("image_id")))
            metric["component_filtered_predicted_presence"] = decisions[case_id]
    return analysis


def complete_object_size_analysis(
    reference: ObjectSizeReference,
    predictions: Mapping[str, Mapping[str, tuple[ObjectComponent, ...]]],
    ranking: list[dict[str, Any]],
    reports: Path,
    *,
    backends: Mapping[str, str],
    progress: bool = False,
) -> tuple[dict[str, Any], str | None, list[dict[str, Any]]]:
    """Evaluate and render one canonical object-size report."""

    if reference.status != "complete":
        for row in ranking:
            row.update(unavailable_object_size_summary())
        return reference.metadata(), None, []
    results = {
        name: evaluate_object_size_model(reference, values)
        for name, values in tqdm(
            predictions.items(),
            total=len(predictions),
            desc="Scoring object sizes",
            unit="model",
            disable=not progress,
            dynamic_ncols=True,
        )
    }
    for row in ranking:
        row.update(results[str(row["model"])].summary)
    path = render_object_size_breakdown(reports, ranking, reference)
    examples = render_large_object_examples(
        reports,
        select_large_examples(reference, results),
        predictions,
        results,
        backends,
        {str(row["model"]): row for row in ranking},
    )
    return reference.metadata(), str(path.relative_to(reports.parent)) if path else None, examples


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
    progress: bool = False,
    progress_description: str = "Extracting object components",
) -> dict[str, tuple[ObjectComponent, ...]]:
    """Extract binary 8-connected objects from final full-image masks."""

    components: dict[str, tuple[ObjectComponent, ...]] = {}
    for case in tqdm(
        cases,
        desc=progress_description,
        unit="mask",
        disable=not progress,
        dynamic_ncols=True,
    ):
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
) -> Path | None:
    if analysis.status != "complete":
        return None
    support = analysis.metadata()["reference_support"]
    column_labels = [group.title() for group in SIZE_GROUPS]
    cells = [
        {
            "model": str(row["model"]),
            "model_key": str(row_index),
            "column": column_labels[index],
            "display_value": value if math.isfinite(value) else None,
            "label": f"{value:.3f}" if math.isfinite(value) else "n/a",
        }
        for row_index, row in enumerate(ranking)
        for index, group in enumerate(SIZE_GROUPS)
        for value in [float(row.get(f"{group}_object_dice", math.nan))]
    ]
    chart = _heatmap_chart(
        cells,
        x_order=column_labels,
        models=ranking,
        title=(
        "Object Dice by foreground area "
        f"(small {support['small']}, medium {support['medium']}, large {support['large']})\n"
        f"small ≤ p10 ({analysis.p10_area:.1f} px²), medium p10–p90, "
        f"large ≥ p90 ({analysis.p90_area:.1f} px²)"
        ),
        legend_title="Macro object Dice",
    )
    path = reports / "object-size-breakdown.png"
    save_chart(chart, path)
    return path


def render_segmentation_metric_breakdown(
    reports: Path,
    ranking: list[dict[str, Any]],
    *,
    title: str = "Segmentation metric breakdown — final reconstructed source images",
    minimum_component_area: float | None = None,
) -> Path:
    """Render pixel metrics and raw/area-filtered image-presence metrics."""

    columns = (
        ("dice", "Mean Dice"),
        ("micro_dice", "Pooled foreground\nDice"),
        ("foreground_precision", "Pixel\nprecision"),
        ("foreground_recall", "Pixel\nrecall"),
        ("raw_presence_precision", "Presence precision\nraw"),
        (
            "component_filtered_presence_precision",
            "Presence precision\narea-filtered",
        ),
        ("raw_positive_image_recall", "Positive-image recall\nraw"),
        (
            "component_filtered_positive_image_recall",
            "Positive-image recall\narea-filtered",
        ),
        ("raw_empty_image_specificity", "Empty specificity\nraw"),
        (
            "component_filtered_empty_image_specificity",
            "Empty specificity\narea-filtered",
        ),
    )
    column_labels = [label for _, label in columns]
    cells = [
        {
            "model": str(row["model"]),
            "model_key": str(row_index),
            "column": label,
            "display_value": value if math.isfinite(value) else None,
            "label": f"{value:.3f}" if math.isfinite(value) else "n/a",
        }
        for row_index, row in enumerate(ranking)
        for key, label in columns
        for value in [float(row.get(key, math.nan))]
    ]
    threshold_note = (
        f"\nArea-filtered presence requires an 8-connected component ≥ "
        f"{minimum_component_area:.1f} px²; raw presence requires any foreground pixel"
        if minimum_component_area is not None
        else "\nArea-filtered presence threshold unavailable; raw presence requires any foreground pixel"
    )
    chart = _heatmap_chart(
        cells,
        x_order=column_labels,
        models=ranking,
        title=title + threshold_note,
        legend_title="Higher is better",
    )
    path = reports / "metric-breakdown.png"
    save_chart(chart, path)
    return path


def render_grouped_metric_breakdown(
    reports: Path,
    ranking: list[dict[str, Any]],
    grouped_by_model: Mapping[str, Mapping[str, Any]],
    *,
    group_splits: Mapping[str, Iterable[str]] | None = None,
) -> Path:
    """Render per-group pooled Dice, ordered by equal-weight group macro Dice."""

    groups = sorted(
        {
            str(group["group"])
            for result in grouped_by_model.values()
            for group in result.get("per_group", [])
        }
    )
    group_labels = [
        f"{group}\n[{'/'.join(_group_split_values(group, group_splits))}]"
        if _group_split_values(group, group_splits)
        else group
        for group in groups
    ]
    columns = ["Macro", *group_labels]

    def macro_sort_key(row: Mapping[str, Any]) -> tuple[bool, float, str]:
        value = float(
            grouped_by_model[str(row["model"])].get("group_macro_dice", math.nan)
        )
        return (
            not math.isfinite(value),
            -value if math.isfinite(value) else 0.0,
            str(row["model"]),
        )

    ordered_ranking = sorted(
        ranking,
        key=macro_sort_key,
    )
    values: list[list[float]] = []
    for row in ordered_ranking:
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
    cells: list[dict[str, Any]] = []
    for row_index in range(array.shape[0]):
        for column_index in range(array.shape[1]):
            value = array[row_index, column_index]
            if column_index == 0 and math.isfinite(value):
                result = grouped_by_model[
                    str(ordered_ranking[row_index]["model"])
                ]
                defined = int(result.get("group_defined_dice_count", 0))
                total = int(result.get("group_count", len(groups)))
                label = f"{value:.3f}\n({defined}/{total})"
            elif column_index > 0 and not math.isfinite(value):
                label = "TN"
                display_value: float | None = 1.0
            else:
                label = f"{value:.3f}" if math.isfinite(value) else "n/a"
                display_value = value if math.isfinite(value) else None
            if column_index == 0:
                display_value = value if math.isfinite(value) else None
            cells.append(
                {
                    "model": str(ordered_ranking[row_index]["model"]),
                    "model_key": str(row_index),
                    "column": columns[column_index],
                    "display_value": display_value,
                    "label": label,
                }
            )
    chart = _heatmap_chart(
        cells,
        x_order=columns,
        models=ordered_ranking,
        title=(
        "Grouped foreground Dice — TP/FP/FN pooled within each group\n"
        "Rows are sorted by macro Dice; support is defined groups / all groups\n"
        "0 = no overlap; green TN = empty reference and prediction (excluded from macro)"
        ),
        legend_title="Foreground Dice (TN = 1)",
    )
    path = reports / "grouped-metric-breakdown.png"
    save_chart(chart, path)
    return path


def render_grouped_presence_metric_breakdown(
    reports: Path,
    ranking: list[dict[str, Any]],
    grouped_by_model: Mapping[str, Mapping[str, Any]],
    *,
    metric: str,
    group_splits: Mapping[str, Iterable[str]] | None = None,
) -> Path:
    """Render an AOI-level area-filtered presence precision, recall, or F1 grid."""

    if metric not in {"precision", "recall", "f1"}:
        raise ValueError("metric must be one of: precision, recall, f1")
    value_key = f"presence_{metric}"
    macro_key = f"group_macro_presence_{metric}"
    defined_key = f"group_defined_presence_{metric}_count"
    groups = sorted(
        {
            str(group["group"])
            for result in grouped_by_model.values()
            for group in result.get("per_group", [])
        }
    )

    def macro_f1_sort_key(row: Mapping[str, Any]) -> tuple[bool, float, str]:
        value = float(
            grouped_by_model[str(row["model"])].get(
                "group_macro_presence_f1", math.nan
            )
        )
        return (
            not math.isfinite(value),
            -value if math.isfinite(value) else 0.0,
            str(row["model"]),
        )

    ordered_ranking = sorted(ranking, key=macro_f1_sort_key)
    group_labels = [
        f"{group}\n[{'/'.join(_group_split_values(group, group_splits))}]"
        if _group_split_values(group, group_splits)
        else group
        for group in groups
    ]
    columns = [f"Macro {metric.upper()}", *group_labels]
    values: list[list[float]] = []
    group_rows: list[dict[str, Mapping[str, Any]]] = []
    for row in ordered_ranking:
        result = grouped_by_model[str(row["model"])]
        lookup = {
            str(group["group"]): group
            for group in result.get("per_group", [])
        }
        group_rows.append(lookup)
        values.append(
            [float(result.get(macro_key, math.nan))]
            + [float(lookup[group].get(value_key, math.nan)) for group in groups]
        )

    array = np.asarray(values, dtype=float)
    display_array = array.copy()
    display_labels: list[list[str]] = []
    for row_index, row in enumerate(ordered_ranking):
        result = grouped_by_model[str(row["model"])]
        macro_value = array[row_index, 0]
        if math.isfinite(macro_value):
            defined = int(result.get(defined_key, 0))
            total = int(result.get("group_count", len(groups)))
            macro_label = f"{macro_value:.3f}\n({defined}/{total})"
        else:
            macro_label = "n/a"
        labels_for_row = [macro_label]
        for column_index, group in enumerate(groups, start=1):
            group_row = group_rows[row_index][group]
            value = array[row_index, column_index]
            positive_cases = int(group_row.get("positive_cases", 0))
            false_positives = int(group_row.get("presence_fp", 0))
            predicted_positives = int(group_row.get("presence_tp", 0)) + false_positives
            if math.isfinite(value):
                label = f"{value:.3f}"
            elif positive_cases == 0 and false_positives == 0:
                display_array[row_index, column_index] = 1.0
                label = "TN"
            elif metric == "precision" and positive_cases > 0 and predicted_positives == 0:
                display_array[row_index, column_index] = 0.0
                label = "MISS"
            elif metric == "recall" and positive_cases == 0 and false_positives > 0:
                display_array[row_index, column_index] = 0.0
                label = "FP"
            else:
                label = "n/a"
            labels_for_row.append(label)
        display_labels.append(labels_for_row)

    cells: list[dict[str, Any]] = []
    for row_index in range(array.shape[0]):
        for column_index in range(array.shape[1]):
            display_value = display_array[row_index, column_index]
            cells.append(
                {
                    "model": str(ordered_ranking[row_index]["model"]),
                    "model_key": str(row_index),
                    "column": columns[column_index],
                    "display_value": display_value if math.isfinite(display_value) else None,
                    "label": display_labels[row_index][column_index],
                }
            )
    chart = _heatmap_chart(
        cells,
        x_order=columns,
        models=ordered_ranking,
        title=(
        f"Area-filtered case-presence {metric} pooled within each AOI\n"
        "Rows are sorted by macro presence F1; macro support is defined AOIs / all AOIs\n"
        "Green TN = empty reference and prediction; MISS/FP = zero display score"
        ),
        legend_title=f"Presence {metric} (TN = 1)",
    )
    path = reports / f"grouped-presence-{metric}.png"
    save_chart(chart, path)
    return path


def render_large_object_examples(
    reports: Path,
    selections: list[dict[str, Any]],
    predictions: Mapping[str, dict[str, tuple[ObjectComponent, ...]]],
    results: Mapping[str, ObjectSizeModelResult],
    backends: Mapping[str, str],
    model_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not selections:
        return []

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
        panels: list[tuple[str, np.ndarray, str | None, Mapping[str, Any] | None]] = [
            ("Source crop", _show_component_panel(crop, (), crop_box), None, None),
            ("Selected ground truth", _show_component_panel(crop, (component,), crop_box), None, None),
        ]
        for model_name in model_names:
            score = results[model_name].reference_scores.get(
                component.component_id, 0.0
            )
            values = predictions[model_name].get(component.image_id, ())
            panels.append(
                (
                    str(model_name),
                    _show_component_panel(crop, values, crop_box),
                    f"{backends.get(model_name, 'unknown')} · matched Dice {score:.3f}",
                    (model_metadata or {}).get(model_name),
                )
            )

        def prepare(
            value: tuple[str, np.ndarray, str | None, Mapping[str, Any] | None]
        ) -> VisualizationItem:
            heading, panel_image, footer, metadata = value
            return VisualizationItem(
                image_path=component.image_path,
                label="",
                panels=(VisualizationPanel(
                    title=heading,
                    image=panel_image,
                    footer=footer,
                    heading=(
                        model_identity_card(metadata, width=403, maximum=48)
                        if metadata is not None
                        else None
                    ),
                ),),
                foreground=np.ones(panel_image.shape[:2], dtype=bool),
            )

        chart = visualize_records(
            panels,
            options=VisualizationOptions(
                samples=None,
                columns=2,
                panel_size=4.2,
                label_mode="wrap",
                show=False,
            ),
            prepare=prepare,
            title=(
                f"Large reference object · {component.area:,} px² · "
                f"{selection['selection_reason']}"
            ),
        )
        stem = bounded_slug(Path(component.relative_path).stem, max_length=96)
        path = output_root / f"{selection_index:02d}-{stem}.png"
        save_chart(chart, path)
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
    image: np.ndarray,
    components: Any,
    crop_box: tuple[int, int, int, int],
) -> np.ndarray:
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
    rendered = np.asarray(image, dtype=np.uint8).copy()
    if not np.any(mask):
        return rendered
    tint = np.asarray((255, 64, 26), dtype=float)
    rendered[mask] = np.uint8(np.round(rendered[mask] * 0.58 + tint * 0.42))
    return draw_mask_outline(
        rendered,
        mask,
        color="#FF401A",
        line_width=2,
        outline_width=4,
        alpha=1.0,
    )
