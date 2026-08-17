from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage
from tqdm import tqdm

from ..errors import DatasetValidationError
from ..tabular import TableLike, frame
from .types import Cohort, Prediction


def evaluate_configuration(
    cohort: Cohort,
    predictions: dict[str, list[Prediction]],
    confidence: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    class_totals: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    all_count_errors: list[float] = []
    relative_count_errors: list[float] = []
    count_agreements: list[float] = []
    distances: list[float] = []
    raw_distances: list[float] = []
    for record in cohort.records:
        pred = [value for value in predictions[record.image_id] if value.score >= confidence]
        gt = list(record.annotations)
        match = optimal_match(gt, pred, cohort.task, cohort.metadata)
        tp, fp, fn = len(match["matches"]), len(match["unmatched_pred"]), len(match["unmatched_gt"])
        row = {
            "image_id": record.image_id,
            "relative_path": record.relative_path,
            "original_id": record.original_id,
            "gt": len(gt),
            "pred": len(pred),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            **_prf(tp, fp, fn),
        }
        if cohort.task == "polo":
            error = len(pred) - len(gt)
            all_count_errors.append(error)
            if len(gt):
                relative_count_errors.append(abs(error) / len(gt))
            count_agreements.append(min(len(pred), len(gt)) / max(len(pred), len(gt)) if max(len(pred), len(gt)) else 1.0)
            local_distances = [item[2].get("distance", math.nan) for item in match["matches"]]
            local_distances = [value for value in local_distances if math.isfinite(value)]
            raw_distances.extend(local_distances)
            local_normalized = [item[2].get("normalized_distance", math.nan) for item in match["matches"]]
            distances.extend(value for value in local_normalized if math.isfinite(value))
            row.update(
                count_error=error,
                count_absolute_error=abs(error),
                localization_error=float(np.mean(local_distances)) if local_distances else math.nan,
                relative_count_error=abs(error) / len(gt) if len(gt) else math.nan,
                count_agreement=min(len(pred), len(gt)) / max(len(pred), len(gt)) if max(len(pred), len(gt)) else 1.0,
            )
        rows.append(row)
        for index, annotation in enumerate(gt):
            class_totals[int(annotation["class_id"])]["gt"] += 1
            if index in {item[0] for item in match["matches"]}:
                class_totals[int(annotation["class_id"])]["tp"] += 1
            else:
                class_totals[int(annotation["class_id"])]["fn"] += 1
        for index, value in enumerate(pred):
            if index not in {item[1] for item in match["matches"]}:
                class_totals[value.class_id]["fp"] += 1

    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    cluster = _cluster_macro(rows)
    ap = compute_ap_metrics(cohort, predictions)
    summary: dict[str, Any] = {
        **_prf(tp, fp, fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support_images": len(rows),
        "support_annotations": tp + fn,
        "support_clusters": len({row["original_id"] for row in rows}),
        "macro_f1": _safe_mean([row["f1"] for row in rows]),
        "cluster_macro_f1": _safe_mean([row["f1"] for row in cluster]),
        **ap["summary"],
    }
    if cohort.task == "polo":
        errors = np.asarray(all_count_errors, dtype=float)
        summary.update(
            count_mae=float(np.mean(np.abs(errors))) if len(errors) else math.nan,
            count_rmse=float(np.sqrt(np.mean(errors**2))) if len(errors) else math.nan,
            count_bias=float(np.mean(errors)) if len(errors) else math.nan,
            count_relative_error=_safe_mean(relative_count_errors),
            count_agreement=_safe_mean(count_agreements),
            localization_distance=_safe_mean(raw_distances),
            radius_normalized_localization_error=_safe_mean(distances),
            notebook_greedy_f1=_notebook_compatibility(cohort, predictions, confidence),
        )
    per_class = []
    for class_id in sorted(cohort.classes):
        totals = class_totals[class_id]
        per_class.append(
            {
                "class_id": class_id,
                "class_name": cohort.classes[class_id],
                "support": int(totals["gt"]),
                "tp": int(totals["tp"]),
                "fp": int(totals["fp"]),
                "fn": int(totals["fn"]),
                **_prf(int(totals["tp"]), int(totals["fp"]), int(totals["fn"])),
                "ap": ap["per_class"].get(class_id, math.nan),
            }
        )
    return {"summary": summary, "per_image": rows, "per_class": per_class, "pr": ap["pr"]}


def binary_metric_breakdown(rows: TableLike) -> dict[str, Any]:
    """Summarize final binary source-image masks without background dominance."""

    data = frame(rows)
    positive = data[data["n_ref"].astype(float).gt(0)]
    empty = data[data["n_ref"].astype(float).eq(0)]
    tp, fp, fn = _confusion_totals(data)
    positive_tp, positive_fp, positive_fn = _confusion_totals(positive)
    positive_detected_cases = int(positive["n_pred"].astype(float).gt(0).sum())
    empty_false_positive_cases = int(empty["n_pred"].astype(float).gt(0).sum())
    empty_false_positive_pixels = int(empty["fp"].sum())
    positive_image_recall = _ratio(positive_detected_cases, len(positive))
    empty_image_specificity = _ratio(
        len(empty) - empty_false_positive_cases, len(empty)
    )
    empty_image_false_positive_rate = _ratio(empty_false_positive_cases, len(empty))
    presence_precision = _ratio(
        positive_detected_cases, positive_detected_cases + empty_false_positive_cases
    )

    return {
        "micro_dice": _ratio(2 * tp, 2 * tp + fp + fn),
        "micro_iou": _ratio(tp, tp + fp + fn),
        "foreground_precision": _ratio(tp, tp + fp),
        "foreground_recall": _ratio(tp, tp + fn),
        "positive_case_dice": _series_mean(positive["dice"]),
        "positive_case_iou": _series_mean(positive["iou"]),
        "positive_micro_dice": _ratio(2 * positive_tp, 2 * positive_tp + positive_fp + positive_fn),
        "positive_micro_iou": _ratio(positive_tp, positive_tp + positive_fp + positive_fn),
        "positive_foreground_precision": _ratio(positive_tp, positive_tp + positive_fp),
        "positive_foreground_recall": _ratio(positive_tp, positive_tp + positive_fn),
        "positive_cases": len(positive),
        "positive_detected_cases": positive_detected_cases,
        "positive_missed_cases": len(positive) - positive_detected_cases,
        "positive_image_recall": positive_image_recall,
        "empty_cases": len(empty),
        "empty_correct_cases": len(empty) - empty_false_positive_cases,
        "empty_false_positive_cases": empty_false_positive_cases,
        "empty_image_specificity": empty_image_specificity,
        "empty_image_false_positive_rate": empty_image_false_positive_rate,
        "empty_false_positive_pixels": empty_false_positive_pixels,
        "empty_mean_false_positive_pixels": _ratio(empty_false_positive_pixels, len(empty)),
        # The historical unsuffixed image-level fields remain the raw
        # any-foreground-pixel metrics. Explicit aliases make comparison with
        # the connected-component-filtered variants unambiguous in reports.
        "presence_precision": presence_precision,
        "raw_positive_image_recall": positive_image_recall,
        "raw_empty_image_specificity": empty_image_specificity,
        "raw_empty_image_false_positive_rate": empty_image_false_positive_rate,
        "raw_presence_precision": presence_precision,
    }


def component_filtered_presence_breakdown(
    rows: TableLike,
    prediction_component_areas: Mapping[str, Sequence[float]],
    minimum_component_area: float,
) -> dict[str, Any]:
    """Score image-level presence after rejecting small predicted components.

    ``rows`` retain the raw, unfiltered masks used for Dice/IoU. This helper
    changes only the binary image-level presence decision: a prediction is
    present when at least one 8-connected foreground component has area
    greater than or equal to ``minimum_component_area``.
    """

    data = frame(rows)
    decisions = component_filtered_presence_decisions(data, prediction_component_areas, minimum_component_area)
    identifiers = _case_ids(data, "Presence rows require a case_id or image_id for component filtering")
    predicted = identifiers.map(decisions).astype(bool)
    positive = data["n_ref"].astype(float).gt(0)
    positive_detected = int((positive & predicted).sum())
    empty_false_positive = int((~positive & predicted).sum())
    positive_cases = int(positive.sum())
    empty_cases = int((~positive).sum())

    return {
        "min_connected_component_area": float(minimum_component_area),
        "component_filtered_positive_detected_cases": positive_detected,
        "component_filtered_positive_missed_cases": positive_cases - positive_detected,
        "component_filtered_positive_image_recall": _ratio(positive_detected, positive_cases),
        "component_filtered_empty_correct_cases": empty_cases - empty_false_positive,
        "component_filtered_empty_false_positive_cases": empty_false_positive,
        "component_filtered_empty_image_specificity": _ratio(empty_cases - empty_false_positive, empty_cases),
        "component_filtered_empty_image_false_positive_rate": _ratio(empty_false_positive, empty_cases),
        "component_filtered_presence_precision": _ratio(positive_detected, positive_detected + empty_false_positive),
    }


def component_filtered_presence_decisions(
    rows: TableLike,
    prediction_component_areas: Mapping[str, Sequence[float]],
    minimum_component_area: float,
) -> dict[str, bool]:
    """Resolve the area-filtered presence decision for every evaluation case."""

    threshold = float(minimum_component_area)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("minimum_component_area must be finite and greater than zero")

    identifiers = _case_ids(
        frame(rows),
        "Presence rows require a case_id or image_id for component filtering",
    )
    missing = sorted(set(identifiers) - set(prediction_component_areas))
    if missing:
        raise ValueError(
            "Prediction component areas are missing evaluation cases: "
            + ", ".join(missing[:5])
        )

    return {
        case_id: any(float(area) >= threshold for area in prediction_component_areas[case_id])
        for case_id in identifiers
    }


def grouped_presence_metric_breakdown(
    rows: TableLike,
    groups: Mapping[str, str],
    predicted_presence: Mapping[str, bool],
) -> dict[str, Any]:
    """Pool image-level presence confusion within groups and macro-average."""

    data = frame(rows)
    identifiers = _case_ids(data, "Grouped presence rows require a case_id or image_id")
    _require_case_mapping(identifiers, groups, "No group was resolved for evaluation case {case_id!r}")
    _require_case_mapping(identifiers, predicted_presence, "No filtered presence decision was resolved for case {case_id!r}")
    reference = data["n_ref"].astype(float).gt(0)
    prediction = identifiers.map(predicted_presence).astype(bool)
    data = data.assign(
        group=identifiers.map(groups).astype(str),
        positive_cases=reference.astype(int),
        empty_cases=(~reference).astype(int),
        presence_tp=(reference & prediction).astype(int),
        presence_fp=(~reference & prediction).astype(int),
        presence_fn=(reference & ~prediction).astype(int),
        presence_tn=(~reference & ~prediction).astype(int),
    )
    per_group = data.groupby("group", sort=True, as_index=False).agg(
        cases=("group", "size"),
        positive_cases=("positive_cases", "sum"),
        empty_cases=("empty_cases", "sum"),
        presence_tp=("presence_tp", "sum"),
        presence_fp=("presence_fp", "sum"),
        presence_fn=("presence_fn", "sum"),
        presence_tn=("presence_tn", "sum"),
    )
    per_group["presence_precision"] = _series_ratio(per_group["presence_tp"], per_group["presence_tp"] + per_group["presence_fp"])
    per_group["presence_recall"] = _series_ratio(per_group["presence_tp"], per_group["presence_tp"] + per_group["presence_fn"])
    per_group["presence_f1"] = _series_ratio(2 * per_group["presence_tp"], 2 * per_group["presence_tp"] + per_group["presence_fp"] + per_group["presence_fn"])

    return {
        "group_count": len(per_group),
        "group_defined_presence_precision_count": int(np.isfinite(per_group["presence_precision"]).sum()),
        "group_defined_presence_recall_count": int(np.isfinite(per_group["presence_recall"]).sum()),
        "group_defined_presence_f1_count": int(np.isfinite(per_group["presence_f1"]).sum()),
        "group_macro_presence_precision": _series_mean(per_group["presence_precision"]),
        "group_macro_presence_recall": _series_mean(per_group["presence_recall"]),
        "group_macro_presence_f1": _series_mean(per_group["presence_f1"]),
        "per_group": per_group.to_dict(orient="records"),
    }


def grouped_binary_metric_breakdown(
    rows: TableLike,
    groups: Mapping[str, str],
) -> dict[str, Any]:
    """Pool pixel confusion within groups, then macro-average across groups."""

    data = frame(rows)
    identifiers = _case_ids(data, "Grouped metric rows require a case_id or image_id")
    _require_case_mapping(identifiers, groups, "No group was resolved for evaluation case {case_id!r}")
    reference = data["n_ref"].astype(float)
    data = data.assign(
        group=identifiers.map(groups).astype(str),
        positive_cases=reference.gt(0).astype(int),
        empty_cases=reference.eq(0).astype(int),
    )
    per_group = data.groupby("group", sort=True, as_index=False).agg(
        cases=("group", "size"),
        positive_cases=("positive_cases", "sum"),
        empty_cases=("empty_cases", "sum"),
        tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"),
    )
    per_group["dice"] = _series_ratio(2 * per_group["tp"], 2 * per_group["tp"] + per_group["fp"] + per_group["fn"])
    per_group["iou"] = _series_ratio(per_group["tp"], per_group["tp"] + per_group["fp"] + per_group["fn"])
    per_group["foreground_precision"] = _series_ratio(per_group["tp"], per_group["tp"] + per_group["fp"])
    per_group["foreground_recall"] = _series_ratio(per_group["tp"], per_group["tp"] + per_group["fn"])

    return {
        "group_count": len(per_group),
        "group_defined_dice_count": int(np.isfinite(per_group["dice"]).sum()),
        "group_macro_dice": _series_mean(per_group["dice"]),
        "group_macro_iou": _series_mean(per_group["iou"]),
        "group_macro_foreground_precision": _series_mean(per_group["foreground_precision"]),
        "group_macro_foreground_recall": _series_mean(per_group["foreground_recall"]),
        "per_group": per_group.to_dict(orient="records"),
    }


def _case_ids(data: pd.DataFrame, error: str) -> pd.Series:
    identifiers = data.get("case_id")
    if identifiers is None:
        identifiers = data.get("image_id")
    elif "image_id" in data:
        identifiers = identifiers.fillna(data["image_id"])
    if identifiers is None or identifiers.isna().any():
        raise ValueError(error)
    return identifiers.astype(str)


def _require_case_mapping(
    identifiers: pd.Series,
    values: Mapping[str, Any],
    error: str,
) -> None:
    missing = identifiers[~identifiers.isin(values)]
    if not missing.empty:
        raise ValueError(error.format(case_id=missing.iloc[0]))


def _confusion_totals(data: pd.DataFrame) -> tuple[int, int, int]:
    totals = data.reindex(columns=["tp", "fp", "fn"], fill_value=0).sum()
    return int(totals["tp"]), int(totals["fp"]), int(totals["fn"])


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def _series_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.astype(float).div(denominator.astype(float).where(denominator.ne(0)))


def _series_mean(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(finite.mean()) if finite.notna().any() else math.nan


def segmentation_binary_metric_breakdown(
    cohort: Cohort,
    predictions: dict[str, list[Prediction]],
    confidence: float,
) -> dict[str, Any]:
    """Score final instance-segmentation polygons as foreground-union masks."""

    rows = segmentation_binary_metric_rows(cohort, predictions, confidence)
    return {
        "dice": _safe_mean([row["dice"] for row in rows]),
        "iou": _safe_mean([row["iou"] for row in rows]),
        **binary_metric_breakdown(rows),
    }


def segmentation_binary_metric_rows(
    cohort: Cohort,
    predictions: dict[str, list[Prediction]],
    confidence: float,
) -> list[dict[str, Any]]:
    """Return per-image binary-mask rows for final instance predictions."""

    if cohort.task != "segment":
        raise ValueError("Binary segmentation breakdown requires a segment cohort")
    rows: list[dict[str, Any]] = []
    for record in cohort.records:
        truth = _polygon_union_mask(
            record.width,
            record.height,
            record.annotations,
            source=f"ground truth {record.relative_path}",
            strict=True,
        )
        selected = [
            prediction
            for prediction in predictions[record.image_id]
            if prediction.score >= confidence
        ]
        prediction = _polygon_union_mask(
            record.width,
            record.height,
            selected,
            source=f"prediction {record.relative_path}",
            strict=False,
        )
        tp = int(np.sum(truth & prediction))
        fp = int(np.sum(~truth & prediction))
        fn = int(np.sum(truth & ~prediction))
        dice_denominator = 2 * tp + fp + fn
        iou_denominator = tp + fp + fn
        rows.append(
            {
                "case_id": record.image_id,
                "image_id": record.image_id,
                "relative_path": record.relative_path,
                "dice": 2 * tp / dice_denominator if dice_denominator else math.nan,
                "iou": tp / iou_denominator if iou_denominator else math.nan,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "n_ref": int(np.sum(truth)),
                "n_pred": int(np.sum(prediction)),
                "prediction_component_areas": _connected_component_areas(
                    prediction
                ),
            }
        )
    return rows


def _connected_component_areas(mask: np.ndarray) -> list[int]:
    labels, count = ndimage.label(
        np.asarray(mask, dtype=bool),
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    if count == 0:
        return []
    counts = np.bincount(labels.ravel(), minlength=count + 1)
    return [int(value) for value in counts[1:]]


def _polygon_union_mask(
    width: int,
    height: int,
    objects: Any,
    *,
    source: str,
    strict: bool,
) -> np.ndarray:
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for item in objects:
        if isinstance(item, Mapping):
            polygon = item.get("polygon")
            polygons = item.get("polygons") or ([polygon] if polygon else [])
        else:
            polygon = item.polygon
            polygons = item.polygons or ([polygon] if polygon else [])
        for points in polygons:
            valid = len(points) >= 3 and all(
                math.isfinite(float(x)) and math.isfinite(float(y))
                for x, y in points
            )
            if not valid:
                if strict:
                    raise DatasetValidationError(
                        f"Cannot rasterize invalid segmentation polygon from {source}"
                    )
                continue
            draw.polygon([(float(x), float(y)) for x, y in points], fill=1)
    return np.asarray(canvas, dtype=np.uint8) > 0


def optimal_match(
    gt: list[dict[str, Any]], predictions: list[Prediction], task: str, metadata: dict[str, Any]
) -> dict[str, Any]:
    if not gt or not predictions:
        return {"matches": [], "unmatched_gt": list(range(len(gt))), "unmatched_pred": list(range(len(predictions)))}
    scores = np.full((len(gt), len(predictions)), -1e6, dtype=float)
    details: dict[tuple[int, int], dict[str, float]] = {}
    for i, annotation in enumerate(gt):
        for j, prediction in enumerate(predictions):
            if int(annotation["class_id"]) != prediction.class_id:
                continue
            value, detail = _similarity(annotation, prediction, task, metadata)
            if _passes(value, task, fixed=True):
                scores[i, j] = value if task != "polo" else -value
                details[(i, j)] = detail
    try:
        from scipy.optimize import linear_sum_assignment

        row_indices, col_indices = linear_sum_assignment(-scores)
        pairs = [(int(i), int(j)) for i, j in zip(row_indices, col_indices) if scores[i, j] > -1e5]
    except ImportError:
        candidates = sorted(
            ((scores[i, j], i, j) for i in range(len(gt)) for j in range(len(predictions)) if scores[i, j] > -1e5),
            reverse=True,
        )
        used_gt: set[int] = set()
        used_pred: set[int] = set()
        pairs = []
        for _, i, j in candidates:
            if i not in used_gt and j not in used_pred:
                used_gt.add(i)
                used_pred.add(j)
                pairs.append((i, j))
    return {
        "matches": [(i, j, details[(i, j)]) for i, j in pairs],
        "unmatched_gt": [i for i in range(len(gt)) if i not in {pair[0] for pair in pairs}],
        "unmatched_pred": [j for j in range(len(predictions)) if j not in {pair[1] for pair in pairs}],
    }


def compute_ap_metrics(cohort: Cohort, predictions: dict[str, list[Prediction]]) -> dict[str, Any]:
    thresholds = [round(value, 2) for value in np.arange(0.5, 0.951, 0.05)]
    if cohort.task == "polo":
        thresholds = [round(value, 1) for value in np.arange(1.0, 0.09, -0.1)]
    per_class_values: dict[int, list[float]] = defaultdict(list)
    pr_primary: dict[int, dict[str, list[float]]] = {}
    all_aps: list[float] = []
    primary_aps: list[float] = []
    for class_id in sorted(cohort.classes):
        gt_by_image = {
            record.image_id: [a for a in record.annotations if int(a["class_id"]) == class_id]
            for record in cohort.records
        }
        count = sum(map(len, gt_by_image.values()))
        if count == 0:
            continue
        model_predictions = sorted(
            (
                (prediction.score, record.image_id, prediction)
                for record in cohort.records
                for prediction in predictions[record.image_id]
                if prediction.class_id == class_id
            ),
            key=lambda value: (-value[0], value[1]),
        )
        for threshold_index, threshold in enumerate(thresholds):
            used: dict[str, set[int]] = defaultdict(set)
            tp: list[int] = []
            fp: list[int] = []
            for _, image_id, prediction in model_predictions:
                choices: list[tuple[float, int]] = []
                for index, annotation in enumerate(gt_by_image[image_id]):
                    if index in used[image_id]:
                        continue
                    value, _ = _similarity(annotation, prediction, cohort.task, cohort.metadata)
                    if _passes(value, cohort.task, threshold=threshold):
                        choices.append((value, index))
                if choices:
                    best = min(choices) if cohort.task == "polo" else max(choices)
                    used[image_id].add(best[1])
                    tp.append(1)
                    fp.append(0)
                else:
                    tp.append(0)
                    fp.append(1)
            recall, precision = _curve(tp, fp, count)
            ap = _interpolated_ap(recall, precision)
            per_class_values[class_id].append(ap)
            all_aps.append(ap)
            if threshold_index == 0:
                primary_aps.append(ap)
                pr_primary[class_id] = {
                    "recall": recall.tolist(),
                    "precision": precision.tolist(),
                    "confidence": [float(value[0]) for value in model_predictions],
                }
    class_mean = {class_id: _safe_mean(values) for class_id, values in per_class_values.items()}
    if cohort.task == "polo":
        summary = {"map100": _safe_mean(primary_aps), "map100_10": _safe_mean(all_aps)}
    else:
        summary = {"ap50": _safe_mean(primary_aps), "map50_95": _safe_mean(all_aps)}
    return {"summary": summary, "per_class": class_mean, "pr": pr_primary}


def paired_statistics(
    rows_by_model: dict[str, list[dict[str, Any]]], *, resamples: int = 10_000, seed: int = 42
) -> list[dict[str, Any]]:
    """Compute every unordered model pair with no designated model reference."""

    return paired_score_statistics(
        {name: pd.Series(_cluster_scores(rows), dtype=float) for name, rows in rows_by_model.items()},
        metric="ultimate_original_macro_f1",
        difference_name="difference_b_minus_a",
        support_name="independent_clusters",
        resamples=resamples,
        seed=seed,
    )


def paired_score_statistics(
    scores_by_model: Mapping[str, pd.Series],
    *,
    metric: str,
    difference_name: str,
    support_name: str,
    resamples: int,
    seed: int,
    include_wins: bool = False,
    include_difference_alias: bool = True,
    progress: bool = False,
    progress_description: str = "Computing paired statistics",
) -> list[dict[str, Any]]:
    """Bootstrap paired score differences from aligned pandas Series."""

    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    raw_p: list[float] = []
    names = list(scores_by_model)
    pairs = [
        (model_a, model_b)
        for left_index, model_a in enumerate(names)
        for model_b in names[left_index + 1 :]
    ]
    for model_a, model_b in tqdm(
        pairs,
        desc=progress_description,
        unit="pair",
        disable=not progress,
        dynamic_ncols=True,
    ):
        paired = pd.concat(
            [scores_by_model[model_a].rename("a"), scores_by_model[model_b].rename("b")],
            axis=1,
            join="inner",
        ).sort_index()
        paired = paired.replace([np.inf, -np.inf], np.nan).dropna()
        differences = (paired["b"] - paired["a"]).to_numpy(dtype=float)
        if not len(differences):
            continue
        samples = rng.choice(
            differences, size=(resamples, len(differences)), replace=True
        ).mean(axis=1)
        signs = rng.choice((-1.0, 1.0), size=(resamples, len(differences)))
        randomized = (differences * signs).mean(axis=1)
        p = float(
            (np.sum(np.abs(randomized) >= abs(differences.mean())) + 1)
            / (resamples + 1)
        )
        raw_p.append(p)
        difference = float(differences.mean())
        row = {
            "model_a": model_a, "model_b": model_b, "metric": metric,
            difference_name: difference, "ci_low": float(np.quantile(samples, 0.025)),
            "ci_high": float(np.quantile(samples, 0.975)), "p_value": p,
            support_name: len(differences),
        }
        if include_difference_alias and difference_name != "difference":
            row["difference"] = difference
        if include_wins:
            row.update(
                wins_model_b=int(np.sum(differences > 0)),
                ties=int(np.sum(differences == 0)),
                wins_model_a=int(np.sum(differences < 0)),
            )
        output.append(row)
    adjusted = [0.0] * len(output)
    running = 0.0
    for rank, index in enumerate(np.argsort(raw_p, kind="stable")):
        running = max(running, min(1.0, raw_p[index] * (len(raw_p) - rank)))
        adjusted[index] = running
    for row, value in zip(output, adjusted):
        row["p_value_holm"] = value
    return output


def bootstrap_metric(rows: list[dict[str, Any]], *, resamples: int = 10_000, seed: int = 42) -> tuple[float, float]:
    return bootstrap_interval(_cluster_scores(rows).values(), resamples=resamples, seed=seed)


def bootstrap_interval(
    values: Sequence[float], *, resamples: int, seed: int
) -> tuple[float, float]:
    data = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(data) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    sampled = rng.choice(data, size=(resamples, len(data)), replace=True).mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def _similarity(annotation: dict[str, Any], prediction: Prediction, task: str, metadata: dict[str, Any]) -> tuple[float, dict[str, float]]:
    if task == "polo":
        if annotation.get("point") is None or prediction.point is None:
            return math.inf, {}
        distance = math.dist(annotation["point"], prediction.point)
        radius = float(annotation.get("radius") or metadata.get("radii", {}).get(str(annotation["class_id"])) or metadata.get("radii", {}).get(annotation["class_id"]) or 1)
        return distance / max(radius, 1e-12), {"distance": distance, "normalized_distance": distance / max(radius, 1e-12)}
    if task == "segment" and annotation.get("polygon") and (
        prediction.polygons or prediction.polygon
    ):
        try:
            from shapely.geometry import Polygon
            from shapely.ops import unary_union

            a = Polygon(annotation["polygon"])
            b = unary_union(
                [
                    Polygon(polygon)
                    for polygon in (
                        prediction.polygons
                        or ([prediction.polygon] if prediction.polygon else [])
                    )
                ]
            )
            union = a.union(b).area
            return (a.intersection(b).area / union if union else 0.0), {}
        except Exception:
            return 0.0, {}
    if task == "pose" and annotation.get("keypoints") and prediction.keypoints:
        sigmas = metadata.get("kpt_oks_sigmas") or [0.1] * len(annotation["keypoints"])
        box = annotation.get("bbox") or _box_from_keypoints(annotation["keypoints"])
        area = max((box[2] - box[0]) * (box[3] - box[1]), 1.0)
        values = []
        for index, (truth, pred) in enumerate(zip(annotation["keypoints"], prediction.keypoints)):
            visible = truth[2] if len(truth) > 2 and truth[2] is not None else 2
            if visible <= 0:
                continue
            squared = (truth[0] - pred[0]) ** 2 + (truth[1] - pred[1]) ** 2
            sigma = float(sigmas[min(index, len(sigmas) - 1)])
            values.append(math.exp(-squared / (2 * area * (2 * sigma) ** 2 + 1e-12)))
        return (_safe_mean(values) if values else 0.0), {}
    return _box_iou(annotation.get("bbox"), prediction.bbox), {}


def _passes(value: float, task: str, *, threshold: float | None = None, fixed: bool = False) -> bool:
    threshold = (1.0 if task == "polo" else 0.5) if fixed else float(threshold)
    return value <= threshold if task == "polo" else value >= threshold


def _box_iou(a: Any, b: Any) -> float:
    if not a or not b:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _box_from_keypoints(points: list[Any]) -> tuple[float, float, float, float]:
    visible = [point for point in points if len(point) < 3 or point[2] is None or point[2] > 0]
    if not visible:
        return 0.0, 0.0, 1.0, 1.0
    return min(p[0] for p in visible), min(p[1] for p in visible), max(p[0] for p in visible), max(p[1] for p in visible)


def _curve(tp: list[int], fp: list[int], count: int) -> tuple[np.ndarray, np.ndarray]:
    if not tp:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
    recall = cum_tp / max(count, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    return recall, precision


def _interpolated_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    if not len(recall):
        return 0.0
    levels = np.linspace(0, 1, 101)
    return float(np.mean([np.max(precision[recall >= level]) if np.any(recall >= level) else 0.0 for level in levels]))


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _cluster_macro(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["original_id"]].append(row)
    result = []
    for original_id, values in grouped.items():
        tp, fp, fn = (sum(row[key] for row in values) for key in ("tp", "fp", "fn"))
        result.append({"original_id": original_id, "tp": tp, "fp": fp, "fn": fn, **_prf(tp, fp, fn)})
    return result


def _cluster_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {row["original_id"]: row["f1"] for row in _cluster_macro(rows)}


def _safe_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else math.nan


def _notebook_compatibility(cohort: Cohort, predictions: dict[str, list[Prediction]], confidence: float) -> float:
    tp = fp = fn = 0
    for record in cohort.records:
        gt = [annotation["point"] for annotation in record.annotations if annotation.get("point") is not None]
        pred = [value.point for value in predictions[record.image_id] if value.score >= confidence and value.point]
        candidates = sorted((math.dist(a, b), i, j) for i, a in enumerate(gt) for j, b in enumerate(pred))
        used_gt: set[int] = set()
        used_pred: set[int] = set()
        for distance, i, j in candidates:
            if distance <= 50 and i not in used_gt and j not in used_pred:
                used_gt.add(i)
                used_pred.add(j)
        tp += len(used_gt)
        fp += len(pred) - len(used_pred)
        fn += len(gt) - len(used_gt)
    return _prf(tp, fp, fn)["f1"]
