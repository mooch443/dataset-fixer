from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

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
    rows_by_model: dict[str, list[dict[str, Any]]], baseline: str, *, resamples: int = 10_000, seed: int = 42
) -> list[dict[str, Any]]:
    if baseline not in rows_by_model:
        raise ValueError(f"Unknown baseline {baseline!r}")
    base = _cluster_scores(rows_by_model[baseline])
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    raw_p: list[float] = []
    for name, rows in rows_by_model.items():
        if name == baseline:
            continue
        scores = _cluster_scores(rows)
        keys = sorted(set(base) & set(scores))
        differences = np.asarray([scores[key] - base[key] for key in keys], dtype=float)
        if not len(differences):
            continue
        samples = rng.choice(differences, size=(resamples, len(differences)), replace=True).mean(axis=1)
        signs = rng.choice((-1.0, 1.0), size=(resamples, len(differences)))
        randomized = (differences * signs).mean(axis=1)
        p = float((np.sum(np.abs(randomized) >= abs(differences.mean())) + 1) / (resamples + 1))
        raw_p.append(p)
        output.append(
            {
                "model": name,
                "baseline": baseline,
                "metric": "ultimate_original_macro_f1",
                "difference": float(differences.mean()),
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "p_value": p,
                "independent_clusters": len(keys),
            }
        )
    order = sorted(range(len(raw_p)), key=lambda index: raw_p[index])
    adjusted = [0.0] * len(raw_p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, raw_p[index] * (len(raw_p) - rank)))
        adjusted[index] = running
    for row, value in zip(output, adjusted):
        row["p_value_holm"] = value
    return output


def bootstrap_metric(rows: list[dict[str, Any]], *, resamples: int = 10_000, seed: int = 42) -> tuple[float, float]:
    values = np.asarray(list(_cluster_scores(rows).values()), dtype=float)
    if len(values) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
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
