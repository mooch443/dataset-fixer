from __future__ import annotations

import json
import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from .dataset import Dataset
from .errors import PredictionCacheMissError, PredictionScoreUnavailableError
from .model import ImagePrediction, Model, ModelCollection, PredictionResult
from .prediction_cache import PredictionCache
from .utils import to_jsonable


@dataclass(frozen=True)
class ThresholdCalibrationResult:
    """Artifacts and per-model operating points from grouped calibration.

    Args:
        location: Directory containing the written calibration artifacts.
        recommendations: Selected ``prediction_threshold`` by model name.
        cache_audit: Per-model cache reuse or rerun decisions.
        threshold_scores: Whole-cohort metrics for every tested threshold.
        fold_scores: Held-out grouped-fold metrics at fold-selected thresholds.
        improvements: Baseline-to-calibrated F1/Dice gains, including grouped
            held-out estimates suitable for concise reporting.
    """

    location: Path
    recommendations: dict[str, float]
    cache_audit: tuple[dict[str, Any], ...]
    threshold_scores: tuple[dict[str, Any], ...]
    fold_scores: tuple[dict[str, Any], ...]
    improvements: tuple[dict[str, Any], ...]


def calibrate_prediction_thresholds(
    models: Any,
    dataset: Dataset,
    *,
    split: str = "val",
    group_by: Callable[[Path], Hashable],
    thresholds: Sequence[float] | Mapping[str, Sequence[float]] = (
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ),
    folds: int = 5,
    seed: int = 42,
    prediction_cache: bool | str | Path | PredictionCache = True,
    rerun_missing_probability_maps: bool = False,
    rerun_missing_instance_predictions: bool = False,
    destination: str | Path | None = None,
    progress: bool = True,
) -> ThresholdCalibrationResult:
    """Calibrate one task-aware ``prediction_threshold`` per model.

    Semantic thresholds operate on cached foreground-probability maps after
    full-image SAHI reconstruction. Instance thresholds operate on cached,
    scored instances and therefore cannot be evaluated below the confidence
    floor used to create that cache. Cache-only reuse is the default; the two
    explicit rerun switches control the only circumstances in which inference
    may occur.

    Args:
        models: Model source specifications, models, or a model collection.
        dataset: Semantic-mask dataset containing images and reference masks.
        split: Dataset split to calibrate.
        group_by: Function mapping each image path to its independent group,
            such as an AOI or island identifier.
        thresholds: Shared threshold grid, or grids keyed by resolved model
            name.
        folds: Maximum number of deterministic group-balanced folds.
        seed: Random seed used before greedily balancing groups across folds.
        prediction_cache: Unified prediction-cache selection. Existing
            dataset-local comparison caches are used when ``True``.
        rerun_missing_probability_maps: Rerun only semantic models whose
            compatible cache lacks reusable foreground probabilities.
        rerun_missing_instance_predictions: Rerun only instance models whose
            compatible scored-instance cache is absent.
        destination: Directory for audit tables, scores, recommendations, and
            the threshold plot.
        progress: Display cohort preparation, cache loading, threshold
            scoring, and any permitted inference progress.

    Returns:
        Calibration artifacts and task-aware per-model recommendations.
    """

    if dataset.format != "semantic_masks":
        raise TypeError("Threshold calibration requires a semantic-mask Dataset")
    if not callable(group_by):
        raise TypeError("group_by must be a callable returning stable group labels")
    if folds < 2:
        raise ValueError("folds must be at least 2")
    collection = Model.load_many(models)
    root = (
        Path(destination).expanduser().resolve()
        if destination is not None
        else dataset.location / "evaluations" / "threshold-calibration"
    )
    root.mkdir(parents=True, exist_ok=True)

    inputs, paths, cache_context = _calibration_cohort(
        dataset,
        split,
        progress=progress,
    )
    groups = [str(group_by(path)) for path in paths]
    fold_ids = _grouped_folds(groups, folds=folds, seed=seed)
    cache_audit: list[dict[str, Any]] = []
    threshold_scores: list[dict[str, Any]] = []
    fold_scores: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    recommendations: dict[str, float] = {}

    model_progress = tqdm(
        collection,
        total=len(collection),
        desc="Calibrating thresholds",
        unit="model",
        disable=not progress,
    )
    for model in model_progress:
        model_progress.set_postfix_str(model.name)
        is_semantic = model._uses_semantic_prediction_threshold()
        allow_rerun = (
            rerun_missing_probability_maps
            if is_semantic
            else rerun_missing_instance_predictions
        )
        try:
            result = _predict_with_prepared_cohort(
                model,
                dataset,
                inputs=inputs,
                cache_context=cache_context,
                split=split,
                prediction_cache=prediction_cache,
                cache_only=True,
                require_probability_maps=is_semantic,
                progress=progress,
            )
            cache_status = "hit"
        except PredictionCacheMissError as error:
            if not allow_rerun:
                cache_audit.append(
                    {
                        "model": model.name,
                        "model_type": model.model_type,
                        "task": model.task,
                        "status": "unavailable",
                        "reason": error.reason,
                        "rerun": False,
                    }
                )
                continue
            try:
                result = _predict_with_prepared_cohort(
                    model,
                    dataset,
                    inputs=inputs,
                    cache_context=cache_context,
                    split=split,
                    prediction_cache=prediction_cache,
                    cache_only=False,
                    require_probability_maps=is_semantic,
                    progress=progress,
                )
            except PredictionScoreUnavailableError as score_error:
                cache_audit.append(
                    {
                        "model": model.name,
                        "model_type": model.model_type,
                        "task": model.task,
                        "status": "unavailable",
                        "reason": score_error.reason,
                        "rerun": True,
                    }
                )
                model.unload()
                continue
            if is_semantic and any(
                record.foreground_probability is None for record in result
            ):
                cache_audit.append(
                    {
                        "model": model.name,
                        "model_type": model.model_type,
                        "task": model.task,
                        "status": "unavailable",
                        "reason": "backend-returned-no-probability-maps",
                        "rerun": True,
                    }
                )
                model.unload()
                continue
            cache_status = "rerun"

        grid = _threshold_grid(thresholds, model)
        candidate_floor = (
            None
            if is_semantic
            else float(
                result.settings.get(
                    "prediction_threshold",
                    result.settings.get("confidence", model.prediction_threshold),
                )
            )
        )
        if candidate_floor is not None:
            grid = tuple(value for value in grid if value >= candidate_floor - 1e-12)
        if not grid:
            cache_audit.append(
                {
                    "model": model.name,
                    "model_type": model.model_type,
                    "task": model.task,
                    "status": "unavailable",
                    "reason": "no-threshold-at-or-above-cached-candidate-floor",
                    "candidate_floor": candidate_floor,
                    "rerun": cache_status == "rerun",
                }
            )
            del result
            model.unload()
            continue

        cache_audit.append(
            {
                "model": model.name,
                "model_type": model.model_type,
                "task": model.task,
                "status": "ready",
                "cache": cache_status,
                "cache_location": result.cache_info.get("location"),
                "candidate_floor": candidate_floor,
                "thresholds": list(grid),
                "rerun": cache_status == "rerun",
            }
        )
        baseline_threshold = float(model.prediction_threshold)
        baseline_available = candidate_floor is None or (
            baseline_threshold >= candidate_floor - 1e-12
        )
        scored_thresholds = tuple(
            sorted(
                set(grid)
                | ({baseline_threshold} if baseline_available else set())
            )
        )
        display_name = (
            model.name
            if len(model.name) <= 42
            else f"{model.name[:20]}…{model.name[-21:]}"
        )
        per_threshold_rows: dict[float, tuple[dict[str, int | float], ...]] = {}
        with tqdm(
            total=len(scored_thresholds) * len(inputs),
            desc=f"Scoring {display_name}",
            unit="case-threshold",
            disable=not progress,
            leave=False,
        ) as score_progress:
            for value in scored_thresholds:
                per_threshold_rows[value] = _case_rows_for_prediction_threshold(
                    result,
                    inputs,
                    threshold=value,
                    progress_bar=score_progress,
                )
        for value in scored_thresholds:
            metrics = _summarize_rows(per_threshold_rows[value])
            threshold_scores.append(
                {
                    "model": model.name,
                    "model_type": model.model_type,
                    "task": model.task,
                    "prediction_threshold": value,
                    "candidate_floor": candidate_floor,
                    "is_candidate": value in grid,
                    "is_baseline": math.isclose(
                        value,
                        baseline_threshold,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ),
                    **metrics,
                }
            )

        unique_folds = sorted(set(fold_ids))
        model_fold_rows: list[dict[str, Any]] = []
        for fold_id in unique_folds:
            train_indices = [
                index for index, value in enumerate(fold_ids) if value != fold_id
            ]
            test_indices = [
                index for index, value in enumerate(fold_ids) if value == fold_id
            ]
            selected = max(
                grid,
                key=lambda value: _selection_key(
                    _summarize_rows(per_threshold_rows[value], train_indices),
                    value,
                ),
            )
            selected_metrics = _summarize_rows(
                per_threshold_rows[selected],
                test_indices,
            )
            baseline_metrics = (
                _summarize_rows(
                    per_threshold_rows[baseline_threshold],
                    test_indices,
                )
                if baseline_available
                else {}
            )
            fold_row = {
                "model": model.name,
                "fold": fold_id,
                "train_groups": len({groups[index] for index in train_indices}),
                "test_groups": len({groups[index] for index in test_indices}),
                "prediction_threshold": selected,
                "baseline_threshold": baseline_threshold,
                **selected_metrics,
                **{
                    f"baseline_{key}": value
                    for key, value in baseline_metrics.items()
                },
            }
            for metric in ("macro_dice", "micro_dice", "presence_f1"):
                fold_row[f"{metric}_gain"] = _metric_difference(
                    selected_metrics.get(metric),
                    baseline_metrics.get(metric),
                )
            model_fold_rows.append(fold_row)
            fold_scores.append(fold_row)

        recommended = max(
            grid,
            key=lambda value: _selection_key(
                _summarize_rows(per_threshold_rows[value]),
                value,
            ),
        )
        recommendations[model.name] = float(recommended)
        baseline_metrics = (
            _summarize_rows(per_threshold_rows[baseline_threshold])
            if baseline_available
            else {}
        )
        recommended_metrics = _summarize_rows(per_threshold_rows[recommended])
        improvement = _calibration_improvement_row(
            model=model,
            baseline_threshold=baseline_threshold,
            recommended_threshold=float(recommended),
            baseline_metrics=baseline_metrics,
            recommended_metrics=recommended_metrics,
            fold_rows=model_fold_rows,
        )
        improvements.append(improvement)
        if progress:
            tqdm.write(_format_improvement(improvement))
        if not is_semantic:
            _publish_instance_threshold_alias(
                model,
                result,
                inputs=inputs,
                dataset=dataset,
                prediction_cache=prediction_cache,
                threshold=float(recommended),
            )
        del per_threshold_rows, result
        model.unload()

    _write_calibration_artifacts(
        root,
        cache_audit=cache_audit,
        threshold_scores=threshold_scores,
        fold_scores=fold_scores,
        improvements=improvements,
        recommendations=recommendations,
        groups=groups,
        fold_ids=fold_ids,
        split=split,
        seed=seed,
    )
    return ThresholdCalibrationResult(
        location=root,
        recommendations=recommendations,
        cache_audit=tuple(cache_audit),
        threshold_scores=tuple(threshold_scores),
        fold_scores=tuple(fold_scores),
        improvements=tuple(improvements),
    )


def _calibration_cohort(
    dataset: Dataset,
    split: str,
    *,
    progress: bool,
) -> tuple[
    tuple[Any, ...],
    tuple[Path, ...],
    dict[str, Any],
]:
    from .model import normalize_model_inputs

    inputs, _, context = normalize_model_inputs(
        dataset,
        split=split,
        progress=progress,
    )
    paths: list[Path] = []
    for value in inputs:
        if value.mask_path is None:
            raise ValueError(f"Calibration input {value.image_id!r} has no reference mask")
        paths.append(value.image_path)
    return (
        inputs,
        tuple(paths),
        dict(context),
    )


def _predict_with_prepared_cohort(
    model: Model,
    dataset: Dataset,
    *,
    inputs: Sequence[Any],
    cache_context: Mapping[str, Any],
    split: str,
    **options: Any,
) -> PredictionResult:
    """Reuse one frozen cohort while preserving ``Model.predict`` cache keys."""

    marker = object()
    previous = getattr(model, "_normalized_prediction_override", marker)
    model._normalized_prediction_override = (
        dataset,
        split,
        tuple(inputs),
        "semantic_segment",
        dict(cache_context),
    )
    try:
        return model.predict(dataset, split=split, **options)
    finally:
        if previous is marker:
            delattr(model, "_normalized_prediction_override")
        else:
            model._normalized_prediction_override = previous


def _publish_instance_threshold_alias(
    model: Model,
    result: PredictionResult,
    *,
    inputs: Sequence[Any],
    dataset: Dataset,
    prediction_cache: bool | str | Path | PredictionCache,
    threshold: float,
) -> None:
    """Publish a higher score cutoff without repeating candidate inference."""

    from .prediction_cache import prediction_cache_key, resolve_prediction_cache
    from .sahi_support import resolve_sahi_settings
    from .semantic_comparison import _raw_semantic_prediction_cache_identity

    cache = resolve_prediction_cache(prediction_cache, source=dataset, default=True)
    if cache is None:
        return
    configured = model._configured_copy(
        overrides={"prediction_threshold": float(threshold)}
    )
    resolved_sahi = (
        resolve_sahi_settings(
            configured.settings,
            resolution=configured.resolution or 480,
        ).as_dict()
        if configured.inference == "sahi"
        else None
    )
    identity = _raw_semantic_prediction_cache_identity(
        configured,
        inputs=inputs,
        resolved_sahi=resolved_sahi,
    )
    key = prediction_cache_key(identity)
    if cache.load(
        key,
        namespace="semantic",
        identity=identity,
        inputs=inputs,
    ) is not None:
        return
    cache.save(
        key,
        PredictionResult(
            model_name=result.model_name,
            model_kind=result.model_kind,
            task=result.task,
            backend=result.backend,
            records=result.records,
            inference_seconds=result.inference_seconds,
            settings={**result.settings, "confidence": float(threshold)},
        ),
        namespace="semantic",
        identity=identity,
        inputs=inputs,
    )


def _threshold_grid(
    thresholds: Sequence[float] | Mapping[str, Sequence[float]],
    model: Model,
) -> tuple[float, ...]:
    values = thresholds.get(model.name, ()) if isinstance(thresholds, Mapping) else thresholds
    grid = tuple(sorted({float(value) for value in values}))
    if not grid or any(not math.isfinite(value) or not 0 <= value <= 1 for value in grid):
        raise ValueError(f"Invalid threshold grid for {model.name!r}")
    return grid


def _grouped_folds(groups: Sequence[str], *, folds: int, seed: int) -> tuple[int, ...]:
    counts: dict[str, int] = {}
    for group in groups:
        counts[group] = counts.get(group, 0) + 1
    if len(counts) < 2:
        raise ValueError("Grouped calibration requires at least two distinct groups")
    fold_count = min(int(folds), len(counts))
    rng = np.random.default_rng(seed)
    shuffled = list(counts)
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda value: counts[value], reverse=True)
    loads = [0] * fold_count
    assignments: dict[str, int] = {}
    for group in shuffled:
        fold_id = min(range(fold_count), key=lambda value: (loads[value], value))
        assignments[group] = fold_id
        loads[fold_id] += counts[group]
    return tuple(assignments[group] for group in groups)


def _case_rows_for_prediction_threshold(
    result: PredictionResult,
    inputs: Sequence[Any],
    *,
    threshold: float,
    progress_bar: Any | None = None,
) -> tuple[dict[str, int | float], ...]:
    rows: list[dict[str, int | float]] = []
    by_id = result.by_id
    if result.task == "semantic_segment":
        for value in inputs:
            record = by_id[value.image_id]
            if record.foreground_probability is None:
                raise PredictionCacheMissError(
                    f"{result.model_name!r} has no foreground probability map",
                    reason="missing-probability-maps",
                )
            prediction = np.asarray(record.foreground_probability) >= threshold
            rows.append(_case_row(_load_reference_mask(value), prediction))
            if progress_bar is not None:
                progress_bar.update(1)
        return tuple(rows)

    for value in inputs:
        score_map = by_id[value.image_id].foreground_score_map()
        if score_map is None:
            raise PredictionScoreUnavailableError(
                f"Instance thresholding for {result.model_name!r} requires "
                f"scored polygons, but {value.image_id!r} has no polygon scores",
                reason="missing-instance-polygon-scores",
            )
        prediction = score_map >= threshold
        rows.append(_case_row(_load_reference_mask(value), prediction))
        if progress_bar is not None:
            progress_bar.update(1)
    return tuple(rows)


def _load_reference_mask(value: Any) -> np.ndarray:
    if value.mask_path is None:
        raise ValueError(f"Calibration input {value.image_id!r} has no reference mask")
    with Image.open(value.mask_path) as opened:
        reference = np.asarray(opened.copy()) > 0
    if reference.shape != (value.height, value.width):
        raise ValueError(f"Reference dimensions changed for {value.relative_path}")
    return reference


def _case_row(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, int | float]:
    ref = np.asarray(reference, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    tp = int(np.count_nonzero(ref & pred))
    fp = int(np.count_nonzero(~ref & pred))
    fn = int(np.count_nonzero(ref & ~pred))
    denominator = 2 * tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_ref": int(np.count_nonzero(ref)),
        "n_pred": int(np.count_nonzero(pred)),
        "dice": 2 * tp / denominator if denominator else math.nan,
    }


def _summarize_rows(
    rows: Sequence[Mapping[str, int | float]],
    indices: Sequence[int] | None = None,
) -> dict[str, float | int]:
    selected = list(rows) if indices is None else [rows[index] for index in indices]
    finite = [float(row["dice"]) for row in selected if math.isfinite(float(row["dice"]))]
    tp = sum(int(row["tp"]) for row in selected)
    fp = sum(int(row["fp"]) for row in selected)
    fn = sum(int(row["fn"]) for row in selected)
    positives = [row for row in selected if int(row["n_ref"]) > 0]
    empties = [row for row in selected if int(row["n_ref"]) == 0]
    detected = sum(int(row["n_pred"]) > 0 for row in positives)
    empty_fp = sum(int(row["n_pred"]) > 0 for row in empties)
    precision_denominator = detected + empty_fp
    presence_precision = detected / precision_denominator if precision_denominator else math.nan
    presence_recall = detected / len(positives) if positives else math.nan
    presence_f1 = (
        2 * presence_precision * presence_recall / (presence_precision + presence_recall)
        if math.isfinite(presence_precision)
        and math.isfinite(presence_recall)
        and presence_precision + presence_recall
        else math.nan
    )
    return {
        "macro_dice": float(np.mean(finite)) if finite else math.nan,
        "micro_dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else math.nan,
        "presence_precision": presence_precision,
        "presence_recall": presence_recall,
        "presence_f1": presence_f1,
        "empty_specificity": (
            (len(empties) - empty_fp) / len(empties) if empties else math.nan
        ),
        "cases": len(selected),
    }


def _selection_key(metrics: Mapping[str, float | int], threshold: float) -> tuple[float, ...]:
    def finite(value: Any) -> float:
        number = float(value)
        return number if math.isfinite(number) else -math.inf

    return (
        finite(metrics["macro_dice"]),
        finite(metrics["micro_dice"]),
        finite(metrics["presence_f1"]),
        float(threshold),
    )


def _finite_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _metric_difference(selected: Any, baseline: Any) -> float:
    selected_value = _finite_number(selected)
    baseline_value = _finite_number(baseline)
    if not math.isfinite(selected_value) or not math.isfinite(baseline_value):
        return math.nan
    return selected_value - baseline_value


def _relative_gain_percent(selected: Any, baseline: Any) -> float:
    selected_value = _finite_number(selected)
    baseline_value = _finite_number(baseline)
    if (
        not math.isfinite(selected_value)
        or not math.isfinite(baseline_value)
        or baseline_value == 0
    ):
        return math.nan
    return 100.0 * (selected_value - baseline_value) / baseline_value


def _mean_finite(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [_finite_number(row.get(key)) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else math.nan


def _calibration_improvement_row(
    *,
    model: Model,
    baseline_threshold: float,
    recommended_threshold: float,
    baseline_metrics: Mapping[str, Any],
    recommended_metrics: Mapping[str, Any],
    fold_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize apparent and held-out threshold-calibration gains."""

    row: dict[str, Any] = {
        "model": model.name,
        "model_type": model.model_type,
        "task": model.task,
        "baseline_threshold": baseline_threshold,
        "recommended_threshold": recommended_threshold,
        "fold_selected_threshold_min": min(
            (float(value["prediction_threshold"]) for value in fold_rows),
            default=math.nan,
        ),
        "fold_selected_threshold_max": max(
            (float(value["prediction_threshold"]) for value in fold_rows),
            default=math.nan,
        ),
    }
    for metric in ("macro_dice", "micro_dice", "presence_f1"):
        baseline = _finite_number(baseline_metrics.get(metric))
        calibrated = _finite_number(recommended_metrics.get(metric))
        row[f"cohort_baseline_{metric}"] = baseline
        row[f"cohort_calibrated_{metric}"] = calibrated
        row[f"cohort_{metric}_gain"] = _metric_difference(calibrated, baseline)
        row[f"cohort_{metric}_relative_gain_pct"] = _relative_gain_percent(
            calibrated,
            baseline,
        )

        cv_baseline = _mean_finite(fold_rows, f"baseline_{metric}")
        cv_calibrated = _mean_finite(fold_rows, metric)
        row[f"cv_baseline_{metric}"] = cv_baseline
        row[f"cv_calibrated_{metric}"] = cv_calibrated
        row[f"cv_{metric}_gain"] = _metric_difference(
            cv_calibrated,
            cv_baseline,
        )
        row[f"cv_{metric}_relative_gain_pct"] = _relative_gain_percent(
            cv_calibrated,
            cv_baseline,
        )
    return row


def _format_improvement(row: Mapping[str, Any]) -> str:
    """Format the primary held-out pixel-F1 result for notebook progress."""

    before = _finite_number(row.get("cv_baseline_macro_dice"))
    after = _finite_number(row.get("cv_calibrated_macro_dice"))
    gain = _finite_number(row.get("cv_macro_dice_gain"))
    relative = _finite_number(row.get("cv_macro_dice_relative_gain_pct"))
    if not all(math.isfinite(value) for value in (before, after, gain, relative)):
        return f"{row['model']}: grouped-CV macro Dice/F1 improvement unavailable"
    return (
        f"{row['model']}: grouped-CV macro Dice/F1 {before:.3f} → {after:.3f} "
        f"({gain * 100:+.2f} percentage points; {relative:+.1f}% relative)"
    )


def _write_calibration_artifacts(
    root: Path,
    *,
    cache_audit: list[dict[str, Any]],
    threshold_scores: list[dict[str, Any]],
    fold_scores: list[dict[str, Any]],
    improvements: list[dict[str, Any]],
    recommendations: dict[str, float],
    groups: Sequence[str],
    fold_ids: Sequence[int],
    split: str,
    seed: int,
) -> None:
    pd.DataFrame(cache_audit).to_csv(root / "cache-audit.csv", index=False)
    scores = pd.DataFrame(threshold_scores)
    scores.to_csv(root / "threshold-scores.csv", index=False)
    pd.DataFrame(fold_scores).to_csv(root / "grouped-cv-folds.csv", index=False)
    pd.DataFrame(improvements).to_csv(
        root / "calibration-improvements.csv",
        index=False,
    )
    pd.DataFrame({"group": groups, "fold": fold_ids}).drop_duplicates().to_csv(
        root / "grouped-cv-assignments.csv",
        index=False,
    )
    payload = {
        "schema": 1,
        "split": split,
        "seed": seed,
        "selection_metric": "macro_dice",
        "tie_breakers": ["micro_dice", "presence_f1", "higher_threshold"],
        "recommendations": {
            name: {"prediction_threshold": threshold}
            for name, threshold in recommendations.items()
        },
        "improvements": improvements,
        "cache_audit": cache_audit,
    }
    (root / "recommendations.json").write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if scores.empty:
        return
    models = list(dict.fromkeys(scores["model"].tolist()))
    figure, axes = plt.subplots(
        len(models),
        1,
        figsize=(9, max(3.0, 2.6 * len(models))),
        squeeze=False,
        sharex=True,
    )
    for axis, model_name in zip(axes[:, 0], models, strict=True):
        selected = scores[scores["model"] == model_name]
        axis.plot(
            selected["prediction_threshold"],
            selected["macro_dice"],
            marker="o",
            label="macro Dice",
        )
        axis.plot(
            selected["prediction_threshold"],
            selected["micro_dice"],
            marker="o",
            label="micro Dice",
        )
        axis.axvline(recommendations[model_name], color="#d95f02", linestyle="--")
        axis.set_title(model_name, fontsize=9)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, loc="best")
    axes[-1, 0].set_xlabel("prediction_threshold")
    figure.suptitle("Full-image threshold calibration (grouped CV cohort)")
    figure.tight_layout()
    figure.savefig(root / "threshold-curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
