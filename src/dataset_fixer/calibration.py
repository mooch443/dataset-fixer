from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from tqdm.auto import tqdm

from .dataset import Dataset
from .errors import PredictionCacheMissError, PredictionScoreUnavailableError
from .model import ImagePrediction, Model, ModelCollection, PredictionResult
from .prediction_cache import PredictionCache
from .comparison.plot_labels import with_model_identities
from .static_rendering import save_chart
from .tabular import chart_data, frame
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
    cache_audit: pd.DataFrame
    threshold_scores: pd.DataFrame
    fold_scores: pd.DataFrame
    improvements: pd.DataFrame


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
    min_connected_component_area: float | None = None,
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
        min_connected_component_area: Minimum 8-connected component area for
            both reference and predicted components in area-filtered metrics.
            ``None`` resolves to the p10 area of held-out reference objects,
            matching comparison reports.
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
    score_cache_root = _calibration_score_cache_root(
        dataset,
        prediction_cache=prediction_cache,
        destination=root,
    )
    reference_fingerprint = str(
        cache_context.get("semantic_cohort_fingerprint") or ""
    )
    resolved_component_area = _resolve_presence_component_area(
        inputs,
        requested=min_connected_component_area,
        cache_root=score_cache_root,
        reference_fingerprint=reference_fingerprint,
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
        grid = _threshold_grid(thresholds, model)
        candidate_floor = (
            None if is_semantic else float(model.prediction_threshold)
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
                    "rerun": False,
                }
            )
            model.unload()
            continue
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
        score_identity = _calibration_score_identity(
            model,
            inputs=inputs,
            reference_fingerprint=reference_fingerprint,
            minimum_component_area=resolved_component_area,
        )
        score_entry = (
            score_cache_root
            / "scores"
            / _calibration_score_key(score_identity)
        )
        per_threshold_rows = {
            threshold: rows
            for threshold in scored_thresholds
            if (
                rows := _load_threshold_case_scores(
                    score_entry,
                    identity=score_identity,
                    threshold=threshold,
                    inputs=inputs,
                )
            )
            is not None
        }
        missing_thresholds = tuple(
            value for value in scored_thresholds if value not in per_threshold_rows
        )
        allow_rerun = (
            rerun_missing_probability_maps
            if is_semantic
            else rerun_missing_instance_predictions
        )
        result: PredictionResult | None = None
        cache_status = "score-hit"
        if missing_thresholds:
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
                            "cached_thresholds": sorted(per_threshold_rows),
                            "missing_thresholds": list(missing_thresholds),
                            "rerun": False,
                        }
                    )
                    model.unload()
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
                    cache_status = "rerun"
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
            assert result is not None
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
            display_name = (
                model.name
                if len(model.name) <= 42
                else f"{model.name[:20]}…{model.name[-21:]}"
            )
            newly_scored = _case_rows_for_prediction_thresholds(
                result,
                inputs,
                thresholds=missing_thresholds,
                minimum_component_area=resolved_component_area,
                description=f"Scoring {display_name}",
                progress=progress,
            )
            for threshold, rows in newly_scored.items():
                _save_threshold_case_scores(
                    score_entry,
                    identity=score_identity,
                    threshold=threshold,
                    rows=rows,
                    inputs=inputs,
                )
                per_threshold_rows[threshold] = rows
        cache_audit.append(
            {
                "model": model.name,
                "model_type": model.model_type,
                "task": model.task,
                "status": "ready",
                "cache": cache_status,
                "cache_location": (
                    result.cache_info.get("location")
                    if result is not None
                    else str(score_entry)
                ),
                "score_cache_location": str(score_entry),
                "cached_thresholds": len(scored_thresholds) - len(missing_thresholds),
                "scored_thresholds": len(missing_thresholds),
                "candidate_floor": candidate_floor,
                "thresholds": list(grid),
                "rerun": cache_status == "rerun",
            }
        )
        for value in scored_thresholds:
            metrics = _summarize_rows(
                per_threshold_rows[value],
                minimum_component_area=resolved_component_area,
            )
            threshold_scores.append(
                {
                    "model": model.name,
                    "model_type": model.model_type,
                    "task": model.task,
                    "prediction_threshold": value,
                    "candidate_floor": candidate_floor,
                    "min_connected_component_area": resolved_component_area,
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
                    _summarize_rows(
                        per_threshold_rows[value],
                        train_indices,
                        minimum_component_area=resolved_component_area,
                    ),
                    value,
                ),
            )
            selected_metrics = _summarize_rows(
                per_threshold_rows[selected],
                test_indices,
                minimum_component_area=resolved_component_area,
            )
            baseline_metrics = (
                _summarize_rows(
                    per_threshold_rows[baseline_threshold],
                    test_indices,
                    minimum_component_area=resolved_component_area,
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
            for metric in (
                "macro_dice",
                "micro_dice",
                "presence_f1",
                "area_filtered_image_presence_f1",
                "area_filtered_component_f1",
            ):
                fold_row[f"{metric}_gain"] = _metric_difference(
                    selected_metrics.get(metric),
                    baseline_metrics.get(metric),
                )
            model_fold_rows.append(fold_row)
            fold_scores.append(fold_row)

        recommended = max(
            grid,
            key=lambda value: _selection_key(
                _summarize_rows(
                    per_threshold_rows[value],
                    minimum_component_area=resolved_component_area,
                ),
                # Selection remains pixel-level macro Dice/F1; component F1
                # is its localization-aware tie-breaker.
                value,
            ),
        )
        recommendations[model.name] = float(recommended)
        baseline_metrics = (
            _summarize_rows(
                per_threshold_rows[baseline_threshold],
                minimum_component_area=resolved_component_area,
            )
            if baseline_available
            else {}
        )
        recommended_metrics = _summarize_rows(
            per_threshold_rows[recommended],
            minimum_component_area=resolved_component_area,
        )
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
        if not is_semantic and result is not None:
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
        minimum_component_area=resolved_component_area,
        model_metadata=[model.describe() for model in collection],
    )
    return ThresholdCalibrationResult(
        location=root,
        recommendations=recommendations,
        cache_audit=frame(cache_audit),
        threshold_scores=frame(threshold_scores),
        fold_scores=frame(fold_scores),
        improvements=frame(improvements),
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


def _calibration_score_cache_root(
    dataset: Dataset,
    *,
    prediction_cache: bool | str | Path | PredictionCache,
    destination: Path,
) -> Path:
    """Resolve persistent incremental calibration-score storage."""

    from .prediction_cache import resolve_prediction_cache

    cache = resolve_prediction_cache(
        prediction_cache,
        source=dataset,
        default=True,
    )
    base = cache.location if cache is not None else destination / ".cache"
    return base / "calibration"


def _calibration_score_identity(
    model: Model,
    *,
    inputs: Sequence[Any],
    reference_fingerprint: str,
    minimum_component_area: float,
) -> dict[str, Any]:
    """Identify threshold-independent scores for one model/reference cohort."""

    from .model import _semantic_image_prediction_cache_identity
    from .sahi_support import resolve_sahi_settings

    resolution = model.resolution or 480
    resolved_sahi = resolve_sahi_settings(
        model.settings,
        resolution=resolution,
    ).as_dict()
    prediction_identity = _semantic_image_prediction_cache_identity(
        model,
        inputs=inputs,
        inference=model.inference,
        resolution=resolution,
        confidence=float(model.confidence),
        postprocess=float(model.postprocess),
        combined_settings=dict(model.settings),
        resolved_sahi=resolved_sahi,
    )
    return {
        "schema": 1,
        "space": "threshold-calibration-case-scores",
        "prediction_identity": prediction_identity,
        "reference_fingerprint": reference_fingerprint,
        "connectivity": 8,
        "min_connected_component_area": float(minimum_component_area),
        "component_matching": "one-to-one-any-pixel-overlap",
        "case_metrics": "tp-fp-fn-dice-and-component-detection-counts",
    }


def _calibration_score_key(identity: Mapping[str, Any]) -> str:
    from .prediction_cache import prediction_cache_key

    return prediction_cache_key(identity)


def _threshold_score_path(entry: Path, threshold: float) -> Path:
    from .prediction_cache import prediction_cache_key

    token = prediction_cache_key(
        {"schema": 1, "prediction_threshold": float(threshold)}
    )[:16]
    return entry / "thresholds" / f"{token}.json"


def _load_threshold_case_scores(
    entry: Path,
    *,
    identity: Mapping[str, Any],
    threshold: float,
    inputs: Sequence[Any],
) -> tuple[dict[str, Any], ...] | None:
    path = _threshold_score_path(entry, threshold)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    cached_threshold = (
        _finite_number(payload.get("prediction_threshold"))
        if isinstance(payload, dict)
        else math.nan
    )
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != 1
        or payload.get("identity") != to_jsonable(dict(identity))
        or not math.isclose(
            cached_threshold,
            float(threshold),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        return None
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != len(inputs):
        return None
    required = {
        "image_id",
        "tp",
        "fp",
        "fn",
        "n_ref",
        "n_pred",
        "dice",
        "component_reference_count",
        "component_prediction_count",
        "component_match_count",
    }
    if any(
        not isinstance(row, dict)
        or row.get("image_id") != value.image_id
        or not required.issubset(row)
        for row, value in zip(rows, inputs)
    ):
        return None
    return tuple(dict(row) for row in rows)


def _save_threshold_case_scores(
    entry: Path,
    *,
    identity: Mapping[str, Any],
    threshold: float,
    rows: Sequence[Mapping[str, Any]],
    inputs: Sequence[Any],
) -> None:
    if len(rows) != len(inputs):
        raise ValueError("Cannot cache incomplete threshold-calibration scores")
    path = _threshold_score_path(entry, threshold)
    payload = {
        "schema": 1,
        "identity": dict(identity),
        "prediction_threshold": float(threshold),
        "rows": [dict(row) for row in rows],
    }
    _write_json_atomic(path, payload)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(to_jsonable(dict(payload)), handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_presence_component_area(
    inputs: Sequence[Any],
    *,
    requested: float | None,
    cache_root: Path,
    reference_fingerprint: str,
    progress: bool,
) -> float:
    if requested is not None:
        value = float(requested)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                "min_connected_component_area must be finite and greater than zero"
            )
        return value
    from .prediction_cache import prediction_cache_key

    identity = {
        "schema": 1,
        "space": "reference-component-area-percentiles",
        "reference_fingerprint": reference_fingerprint,
        "connectivity": 8,
    }
    path = (
        cache_root
        / "reference-components"
        / f"{prediction_cache_key(identity)}.json"
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = None
    if isinstance(payload, dict) and payload.get("identity") == identity:
        value = _finite_number(payload.get("p10_area"))
        if math.isfinite(value) and value > 0:
            return value

    areas: list[int] = []
    for value in tqdm(
        inputs,
        desc="Measuring reference components",
        unit="image",
        disable=not progress,
    ):
        reference = _load_reference_mask(value)
        labels, count = ndimage.label(
            reference,
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        if count:
            counts = np.bincount(labels.ravel(), minlength=count + 1)
            areas.extend(int(area) for area in counts[1:])
    if not areas:
        raise ValueError(
            "Cannot resolve a p10 component area from a reference cohort "
            "without foreground objects"
        )
    resolved = float(np.percentile(np.asarray(areas, dtype=float), 10))
    _write_json_atomic(
        path,
        {
            "schema": 1,
            "identity": identity,
            "p10_area": resolved,
            "object_count": len(areas),
        },
    )
    return resolved


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
            model_metadata=result.model_metadata,
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


def _case_rows_for_prediction_thresholds(
    result: PredictionResult,
    inputs: Sequence[Any],
    *,
    thresholds: Sequence[float],
    minimum_component_area: float,
    description: str,
    progress: bool,
) -> dict[float, tuple[dict[str, Any], ...]]:
    rows: dict[float, list[dict[str, Any]]] = {
        float(threshold): [] for threshold in thresholds
    }
    by_id = result.by_id
    with tqdm(
        total=len(inputs) * len(thresholds),
        desc=description,
        unit="case-threshold",
        disable=not progress,
        leave=False,
    ) as score_progress:
        for value in inputs:
            record = by_id[value.image_id]
            if result.task == "semantic_segment":
                if record.foreground_probability is None:
                    raise PredictionCacheMissError(
                        f"{result.model_name!r} has no foreground probability map",
                        reason="missing-probability-maps",
                    )
                score_map = np.asarray(record.foreground_probability)
            else:
                score_map = record.foreground_score_map()
                if score_map is None:
                    raise PredictionScoreUnavailableError(
                        f"Instance thresholding for {result.model_name!r} requires "
                        f"scored polygons, but {value.image_id!r} has no polygon scores",
                        reason="missing-instance-polygon-scores",
                    )
            reference = _load_reference_mask(value)
            for threshold in thresholds:
                prediction = score_map >= threshold
                rows[float(threshold)].append(
                    {
                        "image_id": value.image_id,
                        **_case_row(
                            reference,
                            prediction,
                            minimum_component_area=minimum_component_area,
                        ),
                    }
                )
                score_progress.update(1)
    return {threshold: tuple(values) for threshold, values in rows.items()}


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
    *,
    minimum_component_area: float,
) -> dict[str, int | float]:
    ref = np.asarray(reference, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    tp = int(np.count_nonzero(ref & pred))
    fp = int(np.count_nonzero(~ref & pred))
    fn = int(np.count_nonzero(ref & ~pred))
    denominator = 2 * tp + fp + fn
    component_reference_count, component_prediction_count, component_match_count = (
        _component_detection_counts(
            ref,
            pred,
            minimum_component_area=minimum_component_area,
        )
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_ref": int(np.count_nonzero(ref)),
        "n_pred": int(np.count_nonzero(pred)),
        "component_reference_count": component_reference_count,
        "component_prediction_count": component_prediction_count,
        "component_match_count": component_match_count,
        "dice": 2 * tp / denominator if denominator else math.nan,
    }


def _component_detection_counts(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    minimum_component_area: float,
) -> tuple[int, int, int]:
    structure = np.ones((3, 3), dtype=np.uint8)
    reference_labels, reference_count = ndimage.label(
        np.asarray(reference, dtype=bool),
        structure=structure,
    )
    prediction_labels, prediction_count = ndimage.label(
        np.asarray(prediction, dtype=bool),
        structure=structure,
    )
    reference_areas = np.bincount(
        reference_labels.ravel(),
        minlength=reference_count + 1,
    )
    prediction_areas = np.bincount(
        prediction_labels.ravel(),
        minlength=prediction_count + 1,
    )
    reference_ids = np.flatnonzero(
        reference_areas[1:] >= minimum_component_area
    ) + 1
    prediction_ids = np.flatnonzero(
        prediction_areas[1:] >= minimum_component_area
    ) + 1
    if not len(reference_ids) or not len(prediction_ids):
        return len(reference_ids), len(prediction_ids), 0

    reference_index = {
        int(component_id): index
        for index, component_id in enumerate(reference_ids)
    }
    prediction_index = {
        int(component_id): index
        for index, component_id in enumerate(prediction_ids)
    }
    overlap = (reference_labels > 0) & (prediction_labels > 0)
    pairs = np.unique(
        np.column_stack(
            (reference_labels[overlap], prediction_labels[overlap])
        ),
        axis=0,
    )
    edges = [
        (reference_index[int(reference_id)], prediction_index[int(prediction_id)])
        for reference_id, prediction_id in pairs
        if int(reference_id) in reference_index
        and int(prediction_id) in prediction_index
    ]
    if not edges:
        return len(reference_ids), len(prediction_ids), 0
    rows, columns = zip(*edges)
    graph = csr_matrix(
        (np.ones(len(edges), dtype=np.uint8), (rows, columns)),
        shape=(len(reference_ids), len(prediction_ids)),
    )
    matching = maximum_bipartite_matching(graph, perm_type="column")
    matches = int(np.count_nonzero(matching >= 0))
    return len(reference_ids), len(prediction_ids), matches


def _summarize_rows(
    rows: Sequence[Mapping[str, int | float]],
    indices: Sequence[int] | None = None,
    *,
    minimum_component_area: float,
) -> dict[str, float | int]:
    data = frame(rows)
    if indices is not None:
        data = data.iloc[list(indices)]
    totals = data[["tp", "fp", "fn"]].sum()
    tp, fp, fn = (int(totals[key]) for key in ("tp", "fp", "fn"))
    positive = data["n_ref"].astype(int).gt(0)
    predicted = data["n_pred"].astype(int).gt(0)
    detected, empty_fp = int((positive & predicted).sum()), int((~positive & predicted).sum())
    presence_precision = _ratio(detected, detected + empty_fp)
    presence_recall = _ratio(detected, int(positive.sum()))
    presence_f1 = _f1(presence_precision, presence_recall)

    filtered_positive = data["component_reference_count"].astype(int).gt(0)
    filtered_predicted = data["component_prediction_count"].astype(int).gt(0)
    filtered_detected = int((filtered_positive & filtered_predicted).sum())
    filtered_fp = int((~filtered_positive & filtered_predicted).sum())
    filtered_precision = _ratio(filtered_detected, filtered_detected + filtered_fp)
    filtered_recall = _ratio(filtered_detected, int(filtered_positive.sum()))
    component_references, component_predictions, component_matches = (
        int(data[column].sum())
        for column in ("component_reference_count", "component_prediction_count", "component_match_count")
    )
    component_precision = _ratio(component_matches, component_predictions)
    component_recall = _ratio(component_matches, component_references)
    component_f1 = _f1(component_precision, component_recall)
    return {
        "macro_dice": _finite_series_mean(data["dice"]),
        "micro_dice": _ratio(2 * tp, 2 * tp + fp + fn),
        "foreground_precision": _ratio(tp, tp + fp),
        "foreground_recall": _ratio(tp, tp + fn),
        "presence_precision": presence_precision,
        "presence_recall": presence_recall,
        "presence_f1": presence_f1,
        # Image-level presence remains location-insensitive: any retained
        # prediction makes an image positive. Keep it as a separate diagnostic.
        "area_filtered_image_presence_precision": filtered_precision,
        "area_filtered_image_presence_recall": filtered_recall,
        "area_filtered_image_presence_f1": _f1(filtered_precision, filtered_recall),
        # Component detection is localization-aware. A match requires at least
        # one shared pixel and is assigned one-to-one, so duplicate predictions
        # remain false positives.
        "area_filtered_component_precision": component_precision,
        "area_filtered_component_recall": component_recall,
        "area_filtered_component_f1": component_f1,
        "area_filtered_component_matches": component_matches,
        "area_filtered_component_predictions": component_predictions,
        "area_filtered_component_references": component_references,
        "area_filtered_empty_specificity": (
            _ratio(int((~filtered_positive).sum()) - filtered_fp, int((~filtered_positive).sum()))
        ),
        "area_filtered_tiny_reference_only_excluded_cases": (
            int((positive & ~filtered_positive).sum())
        ),
        "empty_specificity": _ratio(int((~positive).sum()) - empty_fp, int((~positive).sum())),
        "cases": len(data),
    }


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def _finite_series_mean(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(finite.mean()) if finite.notna().any() else math.nan


def _f1(precision: float, recall: float) -> float:
    if (
        math.isfinite(precision)
        and math.isfinite(recall)
        and precision + recall
    ):
        return 2 * precision * recall / (precision + recall)
    return math.nan


def _selection_key(
    metrics: Mapping[str, float | int],
    threshold: float,
) -> tuple[float, ...]:
    def finite(value: Any) -> float:
        number = float(value)
        return number if math.isfinite(number) else -math.inf

    return (
        finite(metrics["macro_dice"]),
        finite(metrics["micro_dice"]),
        finite(metrics["area_filtered_component_f1"]),
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
    for metric in (
        "macro_dice",
        "micro_dice",
        "foreground_precision",
        "foreground_recall",
        "presence_precision",
        "presence_recall",
        "presence_f1",
        "area_filtered_image_presence_precision",
        "area_filtered_image_presence_recall",
        "area_filtered_image_presence_f1",
        "area_filtered_component_precision",
        "area_filtered_component_recall",
        "area_filtered_component_f1",
        "area_filtered_empty_specificity",
        "empty_specificity",
    ):
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
    """Format the two held-out calibration metrics for notebook progress."""

    before = _finite_number(row.get("cv_baseline_macro_dice"))
    after = _finite_number(row.get("cv_calibrated_macro_dice"))
    gain = _finite_number(row.get("cv_macro_dice_gain"))
    relative = _finite_number(row.get("cv_macro_dice_relative_gain_pct"))
    component_before = _finite_number(
        row.get("cv_baseline_area_filtered_component_f1")
    )
    component_after = _finite_number(
        row.get("cv_calibrated_area_filtered_component_f1")
    )
    precision_before = _finite_number(
        row.get("cv_baseline_area_filtered_component_precision")
    )
    precision_after = _finite_number(
        row.get("cv_calibrated_area_filtered_component_precision")
    )
    specificity_before = _finite_number(
        row.get("cv_baseline_area_filtered_empty_specificity")
    )
    specificity_after = _finite_number(
        row.get("cv_calibrated_area_filtered_empty_specificity")
    )
    if not all(math.isfinite(value) for value in (before, after, gain, relative)):
        return f"{row['model']}: grouped-CV macro Dice/F1 improvement unavailable"
    message = (
        f"{row['model']}: grouped-CV macro Dice/F1 {before:.3f} → {after:.3f} "
        f"({gain * 100:+.2f} percentage points; {relative:+.1f}% relative)"
    )
    if math.isfinite(component_before) and math.isfinite(component_after):
        message += (
            f"; area-filtered component F1 {component_before:.3f} → "
            f"{component_after:.3f}"
        )
    if math.isfinite(precision_before) and math.isfinite(precision_after):
        message += (
            f"; component precision {precision_before:.3f} → "
            f"{precision_after:.3f}"
        )
    if math.isfinite(specificity_before) and math.isfinite(specificity_after):
        message += (
            f"; empty specificity {specificity_before:.3f} → "
            f"{specificity_after:.3f}"
        )
    return message


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
    minimum_component_area: float,
    model_metadata: list[dict[str, Any]],
) -> None:
    frame(cache_audit).to_csv(root / "cache-audit.csv", index=False)
    scores = frame(threshold_scores)
    scores.to_csv(root / "threshold-scores.csv", index=False)
    frame(fold_scores).to_csv(root / "grouped-cv-folds.csv", index=False)
    frame(improvements).to_csv(
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
        "tie_breakers": [
            "micro_dice",
            "area_filtered_component_f1",
            "higher_threshold",
        ],
        "component_detection": {
            "connectivity": 8,
            "min_connected_component_area": minimum_component_area,
            "area_filter_policy": "apply equally to references and predictions",
            "matching": "maximum one-to-one matching on any pixel overlap",
            "overlap_requirement": "at least one shared pixel",
        },
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
    import altair as alt

    models = list(dict.fromkeys(scores["model"].tolist()))
    metric_fields = {
        "macro_dice": "macro Dice",
        "micro_dice": "micro Dice",
        "area_filtered_component_f1": "area-filtered component F1",
        "area_filtered_component_precision": "area-filtered component precision",
    }
    data = scores.melt(
        id_vars=["model", "prediction_threshold"],
        value_vars=list(metric_fields),
        var_name="metric",
        value_name="value",
    )
    data["metric"] = data["metric"].map(metric_fields)
    data["selected"] = data["model"].map(recommendations)
    data = chart_data(data)
    lines = (
        alt.Chart()
        .mark_line(point=True)
        .encode(
            x=alt.X("prediction_threshold:Q", title="prediction threshold"),
            y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1]), title="score"),
            color=alt.Color("metric:N"),
            tooltip=["model:N", "metric:N", "prediction_threshold:Q", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(width=680, height=190)
    )
    rules = (
        alt.Chart()
        .mark_rule(color="#d95f02", strokeDash=[6, 4], strokeWidth=2)
        .encode(x="selected:Q")
    )
    chart = (
        alt.layer(lines, rules, data=data)
        .facet(row=alt.Row("model:N", sort=models, title=None, header=alt.Header(labelFontSize=12)))
        .properties(title="Full-image threshold calibration (grouped CV cohort)")
        .resolve_scale(x="shared", y="shared")
    )
    save_chart(
        with_model_identities(chart, model_metadata),
        root / "threshold-curves.png",
    )
