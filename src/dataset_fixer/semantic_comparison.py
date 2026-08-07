from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from .comparison.reporting import write_csv, write_json
from .errors import DatasetValidationError, ValidationIssue
from .model import ImagePrediction, Model, ModelCollection, ModelInput
from .models import SemanticComparisonResult
from .sahi_support import resolve_sahi_settings
from .utils import (
    IMAGE_SUFFIXES,
    ensure_safe_destination,
    environment_snapshot,
    normalize_split,
    settings_fingerprint,
    sha256_file,
    to_jsonable,
)


@dataclass(frozen=True)
class _SemanticCase:
    case_id: str
    relative_path: Path
    image_path: Path
    mask_path: Path
    width: int
    height: int
    image_sha256: str
    mask_sha256: str


if TYPE_CHECKING:
    from .dataset import Dataset


def compare_nnunet_models(
    export: "Dataset",
    models: Any,
    *,
    split: str,
    baseline: str | None,
    folds: tuple[int | str, ...],
    checkpoint: str,
    device: str,
    workers: int,
    bootstrap_resamples: int,
    seed: int,
    keep_predictions: bool,
    visualize: bool,
    progress: bool,
    destination: str | Path | None,
    inference: str | None = "native",
    resolution: int = 480,
    comparison_unit: str = "model",
    sahi_slice_height: int | None = None,
    sahi_slice_width: int | None = None,
    sahi_overlap: float = 0.2,
    sahi_overlap_height_ratio: float | None = None,
    sahi_overlap_width_ratio: float | None = None,
    sahi_postprocess_type: str = "GREEDYNMM",
    sahi_postprocess_match_metric: str = "IOS",
    sahi_postprocess_class_agnostic: bool = False,
    sahi_model_type: str = "ultralytics",
) -> SemanticComparisonResult:
    """Run official nnU-Net v2 prediction and evaluation for an export."""

    split = normalize_split(split)
    if split not in export.splits:
        raise ValueError(f"Unknown semantic-mask split {split!r}; available splits are {export.splits}")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be 'cpu', 'cuda', or 'mps'")
    if inference is not None and inference not in {"native", "sahi"}:
        raise ValueError("inference must be 'native', 'sahi', or None; 'auto' was removed")
    if comparison_unit not in {"model", "system"}:
        raise ValueError("comparison_unit must be 'model' or 'system'")

    sahi_options = {
        "sahi_slice_height": sahi_slice_height,
        "sahi_slice_width": sahi_slice_width,
        "sahi_overlap": sahi_overlap,
        "sahi_overlap_height_ratio": sahi_overlap_height_ratio,
        "sahi_overlap_width_ratio": sahi_overlap_width_ratio,
        "sahi_postprocess_type": sahi_postprocess_type,
        "sahi_postprocess_match_metric": sahi_postprocess_match_metric,
        "sahi_postprocess_class_agnostic": sahi_postprocess_class_agnostic,
        "sahi_model_type": sahi_model_type,
    }
    effective_sahi_options = {
        key: value for key, value in sahi_options.items() if value is not None
    }

    specs = _parse_models(models, default_folds=folds, default_checkpoint=checkpoint)
    model_backends = {spec.name: inference or spec.inference for spec in specs}
    if comparison_unit == "model" and len(set(model_backends.values())) != 1:
        raise ValueError("Mixed native/SAHI configurations require comparison_unit='system'")
    resolved_sahi_by_model = {
        spec.name: resolve_sahi_settings(
            {**spec.settings, **effective_sahi_options},
            resolution=spec.resolution or resolution,
        ).as_dict()
        for spec in specs
        if model_backends[spec.name] == "sahi"
    }
    if comparison_unit == "model" and len(
        {
            tuple(sorted(configuration.items()))
            for configuration in resolved_sahi_by_model.values()
        }
    ) > 1:
        raise ValueError(
            "Different SAHI tile configurations require comparison_unit='system'"
        )
    if baseline is None:
        baseline = specs[0].name
    if baseline not in {spec.name for spec in specs}:
        raise ValueError(f"Unknown baseline {baseline!r}")
    _require_official_commands(
        "nnUNetv2_predict_from_modelfolder",
        "nnUNetv2_evaluate_folder",
    )
    cases, cohort_fingerprint = _freeze_cohort(export, split)

    resolved_settings = {
        "backend": (
            next(iter(set(model_backends.values())))
            if len(set(model_backends.values())) == 1
            else "mixed"
        ),
        "adapter": "nnunetv2-official",
        "report_schema": 4,
        "canonical_projection": (
            "sahi-feathered-probability-area-pool-argmax"
            if set(model_backends.values()) == {"sahi"}
            else (
                "per-model-see-model_backends"
                if len(set(model_backends.values())) > 1
                else "probability-area-pool-argmax"
            )
        ),
        "split": split,
        "baseline": baseline,
        "device": device,
        "workers": workers,
        "bootstrap_resamples": bootstrap_resamples,
        "seed": seed,
        "keep_predictions": keep_predictions,
        "visualize": visualize,
        "inference": inference,
        "model_backends": model_backends,
        "resolution": resolution,
        "comparison_unit": comparison_unit,
        "sahi_models": resolved_sahi_by_model,
        "models": [
            {
                "name": spec.name,
                "model_folder": spec.model_folder,
                "folds": spec.folds,
                "checkpoint": spec.checkpoint,
                "checkpoint_sha256": spec.checkpoint_sha256,
                "model_sha256": spec.digest,
                "upscale_factor": spec.upscale_factor,
                "device": spec.device or device,
                "workers": spec.workers,
            }
            for spec in specs
        ],
    }
    fingerprint = settings_fingerprint(
        {**to_jsonable(resolved_settings), "cohort_fingerprint": cohort_fingerprint}
    )
    target = (
        Path(destination).expanduser().resolve()
        if destination is not None
        else export.location.parent / f"{export.name}__compare-nnunet__{fingerprint}"
    )
    ensure_safe_destination(export.location, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.building-", dir=target.parent))
    started = time.time()
    limitations = [
        "Metrics are produced by the official nnU-Net v2 folder evaluator on binary foreground masks.",
        "Training/evaluation overlap cannot be independently verified from an nnU-Net model folder alone.",
        "Paired uncertainty treats exported cases as independent; tiled cases from one source image may be correlated.",
        "Dice is undefined when both reference and prediction are empty; finite support is reported separately "
        "from total cohort size.",
    ]
    if any(spec.upscale_factor != 1 for spec in specs):
        limitations.append(
            "Each model received inputs at its configured training-adapter scale; predicted class probabilities "
            "were area-averaged back to the canonical exported resolution before argmax and evaluation."
        )
    if "sahi" in model_backends.values():
        limitations.append(
            "nnU-Net predictions were reconstructed from source-coordinate SAHI tiles "
            "with feathered probability blending before full-image evaluation."
        )

    try:
        canonical_labels = temporary / "cohort" / "canonical" / "labels"
        _prepare_labels(cases, canonical_labels)
        _write_cohort(
            temporary / "evaluation-cohort.jsonl",
            cases,
            split,
            specs,
            projection=(
                "sahi-feathered-probability-area-pool-argmax"
                if set(model_backends.values()) == {"sahi"}
                else (
                    "per-model-see-settings"
                    if len(set(model_backends.values())) > 1
                    else "probability-area-pool-argmax"
                )
            ),
        )

        model_rows: dict[str, list[dict[str, Any]]] = {}
        ranking: list[dict[str, Any]] = []
        official_summaries: dict[str, str] = {}
        native_official_summaries: dict[str, str] = {}
        prediction_dirs: dict[str, Path] = {}
        model_inputs = _model_inputs_from_cases(cases)
        for model_index, spec in enumerate(specs):
            selected_backend = model_backends[spec.name]
            native_labels = temporary / "cohort" / "models" / spec.slug / "labels"
            _prepare_labels(cases, native_labels, upscale_factor=spec.upscale_factor)
            native_prediction_dir = temporary / "native-predictions" / spec.slug
            native_prediction_dir.mkdir(parents=True, exist_ok=True)
            prediction_dir = temporary / "predictions" / spec.slug
            prediction_dir.mkdir(parents=True, exist_ok=True)
            prediction_dirs[spec.name] = prediction_dir
            summary_path = temporary / f"{spec.slug}-metrics.json"
            native_summary_path = temporary / f"{spec.slug}-native-metrics.json"
            print(
                f"Evaluating {spec.name!r} with official nnU-Net v2 "
                f"(folds={spec.folds}, checkpoint={spec.checkpoint}, "
                f"input_scale={spec.upscale_factor}x)"
            )
            prediction_result = spec.predict(
                model_inputs,
                device=spec.device or device,
                progress=progress,
                _keep_native=True,
                inference=selected_backend,
                resolution=resolution,
                **sahi_options,
            )
            inference_seconds = prediction_result.inference_seconds
            _write_semantic_prediction_masks(
                prediction_result.records,
                prediction_dir,
                native_prediction_dir,
            )
            _assert_exact_predictions(native_prediction_dir, cases, spec.name)
            _run_command(
                [
                    "nnUNetv2_evaluate_folder",
                    str(native_labels),
                    str(native_prediction_dir),
                    "-djfile",
                    str(spec.model_folder / "dataset.json"),
                    "-pfile",
                    str(spec.model_folder / "plans.json"),
                    "-o",
                    str(native_summary_path),
                    "-np",
                    str(spec.workers),
                ]
            )
            native_summary = _load_official_summary(native_summary_path, spec.name)
            native_rows = _per_case_rows(native_summary, cases, spec.name)
            _assert_exact_predictions(prediction_dir, cases, spec.name)
            shutil.rmtree(native_prediction_dir)
            _run_command(
                [
                    "nnUNetv2_evaluate_folder",
                    str(canonical_labels),
                    str(prediction_dir),
                    "-djfile",
                    str(spec.model_folder / "dataset.json"),
                    "-pfile",
                    str(spec.model_folder / "plans.json"),
                    "-o",
                    str(summary_path),
                    "-np",
                    str(spec.workers),
                ]
            )
            summary = _load_official_summary(summary_path, spec.name)
            rows = _per_case_rows(summary, cases, spec.name)
            model_rows[spec.name] = rows
            aggregate = summary["foreground_mean"]
            dice = _metric(aggregate, "Dice")
            iou = _metric(aggregate, "IoU")
            native_aggregate = native_summary["foreground_mean"]
            native_dice = _metric(native_aggregate, "Dice")
            native_iou = _metric(native_aggregate, "IoU")
            finite_support = sum(math.isfinite(row["dice"]) for row in rows)
            native_finite_support = sum(
                math.isfinite(row["dice"]) for row in native_rows
            )
            ci_low, ci_high = _bootstrap_interval(
                [row["dice"] for row in rows],
                resamples=bootstrap_resamples,
                seed=seed + model_index,
            )
            ranking.append(
                {
                    "model": spec.name,
                    "backend": selected_backend,
                    "adapter": "nnunetv2-official",
                    "metric": "canonical.foreground_mean.Dice",
                    "score": dice,
                    "dice": dice,
                    "iou": iou,
                    "native_dice": native_dice,
                    "native_iou": native_iou,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "support_cases": finite_support,
                    "native_support_cases": native_finite_support,
                    "cohort_cases": len(cases),
                    "undefined_cases": len(cases) - finite_support,
                    "native_undefined_cases": len(cases) - native_finite_support,
                    "folds": ",".join(spec.folds),
                    "checkpoint": spec.checkpoint,
                    "checkpoint_sha256": spec.checkpoint_sha256,
                    "model_sha256": spec.digest,
                    "model_folder": str(spec.model_folder),
                    "upscale_factor": spec.upscale_factor,
                    "evaluation_resolution": "canonical-export",
                    "projection": (
                        "sahi-feathered-probability-area-pool-argmax"
                        if selected_backend == "sahi"
                        else "probability-area-pool-argmax"
                    ),
                    "native_evaluation_resolution": f"model-input-{spec.upscale_factor}x",
                    "cohort_fingerprint": cohort_fingerprint,
                    "inference_seconds": inference_seconds,
                    "throughput_cases_per_second": (
                        len(cases) / inference_seconds if inference_seconds > 0 else None
                    ),
                }
            )
            sanitized = _sanitize_official_summary(
                summary,
                cases,
                split=split,
                model_slug=spec.slug,
                keep_predictions=keep_predictions,
            )
            sanitized["evaluation_resolution"] = "canonical-export"
            sanitized["projection"] = (
                "sahi-feathered-probability-area-pool-argmax"
                if selected_backend == "sahi"
                else "probability-area-pool-argmax"
            )
            write_json(summary_path, sanitized)
            official_summaries[spec.name] = str(summary_path.relative_to(temporary))
            sanitized_native = _sanitize_official_summary(
                native_summary,
                cases,
                split=split,
                model_slug=spec.slug,
                keep_predictions=False,
            )
            sanitized_native["evaluation_resolution"] = f"model-input-{spec.upscale_factor}x"
            sanitized_native["projection"] = "none"
            write_json(native_summary_path, sanitized_native)
            native_official_summaries[spec.name] = str(
                native_summary_path.relative_to(temporary)
            )

        ranking.sort(key=lambda row: (-_sortable_score(row["score"]), row["model"]))
        for rank, row in enumerate(ranking, start=1):
            row["rank"] = rank
        paired = _paired_statistics(
            model_rows,
            baseline,
            resamples=bootstrap_resamples,
            seed=seed,
        )
        per_case = [row for name in [spec.name for spec in specs] for row in model_rows[name]]
        write_csv(temporary / "ranking.csv", ranking)
        write_csv(temporary / "per-case.csv", per_case)
        write_csv(temporary / "paired-statistics.csv", paired)

        figure_paths: list[str] = []
        qualitative_paths: list[str] = []
        if visualize:
            figure_paths = _render_ranking(temporary, ranking)
            qualitative_paths = _render_qualitative(
                temporary,
                cases,
                prediction_dirs,
                model_rows,
                seed=seed,
            )

        if not keep_predictions:
            shutil.rmtree(temporary / "predictions", ignore_errors=True)
        shutil.rmtree(temporary / "native-predictions", ignore_errors=True)
        shutil.rmtree(temporary / "cohort", ignore_errors=True)

        manifest = {
            "schema": 4,
            "kind": "semantic-mask-model-comparison",
            "backend": resolved_settings["backend"],
            "adapter": "nnunetv2-official",
            "dataset": {
                "name": export.name,
                "location": str(export.location),
                "format": export.manifest.get("format"),
                "class_handling": export.manifest.get("class_handling"),
            },
            "cohort_fingerprint": cohort_fingerprint,
            "cohort_verified": True,
            "split": split,
            "cases": len(cases),
            "baseline": baseline,
            "settings": resolved_settings,
            "settings_fingerprint": fingerprint,
            "ranking": ranking,
            "paired_statistics": paired,
            "official_summaries": official_summaries,
            "native_official_summaries": native_official_summaries,
            "limitations": limitations,
            "figures": figure_paths,
            "qualitative": qualitative_paths,
            "environment": environment_snapshot(),
            "started_at_unix": started,
            "completed_at_unix": time.time(),
        }
        write_json(temporary / "semantic-model-comparison.json", manifest)
        temporary.replace(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(
        f"Semantic model comparison complete: {target}\n"
        f"Cohort verified: yes; cases: {len(cases)}; baseline: {baseline}"
    )
    return SemanticComparisonResult(
        location=target,
        ranking=tuple(ranking),
        cohort_fingerprint=cohort_fingerprint,
        cohort_verified=True,
        split=split,
        baseline=baseline,
        settings=resolved_settings,
        limitations=tuple(limitations),
    )


def compare_semantic_models(
    export: "Dataset",
    models: ModelCollection,
    *,
    split: str,
    baseline: str | None,
    device: str | None,
    inference: str | None,
    confidence: float,
    postprocess: float,
    bootstrap_resamples: int,
    seed: int,
    keep_predictions: bool,
    visualize: bool,
    progress: bool,
    destination: str | Path | None,
    resolution: int = 480,
    comparison_unit: str = "model",
    sahi_slice_height: int | None = None,
    sahi_slice_width: int | None = None,
    sahi_overlap: float = 0.2,
    sahi_overlap_height_ratio: float | None = None,
    sahi_overlap_width_ratio: float | None = None,
    sahi_postprocess_type: str = "GREEDYNMM",
    sahi_postprocess_match_metric: str = "IOS",
    sahi_postprocess_class_agnostic: bool = False,
    sahi_model_type: str = "ultralytics",
) -> SemanticComparisonResult:
    """Compare instance and semantic segmenters in one binary mask space."""

    split = normalize_split(split)
    if split not in export.splits:
        raise ValueError(
            f"Unknown semantic-mask split {split!r}; available splits are {export.splits}"
        )
    if baseline is None:
        baseline = models.models[0].name
    if baseline not in models.names:
        raise ValueError(f"Unknown baseline {baseline!r}")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    if not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1:
        raise ValueError("confidence must be finite and in [0, 1]")
    if not math.isfinite(float(postprocess)) or not 0 <= float(postprocess) <= 1:
        raise ValueError("postprocess must be finite and in [0, 1]")
    if inference is not None and inference not in {"native", "sahi"}:
        raise ValueError("inference must be 'native', 'sahi', or None; 'auto' was removed")
    if comparison_unit not in {"model", "system"}:
        raise ValueError("comparison_unit must be 'model' or 'system'")
    selected_backends = {inference or model.inference for model in models}
    if comparison_unit == "model" and len(selected_backends) != 1:
        raise ValueError("Mixed native/SAHI configurations require comparison_unit='system'")
    sahi_options = {
        "sahi_slice_height": sahi_slice_height,
        "sahi_slice_width": sahi_slice_width,
        "sahi_overlap": sahi_overlap,
        "sahi_overlap_height_ratio": sahi_overlap_height_ratio,
        "sahi_overlap_width_ratio": sahi_overlap_width_ratio,
        "sahi_postprocess_type": sahi_postprocess_type,
        "sahi_postprocess_match_metric": sahi_postprocess_match_metric,
        "sahi_postprocess_class_agnostic": sahi_postprocess_class_agnostic,
        "sahi_model_type": sahi_model_type,
    }
    effective_sahi_options = {
        key: value for key, value in sahi_options.items() if value is not None
    }
    resolved_sahi_by_model = {
        model.name: resolve_sahi_settings(
            {**model.settings, **effective_sahi_options},
            resolution=model.resolution or resolution,
        ).as_dict()
        for model in models
        if (inference or model.inference) == "sahi"
    }
    if comparison_unit == "model" and len(
        {
            tuple(sorted(configuration.items()))
            for configuration in resolved_sahi_by_model.values()
        }
    ) > 1:
        raise ValueError(
            "Different SAHI tile configurations require comparison_unit='system'"
        )
    incompatible = [
        {"model": model.name, "kind": model.kind, "task": model.task}
        for model in models
        if model.task not in {"segment", "semantic_segment"}
    ]
    if incompatible:
        raise DatasetValidationError(
            ValidationIssue(
                "Models cannot be projected to the binary semantic comparison space",
                value=incompatible,
                expected="Ultralytics task='segment' or semantic task='semantic_segment'",
                suggestion="set the correct Model(task=...) or compare incompatible tasks separately",
            )
        )

    cases, cohort_fingerprint = _freeze_cohort(export, split)
    resolved_settings = {
        "backend": "common-semantic-mask",
        "report_schema": 5,
        "comparison_space": "semantic",
        "canonical_projection": "binary-foreground-union",
        "split": split,
        "baseline": baseline,
        "device": device,
        "inference": inference,
        "resolution": resolution,
        "comparison_unit": comparison_unit,
        "sahi_models": resolved_sahi_by_model,
        "confidence": float(confidence),
        "postprocess": float(postprocess),
        "bootstrap_resamples": bootstrap_resamples,
        "seed": seed,
        "keep_predictions": keep_predictions,
        "visualize": visualize,
        "models": [
            {
                **model.describe(),
                "semantic_projection": (
                    "polygon-foreground-union"
                    if model.task == "segment"
                    else "native-semantic-mask"
                ),
            }
            for model in models
        ],
    }
    fingerprint = settings_fingerprint(
        {**to_jsonable(resolved_settings), "cohort_fingerprint": cohort_fingerprint}
    )
    target = (
        Path(destination).expanduser().resolve()
        if destination is not None
        else export.location.parent / f"{export.name}__compare-semantic__{fingerprint}"
    )
    ensure_safe_destination(export.location, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.building-", dir=target.parent))
    started = time.time()
    limitations = [
        "All models are evaluated as binary foreground at the canonical semantic-mask export resolution.",
        "YOLO instance classes and identities are discarded by unioning every retained prediction polygon.",
        "YOLO confidence and postprocessing thresholds do not alter native semantic-model predictions.",
        "Dice is undefined when both reference and prediction are empty; finite support is reported separately.",
        "Training/evaluation overlap cannot be independently verified for models without training provenance.",
    ]

    try:
        _write_cohort(
            temporary / "evaluation-cohort.jsonl",
            cases,
            split,
            list(models.models),
            projection="binary-foreground-union",
        )
        model_inputs = _model_inputs_from_cases(cases)
        model_rows: dict[str, list[dict[str, Any]]] = {}
        prediction_dirs: dict[str, Path] = {}
        ranking: list[dict[str, Any]] = []
        for model_index, model in enumerate(models):
            prediction_dir = temporary / "predictions" / model.slug
            prediction_dir.mkdir(parents=True, exist_ok=True)
            prediction_dirs[model.name] = prediction_dir
            predict_options: dict[str, Any] = {
                "device": model.device or device,
                "progress": progress,
                "inference": inference or model.inference,
                "resolution": model.resolution or resolution,
                **sahi_options,
            }
            if model.kind == "ultralytics":
                predict_options.update(
                    {
                        "confidence": float(confidence),
                        "postprocess": float(postprocess),
                    }
                )
            prediction_result = model.predict(model_inputs, **predict_options)
            projected, projection = _project_semantic_predictions(
                prediction_result.records,
                prediction_result.task,
                cases,
                model.name,
                confidence=float(confidence),
            )
            _write_semantic_prediction_masks(projected, prediction_dir)
            _assert_exact_predictions(prediction_dir, cases, model.name)
            rows = _sample_metric_rows(cases, prediction_dir, model.name)
            model_rows[model.name] = rows
            finite_dice = [row["dice"] for row in rows if math.isfinite(row["dice"])]
            finite_iou = [row["iou"] for row in rows if math.isfinite(row["iou"])]
            dice = float(np.mean(finite_dice)) if finite_dice else math.nan
            iou = float(np.mean(finite_iou)) if finite_iou else math.nan
            tp = sum(int(row["tp"]) for row in rows)
            fp = sum(int(row["fp"]) for row in rows)
            fn = sum(int(row["fn"]) for row in rows)
            micro_dice_denominator = 2 * tp + fp + fn
            micro_iou_denominator = tp + fp + fn
            ci_low, ci_high = _bootstrap_interval(
                [row["dice"] for row in rows],
                resamples=bootstrap_resamples,
                seed=seed + model_index,
            )
            ranking.append(
                {
                    "model": model.name,
                    "model_kind": model.kind,
                    "native_task": prediction_result.task,
                    "backend": prediction_result.backend,
                    "metric": "canonical.macro_foreground.Dice",
                    "score": dice,
                    "dice": dice,
                    "iou": iou,
                    "micro_dice": (
                        2 * tp / micro_dice_denominator
                        if micro_dice_denominator
                        else math.nan
                    ),
                    "micro_iou": (
                        tp / micro_iou_denominator
                        if micro_iou_denominator
                        else math.nan
                    ),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "support_cases": len(finite_dice),
                    "cohort_cases": len(cases),
                    "undefined_cases": len(cases) - len(finite_dice),
                    "evaluation_resolution": "canonical-export",
                    "projection": projection,
                    "cohort_fingerprint": cohort_fingerprint,
                    "model_sha256": model.digest,
                    "inference_seconds": prediction_result.inference_seconds,
                    "throughput_cases_per_second": (
                        len(cases) / prediction_result.inference_seconds
                        if prediction_result.inference_seconds > 0
                        else None
                    ),
                }
            )

        ranking.sort(key=lambda row: (-_sortable_score(row["score"]), row["model"]))
        for rank, row in enumerate(ranking, start=1):
            row["rank"] = rank
        paired = _paired_statistics(
            model_rows,
            baseline,
            resamples=bootstrap_resamples,
            seed=seed,
        )
        per_case = [
            row
            for model in models
            for row in model_rows[model.name]
        ]
        write_csv(temporary / "ranking.csv", ranking)
        write_csv(temporary / "per-case.csv", per_case)
        write_csv(temporary / "paired-statistics.csv", paired)
        figure_paths: list[str] = []
        qualitative_paths: list[str] = []
        if visualize:
            figure_paths = _render_ranking(
                temporary,
                ranking,
                xlabel="Canonical binary foreground mean Dice",
                title="Semantic-space model comparison",
            )
            qualitative_paths = _render_qualitative(
                temporary,
                cases,
                prediction_dirs,
                model_rows,
                seed=seed,
            )
        if not keep_predictions:
            shutil.rmtree(temporary / "predictions", ignore_errors=True)

        manifest = {
            "schema": 5,
            "kind": "semantic-mask-model-comparison",
            "backend": "common-semantic-mask",
            "negotiated_comparison_space": "semantic",
            "dataset": {
                "name": export.name,
                "location": str(export.location),
                "format": export.manifest.get("format"),
                "class_handling": export.manifest.get("class_handling"),
            },
            "cohort_fingerprint": cohort_fingerprint,
            "cohort_verified": True,
            "split": split,
            "cases": len(cases),
            "baseline": baseline,
            "settings": resolved_settings,
            "settings_fingerprint": fingerprint,
            "ranking": ranking,
            "paired_statistics": paired,
            "limitations": limitations,
            "figures": figure_paths,
            "qualitative": qualitative_paths,
            "environment": environment_snapshot(),
            "started_at_unix": started,
            "completed_at_unix": time.time(),
        }
        write_json(temporary / "semantic-model-comparison.json", manifest)
        temporary.replace(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(
        f"Semantic model comparison complete: {target}\n"
        f"Cohort verified: yes; cases: {len(cases)}; baseline: {baseline}; "
        "comparison space: semantic"
    )
    return SemanticComparisonResult(
        location=target,
        ranking=tuple(ranking),
        cohort_fingerprint=cohort_fingerprint,
        cohort_verified=True,
        split=split,
        baseline=baseline,
        settings=resolved_settings,
        limitations=tuple(limitations),
    )


def _project_semantic_predictions(
    records: tuple[ImagePrediction, ...],
    task: str,
    cases: list[_SemanticCase],
    model_name: str,
    *,
    confidence: float,
) -> tuple[tuple[ImagePrediction, ...], str]:
    expected_ids = [case.case_id for case in cases]
    actual_ids = [record.image_id for record in records]
    if actual_ids != expected_ids:
        raise DatasetValidationError(
            ValidationIssue(
                "Model did not return the exact frozen semantic cohort in order",
                source=model_name,
                value={"expected": expected_ids[:5], "actual": actual_ids[:5]},
                expected=f"{len(expected_ids)} ordered predictions",
            )
        )
    projected: list[ImagePrediction] = []
    if task == "semantic_segment":
        for record, case in zip(records, cases):
            if (record.width, record.height) != (case.width, case.height):
                raise DatasetValidationError(
                    f"Semantic prediction dimensions {(record.width, record.height)} do not "
                    f"match {case.relative_path}: {(case.width, case.height)}"
                )
            if record.mask is None:
                raise DatasetValidationError(
                    f"Semantic prediction {model_name}/{record.image_id} has no mask"
                )
            mask = np.asarray(record.mask) > 0
            if mask.shape != (case.height, case.width):
                raise DatasetValidationError(
                    f"Semantic prediction shape {mask.shape} does not match "
                    f"{case.relative_path}: {(case.height, case.width)}"
                )
            projected.append(
                ImagePrediction(
                    image_id=record.image_id,
                    image_path=record.image_path,
                    relative_path=record.relative_path,
                    width=record.width,
                    height=record.height,
                    mask=mask,
                    metadata={**record.metadata, "semantic_projection": "native-semantic-mask"},
                )
            )
        return tuple(projected), "native-semantic-mask"
    if task != "segment":
        raise DatasetValidationError(
            ValidationIssue(
                "Prediction task has no semantic-mask projection",
                source=model_name,
                value=task,
                expected="segment or semantic_segment",
            )
        )
    for record, case in zip(records, cases):
        if (record.width, record.height) != (case.width, case.height):
            raise DatasetValidationError(
                f"YOLO prediction dimensions {(record.width, record.height)} do not "
                f"match {case.relative_path}: {(case.width, case.height)}"
            )
        canvas = Image.new("L", (case.width, case.height), 0)
        draw = ImageDraw.Draw(canvas)
        for index, prediction in enumerate(record.objects, start=1):
            if float(prediction.score) < confidence:
                continue
            polygons = prediction.polygons or (
                [prediction.polygon] if prediction.polygon is not None else []
            )
            if not polygons or any(len(polygon) < 3 for polygon in polygons):
                raise DatasetValidationError(
                    ValidationIssue(
                        "YOLO segmentation prediction has no usable polygon",
                        source=f"{model_name}:{case.relative_path}",
                        line=index,
                        expected="at least three polygon points",
                    )
                )
            if any(
                not math.isfinite(float(x)) or not math.isfinite(float(y))
                for polygon in polygons
                for x, y in polygon
            ):
                raise DatasetValidationError(
                    ValidationIssue(
                        "YOLO segmentation polygon contains non-finite coordinates",
                        source=f"{model_name}:{case.relative_path}",
                        line=index,
                    )
                )
            for polygon in polygons:
                draw.polygon([(float(x), float(y)) for x, y in polygon], fill=1)
        projected.append(
            ImagePrediction(
                image_id=record.image_id,
                image_path=record.image_path,
                relative_path=record.relative_path,
                width=record.width,
                height=record.height,
                mask=np.asarray(canvas, dtype=np.uint8) > 0,
                metadata={"semantic_projection": "polygon-foreground-union"},
            )
        )
    return tuple(projected), "polygon-foreground-union"


def predict_nnunet_model(
    model: Model,
    inputs: tuple[ModelInput, ...],
    *,
    device: str,
    progress: bool,
    keep_native: bool,
    inference: str = "native",
    resolution: int = 480,
    settings: dict[str, Any] | None = None,
) -> tuple[ImagePrediction, ...]:
    """Run the official nnU-Net adapter for :meth:`Model.predict`."""

    if model.kind != "nnunet":
        raise TypeError("predict_nnunet_model requires Model(kind='nnunet')")
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("nnU-Net device must be 'cpu', 'cuda', or 'mps'")
    if inference not in {"native", "sahi"}:
        raise ValueError("inference must be 'native' or 'sahi'; 'auto' was removed")
    _require_official_commands("nnUNetv2_predict_from_modelfolder")
    if inference == "sahi":
        return _predict_nnunet_sahi(
            model,
            inputs,
            device=device,
            progress=progress,
            keep_native=keep_native,
            resolution=resolution,
            settings=dict(settings or {}),
        )
    with tempfile.TemporaryDirectory(prefix="dataset-fixer-nnunet-predict-") as temporary:
        root = Path(temporary)
        prepared_images = root / "cohort" / "models" / model.slug / "images"
        native_predictions = root / "native-predictions" / model.slug
        native_predictions.mkdir(parents=True, exist_ok=True)
        _prepare_model_inputs(
            inputs,
            prepared_images,
            upscale_factor=model.upscale_factor,
            progress=progress,
            model_name=model.name,
        )
        if all(value.mask_path is not None for value in inputs):
            _prepare_model_input_labels(
                inputs,
                root / "cohort" / "canonical" / "labels",
            )
        _run_command(
            [
                "nnUNetv2_predict_from_modelfolder",
                "-i",
                str(prepared_images),
                "-o",
                str(native_predictions),
                "-m",
                str(model.model_folder),
                "-f",
                *model.folds,
                "-chk",
                model.checkpoint,
                "-device",
                device,
                "-npp",
                str(model.workers),
                "-nps",
                str(model.workers),
                "--save_probabilities",
            ]
        )
        _assert_exact_model_predictions(native_predictions, inputs, model.name)
        records: list[ImagePrediction] = []
        for value in inputs:
            native_path = native_predictions / f"{value.image_id}.png"
            with Image.open(native_path) as opened:
                native_mask = np.asarray(opened.convert("L")) > 0
            expected_native = (
                value.height * model.upscale_factor,
                value.width * model.upscale_factor,
            )
            if native_mask.shape != expected_native:
                raise DatasetValidationError(
                    ValidationIssue(
                        "nnU-Net native prediction dimensions do not match its input adapter",
                        source=f"{model.name}/{value.image_id}",
                        value=native_mask.shape,
                        expected=str(expected_native),
                    )
                )
            mask = _canonical_mask_from_probabilities(
                native_predictions / f"{value.image_id}.npz",
                image_id=value.image_id,
                width=value.width,
                height=value.height,
                model_name=model.name,
                upscale_factor=model.upscale_factor,
            )
            records.append(
                ImagePrediction(
                    image_id=value.image_id,
                    image_path=value.image_path,
                    relative_path=value.relative_path,
                    width=value.width,
                    height=value.height,
                    mask=mask,
                    native_mask=native_mask if keep_native else None,
                    metadata={
                        "backend": "native",
                        "adapter": "nnunetv2-official",
                        "upscale_factor": model.upscale_factor,
                        "projection": "probability-area-pool-argmax",
                    },
                )
            )
        return tuple(records)


def _predict_nnunet_sahi(
    model: Model,
    inputs: tuple[ModelInput, ...],
    *,
    device: str,
    progress: bool,
    keep_native: bool,
    resolution: int,
    settings: dict[str, Any],
) -> tuple[ImagePrediction, ...]:
    from .sahi_support import (
        build_tile_manifest,
        resolve_sahi_settings,
        stitch_probability_tiles,
    )

    resolved = resolve_sahi_settings(settings, resolution=resolution)
    records: list[ImagePrediction] = []
    iterator = tqdm(
        inputs,
        desc=f"{model.name} SAHI nnU-Net",
        unit="image",
        disable=not progress,
    )
    with tempfile.TemporaryDirectory(prefix="dataset-fixer-nnunet-sahi-") as temporary:
        root = Path(temporary)
        for value in iterator:
            with Image.open(value.image_path) as opened:
                source_image = opened.convert("RGB")
            if source_image.size != (value.width, value.height):
                raise DatasetValidationError(
                    f"Prediction input dimensions changed while slicing {value.image_path}"
                )
            manifest = build_tile_manifest(
                width=value.width,
                height=value.height,
                settings=resolved,
            )
            image_dir = root / "images" / value.image_id
            prediction_dir = root / "predictions" / value.image_id
            image_dir.mkdir(parents=True, exist_ok=True)
            prediction_dir.mkdir(parents=True, exist_ok=True)
            case_ids: dict[int, str] = {}
            for tile in manifest:
                case_id = f"{value.image_id}__tile_{tile.index:06d}"
                case_ids[tile.index] = case_id
                image = source_image.crop(tile.box)
                if model.upscale_factor != 1:
                    image = image.resize(
                        (
                            tile.width * model.upscale_factor,
                            tile.height * model.upscale_factor,
                        ),
                        Image.Resampling.BICUBIC,
                    )
                image.save(image_dir / f"{case_id}_0000.png", format="PNG")
            _run_command(
                [
                    "nnUNetv2_predict_from_modelfolder",
                    "-i",
                    str(image_dir),
                    "-o",
                    str(prediction_dir),
                    "-m",
                    str(model.model_folder),
                    "-f",
                    *model.folds,
                    "-chk",
                    model.checkpoint,
                    "-device",
                    device,
                    "-npp",
                    str(model.workers),
                    "-nps",
                    str(model.workers),
                    "--save_probabilities",
                ]
            )
            expected_pngs = {f"{case_id}.png" for case_id in case_ids.values()}
            actual_pngs = {
                path.name for path in prediction_dir.glob("*.png") if path.is_file()
            }
            if actual_pngs != expected_pngs:
                raise DatasetValidationError(
                    ValidationIssue(
                        "nnU-Net SAHI tile predictions are incomplete",
                        source=f"{model.name}/{value.image_id}",
                        value={
                            "unexpected": sorted(actual_pngs - expected_pngs),
                            "missing": sorted(expected_pngs - actual_pngs),
                        },
                        expected=f"exactly {len(expected_pngs)} tile predictions",
                    )
                )
            tile_probabilities = [
                (
                    tile,
                    _load_nnunet_tile_probabilities(
                        prediction_dir / f"{case_ids[tile.index]}.npz",
                        expected_shape=(
                            tile.height * model.upscale_factor,
                            tile.width * model.upscale_factor,
                        ),
                        source=f"{model.name}/{value.image_id}/tile-{tile.index}",
                    ),
                )
                for tile in manifest
            ]
            native_probabilities = stitch_probability_tiles(
                width=value.width,
                height=value.height,
                tiles=tile_probabilities,
                scale=model.upscale_factor,
            )
            native_mask = np.argmax(native_probabilities, axis=0).astype(np.uint8)
            if model.upscale_factor == 1:
                canonical_probabilities = native_probabilities
            else:
                canonical_probabilities = native_probabilities.reshape(
                    2,
                    value.height,
                    model.upscale_factor,
                    value.width,
                    model.upscale_factor,
                ).mean(axis=(2, 4))
            mask = np.argmax(canonical_probabilities, axis=0).astype(np.uint8)
            records.append(
                ImagePrediction(
                    image_id=value.image_id,
                    image_path=value.image_path,
                    relative_path=value.relative_path,
                    width=value.width,
                    height=value.height,
                    mask=mask,
                    native_mask=native_mask if keep_native else None,
                    metadata={
                        "backend": "sahi",
                        "adapter": "nnunetv2-official",
                        "upscale_factor": model.upscale_factor,
                        "projection": "sahi-feathered-probability-area-pool-argmax",
                        **resolved.as_dict(),
                    },
                )
            )
            shutil.rmtree(image_dir)
            shutil.rmtree(prediction_dir)
    return tuple(records)


def _load_nnunet_tile_probabilities(
    path: Path,
    *,
    expected_shape: tuple[int, int],
    source: str,
) -> np.ndarray:
    try:
        with np.load(path) as archive:
            probabilities = np.asarray(archive["probabilities"], dtype=np.float32)
    except (OSError, KeyError, ValueError) as exc:
        raise DatasetValidationError(
            f"Unreadable nnU-Net SAHI probability tile {path}: {exc}"
        ) from exc
    if (
        probabilities.ndim < 3
        or probabilities.shape[0] != 2
        or probabilities.shape[-2:] != expected_shape
        or math.prod(probabilities.shape[1:-2]) != 1
    ):
        raise DatasetValidationError(
            ValidationIssue(
                "nnU-Net SAHI probability dimensions do not match the tile adapter",
                source=source,
                value=probabilities.shape,
                expected=f"(2, ..., {expected_shape[0]}, {expected_shape[1]})",
            )
        )
    if not np.all(np.isfinite(probabilities)):
        raise DatasetValidationError(
            ValidationIssue(
                "nnU-Net SAHI probability tile contains non-finite values",
                source=source,
            )
        )
    return probabilities.reshape(2, *expected_shape)


def visualize_nnunet_models(
    cohort: ModelCollection,
    *,
    split: str,
    samples: int,
    examples_per_row: int,
    include_empty: bool,
    seed: int,
    panel_size: float,
    model_title_length: int,
    image_title_length: int,
    progress: bool,
    destination: str | Path | None,
) -> Any:
    """Run sampled official nnU-Net inference and render model masks."""

    export = cohort.source
    from .dataset import Dataset

    if not isinstance(export, Dataset) or export.format != "semantic_masks":
        raise TypeError("Semantic visualization requires a bound semantic-mask Dataset")
    split = normalize_split(split)
    if split not in export.splits:
        raise ValueError(
            f"Unknown semantic-mask split {split!r}; "
            f"available splits are {export.splits}"
        )
    if samples <= 0:
        raise ValueError("samples must be positive")
    if examples_per_row <= 0:
        raise ValueError("examples_per_row must be positive")
    if not math.isfinite(panel_size) or panel_size <= 0:
        raise ValueError("panel_size must be a positive finite number")
    if model_title_length < 5:
        raise ValueError("model_title_length must be at least 5")
    if image_title_length < 5:
        raise ValueError("image_title_length must be at least 5")
    _require_official_commands("nnUNetv2_predict_from_modelfolder")

    cases, _ = _freeze_cohort(export, split)
    selected = _select_visual_cases(
        cases,
        samples=samples,
        include_empty=include_empty,
        seed=seed,
    )

    with tempfile.TemporaryDirectory(prefix="dataset-fixer-semantic-visualize-") as temporary:
        temporary_root = Path(temporary)
        prediction_dirs: dict[str, Path] = {}
        rows_by_model: dict[str, list[dict[str, Any]]] = {}
        model_inputs = _model_inputs_from_cases(selected)
        for spec in cohort.models:
            predictions = temporary_root / "predictions" / spec.slug
            predictions.mkdir(parents=True, exist_ok=True)
            result = spec.predict(
                model_inputs,
                device=spec.device or "cuda",
                progress=progress,
            )
            _write_semantic_prediction_masks(
                result.records,
                predictions,
            )
            prediction_dirs[spec.name] = predictions
            rows_by_model[spec.name] = _sample_metric_rows(
                selected,
                predictions,
                spec.name,
            )

        figure = _render_semantic_grid(
            selected,
            prediction_dirs,
            rows_by_model,
            examples_per_row=examples_per_row,
            panel_size=panel_size,
            model_title_length=model_title_length,
            image_title_length=image_title_length,
        )
        if destination is not None:
            output = _visualization_destination(destination)
            output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(
                output,
                dpi=180,
                bbox_inches="tight",
                facecolor="white",
            )
        return figure


def _parse_models(
    models: Any,
    *,
    default_folds: tuple[int | str, ...],
    default_checkpoint: str,
) -> list[Model]:
    collection = Model.load_many(
        models,
        kind="nnunet",
        folds=default_folds,
        checkpoint=default_checkpoint,
    )
    incompatible = [model.name for model in collection if model.kind != "nnunet"]
    if incompatible:
        raise DatasetValidationError(
            ValidationIssue(
                "Semantic-mask comparison requires official nnU-Net models",
                value=incompatible,
            )
        )
    return list(collection.models)


def _require_official_commands(*commands: str) -> None:
    missing = [
        command
        for command in commands
        if shutil.which(command) is None
    ]
    if missing:
        raise ImportError(
            "Official nnU-Net v2 commands are unavailable: "
            f"{', '.join(missing)}. Install the pinned notebook dependency with "
            "`pip install nnunetv2==2.8.1`."
        )


def _freeze_cohort(
    export: "Dataset",
    split: str,
) -> tuple[list[_SemanticCase], str]:
    image_root = export.image_dirs[split]
    mask_root = export.mask_dirs[split]
    if not image_root.is_dir() or not mask_root.is_dir():
        raise DatasetValidationError(
            ValidationIssue(
                "Semantic-mask split directories are missing",
                source=split,
                value={"images": str(image_root), "masks": str(mask_root)},
            )
        )
    images = sorted(
        path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise DatasetValidationError(f"No images found in semantic-mask split {split!r}")

    cases: list[_SemanticCase] = []
    expected_masks: set[Path] = set()
    digest = hashlib.sha256()
    digest.update(f"semantic-mask-cohort-v2:{split}:canonical-export".encode("utf-8"))
    for index, image_path in enumerate(images):
        relative = image_path.relative_to(image_root)
        mask_path = mask_root / relative.with_suffix(".png")
        expected_masks.add(mask_path.resolve())
        if not mask_path.is_file():
            raise DatasetValidationError(
                ValidationIssue(
                    "Semantic mask is missing for evaluation image",
                    source=str(image_path),
                    expected=str(mask_path),
                )
            )
        with Image.open(image_path) as opened_image, Image.open(mask_path) as opened_mask:
            image = opened_image.convert("RGB")
            mask = opened_mask.convert("L")
            if image.size != mask.size:
                raise DatasetValidationError(
                    f"Semantic mask dimensions {mask.size} do not match image dimensions {image.size}: {mask_path}"
                )
            values = set(mask.getdata())
            if not values <= {0, 1, 255}:
                raise DatasetValidationError(
                    f"Semantic mask contains values outside 0/1/255: {mask_path}: {sorted(values)[:10]}"
                )
            width, height = image.size
        image_sha = sha256_file(image_path)
        mask_sha = sha256_file(mask_path)
        case = _SemanticCase(
            case_id=f"{split}_{index:06d}",
            relative_path=relative,
            image_path=image_path,
            mask_path=mask_path,
            width=width,
            height=height,
            image_sha256=image_sha,
            mask_sha256=mask_sha,
        )
        cases.append(case)
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(image_sha.encode("ascii"))
        digest.update(mask_sha.encode("ascii"))
    actual_masks = {
        path.resolve()
        for path in mask_root.rglob("*.png")
        if path.is_file()
    }
    if actual_masks != expected_masks:
        raise DatasetValidationError(
            ValidationIssue(
                "Semantic-mask split contains orphan or missing masks",
                source=split,
                value={
                    "unexpected": [str(path) for path in sorted(actual_masks - expected_masks)],
                    "missing": [str(path) for path in sorted(expected_masks - actual_masks)],
                },
            )
        )
    return cases, digest.hexdigest()


def _model_inputs_from_cases(
    cases: list[_SemanticCase],
) -> tuple[ModelInput, ...]:
    return tuple(
        ModelInput(
            image_id=case.case_id,
            image_path=case.image_path,
            width=case.width,
            height=case.height,
            relative_path=case.relative_path.as_posix(),
            mask_path=case.mask_path,
        )
        for case in cases
    )


def _prepare_model_input_labels(
    inputs: tuple[ModelInput, ...],
    label_dir: Path,
) -> None:
    label_dir.mkdir(parents=True, exist_ok=True)
    for value in inputs:
        if value.mask_path is None:
            continue
        with Image.open(value.mask_path) as opened_mask:
            mask = opened_mask.convert("L").point(lambda pixel: 1 if pixel else 0)
        mask.save(label_dir / f"{value.image_id}.png", format="PNG", optimize=False)


def _prepare_model_inputs(
    inputs: tuple[ModelInput, ...],
    image_dir: Path,
    *,
    upscale_factor: int,
    progress: bool,
    model_name: str,
) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    iterator = tqdm(
        inputs,
        desc=f"Preparing {model_name} inputs ({upscale_factor}x)",
        unit="image",
        disable=not progress,
    )
    for value in iterator:
        with Image.open(value.image_path) as opened_image:
            image = opened_image.convert("RGB")
        if image.size != (value.width, value.height):
            raise DatasetValidationError(
                f"Prediction input dimensions changed while preparing {value.image_path}"
            )
        if upscale_factor != 1:
            image = image.resize(
                (value.width * upscale_factor, value.height * upscale_factor),
                Image.Resampling.BICUBIC,
            )
        image.save(image_dir / f"{value.image_id}_0000.png", format="PNG")


def _write_semantic_prediction_masks(
    records: tuple[ImagePrediction, ...],
    canonical_dir: Path,
    native_dir: Path | None = None,
) -> None:
    canonical_dir.mkdir(parents=True, exist_ok=True)
    if native_dir is not None:
        native_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        if record.mask is None:
            raise DatasetValidationError(
                f"Semantic prediction {record.image_id!r} has no canonical mask"
            )
        Image.fromarray(np.asarray(record.mask, dtype=np.uint8)).save(
            canonical_dir / f"{record.image_id}.png",
            format="PNG",
            optimize=False,
        )
        if native_dir is not None:
            if record.native_mask is None:
                raise DatasetValidationError(
                    f"Semantic prediction {record.image_id!r} has no native mask"
                )
            Image.fromarray(np.asarray(record.native_mask, dtype=np.uint8)).save(
                native_dir / f"{record.image_id}.png",
                format="PNG",
                optimize=False,
            )


def _assert_exact_model_predictions(
    prediction_dir: Path,
    inputs: tuple[ModelInput, ...],
    model_name: str,
) -> None:
    expected = {f"{value.image_id}.png" for value in inputs}
    actual = {path.name for path in prediction_dir.glob("*.png") if path.is_file()}
    if actual != expected:
        raise DatasetValidationError(
            ValidationIssue(
                "nnU-Net predictions do not match the requested images",
                source=model_name,
                value={
                    "unexpected": sorted(actual - expected)[:20],
                    "missing": sorted(expected - actual)[:20],
                },
                expected=f"exactly {len(expected)} prediction masks",
            )
        )


def _prepare_labels(
    cases: list[_SemanticCase],
    label_dir: Path,
    *,
    upscale_factor: int = 1,
) -> None:
    label_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        with Image.open(case.mask_path) as opened_mask:
            mask = opened_mask.convert("L").point(lambda value: 1 if value else 0)
        if upscale_factor != 1:
            mask = mask.resize(
                (case.width * upscale_factor, case.height * upscale_factor),
                Image.Resampling.NEAREST,
            )
        mask.save(label_dir / f"{case.case_id}.png", format="PNG", optimize=False)


def _prepare_images(
    cases: list[_SemanticCase],
    image_dir: Path,
    *,
    upscale_factor: int,
    progress: bool,
    model_name: str,
) -> None:
    _prepare_model_inputs(
        _model_inputs_from_cases(cases),
        image_dir,
        upscale_factor=upscale_factor,
        progress=progress,
        model_name=model_name,
    )


def _write_cohort(
    path: Path,
    cases: list[_SemanticCase],
    split: str,
    specs: list[Model],
    *,
    projection: str = "probability-area-pool-argmax",
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(
                json.dumps(
                    {
                        "case_id": case.case_id,
                        "split": split,
                        "relative_path": case.relative_path.as_posix(),
                        "source_image": str(case.image_path),
                        "source_mask": str(case.mask_path),
                        "width": case.width,
                        "height": case.height,
                        "evaluation_resolution": "canonical-export",
                        "projection": projection,
                        "model_inputs": {
                            spec.name: {
                                "upscale_factor": spec.upscale_factor,
                                "prepared_width": case.width * spec.upscale_factor,
                                "prepared_height": case.height * spec.upscale_factor,
                            }
                            for spec in specs
                        },
                        "image_sha256": case.image_sha256,
                        "mask_sha256": case.mask_sha256,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def _run_command(command: list[str]) -> None:
    try:
        subprocess.run(
            command,
            check=True,
            text=True,
            # nnU-Net prints several lines and nested progress bars for every
            # case. Capturing keeps large evaluations readable while retaining
            # stdout/stderr for the failure diagnostic below.
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise ImportError(
            f"Official nnU-Net command is unavailable: {command[0]}. "
            "Install `nnunetv2==2.8.1`."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        message = f"Official nnU-Net command failed ({exc.returncode}): {' '.join(command)}"
        if detail:
            message += f"\n{detail[-4000:]}"
        raise RuntimeError(message) from exc


def _assert_exact_predictions(
    prediction_dir: Path,
    cases: list[_SemanticCase],
    model_name: str,
) -> None:
    expected = {f"{case.case_id}.png" for case in cases}
    actual = {path.name for path in prediction_dir.glob("*.png") if path.is_file()}
    if actual != expected:
        raise DatasetValidationError(
            ValidationIssue(
                "nnU-Net predictions do not match the frozen evaluation cohort",
                source=model_name,
                value={
                    "unexpected": sorted(actual - expected)[:20],
                    "missing": sorted(expected - actual)[:20],
                },
                expected=f"exactly {len(expected)} prediction masks",
            )
        )


def _canonicalize_predictions(
    native_prediction_dir: Path,
    canonical_prediction_dir: Path,
    cases: list[_SemanticCase],
    *,
    model_name: str,
    upscale_factor: int,
) -> None:
    """Area-pool native class probabilities onto the frozen source raster."""

    canonical_prediction_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        source = native_prediction_dir / f"{case.case_id}.npz"
        prediction = _canonical_mask_from_probabilities(
            source,
            image_id=case.case_id,
            width=case.width,
            height=case.height,
            model_name=model_name,
            upscale_factor=upscale_factor,
        )
        Image.fromarray(prediction).save(
            canonical_prediction_dir / f"{case.case_id}.png",
            format="PNG",
            optimize=False,
        )


def _canonical_mask_from_probabilities(
    source: Path,
    *,
    image_id: str,
    width: int,
    height: int,
    model_name: str,
    upscale_factor: int,
) -> np.ndarray:
    if not source.is_file():
        raise DatasetValidationError(
            ValidationIssue(
                "nnU-Net probability export is missing",
                source=f"{model_name}/{image_id}",
                value=str(source),
                expected="an .npz file produced by --save_probabilities",
            )
        )
    try:
        with np.load(source) as archive:
            probabilities = np.asarray(archive["probabilities"], dtype=np.float32)
    except (OSError, KeyError, ValueError) as exc:
        raise DatasetValidationError(
            f"Unreadable nnU-Net probability export for {model_name}/{image_id}: {exc}"
        ) from exc
    expected_native_size = (width * upscale_factor, height * upscale_factor)
    expected_spatial_shape = (expected_native_size[1], expected_native_size[0])
    if (
        probabilities.ndim < 3
        or probabilities.shape[0] != 2
        or probabilities.shape[-2:] != expected_spatial_shape
        or math.prod(probabilities.shape[1:-2]) != 1
    ):
        raise DatasetValidationError(
            ValidationIssue(
                "nnU-Net probability dimensions do not match the binary model input adapter",
                source=f"{model_name}/{image_id}",
                value=probabilities.shape,
                expected=f"(2, ..., {expected_native_size[1]}, {expected_native_size[0]})",
            )
        )
    if not np.all(np.isfinite(probabilities)):
        raise DatasetValidationError(
            ValidationIssue(
                "nnU-Net probability export contains non-finite values",
                source=f"{model_name}/{image_id}",
                expected="finite class probabilities",
            )
        )
    probabilities = probabilities.reshape(2, *expected_spatial_shape)
    pooled = probabilities.reshape(
        2,
        height,
        upscale_factor,
        width,
        upscale_factor,
    ).mean(axis=(2, 4))
    return np.argmax(pooled, axis=0).astype(np.uint8)


def _load_official_summary(path: Path, model_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise DatasetValidationError(
            f"Official nnU-Net evaluator did not write its summary for {model_name}: {path}"
        )
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(
            f"Unreadable official nnU-Net evaluation summary for {model_name}: {exc}"
        ) from exc
    if not isinstance(summary.get("foreground_mean"), dict) or not isinstance(
        summary.get("metric_per_case"), list
    ):
        raise DatasetValidationError(
            f"Official nnU-Net evaluation summary for {model_name} lacks foreground_mean or metric_per_case"
        )
    return summary


def _per_case_rows(
    summary: dict[str, Any],
    cases: list[_SemanticCase],
    model_name: str,
) -> list[dict[str, Any]]:
    by_case = {case.case_id: case for case in cases}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in summary["metric_per_case"]:
        case_id = Path(str(result.get("prediction_file", ""))).stem
        case = by_case.get(case_id)
        if case is None:
            raise DatasetValidationError(
                f"Official nnU-Net summary for {model_name} contains unknown case {case_id!r}"
            )
        metrics = result.get("metrics") or {}
        foreground = metrics.get("1") or metrics.get(1)
        if not isinstance(foreground, dict):
            foreground_values = [
                value
                for key, value in metrics.items()
                if str(key) not in {"0", "background"} and isinstance(value, dict)
            ]
            if len(foreground_values) != 1:
                raise DatasetValidationError(
                    f"Official nnU-Net summary for {model_name}/{case_id} is not binary foreground data"
                )
            foreground = foreground_values[0]
        rows.append(
            {
                "model": model_name,
                "case_id": case_id,
                "relative_path": case.relative_path.as_posix(),
                "source_image": str(case.image_path),
                "source_mask": str(case.mask_path),
                "dice": _metric(foreground, "Dice"),
                "iou": _metric(foreground, "IoU"),
                "tp": _metric(foreground, "TP"),
                "fp": _metric(foreground, "FP"),
                "fn": _metric(foreground, "FN"),
                "tn": _metric(foreground, "TN"),
                "n_pred": _metric(foreground, "n_pred"),
                "n_ref": _metric(foreground, "n_ref"),
            }
        )
        seen.add(case_id)
    missing = set(by_case) - seen
    if missing:
        raise DatasetValidationError(
            f"Official nnU-Net summary for {model_name} is missing {len(missing)} frozen cases"
        )
    return rows


def _metric(values: Mapping[str, Any], key: str) -> float:
    value = values.get(key, math.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _bootstrap_interval(
    values: list[float],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if len(array) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    sampled = rng.choice(array, size=(resamples, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def _paired_statistics(
    rows_by_model: dict[str, list[dict[str, Any]]],
    baseline: str,
    *,
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    baseline_scores = {
        row["case_id"]: row["dice"]
        for row in rows_by_model[baseline]
        if math.isfinite(row["dice"])
    }
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    raw_p: list[float] = []
    for name, rows in rows_by_model.items():
        if name == baseline:
            continue
        scores = {
            row["case_id"]: row["dice"]
            for row in rows
            if math.isfinite(row["dice"])
        }
        keys = sorted(set(baseline_scores) & set(scores))
        differences = np.asarray(
            [scores[key] - baseline_scores[key] for key in keys],
            dtype=float,
        )
        if not len(differences):
            continue
        samples = rng.choice(
            differences,
            size=(resamples, len(differences)),
            replace=True,
        ).mean(axis=1)
        signs = rng.choice((-1.0, 1.0), size=(resamples, len(differences)))
        randomized = (differences * signs).mean(axis=1)
        p_value = float(
            (np.sum(np.abs(randomized) >= abs(differences.mean())) + 1)
            / (resamples + 1)
        )
        raw_p.append(p_value)
        output.append(
            {
                "model": name,
                "baseline": baseline,
                "metric": "canonical.per_case.Dice",
                "difference": float(differences.mean()),
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "p_value": p_value,
                "paired_cases": len(differences),
                "wins": int(np.sum(differences > 0)),
                "ties": int(np.sum(differences == 0)),
                "losses": int(np.sum(differences < 0)),
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


def _sanitize_official_summary(
    summary: dict[str, Any],
    cases: list[_SemanticCase],
    *,
    split: str,
    model_slug: str,
    keep_predictions: bool,
) -> dict[str, Any]:
    case_map = {case.case_id: case for case in cases}
    sanitized = json.loads(json.dumps(summary))
    for result in sanitized.get("metric_per_case", []):
        case_id = Path(str(result.get("prediction_file", ""))).stem
        case = case_map.get(case_id)
        if case is None:
            continue
        result["reference_file"] = (
            f"dataset://{split}/masks/0/{case.relative_path.with_suffix('.png').as_posix()}"
        )
        result["prediction_file"] = (
            f"predictions/{model_slug}/{case_id}.png" if keep_predictions else None
        )
        result["source_image"] = f"dataset://{split}/images/{case.relative_path.as_posix()}"
    sanitized["predictions_retained"] = keep_predictions
    return sanitized


def _sortable_score(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return -math.inf
    return parsed if math.isfinite(parsed) else -math.inf


def _render_ranking(
    root: Path,
    ranking: list[dict[str, Any]],
    *,
    xlabel: str = "Canonical probability-pooled foreground mean Dice",
    title: str = "nnU-Net semantic-mask model comparison",
) -> list[str]:
    import matplotlib.pyplot as plt

    ordered = list(reversed(ranking))
    figure, axis = plt.subplots(figsize=(8, max(3.5, 0.7 * len(ordered) + 1.5)))
    names = [row["model"] for row in ordered]
    scores = [
        float(row["dice"]) if math.isfinite(float(row["dice"])) else 0.0
        for row in ordered
    ]
    axis.barh(names, scores, color="#0072B2")
    axis.set_xlim(0, 1)
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    for index, (score, row) in enumerate(zip(scores, ordered)):
        label = f"{float(row['dice']):.3f}" if math.isfinite(float(row["dice"])) else "n/a"
        axis.text(min(score + 0.01, 0.98), index, label, va="center")
    figure.tight_layout()
    path = root / "ranking.png"
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [str(path.relative_to(root))]


def _render_qualitative(
    root: Path,
    cases: list[_SemanticCase],
    prediction_dirs: dict[str, Path],
    rows_by_model: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
) -> list[str]:
    import matplotlib.pyplot as plt

    selected = _select_visual_cases(
        cases,
        samples=8,
        include_empty=False,
        seed=seed,
    )
    figure = _render_semantic_grid(
        selected,
        prediction_dirs,
        rows_by_model,
        examples_per_row=1,
        panel_size=3.0,
        model_title_length=30,
        image_title_length=72,
    )
    output = root / "comparison.png"
    figure.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return [str(output.relative_to(root))]


def _select_visual_cases(
    cases: list[_SemanticCase],
    *,
    samples: int,
    include_empty: bool,
    seed: int,
) -> list[_SemanticCase]:
    eligible: list[_SemanticCase] = []
    for case in cases:
        if include_empty:
            eligible.append(case)
            continue
        with Image.open(case.mask_path) as opened_mask:
            if opened_mask.convert("L").getbbox() is not None:
                eligible.append(case)
    if not eligible:
        eligible = list(cases)
    count = min(samples, len(eligible))
    if count == len(eligible):
        return eligible
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(eligible), size=count, replace=False).tolist())
    return [eligible[index] for index in indices]


def _sample_metric_rows(
    cases: list[_SemanticCase],
    prediction_dir: Path,
    model_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        with Image.open(case.mask_path) as opened_mask:
            truth = np.asarray(opened_mask.convert("L")) > 0
        prediction_path = prediction_dir / f"{case.case_id}.png"
        with Image.open(prediction_path) as opened_prediction:
            prediction = np.asarray(opened_prediction.convert("L")) > 0
        if prediction.shape != truth.shape:
            raise DatasetValidationError(
                f"Prediction dimensions {prediction.shape} do not match "
                f"ground truth {truth.shape}: {prediction_path}"
            )
        metrics = _binary_mask_metrics(truth, prediction)
        rows.append(
            {
                "model": model_name,
                "case_id": case.case_id,
                "relative_path": case.relative_path.as_posix(),
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tn": metrics["tn"],
                "n_ref": metrics["n_ref"],
                "n_pred": metrics["n_pred"],
            }
        )
    return rows


def _binary_mask_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    tp = int(np.sum(truth & prediction))
    fp = int(np.sum(~truth & prediction))
    fn = int(np.sum(truth & ~prediction))
    tn = int(np.sum(~truth & ~prediction))
    dice_denominator = 2 * tp + fp + fn
    iou_denominator = tp + fp + fn
    return {
        "dice": 2 * tp / dice_denominator if dice_denominator else math.nan,
        "iou": tp / iou_denominator if iou_denominator else math.nan,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_ref": int(np.sum(truth)),
        "n_pred": int(np.sum(prediction)),
    }


def _render_semantic_grid(
    cases: list[_SemanticCase],
    prediction_dirs: dict[str, Path],
    rows_by_model: dict[str, list[dict[str, Any]]],
    *,
    examples_per_row: int,
    panel_size: float,
    model_title_length: int,
    image_title_length: int,
) -> Any:
    import matplotlib.pyplot as plt

    if not cases:
        raise ValueError("At least one semantic-mask case is required for visualization")
    model_names = list(prediction_dirs)
    if not model_names:
        raise ValueError("At least one model prediction is required for visualization")
    row_lookup = {
        name: {row["case_id"]: row for row in rows}
        for name, rows in rows_by_model.items()
    }
    panel_count = 2 + len(model_names)
    grid_rows = math.ceil(len(cases) / examples_per_row)
    group_width = panel_size * panel_count
    group_height = panel_size + 0.48
    figure = plt.figure(
        figsize=(
            group_width * examples_per_row,
            group_height * grid_rows,
        ),
    )
    figure.subplots_adjust(
        left=0.015,
        right=0.985,
        top=0.985,
        bottom=0.015,
    )
    outer = figure.add_gridspec(
        grid_rows,
        examples_per_row,
        wspace=0.08,
        hspace=0.18,
    )
    column_titles = [
        "Original",
        "GT",
        *[
            _shorten_middle(name, model_title_length)
            for name in model_names
        ],
    ]

    for index, case in enumerate(cases):
        grid_row = index // examples_per_row
        grid_column = index % examples_per_row
        cell = outer[grid_row, grid_column].subgridspec(
            3,
            panel_count,
            height_ratios=(0.07, 0.07, 0.86),
            hspace=0.01,
            wspace=0.07,
        )
        title_slot = cell[0, :] if grid_row == 0 else cell[0:2, :]
        title_axis = figure.add_subplot(title_slot)
        title_axis.set_axis_off()
        title_axis.text(
            0.5,
            0.5,
            _shorten_middle(case.relative_path.name, image_title_length),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="semibold",
        )
        if grid_row == 0:
            heading_axis = figure.add_subplot(cell[1, :])
            heading_axis.set_axis_off()
            for panel_index, heading in enumerate(column_titles):
                heading_axis.text(
                    (panel_index + 0.5) / panel_count,
                    0.45,
                    heading,
                    ha="center",
                    va="center",
                    fontsize=9,
                )

        with Image.open(case.image_path) as opened_image:
            image = np.asarray(opened_image.convert("RGB"))
        with Image.open(case.mask_path) as opened_mask:
            truth = np.asarray(opened_mask.convert("L")) > 0
        panels: list[np.ndarray] = [image, truth]
        metrics: list[dict[str, Any] | None] = [None, None]
        for name in model_names:
            prediction_path = prediction_dirs[name] / f"{case.case_id}.png"
            with Image.open(prediction_path) as opened_prediction:
                prediction = np.asarray(opened_prediction.convert("L")) > 0
            if prediction.shape != truth.shape:
                raise DatasetValidationError(
                    f"Prediction dimensions {prediction.shape} do not match "
                    f"ground truth {truth.shape}: {prediction_path}"
                )
            panels.append(prediction)
            try:
                metrics.append(row_lookup[name][case.case_id])
            except KeyError as exc:
                raise DatasetValidationError(
                    f"Missing visualization metrics for {name}/{case.case_id}"
                ) from exc

        for panel_index, panel in enumerate(panels):
            axis = figure.add_subplot(cell[2, panel_index])
            if panel_index == 0:
                axis.imshow(panel)
            else:
                axis.imshow(
                    panel,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                )
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            metric = metrics[panel_index]
            if metric is not None:
                axis.set_xlabel(
                    f"Dice={_format_metric(metric['dice'])} · "
                    f"IoU={_format_metric(metric['iou'])}",
                    fontsize=7.5,
                    labelpad=2,
                )
    return figure


def _shorten_middle(value: str, maximum: int) -> str:
    if len(value) <= maximum:
        return value
    left = (maximum - 1) // 2
    right = maximum - 1 - left
    return f"{value[:left]}…{value[-right:]}"


def _format_metric(value: Any) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{parsed:.3f}" if math.isfinite(parsed) else "n/a"


def _visualization_destination(destination: str | Path) -> Path:
    path = Path(destination).expanduser().resolve()
    if path.suffix:
        if path.suffix.lower() != ".png":
            raise ValueError("visualization destination must be a PNG file or directory")
        return path
    return path / "comparison.png"
