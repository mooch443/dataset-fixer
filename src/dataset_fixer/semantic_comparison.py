from __future__ import annotations

import hashlib
import json
import math
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from .comparison.cache import (
    build_staging_dir,
    cache_key,
    default_cache_root,
    load_evaluation_cache,
    save_evaluation_cache,
)
from .comparison.reporting import write_json
from .errors import DatasetValidationError, ValidationIssue
from .model import ImagePrediction, Model, ModelCollection, ModelInput
from .models import SemanticComparisonResult
from .sahi_support import resolve_sahi_settings
from .utils import (
    IMAGE_SUFFIXES,
    environment_snapshot,
    normalize_split,
    package_versions,
    settings_fingerprint,
    sha256_file,
    to_jsonable,
)


# Report presentation evolves independently from prediction/evaluation cache
# identity. A report bump redraws output from completed caches without forcing
# model inference to run again.
SEMANTIC_REPORT_SCHEMA = 11
SEMANTIC_PREDICTION_SCHEMA = 2
SEMANTIC_EVALUATION_CACHE_SCHEMA = 2

# How much in-flight tile memory one SAHI work group may hold, and how much a
# single grouped inference call may allocate for padded inputs and logits.
_NNUNET_SAHI_PROBABILITY_GROUP_BYTES = 512 * 1024 * 1024
_NNUNET_SAHI_MAX_IMAGES_PER_GROUP = 512
_NNUNET_SAHI_INFERENCE_CHUNK_BYTES = 512 * 1024 * 1024


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
    save_prediction_plots: bool,
    progress: bool,
    destination: str | Path | None,
) -> SemanticComparisonResult:
    """Run official nnU-Net v2 prediction and evaluation for an export."""

    seed = 42
    bootstrap_resamples = 10_000
    split = normalize_split(split)
    if split not in export.splits:
        raise ValueError(f"Unknown semantic-mask split {split!r}; available splits are {export.splits}")
    specs = _parse_models(models)
    resolved_devices = {
        spec.name: str(spec._resolved_device())
        for spec in specs
    }
    model_backends = {spec.name: spec.inference for spec in specs}
    resolved_sahi_by_model = {
        spec.name: resolve_sahi_settings(
            spec.settings,
            resolution=spec.resolution or 480,
        ).as_dict()
        for spec in specs
        if model_backends[spec.name] == "sahi"
    }
    model_systems = {
        spec.name: {
            "backend": model_backends[spec.name],
            "resolution": spec.resolution or 480,
            "sahi": resolved_sahi_by_model.get(spec.name),
            "upscale_factor": spec.upscale_factor,
            "nnunet_tta": spec.nnunet_tta,
        }
        for spec in specs
    }
    comparison_unit = (
        "model"
        if len({settings_fingerprint(value) for value in model_systems.values()}) == 1
        else "system"
    )
    cases, cohort_fingerprint = _freeze_cohort(export, split, progress=progress)

    resolved_settings = {
        "backend": (
            next(iter(set(model_backends.values())))
            if len(set(model_backends.values())) == 1
            else "mixed"
        ),
        "adapter": "nnunetv2-official",
        "report_schema": SEMANTIC_REPORT_SCHEMA,
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
        "bootstrap_resamples": bootstrap_resamples,
        "seed": seed,
        "model_backends": model_backends,
        "comparison_unit": comparison_unit,
        "sahi_models": resolved_sahi_by_model,
        "model_systems": model_systems,
        "models": [
            {
                "name": spec.name,
                "model_folder": spec.model_folder,
                "folds": spec.folds,
                "checkpoint": spec.checkpoint,
                "checkpoint_sha256": spec.checkpoint_sha256,
                "model_sha256": spec.digest,
                "upscale_factor": spec.upscale_factor,
                "device": resolved_devices[spec.name],
                "workers": spec.workers,
                "nnunet_tta": spec.nnunet_tta,
            }
            for spec in specs
        ],
    }
    fingerprint = settings_fingerprint(
        {
            "schema": SEMANTIC_REPORT_SCHEMA,
            "cohort_fingerprint": cohort_fingerprint,
            "models": [
                {
                    "name": spec.name,
                    "model_sha256": spec.digest,
                    "backend": model_backends[spec.name],
                    "folds": spec.folds,
                    "checkpoint": spec.checkpoint,
                    "upscale_factor": spec.upscale_factor,
                    "resolution": spec.resolution or 480,
                    "nnunet_tta": spec.nnunet_tta,
                    "sahi": resolved_sahi_by_model.get(spec.name),
                }
                for spec in specs
            ],
        }
    )
    target = (
        Path(destination).expanduser().resolve()
        if destination is not None
        else export.location / "evaluations" / fingerprint
    )
    if target == export.location:
        raise ValueError("Comparison destination cannot replace the dataset")
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = target / "reports" / "result.json"
    if (
        existing.is_file()
        and (target / "reports" / "plots.png").is_file()
        and (target / "reports" / "comparison.png").is_file()
        and (not save_prediction_plots or (target / "predictions").is_dir())
    ):
        cached_manifest = json.loads(existing.read_text(encoding="utf-8"))
        if (
            cached_manifest.get("schema") == SEMANTIC_REPORT_SCHEMA
            and cached_manifest.get("settings_fingerprint") == fingerprint
        ):
            print(f"Reusing complete comparison: {target}")
            return _semantic_result_from_manifest(target, cached_manifest)
    temporary = build_staging_dir(
        target,
        dataset_location=None if destination is not None else export.location,
    )
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
        statistics_key = f"semantic-comparison-{fingerprint}"
        statistics_cache = default_cache_root(export.location)
        cached_statistics = load_evaluation_cache(statistics_cache, statistics_key)
        canonical_labels = temporary / "cohort" / "canonical" / "labels"
        _prepare_labels(cases, canonical_labels)
        model_rows: dict[str, list[dict[str, Any]]] = {}
        ranking: list[dict[str, Any]] = []
        prediction_dirs: dict[str, Path] = {}
        model_inputs = _model_inputs_from_cases(cases)
        for model_index, spec in enumerate(specs):
            selected_backend = model_backends[spec.name]
            cache_payload = {
                "schema": SEMANTIC_EVALUATION_CACHE_SCHEMA,
                "space": "nnunet-semantic",
                "cohort": cohort_fingerprint,
                "model_sha256": spec.digest,
                "backend": selected_backend,
                "folds": spec.folds,
                "checkpoint": spec.checkpoint,
                "upscale_factor": spec.upscale_factor,
                "resolution": spec.resolution or 480,
                "nnunet_tta": spec.nnunet_tta,
                "sahi": resolved_sahi_by_model.get(spec.name),
            }
            cache_identity = cache_key(cache_payload)
            semantic_cache_root = default_cache_root(export.location) / "semantic"
            cache_dir = semantic_cache_root / cache_identity
            legacy_cache_dir = semantic_cache_root / cache_key(
                {
                    "schema": SEMANTIC_EVALUATION_CACHE_SCHEMA,
                    "space": "nnunet-semantic",
                    "cohort": cohort_fingerprint,
                    "model_sha256": spec.digest,
                    "backend": selected_backend,
                    "folds": spec.folds,
                    "checkpoint": spec.checkpoint,
                    "upscale_factor": spec.upscale_factor,
                    "device": resolved_devices[spec.name],
                    "resolution": spec.resolution or 480,
                    "sahi": resolved_sahi_by_model.get(spec.name),
                    "versions": package_versions(),
                }
            )
            compatible_legacy_dirs: tuple[Path, ...] = ()
            if spec.nnunet_tta:
                # nnU-Net TTA was always enabled before it became explicit.
                # Those caches are valid only for an explicit TTA request.
                prior_tta_payload = dict(cache_payload)
                prior_tta_payload.pop("nnunet_tta")
                compatible_legacy_dirs = (
                    semantic_cache_root / cache_key(prior_tta_payload),
                    legacy_cache_dir,
                )
            cache_dir, cached = _load_compatible_semantic_cache(
                cache_dir,
                compatible_legacy_dirs,
                cases,
                cache_identity=cache_payload,
                required_fields=("summary", "native_summary", "native_rows"),
                progress=progress,
                model_name=spec.name,
            )
            if cached is not None:
                prediction_dir = cache_dir / "predictions"
                summary = dict(cached["summary"])
                native_summary = dict(cached["native_summary"])
                rows = _rebase_cached_semantic_rows(
                    cached["rows"], cases, model_name=spec.name
                )
                native_rows = _rebase_cached_semantic_rows(
                    cached["native_rows"], cases, model_name=spec.name
                )
                inference_seconds = float(cached.get("inference_seconds", 0.0))
                execution = dict(cached.get("execution") or {})
                cache_status = "hit"
            else:
                _require_official_commands("nnUNetv2_evaluate_folder")
                if selected_backend == "native":
                    _require_official_commands("nnUNetv2_predict_from_modelfolder")
                native_labels = temporary / "cohort" / "models" / spec.slug / "labels"
                _prepare_labels(cases, native_labels, upscale_factor=spec.upscale_factor)
                native_prediction_dir = temporary / "working" / "native-predictions" / spec.slug
                native_prediction_dir.mkdir(parents=True, exist_ok=True)
                prediction_dir = temporary / "working" / "predictions" / spec.slug
                prediction_dir.mkdir(parents=True, exist_ok=True)
                summary_path = temporary / "working" / f"{spec.slug}-metrics.json"
                native_summary_path = temporary / "working" / f"{spec.slug}-native-metrics.json"
                print(
                    f"Evaluating {spec.name!r} with official nnU-Net v2 "
                    f"(folds={spec.folds}, checkpoint={spec.checkpoint}, "
                    f"input_scale={spec.upscale_factor}x)"
                )
                prediction_result = spec.predict(
                    model_inputs,
                    device=resolved_devices[spec.name],
                    progress=progress,
                    _keep_native=True,
                    inference=selected_backend,
                    resolution=spec.resolution or 480,
                )
                inference_seconds = prediction_result.inference_seconds
                execution = _engine_telemetry(prediction_result.records)
                _write_semantic_prediction_masks(
                    prediction_result.records,
                    prediction_dir,
                    native_prediction_dir,
                    progress=progress,
                    description=f"Saving {spec.name} predictions",
                )
                _assert_exact_predictions(native_prediction_dir, cases, spec.name)
                _run_command(
                    [
                        "nnUNetv2_evaluate_folder", str(native_labels), str(native_prediction_dir),
                        "-djfile", str(spec.model_folder / "dataset.json"),
                        "-pfile", str(spec.model_folder / "plans.json"),
                        "-o", str(native_summary_path), "-np", str(spec.workers),
                    ],
                    progress=progress,
                )
                native_summary = _load_official_summary(native_summary_path, spec.name)
                native_rows = _per_case_rows(native_summary, cases, spec.name)
                _assert_exact_predictions(prediction_dir, cases, spec.name)
                _run_command(
                    [
                        "nnUNetv2_evaluate_folder", str(canonical_labels), str(prediction_dir),
                        "-djfile", str(spec.model_folder / "dataset.json"),
                        "-pfile", str(spec.model_folder / "plans.json"),
                        "-o", str(summary_path), "-np", str(spec.workers),
                    ],
                    progress=progress,
                )
                summary = _load_official_summary(summary_path, spec.name)
                rows = _per_case_rows(summary, cases, spec.name)
                _save_semantic_cache(
                    cache_dir,
                    prediction_dir,
                    {
                        "rows": rows,
                        "native_rows": native_rows,
                        "summary": summary,
                        "native_summary": native_summary,
                        "inference_seconds": inference_seconds,
                        "execution": execution,
                        "cache_identity": cache_payload,
                    },
                    progress=progress,
                    model_name=spec.name,
                )
                prediction_dir = cache_dir / "predictions"
                cache_status = "fresh"
            prediction_dirs[spec.name] = prediction_dir
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
            cached_interval = (
                (cached_statistics or {}).get("intervals", {}).get(spec.name)
            )
            if isinstance(cached_interval, list) and len(cached_interval) == 2:
                ci_low, ci_high = map(float, cached_interval)
            else:
                ci_low, ci_high = _bootstrap_interval(
                    [row["dice"] for row in rows],
                    resamples=bootstrap_resamples,
                    seed=seed + model_index,
                )
            ranking.append(
                {
                    "model": spec.name,
                    **_model_report_fields(spec),
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
                    "nnunet_tta": spec.nnunet_tta,
                    "evaluation_resolution": "canonical-export",
                    "projection": (
                        "sahi-feathered-probability-area-pool-argmax"
                        if selected_backend == "sahi"
                        else "probability-area-pool-argmax"
                    ),
                    "native_evaluation_resolution": f"model-input-{spec.upscale_factor}x",
                    "cohort_fingerprint": cohort_fingerprint,
                    "inference_seconds": inference_seconds,
                    "cache": cache_status,
                    "throughput_cases_per_second": (
                        len(cases) / inference_seconds if inference_seconds > 0 else None
                    ),
                    "execution": execution,
                }
            )

        ranking.sort(key=lambda row: (-_sortable_score(row["score"]), row["model"]))
        for rank, row in enumerate(ranking, start=1):
            row["rank"] = rank
        if cached_statistics is not None and isinstance(cached_statistics.get("paired"), list):
            paired = list(cached_statistics["paired"])
        else:
            paired = _all_pairwise_statistics(
                model_rows,
                resamples=bootstrap_resamples,
                seed=seed,
            )
            save_evaluation_cache(
                statistics_cache,
                statistics_key,
                {
                    "intervals": {
                        row["model"]: [row["ci_low"], row["ci_high"]]
                        for row in ranking
                    },
                    "paired": paired,
                },
            )
        per_case = [row for name in [spec.name for spec in specs] for row in model_rows[name]]
        reports = temporary / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        _render_ranking(reports, ranking)
        _render_qualitative(reports, cases, prediction_dirs, model_rows, seed=seed)
        # Render only the cases the report keeps, so nothing is drawn that is
        # not also referenced in the manifest.
        worst_cases = _bounded_semantic_cases(per_case)
        prediction_paths: list[str] = []
        if save_prediction_plots:
            prediction_paths = _render_semantic_prediction_grids(
                temporary,
                cases,
                prediction_dirs,
                model_rows,
                case_ids=[str(row["case_id"]) for row in worst_cases],
            )
        shutil.rmtree(temporary / "working", ignore_errors=True)
        shutil.rmtree(temporary / "cohort", ignore_errors=True)

        manifest = {
            "schema": SEMANTIC_REPORT_SCHEMA,
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
            "settings": resolved_settings,
            "settings_fingerprint": fingerprint,
            "ranking": ranking,
            "paired_statistics": paired,
            "limitations": limitations,
            "worst_cases": worst_cases,
            "reports": {
                "plots": "reports/plots.png",
                "comparison": "reports/comparison.png",
                "prediction_plots": prediction_paths,
            },
            "environment": environment_snapshot(),
            "started_at_unix": started,
            "completed_at_unix": time.time(),
        }
        write_json(reports / "result.json", manifest)
        if target.exists():
            if not (target / "reports" / "result.json").is_file():
                raise FileExistsError(f"Refusing to replace unrelated comparison destination: {target}")
            shutil.rmtree(target)
        temporary.replace(target)
    except BaseException:
        # BaseException, not Exception: a cancelled run (KeyboardInterrupt)
        # must not leave a partial evaluation behind either.
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    pairing = "paired comparisons: all unordered pairs" if len(specs) > 1 else "single-model evaluation"
    print(
        f"Semantic model comparison complete: {target}\n"
        f"Cohort verified: yes; cases: {len(cases)}; {pairing}"
    )
    return SemanticComparisonResult(
        location=target,
        ranking=tuple(ranking),
        cohort_fingerprint=cohort_fingerprint,
        cohort_verified=True,
        split=split,
        settings=resolved_settings,
        limitations=tuple(limitations),
    )


def compare_semantic_models(
    export: "Dataset",
    models: ModelCollection,
    *,
    split: str,
    save_prediction_plots: bool,
    progress: bool,
    destination: str | Path | None,
) -> SemanticComparisonResult:
    """Compare instance and semantic segmenters in one binary mask space."""

    seed = 42
    bootstrap_resamples = 10_000
    split = normalize_split(split)
    if split not in export.splits:
        raise ValueError(
            f"Unknown semantic-mask split {split!r}; available splits are {export.splits}"
        )
    resolved_devices = {
        model.name: model._resolved_device()
        for model in models
    }
    resolved_sahi_by_model = {
        model.name: resolve_sahi_settings(
            model.settings,
            resolution=model.resolution or 480,
        ).as_dict()
        for model in models
        if model.inference == "sahi"
    }
    model_systems = {
        model.name: {
            "backend": model.inference,
            "resolution": model.resolution or 480,
            "confidence": model.confidence if model.kind == "ultralytics" else None,
            "postprocess": model.postprocess if model.kind == "ultralytics" else None,
            "nnunet_tta": model.nnunet_tta if model.kind == "nnunet" else None,
            "sahi": resolved_sahi_by_model.get(model.name),
        }
        for model in models
    }
    comparison_unit = (
        "model"
        if len({settings_fingerprint(value) for value in model_systems.values()}) == 1
        else "system"
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

    cases, cohort_fingerprint = _freeze_cohort(export, split, progress=progress)
    resolved_settings = {
        "backend": "common-semantic-mask",
        "report_schema": SEMANTIC_REPORT_SCHEMA,
        "comparison_space": "semantic",
        "canonical_projection": "binary-foreground-union",
        "split": split,
        "comparison_unit": comparison_unit,
        "sahi_models": resolved_sahi_by_model,
        "model_systems": model_systems,
        "bootstrap_resamples": bootstrap_resamples,
        "seed": seed,
        "models": [
            {
                **model.describe(),
                "device": resolved_devices[model.name],
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
        {
            "schema": SEMANTIC_REPORT_SCHEMA,
            "cohort_fingerprint": cohort_fingerprint,
            "models": [
                {
                    "name": model.name,
                    "model_sha256": model.digest,
                    "kind": model.kind,
                    "task": model.task,
                    "folds": model.folds,
                    "checkpoint": model.checkpoint,
                    "upscale_factor": model.upscale_factor,
                    "inference": model.inference,
                    "resolution": model.resolution or 480,
                    "confidence": model.confidence,
                    "postprocess": model.postprocess,
                    "settings": model.settings,
                    **(
                        {"nnunet_tta": model.nnunet_tta}
                        if model.kind == "nnunet"
                        else {}
                    ),
                    "sahi": resolved_sahi_by_model.get(model.name),
                }
                for model in models
            ],
        }
    )
    target = (
        Path(destination).expanduser().resolve()
        if destination is not None
        else export.location / "evaluations" / fingerprint
    )
    if target == export.location:
        raise ValueError("Comparison destination cannot replace the dataset")
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = target / "reports" / "result.json"
    if (
        existing.is_file()
        and (target / "reports" / "plots.png").is_file()
        and (target / "reports" / "comparison.png").is_file()
        and (not save_prediction_plots or (target / "predictions").is_dir())
    ):
        cached_manifest = json.loads(existing.read_text(encoding="utf-8"))
        if (
            cached_manifest.get("schema") == SEMANTIC_REPORT_SCHEMA
            and cached_manifest.get("settings_fingerprint") == fingerprint
        ):
            print(f"Reusing complete comparison: {target}")
            return _semantic_result_from_manifest(target, cached_manifest)
    temporary = build_staging_dir(
        target,
        dataset_location=None if destination is not None else export.location,
    )
    started = time.time()
    limitations = [
        "All models are evaluated as binary foreground at the canonical semantic-mask export resolution.",
        "YOLO instance classes and identities are discarded by unioning every retained prediction polygon.",
        "YOLO confidence and postprocessing thresholds do not alter native semantic-model predictions.",
        "Dice is undefined when both reference and prediction are empty; finite support is reported separately.",
        "Training/evaluation overlap cannot be independently verified for models without training provenance.",
    ]

    try:
        statistics_key = f"semantic-comparison-{fingerprint}"
        statistics_cache = default_cache_root(export.location)
        cached_statistics = load_evaluation_cache(statistics_cache, statistics_key)
        model_inputs = _model_inputs_from_cases(cases)
        model_rows: dict[str, list[dict[str, Any]]] = {}
        prediction_dirs: dict[str, Path] = {}
        ranking: list[dict[str, Any]] = []
        projection_warnings: list[dict[str, Any]] = []
        for model_index, model in enumerate(models):
            confidence = model.confidence
            postprocess = model.postprocess
            predict_options: dict[str, Any] = {
                "device": resolved_devices[model.name],
                "progress": progress,
                "inference": model.inference,
                "resolution": model.resolution or 480,
                "batch_size": model.batch_size,
            }
            if model.kind == "ultralytics":
                predict_options.update(
                    {
                        "confidence": float(confidence),
                        "postprocess": float(postprocess),
                    }
                )
            cache_payload = {
                "schema": SEMANTIC_EVALUATION_CACHE_SCHEMA,
                "space": "binary-semantic",
                "cohort": cohort_fingerprint,
                "model_sha256": model.digest,
                "kind": model.kind,
                "task": model.task,
                "folds": model.folds,
                "checkpoint": model.checkpoint,
                "upscale_factor": model.upscale_factor,
                "inference": model.inference,
                "resolution": model.resolution or 480,
                "settings": model.settings,
                **(
                    {"nnunet_tta": model.nnunet_tta}
                    if model.kind == "nnunet"
                    else {}
                ),
            }
            cache_identity = cache_key(cache_payload)
            semantic_cache_root = default_cache_root(export.location) / "semantic"
            cache_dir = semantic_cache_root / cache_identity
            legacy_cache_dir = semantic_cache_root / cache_key(
                {
                    "schema": SEMANTIC_EVALUATION_CACHE_SCHEMA,
                    "space": "binary-semantic",
                    "cohort": cohort_fingerprint,
                    "model_sha256": model.digest,
                    "kind": model.kind,
                    "task": model.task,
                    "model_settings": model.settings,
                    "folds": model.folds,
                    "checkpoint": model.checkpoint,
                    "upscale_factor": model.upscale_factor,
                    "workers": model.workers,
                    "batch_size": model.batch_size,
                    "settings": {
                        key: value
                        for key, value in predict_options.items()
                        if key not in {"progress", "nnunet_tta"}
                    },
                    "versions": package_versions(),
                }
            )
            compatible_legacy_dirs: tuple[Path, ...]
            if model.kind != "nnunet":
                compatible_legacy_dirs = (legacy_cache_dir,)
            elif model.nnunet_tta:
                # Prior releases always enabled nnU-Net TTA but did not record
                # that fact in the identity. They are compatible only when the
                # caller explicitly requests the historical TTA behavior.
                prior_tta_payload = dict(cache_payload)
                prior_tta_payload.pop("nnunet_tta")
                compatible_legacy_dirs = (
                    semantic_cache_root / cache_key(prior_tta_payload),
                    legacy_cache_dir,
                )
            else:
                compatible_legacy_dirs = ()
            cache_dir, cached = _load_compatible_semantic_cache(
                cache_dir,
                compatible_legacy_dirs,
                cases,
                cache_identity=cache_payload,
                required_fields=("projection", "native_task", "backend"),
                progress=progress,
                model_name=model.name,
            )
            if cached is not None:
                prediction_dir = cache_dir / "predictions"
                rows = _rebase_cached_semantic_rows(
                    cached["rows"], cases, model_name=model.name
                )
                projection = str(cached["projection"])
                native_task = str(cached["native_task"])
                prediction_backend = str(cached["backend"])
                inference_seconds = float(cached.get("inference_seconds", 0.0))
                model_warnings = list(cached.get("warnings") or [])
                cache_status = "hit"
            else:
                prediction_dir = temporary / "working" / "predictions" / model.slug
                prediction_dir.mkdir(parents=True, exist_ok=True)
                prediction_result = model.predict(model_inputs, **predict_options)
                projected, projection, model_warnings = _project_semantic_predictions(
                    prediction_result.records,
                    prediction_result.task,
                    cases,
                    model.name,
                    confidence=float(confidence),
                )
                _write_semantic_prediction_masks(
                    projected,
                    prediction_dir,
                    progress=progress,
                    description=f"Saving {model.name} predictions",
                )
                _assert_exact_predictions(prediction_dir, cases, model.name)
                rows = _sample_metric_rows(
                    cases, prediction_dir, model.name, progress=progress
                )
                native_task = prediction_result.task
                prediction_backend = prediction_result.backend
                inference_seconds = prediction_result.inference_seconds
                _save_semantic_cache(
                    cache_dir,
                    prediction_dir,
                    {
                        "rows": rows,
                        "projection": projection,
                        "native_task": native_task,
                        "backend": prediction_backend,
                        "inference_seconds": inference_seconds,
                        "warnings": model_warnings,
                        "cache_identity": cache_payload,
                    },
                    progress=progress,
                    model_name=model.name,
                )
                prediction_dir = cache_dir / "predictions"
                cache_status = "fresh"
            projection_warnings.extend(model_warnings)
            prediction_dirs[model.name] = prediction_dir
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
            cached_interval = (
                (cached_statistics or {}).get("intervals", {}).get(model.name)
            )
            if isinstance(cached_interval, list) and len(cached_interval) == 2:
                ci_low, ci_high = map(float, cached_interval)
            else:
                ci_low, ci_high = _bootstrap_interval(
                    [row["dice"] for row in rows],
                    resamples=bootstrap_resamples,
                    seed=seed + model_index,
                )
            ranking.append(
                {
                    "model": model.name,
                    **_model_report_fields(model),
                    "model_kind": model.kind,
                    "native_task": native_task,
                    "backend": prediction_backend,
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
                    "inference_seconds": inference_seconds,
                    "cache": cache_status,
                    "warning_count": len(model_warnings),
                    "throughput_cases_per_second": (
                        len(cases) / inference_seconds
                        if inference_seconds > 0
                        else None
                    ),
                }
            )

        ranking.sort(key=lambda row: (-_sortable_score(row["score"]), row["model"]))
        for rank, row in enumerate(ranking, start=1):
            row["rank"] = rank
        if cached_statistics is not None and isinstance(cached_statistics.get("paired"), list):
            paired = list(cached_statistics["paired"])
        else:
            paired = _all_pairwise_statistics(
                model_rows,
                resamples=bootstrap_resamples,
                seed=seed,
            )
            save_evaluation_cache(
                statistics_cache,
                statistics_key,
                {
                    "intervals": {
                        row["model"]: [row["ci_low"], row["ci_high"]]
                        for row in ranking
                    },
                    "paired": paired,
                },
            )
        per_case = [
            row
            for model in models
            for row in model_rows[model.name]
        ]
        reports = temporary / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        _render_ranking(
            reports,
            ranking,
            xlabel="Canonical binary foreground mean Dice",
            title="Semantic-space model comparison",
        )
        _render_qualitative(reports, cases, prediction_dirs, model_rows, seed=seed)
        # Render only the cases the report keeps, so nothing is drawn that is
        # not also referenced in the manifest.
        worst_cases = _bounded_semantic_cases(per_case)
        prediction_paths: list[str] = []
        if save_prediction_plots:
            prediction_paths = _render_semantic_prediction_grids(
                temporary,
                cases,
                prediction_dirs,
                model_rows,
                case_ids=[str(row["case_id"]) for row in worst_cases],
            )
        shutil.rmtree(temporary / "working", ignore_errors=True)

        manifest = {
            "schema": SEMANTIC_REPORT_SCHEMA,
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
            "settings": resolved_settings,
            "settings_fingerprint": fingerprint,
            "ranking": ranking,
            "paired_statistics": paired,
            "limitations": limitations,
            "warnings": projection_warnings,
            "worst_cases": worst_cases,
            "reports": {
                "plots": "reports/plots.png",
                "comparison": "reports/comparison.png",
                "prediction_plots": prediction_paths,
            },
            "environment": environment_snapshot(),
            "started_at_unix": started,
            "completed_at_unix": time.time(),
        }
        write_json(reports / "result.json", manifest)
        if target.exists():
            if not (target / "reports" / "result.json").is_file():
                raise FileExistsError(f"Refusing to replace unrelated comparison destination: {target}")
            shutil.rmtree(target)
        temporary.replace(target)
    except BaseException:
        # BaseException, not Exception: a cancelled run (KeyboardInterrupt)
        # must not leave a partial evaluation behind either.
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    pairing = "paired comparisons: all unordered pairs" if len(models) > 1 else "single-model evaluation"
    print(
        f"Semantic model comparison complete: {target}\n"
        f"Cohort verified: yes; cases: {len(cases)}; {pairing}; "
        "comparison space: semantic"
    )
    if projection_warnings:
        print(
            f"Warnings: skipped {len(projection_warnings)} invalid segmentation "
            f"object(s); details: {target / 'reports' / 'result.json'}"
        )
    return SemanticComparisonResult(
        location=target,
        ranking=tuple(ranking),
        cohort_fingerprint=cohort_fingerprint,
        cohort_verified=True,
        split=split,
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
) -> tuple[tuple[ImagePrediction, ...], str, list[dict[str, Any]]]:
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
        return tuple(projected), "native-semantic-mask", []
    if task != "segment":
        raise DatasetValidationError(
            ValidationIssue(
                "Prediction task has no semantic-mask projection",
                source=model_name,
                value=task,
                expected="segment or semantic_segment",
            )
        )
    warnings: list[dict[str, Any]] = []
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
            usable: list[list[tuple[float, float]]] = []
            for polygon in polygons:
                reason = None
                if len(polygon) < 3:
                    reason = "fewer than three polygon points"
                elif any(
                    not math.isfinite(float(x)) or not math.isfinite(float(y))
                    for x, y in polygon
                ):
                    reason = "non-finite polygon coordinates"
                if reason is not None:
                    warnings.append(
                        {
                            "model": model_name,
                            "case_id": case.case_id,
                            "relative_path": case.relative_path.as_posix(),
                            "object_index": index,
                            "reason": reason,
                            "action": "skipped-object",
                        }
                    )
                    continue
                usable.append(polygon)
            if not polygons:
                warnings.append(
                    {
                        "model": model_name,
                        "case_id": case.case_id,
                        "relative_path": case.relative_path.as_posix(),
                        "object_index": index,
                        "reason": "no polygon returned by segmentation model",
                        "action": "skipped-object",
                    }
                )
            for polygon in usable:
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
    return tuple(projected), "polygon-foreground-union", warnings


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
    batch_size: int = -1,
) -> tuple[ImagePrediction, ...]:
    """Run the official nnU-Net adapter for :meth:`Model.predict`."""

    if model.kind != "nnunet":
        raise TypeError("predict_nnunet_model requires Model(kind='nnunet')")
    from .nnunet_engine import require_nnunet

    require_nnunet()
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("nnU-Net device must be 'cpu', 'cuda', or 'mps'")
    if inference not in {"native", "sahi"}:
        raise ValueError("inference must be 'native' or 'sahi'; 'auto' was removed")
    if inference == "sahi":
        # Sliced prediction runs in process against nnU-Net's Python API; only
        # whole-image prediction still shells out to the official CLI.
        return _predict_nnunet_sahi(
            model,
            inputs,
            device=device,
            progress=progress,
            keep_native=keep_native,
            resolution=resolution,
            settings=dict(settings or {}),
            batch_size=batch_size,
        )
    _require_official_commands("nnUNetv2_predict_from_modelfolder")
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
                "--disable_progress_bar",
                *([] if model.nnunet_tta else ["--disable_tta"]),
            ],
            progress=progress,
            progress_total=len(inputs),
            progress_directory=native_predictions,
            progress_description="nnU-Net native prediction",
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
    batch_size: int,
) -> tuple[ImagePrediction, ...]:
    """Predict SAHI tiles in process, as real network minibatches.

    The canonical manifest, tile geometry, overlap, TTA, upscale adapter,
    feathered probability stitching, and argmax-after-stitch behavior are
    unchanged; only the execution engine differs from tile-by-tile CLI runs.
    """

    from .nnunet_engine import EngineTelemetry, load_session
    from .sahi_support import (
        build_tile_manifest,
        resolve_sahi_settings,
        stitch_probability_tiles,
    )

    if progress:
        checkpoints = ", ".join(str(path) for path in model.checkpoint_files)
        print(
            "nnU-Net execution: in-process Python API for SAHI (no CLI command)\n"
            f"  model_folder: {model.model_folder}\n"
            f"  folds: {', '.join(model.folds)}\n"
            f"  checkpoint: {model.checkpoint}\n"
            f"  checkpoint_files: {checkpoints}\n"
            f"  device: {device}\n"
            f"  batch_size: {batch_size}\n"
            f"  workers: {model.workers}\n"
            f"  TTA: {'enabled' if model.nnunet_tta else 'disabled'}"
        )
    manifest_progress = tqdm(
        total=len(inputs),
        desc="Planning nnU-Net SAHI tiles",
        unit="image",
        file=sys.stdout,
        dynamic_ncols=True,
        disable=not progress,
    )
    try:
        resolved = resolve_sahi_settings(settings, resolution=resolution)
        manifests = {}
        for value in inputs:
            manifests[value.image_id] = build_tile_manifest(
                width=value.width,
                height=value.height,
                settings=resolved,
            )
            manifest_progress.update(1)
    finally:
        manifest_progress.close()
    total_tiles = sum(len(manifests[value.image_id]) for value in inputs)
    if progress:
        print(
            f"SAHI tile plan ready: {len(inputs):,} source images, "
            f"{total_tiles:,} logical tiles. Loading nnU-Net checkpoint…"
        )

    checkpoint_progress = tqdm(
        desc="Loading nnU-Net checkpoint",
        unit="s",
        file=sys.stdout,
        dynamic_ncols=True,
        disable=not progress,
    )
    refresh = getattr(checkpoint_progress, "refresh", None)
    if refresh is not None:
        refresh()
    checkpoint_started = time.perf_counter()
    checkpoint_stop = threading.Event()

    def update_checkpoint_elapsed() -> None:
        reported = 0
        while not checkpoint_stop.wait(1.0):
            elapsed = int(time.perf_counter() - checkpoint_started)
            if elapsed > reported:
                checkpoint_progress.update(elapsed - reported)
                reported = elapsed

    checkpoint_thread = (
        threading.Thread(
            target=update_checkpoint_elapsed,
            name="dataset-fixer-nnunet-load-progress",
            daemon=True,
        )
        if progress
        else None
    )
    if checkpoint_thread is not None:
        checkpoint_thread.start()
    try:
        session = model._runtime_model(
            (
                "nnunet-sahi",
                device,
                tuple(model.folds),
                model.checkpoint,
                model.workers,
                batch_size,
                model.nnunet_tta,
            ),
            lambda: load_session(
                model_folder=model.model_folder,
                folds=model.folds,
                checkpoint=model.checkpoint,
                device=device,
                workers=model.workers,
                batch_size=batch_size,
                use_tta=model.nnunet_tta,
            ),
        )
    finally:
        checkpoint_stop.set()
        if checkpoint_thread is not None:
            checkpoint_thread.join(timeout=1.0)
        checkpoint_progress.close()
    if progress:
        print(
            f"nnU-Net runtime ready in "
            f"{time.perf_counter() - checkpoint_started:.1f}s; starting inference."
        )
    telemetry = EngineTelemetry(
        device=device,
        plan_batch_size=session.plan_batch_size,
        requested_batch_size=session.requested_batch_size,
        resolved_batch_size=session.requested_batch_size,
        tiles=total_tiles,
        sources=len(inputs),
        folds=tuple(model.folds),
        tta=session.use_tta,
        workers=session.workers,
    )
    records: list[ImagePrediction] = []
    groups = _nnunet_sahi_source_groups(
        inputs,
        manifests,
        upscale_factor=model.upscale_factor,
        classes=session.num_classes,
    )
    inference_started = time.perf_counter()
    completed_tiles = 0
    last_text_report = inference_started
    warmup_completed_tiles = 0
    steady_state_started: float | None = None
    if progress:
        print(
            f"nnU-Net SAHI inference: 0/{total_tiles:,} tiles across "
            f"{len(groups):,} work groups; requested batch {session.requested_batch_size}.",
            flush=True,
        )
    tile_progress = tqdm(
        total=total_tiles,
        desc="nnU-Net SAHI tiles",
        unit="tile",
        file=sys.stderr,
        dynamic_ncols=True,
        disable=not progress,
    )

    def write_progress(message: str) -> None:
        print(message, flush=True)

    try:
        for group_index, group in enumerate(groups, start=1):
            group_tiles = sum(len(manifests[value.image_id]) for value in group)
            if progress and group_index == 1:
                write_progress(
                    f"Preparing first work group on CPU: {len(group):,} source "
                    f"image(s), {group_tiles:,} tiles. This is the CUDA warm-up "
                    "group and is excluded from the steady-state ETA."
                )
            started = time.perf_counter()
            spans: dict[str, tuple[int, int]] = {}
            images: list[np.ndarray] = []
            for value in group:
                tiles = _slice_source_tiles(
                    value,
                    manifests[value.image_id],
                    upscale_factor=model.upscale_factor,
                )
                spans[value.image_id] = (len(images), len(images) + len(tiles))
                images.extend(tiles)
            prepared = session.preprocess_many(images)
            del images
            preprocess_seconds = time.perf_counter() - started
            telemetry.preprocess_seconds += preprocess_seconds
            if progress and group_index == 1:
                write_progress(
                    f"First work group preprocessed in {preprocess_seconds:.1f}s; "
                    "sending tiles to CUDA."
                )

            # Folds are outermost inside predict_logits, so a multi-fold model
            # loads each fold once for the whole group rather than per tile.
            started = time.perf_counter()
            logits: list[Any] = [None] * len(prepared)
            forward_batch_sizes: list[int] = []

            def report_oom_backoff(
                attempted: int,
                retry: int,
                retry_number: int,
                error: str,
            ) -> None:
                detail = " ".join(error.split())
                if len(detail) > 500:
                    detail = detail[:497] + "..."
                write_progress(
                    f"CUDA OOM during nnU-Net forward at batch {attempted}; "
                    f"cleared temporary CUDA allocations and retrying with batch "
                    f"{retry} (OOM retry {retry_number}).\n"
                    f"  CUDA error: {detail}"
                )

            for indices in _equal_shape_batches(
                [array for array, _ in prepared],
                classes=session.num_classes,
                minimum=session.resolved_batch_size,
            ):
                predicted = session.predict_logits(
                    [prepared[index][0] for index in indices],
                    on_batch=forward_batch_sizes.append,
                    on_oom=report_oom_backoff if progress else None,
                )
                for index, tile_logits in zip(indices, predicted):
                    logits[index] = tile_logits
                tile_progress.update(len(indices))
                completed_tiles += len(indices)
            inference_seconds = time.perf_counter() - started
            telemetry.inference_seconds += inference_seconds
            telemetry.resolved_batch_size = session.resolved_batch_size
            telemetry.oom_retries = session.oom_retries

            started = time.perf_counter()
            probabilities = session.to_probabilities_many(
                [(logits[index], prepared[index][1]) for index in range(len(prepared))]
            )
            conversion_seconds = time.perf_counter() - started
            telemetry.conversion_seconds += conversion_seconds
            del logits, prepared

            started = time.perf_counter()
            for value in group:
                first, last = spans[value.image_id]
                records.append(
                    _stitch_sahi_source(
                        value,
                        manifests[value.image_id],
                        probabilities[first:last],
                        model=model,
                        keep_native=keep_native,
                        resolved=resolved,
                    )
                )
                # Release each completed source image promptly.
                probabilities[first:last] = [None] * (last - first)
            stitch_seconds = time.perf_counter() - started
            telemetry.stitch_seconds += stitch_seconds
            del probabilities
            now = time.perf_counter()
            if group_index == 1:
                warmup_completed_tiles = completed_tiles
                steady_state_started = now
                last_text_report = now
                if progress:
                    write_progress(
                        f"nnU-Net CUDA warm-up complete: {completed_tiles:,} tiles; "
                        f"CPU prep {preprocess_seconds:.1f}s, GPU {inference_seconds:.1f}s, "
                        f"CPU post {conversion_seconds + stitch_seconds:.1f}s; "
                        f"active batch cap {session.resolved_batch_size}, actual forward "
                        f"batches {_format_batch_sizes(forward_batch_sizes)}, OOM retries "
                        f"{session.oom_retries}. Steady-state ETA will follow."
                    )
                    if completed_tiles == total_tiles:
                        write_progress(
                            f"nnU-Net SAHI progress: {completed_tiles:,}/{total_tiles:,} "
                            "tiles (100.0%); run completed within the warm-up group."
                        )
                continue
            if (
                progress
                and (
                    now - last_text_report >= 15
                    or completed_tiles == total_tiles
                )
            ):
                elapsed = max(now - (steady_state_started or inference_started), 1e-9)
                measured_tiles = completed_tiles - warmup_completed_tiles
                rate = measured_tiles / elapsed
                remaining = (
                    (total_tiles - completed_tiles) / rate if rate > 0 else math.inf
                )
                eta = (
                    _human_duration(remaining)
                    if math.isfinite(remaining)
                    else "unknown"
                )
                write_progress(
                    f"nnU-Net SAHI progress: {completed_tiles:,}/{total_tiles:,} "
                    f"tiles ({completed_tiles / total_tiles:.1%}), "
                    f"{rate:.2f} tile/s, ETA {eta}; group {group_index:,}/"
                    f"{len(groups):,} phases: CPU prep {preprocess_seconds:.1f}s, "
                    f"GPU {inference_seconds:.1f}s, CPU post "
                    f"{conversion_seconds + stitch_seconds:.1f}s; active batch cap "
                    f"{session.resolved_batch_size}, actual forward batches "
                    f"{_format_batch_sizes(forward_batch_sizes)}, OOM retries "
                    f"{session.oom_retries}."
                )
                last_text_report = now
    finally:
        tile_progress.close()
        telemetry.weight_loads = session.weight_loads
        telemetry.forward_passes = session.forward_passes
        session.release()
    for record in records:
        record.metadata.update(telemetry.as_dict())
    return tuple(records)


def _slice_source_tiles(
    value: ModelInput,
    manifest: tuple[Any, ...],
    *,
    upscale_factor: int,
) -> list[np.ndarray]:
    """Cut one source image into its canonical SAHI tiles at model input scale.

    Tiles are returned in manifest order, which is how stitching pairs them
    back with their source-coordinate boxes.
    """

    with Image.open(value.image_path) as opened:
        source_image = opened.convert("RGB")
    if source_image.size != (value.width, value.height):
        raise DatasetValidationError(
            f"Prediction input dimensions changed while slicing {value.image_path}"
        )
    tiles: list[np.ndarray] = []
    for tile in manifest:
        image = source_image.crop(tile.box)
        if upscale_factor != 1:
            image = image.resize(
                (tile.width * upscale_factor, tile.height * upscale_factor),
                Image.Resampling.BICUBIC,
            )
        tiles.append(np.asarray(image))
    return tiles


def _equal_shape_batches(
    prepared: list[np.ndarray],
    *,
    classes: int = 2,
    minimum: int = 1,
) -> list[list[int]]:
    """Group equally shaped preprocessed tiles into bounded inference calls.

    Only equally shaped tiles can share a network minibatch. Each group is then
    split so one call's padded inputs and accumulated logits stay within a
    fixed memory budget.
    """

    grouped: dict[tuple[int, ...], list[int]] = {}
    for index, array in enumerate(prepared):
        grouped.setdefault(tuple(array.shape), []).append(index)
    batches: list[list[int]] = []
    for shape in sorted(grouped):
        area = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        # Padded float32 input, plus fold logits and fold totals in half.
        per_tile = shape[0] * area * 4 + 2 * classes * area * 2
        limit = max(minimum, _NNUNET_SAHI_INFERENCE_CHUNK_BYTES // max(1, per_tile))
        indices = grouped[shape]
        batches.extend(
            indices[start : start + limit] for start in range(0, len(indices), limit)
        )
    return batches


def _stitch_sahi_source(
    value: ModelInput,
    manifest: tuple[Any, ...],
    probabilities: list[np.ndarray],
    *,
    model: Model,
    keep_native: bool,
    resolved: Any,
) -> ImagePrediction:
    from .sahi_support import stitch_probability_tiles

    tile_probabilities = [
        (
            tile,
            _validate_tile_probabilities(
                probabilities[index],
                expected_shape=(
                    tile.height * model.upscale_factor,
                    tile.width * model.upscale_factor,
                ),
                source=f"{model.name}/{value.image_id}/tile-{tile.index}",
            ),
        )
        for index, tile in enumerate(manifest)
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
            native_probabilities.shape[0],
            value.height,
            model.upscale_factor,
            value.width,
            model.upscale_factor,
        ).mean(axis=(2, 4))
    mask = np.argmax(canonical_probabilities, axis=0).astype(np.uint8)
    return ImagePrediction(
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
            "tile_count": len(manifest),
            **resolved.as_dict(),
        },
    )


def _nnunet_sahi_source_groups(
    inputs: tuple[ModelInput, ...],
    manifests: dict[str, tuple[Any, ...]],
    *,
    upscale_factor: int,
    classes: int,
) -> tuple[tuple[ModelInput, ...], ...]:
    """Bound in-flight work by the tile memory one group holds at once.

    A group's preprocessed inputs and converted probabilities are all resident
    until its source images have been stitched, so both are budgeted here.
    """

    channels = 3
    groups: list[tuple[ModelInput, ...]] = []
    pending: list[ModelInput] = []
    pending_bytes = 0
    for value in inputs:
        estimated_bytes = sum(
            (channels + classes)
            * tile.width
            * upscale_factor
            * tile.height
            * upscale_factor
            * np.dtype(np.float32).itemsize
            for tile in manifests[value.image_id]
        )
        if pending and (
            pending_bytes + estimated_bytes > _NNUNET_SAHI_PROBABILITY_GROUP_BYTES
            or len(pending) >= _NNUNET_SAHI_MAX_IMAGES_PER_GROUP
        ):
            groups.append(tuple(pending))
            pending = []
            pending_bytes = 0
        pending.append(value)
        pending_bytes += estimated_bytes
    if pending:
        groups.append(tuple(pending))
    return tuple(groups)


def _engine_telemetry(records: tuple[ImagePrediction, ...]) -> dict[str, Any]:
    """Lift the shared execution facts a prediction run recorded per image."""

    if not records:
        return {}
    metadata = records[0].metadata
    return {
        key: value
        for key, value in metadata.items()
        if key.startswith("nnunet_")
    }


def _validate_tile_probabilities(
    probabilities: np.ndarray,
    *,
    expected_shape: tuple[int, int],
    source: str,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float32)
    if (
        values.ndim < 3
        or values.shape[0] != 2
        or values.shape[-2:] != expected_shape
        or math.prod(values.shape[1:-2]) != 1
    ):
        raise DatasetValidationError(
            ValidationIssue(
                "nnU-Net SAHI probability dimensions do not match the tile adapter",
                source=source,
                value=values.shape,
                expected=f"(2, ..., {expected_shape[0]}, {expected_shape[1]})",
            )
        )
    if not np.all(np.isfinite(values)):
        raise DatasetValidationError(
            ValidationIssue(
                "nnU-Net SAHI probability tile contains non-finite values",
                source=source,
            )
        )
    return values.reshape(2, *expected_shape)


def visualize_nnunet_models(
    export: "Dataset",
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

    from .dataset import Dataset

    if not isinstance(export, Dataset) or export.format != "semantic_masks":
        raise TypeError("Semantic visualization requires a semantic-mask Dataset")
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
    if any(spec.inference != "sahi" for spec in cohort.models):
        _require_official_commands("nnUNetv2_predict_from_modelfolder")

    cases, _ = _freeze_cohort(export, split, progress=progress)
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
                device=spec._resolved_device(),
                progress=progress,
            )
            _write_semantic_prediction_masks(
                result.records,
                predictions,
                progress=progress,
                description=f"Saving {spec.name} predictions",
            )
            prediction_dirs[spec.name] = predictions
            rows_by_model[spec.name] = _sample_metric_rows(
                selected,
                predictions,
                spec.name,
                progress=progress,
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


def _parse_models(models: Any) -> list[Model]:
    collection = Model.load_many(models)
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
    *,
    progress: bool = False,
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
    cohort_progress = tqdm(
        total=len(images),
        desc="Freezing semantic cohort",
        unit="cohort image",
        disable=not progress,
    )
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
            image_size = opened_image.size
            if opened_mask.mode != "L":
                raise DatasetValidationError(
                    f"Semantic mask must be single-channel L: {mask_path}: {opened_mask.mode}"
                )
            if image_size != opened_mask.size:
                raise DatasetValidationError(
                    f"Semantic mask dimensions {opened_mask.size} do not match image "
                    f"dimensions {image_size}: {mask_path}"
                )
            histogram = opened_mask.histogram()
            values = {value for value, count in enumerate(histogram) if count}
            if not values <= {0, 1, 255}:
                raise DatasetValidationError(
                    f"Semantic mask contains values outside 0/1/255: {mask_path}: {sorted(values)[:10]}"
                )
            width, height = image_size
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
        cohort_progress.update(1)
    cohort_progress.close()
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


def _rebase_cached_semantic_rows(
    rows: Iterable[Mapping[str, Any]],
    cases: list[_SemanticCase],
    *,
    model_name: str,
) -> list[dict[str, Any]]:
    """Refresh location-only row metadata after a dataset/cache is moved."""

    by_id = {case.case_id: case for case in cases}
    output: list[dict[str, Any]] = []
    for row in rows:
        value = dict(row)
        case = by_id[str(value["case_id"])]
        value.update(
            {
                "model": model_name,
                "relative_path": case.relative_path.as_posix(),
                "source_image": str(case.image_path),
                "source_mask": str(case.mask_path),
            }
        )
        output.append(value)
    return output


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
    *,
    progress: bool = False,
    description: str = "Saving semantic predictions",
) -> None:
    canonical_dir.mkdir(parents=True, exist_ok=True)
    if native_dir is not None:
        native_dir.mkdir(parents=True, exist_ok=True)
    for record in tqdm(
        records,
        desc=description,
        unit="mask",
        disable=not progress,
    ):
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


def _run_command(
    command: list[str],
    *,
    progress: bool = False,
    progress_total: int | None = None,
    progress_directory: Path | None = None,
    progress_description: str = "nnU-Net command",
) -> None:
    if progress:
        print(f"Running nnU-Net command: {shlex.join(command)}")
    try:
        if progress and progress_total is not None and progress_directory is not None:
            _run_command_with_output_progress(
                command,
                total=progress_total,
                output_directory=progress_directory,
                description=progress_description,
            )
        else:
            subprocess.run(
                command,
                check=True,
                text=True,
                # nnU-Net prints several lines and nested progress bars for every
                # case. Capturing keeps non-interactive evaluations readable while
                # retaining stdout/stderr for the failure diagnostic below.
                capture_output=not progress,
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


def _run_command_with_output_progress(
    command: list[str],
    *,
    total: int,
    output_directory: Path,
    description: str,
) -> None:
    """Replace nnU-Net's per-case chatter with one exported-case progress bar."""

    total = max(0, int(total))
    progress_bar = tqdm(
        total=total,
        desc=description,
        unit="case",
        dynamic_ncols=True,
    )
    stop = threading.Event()
    completed = 0

    def monitor_outputs() -> None:
        nonlocal completed
        while True:
            available = min(
                total,
                sum(1 for path in output_directory.glob("*.png") if path.is_file()),
            )
            if available > completed:
                progress_bar.update(available - completed)
                completed = available
            if stop.wait(0.1):
                # One final scan catches outputs published between the last
                # interval and process completion.
                available = min(
                    total,
                    sum(
                        1
                        for path in output_directory.glob("*.png")
                        if path.is_file()
                    ),
                )
                if available > completed:
                    progress_bar.update(available - completed)
                    completed = available
                return

    monitor = threading.Thread(
        target=monitor_outputs,
        name="dataset-fixer-nnunet-native-progress",
        daemon=True,
    )
    monitor.start()
    succeeded = False
    try:
        with tempfile.TemporaryDirectory(
            prefix="dataset-fixer-nnunet-command-log-"
        ) as temporary:
            log_path = Path(temporary) / "nnunet.log"
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    subprocess.run(
                        command,
                        check=True,
                        text=True,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        capture_output=False,
                    )
                succeeded = True
            except subprocess.CalledProcessError as exc:
                exc.stdout = log_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
                raise
    finally:
        stop.set()
        monitor.join()
        if succeeded and completed < total:
            # A mocked command or an unusual exporter may finish without PNGs
            # appearing incrementally. Successful completion is authoritative.
            progress_bar.update(total - completed)
        progress_bar.close()


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


def _all_pairwise_statistics(
    rows_by_model: dict[str, list[dict[str, Any]]],
    *,
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Compare every unordered model pair without designating a reference."""

    scores_by_model = {
        name: {
            row["case_id"]: row["dice"]
            for row in rows
            if math.isfinite(row["dice"])
        }
        for name, rows in rows_by_model.items()
    }
    names = list(rows_by_model)
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    raw_p: list[float] = []
    for left_index, model_a in enumerate(names):
        for model_b in names[left_index + 1 :]:
            scores_a = scores_by_model[model_a]
            scores_b = scores_by_model[model_b]
            keys = sorted(set(scores_a) & set(scores_b))
            differences = np.asarray(
                [scores_b[key] - scores_a[key] for key in keys],
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
                    "model_a": model_a,
                    "model_b": model_b,
                    "metric": "canonical.per_case.Dice",
                    "difference_model_b_minus_model_a": float(differences.mean()),
                    "ci_low": float(np.quantile(samples, 0.025)),
                    "ci_high": float(np.quantile(samples, 0.975)),
                    "p_value": p_value,
                    "paired_cases": len(differences),
                    "wins_model_b": int(np.sum(differences > 0)),
                    "ties": int(np.sum(differences == 0)),
                    "wins_model_a": int(np.sum(differences < 0)),
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


def _sortable_score(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return -math.inf
    return parsed if math.isfinite(parsed) else -math.inf


def _model_report_fields(model: Model) -> dict[str, Any]:
    resolution = model.effective_resolution
    if resolution is None:
        resolution_label = "unknown"
        resolution_size = None
    elif resolution[0] == resolution[1]:
        resolution_label = f"{resolution[0]}px"
        resolution_size = list(resolution)
    else:
        resolution_label = f"{resolution[1]}x{resolution[0]}px"
        resolution_size = list(resolution)
    checkpoint_hash = model.checkpoint_sha256 or model.digest
    return {
        "model_source": model.source_key,
        "source_dataset_zip": model.source_dataset_zip,
        "model_type": model.model_type,
        "effective_prediction_resolution": resolution_label,
        "effective_prediction_size": resolution_size,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_sha256_short": checkpoint_hash[:8],
        "model_sha256_short": model.digest[:8],
    }


def _shorten_plot_text(value: str, limit: int = 64) -> str:
    if len(value) <= limit:
        return value
    left = (limit - 1) // 2
    right = limit - left - 1
    return f"{value[:left]}…{value[-right:]}"


def _ranking_plot_label(row: Mapping[str, Any]) -> str:
    name = str(row.get("model") or row.get("model_source") or "model")
    parts = name.split("__")
    if len(parts) >= 2:
        return "\n".join(
            (
                _shorten_plot_text(parts[0]),
                _shorten_plot_text("__".join(parts[1:])),
            )
        )
    return _shorten_plot_text(name)


def _render_ranking(
    root: Path,
    ranking: list[dict[str, Any]],
    *,
    xlabel: str = "Canonical probability-pooled foreground mean Dice",
    title: str = "nnU-Net semantic-mask model comparison",
) -> list[str]:
    import matplotlib.pyplot as plt

    ordered = list(reversed(ranking))
    figure, axis = plt.subplots(
        figsize=(11.5, max(4.0, 1.05 * len(ordered) + 1.5))
    )
    names = [_ranking_plot_label(row) for row in ordered]
    scores = [
        float(row["dice"]) if math.isfinite(float(row["dice"])) else 0.0
        for row in ordered
    ]
    positions = np.arange(len(ordered))
    axis.barh(positions, scores, color="#0072B2")
    axis.set_yticks(positions, names, fontsize=8.5, linespacing=1.15)
    axis.set_xlim(0, 1)
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    for index, (score, row) in enumerate(zip(scores, ordered)):
        label = f"{float(row['dice']):.3f}" if math.isfinite(float(row["dice"])) else "n/a"
        axis.text(min(score + 0.01, 0.98), index, label, va="center")
    figure.tight_layout()
    path = root / "plots.png"
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


def _render_semantic_prediction_grids(
    root: Path,
    cases: list[_SemanticCase],
    prediction_dirs: dict[str, Path],
    rows_by_model: dict[str, list[dict[str, Any]]],
    *,
    case_ids: Iterable[str] | None = None,
) -> list[str]:
    """Write one image-level comparison with at most two models per row.

    Only ``case_ids`` are rendered. A full cohort is far more output than
    anyone inspects -- a thousand-image split produced a thousand figures and
    gigabytes of PNGs -- so the caller selects the cases worth keeping and
    nothing else is drawn.
    """

    import matplotlib.pyplot as plt

    model_names = list(prediction_dirs)
    row_lookup = {
        name: {str(row["case_id"]): row for row in rows}
        for name, rows in rows_by_model.items()
    }
    if case_ids is not None:
        selected = set(case_ids)
        cases = [case for case in cases if case.case_id in selected]
    output_root = root / "predictions"
    rendered: list[str] = []
    for case in cases:
        with Image.open(case.mask_path) as opened:
            truth = np.asarray(opened.convert("L")) > 0
        predictions = {}
        for name in model_names:
            with Image.open(prediction_dirs[name] / f"{case.case_id}.png") as opened:
                predictions[name] = np.asarray(opened.convert("L")) > 0
        # A case with no reference and no prediction draws an empty overlay on
        # every panel, so there is nothing in it to look at.
        if not truth.any() and not any(value.any() for value in predictions.values()):
            continue
        columns = min(2, len(model_names))
        rows = math.ceil(len(model_names) / columns)
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(5 * columns, 5 * rows),
            squeeze=False,
        )
        with Image.open(case.image_path) as opened:
            image = np.asarray(opened.convert("RGB"), dtype=np.float32)
        for index, name in enumerate(model_names):
            row, column = divmod(index, columns)
            prediction = predictions[name]
            overlay = image.copy()
            overlay[truth] = 0.55 * overlay[truth] + 0.45 * np.asarray([0, 200, 90])
            overlay[prediction] = 0.55 * overlay[prediction] + 0.45 * np.asarray([215, 50, 160])
            metric = row_lookup[name][case.case_id]
            axes[row, column].imshow(overlay.astype(np.uint8))
            axes[row, column].set_title(
                f"{_multiline_model_title(name, 36)}\n"
                f"Dice={_format_metric(metric['dice'])} · "
                f"IoU={_format_metric(metric['iou'])}"
            )
            axes[row, column].axis("off")
        for index in range(len(model_names), rows * columns):
            row, column = divmod(index, columns)
            axes[row, column].axis("off")
        figure.suptitle(f"{case.relative_path} · green=truth, magenta=prediction")
        figure.tight_layout()
        relative = case.relative_path.with_suffix(".png")
        output = output_root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        rendered.append(str(output.relative_to(root)))
    return rendered


def _bounded_semantic_cases(
    rows: list[dict[str, Any]],
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id") or "")
        entry = by_case.setdefault(
            case_id,
            {
                "case_id": case_id,
                "relative_path": row.get("relative_path"),
                "models": [],
            },
        )
        entry["models"].append(
            {
                "model": row.get("model"),
                "dice": row.get("dice"),
                "iou": row.get("iou"),
                "fp": row.get("fp"),
                "fn": row.get("fn"),
            }
        )
    values = list(by_case.values())
    for value in values:
        finite = [
            float(row["dice"])
            for row in value["models"]
            if math.isfinite(float(row["dice"]))
        ]
        value["mean_dice"] = float(np.mean(finite)) if finite else None
    return sorted(
        values,
        key=lambda value: (
            math.inf if value["mean_dice"] is None else float(value["mean_dice"]),
            str(value["relative_path"]),
        ),
    )[:limit]


def _semantic_result_from_manifest(
    target: Path,
    manifest: dict[str, Any],
) -> SemanticComparisonResult:
    return SemanticComparisonResult(
        location=target,
        ranking=tuple(manifest.get("ranking") or ()),
        cohort_fingerprint=str(manifest.get("cohort_fingerprint") or ""),
        cohort_verified=bool(manifest.get("cohort_verified")),
        split=str(manifest.get("split") or "val"),
        settings=dict(manifest.get("settings") or {}),
        limitations=tuple(str(value) for value in manifest.get("limitations") or ()),
    )


def _load_compatible_semantic_cache(
    cache_dir: Path,
    legacy_cache_dirs: Iterable[Path],
    cases: list[_SemanticCase],
    *,
    cache_identity: dict[str, Any],
    required_fields: Iterable[str] = (),
    progress: bool = False,
    model_name: str | None = None,
) -> tuple[Path, dict[str, Any] | None]:
    """Load the current semantic cache key or a compatible historical key.

    First probe the canonical key, then discover entries whose stored logical
    identity matches even if an archive or older implementation placed them
    under a different directory hash. Cache identity also used to include
    execution-only values such as device, batching, worker count, and installed
    package versions, so the exact historical hash is checked as a final
    compatibility path. Every verified hit is promoted to the current key.
    """

    discovered: list[Path] = []
    unverifiable: list[tuple[Path, str]] = []
    expected_case_ids = {case.case_id for case in cases}
    if cache_dir.parent.is_dir():
        for metadata_path in cache_dir.parent.glob("*/evaluation.json"):
            if metadata_path.parent.name.startswith("."):
                # Cache publication is an atomic rename from a hidden staging
                # directory. Never discover or reuse unpublished work, even if
                # interruption happened after its metadata was written.
                continue
            try:
                value = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            stored_identity = value.get("cache_identity") if isinstance(value, dict) else None
            if (
                isinstance(stored_identity, dict)
                and cache_key(stored_identity) == cache_key(cache_identity)
            ):
                discovered.append(metadata_path.parent)
            elif not isinstance(stored_identity, dict) and isinstance(value, dict):
                rows = value.get("rows")
                if (
                    model_name is not None
                    and isinstance(rows, list)
                    and {str(row.get("model")) for row in rows} == {model_name}
                    and {str(row.get("case_id")) for row in rows} == expected_case_ids
                ):
                    unverifiable.append(
                        (
                            metadata_path.parent,
                            "legacy entry has no stored logical identity, so model bytes/settings cannot be verified",
                        )
                    )

    candidates = (cache_dir, *discovered, *legacy_cache_dirs)
    seen: set[Path] = set()
    invalid: list[tuple[Path, str]] = list(unverifiable)
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        cached, status = _inspect_semantic_cache(
            candidate,
            cases,
            progress=progress,
            model_name=model_name,
        )
        if cached is None:
            if status != "not found":
                invalid.append((candidate, status))
            continue
        missing_fields = [field for field in required_fields if field not in cached]
        if missing_fields:
            invalid.append(
                (candidate, f"metadata is missing fields: {', '.join(missing_fields)}")
            )
            continue
        if candidate != cache_dir:
            promoted = _promote_semantic_cache(
                candidate,
                cache_dir,
                {**cached, "cache_identity": cache_identity},
            )
            if promoted:
                candidate = cache_dir
            if progress:
                print(
                    f"Cache hit: {model_name or 'semantic model'} "
                    f"({len(cases)} completed masks; compatible historical cache key)"
                )
        elif progress:
            print(
                f"Cache hit: {model_name or 'semantic model'} "
                f"({len(cases)} completed masks)"
            )
        return candidate, cached
    if progress:
        for candidate, reason in invalid:
            print(
                f"Cache invalid: {model_name or 'semantic model'} at {candidate}: {reason}"
            )
        print(
            f"Cache miss: {model_name or 'semantic model'} "
            "(no valid completed prediction cache; running inference)"
        )
    return cache_dir, None


def _promote_semantic_cache(
    source: Path,
    destination: Path,
    metadata: dict[str, Any],
) -> bool:
    """Atomically move a verified historical cache entry to its current key."""

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        source.replace(destination)
    except OSError:
        # Recognition must still succeed on read-only or cross-device caches.
        return False
    try:
        write_json(destination / "evaluation.json", metadata)
    except OSError:
        # The directory name now carries the current identity; the already
        # verified metadata remains sufficient if refreshing it is impossible.
        pass
    return True


def _load_semantic_cache(
    cache_dir: Path,
    cases: list[_SemanticCase],
    *,
    progress: bool = False,
    model_name: str | None = None,
) -> dict[str, Any] | None:
    value, _ = _inspect_semantic_cache(
        cache_dir,
        cases,
        progress=progress,
        model_name=model_name,
    )
    return value


def _inspect_semantic_cache(
    cache_dir: Path,
    cases: list[_SemanticCase],
    *,
    progress: bool = False,
    model_name: str | None = None,
) -> tuple[dict[str, Any] | None, str]:
    metadata = cache_dir / "evaluation.json"
    predictions = cache_dir / "predictions"
    metadata_exists = metadata.is_file()
    predictions_exist = predictions.is_dir()
    if not metadata_exists and not predictions_exist:
        return None, "not found"
    if not metadata_exists:
        return None, "incomplete publication: evaluation.json is missing"
    if not predictions_exist:
        return None, "incomplete publication: predictions directory is missing"
    expected = {f"{case.case_id}.png" for case in cases}
    paths = list(predictions.glob("*.png"))
    actual = {
        path.name
        for path in tqdm(
            paths,
            desc=f"Loading {model_name or 'semantic'} cache",
            unit="mask",
            disable=not progress,
        )
        if path.is_file()
    }
    if actual != expected:
        return None, (
            f"prediction set mismatch (expected {len(expected)}, found {len(actual)}, "
            f"missing {len(expected - actual)}, unexpected {len(actual - expected)})"
        )
    try:
        value = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "evaluation.json is unreadable"
    if not isinstance(value, dict) or value.get("schema") != SEMANTIC_EVALUATION_CACHE_SCHEMA:
        return None, "evaluation.json has an incompatible schema"
    rows = value.get("rows")
    if not isinstance(rows, list) or {str(row.get("case_id")) for row in rows} != {
        case.case_id for case in cases
    }:
        return None, "evaluation rows do not match the frozen cohort"
    return value, "valid"


def _save_semantic_cache(
    cache_dir: Path,
    prediction_dir: Path,
    metadata: dict[str, Any],
    *,
    progress: bool = False,
    model_name: str | None = None,
) -> None:
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{cache_dir.name}.building-", dir=cache_dir.parent)
    )
    try:
        staged_predictions = staging / "predictions"
        staged_predictions.mkdir(parents=True)
        prediction_paths = sorted(
            path for path in prediction_dir.glob("*.png") if path.is_file()
        )
        for path in tqdm(
            prediction_paths,
            desc=f"Caching {model_name or 'semantic'} predictions",
            unit="mask",
            disable=not progress,
        ):
            shutil.copy2(path, staged_predictions / path.name)
        write_json(
            staging / "evaluation.json",
            {"schema": SEMANTIC_EVALUATION_CACHE_SCHEMA, **metadata},
        )
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        staging.replace(cache_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


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
    *,
    progress: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in tqdm(
        cases,
        desc=f"Scoring {model_name}",
        unit="case",
        disable=not progress,
    ):
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
    column_titles = [
        "Original",
        "GT",
        *[
            _multiline_model_title(name, model_title_length)
            for name in model_names
        ],
    ]
    heading_lines = max(title.count("\n") + 1 for title in column_titles)
    panel_count = 2 + len(model_names)
    grid_rows = math.ceil(len(cases) / examples_per_row)
    group_width = panel_size * panel_count
    image_title_height = 0.28
    heading_height = max(0.3, 0.18 * heading_lines)
    group_height = panel_size + image_title_height + heading_height
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
    for index, case in enumerate(cases):
        grid_row = index // examples_per_row
        grid_column = index % examples_per_row
        cell = outer[grid_row, grid_column].subgridspec(
            3,
            panel_count,
            height_ratios=(image_title_height, heading_height, panel_size),
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
                    fontsize=8,
                    linespacing=1.15,
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


def _multiline_model_title(value: str, line_width: int) -> str:
    """Format canonical model identities as readable narrow-column titles."""

    parts = value.split("__")
    if len(parts) >= 4:
        lines = textwrap.wrap(
            parts[0],
            width=line_width,
            break_long_words=True,
            break_on_hyphens=True,
        )
        lines.append(parts[1])
        lines.append(" · ".join(parts[2:]))
    else:
        lines = textwrap.wrap(
            value,
            width=line_width,
            break_long_words=True,
            break_on_hyphens=True,
        )
    if len(lines) > 5:
        remainder = "".join(lines[4:])
        lines = [*lines[:4], _shorten_middle(remainder, line_width)]
    return "\n".join(lines)


def _format_metric(value: Any) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{parsed:.3f}" if math.isfinite(parsed) else "n/a"


def _human_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _format_batch_sizes(values: Iterable[int]) -> str:
    counts = Counter(int(value) for value in values)
    if not counts:
        return "none"
    return ", ".join(
        f"{size}×{counts[size]}" for size in sorted(counts, reverse=True)
    )


def _visualization_destination(destination: str | Path) -> Path:
    path = Path(destination).expanduser().resolve()
    if path.suffix:
        if path.suffix.lower() != ".png":
            raise ValueError("visualization destination must be a PNG file or directory")
        return path
    return path / "comparison.png"
