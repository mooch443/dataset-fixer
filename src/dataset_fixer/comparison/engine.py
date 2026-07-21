from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import DatasetValidationError, ValidationIssue
from ..utils import ensure_safe_destination, environment_snapshot, package_versions, settings_fingerprint, to_jsonable
from .cache import (
    append_migration_log,
    cache_key,
    default_cache_root,
    import_notebook_cache,
    load_package_cache,
    model_cache_dir,
    model_hash,
    notebook_cache_dirs,
    notebook_cache_basename,
    notebook_dataset_hash,
    save_package_cache,
    write_notebook_numpy_cache,
)
from .cohort import check_training_provenance, freeze_cohort, write_cohort
from .inference import resolve_backend, run_inference
from .metrics import bootstrap_metric, evaluate_configuration, paired_statistics
from .reporting import render_figures, render_qualitative, write_csv, write_json, write_tables
from .specs import parse_models
from .types import Cohort, ComparisonResult, ModelSpec, Prediction

if TYPE_CHECKING:
    from ..dataset import Dataset


def compare_models(
    dataset: "Dataset",
    models: Any,
    *,
    split: str = "val",
    baseline: str | None = None,
    inference: str = "auto",
    protocol: str = "validation",
    calibration_split: str | None = None,
    training_provenance: str = "required",
    confidence_thresholds: tuple[float, ...] = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85),
    postprocess_thresholds: tuple[float, ...] = (0.75, 0.85, 0.95),
    resolution: int = 480,
    comparison_unit: str = "model",
    cache: bool | str | Path = True,
    notebook_cache: str | Path | None = None,
    write_notebook_cache: bool = False,
    allow_unverified_cache: bool = False,
    visualize: bool = True,
    progress: bool = True,
    destination: str | Path | None = None,
    device: str | None = None,
    seed: int = 42,
    bootstrap_resamples: int = 10_000,
    **inference_settings: Any,
) -> ComparisonResult:
    """Evaluate multiple model configurations on one cryptographically frozen cohort."""

    started = time.time()
    protocol = protocol.lower()
    if protocol not in {"validation", "locked", "calibrate_then_test"}:
        raise ValueError("protocol must be 'validation', 'locked', or 'calibrate_then_test'")
    if comparison_unit not in {"model", "system"}:
        raise ValueError("comparison_unit must be 'model' or 'system'")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    if allow_unverified_cache and training_provenance == "required":
        raise ValueError(
            "allow_unverified_cache is exploratory and cannot be combined with training_provenance='required'"
        )
    specs = parse_models(
        models,
        default_resolution=resolution,
        confidence_thresholds=tuple(confidence_thresholds),
        postprocess_thresholds=tuple(postprocess_thresholds),
    )
    if baseline is None:
        baseline = specs[0].name
    if baseline not in {spec.name for spec in specs}:
        raise ValueError(f"Unknown baseline {baseline!r}")
    cohort = freeze_cohort(dataset, split)
    calibration: Cohort | None = None
    if protocol == "calibrate_then_test":
        if not calibration_split:
            raise ValueError("calibration_split is required for protocol='calibrate_then_test'")
        calibration = freeze_cohort(dataset, calibration_split)
        if calibration.fingerprint == cohort.fingerprint or calibration.split == cohort.split:
            raise DatasetValidationError("Calibration and evaluation splits must be distinct frozen cohorts")
    backend = resolve_backend(inference, cohort.task)
    if comparison_unit != "system" and any(spec.inference_overrides.get("inference", backend) != backend for spec in specs):
        raise ValueError("Mixed native/SAHI configurations require comparison_unit='system'")
    overlap, provenance_complete, leakage, limitations = check_training_provenance(
        specs, cohort, training_provenance
    )
    independent_clusters = len({record.original_id for record in cohort.records})
    if independent_clusters < 10:
        limitations.append(
            f"Only {independent_clusters} independent ultimate-original clusters are available; paired uncertainty estimates may be unstable."
        )

    resolved_settings = {
        "split": cohort.split,
        "baseline": baseline,
        "inference_requested": inference,
        "inference_resolved": backend,
        "protocol": protocol,
        "calibration_split": calibration.split if calibration else None,
        "training_provenance": training_provenance,
        "confidence_thresholds": confidence_thresholds,
        "postprocess_thresholds": postprocess_thresholds,
        "resolution": resolution,
        "comparison_unit": comparison_unit,
        "cache": bool(cache),
        "visualize": visualize,
        "device": device,
        "seed": seed,
        "bootstrap_resamples": bootstrap_resamples,
        **inference_settings,
    }
    fingerprint = settings_fingerprint(to_jsonable(resolved_settings))
    target = (
        Path(destination).expanduser().resolve()
        if destination
        else dataset.location.parent / f"{dataset.name}__compare-models__{fingerprint}"
    )
    ensure_safe_destination(dataset.location, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.building-", dir=target.parent))
    cache_root = _cache_root(cache, dataset.location)
    cache_migration_log = temporary / "cache_migrations.jsonl"
    package_versions_snapshot = package_versions()
    print(
        f"Comparing {len(specs)} models on frozen {cohort.split!r} cohort "
        f"({len(cohort.records)} images, fingerprint {cohort.fingerprint[:12]})\nDestination: {target}\n"
        f"Backend: requested={inference}, resolved={backend}; protocol={protocol}"
    )
    try:
        write_cohort(cohort, temporary / "evaluation-cohort.jsonl")
        cache_migration_log.touch()
        (temporary / "predictions").mkdir(parents=True, exist_ok=True)
        if calibration:
            write_cohort(calibration, temporary / "calibration-cohort.jsonl")
        write_json(temporary / "reports" / "leakage_audit.json", leakage)
        write_json(
            temporary / "reports" / "cohort_audit.json",
            {
                "fingerprint": cohort.fingerprint,
                "verified": True,
                "split": cohort.split,
                "images": len(cohort.records),
                "annotations": sum(len(record.annotations) for record in cohort.records),
                "independent_clusters": independent_clusters,
            },
        )
        model_outputs: dict[str, dict[str, Any]] = {}
        cache_audit: dict[str, Any] = {}
        model_hashes: dict[str, str] = {}
        for spec in specs:
            sha = model_hash(spec.path)
            model_hashes[spec.name] = sha
            spec_backend = str(spec.inference_overrides.get("inference", backend))
            spec_backend = resolve_backend(spec_backend, cohort.task)
            model_outputs[spec.name] = _evaluate_model(
                spec,
                cohort,
                calibration,
                backend=spec_backend,
                protocol=protocol,
                cache_root=cache_root,
                notebook_cache=Path(notebook_cache).expanduser().resolve() if notebook_cache else None,
                cache_log=cache_migration_log,
                model_sha=sha,
                device=device,
                progress=progress,
                settings={**inference_settings, **spec.inference_overrides},
                allow_unverified_cache=allow_unverified_cache,
            )
            cache_audit[spec.name] = model_outputs[spec.name]["cache"]

        primary_metric = "map100_10" if cohort.task == "polo" else "map50_95"
        ranking: list[dict[str, Any]] = []
        full_grid: list[dict[str, Any]] = []
        per_image: list[dict[str, Any]] = []
        per_class: list[dict[str, Any]] = []
        pr_data: dict[str, Any] = {}
        best_rows: dict[str, list[dict[str, Any]]] = {}
        best_predictions: dict[str, dict[str, list[Prediction]]] = {}
        best_confidences: dict[str, float] = {}
        for spec in specs:
            output = model_outputs[spec.name]
            grid_rows = output["grid"]
            full_grid.extend({"model": spec.name, **row["summary"], "confidence": row["confidence"], "postprocess": row["postprocess"], "score": row["summary"][primary_metric]} for row in grid_rows)
            best = output["best"]
            ci_low, ci_high = bootstrap_metric(best["per_image"], resamples=bootstrap_resamples, seed=seed)
            duration = float(output["timing"].get("inference_seconds", 0))
            rank_row = {
                "model": spec.name,
                "configuration": f"{spec.name}@{spec.resolution}/{output['backend']}",
                "backend": output["backend"],
                "metric": primary_metric,
                "score": best["summary"][primary_metric],
                "uncertainty_metric": "ultimate_original_macro_f1",
                "uncertainty_score": best["summary"]["cluster_macro_f1"],
                "ci_low": ci_low,
                "ci_high": ci_high,
                "confidence": output["best_confidence"],
                "postprocess": output["best_postprocess"],
                "resolution": spec.resolution,
                "cohort_fingerprint": cohort.fingerprint,
                "support_images": best["summary"]["support_images"],
                "support_annotations": best["summary"]["support_annotations"],
                "support_clusters": best["summary"]["support_clusters"],
                "tp": best["summary"]["tp"], "fp": best["summary"]["fp"], "fn": best["summary"]["fn"],
                "precision": best["summary"]["precision"], "recall": best["summary"]["recall"], "f1": best["summary"]["f1"],
                "inference_seconds": duration,
                "throughput_images_per_second": len(cohort.records) / duration if duration > 0 else None,
                "protocol_label": "validation/model-selection" if protocol == "validation" else protocol,
            }
            ranking.append(rank_row)
            best_rows[spec.name] = best["per_image"]
            per_image.extend({"model": spec.name, **row} for row in best["per_image"])
            per_class.extend({"model": spec.name, **row} for row in best["per_class"])
            pr_data[spec.name] = best["pr"]
            best_predictions[spec.name] = output["predictions"][output["best_postprocess"]]
            best_confidences[spec.name] = output["best_confidence"]
        ranking.sort(key=lambda row: (-row["score"], row["model"]))
        for index, row in enumerate(ranking, start=1):
            row["rank"] = index
        _assert_ranking_invariants(ranking, cohort)
        paired = paired_statistics(best_rows, baseline, resamples=bootstrap_resamples, seed=seed)

        write_csv(temporary / "metrics" / "ranking.csv", ranking)
        write_csv(temporary / "metrics" / "full_grid.csv", full_grid)
        write_csv(temporary / "metrics" / "per_image.csv", per_image)
        write_csv(temporary / "metrics" / "per_class.csv", per_class)
        write_csv(temporary / "metrics" / "paired_statistics.csv", paired)
        write_tables(temporary, ranking)
        write_json(temporary / "reports" / "cache_audit.json", cache_audit)
        write_json(
            temporary / "predictions" / "cache-locations.json",
            {name: value.get("root") for name, value in cache_audit.items()},
        )
        write_json(temporary / "reports" / "limitations.json", {"limitations": limitations})
        figure_paths: list[str] = []
        qualitative_paths: list[str] = []
        figure_metadata = {
            "cohort_fingerprint": cohort.fingerprint,
            "model_hashes": model_hashes,
            "backends": {name: value["backend"] for name, value in model_outputs.items()},
            "cache_sources": {name: value.get("source") for name, value in cache_audit.items()},
            "independent_clusters": independent_clusters,
            "metric_definition": primary_metric,
        }
        if visualize:
            figure_paths = render_figures(
                temporary, cohort=cohort, ranking=ranking, grid=full_grid, per_class=per_class,
                paired=paired, per_image=per_image, pr_data=pr_data, cache_audit=cache_audit,
                leakage_audit=leakage,
                metadata=figure_metadata,
            )
            qualitative_paths = render_qualitative(
                temporary, cohort, best_predictions, best_confidences,
                [row["model"] for row in ranking],
                baseline=baseline,
            )
        if write_notebook_cache:
            _export_notebook_caches(
                temporary, specs, cohort, model_outputs, model_hashes,
            )

        cache_verified = bool(cache_root) and all(bool(value.get("verified")) for value in cache_audit.values())
        cache_statistics = {
            "models": cache_audit,
            "package_hits": sum(value.get("source") == "package" for value in cache_audit.values()),
            "notebook_imports": sum(str(value.get("source", "")).startswith("notebook") for value in cache_audit.values()),
            "fresh_inference": sum(value.get("source") == "fresh" for value in cache_audit.values()),
        }
        manifest = {
            "schema": 1,
            "kind": "model-comparison",
            "dataset": {"name": dataset.name, "location": str(dataset.location), "task": cohort.task},
            "cohort_fingerprint": cohort.fingerprint,
            "cohort_verified": True,
            "training_overlap_detected": overlap,
            "training_provenance_complete": provenance_complete,
            "cache_verified": cache_verified,
            "cache_statistics": cache_statistics,
            "protocol": protocol,
            "settings": resolved_settings,
            "settings_fingerprint": fingerprint,
            "model_hashes": model_hashes,
            "ranking": ranking,
            "limitations": limitations,
            "figures": figure_paths,
            "qualitative": qualitative_paths,
            "environment": environment_snapshot(),
            "started_at_unix": started,
            "completed_at_unix": time.time(),
        }
        write_json(temporary / "model-comparison.json", manifest)
        temporary.replace(target)
    except Exception:
        _remove_build_dir(temporary)
        raise
    print(
        f"Comparison complete: {target}\nCohort verified: yes; training overlap: "
        f"{'detected' if overlap else 'none detected'}; cache verified: {'yes' if cache_verified else 'no'}"
    )
    return ComparisonResult(
        location=target,
        ranking=tuple(ranking),
        cohort_fingerprint=cohort.fingerprint,
        cohort_verified=True,
        training_overlap_detected=overlap,
        training_provenance_complete=provenance_complete,
        cache_verified=cache_verified,
        cache_statistics=cache_statistics,
        protocol=protocol,
        settings=resolved_settings,
        limitations=tuple(limitations),
    )


def _evaluate_model(
    spec: ModelSpec,
    cohort: Cohort,
    calibration: Cohort | None,
    *, backend: str, protocol: str, cache_root: Path | None, notebook_cache: Path | None,
    cache_log: Path, model_sha: str, device: str | None, progress: bool,
    settings: dict[str, Any], allow_unverified_cache: bool,
) -> dict[str, Any]:
    confidences = spec.confidence_thresholds or ()
    posts = spec.postprocess_thresholds or ()
    if protocol == "locked":
        if spec.locked_confidence is not None: confidences = (spec.locked_confidence,)
        if spec.locked_postprocess is not None: posts = (spec.locked_postprocess,)
        if len(confidences) != 1 or len(posts) != 1:
            raise ValueError(f"Locked protocol requires one confidence and postprocess setting for {spec.name}")
    floor = min(confidences)

    def get_predictions(active: Cohort, active_posts: tuple[float, ...]) -> tuple[dict[float, dict[str, list[Prediction]]], dict[str, Any], dict[str, float]]:
        payload = {
            "model_sha256": model_sha, "cohort_fingerprint": active.fingerprint, "task": active.task,
            "classes": active.classes, "backend": backend, "versions": package_versions(),
            "resolution": spec.resolution, "confidence_floor": floor, "postprocess_thresholds": active_posts,
            "device": device, "settings": settings,
        }
        root = model_cache_dir(cache_root, spec.name, cache_key(payload)) if cache_root else None
        loaded: dict[float, dict[str, list[Prediction]]] = {}
        shards = 0
        source = "fresh"
        imported_verified = False
        if root:
            loaded, shards, complete = load_package_cache(root, active, active_posts)
            if complete:
                source = "package"
        if not loaded and active.task == "polo" and backend == "sahi" and len(active.classes) == 1:
            legacy_dirs = notebook_cache_dirs(Path(active.records[0].image_path).parents[2], notebook_cache)
            imported, import_meta = import_notebook_cache(
                legacy_dirs,
                model_sha256=model_sha, resolution=spec.resolution, confidence_floor=floor,
                thresholds=active_posts, cohort=active,
                expected_key={
                    "device": str(device),
                    "overlap_height_ratio": float(settings.get("overlap_height_ratio", settings.get("overlap", .2))),
                    "overlap_width_ratio": float(settings.get("overlap_width_ratio", settings.get("overlap", .2))),
                    "postprocess_class_agnostic": bool(settings.get("postprocess_class_agnostic", False)),
                    "model_type": str(settings.get("model_type", "ultralytics")),
                },
                allow_unverified=allow_unverified_cache,
            )
            if imported:
                loaded = imported; source = import_meta["format"]
                imported_verified = bool(import_meta.get("verified"))
                append_migration_log(cache_log, {"event": "import", **import_meta, "model": spec.name})
            else:
                candidates = [
                    str(path)
                    for directory in legacy_dirs if directory.is_dir()
                    for pattern in ("*.gridcache_v2", "*.gridcache.pkl")
                    for path in directory.glob(pattern)
                ]
                if candidates:
                    append_migration_log(
                        cache_log,
                        {"event": "rejection", "model": spec.name, "candidates": candidates,
                         "reason": "no candidate passed key, structure, cohort, and content validation"},
                    )
        start = time.perf_counter()
        predictions, timings = run_inference(
            spec, active, backend=backend, thresholds=active_posts, confidence_floor=floor, device=device,
            progress=progress, settings=settings, existing=loaded,
            on_threshold=(lambda threshold, values: save_package_cache(root, active, payload, {**loaded, threshold: values})) if root else None,
        )
        inference_seconds = time.perf_counter() - start if source == "fresh" or len(loaded) < len(active_posts) else 0.0
        if root:
            save_package_cache(root, active, payload, predictions)
            verified_loaded, verified_shards, verified = load_package_cache(root, active, active_posts)
            if not verified or set(verified_loaded) != set(map(float, active_posts)):
                raise DatasetValidationError("Package prediction cache failed post-write verification")
            shards = verified_shards
        else:
            verified = False
        content_verified = imported_verified if source.startswith("notebook") else (verified if root else source == "fresh")
        return predictions, {"source": source, "verified": content_verified, "shards": shards, "root": str(root) if root else None}, {"inference_seconds": inference_seconds, **timings}

    grid: list[dict[str, Any]] = []
    tune_grid: list[dict[str, Any]] = []
    metric_name = "map100_10" if cohort.task == "polo" else "map50_95"
    if calibration is not None:
        calibration_predictions, calibration_cache, calibration_timing = get_predictions(calibration, posts)
        for post in posts:
            for confidence in confidences:
                tuned = evaluate_configuration(calibration, calibration_predictions[float(post)], confidence)
                tune_grid.append({"confidence": confidence, "postprocess": post, **tuned})
        selected = max(tune_grid, key=lambda row: (row["summary"][metric_name], row["summary"]["f1"], -row["confidence"], -row["postprocess"]))
        selected_post = float(selected["postprocess"])
        evaluation_predictions, cache_info, timing = get_predictions(cohort, (selected_post,))
        cache_info["calibration"] = calibration_cache
        timing["calibration_inference_seconds"] = calibration_timing.get("inference_seconds", 0)
        evaluated = evaluate_configuration(cohort, evaluation_predictions[selected_post], float(selected["confidence"]))
        best = {"confidence": selected["confidence"], "postprocess": selected_post, **evaluated}
        grid = [best]
    else:
        evaluation_predictions, cache_info, timing = get_predictions(cohort, posts)
        for post in posts:
            for confidence in confidences:
                evaluated = evaluate_configuration(cohort, evaluation_predictions[float(post)], confidence)
                grid.append({"confidence": confidence, "postprocess": post, **evaluated})
        selected = max(grid, key=lambda row: (row["summary"][metric_name], row["summary"]["f1"], -row["confidence"], -row["postprocess"]))
        best = selected
    return {
        "backend": backend, "predictions": evaluation_predictions, "cache": cache_info, "timing": timing,
        "grid": grid, "best": best, "best_confidence": selected["confidence"], "best_postprocess": selected["postprocess"],
    }


def _cache_root(cache: bool | str | Path, location: Path) -> Path | None:
    if cache is False: return None
    if isinstance(cache, (str, Path)) and not isinstance(cache, bool): return Path(cache).expanduser().resolve()
    return default_cache_root(location)


def _assert_ranking_invariants(rows: list[dict[str, Any]], cohort: Cohort) -> None:
    expected = (cohort.fingerprint, len(cohort.records), sum(len(r.annotations) for r in cohort.records), len({r.original_id for r in cohort.records}))
    for row in rows:
        actual = (row["cohort_fingerprint"], row["support_images"], row["support_annotations"], row["support_clusters"])
        if actual != expected:
            raise DatasetValidationError(
                ValidationIssue("Ranking denominators differ from the frozen cohort", source=row["model"], value=actual, expected=str(expected))
            )


def _export_notebook_caches(
    root: Path, specs: list[ModelSpec], cohort: Cohort, outputs: dict[str, dict[str, Any]],
    hashes: dict[str, str],
) -> None:
    if cohort.task != "polo" or len(cohort.classes) != 1:
        raise ValueError("write_notebook_cache requires a single-class POLO cohort")
    export_root = root / "predictions" / "notebook-cache"
    first = cohort.records[0].image_path
    dataset_root = first.parent.parent
    if first.parent.name != "images" or not (dataset_root / "labels").is_dir():
        raise ValueError(
            "Notebook cache export requires the notebook split layout "
            "<split>/images/* and <split>/labels/* (nested canonical images/{split} is not directly loadable by the notebook)"
        )
    dataset_sha = notebook_dataset_hash(dataset_root)
    for spec in specs:
        output = outputs[spec.name]
        if output["backend"] != "sahi":
            raise ValueError("write_notebook_cache requires POLO predictions produced through SAHI")
        confidence_floor = min(spec.confidence_thresholds or ())
        key = {
            "cache_version": 2, "cache_format": "gridcache_v2_numpy_sharded",
            "model_path": str(spec.path), "model_hash": hashes[spec.name],
            "dataset_root": str(dataset_root.resolve()), "dataset_hash": dataset_sha,
            "resolution": spec.resolution, "min_conf": confidence_floor,
            "iou_list": sorted(output["predictions"]), "device": None,
            "overlap_height_ratio": spec.inference_overrides.get("overlap_height_ratio", .2),
            "overlap_width_ratio": spec.inference_overrides.get("overlap_width_ratio", .2),
            "postprocess_class_agnostic": spec.inference_overrides.get("postprocess_class_agnostic", False),
            "model_type": spec.inference_overrides.get("model_type", "ultralytics"),
        }
        target = export_root / notebook_cache_basename(spec.path, key, numpy=True)
        write_notebook_numpy_cache(target, key=key, cohort=cohort, predictions=output["predictions"])
        imported, imported_meta = import_notebook_cache(
            [export_root], model_sha256=hashes[spec.name], resolution=spec.resolution,
            confidence_floor=confidence_floor, thresholds=tuple(sorted(output["predictions"])),
            cohort=cohort,
            expected_key={
                "device": "None",
                "overlap_height_ratio": float(key["overlap_height_ratio"]),
                "overlap_width_ratio": float(key["overlap_width_ratio"]),
                "postprocess_class_agnostic": bool(key["postprocess_class_agnostic"]),
                "model_type": str(key["model_type"]),
            },
        )
        if imported is None or not imported_meta or not imported_meta.get("verified"):
            raise DatasetValidationError("Notebook cache round-trip verification failed")
        _assert_notebook_roundtrip(output["predictions"], imported)
        append_migration_log(root / "cache_migrations.jsonl", {"event": "export", "model": spec.name, "target": str(target)})


def _remove_build_dir(path: Path) -> None:
    import shutil
    shutil.rmtree(path, ignore_errors=True)


def _assert_notebook_roundtrip(
    expected: dict[float, dict[str, list[Prediction]]],
    actual: dict[float, dict[str, list[Prediction]]],
) -> None:
    if set(expected) != set(actual):
        raise DatasetValidationError("Notebook cache round trip changed postprocessing thresholds")
    for threshold, by_image in expected.items():
        if list(by_image) != list(actual[threshold]):
            raise DatasetValidationError("Notebook cache round trip changed image ordering")
        for image_id, predictions in by_image.items():
            restored = actual[threshold][image_id]
            if len(predictions) != len(restored):
                raise DatasetValidationError("Notebook cache round trip changed prediction counts")
            for left, right in zip(predictions, restored):
                if left.point is None or right.point is None:
                    raise DatasetValidationError("Notebook cache round trip lost POLO points")
                if any(abs(a - b) > 1e-3 for a, b in zip(left.point, right.point)) or abs(left.score - right.score) > 1e-5:
                    raise DatasetValidationError("Notebook cache round trip changed prediction values")
