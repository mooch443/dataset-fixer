from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ..errors import DatasetValidationError, ValidationIssue
from ..sahi_support import reject_legacy_sahi_settings, resolve_sahi_settings
from ..utils import environment_snapshot, settings_fingerprint, to_jsonable
from .cache import (
    build_staging_dir,
    cache_key,
    default_cache_root,
    load_evaluation_cache,
    load_package_cache,
    model_cache_dir,
    save_package_cache,
    save_evaluation_cache,
)
from .cohort import check_training_provenance, freeze_cohort
from .inference import resolve_backend, run_inference
from .metrics import bootstrap_metric, evaluate_configuration, paired_statistics
from .reporting import (
    combine_report_plots,
    render_figures,
    render_prediction_grids,
    render_qualitative,
    write_json,
)
from .specs import parse_models
from .types import Cohort, ComparisonResult, ModelSpec, Prediction

if TYPE_CHECKING:
    from ..dataset import Dataset


def _compare_models(
    dataset: "Dataset",
    models: Any,
    *,
    split: str = "val",
    save_prediction_plots: bool = False,
    progress: bool = True,
    destination: str | Path | None = None,
    errors: Literal["raise", "skip"] = "raise",
) -> ComparisonResult:
    """Evaluate multiple model configurations on one cryptographically frozen cohort."""

    started = time.time()
    protocol = "fixed"
    seed = 42
    bootstrap_resamples = 10_000
    specs = parse_models(models)
    cohort = freeze_cohort(dataset, split, progress=progress)
    model_backends = {
        spec.name: resolve_backend(
            str(spec.inference_overrides.get("inference", "native")), cohort.task
        )
        for spec in specs
    }
    model_systems: dict[str, dict[str, Any]] = {}
    for spec in specs:
        settings = dict(spec.inference_overrides)
        reject_legacy_sahi_settings(settings)
        system = {
            "resolution": spec.resolution,
            "backend": model_backends[spec.name],
            "confidence": spec.resolved_model.confidence,
            "postprocess": spec.resolved_model.postprocess,
        }
        if model_backends[spec.name] == "sahi":
            system["sahi"] = resolve_sahi_settings(
                settings, resolution=spec.resolution
            ).as_dict()
        model_systems[spec.name] = system
    comparison_unit = (
        "model"
        if len({settings_fingerprint(value) for value in model_systems.values()}) == 1
        else "system"
    )
    overlap, provenance_complete, leakage, limitations = check_training_provenance(
        specs, cohort, "warn"
    )
    geometry_skips = list(getattr(dataset, "_geometry_skip_audit", ()))
    if geometry_skips:
        limitations.append(
            f"Skipped {len(geometry_skips)} oversized evaluation image(s); "
            "details are recorded in settings.source_size_policy."
        )
    independent_clusters = len({record.original_id for record in cohort.records})
    if independent_clusters < 10:
        limitations.append(
            f"Only {independent_clusters} independent ultimate-original clusters are available; paired uncertainty estimates may be unstable."
        )

    resolved_settings = {
        "report_schema": 6,
        "split": cohort.split,
        "protocol": protocol,
        "training_provenance": "verify-when-configured",
        "comparison_unit": comparison_unit,
        "seed": seed,
        "bootstrap_resamples": bootstrap_resamples,
        "models": model_systems,
        "source_size_policy": {
            "errors": errors,
            "maximum_size": to_jsonable(
                getattr(dataset, "_geometry_maximum_size", None)
            ),
            "smaller_or_equal": "retain",
            "oversized": "skip" if errors == "skip" else "raise",
            "skipped_inputs": geometry_skips,
        },
    }
    model_hashes = {spec.name: spec.resolved_model.digest for spec in specs}
    fingerprint = settings_fingerprint(
        to_jsonable(
            {
                "schema": 5,
                "cohort": cohort.fingerprint,
                "models": [
                    {
                        "name": spec.name,
                        "sha256": model_hashes[spec.name],
                        "resolution": spec.resolution,
                        "settings": spec.inference_overrides,
                    }
                    for spec in specs
                ],
                "settings": resolved_settings,
            }
        )
    )
    target = (
        Path(destination).expanduser().resolve()
        if destination
        else dataset.location / "evaluations" / fingerprint
    )
    if target == dataset.location:
        raise ValueError("Comparison destination cannot replace the dataset")
    target.parent.mkdir(parents=True, exist_ok=True)
    existing_result = target / "reports" / "result.json"
    if (
        existing_result.is_file()
        and (target / "reports" / "plots.png").is_file()
        and (target / "reports" / "comparison.png").is_file()
        and (not save_prediction_plots or (target / "predictions").is_dir())
    ):
        cached_manifest = json.loads(existing_result.read_text(encoding="utf-8"))
        if cached_manifest.get("schema") == 6:
            print(f"Reusing complete comparison: {target}")
            return _result_from_manifest(target, cached_manifest)
    temporary = build_staging_dir(
        target,
        dataset_location=None if destination else dataset.location,
    )
    cache_root = default_cache_root(dataset.location)
    print(
        f"Comparing {len(specs)} models on frozen {cohort.split!r} cohort "
        f"({len(cohort.records)} images, fingerprint {cohort.fingerprint[:12]})\nDestination: {target}\n"
        f"Systems: {comparison_unit}; protocol={protocol}"
    )
    try:
        statistics_key = f"comparison-{fingerprint}"
        cached_statistics = load_evaluation_cache(cache_root, statistics_key)
        model_outputs: dict[str, dict[str, Any]] = {}
        cache_audit: dict[str, Any] = {}
        for spec in specs:
            sha = model_hashes[spec.name]
            spec_backend = model_backends[spec.name]
            model_outputs[spec.name] = _evaluate_model(
                spec,
                cohort,
                backend=spec_backend,
                protocol=protocol,
                cache_root=cache_root,
                model_sha=sha,
                device=spec.resolved_model.device,
                progress=progress,
                settings=dict(spec.inference_overrides),
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
            cached_interval = (
                (cached_statistics or {}).get("intervals", {}).get(spec.name)
            )
            if isinstance(cached_interval, list) and len(cached_interval) == 2:
                ci_low, ci_high = map(float, cached_interval)
            else:
                ci_low, ci_high = bootstrap_metric(
                    best["per_image"], resamples=bootstrap_resamples, seed=seed
                )
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
                "protocol_label": protocol,
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
        if cached_statistics is not None and isinstance(cached_statistics.get("paired"), list):
            paired = list(cached_statistics["paired"])
        else:
            paired = paired_statistics(
                best_rows, resamples=bootstrap_resamples, seed=seed
            )
            save_evaluation_cache(
                cache_root,
                statistics_key,
                {
                    "intervals": {
                        row["model"]: [row["ci_low"], row["ci_high"]]
                        for row in ranking
                    },
                    "paired": paired,
                },
            )

        figure_metadata = {
            "cohort_fingerprint": cohort.fingerprint,
            "model_hashes": model_hashes,
            "backends": {name: value["backend"] for name, value in model_outputs.items()},
            "cache_sources": {name: value.get("source") for name, value in cache_audit.items()},
            "independent_clusters": independent_clusters,
            "metric_definition": primary_metric,
        }
        render_figures(
            temporary, cohort=cohort, ranking=ranking, grid=full_grid, per_class=per_class,
            paired=paired, per_image=per_image, pr_data=pr_data, cache_audit=cache_audit,
            leakage_audit=leakage,
            metadata=figure_metadata,
        )
        render_qualitative(
            temporary, cohort, best_predictions, best_confidences,
            [row["model"] for row in ranking],
            seed=seed,
        )
        reports_dir = temporary / "reports"
        combine_report_plots((temporary / "figures",), reports_dir / "plots.png")
        combine_report_plots((temporary / "qualitative",), reports_dir / "comparison.png")
        prediction_paths: list[str] = []
        if save_prediction_plots:
            prediction_paths = render_prediction_grids(
                temporary,
                cohort,
                best_predictions,
                best_confidences,
                [row["model"] for row in ranking],
            )

        cache_verified = bool(cache_root) and all(bool(value.get("verified")) for value in cache_audit.values())
        cache_statistics = {
            "models": cache_audit,
            "prediction_hits": sum(value.get("source") == "package" for value in cache_audit.values()),
            "evaluation_hits": sum(value.get("evaluation") == "hit" for value in cache_audit.values()),
            "fresh_inference": sum(value.get("source") == "fresh" for value in cache_audit.values()),
        }
        manifest = {
            "schema": 6,
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
            "reports": {
                "plots": "reports/plots.png",
                "comparison": "reports/comparison.png",
                "prediction_plots": prediction_paths,
            },
            "paired_statistics": paired,
            "worst_cases": _bounded_worst_cases(per_image),
            "environment": environment_snapshot(),
            "started_at_unix": started,
            "completed_at_unix": time.time(),
        }
        write_json(reports_dir / "result.json", manifest)
        if target.exists():
            previous = target / "reports" / "result.json"
            if not previous.is_file():
                raise FileExistsError(f"Refusing to replace unrelated comparison destination: {target}")
            shutil.rmtree(target)
        temporary.replace(target)
    except BaseException:
        # BaseException, not Exception: a cancelled run (KeyboardInterrupt)
        # must not leave a partial evaluation behind either.
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
    *, backend: str, protocol: str, cache_root: Path,
    model_sha: str, device: str | None, progress: bool,
    settings: dict[str, Any],
) -> dict[str, Any]:
    confidence = float(spec.confidence)
    postprocess = float(spec.postprocess)
    floor = confidence
    evaluation_key = cache_key(
        {
            "schema": 1,
            "model_sha256": model_sha,
            "cohort_fingerprint": cohort.fingerprint,
            "task": cohort.task,
            "classes": cohort.classes,
            "backend": backend,
            "resolution": spec.resolution,
            "confidence": confidence,
            "postprocess": postprocess,
            "protocol": protocol,
            "settings": settings,
        }
    )

    def get_predictions(active: Cohort, active_posts: tuple[float, ...]) -> tuple[dict[float, dict[str, list[Prediction]]], dict[str, Any], dict[str, float]]:
        # Execution choices such as device and batch size are deliberately not
        # part of this payload. They change how inference runs, not its logical
        # model/dataset/prediction identity.
        payload = {
            "model_sha256": model_sha, "cohort_fingerprint": active.fingerprint, "task": active.task,
            "classes": active.classes, "backend": backend,
            "resolution": spec.resolution, "confidence_floor": floor, "postprocess_thresholds": active_posts,
            "settings": settings,
        }
        prediction_identity = {
            key: value
            for key, value in payload.items()
            if key != "postprocess_thresholds"
        }
        root = model_cache_dir(cache_root, spec.name, cache_key(prediction_identity))
        loaded: dict[float, dict[str, list[Prediction]]] = {}
        shards = 0
        source = "fresh"
        loaded, shards, complete = load_package_cache(
            root, active, active_posts, progress=progress
        )
        if complete:
            source = "package"
            if progress:
                print(f"Cache hit: {spec.name} ({shards} prediction shards)")
        checkpointed = dict(loaded)

        def checkpoint_threshold(
            threshold: float,
            values: dict[str, list[Prediction]],
        ) -> None:
            checkpointed[float(threshold)] = values
            save_package_cache(
                root,
                active,
                payload,
                checkpointed,
                progress=progress,
            )

        start = time.perf_counter()
        predictions, timings = run_inference(
            spec, active, backend=backend, thresholds=active_posts, confidence_floor=floor, device=device,
            progress=progress, settings=settings, existing=loaded,
            on_threshold=checkpoint_threshold,
        )
        inference_seconds = time.perf_counter() - start if source == "fresh" or len(loaded) < len(active_posts) else 0.0
        if not complete and set(checkpointed) != {
            float(value) for value in active_posts
        }:
            # Third-party/test inference adapters may not invoke the checkpoint
            # callback. Persist the completed result before metric/report work.
            save_package_cache(
                root, active, payload, predictions, progress=progress
            )
        if complete:
            verified_loaded, verified_shards, verified = loaded, shards, True
        else:
            verified_loaded, verified_shards, verified = load_package_cache(
                root, active, active_posts, progress=progress
            )
        if not verified or set(verified_loaded) != set(map(float, active_posts)):
            raise DatasetValidationError("Package prediction cache failed post-write verification")
        shards = verified_shards
        return predictions, {"source": source, "verified": verified, "shards": shards, "root": str(root)}, {"inference_seconds": inference_seconds, **timings}

    grid: list[dict[str, Any]] = []
    cached_evaluation = load_evaluation_cache(cache_root, evaluation_key)
    if cached_evaluation is not None:
        selected_post = float(cached_evaluation["best_postprocess"])
        evaluation_predictions, cache_info, timing = get_predictions(cohort, (selected_post,))
        cache_info["evaluation"] = "hit"
        return {
            "backend": backend,
            "predictions": evaluation_predictions,
            "cache": cache_info,
            "timing": timing,
            "grid": cached_evaluation["grid"],
            "best": cached_evaluation["best"],
            "best_confidence": float(cached_evaluation["best_confidence"]),
            "best_postprocess": selected_post,
            "settings": settings,
        }
    evaluation_predictions, cache_info, timing = get_predictions(
        cohort, (postprocess,)
    )
    selected_post = postprocess
    selected_confidence = confidence
    evaluated = evaluate_configuration(
        cohort, evaluation_predictions[selected_post], selected_confidence
    )
    selected = {
        "confidence": selected_confidence,
        "postprocess": selected_post,
        **evaluated,
    }
    grid = [selected]
    best = selected
    cached_result = {
        "grid": grid,
        "best": best,
        "best_confidence": selected["confidence"],
        "best_postprocess": selected["postprocess"],
    }
    save_evaluation_cache(cache_root, evaluation_key, cached_result)
    cache_info["evaluation"] = "fresh"
    return {
        "backend": backend, "predictions": evaluation_predictions, "cache": cache_info, "timing": timing,
        **cached_result,
        "settings": settings,
    }


def _assert_ranking_invariants(rows: list[dict[str, Any]], cohort: Cohort) -> None:
    expected = (cohort.fingerprint, len(cohort.records), sum(len(r.annotations) for r in cohort.records), len({r.original_id for r in cohort.records}))
    for row in rows:
        actual = (row["cohort_fingerprint"], row["support_images"], row["support_annotations"], row["support_clusters"])
        if actual != expected:
            raise DatasetValidationError(
                ValidationIssue("Ranking denominators differ from the frozen cohort", source=row["model"], value=actual, expected=str(expected))
            )


def _remove_build_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _bounded_worst_cases(rows: list[dict[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    by_image: dict[str, dict[str, Any]] = {}
    for row in rows:
        image_id = str(row.get("image_id") or row.get("relative_path") or "")
        entry = by_image.setdefault(
            image_id,
            {
                "image_id": image_id,
                "relative_path": row.get("relative_path"),
                "error_count": 0,
                "models": [],
            },
        )
        error_count = int(row.get("fp", 0)) + int(row.get("fn", 0))
        entry["error_count"] += error_count
        entry["models"].append(
            {
                "model": row.get("model"),
                "fp": int(row.get("fp", 0)),
                "fn": int(row.get("fn", 0)),
                "f1": row.get("f1"),
            }
        )
    return sorted(
        by_image.values(),
        key=lambda value: (-int(value["error_count"]), str(value["relative_path"])),
    )[:limit]


def _result_from_manifest(target: Path, manifest: dict[str, Any]) -> ComparisonResult:
    return ComparisonResult(
        location=target,
        ranking=tuple(manifest.get("ranking") or ()),
        cohort_fingerprint=str(manifest.get("cohort_fingerprint") or ""),
        cohort_verified=bool(manifest.get("cohort_verified")),
        training_overlap_detected=bool(manifest.get("training_overlap_detected")),
        training_provenance_complete=bool(manifest.get("training_provenance_complete")),
        cache_verified=bool(manifest.get("cache_verified")),
        cache_statistics=dict(manifest.get("cache_statistics") or {}),
        protocol=str(manifest.get("protocol") or "fixed"),
        settings=dict(manifest.get("settings") or {}),
        limitations=tuple(str(value) for value in manifest.get("limitations") or ()),
    )
