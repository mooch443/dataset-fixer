"""Small, explicit Weights & Biases integration helpers."""

from __future__ import annotations

import importlib
import inspect
import json
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from ..bundle import Bundle, Config, Outcome
from ..convert import Prepared
from ..geometry import Geometry


_HELDOUT_BREAKDOWN_FIELDS = (
    "dice",
    "iou",
    "micro_dice",
    "micro_iou",
    "foreground_precision",
    "foreground_recall",
    "positive_case_dice",
    "positive_case_iou",
    "positive_micro_dice",
    "positive_micro_iou",
    "positive_foreground_precision",
    "positive_foreground_recall",
    "positive_cases",
    "positive_detected_cases",
    "positive_missed_cases",
    "positive_image_recall",
    "empty_cases",
    "empty_correct_cases",
    "empty_false_positive_cases",
    "empty_image_specificity",
    "empty_image_false_positive_rate",
    "empty_false_positive_pixels",
    "empty_mean_false_positive_pixels",
)


def _size(value: tuple[int, int] | None) -> str | None:
    return f"{value[0]}x{value[1]}" if value is not None else None


def _imgsz(value: tuple[int, int] | None) -> int | list[int] | None:
    """Return the framework-neutral alias used by scalar W&B analyses."""

    if value is None:
        return None
    return value[0] if value[0] == value[1] else list(value)


def _dataset_source(value: Prepared | Mapping[str, Any]) -> str | None:
    """Return only the portable source folder/ZIP basename."""

    if isinstance(value, Prepared):
        return value.source_name
    dataset = dict(value)
    source = dataset.get("dataset_source") or dataset.get("source_dataset_zip")
    if source is None:
        nested = dataset.get("source_dataset")
        if isinstance(nested, Mapping):
            source = nested.get("basename") or nested.get("name") or nested.get("path")
    return Path(str(source)).name if source else None


def _model_family(framework: Any, task: Any, preparation: Any) -> str | None:
    """Collapse framework/task metadata into one comparable model family."""

    prepared = str(preparation or "").strip().lower().replace("_", "-")
    if prepared in {"nnunet", "yolo-sem", "yolo-seg"}:
        return prepared
    framework_name = str(framework or "").strip().lower()
    task_name = str(task or "").strip().lower().replace("_", "-")
    if "nnunet" in framework_name:
        return "nnunet"
    if "ultralytics" in framework_name or "yolo" in framework_name:
        return "yolo-sem" if "semantic" in task_name else "yolo-seg"
    return None


def _split_image_values(value: Prepared | Mapping[str, Any]) -> dict[str, int]:
    statistics = value.split_statistics if isinstance(value, Prepared) else value.get(
        "split_statistics", {}
    )
    aliases = {
        "train": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "test",
    }
    result: dict[str, int] = {}
    for split, raw in dict(statistics or {}).items():
        canonical = aliases.get(str(split).lower())
        if canonical is None or not isinstance(raw, Mapping) or raw.get("images") is None:
            continue
        result[f"dataset_{canonical}_images"] = int(raw["images"])
    return result


def _nested(value: Mapping[str, Any], *path: str) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _native_training_aliases(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize equivalent native trainer fields already present on a run."""

    candidates: dict[str, tuple[tuple[str, ...], ...]] = {
        "epochs": (("epochs",), ("train_args", "epochs"), ("hparas", "num_epochs")),
        "batch_size": (
            ("batch_size",),
            ("resolved_batch_size",),
            ("reproducibility", "resolved_batch_size"),
            ("hparas", "batch_size"),
            ("train_args", "batch"),
        ),
        "initial_lr": (
            ("initial_lr",),
            ("train_args", "lr0"),
            ("hparas", "initial_lr"),
        ),
        "weight_decay": (
            ("weight_decay",),
            ("train_args", "weight_decay"),
            ("hparas", "weight_decay"),
        ),
        "trainer": (("trainer",), ("reproducibility", "trainer")),
    }
    aliases: dict[str, Any] = {}
    for field, paths in candidates.items():
        for path in paths:
            selected = _nested(value, *path)
            if selected is None:
                continue
            if field == "batch_size":
                if isinstance(selected, bool):
                    continue
                try:
                    selected = int(selected)
                except (TypeError, ValueError):
                    continue
                if selected <= 0:
                    continue
            aliases[field] = selected
            break
    return aliases


def _values(config: Config | Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if isinstance(config, Config):
        geometry = (
            config.geometry
            if isinstance(config.geometry, Geometry)
            else Geometry.create(**dict(config.geometry))
        )
        dataset = config.dataset
        dataset_source = _dataset_source(dataset)
        if isinstance(dataset, Prepared):
            dataset_hash = dataset.content_sha256
            preparation = dataset.kind.value
        else:
            dataset_hash = dataset.get("content_sha256")
            preparation = dataset.get("preparation_kind")
        values = {
            "model_name": config.name,
            "framework": config.framework,
            "task": config.task,
            "native_tile_size": list(geometry.native_tile_size) if geometry.native_tile_size else None,
            "upscale_factor": geometry.upscale_factor,
            "model_input_size": list(geometry.input_size) if geometry.input_size else None,
            "imgsz": _imgsz(geometry.input_size),
            "dataset_content_sha256": dataset_hash,
            "dataset_preparation": preparation,
            "model_family": _model_family(config.framework, config.task, preparation),
            "dataset_source": dataset_source,
            "source_dataset_zip": (
                dataset_source
                if dataset_source and dataset_source.lower().endswith(".zip")
                else None
            ),
            **_split_image_values(dataset),
            **dict(config.training),
        }
    else:
        values = dict(config)
        if not values.get("dataset_source"):
            dataset_source = _dataset_source(values)
            if dataset_source:
                values["dataset_source"] = dataset_source
        geometry = Geometry.create(
            native_tile_size=values.get("native_tile_size"),
            upscale_factor=values.get("upscale_factor"),
            input_size=values.get("model_input_size", values.get("input_size")),
        )
        values["native_tile_size"] = (
            list(geometry.native_tile_size) if geometry.native_tile_size else None
        )
        values["upscale_factor"] = geometry.upscale_factor
        values["model_input_size"] = list(geometry.input_size) if geometry.input_size else None
        values.setdefault("imgsz", _imgsz(geometry.input_size))
        values.setdefault(
            "model_family",
            _model_family(
                values.get("framework"),
                values.get("task"),
                values.get("dataset_preparation", values.get("preparation_kind")),
            ),
        )
    # Only searchable top-level values are written.  There is intentionally no
    # duplicate nested metadata block.
    cleaned = {str(key): value for key, value in values.items() if value is not None}
    tags = [
        str(cleaned[key]).strip().lower().replace("_", "-")
        for key in ("framework", "task")
        if cleaned.get(key)
    ]
    if geometry.native_tile_size:
        tags.append(f"native-{_size(geometry.native_tile_size)}")
    if geometry.upscale_factor:
        tags.append(f"scale-{geometry.upscale_factor}x")
    if geometry.input_size:
        tags.append(f"input-{_size(geometry.input_size)}")
    dataset_source = cleaned.get("dataset_source")
    if dataset_source:
        tags.append(str(dataset_source))
    return cleaned, list(dict.fromkeys(tags))


def configure(run: Any, config: Config | Mapping[str, Any]) -> Any:
    """Write the same searchable config keys and tags for every training task.

    Parameters:
        run: Existing active W&B run or public API run object.
        config: Bundle configuration or equivalent top-level values.

    Returns:
        The configured run object.
    """

    if run is None:
        raise ValueError("wandb.configure() requires an explicit existing run")
    values, tags = _values(config)
    target_config = getattr(run, "config", None)
    if target_config is None:
        raise TypeError("The supplied W&B run has no config")
    for key, value in _native_training_aliases(target_config).items():
        values.setdefault(key, value)
    try:
        target_config.update(values, allow_val_change=True)
    except TypeError:
        target_config.update(values)
    existing = list(getattr(run, "tags", ()) or ())
    dataset_source = values.get("dataset_source")
    if dataset_source and str(dataset_source).lower().endswith(".zip"):
        legacy_source_tag = f"source-zip-{Path(str(dataset_source)).stem}"
        existing = [tag for tag in existing if tag != legacy_source_tag]
    run.tags = tuple(dict.fromkeys([*existing, *tags]))
    update = getattr(run, "update", None)
    if callable(update):
        update()
    return run


def _active_run() -> Any:
    try:
        sdk = importlib.import_module("wandb")
    except ImportError:
        return None
    return getattr(sdk, "run", None)


def _set_summary(run: Any, values: Mapping[str, Any]) -> None:
    summary = getattr(run, "summary", None)
    if summary is None:
        return
    for key, value in values.items():
        summary[key] = value
    update = getattr(summary, "update", None)
    if callable(update):
        try:
            update()
        except TypeError:
            pass


def _comparison_result_file(value: Any) -> tuple[Path, Path]:
    """Return the completed comparison manifest and its portable report root."""

    location = Path(str(value)).expanduser()
    if location.is_file():
        return location, location.parent
    return location / "reports" / "result.json", location


def _matching_ranking_row(rows: list[Any], run: Any) -> Mapping[str, Any] | None:
    candidates = [row for row in rows if isinstance(row, Mapping)]
    if len(candidates) == 1:
        return candidates[0]

    run_id = str(getattr(run, "id", "") or "").strip()
    if not run_id:
        return None
    matches = []
    for row in candidates:
        model = str(row.get("model", ""))
        source = str(row.get("model_source", ""))
        if source == run_id or source.rstrip("/").endswith(f"/{run_id}"):
            matches.append(row)
        elif f"__{run_id}__" in model:
            matches.append(row)
    return matches[0] if len(matches) == 1 else None


def _heldout_breakdown_values(
    comparison_report: Any,
    run: Any,
) -> tuple[dict[str, Any], str | None]:
    """Read all held-out metrics from one atomically completed comparison."""

    result_file, report_root = _comparison_result_file(comparison_report)
    try:
        manifest = json.loads(result_file.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {}, f"cannot read {result_file}: {type(exc).__name__}: {exc}"

    if not isinstance(manifest, Mapping):
        return {}, f"invalid comparison manifest in {result_file}"
    kind = manifest.get("kind")
    dataset = manifest.get("dataset")
    native_segment_report = (
        kind == "model-comparison"
        and isinstance(dataset, Mapping)
        and dataset.get("task") == "segment"
    )
    if kind != "semantic-mask-model-comparison" and not native_segment_report:
        return {}, f"report has no binary segmentation comparison space: {result_file}"
    if manifest.get("cohort_verified") is not True or manifest.get("completed_at_unix") is None:
        return {}, f"comparison report is not marked complete and cohort-verified: {result_file}"

    ranking = manifest.get("ranking")
    if not isinstance(ranking, list):
        return {}, f"comparison report has no ranking: {result_file}"
    row = _matching_ranking_row(ranking, run)
    if row is None:
        return {}, (
            "comparison report contains multiple models and none uniquely matches "
            f"W&B run {getattr(run, 'id', None)!r}: {result_file}"
        )
    missing = [field for field in _HELDOUT_BREAKDOWN_FIELDS if field not in row]
    if missing:
        return {}, (
            "comparison report predates the complete held-out breakdown; rerun evaluation "
            f"with the current dataset-fixer (missing: {', '.join(missing)}): {result_file}"
        )

    try:
        source_file = result_file.relative_to(report_root).as_posix()
    except ValueError:
        source_file = result_file.name
    values = {
        "schema": 1,
        "source": "completed dataset-fixer semantic comparison report",
        "case_unit": "final postprocessed source image",
        "source_artifact": report_root.name,
        "source_file": source_file,
        **{field: row[field] for field in _HELDOUT_BREAKDOWN_FIELDS},
    }
    return {f"heldout_breakdown/{key}": value for key, value in values.items()}, None


def upload(
    run: Any,
    bundle: Bundle,
    outcome: Outcome | None = None,
) -> Bundle:
    """Upload a local bundle to an existing run without creating or logging in.

    Missing runs and all authentication/network failures leave the ZIP intact
    and return a :class:`Bundle` describing the local result.

    Parameters:
        run: Explicit existing W&B run, or ``None`` to use ``wandb.run``.
        bundle: Local bundle returned by :func:`dataset_fixer.bundle.create`.
        outcome: Optional result metadata written after a successful upload.

    Returns:
        A bundle retaining its local path and recording any remote outcome.
    """

    if not isinstance(bundle, Bundle):
        raise TypeError("bundle must be bundle.Bundle")
    selected = run if run is not None else _active_run()
    if selected is None or not getattr(selected, "id", None):
        message = (
            f"No active W&B run; kept local bundle {bundle.path} "
            f"({bundle.size:,} bytes, sha256={bundle.sha256})."
        )
        print(message)
        return replace(bundle, warnings=(*bundle.warnings, message))
    try:
        print(f"Uploading model bundle to W&B: {bundle.path.name} ({bundle.size:,} bytes) ...")
        remote_url = None
        upload_file = getattr(selected, "upload_file", None)
        if callable(upload_file):
            try:
                supports_root = "root" in inspect.signature(upload_file).parameters
            except (TypeError, ValueError):
                supports_root = False
            if supports_root:
                # Public API runs need ``root`` to preserve the bundle's
                # basename as the remote file name.
                remote = upload_file(str(bundle.path), root=str(bundle.path.parent))
            else:
                remote = upload_file(str(bundle.path))
            remote_url = getattr(remote, "url", None)
        else:
            save = getattr(selected, "save", None)
            if not callable(save):
                raise TypeError("The supplied W&B run cannot upload files")
            remote = save(
                str(bundle.path),
                base_path=str(bundle.path.parent),
                policy="now",
            )
            if isinstance(remote, (list, tuple)) and remote:
                remote_url = getattr(remote[0], "url", None)
            else:
                remote_url = getattr(remote, "url", None)
        summary_values: dict[str, Any] = {
            "evaluation_bundle": bundle.path.name,
            "evaluation_bundle_sha256": bundle.sha256,
            "evaluation_bundle_size": bundle.size,
        }
        if outcome is not None:
            summary_values.update(
                {
                    "selected_epoch": outcome.selected_epoch,
                    "selection_metric": outcome.selection_metric,
                    "selection_value": outcome.selection_value,
                    **dict(outcome.metrics),
                }
            )
            comparison_report = outcome.metrics.get("comparison_report")
            if comparison_report is not None:
                breakdown, reason = _heldout_breakdown_values(
                    comparison_report,
                    selected,
                )
                if reason is not None:
                    warnings.warn(
                        f"Held-out breakdown was not published: {reason}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    # Explicit caller-supplied summary fields remain authoritative.
                    for key, value in breakdown.items():
                        summary_values.setdefault(key, value)
                    print(
                        "Published 28 heldout_breakdown fields from the completed "
                        "semantic comparison report."
                    )
        _set_summary(
            selected,
            {key: value for key, value in summary_values.items() if value is not None},
        )
        return replace(bundle, uploaded=True, remote_url=remote_url)
    except Exception as exc:
        message = (
            f"W&B upload failed ({type(exc).__name__}: {exc}); local bundle remains at "
            f"{bundle.path} ({bundle.size:,} bytes, sha256={bundle.sha256})."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return replace(bundle, warnings=(*bundle.warnings, message))


__all__ = ["configure", "upload"]
