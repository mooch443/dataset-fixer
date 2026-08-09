"""Small, explicit Weights & Biases integration helpers."""

from __future__ import annotations

import importlib
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from ..bundle import Bundle, Config, Outcome
from ..convert import Prepared
from ..geometry import Geometry


def _size(value: tuple[int, int] | None) -> str | None:
    return f"{value[0]}x{value[1]}" if value is not None else None


def _values(config: Config | Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if isinstance(config, Config):
        geometry = (
            config.geometry
            if isinstance(config.geometry, Geometry)
            else Geometry.create(**dict(config.geometry))
        )
        dataset = config.dataset
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
            "dataset_content_sha256": dataset_hash,
            "dataset_preparation": preparation,
            **dict(config.training),
        }
    else:
        values = dict(config)
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
    return cleaned, list(dict.fromkeys(tags))


def configure(run: Any, config: Config | Mapping[str, Any]) -> Any:
    """Write the same searchable config keys and tags for every training task."""

    if run is None:
        raise ValueError("wandb.configure() requires an explicit existing run")
    values, tags = _values(config)
    target_config = getattr(run, "config", None)
    if target_config is None:
        raise TypeError("The supplied W&B run has no config")
    try:
        target_config.update(values, allow_val_change=True)
    except TypeError:
        target_config.update(values)
    existing = list(getattr(run, "tags", ()) or ())
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


def upload(
    run: Any,
    bundle: Bundle,
    outcome: Outcome | None = None,
) -> Bundle:
    """Upload a local bundle to an existing run without creating or logging in.

    Missing runs and all authentication/network failures leave the ZIP intact
    and return a :class:`Bundle` describing the local result.
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
