from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..errors import DatasetValidationError, ValidationIssue
from ..model import Model, ModelCollection
from .types import ModelSpec


def parse_models(
    models: Any,
    *,
    default_resolution: int,
    confidence_thresholds: tuple[float, ...],
    postprocess_thresholds: tuple[float, ...],
) -> list[ModelSpec]:
    if isinstance(models, ModelCollection):
        items = [(model.name, model) for model in models]
    elif isinstance(models, Model):
        items = [(models.name, models)]
    elif isinstance(models, (str, Path)):
        items = [(Path(models).stem, models)]
    elif isinstance(models, Mapping):
        items = list(models.items())
    elif isinstance(models, Sequence):
        items = [
            (value.name, value) if isinstance(value, Model) else (Path(value).stem, value)
            for value in models
        ]
    else:
        raise TypeError("models must be a checkpoint path, a sequence of paths, or a name-to-model mapping")

    issues: list[ValidationIssue] = []
    result: list[ModelSpec] = []
    seen: set[str] = set()
    for raw_name, value in items:
        name = str(raw_name).strip()
        if not name or name in seen:
            issues.append(ValidationIssue("Model names must be non-empty and unique", value=name))
            continue
        seen.add(name)
        settings = dict(value) if isinstance(value, Mapping) else {"path": value}
        loaded = value if isinstance(value, Model) else None
        if loaded is not None:
            settings = {
                "path": loaded.path,
                "training_dataset": loaded.training_dataset,
                "resolution": loaded.resolution or default_resolution,
                "inference": loaded.inference,
                **loaded.settings,
            }
        if "path" not in settings and "model_folder" not in settings:
            issues.append(ValidationIssue("Model specification is missing path", source=name))
            continue
        raw_path = settings.get("path", settings.get("model_folder"))
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            issues.append(
                ValidationIssue(
                    "Model checkpoint does not exist or is not a file",
                    source=name,
                    value=str(path),
                    suggestion="supply the exact checkpoint path",
                )
            )
            continue
        resolution = int(settings.get("resolution", default_resolution))
        if resolution <= 0:
            issues.append(ValidationIssue("Model resolution must be positive", source=name, value=resolution))
            continue
        conf = _thresholds(settings.get("confidence_thresholds", confidence_thresholds), "confidence", name, issues)
        post = _thresholds(settings.get("postprocess_thresholds", postprocess_thresholds), "postprocess", name, issues)
        adapter_settings = {
            key: value
            for key, value in settings.items()
            if key
            not in {
                "path",
                "model_folder",
                "training_dataset",
                "resolution",
                "confidence_thresholds",
                "postprocess_thresholds",
                "locked_confidence",
                "locked_postprocess",
                "inference",
            }
        }
        resolved_model = loaded or Model(
            path,
            name=name,
            resolution=resolution,
            training_dataset=settings.get("training_dataset"),
            inference=str(settings.get("inference", "auto")),
            settings=adapter_settings,
        )
        training = settings.get("training_dataset") or resolved_model.training_dataset
        result.append(
            ModelSpec(
                name=name,
                path=path,
                training_dataset=Path(training).expanduser().resolve() if training else None,
                resolution=resolution,
                confidence_thresholds=conf,
                postprocess_thresholds=post,
                locked_confidence=_optional_probability(settings.get("locked_confidence"), name, issues),
                locked_postprocess=_optional_probability(settings.get("locked_postprocess"), name, issues),
                inference_overrides={
                    key: value
                    for key, value in settings.items()
                    if key
                    not in {
                        "path", "model_folder", "training_dataset", "resolution", "confidence_thresholds",
                        "postprocess_thresholds", "locked_confidence", "locked_postprocess",
                    }
                },
                model=resolved_model,
            )
        )
    if issues:
        raise DatasetValidationError(issues)
    if not result:
        raise ValueError("At least one model is required")
    return result


def _thresholds(value: Any, kind: str, name: str, issues: list[ValidationIssue]) -> tuple[float, ...]:
    try:
        values = tuple(sorted({float(item) for item in value}))
    except (TypeError, ValueError):
        issues.append(ValidationIssue(f"Invalid {kind} thresholds", source=name, value=value))
        return ()
    if not values or any(not 0 <= item <= 1 for item in values):
        issues.append(
            ValidationIssue(
                f"Invalid {kind} thresholds",
                source=name,
                value=values,
                expected="one or more finite values in [0, 1]",
            )
        )
    return values


def _optional_probability(value: Any, name: str, issues: list[ValidationIssue]) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        issues.append(ValidationIssue("Invalid locked threshold", source=name, value=value))
        return None
    if not 0 <= result <= 1:
        issues.append(ValidationIssue("Locked threshold must be in [0, 1]", source=name, value=result))
    return result
