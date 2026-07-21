from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from ..errors import DatasetValidationError, ValidationIssue
from .types import ModelSpec


def parse_models(
    models: Any,
    *,
    default_resolution: int,
    confidence_thresholds: tuple[float, ...],
    postprocess_thresholds: tuple[float, ...],
) -> list[ModelSpec]:
    if isinstance(models, (str, Path)):
        items = [(Path(models).stem, models)]
    elif isinstance(models, Mapping):
        items = list(models.items())
    elif isinstance(models, Sequence):
        items = [(Path(value).stem, value) for value in models]
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
        if "path" not in settings:
            issues.append(ValidationIssue("Model specification is missing path", source=name))
            continue
        path = Path(settings["path"]).expanduser().resolve()
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
        training = settings.get("training_dataset") or _training_dataset_from_args(path)
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
                        "path", "training_dataset", "resolution", "confidence_thresholds",
                        "postprocess_thresholds", "locked_confidence", "locked_postprocess",
                    }
                },
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


def _training_dataset_from_args(checkpoint: Path) -> str | None:
    for directory in (checkpoint.parent, checkpoint.parent.parent):
        args = directory / "args.yaml"
        if not args.is_file():
            continue
        try:
            payload = yaml.safe_load(args.read_text(encoding="utf-8")) or {}
            data = payload.get("data")
            if not data:
                continue
            path = Path(str(data)).expanduser()
            if not path.is_absolute():
                path = (args.parent / path).resolve()
            if path.suffix.lower() in {".yaml", ".yml"}:
                return str(path)
            return str(path.resolve())
        except (OSError, TypeError, yaml.YAMLError):
            continue
    return None
