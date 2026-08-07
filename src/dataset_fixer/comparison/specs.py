from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..errors import DatasetValidationError, ValidationIssue
from ..model import Model, ModelCollection
from .types import ModelSpec


def parse_models(models: Any) -> list[ModelSpec]:
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
                "resolution": loaded.resolution or 480,
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
        resolution = int(settings.get("resolution", 480))
        if resolution <= 0:
            issues.append(ValidationIssue("Model resolution must be positive", source=name, value=resolution))
            continue
        adapter_settings = {
            key: value
            for key, value in settings.items()
            if key
            not in {
                "path",
                "model_folder",
                "training_dataset",
                "resolution",
                "confidence",
                "postprocess",
                "inference",
            }
        }
        resolved_model = loaded or Model(
            path,
            name=name,
            resolution=resolution,
            training_dataset=settings.get("training_dataset"),
            inference=str(settings.get("inference", "native")),
            confidence=float(settings.get("confidence", 0.25)),
            postprocess=float(settings.get("postprocess", 0.7)),
            settings=adapter_settings,
        )
        training = settings.get("training_dataset") or resolved_model.training_dataset
        result.append(
            ModelSpec(
                name=name,
                path=path,
                training_dataset=Path(training).expanduser().resolve() if training else None,
                resolution=resolution,
                confidence=resolved_model.confidence,
                postprocess=resolved_model.postprocess,
                inference_overrides={
                    key: value
                    for key, value in settings.items()
                    if key
                    not in {
                        "path", "model_folder", "training_dataset", "resolution",
                        "confidence", "postprocess",
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
