"""Normalized model/dataset geometry and pre-inference compatibility checks."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, TypeVar

from PIL import Image

from .errors import DatasetValidationError, ValidationIssue


Size = tuple[int, int]
InputT = TypeVar("InputT")


def normalize_errors(errors: str) -> Literal["raise", "skip"]:
    """Normalize the shared recoverable-error policy."""

    value = errors.lower()
    if value not in {"raise", "skip"}:
        raise ValueError("errors must be 'raise' or 'skip'")
    return value


def exceeds_size(size: Size, maximum: Size) -> bool:
    """Return whether either image dimension exceeds its declared maximum."""

    return size[0] > maximum[0] or size[1] > maximum[1]


def filter_inputs_by_size(
    inputs: Sequence[InputT],
    *,
    maximum: Size | None,
    errors: str,
    source: str,
) -> tuple[tuple[InputT, ...], tuple[dict[str, Any], ...]]:
    """Keep at-most-maximum inputs and raise or audit oversized inputs."""

    policy = normalize_errors(errors)
    if maximum is None:
        return tuple(inputs), ()
    kept: list[InputT] = []
    skipped: list[dict[str, Any]] = []
    for value in inputs:
        size = (int(getattr(value, "height")), int(getattr(value, "width")))
        if not exceeds_size(size, maximum):
            kept.append(value)
            continue
        skipped.append(
            {
                "source": str(getattr(value, "image_path", "")),
                "relative_path": str(getattr(value, "relative_path", "")),
                "actual_size": list(size),
                "maximum_size": list(maximum),
                "reason": "image exceeds native_tile_size",
            }
        )
    if skipped and policy == "raise":
        raise DatasetValidationError(
            ValidationIssue(
                "Prediction input exceeds native_tile_size",
                source=source,
                value=skipped[:20],
                expected={"maximum_size": maximum, "rule": "height and width must be at most the maximum"},
                suggestion="Pass errors='skip' to omit oversized inputs.",
            )
        )
    if skipped and not kept:
        raise DatasetValidationError(
            ValidationIssue(
                "No usable images remain after skipping oversized inputs",
                source=source,
                value=skipped[:20],
                expected={"maximum_size": maximum},
            )
        )
    return tuple(kept), tuple(skipped)


def normalize_size(value: Any, *, field: str) -> Size | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer or two-item size")
    if isinstance(value, (int, float)) and int(value) == value:
        result = (int(value), int(value))
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        result = (int(value[0]), int(value[1]))
    else:
        raise ValueError(f"{field} must be a positive integer or two-item size")
    if min(result) <= 0:
        raise ValueError(f"{field} must be positive, got {result}")
    return result


def normalize_factor(value: Any, *, field: str = "upscale_factor") -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a positive integer")
    result = int(value)
    if result <= 0 or result != value:
        raise ValueError(f"{field} must be a positive integer")
    return result


@dataclass(frozen=True)
class Geometry:
    """Normalized source-tile, scale, and adapter-input geometry.

    Parameters:
        native_tile_size: Source tile height and width before upscaling.
        upscale_factor: Integer scale applied to the source tiles.
        input_size: Adapter/model input height and width.
    """

    native_tile_size: Size | None = None
    upscale_factor: int | None = None
    input_size: Size | None = None

    @classmethod
    def create(
        cls,
        *,
        native_tile_size: Any = None,
        upscale_factor: Any = None,
        input_size: Any = None,
        source: str | None = None,
    ) -> "Geometry":
        """Normalize geometry values and infer one missing field when possible.

        Parameters:
            native_tile_size: Source tile edge or two-item size before scaling.
            upscale_factor: Positive integer source-to-input scale.
            input_size: Model input edge or two-item size.
            source: Optional source name included in validation errors.

        Returns:
            Consistent normalized geometry.
        """

        native = normalize_size(native_tile_size, field="native_tile_size")
        factor = normalize_factor(upscale_factor)
        model_input = normalize_size(input_size, field="input_size")
        if factor is None and native is not None and model_input is not None:
            ratios = (model_input[0] / native[0], model_input[1] / native[1])
            if ratios[0] == ratios[1] and ratios[0].is_integer():
                factor = int(ratios[0])
        if model_input is None and native is not None and factor is not None:
            model_input = (native[0] * factor, native[1] * factor)
        if native is None and model_input is not None and factor is not None:
            if model_input[0] % factor or model_input[1] % factor:
                raise DatasetValidationError(
                    ValidationIssue(
                        "Model input size is not divisible by upscale_factor",
                        source=source,
                        value={"input_size": model_input, "upscale_factor": factor},
                    )
                )
            native = (model_input[0] // factor, model_input[1] // factor)
        if native is not None and factor is not None and model_input is not None:
            expected = (native[0] * factor, native[1] * factor)
            if model_input != expected:
                raise DatasetValidationError(
                    ValidationIssue(
                        "Contradictory model geometry",
                        source=source,
                        value={
                            "native_tile_size": native,
                            "upscale_factor": factor,
                            "input_size": model_input,
                        },
                        expected={"input_size": expected},
                    )
                )
        return cls(native, factor, model_input)

    @property
    def complete(self) -> bool:
        return all(
            value is not None
            for value in (self.native_tile_size, self.upscale_factor, self.input_size)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "native_tile_size": list(self.native_tile_size) if self.native_tile_size else None,
            "upscale_factor": self.upscale_factor,
            "input_size": list(self.input_size) if self.input_size else None,
        }


def first_value(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def from_metadata(*values: Mapping[str, Any] | None, source: str | None = None) -> Geometry:
    """Read geometry from new or legacy metadata dictionaries."""

    mappings = [dict(value or {}) for value in values]
    native = first_value(
        *(mapping.get("native_tile_size") for mapping in mappings)
    )
    factor = first_value(*(mapping.get("upscale_factor") for mapping in mappings))
    model_input = first_value(
        *(
            first_value(
                mapping.get("input_size"),
                mapping.get("model_input_size"),
                mapping.get("adapter_output_size"),
                mapping.get("imgsz"),
                mapping.get("resolution"),
            )
            for mapping in mappings
        )
    )
    return Geometry.create(
        native_tile_size=native,
        upscale_factor=factor,
        input_size=model_input,
        source=source,
    )


def infer_dataset_geometry(dataset: Any, split: str) -> tuple[Geometry, bool, set[Size]]:
    """Infer declared geometry, whether tiling is proven, and actual image sizes."""

    manifest = dict(getattr(dataset, "manifest", {}) or {})
    declared = dict(manifest.get("geometry") or {})
    dataset_metadata = dict(manifest.get("dataset") or {})
    history = list(manifest.get("history") or [])
    tiled = bool(declared.get("tiled") or dataset_metadata.get("tiled"))
    if not declared.get("native_tile_size"):
        for event in reversed(history):
            operation = str(event.get("operation") or "")
            settings = dict(event.get("settings") or {})
            if operation == "tile" or operation.startswith("tile-"):
                declared["native_tile_size"] = settings.get("tile_size")
                tiled = True
                break
    geometry = from_metadata(declared, dataset_metadata, manifest, source=dataset.name)

    samples = [sample for sample in dataset._samples if sample.split == split]
    actual: set[Size] = set()
    for sample in samples:
        width = getattr(sample, "width", None)
        height = getattr(sample, "height", None)
        if width and height:
            actual.add((int(height), int(width)))
        else:
            with Image.open(Path(sample.image_path)) as opened:
                actual.add((opened.height, opened.width))
    return geometry, tiled, actual


def validate_collection_geometry(
    dataset: Any,
    collection: Any,
    *,
    split: str,
    errors: str = "raise",
) -> Any:
    """Validate shared geometry and return a view omitting allowed oversize skips."""

    policy = normalize_errors(errors)
    dataset_geometry, tiled, _ = infer_dataset_geometry(dataset, split)

    problems: list[dict[str, Any]] = []
    maxima: list[Size] = []
    if dataset_geometry.native_tile_size is not None:
        maxima.append(dataset_geometry.native_tile_size)
    for model in collection:
        geometry = model.geometry
        if geometry.native_tile_size is not None:
            maxima.append(geometry.native_tile_size)
        if tiled and not geometry.complete:
            missing = [
                field
                for field, value in geometry.as_dict().items()
                if value is None
            ]
            problems.append(
                {
                    "model": model.name,
                    "problem": "training geometry is unproven",
                    "missing": missing,
                }
            )
            continue
    if problems:
        raise DatasetValidationError(
            ValidationIssue(
                "Dataset/model geometry mismatch; inference was not started",
                source=dataset.name,
                value=problems,
                expected={
                    "dataset_geometry": dataset_geometry.as_dict(),
                    "rule": (
                        "source images may be smaller than native_tile_size; "
                        "tiled datasets require proven model geometry"
                    ),
                },
                suggestion=(
                    "Use ModelCollection.configure() to supply missing standalone-checkpoint "
                    "geometry, or choose a dataset prepared with the model's training geometry."
                ),
            )
        )
    maximum = (
        (min(size[0] for size in maxima), min(size[1] for size in maxima))
        if maxima
        else None
    )
    selected = [sample for sample in dataset._samples if sample.split == split]
    _, skipped = filter_inputs_by_size(
        selected,
        maximum=maximum,
        errors=policy,
        source=dataset.name,
    )
    active = copy.copy(dataset)
    if skipped:
        skipped_paths = {str(Path(row["source"]).resolve()) for row in skipped}
        active._samples = [
            sample
            for sample in dataset._samples
            if sample.split != split
            or str(Path(sample.image_path).resolve()) not in skipped_paths
        ]
    active._geometry_skip_audit = tuple(
        {**row, "split": split, "errors": policy} for row in skipped
    )
    active._geometry_errors = policy
    active._geometry_maximum_size = maximum
    return active
