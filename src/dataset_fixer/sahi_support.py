from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from .errors import DatasetValidationError, ValidationIssue


LEGACY_SAHI_SETTINGS = {
    "slice_height": "sahi_slice_height",
    "slice_width": "sahi_slice_width",
    "overlap": "sahi_overlap",
    "overlap_height_ratio": "sahi_overlap_height_ratio",
    "overlap_width_ratio": "sahi_overlap_width_ratio",
    "postprocess_type": "sahi_postprocess_type",
    "postprocess_match_metric": "sahi_postprocess_match_metric",
    "postprocess_class_agnostic": "sahi_postprocess_class_agnostic",
    "model_type": "sahi_model_type",
}


@dataclass(frozen=True)
class SahiSettings:
    slice_height: int
    slice_width: int
    overlap_height_ratio: float
    overlap_width_ratio: float
    postprocess_type: str
    postprocess_match_metric: str
    postprocess_class_agnostic: bool
    model_type: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "sahi_slice_height": self.slice_height,
            "sahi_slice_width": self.slice_width,
            "sahi_overlap_height_ratio": self.overlap_height_ratio,
            "sahi_overlap_width_ratio": self.overlap_width_ratio,
            "sahi_postprocess_type": self.postprocess_type,
            "sahi_postprocess_match_metric": self.postprocess_match_metric,
            "sahi_postprocess_class_agnostic": self.postprocess_class_agnostic,
            "sahi_model_type": self.model_type,
            "sahi_stitching": "feathered-probabilities",
            "sahi_whole_image_pass": False,
        }


@dataclass(frozen=True, order=True)
class SahiTile:
    top: int
    left: int
    bottom: int
    right: int
    index: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom


def sahi_available() -> bool:
    return importlib.util.find_spec("sahi") is not None


def require_sahi() -> None:
    if not sahi_available():
        raise ImportError(
            "SAHI inference was requested but SAHI is not installed; "
            "reinstall dataset-fixer"
        )


def reject_legacy_sahi_settings(settings: Mapping[str, Any]) -> None:
    renamed = {
        key: LEGACY_SAHI_SETTINGS[key]
        for key in LEGACY_SAHI_SETTINGS
        if key in settings
    }
    if "sahi_mode" in settings:
        raise ValueError(
            "sahi_mode was removed; inference='sahi' now always performs sliced-only inference"
        )
    if renamed:
        migration = ", ".join(f"{old}->{new}" for old, new in sorted(renamed.items()))
        raise ValueError(f"Unprefixed SAHI settings were removed; rename {migration}")


def resolve_sahi_settings(
    settings: Mapping[str, Any],
    *,
    resolution: int | None,
) -> SahiSettings:
    reject_legacy_sahi_settings(settings)
    default_size = int(resolution or 480)
    raw_height = settings.get("sahi_slice_height")
    raw_width = settings.get("sahi_slice_width")
    slice_height = default_size if raw_height is None else _positive_int(raw_height, "sahi_slice_height")
    slice_width = default_size if raw_width is None else _positive_int(raw_width, "sahi_slice_width")
    overlap = _overlap(settings.get("sahi_overlap", 0.2), "sahi_overlap")
    overlap_height = _overlap(
        settings.get("sahi_overlap_height_ratio", overlap),
        "sahi_overlap_height_ratio",
    )
    overlap_width = _overlap(
        settings.get("sahi_overlap_width_ratio", overlap),
        "sahi_overlap_width_ratio",
    )
    postprocess_type = str(
        settings.get("sahi_postprocess_type", "GREEDYNMM")
    ).upper()
    if postprocess_type not in {"GREEDYNMM", "NMM", "NMS", "LSNMS"}:
        raise ValueError(
            "sahi_postprocess_type must be 'GREEDYNMM', 'NMM', 'NMS', or 'LSNMS'"
        )
    match_metric = str(
        settings.get("sahi_postprocess_match_metric", "IOS")
    ).upper()
    if match_metric not in {"IOU", "IOS"}:
        raise ValueError("sahi_postprocess_match_metric must be 'IOU' or 'IOS'")
    model_type = str(settings.get("sahi_model_type", "ultralytics")).strip()
    if not model_type:
        raise ValueError("sahi_model_type must be non-empty")
    return SahiSettings(
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height,
        overlap_width_ratio=overlap_width,
        postprocess_type=postprocess_type,
        postprocess_match_metric=match_metric,
        postprocess_class_agnostic=bool(
            settings.get("sahi_postprocess_class_agnostic", False)
        ),
        model_type=model_type,
    )


def build_tile_manifest(
    *,
    width: int,
    height: int,
    settings: SahiSettings,
) -> tuple[SahiTile, ...]:
    require_sahi()
    from sahi.slicing import get_slice_bboxes

    if width <= 0 or height <= 0:
        raise ValueError("SAHI source dimensions must be positive")
    boxes = get_slice_bboxes(
        image_height=height,
        image_width=width,
        slice_height=settings.slice_height,
        slice_width=settings.slice_width,
        overlap_height_ratio=settings.overlap_height_ratio,
        overlap_width_ratio=settings.overlap_width_ratio,
        auto_slice_resolution=False,
    )
    normalized = sorted(
        {
            (int(top), int(left), int(bottom), int(right))
            for left, top, right, bottom in boxes
        }
    )
    tiles = tuple(
        SahiTile(top, left, bottom, right, index)
        for index, (top, left, bottom, right) in enumerate(normalized)
    )
    if not tiles:
        raise DatasetValidationError("SAHI produced no tiles")
    coverage = np.zeros((height, width), dtype=np.bool_)
    for tile in tiles:
        if not (
            0 <= tile.left < tile.right <= width
            and 0 <= tile.top < tile.bottom <= height
        ):
            raise DatasetValidationError(
                ValidationIssue(
                    "SAHI produced an invalid tile",
                    value=tile.box,
                    expected=f"inside source dimensions {(width, height)}",
                )
            )
        coverage[tile.top : tile.bottom, tile.left : tile.right] = True
    if not bool(np.all(coverage)):
        missing = int(coverage.size - np.count_nonzero(coverage))
        raise DatasetValidationError(
            ValidationIssue(
                "SAHI tile manifest does not cover the complete image",
                value={"uncovered_pixels": missing},
                expected="complete source-pixel coverage",
            )
        )
    return tiles


def class_map_probabilities(
    class_map: np.ndarray,
    *,
    num_classes: int | None = None,
) -> np.ndarray:
    values = np.asarray(class_map)
    if values.ndim != 2:
        raise ValueError(f"Semantic class map must be two-dimensional, got {values.shape}")
    if not np.issubdtype(values.dtype, np.integer):
        if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
            raise ValueError("Semantic class map must contain finite integer IDs")
        values = values.astype(np.int64)
    if values.size and (int(values.min()) < 0 or bool(np.any(values == 255))):
        raise ValueError("Semantic prediction contains a negative or ignore class ID")
    observed = int(values.max()) + 1 if values.size else 1
    classes = max(observed, int(num_classes or 0), 2)
    return np.eye(classes, dtype=np.float32)[values.astype(np.int64)].transpose(2, 0, 1)


def stitch_probability_tiles(
    *,
    width: int,
    height: int,
    tiles: Iterable[tuple[SahiTile, np.ndarray]],
    scale: int = 1,
) -> np.ndarray:
    if isinstance(scale, bool) or not isinstance(scale, int) or scale <= 0:
        raise ValueError("Semantic stitching scale must be a positive integer")
    tile_rows = list(tiles)
    if not tile_rows:
        raise ValueError("Semantic stitching requires at least one tile")
    first = np.asarray(tile_rows[0][1])
    channels = int(first.reshape((-1,) + first.shape[-2:]).shape[0])
    canvas = np.zeros((channels, height * scale, width * scale), dtype=np.float32)
    weights = np.zeros((height * scale, width * scale), dtype=np.float32)
    for tile, raw in tile_rows:
        probabilities = np.asarray(raw, dtype=np.float32)
        if probabilities.ndim < 3 or math.prod(probabilities.shape[1:-2]) != 1:
            raise ValueError(f"Semantic tile probabilities have invalid shape {probabilities.shape}")
        probabilities = probabilities.reshape(probabilities.shape[0], *probabilities.shape[-2:])
        expected = (tile.height * scale, tile.width * scale)
        if probabilities.shape[0] != channels or probabilities.shape[-2:] != expected:
            raise ValueError(
                "Semantic tile probabilities do not match the tile manifest: "
                f"{probabilities.shape}, expected ({channels}, {expected[0]}, {expected[1]})"
            )
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("Semantic tile probabilities contain non-finite values")
        feather = _feather_weights(*expected)
        top, left = tile.top * scale, tile.left * scale
        bottom, right = tile.bottom * scale, tile.right * scale
        canvas[:, top:bottom, left:right] += probabilities * feather[None]
        weights[top:bottom, left:right] += feather
    if np.any(weights <= 0) or not np.all(np.isfinite(weights)):
        raise DatasetValidationError(
            "Semantic tile stitching left uncovered or invalid full-image pixels"
        )
    return canvas / weights[None]


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0 or parsed != value:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _overlap(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite fraction in [0, 1)") from exc
    if not math.isfinite(parsed) or not 0 <= parsed < 1:
        raise ValueError(f"{name} must be a finite fraction in [0, 1)")
    return parsed


def _feather_weights(height: int, width: int, floor: float = 0.05) -> np.ndarray:
    def axis(size: int) -> np.ndarray:
        if size <= 2:
            return np.ones(size, dtype=np.float32)
        return np.maximum(np.hanning(size).astype(np.float32), floor)

    return np.outer(axis(height), axis(width)).astype(np.float32)
