from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely import make_valid
from shapely.geometry import Polygon

from .models import Annotation


@dataclass(frozen=True)
class PolygonRepairConfig:
    """Raster-first recovery settings for slit-encoded segmentation polygons.

    Parameters:
        closing_kernel_px: Positive odd diameter of the elliptical classical
            closing kernel, in source-image pixels. Larger values seal wider
            slits and round contours more strongly.
    """

    closing_kernel_px: int = 5

    def __post_init__(self) -> None:
        if (
            isinstance(self.closing_kernel_px, bool)
            or not isinstance(self.closing_kernel_px, int)
            or self.closing_kernel_px < 1
            or self.closing_kernel_px % 2 == 0
        ):
            raise ValueError("closing_kernel_px must be a positive odd integer")

    @classmethod
    def _parse(
        cls,
        value: PolygonRepairConfig | Mapping[str, Any] | None,
    ) -> PolygonRepairConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise TypeError("polygon_repair must be a PolygonRepairConfig, mapping, or None")

    def _to_dict(self) -> dict[str, Any]:
        return {"closing_kernel_px": self.closing_kernel_px}


def rasterize_annotation(
    annotation: Annotation,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Rasterize one segmentation annotation while preserving interior holes."""

    if annotation.polygon:
        mask = np.zeros((height, width), dtype=np.uint8)
        rings = [annotation.polygon, *(annotation.polygon_holes or [])]
        cv2.fillPoly(mask, [_integer_contour(ring) for ring in rings], 1)
        return mask
    if annotation.rle is None:
        return np.zeros((height, width), dtype=np.uint8)
    if "multipart" in annotation.rle:
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        for flat in annotation.rle["multipart"]:
            points = [
                (float(flat[index]), float(flat[index + 1]))
                for index in range(0, len(flat), 2)
            ]
            if len(points) >= 3:
                draw.polygon(points, fill=1)
        return np.asarray(mask, dtype=np.uint8)

    from pycocotools import mask as mask_utils

    decoded = np.asarray(mask_utils.decode(annotation.rle), dtype=np.uint8)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2).astype(np.uint8)
    if decoded.shape != (height, width):
        raise ValueError(
            f"RLE mask shape {decoded.shape} does not match image shape {(height, width)}"
        )
    return (decoded > 0).astype(np.uint8)


def rasterize_annotations(
    annotations: Iterable[Annotation],
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Rasterize a foreground union through the canonical segmentation path."""

    mask = np.zeros((height, width), dtype=np.uint8)
    for annotation in annotations:
        mask |= rasterize_annotation(annotation, width=width, height=height)
    return mask


def repair_polygon_annotation(
    annotation: Annotation,
    *,
    width: int,
    height: int,
    config: PolygonRepairConfig,
) -> list[Annotation]:
    """Rasterize, close, and vectorize one polygon and its hole hierarchy.

    YOLO paths sometimes encode holes by walking from the exterior to an interior
    ring and back along the same narrow bridge. Treating that path as ordinary
    vector topology can discard the holes. OpenCV's even-odd raster fill retains
    the intended mask; a small closing seals only the rasterized bridge artifacts
    before connected exteriors and holes are extracted again.
    """

    if not annotation.polygon or len(annotation.polygon) < 3:
        return [annotation]

    mask = rasterize_annotation(annotation, width=width, height=height)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.closing_kernel_px, config.closing_kernel_px),
    )
    repaired_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(
        repaired_mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if hierarchy is None:
        return []

    hierarchy_rows = hierarchy[0]
    polygonal_parts: list[Polygon] = []
    for index, row in enumerate(hierarchy_rows):
        if int(row[3]) >= 0:
            continue
        exterior = _contour_points(contours[index])
        if len(exterior) < 3:
            continue
        holes: list[list[tuple[float, float]]] = []
        child = int(row[2])
        while child >= 0:
            ring = _contour_points(contours[child])
            if len(ring) >= 3:
                holes.append(ring)
            child = int(hierarchy_rows[child][0])
        candidate = Polygon(exterior, holes or None)
        geometry = candidate if candidate.is_valid else make_valid(candidate)
        polygonal_parts.extend(_polygonal_components(geometry))

    polygonal_parts = sorted(
        (
            part
            for part in polygonal_parts
            if not part.is_empty and part.is_valid and part.area > 0
        ),
        key=lambda part: (-part.area, *part.bounds),
    )
    repaired_annotations: list[Annotation] = []
    for index, part in enumerate(polygonal_parts, start=1):
        points = [(float(x), float(y)) for x, y in list(part.exterior.coords)[:-1]]
        holes = [
            [(float(x), float(y)) for x, y in list(ring.coords)[:-1]]
            for ring in part.interiors
        ]
        if len(points) < 3:
            continue
        min_x, min_y, max_x, max_y = map(float, part.bounds)
        source_id = annotation.source_id
        if len(polygonal_parts) > 1 and source_id is not None:
            source_id = f"{source_id}#repair-{index}"
        repaired_annotations.append(
            annotation.clone(
                bbox=(min_x, min_y, max_x, max_y),
                polygon=points,
                polygon_holes=holes or None,
                source_id=source_id,
            )
        )
    return repaired_annotations


def _integer_contour(ring: list[tuple[float, float]]) -> np.ndarray:
    return np.rint(np.asarray(ring, dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)


def _contour_points(contour: np.ndarray) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in contour.reshape(-1, 2)]


def _polygonal_components(geometry: Any) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    return [
        polygon
        for component in getattr(geometry, "geoms", ())
        for polygon in _polygonal_components(component)
    ]
