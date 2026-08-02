from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

import numpy as np
from PIL import Image, ImageOps
from shapely.errors import ShapelyError
from shapely.geometry import Point, Polygon, box as shapely_box
from shapely.ops import unary_union
from shapely.validation import explain_validity
from tqdm.auto import tqdm

from .augmentation import apply_virtual_view
from .errors import DatasetValidationError, ValidationIssue
from .models import Annotation, Sample, Task
from .operations import _builder, _print_start, _publish
from .utils import normalize_split
from .visualization import (
    save_coverage_annotated_original,
    save_label_coverage_summary,
    save_source_pixel_coverage_summary,
    save_tiling_count_summary,
    save_tiling_preview,
)

if TYPE_CHECKING:
    from .dataset import Dataset


class _SkippableTileGeometryError(DatasetValidationError):
    """A crop-specific geometry failure that ``errors='skip'`` may reject."""


def tile_dataset(
    dataset: "Dataset",
    *,
    mode: str,
    destination: str | Path | None,
    name: str | None,
    splits: Iterable[str] | None,
    tile_size: int,
    overlap: float,
    min_area_ratio: float,
    negative_tiles: str | float,
    allow_lossy: bool,
    background_filter: Callable[[Image.Image], bool] | None,
    background_filter_description: dict[str, Any] | None,
    errors: str,
    visualize: bool,
    progress: bool,
    dry_run: bool,
    settings: dict[str, Any],
    validate_output: bool = True,
) -> "Dataset":
    mode = mode.lower()
    if mode == "grid":
        return _tile_grid(
            dataset,
            destination=destination,
            name=name,
            splits=splits,
            tile_size=tile_size,
            overlap=overlap,
            min_area_ratio=min_area_ratio,
            negative_tiles=negative_tiles,
            allow_lossy=allow_lossy,
            background_filter=background_filter,
            background_filter_description=background_filter_description,
            errors=errors,
            visualize=visualize,
            progress=progress,
            dry_run=dry_run,
            validate_output=validate_output,
        )
    if mode == "coverage":
        return _tile_coverage(
            dataset,
            destination=destination,
            name=name,
            splits=splits,
            tile_size=tile_size,
            min_area_ratio=min_area_ratio,
            allow_lossy=allow_lossy,
            background_filter=background_filter,
            background_filter_description=background_filter_description,
            errors=errors,
            visualize=visualize,
            progress=progress,
            dry_run=dry_run,
            overrides=settings,
            validate_output=validate_output,
        )
    raise ValueError("mode must be 'grid' or 'coverage'")


def grid_boxes(width: int, height: int, tile_size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    if width <= tile_size and height <= tile_size:
        return [(0, 0, width, height)]
    stride = max(1, int(round(tile_size * (1 - overlap))))

    def starts(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        values = list(range(0, max(1, length - tile_size + 1), stride))
        last = length - tile_size
        if not values or values[-1] != last:
            values.append(last)
        return values

    return [(x, y, min(x + tile_size, width), min(y + tile_size, height)) for y in starts(height) for x in starts(width)]


def _tile_grid(
    dataset: "Dataset",
    *,
    destination: str | Path | None,
    name: str | None,
    splits: Iterable[str] | None,
    tile_size: int,
    overlap: float,
    min_area_ratio: float,
    negative_tiles: str | float,
    allow_lossy: bool,
    background_filter: Callable[[Image.Image], bool] | None,
    background_filter_description: dict[str, Any] | None,
    errors: str,
    visualize: bool,
    progress: bool,
    dry_run: bool,
    validate_output: bool,
) -> "Dataset":
    if not 0 <= min_area_ratio <= 1:
        raise ValueError("min_area_ratio must be in [0, 1]")
    selected = {normalize_split(s) for s in splits} if splits else set(dataset.splits)
    samples = [s for s in dataset._samples if s.split in selected]
    total_tiles = sum(len(grid_boxes(s.width, s.height, tile_size, overlap)) for s in samples)
    op_settings = {
        "mode": "grid",
        "tile_size": tile_size,
        "overlap": overlap,
        "min_area_ratio": min_area_ratio,
        "negative_tiles": negative_tiles,
        "allow_lossy": allow_lossy,
        "background_filter": background_filter_description,
        "errors": errors,
        "splits": sorted(selected),
        "visualize": visualize,
        "estimated_tiles": total_tiles,
    }
    builder = _builder(dataset, destination, name, "tile-grid", op_settings)
    _clear_inherited_tiling_reports(builder)
    try:
        _print_start(builder, samples, op_settings)
        if dry_run:
            builder.cleanup()
            return dataset

        positive_counts: Counter[str] = Counter()
        background_filter_stats: Counter[str] = Counter()
        skipped_geometry: list[dict[str, Any]] = []
        negatives: dict[
            str,
            list[tuple[Sample, tuple[int, int, int, int], Path, dict[str, Any]]],
        ] = defaultdict(list)
        iterator = tqdm(total=total_tiles, desc="Generating grid tiles", unit="tile", disable=not progress)
        for sample in samples:
            boxes = grid_boxes(sample.width, sample.height, tile_size, overlap)
            if len(boxes) == 1 and boxes[0] == (0, 0, sample.width, sample.height):
                provenance = {
                    "crop": [0, 0, sample.width, sample.height],
                    "scale": 1.0,
                    "zoom": 1.0,
                    "tile_mode": "grid",
                }
                if sample.annotations:
                    builder.add_copy(sample, split=sample.split, provenance=provenance)
                    positive_counts[sample.split] += 1
                else:
                    if background_filter is not None and negative_tiles != "none":
                        with Image.open(sample.image_path) as opened:
                            candidate_image = ImageOps.exif_transpose(opened).convert("RGB")
                        if not _background_candidate_is_accepted(
                            candidate_image,
                            background_filter,
                            sample=sample,
                            crop=boxes[0],
                            origin="grid-pass-through",
                            stats=background_filter_stats,
                        ):
                            iterator.update(1)
                            continue
                    provenance.update(
                        _background_filter_provenance(background_filter_description)
                    )
                    negatives[sample.split].append(
                        (
                            sample,
                            boxes[0],
                            sample.relative_path,
                            provenance,
                        )
                    )
                iterator.update(1)
                continue
            with Image.open(sample.image_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                for left, top, right, bottom in boxes:
                    crop = (left, top, right, bottom)
                    candidate_warnings: list[str] = []
                    try:
                        transformed = _transform_annotations(
                            sample,
                            crop,
                            dataset.task,
                            min_area_ratio,
                            allow_lossy,
                            candidate_warnings,
                        )
                    except _SkippableTileGeometryError as exc:
                        if errors == "raise":
                            raise
                        _record_skipped_geometry(
                            builder,
                            skipped_geometry,
                            exc,
                            sample=sample,
                            crop=crop,
                            mode="grid",
                        )
                        iterator.update(1)
                        continue
                    builder.warnings.extend(candidate_warnings)
                    rel = sample.relative_path.parent / f"{sample.relative_path.stem}__x{left}_y{top}_w{right-left}_h{bottom-top}.jpg"
                    provenance = {"crop": [left, top, right, bottom], "scale": 1.0, "zoom": 1.0, "tile_mode": "grid"}
                    if transformed:
                        builder.add_image(
                            sample,
                            image.crop((left, top, right, bottom)),
                            split=sample.split,
                            relative_path=rel,
                            annotations=transformed,
                            provenance=provenance,
                        )
                        positive_counts[sample.split] += 1
                    else:
                        if background_filter is not None and negative_tiles != "none":
                            candidate_image = image.crop(crop)
                            if not _background_candidate_is_accepted(
                                candidate_image,
                                background_filter,
                                sample=sample,
                                crop=crop,
                                origin="grid-window",
                                stats=background_filter_stats,
                            ):
                                iterator.update(1)
                                continue
                        provenance.update(
                            _background_filter_provenance(background_filter_description)
                        )
                        negatives[sample.split].append(
                            (sample, (left, top, right, bottom), rel, provenance)
                        )
                    iterator.update(1)
        iterator.close()

        chosen_negatives: list[
            tuple[Sample, tuple[int, int, int, int], Path, dict[str, Any]]
        ] = []
        for split in sorted(selected):
            candidates = negatives[split]
            if negative_tiles == "all":
                chosen = candidates
            elif negative_tiles == "none":
                chosen = []
            elif isinstance(negative_tiles, (int, float)):
                ratio = float(negative_tiles)
                if not 0 <= ratio < 1:
                    raise ValueError("numeric negative_tiles must be in [0, 1)")
                count = _background_images_for_ratio(
                    positive_counts[split],
                    ratio,
                )
                if count > len(candidates):
                    raise DatasetValidationError(
                        ValidationIssue(
                            "Grid tiling cannot satisfy the requested final background fraction",
                            source=split,
                            value={
                                "positive_output_images": positive_counts[split],
                                "available_background_windows": len(candidates),
                                "background_filter_rejections": background_filter_stats[
                                    "rejected"
                                ],
                                "requested_background_images": count,
                                "negative_tiles": ratio,
                            },
                            expected="enough empty grid windows to reach the requested final fraction",
                            suggestion=(
                                "lower negative_tiles, use negative_tiles='all', "
                                + (
                                    "relax background_filter, "
                                    if background_filter is not None
                                    else ""
                                )
                                + "or provide more background images"
                            ),
                        )
                    )
                chosen = (
                    candidates
                    if count == len(candidates)
                    else random.Random(f"42:{split}").sample(candidates, count)
                )
            else:
                raise ValueError(
                    "negative_tiles must be 'all', 'none', or a final background fraction in [0, 1)"
                )
            chosen_negatives.extend(chosen)
        for sample, crop, rel, provenance in chosen_negatives:
            with Image.open(sample.image_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB").crop(crop)
            builder.add_image(sample, image, split=sample.split, relative_path=rel, annotations=[], provenance=provenance)
        _write_tiling_skip_report(builder, errors, skipped_geometry)
        _write_background_filter_report(
            builder,
            background_filter_description,
            background_filter_stats,
        )
        _raise_if_background_filter_removed_every_output(
            builder,
            background_filter_description,
            background_filter_stats,
        )
        _raise_if_every_tile_was_skipped(builder, errors, skipped_geometry)
        _write_tiling_class_counts(
            builder,
            samples,
            dataset._metadata.names,
            visualize=visualize,
        )
        if visualize:
            boxes_by_source = {
                str(sample.image_path): (
                    []
                    if (
                        boxes := grid_boxes(
                            sample.width,
                            sample.height,
                            tile_size,
                            overlap,
                        )
                    )
                    == [(0, 0, sample.width, sample.height)]
                    else boxes
                )
                for sample in samples
            }
            preview_items = _select_tiling_preview_items(
                samples,
                boxes_by_source,
                {record["parent_image"] for record in builder.records},
                small_limit=tile_size,
            )
            if preview_items:
                preview = save_tiling_preview(
                    preview_items,
                    dataset.task,
                    dataset._metadata,
                    builder.reports_dir / "grid_preview.jpg",
                    mode="grid",
                )
                builder.visuals.append(str(preview.relative_to(builder.staging)))
            _save_staging_contact_sheet(builder, dataset.task, "reports/grid_tiles_audit.jpg")
        return _publish(builder, progress=progress, validate_output=validate_output)
    except Exception:
        builder.cleanup()
        raise


def _transform_annotations(
    sample: Sample,
    crop: tuple[int, int, int, int],
    task: Task,
    min_area_ratio: float,
    allow_lossy: bool,
    warnings: list[str],
) -> list[Annotation]:
    result: list[Annotation] = []
    for annotation_index, annotation in enumerate(sample.annotations):
        transformed = _transform_annotation(
            annotation,
            crop,
            task,
            min_area_ratio,
            allow_lossy,
            warnings,
            source_image=sample.image_path,
            annotation_index=annotation_index,
        )
        if transformed is not None:
            result.append(transformed)
    return result


def _transform_annotation(
    annotation: Annotation,
    crop: tuple[int, int, int, int],
    task: Task,
    min_area_ratio: float,
    allow_lossy: bool,
    warnings: list[str],
    *,
    source_image: str | Path | None = None,
    annotation_index: int | None = None,
) -> Annotation | None:
    left, top, right, bottom = crop
    crop_shape = shapely_box(left, top, right, bottom)
    if task is Task.POLO:
        assert annotation.point is not None and annotation.radius is not None
        x, y = annotation.point
        radius = annotation.radius
        if left <= x - radius and x + radius <= right and top <= y - radius and y + radius <= bottom:
            return annotation.clone(point=(x - left, y - top))
        if allow_lossy and left <= x < right and top <= y < bottom:
            clipped_radius = min(radius, x - left, right - x, y - top, bottom - y)
            if clipped_radius > 0:
                warnings.append(f"Clipped POLO radius for annotation {annotation.source_id}")
                return annotation.clone(
                    point=(x - left, y - top),
                    radius=float(clipped_radius),
                )
        return None
    if annotation.bbox is None:
        return None
    x1, y1, x2, y2 = annotation.bbox
    ix1, iy1, ix2, iy2 = max(x1, left), max(y1, top), min(x2, right), min(y2, bottom)
    original_area = max(0, x2 - x1) * max(0, y2 - y1)
    intersection_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if original_area <= 0 or intersection_area / original_area < min_area_ratio:
        return None
    new_bbox = (ix1 - left, iy1 - top, ix2 - left, iy2 - top)
    if task is Task.DETECT:
        return annotation.clone(bbox=new_bbox)
    if task is Task.POSE:
        keypoints = []
        for x, y, visibility in annotation.keypoints or []:
            if left <= x < right and top <= y < bottom and visibility != 0:
                keypoints.append((x - left, y - top, visibility))
            else:
                keypoints.append((0.0, 0.0, 0.0 if visibility is not None else None))
        if any(x != 0 or y != 0 for x, y, _ in keypoints):
            return annotation.clone(bbox=new_bbox, keypoints=keypoints)
        return None
    if task is Task.SEGMENT:
        if not annotation.polygon:
            if allow_lossy:
                warnings.append(f"Dropped non-polygon segmentation {annotation.source_id} during tiling")
                return None
            raise _segmentation_crop_error(
                "Segmentation annotation has no polygon geometry",
                annotation,
                crop,
                source_image=source_image,
                annotation_index=annotation_index,
                detail="The annotation is RLE or multipart data and cannot be written as one YOLO polygon.",
            )
        try:
            source_polygon = Polygon(annotation.polygon)
        except (ShapelyError, TypeError, ValueError) as exc:
            raise _segmentation_crop_error(
                "Could not construct the source segmentation polygon",
                annotation,
                crop,
                source_image=source_image,
                annotation_index=annotation_index,
                exception=exc,
            ) from exc
        if source_polygon.is_empty or source_polygon.area <= 0 or not source_polygon.is_valid:
            raise _segmentation_crop_error(
                "Source segmentation is invalid before crop intersection",
                annotation,
                crop,
                source_image=source_image,
                annotation_index=annotation_index,
                source_geometry=source_polygon,
                detail=explain_validity(source_polygon),
            )
        try:
            intersection = source_polygon.intersection(crop_shape)
        except ShapelyError as exc:
            raise _segmentation_crop_error(
                "Shapely failed to intersect the segmentation with the crop",
                annotation,
                crop,
                source_image=source_image,
                annotation_index=annotation_index,
                source_geometry=source_polygon,
                exception=exc,
            ) from exc
        if intersection.is_empty or intersection.area / source_polygon.area < min_area_ratio:
            return None
        if intersection.geom_type == "MultiPolygon":
            if not allow_lossy:
                raise _segmentation_crop_error(
                    "Crop split the segmentation into disconnected polygon components",
                    annotation,
                    crop,
                    source_image=source_image,
                    annotation_index=annotation_index,
                    source_geometry=source_polygon,
                    result_geometry=intersection,
                    detail=f"The result contains {len(intersection.geoms)} disconnected polygons.",
                )
            intersection = max(intersection.geoms, key=lambda geom: geom.area)
            warnings.append(f"Kept largest segment fragment for annotation {annotation.source_id}")
        if intersection.geom_type != "Polygon":
            if not allow_lossy:
                component_types = sorted(
                    {geom.geom_type for geom in getattr(intersection, "geoms", ())}
                )
                raise _segmentation_crop_error(
                    "Cropped segmentation produced unsupported mixed geometry",
                    annotation,
                    crop,
                    source_image=source_image,
                    annotation_index=annotation_index,
                    source_geometry=source_polygon,
                    result_geometry=intersection,
                    detail=(
                        f"Shapely returned {intersection.geom_type} with component types "
                        f"{component_types or ['unknown']}; YOLO can encode only one polygon exterior."
                    ),
                )
            polygon_parts = [geom for geom in getattr(intersection, "geoms", ()) if geom.geom_type == "Polygon"]
            if not polygon_parts:
                warnings.append(f"Dropped unsupported cropped segmentation {annotation.source_id}")
                return None
            intersection = max(polygon_parts, key=lambda geom: geom.area)
        if intersection.interiors:
            if not allow_lossy:
                raise _segmentation_crop_error(
                    "Cropped segmentation contains interior holes",
                    annotation,
                    crop,
                    source_image=source_image,
                    annotation_index=annotation_index,
                    source_geometry=source_polygon,
                    result_geometry=intersection,
                    detail=f"The result contains {len(intersection.interiors)} interior ring(s).",
                )
            warnings.append(f"Removed holes from cropped segmentation {annotation.source_id}")
        polygon = [(float(x - left), float(y - top)) for x, y in list(intersection.exterior.coords)[:-1]]
        if len(polygon) < 3:
            return None
        xs, ys = zip(*polygon)
        return annotation.clone(polygon=polygon, bbox=(min(xs), min(ys), max(xs), max(ys)))
    return None


def _segmentation_crop_error(
    message: str,
    annotation: Annotation,
    crop: tuple[int, int, int, int],
    *,
    source_image: str | Path | None,
    annotation_index: int | None,
    source_geometry: Any | None = None,
    result_geometry: Any | None = None,
    detail: str | None = None,
    exception: Exception | None = None,
) -> _SkippableTileGeometryError:
    """Build a diagnostic, candidate-local segmentation geometry error."""

    value: dict[str, Any] = {
        "annotation_index": annotation_index,
        "annotation_source_id": annotation.source_id,
        "class_id": annotation.class_id,
        "crop_xyxy": list(crop),
        "source_vertex_count": len(annotation.polygon or []),
    }
    if detail:
        value["geometry_detail"] = detail
    if exception is not None:
        value["exception"] = f"{type(exception).__name__}: {exception}"
    if source_geometry is not None:
        value["source_geometry"] = _geometry_summary(source_geometry)
    if result_geometry is not None:
        value["result_geometry"] = _geometry_summary(result_geometry)
    return _SkippableTileGeometryError(
        ValidationIssue(
            message,
            source=str(source_image) if source_image is not None else None,
            value=value,
            expected="one non-empty, hole-free Polygon that a YOLO segmentation row can encode",
            suggestion=(
                "use allow_lossy=True to keep the largest representable polygon where possible, "
                "or errors='skip' to reject this tile and continue"
            ),
        )
    )


def _geometry_summary(geometry: Any) -> dict[str, Any]:
    components = list(getattr(geometry, "geoms", ()))

    def finite(value: Any) -> float | None:
        number = float(value)
        return number if math.isfinite(number) else None

    summary: dict[str, Any] = {
        "type": geometry.geom_type,
        "is_valid": bool(geometry.is_valid),
        "is_empty": bool(geometry.is_empty),
        "area": finite(geometry.area),
        "bounds": [finite(value) for value in geometry.bounds],
    }
    if components:
        summary["component_count"] = len(components)
        summary["component_types"] = [component.geom_type for component in components]
        summary["component_areas"] = [float(component.area) for component in components]
    if geometry.geom_type == "Polygon":
        summary["interior_ring_count"] = len(geometry.interiors)
    if not geometry.is_valid:
        summary["validity_reason"] = explain_validity(geometry)
    return summary


def _crop_from_geometry_error(
    error: _SkippableTileGeometryError,
) -> tuple[int, int, int, int] | None:
    issue = error.issues[0]
    if not isinstance(issue.value, dict):
        return None
    crop = issue.value.get("crop_xyxy")
    if not isinstance(crop, (list, tuple)) or len(crop) != 4:
        return None
    try:
        return tuple(int(value) for value in crop)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _record_skipped_geometry(
    builder: Any,
    rows: list[dict[str, Any]],
    error: _SkippableTileGeometryError,
    *,
    sample: Sample,
    crop: tuple[int, int, int, int] | None,
    mode: str,
    attempt: int | None = None,
    focus_annotation_index: int | None = None,
) -> None:
    issue = error.issues[0]
    rows.append(
        {
            "mode": mode,
            "split": sample.split,
            "source_image": str(sample.image_path),
            "relative_image": sample.relative_path.as_posix(),
            "crop_xyxy": list(crop) if crop is not None else None,
            "attempt": attempt,
            "focus_annotation_index": focus_annotation_index,
            "reason": issue.message,
            "details": issue.value,
            "expected": issue.expected,
            "suggestion": issue.suggestion,
        }
    )


def _write_tiling_skip_report(
    builder: Any,
    errors: str,
    rows: list[dict[str, Any]],
) -> None:
    if errors != "skip":
        return
    reasons = Counter(str(row["reason"]) for row in rows)
    report = {
        "errors": "skip",
        "skipped_candidates": len(rows),
        "reason_counts": dict(sorted(reasons.items())),
        "items": rows,
    }
    builder.reports_dir.mkdir(parents=True, exist_ok=True)
    (builder.reports_dir / "tiling_skips.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    if rows:
        builder.warnings.append(
            f"Skipped {len(rows)} tile candidate(s) with non-exportable geometry; "
            "see reports/tiling_skips.json"
        )


def _raise_if_every_tile_was_skipped(
    builder: Any,
    errors: str,
    rows: list[dict[str, Any]],
) -> None:
    if errors != "skip" or not rows or builder.records:
        return
    reasons = Counter(str(row["reason"]) for row in rows)
    raise DatasetValidationError(
        ValidationIssue(
            "Tiling produced no output because every usable tile candidate was skipped",
            value={
                "skipped_candidates": len(rows),
                "reason_counts": dict(sorted(reasons.items())),
            },
            expected="at least one tile with exportable annotations",
            suggestion=(
                "use allow_lossy=True where acceptable, adjust the crop settings, or run with "
                "errors='raise' to inspect the first full geometry diagnostic"
            ),
        )
    )


def _background_candidate_is_accepted(
    image: Image.Image,
    predicate: Callable[[Image.Image], Any],
    *,
    sample: Sample,
    crop: tuple[int, int, int, int] | None,
    origin: str,
    stats: Counter[str],
) -> bool:
    """Evaluate a user predicate without allowing it to mutate output pixels."""

    stats["evaluated"] += 1
    stats[f"evaluated::{origin}"] += 1
    try:
        result = predicate(image.copy())
    except Exception as exc:
        raise DatasetValidationError(
            ValidationIssue(
                "Background filter raised an exception",
                source=str(sample.image_path),
                value={
                    "origin": origin,
                    "crop_xyxy": list(crop) if crop is not None else None,
                    "exception": f"{type(exc).__name__}: {exc}",
                },
                expected="a predicate that returns one truthy value to keep or falsey value to discard",
                suggestion="fix the background_filter callback or remove it",
            )
        ) from exc
    try:
        accepted = bool(result)
    except Exception as exc:
        raise DatasetValidationError(
            ValidationIssue(
                "Background filter returned a value that is not one boolean decision",
                source=str(sample.image_path),
                value={
                    "origin": origin,
                    "crop_xyxy": list(crop) if crop is not None else None,
                    "return_type": type(result).__name__,
                    "exception": f"{type(exc).__name__}: {exc}",
                },
                expected="one truthy value to keep or falsey value to discard",
                suggestion="reduce array-valued results with bool(...), any(), or all()",
            )
        ) from exc
    decision = "accepted" if accepted else "rejected"
    stats[decision] += 1
    stats[f"{decision}::{origin}"] += 1
    return accepted


def _background_filter_provenance(
    description: dict[str, Any] | None,
) -> dict[str, Any]:
    if description is None:
        return {}
    return {
        "background_filter": description,
        "background_filter_result": "accepted",
    }


def _write_background_filter_report(
    builder: Any,
    description: dict[str, Any] | None,
    stats: Counter[str],
) -> None:
    if description is None:
        return
    by_origin: dict[str, dict[str, int | float]] = defaultdict(dict)
    for key, count in stats.items():
        if "::" not in key:
            continue
        decision, origin = key.split("::", 1)
        by_origin[origin][decision] = int(count)
    for origin_stats in by_origin.values():
        evaluated = int(origin_stats.get("evaluated", 0))
        accepted = int(origin_stats.get("accepted", 0))
        rejected = int(origin_stats.get("rejected", 0))
        origin_stats["accepted_percentage"] = (
            100.0 * accepted / evaluated if evaluated else 0.0
        )
        origin_stats["rejected_percentage"] = (
            100.0 * rejected / evaluated if evaluated else 0.0
        )
    evaluated = int(stats["evaluated"])
    accepted = int(stats["accepted"])
    rejected = int(stats["rejected"])
    payload = {
        "filter": description,
        "semantics": "truthy keeps a background candidate; falsey discards it",
        "input": "RGB PIL.Image.Image after any applicable transform, crop, and resize",
        "evaluated_candidates": evaluated,
        "accepted_candidates": accepted,
        "rejected_candidates": rejected,
        "accepted_percentage": 100.0 * accepted / evaluated if evaluated else 0.0,
        "rejected_percentage": 100.0 * rejected / evaluated if evaluated else 0.0,
        "by_origin": dict(sorted(by_origin.items())),
    }
    (builder.reports_dir / "background_filter.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if stats["rejected"]:
        builder.warnings.append(
            f"Background filter rejected {stats['rejected']} candidate(s); "
            "see reports/background_filter.json"
        )


def _raise_if_background_filter_removed_every_output(
    builder: Any,
    description: dict[str, Any] | None,
    stats: Counter[str],
) -> None:
    if description is None or not stats["rejected"] or builder.records:
        return
    raise DatasetValidationError(
        ValidationIssue(
            "Background filter rejected every output candidate",
            value={
                "evaluated_candidates": int(stats["evaluated"]),
                "accepted_candidates": int(stats["accepted"]),
                "rejected_candidates": int(stats["rejected"]),
                "filter": description,
            },
            expected="at least one positive tile or accepted background candidate",
            suggestion="relax background_filter or include positive tiles in the selected splits",
        )
    )


def _scale_annotation(annotation: Annotation, scale: float, task: Task, radius_multiplier: float) -> Annotation:
    updates: dict[str, Any] = {}
    if annotation.bbox is not None:
        updates["bbox"] = tuple(value * scale for value in annotation.bbox)
    if annotation.polygon is not None:
        updates["polygon"] = [(x * scale, y * scale) for x, y in annotation.polygon]
    if annotation.keypoints is not None:
        updates["keypoints"] = [(x * scale, y * scale, visibility) for x, y, visibility in annotation.keypoints]
    if annotation.point is not None:
        updates["point"] = (annotation.point[0] * scale, annotation.point[1] * scale)
    if task is Task.POLO and annotation.radius is not None:
        updates["radius"] = annotation.radius * scale * radius_multiplier
    return annotation.clone(**updates)


def _tile_coverage(
    dataset: "Dataset",
    *,
    destination: str | Path | None,
    name: str | None,
    splits: Iterable[str] | None,
    tile_size: int,
    min_area_ratio: float,
    allow_lossy: bool,
    background_filter: Callable[[Image.Image], bool] | None,
    background_filter_description: dict[str, Any] | None,
    errors: str,
    visualize: bool,
    progress: bool,
    dry_run: bool,
    overrides: dict[str, Any],
    validate_output: bool,
) -> "Dataset":
    selected = {normalize_split(s) for s in splits} if splits else set(dataset.splits)
    samples = [s for s in dataset._samples if s.split in selected]
    cfg = dict(overrides)
    cfg["tile_size"] = int(tile_size)
    cfg["min_area_ratio"] = float(min_area_ratio)
    cfg["allow_lossy"] = bool(allow_lossy)
    cfg["background_filter"] = background_filter_description
    cfg["errors"] = errors
    if cfg["large_image_threshold"] is None:
        cfg["large_image_threshold"] = int(tile_size)
    _validate_coverage_settings(cfg)
    if cfg["dense_neighbor_radius_px"] is None:
        cfg["dense_neighbor_radius_px"] = tile_size * 0.5
    cfg["mode"] = "coverage"
    cfg["splits"] = sorted(selected)
    cfg["visualize"] = visualize
    dense_target = max(
        int(cfg["target_appearances_per_object"]),
        int(cfg["sparse_appearances_per_object"]),
    )
    cap = cfg["max_tiles_per_source_image"]
    cfg["estimated_tiles"] = sum(
        1
        if max(sample.width, sample.height) <= int(cfg["large_image_threshold"])
        else min(int(cap) if cap is not None else len(sample.annotations) * dense_target, len(sample.annotations) * dense_target)
        for sample in samples
    )
    rng = random.Random(int(cfg["seed"]))
    builder = _builder(dataset, destination, name, "tile-coverage", cfg)
    _clear_inherited_tiling_reports(builder)
    try:
        _print_start(builder, samples, cfg)
        if dry_run:
            builder.cleanup()
            return dataset

        coverage_rows: list[dict[str, Any]] = []
        image_rows: list[dict[str, Any]] = []
        class_totals: dict[tuple[str, int], Counter] = defaultdict(Counter)
        split_summary: dict[str, Counter] = defaultdict(Counter)
        crop_transform_stats: Counter[str] = Counter()
        background_filter_stats: Counter[str] = Counter()
        skipped_geometry: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        small_backgrounds: dict[str, list[Sample]] = defaultdict(list)
        small_preview_sources: list[Sample] = []
        iterator = tqdm(samples, desc="Coverage tiling", unit="image", disable=not progress)
        for sample in iterator:
            if max(sample.width, sample.height) <= int(cfg["large_image_threshold"]):
                annotations = [
                    _coverage_small_annotation(annotation, dataset.task, cfg)
                    for annotation in sample.annotations
                ]
                targets = {idx: 1 for idx in range(len(sample.annotations))}
                counts = {idx: 1 for idx in range(len(sample.annotations))}
                _append_coverage_rows(sample, targets, counts, False, coverage_rows, image_rows, class_totals, cfg)
                if not annotations:
                    small_backgrounds[sample.split].append(sample)
                    split_summary[sample.split]["candidate_background_source_images"] += 1
                    continue
                with Image.open(sample.image_path) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                    rel = sample.relative_path.with_suffix(".jpg")
                    builder.add_image(sample, image, split=sample.split, relative_path=rel, annotations=annotations, jpeg_quality=int(cfg["jpeg_quality"]))
                small_preview_sources.append(sample)
                split_summary[sample.split].update(
                    total_output_images=1,
                    copied_small_images=1,
                    positive_output_images=1,
                )
                iterator.set_postfix(produced=len(builder.records), refresh=False)
                continue

            targets = _coverage_targets(sample, cfg)
            coverage_types = {
                idx: _coverage_type(sample, idx, cfg)
                for idx in range(len(sample.annotations))
            }
            generated: list[dict[str, Any]] = []
            attempts = 0
            max_attempts = max(1, sum(targets.values()) * int(cfg["max_attempts_per_target"]))
            provisional = Counter()
            use_virtual_camera = _crop_pipeline_enabled(sample.split, cfg)
            source_image: np.ndarray | None = None
            if use_virtual_camera:
                with Image.open(sample.image_path) as opened:
                    source_image = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
            while attempts < max_attempts and any(provisional[i] < targets[i] for i in targets):
                needed = [i for i in targets if provisional[i] < targets[i]]
                focus_idx = rng.choice(needed)
                attempts += 1
                if use_virtual_camera:
                    assert source_image is not None
                    try:
                        candidate = _virtual_positive_candidate(
                            sample,
                            focus_idx,
                            source_image,
                            dataset.task,
                            cfg,
                            rng,
                            attempts,
                            dataset._metadata.flip_idx,
                            crop_transform_stats,
                        )
                    except _SkippableTileGeometryError as exc:
                        if errors == "raise":
                            raise
                        _record_skipped_geometry(
                            builder,
                            skipped_geometry,
                            exc,
                            sample=sample,
                            crop=_crop_from_geometry_error(exc),
                            mode="coverage-virtual",
                            attempt=attempts,
                            focus_annotation_index=focus_idx,
                        )
                        crop_transform_stats["rejected_geometry"] += 1
                        continue
                    if candidate is None:
                        continue
                    indices = candidate["indices"]
                    if any(provisional[idx] >= targets[idx] for idx in indices):
                        crop_transform_stats["rejected_already_satisfied"] += 1
                        continue
                    generated.append(candidate)
                    provisional.update(indices)
                    crop_transform_stats["accepted_positive_tiles"] += 1
                    continue
                crop = _make_crop_containing(
                    sample.annotations[focus_idx],
                    sample.width,
                    sample.height,
                    dataset.task,
                    cfg,
                    rng,
                )
                if crop is None:
                    continue
                if not allow_lossy and any(
                    _annotation_is_cut_by_crop(annotation, crop, dataset.task, cfg)
                    for annotation in sample.annotations
                ):
                    continue
                adjusted: list[Annotation] = []
                indices: list[int] = []
                scale = tile_size / (crop[2] - crop[0])
                candidate_warnings: list[str] = []
                try:
                    for idx, annotation in enumerate(sample.annotations):
                        source_annotation = _coverage_source_annotation(annotation, dataset.task, cfg)
                        transformed = _transform_annotation(
                            source_annotation,
                            crop,
                            dataset.task,
                            min_area_ratio,
                            allow_lossy,
                            candidate_warnings,
                            source_image=sample.image_path,
                            annotation_index=idx,
                        )
                        if transformed is not None:
                            adjusted.append(
                                _scale_annotation(
                                    transformed,
                                    scale,
                                    dataset.task,
                                    float(cfg["radius_multiplier"]),
                                )
                            )
                            indices.append(idx)
                except _SkippableTileGeometryError as exc:
                    if errors == "raise":
                        raise
                    _record_skipped_geometry(
                        builder,
                        skipped_geometry,
                        exc,
                        sample=sample,
                        crop=crop,
                        mode="coverage",
                        attempt=attempts,
                        focus_annotation_index=focus_idx,
                    )
                    crop_transform_stats["rejected_geometry"] += 1
                    continue
                if not indices:
                    continue
                if any(provisional[idx] >= targets[idx] for idx in indices):
                    continue
                generated.append(
                    {
                        "box": crop,
                        "annotations": adjusted,
                        "indices": indices,
                        "warnings": candidate_warnings,
                    }
                )
                provisional.update(indices)
            generated_before_cap = len(generated)
            cap = cfg["max_tiles_per_source_image"]
            if cap is not None and len(generated) > int(cap):
                generated = rng.sample(generated, int(cap))
                split_summary[sample.split]["positive_tiles_dropped_by_source_cap"] += generated_before_cap - len(generated)
            counts = Counter()
            for tile_idx, tile in enumerate(generated):
                if tile.get("image") is not None:
                    crop_image = tile["image"]
                else:
                    with Image.open(sample.image_path) as opened:
                        image = ImageOps.exif_transpose(opened).convert("RGB")
                        crop_image = image.crop(tile["box"])
                        if crop_image.size != (tile_size, tile_size):
                            crop_image = crop_image.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
                rel = sample.relative_path.parent / f"{sample.relative_path.stem}_tile_{tile_idx}.jpg"
                provenance = {
                    "crop": list(tile["box"]),
                    "tile_index": tile_idx,
                    "tile_mode": "coverage",
                    "zoom": tile_size / (tile["box"][2] - tile["box"][0]),
                    "scale": tile_size / (tile["box"][2] - tile["box"][0]),
                    "source_annotation_indices": tile["indices"],
                    **tile.get("provenance", {}),
                }
                builder.add_image(
                    sample,
                    crop_image,
                    split=sample.split,
                    relative_path=rel,
                    annotations=tile["annotations"],
                    provenance=provenance,
                    jpeg_quality=int(cfg["jpeg_quality"]),
                )
                builder.warnings.extend(tile.get("warnings", []))
                counts.update(tile["indices"])
            split_summary[sample.split].update(
                total_output_images=len(generated),
                positive_output_images=len(generated),
                tiled_output_images=len(generated),
                positive_tiled_images=len(generated),
                generated_positive_tiles_before_cap=generated_before_cap,
                produced_label_entries=sum(len(tile["annotations"]) for tile in generated),
            )
            record = {
                "sample": sample,
                "targets": targets,
                "coverage_types": coverage_types,
                "counts": dict(counts),
                # Only source-coordinate boxes can be overlaid on the original.
                "tile_boxes": [tile["box"] for tile in generated if tile.get("image") is None],
                "background_boxes": [],
                "background_box_origins": [],
                "background_tiles": [],
                "next_tile_idx": len(generated),
            }
            records.append(record)
            _append_coverage_rows(sample, targets, counts, True, coverage_rows, image_rows, class_totals, cfg)
            iterator.set_postfix(produced=len(builder.records), refresh=False)
            missed = sum(max(0, targets[i] - counts[i]) for i in targets)
            if missed:
                if errors == "skip":
                    builder.warnings.append(
                        f"{sample.split}/{sample.relative_path}: missed {missed} requested "
                        "object appearances after candidate rejection; see reports/tiling_skips.json"
                    )
                elif use_virtual_camera:
                    raise DatasetValidationError(
                        ValidationIssue(
                            "Virtual-camera coverage tiling exhausted its retry budget",
                            source=f"{sample.split}/{sample.relative_path}",
                            value={
                                "missed_object_appearances": missed,
                                "attempts": attempts,
                                "max_attempts": max_attempts,
                                "crop_transform_rejections": dict(crop_transform_stats),
                            },
                            expected=(
                                "a transformed full-source view with enough real source-pixel area "
                                "for every requested crop"
                            ),
                            suggestion=(
                                "increase max_attempts_per_target, use a smaller tile_size, narrow "
                                "scale_range or the crop transform, or reduce transform padding"
                            ),
                        )
                    )
                elif not allow_lossy:
                    raise DatasetValidationError(
                        ValidationIssue(
                            "Lossless coverage tiling could not replace boundary-cut candidates",
                            source=f"{sample.split}/{sample.relative_path}",
                            value={
                                "missed_object_appearances": missed,
                                "attempts": attempts,
                                "max_attempts": max_attempts,
                            },
                            expected="all requested object appearances to use complete, uncut annotations",
                            suggestion=(
                                "increase max_attempts_per_target, use a larger tile_size, narrow scale_range, "
                                "or explicitly set allow_lossy=True to permit clipped annotations"
                            ),
                        )
                    )
                else:
                    builder.warnings.append(
                        f"{sample.split}/{sample.relative_path}: missed {missed} requested object appearances"
                    )

        for split in selected:
            positive_count = int(split_summary[split]["positive_output_images"])
            desired = _background_images_for_ratio(
                positive_count,
                float(cfg["background_ratio"]),
            )
            candidates = small_backgrounds[split]
            split_records = [r for r in records if r["sample"].split == split]
            empty_source_records = [
                record for record in split_records if not record["sample"].annotations
            ]
            populated_records = [
                record for record in split_records if record["sample"].annotations
            ]

            # Reserve half of the final background-image target for each source
            # type. For odd totals, wholly empty sources receive the extra image.
            target_from_empty_sources = (desired + 1) // 2
            target_from_populated_space = desired // 2
            kept_sources: list[Sample] = []
            unused_source_candidates = list(candidates)

            def take_empty_source_images(count: int) -> int:
                if count <= 0 or not unused_source_candidates:
                    return 0
                if background_filter is None:
                    selected_count = min(count, len(unused_source_candidates))
                    selected = (
                        list(unused_source_candidates)
                        if selected_count == len(unused_source_candidates)
                        else rng.sample(unused_source_candidates, selected_count)
                    )
                    evaluated = selected
                else:
                    ordered = list(unused_source_candidates)
                    rng.shuffle(ordered)
                    selected = []
                    evaluated = []
                    for candidate in ordered:
                        if len(selected) >= count:
                            break
                        evaluated.append(candidate)
                        with Image.open(candidate.image_path) as opened:
                            candidate_image = ImageOps.exif_transpose(opened).convert(
                                "RGB"
                            )
                        if _background_candidate_is_accepted(
                            candidate_image,
                            background_filter,
                            sample=candidate,
                            crop=(0, 0, candidate.width, candidate.height),
                            origin="coverage-background-copy",
                            stats=background_filter_stats,
                        ):
                            selected.append(candidate)
                evaluated_paths = {str(sample.image_path) for sample in evaluated}
                unused_source_candidates[:] = [
                    sample
                    for sample in unused_source_candidates
                    if str(sample.image_path) not in evaluated_paths
                ]
                kept_sources.extend(selected)
                return len(selected)

            empty_source_count = take_empty_source_images(target_from_empty_sources)
            empty_source_count += _allocate_backgrounds(
                empty_source_records,
                target_from_empty_sources - empty_source_count,
                dataset.task,
                cfg,
                rng,
                origin="empty_source_image",
                flip_idx=dataset._metadata.flip_idx,
                stats=crop_transform_stats,
                background_filter=background_filter,
                filter_stats=background_filter_stats,
            )
            populated_space_count = _allocate_backgrounds(
                populated_records,
                target_from_populated_space,
                dataset.task,
                cfg,
                rng,
                origin="populated_image_empty_space",
                flip_idx=dataset._metadata.flip_idx,
                stats=crop_transform_stats,
                background_filter=background_filter,
                filter_stats=background_filter_stats,
            )

            fallback_reasons: list[str] = []
            if empty_source_count < target_from_empty_sources:
                fallback_reasons.append(
                    "wholly empty source images could not supply their equal share"
                )
            if populated_space_count < target_from_populated_space:
                fallback_reasons.append(
                    "populated images did not provide enough object-free crop locations"
                )

            # Preserve the overall fraction even if one source type cannot meet
            # its half by cross-filling from any remaining candidate pool.
            remaining = desired - empty_source_count - populated_space_count
            if remaining > 0:
                added = take_empty_source_images(remaining)
                empty_source_count += added
                remaining -= added
            if remaining > 0:
                added = _allocate_backgrounds(
                    empty_source_records,
                    remaining,
                    dataset.task,
                    cfg,
                    rng,
                    origin="empty_source_image",
                    flip_idx=dataset._metadata.flip_idx,
                    stats=crop_transform_stats,
                    background_filter=background_filter,
                    filter_stats=background_filter_stats,
                )
                empty_source_count += added
                remaining -= added
            if remaining > 0:
                added = _allocate_backgrounds(
                    populated_records,
                    remaining,
                    dataset.task,
                    cfg,
                    rng,
                    origin="populated_image_empty_space",
                    flip_idx=dataset._metadata.flip_idx,
                    stats=crop_transform_stats,
                    background_filter=background_filter,
                    filter_stats=background_filter_stats,
                )
                populated_space_count += added
                remaining -= added

            for sample in kept_sources:
                with Image.open(sample.image_path) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                    rel = sample.relative_path.with_suffix(".jpg")
                    builder.add_image(
                        sample,
                        image,
                        split=split,
                        relative_path=rel,
                        annotations=[],
                        provenance={
                            "tile_mode": "coverage-background-copy",
                            "background_source": "empty_source_image",
                            "zoom": 1.0,
                            "scale": 1.0,
                            **_background_filter_provenance(
                                background_filter_description
                            ),
                        },
                        jpeg_quality=int(cfg["jpeg_quality"]),
                    )
                split_summary[split].update(
                    total_output_images=1,
                    empty_output_images=1,
                    copied_small_images=1,
                    copied_background_images=1,
                )
                small_preview_sources.append(sample)

            actual = empty_source_count + populated_space_count
            balanced = (
                empty_source_count == target_from_empty_sources
                and populated_space_count == target_from_populated_space
            )
            split_summary[split].update(
                target_background_images=desired,
                actual_background_images=actual,
                target_background_from_empty_source_images=target_from_empty_sources,
                target_background_from_populated_image_space=target_from_populated_space,
                background_from_empty_source_images=empty_source_count,
                background_from_populated_image_space=populated_space_count,
                background_source_balance_fallback_images=(
                    abs(empty_source_count - target_from_empty_sources)
                    if actual == desired
                    else 0
                ),
                candidate_empty_source_images=len(candidates) + len(empty_source_records),
                candidate_populated_source_images=len(populated_records),
                dropped_background_source_images=len(candidates) - len(kept_sources),
                missed_background_images=max(0, desired - actual),
            )
            if actual == desired and not balanced:
                builder.warnings.append(
                    f"{split}: reached the requested {float(cfg['background_ratio']):.1%} "
                    "background fraction, but could not use an equal mix of wholly empty "
                    "source images and empty regions from populated images: "
                    + "; ".join(fallback_reasons)
                )
            if actual < desired:
                raise DatasetValidationError(
                    ValidationIssue(
                        "Coverage tiling cannot satisfy the requested final background fraction",
                        source=split,
                        value={
                            "positive_output_images": positive_count,
                            "produced_background_images": actual,
                            "requested_background_images": desired,
                            "background_ratio": float(cfg["background_ratio"]),
                            "requested_from_empty_source_images": target_from_empty_sources,
                            "produced_from_empty_source_images": empty_source_count,
                            "available_small_empty_source_images": len(candidates),
                            "large_empty_source_images": len(empty_source_records),
                            "requested_from_populated_image_space": target_from_populated_space,
                            "produced_from_populated_image_space": populated_space_count,
                            "populated_source_images": len(populated_records),
                            "background_filter": background_filter_description,
                            "background_filter_rejections": int(
                                background_filter_stats["rejected"]
                            ),
                            "reason": "; ".join(fallback_reasons)
                            or "all background candidate pools were exhausted",
                        },
                        expected="enough object-free source images or crop locations to reach the requested final fraction",
                        suggestion=(
                            "lower background_ratio, raise max_background_attempts_per_tile or "
                            "max_tiles_per_source_image, "
                            + (
                                "relax background_filter, "
                                if background_filter is not None
                                else ""
                            )
                            + "or provide more background imagery"
                        ),
                    )
                )
            for record in split_records:
                sample = record["sample"]
                with Image.open(sample.image_path) as opened:
                    source = ImageOps.exif_transpose(opened).convert("RGB")
                    for crop, origin in zip(
                        record["background_boxes"],
                        record["background_box_origins"],
                    ):
                        crop_image = source.crop(crop)
                        if crop_image.size != (tile_size, tile_size):
                            crop_image = crop_image.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
                        index = record["next_tile_idx"]
                        rel = sample.relative_path.parent / f"{sample.relative_path.stem}_tile_{index}.jpg"
                        builder.add_image(
                            sample,
                            crop_image,
                            split=split,
                            relative_path=rel,
                            annotations=[],
                            provenance={
                                "crop": list(crop),
                                "tile_index": index,
                                "tile_mode": f"coverage-background-{origin}",
                                "background_source": origin,
                                "zoom": tile_size / (crop[2] - crop[0]),
                                "scale": tile_size / (crop[2] - crop[0]),
                                **_background_filter_provenance(
                                    background_filter_description
                                ),
                            },
                            jpeg_quality=int(cfg["jpeg_quality"]),
                        )
                        record["next_tile_idx"] += 1
                        split_summary[split].update(total_output_images=1, empty_output_images=1, tiled_output_images=1, empty_tiled_images=1)
                for tile in record["background_tiles"]:
                    index = record["next_tile_idx"]
                    rel = sample.relative_path.parent / f"{sample.relative_path.stem}_tile_{index}.jpg"
                    builder.add_image(
                        sample,
                        tile["image"],
                        split=split,
                        relative_path=rel,
                        annotations=[],
                        provenance={
                            "crop": list(tile["box"]),
                            "tile_index": index,
                            "tile_mode": f"coverage-background-augmented-{tile['origin']}",
                            "background_source": tile["origin"],
                            "zoom": tile_size / (tile["box"][2] - tile["box"][0]),
                            "scale": tile_size / (tile["box"][2] - tile["box"][0]),
                            **tile["provenance"],
                        },
                        jpeg_quality=int(cfg["jpeg_quality"]),
                    )
                    record["next_tile_idx"] += 1
                    split_summary[split].update(
                        total_output_images=1,
                        empty_output_images=1,
                        tiled_output_images=1,
                        empty_tiled_images=1,
                    )
                if visualize and (sample.annotations or record["background_boxes"]):
                    output = builder.staging / "coverage_summary" / "annotated_originals" / split / f"{sample.image_path.stem}_coverage.jpg"
                    save_coverage_annotated_original(
                        sample,
                        record["counts"],
                        record["targets"],
                        record["coverage_types"],
                        record["background_boxes"],
                        output,
                        cfg,
                    )
                    builder.visuals.append(str(output.relative_to(builder.staging)))

        _write_tiling_skip_report(builder, errors, skipped_geometry)
        _write_background_filter_report(
            builder,
            background_filter_description,
            background_filter_stats,
        )
        _raise_if_background_filter_removed_every_output(
            builder,
            background_filter_description,
            background_filter_stats,
        )
        _raise_if_every_tile_was_skipped(builder, errors, skipped_geometry)
        source_pixel_rows = _source_pixel_coverage_rows(samples, builder.records)
        coverage_visuals = _write_coverage_reports(
            builder.staging / "coverage_summary",
            coverage_rows,
            image_rows,
            source_pixel_rows,
            class_totals,
            split_summary,
            selected,
            background_ratio=float(cfg["background_ratio"]),
            visualize=visualize,
        )
        builder.visuals.extend(
            str(output.relative_to(builder.staging)) for output in coverage_visuals
        )
        if cfg.get("crop_pipeline") is not None:
            accepted_virtual_records = [
                record
                for record in builder.records
                if record.get("crop_transform_seed") is not None
            ]
            accepted_seeds = [
                int(record["crop_transform_seed"])
                for record in accepted_virtual_records
            ]
            accepted_by_split = Counter(
                str(record["output_split"])
                for record in accepted_virtual_records
            )
            transformed_splits = sorted(
                split for split in selected if _crop_pipeline_enabled(split, cfg)
            )
            unchanged_splits: dict[str, str] = {}
            if "val" in selected and "val" not in transformed_splits:
                unchanged_splits["val"] = (
                    "augment_val=False; set augment_val=True to sample a fresh "
                    "virtual camera for every produced validation crop"
                )
            if "test" in selected:
                unchanged_splits["test"] = (
                    "test crops are never transformed by crop_transforms"
                )
            crop_report = {
                "pipeline": cfg["crop_pipeline"],
                "augment_val": bool(cfg.get("augment_val")),
                "transformed_splits": transformed_splits,
                "unchanged_splits": unchanged_splits,
                "seed": int(cfg["seed"]),
                "stats": dict(sorted(crop_transform_stats.items())),
                "accepted_virtual_camera_crops": len(accepted_virtual_records),
                "distinct_accepted_virtual_camera_seeds": len(set(accepted_seeds)),
                "fresh_seed_per_accepted_crop": (
                    len(accepted_seeds) == len(set(accepted_seeds))
                ),
                "accepted_virtual_camera_crops_by_split": dict(
                    sorted(accepted_by_split.items())
                ),
                "valid_pixel_requirement": 1.0,
                "coordinate_order": "full-source transform, then coverage crop",
                "sampling_unit": (
                    "one independently seeded full-source virtual camera per crop candidate"
                ),
            }
            (builder.reports_dir / "crop_augmentation.json").write_text(
                json.dumps(crop_report, indent=2, sort_keys=True), encoding="utf-8"
            )
            if "val" in unchanged_splits:
                builder.warnings.append(
                    "crop_transforms left validation crops in their source orientation because "
                    "augment_val=False; set augment_val=True to sample a fresh virtual camera "
                    "for every produced validation crop"
                )
        _write_tiling_class_counts(
            builder,
            samples,
            dataset._metadata.names,
            visualize=visualize,
        )
        if visualize:
            boxes_by_source = {
                str(record["sample"].image_path): [
                    *record["tile_boxes"],
                    *record["background_boxes"],
                ]
                for record in records
            }
            boxes_by_source.update(
                {str(sample.image_path): [] for sample in small_preview_sources}
            )
            preview_items = _select_tiling_preview_items(
                samples,
                boxes_by_source,
                {record["parent_image"] for record in builder.records},
                small_limit=int(cfg["large_image_threshold"]),
            )
            if preview_items:
                preview = save_tiling_preview(
                    preview_items,
                    dataset.task,
                    dataset._metadata,
                    builder.reports_dir / "coverage_preview.jpg",
                    mode="coverage",
                )
                builder.visuals.append(str(preview.relative_to(builder.staging)))
            _save_staging_contact_sheet(builder, dataset.task, "coverage_summary/random_tile_contact_sheet.jpg")
        return _publish(builder, progress=progress, validate_output=validate_output)
    except Exception:
        builder.cleanup()
        raise


def _coverage_targets(sample: Sample, cfg: dict[str, Any]) -> dict[int, int]:
    targets: dict[int, int] = {}
    for index, annotation in enumerate(sample.annotations):
        override = _coverage_override(annotation, cfg)
        if override is not None:
            targets[index] = override
            continue
        targets[index] = int(
            cfg[
                "target_appearances_per_object"
                if _coverage_type(sample, index, cfg) == "dense"
                else "sparse_appearances_per_object"
            ]
        )
    return targets


def _crop_pipeline_enabled(split: str, cfg: dict[str, Any]) -> bool:
    if cfg.get("crop_pipeline") is None:
        return False
    return split == "train" or (split == "val" and bool(cfg.get("augment_val")))


def _virtual_positive_candidate(
    sample: Sample,
    focus_idx: int,
    source_image: np.ndarray,
    task: Task,
    cfg: dict[str, Any],
    rng: random.Random,
    attempt: int,
    flip_idx: list[int] | None,
    stats: Counter[str],
) -> dict[str, Any] | None:
    seed = _virtual_crop_seed(int(cfg["seed"]), sample, "positive", attempt, focus_idx)
    stats["virtual_view_attempts"] += 1
    (
        view_image,
        view_annotations,
        source_indices,
        validity,
        applied,
        transform_warnings,
        partial_indices,
    ) = apply_virtual_view(
        sample,
        task,
        cfg["crop_pipeline"],
        seed,
        source_image=source_image,
        flip_idx=flip_idx,
    )
    focus_annotation = next(
        (
            annotation
            for annotation, source_index in zip(view_annotations, source_indices)
            if source_index == focus_idx
        ),
        None,
    )
    if focus_annotation is None:
        stats["rejected_missing_focus"] += 1
        return None
    view_cfg = dict(cfg)
    view_cfg["require_requested_crop_size"] = True
    if task is Task.POLO:
        view_cfg["polo_radius_px"] = None
    crop = _make_crop_containing(
        focus_annotation,
        int(view_image.shape[1]),
        int(view_image.shape[0]),
        task,
        view_cfg,
        rng,
    )
    if crop is None:
        stats["rejected_no_containing_crop"] += 1
        return None
    if not _crop_is_fully_valid(validity, crop):
        stats["rejected_invalid_source_pixels"] += 1
        return None
    if not bool(cfg["allow_lossy"]):
        for annotation, source_index in zip(view_annotations, source_indices):
            if not _annotation_intersects_crop(annotation, crop, task, view_cfg):
                continue
            if source_index in partial_indices or _annotation_is_cut_by_crop(
                annotation, crop, task, view_cfg
            ):
                stats["rejected_strict_annotation_cut"] += 1
                return None

    adjusted: list[Annotation] = []
    indices: list[int] = []
    scale = int(cfg["tile_size"]) / (crop[2] - crop[0])
    accepted_warnings = list(transform_warnings)
    for annotation, source_index in zip(view_annotations, source_indices):
        crop_warnings: list[str] = []
        transformed = _transform_annotation(
            annotation,
            crop,
            task,
            float(cfg["min_area_ratio"]),
            bool(cfg["allow_lossy"]),
            crop_warnings,
            source_image=sample.image_path,
            annotation_index=source_index,
        )
        if transformed is None:
            continue
        adjusted.append(
            _scale_annotation(
                transformed,
                scale,
                task,
                float(cfg["radius_multiplier"]),
            )
        )
        indices.append(source_index)
        accepted_warnings.extend(crop_warnings)
        if source_index in partial_indices:
            accepted_warnings.append(
                f"Accepted a clipped virtual-view annotation {sample.image_path}:{source_index}"
            )
    if not indices:
        stats["rejected_empty_after_crop"] += 1
        return None
    if accepted_warnings:
        stats["accepted_lossy_annotations"] += len(accepted_warnings)
    cropped = Image.fromarray(view_image).crop(crop)
    tile_size = int(cfg["tile_size"])
    if cropped.size != (tile_size, tile_size):
        cropped = cropped.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
    return {
        "box": crop,
        "annotations": adjusted,
        "indices": indices,
        "image": cropped,
        "provenance": {
            "tile_mode": "coverage-augmented",
            "crop_coordinate_space": "transformed_view",
            "source_context": [0, 0, sample.width, sample.height],
            "transformed_view_size": [int(view_image.shape[1]), int(view_image.shape[0])],
            "crop_transform_seed": seed,
            "crop_transform_attempt": attempt,
            "crop_pipeline": cfg["crop_pipeline"],
            "crop_albumentations_applied": applied,
            "valid_pixel_fraction": 1.0,
            "validity_result": "all_output_pixels_map_to_source",
            "crop_transform_warnings": accepted_warnings,
            "lossy_clipping": bool(accepted_warnings),
        },
        "warnings": accepted_warnings,
    }


def _crop_is_fully_valid(
    validity: np.ndarray,
    crop: tuple[int, int, int, int],
) -> bool:
    left, top, right, bottom = crop
    if left < 0 or top < 0 or right > validity.shape[1] or bottom > validity.shape[0]:
        return False
    selected = validity[top:bottom, left:right]
    return selected.shape == (bottom - top, right - left) and bool(selected.all())


def _virtual_crop_seed(
    base_seed: int,
    sample: Sample,
    role: str,
    attempt: int,
    focus_idx: int | None,
) -> int:
    value = (
        f"{base_seed}:{sample.split}:{sample.relative_path.as_posix()}:"
        f"{role}:{attempt}:{focus_idx}"
    ).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big")


def _coverage_override(annotation: Annotation, cfg: dict[str, Any]) -> int | None:
    overrides = cfg["object_appearance_overrides"]
    override = overrides.get(annotation.source_id)
    if override is None and annotation.source_id is not None:
        override = overrides.get(str(annotation.source_id))
    return int(override) if override is not None else None


def _coverage_type(sample: Sample, index: int, cfg: dict[str, Any]) -> str:
    annotation = sample.annotations[index]
    if _coverage_override(annotation, cfg) is not None:
        return "override"
    anchor = _annotation_anchor(annotation)
    radius = float(cfg["dense_neighbor_radius_px"])
    nearby = sum(
        other_index != index
        and math.hypot(
            anchor[0] - _annotation_anchor(other)[0],
            anchor[1] - _annotation_anchor(other)[1],
        )
        <= radius
        for other_index, other in enumerate(sample.annotations)
    )
    return (
        "dense"
        if nearby >= int(cfg["min_nearby_objects_for_full_coverage"])
        else "sparse"
    )


def _annotation_anchor(annotation: Annotation) -> tuple[float, float]:
    if annotation.point is not None:
        return annotation.point
    if annotation.bbox is not None:
        x1, y1, x2, y2 = annotation.bbox
        return (x1 + x2) / 2, (y1 + y2) / 2
    if annotation.polygon:
        xs, ys = zip(*annotation.polygon)
        return sum(xs) / len(xs), sum(ys) / len(ys)
    raise DatasetValidationError(f"Annotation {annotation.source_id} has no crop anchor")


def _polo_radius(annotation: Annotation, cfg: dict[str, Any]) -> float:
    configured = cfg["polo_radius_px"]
    if configured is not None:
        return float(configured)
    if annotation.radius is None:
        raise DatasetValidationError(f"POLO annotation {annotation.source_id} has no radius")
    return float(annotation.radius)


def _coverage_source_annotation(annotation: Annotation, task: Task, cfg: dict[str, Any]) -> Annotation:
    return annotation.clone(radius=_polo_radius(annotation, cfg)) if task is Task.POLO else annotation.clone()


def _coverage_small_annotation(annotation: Annotation, task: Task, cfg: dict[str, Any]) -> Annotation:
    if task is not Task.POLO:
        return annotation.clone()
    return annotation.clone(
        radius=_polo_radius(annotation, cfg) * float(cfg["radius_multiplier"])
    )


def _annotation_bounds(annotation: Annotation, task: Task, cfg: dict[str, Any]) -> tuple[float, float, float, float]:
    if task is Task.POLO:
        assert annotation.point is not None
        radius = _polo_radius(annotation, cfg)
        x, y = annotation.point
        return x - radius, y - radius, x + radius, y + radius
    if annotation.bbox is not None:
        return annotation.bbox
    if annotation.polygon:
        xs, ys = zip(*annotation.polygon)
        return min(xs), min(ys), max(xs), max(ys)
    raise DatasetValidationError(f"Annotation {annotation.source_id} has no crop bounds")


def _make_crop_containing(
    annotation: Annotation,
    width: int,
    height: int,
    task: Task,
    cfg: dict[str, Any],
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    scale = rng.uniform(*cfg["scale_range"])
    requested_dim = max(1, int(cfg["tile_size"] / scale))
    if cfg.get("require_requested_crop_size") and (
        requested_dim > width or requested_dim > height
    ):
        return None
    crop_dim = min(requested_dim, width, height)
    x1, y1, x2, y2 = _annotation_bounds(annotation, task, cfg)
    if x2 - x1 > crop_dim or y2 - y1 > crop_dim:
        if not bool(cfg.get("allow_lossy")):
            return None
        # Lossy coverage still anchors the candidate on the requested object,
        # but permits the final crop to clip geometry larger than the sampled
        # camera window.
        anchor_x, anchor_y = _annotation_anchor(annotation)
        max_left, max_top = width - crop_dim, height - crop_dim
        left_min = max(0, int(math.ceil(anchor_x - crop_dim)))
        left_max = min(max_left, int(math.floor(anchor_x)))
        top_min = max(0, int(math.ceil(anchor_y - crop_dim)))
        top_max = min(max_top, int(math.floor(anchor_y)))
        if left_min > left_max or top_min > top_max:
            return None
        left = rng.randint(left_min, left_max)
        top = rng.randint(top_min, top_max)
        return left, top, left + crop_dim, top + crop_dim
    max_left, max_top = width - crop_dim, height - crop_dim
    left_min = max(0, int(math.ceil(x2 - crop_dim)))
    left_max = min(max_left, int(math.floor(x1)))
    top_min = max(0, int(math.ceil(y2 - crop_dim)))
    top_max = min(max_top, int(math.floor(y1)))
    if left_min > left_max or top_min > top_max:
        return None
    left, top = rng.randint(left_min, left_max), rng.randint(top_min, top_max)
    return left, top, left + crop_dim, top + crop_dim


def _random_crop(
    width: int,
    height: int,
    cfg: dict[str, Any],
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    scale = rng.uniform(*cfg["scale_range"])
    requested_dim = max(1, int(cfg["tile_size"] / scale))
    if cfg.get("require_requested_crop_size") and (
        requested_dim > width or requested_dim > height
    ):
        return None
    dim = min(requested_dim, width, height)
    left = rng.randint(0, width - dim) if width > dim else 0
    top = rng.randint(0, height - dim) if height > dim else 0
    return left, top, left + dim, top + dim


def _annotation_intersects_crop(
    annotation: Annotation,
    crop: tuple[int, int, int, int],
    task: Task,
    cfg: dict[str, Any],
) -> bool:
    left, top, right, bottom = crop
    if task is Task.POLO:
        assert annotation.point is not None
        x, y = annotation.point
        radius = _polo_radius(annotation, cfg)
        closest_x, closest_y = min(max(x, left), right), min(max(y, top), bottom)
        return (x - closest_x) ** 2 + (y - closest_y) ** 2 <= radius**2
    x1, y1, x2, y2 = _annotation_bounds(annotation, task, cfg)
    return not (x2 <= left or right <= x1 or y2 <= top or bottom <= y1)


def _annotation_is_cut_by_crop(
    annotation: Annotation,
    crop: tuple[int, int, int, int],
    task: Task,
    cfg: dict[str, Any],
) -> bool:
    """Whether a crop contains only part of a source annotation geometry."""

    left, top, right, bottom = crop
    crop_shape = shapely_box(left, top, right, bottom)
    if task is Task.POLO:
        if annotation.point is None:
            return False
        configured = cfg.get("polo_radius_px")
        radius = float(configured if configured is not None else annotation.radius or 0.0)
        x, y = annotation.point
        circle = Point(x, y).buffer(radius)
        return circle.intersects(crop_shape) and not crop_shape.covers(circle)
    if task is Task.SEGMENT and annotation.polygon:
        geometry = Polygon(annotation.polygon)
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        intersection = geometry.intersection(crop_shape)
        return not intersection.is_empty and intersection.area > 0 and not crop_shape.covers(geometry)
    if annotation.bbox is None:
        return False
    x1, y1, x2, y2 = annotation.bbox
    intersection_width = max(0.0, min(x2, right) - max(x1, left))
    intersection_height = max(0.0, min(y2, bottom) - max(y1, top))
    intersects = intersection_width > 0 and intersection_height > 0
    contained = left <= x1 and y1 >= top and x2 <= right and y2 <= bottom
    return intersects and not contained


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _allocate_backgrounds(
    records: list[dict[str, Any]],
    desired: int,
    task: Task,
    cfg: dict[str, Any],
    rng: random.Random,
    *,
    origin: str,
    flip_idx: list[int] | None = None,
    stats: Counter[str] | None = None,
    background_filter: Callable[[Image.Image], bool] | None = None,
    filter_stats: Counter[str] | None = None,
) -> int:
    allocated = 0
    stats = stats if stats is not None else Counter()
    filter_stats = filter_stats if filter_stats is not None else Counter()
    cap = cfg["max_tiles_per_source_image"]
    source_cache: dict[str, np.ndarray] = {}
    for attempt in range(1, desired * int(cfg["max_background_attempts_per_tile"]) + 1):
        if allocated >= desired:
            break
        eligible = [
            record
            for record in records
            if cap is None
            or record["next_tile_idx"]
            + len(record["background_boxes"])
            + len(record.get("background_tiles", []))
            < int(cap)
        ]
        if not eligible:
            break
        record = rng.choice(eligible)
        sample = record["sample"]
        if _crop_pipeline_enabled(sample.split, cfg):
            cache_key = str(sample.image_path)
            source_image = source_cache.get(cache_key)
            if source_image is None:
                with Image.open(sample.image_path) as opened:
                    source_image = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
                source_cache.clear()
                source_cache[cache_key] = source_image
            seed = _virtual_crop_seed(
                int(cfg["seed"]),
                sample,
                f"background-{origin}",
                attempt,
                None,
            )
            stats["virtual_background_attempts"] += 1
            (
                view_image,
                annotations,
                _,
                validity,
                applied,
                transform_warnings,
                _,
            ) = apply_virtual_view(
                sample,
                task,
                cfg["crop_pipeline"],
                seed,
                source_image=source_image,
                flip_idx=flip_idx,
            )
            view_cfg = dict(cfg)
            view_cfg["require_requested_crop_size"] = True
            if task is Task.POLO:
                view_cfg["polo_radius_px"] = None
            crop = _random_crop(
                int(view_image.shape[1]),
                int(view_image.shape[0]),
                view_cfg,
                rng,
            )
            if crop is None:
                stats["rejected_background_insufficient_view_size"] += 1
                continue
            if not _crop_is_fully_valid(validity, crop):
                stats["rejected_background_invalid_source_pixels"] += 1
                continue
            if any(
                _annotation_intersects_crop(annotation, crop, task, view_cfg)
                for annotation in annotations
            ):
                stats["rejected_background_with_annotation"] += 1
                continue
            crop_image = Image.fromarray(view_image).crop(crop)
            tile_size = int(cfg["tile_size"])
            if crop_image.size != (tile_size, tile_size):
                crop_image = crop_image.resize(
                    (tile_size, tile_size), Image.Resampling.BILINEAR
                )
            if background_filter is not None and not _background_candidate_is_accepted(
                crop_image,
                background_filter,
                sample=sample,
                crop=crop,
                origin=f"coverage-virtual-{origin}",
                stats=filter_stats,
            ):
                continue
            record.setdefault("background_tiles", []).append(
                {
                    "box": crop,
                    "image": crop_image,
                    "origin": origin,
                    "provenance": {
                        "crop_coordinate_space": "transformed_view",
                        "source_context": [0, 0, sample.width, sample.height],
                        "transformed_view_size": [
                            int(view_image.shape[1]),
                            int(view_image.shape[0]),
                        ],
                        "crop_transform_seed": seed,
                        "crop_transform_attempt": attempt,
                        "crop_pipeline": cfg["crop_pipeline"],
                        "crop_albumentations_applied": applied,
                        "valid_pixel_fraction": 1.0,
                        "validity_result": "all_output_pixels_map_to_source",
                        "crop_transform_warnings": transform_warnings,
                        "lossy_clipping": False,
                        **_background_filter_provenance(
                            cfg.get("background_filter")
                        ),
                    },
                }
            )
            stats["accepted_background_tiles"] += 1
            allocated += 1
            continue
        crop = _random_crop(sample.width, sample.height, cfg, rng)
        if crop is None:
            continue
        if any(_annotation_intersects_crop(annotation, crop, task, cfg) for annotation in sample.annotations):
            continue
        if any(_boxes_intersect(crop, existing) for existing in record["background_boxes"]):
            continue
        if background_filter is not None:
            cache_key = str(sample.image_path)
            source_image = source_cache.get(cache_key)
            if source_image is None:
                with Image.open(sample.image_path) as opened:
                    source_image = np.asarray(
                        ImageOps.exif_transpose(opened).convert("RGB")
                    )
                source_cache.clear()
                source_cache[cache_key] = source_image
            crop_image = Image.fromarray(source_image).crop(crop)
            tile_size = int(cfg["tile_size"])
            if crop_image.size != (tile_size, tile_size):
                crop_image = crop_image.resize(
                    (tile_size, tile_size), Image.Resampling.BILINEAR
                )
            if not _background_candidate_is_accepted(
                crop_image,
                background_filter,
                sample=sample,
                crop=crop,
                origin=f"coverage-{origin}",
                stats=filter_stats,
            ):
                continue
        record["background_boxes"].append(crop)
        record["background_box_origins"].append(origin)
        allocated += 1
    return allocated


def _background_images_for_ratio(positive_images: int, background_ratio: float) -> int:
    """Return the nearest background count for a final output fraction."""

    if positive_images <= 0 or background_ratio <= 0:
        return 0
    exact = positive_images * background_ratio / (1.0 - background_ratio)
    candidates = {int(math.floor(exact)), int(math.ceil(exact))}
    return min(
        candidates,
        key=lambda count: (
            abs(count / (positive_images + count) - background_ratio),
            count,
        ),
    )


def _append_coverage_rows(
    sample: Sample,
    targets: dict[int, int],
    counts: dict[int, int] | Counter,
    is_large: bool,
    coverage_rows: list[dict[str, Any]],
    image_rows: list[dict[str, Any]],
    class_totals: dict[tuple[str, int], Counter],
    cfg: dict[str, Any],
) -> None:
    for idx, annotation in enumerate(sample.annotations):
        count, target = int(counts.get(idx, 0)), targets[idx]
        anchor = _annotation_anchor(annotation)
        coverage_type = _coverage_type(sample, idx, cfg)
        coverage_rows.append(
            {
                "split": sample.split,
                "image": sample.image_path.name,
                "label_idx": idx,
                "source_id": annotation.source_id,
                "class_id": annotation.class_id,
                "x_norm": anchor[0] / sample.width,
                "y_norm": anchor[1] / sample.height,
                "actual_coverages": count,
                "requested_coverages": target,
                "covered_at_least_once": count >= 1,
                "percent_of_requested": 100 * count / target if target else 0,
                "source_image_width": sample.width,
                "source_image_height": sample.height,
                "is_large_image": is_large,
                "coverage_type": coverage_type,
            }
        )
        totals = class_totals[(sample.split, annotation.class_id)]
        totals.update(original_labels=1, requested_coverages=target, actual_coverages=count, labels_covered_at_least_once=int(count >= 1))
    total = len(sample.annotations)
    hit = sum(counts.get(i, 0) >= 1 for i in range(total))
    requested, actual = sum(targets.values()), sum(counts.get(i, 0) for i in range(total))
    dense = sum(
        _coverage_type(sample, index, cfg) == "dense"
        for index in range(total)
    )
    overrides = sum(
        _coverage_type(sample, index, cfg) == "override"
        for index in range(total)
    )
    image_rows.append(
        {
            "split": sample.split,
            "image": sample.image_path.name,
            "total_labels": total,
            "dense_labels": dense,
            "sparse_labels": total - dense - overrides,
            "override_labels": overrides,
            "labels_covered_at_least_once": hit,
            "labels_never_covered": total - hit,
            "percent_labels_covered_at_least_once": 100 * hit / total if total else 0,
            "actual_coverages": actual,
            "requested_coverages": requested,
            "percent_requested_coverages": 100 * actual / requested if requested else 0,
            "is_large_image": is_large,
        }
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


_MATRIX_NUMBER = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_MATRIX_TRANSFORMS = {
    "Affine",
    "Rotate",
    "SafeRotate",
    "ShiftScaleRotate",
}
_KNOWN_GEOMETRIC_TOKENS = (
    "Affine",
    "Crop",
    "D4",
    "Distortion",
    "Elastic",
    "Flip",
    "Grid",
    "Pad",
    "Perspective",
    "Piecewise",
    "Resize",
    "Rotate",
    "Scale",
    "ThinPlate",
    "Transpose",
)


def _matrix3(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, str):
        values = [float(item) for item in _MATRIX_NUMBER.findall(value)]
        if len(values) != 9:
            return None
        matrix = np.asarray(values, dtype=np.float64).reshape(3, 3)
    else:
        matrix = np.asarray(value, dtype=np.float64)
    return matrix if matrix.shape == (3, 3) and np.isfinite(matrix).all() else None


def _flatten_applied_transforms(transforms: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if not isinstance(value, dict) or not value.get("applied"):
            return
        children = value.get("transforms")
        if isinstance(children, list):
            for child in children:
                visit(child)
            return
        output.append(value)

    for transform in transforms if isinstance(transforms, list) else []:
        visit(transform)
    return output


def _translation(x: float, y: float) -> np.ndarray:
    return np.asarray(((1.0, 0.0, x), (0.0, 1.0, y), (0.0, 0.0, 1.0)))


def _scale(x: float, y: float) -> np.ndarray:
    return np.asarray(((x, 0.0, 0.0), (0.0, y, 0.0), (0.0, 0.0, 1.0)))


def _virtual_view_projective_matrix(
    transforms: Any,
    source_width: int,
    source_height: int,
) -> tuple[np.ndarray | None, str | None]:
    """Compose supported replay geometry from source into transformed-view space."""

    combined = np.eye(3, dtype=np.float64)
    width = float(source_width)
    height = float(source_height)
    for transform in _flatten_applied_transforms(transforms):
        name = str(transform.get("__class_fullname__", "")).rsplit(".", 1)[-1]
        params = transform.get("params") if isinstance(transform.get("params"), dict) else {}
        local = np.eye(3, dtype=np.float64)
        output_width, output_height = width, height

        if name in _MATRIX_TRANSFORMS:
            matrix = _matrix3(params.get("matrix"))
            if matrix is None:
                return None, f"{name} did not expose a usable 3x3 replay matrix"
            local = matrix
            output_shape = params.get("output_shape")
            if isinstance(output_shape, (list, tuple)) and len(output_shape) >= 2:
                output_height, output_width = map(float, output_shape[:2])
        elif name == "Perspective":
            matrix = _matrix3(params.get("matrix"))
            if matrix is None:
                return None, "Perspective did not expose a usable 3x3 replay matrix"
            local = matrix
            max_width = float(params.get("max_width") or width)
            max_height = float(params.get("max_height") or height)
            if bool(transform.get("keep_size", True)):
                local = _scale(width / max_width, height / max_height) @ local
            else:
                output_width, output_height = max_width, max_height
        elif name == "Resize":
            output_width = float(transform.get("width") or width)
            output_height = float(transform.get("height") or height)
            local = _scale(output_width / width, output_height / height)
        elif name in {"Crop", "CenterCrop", "RandomCrop", "RandomSizedCrop", "RandomResizedCrop"}:
            crop = params.get("crop_coords")
            if not isinstance(crop, (list, tuple)) or len(crop) != 4:
                return None, f"{name} did not expose crop_coords in replay"
            left, top, right, bottom = map(float, crop)
            pad = params.get("pad_params")
            pad_top = pad_left = 0.0
            if isinstance(pad, dict):
                pad_top = float(pad.get("pad_top", 0.0))
                pad_left = float(pad.get("pad_left", 0.0))
            elif isinstance(pad, (list, tuple)) and len(pad) == 4:
                pad_top, _, pad_left, _ = map(float, pad)
            local = _translation(pad_left - left, pad_top - top)
            output_width, output_height = right - left, bottom - top
            requested_size = transform.get("size")
            if name == "RandomResizedCrop" and isinstance(requested_size, (list, tuple)) and len(requested_size) == 2:
                requested_height, requested_width = map(float, requested_size)
                local = _scale(
                    requested_width / output_width,
                    requested_height / output_height,
                ) @ local
                output_width, output_height = requested_width, requested_height
        elif name == "PadIfNeeded":
            pad_left = float(params.get("pad_left", 0.0))
            pad_right = float(params.get("pad_right", 0.0))
            pad_top = float(params.get("pad_top", 0.0))
            pad_bottom = float(params.get("pad_bottom", 0.0))
            local = _translation(pad_left, pad_top)
            output_width = width + pad_left + pad_right
            output_height = height + pad_top + pad_bottom
        elif name == "HorizontalFlip":
            local = np.asarray(((-1.0, 0.0, width), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
        elif name == "VerticalFlip":
            local = np.asarray(((1.0, 0.0, 0.0), (0.0, -1.0, height), (0.0, 0.0, 1.0)))
        elif name == "Transpose":
            local = np.asarray(((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
            output_width, output_height = height, width
        elif name == "RandomRotate90":
            factor = int(params.get("factor", 0)) % 4
            if factor == 1:
                local = np.asarray(((0.0, 1.0, 0.0), (-1.0, 0.0, width), (0.0, 0.0, 1.0)))
                output_width, output_height = height, width
            elif factor == 2:
                local = np.asarray(((-1.0, 0.0, width), (0.0, -1.0, height), (0.0, 0.0, 1.0)))
            elif factor == 3:
                local = np.asarray(((0.0, -1.0, height), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
                output_width, output_height = height, width
        elif any(token in name for token in _KNOWN_GEOMETRIC_TOKENS):
            return None, f"{name} has no supported inverse source-footprint mapping"

        combined = local @ combined
        width, height = output_width, output_height
    return combined, None


def _source_footprint(
    record: dict[str, Any],
    width: int,
    height: int,
) -> tuple[Any | None, str | None]:
    source_bounds = shapely_box(0.0, 0.0, float(width), float(height))
    crop = record.get("crop")
    if crop is None:
        return source_bounds, None
    left, top, right, bottom = map(float, crop)
    if record.get("crop_coordinate_space") != "transformed_view":
        return shapely_box(left, top, right, bottom).intersection(source_bounds), None
    matrix, reason = _virtual_view_projective_matrix(
        record.get("crop_albumentations_applied"),
        width,
        height,
    )
    if matrix is None:
        return None, reason or "virtual transform is not invertible"
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None, "virtual transform matrix is singular"
    points: list[tuple[float, float]] = []
    for x, y in ((left, top), (right, top), (right, bottom), (left, bottom)):
        projected = inverse @ np.asarray((x, y, 1.0), dtype=np.float64)
        if abs(float(projected[2])) < 1e-12:
            return None, "inverse virtual footprint contains a point at infinity"
        points.append(
            (
                float(projected[0] / projected[2]),
                float(projected[1] / projected[2]),
            )
        )
    footprint = Polygon(points)
    if not footprint.is_valid:
        footprint = footprint.buffer(0)
    return footprint.intersection(source_bounds), None


def _source_pixel_coverage_rows(
    samples: list[Sample],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_parent[str(Path(record["parent_image"]).resolve())].append(record)

    rows: list[dict[str, Any]] = []
    for sample in samples:
        parent = str(sample.image_path.resolve())
        source_records = by_parent.get(parent, [])
        footprints: list[Any] = []
        reasons: Counter[str] = Counter()
        for record in source_records:
            footprint, reason = _source_footprint(record, sample.width, sample.height)
            if footprint is not None and not footprint.is_empty:
                footprints.append(footprint)
            if reason is not None:
                reasons[reason] += 1
        covered = float(unary_union(footprints).area) if footprints else 0.0
        total = float(sample.width * sample.height)
        original_image = (
            (sample.provenance or {}).get("original_image") or str(sample.image_path)
        )
        rows.append(
            {
                "split": sample.split,
                "original_image": original_image,
                "tiling_source_image": str(sample.image_path),
                "source_width": sample.width,
                "source_height": sample.height,
                "output_tiles": len(source_records),
                "projected_tiles": len(source_records) - sum(reasons.values()),
                "unsupported_tiles": sum(reasons.values()),
                "covered_source_area_px": covered,
                "source_area_px": total,
                "source_pixel_coverage_percent": 100.0 * covered / total if total else 0.0,
                "coverage_status": "exact" if not reasons else "partial_unsupported_transform",
                "unsupported_reasons": "; ".join(
                    f"{reason} ({count})" for reason, count in sorted(reasons.items())
                ),
            }
        )
    return rows


def _source_pixel_coverage_payload(
    rows: list[dict[str, Any]],
    splits: set[str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "definition": (
            "For each tiling source image, inverse-map accepted tile footprints into "
            "source coordinates, union overlaps, and divide union area by source width*height."
        ),
        "units": "continuous source-pixel area",
        "unsupported_policy": (
            "Non-projective transforms without an inverse footprint are excluded from exact "
            "aggregates and reported per source instead of being approximated silently."
        ),
        "splits": {},
    }
    for split in [*sorted(splits), "all"]:
        selected = rows if split == "all" else [row for row in rows if row["split"] == split]
        exact = [row for row in selected if row["coverage_status"] == "exact"]
        values = [float(row["source_pixel_coverage_percent"]) for row in exact]
        covered = sum(float(row["covered_source_area_px"]) for row in exact)
        total = sum(float(row["source_area_px"]) for row in exact)
        payload["splits"][split] = {
            "source_images": len(selected),
            "represented_source_images": sum(int(row["output_tiles"]) > 0 for row in selected),
            "exact_source_images": len(exact),
            "unsupported_source_images": len(selected) - len(exact),
            "output_tiles": sum(int(row["output_tiles"]) for row in selected),
            "covered_source_area_px": covered,
            "source_area_px": total,
            "pixel_weighted_coverage_percent": 100.0 * covered / total if total else 0.0,
            "mean_per_source_coverage_percent": statistics.mean(values) if values else 0.0,
            "median_per_source_coverage_percent": statistics.median(values) if values else 0.0,
            "minimum_per_source_coverage_percent": min(values) if values else 0.0,
            "maximum_per_source_coverage_percent": max(values) if values else 0.0,
            "fully_covered_source_images": sum(value >= 99.999 for value in values),
        }
    return payload


def _write_coverage_reports(
    root: Path,
    coverage_rows: list[dict[str, Any]],
    image_rows: list[dict[str, Any]],
    source_pixel_rows: list[dict[str, Any]],
    class_totals: dict[tuple[str, int], Counter],
    split_summary: dict[str, Counter],
    splits: set[str],
    *,
    background_ratio: float,
    visualize: bool,
) -> list[Path]:
    visuals: list[Path] = []
    _write_csv(root / "label_coverage.csv", coverage_rows)
    aggregate_image_rows = list(image_rows)
    for split in [*sorted(splits), "all"]:
        selected_rows = image_rows if split == "all" else [row for row in image_rows if row["split"] == split]
        total_labels = sum(row["total_labels"] for row in selected_rows)
        hit = sum(row["labels_covered_at_least_once"] for row in selected_rows)
        actual = sum(row["actual_coverages"] for row in selected_rows)
        requested = sum(row["requested_coverages"] for row in selected_rows)
        aggregate_image_rows.append(
            {
                "split": split,
                "image": "__TOTAL__",
                "total_labels": total_labels,
                "dense_labels": sum(row["dense_labels"] for row in selected_rows),
                "sparse_labels": sum(row["sparse_labels"] for row in selected_rows),
                "override_labels": sum(row["override_labels"] for row in selected_rows),
                "labels_covered_at_least_once": hit,
                "labels_never_covered": total_labels - hit,
                "percent_labels_covered_at_least_once": 100 * hit / total_labels if total_labels else 0,
                "actual_coverages": actual,
                "requested_coverages": requested,
                "percent_requested_coverages": 100 * actual / requested if requested else 0,
                "is_large_image": "",
            }
        )
    _write_csv(root / "label_hit_summary.csv", aggregate_image_rows)
    class_rows = []
    for (split, class_id), totals in sorted(class_totals.items()):
        requested, actual = totals["requested_coverages"], totals["actual_coverages"]
        original, hit = totals["original_labels"], totals["labels_covered_at_least_once"]
        class_rows.append(
            {
                "split": split,
                "class_id": class_id,
                "original_labels": original,
                "labels_covered_at_least_once": hit,
                "percent_labels_covered_at_least_once": 100 * hit / original if original else 0,
                "actual_coverages": actual,
                "requested_coverages": requested,
                "percent_of_requested": 100 * actual / requested if requested else 0,
            }
        )
    for class_id in sorted({class_id for _, class_id in class_totals}):
        totals = sum((class_totals[(split, class_id)] for split in splits), Counter())
        requested, actual = totals["requested_coverages"], totals["actual_coverages"]
        original, hit = totals["original_labels"], totals["labels_covered_at_least_once"]
        class_rows.append(
            {
                "split": "all",
                "class_id": class_id,
                "original_labels": original,
                "labels_covered_at_least_once": hit,
                "percent_labels_covered_at_least_once": 100 * hit / original if original else 0,
                "actual_coverages": actual,
                "requested_coverages": requested,
                "percent_of_requested": 100 * actual / requested if requested else 0,
            }
        )
    _write_csv(root / "class_coverage_summary.csv", class_rows)
    keys = sorted({key for summary in split_summary.values() for key in summary})

    def tile_row(split: str, summary: Counter) -> dict[str, Any]:
        total = summary["total_output_images"]
        return {
            "split": split,
            **{key: summary[key] for key in keys},
            "background_fraction": summary["empty_output_images"] / total if total else 0.0,
        }

    tile_rows = [tile_row(split, split_summary[split]) for split in sorted(splits)]
    combined = Counter()
    for split in splits:
        combined.update(split_summary[split])
    tile_rows.append(tile_row("all", combined))
    _write_csv(root / "tile_summary.csv", tile_rows)
    _write_csv(root / "source_pixel_coverage.csv", source_pixel_rows)
    (root / "source_pixel_coverage.json").write_text(
        json.dumps(
            _source_pixel_coverage_payload(source_pixel_rows, splits),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if visualize:
        visuals.append(
            save_source_pixel_coverage_summary(
                source_pixel_rows,
                root / "source_pixel_coverage.jpg",
            )
        )
        visuals.append(
            save_label_coverage_summary(
                image_rows,
                root / "label_coverage.jpg",
            )
        )

    sampling_payload: dict[str, Any] = {
        "definition": (
            "background_ratio is the fraction of output images with no annotations; "
            "class annotation instances are not part of this calculation"
        ),
        "requested_background_fraction": background_ratio,
        "target_formula": (
            "nearest whole number to "
            "positive_output_images * background_ratio / (1 - background_ratio)"
        ),
        "source_policy": (
            "target an equal mix of wholly empty source images and object-free "
            "regions cropped from populated images, then cross-fill if one source is insufficient"
        ),
        "splits": {},
    }
    summaries = {
        **{split: split_summary[split] for split in sorted(splits)},
        "all": combined,
    }
    for split, summary in summaries.items():
        total = int(summary["total_output_images"])
        target = int(summary["target_background_images"])
        actual = int(summary["actual_background_images"])
        target_empty = int(summary["target_background_from_empty_source_images"])
        target_populated = int(summary["target_background_from_populated_image_space"])
        actual_empty = int(summary["background_from_empty_source_images"])
        actual_populated = int(summary["background_from_populated_image_space"])
        if actual < target:
            status = "target missed"
            reason = (
                "candidate pools were exhausted; inspect missed_background_images "
                "and the candidate-source counts"
            )
        elif actual_empty == target_empty and actual_populated == target_populated:
            status = "target and equal source mix met"
            reason = None
        else:
            status = "overall target met with source fallback"
            deficient = []
            if actual_empty < target_empty:
                deficient.append("wholly empty source images")
            if actual_populated < target_populated:
                deficient.append("object-free regions in populated images")
            reason = f"insufficient candidates from {', '.join(deficient)}"
        sampling_payload["splits"][split] = {
            "status": status,
            "reason": reason,
            "positive_output_images": int(summary["positive_output_images"]),
            "target_background_images": target,
            "actual_background_images": actual,
            "total_output_images": total,
            "actual_background_fraction": actual / total if total else 0.0,
            "target_from_empty_source_images": target_empty,
            "actual_from_empty_source_images": actual_empty,
            "candidate_empty_source_images": int(summary["candidate_empty_source_images"]),
            "target_from_populated_image_space": target_populated,
            "actual_from_populated_image_space": actual_populated,
            "candidate_populated_source_images": int(summary["candidate_populated_source_images"]),
        }
    (root / "background_sampling.json").write_text(
        json.dumps(sampling_payload, indent=2),
        encoding="utf-8",
    )
    return visuals


def _write_tiling_class_counts(
    builder: Any,
    before_samples: list[Sample],
    names: dict[int, str],
    *,
    visualize: bool,
) -> None:
    before_counts = Counter(
        annotation.class_id
        for sample in before_samples
        for annotation in sample.annotations
    )
    after_counts = Counter(
        annotation.class_id
        for sample in builder.output_samples
        for annotation in sample.annotations
    )
    before_background = sum(not sample.annotations for sample in before_samples)
    after_background = sum(not sample.annotations for sample in builder.output_samples)
    before_annotated = len(before_samples) - before_background
    after_annotated = len(builder.output_samples) - after_background
    named = {str(class_id): class_name for class_id, class_name in sorted(names.items())}
    payload = {
        "definition": (
            "annotation_counts count annotation instances; image_composition counts images. "
            "Only image_composition is used to calculate a tiling background fraction"
        ),
        "operation": builder.operation,
        "annotation_counts": {
            "before": {
                str(class_id): before_counts.get(class_id, 0)
                for class_id in sorted(names)
            },
            "after": {
                str(class_id): after_counts.get(class_id, 0)
                for class_id in sorted(names)
            },
            "names": named,
        },
        "image_composition": {
            "before": {
                "annotated": before_annotated,
                "background": before_background,
                "total": len(before_samples),
                "background_fraction": (
                    before_background / len(before_samples) if before_samples else 0.0
                ),
            },
            "after": {
                "annotated": after_annotated,
                "background": after_background,
                "total": len(builder.output_samples),
                "background_fraction": (
                    after_background / len(builder.output_samples)
                    if builder.output_samples
                    else 0.0
                ),
            },
        },
        # Kept for report compatibility. These mixed-unit fields are deprecated;
        # new consumers should use the two explicitly named sections above.
        "before": {
            **{str(class_id): before_counts.get(class_id, 0) for class_id in sorted(names)},
            "background": before_background,
        },
        "after": {
            **{str(class_id): after_counts.get(class_id, 0) for class_id in sorted(names)},
            "background": after_background,
        },
        "names_before": {**named, "background": "background"},
        "names_after": {**named, "background": "background"},
    }
    (builder.reports_dir / "class_counts.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    if visualize:
        before_named = {
            class_name: before_counts.get(class_id, 0)
            for class_id, class_name in sorted(names.items())
        }
        after_named = {
            class_name: after_counts.get(class_id, 0)
            for class_id, class_name in sorted(names.items())
        }
        output = save_tiling_count_summary(
            before_named,
            after_named,
            {
                "before": {
                    "annotated": before_annotated,
                    "background": before_background,
                },
                "after": {
                    "annotated": after_annotated,
                    "background": after_background,
                },
            },
            builder.reports_dir / "class_counts.jpg",
        )
        builder.visuals.append(str(output.relative_to(builder.staging)))


def _select_tiling_preview_items(
    samples: list[Sample],
    boxes_by_source: dict[str, list[tuple[int, int, int, int]]],
    kept_parent_paths: set[str],
    *,
    small_limit: int,
) -> list[tuple[Sample, list[tuple[int, int, int, int]], str]]:
    """Select one pass-through source and three genuinely cropped sources."""

    available = [
        sample
        for sample in samples
        if str(sample.image_path) in kept_parent_paths
    ]
    small = sorted(
        (
            sample
            for sample in available
            if max(sample.width, sample.height) <= small_limit
            and not boxes_by_source.get(str(sample.image_path))
        ),
        key=lambda sample: (
            -bool(sample.annotations),
            -(sample.width * sample.height),
            str(sample.relative_path),
        ),
    )
    tiled = sorted(
        (
            sample
            for sample in available
            if boxes_by_source.get(str(sample.image_path))
            and not (
                len(boxes_by_source[str(sample.image_path)]) == 1
                and boxes_by_source[str(sample.image_path)][0]
                == (0, 0, sample.width, sample.height)
            )
        ),
        key=lambda sample: (
            -bool(sample.annotations),
            -len(boxes_by_source[str(sample.image_path)]),
            -(sample.width * sample.height),
            str(sample.relative_path),
        ),
    )
    output: list[tuple[Sample, list[tuple[int, int, int, int]], str]] = []
    if small:
        output.append((small[0], [], "Small image copied unchanged"))
    tiled_limit = 3 if output else 4
    for sample in tiled[:tiled_limit]:
        boxes = boxes_by_source[str(sample.image_path)]
        output.append((sample, boxes, f"Tiled source · {len(boxes)} crop windows"))
    return output


def _clear_inherited_tiling_reports(builder: Any) -> None:
    """Hide class counts from an earlier operation until tiling replaces them."""

    for filename in (
        "class_counts.json",
        "class_counts.jpg",
        "background_filter.json",
    ):
        inherited = builder.reports_dir / filename
        if inherited.exists():
            inherited.unlink()


def _validate_coverage_settings(cfg: dict[str, Any]) -> None:
    if int(cfg["tile_size"]) <= 0:
        raise ValueError("tile_size must be positive")
    try:
        low, high = map(float, cfg["scale_range"])
    except (TypeError, ValueError) as exc:
        raise ValueError("scale_range must contain exactly two numeric values") from exc
    if low <= 0 or high < low:
        raise ValueError("scale_range must contain two positive ascending values")
    if cfg["polo_radius_px"] is not None and float(cfg["polo_radius_px"]) <= 0:
        raise ValueError("polo_radius_px must be positive or None")
    if float(cfg["radius_multiplier"]) <= 0:
        raise ValueError("radius_multiplier must be positive")
    for key in (
        "target_appearances_per_object",
        "sparse_appearances_per_object",
        "max_attempts_per_target",
        "max_background_attempts_per_tile",
    ):
        if int(cfg[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if int(cfg["min_nearby_objects_for_full_coverage"]) < 0:
        raise ValueError("min_nearby_objects_for_full_coverage must be non-negative")
    if int(cfg["large_image_threshold"]) < 0:
        raise ValueError("large_image_threshold must be non-negative or None")
    if cfg["max_tiles_per_source_image"] is not None and int(cfg["max_tiles_per_source_image"]) <= 0:
        raise ValueError("max_tiles_per_source_image must be positive or None")
    if not 0 <= float(cfg["background_ratio"]) < 1:
        raise ValueError("background_ratio must be in [0, 1)")
    if not 0 <= float(cfg["min_area_ratio"]) <= 1:
        raise ValueError("min_area_ratio must be in [0, 1]")
    if not 1 <= int(cfg["jpeg_quality"]) <= 100:
        raise ValueError("jpeg_quality must be in [1, 100]")
    for source_id, target in cfg["object_appearance_overrides"].items():
        if int(target) <= 0:
            raise ValueError(f"object_appearance_overrides[{source_id!r}] must be positive")


def _save_staging_contact_sheet(builder, task: Task, relative_output: str) -> None:
    from .dataset import Dataset

    builder.write_yaml()
    staged = Dataset.open(builder.staging, task=task, progress=False)
    output = builder.staging / relative_output
    staged.visualize(split=None, n=12, seed=42, columns=3, save_to=output, show=False)
    builder.visuals.append(str(output.relative_to(builder.staging)))
