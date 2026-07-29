from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from PIL import Image, ImageOps
from shapely.geometry import Polygon, box as shapely_box
from tqdm.auto import tqdm

from .errors import DatasetValidationError, ValidationIssue
from .models import Annotation, Sample, Task
from .operations import _builder, _print_start, _publish
from .utils import normalize_split
from .visualization import (
    save_coverage_annotated_original,
    save_tiling_count_summary,
    save_tiling_preview,
)

if TYPE_CHECKING:
    from .dataset import Dataset


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
                    transformed = _transform_annotations(
                        sample, (left, top, right, bottom), dataset.task, min_area_ratio, allow_lossy, builder.warnings
                    )
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
                                "requested_background_images": count,
                                "negative_tiles": ratio,
                            },
                            expected="enough empty grid windows to reach the requested final fraction",
                            suggestion="lower negative_tiles, use negative_tiles='all', or provide more background images",
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
    for annotation in sample.annotations:
        transformed = _transform_annotation(
            annotation,
            crop,
            task,
            min_area_ratio,
            allow_lossy,
            warnings,
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
) -> Annotation | None:
    left, top, right, bottom = crop
    crop_shape = shapely_box(left, top, right, bottom)
    if task is Task.POLO:
        assert annotation.point is not None and annotation.radius is not None
        x, y = annotation.point
        radius = annotation.radius
        if left <= x - radius and x + radius <= right and top <= y - radius and y + radius <= bottom:
            return annotation.clone(point=(x - left, y - top))
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
            raise DatasetValidationError("RLE/multipart segmentation requires allow_lossy=True before tiling")
        source_polygon = Polygon(annotation.polygon)
        intersection = source_polygon.intersection(crop_shape)
        if intersection.is_empty or intersection.area / source_polygon.area < min_area_ratio:
            return None
        if intersection.geom_type == "MultiPolygon":
            if not allow_lossy:
                raise DatasetValidationError(
                    "Crop split one segmentation into disconnected components; pass allow_lossy=True"
                )
            intersection = max(intersection.geoms, key=lambda geom: geom.area)
            warnings.append(f"Kept largest segment fragment for annotation {annotation.source_id}")
        if intersection.geom_type != "Polygon":
            if not allow_lossy:
                raise DatasetValidationError("Cropped segmentation has unsupported geometry")
            polygon_parts = [geom for geom in getattr(intersection, "geoms", ()) if geom.geom_type == "Polygon"]
            if not polygon_parts:
                warnings.append(f"Dropped unsupported cropped segmentation {annotation.source_id}")
                return None
            intersection = max(polygon_parts, key=lambda geom: geom.area)
        if intersection.interiors:
            if not allow_lossy:
                raise DatasetValidationError("Cropped segmentation has holes; pass allow_lossy=True")
            warnings.append(f"Removed holes from cropped segmentation {annotation.source_id}")
        polygon = [(float(x - left), float(y - top)) for x, y in list(intersection.exterior.coords)[:-1]]
        if len(polygon) < 3:
            return None
        xs, ys = zip(*polygon)
        return annotation.clone(polygon=polygon, bbox=(min(xs), min(ys), max(xs), max(ys)))
    return None


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
            while attempts < max_attempts and any(provisional[i] < targets[i] for i in targets):
                needed = [i for i in targets if provisional[i] < targets[i]]
                focus_idx = rng.choice(needed)
                crop = _make_crop_containing(
                    sample.annotations[focus_idx],
                    sample.width,
                    sample.height,
                    dataset.task,
                    cfg,
                    rng,
                )
                attempts += 1
                if crop is None:
                    continue
                adjusted: list[Annotation] = []
                indices: list[int] = []
                scale = tile_size / (crop[2] - crop[0])
                for idx, annotation in enumerate(sample.annotations):
                    source_annotation = _coverage_source_annotation(annotation, dataset.task, cfg)
                    transformed = _transform_annotation(
                        source_annotation,
                        crop,
                        dataset.task,
                        min_area_ratio,
                        allow_lossy,
                        builder.warnings,
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
                if not indices:
                    continue
                if any(provisional[idx] >= targets[idx] for idx in indices):
                    continue
                generated.append({"box": crop, "annotations": adjusted, "indices": indices})
                provisional.update(indices)
            generated_before_cap = len(generated)
            cap = cfg["max_tiles_per_source_image"]
            if cap is not None and len(generated) > int(cap):
                generated = rng.sample(generated, int(cap))
                split_summary[sample.split]["positive_tiles_dropped_by_source_cap"] += generated_before_cap - len(generated)
            counts = Counter()
            for tile_idx, tile in enumerate(generated):
                with Image.open(sample.image_path) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                    crop_image = image.crop(tile["box"])
                    if crop_image.size != (tile_size, tile_size):
                        crop_image = crop_image.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
                rel = sample.relative_path.parent / f"{sample.relative_path.stem}_tile_{tile_idx}.jpg"
                builder.add_image(
                    sample,
                    crop_image,
                    split=sample.split,
                    relative_path=rel,
                    annotations=tile["annotations"],
                    provenance={
                        "crop": list(tile["box"]),
                        "tile_index": tile_idx,
                        "tile_mode": "coverage",
                        "zoom": tile_size / (tile["box"][2] - tile["box"][0]),
                        "scale": tile_size / (tile["box"][2] - tile["box"][0]),
                        "source_annotation_indices": tile["indices"],
                    },
                    jpeg_quality=int(cfg["jpeg_quality"]),
                )
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
                "tile_boxes": [tile["box"] for tile in generated],
                "background_boxes": [],
                "background_box_origins": [],
                "next_tile_idx": len(generated),
            }
            records.append(record)
            _append_coverage_rows(sample, targets, counts, True, coverage_rows, image_rows, class_totals, cfg)
            iterator.set_postfix(produced=len(builder.records), refresh=False)
            missed = sum(max(0, targets[i] - counts[i]) for i in targets)
            if missed:
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
                selected_count = min(count, len(unused_source_candidates))
                selected = (
                    list(unused_source_candidates)
                    if selected_count == len(unused_source_candidates)
                    else rng.sample(unused_source_candidates, selected_count)
                )
                selected_paths = {str(sample.image_path) for sample in selected}
                unused_source_candidates[:] = [
                    sample
                    for sample in unused_source_candidates
                    if str(sample.image_path) not in selected_paths
                ]
                kept_sources.extend(selected)
                return selected_count

            empty_source_count = take_empty_source_images(target_from_empty_sources)
            empty_source_count += _allocate_backgrounds(
                empty_source_records,
                target_from_empty_sources - empty_source_count,
                dataset.task,
                cfg,
                rng,
                origin="empty_source_image",
            )
            populated_space_count = _allocate_backgrounds(
                populated_records,
                target_from_populated_space,
                dataset.task,
                cfg,
                rng,
                origin="populated_image_empty_space",
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
                            "reason": "; ".join(fallback_reasons)
                            or "all background candidate pools were exhausted",
                        },
                        expected="enough object-free source images or crop locations to reach the requested final fraction",
                        suggestion=(
                            "lower background_ratio, raise max_background_attempts_per_tile or "
                            "max_tiles_per_source_image, or provide more background imagery"
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
                            },
                            jpeg_quality=int(cfg["jpeg_quality"]),
                        )
                        record["next_tile_idx"] += 1
                        split_summary[split].update(total_output_images=1, empty_output_images=1, tiled_output_images=1, empty_tiled_images=1)
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

        _write_coverage_reports(
            builder.staging / "coverage_summary",
            coverage_rows,
            image_rows,
            class_totals,
            split_summary,
            selected,
            background_ratio=float(cfg["background_ratio"]),
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
    crop_dim = min(max(1, int(cfg["tile_size"] / scale)), width, height)
    x1, y1, x2, y2 = _annotation_bounds(annotation, task, cfg)
    if x2 - x1 > crop_dim or y2 - y1 > crop_dim:
        return None
    max_left, max_top = width - crop_dim, height - crop_dim
    left_min = max(0, int(math.ceil(x2 - crop_dim)))
    left_max = min(max_left, int(math.floor(x1)))
    top_min = max(0, int(math.ceil(y2 - crop_dim)))
    top_max = min(max_top, int(math.floor(y1)))
    if left_min > left_max or top_min > top_max:
        return None
    left, top = rng.randint(left_min, left_max), rng.randint(top_min, top_max)
    return left, top, left + crop_dim, top + crop_dim


def _random_crop(width: int, height: int, cfg: dict[str, Any], rng: random.Random) -> tuple[int, int, int, int]:
    scale = rng.uniform(*cfg["scale_range"])
    dim = min(max(1, int(cfg["tile_size"] / scale)), width, height)
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
) -> int:
    allocated = 0
    cap = cfg["max_tiles_per_source_image"]
    for _ in range(desired * int(cfg["max_background_attempts_per_tile"])):
        if allocated >= desired:
            break
        eligible = [r for r in records if cap is None or r["next_tile_idx"] + len(r["background_boxes"]) < int(cap)]
        if not eligible:
            break
        record = rng.choice(eligible)
        sample = record["sample"]
        crop = _random_crop(sample.width, sample.height, cfg, rng)
        if any(_annotation_intersects_crop(annotation, crop, task, cfg) for annotation in sample.annotations):
            continue
        if any(_boxes_intersect(crop, existing) for existing in record["background_boxes"]):
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


def _write_coverage_reports(
    root: Path,
    coverage_rows: list[dict[str, Any]],
    image_rows: list[dict[str, Any]],
    class_totals: dict[tuple[str, int], Counter],
    split_summary: dict[str, Counter],
    splits: set[str],
    *,
    background_ratio: float,
) -> None:
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

    for filename in ("class_counts.json", "class_counts.jpg"):
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
