from __future__ import annotations

import csv
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
from .visualization import save_coverage_annotated_original, save_grid_preview

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
        )
    if mode == "coverage":
        return _tile_coverage(
            dataset,
            destination=destination,
            name=name,
            splits=splits,
            tile_size=tile_size,
            visualize=visualize,
            progress=progress,
            dry_run=dry_run,
            overrides=settings,
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
    try:
        if visualize and samples:
            sample = max(samples, key=lambda s: s.width * s.height)
            preview = save_grid_preview(sample, grid_boxes(sample.width, sample.height, tile_size, overlap), builder.reports_dir / "grid_preview.jpg")
            builder.visuals.append(str(preview.relative_to(builder.staging)))
            print(f"Grid sanity preview: {preview}")
        _print_start(builder, samples, op_settings)
        if dry_run:
            builder.cleanup()
            return dataset

        positive_count = 0
        negatives: list[tuple[Sample, tuple[int, int, int, int], Path, dict[str, Any]]] = []
        iterator = tqdm(total=total_tiles, desc="Generating grid tiles", unit="tile", disable=not progress)
        for sample in samples:
            boxes = grid_boxes(sample.width, sample.height, tile_size, overlap)
            if len(boxes) == 1 and boxes[0] == (0, 0, sample.width, sample.height):
                builder.add_copy(sample, split=sample.split)
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
                        positive_count += 1
                    else:
                        negatives.append((sample, (left, top, right, bottom), rel, provenance))
                    iterator.update(1)
        iterator.close()

        if negative_tiles == "all":
            chosen_negatives = negatives
        elif negative_tiles == "none":
            chosen_negatives = []
        elif isinstance(negative_tiles, (int, float)):
            if float(negative_tiles) < 0:
                raise ValueError("negative tile ratio must be non-negative")
            count = min(len(negatives), int(round(positive_count * float(negative_tiles))))
            chosen_negatives = random.Random(42).sample(negatives, count)
        else:
            raise ValueError("negative_tiles must be 'all', 'none', or a non-negative ratio")
        for sample, crop, rel, provenance in chosen_negatives:
            with Image.open(sample.image_path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB").crop(crop)
            builder.add_image(sample, image, split=sample.split, relative_path=rel, annotations=[], provenance=provenance)
        if visualize:
            _save_staging_contact_sheet(builder, dataset.task, "reports/grid_tiles_audit.jpg")
        return _publish(builder, progress=progress)
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
    left, top, right, bottom = crop
    result: list[Annotation] = []
    crop_shape = shapely_box(left, top, right, bottom)
    for annotation in sample.annotations:
        if task is Task.POLO:
            assert annotation.point is not None and annotation.radius is not None
            x, y = annotation.point
            r = annotation.radius
            if left <= x - r and x + r <= right and top <= y - r and y + r <= bottom:
                result.append(annotation.clone(point=(x - left, y - top)))
            continue
        if annotation.bbox is None:
            continue
        x1, y1, x2, y2 = annotation.bbox
        ix1, iy1, ix2, iy2 = max(x1, left), max(y1, top), min(x2, right), min(y2, bottom)
        original_area = max(0, x2 - x1) * max(0, y2 - y1)
        intersection_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if original_area <= 0 or intersection_area / original_area < min_area_ratio:
            continue
        new_bbox = (ix1 - left, iy1 - top, ix2 - left, iy2 - top)
        if task is Task.DETECT:
            result.append(annotation.clone(bbox=new_bbox))
        elif task is Task.POSE:
            keypoints = []
            for x, y, visibility in annotation.keypoints or []:
                if left <= x < right and top <= y < bottom and visibility != 0:
                    keypoints.append((x - left, y - top, visibility))
                else:
                    keypoints.append((0.0, 0.0, 0.0 if visibility is not None else None))
            if any(x != 0 or y != 0 for x, y, _ in keypoints):
                result.append(annotation.clone(bbox=new_bbox, keypoints=keypoints))
        elif task is Task.SEGMENT:
            if not annotation.polygon:
                if allow_lossy:
                    warnings.append(f"Dropped non-polygon segmentation {annotation.source_id} during tiling")
                    continue
                raise DatasetValidationError("RLE/multipart segmentation requires allow_lossy=True before grid tiling")
            intersection = Polygon(annotation.polygon).intersection(crop_shape)
            if intersection.is_empty or intersection.area / Polygon(annotation.polygon).area < min_area_ratio:
                continue
            if intersection.geom_type == "MultiPolygon":
                if not allow_lossy:
                    raise DatasetValidationError("Crop split one segmentation into disconnected components; pass allow_lossy=True")
                intersection = max(intersection.geoms, key=lambda geom: geom.area)
                warnings.append(f"Kept largest segment fragment for annotation {annotation.source_id}")
            if intersection.geom_type != "Polygon" or intersection.interiors:
                if not allow_lossy:
                    raise DatasetValidationError("Cropped segmentation has holes or unsupported geometry")
                warnings.append(f"Simplified cropped segmentation {annotation.source_id}")
            polygon = [(float(x - left), float(y - top)) for x, y in list(intersection.exterior.coords)[:-1]]
            xs, ys = zip(*polygon)
            result.append(annotation.clone(polygon=polygon, bbox=(min(xs), min(ys), max(xs), max(ys))))
    return result


COVERAGE_DEFAULTS: dict[str, Any] = {
    "large_image_threshold": 1000,
    "scale_range": (0.75, 1.25),
    "fixed_polo_radius_px": 15.0,
    "radius_multiplier": 1.0,
    "target_coverage_per_label": 5,
    "sparse_coverage_per_label": 1,
    "min_nearby_labels_for_full_coverage": 5,
    "dense_neighbor_radius_px": None,
    "max_bg_ratio": 0.1,
    "max_attempts_per_needed_crop": 15,
    "max_bg_attempts_per_tile": 15,
    "max_total_tiles_per_source_image": 100,
    "jpeg_quality": 95,
    "seed": 42,
}


def _tile_coverage(
    dataset: "Dataset",
    *,
    destination: str | Path | None,
    name: str | None,
    splits: Iterable[str] | None,
    tile_size: int,
    visualize: bool,
    progress: bool,
    dry_run: bool,
    overrides: dict[str, Any],
) -> "Dataset":
    if dataset.task is not Task.POLO:
        raise ValueError("coverage tiling is POLO-specific; use mode='grid' for other tasks")
    selected = {normalize_split(s) for s in splits} if splits else set(dataset.splits)
    samples = [s for s in dataset._samples if s.split in selected]
    cfg = dict(COVERAGE_DEFAULTS)
    cfg.update(overrides)
    cfg["tile_size"] = int(tile_size)
    _validate_coverage_settings(cfg)
    if cfg["dense_neighbor_radius_px"] is None:
        cfg["dense_neighbor_radius_px"] = tile_size * 0.5
    cfg["mode"] = "coverage"
    cfg["splits"] = sorted(selected)
    cfg["visualize"] = visualize
    dense_target = max(int(cfg["target_coverage_per_label"]), int(cfg["sparse_coverage_per_label"]))
    cap = cfg["max_total_tiles_per_source_image"]
    cfg["estimated_tiles"] = sum(
        1
        if sample.width <= int(cfg["large_image_threshold"])
        else min(int(cap) if cap is not None else len(sample.annotations) * dense_target, len(sample.annotations) * dense_target)
        for sample in samples
    )
    rng = random.Random(int(cfg["seed"]))
    builder = _builder(dataset, destination, name, "tile-coverage", cfg)
    try:
        if visualize and samples:
            preview_sample = max(samples, key=lambda s: s.width * s.height)
            preview_boxes = []
            if preview_sample.annotations:
                first = preview_sample.annotations[0]
                candidate = _make_crop_containing(first, preview_sample.width, preview_sample.height, cfg, rng)
                if candidate:
                    preview_boxes.append(candidate)
            preview = save_grid_preview(preview_sample, preview_boxes, builder.reports_dir / "coverage_preview.jpg")
            builder.visuals.append(str(preview.relative_to(builder.staging)))
            print(f"Coverage sanity preview: {preview}")
        _print_start(builder, samples, cfg)
        if dry_run:
            builder.cleanup()
            return dataset

        coverage_rows: list[dict[str, Any]] = []
        image_rows: list[dict[str, Any]] = []
        class_totals: dict[tuple[str, int], Counter] = defaultdict(Counter)
        split_summary: dict[str, Counter] = defaultdict(Counter)
        records: list[dict[str, Any]] = []
        iterator = tqdm(samples, desc="Coverage tiling", unit="image", disable=not progress)
        for sample in iterator:
            if sample.width <= int(cfg["large_image_threshold"]):
                annotations = [a.clone(radius=float(cfg["fixed_polo_radius_px"])) for a in sample.annotations]
                with Image.open(sample.image_path) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                    rel = sample.relative_path.with_suffix(".jpg")
                    builder.add_image(sample, image, split=sample.split, relative_path=rel, annotations=annotations, jpeg_quality=int(cfg["jpeg_quality"]))
                targets = {idx: 1 for idx in range(len(sample.annotations))}
                counts = {idx: 1 for idx in range(len(sample.annotations))}
                _append_coverage_rows(sample, targets, counts, False, coverage_rows, image_rows, class_totals, cfg)
                split_summary[sample.split].update(total_output_images=1, copied_small_images=1)
                split_summary[sample.split]["positive_output_images" if annotations else "empty_output_images"] += 1
                iterator.set_postfix(produced=len(builder.records), refresh=False)
                continue

            targets = _coverage_targets(sample, cfg)
            generated: list[dict[str, Any]] = []
            attempts = 0
            max_attempts = max(1, sum(targets.values()) * int(cfg["max_attempts_per_needed_crop"]))
            provisional = Counter()
            while attempts < max_attempts and any(provisional[i] < targets[i] for i in targets):
                needed = [i for i in targets if provisional[i] < targets[i]]
                focus_idx = rng.choice(needed)
                crop = _make_crop_containing(sample.annotations[focus_idx], sample.width, sample.height, cfg, rng)
                attempts += 1
                if crop is None:
                    continue
                adjusted: list[Annotation] = []
                indices: list[int] = []
                for idx, annotation in enumerate(sample.annotations):
                    transformed = _adjust_polo(annotation, crop, cfg)
                    if transformed is not None:
                        adjusted.append(transformed)
                        indices.append(idx)
                if not indices:
                    continue
                generated.append({"box": crop, "annotations": adjusted, "indices": indices})
                provisional.update(indices)
            generated_before_cap = len(generated)
            cap = cfg["max_total_tiles_per_source_image"]
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
                "counts": dict(counts),
                "background_boxes": [],
                "next_tile_idx": len(generated),
            }
            records.append(record)
            _append_coverage_rows(sample, targets, counts, True, coverage_rows, image_rows, class_totals, cfg)
            iterator.set_postfix(produced=len(builder.records), refresh=False)
            missed = sum(max(0, targets[i] - counts[i]) for i in targets)
            if missed:
                builder.warnings.append(
                    f"{sample.split}/{sample.relative_path}: missed {missed} requested label coverages"
                )

        for split in selected:
            split_records = [r for r in records if r["sample"].split == split]
            desired = int(round(split_summary[split]["positive_output_images"] * float(cfg["max_bg_ratio"])))
            allocated = _allocate_backgrounds(split_records, desired, cfg, rng)
            split_summary[split]["target_empty_images_from_file_ratio"] = desired
            split_summary[split]["missed_empty_images_from_file_ratio"] = max(0, desired - allocated)
            if allocated < desired:
                builder.warnings.append(
                    f"{split}: allocated {allocated}/{desired} requested background tiles without touching point circles"
                )
            for record in split_records:
                sample = record["sample"]
                with Image.open(sample.image_path) as opened:
                    source = ImageOps.exif_transpose(opened).convert("RGB")
                    for crop in record["background_boxes"]:
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
                                "tile_mode": "coverage-background",
                                "zoom": tile_size / (crop[2] - crop[0]),
                                "scale": tile_size / (crop[2] - crop[0]),
                            },
                            jpeg_quality=int(cfg["jpeg_quality"]),
                        )
                        record["next_tile_idx"] += 1
                        split_summary[split].update(total_output_images=1, empty_output_images=1, tiled_output_images=1, empty_tiled_images=1)
                if visualize:
                    output = builder.staging / "coverage_summary" / "annotated_originals" / split / f"{sample.image_path.stem}_coverage.jpg"
                    save_coverage_annotated_original(
                        sample, record["counts"], record["targets"], record["background_boxes"], output, cfg
                    )
                    builder.visuals.append(str(output.relative_to(builder.staging)))

        _write_coverage_reports(builder.staging / "coverage_summary", coverage_rows, image_rows, class_totals, split_summary, selected)
        if visualize:
            _save_staging_contact_sheet(builder, dataset.task, "coverage_summary/random_tile_contact_sheet.jpg")
        return _publish(builder, progress=progress)
    except Exception:
        builder.cleanup()
        raise


def _coverage_targets(sample: Sample, cfg: dict[str, Any]) -> dict[int, int]:
    targets: dict[int, int] = {}
    radius = float(cfg["dense_neighbor_radius_px"])
    for i, annotation in enumerate(sample.annotations):
        assert annotation.point is not None
        nearby = sum(
            j != i
            and other.point is not None
            and math.hypot(annotation.point[0] - other.point[0], annotation.point[1] - other.point[1]) <= radius
            for j, other in enumerate(sample.annotations)
        )
        targets[i] = int(
            cfg["target_coverage_per_label"]
            if nearby >= int(cfg["min_nearby_labels_for_full_coverage"])
            else cfg["sparse_coverage_per_label"]
        )
    return targets


def _make_crop_containing(
    annotation: Annotation, width: int, height: int, cfg: dict[str, Any], rng: random.Random
) -> tuple[int, int, int, int] | None:
    assert annotation.point is not None
    scale = rng.uniform(*cfg["scale_range"])
    crop_dim = int(cfg["tile_size"] / scale)
    crop_w, crop_h = min(crop_dim, width), min(crop_dim, height)
    x, y = annotation.point
    radius = float(cfg["fixed_polo_radius_px"])
    max_left, max_top = width - crop_w, height - crop_h
    left_min = max(0, int(math.ceil(x + radius - crop_w)))
    left_max = min(max_left, int(math.floor(x - radius)))
    top_min = max(0, int(math.ceil(y + radius - crop_h)))
    top_max = min(max_top, int(math.floor(y - radius)))
    if left_min > left_max or top_min > top_max:
        return None
    left, top = rng.randint(left_min, left_max), rng.randint(top_min, top_max)
    return left, top, left + crop_w, top + crop_h


def _adjust_polo(annotation: Annotation, crop: tuple[int, int, int, int], cfg: dict[str, Any]) -> Annotation | None:
    assert annotation.point is not None
    left, top, right, bottom = crop
    x, y = annotation.point
    radius = float(cfg["fixed_polo_radius_px"])
    if not (left <= x - radius and x + radius < right and top <= y - radius and y + radius < bottom):
        return None
    scale = cfg["tile_size"] / (right - left)
    return annotation.clone(
        point=((x - left) * scale, (y - top) * scale),
        radius=radius * scale * float(cfg["radius_multiplier"]),
    )


def _random_crop(width: int, height: int, cfg: dict[str, Any], rng: random.Random) -> tuple[int, int, int, int]:
    scale = rng.uniform(*cfg["scale_range"])
    dim = int(cfg["tile_size"] / scale)
    crop_w, crop_h = min(dim, width), min(dim, height)
    left = rng.randint(0, width - crop_w) if width > crop_w else 0
    top = rng.randint(0, height - crop_h) if height > crop_h else 0
    return left, top, left + crop_w, top + crop_h


def _circle_intersects(annotation: Annotation, crop: tuple[int, int, int, int], radius: float) -> bool:
    assert annotation.point is not None
    x, y = annotation.point
    left, top, right, bottom = crop
    closest_x, closest_y = min(max(x, left), right), min(max(y, top), bottom)
    return (x - closest_x) ** 2 + (y - closest_y) ** 2 <= radius**2


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _allocate_backgrounds(records: list[dict[str, Any]], desired: int, cfg: dict[str, Any], rng: random.Random) -> int:
    allocated = 0
    cap = cfg["max_total_tiles_per_source_image"]
    for _ in range(desired * int(cfg["max_bg_attempts_per_tile"])):
        if allocated >= desired:
            break
        eligible = [r for r in records if cap is None or r["next_tile_idx"] + len(r["background_boxes"]) < int(cap)]
        if not eligible:
            break
        record = rng.choice(eligible)
        sample = record["sample"]
        crop = _random_crop(sample.width, sample.height, cfg, rng)
        if any(_circle_intersects(a, crop, float(cfg["fixed_polo_radius_px"])) for a in sample.annotations):
            continue
        if any(_boxes_intersect(crop, existing) for existing in record["background_boxes"]):
            continue
        record["background_boxes"].append(crop)
        allocated += 1
    return allocated


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
        coverage_rows.append(
            {
                "split": sample.split,
                "image": sample.image_path.name,
                "label_idx": idx,
                "class_id": annotation.class_id,
                "x_norm": annotation.point[0] / sample.width if annotation.point else "",
                "y_norm": annotation.point[1] / sample.height if annotation.point else "",
                "actual_coverages": count,
                "requested_coverages": target,
                "covered_at_least_once": count >= 1,
                "percent_of_requested": 100 * count / target if target else 0,
                "source_image_width": sample.width,
                "source_image_height": sample.height,
                "is_large_image": is_large,
                "coverage_type": "dense" if target == cfg["target_coverage_per_label"] else "sparse",
            }
        )
        totals = class_totals[(sample.split, annotation.class_id)]
        totals.update(original_labels=1, requested_coverages=target, actual_coverages=count, labels_covered_at_least_once=int(count >= 1))
    total = len(sample.annotations)
    hit = sum(counts.get(i, 0) >= 1 for i in range(total))
    requested, actual = sum(targets.values()), sum(counts.get(i, 0) for i in range(total))
    dense = sum(v == cfg["target_coverage_per_label"] for v in targets.values())
    image_rows.append(
        {
            "split": sample.split,
            "image": sample.image_path.name,
            "total_labels": total,
            "dense_labels": dense,
            "sparse_labels": total - dense,
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
    tile_rows = [{"split": split, **{key: split_summary[split][key] for key in keys}} for split in sorted(splits)]
    tile_rows.append({"split": "all", **{key: sum(split_summary[s][key] for s in splits) for key in keys}})
    _write_csv(root / "tile_summary.csv", tile_rows)


def _validate_coverage_settings(cfg: dict[str, Any]) -> None:
    if int(cfg["tile_size"]) <= 0:
        raise ValueError("tile_size must be positive")
    low, high = map(float, cfg["scale_range"])
    if low <= 0 or high < low:
        raise ValueError("scale_range must contain two positive ascending values")
    for key in ("fixed_polo_radius_px", "radius_multiplier"):
        if float(cfg[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("target_coverage_per_label", "sparse_coverage_per_label", "max_attempts_per_needed_crop"):
        if int(cfg[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 <= float(cfg["max_bg_ratio"]) <= 1:
        raise ValueError("max_bg_ratio must be in [0, 1]")


def _save_staging_contact_sheet(builder, task: Task, relative_output: str) -> None:
    from .dataset import Dataset

    builder.write_yaml()
    staged = Dataset.open(builder.staging, task=task, progress=False)
    output = builder.staging / relative_output
    staged.visualize(split=None, n=12, seed=42, columns=3, save_to=output, show=False)
    builder.visuals.append(str(output.relative_to(builder.staging)))
