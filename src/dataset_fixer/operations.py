from __future__ import annotations

import inspect
import json
import math
import random
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Hashable, Iterable

from tqdm.auto import tqdm

from .errors import DatasetValidationError, ValidationIssue
from .models import Annotation, DatasetMetadata, Sample, Task
from .planning import select_empty_images
from .utils import ensure_safe_destination, normalize_split, settings_fingerprint, slugify, to_jsonable
from .visualization import (
    save_class_count_summary,
    save_class_removal_preview,
    save_empty_image_balance_summary,
    save_split_preview,
    save_split_summary,
)
from .writer import OutputBuilder

if TYPE_CHECKING:
    from .dataset import Dataset


def split_dataset(
    dataset: "Dataset",
    ratios: dict[str, float],
    *,
    destination: str | Path | None,
    name: str | None,
    source_splits: Iterable[str] | None,
    group_by: Callable[[Path], Hashable] | None,
    assign: Callable[[Path], str | None] | None,
    seed: int,
    visualize: bool,
    progress: bool,
    dry_run: bool,
    validate_output: bool = True,
) -> "Dataset":
    normalized = {normalize_split(k): float(v) for k, v in ratios.items()}
    if any(v < 0 for v in normalized.values()) or not math.isclose(sum(normalized.values()), 1.0, abs_tol=1e-9):
        raise ValueError(f"Split ratios must be non-negative and sum to 1.0, got {normalized}")
    selected = {normalize_split(s) for s in source_splits} if source_splits else set(dataset.splits)
    samples = [sample for sample in dataset._samples if sample.split in selected]
    if not samples:
        raise DatasetValidationError("No images selected for splitting")

    groups: dict[Hashable, list[Sample]] = defaultdict(list)
    locked: dict[Hashable, str] = {}
    assignments: dict[str, str] = {}
    for sample in samples:
        group = group_by(sample.image_path) if group_by else str(sample.image_path)
        groups[group].append(sample)
        target = assign(sample.image_path) if assign else None
        if target is not None:
            target = normalize_split(target)
            if target not in normalized:
                raise ValueError(f"Assignment callback returned {target!r}, which is absent from ratios")
            if group in locked and locked[group] != target:
                raise DatasetValidationError(
                    ValidationIssue(
                        "One physical group received conflicting explicit split assignments",
                        value={"group": repr(group), "first": locked[group], "second": target},
                    )
                )
            locked[group] = target

    counts = Counter()
    for group, target in locked.items():
        counts[target] += len(groups[group])
        for sample in groups[group]:
            assignments[str(sample.image_path)] = target
    rng = random.Random(seed)
    remaining = [group for group in groups if group not in locked]
    rng.shuffle(remaining)
    target_counts = {split: ratio * len(samples) for split, ratio in normalized.items()}
    for group in remaining:
        eligible = [split for split, ratio in normalized.items() if ratio > 0]
        target = max(eligible, key=lambda split: (target_counts[split] - counts[split]) / max(target_counts[split], 1))
        counts[target] += len(groups[group])
        for sample in groups[group]:
            assignments[str(sample.image_path)] = target

    resolved_groups = {
        repr(group): {
            "size": len(group_samples),
            "split": assignments[str(group_samples[0].image_path)],
            "images": [str(s.image_path) for s in group_samples],
        }
        for group, group_samples in groups.items()
    }
    settings = {
        "ratios": normalized,
        "source_splits": sorted(selected),
        "seed": seed,
        "group_by": _callback_description(group_by),
        "assign": _callback_description(assign),
        "resolved_groups": resolved_groups,
        "resolved_assignments": assignments,
        "visualize": visualize,
    }
    builder = _builder(dataset, destination, name, "split", settings)
    try:
        if visualize:
            preview = save_split_preview(samples, assignments, builder.reports_dir / "split_preview.jpg")
            builder.visuals.append(str(preview.relative_to(builder.staging)))
            print(f"Split sanity preview: {preview}")
        _print_start(builder, samples, settings)
        if dry_run:
            print("Dry run complete; no dataset was published.")
            builder.cleanup()
            return dataset
        iterator = tqdm(samples, desc="Writing split dataset", unit="image", disable=not progress)
        for sample in iterator:
            target = assignments[str(sample.image_path)]
            builder.add_copy(sample, split=target, provenance={"split_group": next(k for k, v in resolved_groups.items() if str(sample.image_path) in v["images"])})
        if visualize:
            summary = save_split_summary(samples, assignments, builder.reports_dir / "split_summary.jpg")
            builder.visuals.append(str(summary.relative_to(builder.staging)))
        return _publish(builder, progress=progress, validate_output=validate_output)
    except Exception:
        builder.cleanup()
        raise


def remove_classes(
    dataset: "Dataset",
    classes: Iterable[str | int],
    *,
    destination: str | Path | None,
    name: str | None,
    splits: Iterable[str] | None,
    drop_empty_images: bool,
    visualize: bool,
    progress: bool,
    dry_run: bool,
    validate_output: bool = True,
) -> "Dataset":
    selected_splits = {normalize_split(s) for s in splits} if splits else set(dataset.splits)
    selected_samples = [s for s in dataset._samples if s.split in selected_splits]
    reverse: dict[str, list[int]] = defaultdict(list)
    for class_id, class_name in dataset._metadata.names.items():
        reverse[class_name].append(class_id)
    removed: set[int] = set()
    for selector in classes:
        if isinstance(selector, int):
            if selector not in dataset._metadata.names:
                raise ValueError(f"Unknown class ID {selector}; available IDs are {sorted(dataset._metadata.names)}")
            removed.add(selector)
        else:
            matches = reverse.get(selector, [])
            if len(matches) != 1:
                raise ValueError(f"Class name {selector!r} matched {len(matches)} classes; available names are {list(reverse)}")
            removed.add(matches[0])
    if not removed:
        raise ValueError("At least one class must be removed")
    remaining = [class_id for class_id in sorted(dataset._metadata.names) if class_id not in removed]
    if not remaining:
        raise DatasetValidationError("Removing these classes would leave the dataset with no classes")
    mapping = {old: new for new, old in enumerate(remaining)}
    metadata = _remap_metadata(dataset._metadata, mapping)
    settings = {
        "removed_classes": {class_id: dataset._metadata.names[class_id] for class_id in sorted(removed)},
        "splits": sorted(selected_splits),
        "drop_empty_images": drop_empty_images,
        "class_mapping": mapping,
        "visualize": visualize,
    }
    builder = _builder(dataset, destination, name, "remove-classes", settings, metadata=metadata)
    try:
        if visualize:
            preview_sample = next((s for s in selected_samples if any(a.class_id in removed for a in s.annotations)), selected_samples[0])
            preview = save_class_removal_preview(
                preview_sample, removed, dataset.task, dataset._metadata, builder.reports_dir / "remove_classes_preview.jpg"
            )
            builder.visuals.append(str(preview.relative_to(builder.staging)))
            print(f"Class-removal sanity preview: {preview}")
        _print_start(builder, selected_samples, settings)
        if dry_run:
            print("Dry run complete; no dataset was published.")
            builder.cleanup()
            return dataset
        before_counts = Counter(a.class_id for s in selected_samples for a in s.annotations)
        before_background = sum(not sample.annotations for sample in selected_samples)
        after_counts: Counter[int] = Counter()
        after_background = 0
        iterator = tqdm(selected_samples, desc="Removing classes", unit="image", disable=not progress)
        for sample in iterator:
            annotations = [a.clone(class_id=mapping[a.class_id]) for a in sample.annotations if a.class_id in mapping]
            if drop_empty_images and not annotations:
                continue
            after_counts.update(annotation.class_id for annotation in annotations)
            after_background += not annotations
            builder.add_copy(sample, split=sample.split, annotations=annotations, provenance={"class_mapping": mapping})
        (builder.reports_dir / "class_counts.json").write_text(
            json.dumps(
                {
                    "definition": "class values count annotations; background counts images with no annotations",
                    "before": {
                        **{str(class_id): before_counts.get(class_id, 0) for class_id in sorted(dataset._metadata.names)},
                        "background": before_background,
                    },
                    "after": {
                        **{str(class_id): after_counts.get(class_id, 0) for class_id in sorted(metadata.names)},
                        "background": after_background,
                    },
                    "names_before": {
                        **{str(class_id): class_name for class_id, class_name in sorted(dataset._metadata.names.items())},
                        "background": "background",
                    },
                    "names_after": {
                        **{str(class_id): class_name for class_id, class_name in sorted(metadata.names.items())},
                        "background": "background",
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if visualize:
            before_named = {
                class_name: before_counts.get(class_id, 0)
                for class_id, class_name in sorted(dataset._metadata.names.items())
            }
            before_named["background"] = before_background
            after_named = {
                class_name: after_counts.get(class_id, 0)
                for class_id, class_name in sorted(metadata.names.items())
            }
            after_named["background"] = after_background
            summary = save_class_count_summary(
                before_named,
                after_named,
                builder.reports_dir / "class_counts.jpg",
            )
            builder.visuals.append(str(summary.relative_to(builder.staging)))
        return _publish(
            builder,
            class_mapping=mapping,
            progress=progress,
            validate_output=validate_output,
        )
    except Exception:
        builder.cleanup()
        raise


def export_dataset(
    dataset: "Dataset",
    *,
    destination: str | Path | None,
    name: str | None,
    splits: Iterable[str] | None,
    allow_lossy: bool,
    visualize: bool,
    progress: bool,
    dry_run: bool,
    validate_output: bool = True,
) -> "Dataset":
    selected = {normalize_split(s) for s in splits} if splits else set(dataset.splits)
    samples = [s for s in dataset._samples if s.split in selected]
    settings = {"splits": sorted(selected), "allow_lossy": allow_lossy, "visualize": visualize}
    builder = _builder(dataset, destination, name, "export", settings)
    try:
        if visualize and samples:
            from .visualization import visualize_samples

            path = builder.reports_dir / "export_preview.jpg"
            visualize_samples(samples, dataset.task, dataset._metadata, split=samples[0].split, n=1, seed=42, columns=1, save_to=path, show=False)
            builder.visuals.append(str(path.relative_to(builder.staging)))
            print(f"Export sanity preview: {path}")
        _print_start(builder, samples, settings)
        if dry_run:
            builder.cleanup()
            return dataset
        iterator = tqdm(samples, desc="Exporting YOLO dataset", unit="image", disable=not progress)
        for sample in iterator:
            annotations = [_make_representable(a, allow_lossy, builder) for a in sample.annotations]
            builder.add_copy(sample, split=sample.split, annotations=annotations)
        return _publish(builder, progress=progress, validate_output=validate_output)
    except Exception:
        builder.cleanup()
        raise


def rebalance_empty_dataset(
    dataset: "Dataset",
    max_empty_fraction: float,
    *,
    destination: str | Path | None,
    name: str | None,
    splits: Iterable[str] | None,
    seed: int,
    visualize: bool,
    progress: bool,
    dry_run: bool,
    validate_output: bool = True,
) -> "Dataset":
    selected = {normalize_split(split) for split in splits} if splits else set(dataset.splits)
    kept, summary = select_empty_images(
        dataset._samples,
        max_empty_fraction=float(max_empty_fraction),
        selected_splits=selected,
        seed=seed,
    )
    settings = {
        "max_empty_fraction": float(max_empty_fraction),
        "splits": sorted(selected),
        "seed": seed,
        "summary": summary,
        "visualize": visualize,
    }
    builder = _builder(dataset, destination, name, "rebalance-empty", settings)
    try:
        _print_start(builder, kept, settings)
        if dry_run:
            builder.cleanup()
            return dataset
        iterator = tqdm(kept, desc="Writing empty-image balance", unit="image", disable=not progress)
        for sample in iterator:
            builder.add_copy(
                sample,
                split=sample.split,
                provenance={"empty_image": not bool(sample.annotations)},
            )
        (builder.reports_dir / "empty_image_balance.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        if visualize and kept:
            preview = builder.reports_dir / "empty_image_balance.jpg"
            save_empty_image_balance_summary(summary, preview)
            builder.visuals.append(str(preview.relative_to(builder.staging)))
        return _publish(builder, progress=progress, validate_output=validate_output)
    except Exception:
        builder.cleanup()
        raise


def _make_representable(annotation: Annotation, allow_lossy: bool, builder: OutputBuilder) -> Annotation:
    if annotation.rle is None:
        return annotation
    if not allow_lossy:
        raise DatasetValidationError(
            ValidationIssue(
                "Segmentation cannot be represented as one YOLO polygon",
                value=annotation.source_id,
                suggestion="re-run export with allow_lossy=True",
            )
        )
    if "multipart" in annotation.rle:
        segments = annotation.rle["multipart"]
        largest = max(segments, key=_flat_polygon_area)
        polygon = [(float(largest[i]), float(largest[i + 1])) for i in range(0, len(largest), 2)]
    else:
        try:
            import cv2
            import numpy as np
            from pycocotools import mask as mask_utils
        except ImportError as exc:
            raise ImportError("RLE conversion requires dataset-fixer[coco-rle] and OpenCV") from exc
        decoded = mask_utils.decode(annotation.rle).astype("uint8")
        contours, _ = cv2.findContours(decoded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [contour.reshape(-1, 2) for contour in contours if contour.size >= 6]
        if not valid:
            raise DatasetValidationError(f"RLE annotation {annotation.source_id} produced no valid polygon")
        polygon = [tuple(map(float, point)) for point in max(valid, key=cv2.contourArea)]
    builder.warnings.append(f"Lossy segmentation conversion for annotation {annotation.source_id}")
    return annotation.clone(polygon=polygon, rle=None)


def _flat_polygon_area(flat: list[float]) -> float:
    pts = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]))) / 2


def _remap_metadata(metadata: DatasetMetadata, mapping: dict[int, int]) -> DatasetMetadata:
    result = metadata.copy()
    result.names = {mapping[old]: metadata.names[old] for old in mapping}
    result.radii = {mapping[old]: metadata.radii[old] for old in mapping if old in metadata.radii}
    result.kpt_names = {mapping[old]: metadata.kpt_names[old] for old in mapping if old in metadata.kpt_names}
    return result


def _callback_description(callback: Callable | None) -> dict[str, Any] | None:
    if callback is None:
        return None
    try:
        source = inspect.getsource(callback).strip()
    except (OSError, TypeError):
        source = None
    return {
        "module": getattr(callback, "__module__", None),
        "qualname": getattr(callback, "__qualname__", None),
        "source": source,
    }


def _builder(
    dataset: "Dataset",
    destination: str | Path | None,
    name: str | None,
    operation: str,
    settings: dict[str, Any],
    *,
    metadata: DatasetMetadata | None = None,
) -> OutputBuilder:
    fingerprint = settings_fingerprint(settings)
    detail = _operation_detail(operation, settings)
    derived_name = slugify(name or f"{dataset.name}__{detail}__{fingerprint}")
    dest = Path(destination).expanduser() if destination is not None else dataset.location.parent / derived_name
    dest = dest.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    ensure_safe_destination(dataset.location, dest)
    return OutputBuilder(
        source_root=dataset.location,
        source_name=dataset.name,
        destination=dest,
        name=name or dest.name,
        task=dataset.task,
        metadata=metadata or dataset._metadata.copy(),
        operation=operation,
        settings=settings,
        parent_manifest=dataset._manifest,
    )


def _operation_detail(operation: str, settings: dict[str, Any]) -> str:
    if operation == "split":
        return "split-" + "-".join(f"{k}{int(v * 100)}" for k, v in settings["ratios"].items())
    if operation == "remove-classes":
        values = list(settings["removed_classes"].values())
        return "remove-" + "-".join(map(slugify, values[:3]))
    if operation == "export":
        return "yolo"
    if operation == "tile-grid":
        return f"tile-grid-{settings['tile_size']}-o{int(round(settings['overlap'] * 100))}"
    if operation == "tile-coverage":
        return f"tile-coverage-{settings['tile_size']}"
    return operation


def _print_start(builder: OutputBuilder, samples: list[Sample], settings: dict[str, Any]) -> None:
    print(f"\n{builder.operation}: {builder.name}")
    print(f"Destination: {builder.destination}")
    print(f"Images: {len(samples)} | splits: {sorted({s.split for s in samples})}")
    print(f"Settings: {json.dumps(to_jsonable(settings), sort_keys=True)}")
    print("Validation: source passed load-time consistency checks")
    estimated = settings.get("estimated_tiles", len(samples))
    print(f"Estimated work: {estimated} {'tiles' if 'estimated_tiles' in settings else 'images'}")


def _publish(
    builder: OutputBuilder,
    class_mapping: dict[int, int] | None = None,
    *,
    progress: bool = True,
    validate_output: bool = True,
) -> "Dataset":
    from .dataset import Dataset

    manifest = builder.publish(
        class_mapping=class_mapping,
        progress=progress,
        validate=validate_output,
    )
    if progress and validate_output:
        print("Reusing validated in-memory index; published dataset is not rescanned.")
    result = builder.result_dataset(manifest)
    duration = time.time() - builder.started
    print(f"\nCreated {result.name}")
    print(f"Location: {result.location}")
    print(f"data.yaml: {result.data_yaml}")
    manifest = result._manifest
    annotations = sum(len(sample.annotations) for sample in result._samples)
    throughput = len(result._samples) / duration if duration else 0.0
    print(f"Settings fingerprint: {manifest.get('settings_fingerprint')}")
    print(
        f"Duration: {duration:.2f}s | throughput: {throughput:.2f} images/s | "
        f"images: {len(result._samples)} | annotations: {annotations} | training_ready: {result.training_ready}"
    )
    if manifest.get("warnings"):
        print(f"Warnings: {len(manifest['warnings'])} (see dataset-fixer.json)")
    if manifest.get("visuals"):
        print(f"Visual audits: {manifest['visuals']}")
    print(f"Reports: {result.location / 'reports'}")
    return result
