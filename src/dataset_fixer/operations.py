from __future__ import annotations

import inspect
import json
import random
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Hashable, Iterable, Mapping

from tqdm.auto import tqdm

from .errors import DatasetValidationError, ValidationIssue
from .models import Annotation, DatasetMetadata, Sample, Task
from .planning import (
    normalize_split_ratios,
    project_move_images_with_classes,
    project_move_n_groups,
    resolve_class_selectors,
    resolve_removed_classes,
    resolve_renamed_classes,
    select_empty_images,
)
from .split_group_audit import (
    audit_split_groups,
    print_split_group_audit,
    write_split_group_audit,
)
from .utils import ensure_safe_destination, normalize_split, settings_fingerprint, slugify, to_jsonable
from .validation_audit import stage_load_validation_audit
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
    group_by: Callable[[Path], Hashable],
    assign: Callable[[Path], str | None] | None,
    seed: int,
    visualize: bool,
    progress: bool,
    dry_run: bool,
    validate_output: bool = True,
) -> "Dataset":
    normalized = normalize_split_ratios(ratios)
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

    # Ratios apply to the annotated images as well as to the total, because a
    # split that is correct overall can still starve one side of labels when
    # most images are background.
    group_annotated = {
        group: sum(bool(sample.annotations) for sample in group_samples)
        for group, group_samples in groups.items()
    }
    annotated_total = sum(group_annotated.values())

    counts = Counter()
    annotated_counts = Counter()
    for group, target in locked.items():
        counts[target] += len(groups[group])
        annotated_counts[target] += group_annotated[group]
        for sample in groups[group]:
            assignments[str(sample.image_path)] = target
    rng = random.Random(seed)
    remaining = [group for group in groups if group not in locked]
    rng.shuffle(remaining)
    # Place annotated groups first: they are the scarce resource, and
    # background groups can then fill whatever total quota is left.
    remaining.sort(key=lambda group: -group_annotated[group])
    target_counts = {split: ratio * len(samples) for split, ratio in normalized.items()}
    target_annotated = {
        split: ratio * annotated_total for split, ratio in normalized.items()
    }

    def deficits(split: str) -> tuple[float, float]:
        return (
            (target_annotated[split] - annotated_counts[split])
            / max(target_annotated[split], 1),
            (target_counts[split] - counts[split]) / max(target_counts[split], 1),
        )

    for group in remaining:
        eligible = [split for split, ratio in normalized.items() if ratio > 0]
        carries_labels = group_annotated[group] > 0
        target = max(
            eligible,
            key=lambda split: (
                deficits(split) if carries_labels else deficits(split)[::-1]
            ),
        )
        counts[target] += len(groups[group])
        annotated_counts[target] += group_annotated[group]
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
        "ratio_targets": "total_and_annotated_images",
        "source_splits": sorted(selected),
        "seed": seed,
        "group_by": _callback_description(group_by),
        "assign": _callback_description(assign),
        "resolved_groups": resolved_groups,
        "resolved_assignments": assignments,
        "resolved_distribution": {
            split: {
                "images": int(counts[split]),
                "annotated_images": int(annotated_counts[split]),
                "background_images": int(counts[split] - annotated_counts[split]),
                "image_fraction": counts[split] / len(samples) if samples else 0.0,
                "annotated_fraction": (
                    annotated_counts[split] / annotated_total if annotated_total else 0.0
                ),
                "requested_fraction": normalized[split],
            }
            for split in normalized
        },
        "visualize": visualize,
    }
    builder = _builder(dataset, destination, name, "split", settings)
    try:
        if visualize:
            group_lookup = {
                image: group
                for group, details in resolved_groups.items()
                for image in details["images"]
            }
            preview = save_split_preview(
                samples,
                assignments,
                group_lookup,
                dataset.task,
                dataset._metadata,
                builder.reports_dir / "split_preview.jpg",
            )
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


def move_images_with_classes(
    dataset: "Dataset",
    classes: Iterable[str | int],
    *,
    to_split: str,
    destination: str | Path | None,
    name: str | None,
    source_splits: Iterable[str] | None,
    group_by: Callable[[Path], Hashable] | None,
    visualize: bool,
    progress: bool,
    dry_run: bool,
    validate_output: bool = True,
) -> "Dataset":
    """Materialize an annotation-aware whole-image split move."""

    target = normalize_split(to_split)
    selected = (
        {normalize_split(split) for split in source_splits}
        if source_splits is not None
        else set(dataset.splits)
    )
    if not selected:
        raise ValueError("source_splits must select at least one split")
    class_ids = resolve_class_selectors(dataset._metadata, classes)
    projected, summary, details = project_move_images_with_classes(
        dataset._samples,
        selected_splits=selected,
        class_ids=class_ids,
        to_split=target,
        group_by=group_by,
    )
    settings = {
        "selected_classes": {
            class_id: dataset._metadata.names[class_id]
            for class_id in sorted(class_ids)
        },
        "source_splits": sorted(selected),
        "to_split": target,
        "group_by": _callback_description(group_by),
        **summary,
        "visualize": visualize,
    }
    assignments = {
        str(source.image_path): output.split
        for source, output in zip(dataset._samples, projected)
    }
    builder = _builder(
        dataset,
        destination,
        name,
        "move-images-with-classes",
        settings,
    )
    try:
        _print_start(builder, dataset._samples, settings)
        if dry_run:
            print("Dry run complete; no dataset was published.")
            builder.cleanup()
            return dataset
        iterator = tqdm(
            dataset._samples,
            desc="Moving class-containing images",
            unit="image",
            disable=not progress,
        )
        for sample in iterator:
            path = str(sample.image_path)
            target_for_sample = assignments[path]
            detail = details[path]
            matched_class_ids = sorted(
                {
                    annotation.class_id
                    for annotation in sample.annotations
                    if annotation.class_id in class_ids
                }
            ) if sample.split in selected else []
            provenance = {"split_group": detail["group"]}
            if detail["selected_by_group"]:
                provenance["class_move"] = {
                    "matched_class_ids": matched_class_ids,
                    "matched_class_names": [
                        dataset._metadata.names[class_id]
                        for class_id in matched_class_ids
                    ],
                    "direct_class_match": detail["direct_class_match"],
                    "group_expansion": not detail["direct_class_match"],
                    "group": detail["group"],
                    "from_split": sample.split,
                    "to_split": target_for_sample,
                    "moved": sample.split != target_for_sample,
                }
            builder.add_copy(
                sample,
                split=target_for_sample,
                provenance=provenance or None,
            )
        (builder.reports_dir / "class_move_summary.json").write_text(
            json.dumps(settings, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if visualize:
            summary_path = save_split_summary(
                dataset._samples,
                assignments,
                builder.reports_dir / "class_move_summary.jpg",
            )
            builder.visuals.append(str(summary_path.relative_to(builder.staging)))
        return _publish(builder, progress=progress, validate_output=validate_output)
    except Exception:
        builder.cleanup()
        raise


def move_n_groups(
    dataset: "Dataset",
    *,
    n: int,
    from_split: str,
    to_split: str,
    group_by: Callable[[Path], Hashable],
    seed: int,
    destination: str | Path | None,
    name: str | None,
    visualize: bool,
    progress: bool,
    dry_run: bool,
    validate_output: bool = True,
) -> "Dataset":
    """Materialize a deterministic, group-atomic split move."""

    source = normalize_split(from_split)
    target = normalize_split(to_split)
    projected, summary, details = project_move_n_groups(
        dataset._samples,
        n=n,
        from_split=source,
        to_split=target,
        group_by=group_by,
        seed=seed,
    )
    settings = {
        "from_split": source,
        "to_split": target,
        "group_by": _callback_description(group_by),
        "seed": seed,
        **summary,
        "visualize": visualize,
    }
    assignments = {
        str(input_sample.image_path): output_sample.split
        for input_sample, output_sample in zip(dataset._samples, projected)
    }
    builder = _builder(dataset, destination, name, "move-n-groups", settings)
    try:
        _print_start(builder, dataset._samples, settings)
        if dry_run:
            print("Dry run complete; no dataset was published.")
            builder.cleanup()
            return dataset
        iterator = tqdm(
            dataset._samples,
            desc="Moving physical groups",
            unit="image",
            disable=not progress,
        )
        for sample in iterator:
            path = str(sample.image_path)
            detail = details[path]
            provenance: dict[str, Any] = {"split_group": detail["group"]}
            if detail["selected"]:
                provenance["group_move"] = {
                    "group": detail["group"],
                    "from_split": sample.split,
                    "to_split": assignments[path],
                    "moved": detail["moved"],
                }
            builder.add_copy(
                sample,
                split=assignments[path],
                provenance=provenance,
            )
        (builder.reports_dir / "group_move_summary.json").write_text(
            json.dumps(settings, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if visualize:
            summary_path = save_split_summary(
                dataset._samples,
                assignments,
                builder.reports_dir / "group_move_summary.jpg",
            )
            builder.visuals.append(str(summary_path.relative_to(builder.staging)))
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
    merge_into: str | int | None = None,
    drop_empty_images: bool,
    drop_containing_images: bool,
    visualize: bool,
    progress: bool,
    dry_run: bool,
    validate_output: bool = True,
) -> "Dataset":
    if drop_containing_images and merge_into is not None:
        raise ValueError(
            "drop_containing_images and merge_into are mutually exclusive"
        )
    selected_splits = {normalize_split(s) for s in splits} if splits else set(dataset.splits)
    selected_samples = [s for s in dataset._samples if s.split in selected_splits]
    removed, mapping, metadata = resolve_removed_classes(
        dataset._metadata,
        classes,
        merge_into=merge_into,
    )
    settings = {
        "removed_classes": {class_id: dataset._metadata.names[class_id] for class_id in sorted(removed)},
        "splits": sorted(selected_splits),
        "drop_empty_images": drop_empty_images,
        "drop_containing_images": drop_containing_images,
        "dropped_containing_images": sum(
            sample.split in selected_splits
            and any(annotation.class_id in removed for annotation in sample.annotations)
            for sample in dataset._samples
        ) if drop_containing_images else 0,
        "class_mapping": mapping,
        "visualize": visualize,
    }
    if merge_into is not None:
        output_class_id = mapping[next(iter(removed))]
        settings["merge_into"] = {
            "selector": merge_into,
            "output_class_id": output_class_id,
            "output_class_name": metadata.names[output_class_id],
        }
    builder = _builder(dataset, destination, name, "remove-classes", settings, metadata=metadata)
    try:
        if visualize:
            preview_sample = next((s for s in selected_samples if any(a.class_id in removed for a in s.annotations)), selected_samples[0])
            preview = save_class_removal_preview(
                preview_sample,
                mapping,
                dataset.task,
                dataset._metadata,
                metadata,
                builder.reports_dir / "remove_classes_preview.jpg",
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
        dropped_containing = 0
        iterator = tqdm(selected_samples, desc="Removing classes", unit="image", disable=not progress)
        for sample in iterator:
            if drop_containing_images and any(
                annotation.class_id in removed for annotation in sample.annotations
            ):
                dropped_containing += 1
                continue
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
                    "images": {
                        "before": len(selected_samples),
                        "after": len(builder.records),
                        "dropped_containing_removed_classes": dropped_containing,
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
    visualize_kwargs: Mapping[str, Any],
    visualize_kwargs_description: Mapping[str, Any],
    progress: bool,
    dry_run: bool,
    validate_output: bool = True,
) -> "Dataset":
    selected = {normalize_split(s) for s in splits} if splits else set(dataset.splits)
    samples = [s for s in dataset._samples if s.split in selected]
    group_validation, group_report = audit_split_groups(
        dataset,
        exported_splits=selected,
    )
    print_split_group_audit(group_report)
    settings = {
        "splits": sorted(selected),
        "allow_lossy": allow_lossy,
        "visualize": visualize,
        "visualize_kwargs": dict(visualize_kwargs_description),
    }
    builder = _builder(dataset, destination, name, "export", settings)
    builder.visualize_kwargs = dict(visualize_kwargs)
    builder.validation_details["split_group_isolation"] = group_validation
    try:
        write_split_group_audit(builder.reports_dir, group_report)
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


def rename_classes(
    dataset: "Dataset",
    renames: Mapping[str | int, str],
    *,
    destination: str | Path | None,
    name: str | None,
    progress: bool,
    dry_run: bool,
    validate_output: bool = True,
) -> "Dataset":
    renamed, metadata = resolve_renamed_classes(dataset._metadata, renames)
    settings = {
        "renamed_classes": renamed,
        "class_ids_changed": False,
    }
    builder = _builder(dataset, destination, name, "rename-classes", settings, metadata=metadata)
    try:
        inherited_preview = builder.reports_dir / "rename_classes_summary.jpg"
        if inherited_preview.exists():
            inherited_preview.unlink()
        _print_start(builder, dataset._samples, settings)
        if dry_run:
            print("Dry run complete; no dataset was published.")
            builder.cleanup()
            return dataset
        iterator = tqdm(dataset._samples, desc="Renaming classes", unit="image", disable=not progress)
        for sample in iterator:
            builder.add_copy(
                sample,
                split=sample.split,
                provenance={"class_renames": renamed},
            )
        (builder.reports_dir / "class_renames.json").write_text(
            json.dumps(to_jsonable(renamed), indent=2, sort_keys=True),
            encoding="utf-8",
        )
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
            raise ImportError("RLE conversion requires pycocotools and OpenCV; reinstall dataset-fixer") from exc
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
    builder = OutputBuilder(
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
    load_validation, load_visualization = stage_load_validation_audit(
        dataset._validation_audit,
        dataset._validation_audit_visualization,
        builder.reports_dir,
    )
    builder.validation_details["load_validation"] = load_validation
    if load_visualization is not None:
        builder.visuals.append(str(load_visualization.relative_to(builder.staging)))
    builder.warnings.extend(dataset._warnings)
    return builder


def _operation_detail(operation: str, settings: dict[str, Any]) -> str:
    if operation == "split":
        return "split-" + "-".join(f"{k}{int(v * 100)}" for k, v in settings["ratios"].items())
    if operation == "move-images-with-classes":
        values = list(settings["selected_classes"].values())
        return (
            "move-"
            + "-".join(map(slugify, values[:3]))
            + f"-to-{settings['to_split']}"
        )
    if operation == "move-n-groups":
        return (
            f"move-{settings['selected_groups']}-groups-"
            f"{settings['from_split']}-to-{settings['to_split']}"
        )
    if operation == "remove-classes":
        values = list(settings["removed_classes"].values())
        return "remove-" + "-".join(map(slugify, values[:3]))
    if operation == "rename-classes":
        values = [item["to"] for item in settings["renamed_classes"].values()]
        return "rename-" + "-".join(map(slugify, values[:3]))
    if operation == "export":
        return "yolo"
    if operation == "tile-grid":
        return f"tile-grid-{settings['tile_size']}-o{int(round(settings['overlap'] * 100))}"
    if operation == "tile-coverage":
        return f"tile-coverage-{settings['tile_size']}"
    if operation == "augment":
        return f"augment-{settings['copies']}x"
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
        print(f"Warnings: {len(manifest['warnings'])} (see reports/dataset-info.json)")
    if manifest.get("visuals"):
        print(f"Visual audits: {manifest['visuals']}")
    print(f"Reports: {result.location / 'reports'}")
    return result
