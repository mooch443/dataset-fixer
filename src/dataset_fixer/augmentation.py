from __future__ import annotations

import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tqdm.auto import tqdm

from .errors import DatasetValidationError, ValidationIssue
from .models import Annotation, Sample, Task
from .operations import _builder, _print_start, _publish
from .utils import normalize_split, to_jsonable
from .visualization import save_class_count_summary

if TYPE_CHECKING:
    from .dataset import Dataset


def serialize_pipeline(transforms: Any, compose_args: dict[str, Any]) -> dict[str, Any]:
    """Validate and serialize an Albumentations pipeline at planning time."""

    A = _albumentations()
    reserved = {"bbox_params", "keypoint_params", "additional_targets", "seed", "save_applied_params"}
    conflicting = sorted(reserved & compose_args.keys())
    if conflicting:
        raise TypeError(
            f"dataset-fixer controls {', '.join(conflicting)} so annotations remain synchronized; "
            "pass ordinary Albumentations Compose arguments only"
        )
    if isinstance(transforms, dict):
        if compose_args:
            raise TypeError("compose arguments cannot be combined with a serialized Albumentations pipeline")
        serialized = transforms
        try:
            A.from_dict(serialized)
        except Exception as exc:
            raise ValueError(f"Invalid serialized Albumentations pipeline: {exc}") from exc
        _reject_target_processors(serialized)
        return to_jsonable(serialized)
    try:
        if isinstance(transforms, A.Compose):
            if compose_args:
                raise TypeError("compose arguments cannot be combined with an existing Compose pipeline")
            pipeline = transforms
        elif isinstance(transforms, (A.BasicTransform, A.core.composition.BaseCompose)):
            pipeline = A.Compose([transforms], **compose_args)
        else:
            if not isinstance(transforms, Sequence) or isinstance(transforms, (str, bytes)):
                raise TypeError("transforms must be an Albumentations Compose, serialized dict, or transform sequence")
            pipeline = A.Compose(list(transforms), **compose_args)
        serialized = A.to_dict(pipeline)
        # Rebuild now so Lambda and other non-serializable transforms fail before export.
        A.from_dict(serialized)
    except TypeError:
        raise
    except Exception as exc:
        raise DatasetValidationError(
            ValidationIssue(
                "Albumentations pipeline is not reproducibly serializable",
                value=repr(transforms),
                expected="a pipeline supported by albumentations.to_dict()/from_dict()",
                suggestion="replace Lambda/custom transforms with serializable Albumentations transforms",
            )
        ) from exc
    _reject_target_processors(serialized)
    return to_jsonable(serialized)


def augment_dataset(
    dataset: "Dataset",
    *,
    pipeline: dict[str, Any],
    destination: str | Path | None,
    name: str | None,
    splits: Iterable[str] | None,
    copies: int,
    include_original: bool,
    min_area: float,
    min_visibility: float,
    allow_lossy: bool,
    seed: int,
    visualize: bool,
    progress: bool,
    dry_run: bool,
    validate_output: bool = True,
) -> "Dataset":
    if copies < 1:
        raise ValueError("copies must be at least 1")
    if min_area < 0:
        raise ValueError("min_area must be non-negative")
    if not 0 <= min_visibility <= 1:
        raise ValueError("min_visibility must be in [0, 1]")
    selected = {normalize_split(split) for split in splits} if splits else {"train"}
    missing = selected - set(dataset.splits)
    if missing:
        raise ValueError(f"Unknown augmentation splits {sorted(missing)}; available splits are {dataset.splits}")
    selected_samples = [sample for sample in dataset._samples if sample.split in selected]
    if not selected_samples:
        raise DatasetValidationError("No images selected for augmentation")
    settings = {
        "pipeline": pipeline,
        "splits": sorted(selected),
        "copies": copies,
        "include_original": include_original,
        "min_area": min_area,
        "min_visibility": min_visibility,
        "allow_lossy": allow_lossy,
        "seed": seed,
        "visualize": visualize,
    }
    builder = _builder(dataset, destination, name, "augment", settings)
    dropped = 0
    lossy = 0
    generated = 0
    before_counts = Counter(
        annotation.class_id for sample in dataset._samples for annotation in sample.annotations
    )
    before_background = sum(not sample.annotations for sample in dataset._samples)
    after_counts: Counter[int] = Counter()
    after_background = 0
    try:
        _assert_output_names(dataset._samples, selected, copies, include_original)
        if visualize:
            preview_source = next((sample for sample in selected_samples if sample.annotations), selected_samples[0])
            preview_seed = _sample_seed(seed, preview_source, 1)
            image, annotations, _, _ = _apply(
                preview_source,
                dataset.task,
                pipeline,
                preview_seed,
                flip_idx=dataset._metadata.flip_idx,
                min_area=min_area,
                min_visibility=min_visibility,
                allow_lossy=allow_lossy,
            )
            preview = _save_preview(
                preview_source,
                image,
                annotations,
                dataset,
                builder.reports_dir / "augmentation_preview.jpg",
            )
            builder.visuals.append(str(preview.relative_to(builder.staging)))
            print(f"Augmentation sanity preview: {preview}")
        total = len(dataset._samples) + len(selected_samples) * copies
        if not include_original:
            total -= len(selected_samples)
        _print_start(builder, dataset._samples, settings)
        print(f"Estimated work: {total} output images ({len(selected_samples) * copies} augmented)")
        if dry_run:
            builder.cleanup()
            return dataset
        with tqdm(
            total=total,
            desc="Applying Albumentations",
            unit="output image",
            disable=not progress,
        ) as output_progress:
            for sample in dataset._samples:
                if sample.split not in selected:
                    builder.add_copy(sample, split=sample.split)
                    after_counts.update(annotation.class_id for annotation in sample.annotations)
                    after_background += not sample.annotations
                    output_progress.update()
                    continue
                if include_original:
                    builder.add_copy(
                        sample,
                        split=sample.split,
                        provenance={"augmentation_index": 0, "augmentation": "original"},
                    )
                    after_counts.update(annotation.class_id for annotation in sample.annotations)
                    after_background += not sample.annotations
                    output_progress.update()
                for augmentation_index in range(1, copies + 1):
                    image, annotations, applied, warnings = _apply(
                        sample,
                        dataset.task,
                        pipeline,
                        _sample_seed(seed, sample, augmentation_index),
                        flip_idx=dataset._metadata.flip_idx,
                        min_area=min_area,
                        min_visibility=min_visibility,
                        allow_lossy=allow_lossy,
                    )
                    dropped += len(sample.annotations) - len(annotations)
                    lossy += len(warnings)
                    builder.warnings.extend(warnings)
                    relative_path = _augmented_path(sample.relative_path, augmentation_index)
                    builder.add_image(
                        sample,
                        Image.fromarray(image),
                        split=sample.split,
                        relative_path=relative_path,
                        annotations=annotations,
                        provenance={
                            "augmentation_index": augmentation_index,
                            "augmentation_seed": _sample_seed(seed, sample, augmentation_index),
                            "albumentations_applied": applied,
                        },
                    )
                    after_counts.update(annotation.class_id for annotation in annotations)
                    after_background += not annotations
                    generated += 1
                    output_progress.update()
        report = {
            "generated_images": generated,
            "dropped_annotations": dropped,
            "lossy_annotations": lossy,
            "settings": settings,
        }
        (builder.reports_dir / "augmentation.json").write_text(
            json.dumps(to_jsonable(report), indent=2, sort_keys=True), encoding="utf-8"
        )
        count_report = {
            "definition": "class values count annotations; background counts images with no annotations",
            "before": {
                **{
                    str(class_id): before_counts.get(class_id, 0)
                    for class_id in sorted(dataset._metadata.names)
                },
                "background": before_background,
            },
            "after": {
                **{
                    str(class_id): after_counts.get(class_id, 0)
                    for class_id in sorted(dataset._metadata.names)
                },
                "background": after_background,
            },
            "names": {
                **{
                    str(class_id): class_name
                    for class_id, class_name in sorted(dataset._metadata.names.items())
                },
                "background": "background",
            },
        }
        (builder.reports_dir / "augmentation_class_counts.json").write_text(
            json.dumps(count_report, indent=2, sort_keys=True), encoding="utf-8"
        )
        if visualize:
            before_named = {
                class_name: before_counts.get(class_id, 0)
                for class_id, class_name in sorted(dataset._metadata.names.items())
            }
            before_named["background"] = before_background
            after_named = {
                class_name: after_counts.get(class_id, 0)
                for class_id, class_name in sorted(dataset._metadata.names.items())
            }
            after_named["background"] = after_background
            chart = save_class_count_summary(
                before_named,
                after_named,
                builder.reports_dir / "augmentation_class_counts.jpg",
                title="Class counts before and after augmentation",
            )
            builder.visuals.append(str(chart.relative_to(builder.staging)))
        return _publish(builder, progress=progress, validate_output=validate_output)
    except Exception:
        builder.cleanup()
        raise


def _apply(
    sample: Sample,
    task: Task,
    serialized: dict[str, Any],
    seed: int,
    *,
    flip_idx: list[int] | None,
    min_area: float,
    min_visibility: float,
    allow_lossy: bool,
) -> tuple[np.ndarray, list[Annotation], Any, list[str]]:
    A = _albumentations()
    with Image.open(sample.image_path) as opened:
        source_image = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
    transform = _target_pipeline(A, serialized, seed, min_area, min_visibility)
    bboxes: list[tuple[float, float, float, float]] = []
    bbox_ids: list[int] = []
    keypoints: list[tuple[float, float]] = []
    keypoint_ids: list[int] = []
    masks: list[np.ndarray] = []

    if task in {Task.DETECT, Task.POSE}:
        for index, annotation in enumerate(sample.annotations):
            if annotation.bbox is None:
                continue
            bboxes.append(annotation.bbox)
            bbox_ids.append(index)
    if task is Task.POSE:
        for annotation_index, annotation in enumerate(sample.annotations):
            for keypoint_index, (x, y, _) in enumerate(annotation.keypoints or []):
                keypoints.append((x, y))
                keypoint_ids.append(_keypoint_tag(annotation_index, keypoint_index))
    elif task is Task.POLO:
        for annotation_index, annotation in enumerate(sample.annotations):
            if annotation.point is None:
                continue
            keypoints.append(annotation.point)
            keypoint_ids.append(_keypoint_tag(annotation_index, 0))
            masks.append(_polo_mask(sample, annotation))
    elif task is Task.SEGMENT:
        masks = [_polygon_mask(sample, annotation) for annotation in sample.annotations]

    inputs: dict[str, Any] = {
        "image": source_image,
        "bboxes": np.asarray(bboxes, dtype=np.float32).reshape(-1, 4),
        "bbox_ids": np.asarray(bbox_ids, dtype=np.int64),
        "keypoints": np.asarray(keypoints, dtype=np.float32).reshape(-1, 2),
        "keypoint_ids": np.asarray(keypoint_ids, dtype=np.int64),
    }
    if masks:
        inputs["masks"] = np.stack(masks)
    try:
        result = transform(**inputs)
    except Exception as exc:
        raise DatasetValidationError(
            ValidationIssue(
                "Albumentations failed for an image",
                source=str(sample.image_path),
                value=str(exc),
                suggestion="check that every selected transform supports this dataset's annotation targets",
            )
        ) from exc
    image = np.asarray(result["image"])
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise DatasetValidationError(
            ValidationIssue(
                "Albumentations produced a non-exportable image array",
                source=str(sample.image_path),
                value={"shape": list(image.shape), "dtype": str(image.dtype)},
                expected="an HWC RGB uint8 image",
                suggestion="remove Normalize, ToTensor, or transforms that change the exported image type",
            )
        )
    height, width = image.shape[:2]
    warnings: list[str] = []
    applied = result.get("applied_transforms", [])
    if task is Task.DETECT:
        annotations = _boxes_after(sample, result, width, height)
    elif task is Task.POSE:
        annotations = _pose_after(
            sample,
            result,
            width,
            height,
            flip_idx=flip_idx,
            horizontal_flip=_horizontal_flip_applied(applied),
        )
    elif task is Task.SEGMENT:
        annotations, warnings = _segments_after(sample, result, allow_lossy)
    else:
        annotations, warnings = _polo_after(sample, result, width, height, allow_lossy)
    return image, annotations, applied, warnings


def _target_pipeline(A: Any, serialized: dict[str, Any], seed: int, min_area: float, min_visibility: float):
    base = A.from_dict(serialized)
    kwargs: dict[str, Any] = {
        "transforms": list(base.transforms),
        "bbox_params": A.BboxParams(
            format="pascal_voc",
            label_fields=["bbox_ids"],
            min_area=min_area,
            min_visibility=min_visibility,
        ),
        "keypoint_params": A.KeypointParams(
            format="xy", label_fields=["keypoint_ids"], remove_invisible=False
        ),
        "p": float(getattr(base, "p", 1.0)),
        "is_check_shapes": bool(getattr(base, "is_check_shapes", True)),
        "seed": seed,
        "save_applied_params": True,
    }
    signature = inspect.signature(A.Compose)
    if "strict" in signature.parameters:
        kwargs["strict"] = bool(getattr(base, "strict", False))
    if "mask_interpolation" in signature.parameters and getattr(base, "mask_interpolation", None) is not None:
        kwargs["mask_interpolation"] = base.mask_interpolation
    if "telemetry" in signature.parameters:
        kwargs["telemetry"] = False
    return A.Compose(**kwargs)


def _boxes_after(sample: Sample, result: dict[str, Any], width: int, height: int) -> list[Annotation]:
    output: list[Annotation] = []
    for bbox, annotation_index in zip(result["bboxes"], result["bbox_ids"]):
        clipped = _clip_bbox(bbox, width, height)
        if clipped is not None:
            output.append(sample.annotations[int(annotation_index)].clone(bbox=clipped))
    return output


def _pose_after(
    sample: Sample,
    result: dict[str, Any],
    width: int,
    height: int,
    *,
    flip_idx: list[int] | None,
    horizontal_flip: bool,
) -> list[Annotation]:
    transformed_points = {
        int(tag): (float(point[0]), float(point[1]))
        for point, tag in zip(result["keypoints"], result["keypoint_ids"])
    }
    output: list[Annotation] = []
    for bbox, annotation_index_value in zip(result["bboxes"], result["bbox_ids"]):
        annotation_index = int(annotation_index_value)
        annotation = sample.annotations[annotation_index]
        points: list[tuple[float, float, float | None]] = []
        visible = 0
        for keypoint_index, (_, _, visibility) in enumerate(annotation.keypoints or []):
            x, y = transformed_points.get(_keypoint_tag(annotation_index, keypoint_index), (0.0, 0.0))
            if visibility != 0 and 0 <= x < width and 0 <= y < height:
                points.append((x, y, visibility))
                visible += 1
            else:
                points.append((0.0, 0.0, 0.0))
        if horizontal_flip and flip_idx is not None:
            if len(flip_idx) != len(points):
                raise DatasetValidationError(
                    f"flip_idx has {len(flip_idx)} entries but pose annotation has {len(points)} keypoints"
                )
            points = [points[index] for index in flip_idx]
        clipped = _clip_bbox(bbox, width, height)
        if clipped is not None and visible:
            output.append(annotation.clone(bbox=clipped, keypoints=points))
    return output


def _horizontal_flip_applied(applied: Any) -> bool:
    names = []
    for item in applied or []:
        if isinstance(item, (list, tuple)) and item:
            names.append(str(item[0]).rsplit(".", 1)[-1])
        elif isinstance(item, dict):
            names.append(str(item.get("__class_fullname__") or item.get("name") or "").rsplit(".", 1)[-1])
    return names.count("HorizontalFlip") % 2 == 1


def _segments_after(
    sample: Sample, result: dict[str, Any], allow_lossy: bool
) -> tuple[list[Annotation], list[str]]:
    import cv2

    output: list[Annotation] = []
    warnings: list[str] = []
    for annotation, transformed_mask in zip(sample.annotations, result.get("masks", [])):
        mask = (np.asarray(transformed_mask) > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if cv2.contourArea(contour) > 0]
        if not contours:
            continue
        if len(contours) > 1 and not allow_lossy:
            raise DatasetValidationError(
                ValidationIssue(
                    "Augmentation produced a disconnected instance mask",
                    source=str(sample.image_path),
                    value=annotation.source_id,
                    expected="one YOLO-representable polygon",
                    suggestion="use transforms that preserve connectivity or set allow_lossy=True",
                )
            )
        contour = max(contours, key=cv2.contourArea)
        if len(contours) > 1:
            warnings.append(f"Kept the largest transformed polygon for {sample.image_path}:{annotation.source_id}")
        polygon = [(float(point[0][0]), float(point[0][1])) for point in contour]
        if len(polygon) < 3:
            continue
        xs, ys = zip(*polygon)
        output.append(annotation.clone(polygon=polygon, bbox=(min(xs), min(ys), max(xs), max(ys))))
    return output, warnings


def _polo_after(
    sample: Sample,
    result: dict[str, Any],
    width: int,
    height: int,
    allow_lossy: bool,
) -> tuple[list[Annotation], list[str]]:
    import cv2

    centers = {
        int(tag): (float(point[0]), float(point[1]))
        for point, tag in zip(result["keypoints"], result["keypoint_ids"])
    }
    output: list[Annotation] = []
    warnings: list[str] = []
    for index, (annotation, transformed_mask) in enumerate(zip(sample.annotations, result.get("masks", []))):
        center = centers.get(_keypoint_tag(index, 0))
        if center is None or not (0 <= center[0] < width and 0 <= center[1] < height):
            continue
        mask = (np.asarray(transformed_mask) > 0).astype(np.uint8)
        if not mask.any() or mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        boundary = max(contours, key=cv2.contourArea).reshape(-1, 2)
        distances = np.sqrt((boundary[:, 0] - center[0]) ** 2 + (boundary[:, 1] - center[1]) ** 2)
        radius = float(np.median(distances))
        distortion = float(np.percentile(distances, 90) - np.percentile(distances, 10))
        tolerance = max(2.5, radius * 0.15)
        if distortion > tolerance:
            if not allow_lossy:
                raise DatasetValidationError(
                    ValidationIssue(
                        "Albumentations transformed a POLO circle into a non-circular shape",
                        source=str(sample.image_path),
                        value={"annotation": annotation.source_id, "radial_spread": distortion},
                        expected="a scalar-radius-representable point",
                        suggestion="use isotropic geometry or set allow_lossy=True",
                    )
                )
            radius = float(distances.max())
            warnings.append(
                f"Used an enclosing radius for distorted POLO point {sample.image_path}:{annotation.source_id}"
            )
        if radius > 0:
            output.append(annotation.clone(point=center, radius=radius))
    return output, warnings


def _polygon_mask(sample: Sample, annotation: Annotation) -> np.ndarray:
    if not annotation.polygon:
        raise DatasetValidationError(f"Segmentation annotation {annotation.source_id} has no polygon")
    mask = Image.new("L", (sample.width, sample.height), 0)
    ImageDraw.Draw(mask).polygon(annotation.polygon, fill=1)
    return np.asarray(mask, dtype=np.uint8)


def _polo_mask(sample: Sample, annotation: Annotation) -> np.ndarray:
    if annotation.point is None or annotation.radius is None:
        raise DatasetValidationError(f"POLO annotation {annotation.source_id} has no point/radius")
    x, y = annotation.point
    radius = annotation.radius
    mask = Image.new("L", (sample.width, sample.height), 0)
    ImageDraw.Draw(mask).ellipse((x - radius, y - radius, x + radius, y + radius), fill=1)
    return np.asarray(mask, dtype=np.uint8)


def _clip_bbox(values: Any, width: int, height: int) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = map(float, values[:4])
    clipped = (max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2))
    return clipped if clipped[2] > clipped[0] and clipped[3] > clipped[1] else None


def _keypoint_tag(annotation_index: int, keypoint_index: int) -> int:
    return annotation_index * 1_000_000 + keypoint_index


def _sample_seed(seed: int, sample: Sample, augmentation_index: int) -> int:
    value = f"{seed}:{sample.split}:{sample.relative_path.as_posix()}:{augmentation_index}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big")


def _augmented_path(path: Path, augmentation_index: int) -> Path:
    return path.with_name(f"{path.stem}__aug-{augmentation_index:03d}{path.suffix.lower()}")


def _assert_output_names(samples: list[Sample], selected: set[str], copies: int, include_original: bool) -> None:
    paths: set[tuple[str, str]] = set()
    for sample in samples:
        if sample.split not in selected or include_original:
            candidates = [sample.relative_path]
        else:
            candidates = []
        if sample.split in selected:
            candidates.extend(_augmented_path(sample.relative_path, index) for index in range(1, copies + 1))
        for candidate in candidates:
            key = sample.split, candidate.as_posix().casefold()
            if key in paths:
                raise DatasetValidationError(f"Augmentation output filename collision: {sample.split}/{candidate}")
            paths.add(key)


def _save_preview(
    source: Sample,
    augmented_image: np.ndarray,
    annotations: list[Annotation],
    dataset: "Dataset",
    output: Path,
) -> Path:
    from .visualization import _draw_sample
    import matplotlib.pyplot as plt

    temporary = output.with_name(".augmentation-preview-image.png")
    Image.fromarray(augmented_image).save(temporary)
    augmented = Sample(
        temporary,
        source.relative_path,
        source.split,
        int(augmented_image.shape[1]),
        int(augmented_image.shape[0]),
        annotations,
    )
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        _draw_sample(axes[0], source, dataset.task, dataset._metadata)
        axes[0].set_title("Before")
        _draw_sample(axes[1], augmented, dataset.task, dataset._metadata)
        axes[1].set_title("Albumentations preview")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight", dpi=160)
        plt.close(fig)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _albumentations():
    try:
        import albumentations as A
    except ImportError as exc:
        raise ImportError(
            "Dataset.augment() requires Albumentations; install it with "
            "`pip install 'dataset-fixer[augment]'`"
        ) from exc
    return A


def _reject_target_processors(serialized: dict[str, Any]) -> None:
    transform = serialized.get("transform") if isinstance(serialized, dict) else None
    if not isinstance(transform, dict):
        return
    if str(transform.get("__class_fullname__", "")).rsplit(".", 1)[-1] != "Compose":
        raise TypeError("The serialized augmentation root must be an Albumentations Compose pipeline")
    configured = [
        name
        for name in ("bbox_params", "keypoint_params", "additional_targets")
        if transform.get(name) not in (None, {}, [])
    ]
    if configured:
        raise TypeError(
            "An existing Compose pipeline may not configure "
            f"{', '.join(configured)}; dataset-fixer installs task-aware processors automatically"
        )
