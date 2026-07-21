from __future__ import annotations

import hashlib
import math
from pathlib import Path

from shapely.geometry import Polygon
from tqdm.auto import tqdm

from .errors import DatasetValidationError, ValidationIssue
from .models import DatasetMetadata, Sample, Task


def validate_dataset(
    samples: list[Sample], metadata: DatasetMetadata, task: Task, *, deep: bool = False, progress: bool = False
) -> list[str]:
    issues: list[ValidationIssue] = []
    warnings: list[str] = []
    if not samples:
        issues.append(ValidationIssue("Dataset contains no images"))
    if sorted(metadata.names) != list(range(len(metadata.names))):
        issues.append(
            ValidationIssue(
                "Class IDs must be contiguous and zero-based",
                value=sorted(metadata.names),
                expected=f"0..{max(0, len(metadata.names) - 1)}",
            )
        )
    duplicate_names = sorted({name for name in metadata.names.values() if list(metadata.names.values()).count(name) > 1})
    if duplicate_names:
        issues.append(
            ValidationIssue(
                "Class names must be unique",
                value=duplicate_names,
                suggestion="rename duplicate classes before loading",
            )
        )
    if task is Task.POSE:
        if metadata.kpt_shape is None:
            issues.append(ValidationIssue("Pose dataset is missing kpt_shape metadata"))
        else:
            nkpt, ndim = metadata.kpt_shape
            if nkpt <= 0 or ndim not in {2, 3}:
                issues.append(ValidationIssue("Invalid kpt_shape", value=metadata.kpt_shape, expected="[N, 2] or [N, 3]"))
            if metadata.flip_idx is not None and len(metadata.flip_idx) != nkpt:
                issues.append(ValidationIssue("flip_idx length does not match kpt_shape", value=metadata.flip_idx))
            elif metadata.flip_idx is not None and sorted(metadata.flip_idx) != list(range(nkpt)):
                issues.append(ValidationIssue("flip_idx must be a permutation of keypoint indices", value=metadata.flip_idx))
            for class_id, keypoint_names in metadata.kpt_names.items():
                if class_id not in metadata.names or len(keypoint_names) != nkpt:
                    issues.append(
                        ValidationIssue(
                            "kpt_names must contain one name per keypoint for a valid class",
                            value={class_id: keypoint_names},
                        )
                    )
            if metadata.kpt_oks_sigmas is not None and (
                len(metadata.kpt_oks_sigmas) != nkpt or any(v <= 0 for v in metadata.kpt_oks_sigmas)
            ):
                issues.append(ValidationIssue("kpt_oks_sigmas must contain one positive value per keypoint"))
    if task is Task.POLO:
        missing_radii = sorted(set(metadata.names) - set(metadata.radii))
        if missing_radii:
            issues.append(
                ValidationIssue(
                    "POLO dataset is missing class-level radii",
                    value=missing_radii,
                    expected="one positive radius per class in data.yaml",
                )
            )
        for class_id, radius in metadata.radii.items():
            if class_id not in metadata.names or radius <= 0 or not math.isfinite(radius):
                issues.append(ValidationIssue("Invalid POLO class radius", value={class_id: radius}))

    seen_paths: dict[Path, str] = {}
    seen_hashes: dict[str, tuple[Path, str]] = {}
    iterator = tqdm(samples, desc="Validating dataset", unit="image", disable=not progress)
    for sample in iterator:
        resolved = sample.image_path.resolve()
        if resolved in seen_paths:
            message = (
                "Same image appears in multiple splits"
                if seen_paths[resolved] != sample.split
                else "Same image is listed more than once in a split"
            )
            issues.append(ValidationIssue(message, source=str(resolved)))
        seen_paths[resolved] = sample.split
        if sample.width <= 0 or sample.height <= 0:
            issues.append(ValidationIssue("Image dimensions must be positive", source=str(resolved)))
        if deep:
            digest = _hash_file(resolved)
            if digest in seen_hashes and seen_hashes[digest][1] != sample.split:
                issues.append(
                    ValidationIssue(
                        "Byte-identical images appear in multiple splits",
                        source=str(resolved),
                        suggestion=f"also present at {seen_hashes[digest][0]}",
                    )
                )
            seen_hashes[digest] = (resolved, sample.split)
        for annotation in sample.annotations:
            annotation_source = (
                f"{resolved} [annotation {annotation.source_id}]" if annotation.source_id is not None else str(resolved)
            )
            if task is Task.DETECT and annotation.bbox is None:
                issues.append(ValidationIssue("Detection annotation is missing a bounding box", source=annotation_source))
            elif task is Task.SEGMENT and annotation.polygon is None and annotation.rle is None:
                issues.append(ValidationIssue("Segmentation annotation is missing mask geometry", source=annotation_source))
            elif task is Task.POSE and (annotation.bbox is None or annotation.keypoints is None):
                issues.append(ValidationIssue("Pose annotation requires a box and keypoints", source=annotation_source))
            elif task is Task.POLO and annotation.point is None:
                issues.append(ValidationIssue("POLO annotation requires a point and radius", source=annotation_source))
            if annotation.class_id not in metadata.names:
                issues.append(
                    ValidationIssue(
                        "Annotation class is missing from names",
                        source=annotation_source,
                        value=annotation.class_id,
                        expected=f"one of {sorted(metadata.names)}",
                    )
                )
            if annotation.bbox is not None:
                x1, y1, x2, y2 = annotation.bbox
                if not all(math.isfinite(v) for v in annotation.bbox) or x2 <= x1 or y2 <= y1:
                    issues.append(ValidationIssue("Invalid bounding box", source=annotation_source, value=annotation.bbox))
                if x1 < -0.01 * sample.width or y1 < -0.01 * sample.height or x2 > 1.01 * sample.width or y2 > 1.01 * sample.height:
                    issues.append(ValidationIssue("Bounding box lies outside normalized bounds", source=annotation_source, value=annotation.bbox))
            if annotation.polygon is not None:
                if len(annotation.polygon) < 3:
                    issues.append(ValidationIssue("Polygon needs at least three points", source=annotation_source))
                else:
                    polygon = Polygon(annotation.polygon)
                    if polygon.is_empty or polygon.area <= 0 or not polygon.is_valid:
                        issues.append(ValidationIssue("Invalid or self-intersecting polygon", source=annotation_source))
                    if any(
                        x < -0.01 * sample.width
                        or y < -0.01 * sample.height
                        or x > 1.01 * sample.width
                        or y > 1.01 * sample.height
                        for x, y in annotation.polygon
                    ):
                        issues.append(ValidationIssue("Polygon lies outside normalized image bounds", source=annotation_source))
            if annotation.keypoints is not None:
                if metadata.kpt_shape and len(annotation.keypoints) != metadata.kpt_shape[0]:
                    issues.append(ValidationIssue("Keypoint count does not match kpt_shape", source=annotation_source))
                for x, y, visibility in annotation.keypoints:
                    if not all(math.isfinite(v) for v in (x, y)):
                        issues.append(ValidationIssue("Non-finite keypoint", source=annotation_source))
                    if visibility is not None and visibility not in {0, 1, 2, 0.0, 1.0, 2.0}:
                        issues.append(ValidationIssue("Keypoint visibility must be 0, 1, or 2", source=annotation_source, value=visibility))
                    if visibility != 0 and not (0 <= x <= sample.width and 0 <= y <= sample.height):
                        issues.append(ValidationIssue("Visible keypoint lies outside image bounds", source=annotation_source, value=(x, y)))
            if annotation.point is not None:
                x, y = annotation.point
                if not (0 <= x <= sample.width and 0 <= y <= sample.height):
                    issues.append(ValidationIssue("POLO point lies outside image", source=annotation_source, value=annotation.point))
                if annotation.radius is None or annotation.radius <= 0 or not math.isfinite(annotation.radius):
                    issues.append(ValidationIssue("POLO radius must be positive and finite", source=annotation_source, value=annotation.radius))
            if annotation.rle is not None:
                warnings.append(f"{resolved}: segmentation requires explicit allow_lossy=True for YOLO export")
    if issues:
        raise DatasetValidationError(issues)
    return warnings


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
