from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import yaml
from PIL import Image, ImageOps
from shapely.geometry import Polygon
from tqdm.auto import tqdm

from .errors import DatasetValidationError, ValidationIssue
from .io import _label_path_for_image, _parse_yolo_line
from .models import Annotation, DatasetMetadata, Sample, Task
from .validation_audit import ValidationFailureExample, add_failure_example


MAX_STAGED_VALIDATION_ISSUES = 100


def validate_staged_yolo_output(
    root: Path,
    samples: list[Sample],
    records: list[dict[str, Any]],
    metadata: DatasetMetadata,
    task: Task,
    manifest: dict[str, Any],
    *,
    progress: bool = True,
) -> bool:
    """Validate a staged YOLO tree without building a second dataset index.

    The in-memory output index is validated first. Image files and labels are
    then reopened one at a time, provenance JSONL is parsed one line at a time,
    and orphan labels are checked from compact relative-path keys. Peak memory
    is therefore bounded by the existing output index rather than another full
    copy of samples, annotations, and provenance.
    """

    root = root.resolve()
    validate_dataset(samples, metadata, task, progress=False)
    issues: list[ValidationIssue] = []
    omitted_issues = 0

    def add_issue(issue: ValidationIssue) -> None:
        nonlocal omitted_issues
        if len(issues) < MAX_STAGED_VALIDATION_ISSUES:
            issues.append(issue)
        else:
            omitted_issues += 1

    _validate_staged_metadata_files(root, samples, metadata, task, manifest, add_issue)

    expected_labels: set[str] = set()
    iterator = tqdm(
        enumerate(samples),
        total=len(samples),
        desc="Streaming staged validation",
        unit="image",
        disable=not progress,
    )
    for index, sample in iterator:
        record = records[index] if index < len(records) else {}
        expected_image = root / sample.split / "images" / sample.relative_path
        if sample.image_path.resolve() != expected_image.resolve():
            add_issue(
                ValidationIssue(
                    "Staged sample points outside its canonical output path",
                    source=str(sample.image_path),
                    expected=str(expected_image),
                )
            )
        actual_width: int | None = None
        actual_height: int | None = None
        try:
            with Image.open(expected_image) as opened:
                oriented = ImageOps.exif_transpose(opened)
                actual_width, actual_height = oriented.size
                oriented.load()
        except Exception as error:
            add_issue(
                ValidationIssue(
                    f"Unreadable staged image: {error}",
                    source=str(expected_image),
                )
            )
        else:
            if (actual_width, actual_height) != (sample.width, sample.height):
                add_issue(
                    ValidationIssue(
                        "Staged image dimensions differ from the output index",
                        source=str(expected_image),
                        value=(actual_width, actual_height),
                        expected=str((sample.width, sample.height)),
                    )
                )

        label_path = _label_path_for_image(expected_image, sample.relative_path)
        expected_labels.add(label_path.relative_to(root).as_posix())
        annotation_count = 0
        if not label_path.is_file():
            add_issue(
                ValidationIssue(
                    "Staged image is missing its label file",
                    source=str(label_path),
                )
            )
        else:
            try:
                actual_sample = Sample(
                    expected_image,
                    sample.relative_path,
                    sample.split,
                    actual_width or sample.width,
                    actual_height or sample.height,
                )
                with label_path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            continue
                        annotation_count += 1
                        try:
                            annotation = _parse_yolo_line(
                                line,
                                task,
                                metadata,
                                actual_width or sample.width,
                                actual_height or sample.height,
                            )
                        except Exception as error:
                            add_issue(
                                ValidationIssue(
                                    str(error),
                                    source=str(label_path),
                                    line=line_number,
                                    value=line.strip(),
                                )
                            )
                            continue
                        for issue in _validate_annotation(
                            annotation,
                            actual_sample,
                            metadata,
                            task,
                            f"{label_path}:{line_number}",
                        ):
                            add_issue(issue)
            except OSError as error:
                add_issue(
                    ValidationIssue(
                        f"Unreadable staged label file: {error}",
                        source=str(label_path),
                    )
                )
        expected_count = len(sample.annotations)
        recorded_count = record.get("output_annotation_count")
        if annotation_count != expected_count or recorded_count != expected_count:
            add_issue(
                ValidationIssue(
                    "Staged label count differs from output metadata",
                    source=str(label_path),
                    value={"label": annotation_count, "provenance": recorded_count},
                    expected=str(expected_count),
                )
            )

    if len(samples) != len(records):
        add_issue(
            ValidationIssue(
                "Output sample and provenance-record counts differ",
                value={"samples": len(samples), "records": len(records)},
                expected="one provenance record per output image",
            )
        )

    for label_path in root.rglob("*.txt"):
        if "labels" not in label_path.parts or label_path.name in {"train.txt", "val.txt", "test.txt"}:
            continue
        relative = label_path.relative_to(root).as_posix()
        if relative not in expected_labels:
            add_issue(
                ValidationIssue(
                    "Label has no image in the configured dataset splits",
                    source=str(label_path),
                    suggestion="add the image to data.yaml, move the label, or remove the orphan",
                )
            )

    _validate_staged_provenance(root, records, add_issue)
    if omitted_issues:
        issues.append(
            ValidationIssue(
                f"{omitted_issues} additional staged validation error(s) omitted",
                suggestion="fix the first reported errors and export again",
            )
        )
    if issues:
        raise DatasetValidationError(issues)

    split_counts = Counter(sample.split for sample in samples)
    return bool(
        split_counts["train"]
        and split_counts["val"]
        and metadata.names
        and (task is not Task.POSE or metadata.kpt_shape)
        and (task is not Task.POLO or metadata.radii)
    )


def _validate_staged_metadata_files(
    root: Path,
    samples: list[Sample],
    metadata: DatasetMetadata,
    task: Task,
    manifest: dict[str, Any],
    add_issue,
) -> None:
    yaml_path = root / "data.yaml"
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except Exception as error:
        add_issue(ValidationIssue(f"Unreadable staged data.yaml: {error}", source=str(yaml_path)))
    else:
        configured_root = Path(str(data.get("path") or yaml_path.parent))
        if not configured_root.is_absolute():
            configured_root = yaml_path.parent / configured_root
        if configured_root.resolve() != root:
            add_issue(
                ValidationIssue(
                    "Staged data.yaml points to the wrong dataset root",
                    source=str(yaml_path),
                    value=str(configured_root.resolve()),
                    expected=str(root),
                )
            )
        yaml_names = data.get("names") or {}
        parsed_names = (
            {index: str(value) for index, value in enumerate(yaml_names)}
            if isinstance(yaml_names, list)
            else {int(key): str(value) for key, value in yaml_names.items()}
        )
        if parsed_names != metadata.names:
            add_issue(
                ValidationIssue(
                    "Staged data.yaml class names differ from output metadata",
                    source=str(yaml_path),
                    value=parsed_names,
                    expected=str(metadata.names),
                )
            )
        available = {sample.split for sample in samples}
        for split in ("train", "val", "test"):
            expected = f"{split}/images" if split in available else None
            if data.get(split) != expected:
                add_issue(
                    ValidationIssue(
                        "Staged data.yaml split entry is incorrect",
                        source=str(yaml_path),
                        value={split: data.get(split)},
                        expected=str({split: expected}),
                    )
                )
        if task is Task.POSE and not data.get("kpt_shape"):
            add_issue(ValidationIssue("Staged pose data.yaml is missing kpt_shape", source=str(yaml_path)))
        if task is Task.POLO and not data.get("radii"):
            add_issue(ValidationIssue("Staged POLO data.yaml is missing radii", source=str(yaml_path)))

    manifest_path = root / "dataset-fixer.json"
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        add_issue(
            ValidationIssue(
                "Staged dataset-fixer manifest is missing or empty",
                source=str(manifest_path),
            )
        )
    if manifest.get("format") != "yolo" or manifest.get("task") != task.value:
        add_issue(
            ValidationIssue(
                "Staged manifest format or task is incorrect",
                source=str(manifest_path),
                value={"format": manifest.get("format"), "task": manifest.get("task")},
                expected=str({"format": "yolo", "task": task.value}),
            )
        )


def _validate_staged_provenance(root: Path, records: list[dict[str, Any]], add_issue) -> None:
    provenance_path = root / "provenance.jsonl"
    record_count = 0
    try:
        with provenance_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record_count += 1
                try:
                    record = json.loads(line)
                    output_image = str(Path(record["output_image"]))
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    add_issue(
                        ValidationIssue(
                            "Invalid provenance record",
                            source=str(provenance_path),
                            line=line_number,
                            value=str(error),
                        )
                    )
                    continue
                if record_count > len(records):
                    add_issue(
                        ValidationIssue(
                            "Unexpected extra provenance record",
                            source=str(provenance_path),
                            line=line_number,
                            value=output_image,
                        )
                    )
                    continue
                expected = str(Path(records[record_count - 1]["output_image"]))
                if output_image != expected:
                    add_issue(
                        ValidationIssue(
                            "Provenance output path differs from the staged image index",
                            source=str(provenance_path),
                            line=line_number,
                            value=output_image,
                            expected=expected,
                        )
                    )
    except OSError as error:
        add_issue(
            ValidationIssue(
                f"Unreadable staged provenance: {error}",
                source=str(provenance_path),
            )
        )
        return
    if record_count != len(records):
        add_issue(
            ValidationIssue(
                "Staged provenance count differs from output image count",
                source=str(provenance_path),
                value=record_count,
                expected=str(len(records)),
            )
        )


def validate_dataset(
    samples: list[Sample],
    metadata: DatasetMetadata,
    task: Task,
    *,
    deep: bool = False,
    progress: bool = False,
    errors: Literal["raise", "skip"] = "raise",
    failure_examples: list[ValidationFailureExample] | None = None,
) -> list[str]:
    if errors not in {"raise", "skip"}:
        raise ValueError("errors must be 'raise' or 'skip'")
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
    valid_samples: list[Sample] | None = [] if errors == "skip" else None
    iterator = tqdm(samples, desc="Validating dataset", unit="image", disable=not progress)
    for sample in iterator:
        resolved = sample.image_path.resolve()
        sample_issues: list[ValidationIssue] = []
        digest: str | None = None
        if resolved in seen_paths:
            message = (
                "Same image appears in multiple splits"
                if seen_paths[resolved] != sample.split
                else "Same image is listed more than once in a split"
            )
            sample_issues.append(ValidationIssue(message, source=str(resolved)))
        if sample.width <= 0 or sample.height <= 0:
            sample_issues.append(ValidationIssue("Image dimensions must be positive", source=str(resolved)))
        if deep:
            try:
                digest = _hash_file(resolved)
            except OSError as exc:
                sample_issues.append(ValidationIssue(f"Could not hash image: {exc}", source=str(resolved)))
            else:
                if digest in seen_hashes and seen_hashes[digest][1] != sample.split:
                    sample_issues.append(
                        ValidationIssue(
                            "Byte-identical images appear in multiple splits",
                            source=str(resolved),
                            suggestion=f"also present at {seen_hashes[digest][0]}",
                        )
                    )
        if sample_issues and errors == "skip":
            messages = "; ".join(dict.fromkeys(issue.message for issue in sample_issues))
            details = " | ".join(issue.format() for issue in sample_issues)
            warning = f"Skipped invalid image: {details}"
            warnings.append(warning)
            add_failure_example(
                failure_examples,
                ValidationFailureExample(
                    warning=warning,
                    summary=messages,
                    image_path=sample.image_path,
                    relative_path=sample.relative_path,
                    split=sample.split,
                    width=sample.width,
                    height=sample.height,
                ),
            )
            continue
        seen_paths[resolved] = sample.split
        if digest is not None:
            seen_hashes[digest] = (resolved, sample.split)
        issues.extend(sample_issues)
        valid_annotations = []
        for annotation in sample.annotations:
            annotation_source = (
                f"{resolved} [annotation {annotation.source_id}]" if annotation.source_id is not None else str(resolved)
            )
            annotation_issues = _validate_annotation(annotation, sample, metadata, task, annotation_source)
            if annotation_issues and errors == "skip":
                messages = "; ".join(dict.fromkeys(issue.message for issue in annotation_issues))
                warning = f"Skipped invalid annotation {annotation_source}: {messages}"
                warnings.append(warning)
                add_failure_example(
                    failure_examples,
                    ValidationFailureExample(
                        warning=warning,
                        summary=messages,
                        image_path=sample.image_path,
                        relative_path=sample.relative_path,
                        split=sample.split,
                        width=sample.width,
                        height=sample.height,
                        annotation=annotation.clone(),
                    ),
                )
                continue
            issues.extend(annotation_issues)
            valid_annotations.append(annotation)
            if annotation.rle is not None and not annotation_issues:
                warnings.append(f"{resolved}: segmentation requires explicit allow_lossy=True for YOLO export")
        if errors == "skip":
            sample.annotations = valid_annotations
        if valid_samples is not None:
            valid_samples.append(sample)
    if errors == "skip":
        assert valid_samples is not None
        samples[:] = valid_samples
        if not samples:
            issues.append(ValidationIssue("Dataset contains no valid images after skipping recoverable errors"))
    if issues:
        raise DatasetValidationError(issues)
    return warnings


def _validate_annotation(
    annotation: Annotation,
    sample: Sample,
    metadata: DatasetMetadata,
    task: Task,
    source: str,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if task is Task.DETECT and annotation.bbox is None:
        issues.append(ValidationIssue("Detection annotation is missing a bounding box", source=source))
    elif task is Task.SEGMENT and annotation.polygon is None and annotation.rle is None:
        issues.append(ValidationIssue("Segmentation annotation is missing mask geometry", source=source))
    elif task is Task.POSE and (annotation.bbox is None or annotation.keypoints is None):
        issues.append(ValidationIssue("Pose annotation requires a box and keypoints", source=source))
    elif task is Task.POLO and annotation.point is None:
        issues.append(ValidationIssue("POLO annotation requires a point and radius", source=source))
    if annotation.class_id not in metadata.names:
        issues.append(
            ValidationIssue(
                "Annotation class is missing from names",
                source=source,
                value=annotation.class_id,
                expected=f"one of {sorted(metadata.names)}",
            )
        )
    if annotation.bbox is not None:
        x1, y1, x2, y2 = annotation.bbox
        if not all(math.isfinite(v) for v in annotation.bbox) or x2 <= x1 or y2 <= y1:
            issues.append(ValidationIssue("Invalid bounding box", source=source, value=annotation.bbox))
        if (
            x1 < -0.01 * sample.width
            or y1 < -0.01 * sample.height
            or x2 > 1.01 * sample.width
            or y2 > 1.01 * sample.height
        ):
            issues.append(
                ValidationIssue("Bounding box lies outside normalized bounds", source=source, value=annotation.bbox)
            )
    if annotation.polygon is not None:
        if len(annotation.polygon) < 3:
            issues.append(ValidationIssue("Polygon needs at least three points", source=source))
        else:
            try:
                polygon = Polygon(annotation.polygon)
                invalid_polygon = polygon.is_empty or polygon.area <= 0 or not polygon.is_valid
            except Exception:
                invalid_polygon = True
            if invalid_polygon:
                issues.append(ValidationIssue("Invalid or self-intersecting polygon", source=source))
            if any(
                x < -0.01 * sample.width
                or y < -0.01 * sample.height
                or x > 1.01 * sample.width
                or y > 1.01 * sample.height
                for x, y in annotation.polygon
            ):
                issues.append(ValidationIssue("Polygon lies outside normalized image bounds", source=source))
    if annotation.keypoints is not None:
        if metadata.kpt_shape and len(annotation.keypoints) != metadata.kpt_shape[0]:
            issues.append(ValidationIssue("Keypoint count does not match kpt_shape", source=source))
        for x, y, visibility in annotation.keypoints:
            if not all(math.isfinite(v) for v in (x, y)):
                issues.append(ValidationIssue("Non-finite keypoint", source=source))
            if visibility is not None and visibility not in {0, 1, 2, 0.0, 1.0, 2.0}:
                issues.append(
                    ValidationIssue("Keypoint visibility must be 0, 1, or 2", source=source, value=visibility)
                )
            if visibility != 0 and not (0 <= x <= sample.width and 0 <= y <= sample.height):
                issues.append(ValidationIssue("Visible keypoint lies outside image bounds", source=source, value=(x, y)))
    if annotation.point is not None:
        x, y = annotation.point
        if not (0 <= x <= sample.width and 0 <= y <= sample.height):
            issues.append(ValidationIssue("POLO point lies outside image", source=source, value=annotation.point))
        if annotation.radius is None or annotation.radius <= 0 or not math.isfinite(annotation.radius):
            issues.append(
                ValidationIssue("POLO radius must be positive and finite", source=source, value=annotation.radius)
            )
    return issues


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
