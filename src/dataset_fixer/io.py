from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

import yaml
from PIL import Image, ImageOps
from tqdm.auto import tqdm

from .errors import DatasetValidationError, ValidationIssue
from .artifacts import DATASET_INFO_NAME, dataset_info_path
from .models import Annotation, DatasetMetadata, Sample, Task
from .utils import IMAGE_SUFFIXES, image_files, normalize_split


def load_source(
    location: Path,
    *,
    task: Task | None,
    name: str | None,
    names: dict[int, str] | list[str] | None,
    radii: dict[int, float] | None,
    progress: bool,
    errors: Literal["raise", "skip"] = "raise",
    warnings: list[str] | None = None,
) -> tuple[Path, str, Task, DatasetMetadata, list[Sample], dict[str, Any]]:
    warnings = warnings if warnings is not None else []
    location = location.expanduser().resolve()
    if not location.exists():
        raise FileNotFoundError(f"Dataset does not exist: {location}")

    if location.suffix.lower() == ".json":
        return _load_coco(
            location,
            task=task,
            name=name,
            progress=progress,
            errors=errors,
            warnings=warnings,
        )

    root = location.parent if location.suffix.lower() in {".yaml", ".yml"} else location
    yaml_path = location if location.suffix.lower() in {".yaml", ".yml"} else _find_yaml(root)
    if yaml_path is not None:
        return _load_yolo(
            yaml_path,
            task=task,
            name=name,
            names_override=names,
            radii_override=radii,
            progress=progress,
            errors=errors,
            warnings=warnings,
        )

    coco_files = sorted(root.rglob("*.json"))
    coco_files = [
        p
        for p in coco_files
        if p.name not in {"dataset-fixer.json", DATASET_INFO_NAME, "source.json"}
        and not {".cache", "evaluations", "reports"}.intersection(p.relative_to(root).parts)
    ]
    if coco_files:
        return _load_coco(
            root,
            task=task,
            name=name,
            progress=progress,
            errors=errors,
            warnings=warnings,
        )

    return _load_flat_yolo(
        root,
        task=task,
        name=name,
        names_override=names,
        radii_override=radii,
        progress=progress,
        errors=errors,
        warnings=warnings,
    )


def _find_yaml(root: Path) -> Path | None:
    direct = [root / "data.yaml", root / "dataset.yaml", root / "data.yml"]
    found = [p for p in direct if p.is_file()]
    if found:
        return found[0]
    nested = sorted([*root.glob("*.yaml"), *root.glob("*.yml")])
    return nested[0] if len(nested) == 1 else None


def _parse_names(value: Any) -> dict[int, str]:
    if isinstance(value, list):
        return {i: str(v) for i, v in enumerate(value)}
    if isinstance(value, dict):
        return {int(k): str(v) for k, v in value.items()}
    return {}


def _load_yolo(
    yaml_path: Path,
    *,
    task: Task | None,
    name: str | None,
    names_override: dict[int, str] | list[str] | None,
    radii_override: dict[int, float] | None,
    progress: bool,
    errors: Literal["raise", "skip"],
    warnings: list[str],
) -> tuple[Path, str, Task, DatasetMetadata, list[Sample], dict[str, Any]]:
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    yaml_dir = yaml_path.parent.resolve()
    configured_root = Path(raw.get("path") or yaml_dir)
    root = configured_root if configured_root.is_absolute() else (yaml_dir / configured_root)
    root = root.resolve()

    parsed_names = _parse_names(names_override if names_override is not None else raw.get("names"))
    parsed_radii = {int(k): float(v) for k, v in (radii_override or raw.get("radii") or {}).items()}
    metadata = DatasetMetadata(
        names=parsed_names,
        channels=int(raw.get("channels", 3)),
        radii=parsed_radii,
        kpt_shape=tuple(raw["kpt_shape"]) if raw.get("kpt_shape") else None,
        flip_idx=[int(v) for v in raw["flip_idx"]] if raw.get("flip_idx") is not None else None,
        kpt_names={int(k): list(v) for k, v in (raw.get("kpt_names") or {}).items()},
        kpt_oks_sigmas=[float(v) for v in raw["kpt_oks_sigmas"]] if raw.get("kpt_oks_sigmas") else None,
        extra={
            k: v
            for k, v in raw.items()
            if k not in {"path", "train", "val", "valid", "validation", "test", "names", "nc", "channels", "radii", "kpt_shape", "flip_idx", "kpt_names", "kpt_oks_sigmas", "download"}
        },
    )

    split_images: list[tuple[str, Path, Path]] = []
    for key, value in raw.items():
        if key not in {"train", "val", "valid", "validation", "test"} or value is None or value == "":
            continue
        split = normalize_split(key)
        for image_path, rel in _expand_yolo_split(
            value,
            root=root,
            yaml_dir=yaml_dir,
            errors=errors,
            warnings=warnings,
        ):
            split_images.append((split, image_path, rel))

    if not split_images:
        raise DatasetValidationError(
            ValidationIssue(
                "No images resolved from data.yaml split entries",
                source=str(yaml_path),
                expected="at least one image in train, val, or test",
                suggestion="check path and split paths in data.yaml",
            )
        )

    resolved_task = task or _infer_yolo_task(
        [(image, relative) for _, image, relative in split_images],
        metadata,
        errors=errors,
        warnings=warnings,
    )
    if resolved_task is None:
        raise DatasetValidationError("Could not infer task from empty labels; pass task='detect', 'segment', 'pose', or 'polo'")
    samples = _parse_yolo_images(
        split_images,
        resolved_task,
        metadata,
        progress=progress,
        errors=errors,
        warnings=warnings,
    )
    if not metadata.names:
        max_id = max((a.class_id for s in samples for a in s.annotations), default=-1)
        metadata.names = {i: f"class_{i}" for i in range(max_id + 1)}
    manifest = _load_manifest(yaml_dir, errors=errors, warnings=warnings)
    dataset_name = name or manifest.get("name") or raw.get("name") or root.name
    return root, dataset_name, resolved_task, metadata, samples, manifest


def _expand_yolo_split(
    value: Any,
    *,
    root: Path,
    yaml_dir: Path,
    errors: Literal["raise", "skip"],
    warnings: list[str],
) -> list[tuple[Path, Path]]:
    values = value if isinstance(value, list) else [value]
    result: list[tuple[Path, Path]] = []
    for item in values:
        path = Path(str(item))
        path = path if path.is_absolute() else (root / path)
        if not path.exists() and str(item).startswith("./"):
            path = yaml_dir / str(item)[2:]
        path = path.resolve()
        if path.suffix.lower() == ".txt" and path.is_file():
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                issue = ValidationIssue(f"Unreadable image-list file: {exc}", source=str(path))
                if errors == "raise":
                    raise DatasetValidationError(issue) from exc
                warnings.append(f"Skipped invalid split entry: {issue.format()}")
                continue
            for line in lines:
                if not line.strip():
                    continue
                image = Path(line.strip())
                if not image.is_absolute():
                    image = (path.parent / image).resolve()
                result.append((image, _relative_image_path(image)))
        elif path.is_dir():
            images = image_files(path)
            base = _image_base(path)
            result.extend((image, image.relative_to(base)) for image in images)
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            result.append((path, Path(path.name)))
        else:
            issue = ValidationIssue("Split path does not exist or contains no supported images", source=str(path))
            if errors == "raise":
                raise DatasetValidationError(issue)
            warnings.append(f"Skipped invalid split entry: {issue.format()}")
    return result


def _image_base(path: Path) -> Path:
    if (path / "images").is_dir():
        return path / "images"
    if path.name == "images":
        return path
    return path


def _relative_image_path(image: Path) -> Path:
    parts = image.parts
    indices = [i for i, part in enumerate(parts) if part == "images"]
    if indices:
        return Path(*parts[indices[-1] + 1 :])
    return Path(image.name)


def _label_path_for_image(image: Path, relative_path: Path | None = None) -> Path:
    """Resolve a YOLO label without confusing nested ``images`` directories.

    When the caller knows an image's path relative to its configured image
    root, use that relationship to locate the sibling label root. This keeps
    paths such as ``train/images/example.jpg`` intact and pairs them with
    ``<label-root>/train/images/example.txt``. The heuristic fallback preserves
    support for standalone and image-list inputs that do not expose a root.
    """

    if relative_path is not None:
        relative = Path(relative_path)
        if relative.parts and not relative.is_absolute() and ".." not in relative.parts:
            image_root = image
            for _ in relative.parts:
                image_root = image_root.parent
            if image_root.name == "images" and image_root / relative == image:
                return image_root.with_name("labels") / relative.with_suffix(".txt")
    parts = list(image.parts)
    indices = [i for i, part in enumerate(parts) if part == "images"]
    if indices:
        parts[indices[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.parent.parent / "labels" / f"{image.stem}.txt"


def _infer_yolo_task(
    images: list[tuple[Path, Path]],
    metadata: DatasetMetadata,
    *,
    errors: Literal["raise", "skip"],
    warnings: list[str],
) -> Task | None:
    if metadata.kpt_shape:
        return Task.POSE
    for image, relative_path in images:
        label = _label_path_for_image(image, relative_path)
        if not label.is_file():
            continue
        try:
            lines = label.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            issue = ValidationIssue(f"Unreadable label file during task inference: {exc}", source=str(label))
            if errors == "raise":
                raise DatasetValidationError(issue) from exc
            warnings.append(f"Skipped invalid label file: {issue.format()}")
            continue
        for line in lines:
            cols = line.split()
            if not cols:
                continue
            if len(cols) == 4:
                return Task.POLO
            if len(cols) == 5:
                return Task.DETECT
            if len(cols) >= 7 and len(cols) % 2 == 1:
                return Task.SEGMENT
    return None


def _parse_yolo_images(
    split_images: list[tuple[str, Path, Path]],
    task: Task,
    metadata: DatasetMetadata,
    *,
    progress: bool,
    errors: Literal["raise", "skip"],
    warnings: list[str],
) -> list[Sample]:
    samples: list[Sample] = []
    iterator = tqdm(split_images, desc="Loading YOLO dataset", unit="image", disable=not progress)
    issues: list[ValidationIssue] = []
    for split, image_path, relative_path in iterator:
        if not image_path.is_file():
            issue = ValidationIssue("Image referenced by dataset does not exist", source=str(image_path))
            if errors == "skip":
                warnings.append(f"Skipped invalid image: {issue.format()}")
            else:
                issues.append(issue)
            continue
        try:
            with Image.open(image_path) as opened:
                image = ImageOps.exif_transpose(opened)
                width, height = image.size
        except Exception as exc:
            issue = ValidationIssue(f"Unreadable image: {exc}", source=str(image_path))
            if errors == "skip":
                warnings.append(f"Skipped invalid image: {issue.format()}")
            else:
                issues.append(issue)
            continue
        annotations: list[Annotation] = []
        label_path = _label_path_for_image(image_path, relative_path)
        if label_path.is_file():
            try:
                label_lines = label_path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                issue = ValidationIssue(f"Unreadable label file: {exc}", source=str(label_path))
                if errors == "skip":
                    warnings.append(f"Skipped invalid label file: {issue.format()}")
                    label_lines = []
                else:
                    issues.append(issue)
                    label_lines = []
            for line_no, line in enumerate(label_lines, 1):
                if not line.strip():
                    continue
                try:
                    annotation = _parse_yolo_line(line, task, metadata, width, height)
                    annotation.source_id = f"{relative_path}:{line_no}"
                    annotations.append(annotation)
                except Exception as exc:
                    issue = ValidationIssue(
                        str(exc),
                        source=str(label_path),
                        line=line_no,
                        value=line,
                        suggestion="fix or remove this label row",
                    )
                    if errors == "skip":
                        warnings.append(f"Skipped invalid annotation: {issue.format()}")
                    else:
                        issues.append(issue)
        samples.append(Sample(image_path, relative_path, split, width, height, annotations))
    if issues:
        raise DatasetValidationError(issues)
    return samples


def _parse_yolo_line(line: str, task: Task, metadata: DatasetMetadata, width: int, height: int) -> Annotation:
    values = [float(v) for v in line.split()]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("Label contains non-finite or missing numeric values")
    class_id = int(values[0])
    if values[0] != class_id:
        raise ValueError("Class ID must be an integer")
    if task is Task.POLO:
        if len(values) != 4:
            raise ValueError(f"POLO labels require 4 columns, found {len(values)}")
        _, radius, xn, yn = values
        return Annotation(class_id, point=(xn * width, yn * height), radius=radius)
    if task is Task.DETECT:
        if len(values) != 5:
            raise ValueError(f"Detection labels require 5 columns, found {len(values)}")
        _, xn, yn, wn, hn = values
        return Annotation(class_id, bbox=_norm_box_to_xyxy(xn, yn, wn, hn, width, height))
    if task is Task.SEGMENT:
        if len(values) < 7 or len(values) % 2 == 0:
            raise ValueError("Segmentation labels require class plus at least three x/y points")
        polygon = [(values[i] * width, values[i + 1] * height) for i in range(1, len(values), 2)]
        xs, ys = zip(*polygon)
        return Annotation(class_id, bbox=(min(xs), min(ys), max(xs), max(ys)), polygon=polygon)
    if metadata.kpt_shape is None:
        raise ValueError("Pose labels require kpt_shape in data.yaml")
    nkpt, ndim = metadata.kpt_shape
    expected = 5 + nkpt * ndim
    if len(values) != expected:
        raise ValueError(f"Pose labels require {expected} columns for kpt_shape={metadata.kpt_shape}, found {len(values)}")
    _, xn, yn, wn, hn, *flat = values
    keypoints = []
    for i in range(nkpt):
        x = flat[i * ndim] * width
        y = flat[i * ndim + 1] * height
        visibility = flat[i * ndim + 2] if ndim == 3 else None
        keypoints.append((x, y, visibility))
    return Annotation(class_id, bbox=_norm_box_to_xyxy(xn, yn, wn, hn, width, height), keypoints=keypoints)


def _norm_box_to_xyxy(xn: float, yn: float, wn: float, hn: float, width: int, height: int) -> tuple[float, float, float, float]:
    cx, cy, bw, bh = xn * width, yn * height, wn * width, hn * height
    return cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2


def _load_flat_yolo(
    root: Path,
    *,
    task: Task | None,
    name: str | None,
    names_override: dict[int, str] | list[str] | None,
    radii_override: dict[int, float] | None,
    progress: bool,
    errors: Literal["raise", "skip"],
    warnings: list[str],
) -> tuple[Path, str, Task, DatasetMetadata, list[Sample], dict[str, Any]]:
    split_directories = [
        (split, root / split / "images")
        for split in ("train", "val", "test")
        if (root / split / "images").is_dir()
    ]
    if split_directories:
        split_images = [
            (split, image, image.relative_to(images_dir))
            for split, images_dir in split_directories
            for image in image_files(images_dir)
        ]
        searched = root
    else:
        images_dir = root / "images" if (root / "images").is_dir() else root
        split_images = [
            ("train", image, image.relative_to(images_dir))
            for image in image_files(images_dir)
        ]
        searched = images_dir
    if not split_images:
        raise DatasetValidationError(f"No supported images found in {searched}")
    metadata = DatasetMetadata(
        names=_parse_names(names_override), radii={int(k): float(v) for k, v in (radii_override or {}).items()}
    )
    resolved_task = task or _infer_yolo_task(
        [(image, relative) for _, image, relative in split_images],
        metadata,
        errors=errors,
        warnings=warnings,
    )
    if resolved_task is None:
        raise DatasetValidationError("Could not infer task from flat dataset; pass task explicitly")
    samples = _parse_yolo_images(
        split_images,
        resolved_task,
        metadata,
        progress=progress,
        errors=errors,
        warnings=warnings,
    )
    if not metadata.names:
        max_id = max((a.class_id for s in samples for a in s.annotations), default=-1)
        metadata.names = {i: f"class_{i}" for i in range(max_id + 1)}
    return root, name or root.name, resolved_task, metadata, samples, _load_manifest(
        root,
        errors=errors,
        warnings=warnings,
    )


def _load_coco(
    source: Path,
    *,
    task: Task | None,
    name: str | None,
    progress: bool,
    errors: Literal["raise", "skip"],
    warnings: list[str],
) -> tuple[Path, str, Task, DatasetMetadata, list[Sample], dict[str, Any]]:
    root = source.parent if source.is_file() else source
    json_files = [source] if source.is_file() else sorted(
        p
        for p in source.rglob("*.json")
        if p.name not in {"dataset-fixer.json", DATASET_INFO_NAME, "source.json"}
        and not {".cache", "evaluations", "reports"}.intersection(p.relative_to(source).parts)
    )
    if not json_files:
        raise DatasetValidationError(f"No COCO JSON annotations found in {source}")

    loaded: list[tuple[Path, dict[str, Any]]] = []
    load_issues: list[ValidationIssue] = []
    for path in json_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("top-level JSON value must be an object")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            issue = ValidationIssue(f"Unreadable annotation JSON: {exc}", source=str(path))
            if errors == "skip":
                warnings.append(f"Skipped invalid annotation file: {issue.format()}")
            else:
                load_issues.append(issue)
            continue
        loaded.append((path, data))
    if load_issues:
        raise DatasetValidationError(load_issues)
    valid = [(p, d) for p, d in loaded if all(k in d for k in ("images", "annotations", "categories"))]
    if not valid:
        raise DatasetValidationError("JSON files do not contain COCO images, annotations, and categories arrays")
    inferred = _infer_coco_task([d for _, d in valid])
    resolved_task = task or inferred
    if resolved_task is None:
        raise DatasetValidationError("COCO contains multiple annotation geometries; pass task explicitly")

    category_names: dict[int, str] = {}
    category_map: dict[int, int] = {}
    categories = valid[0][1]["categories"]
    try:
        sorted_categories = sorted(categories, key=lambda category: int(category["id"]))
        source_category_ids = [int(category["id"]) for category in sorted_categories]
        if len(source_category_ids) != len(set(source_category_ids)):
            raise ValueError("category IDs must be unique")
        for new_id, category in enumerate(sorted_categories):
            category_map[int(category["id"])] = new_id
            category_names[new_id] = str(category["name"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetValidationError(
            ValidationIssue(
                f"COCO reference category schema is unusable: {exc}",
                source=str(valid[0][0]),
            )
        ) from exc

    reference_categories = {(int(category["id"]), str(category["name"])) for category in categories}
    reference_keypoints = {
        int(category["id"]): tuple(category.get("keypoints") or []) for category in categories
    }
    reference_skeletons = {
        int(category["id"]): tuple(tuple(edge) for edge in category.get("skeleton") or []) for category in categories
    }
    schema_issues: list[ValidationIssue] = []
    sanitized_valid: list[tuple[Path, dict[str, Any]]] = []
    for json_path, data in valid:
        try:
            actual_categories = {(int(category["id"]), str(category["name"])) for category in data["categories"]}
        except (KeyError, TypeError, ValueError) as exc:
            issue = ValidationIssue(f"Invalid COCO category schema: {exc}", source=str(json_path))
            if errors == "skip":
                warnings.append(f"Skipped incompatible annotation file: {issue.format()}")
                continue
            schema_issues.append(issue)
            continue
        if actual_categories != reference_categories:
            issue = ValidationIssue(
                "COCO category schemas differ between split files",
                source=str(json_path),
                value=sorted(actual_categories),
                expected=str(sorted(reference_categories)),
            )
            if errors == "skip":
                warnings.append(f"Skipped incompatible annotation file: {issue.format()}")
                continue
            schema_issues.append(issue)
        if resolved_task is Task.POSE:
            actual_keypoints = {
                int(category["id"]): tuple(category.get("keypoints") or []) for category in data["categories"]
            }
            actual_skeletons = {
                int(category["id"]): tuple(tuple(edge) for edge in category.get("skeleton") or [])
                for category in data["categories"]
            }
            if actual_keypoints != reference_keypoints or actual_skeletons != reference_skeletons:
                issue = ValidationIssue(
                    "COCO pose keypoint or skeleton schemas differ between splits",
                    source=str(json_path),
                )
                if errors == "skip":
                    warnings.append(f"Skipped incompatible annotation file: {issue.format()}")
                    continue
                schema_issues.append(issue)

        clean_images: list[dict[str, Any]] = []
        image_ids: set[int] = set()
        for image in data["images"]:
            try:
                image_id = int(image["id"])
            except (KeyError, TypeError, ValueError) as exc:
                issue = ValidationIssue(f"Invalid COCO image record: {exc}", source=str(json_path), value=image)
            else:
                issue = (
                    ValidationIssue("COCO image IDs must be unique", source=str(json_path), value=image_id)
                    if image_id in image_ids
                    else None
                )
            if issue is not None:
                if errors == "skip":
                    warnings.append(f"Skipped invalid image record: {issue.format()}")
                    continue
                schema_issues.append(issue)
                continue
            image_ids.add(image_id)
            clean_images.append(image)

        clean_annotations: list[dict[str, Any]] = []
        annotation_ids: set[Any] = set()
        for annotation in data["annotations"]:
            annotation_id = annotation.get("id")
            issue = None
            if annotation_id in annotation_ids:
                issue = ValidationIssue(
                    "COCO annotation IDs must be unique",
                    source=str(json_path),
                    value=annotation_id,
                )
            else:
                try:
                    image_id = int(annotation.get("image_id", -1))
                    category_id = int(annotation.get("category_id", -1))
                except (TypeError, ValueError) as exc:
                    issue = ValidationIssue(
                        f"Invalid COCO annotation reference: {exc}",
                        source=str(json_path),
                        value={"annotation_id": annotation_id},
                    )
                else:
                    if image_id not in image_ids:
                        issue = ValidationIssue(
                            "COCO annotation references an unknown image",
                            source=str(json_path),
                            value={"annotation_id": annotation_id, "image_id": image_id},
                        )
                    elif category_id not in category_map:
                        issue = ValidationIssue(
                            "COCO annotation references an unknown category",
                            source=str(json_path),
                            value={"annotation_id": annotation_id, "category_id": category_id},
                        )
            if issue is not None:
                if errors == "skip":
                    warnings.append(f"Skipped invalid annotation: {issue.format()}")
                    continue
                schema_issues.append(issue)
                continue
            annotation_ids.add(annotation_id)
            clean_annotations.append(annotation)
        clean_data = dict(data)
        clean_data["images"] = clean_images
        clean_data["annotations"] = clean_annotations
        sanitized_valid.append((json_path, clean_data))
    if schema_issues:
        raise DatasetValidationError(schema_issues)
    valid = sanitized_valid
    if not valid or not any(data["images"] for _, data in valid):
        raise DatasetValidationError("COCO contains no valid images after skipping recoverable errors")

    metadata = DatasetMetadata(names=category_names)
    if resolved_task is Task.POSE:
        keypoint_lengths = {len(c.get("keypoints", [])) for c in categories if c.get("keypoints")}
        if len(keypoint_lengths) > 1:
            raise DatasetValidationError("COCO pose categories have inconsistent keypoint schemas")
        if keypoint_lengths:
            nkpt = next(iter(keypoint_lengths))
            metadata.kpt_shape = (nkpt, 3)
            metadata.kpt_names = {
                category_map[int(c["id"])]: list(c.get("keypoints", [])) for c in categories if c.get("keypoints")
            }
            skeletons = {
                category_map[int(c["id"])]: c.get("skeleton", []) for c in categories if c.get("skeleton")
            }
            if skeletons:
                metadata.extra["skeleton"] = skeletons

    samples: list[Sample] = []
    issues: list[ValidationIssue] = []
    output_paths: dict[tuple[str, Path], Path] = {}
    for json_path, data in valid:
        split = _split_from_filename(json_path.name)
        by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in data["annotations"]:
            by_image.setdefault(int(ann["image_id"]), []).append(ann)
        iterator = tqdm(data["images"], desc=f"Loading COCO {split}", unit="image", disable=not progress)
        for image_record in iterator:
            try:
                image_id = int(image_record["id"])
                file_name = str(image_record["file_name"])
                width, height = int(image_record["width"]), int(image_record["height"])
            except (KeyError, TypeError, ValueError) as exc:
                issue = ValidationIssue(
                    f"Invalid COCO image record: {exc}",
                    source=str(json_path),
                    value=image_record,
                )
                if errors == "skip":
                    warnings.append(f"Skipped invalid image record: {issue.format()}")
                else:
                    issues.append(issue)
                continue
            image_path = _resolve_coco_image(root, json_path, file_name, split)
            if image_path is None:
                issue = ValidationIssue(
                    "COCO image file not found",
                    source=str(json_path),
                    value=file_name,
                )
                if errors == "skip":
                    warnings.append(f"Skipped invalid image: {issue.format()}")
                else:
                    issues.append(issue)
                continue
            relative_path = _canonical_coco_relative_path(root, image_path, split)
            output_key = (split, relative_path)
            if output_key in output_paths:
                issues.append(
                    ValidationIssue(
                        "COCO images would map to the same canonical output path",
                        source=str(json_path),
                        value={
                            "output": str(Path(split) / "images" / relative_path),
                            "first": str(output_paths[output_key]),
                            "second": str(image_path),
                        },
                        suggestion="make COCO file_name paths unique within each split",
                    )
                )
                continue
            output_paths[output_key] = image_path
            try:
                with Image.open(image_path) as opened:
                    actual_width, actual_height = ImageOps.exif_transpose(opened).size
            except Exception as exc:
                issue = ValidationIssue(f"Unreadable COCO image: {exc}", source=str(image_path))
                if errors == "skip":
                    warnings.append(f"Skipped invalid image: {issue.format()}")
                else:
                    issues.append(issue)
                continue
            if (width, height) != (actual_width, actual_height):
                issue = ValidationIssue(
                    "COCO image dimensions do not match the image file after EXIF orientation",
                    source=str(image_path),
                    value={"coco": [width, height], "actual": [actual_width, actual_height]},
                    suggestion="correct the COCO image width and height",
                )
                if errors == "skip":
                    warnings.append(f"Skipped invalid image: {issue.format()}")
                else:
                    issues.append(issue)
                continue
            annotations: list[Annotation] = []
            for ann in by_image.get(image_id, []):
                try:
                    class_id = category_map[int(ann["category_id"])]
                except KeyError:
                    issues.append(ValidationIssue("Annotation references unknown category", source=str(json_path), value=ann))
                    continue
                try:
                    bbox_raw = ann.get("bbox")
                    bbox = None
                    if bbox_raw:
                        x, y, w, h = map(float, bbox_raw)
                        bbox = (x, y, x + w, y + h)
                    annotation = Annotation(class_id, bbox=bbox, source_id=ann.get("id"))
                    if resolved_task is Task.SEGMENT:
                        segmentation = ann.get("segmentation")
                        if isinstance(segmentation, dict):
                            annotation.rle = segmentation
                        elif isinstance(segmentation, list):
                            if len(segmentation) != 1:
                                annotation.rle = {"multipart": segmentation}
                            elif segmentation:
                                flat = segmentation[0]
                                annotation.polygon = [
                                    (float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat), 2)
                                ]
                    elif resolved_task is Task.POSE:
                        flat = ann.get("keypoints") or []
                        annotation.keypoints = [
                            (float(flat[i]), float(flat[i + 1]), float(flat[i + 2]))
                            for i in range(0, len(flat), 3)
                        ]
                except (IndexError, TypeError, ValueError) as exc:
                    source_id = ann.get("id")
                    source = f"{json_path} [annotation {source_id}]" if source_id is not None else str(json_path)
                    issue = ValidationIssue(
                        f"Malformed COCO annotation: {exc}",
                        source=source,
                        suggestion="fix or remove this annotation",
                    )
                    if errors == "skip":
                        warnings.append(f"Skipped invalid annotation: {issue.format()}")
                    else:
                        issues.append(issue)
                    continue
                annotations.append(annotation)
            samples.append(
                Sample(image_path, relative_path, split, width, height, annotations)
            )
    if issues:
        raise DatasetValidationError(issues)
    info = valid[0][1].get("info") or {}
    dataset_name = name or info.get("name") or info.get("description") or root.name
    return root, dataset_name, resolved_task, metadata, samples, _load_manifest(
        root,
        errors=errors,
        warnings=warnings,
    )


def _infer_coco_task(data_sets: list[dict[str, Any]]) -> Task | None:
    found: set[Task] = set()
    for data in data_sets:
        for ann in data.get("annotations", []):
            if ann.get("keypoints"):
                found.add(Task.POSE)
            if ann.get("segmentation"):
                found.add(Task.SEGMENT)
            if ann.get("bbox"):
                found.add(Task.DETECT)
    if len(found) == 1:
        return next(iter(found))
    return None


def _split_from_filename(name: str) -> str:
    lower = name.lower()
    if "test" in lower:
        return "test"
    if "val" in lower or "valid" in lower:
        return "val"
    return "train"


def _resolve_coco_image(root: Path, json_path: Path, file_name: str, split: str) -> Path | None:
    rel = Path(file_name)
    candidates = [
        root / rel,
        root / "images" / rel,
        root / "images" / split / rel,
        root / split / rel,
        json_path.parent / rel,
        json_path.parent.parent / rel,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = list(root.rglob(rel.name))
    return matches[0].resolve() if len(matches) == 1 else None


def _canonical_coco_relative_path(root: Path, image_path: Path, split: str) -> Path:
    """Return a lossless path relative to the most specific known image root."""

    candidates = (
        root / split / "images",
        root / "images" / split,
        root / "images",
        root / split,
        root,
    )
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            relative = image_path.relative_to(resolved)
        except ValueError:
            continue
        if relative.parts:
            return relative
    return Path(image_path.name)


def _load_manifest(
    root: Path,
    *,
    errors: Literal["raise", "skip"],
    warnings: list[str],
) -> dict[str, Any]:
    path = dataset_info_path(root)
    if not path.is_file():
        # Old datasets remain readable in memory, but all writers emit the new
        # compact reports layout.
        legacy = root / "dataset-fixer.json"
        path = legacy if legacy.is_file() else path
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        issue = ValidationIssue(f"Unreadable dataset-fixer manifest: {exc}", source=str(path))
        if errors == "raise":
            raise DatasetValidationError(issue) from exc
        warnings.append(f"Ignored invalid manifest: {issue.format()}")
        return {}


def annotation_to_yolo(annotation: Annotation, task: Task, width: int, height: int, metadata: DatasetMetadata) -> str:
    cls = annotation.class_id
    if task is Task.POLO:
        assert annotation.point is not None and annotation.radius is not None
        return f"{cls} {annotation.radius:.6f} {annotation.point[0] / width:.6f} {annotation.point[1] / height:.6f}"
    if task is Task.DETECT:
        assert annotation.bbox is not None
        x1, y1, x2, y2 = annotation.bbox
        return f"{cls} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
    if task is Task.SEGMENT:
        if not annotation.polygon:
            raise ValueError("Segmentation annotation is not representable as one YOLO polygon")
        coords = " ".join(f"{x / width:.6f} {y / height:.6f}" for x, y in annotation.polygon)
        return f"{cls} {coords}"
    assert annotation.bbox is not None and annotation.keypoints is not None and metadata.kpt_shape is not None
    x1, y1, x2, y2 = annotation.bbox
    values = [
        str(cls),
        f"{(x1 + x2) / 2 / width:.6f}",
        f"{(y1 + y2) / 2 / height:.6f}",
        f"{(x2 - x1) / width:.6f}",
        f"{(y2 - y1) / height:.6f}",
    ]
    ndim = metadata.kpt_shape[1]
    for x, y, visibility in annotation.keypoints:
        values.extend((f"{x / width:.6f}", f"{y / height:.6f}"))
        if ndim == 3:
            values.append(f"{0.0 if visibility is None else visibility:.6f}")
    return " ".join(values)
