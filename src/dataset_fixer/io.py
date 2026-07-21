from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageOps
from tqdm.auto import tqdm

from .errors import DatasetValidationError, ValidationIssue
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
) -> tuple[Path, str, Task, DatasetMetadata, list[Sample], dict[str, Any]]:
    location = location.expanduser().resolve()
    if not location.exists():
        raise FileNotFoundError(f"Dataset does not exist: {location}")

    if location.suffix.lower() == ".json":
        return _load_coco(location, task=task, name=name, progress=progress)

    root = location.parent if location.suffix.lower() in {".yaml", ".yml"} else location
    yaml_path = location if location.suffix.lower() in {".yaml", ".yml"} else _find_yaml(root)
    if yaml_path is not None:
        return _load_yolo(yaml_path, task=task, name=name, names_override=names, radii_override=radii, progress=progress)

    coco_files = sorted(root.rglob("*.json"))
    coco_files = [p for p in coco_files if p.name not in {"dataset-fixer.json"}]
    if coco_files:
        return _load_coco(root, task=task, name=name, progress=progress)

    return _load_flat_yolo(root, task=task, name=name, names_override=names, radii_override=radii, progress=progress)


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
        for image_path, rel in _expand_yolo_split(value, root=root, yaml_dir=yaml_dir):
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

    resolved_task = task or _infer_yolo_task([p for _, p, _ in split_images], metadata)
    if resolved_task is None:
        raise DatasetValidationError("Could not infer task from empty labels; pass task='detect', 'segment', 'pose', or 'polo'")
    samples = _parse_yolo_images(split_images, resolved_task, metadata, progress=progress)
    if not metadata.names:
        max_id = max((a.class_id for s in samples for a in s.annotations), default=-1)
        metadata.names = {i: f"class_{i}" for i in range(max_id + 1)}
    manifest = _load_manifest(yaml_dir)
    dataset_name = name or manifest.get("name") or raw.get("name") or root.name
    return root, dataset_name, resolved_task, metadata, samples, manifest


def _expand_yolo_split(value: Any, *, root: Path, yaml_dir: Path) -> list[tuple[Path, Path]]:
    values = value if isinstance(value, list) else [value]
    result: list[tuple[Path, Path]] = []
    for item in values:
        path = Path(str(item))
        path = path if path.is_absolute() else (root / path)
        if not path.exists() and str(item).startswith("./"):
            path = yaml_dir / str(item)[2:]
        path = path.resolve()
        if path.suffix.lower() == ".txt" and path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
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
            raise DatasetValidationError(
                ValidationIssue("Split path does not exist or contains no supported images", source=str(path))
            )
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


def _label_path_for_image(image: Path) -> Path:
    parts = list(image.parts)
    indices = [i for i, part in enumerate(parts) if part == "images"]
    if indices:
        parts[indices[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.parent.parent / "labels" / f"{image.stem}.txt"


def _infer_yolo_task(images: list[Path], metadata: DatasetMetadata) -> Task | None:
    if metadata.kpt_shape:
        return Task.POSE
    for image in images:
        label = _label_path_for_image(image)
        if not label.is_file():
            continue
        for line in label.read_text(encoding="utf-8").splitlines():
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
    split_images: list[tuple[str, Path, Path]], task: Task, metadata: DatasetMetadata, *, progress: bool
) -> list[Sample]:
    samples: list[Sample] = []
    iterator = tqdm(split_images, desc="Loading YOLO dataset", unit="image", disable=not progress)
    issues: list[ValidationIssue] = []
    for split, image_path, relative_path in iterator:
        if not image_path.is_file():
            issues.append(ValidationIssue("Image referenced by dataset does not exist", source=str(image_path)))
            continue
        try:
            with Image.open(image_path) as opened:
                image = ImageOps.exif_transpose(opened)
                width, height = image.size
        except Exception as exc:
            issues.append(ValidationIssue(f"Unreadable image: {exc}", source=str(image_path)))
            continue
        annotations: list[Annotation] = []
        label_path = _label_path_for_image(image_path)
        if label_path.is_file():
            for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    annotation = _parse_yolo_line(line, task, metadata, width, height)
                    annotation.source_id = f"{relative_path}:{line_no}"
                    annotations.append(annotation)
                except Exception as exc:
                    issues.append(
                        ValidationIssue(
                            str(exc), source=str(label_path), line=line_no, value=line, suggestion="fix or remove this label row"
                        )
                    )
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
) -> tuple[Path, str, Task, DatasetMetadata, list[Sample], dict[str, Any]]:
    images_dir = root / "images" if (root / "images").is_dir() else root
    images = image_files(images_dir)
    if not images:
        raise DatasetValidationError(f"No supported images found in {images_dir}")
    metadata = DatasetMetadata(
        names=_parse_names(names_override), radii={int(k): float(v) for k, v in (radii_override or {}).items()}
    )
    resolved_task = task or _infer_yolo_task(images, metadata)
    if resolved_task is None:
        raise DatasetValidationError("Could not infer task from flat dataset; pass task explicitly")
    split_images = [("train", p, p.relative_to(images_dir)) for p in images]
    samples = _parse_yolo_images(split_images, resolved_task, metadata, progress=progress)
    if not metadata.names:
        max_id = max((a.class_id for s in samples for a in s.annotations), default=-1)
        metadata.names = {i: f"class_{i}" for i in range(max_id + 1)}
    return root, name or root.name, resolved_task, metadata, samples, _load_manifest(root)


def _load_coco(
    source: Path, *, task: Task | None, name: str | None, progress: bool
) -> tuple[Path, str, Task, DatasetMetadata, list[Sample], dict[str, Any]]:
    root = source.parent if source.is_file() else source
    json_files = [source] if source.is_file() else sorted(
        p for p in source.rglob("*.json") if p.name not in {"dataset-fixer.json"}
    )
    if not json_files:
        raise DatasetValidationError(f"No COCO JSON annotations found in {source}")

    loaded = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in json_files]
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
    for new_id, category in enumerate(sorted(categories, key=lambda c: int(c["id"]))):
        category_map[int(category["id"])] = new_id
        category_names[new_id] = str(category["name"])

    reference_categories = {(int(category["id"]), str(category["name"])) for category in categories}
    reference_keypoints = {
        int(category["id"]): tuple(category.get("keypoints") or []) for category in categories
    }
    reference_skeletons = {
        int(category["id"]): tuple(tuple(edge) for edge in category.get("skeleton") or []) for category in categories
    }
    schema_issues: list[ValidationIssue] = []
    for json_path, data in valid:
        actual_categories = {(int(category["id"]), str(category["name"])) for category in data["categories"]}
        if actual_categories != reference_categories:
            schema_issues.append(
                ValidationIssue(
                    "COCO category schemas differ between split files",
                    source=str(json_path),
                    value=sorted(actual_categories),
                    expected=str(sorted(reference_categories)),
                )
            )
        image_ids = [int(image["id"]) for image in data["images"]]
        if len(image_ids) != len(set(image_ids)):
            schema_issues.append(ValidationIssue("COCO image IDs must be unique", source=str(json_path)))
        annotation_ids = [annotation.get("id") for annotation in data["annotations"]]
        if len(annotation_ids) != len(set(annotation_ids)):
            schema_issues.append(ValidationIssue("COCO annotation IDs must be unique", source=str(json_path)))
        for annotation in data["annotations"]:
            if int(annotation.get("image_id", -1)) not in set(image_ids):
                schema_issues.append(
                    ValidationIssue(
                        "COCO annotation references an unknown image",
                        source=str(json_path),
                        value={"annotation_id": annotation.get("id"), "image_id": annotation.get("image_id")},
                    )
                )
            if int(annotation.get("category_id", -1)) not in category_map:
                schema_issues.append(
                    ValidationIssue(
                        "COCO annotation references an unknown category",
                        source=str(json_path),
                        value={"annotation_id": annotation.get("id"), "category_id": annotation.get("category_id")},
                    )
                )
        if resolved_task is Task.POSE:
            actual_keypoints = {
                int(category["id"]): tuple(category.get("keypoints") or []) for category in data["categories"]
            }
            actual_skeletons = {
                int(category["id"]): tuple(tuple(edge) for edge in category.get("skeleton") or [])
                for category in data["categories"]
            }
            if actual_keypoints != reference_keypoints or actual_skeletons != reference_skeletons:
                schema_issues.append(
                    ValidationIssue("COCO pose keypoint or skeleton schemas differ between splits", source=str(json_path))
                )
    if schema_issues:
        raise DatasetValidationError(schema_issues)

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
    for json_path, data in valid:
        split = _split_from_filename(json_path.name)
        by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in data["annotations"]:
            by_image.setdefault(int(ann["image_id"]), []).append(ann)
        iterator = tqdm(data["images"], desc=f"Loading COCO {split}", unit="image", disable=not progress)
        for image_record in iterator:
            image_id = int(image_record["id"])
            image_path = _resolve_coco_image(root, json_path, str(image_record["file_name"]), split)
            if image_path is None:
                issues.append(ValidationIssue("COCO image file not found", source=str(json_path), value=image_record["file_name"]))
                continue
            width, height = int(image_record["width"]), int(image_record["height"])
            try:
                with Image.open(image_path) as opened:
                    actual_width, actual_height = ImageOps.exif_transpose(opened).size
            except Exception as exc:
                issues.append(ValidationIssue(f"Unreadable COCO image: {exc}", source=str(image_path)))
                continue
            if (width, height) != (actual_width, actual_height):
                issues.append(
                    ValidationIssue(
                        "COCO image dimensions do not match the image file after EXIF orientation",
                        source=str(image_path),
                        value={"coco": [width, height], "actual": [actual_width, actual_height]},
                        suggestion="correct the COCO image width and height",
                    )
                )
                continue
            annotations: list[Annotation] = []
            for ann in by_image.get(image_id, []):
                try:
                    class_id = category_map[int(ann["category_id"])]
                except KeyError:
                    issues.append(ValidationIssue("Annotation references unknown category", source=str(json_path), value=ann))
                    continue
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
                            annotation.polygon = [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat), 2)]
                elif resolved_task is Task.POSE:
                    flat = ann.get("keypoints") or []
                    annotation.keypoints = [
                        (float(flat[i]), float(flat[i + 1]), float(flat[i + 2])) for i in range(0, len(flat), 3)
                    ]
                annotations.append(annotation)
            samples.append(
                Sample(image_path, Path(str(image_record["file_name"])), split, width, height, annotations)
            )
    if issues:
        raise DatasetValidationError(issues)
    info = valid[0][1].get("info") or {}
    dataset_name = name or info.get("name") or info.get("description") or root.name
    return root, dataset_name, resolved_task, metadata, samples, _load_manifest(root)


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


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / "dataset-fixer.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
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
