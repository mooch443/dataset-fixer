from __future__ import annotations

import builtins
import hashlib
import io
import json
import pickle
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import DatasetValidationError, ValidationIssue
from ..utils import sha256_file, to_jsonable
from .types import Cohort, Prediction

CACHE_SCHEMA = 4
NOTEBOOK_PICKLE_SCHEMA = 1
NOTEBOOK_NUMPY_SCHEMA = 3


def token(value: float) -> str:
    return f"{float(value):.8f}".rstrip("0").rstrip(".").replace(".", "p")


def cache_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(to_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def model_hash(path: Path) -> str:
    return sha256_file(path)


def default_cache_root(dataset_location: Path) -> Path:
    return dataset_location.parent / ".dataset-fixer-cache" / "model-comparison"


def model_cache_dir(cache_root: Path, model_name: str, key: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)[:100]
    return cache_root / f"{safe}__{key[:24]}"


def _prediction_arrays(predictions: list[Prediction]) -> dict[str, np.ndarray]:
    n = len(predictions)
    boxes = np.full((n, 4), np.nan, dtype=np.float32)
    points = np.full((n, 2), np.nan, dtype=np.float32)
    radii = np.full(n, np.nan, dtype=np.float32)
    extras: list[str] = []
    for index, prediction in enumerate(predictions):
        if prediction.bbox is not None:
            boxes[index] = prediction.bbox
        if prediction.point is not None:
            points[index] = prediction.point
        if prediction.radius is not None:
            radii[index] = prediction.radius
        extras.append(
            json.dumps(
                to_jsonable(
                    {
                        "polygon": prediction.polygon,
                        "polygons": prediction.polygons,
                        "keypoints": prediction.keypoints,
                        "metadata": prediction.metadata,
                    }
                ),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    max_len = max((len(value) for value in extras), default=1)
    return {
        "class_ids": np.asarray([p.class_id for p in predictions], dtype=np.int32),
        "scores": np.asarray([p.score for p in predictions], dtype=np.float32),
        "boxes": boxes,
        "points": points,
        "radii": radii,
        "extras": np.asarray(extras, dtype=f"<U{max_len}"),
    }


def _predictions_from_arrays(data: Any, source: Path) -> list[Prediction]:
    required = {"class_ids", "scores", "boxes", "points", "radii", "extras"}
    if not required.issubset(data.files):
        raise DatasetValidationError(
            ValidationIssue("Prediction cache shard is missing arrays", source=str(source), value=sorted(data.files))
        )
    n = len(data["class_ids"])
    if any(len(data[key]) != n for key in required):
        raise DatasetValidationError(ValidationIssue("Prediction cache arrays have inconsistent lengths", source=str(source)))
    if data["boxes"].shape != (n, 4) or data["points"].shape != (n, 2):
        raise DatasetValidationError(ValidationIssue("Prediction cache geometry arrays have invalid shapes", source=str(source)))
    if not np.isfinite(data["scores"]).all():
        raise DatasetValidationError(ValidationIssue("Prediction scores contain non-finite values", source=str(source)))
    result: list[Prediction] = []
    for i in range(n):
        extra = json.loads(str(data["extras"][i]))
        box = tuple(float(v) for v in data["boxes"][i]) if np.isfinite(data["boxes"][i]).all() else None
        point = tuple(float(v) for v in data["points"][i]) if np.isfinite(data["points"][i]).all() else None
        radius = float(data["radii"][i]) if np.isfinite(data["radii"][i]) else None
        result.append(
            Prediction(
                class_id=int(data["class_ids"][i]),
                score=float(data["scores"][i]),
                bbox=box,
                point=point,
                radius=radius,
                polygon=[tuple(map(float, p)) for p in extra.get("polygon") or []] or None,
                polygons=[
                    [tuple(map(float, point)) for point in polygon]
                    for polygon in extra.get("polygons") or []
                ]
                or None,
                keypoints=[tuple(p) for p in extra.get("keypoints") or []] or None,
                metadata=extra.get("metadata") or {},
            )
        )
    return result


def load_package_cache(
    root: Path, cohort: Cohort, thresholds: tuple[float, ...]
) -> tuple[dict[float, dict[str, list[Prediction]]], int, bool]:
    manifest_path = root / "cache-manifest.json"
    if not manifest_path.is_file():
        return {}, 0, False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}, 0, False
    if manifest.get("schema") != CACHE_SCHEMA or manifest.get("cohort_fingerprint") != cohort.fingerprint:
        return {}, 0, False
    loaded: dict[float, dict[str, list[Prediction]]] = {}
    shards = 0
    for threshold in thresholds:
        by_image: dict[str, list[Prediction]] = {}
        for record in cohort.records:
            path = root / "images" / record.image_id / f"predictions-{token(threshold)}.npz"
            if not path.is_file():
                break
            try:
                with np.load(path, allow_pickle=False) as data:
                    values = _predictions_from_arrays(data, path)
                    _validate_cached_predictions(values, record, cohort, path)
                    by_image[record.image_id] = values
            except Exception:
                break
            shards += 1
        if len(by_image) == len(cohort.records):
            loaded[float(threshold)] = by_image
    complete = set(loaded) == {float(v) for v in thresholds} and (root / "complete.json").is_file()
    return loaded, shards, complete


def save_package_cache(
    root: Path,
    cohort: Cohort,
    payload: dict[str, Any],
    predictions: dict[float, dict[str, list[Prediction]]],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": CACHE_SCHEMA,
        "cohort_fingerprint": cohort.fingerprint,
        "settings": to_jsonable(payload),
        "created_at_unix": time.time(),
    }
    _write_json_atomic(root / "cache-manifest.json", manifest)
    _write_json_atomic(
        root / "cohort.json",
        {"fingerprint": cohort.fingerprint, "image_ids": [record.image_id for record in cohort.records]},
    )
    for threshold, by_image in predictions.items():
        for record in cohort.records:
            if record.image_id not in by_image:
                raise DatasetValidationError("Cannot cache incomplete model predictions")
            output = root / "images" / record.image_id / f"predictions-{token(threshold)}.npz"
            output.parent.mkdir(parents=True, exist_ok=True)
            arrays = _prediction_arrays(by_image[record.image_id])
            with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".npz", delete=False) as handle:
                temporary = Path(handle.name)
            try:
                np.savez_compressed(temporary, **arrays)
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)
    _write_json_atomic(
        root / "complete.json",
        {"cohort_fingerprint": cohort.fingerprint, "thresholds": sorted(predictions), "completed_at_unix": time.time()},
    )


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(to_jsonable(value), sort_keys=True, indent=2), encoding="utf-8")
    temporary.replace(path)


class RestrictedUnpickler(pickle.Unpickler):
    _allowed = {
        ("builtins", name)
        for name in ("dict", "list", "tuple", "set", "frozenset", "str", "bytes", "int", "float", "bool")
    }

    def find_class(self, module: str, name: str):
        if (module, name) in self._allowed:
            return getattr(builtins, name)
        raise pickle.UnpicklingError(f"forbidden pickle global: {module}.{name}")


def restricted_pickle_load(path: Path) -> Any:
    return RestrictedUnpickler(io.BytesIO(path.read_bytes())).load()


def notebook_cache_dirs(dataset_location: Path, explicit: Path | None = None) -> list[Path]:
    values: list[Path] = []
    if explicit:
        values.append(explicit)
    if dataset_location.name.lower() in {"train", "val", "valid", "validation", "test"}:
        values.append(dataset_location.parent / ".sahi_eval_cache")
    values.extend([dataset_location / ".sahi_eval_cache", dataset_location.parent / ".sahi_eval_cache"])
    unique: list[Path] = []
    for value in values:
        resolved = value.expanduser().resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def import_notebook_cache(
    directories: list[Path],
    *,
    model_sha256: str,
    resolution: int,
    confidence_floor: float,
    thresholds: tuple[float, ...],
    cohort: Cohort,
    expected_key: dict[str, Any] | None = None,
    allow_unverified: bool = False,
) -> tuple[dict[float, dict[str, list[Prediction]]] | None, dict[str, Any] | None]:
    for directory in directories:
        if not directory.is_dir():
            continue
        for candidate in sorted(directory.glob("*.gridcache_v3")):
            imported = _load_notebook_numpy(
                candidate,
                model_sha256=model_sha256,
                resolution=resolution,
                confidence_floor=confidence_floor,
                thresholds=thresholds,
                cohort=cohort,
                expected_key=expected_key,
                allow_unverified=allow_unverified,
            )
            if imported is not None:
                values, verified = imported
                return values, {"format": "notebook_numpy", "source": str(candidate), "verified": verified}
        for candidate in sorted(directory.glob("*.gridcache.pkl")):
            imported = _load_notebook_pickle(
                candidate,
                model_sha256=model_sha256,
                resolution=resolution,
                confidence_floor=confidence_floor,
                thresholds=thresholds,
                cohort=cohort,
                expected_key=expected_key,
                allow_unverified=allow_unverified,
            )
            if imported is not None:
                values, verified = imported
                return values, {"format": "notebook_pickle", "source": str(candidate), "verified": verified}
    return None, None


def _valid_notebook_key(
    key: dict[str, Any], model_sha256: str, resolution: int, confidence_floor: float,
    thresholds: tuple[float, ...], expected: dict[str, Any] | None = None,
) -> bool:
    valid = (
        key.get("model_hash") == model_sha256
        and int(key.get("resolution", -1)) == int(resolution)
        and abs(float(key.get("min_conf", -1)) - float(confidence_floor)) < 1e-9
        and sorted(map(float, key.get("iou_list", []))) == sorted(map(float, thresholds))
    )
    if not valid:
        return False
    for name, value in (expected or {}).items():
        if name not in key:
            return False
        actual = key[name]
        if isinstance(value, float):
            if abs(float(actual) - value) > 1e-9:
                return False
        elif actual != value:
            return False
    return True


def notebook_dataset_hash(dataset_root: Path) -> str:
    """Reproduce the notebook's content/path hash exactly."""

    images = dataset_root / "images"
    labels = dataset_root / "labels"
    if not images.is_dir() or not labels.is_dir():
        raise FileNotFoundError(dataset_root)
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = sorted(p for p in images.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)
    files += sorted(p for p in labels.rglob("*") if p.is_file() and p.suffix.lower() == ".txt")
    digest = hashlib.sha256()
    digest.update(str(dataset_root.resolve()).encode("utf-8"))
    for path in files:
        digest.update(path.relative_to(dataset_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def notebook_cache_basename(model_path: Path, key: dict[str, Any], *, numpy: bool) -> str:
    payload = json.dumps(to_jsonable(key), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_path.stem)[:120]
    return f"{stem}__{digest}.gridcache_v3" if numpy else f"{stem}__{digest}.gridcache.pkl"


def _notebook_identity_verified(key: dict[str, Any], image_paths: list[str], cohort: Cohort) -> bool:
    root = Path(str(key.get("dataset_root", ""))).expanduser()
    try:
        if notebook_dataset_hash(root) != key.get("dataset_hash"):
            return False
    except (OSError, ValueError):
        return False
    lookup = _cohort_by_basename(cohort)
    if lookup is None or len(image_paths) != len(cohort.records):
        return False
    try:
        for raw in image_paths:
            old = Path(raw)
            record = lookup[old.name]
            if not old.is_file() or sha256_file(old) != record.image_sha256:
                return False
    except (KeyError, OSError):
        return False
    return True


def _cohort_by_basename(cohort: Cohort) -> dict[str, Any] | None:
    result: dict[str, Any] = {}
    for record in cohort.records:
        name = Path(record.relative_path).name
        if name in result:
            return None
        result[name] = record
    return result


def _validate_notebook_gt(record: Any, rows: np.ndarray) -> bool:
    gt = [annotation for annotation in record.annotations if annotation.get("point") is not None]
    if len(gt) != len(rows):
        return False
    expected = sorted(
        (int(a["class_id"]), float(a.get("radius") or 0), float(a["point"][0]), float(a["point"][1])) for a in gt
    )
    actual = sorted((int(row[0]), float(row[1]), float(row[2]), float(row[3])) for row in rows)
    return all(np.allclose(a, b, rtol=1e-5, atol=1e-3) for a, b in zip(expected, actual))


def _validate_cached_predictions(values: list[Prediction], record: Any, cohort: Cohort, source: Path) -> None:
    for index, value in enumerate(values):
        if value.class_id not in cohort.classes:
            raise DatasetValidationError(
                ValidationIssue("Cached prediction has an unknown class ID", source=str(source), line=index + 1, value=value.class_id)
            )
        if not np.isfinite(value.score) or not 0 <= value.score <= 1:
            raise DatasetValidationError(
                ValidationIssue("Cached prediction score must be finite and in [0, 1]", source=str(source), line=index + 1, value=value.score)
            )
        if value.bbox is not None:
            x1, y1, x2, y2 = value.bbox
            if not all(np.isfinite(value.bbox)) or x2 < x1 or y2 < y1 or x1 < -1e-3 or y1 < -1e-3 or x2 > record.width + 1e-3 or y2 > record.height + 1e-3:
                raise DatasetValidationError(ValidationIssue("Cached prediction box is invalid or outside the image", source=str(source), line=index + 1, value=value.bbox))
        if value.point is not None:
            x, y = value.point
            if not np.isfinite([x, y]).all() or not (-1e-3 <= x <= record.width + 1e-3 and -1e-3 <= y <= record.height + 1e-3):
                raise DatasetValidationError(ValidationIssue("Cached prediction point is outside the image", source=str(source), line=index + 1, value=value.point))
        if cohort.task == "polo" and value.point is None:
            raise DatasetValidationError(ValidationIssue("POLO cache prediction is missing a point", source=str(source), line=index + 1))
        if cohort.task in {"detect", "segment", "pose"} and value.bbox is None:
            raise DatasetValidationError(ValidationIssue("Cached instance prediction is missing a box", source=str(source), line=index + 1))


def _load_notebook_numpy(
    root: Path,
    *,
    model_sha256: str,
    resolution: int,
    confidence_floor: float,
    thresholds: tuple[float, ...],
    cohort: Cohort,
    expected_key: dict[str, Any] | None = None,
    allow_unverified: bool = False,
) -> tuple[dict[float, dict[str, list[Prediction]]], bool] | None:
    try:
        key = json.loads((root / "key.json").read_text(encoding="utf-8"))
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        if key.get("cache_version") != NOTEBOOK_NUMPY_SCHEMA or key.get("cache_format") != "gridcache_v3_numpy_sharded":
            return None
        if meta.get("cache_version") != NOTEBOOK_NUMPY_SCHEMA or meta.get("cache_format") != "gridcache_v3_numpy_sharded":
            return None
        if not _valid_notebook_key(key, model_sha256, resolution, confidence_floor, thresholds, expected_key):
            return None
        names = list(meta["image_names"])
        verified = _notebook_identity_verified(key, list(meta.get("image_paths") or []), cohort)
        if not verified and not allow_unverified:
            return None
        if int(meta.get("skipped", 0)):
            return None
        lookup = _cohort_by_basename(cohort)
        expected_names = [Path(record.relative_path).name for record in cohort.records]
        if lookup is None or names != expected_names:
            return None
        shapes = np.load(root / "image_shapes.npy", allow_pickle=False)
        gt_offsets = np.load(root / "gt_offsets.npy", allow_pickle=False)
        gt_points = np.load(root / "gt_points.npy", allow_pickle=False)
        if shapes.shape != (len(names), 2) or gt_offsets.shape != (len(names) + 1,) or gt_points.ndim != 2 or gt_points.shape[1] != 4:
            return None
        if not np.issubdtype(shapes.dtype, np.integer) or not np.issubdtype(gt_offsets.dtype, np.integer):
            return None
        if not np.isfinite(gt_points).all() or np.any(gt_points[:, 0] != np.floor(gt_points[:, 0])):
            return None
        if gt_offsets[0] != 0 or gt_offsets[-1] != len(gt_points) or np.any(np.diff(gt_offsets) < 0):
            return None
        output: dict[float, dict[str, list[Prediction]]] = {}
        for index, name in enumerate(names):
            record = lookup[name]
            if tuple(map(int, shapes[index])) != (record.height, record.width):
                return None
            if not _validate_notebook_gt(record, gt_points[int(gt_offsets[index]) : int(gt_offsets[index + 1])]):
                return None
        for threshold in thresholds:
            suffix = token(threshold)
            offsets = np.load(root / f"pred_offsets_iou_{suffix}.npy", allow_pickle=False)
            points = np.load(root / f"pred_points_iou_{suffix}.npy", allow_pickle=False)
            if offsets.shape != (len(names) + 1,) or points.ndim != 2 or points.shape[1] != 7:
                return None
            if not np.issubdtype(offsets.dtype, np.integer):
                return None
            if offsets[0] != 0 or offsets[-1] != len(points) or np.any(np.diff(offsets) < 0):
                return None
            by_image: dict[str, list[Prediction]] = {}
            for index, name in enumerate(names):
                record = lookup[name]
                rows = points[int(offsets[index]) : int(offsets[index + 1])]
                if not np.isfinite(rows[:, [0, 1, 3, 4, 5, 6]]).all():
                    return None
                by_image[record.image_id] = [
                    Prediction(
                        class_id=0,
                        score=float(row[2]) if np.isfinite(row[2]) else 1.0,
                        point=(float(row[0]), float(row[1])),
                        bbox=tuple(map(float, row[3:7])),
                        metadata={"cache_source": "notebook_numpy"},
                    )
                    for row in rows
                ]
                _validate_cached_predictions(by_image[record.image_id], record, cohort, root)
            output[float(threshold)] = by_image
        return output, verified
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _load_notebook_pickle(
    path: Path,
    *,
    model_sha256: str,
    resolution: int,
    confidence_floor: float,
    thresholds: tuple[float, ...],
    cohort: Cohort,
    expected_key: dict[str, Any] | None = None,
    allow_unverified: bool = False,
) -> tuple[dict[float, dict[str, list[Prediction]]], bool] | None:
    try:
        payload = restricted_pickle_load(path)
        key = payload["key"]
        images = payload["images"]
        if int(key.get("cache_version", -1)) != NOTEBOOK_PICKLE_SCHEMA or int(payload.get("skipped", 0)):
            return None
        if not _valid_notebook_key(key, model_sha256, resolution, confidence_floor, thresholds, expected_key):
            return None
        lookup = _cohort_by_basename(cohort)
        if lookup is None or [row["image_name"] for row in images] != [Path(record.relative_path).name for record in cohort.records]:
            return None
        verified = _notebook_identity_verified(key, [str(row.get("image_path", "")) for row in images], cohort)
        if not verified and not allow_unverified:
            return None
        output = {float(threshold): {} for threshold in thresholds}
        for image in images:
            record = lookup[image["image_name"]]
            if list(image["image_shape"]) != [record.height, record.width]:
                return None
            gt_rows = np.asarray(
                [[g.get("class_id", 0), g.get("radius", 0), g["x"], g["y"]] for g in image["gt_points"]],
                dtype=float,
            ).reshape(-1, 4)
            if not _validate_notebook_gt(record, gt_rows):
                return None
            for threshold in thresholds:
                raw = image["predictions_by_iou"].get(float(threshold))
                if raw is None:
                    raw = image["predictions_by_iou"].get(str(float(threshold)))
                if raw is None:
                    return None
                output[float(threshold)][record.image_id] = [
                    Prediction(
                        class_id=int(p.get("class_id", 0)),
                        score=float(p.get("score") if p.get("score") is not None else 1.0),
                        point=(float(p["x"]), float(p["y"])),
                        bbox=tuple(map(float, p.get("bbox"))) if p.get("bbox") else None,
                        metadata={"cache_source": "notebook_pickle"},
                    )
                    for p in raw
                ]
                _validate_cached_predictions(output[float(threshold)][record.image_id], record, cohort, path)
        return output, verified
    except Exception:
        return None


def append_migration_log(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": time.time(), **to_jsonable(event)}, sort_keys=True) + "\n")


def write_notebook_numpy_cache(
    root: Path,
    *,
    key: dict[str, Any],
    cohort: Cohort,
    predictions: dict[float, dict[str, list[Prediction]]],
) -> None:
    if cohort.task != "polo":
        raise ValueError("Notebook cache export is supported only for POLO comparisons")
    temporary = root.with_name(root.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    names = [Path(record.relative_path).name for record in cohort.records]
    if len(names) != len(set(names)):
        raise ValueError("Notebook cache export requires unique image basenames")
    gt_offsets = [0]
    gt_rows: list[list[float]] = []
    for record in cohort.records:
        for annotation in record.annotations:
            if annotation.get("point") is not None:
                gt_rows.append(
                    [annotation["class_id"], annotation.get("radius") or 0, *annotation["point"]]
                )
        gt_offsets.append(len(gt_rows))
    _write_json_atomic(temporary / "key.json", key)
    _write_json_atomic(
        temporary / "meta.json",
        {
            "cache_format": "gridcache_v3_numpy_sharded",
            "cache_version": NOTEBOOK_NUMPY_SCHEMA,
            "created_at_unix": time.time(),
            "skipped": 0,
            "num_images": len(cohort.records),
            "image_names": names,
            "image_paths": [str(record.image_path) for record in cohort.records],
            "labels_paths": [
                str(record.image_path.parent.parent / "labels" / Path(record.relative_path).with_suffix(".txt").name)
                for record in cohort.records
            ],
            "iou_values": sorted(predictions),
        },
    )
    np.save(temporary / "image_shapes.npy", np.asarray([[r.height, r.width] for r in cohort.records], dtype=np.int32), allow_pickle=False)
    np.save(temporary / "gt_offsets.npy", np.asarray(gt_offsets, dtype=np.int64), allow_pickle=False)
    np.save(temporary / "gt_points.npy", np.asarray(gt_rows, dtype=np.float32).reshape(-1, 4), allow_pickle=False)
    for threshold, by_image in predictions.items():
        offsets = [0]
        rows: list[list[float]] = []
        for record in cohort.records:
            for prediction in by_image[record.image_id]:
                if prediction.point is None:
                    raise ValueError("Notebook cache export requires point predictions")
                bbox = prediction.bbox or (np.nan, np.nan, np.nan, np.nan)
                rows.append([*prediction.point, prediction.score, *bbox])
            offsets.append(len(rows))
        suffix = token(threshold)
        np.save(temporary / f"pred_offsets_iou_{suffix}.npy", np.asarray(offsets, dtype=np.int64), allow_pickle=False)
        np.save(temporary / f"pred_points_iou_{suffix}.npy", np.asarray(rows, dtype=np.float32).reshape(-1, 7), allow_pickle=False)
    if root.exists():
        shutil.rmtree(root)
    temporary.replace(root)
