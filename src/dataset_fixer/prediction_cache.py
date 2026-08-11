from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

import numpy as np
from PIL import Image

from .errors import DatasetValidationError
from .sources import cache_root as package_cache_root
from .utils import sha256_file, to_jsonable

if TYPE_CHECKING:
    from .dataset import Dataset
    from .model import ImagePrediction, ModelInput, PredictionResult


RAW_RESULT_CACHE_SCHEMA = 1
PredictionCacheNamespace = Literal["predictions", "semantic"]


def prediction_cache_key(payload: Mapping[str, Any]) -> str:
    """Return the stable SHA-256 key used for prediction cache identities."""

    return hashlib.sha256(
        json.dumps(
            to_jsonable(dict(payload)),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class PredictionCache:
    """Shared, verified storage for model prediction results.

    location:
        Established cache base containing the ``predictions`` and ``semantic``
        namespaces. Dataset-local caches therefore resolve to
        ``<dataset>/.cache/evaluations`` rather than introducing a competing
        tree.
    """

    def __init__(self, location: str | Path) -> None:
        self._location = Path(location).expanduser().resolve()

    @property
    def location(self) -> Path:
        """Resolved cache base directory."""

        return self._location

    @classmethod
    def for_dataset(cls, dataset: "Dataset") -> "PredictionCache":
        """Use the established comparison cache belonging to a dataset.

        dataset:
            Dataset whose ``.cache/evaluations`` directory should be used.
        """

        from .comparison.cache import default_cache_root

        return cls(default_cache_root(dataset.location))

    @classmethod
    def package_default(cls) -> "PredictionCache":
        """Use package-managed storage for non-dataset prediction inputs."""

        return cls(package_cache_root() / "evaluations")

    def namespace(self, value: PredictionCacheNamespace) -> Path:
        """Return the directory for one cache namespace.

        value:
            Either ``"predictions"`` or ``"semantic"``.
        """

        if value not in {"predictions", "semantic"}:
            raise ValueError("prediction cache namespace must be 'predictions' or 'semantic'")
        return self.location / value

    def entry(self, key: str, *, namespace: PredictionCacheNamespace) -> Path:
        """Return the directory for one content-addressed cache entry.

        key:
            Lowercase hexadecimal cache identity.
        namespace:
            Cache namespace containing the entry.
        """

        if not key or any(character not in "0123456789abcdef" for character in key):
            raise ValueError("prediction cache key must be a lowercase hexadecimal digest")
        return self.namespace(namespace) / key

    def load(
        self,
        key: str,
        *,
        namespace: PredictionCacheNamespace,
        identity: Mapping[str, Any],
        inputs: Sequence["ModelInput"],
    ) -> "PredictionResult | None":
        """Load and validate one complete raw prediction result.

        key:
            Content-addressed cache identity.
        namespace:
            Cache namespace containing the entry.
        identity:
            Complete identity payload expected to hash to ``key``.
        inputs:
            Ordered model inputs that the result must cover.
        """

        root = self.entry(key, namespace=namespace) / "raw-result"
        manifest_path = root / "manifest.json"
        complete_path = root / "complete.json"
        if not manifest_path.is_file() or not complete_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema") != RAW_RESULT_CACHE_SCHEMA
            or manifest.get("key") != key
            or prediction_cache_key(manifest.get("identity") or {}) != key
            or prediction_cache_key(identity) != key
            or not isinstance(complete, dict)
            or complete.get("key") != key
        ):
            return None

        expected = _input_manifest(inputs)
        if manifest.get("inputs") != expected:
            return None
        result_value = manifest.get("result")
        records_value = manifest.get("records")
        if not isinstance(result_value, dict) or not isinstance(records_value, list):
            return None
        if len(records_value) != len(inputs):
            return None

        from .comparison.types import Prediction
        from .model import ImagePrediction, PredictionResult

        records: list[ImagePrediction] = []
        for source, stored in zip(inputs, records_value):
            if not isinstance(stored, dict) or stored.get("image_id") != source.image_id:
                return None
            if (
                stored.get("relative_path") != source.relative_path
                or stored.get("width") != source.width
                or stored.get("height") != source.height
            ):
                return None
            try:
                metadata = dict(stored.get("metadata") or {})
            except (TypeError, ValueError):
                return None

            mask = _load_mask(root, stored.get("mask"), source.width, source.height)
            if stored.get("mask") is not None and mask is None:
                return None
            native_mask = _load_mask(root, stored.get("native_mask"), None, None)
            if stored.get("native_mask") is not None and native_mask is None:
                return None

            objects: list[Prediction] = []
            raw_objects = stored.get("objects") or []
            if isinstance(raw_objects, str):
                objects_path = root / raw_objects
                if not objects_path.is_file():
                    return None
                try:
                    raw_objects = json.loads(objects_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    return None
            if not isinstance(raw_objects, list):
                return None
            try:
                for value in raw_objects:
                    objects.append(_prediction_from_json(value))
                _validate_prediction_geometry(objects, source.width, source.height)
            except (TypeError, ValueError, KeyError):
                return None
            records.append(
                ImagePrediction(
                    image_id=source.image_id,
                    image_path=source.image_path,
                    relative_path=source.relative_path,
                    width=source.width,
                    height=source.height,
                    objects=tuple(objects),
                    mask=mask,
                    native_mask=native_mask,
                    metadata=metadata,
                )
            )

        try:
            inference_seconds = float(result_value.get("inference_seconds", 0.0))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(inference_seconds) or inference_seconds < 0:
            return None
        info = {
            "status": "hit",
            "verified": True,
            "key": key,
            "namespace": namespace,
            "location": str(self.entry(key, namespace=namespace)),
        }
        return PredictionResult(
            model_name=str(result_value.get("model_name") or "cached-model"),
            model_kind=str(result_value.get("model_kind") or "ultralytics"),
            task=str(result_value.get("task") or "detect"),
            backend=str(result_value.get("backend") or "native"),
            records=tuple(records),
            inference_seconds=inference_seconds,
            settings=dict(result_value.get("settings") or {}),
            cache_info=info,
        )

    def save(
        self,
        key: str,
        result: "PredictionResult",
        *,
        namespace: PredictionCacheNamespace,
        identity: Mapping[str, Any],
        inputs: Sequence["ModelInput"],
    ) -> "PredictionResult":
        """Atomically publish one complete raw prediction result.

        key:
            Content-addressed cache identity.
        result:
            Complete model prediction result to publish.
        namespace:
            Cache namespace containing the entry.
        identity:
            Complete identity payload expected to hash to ``key``.
        inputs:
            Ordered model inputs covered by ``result``.
        """

        if prediction_cache_key(identity) != key:
            raise ValueError("prediction cache identity does not match its key")
        if len(result.records) != len(inputs):
            raise DatasetValidationError(
                "Cannot cache a prediction result that does not cover every input"
            )
        entry = self.entry(key, namespace=namespace)
        entry.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".raw-result.building-", dir=entry))
        try:
            records: list[dict[str, Any]] = []
            for source, record in zip(inputs, result.records):
                if (
                    record.image_id != source.image_id
                    or record.relative_path != source.relative_path
                    or record.width != source.width
                    or record.height != source.height
                ):
                    raise DatasetValidationError(
                        "Cannot cache predictions whose ordered inputs do not match"
                    )
                mask_path = _save_mask(staging, "masks", record.image_id, record.mask)
                native_path = _save_mask(
                    staging,
                    "native-masks",
                    record.image_id,
                    record.native_mask,
                )
                objects_path = _save_objects(staging, record.image_id, record.objects)
                records.append(
                    {
                        "image_id": record.image_id,
                        "relative_path": record.relative_path,
                        "width": record.width,
                        "height": record.height,
                        "mask": mask_path,
                        "native_mask": native_path,
                        "objects": objects_path,
                        "metadata": to_jsonable(record.metadata),
                    }
                )
            manifest = {
                "schema": RAW_RESULT_CACHE_SCHEMA,
                "key": key,
                "identity": to_jsonable(dict(identity)),
                "created_at_unix": time.time(),
                "inputs": _input_manifest(inputs),
                "result": {
                    "model_name": result.model_name,
                    "model_kind": result.model_kind,
                    "task": result.task,
                    "backend": result.backend,
                    "inference_seconds": result.inference_seconds,
                    "settings": to_jsonable(result.settings),
                },
                "records": records,
            }
            _write_json(staging / "manifest.json", manifest)
            _write_json(
                staging / "complete.json",
                {"key": key, "completed_at_unix": time.time(), "records": len(records)},
            )
            target = entry / "raw-result"
            previous = entry / ".raw-result.replaced"
            if previous.exists():
                shutil.rmtree(previous)
            if target.exists():
                target.replace(previous)
            try:
                staging.replace(target)
            except Exception:
                if previous.exists() and not target.exists():
                    previous.replace(target)
                raise
            if previous.exists():
                shutil.rmtree(previous)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        info = {
            "status": "fresh",
            "verified": True,
            "key": key,
            "namespace": namespace,
            "location": str(entry),
        }
        return replace(result, cache_info=info)

    def __repr__(self) -> str:
        return f"PredictionCache(location={str(self.location)!r})"


def resolve_prediction_cache(
    value: bool | str | Path | PredictionCache | None,
    *,
    source: Any = None,
    default: bool,
) -> PredictionCache | None:
    """Normalize a public prediction-cache option."""

    if value is None:
        value = default
    if value is False:
        return None
    if isinstance(value, PredictionCache):
        return value
    if isinstance(value, (str, Path)):
        return PredictionCache(value)
    if value is not True:
        raise TypeError(
            "prediction_cache must be a boolean, path, PredictionCache, or None"
        )
    from .dataset import Dataset

    if isinstance(source, Dataset):
        return PredictionCache.for_dataset(source)
    return PredictionCache.package_default()


def prediction_input_fingerprint(inputs: Sequence["ModelInput"]) -> str:
    """Fingerprint ordered prediction inputs without ground-truth annotations."""

    return prediction_cache_key({"schema": 1, "inputs": _input_manifest(inputs)})


def _input_manifest(inputs: Sequence["ModelInput"]) -> list[dict[str, Any]]:
    return [
        {
            "image_id": value.image_id,
            "relative_path": value.relative_path,
            "width": value.width,
            "height": value.height,
            "image_sha256": (
                value.image_sha256
                if getattr(value, "image_sha256", None)
                else sha256_file(value.image_path)
            ),
        }
        for value in inputs
    ]


def _save_mask(
    root: Path,
    folder: str,
    image_id: str,
    value: np.ndarray | None,
) -> str | None:
    if value is None:
        return None
    mask = np.asarray(value)
    if mask.ndim != 2 or not np.all(np.isfinite(mask)):
        raise DatasetValidationError(f"Prediction mask {image_id!r} is not a finite 2D array")
    if np.any(mask < 0) or np.any(mask != np.floor(mask)):
        raise DatasetValidationError(f"Prediction mask {image_id!r} has invalid class IDs")
    maximum = int(mask.max(initial=0))
    dtype = np.uint8 if maximum <= 255 else np.uint16
    relative = Path(folder) / f"{image_id}.png"
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(dtype)).save(destination, format="PNG")
    return relative.as_posix()


def _load_mask(
    root: Path,
    value: Any,
    width: int | None,
    height: int | None,
) -> np.ndarray | None:
    if value is None:
        return None
    path = root / str(value)
    if not path.is_file():
        return None
    try:
        with Image.open(path) as opened:
            mask = np.asarray(opened.copy())
    except OSError:
        return None
    if mask.ndim != 2:
        return None
    if width is not None and height is not None and mask.shape != (height, width):
        return None
    return mask


def _prediction_to_json(value: Any) -> dict[str, Any]:
    return to_jsonable(
        {
            "class_id": value.class_id,
            "score": value.score,
            "bbox": value.bbox,
            "point": value.point,
            "radius": value.radius,
            "polygon": value.polygon,
            "polygons": value.polygons,
            "keypoints": value.keypoints,
            "metadata": value.metadata,
        }
    )


def _save_objects(root: Path, image_id: str, values: Sequence[Any]) -> str | None:
    if not values:
        return None
    relative = Path("objects") / f"{image_id}.json"
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json(destination, [_prediction_to_json(value) for value in values])
    return relative.as_posix()


def _prediction_from_json(value: Mapping[str, Any]) -> Any:
    from .comparison.types import Prediction

    score = float(value["score"])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("prediction score is outside [0, 1]")
    return Prediction(
        class_id=int(value["class_id"]),
        score=score,
        bbox=tuple(map(float, value["bbox"])) if value.get("bbox") is not None else None,
        point=tuple(map(float, value["point"])) if value.get("point") is not None else None,
        radius=float(value["radius"]) if value.get("radius") is not None else None,
        polygon=[tuple(map(float, point)) for point in value.get("polygon") or []] or None,
        polygons=[
            [tuple(map(float, point)) for point in polygon]
            for polygon in value.get("polygons") or []
        ]
        or None,
        keypoints=[tuple(point) for point in value.get("keypoints") or []] or None,
        metadata=dict(value.get("metadata") or {}),
    )


def _validate_prediction_geometry(values: Sequence[Any], width: int, height: int) -> None:
    for value in values:
        if value.class_id < 0:
            raise ValueError("prediction class ID is negative")
        if value.bbox is not None:
            if len(value.bbox) != 4 or not np.isfinite(value.bbox).all():
                raise ValueError("prediction box is invalid")
            x1, y1, x2, y2 = value.bbox
            if x2 < x1 or y2 < y1 or x1 < -1e-3 or y1 < -1e-3:
                raise ValueError("prediction box is invalid")
            if x2 > width + 1e-3 or y2 > height + 1e-3:
                raise ValueError("prediction box is outside the image")
        if value.point is not None:
            if len(value.point) != 2 or not np.isfinite(value.point).all():
                raise ValueError("prediction point is invalid")
            x, y = value.point
            if not (-1e-3 <= x <= width + 1e-3 and -1e-3 <= y <= height + 1e-3):
                raise ValueError("prediction point is outside the image")
        polygons = value.polygons or ([value.polygon] if value.polygon else [])
        for polygon in polygons:
            if any(
                len(point) != 2
                or not np.isfinite(point).all()
                or not (-1e-3 <= point[0] <= width + 1e-3)
                or not (-1e-3 <= point[1] <= height + 1e-3)
                for point in polygon
            ):
                raise ValueError("prediction polygon is invalid or outside the image")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(to_jsonable(value), indent=2, sort_keys=True),
        encoding="utf-8",
    )
