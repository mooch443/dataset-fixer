from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from ..errors import DatasetValidationError, ValidationIssue
from ..utils import sha256_file, to_jsonable
from .types import Cohort, Prediction

CACHE_SCHEMA = 5
EVALUATION_CACHE_SCHEMA = 1


def token(value: float) -> str:
    return f"{float(value):.8f}".rstrip("0").rstrip(".").replace(".", "p")


def cache_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(to_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def model_hash(path: Path) -> str:
    return sha256_file(path)


def default_cache_root(dataset_location: Path) -> Path:
    return dataset_location / ".cache" / "evaluations"


def build_staging_dir(target: Path, *, dataset_location: Path | None) -> Path:
    """Create the private directory an in-progress evaluation is built in.

    Default destinations stage under the dataset's ``.cache`` so a cancelled or
    failed run never leaves a partial build beside published evaluations. An
    explicit destination stages beside itself, which keeps publication an
    atomic same-filesystem rename.
    """

    root = (
        dataset_location / ".cache" / "builds"
        if dataset_location is not None
        else target.parent
    )
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{target.name}.building-", dir=root))


def model_cache_dir(cache_root: Path, model_name: str, key: str) -> Path:
    # model_name is accepted for call-site clarity but deliberately excluded
    # from identity: renaming identical model bytes must retain the cache.
    return cache_root / "predictions" / key


def load_evaluation_cache(root: Path, key: str) -> dict[str, Any] | None:
    path = root / "metrics" / f"{key}.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(value, dict)
        or value.get("schema") != EVALUATION_CACHE_SCHEMA
        or value.get("key") != key
        or not isinstance(value.get("result"), dict)
    ):
        return None
    return dict(value["result"])


def save_evaluation_cache(root: Path, key: str, result: dict[str, Any]) -> None:
    _write_json_atomic(
        root / "metrics" / f"{key}.json",
        {
            "schema": EVALUATION_CACHE_SCHEMA,
            "key": key,
            "created_at_unix": time.time(),
            "result": result,
        },
    )


def _prediction_arrays(predictions: list[Prediction]) -> dict[str, np.ndarray]:
    count = len(predictions)
    boxes = np.full((count, 4), np.nan, dtype=np.float32)
    points = np.full((count, 2), np.nan, dtype=np.float32)
    radii = np.full(count, np.nan, dtype=np.float32)
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
    maximum = max((len(value) for value in extras), default=1)
    return {
        "class_ids": np.asarray([value.class_id for value in predictions], dtype=np.int32),
        "scores": np.asarray([value.score for value in predictions], dtype=np.float32),
        "boxes": boxes,
        "points": points,
        "radii": radii,
        "extras": np.asarray(extras, dtype=f"<U{maximum}"),
    }


def _predictions_from_arrays(data: Any, source: Path) -> list[Prediction]:
    required = {"class_ids", "scores", "boxes", "points", "radii", "extras"}
    if not required.issubset(data.files):
        raise DatasetValidationError(
            ValidationIssue(
                "Prediction cache shard is missing arrays",
                source=str(source),
                value=sorted(data.files),
            )
        )
    count = len(data["class_ids"])
    if any(len(data[key]) != count for key in required):
        raise DatasetValidationError(
            ValidationIssue("Prediction cache arrays have inconsistent lengths", source=str(source))
        )
    if data["boxes"].shape != (count, 4) or data["points"].shape != (count, 2):
        raise DatasetValidationError(
            ValidationIssue("Prediction cache geometry arrays have invalid shapes", source=str(source))
        )
    if not np.isfinite(data["scores"]).all():
        raise DatasetValidationError(
            ValidationIssue("Prediction scores contain non-finite values", source=str(source))
        )
    output: list[Prediction] = []
    for index in range(count):
        extra = json.loads(str(data["extras"][index]))
        box = (
            tuple(float(value) for value in data["boxes"][index])
            if np.isfinite(data["boxes"][index]).all()
            else None
        )
        point = (
            tuple(float(value) for value in data["points"][index])
            if np.isfinite(data["points"][index]).all()
            else None
        )
        radius = float(data["radii"][index]) if np.isfinite(data["radii"][index]) else None
        output.append(
            Prediction(
                class_id=int(data["class_ids"][index]),
                score=float(data["scores"][index]),
                bbox=box,
                point=point,
                radius=radius,
                polygon=[tuple(map(float, value)) for value in extra.get("polygon") or []] or None,
                polygons=[
                    [tuple(map(float, point_value)) for point_value in polygon]
                    for polygon in extra.get("polygons") or []
                ]
                or None,
                keypoints=[tuple(value) for value in extra.get("keypoints") or []] or None,
                metadata=extra.get("metadata") or {},
            )
        )
    return output


def load_package_cache(
    root: Path,
    cohort: Cohort,
    thresholds: tuple[float, ...],
    *,
    progress: bool = False,
) -> tuple[dict[float, dict[str, list[Prediction]]], int, bool]:
    manifest_path = root / "cache-manifest.json"
    if not manifest_path.is_file():
        return {}, 0, False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, 0, False
    if manifest.get("schema") != CACHE_SCHEMA or manifest.get("cohort_fingerprint") != cohort.fingerprint:
        return {}, 0, False
    loaded: dict[float, dict[str, list[Prediction]]] = {}
    shards = 0
    cache_progress = tqdm(
        total=len(thresholds) * len(cohort.records),
        desc="Loading prediction cache",
        unit="shard",
        disable=not progress,
    )
    try:
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
                cache_progress.update(1)
            if len(by_image) == len(cohort.records):
                loaded[float(threshold)] = by_image
    finally:
        cache_progress.close()
    complete = (
        set(loaded) == {float(value) for value in thresholds}
        and (root / "complete.json").is_file()
    )
    return loaded, shards, complete


def save_package_cache(
    root: Path,
    cohort: Cohort,
    payload: dict[str, Any],
    predictions: dict[float, dict[str, list[Prediction]]],
    *,
    progress: bool = False,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        root / "cache-manifest.json",
        {
            "schema": CACHE_SCHEMA,
            "cohort_fingerprint": cohort.fingerprint,
            "settings": to_jsonable(payload),
            "created_at_unix": time.time(),
        },
    )
    cache_progress = tqdm(
        total=len(predictions) * len(cohort.records),
        desc="Saving prediction cache",
        unit="shard",
        disable=not progress,
    )
    try:
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
                cache_progress.update(1)
    finally:
        cache_progress.close()
    _write_json_atomic(
        root / "complete.json",
        {
            "cohort_fingerprint": cohort.fingerprint,
            "thresholds": sorted(predictions),
            "completed_at_unix": time.time(),
        },
    )


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(to_jsonable(value), sort_keys=True, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_cached_predictions(
    values: list[Prediction],
    record: Any,
    cohort: Cohort,
    source: Path,
) -> None:
    for index, value in enumerate(values):
        if value.class_id not in cohort.classes:
            raise DatasetValidationError(
                ValidationIssue(
                    "Cached prediction has an unknown class ID",
                    source=str(source),
                    line=index + 1,
                    value=value.class_id,
                )
            )
        if not np.isfinite(value.score) or not 0 <= value.score <= 1:
            raise DatasetValidationError(
                ValidationIssue(
                    "Cached prediction score must be finite and in [0, 1]",
                    source=str(source),
                    line=index + 1,
                    value=value.score,
                )
            )
        if value.bbox is not None:
            x1, y1, x2, y2 = value.bbox
            if (
                not all(np.isfinite(value.bbox))
                or x2 < x1
                or y2 < y1
                or x1 < -1e-3
                or y1 < -1e-3
                or x2 > record.width + 1e-3
                or y2 > record.height + 1e-3
            ):
                raise DatasetValidationError(
                    ValidationIssue(
                        "Cached prediction box is invalid or outside the image",
                        source=str(source),
                        line=index + 1,
                        value=value.bbox,
                    )
                )
        if value.point is not None:
            x, y = value.point
            if not np.isfinite([x, y]).all() or not (
                -1e-3 <= x <= record.width + 1e-3
                and -1e-3 <= y <= record.height + 1e-3
            ):
                raise DatasetValidationError(
                    ValidationIssue(
                        "Cached prediction point is outside the image",
                        source=str(source),
                        line=index + 1,
                        value=value.point,
                    )
                )
        if cohort.task == "polo" and value.point is None:
            raise DatasetValidationError(
                ValidationIssue(
                    "POLO cache prediction is missing a point",
                    source=str(source),
                    line=index + 1,
                )
            )
        if cohort.task in {"detect", "segment", "pose"} and value.bbox is None:
            raise DatasetValidationError(
                ValidationIssue(
                    "Cached instance prediction is missing a box",
                    source=str(source),
                    line=index + 1,
                )
            )
