from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from ..errors import DatasetValidationError, ValidationIssue
from ..sahi_support import (
    SahiSettings,
    SahiTile,
    build_tile_manifest,
    class_map_probabilities,
    resolve_sahi_settings,
    sahi_available,
    stitch_probability_tiles,
)
from .types import Cohort, ModelSpec, Prediction

if TYPE_CHECKING:
    from ..model import Model, ModelInput, PredictionTask


_ULTRALYTICS_SAHI_BATCH_PIXELS = 4 * 1024 * 1024
_ULTRALYTICS_SAHI_MAX_BATCH_SIZE = 32


def resolve_backend(requested: str, task: str) -> str:
    requested = requested.lower()
    if requested not in {"native", "sahi"}:
        raise ValueError("inference must be 'native' or 'sahi'; 'auto' was removed")
    if requested == "sahi" and not sahi_available():
        raise ImportError("SAHI inference was requested but SAHI is not installed; install dataset-fixer[sahi]")
    return requested


def run_inference(
    spec: ModelSpec,
    cohort: Cohort,
    *,
    backend: str,
    thresholds: tuple[float, ...],
    confidence_floor: float,
    device: str | None,
    progress: bool,
    settings: dict[str, Any],
    existing: dict[float, dict[str, list[Prediction]]] | None = None,
    on_threshold: Callable[[float, dict[str, list[Prediction]]], None] | None = None,
) -> tuple[dict[float, dict[str, list[Prediction]]], dict[str, float]]:
    predictions = dict(existing or {})
    timings: dict[str, float] = {}
    for threshold in thresholds:
        threshold = float(threshold)
        if threshold in predictions:
            continue
        start = time.perf_counter()
        result = spec.resolved_model.predict(
            cohort,
            inference=backend,
            resolution=spec.resolution,
            confidence=confidence_floor,
            postprocess=threshold,
            device=device,
            progress=progress,
            settings=settings,
        )
        by_image = {
            record.image_id: list(record.objects)
            for record in result.records
        }
        _assert_exact_predictions(cohort, by_image, spec.name)
        predictions[threshold] = by_image
        timings[f"postprocess_{threshold:g}"] = time.perf_counter() - start
        if on_threshold:
            on_threshold(threshold, by_image)
    return predictions, timings


def _run_native(
    spec: ModelSpec,
    cohort: Cohort,
    threshold: float,
    confidence_floor: float,
    device: str | None,
    progress: bool,
    settings: dict[str, Any],
) -> dict[str, list[Prediction]]:
    result = spec.resolved_model.predict(
        cohort,
        inference="native",
        resolution=spec.resolution,
        confidence=confidence_floor,
        postprocess=threshold,
        device=device,
        progress=progress,
        settings=settings,
    )
    return {record.image_id: list(record.objects) for record in result.records}


def predict_model_inputs(
    model: "Model",
    inputs: tuple["ModelInput", ...],
    *,
    task: str | None,
    backend: str,
    resolution: int,
    confidence: float,
    postprocess: float,
    device: str | None,
    progress: bool,
    settings: dict[str, Any],
) -> tuple[dict[str, list[Prediction] | np.ndarray], "PredictionTask"]:
    """Adapter entry point used by the public :class:`Model` API."""

    if backend == "sahi":
        values, resolved_task = _predict_sahi_inputs(
            model,
            inputs,
            task=task,
            threshold=postprocess,
            confidence_floor=confidence,
            resolution=resolution,
            device=device,
            progress=progress,
            settings=settings,
        )
        return values, resolved_task
    return _predict_native_inputs(
        model,
        inputs,
        task=task,
        threshold=postprocess,
        confidence_floor=confidence,
        resolution=resolution,
        device=device,
        progress=progress,
        settings=settings,
    )


def _predict_native_inputs(
    model: "Model",
    inputs: tuple["ModelInput", ...],
    *,
    task: str | None,
    threshold: float,
    confidence_floor: float,
    resolution: int,
    device: str | None,
    progress: bool,
    settings: dict[str, Any],
) -> tuple[dict[str, list[Prediction] | np.ndarray], "PredictionTask"]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Native inference requires Ultralytics; install dataset-fixer[comparison]") from exc
    loaded = model._runtime_model(
        ("ultralytics-native",),
        lambda: YOLO(str(model.path)),
    )
    detected_task = task or model.task or getattr(loaded, "task", None)
    if detected_task is None:
        detected_task = getattr(getattr(loaded, "model", None), "task", None)
    detected_task = _canonical_task(detected_task)
    if detected_task not in {"detect", "segment", "pose", "polo", "semantic_segment"}:
        raise DatasetValidationError(
            ValidationIssue(
                "Could not determine the Ultralytics model task",
                source=model.name,
                value=detected_task,
                expected="detect, segment, pose, polo/locate, or semantic/semantic_segment",
                suggestion="pass task=... when constructing Model",
            )
        )
    kwargs: dict[str, Any] = {
        "imgsz": resolution,
        "conf": confidence_floor,
        "iou": threshold,
        "verbose": False,
        "stream": False,
        "augment": bool(settings.get("augment", False)),
    }
    if device is not None:
        kwargs["device"] = device
    if settings.get("precision") == "half":
        kwargs["half"] = True
    output: dict[str, list[Prediction] | np.ndarray] = {}
    iterator = tqdm(
        inputs,
        total=len(inputs),
        desc=f"{model.name} native {threshold:g}",
        disable=not progress,
    )
    for record in iterator:
        # Ultralytics converts a list of path strings to image arrays internally and
        # then reports synthetic paths such as ``image0.jpg``. Inference one image
        # at a time so result identity remains independently verifiable instead of
        # trusting positional ordering from a batched loader.
        results = loaded.predict(source=str(record.image_path), **kwargs)
        if len(results) != 1:
            raise DatasetValidationError(
                ValidationIssue(
                    "Native inference did not return exactly one result for a cohort image",
                    source=f"{model.name}: {record.image_path}",
                    value=len(results),
                    expected="exactly one result",
                )
            )
        result = results[0]
        result_path = getattr(result, "path", None)
        if result_path is not None:
            from pathlib import Path

            if Path(result_path).expanduser().resolve() != record.image_path:
                raise DatasetValidationError(
                    ValidationIssue(
                        "Native inference reordered or substituted a cohort image",
                        source=model.name,
                        value=str(result_path),
                        expected=str(record.image_path),
                    )
                )
        if detected_task == "semantic_segment":
            output[record.image_id] = _semantic_class_map(
                result,
                expected_shape=(record.height, record.width),
                source=f"{model.name}: {record.relative_path}",
            )
        else:
            output[record.image_id] = _parse_native_result(result, detected_task)
    return output, detected_task  # type: ignore[return-value]


def _parse_native_result(result: Any, task: str) -> list[Prediction]:
    if task == "polo":
        locations = getattr(result, "locations", None)
        if locations is None:
            return []
        xy = _tolist(locations.xy)
        scores = _tolist(locations.conf)
        classes = _tolist(locations.cls)
        radii = _tolist(getattr(locations, "radii", []))
        return [
            Prediction(
                class_id=int(classes[i]),
                score=float(scores[i]),
                point=(float(point[0]), float(point[1])),
                radius=float(radii[i]) if i < len(radii) else None,
                metadata={"backend": "native"},
            )
            for i, point in enumerate(xy)
        ]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    xyxy = _tolist(boxes.xyxy)
    scores = _tolist(boxes.conf)
    classes = _tolist(boxes.cls)
    polygons = list(getattr(getattr(result, "masks", None), "xy", []) or [])
    keypoint_rows = _tolist(getattr(getattr(result, "keypoints", None), "data", []))
    output: list[Prediction] = []
    for i, box in enumerate(xyxy):
        polygon = _tolist(polygons[i]) if i < len(polygons) else None
        keypoints = keypoint_rows[i] if i < len(keypoint_rows) else None
        output.append(
            Prediction(
                class_id=int(classes[i]),
                score=float(scores[i]),
                bbox=tuple(map(float, box[:4])),
                polygon=[tuple(map(float, point[:2])) for point in polygon] if polygon else None,
                polygons=(
                    [[tuple(map(float, point[:2])) for point in polygon]]
                    if polygon
                    else None
                ),
                keypoints=[tuple(map(float, point[:3])) for point in keypoints] if keypoints else None,
                metadata={"backend": "native"},
            )
        )
    return output


def _run_sahi(
    spec: ModelSpec,
    cohort: Cohort,
    threshold: float,
    confidence_floor: float,
    device: str | None,
    progress: bool,
    settings: dict[str, Any],
) -> dict[str, list[Prediction]]:
    result = spec.resolved_model.predict(
        cohort,
        inference="sahi",
        resolution=spec.resolution,
        confidence=confidence_floor,
        postprocess=threshold,
        device=device,
        progress=progress,
        settings=settings,
    )
    return {record.image_id: list(record.objects) for record in result.records}


def _predict_sahi_inputs(
    source_model: "Model",
    inputs: tuple["ModelInput", ...],
    *,
    task: str | None,
    threshold: float,
    confidence_floor: float,
    resolution: int,
    device: str | None,
    progress: bool,
    settings: dict[str, Any],
) -> tuple[dict[str, list[Prediction] | np.ndarray], "PredictionTask"]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "SAHI inference for supported models requires dataset-fixer[sahi]"
        ) from exc
    resolved = resolve_sahi_settings(settings, resolution=resolution)
    if resolved.model_type.lower() not in {
        "ultralytics",
        "polo",
        "polo26",
        "polov8",
        "locate",
    }:
        raise ValueError(
            "sahi_model_type must identify the Ultralytics adapter "
            "('ultralytics', 'polo', 'polo26', 'polov8', or 'locate')"
        )
    canonical_task = _canonical_task(task)
    loaded = source_model._runtime_model(
        ("ultralytics-sahi", device or "cpu", int(resolution)),
        lambda: YOLO(str(source_model.path)),
    )
    detected_task = canonical_task or _canonical_task(getattr(loaded, "task", None))
    if detected_task not in {"detect", "segment", "pose", "polo", "semantic_segment"}:
        raise DatasetValidationError(
            ValidationIssue(
                "Could not determine the SAHI model task",
                source=source_model.name,
                value=detected_task,
                expected="detect, segment, pose, polo/locate, or semantic",
            )
        )
    output: dict[str, list[Prediction] | np.ndarray] = {}
    iterator = tqdm(
        inputs,
        desc=f"{source_model.name} SAHI {threshold:g}",
        disable=not progress,
    )
    for record in iterator:
        with Image.open(record.image_path) as opened:
            source_image = opened.convert("RGB")
        if source_image.size != (record.width, record.height):
            raise DatasetValidationError(
                f"Prediction input dimensions changed while slicing {record.image_path}"
            )
        manifest = build_tile_manifest(
            width=record.width,
            height=record.height,
            settings=resolved,
        )
        raw_objects: list[Prediction] = []
        semantic_tiles: list[tuple[SahiTile, np.ndarray]] = []
        batch_size = max(
            1,
            min(
                _ULTRALYTICS_SAHI_MAX_BATCH_SIZE,
                _ULTRALYTICS_SAHI_BATCH_PIXELS // max(1, resolution * resolution),
            ),
        )
        for offset in range(0, len(manifest), batch_size):
            tile_batch = manifest[offset : offset + batch_size]
            crops = [source_image.crop(tile.box) for tile in tile_batch]
            results = _predict_ultralytics_tiles(
                loaded,
                crops,
                resolution=resolution,
                confidence=confidence_floor,
                postprocess=threshold,
                device=device,
                settings=settings,
                source=f"{source_model.name}:{record.relative_path}:tiles-{offset}-{offset + len(tile_batch) - 1}",
            )
            for tile, result in zip(tile_batch, results):
                if detected_task == "semantic_segment":
                    semantic_tiles.append(
                        (
                            tile,
                            _semantic_probabilities(
                                result,
                                expected_shape=(tile.height, tile.width),
                                num_classes=_model_class_count(loaded),
                                source=f"{source_model.name}:{record.relative_path}:tile-{tile.index}",
                            ),
                        )
                    )
                    continue
                tile_objects = _parse_native_result(result, detected_task)
                raw_objects.extend(
                    _shift_tile_prediction(
                        value,
                        tile,
                        loaded,
                        full_width=record.width,
                        full_height=record.height,
                    )
                    for value in tile_objects
                )
        if detected_task == "semantic_segment":
            probabilities = stitch_probability_tiles(
                width=record.width,
                height=record.height,
                tiles=semantic_tiles,
            )
            output[record.image_id] = np.argmax(probabilities, axis=0).astype(
                _class_map_dtype(probabilities.shape[0])
            )
        elif detected_task in {"pose", "polo"}:
            output[record.image_id] = _postprocess_payload_predictions(
                raw_objects,
                task=detected_task,
                threshold=threshold,
                settings=resolved,
            )
        else:
            output[record.image_id] = _postprocess_object_predictions(
                raw_objects,
                task=detected_task,
                width=record.width,
                height=record.height,
                threshold=threshold,
                settings=resolved,
            )
    return output, detected_task  # type: ignore[return-value]


def _sahi_prediction(
    value: Any,
    task: str,
    post_type: str,
    post_metric: str,
    threshold: float,
) -> Prediction:
    bbox = tuple(map(float, value.bbox.to_xyxy()))
    score = float(value.score.value)
    class_id = int(value.category.id)
    polygon = None
    polygons = None
    if task == "segment" and getattr(value, "mask", None) is not None:
        segmentation = getattr(value.mask, "segmentation", None)
        if segmentation and isinstance(segmentation[0], (list, tuple)):
            polygons = [
                [
                    (float(row[i]), float(row[i + 1]))
                    for i in range(0, len(row) - 1, 2)
                ]
                for row in segmentation
                if len(row) >= 6
            ]
            polygon = max(polygons, key=_polygon_area) if polygons else None
    return Prediction(
        class_id=class_id,
        score=score,
        bbox=bbox,
        polygon=polygon,
        polygons=polygons,
        metadata={
            "backend": "sahi",
            "source_box": bbox,
            "sahi_postprocess_type": post_type,
            "sahi_postprocess_match_metric": post_metric,
            "sahi_postprocess_threshold": threshold,
        },
    )


def _predict_ultralytics_tiles(
    model: Any,
    images: list[Image.Image],
    *,
    resolution: int,
    confidence: float,
    postprocess: float,
    device: str | None,
    settings: dict[str, Any],
    source: str,
) -> list[Any]:
    arrays = [
        np.ascontiguousarray(np.asarray(image, dtype=np.uint8)[:, :, ::-1])
        for image in images
    ]
    kwargs: dict[str, Any] = {
        "source": arrays[0] if len(arrays) == 1 else arrays,
        "imgsz": resolution,
        "conf": confidence,
        "iou": postprocess,
        "verbose": False,
        "stream": False,
        "augment": False,
    }
    if device is not None:
        kwargs["device"] = device
    if settings.get("precision") == "half":
        kwargs["half"] = True
    results = model.predict(**kwargs)
    if len(results) != len(images):
        raise DatasetValidationError(
            ValidationIssue(
                "SAHI tile batch did not return exactly one result per tile",
                source=source,
                value=len(results),
                expected=f"exactly {len(images)} ordered tile results",
            )
        )
    return list(results)


def _shift_tile_prediction(
    value: Prediction,
    tile: SahiTile,
    model: Any,
    *,
    full_width: int,
    full_height: int,
) -> Prediction:
    bbox = (
        (
            value.bbox[0] + tile.left,
            value.bbox[1] + tile.top,
            value.bbox[2] + tile.left,
            value.bbox[3] + tile.top,
        )
        if value.bbox is not None
        else None
    )
    point = (
        (value.point[0] + tile.left, value.point[1] + tile.top)
        if value.point is not None
        else None
    )
    polygons = value.polygons or ([value.polygon] if value.polygon else None)
    shifted_polygons = (
        [
            [(x + tile.left, y + tile.top) for x, y in polygon]
            for polygon in polygons
        ]
        if polygons
        else None
    )
    keypoints = (
        [
            (point_row[0] + tile.left, point_row[1] + tile.top, point_row[2])
            for point_row in value.keypoints
        ]
        if value.keypoints
        else None
    )
    radius = value.radius
    if point is not None and bbox is None:
        radius = radius if radius is not None else _model_polo_radius(model, value.class_id)
        proxy = max(float(radius) if radius is not None else 1.0, 1e-3)
        bbox = (
            max(0.0, point[0] - proxy),
            max(0.0, point[1] - proxy),
            min(float(full_width), point[0] + proxy),
            min(float(full_height), point[1] + proxy),
        )
    return Prediction(
        class_id=value.class_id,
        score=value.score,
        bbox=bbox,
        point=point,
        radius=radius,
        polygon=(
            max(shifted_polygons, key=_polygon_area) if shifted_polygons else None
        ),
        polygons=shifted_polygons,
        keypoints=keypoints,
        metadata={**value.metadata, "backend": "sahi", "tile": tile.box},
    )


def _postprocess_object_predictions(
    predictions: list[Prediction],
    *,
    task: str,
    width: int,
    height: int,
    threshold: float,
    settings: SahiSettings,
) -> list[Prediction]:
    if not predictions:
        return []
    try:
        from sahi.postprocess.combine import (
            GreedyNMMPostprocess,
            LSNMSPostprocess,
            NMMPostprocess,
            NMSPostprocess,
        )
        from sahi.prediction import ObjectPrediction
    except ImportError as exc:
        raise ImportError("SAHI inference requires dataset-fixer[sahi]") from exc
    classes = {
        "GREEDYNMM": GreedyNMMPostprocess,
        "NMM": NMMPostprocess,
        "NMS": NMSPostprocess,
        "LSNMS": LSNMSPostprocess,
    }
    objects = []
    for value in predictions:
        if value.bbox is None:
            raise DatasetValidationError("SAHI object prediction is missing its proxy box")
        polygons = value.polygons or ([value.polygon] if value.polygon else None)
        segmentation = (
            [[coordinate for point in polygon for coordinate in point] for polygon in polygons]
            if task == "segment" and polygons
            else None
        )
        objects.append(
            ObjectPrediction(
                bbox=list(value.bbox),
                category_id=value.class_id,
                category_name=str(value.class_id),
                score=value.score,
                segmentation=segmentation,
                full_shape=[height, width],
            )
        )
    processor = classes[settings.postprocess_type](
        match_threshold=threshold,
        match_metric=settings.postprocess_match_metric,
        class_agnostic=settings.postprocess_class_agnostic,
    )
    return [
        _sahi_prediction(
            value,
            task,
            settings.postprocess_type,
            settings.postprocess_match_metric,
            threshold,
        )
        for value in processor(objects)
    ]


def _postprocess_payload_predictions(
    predictions: list[Prediction],
    *,
    task: str,
    threshold: float,
    settings: SahiSettings,
) -> list[Prediction]:
    groups = _prediction_groups(
        predictions,
        postprocess_type=settings.postprocess_type,
        match_metric=settings.postprocess_match_metric,
        threshold=threshold,
        class_agnostic=settings.postprocess_class_agnostic,
    )
    output: list[Prediction] = []
    merge_payload = settings.postprocess_type in {"GREEDYNMM", "NMM"}
    for indices in groups:
        members = [predictions[index] for index in indices]
        winner = max(members, key=lambda value: value.score)
        if not merge_payload:
            output.append(winner)
            continue
        boxes = [value.bbox for value in members if value.bbox is not None]
        bbox = (
            (
                min(value[0] for value in boxes),
                min(value[1] for value in boxes),
                max(value[2] for value in boxes),
                max(value[3] for value in boxes),
            )
            if boxes
            else None
        )
        if task == "pose":
            lengths = {len(value.keypoints or []) for value in members}
            if len(lengths) != 1 or not lengths or 0 in lengths:
                raise DatasetValidationError(
                    ValidationIssue(
                        "Cannot merge SAHI pose predictions with inconsistent skeletons",
                        value=sorted(lengths),
                        expected="one non-zero keypoint count per merge group",
                    )
                )
            keypoints = [
                _merge_keypoint([value.keypoints[index] for value in members], members)
                for index in range(next(iter(lengths)))
            ]
            output.append(
                Prediction(
                    class_id=winner.class_id,
                    score=winner.score,
                    bbox=bbox,
                    keypoints=keypoints,
                    metadata=_merged_metadata(winner, members, settings, threshold),
                )
            )
        else:
            point = _weighted_point(members)
            radii = [(value.radius, value.score) for value in members if value.radius is not None]
            radius = (
                sum(float(value) * score for value, score in radii)
                / sum(score for _, score in radii)
                if radii and sum(score for _, score in radii) > 0
                else winner.radius
            )
            output.append(
                Prediction(
                    class_id=winner.class_id,
                    score=winner.score,
                    bbox=bbox,
                    point=point,
                    radius=radius,
                    metadata=_merged_metadata(winner, members, settings, threshold),
                )
            )
    return output


def _prediction_groups(
    predictions: list[Prediction],
    *,
    postprocess_type: str,
    match_metric: str,
    threshold: float,
    class_agnostic: bool,
) -> list[list[int]]:
    remaining = sorted(range(len(predictions)), key=lambda i: (-predictions[i].score, i))
    if postprocess_type == "NMM":
        groups: list[list[int]] = []
        unseen = set(remaining)
        for seed in remaining:
            if seed not in unseen:
                continue
            unseen.remove(seed)
            group = [seed]
            queue = [seed]
            while queue:
                current = queue.pop(0)
                matches = [
                    candidate
                    for candidate in remaining
                    if candidate in unseen
                    and _payload_match(
                        predictions[current],
                        predictions[candidate],
                        match_metric,
                        threshold,
                        class_agnostic,
                    )
                ]
                for candidate in matches:
                    unseen.remove(candidate)
                    group.append(candidate)
                    queue.append(candidate)
            groups.append(group)
        return groups
    groups = []
    while remaining:
        winner = remaining.pop(0)
        matches = [
            candidate
            for candidate in remaining
            if _payload_match(
                predictions[winner],
                predictions[candidate],
                match_metric,
                threshold,
                class_agnostic,
            )
        ]
        remaining = [candidate for candidate in remaining if candidate not in matches]
        groups.append(
            [winner, *matches]
            if postprocess_type == "GREEDYNMM"
            else [winner]
        )
    return groups


def _payload_match(
    left: Prediction,
    right: Prediction,
    metric: str,
    threshold: float,
    class_agnostic: bool,
) -> bool:
    if not class_agnostic and left.class_id != right.class_id:
        return False
    if left.bbox is None or right.bbox is None:
        return False
    lx1, ly1, lx2, ly2 = left.bbox
    rx1, ry1, rx2, ry2 = right.bbox
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    denominator = (
        min(left_area, right_area)
        if metric == "IOS"
        else left_area + right_area - intersection
    )
    return denominator > 0 and intersection / denominator >= threshold


def _merge_keypoint(
    points: list[tuple[float, float, float | None]],
    members: list[Prediction],
) -> tuple[float, float, float | None]:
    rows = []
    for point, member in zip(points, members):
        confidence = point[2] if len(point) > 2 else None
        keypoint_weight = max(float(confidence), 0.0) if confidence is not None else 1.0
        weight = max(float(member.score), 0.0) * keypoint_weight
        if weight > 0 and math.isfinite(point[0]) and math.isfinite(point[1]):
            rows.append((point, weight))
    if not rows:
        return points[0]
    denominator = sum(weight for _, weight in rows)
    x = sum(point[0] * weight for point, weight in rows) / denominator
    y = sum(point[1] * weight for point, weight in rows) / denominator
    confidences = [
        (float(point[2]), weight)
        for point, weight in rows
        if len(point) > 2 and point[2] is not None
    ]
    confidence = (
        sum(value * weight for value, weight in confidences)
        / sum(weight for _, weight in confidences)
        if confidences
        else None
    )
    return x, y, confidence


def _weighted_point(members: list[Prediction]) -> tuple[float, float]:
    rows = [(value.point, max(value.score, 0.0)) for value in members if value.point]
    denominator = sum(weight for _, weight in rows)
    if not rows:
        raise DatasetValidationError("SAHI POLO merge group contains no points")
    if denominator <= 0:
        return rows[0][0]
    return (
        sum(point[0] * weight for point, weight in rows) / denominator,
        sum(point[1] * weight for point, weight in rows) / denominator,
    )


def _merged_metadata(
    winner: Prediction,
    members: list[Prediction],
    settings: SahiSettings,
    threshold: float,
) -> dict[str, Any]:
    return {
        **winner.metadata,
        "backend": "sahi",
        "merged_predictions": len(members),
        "sahi_postprocess_type": settings.postprocess_type,
        "sahi_postprocess_match_metric": settings.postprocess_match_metric,
        "sahi_postprocess_threshold": threshold,
    }


def _semantic_class_map(
    result: Any,
    *,
    expected_shape: tuple[int, int],
    source: str,
) -> np.ndarray:
    semantic = getattr(result, "semantic_mask", None)
    raw = getattr(semantic, "data", semantic)
    if raw is None:
        raise DatasetValidationError(
            ValidationIssue(
                "Semantic model result has no semantic_mask",
                source=source,
                expected="an Ultralytics task='semantic' result",
            )
        )
    values = np.asarray(_tolist(raw))
    values = np.squeeze(values)
    if values.shape != expected_shape:
        raise DatasetValidationError(
            ValidationIssue(
                "Semantic model result dimensions do not match its source image",
                source=source,
                value=values.shape,
                expected=str(expected_shape),
            )
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values != np.floor(values)):
        raise DatasetValidationError(
            ValidationIssue("Semantic model result contains invalid class IDs", source=source)
        )
    return values.astype(_class_map_dtype(int(values.max(initial=0)) + 1))


def _semantic_probabilities(
    result: Any,
    *,
    expected_shape: tuple[int, int],
    num_classes: int | None,
    source: str,
) -> np.ndarray:
    probabilities = getattr(result, "semantic_probabilities", None)
    logits = getattr(result, "semantic_logits", None)
    raw = probabilities if probabilities is not None else logits
    if raw is None:
        return class_map_probabilities(
            _semantic_class_map(result, expected_shape=expected_shape, source=source),
            num_classes=num_classes,
        )
    values = np.asarray(_tolist(raw), dtype=np.float32)
    while values.ndim > 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim == 2:
        values = values[None]
    if values.ndim != 3 or values.shape[-2:] != expected_shape:
        raise DatasetValidationError(
            ValidationIssue(
                "Semantic probability/logit dimensions do not match their tile",
                source=source,
                value=values.shape,
                expected=f"(classes, {expected_shape[0]}, {expected_shape[1]})",
            )
        )
    if logits is not None:
        if values.shape[0] == 1:
            foreground = 1.0 / (1.0 + np.exp(-values[0]))
            values = np.stack((1.0 - foreground, foreground))
        else:
            values = values - np.max(values, axis=0, keepdims=True)
            values = np.exp(values)
            values /= np.sum(values, axis=0, keepdims=True)
    elif values.shape[0] == 1:
        values = np.stack((1.0 - values[0], values[0]))
    if not np.all(np.isfinite(values)):
        raise DatasetValidationError(
            ValidationIssue("Semantic probabilities contain non-finite values", source=source)
        )
    return values


def _model_class_count(model: Any) -> int | None:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return len(names)
    if isinstance(names, (list, tuple)):
        return len(names)
    return None


def _model_polo_radius(model: Any, class_id: int) -> float | None:
    radii = getattr(model, "radii", None)
    if radii is None:
        radii = getattr(getattr(model, "predictor", None), "radii", None)
    if isinstance(radii, dict):
        value = radii.get(class_id, radii.get(str(class_id)))
        if isinstance(value, dict):
            value = value.get("radius", value.get("value", value.get("r")))
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _canonical_task(value: Any) -> Any:
    if value is None:
        return None
    normalized = str(value).lower()
    return {"locate": "polo", "semantic": "semantic_segment"}.get(
        normalized, normalized
    )


def _class_map_dtype(classes: int) -> Any:
    return np.uint8 if classes <= 256 else np.uint16


def _polygon_area(polygon: list[tuple[float, float]]) -> float:
    if len(polygon) < 3:
        return 0.0
    return abs(
        sum(
            polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
            - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
            for index in range(len(polygon))
        )
    ) / 2


def _assert_exact_predictions(cohort: Cohort, values: dict[str, list[Prediction]], model_name: str) -> None:
    expected = [record.image_id for record in cohort.records]
    actual = list(values)
    if actual != expected:
        missing = [item for item in expected if item not in values]
        added = [item for item in actual if item not in set(expected)]
        raise DatasetValidationError(
            ValidationIssue(
                "Model did not return the exact frozen evaluation cohort in order",
                source=model_name,
                value={"missing": missing[:5], "added": added[:5], "reordered": set(actual) == set(expected)},
                expected=f"{len(expected)} ordered image results",
            )
        )
    issues: list[ValidationIssue] = []
    for record in cohort.records:
        for index, prediction in enumerate(values[record.image_id], start=1):
            if prediction.class_id not in cohort.classes:
                issues.append(ValidationIssue("Prediction uses a class outside the frozen schema", source=f"{model_name}:{record.relative_path}", line=index, value=prediction.class_id))
            if not math.isfinite(prediction.score) or not 0 <= prediction.score <= 1:
                issues.append(ValidationIssue("Prediction score is not finite and in [0, 1]", source=f"{model_name}:{record.relative_path}", line=index, value=prediction.score))
            if cohort.task == "polo" and prediction.point is None:
                issues.append(ValidationIssue("POLO prediction has no point", source=f"{model_name}:{record.relative_path}", line=index))
            if cohort.task in {"detect", "segment", "pose"} and prediction.bbox is None:
                issues.append(ValidationIssue("Instance prediction has no box", source=f"{model_name}:{record.relative_path}", line=index))
    if issues:
        raise DatasetValidationError(issues)


def _tolist(value: Any) -> list[Any]:
    if value is None:
        return []
    try:
        value = value.detach().cpu()
    except AttributeError:
        pass
    try:
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    except AttributeError:
        return list(value)
