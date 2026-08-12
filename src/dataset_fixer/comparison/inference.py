from __future__ import annotations

import gc
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image
from tqdm import tqdm

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


_ULTRALYTICS_AUTO_BATCH_PIXELS = 64 * 1024 * 1024
_ULTRALYTICS_AUTO_BATCH_MAX = 128


@dataclass(frozen=True)
class SemanticOutput:
    """One semantic class map plus reusable canonical foreground scores."""

    class_map: np.ndarray
    foreground_probability: np.ndarray | None
    probability_source: str


_SEMANTIC_PROBABILITY_PREDICTOR: type[Any] | None = None


def _semantic_probability_predictor() -> type[Any]:
    """Return an Ultralytics semantic predictor that retains scaled logits."""

    global _SEMANTIC_PROBABILITY_PREDICTOR
    if _SEMANTIC_PROBABILITY_PREDICTOR is not None:
        return _SEMANTIC_PROBABILITY_PREDICTOR

    import torch
    import torch.nn.functional as torch_functional
    from ultralytics.models.yolo.semantic.predict import (
        SemanticSegmentationPredictor,
    )
    from ultralytics.utils import ops

    class ProbabilitySemanticSegmentationPredictor(
        SemanticSegmentationPredictor
    ):
        """Preserve pre-argmax semantic logits on each Results object."""

        def postprocess(self, preds: Any, img: Any, orig_imgs: Any) -> list[Any]:
            raw = preds[0] if isinstance(preds, (tuple, list)) else preds
            originals = (
                orig_imgs
                if isinstance(orig_imgs, list)
                else ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
            )
            output = super().postprocess(preds, img, orig_imgs)
            for prediction, original, result in zip(
                raw,
                originals,
                output,
                strict=True,
            ):
                if prediction.ndim == 2:
                    # Exported graphs with an in-graph ArgMax have already
                    # destroyed the score information and are not calibratable.
                    continue
                logits = prediction[None].float()
                if logits.shape[2:] != img.shape[2:]:
                    logits = torch_functional.interpolate(
                        logits,
                        img.shape[2:],
                        mode="bilinear",
                    )
                logits = ops.scale_masks(logits, original.shape[:2])[0]
                result.semantic_logits = logits.detach().to(
                    device="cpu",
                    dtype=torch.float16,
                )
            return output

    ProbabilitySemanticSegmentationPredictor.__name__ = (
        "DatasetFixerProbabilitySemanticSegmentationPredictor"
    )
    _SEMANTIC_PROBABILITY_PREDICTOR = ProbabilitySemanticSegmentationPredictor
    return ProbabilitySemanticSegmentationPredictor


def _predict_ultralytics(
    model: Any,
    *,
    source: Any,
    semantic: bool,
    options: dict[str, Any],
) -> list[Any]:
    """Predict while retaining semantic score maps when the backend has them."""

    if not semantic:
        return list(model.predict(source=source, **options))
    predictor = _semantic_probability_predictor()
    if not isinstance(getattr(model, "predictor", None), predictor):
        model.predictor = None
    return list(model.predict(source=source, predictor=predictor, **options))


def resolve_backend(requested: str, task: str) -> str:
    requested = requested.lower()
    if requested not in {"native", "sahi"}:
        raise ValueError("inference must be 'native' or 'sahi'; 'auto' was removed")
    if requested == "sahi" and not sahi_available():
        raise ImportError("SAHI inference was requested but SAHI is unavailable; reinstall dataset-fixer")
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
            # Comparison owns its schema-5 threshold shards. Avoid a second
            # package-default direct-prediction cache for this internal call.
            prediction_cache=False,
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
        prediction_cache=False,
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
    batch_size: int,
    foreground_probability_threshold: float | None = None,
) -> tuple[
    dict[str, list[Prediction] | SemanticOutput],
    "PredictionTask",
    dict[str, Any],
]:
    """Adapter entry point used by the public :class:`Model` API."""

    if backend == "sahi":
        return _predict_sahi_inputs(
            model,
            inputs,
            task=task,
            threshold=postprocess,
            confidence_floor=confidence,
            resolution=resolution,
            device=device,
            progress=progress,
            settings=settings,
            batch_size=batch_size,
            foreground_probability_threshold=foreground_probability_threshold,
        )
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
        batch_size=batch_size,
        foreground_probability_threshold=foreground_probability_threshold,
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
    batch_size: int,
    foreground_probability_threshold: float | None = None,
) -> tuple[
    dict[str, list[Prediction] | SemanticOutput],
    "PredictionTask",
    dict[str, Any],
]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Native inference requires Ultralytics; reinstall dataset-fixer") from exc
    runtime_key = ("ultralytics", device or "auto")
    reused_runtime = runtime_key in model._runtime
    loaded = model._runtime_model(
        runtime_key,
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
    output: dict[str, list[Prediction] | SemanticOutput] = {}
    progress_bar = tqdm(
        total=len(inputs),
        desc=f"{model.name} native {threshold:g}",
        unit="image",
        disable=not progress,
    )
    batch_key = ("ultralytics-batch", "native", device or "auto", int(resolution))
    preferred_batch = model._runtime.get(batch_key) if batch_size == -1 else None

    def predict_batch(batch: list["ModelInput"]) -> list[Any]:
        paths = [str(record.image_path) for record in batch]
        source: str | list[str] = paths[0] if len(paths) == 1 else paths
        results = _predict_ultralytics(
            loaded,
            source=source,
            semantic=detected_task == "semantic_segment",
            options=kwargs,
        )
        if len(results) != len(batch):
            raise DatasetValidationError(
                ValidationIssue(
                    "Native inference did not return exactly one ordered result per cohort image",
                    source=model.name,
                    value=len(results),
                    expected=f"exactly {len(batch)} results",
                )
            )
        for record, result in zip(batch, results):
            result_path = getattr(result, "path", None)
            # Ultralytics preserves real path sources. Array-backed exporters
            # may report synthetic paths; only validate paths that resolve to
            # an existing file so positional batching remains supported.
            if result_path is None:
                continue
            from pathlib import Path

            reported = Path(str(result_path)).expanduser()
            if reported.exists() and reported.resolve() != record.image_path:
                raise DatasetValidationError(
                    ValidationIssue(
                        "Native inference reordered or substituted a cohort image",
                        source=model.name,
                        value=str(result_path),
                        expected=str(record.image_path),
                    )
                )
        return results

    def consume_batch(batch: list["ModelInput"], results: list[Any]) -> None:
        for record, result in zip(batch, results):
            if detected_task == "semantic_segment":
                output[record.image_id] = _semantic_output(
                    result,
                    expected_shape=(record.height, record.width),
                    num_classes=_model_class_count(loaded),
                    threshold=foreground_probability_threshold,
                    source=f"{model.name}: {record.relative_path}",
                )
            else:
                output[record.image_id] = _parse_native_result(result, detected_task)

    try:
        telemetry = _adaptive_batches(
            list(inputs),
            predict_batch,
            consume_batch,
            requested=batch_size,
            device=device,
            resolution=resolution,
            progress_bar=progress_bar,
            source=model.name,
            preferred=preferred_batch,
        )
    finally:
        progress_bar.close()
    telemetry["runtime_reused"] = reused_runtime
    if batch_size == -1:
        model._runtime[batch_key] = telemetry["resolved_batch_size"]
    return output, detected_task, telemetry  # type: ignore[return-value]


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


def _adaptive_batches(
    items: list[Any],
    predict: Callable[[list[Any]], list[Any]],
    consume: Callable[[list[Any], list[Any]], None],
    *,
    requested: int,
    device: str | None,
    resolution: int,
    progress_bar: Any,
    source: str,
    preferred: int | None = None,
) -> dict[str, Any]:
    """Run ordered official-API batches with recoverable OOM backoff.

    Ultralytics documents ``batch=-1`` for training AutoBatch, not prediction.
    Prediction therefore probes a resolution-aware cohort batch here and
    halves only the failed chunk. Successful chunks are immediately converted
    to lightweight package records so result tensors do not accumulate on the
    accelerator.
    """

    if requested == -1:
        size = (
            min(len(items), int(preferred))
            if preferred is not None
            else min(
                len(items),
                _ULTRALYTICS_AUTO_BATCH_MAX,
                max(
                    1,
                    _ULTRALYTICS_AUTO_BATCH_PIXELS
                    // max(1, resolution * resolution),
                ),
            )
        )
        if str(device).lower() == "cpu":
            size = min(size, 32)
    else:
        size = min(len(items), requested, _ULTRALYTICS_AUTO_BATCH_MAX)
    size = max(1, size)
    initial = size
    retries = 0
    offset = 0
    while offset < len(items):
        active = items[offset : offset + size]
        try:
            results = predict(active)
            consume(active, results)
            del results
        except BaseException as error:
            if not _is_out_of_memory(error):
                raise
            retries += 1
            _clear_accelerator_memory(device)
            if size <= 1:
                raise DatasetValidationError(
                    ValidationIssue(
                        "Inference exhausted accelerator memory at batch size 1",
                        source=source,
                        value=str(error),
                        suggestion=(
                            "use a smaller model/input size or select a device with more memory"
                        ),
                    )
                ) from error
            size = max(1, size // 2)
            continue
        offset += len(active)
        progress_bar.update(len(active))
    return {
        "requested_batch_size": requested,
        "initial_batch_size": initial,
        "resolved_batch_size": size,
        "oom_retries": retries,
        "inference_device": device,
    }


def _is_out_of_memory(error: BaseException) -> bool:
    current: BaseException | None = error
    while current is not None:
        name = type(current).__name__.lower()
        message = str(current).lower()
        if "outofmemory" in name or any(
            marker in message
            for marker in (
                "out of memory",
                "mps backend out of memory",
                "failed to allocate",
                "cannot allocate memory",
                "can't allocate memory",
                "insufficient memory",
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _clear_accelerator_memory(device: str | None) -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    selected = str(device or "").lower()
    if selected in {"", "cuda"} and torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch, "mps", None)
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(
        mps_backend is not None
        and hasattr(mps_backend, "is_available")
        and mps_backend.is_available()
    )
    if (
        selected in {"", "mps"}
        and mps_available
        and mps is not None
        and hasattr(mps, "empty_cache")
    ):
        try:
            mps.empty_cache()
        except RuntimeError:
            pass


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
        prediction_cache=False,
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
    batch_size: int,
    foreground_probability_threshold: float | None = None,
) -> tuple[
    dict[str, list[Prediction] | SemanticOutput],
    "PredictionTask",
    dict[str, Any],
]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "SAHI inference for supported models requires dataset-fixer"
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
    runtime_key = ("ultralytics", device or "auto")
    reused_runtime = runtime_key in source_model._runtime
    loaded = source_model._runtime_model(
        runtime_key,
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
    output: dict[str, list[Prediction] | SemanticOutput] = {}
    manifests = {
        record.image_id: build_tile_manifest(
            width=record.width,
            height=record.height,
            settings=resolved,
        )
        for record in inputs
    }
    descriptors = [
        (record, tile)
        for record in inputs
        for tile in manifests[record.image_id]
    ]
    raw_by_image: dict[str, list[Prediction]] = {
        record.image_id: [] for record in inputs
    }
    semantic_by_image: dict[str, list[tuple[SahiTile, np.ndarray]]] = {
        record.image_id: [] for record in inputs
    }
    semantic_probability_sources: dict[str, set[str]] = {
        record.image_id: set() for record in inputs
    }
    tile_progress = tqdm(
        total=len(descriptors),
        desc=f"{source_model.name} SAHI tiles",
        unit="tile",
        disable=not progress,
    )
    batch_key = (
        "ultralytics-batch",
        "sahi",
        device or "auto",
        int(resolution),
    )
    preferred_batch = (
        source_model._runtime.get(batch_key) if batch_size == -1 else None
    )

    def predict_tiles(batch: list[tuple["ModelInput", SahiTile]]) -> list[Any]:
        images: dict[str, Image.Image] = {}
        crops: list[Image.Image] = []
        try:
            for record, tile in batch:
                if record.image_id not in images:
                    with Image.open(record.image_path) as opened:
                        source_image = opened.convert("RGB")
                    if source_image.size != (record.width, record.height):
                        raise DatasetValidationError(
                            f"Prediction input dimensions changed while slicing {record.image_path}"
                        )
                    images[record.image_id] = source_image
                crops.append(images[record.image_id].crop(tile.box))
            return _predict_ultralytics_tiles(
                loaded,
                crops,
                resolution=resolution,
                confidence=confidence_floor,
                postprocess=threshold,
                device=device,
                settings=settings,
                source=f"{source_model.name}:SAHI-tile-batch",
            )
        finally:
            for image in crops:
                image.close()
            for image in images.values():
                image.close()

    def consume_tiles(
        batch: list[tuple["ModelInput", SahiTile]],
        results: list[Any],
    ) -> None:
        for (record, tile), result in zip(batch, results):
            if detected_task == "semantic_segment":
                if getattr(result, "semantic_probabilities", None) is not None:
                    semantic_probability_sources[record.image_id].add(
                        "model-probabilities"
                    )
                elif getattr(result, "semantic_logits", None) is not None:
                    semantic_probability_sources[record.image_id].add("model-logits")
                else:
                    semantic_probability_sources[record.image_id].add(
                        "class-map-fallback"
                    )
                semantic_by_image[record.image_id].append(
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
            raw_by_image[record.image_id].extend(
                _shift_tile_prediction(
                    value,
                    tile,
                    loaded,
                    full_width=record.width,
                    full_height=record.height,
                )
                for value in tile_objects
            )

    try:
        telemetry = _adaptive_batches(
            descriptors,
            predict_tiles,
            consume_tiles,
            requested=batch_size,
            device=device,
            resolution=resolution,
            progress_bar=tile_progress,
            source=source_model.name,
            preferred=preferred_batch,
        )
    finally:
        tile_progress.close()

    image_progress = tqdm(
        inputs,
        desc=f"{source_model.name} SAHI images",
        unit="image",
        disable=not progress,
    )
    for record in image_progress:
        raw_objects = raw_by_image[record.image_id]
        if detected_task == "semantic_segment":
            probabilities = stitch_probability_tiles(
                width=record.width,
                height=record.height,
                tiles=semantic_by_image[record.image_id],
            )
            sources = semantic_probability_sources[record.image_id]
            probability_source = (
                next(iter(sources))
                if len(sources) == 1
                else "mixed:" + ",".join(sorted(sources))
            )
            output[record.image_id] = _semantic_output_from_probabilities(
                probabilities,
                threshold=foreground_probability_threshold,
                probability_source=probability_source,
                retain_probability="class-map-fallback" not in sources,
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
    telemetry["runtime_reused"] = reused_runtime
    if batch_size == -1:
        source_model._runtime[batch_key] = telemetry["resolved_batch_size"]
    telemetry["sahi_tiles"] = len(descriptors)
    telemetry["sahi_cross_image_batching"] = True
    return output, detected_task, telemetry  # type: ignore[return-value]


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
    results = _predict_ultralytics(
        model,
        source=kwargs.pop("source"),
        semantic=_canonical_task(getattr(model, "task", None))
        == "semantic_segment",
        options=kwargs,
    )
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
        raise ImportError("SAHI inference requires SAHI; reinstall dataset-fixer") from exc
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
            binary_logits = values[0]
            foreground = np.empty_like(binary_logits, dtype=np.float32)
            nonnegative = binary_logits >= 0
            foreground[nonnegative] = 1.0 / (
                1.0 + np.exp(-binary_logits[nonnegative])
            )
            negative_exp = np.exp(binary_logits[~nonnegative])
            foreground[~nonnegative] = negative_exp / (1.0 + negative_exp)
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


def _semantic_output(
    result: Any,
    *,
    expected_shape: tuple[int, int],
    num_classes: int | None,
    threshold: float | None,
    source: str,
) -> SemanticOutput:
    """Normalize a semantic result without pretending a class map is calibrated."""

    has_probabilities = getattr(result, "semantic_probabilities", None) is not None
    has_logits = getattr(result, "semantic_logits", None) is not None
    probabilities = _semantic_probabilities(
        result,
        expected_shape=expected_shape,
        num_classes=num_classes,
        source=source,
    )
    probability_source = (
        "model-probabilities"
        if has_probabilities
        else "model-logits"
        if has_logits
        else "class-map-fallback"
    )
    return _semantic_output_from_probabilities(
        probabilities,
        threshold=threshold,
        probability_source=probability_source,
        retain_probability=has_probabilities or has_logits,
    )


def _semantic_output_from_probabilities(
    probabilities: np.ndarray,
    *,
    threshold: float | None,
    probability_source: str,
    retain_probability: bool = True,
) -> SemanticOutput:
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] < 2:
        raise DatasetValidationError(
            "Semantic probabilities must have at least background and foreground channels"
        )
    argmax = np.argmax(values, axis=0)
    if threshold is None:
        class_map = argmax.astype(_class_map_dtype(values.shape[0]))
    else:
        foreground = 1.0 - values[0]
        foreground_class = np.argmax(values[1:], axis=0) + 1
        class_map = np.where(
            foreground >= float(threshold),
            foreground_class,
            0,
        ).astype(_class_map_dtype(values.shape[0]))
    foreground_probability = (
        np.asarray(1.0 - values[0], dtype=np.float32)
        if retain_probability
        else None
    )
    return SemanticOutput(
        class_map=class_map,
        foreground_probability=foreground_probability,
        probability_source=probability_source,
    )


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
