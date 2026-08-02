from __future__ import annotations

import importlib.util
import math
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from tqdm.auto import tqdm

from ..errors import DatasetValidationError, ValidationIssue
from .types import Cohort, ModelSpec, Prediction

if TYPE_CHECKING:
    from ..model import Model, ModelInput, PredictionTask


def sahi_available() -> bool:
    return importlib.util.find_spec("sahi") is not None


def resolve_backend(requested: str, task: str) -> str:
    requested = requested.lower()
    if requested not in {"auto", "native", "sahi"}:
        raise ValueError("inference must be 'auto', 'native', or 'sahi'")
    if task == "pose":
        if requested == "sahi":
            raise DatasetValidationError(
                ValidationIssue(
                    "SAHI cannot preserve the complete pose prediction schema",
                    expected="native inference for pose",
                    suggestion="use inference='native' or 'auto'",
                )
            )
        return "native"
    if requested == "auto":
        return "sahi" if sahi_available() else "native"
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
) -> tuple[dict[str, list[Prediction]], "PredictionTask"]:
    """Adapter entry point used by the public :class:`Model` API."""

    if backend == "sahi":
        if task is None:
            raise DatasetValidationError(
                ValidationIssue(
                    "SAHI prediction requires a known model task",
                    source=model.name,
                    expected="task metadata in args.yaml or Model(..., task=...)",
                )
            )
        values = _predict_sahi_inputs(
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
        return values, task  # type: ignore[return-value]
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
) -> tuple[dict[str, list[Prediction]], "PredictionTask"]:
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
    if detected_task not in {"detect", "segment", "pose", "polo"}:
        raise DatasetValidationError(
            ValidationIssue(
                "Could not determine the Ultralytics model task",
                source=model.name,
                value=detected_task,
                expected="detect, segment, pose, or polo",
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
    output: dict[str, list[Prediction]] = {}
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
    task: str,
    threshold: float,
    confidence_floor: float,
    resolution: int,
    device: str | None,
    progress: bool,
    settings: dict[str, Any],
) -> dict[str, list[Prediction]]:
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_prediction, get_sliced_prediction
    except ImportError as exc:
        raise ImportError("SAHI inference requires the optional dataset-fixer[sahi] extra") from exc
    model_type = str(settings.get("model_type", "ultralytics"))
    model = source_model._runtime_model(
        (
            "sahi",
            model_type,
            float(confidence_floor),
            device or "cpu",
            int(resolution),
        ),
        lambda: AutoDetectionModel.from_pretrained(
            model_type=model_type,
            model_path=str(source_model.path),
            confidence_threshold=confidence_floor,
            device=device or "cpu",
            image_size=resolution,
        ),
    )
    mode = str(settings.get("sahi_mode", "sliced"))
    if mode not in {"standard", "sliced", "combined"}:
        raise ValueError("sahi_mode must be 'standard', 'sliced', or 'combined'")
    overlap_h = float(settings.get("overlap_height_ratio", settings.get("overlap", 0.2)))
    overlap_w = float(settings.get("overlap_width_ratio", settings.get("overlap", 0.2)))
    post_type = str(settings.get("postprocess_type", "GREEDYNMM")).upper()
    post_metric = str(settings.get("postprocess_match_metric", "IOS")).upper()
    output: dict[str, list[Prediction]] = {}
    iterator = tqdm(
        inputs,
        desc=f"{source_model.name} SAHI {threshold:g}",
        disable=not progress,
    )
    for record in iterator:
        common = {
            "detection_model": model,
            "postprocess_type": post_type,
            "postprocess_match_metric": post_metric,
            "postprocess_match_threshold": threshold,
            "postprocess_class_agnostic": bool(settings.get("postprocess_class_agnostic", False)),
            "verbose": 0,
        }
        if mode == "standard":
            # SAHI's standard prediction entry point does not accept sliced
            # postprocessing parameters. The threshold is still part of the
            # configuration/cache identity and is reported as inapplicable.
            result = get_prediction(
                str(record.image_path), detection_model=model, verbose=0
            )
        else:
            result = get_sliced_prediction(
                str(record.image_path),
                slice_height=int(settings.get("slice_height", resolution)),
                slice_width=int(settings.get("slice_width", resolution)),
                overlap_height_ratio=overlap_h,
                overlap_width_ratio=overlap_w,
                perform_standard_pred=mode == "combined",
                **common,
            )
        output[record.image_id] = [
            _sahi_prediction(value, task, post_type, post_metric, threshold)
            for value in result.object_prediction_list
        ]
    return output


def _sahi_prediction(value: Any, task: str, post_type: str, post_metric: str, threshold: float) -> Prediction:
    bbox = tuple(map(float, value.bbox.to_xyxy()))
    score = float(value.score.value)
    class_id = int(value.category.id)
    polygon = None
    if task == "segment" and getattr(value, "mask", None) is not None:
        segmentation = getattr(value.mask, "segmentation", None)
        if segmentation and isinstance(segmentation[0], (list, tuple)):
            row = segmentation[0]
            polygon = [(float(row[i]), float(row[i + 1])) for i in range(0, len(row) - 1, 2)]
    point = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2) if task == "polo" else None
    return Prediction(
        class_id=class_id,
        score=score,
        bbox=bbox,
        point=point,
        polygon=polygon,
        metadata={
            "backend": "sahi",
            "source_box": bbox,
            "postprocess_type": post_type,
            "postprocess_match_metric": post_metric,
            "postprocess_threshold": threshold,
        },
    )


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
        return value.tolist()
    except AttributeError:
        return list(value)
