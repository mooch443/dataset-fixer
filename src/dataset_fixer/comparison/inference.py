from __future__ import annotations

import importlib.util
import math
import time
from collections.abc import Callable
from typing import Any

from tqdm.auto import tqdm

from ..errors import DatasetValidationError, ValidationIssue
from .types import Cohort, ModelSpec, Prediction


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
        if backend == "sahi":
            by_image = _run_sahi(
                spec, cohort, threshold, confidence_floor, device, progress, settings
            )
        else:
            by_image = _run_native(
                spec, cohort, threshold, confidence_floor, device, progress, settings
            )
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
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Native inference requires Ultralytics; install dataset-fixer[comparison]") from exc
    model = YOLO(str(spec.path))
    sources = [str(record.image_path) for record in cohort.records]
    kwargs: dict[str, Any] = {
        "source": sources,
        "imgsz": spec.resolution,
        "conf": confidence_floor,
        "iou": threshold,
        "verbose": False,
        "stream": True,
        "augment": bool(settings.get("augment", False)),
    }
    if device is not None:
        kwargs["device"] = device
    if settings.get("precision") == "half":
        kwargs["half"] = True
    iterator = model.predict(**kwargs)
    iterator = tqdm(iterator, total=len(sources), desc=f"{spec.name} native {threshold:g}", disable=not progress)
    output: dict[str, list[Prediction]] = {}
    for index, result in enumerate(iterator):
        if index >= len(cohort.records):
            raise DatasetValidationError(f"{spec.name} returned extra inference results")
        record = cohort.records[index]
        result_path = getattr(result, "path", None)
        if result_path is not None:
            from pathlib import Path

            if Path(result_path).expanduser().resolve() != record.image_path:
                raise DatasetValidationError(
                    ValidationIssue(
                        "Native inference reordered or substituted a cohort image",
                        source=spec.name,
                        value=str(result_path),
                        expected=str(record.image_path),
                    )
                )
        output[record.image_id] = _parse_native_result(result, cohort.task)
    return output


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
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_prediction, get_sliced_prediction
    except ImportError as exc:
        raise ImportError("SAHI inference requires the optional dataset-fixer[sahi] extra") from exc
    model = AutoDetectionModel.from_pretrained(
        model_type=str(settings.get("model_type", "ultralytics")),
        model_path=str(spec.path),
        confidence_threshold=confidence_floor,
        device=device or "cpu",
        image_size=spec.resolution,
    )
    mode = str(settings.get("sahi_mode", "sliced"))
    if mode not in {"standard", "sliced", "combined"}:
        raise ValueError("sahi_mode must be 'standard', 'sliced', or 'combined'")
    overlap_h = float(settings.get("overlap_height_ratio", settings.get("overlap", 0.2)))
    overlap_w = float(settings.get("overlap_width_ratio", settings.get("overlap", 0.2)))
    post_type = str(settings.get("postprocess_type", "GREEDYNMM")).upper()
    post_metric = str(settings.get("postprocess_match_metric", "IOS")).upper()
    output: dict[str, list[Prediction]] = {}
    iterator = tqdm(cohort.records, desc=f"{spec.name} SAHI {threshold:g}", disable=not progress)
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
                slice_height=int(settings.get("slice_height", spec.resolution)),
                slice_width=int(settings.get("slice_width", spec.resolution)),
                overlap_height_ratio=overlap_h,
                overlap_width_ratio=overlap_w,
                perform_standard_pred=mode == "combined",
                **common,
            )
        output[record.image_id] = [
            _sahi_prediction(value, cohort.task, post_type, post_metric, threshold)
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
