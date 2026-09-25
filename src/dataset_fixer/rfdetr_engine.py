"""Thin RF-DETR checkpoint and prediction adapters; shared records/rendering stay upstream."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class NotRFDETRCheckpoint(ValueError):
    """A file has no RF-DETR identity; another backend may support it."""


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    import torch
    try:
        saved = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise NotRFDETRCheckpoint(f"Cannot read RF-DETR checkpoint metadata: {path}") from exc
    if not isinstance(saved, dict):
        raise NotRFDETRCheckpoint(f"RF-DETR checkpoint metadata is missing: {path}. Load its dataset-fixer bundle instead.")
    raw_config = saved.get("model_config") or {}
    config = dict(raw_config) if isinstance(raw_config, dict) else {}
    args = saved.get("args") or {}
    args = args if isinstance(args, dict) else getattr(args, "__dict__", {})
    name = saved.get("model_name") or config.get("model_name")
    source = str(args.get("pretrain_weights") or path.name).lower().replace("_", "-")
    if not ("rfdetr_version" in saved or str(name).startswith("RFDETR") or "rf-detr" in source or "rfdetr" in source):
        raise NotRFDETRCheckpoint(f"Checkpoint has no RF-DETR identity: {path}")
    if not name:
        # The native release catalogue also identifies legacy run-file weights.
        from rfdetr.detr import _CHECKPOINT_MODEL_MAP_ENTRIES
        name = next((variant for key, variant in _CHECKPOINT_MODEL_MAP_ENTRIES if key in source), None)
    if not name:
        raise ValueError("RF-DETR checkpoint must identify its model variant")
    if not config:
        import rfdetr
        factory = getattr(rfdetr, name)._model_config_class
        config = factory().model_dump()
        config.update({key: value for key, value in args.items() if key in factory.model_fields})
    # Legacy metadata may predate alignment of the fine-tuned prediction heads.
    state = saved.get("model") or {}
    if isinstance(state.get("_kp_active_mask"), torch.Tensor):
        config["num_keypoints_per_class"] = [int(n) for n in state["_kp_active_mask"].sum(dim=1).tolist()]
    if isinstance(state.get("class_embed.weight"), torch.Tensor):
        config["num_classes"] = int(state["class_embed.weight"].shape[0]) - 1
    task = "pose" if config.get("use_grouppose_keypoints") else "segment" if config.get("segmentation_head") else "detect"
    schema = {}
    if args.get("class_names"):
        schema["classes"] = {str(i): value for i, value in enumerate(args["class_names"])}
    counts = config.get("num_keypoints_per_class")
    if counts:
        schema["keypoint_counts"] = counts
        pairs = args.get("keypoint_flip_pairs") or []
        if pairs and len(set(counts)) == 1:
            flip = list(range(counts[0]))
            for i, j in zip(pairs[::2], pairs[1::2]):
                flip[i], flip[j] = j, i
            schema["flip_idx"] = flip
    return {"framework": "rfdetr", "task": task, "model_name": name, "model_config": config, "checkpoint_schema": schema,
            "class_names": args.get("class_names"), "epoch": saved.get("epoch"),
            "dataset_schema": saved.get("dataset_schema"), "training_config": args}


def resolution_overrides(config: dict, resolution: int | None) -> dict:
    if resolution is None:
        return {}
    patch = int(config["patch_size"])
    block = patch * int(config["num_windows"])
    if resolution <= 0 or resolution % block:
        raise ValueError(f"RF-DETR resolution {resolution} must be divisible by {block}")
    changes = {"resolution": resolution}
    previous = config.get("resolution")
    pe = config.get("positional_encoding_size")
    # Preserve deliberately fixed PE grids; resize grids derived from resolution.
    if previous and (pe is None or pe == int(previous) // patch):
        changes["positional_encoding_size"] = resolution // patch
    return changes


def predict_inputs(model, inputs, *, resolution, confidence, device, progress, backend):
    if backend != "native":
        raise ValueError("RF-DETR currently supports inference='native'")
    import rfdetr
    import torch
    device = str(torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")))
    dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
    from tqdm.auto import tqdm
    from .comparison.types import Prediction
    key = ("rfdetr", resolution, device)
    native = model._runtime.get(key)
    if native is None:
        metadata = checkpoint_metadata(model.path)
        overrides = resolution_overrides(metadata["model_config"], resolution)
        overrides["device"] = device
        native = rfdetr.from_checkpoint(str(model.path), **overrides)
        # RF-DETR 1.8.3 reads the task here when decoding optimized tuple outputs,
        # but its ModelContext constructor omits the config (pose becomes masks).
        native.model.model_config = native.model_config
        # Native export + dtype casting avoid JIT startup costs for small evals.
        # Keep the original module intact; only the inference copy is optimized.
        with torch.inference_mode():
            native.optimize_for_inference(compile=False, dtype=dtype)
        model._runtime[key] = native
    task = model.task or checkpoint_metadata(model.path)["task"]
    output = {}
    for item in tqdm(inputs, desc="RF-DETR predictions", disable=not progress):
        with torch.inference_mode():
            result = native.predict(str(item.image_path), threshold=confidence, include_source_image=False)
        objects = []
        boxes = result.data["xyxy"] if task == "pose" else result.xyxy
        scores = result.detection_confidence if task == "pose" else result.confidence
        for i, box in enumerate(boxes):
            points = None
            polygons = None
            if task == "pose":
                points = [(float(x), float(y), float(c)) for (x, y), c in zip(result.xy[i], result.keypoint_confidence[i])]
            if task == "segment" and result.mask is not None:
                import cv2
                contours, _ = cv2.findContours(result.mask[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                polygons = [c.reshape(-1, 2).astype(float).tolist() for c in contours if len(c) >= 3]
            objects.append(Prediction(class_id=int(result.class_id[i]), score=float(scores[i]), bbox=tuple(map(float, box)),
                                      keypoints=points, polygons=polygons, polygon=max(polygons, key=len) if polygons else None,
                                      # Pose uncertainty fusion can legitimately amplify scores above 1.
                                      metadata={"backend": "rfdetr", "score_domain": "nonnegative" if task == "pose" else "probability"}))
        output[item.image_id] = objects
    return output, task, {"resolved_batch_size": 1, "backend": "rfdetr", "device": device,
                          "optimized_for_inference": True, "inference_dtype": str(dtype).removeprefix("torch.")}
