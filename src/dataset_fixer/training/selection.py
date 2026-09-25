"""Resolve backend capabilities without maintaining a second model catalogue."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import ModelTypes, TrainingConfig
from ..model_sources import resolve_model_source
from ..sources import sha256_progress


@dataclass
class Selection:
    family: ModelTypes
    task: str
    name: str
    weights: Path | None = None
    resume: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


def infer_task(dataset, explicit=None):
    from ..model import _normalize_prediction_task
    inferred = "semantic_segment" if dataset.format == "semantic_masks" else dataset.task.value
    selected = _normalize_prediction_task(getattr(explicit, "value", explicit)) if explicit is not None else inferred
    if selected != inferred and not (inferred == "segment" and selected == "semantic_segment"):
        raise ValueError(f"model_type={selected!r} is incompatible with dataset task={inferred!r}")
    return selected


def _yolo_name(task, version, size):
    from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES
    suffix = {"detect": "", "segment": "-seg", "pose": "-pose", "semantic_segment": "-sem", "polo": "-locate"}[task]
    candidates = []
    for name in GITHUB_ASSETS_NAMES:
        match = re.fullmatch(r"yolo(v?)(\d+)([a-z]+)(-[a-z]+)?\.pt", name)
        if match and (match[4] or "") == suffix:
            if version is None or match[2] == str(version):
                if size is None or match[3] == str(size):
                    candidates.append((int(match[2]), match[3], name))
    if not candidates:
        raise ValueError(f"Installed Ultralytics has no pretrained YOLO version={version}, size={size}, task={task}; supply compatible weights or install a backend offering it")
    newest = max(v for v, _, _ in candidates)
    choices = sorted((s, name) for v, s, name in candidates if v == newest)
    return next((name for s, name in choices if s == "n"), choices[0][1])


def _rf_variant(task, size):
    import rfdetr
    if task == "pose":
        if size not in {None, "preview", "x", "xl"}:
            raise ValueError("RF-DETR pose currently offers size='preview'")
        return "RFDETRKeypointPreview"
    names = {"n": "Nano", "s": "Small", "m": "Medium", "l": "Large", "x": "XLarge", "xl": "XLarge", "2xl": "2XLarge"}
    if size is None:
        size = "s"
    if size not in names:
        raise ValueError(f"Unknown RF-DETR size: {size}")
    name = "RFDETR" + ("Seg" if task == "segment" else "") + names[size]
    if not hasattr(rfdetr, name):
        raise ValueError(f"Installed RF-DETR does not offer {name}")
    return name


def select(dataset, *, type=None, version=None, s=None, model_type=None, weights=None, resume=None, config=TrainingConfig()):
    if weights is not None and resume is not None:
        raise ValueError("weights initializes a new run; resume continues one. Supply only one.")
    task = infer_task(dataset, model_type)
    if type == ModelTypes.NNUNET and task == "segment" and model_type is None:
        task = "semantic_segment"
    selection = None
    source = weights if weights is not None else resume
    if source is not None:
        resolved = resolve_model_source(source, progress=False)
        family = ModelTypes(resolved.options.get("kind", "ultralytics"))
        if family == ModelTypes.NNUNET and task == "segment" and model_type is None:
            task = "semantic_segment"
        if type is not None and ModelTypes(type) != family:
            raise ValueError("Requested model family conflicts with checkpoint metadata")
        from ..model import _normalize_prediction_task
        saved_task = _normalize_prediction_task(resolved.options.get("task"))
        if saved_task is not None and saved_task != task:
            raise ValueError(f"Checkpoint task {saved_task} does not match dataset task {task}")
        meta = dict(resolved.manifest)
        model_meta = dict(meta.get("model") or {})
        if weights is not None and model_meta.get("selection_role", "best") != "best":
            raise ValueError("This run has recovery weights only; no best checkpoint was selected")
        schema = meta.get("dataset_schema") or dict(meta.get("model") or {}).get("dataset_schema")
        if schema and schema != dataset_schema(dataset):
            raise ValueError("Checkpoint class/keypoint schema differs from the dataset; implicit head replacement is disabled")
        partial_schema = meta.get("checkpoint_schema", {})
        if any(dataset_schema(dataset).get(key) != value for key, value in partial_schema.items()):
            raise ValueError("Checkpoint class/keypoint schema differs from the dataset")
        path = resolved.path
        if family == ModelTypes.NNUNET:
            import json
            plans = json.loads((path / "plans.json").read_text())
            model_meta.setdefault("planner", plans.get("experiment_planner_used", "nnUNetPlannerResEncM"))
            meta["model"] = model_meta
            fold = resolved.options.get("folds", (0,))[0]
            checkpoint = resolved.options.get("checkpoint", "checkpoint_best.pth")
            if resume is not None and "checkpoint" not in resolved.options:
                checkpoint = next((name for name in ("checkpoint_latest.pth", "checkpoint_final.pth")
                                   if (path / f"fold_{fold}" / name).is_file()), checkpoint)
            path = path / f"fold_{fold}" / checkpoint
            if not path.is_file():
                raise FileNotFoundError(f"No selected checkpoint: {path}")
            meta["model_folder"] = str(resolved.path)
        if resume is not None and model_meta.get("latest_checkpoint"):
            candidate = path.parent / model_meta["latest_checkpoint"]
            if not candidate.is_file():
                raise FileNotFoundError(f"Bundle resumable checkpoint is missing: {candidate}")
            path = candidate
            from ..model_sources import _validate_checkpoint
            _validate_checkpoint(path, meta, progress=False)
        if resume is not None:
            import torch
            saved = torch.load(path, map_location="cpu", weights_only=False)
            optimizer = saved.get("optimizer_states") if family == ModelTypes.RFDETR else saved.get("optimizer_state") if family == ModelTypes.NNUNET else saved.get("optimizer")
            if not optimizer:
                raise ValueError("This checkpoint has no optimizer state; use weights= for a new run")
            saved_resolution = resolved.options.get("resolution") or resolved.geometry.input_size
            if isinstance(saved_resolution, (list, tuple)) and len(set(saved_resolution)) == 1:
                saved_resolution = saved_resolution[0]
            if config.resolution and saved_resolution not in {None, config.resolution}:
                raise ValueError("Change resolution with weights=, not full-state resume")
        name = model_meta["planner"] if family == ModelTypes.NNUNET else resolved.options.get("model_type", path.stem)
        if version is not None or s is not None:
            if family == ModelTypes.YOLO:
                parts = re.fullmatch(r"yolo(?:v)?(\d+)([a-z]+)(?:-[a-z]+)?", name.removesuffix(".pt"))
                if parts is None:
                    raise ValueError("Checkpoint does not identify a YOLO architecture; omit version/s selectors")
                expected = _yolo_name(task, version if version is not None else parts[1], s or parts[2]).removesuffix(".pt")
            elif family == ModelTypes.RFDETR:
                if version is not None:
                    raise ValueError("RF-DETR variants use s=, not version=")
                expected = _rf_variant(task, s)
            else:
                if version not in {None, 2}:
                    raise ValueError("Only nnU-Net v2 is supported")
                planner = model_meta.get("planner", name)
                expected = {"m": "nnUNetPlannerResEncM", "l": "nnUNetPlannerResEncL", "xl": "nnUNetPlannerResEncXL"}.get(s, planner if s is None else "invalid")
                name = planner
            if name.removesuffix(".pt") != expected:
                raise ValueError(f"Requested architecture {expected} conflicts with checkpoint {name}")
        selection = Selection(family, task, name, path if weights is not None else None, path if resume is not None else None, meta,
                              {"source": str(source), "sha256": sha256_progress(path, progress=False), **dict(meta.get("source_artifact") or {})})
    if selection is None:
        family = ModelTypes(type or ModelTypes.YOLO)
        if family == ModelTypes.YOLO:
            name = _yolo_name(task, version, s)
        elif family == ModelTypes.RFDETR:
            if version is not None:
                raise ValueError("RF-DETR variants use s=; version is a YOLO architecture selector")
            name = _rf_variant(task, s)
        else:
            if version not in {None, 2}:
                raise ValueError("Only nnU-Net v2 is supported")
            presets = {None: "nnUNetPlannerResEncM", "m": "nnUNetPlannerResEncM", "l": "nnUNetPlannerResEncL", "xl": "nnUNetPlannerResEncXL"}
            if s not in presets:
                raise ValueError("nnU-Net size must be m, l, or xl")
            name = presets[s]
        selection = Selection(family, task, name)
    supported = {ModelTypes.YOLO: {"detect", "segment", "pose", "polo", "semantic_segment"}, ModelTypes.RFDETR: {"detect", "segment", "pose"}, ModelTypes.NNUNET: {"semantic_segment"}}
    if task not in supported[selection.family]:
        raise ValueError(f"{selection.family.name} does not support dataset task {task}")
    return selection


def dataset_schema(dataset):
    shape = dataset._metadata.kpt_shape
    return {"classes": {str(k): v for k, v in dataset.classes.items()}, "kpt_shape": list(shape) if shape else None,
            "keypoint_counts": [shape[0]] * len(dataset.classes) if shape else None,
            "kpt_names": {str(k): v for k, v in dataset._metadata.kpt_names.items()}, "flip_idx": dataset._metadata.flip_idx}
