"""Render actual native augmentation samples through dataset-fixer's renderer."""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from .config import ModelTypes, TrainingConfig
from .selection import select
from .augmentations import validate_augmentations


def _annotations(target, width, height):
    from ..models import Annotation
    rows = []
    boxes = np.asarray(target.get("boxes", []))
    labels = np.asarray(target.get("labels", []))
    points = np.asarray(target.get("keypoints", []))
    masks = np.asarray(target.get("masks", []))
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x, y, w, h = box * [width, height, width, height]
        keypoints = None
        polygon = None
        if len(points):
            keypoints = [(float(p[0] * width), float(p[1] * height), float(p[2]) if len(p) > 2 else 2.0) for p in points[i]]
        if len(masks):
            import cv2
            mask = cv2.resize(masks[i].astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                polygon = max(contours, key=len).reshape(-1, 2).astype(float).tolist()
        rows.append(Annotation(int(label), bbox=(x-w/2, y-h/2, x+w/2, y+h/2), keypoints=keypoints, polygon=polygon))
    return rows


def preview_augmentations(dataset, augmentations=None, *, type=None, version=None, s=None,
                          weights=None, model_type=None, config=None, samples=3, destination=None, show=True):
    """Preview native training inputs, including geometry, without loading model weights.

    YOLO accepts native augmentation settings; RF-DETR accepts its augmentation
    dictionary; nnU-Net accepts a callable customizing its native transform.
    The same object is passed unchanged to train().

    Parameters:
        dataset: Validated Dataset or source accepted by Dataset.open().
        augmentations: The same native augmentation configuration used by train().
        type: Selected native backend.
        version: Installed YOLO version selector.
        s: Native model size/variant selector.
        weights: Optional initialization source used to infer the architecture.
        model_type: Optional explicit dataset task override.
        config: Shared training settings controlling input size and native options.
        samples: Maximum native training samples to render.
        destination: Optional image/PDF/SVG output path.
        show: Display the chart in the notebook.
    """
    from ..dataset import Dataset
    from ..models import Sample
    from ..static_rendering import save_chart
    from ..visualization import VisualizationItem, VisualizationPanel, VisualizationOptions, visualize_records, render_annotated_sample
    from .backends import prepare_data, rfdetr_configs, train_nnunet, _yolo_options
    if not isinstance(dataset, Dataset):
        dataset = Dataset.open(dataset)
    config = config or TrainingConfig()
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    import random
    import torch
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    selection = select(dataset, type=type, version=version, s=s, weights=weights, model_type=model_type, config=config)
    validate_augmentations(selection.family, config, augmentations)
    with tempfile.TemporaryDirectory(prefix="dataset-fixer-preview-") as temporary:
        root = Path(temporary)
        result = SimpleNamespace(dataset=dataset, selection=selection, config=config, output_dir=root,
                                 metadata={}, auxiliary_files={})
        prepared = prepare_data(result)
        rows = []
        if selection.family == ModelTypes.RFDETR:
            from rfdetr.datasets.yolo import build_roboflow_from_yolo
            from rfdetr.utilities.reproducibility import seed_all
            seed_all(config.seed)
            variant, mc, options = rfdetr_configs(selection, config, prepared, augmentations)
            for name in ("native_callbacks", "trainer", "trainer_options"):
                options.pop(name, None)
            tc = variant._train_config_class(dataset_dir=str(prepared.location), dataset_file="yolo", **options)
            native = build_roboflow_from_yolo("train", SimpleNamespace(**{**mc.model_dump(), **tc.model_dump()}), mc.resolution)
            for i in range(min(samples, len(native))):
                image, target = native[i]
                pixels = image.numpy().transpose(1, 2, 0)
                pixels = np.clip((pixels * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255, 0, 255).astype(np.uint8)
                rows.append((pixels, _annotations(target, pixels.shape[1], pixels.shape[0]), None))
        elif selection.family == ModelTypes.YOLO:
            from ultralytics.cfg import get_cfg
            from ultralytics.data.build import build_yolo_dataset
            import yaml
            values = yaml.safe_load(prepared.data_yaml.read_text())
            options = _yolo_options(config, augmentations)
            for name in ("native_callbacks", "trainer"):
                options.pop(name, None)
            task = "semantic" if selection.task == "semantic_segment" else selection.task
            hyp = get_cfg(overrides={**options, "task": task})
            native = build_yolo_dataset(hyp, str(prepared.location / "train" / "images"),
                                        batch=hyp.batch if isinstance(hyp.batch, int) and hyp.batch > 0 else 1,
                                        data=values, mode="train")
            for i in range(min(samples, len(native))):
                item = native[i]
                pixels = item["img"].numpy().transpose(1, 2, 0)
                target = {"boxes": item.get("bboxes", []), "labels": np.asarray(item.get("cls", [])).reshape(-1), "keypoints": item.get("keypoints", [])}
                if task == "segment":
                    masks = np.asarray(item["masks"])
                    target["masks"] = np.stack([masks[0] == i + 1 for i in range(len(target["labels"]))]) if hyp.overlap_mask and len(target["labels"]) else masks
                semantic = np.asarray(item["semantic_mask"]).squeeze() if task == "semantic" else None
                rows.append((pixels, _annotations(target, pixels.shape[1], pixels.shape[0]), semantic))
        else:
            def collect(trainer):
                import torch
                old_threads = torch.get_num_threads()
                loaders = trainer.get_dataloaders()
                try:
                    for _ in range(samples):
                        batch = next(loaders[0])
                        pixels = batch["data"][0].numpy().transpose(1, 2, 0)
                        pixels = np.clip((pixels - pixels.min()) / max(float(np.ptp(pixels)), 1e-8) * 255, 0, 255).astype(np.uint8)
                        target = batch["target"][0] if isinstance(batch["target"], list) else batch["target"]
                        rows.append((pixels, [], target[0, 0].numpy()))
                finally:
                    for loader in loaders:
                        if hasattr(loader, "_finish"):
                            loader._finish()
                    torch.set_num_threads(old_threads)
            train_nnunet(result, prepared, augmentations, preview=collect)
        items = []
        for i, (pixels, annotations, semantic) in enumerate(rows):
            if pixels.shape[-1] == 1:
                pixels = np.repeat(pixels, 3, axis=-1)
            path = root / f"sample-{i}.png"
            Image.fromarray(pixels).save(path)
            sample = Sample(path, Path(path.name), "train", pixels.shape[1], pixels.shape[0], annotations)
            scale = max(1, 384 / min(sample.width, sample.height))
            display_size = (round(sample.width * scale), round(sample.height * scale))
            rendered = np.asarray(render_annotated_sample(sample, dataset.task, dataset._metadata, resize_to=display_size))
            pixels = np.asarray(Image.fromarray(pixels).resize(display_size, Image.Resampling.LANCZOS))
            mask = None
            if semantic is not None:
                import cv2
                foreground = (semantic > 0) & (semantic != 255)
                mask = cv2.resize(foreground.astype(np.uint8), display_size, interpolation=cv2.INTER_NEAREST) > 0
            items.append(VisualizationItem(path, f"Training sample {i+1}",
                panels=(VisualizationPanel(title="Augmented image", image=pixels), VisualizationPanel(title="Training targets", image=rendered, mask=mask, color="#ff2020")),
                foreground=np.ones(pixels.shape[:2], dtype=bool)))
        chart = visualize_records(items, options=VisualizationOptions(samples=None, columns=1, panel_size=4, show=show),
                                  prepare=lambda item: item, title=f"{selection.name} — native augmentation pipeline")
        if destination is not None:
            save_chart(chart, Path(destination))
        return chart
