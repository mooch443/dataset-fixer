"""Render actual native augmentation samples through dataset-fixer's renderer."""
from __future__ import annotations

import json
import random
import tempfile
from contextlib import contextmanager
from dataclasses import replace
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


@contextmanager
def _preview_seed(seed):
    """Keep preview randomness reproducible without changing the caller's RNGs."""
    import torch
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.random.default_generator.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def visualize_augmentations(dataset, *, augmentations, weights, type, version, s,
                           model_type, config, split, options):
    """Prepare native targets, then delegate all presentation to visualization."""
    if dataset.task.value == "id":
        from ..id.evaluation import augmentation_preview
        return augmentation_preview(dataset, augmentations, config, split, options, weights=weights)
    from ..convert import _semantic_case_id, _yolo_relative_path
    from ..planning import clone_sample
    from ..visualization import annotation_item, finish_visualization, select_visualization_records, visualize_records, VisualizationItem
    from .backends import prepare_data

    candidates = [sample for sample in dataset._samples if split is None or sample.split == split]
    selected = select_visualization_records(candidates, options)
    config = replace(config or TrainingConfig(), seed=options.seed)
    selection = select(dataset, type=type, version=version, s=s, weights=weights,
                       model_type=model_type, config=config)
    if config.resolution is None and selection.resolution is not None and not ({"imgsz", "model"} & config.backend_options.keys()):
        config = replace(config, resolution=selection.resolution)
    validate_augmentations(selection.family, config, augmentations)
    # Preview the training pipeline over the requested split(s), retaining the
    # full eligible pool for native mosaic/mixing. Only this private view is relabelled.
    preview_samples = []
    for sample in candidates:
        copied = clone_sample(sample, split="train")
        copied.relative_path = Path(sample.split) / sample.relative_path
        preview_samples.append(copied)
    view = dataset._materialized_view(preview_samples, view={"kind": "augmentation-preview", "split": split})
    originals = {copied.relative_path: original for copied, original in zip(preview_samples, candidates)}
    if selection.family == ModelTypes.NNUNET:
        config = replace(config, workers=0)  # deterministic native transforms, no background workers
    with tempfile.TemporaryDirectory(prefix="dataset-fixer-preview-") as temporary, _preview_seed(options.seed):
        result = SimpleNamespace(dataset=view, selection=selection, config=config, output_dir=Path(temporary),
                                 metadata={}, auxiliary_files={})
        prepared = prepare_data(result)
        by_source = {sample.image_path.resolve(): originals[sample.relative_path]
                     for sample in result.dataset._samples}
        if selection.family == ModelTypes.NNUNET:
            source_lookup = {_semantic_case_id(sample): by_source[sample.image_path.resolve()]
                             for sample in result.dataset._samples}
        elif "cases" in prepared.paths:
            cases = json.loads(prepared.paths["cases"].read_text())
            source_lookup = {
                str((prepared.location / row["prepared_image"]).resolve()):
                by_source[Path(row["source_image"]).resolve()] for row in cases
            }
        else:
            source_lookup = {str((prepared.location / "train" / "images" / _yolo_relative_path(
                sample, flat=selection.family == ModelTypes.RFDETR)).resolve()): originals[sample.relative_path]
                for sample in preview_samples}
        rows = _native_samples(result, prepared, augmentations, source_lookup, selected)
        items = []
        for pixels, annotations, semantic, source in rows:
            if pixels.shape[-1] == 1:
                pixels = np.repeat(pixels, 3, axis=-1)
            sample = replace(source, width=pixels.shape[1], height=pixels.shape[0], annotations=annotations)
            mask = None
            if semantic is not None:
                import cv2
                foreground = ((semantic > 0) & (semantic != 255)).astype(np.uint8)
                mask = cv2.resize(foreground, (sample.width, sample.height), interpolation=cv2.INTER_NEAREST) > 0
            augmented = annotation_item(sample, dataset.task, dataset._metadata, options,
                                        image=Image.fromarray(pixels), mask=mask, paired=False)
            original = annotation_item(source, dataset.task, dataset._metadata, options, paired=False)
            items.append(VisualizationItem(source.image_path, augmented.label,
                (replace(original.panels[0], title="Original"), replace(augmented.panels[0], title="Augmented")),
                augmented.foreground))
        chart = visualize_records(items, options=replace(options, samples=None), prepare=lambda item: item,
                                  title=f"{selection.name} — native augmentation pipeline")
        return finish_visualization(chart, options)


def _native_samples(result, prepared, augmentations, source_lookup, selected):
    """Thin adapters from native data loaders to RGB pixels and domain targets."""
    from .backends import rfdetr_configs, train_nnunet, _yolo_options
    selection, config = result.selection, result.config
    rows = []
    selected_paths = [sample.image_path.resolve() for sample in selected]

    def indices(paths):
        lookup = {source_lookup[str(Path(path).resolve())].image_path.resolve(): i for i, path in enumerate(paths)}
        return [lookup[path] for path in selected_paths]

    if selection.family == ModelTypes.RFDETR:
        from rfdetr.datasets.yolo import build_roboflow_from_yolo
        variant, mc, native_options = rfdetr_configs(selection, config, prepared, augmentations)
        for name in ("native_callbacks", "trainer", "trainer_options"):
            native_options.pop(name, None)
        tc = variant._train_config_class(dataset_dir=str(prepared.location), dataset_file="yolo", **native_options)
        native = build_roboflow_from_yolo("train", SimpleNamespace(**{**mc.model_dump(), **tc.model_dump()}), mc.resolution)
        paths = [native.sv_dataset.get_image_info(i).image_path for i in range(len(native))]
        for i in indices(paths):
            image, target = native[i]
            pixels = image.numpy().transpose(1, 2, 0)
            pixels = np.clip((pixels * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255, 0, 255).astype(np.uint8)
            rows.append((pixels, _annotations(target, pixels.shape[1], pixels.shape[0]), None, source_lookup[str(Path(paths[i]).resolve())]))
    elif selection.family == ModelTypes.YOLO:
        from ultralytics.cfg import get_cfg
        from ultralytics.data.build import build_yolo_dataset
        import yaml
        values = yaml.safe_load(prepared.data_yaml.read_text())
        native_options = _yolo_options(config, augmentations)
        for name in ("native_callbacks", "trainer"):
            native_options.pop(name, None)
        task = "semantic" if selection.task == "semantic_segment" else selection.task
        hyp = get_cfg(overrides={**native_options, "task": task})
        native = build_yolo_dataset(hyp, str(prepared.location / "train" / "images"),
                                    batch=hyp.batch if isinstance(hyp.batch, int) and hyp.batch > 0 else 1,
                                    data=values, mode="train")
        for i in indices(native.im_files):
            item = native[i]
            pixels = item["img"].numpy().transpose(1, 2, 0)
            target = {"boxes": item.get("bboxes", []), "labels": np.asarray(item.get("cls", [])).reshape(-1),
                      "keypoints": item.get("keypoints", [])}
            if task == "segment":
                masks = np.asarray(item["masks"])
                target["masks"] = np.stack([masks[0] == j + 1 for j in range(len(target["labels"]))]) if hyp.overlap_mask and len(target["labels"]) else masks
            semantic = np.asarray(item["semantic_mask"]).squeeze() if task == "semantic" else None
            rows.append((pixels, _annotations(target, pixels.shape[1], pixels.shape[0]), semantic,
                         source_lookup[str(Path(native.im_files[i]).resolve())]))
    else:
        def collect(trainer):
            import torch
            old_threads = torch.get_num_threads()
            loaders = ()
            try:
                keys = list(source_lookup)
                trainer.do_split = lambda: (keys, keys)
                loaders = trainer.get_dataloaders()
                by_path = {source.image_path.resolve(): key for key, source in source_lookup.items()}
                for path in selected_paths:
                    key = by_path[path]
                    loaders[0].data_loader.indices = [key]
                    batch = next(loaders[0])
                    pixels = batch["data"][0].numpy().transpose(1, 2, 0)
                    pixels = np.clip((pixels - pixels.min()) / max(float(np.ptp(pixels)), 1e-8) * 255, 0, 255).astype(np.uint8)
                    target = batch["target"][0] if isinstance(batch["target"], list) else batch["target"]
                    rows.append((pixels, [], target[0, 0].numpy(), source_lookup[key]))
            finally:
                for loader in loaders:
                    if hasattr(loader, "_finish"):
                        loader._finish()
                torch.set_num_threads(old_threads)
        train_nnunet(result, prepared, augmentations, preview=collect)
    return rows



def publish_augmentation_preview(result, augmentations):
    """Shared startup artifact, rendered locally once and logged to the owned run."""
    if result.config.augmentation_samples == 0:
        return
    from ..visualization import VisualizationOptions
    config = result.config
    if result.selection.family == ModelTypes.ID:
        config = replace(config, backend_options={**config.backend_options,
                         "preprocessing": result.metadata["identity"]["preprocessing"]})
    path = result.output_dir / "augmentations.png"
    options = VisualizationOptions(samples=config.augmentation_samples, columns=3, seed=config.seed, show=False, destination=path)
    with _preview_seed(config.seed):
        visualize_augmentations(result.dataset, augmentations=augmentations,
            weights=result.selection.weights, type=result.selection.family, version=None, s=None,
            model_type=result.selection.task, config=config, split="train", options=options)
    result.auxiliary_files["training/augmentations.png"] = path
    if result.wandb_run is not None:
        import wandb
        result.wandb_run.log({"training/augmentations": wandb.Image(str(path))})
