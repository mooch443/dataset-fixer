"""Backbone reconstruction and one embedding/prediction path for both query modes."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import hashlib
import json
import random
import time

import numpy as np

from .data import IDENTITY_IMAGE_SUFFIXES, ImageRef, SampleIndex, index_of, _file_record

# Factories reconstruct architectures; checkpoints contain tensors, not pickled models.
_BACKBONES = {}
_INSTANCES = {}
_PRESETS = {"resnet18": "resnet18", "efficientnet_v2_s": "efficientnet_v2_s",
            "efficientnetv2_s": "efficientnet_v2_s", "convnext_tiny": "convnext_tiny"}


def register_backbone(name, factory, *, feature_dim, preprocessing, model=None):
    """Register a custom encoder (or complete model producing N×D features).

    The zero-argument factory must rebuild the same architecture in each process.
    preprocessing supplies resolution, mean and std. Optional model registers an
    already initialized instance for use as backbone=model in a new training run.
    Register the same name/factory before loading its checkpoints elsewhere.

    Parameters:
        name: Stable custom architecture identifier.
        factory: Zero-argument callable reconstructing the encoder architecture.
        feature_dim: Dimension returned by the encoder for each image.
        preprocessing: Resolution, mean, std, and optional interpolation definition.
        model: Optional initialized model instance associated with this registration.
    """
    if not name or name in _PRESETS or name.startswith("timm:"):
        raise ValueError("Choose a distinct custom backbone name")
    if not callable(factory) or not isinstance(feature_dim, int) or feature_dim < 1:
        raise ValueError("factory must be callable and feature_dim must be positive")
    prep = validate_preprocessing(preprocessing)
    _BACKBONES[name] = (factory, feature_dim, prep)
    if model is not None:
        _INSTANCES[id(model)] = name
    return name


def backbone_name(value):
    if isinstance(value, str):
        return value
    if id(value) in _INSTANCES:
        return _INSTANCES[id(value)]
    matches = [name for name, (factory, _, _) in _BACKBONES.items()
               if isinstance(factory, type) and isinstance(value, factory)]
    if len(matches) == 1:
        return matches[0]
    raise ValueError("Register the custom model with register_backbone(..., model=model) before training; a reconstruction factory is required")


def validate_preprocessing(value):
    prep = dict(value)
    prep.setdefault("resolution", 224)
    prep.setdefault("mean", [0.485, 0.456, 0.406])
    prep.setdefault("std", [0.229, 0.224, 0.225])
    prep.setdefault("interpolation", "bilinear")
    if not isinstance(prep["resolution"], int) or prep["resolution"] < 1:
        raise ValueError("Preprocessing resolution must be a positive integer")
    if len(prep["mean"]) != 3 or len(prep["std"]) != 3 or not np.isfinite([*prep["mean"], *prep["std"]]).all() or min(prep["std"]) <= 0:
        raise ValueError("Preprocessing requires three finite channel means and positive standard deviations")
    return prep


def build_backbone(name, *, pretrained=True, instance=None, resolution=None):
    import torch
    if name in _BACKBONES:
        factory, dim, prep = _BACKBONES[name]
        return instance if instance is not None else factory(), dim, dict(prep)
    if instance is not None:
        raise ValueError("Custom model instances require a registered factory")
    if name in _PRESETS:
        from torchvision import models
        name = _PRESETS[name]
        weights = models.get_model_weights(name).DEFAULT
        encoder = models.get_model(name, weights=weights if pretrained else None)
        transform = weights.transforms()
        prep = dict(resolution=transform.crop_size[0], mean=transform.mean, std=transform.std,
                    interpolation=transform.interpolation.value)
        if name == "resnet18":
            dim, encoder.fc = encoder.fc.in_features, torch.nn.Identity()
        elif name == "convnext_tiny":
            dim = encoder.classifier[-1].in_features
            encoder.classifier[-1] = torch.nn.Identity()
        else:
            dim = encoder.classifier[-1].in_features
            encoder.classifier[-1] = torch.nn.Identity()
        return encoder, dim, prep
    import timm
    encoder = timm.create_model(name.removeprefix("timm:"), pretrained=pretrained, num_classes=0)
    if resolution is not None and hasattr(encoder, "set_input_size"):
        encoder.set_input_size(img_size=resolution)
    config = timm.data.resolve_model_data_config(encoder)
    prep = dict(resolution=config["input_size"][-1], mean=list(config["mean"]),
                std=list(config["std"]), interpolation=config["interpolation"])
    return encoder, encoder.num_features, prep


def default_augmentations():
    """Return a fresh wolf/animal crop augmentation list.

    Preserve orientation (±5°, no shear/flips), with proportional zoom, modest
    cropping, bidirectional color/intensity changes and random-pixel occlusions.
    Customize this ordinary Albumentations list or pass [] to disable augmentation.
    """
    import albumentations as A
    return [
        A.RandomCropFromBorders(crop_left=0.1, crop_right=0.1, crop_top=0.1, crop_bottom=0.1, p=0.35),
        A.Affine(scale=(0.8, 1.6), translate_percent=(-0.18, 0.18), rotate=(-5, 5),
                 shear=0, keep_ratio=True, balanced_scale=True, fill=0, p=0.7),
        A.HueSaturationValue(hue_shift_limit=18, sat_shift_limit=20, val_shift_limit=50, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=(-0.15, 0.15), contrast_limit=(-0.15, 0.15), p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 0.6), p=0.1),
        A.Erasing(scale=(0.02, 0.12), ratio=(0.3, 3.3), fill="random", p=0.35),
        A.Erasing(scale=(0.02, 0.06), ratio=(0.3, 3.3), fill="random", p=0.2),
        A.ToGray(p=0.25),
    ]


def augmentation_pipeline(value, *, seed=7):
    """Clone the serialized pipeline, including its private RNG, on every build."""
    import albumentations as A
    from ..augmentation import serialize_pipeline
    if value is None:
        value = default_augmentations()
    serialized = serialize_pipeline(value, {})
    def inspect(node):
        if isinstance(node, dict):
            kind = node.get("__class_fullname__", "").split(".")[-1]
            if kind in {"Normalize", "ToTensorV2", "ToTensor3D"}:
                raise ValueError("Model preprocessing owns normalization/tensor conversion; custom augmentations must return image pixels")
            for child in node.values():
                inspect(child)
        elif isinstance(node, list):
            for child in node:
                inspect(child)
    inspect(serialized)
    pipeline = A.from_dict(serialized)
    seed_augmentations(pipeline, seed)
    return pipeline, serialized


def seed_augmentations(pipeline, seed):
    """Seed one shared stream, so successive transforms consume distinct draws.

    Compose.set_random_seed gives each child an identical *separate* stream.
    Resetting those streams per image correlates their probability decisions.
    Albumentations' set_random_state shares streams instead, without touching
    Python/NumPy global state. Training and preview use this same policy.
    """
    pipeline.set_random_state(np.random.default_rng(int(seed)), random.Random(int(seed)))


def preprocess(image, preprocessing):
    """Final deterministic resize/normalization, after any body-part cropping."""
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import resize, normalize
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 1:
        raise ValueError("Augmentations must return a nonempty H×W×3 uint8 image")
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255)
    size = preprocessing["resolution"]
    if tensor.shape[-2:] != (size, size):
        tensor = resize(tensor, [size, size], interpolation=InterpolationMode(preprocessing.get("interpolation", "bilinear")), antialias=True)
    return normalize(tensor, preprocessing["mean"], preprocessing["std"])


def load_checkpoint(path):
    import torch
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(saved, dict) or saved.get("identity", {}).get("framework") != "id":
        raise ValueError("Not a dataset-fixer identity checkpoint")
    return saved


def runtime(model):
    if "identity" not in model._runtime:
        from .training import _IdentityModule
        saved = load_checkpoint(model.path)
        meta = saved["identity"]
        module = _IdentityModule(meta, pretrained=False)
        module.load_state_dict(saved["state_dict"])
        module.eval()
        model._runtime["identity"] = module
    return model._runtime["identity"]


def fingerprint(model):
    """Identity of inference tensors/preprocessing, independent of optimizer/head packaging."""
    module = runtime(model)
    if "identity_fingerprint" not in model._runtime:
        digest = hashlib.sha256(json.dumps({key: module.identity.get(key) for key in
            ("backbone", "embedding_dim", "preprocessing")}, sort_keys=True).encode())
        for name, tensor in sorted(module.state_dict().items()):
            if name.startswith(("encoder.", "projection.")):
                array = tensor.detach().cpu().contiguous()
                digest.update(f"{name}:{array.dtype}:{tuple(array.shape)}".encode())
                digest.update(array.reshape(-1).view(__import__("torch").uint8).numpy().tobytes())
        model._runtime["identity_fingerprint"] = digest.hexdigest()
    return model._runtime["identity_fingerprint"]


def input_index(source, split=None):
    from ..dataset import Dataset
    from ..models import Sample
    if isinstance(source, Dataset):
        index = index_of(source)
        return index.where(split=split) if split is not None else index
    if isinstance(source, SampleIndex):
        return source.where(split=split) if split is not None else source
    if hasattr(source, "records"):
        source = source.records
    paths = [source] if isinstance(source, (str, Path, Sample, ImageRef)) else list(source)
    if paths and isinstance(paths[0], Sample):
        from .data import record_from_sample
        return SampleIndex([record_from_sample(sample) for sample in paths])
    if paths and isinstance(paths[0], ImageRef):
        return SampleIndex([_file_record(dict(path=ref.path, array=ref.array, row=ref.row, crop=ref.crop,
            trim_black_borders=ref.trim_black_borders), Path.cwd()) for ref in paths])
    if paths and hasattr(paths[0], "image_path"):
        records = []
        for value in paths:
            provenance = getattr(value, "provenance", {})
            ref = ImageRef(*json.loads(provenance["image_ref"])) if provenance.get("image_ref") else ImageRef(str(value.image_path))
            records.append(_file_record(dict(path=ref.path, array=ref.array, row=ref.row, crop=ref.crop,
                trim_black_borders=ref.trim_black_borders,
                identity_id=provenance.get("identity_id"), clip_id=provenance.get("clip_id")), Path.cwd()))
        return SampleIndex(records)
    if len(paths) == 1 and Path(paths[0]).is_dir():
        paths = sorted(p for p in Path(paths[0]).rglob("*") if p.is_file() and p.suffix.lower() in IDENTITY_IMAGE_SUFFIXES)
    return SampleIndex([_file_record({"path": str(Path(p).resolve())}, Path.cwd()) for p in paths])


def embed(model, source, *, split=None, batch_size=None, device=None, prediction_cache=True, cache_only=False,
          progress=True, _on_progress=None):
    import torch
    from ..model import ImagePrediction, ModelInput, PredictionResult
    from ..prediction_cache import PredictionCache, prediction_cache_key
    from ._progress import batches
    index = input_index(source, split)
    module = runtime(model)
    device = device or model._device or "cpu"
    if device.startswith("cuda:") and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but unavailable")
    module.to(device)
    size = batch_size if batch_size is not None else (model._batch_size if model._batch_size and model._batch_size > 0 else 64)
    if size < 1:
        raise ValueError("Identity batch_size must be positive")
    # Cache each bounded batch using the canonical cache serializer. Pixel hashes
    # avoid hashing an entire NPZ for each row and detect derived/duplicate crops.
    cache = prediction_cache if isinstance(prediction_cache, PredictionCache) else (
        PredictionCache(prediction_cache) if isinstance(prediction_cache, (str, Path)) else PredictionCache.package_default())
    records, hits = [], 0
    started = time.monotonic()
    for start, end in batches(len(index), size, stage="embeddings", description=f"ID embeddings ({device})",
                              progress=progress, on_progress=_on_progress):
        samples = [index[i] for i in range(start, end)]
        pixels = [s.image_ref.read() for s in samples]
        hashes = [hashlib.sha256(str(p.shape).encode()+p.tobytes()).hexdigest() for p in pixels]
        inputs = [ModelInput(hashlib.sha256(s.image_ref.key.encode()).hexdigest(), s.image_path, p.shape[1], p.shape[0],
                            relative_path=s.image_ref.key, image_sha256=digest) for s, p, digest in zip(samples, pixels, hashes)]
        identity = dict(task="id", model=fingerprint(model), preprocessing=module.identity["preprocessing"],
                        images=[dict(key=s.image_ref.key, sha256=digest) for s, digest in zip(samples, hashes)])
        key = prediction_cache_key(identity)
        cached = cache.load(key, namespace="predictions", identity=identity, inputs=inputs) if prediction_cache else None
        if cached is None:
            if cache_only:
                raise FileNotFoundError("Identity embeddings are not present in the verified prediction cache")
            with torch.inference_mode():
                vectors = module(torch.stack([preprocess(p, module.identity["preprocessing"]) for p in pixels]).to(device)).cpu().numpy()
            rows = tuple(ImagePrediction(i.image_id, i.image_path, i.relative_path, i.width, i.height,
                        metadata={"embedding": vector.tolist(), "pixel_sha256": digest}) for i, vector, digest in zip(inputs, vectors, hashes))
            cached = PredictionResult(model.name, "id", "id", "torch", rows, time.monotonic()-started)
            if prediction_cache:
                cache.save(key, cached, namespace="predictions", identity=identity, inputs=inputs)
        else:
            hits += len(cached)
        # Ground truth is presentation/evaluation context, never a prediction input.
        records.extend(replace(r, metadata={**r.metadata, "reference_identity": s.identity_id, "clip_id": s.clip_id,
                       "frame_index": s.frame_index, "local_id": s.local_id,
                       "original_id": s.provenance.get("original_id"), "tracklet_id": s.provenance.get("tracklet_id"), "source_ref": s.image_ref.key})
                       for r, s in zip(cached.records, samples))
    return PredictionResult(model.name, "id", "id", "torch", tuple(records), time.monotonic()-started,
                            settings={"model_fingerprint": fingerprint(model)}, cache_info={"hits": hits, "misses": len(index)-hits}, model_metadata=model.describe())


def predict(model, source, *, gallery, mode="closed", rejection=None, split=None, batch_size=None,
            device=None, prediction_cache=True, cache_only=False, track_by=None, progress=True):
    if mode not in {"closed", "open"}:
        raise ValueError("mode must be 'closed' or 'open'")
    if gallery is None:
        raise ValueError("Identification requires an enrollment gallery; use model.enroll(reference_crops)")
    gallery.validate(model)
    if mode == "open" and rejection is None:
        raise ValueError("Open queries require validation-calibrated rejection")
    if rejection is not None:
        rejection.validate(model, gallery)
    result = embed(model, source, split=split, batch_size=batch_size, device=device,
                   prediction_cache=prediction_cache, cache_only=cache_only, progress=progress)
    if not result.records:
        return result
    vectors = np.asarray([r.embedding for r in result])
    if track_by:
        fields = (track_by,) if isinstance(track_by, str) else tuple(track_by)
        if any(key not in {"clip_id", "local_id", "tracklet_id"} for key in fields):
            raise ValueError("track_by accepts supplied clip/local tracking groups, never identity labels")
        groups = {}
        for i, record in enumerate(result):
            key = tuple(record.metadata.get(field) for field in fields)
            if None in key:
                raise ValueError("Tracking groups are missing on query observations")
            groups.setdefault(key, []).append(i)
        for rows in groups.values():
            vector = vectors[rows].mean(axis=0)
            vectors[rows] = vector / max(np.linalg.norm(vector), 1e-12)
    scores = gallery.scores(vectors)
    records = []
    for record, row in zip(result, scores):
        order = np.argsort(-row, kind="stable")
        best = order[0]
        rejected = bool(mode == "open" and float(row[best]) < rejection.threshold)
        records.append(replace(record, metadata={**record.metadata,
            "identity_id": None if rejected else gallery.identities[best], "similarity": float(row[best]),
            "rejected": rejected, "ranking": [(gallery.identities[j], float(row[j])) for j in order]}))
    return replace(result, records=tuple(records), settings={**result.settings, "mode": mode, "gallery_fingerprint": gallery.fingerprint})
