"""Validate native augmentation settings without backends' warn-and-skip fallbacks."""
from __future__ import annotations


def yolo_augmentations(value):
    from ultralytics.cfg import get_cfg
    options = dict(value) if isinstance(value, dict) else {"augmentations": value}
    get_cfg(overrides=options)
    return options


def validate_yolo_transforms(value):
    if value is None:
        return
    if not isinstance(value, (list, tuple)):
        raise ValueError("YOLO custom augmentations must be a list of Albumentations transforms")
    from ultralytics.data.augment import Albumentations
    native = Albumentations(transforms=value)
    if native.transform is None:
        raise ValueError("Invalid YOLO Albumentations configuration; the native backend would skip it")
    native.transform.strict = True


def validate_rfdetr_augmentations(value, *, flip_pairs=None):
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError("RF-DETR augmentations must map transform names to parameter dictionaries")
    from rfdetr.datasets.transforms import ALBUMENTATIONS_CONTAINERS, AlbumentationsWrapper, _build_albu_transform
    from rfdetr.datasets._aug_utils import filter_keypoint_hflip_augmentations

    def reject(message, *args):
        raise ValueError(message % args)

    def check_containers(name, params):
        if name not in ALBUMENTATIONS_CONTAINERS:
            return
        if isinstance(params, list):
            params = {"transforms": params}
        if not isinstance(params, dict):
            return  # the native builder reports malformed parameters
        if name in {"OneOf", "Sequential"} and params.get("p", 1.0) != 1.0:
            raise ValueError(f"RF-DETR forces {name}.p=1.0; set probabilities on its children")
        for entry in params.get("transforms", []) if isinstance(params.get("transforms", []), list) else []:
            if isinstance(entry, dict) and len(entry) == 1:
                check_containers(*next(iter(entry.items())))

    filter_keypoint_hflip_augmentations(value, include_keypoints=flip_pairs == [], warn=reject)
    for name, params in value.items():
        check_containers(name, params)
        if name in ALBUMENTATIONS_CONTAINERS and isinstance(params, list):
            params = {"transforms": params}
        if not isinstance(params, dict):
            raise ValueError(f"Invalid RF-DETR augmentation {name!r}: expected a parameter dictionary")
        try:
            transform = _build_albu_transform(name, params)
            wrapper = AlbumentationsWrapper(transform, keypoint_flip_pairs=flip_pairs)
            wrapper.transform.strict = True
        except Exception as exc:
            raise ValueError(f"Invalid RF-DETR augmentation {name!r}: {exc}") from exc


def validate_augmentations(family, config, value):
    from .config import ModelTypes
    if family == ModelTypes.YOLO:
        from .backends import _yolo_options
        _yolo_options(config, value)
    elif family == ModelTypes.RFDETR:
        validate_rfdetr_augmentations(value if value is not None else config.backend_options.get("aug_config"))
    elif value is not None and not callable(value):
        raise ValueError("nnU-Net augmentations must be a callable customizing its native transform pipeline")
