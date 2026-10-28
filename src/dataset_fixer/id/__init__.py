"""Individual identification with one encoder/gallery for closed and open queries.

Training dependencies are loaded only when training or using a model.
"""
from .data import from_dataset, stats
from .evaluation import Protocol, RejectionCalibration
from .gallery import Gallery
from .model import register_backbone, default_augmentations
from .inspection import visualize_attention


def train(data, *, backbone=None, **kwargs):
    """Train via the shared lifecycle, returning the existing TrainingResult.

    Parameters:
        data: Identity Dataset or validated Protocol.
        backbone: Predefined/timm name or registered custom model instance.

    Other keyword arguments are identical to dataset_fixer.train().
    """
    from ..training import train as shared_train
    return shared_train(data, type="id", backbone=backbone, **kwargs)


__all__ = ["train", "from_dataset", "register_backbone", "default_augmentations", "stats", "visualize_attention",
           "Protocol", "Gallery", "RejectionCalibration"]
