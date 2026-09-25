"""Safe, reproducible computer-vision dataset transformations."""

from .dataset import Dataset
from .dataset_comparison import DatasetComparisonResult
from .calibration import ThresholdCalibrationResult, calibrate_prediction_thresholds
from .comparison.types import ComparisonResult
from .errors import (
    DatasetValidationError,
    PredictionCacheMissError,
    PredictionScoreUnavailableError,
)
from .geometry import Geometry
from .model import ImagePrediction, Model, ModelCollection, ModelInput, PredictionResult
from .models import SemanticComparisonResult, Task
from .prediction_cache import PredictionCache
from .segmentation import PolygonRepairConfig
from .comparison.plot_labels import (
    ModelBadge,
    ModelPresentation,
    model_badges,
    model_full_label,
    model_label,
    model_presentation,
)
from .tracing import DatasetTrace, DatasetTraceNode, SampleTrace
from .utils import bounded_slug
from .training import (train, ModelTypes, TrainingConfig, CheckpointConfig, CheckpointProvider,
                       Checkpoints, WandbConfig, TrainingEvent, TrainingSession, TrainingResult,
                       preview_augmentations)

try:
    from ._version import __version__
except ImportError:  # source tree before setuptools-scm has generated the file
    __version__ = "0.1.0"

__all__ = [
    "train", "ModelTypes", "TrainingConfig", "CheckpointConfig", "CheckpointProvider", "Checkpoints",
    "WandbConfig", "TrainingEvent", "TrainingSession", "TrainingResult", "preview_augmentations",
    "Dataset",
    "DatasetComparisonResult",
    "calibrate_prediction_thresholds",
    "ComparisonResult",
    "DatasetValidationError",
    "DatasetTrace",
    "DatasetTraceNode",
    "ImagePrediction",
    "Geometry",
    "Model",
    "ModelBadge",
    "ModelPresentation",
    "ModelCollection",
    "ModelInput",
    "PredictionResult",
    "PredictionCache",
    "PredictionCacheMissError",
    "PredictionScoreUnavailableError",
    "PolygonRepairConfig",
    "SemanticComparisonResult",
    "SampleTrace",
    "Task",
    "ThresholdCalibrationResult",
    "bounded_slug",
    "model_badges",
    "model_full_label",
    "model_label",
    "model_presentation",
    "__version__",
]
