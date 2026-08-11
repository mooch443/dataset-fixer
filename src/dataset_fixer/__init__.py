"""Safe, reproducible computer-vision dataset transformations."""

from .dataset import Dataset
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
from .tracing import DatasetTrace, DatasetTraceNode, SampleTrace

try:
    from ._version import __version__
except ImportError:  # source tree before setuptools-scm has generated the file
    __version__ = "0.1.0"

__all__ = [
    "Dataset",
    "calibrate_prediction_thresholds",
    "ComparisonResult",
    "DatasetValidationError",
    "DatasetTrace",
    "DatasetTraceNode",
    "ImagePrediction",
    "Geometry",
    "Model",
    "ModelCollection",
    "ModelInput",
    "PredictionResult",
    "PredictionCache",
    "PredictionCacheMissError",
    "PredictionScoreUnavailableError",
    "SemanticComparisonResult",
    "SampleTrace",
    "Task",
    "ThresholdCalibrationResult",
    "__version__",
]
