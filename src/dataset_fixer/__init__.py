"""Safe, reproducible computer-vision dataset transformations."""

from .dataset import Dataset
from .comparison.types import ComparisonResult
from .errors import DatasetValidationError
from .model import ImagePrediction, Model, ModelCollection, ModelInput, PredictionResult
from .models import SemanticComparisonResult, Task
from .tracing import DatasetTrace, DatasetTraceNode, SampleTrace

try:
    from ._version import __version__
except ImportError:  # source tree before setuptools-scm has generated the file
    __version__ = "0.1.0"

__all__ = [
    "Dataset",
    "ComparisonResult",
    "DatasetValidationError",
    "DatasetTrace",
    "DatasetTraceNode",
    "ImagePrediction",
    "Model",
    "ModelCollection",
    "ModelInput",
    "PredictionResult",
    "SemanticComparisonResult",
    "SampleTrace",
    "Task",
    "__version__",
]
