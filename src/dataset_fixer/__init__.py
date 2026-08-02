"""Safe, reproducible computer-vision dataset transformations."""

from .dataset import Dataset
from .comparison.types import ComparisonResult
from .errors import DatasetValidationError
from .models import SemanticComparisonResult, SemanticMaskExport, Task

try:
    from ._version import __version__
except ImportError:  # source tree before setuptools-scm has generated the file
    __version__ = "0.1.0"

__all__ = [
    "Dataset",
    "ComparisonResult",
    "DatasetValidationError",
    "SemanticComparisonResult",
    "SemanticMaskExport",
    "Task",
    "__version__",
]
