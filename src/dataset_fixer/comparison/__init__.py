"""Model-comparison result type and lazy internal entry point."""

from .types import ComparisonResult


def compare_models(*args, **kwargs):
    # Keep ordinary dataset loading free from comparison/plotting imports.
    from .engine import compare_models as implementation

    return implementation(*args, **kwargs)

__all__ = ["ComparisonResult", "compare_models"]
