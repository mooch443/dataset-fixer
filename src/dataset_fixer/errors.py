from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationIssue:
    message: str
    source: str | None = None
    line: int | None = None
    value: object | None = None
    expected: str | None = None
    suggestion: str | None = None

    def format(self) -> str:
        where = self.source or "dataset"
        if self.line is not None:
            where += f":{self.line}"
        parts = [f"{where}: {self.message}"]
        if self.value is not None:
            parts.append(f"value={self.value!r}")
        if self.expected:
            parts.append(f"expected {self.expected}")
        if self.suggestion:
            parts.append(f"fix: {self.suggestion}")
        return "; ".join(parts)


class DatasetValidationError(ValueError):
    """Raised when data or model geometry is unsafe or incompatible.

    Parameters:
        issues: One message, one structured validation issue, or a list of
            structured issues to include in the exception.
    """

    def __init__(self, issues: str | ValidationIssue | list[ValidationIssue]):
        if isinstance(issues, str):
            self.issues = [ValidationIssue(issues)]
        elif isinstance(issues, ValidationIssue):
            self.issues = [issues]
        else:
            self.issues = issues
        body = "\n".join(f"  {i + 1}. {issue.format()}" for i, issue in enumerate(self.issues))
        super().__init__(f"Dataset validation failed with {len(self.issues)} error(s):\n{body}")


class PredictionCacheMissError(RuntimeError):
    """Raised when cache-only prediction cannot return the requested data.

    ``reason`` is machine-readable so calibration workflows can distinguish a
    completely missing prediction from a valid legacy hard-mask cache that
    lacks probability maps.

    Args:
        message: Human-readable description of the unavailable cache product.
        reason: Stable machine-readable cache-miss category.
    """

    def __init__(self, message: str, *, reason: str = "missing") -> None:
        self.reason = reason
        super().__init__(message)


class PredictionScoreUnavailableError(RuntimeError):
    """Raised when a requested threshold cannot be applied to model output.

    Semantic probability thresholds require logits or probabilities. A hard
    class map remains a valid prediction artifact, but it cannot be calibrated
    or re-thresholded without inventing score information that the backend did
    not return.

    Args:
        message: Human-readable explanation of the unavailable score product.
        reason: Stable machine-readable failure category.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: str = "missing-semantic-probabilities",
    ) -> None:
        self.reason = reason
        super().__init__(message)
