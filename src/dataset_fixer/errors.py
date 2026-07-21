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
    """Raised when a dataset is unsafe or incompatible before any training starts."""

    def __init__(self, issues: str | ValidationIssue | list[ValidationIssue]):
        if isinstance(issues, str):
            self.issues = [ValidationIssue(issues)]
        elif isinstance(issues, ValidationIssue):
            self.issues = [issues]
        else:
            self.issues = issues
        body = "\n".join(f"  {i + 1}. {issue.format()}" for i, issue in enumerate(self.issues))
        super().__init__(f"Dataset validation failed with {len(self.issues)} error(s):\n{body}")

