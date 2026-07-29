from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    training_dataset: Path | None = None
    resolution: int = 480
    confidence_thresholds: tuple[float, ...] | None = None
    postprocess_thresholds: tuple[float, ...] | None = None
    locked_confidence: float | None = None
    locked_postprocess: float | None = None
    inference_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    class_id: int
    score: float
    bbox: tuple[float, float, float, float] | None = None
    point: tuple[float, float] | None = None
    radius: float | None = None
    polygon: list[tuple[float, float]] | None = None
    keypoints: list[tuple[float, float, float | None]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CohortRecord:
    image_id: str
    image_path: Path
    relative_path: str
    split: str
    width: int
    height: int
    image_sha256: str
    annotation_sha256: str
    original_id: str
    annotations: tuple[dict[str, Any], ...]
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Cohort:
    split: str
    fingerprint: str
    records: tuple[CohortRecord, ...]
    task: str
    classes: dict[int, str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ComparisonResult:
    """Verified model-comparison outcome and artifact location.

    ``ranking`` contains ordered metric summaries. The verification booleans
    report cohort integrity, training-data overlap/provenance, and cache trust.
    ``settings`` records the exact protocol and inference configuration, while
    ``limitations`` lists caveats that should accompany interpretation.
    """

    location: Path
    ranking: tuple[dict[str, Any], ...]
    cohort_fingerprint: str
    cohort_verified: bool
    training_overlap_detected: bool
    training_provenance_complete: bool
    cache_verified: bool
    cache_statistics: dict[str, Any]
    protocol: str
    settings: dict[str, Any]
    limitations: tuple[str, ...]

    def __repr__(self) -> str:
        return (
            f"ComparisonResult(models={len(self.ranking)}, protocol={self.protocol!r}, "
            f"cohort={self.cohort_fingerprint[:12]!r}, location={str(self.location)!r})"
        )
