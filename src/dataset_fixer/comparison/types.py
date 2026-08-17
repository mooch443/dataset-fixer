from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path
    training_dataset: Path | None = None
    resolution: int = 480
    confidence: float = 0.25
    postprocess: float = 0.7
    inference_overrides: dict[str, Any] = field(default_factory=dict)
    model: Any | None = field(default=None, repr=False, compare=False)

    @cached_property
    def resolved_model(self) -> Any:
        """Generic :class:`dataset_fixer.Model` used for prediction."""

        if self.model is not None:
            return self.model
        from ..model import Model

        return Model(
            self.path,
            name=self.name,
            resolution=self.resolution,
            training_dataset=self.training_dataset,
            inference=str(self.inference_overrides.get("inference", "native")),
            settings={
                key: value
                for key, value in self.inference_overrides.items()
                if key != "inference"
            },
        )


@dataclass
class Prediction:
    class_id: int
    score: float
    bbox: tuple[float, float, float, float] | None = None
    point: tuple[float, float] | None = None
    radius: float | None = None
    polygon: list[tuple[float, float]] | None = None
    polygons: list[list[tuple[float, float]]] | None = None
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

    Parameters:
        location: Directory containing the comparison report and artifacts.
        ranking: Ordered per-model metric summaries.
        cohort_fingerprint: Content identity of the frozen evaluation cohort.
        cohort_verified: Whether cohort inputs passed integrity verification.
        training_overlap_detected: Whether evaluation inputs overlap any known
            model training data.
        training_provenance_complete: Whether training provenance was available
            for every model.
        cache_verified: Whether reused predictions passed identity checks.
        cache_statistics: Counts and details for prediction-cache reuse.
        protocol: Comparison protocol identifier.
        settings: Exact comparison and inference configuration.
        limitations: Caveats that should accompany interpretation.
    """

    location: Path
    ranking: pd.DataFrame
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
