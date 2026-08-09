from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal


class Task(str, Enum):
    """Supported annotation geometries.

    Values are accepted anywhere a public API requests a task and serialize
    directly to the corresponding lowercase string.
    """

    DETECT = "detect"
    SEGMENT = "segment"
    POSE = "pose"
    POLO = "polo"

    @classmethod
    def parse(
        cls,
        value: Literal["detect", "segment", "pose", "polo", "bbox", "boxes", "keypoints", "locate", "point"]
        | "Task"
        | None,
    ) -> "Task | None":
        """Normalize a task value or supported convenience alias."""
        if value is None or isinstance(value, cls):
            return value
        aliases = {"bbox": "detect", "boxes": "detect", "keypoints": "pose", "locate": "polo", "point": "polo"}
        return cls(aliases.get(value.lower(), value.lower()))


@dataclass
class Annotation:
    class_id: int
    bbox: tuple[float, float, float, float] | None = None  # absolute xyxy
    polygon: list[tuple[float, float]] | None = None
    rle: dict[str, Any] | None = None
    keypoints: list[tuple[float, float, float | None]] | None = None
    point: tuple[float, float] | None = None
    radius: float | None = None
    source_id: str | int | None = None

    def clone(self, **updates: Any) -> "Annotation":
        values = {
            "class_id": self.class_id,
            "bbox": self.bbox,
            "polygon": list(self.polygon) if self.polygon else None,
            "rle": dict(self.rle) if self.rle else None,
            "keypoints": list(self.keypoints) if self.keypoints else None,
            "point": self.point,
            "radius": self.radius,
            "source_id": self.source_id,
        }
        values.update(updates)
        return Annotation(**values)


@dataclass
class Sample:
    image_path: Path
    relative_path: Path
    split: str
    width: int
    height: int
    annotations: list[Annotation] = field(default_factory=list)
    source_sha256: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetMetadata:
    names: dict[int, str]
    channels: int = 3
    radii: dict[int, float] = field(default_factory=dict)
    kpt_shape: tuple[int, int] | None = None
    flip_idx: list[int] | None = None
    kpt_names: dict[int, list[str]] = field(default_factory=dict)
    kpt_oks_sigmas: list[float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "DatasetMetadata":
        return DatasetMetadata(
            names=dict(self.names),
            channels=self.channels,
            radii=dict(self.radii),
            kpt_shape=self.kpt_shape,
            flip_idx=list(self.flip_idx) if self.flip_idx else None,
            kpt_names={k: list(v) for k, v in self.kpt_names.items()},
            kpt_oks_sigmas=list(self.kpt_oks_sigmas) if self.kpt_oks_sigmas else None,
            extra=dict(self.extra),
        )


@dataclass(frozen=True)
class SemanticComparisonResult:
    """Model comparison outcome for a frozen semantic-mask cohort."""

    location: Path
    ranking: tuple[dict[str, Any], ...]
    cohort_fingerprint: str
    cohort_verified: bool
    split: str
    baseline: str | None
    settings: dict[str, Any]
    limitations: tuple[str, ...]

    def __repr__(self) -> str:
        return (
            f"SemanticComparisonResult(models={len(self.ranking)}, split={self.split!r}, "
            f"cohort={self.cohort_fingerprint[:12]!r}, location={str(self.location)!r})"
        )
