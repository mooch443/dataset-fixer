from __future__ import annotations

import json
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
class SemanticMaskExport:
    """Published binary semantic-mask dataset and its canonical directories."""

    name: str
    location: Path
    manifest: dict[str, Any]
    manifest_path: Path
    splits: tuple[str, ...]
    image_dirs: dict[str, Path]
    mask_dirs: dict[str, Path]

    def __repr__(self) -> str:
        return (
            f"SemanticMaskExport(name={self.name!r}, splits={self.splits!r}, "
            f"location={str(self.location)!r})"
        )

    @classmethod
    def open(cls, location: str | Path) -> "SemanticMaskExport":
        """Load a previously published semantic-mask export.

        ``location`` may be the export directory or its ``dataset-fixer.json``
        manifest. The manifest and every declared image/mask split directory
        are validated before the artifact is returned.
        """

        from .errors import DatasetValidationError, ValidationIssue

        requested = Path(location).expanduser().resolve()
        manifest_path = requested if requested.is_file() else requested / "dataset-fixer.json"
        root = manifest_path.parent
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise DatasetValidationError(
                ValidationIssue(
                    "Semantic-mask export manifest is missing",
                    source=str(manifest_path),
                    expected="dataset-fixer.json",
                )
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetValidationError(
                f"Unreadable semantic-mask export manifest {manifest_path}: {exc}"
            ) from exc
        if manifest.get("format") != "semantic_masks":
            raise DatasetValidationError(
                ValidationIssue(
                    "Manifest is not a semantic-mask export",
                    source=str(manifest_path),
                    value=manifest.get("format"),
                    expected="format='semantic_masks'",
                )
            )
        declared_splits = manifest.get("splits")
        if not isinstance(declared_splits, list) or not declared_splits:
            raise DatasetValidationError(
                ValidationIssue(
                    "Semantic-mask manifest has no splits",
                    source=str(manifest_path),
                    expected="a non-empty splits list",
                )
            )
        unknown = {str(split) for split in declared_splits} - {"train", "val", "test"}
        if unknown:
            raise DatasetValidationError(
                ValidationIssue(
                    "Semantic-mask manifest has unsupported splits",
                    source=str(manifest_path),
                    value=sorted(unknown),
                    expected="train, val, or test",
                )
            )
        declared = {str(split) for split in declared_splits}
        splits = tuple(split for split in ("train", "val", "test") if split in declared)
        image_dirs = {split: root / split / "images" for split in splits}
        mask_dirs = {split: root / split / "masks" / "0" for split in splits}
        missing = [
            str(path)
            for split in splits
            for path in (image_dirs[split], mask_dirs[split])
            if not path.is_dir()
        ]
        if missing:
            raise DatasetValidationError(
                ValidationIssue(
                    "Semantic-mask split directories are missing",
                    source=str(root),
                    value=missing,
                    expected="<split>/images and <split>/masks/0",
                )
            )
        return cls(
            name=str(manifest.get("name") or root.name),
            location=root,
            manifest=manifest,
            manifest_path=manifest_path,
            splits=splits,
            image_dirs=image_dirs,
            mask_dirs=mask_dirs,
        )

    def compare_models(
        self,
        models: Any,
        *,
        split: Literal["train", "val", "test"] = "val",
        baseline: str | None = None,
        folds: tuple[int | str, ...] = (0,),
        checkpoint: str = "checkpoint_final.pth",
        device: Literal["cpu", "cuda", "mps"] = "cuda",
        workers: int = 2,
        bootstrap_resamples: int = 10_000,
        seed: int = 42,
        keep_predictions: bool = True,
        visualize: bool = True,
        progress: bool = True,
        destination: str | Path | None = None,
    ) -> "SemanticComparisonResult":
        """Compare official nnU-Net v2 models on one frozen mask cohort.

        Parameters:
            models: A trained nnU-Net model folder, a sequence of folders, or
                a name-to-folder/configuration mapping. Configuration mappings
                accept ``model_folder`` (or ``path``), ``folds``, and
                ``checkpoint``, plus a model-specific ``upscale_factor`` that
                reproduces any resizing applied before nnU-Net training. The
                model folder contains ``dataset.json``, ``plans.json``, and
                ``fold_*`` directories.
            split: Exported image/mask split used as the fixed evaluation cohort.
            baseline: Model name used for paired per-case Dice deltas; defaults
                to the first model.
            folds: Default trained folds used for prediction.
            checkpoint: Default checkpoint filename within every selected fold.
            device: Device passed to official nnU-Net prediction.
            workers: Official preprocessing, segmentation-export, and evaluation
                worker count.
            bootstrap_resamples: Paired case-level bootstrap sample count.
            seed: Deterministic bootstrap and qualitative-sample seed.
            keep_predictions: Retain per-model predicted masks in the report.
            visualize: Write a ranking plot and qualitative mask comparison.
            progress: Show semantic-cohort preparation progress. Official
                nnU-Net per-case command output is captured silently and is
                included only when a command fails.
            destination: Comparison-report directory. Existing paths are never
                overwritten.

        Returns:
            A :class:`SemanticComparisonResult` with the official Dice/IoU
            ranking, frozen cohort identity, settings, limitations, and report
            location.
        """

        from .semantic_comparison import compare_nnunet_models

        return compare_nnunet_models(
            self,
            models,
            split=split,
            baseline=baseline,
            folds=folds,
            checkpoint=checkpoint,
            device=device,
            workers=workers,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
            keep_predictions=keep_predictions,
            visualize=visualize,
            progress=progress,
            destination=destination,
        )


@dataclass(frozen=True)
class SemanticComparisonResult:
    """Official nnU-Net comparison outcome for a frozen semantic-mask cohort."""

    location: Path
    ranking: tuple[dict[str, Any], ...]
    cohort_fingerprint: str
    cohort_verified: bool
    split: str
    baseline: str
    settings: dict[str, Any]
    limitations: tuple[str, ...]

    def __repr__(self) -> str:
        return (
            f"SemanticComparisonResult(models={len(self.ranking)}, split={self.split!r}, "
            f"cohort={self.cohort_fingerprint[:12]!r}, location={str(self.location)!r})"
        )
