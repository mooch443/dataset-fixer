from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable, Mapping, Sequence

from .errors import DatasetValidationError, ValidationIssue
from .io import _label_path_for_image, load_source
from .models import DatasetMetadata, Sample, Task
from .operations import export_dataset, remove_classes, split_dataset
from .tiling import tile_dataset
from .utils import normalize_split
from .validation import validate_dataset
from .visualization import visualize_samples


class Dataset:
    """A validated, immutable view of a YOLO or COCO dataset."""

    def __init__(
        self,
        *,
        location: Path,
        name: str,
        task: Task,
        metadata: DatasetMetadata,
        samples: list[Sample],
        manifest: dict[str, Any],
        data_yaml: Path | None,
        source_format: str,
        warnings: list[str],
    ) -> None:
        self._location = location.resolve()
        self._name = name
        self._task = task
        self._metadata = metadata
        self._samples = samples
        self._manifest = manifest
        self._data_yaml = data_yaml.resolve() if data_yaml else None
        self._source_format = source_format
        self._warnings = tuple(warnings)
        self._provenance = _load_provenance(self._location, samples)
        for sample in self._samples:
            key = str(Path("images") / sample.split / sample.relative_path)
            if key in self._provenance:
                sample.provenance = dict(self._provenance[key])

    @classmethod
    def open(
        cls,
        location: str | Path,
        *,
        task: str | Task | None = None,
        name: str | None = None,
        names: Mapping[int, str] | Sequence[str] | None = None,
        radii: Mapping[int, float] | None = None,
        deep: bool = False,
        progress: bool = True,
    ) -> "Dataset":
        """Load a dataset and run complete consistency validation immediately."""

        requested = Path(location).expanduser().resolve()
        if names is None or isinstance(names, (list, tuple)):
            parsed_names = list(names) if names is not None else None
        else:
            parsed_names = {int(key): str(value) for key, value in names.items()}
        parsed_radii = {int(key): float(value) for key, value in radii.items()} if radii else None
        root, resolved_name, resolved_task, metadata, samples, manifest = load_source(
            requested,
            task=Task.parse(task),
            name=name,
            names=parsed_names,
            radii=parsed_radii,
            progress=progress,
        )
        warnings = validate_dataset(samples, metadata, resolved_task, deep=deep, progress=progress)
        yaml_path = _resolve_data_yaml(requested, root)
        source_format = (
            "yolo"
            if yaml_path is not None or any(path.is_file() for path in (root / "labels").rglob("*.txt"))
            else "coco"
        )
        if source_format == "yolo":
            _assert_no_orphan_labels(root, samples)
        return cls(
            location=root,
            name=resolved_name or "dataset",
            task=resolved_task,
            metadata=metadata,
            samples=samples,
            manifest=manifest,
            data_yaml=yaml_path,
            source_format=source_format,
            warnings=warnings,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def location(self) -> Path:
        return self._location

    @property
    def data_yaml(self) -> Path | None:
        """Canonical training YAML, or ``None`` for a not-yet-exported COCO source."""

        return self._data_yaml

    @property
    def task(self) -> Task:
        return self._task

    @property
    def splits(self) -> tuple[str, ...]:
        present = {sample.split for sample in self._samples}
        return tuple(split for split in ("train", "val", "test") if split in present)

    @property
    def classes(self) -> dict[int, str]:
        return dict(self._metadata.names)

    @property
    def settings(self) -> dict[str, Any]:
        return dict(self._manifest.get("settings") or {})

    @property
    def history(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self._manifest.get("history") or [])

    @property
    def provenance(self) -> dict[str, dict[str, Any]]:
        return {key: dict(value) for key, value in self._provenance.items()}

    @property
    def training_ready(self) -> bool:
        try:
            self.assert_trainable(backend=False)
        except DatasetValidationError:
            return False
        return True

    def split(
        self,
        ratios: Mapping[str, float],
        *,
        destination: str | Path | None = None,
        name: str | None = None,
        source_splits: Iterable[str] | None = None,
        group_by: Callable[[Path], Hashable] | None = None,
        assign: Callable[[Path], str | None] | None = None,
        seed: int = 42,
        visualize: bool = True,
        progress: bool = True,
        dry_run: bool = False,
    ) -> "Dataset":
        return split_dataset(
            self,
            dict(ratios),
            destination=destination,
            name=name,
            source_splits=source_splits,
            group_by=group_by,
            assign=assign,
            seed=seed,
            visualize=visualize,
            progress=progress,
            dry_run=dry_run,
        )

    def remove_classes(
        self,
        classes: Iterable[str | int],
        *,
        destination: str | Path | None = None,
        name: str | None = None,
        splits: Iterable[str] | None = None,
        drop_empty_images: bool = False,
        visualize: bool = True,
        progress: bool = True,
        dry_run: bool = False,
    ) -> "Dataset":
        return remove_classes(
            self,
            classes,
            destination=destination,
            name=name,
            splits=splits,
            drop_empty_images=drop_empty_images,
            visualize=visualize,
            progress=progress,
            dry_run=dry_run,
        )

    def tile(
        self,
        *,
        mode: str = "grid",
        destination: str | Path | None = None,
        name: str | None = None,
        splits: Iterable[str] | None = None,
        tile_size: int = 480,
        overlap: float = 0.2,
        min_area_ratio: float = 0.1,
        negative_tiles: str | float = "all",
        allow_lossy: bool = False,
        visualize: bool = True,
        progress: bool = True,
        dry_run: bool = False,
        **settings: Any,
    ) -> "Dataset":
        return tile_dataset(
            self,
            mode=mode,
            destination=destination,
            name=name,
            splits=splits,
            tile_size=tile_size,
            overlap=overlap,
            min_area_ratio=min_area_ratio,
            negative_tiles=negative_tiles,
            allow_lossy=allow_lossy,
            visualize=visualize,
            progress=progress,
            dry_run=dry_run,
            settings=settings,
        )

    def export(
        self,
        *,
        destination: str | Path | None = None,
        name: str | None = None,
        splits: Iterable[str] | None = None,
        allow_lossy: bool = False,
        visualize: bool = True,
        progress: bool = True,
        dry_run: bool = False,
    ) -> "Dataset":
        return export_dataset(
            self,
            destination=destination,
            name=name,
            splits=splits,
            allow_lossy=allow_lossy,
            visualize=visualize,
            progress=progress,
            dry_run=dry_run,
        )

    def compare_models(
        self,
        models: Any,
        *,
        split: str = "val",
        baseline: str | None = None,
        inference: str = "auto",
        protocol: str = "validation",
        calibration_split: str | None = None,
        training_provenance: str = "required",
        confidence_thresholds: tuple[float, ...] = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85),
        postprocess_thresholds: tuple[float, ...] = (0.75, 0.85, 0.95),
        resolution: int = 480,
        comparison_unit: str = "model",
        cache: bool | str | Path = True,
        notebook_cache: str | Path | None = None,
        write_notebook_cache: bool = False,
        allow_unverified_cache: bool = False,
        visualize: bool = True,
        progress: bool = True,
        destination: str | Path | None = None,
        device: str | None = None,
        seed: int = 42,
        bootstrap_resamples: int = 10_000,
        **inference_settings: Any,
    ):
        """Compare model-plus-inference configurations on one frozen cohort."""

        from .comparison import compare_models

        return compare_models(
            self, models, split=split, baseline=baseline, inference=inference,
            protocol=protocol, calibration_split=calibration_split,
            training_provenance=training_provenance,
            confidence_thresholds=confidence_thresholds,
            postprocess_thresholds=postprocess_thresholds, resolution=resolution,
            comparison_unit=comparison_unit, cache=cache,
            notebook_cache=notebook_cache, write_notebook_cache=write_notebook_cache,
            allow_unverified_cache=allow_unverified_cache, visualize=visualize,
            progress=progress, destination=destination, device=device, seed=seed,
            bootstrap_resamples=bootstrap_resamples, **inference_settings,
        )

    def visualize(
        self,
        *,
        split: str | None = "train",
        n: int = 12,
        seed: int = 42,
        columns: int = 3,
        save_to: str | Path | None = None,
        show: bool = True,
    ):
        if n <= 0 or columns <= 0:
            raise ValueError("n and columns must be positive")
        normalized = normalize_split(split) if split is not None else None
        if normalized is not None and normalized not in self.splits:
            raise ValueError(f"Unknown split {split!r}; available splits are {self.splits}")
        destination = Path(save_to).expanduser().resolve() if save_to else None
        return visualize_samples(
            self._samples,
            self.task,
            self._metadata,
            split=normalized,
            n=n,
            seed=seed,
            columns=columns,
            save_to=destination,
            show=show,
        )

    def assert_trainable(
        self,
        *,
        required_splits: Iterable[str] = ("train", "val"),
        backend: bool | str = "auto",
    ) -> None:
        """Raise before training if structural or installed-backend checks fail."""

        issues: list[ValidationIssue] = []
        required = tuple(normalize_split(split) for split in required_splits)
        by_split = {split: [sample for sample in self._samples if sample.split == split] for split in self.splits}
        for split in required:
            if not by_split.get(split):
                issues.append(
                    ValidationIssue(
                        "Required training split is missing or empty",
                        value=split,
                        expected=f"non-empty {split} split",
                        suggestion="create the split with dataset.split(...) or change required_splits",
                    )
                )
        if not self._metadata.names:
            issues.append(ValidationIssue("Dataset has no class names"))
        if self._source_format == "yolo" and (self.data_yaml is None or not self.data_yaml.is_file()):
            issues.append(ValidationIssue("Canonical data.yaml is missing", source=str(self.location)))
        if self.task is Task.POSE and not self._metadata.kpt_shape:
            issues.append(ValidationIssue("Pose metadata is missing kpt_shape"))
        if self.task is Task.POLO and not self._metadata.radii:
            issues.append(
                ValidationIssue(
                    "POLO metadata is missing class-level radii",
                    expected="a positive radius for every class in data.yaml",
                )
            )
        if issues:
            raise DatasetValidationError(issues)

        should_check_backend = backend is True or (
            isinstance(backend, str) and backend.lower() not in {"auto", "false", "none"}
        )
        if backend == "auto":
            try:
                import ultralytics  # noqa: F401
            except ImportError:
                should_check_backend = False
            else:
                should_check_backend = self.data_yaml is not None
        if should_check_backend:
            if self.data_yaml is None:
                raise DatasetValidationError("COCO sources must be exported before backend training checks")
            try:
                from ultralytics.data.utils import check_det_dataset

                check_det_dataset(str(self.data_yaml), autodownload=False)
            except Exception as exc:
                raise DatasetValidationError(
                    ValidationIssue(
                        "Installed Ultralytics backend rejected the dataset",
                        source=str(self.data_yaml),
                        value=str(exc),
                        suggestion="review the backend message and dataset-fixer validation report",
                    )
                ) from exc

    def __repr__(self) -> str:
        return (
            f"Dataset(name={self.name!r}, task={self.task.value!r}, "
            f"splits={self.splits!r}, location={str(self.location)!r})"
        )


def _resolve_data_yaml(requested: Path, root: Path) -> Path | None:
    if requested.suffix.lower() in {".yaml", ".yml"} and requested.is_file():
        return requested
    for candidate in (root / "data.yaml", root / "dataset.yaml", root / "data.yml"):
        if candidate.is_file():
            return candidate
    return None


def _load_provenance(root: Path, samples: list[Sample]) -> dict[str, dict[str, Any]]:
    path = root / "provenance.jsonl"
    if not path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    issues: list[ValidationIssue] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            records[str(Path(record["output_image"]))] = record
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            issues.append(
                ValidationIssue(
                    "Invalid provenance record",
                    source=str(path),
                    line=line_number,
                    value=str(exc),
                )
            )
    if issues:
        raise DatasetValidationError(issues)
    expected = {str(Path("images") / sample.split / sample.relative_path) for sample in samples}
    missing = expected - records.keys()
    if missing:
        raise DatasetValidationError(
            ValidationIssue(
                "Derived dataset is missing image provenance records",
                source=str(path),
                value=sorted(missing)[:10],
                expected="one record for every output image",
            )
        )
    return records


def _assert_no_orphan_labels(root: Path, samples: list[Sample]) -> None:
    expected = {_label_path_for_image(sample.image_path).resolve() for sample in samples}
    actual = {
        path.resolve()
        for path in root.rglob("*.txt")
        if "labels" in path.parts and path.name not in {"train.txt", "val.txt", "test.txt"}
    }
    orphaned = sorted(actual - expected)
    if orphaned:
        raise DatasetValidationError(
            [
                ValidationIssue(
                    "Label has no image in the configured dataset splits",
                    source=str(path),
                    suggestion="add the image to data.yaml, move the label, or remove the orphan",
                )
                for path in orphaned
            ]
        )
