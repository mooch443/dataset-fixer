from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable, Mapping, Sequence

from .errors import DatasetValidationError, ValidationIssue
from .io import _label_path_for_image, load_source
from .models import DatasetMetadata, Sample, Task
from .operations import (
    export_dataset,
    rebalance_empty_dataset,
    remove_classes as materialize_remove_classes,
    split_dataset,
)
from .planning import (
    PlannedOperation,
    derived_name,
    plan_split,
    project_remove_classes,
    resolve_removed_classes,
    select_empty_images,
)
from .tiling import tile_dataset
from .utils import ensure_safe_destination, normalize_split, settings_fingerprint, slugify
from .validation import validate_dataset
from .visualization import visualize_samples


class Dataset:
    """A validated dataset or immutable virtual transformation pipeline."""

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
        base: "Dataset | None" = None,
        plan: tuple[PlannedOperation, ...] = (),
        projection_exact: bool = True,
        planned_splits: tuple[str, ...] | None = None,
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
        self._base = base
        self._plan = plan
        self._projection_exact = projection_exact
        self._planned_splits = planned_splits
        self._provenance = _load_provenance(self._location, samples)
        for sample in self._samples:
            try:
                key = str(sample.image_path.resolve().relative_to(self._location))
            except ValueError:
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
            if yaml_path is not None
            or any(path.is_file() and "labels" in path.parts for path in root.rglob("*.txt"))
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
        """Canonical training YAML, or ``None`` while transformations are pending."""
        return None if self._plan else self._data_yaml

    @property
    def task(self) -> Task:
        return self._task

    @property
    def splits(self) -> tuple[str, ...]:
        if self._planned_splits is not None:
            return self._planned_splits
        present = {sample.split for sample in self._samples}
        return tuple(split for split in ("train", "val", "test") if split in present)

    @property
    def classes(self) -> dict[int, str]:
        return dict(self._metadata.names)

    @property
    def settings(self) -> dict[str, Any]:
        if self._plan:
            return {"pending_operations": [operation.public_record() for operation in self._plan]}
        return dict(self._manifest.get("settings") or {})

    @property
    def history(self) -> tuple[dict[str, Any], ...]:
        stored = [dict(item) for item in self._manifest.get("history") or []]
        return tuple([*stored, *(operation.public_record() for operation in self._plan)])

    @property
    def provenance(self) -> dict[str, dict[str, Any]]:
        return {key: dict(value) for key, value in self._provenance.items()}

    @property
    def training_ready(self) -> bool:
        if self._plan:
            return False
        try:
            self.assert_trainable(backend=False)
        except DatasetValidationError:
            return False
        return True

    def split(
        self,
        ratios: Mapping[str, float],
        *,
        name: str | None = None,
        source_splits: Iterable[str] | None = None,
        group_by: Callable[[Path], Hashable] | None = None,
        assign: Callable[[Path], str | None] | None = None,
        seed: int = 42,
        visualize: bool = True,
        progress: bool = True,
    ) -> "Dataset":
        source_split_values = tuple(source_splits) if source_splits is not None else None
        projected, settings, _ = plan_split(
            self._samples, ratios, source_splits=source_split_values, group_by=group_by, assign=assign, seed=seed
        )
        settings["visualize"] = visualize
        operation = PlannedOperation(
            "split",
            {
                "ratios": dict(ratios), "source_splits": source_split_values,
                "group_by": group_by, "assign": assign, "seed": seed, "visualize": visualize,
            },
            settings,
        )
        return self._with_plan(
            operation, samples=projected, name=name,
            planned_splits=tuple(split for split in ("train", "val", "test") if settings["ratios"].get(split, 0) > 0),
        )

    def remove_classes(
        self,
        classes: Iterable[str | int],
        *,
        name: str | None = None,
        splits: Iterable[str] | None = None,
        drop_empty_images: bool = False,
        visualize: bool = True,
        progress: bool = True,
    ) -> "Dataset":
        selectors = tuple(classes)
        split_values = tuple(splits) if splits is not None else None
        removed, mapping, metadata = resolve_removed_classes(self._metadata, selectors)
        selected = {normalize_split(split) for split in split_values} if split_values else set(self.splits)
        projected = project_remove_classes(
            self._samples, selected_splits=selected, mapping=mapping, drop_empty_images=drop_empty_images
        )
        settings = {
            "removed_classes": {class_id: self._metadata.names[class_id] for class_id in sorted(removed)},
            "splits": sorted(selected), "drop_empty_images": drop_empty_images,
            "class_mapping": mapping, "visualize": visualize,
        }
        operation = PlannedOperation(
            "remove-classes",
            {
                "classes": selectors, "splits": split_values,
                "drop_empty_images": drop_empty_images, "visualize": visualize,
            },
            settings,
        )
        return self._with_plan(
            operation,
            samples=projected,
            metadata=metadata,
            name=name,
            planned_splits=tuple(split for split in ("train", "val", "test") if split in selected),
        )

    def rebalance_empty(
        self,
        max_empty_fraction: float,
        *,
        splits: Iterable[str] | None = ("train",),
        seed: int = 42,
        name: str | None = None,
        visualize: bool = True,
        progress: bool = True,
    ) -> "Dataset":
        """Deterministically cap empty images without duplicating source images."""

        split_values = tuple(splits) if splits is not None else None
        selected = {normalize_split(split) for split in split_values} if split_values else set(self.splits)
        projected, summary = select_empty_images(
            self._samples,
            max_empty_fraction=float(max_empty_fraction),
            selected_splits=selected,
            seed=seed,
        )
        settings = {
            "max_empty_fraction": float(max_empty_fraction), "splits": sorted(selected),
            "seed": seed, "summary": summary if self._projection_exact else "resolved during export",
            "visualize": visualize,
        }
        operation = PlannedOperation(
            "rebalance-empty",
            {
                "max_empty_fraction": float(max_empty_fraction),
                "splits": split_values,
                "seed": seed,
                "visualize": visualize,
            },
            settings,
        )
        return self._with_plan(
            operation,
            samples=projected if self._projection_exact else self._samples,
            name=name,
        )

    def tile(
        self,
        *,
        mode: str = "grid",
        name: str | None = None,
        splits: Iterable[str] | None = None,
        tile_size: int = 480,
        overlap: float = 0.2,
        min_area_ratio: float = 0.1,
        negative_tiles: str | float = "all",
        allow_lossy: bool = False,
        visualize: bool = True,
        progress: bool = True,
        **settings: Any,
    ) -> "Dataset":
        retired = sorted({"destination", "dry_run"} & settings.keys())
        if retired:
            raise TypeError(
                f"{', '.join(retired)} can only be used with export(); tile() is an in-memory planning operation"
            )
        mode = mode.lower()
        if mode not in {"grid", "coverage"}:
            raise ValueError("mode must be 'grid' or 'coverage'")
        if mode == "coverage" and self.task is not Task.POLO:
            raise DatasetValidationError("Coverage tiling is only available for task='polo'")
        split_values = tuple(splits) if splits is not None else None
        public_settings = {
            "mode": mode, "tile_size": tile_size, "overlap": overlap,
            "min_area_ratio": min_area_ratio, "negative_tiles": negative_tiles,
            "allow_lossy": allow_lossy,
            "splits": sorted({normalize_split(split) for split in split_values} if split_values else set(self.splits)),
            "visualize": visualize, **settings,
        }
        operation = PlannedOperation(
            "tile",
            {
                "mode": mode, "splits": split_values, "tile_size": tile_size,
                "overlap": overlap, "min_area_ratio": min_area_ratio, "negative_tiles": negative_tiles,
                "allow_lossy": allow_lossy, "visualize": visualize, "settings": dict(settings),
            },
            public_settings,
        )
        return self._with_plan(operation, samples=self._samples, name=name, projection_exact=False)

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
        if self._plan:
            return self._export_plan(
                destination=destination,
                name=name,
                splits=splits,
                allow_lossy=allow_lossy,
                visualize=visualize,
                progress=progress,
                dry_run=dry_run,
            )
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

        if self._plan:
            raise DatasetValidationError(
                "Model comparison requires a fixed on-disk cohort; call dataset.export(...) first"
            )

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
        if self._plan and not self._projection_exact:
            raise DatasetValidationError(
                "This pipeline contains deferred tiling, so exact output pixels do not exist yet; "
                "call export() before visualizing tiled results"
            )
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

        if self._plan:
            raise DatasetValidationError(
                ValidationIssue(
                    "This dataset has pending virtual transformations",
                    value=[operation.kind for operation in self._plan],
                    expected="an exported dataset",
                    suggestion="call dataset.export(...) and train using the returned dataset.data_yaml",
                )
            )

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
        counts = self._summary_counts()
        operations = [operation.kind for operation in self._plan]
        return (
            f"Dataset(name={self.name!r}, task={self.task.value!r}, classes={self.classes!r}, "
            f"splits={counts['splits']!r}, images={counts['images']!r}, "
            f"empty={counts['empty']!r}, materialized={not bool(self._plan)}, "
            f"pending={operations!r}, location={str(self.location)!r})"
        )

    def __str__(self) -> str:
        counts = self._summary_counts()
        classes = ", ".join(f"{class_id}:{name}" for class_id, name in self.classes.items()) or "none"
        empty = counts["empty"]
        empty_text = "pending export" if empty is None else f"{empty} ({counts['empty_fraction']:.1%})"
        state = "virtual pipeline" if self._plan else "materialized"
        lines = [
            f"Dataset {self.name!r} [{self.task.value}; {state}]",
            f"  classes: {classes}",
            f"  splits: {counts['splits']}",
            f"  images: {counts['images']} | annotations: {counts['annotations']} | empty: {empty_text}",
            f"  location: {self.location}",
        ]
        if self._plan:
            lines.append("  pending: " + " → ".join(operation.kind for operation in self._plan))
            lines.append("  export required: data_yaml and training_ready are unavailable until export()")
        elif self.data_yaml is not None:
            lines.append(f"  data_yaml: {self.data_yaml}")
        return "\n".join(lines)

    def _summary_counts(self) -> dict[str, Any]:
        split_counts = {split: sum(sample.split == split for sample in self._samples) for split in self.splits}
        if self._plan and not self._projection_exact:
            return {
                "splits": split_counts,
                "images": "pending export",
                "annotations": "pending export",
                "empty": None,
                "empty_fraction": 0.0,
            }
        images = len(self._samples)
        empty = sum(not sample.annotations for sample in self._samples)
        return {
            "splits": split_counts,
            "images": images,
            "annotations": sum(len(sample.annotations) for sample in self._samples),
            "empty": empty,
            "empty_fraction": empty / images if images else 0.0,
        }

    def _with_plan(
        self,
        operation: PlannedOperation,
        *,
        samples: list[Sample],
        metadata: DatasetMetadata | None = None,
        name: str | None = None,
        projection_exact: bool | None = None,
        planned_splits: tuple[str, ...] | None = None,
    ) -> "Dataset":
        virtual_name = slugify(name) if name else derived_name(self.name, operation.kind, operation.settings)
        return Dataset(
            location=self.location,
            name=virtual_name,
            task=self.task,
            metadata=metadata or self._metadata.copy(),
            samples=samples,
            manifest=self._manifest,
            data_yaml=self._data_yaml,
            source_format=self._source_format,
            warnings=list(self._warnings),
            base=self._base or self,
            plan=(*self._plan, operation),
            projection_exact=self._projection_exact if projection_exact is None else projection_exact,
            planned_splits=planned_splits if planned_splits is not None else self._planned_splits,
        )

    def _export_plan(
        self,
        *,
        destination: str | Path | None,
        name: str | None,
        splits: Iterable[str] | None,
        allow_lossy: bool,
        visualize: bool,
        progress: bool,
        dry_run: bool,
    ) -> "Dataset":
        base = self._base
        if base is None:
            raise RuntimeError("Virtual dataset is missing its materialized base")
        export_settings = {
            "pipeline": [operation.public_record() for operation in self._plan],
            "splits": sorted(normalize_split(split) for split in splits) if splits else list(self.splits),
            "allow_lossy": allow_lossy,
        }
        final_name = slugify(name or self.name)
        final_destination = (
            Path(destination).expanduser().resolve()
            if destination is not None
            else (base.location.parent / f"{final_name}__export__{settings_fingerprint(export_settings)}").resolve()
        )
        final_destination.parent.mkdir(parents=True, exist_ok=True)
        ensure_safe_destination(base.location, final_destination)
        print(self)
        print(f"\nExport destination: {final_destination}")
        if dry_run:
            print("Dry run complete; the virtual pipeline was not materialized.")
            return self

        with tempfile.TemporaryDirectory(
            prefix=f".{final_destination.name}.pipeline-", dir=final_destination.parent
        ) as temporary:
            current = base
            temporary_root = Path(temporary)
            for index, operation in enumerate(self._plan, start=1):
                step_destination = temporary_root / f"step-{index:03d}-{operation.kind}"
                print(f"\nMaterializing pipeline step {index}/{len(self._plan)}: {operation.kind}")
                kwargs = dict(operation.kwargs)
                kwargs["destination"] = step_destination
                kwargs["name"] = None
                kwargs["progress"] = progress
                kwargs["dry_run"] = False
                kwargs["validate_output"] = False
                if operation.kind == "split":
                    current = split_dataset(current, **kwargs)
                elif operation.kind == "remove-classes":
                    current = materialize_remove_classes(current, **kwargs)
                elif operation.kind == "rebalance-empty":
                    current = rebalance_empty_dataset(current, **kwargs)
                elif operation.kind == "tile":
                    tile_settings = kwargs.pop("settings")
                    current = tile_dataset(current, **kwargs, settings=tile_settings)
                else:
                    raise RuntimeError(f"Unknown planned operation {operation.kind!r}")
            return export_dataset(
                current,
                destination=final_destination,
                name=final_name,
                splits=splits,
                allow_lossy=allow_lossy,
                visualize=visualize,
                progress=progress,
                dry_run=False,
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
    expected: set[str] = set()
    for sample in samples:
        try:
            expected.add(str(sample.image_path.resolve().relative_to(root.resolve())))
        except ValueError:
            # Virtual projections still point to immutable source images.
            continue
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
