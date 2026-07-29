from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Hashable, Iterable, Literal, Mapping, Sequence

from .augmentation import augment_dataset, serialize_pipeline
from .errors import DatasetValidationError, ValidationIssue
from .io import _label_path_for_image, load_source
from .models import DatasetMetadata, Sample, Task
from .operations import (
    export_dataset,
    rebalance_empty_dataset,
    remove_classes as materialize_remove_classes,
    rename_classes as materialize_rename_classes,
    split_dataset,
)
from .planning import (
    PlannedOperation,
    derived_name,
    plan_split,
    project_remove_classes,
    resolve_removed_classes,
    resolve_renamed_classes,
    select_empty_images,
)
from .tiling import tile_dataset
from .utils import ensure_safe_destination, normalize_split, settings_fingerprint, slugify
from .validation import validate_dataset
from .visualization import visualize_samples

if TYPE_CHECKING:
    from .comparison.types import ComparisonResult


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
        errors: Literal["raise", "skip"] = "raise",
    ) -> None:
        self._location = location.resolve()
        self._name = name
        self._task = task
        self._metadata = metadata
        self._samples = samples
        self._manifest = manifest
        self._data_yaml = data_yaml.resolve() if data_yaml else None
        self._source_format = source_format
        self._errors = errors
        self._base = base
        self._plan = plan
        self._projection_exact = projection_exact
        self._planned_splits = planned_splits
        self._provenance = _load_provenance(self._location, samples, errors=errors, warnings=warnings)
        self._warnings = tuple(warnings)
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
        task: Literal["detect", "segment", "pose", "polo"] | Task | None = None,
        name: str | None = None,
        names: Mapping[int, str] | Sequence[str] | None = None,
        radii: Mapping[int, float] | None = None,
        deep: bool = False,
        errors: Literal["raise", "skip"] = "raise",
        progress: bool = True,
    ) -> "Dataset":
        """Load YOLO or COCO data, infer metadata, and validate it.

        Parameters:
            location: Dataset root, YOLO YAML file, or COCO JSON file/root.
            task: Annotation task. Pass a :class:`Task` or one of ``"detect"``,
                ``"segment"``, ``"pose"``, or ``"polo"`` when inference is
                ambiguous.
            name: Optional display name overriding source metadata.
            names: Optional zero-based class-name sequence or ID/name mapping.
            radii: Optional POLO class-radius mapping, in source pixels.
            deep: Hash image bytes to detect duplicate content across splits.
            errors: ``"raise"`` fails on the first validation batch.
                ``"skip"`` virtually omits recoverably bad images,
                annotations, duplicate records, and orphan labels while
                retaining an audit trail in :attr:`warnings`. Source files are
                never changed. Errors that make the dataset unusable still
                raise.
            progress: Show image-loading and validation progress bars.

        Returns:
            A materialized, validated dataset index.
        """

        requested = Path(location).expanduser().resolve()
        errors = errors.lower()
        if errors not in {"raise", "skip"}:
            raise ValueError("errors must be 'raise' or 'skip'")
        if names is None or isinstance(names, (list, tuple)):
            parsed_names = list(names) if names is not None else None
        else:
            parsed_names = {int(key): str(value) for key, value in names.items()}
        parsed_radii = {int(key): float(value) for key, value in radii.items()} if radii else None
        warnings: list[str] = []
        root, resolved_name, resolved_task, metadata, samples, manifest = load_source(
            requested,
            task=Task.parse(task),
            name=name,
            names=parsed_names,
            radii=parsed_radii,
            progress=progress,
            errors=errors,
            warnings=warnings,
        )
        warnings.extend(
            validate_dataset(
                samples,
                metadata,
                resolved_task,
                deep=deep,
                progress=progress,
                errors=errors,
            )
        )
        yaml_path = _resolve_data_yaml(requested, root)
        source_format = (
            "yolo"
            if yaml_path is not None
            or any(path.is_file() and "labels" in path.parts for path in root.rglob("*.txt"))
            else "coco"
        )
        if source_format == "yolo":
            _assert_no_orphan_labels(root, samples, errors=errors, warnings=warnings)
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
            errors=errors,
        )

    @property
    def name(self) -> str:
        """Human-readable dataset or virtual-pipeline name."""
        return self._name

    @property
    def location(self) -> Path:
        """Absolute source root, or output root for a materialized derivative."""
        return self._location

    @property
    def data_yaml(self) -> Path | None:
        """Canonical training YAML, or ``None`` while transformations are pending."""
        return None if self._plan else self._data_yaml

    @property
    def task(self) -> Task:
        """Validated annotation task shared by every sample."""
        return self._task

    @property
    def splits(self) -> tuple[str, ...]:
        """Available canonical split names in train/validation/test order."""
        if self._planned_splits is not None:
            return self._planned_splits
        present = {sample.split for sample in self._samples}
        return tuple(split for split in ("train", "val", "test") if split in present)

    @property
    def classes(self) -> dict[int, str]:
        """Copy of the contiguous zero-based class ID/name mapping."""
        return dict(self._metadata.names)

    @property
    def warnings(self) -> tuple[str, ...]:
        """Non-fatal validation and transformation warnings."""
        return self._warnings

    @property
    def settings(self) -> dict[str, Any]:
        """Effective materialized settings or pending virtual-operation records."""
        if self._plan:
            return {"pending_operations": [operation.public_record() for operation in self._plan]}
        return dict(self._manifest.get("settings") or {})

    @property
    def history(self) -> tuple[dict[str, Any], ...]:
        """Immutable transformation history, including pending operations."""
        stored = [dict(item) for item in self._manifest.get("history") or []]
        return tuple([*stored, *(operation.public_record() for operation in self._plan)])

    @property
    def provenance(self) -> dict[str, dict[str, Any]]:
        """Per-output-image lineage keyed by relative output image path."""
        return {key: dict(value) for key, value in self._provenance.items()}

    @property
    def training_ready(self) -> bool:
        """Whether structural trainability checks pass with no pending operations."""
        if self._plan:
            return False
        try:
            self.assert_trainable(backend=False)
        except DatasetValidationError:
            return False
        return True

    def split(
        self,
        ratios: Mapping[Literal["train", "val", "test"], float],
        *,
        name: str | None = None,
        source_splits: Iterable[Literal["train", "val", "test"]] | None = None,
        group_by: Callable[[Path], Hashable] | None = None,
        assign: Callable[[Path], Literal["train", "val", "test"] | None] | None = None,
        seed: int = 42,
        visualize: bool = True,
        progress: bool = True,
    ) -> "Dataset":
        """Plan a deterministic, leakage-aware reassignment of whole images.

        Parameters:
            ratios: Target fractions keyed by ``"train"``, ``"val"``, and/or
                ``"test"``. Positive values are normalized to sum to one.
            name: Optional virtual-derivative name.
            source_splits: Existing splits eligible for reassignment. ``None``
                uses all splits.
            group_by: Optional callback from image path to a stable group key.
                Images with the same key remain in one output split.
            assign: Optional callback that pins an image to a split. Return
                ``None`` to let ratio-based assignment decide.
            seed: Deterministic shuffle seed for unpinned groups.
            visualize: Produce split previews and count reports at export.
            progress: Show export-time progress.

        Returns:
            A virtual dataset with projected split membership.
        """

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
        splits: Iterable[Literal["train", "val", "test"]] | None = None,
        drop_empty_images: bool = False,
        visualize: bool = True,
        progress: bool = True,
    ) -> "Dataset":
        """Plan class removal and compact all surviving class IDs.

        Parameters:
            classes: Class names or integer IDs to remove.
            name: Optional virtual-derivative name.
            splits: Splits affected by removal; ``None`` selects all splits.
            drop_empty_images: Remove images that become annotation-free.
                Otherwise they remain valid negative/background examples.
            visualize: Produce before/after class-count audits at export.
            progress: Show export-time progress.

        Returns:
            A virtual dataset with projected annotations and class metadata.
        """

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

    def rename_classes(
        self,
        renames: Mapping[str | int, str],
        *,
        name: str | None = None,
        visualize: bool = True,
        progress: bool = True,
    ) -> "Dataset":
        """Plan class-name changes without changing class IDs or annotations.

        Parameters:
            renames: Mapping from existing class names or integer IDs to new
                non-empty names. Final class names must remain unique.
            name: Optional virtual-derivative name.
            visualize: Produce a before/after class-name audit table at export.
            progress: Show export-time copying progress.

        Returns:
            A virtual dataset with projected class metadata. All image pixels,
            annotation geometry, class IDs, POLO radii, and pose metadata remain
            unchanged.
        """

        requested = dict(renames)
        renamed, metadata = resolve_renamed_classes(self._metadata, requested)
        settings = {
            "renamed_classes": renamed,
            "class_ids_changed": False,
            "visualize": visualize,
        }
        operation = PlannedOperation(
            "rename-classes",
            {
                "renames": requested,
                "visualize": visualize,
            },
            settings,
        )
        return self._with_plan(
            operation,
            samples=self._samples,
            metadata=metadata,
            name=name,
        )

    def rebalance_empty(
        self,
        max_empty_fraction: float,
        *,
        splits: Iterable[Literal["train", "val", "test"]] | None = ("train",),
        seed: int = 42,
        name: str | None = None,
        visualize: bool = True,
        progress: bool = True,
    ) -> "Dataset":
        """Plan deterministic downsampling of annotation-free images.

        Parameters:
            max_empty_fraction: Maximum fraction of selected output images that
                may have no annotations, in ``[0, 1)``.
            splits: Splits to rebalance. ``None`` selects all splits.
            seed: Deterministic negative-image sampling seed.
            name: Optional virtual-derivative name.
            visualize: Produce before/after background-balance reports.
            progress: Show export-time progress.

        Returns:
            A virtual dataset; positive images are never duplicated or removed.
        """

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
        mode: Literal["grid", "coverage"] = "grid",
        name: str | None = None,
        splits: Iterable[Literal["train", "val", "test"]] | None = None,
        tile_size: int = 480,
        overlap: float = 0.2,
        min_area_ratio: float = 0.1,
        negative_tiles: Literal["all", "none"] | float = "all",
        scale_range: tuple[float, float] = (0.75, 1.25),
        target_appearances_per_object: int = 5,
        sparse_appearances_per_object: int = 1,
        object_appearance_overrides: Mapping[str | int, int] | None = None,
        min_nearby_objects_for_full_coverage: int = 5,
        dense_neighbor_radius_px: float | None = None,
        background_ratio: float = 0.1,
        large_image_threshold: int | None = None,
        max_attempts_per_target: int = 15,
        max_background_attempts_per_tile: int = 15,
        max_tiles_per_source_image: int | None = 100,
        polo_radius_px: float | None = 15.0,
        radius_multiplier: float = 1.0,
        seed: int = 42,
        jpeg_quality: int = 95,
        allow_lossy: bool = False,
        visualize: bool = True,
        progress: bool = True,
    ) -> "Dataset":
        """Plan task-aware grid tiles or coverage-targeted random crops.

        Both modes support detection, segmentation, pose, and POLO annotations.
        The operation is virtual; call :meth:`export` to write pixels and labels.

        Modes:
            ``"grid"``:
                Deterministic edge-aligned source windows. Windows are not
                resized, so there is no random crop or zoom. ``overlap`` and
                ``negative_tiles`` control stride and empty-window retention.
            ``"coverage"``:
                Random object-containing crops sampled from ``scale_range`` and
                resized to ``tile_size``. Sampling tries to reach an appearance
                target for every source object. Set both appearance parameters
                to the same value for a uniform target.

        Common parameters:
            name: Optional name for the virtual derivative.
            splits: Splits to tile; defaults to every available split.
            tile_size: Grid window edge or coverage output edge, in pixels.
            min_area_ratio: Minimum retained fraction of an object's original
                box/mask area. Applies to clipped detection, segmentation, and
                pose annotations in either mode.
            allow_lossy: Permit dropping RLE/multipart masks or keeping the
                largest polygon fragment when one YOLO polygon cannot represent
                the crop exactly.
            visualize: Generate previews and audit images during export.
            progress: Show materialization progress during export.

        Grid parameters:
            overlap: Fractional overlap between adjacent windows in ``[0, 1)``.
            negative_tiles: ``"all"`` keeps every empty window, ``"none"``
                removes them, and a non-negative float caps empty windows to
                that ratio relative to positive windows.

        Coverage parameters:
            scale_range: Inclusive random zoom range. A sampled zoom ``z`` uses
                a source crop edge of approximately ``tile_size / z``.
            target_appearances_per_object: Requested appearances for objects
                with at least ``min_nearby_objects_for_full_coverage`` neighbors.
            sparse_appearances_per_object: Requested appearances for less-dense
                objects.
            object_appearance_overrides: Optional exact targets keyed by an
                annotation's ``source_id`` (integer or string).
            min_nearby_objects_for_full_coverage: Neighbor count that selects
                the dense target.
            dense_neighbor_radius_px: Source-pixel radius used for neighbor
                counting; defaults to half ``tile_size``.
            background_ratio: Requested object-free crops relative to positive
                coverage crops.
            large_image_threshold: Images whose largest edge is at or below
                this value are copied once instead of coverage-sampled.
                ``None`` uses ``tile_size``.
            max_attempts_per_target: Random-sampling budget per requested
                object appearance.
            max_background_attempts_per_tile: Rejection-sampling budget for
                each requested background crop.
            max_tiles_per_source_image: Positive-plus-background cap per large
                source image; ``None`` disables the cap.
            polo_radius_px: POLO source-space radius used for containment and
                output labels. ``None`` uses each annotation's own radius.
            radius_multiplier: Additional POLO output-radius multiplier.
            seed: Deterministic coverage-sampling seed.
            jpeg_quality: JPEG quality for resized coverage tiles.

        Returns:
            A virtual dataset pipeline with deferred pixel generation.
        """

        mode = mode.lower()
        if mode not in {"grid", "coverage"}:
            raise ValueError("mode must be 'grid' or 'coverage'")
        split_values = tuple(splits) if splits is not None else None
        appearance_overrides = dict(object_appearance_overrides or {})
        if mode == "coverage" and appearance_overrides and self._projection_exact:
            source_ids = {
                value
                for sample in self._samples
                for annotation in sample.annotations
                if annotation.source_id is not None
                for value in (annotation.source_id, str(annotation.source_id))
            }
            unknown_overrides = [key for key in appearance_overrides if key not in source_ids]
            if unknown_overrides:
                raise ValueError(
                    f"Unknown object_appearance_overrides keys {unknown_overrides}; "
                    "use annotation source IDs from load warnings or coverage reports"
                )
        coverage_settings = {
            "scale_range": tuple(scale_range),
            "target_appearances_per_object": target_appearances_per_object,
            "sparse_appearances_per_object": sparse_appearances_per_object,
            "object_appearance_overrides": appearance_overrides,
            "min_nearby_objects_for_full_coverage": min_nearby_objects_for_full_coverage,
            "dense_neighbor_radius_px": dense_neighbor_radius_px,
            "background_ratio": background_ratio,
            "large_image_threshold": large_image_threshold,
            "max_attempts_per_target": max_attempts_per_target,
            "max_background_attempts_per_tile": max_background_attempts_per_tile,
            "max_tiles_per_source_image": max_tiles_per_source_image,
            "polo_radius_px": polo_radius_px,
            "radius_multiplier": radius_multiplier,
            "seed": seed,
            "jpeg_quality": jpeg_quality,
        }
        public_settings = {
            "mode": mode, "tile_size": tile_size, "overlap": overlap,
            "min_area_ratio": min_area_ratio, "negative_tiles": negative_tiles,
            "allow_lossy": allow_lossy,
            "splits": sorted({normalize_split(split) for split in split_values} if split_values else set(self.splits)),
            "visualize": visualize,
            **(coverage_settings if mode == "coverage" else {}),
        }
        operation = PlannedOperation(
            "tile",
            {
                "mode": mode, "splits": split_values, "tile_size": tile_size,
                "overlap": overlap, "min_area_ratio": min_area_ratio, "negative_tiles": negative_tiles,
                "allow_lossy": allow_lossy, "visualize": visualize,
                "settings": coverage_settings if mode == "coverage" else {},
            },
            public_settings,
        )
        return self._with_plan(operation, samples=self._samples, name=name, projection_exact=False)

    def augment(
        self,
        transforms: Any,
        *,
        copies: int = 1,
        splits: Iterable[Literal["train", "val", "test"]] | None = ("train",),
        include_original: bool = True,
        min_area: float = 0.0,
        min_visibility: float = 0.0,
        allow_lossy: bool = False,
        seed: int = 42,
        name: str | None = None,
        visualize: bool = True,
        progress: bool = True,
        **compose_args: Any,
    ) -> "Dataset":
        """Plan reproducible, task-aware Albumentations copies for selected splits.

        Detection boxes, segmentation masks, pose keypoints, and POLO circles
        stay synchronized with image transforms. Dataset files are created only
        by :meth:`export`.

        Parameters:
            transforms: Albumentations ``Compose`` object, one transform, a
                transform sequence, or an ``albumentations.to_dict()`` result.
            copies: Augmented outputs generated per selected source image.
            splits: Splits to augment; defaults to training only.
            include_original: Keep each selected source image alongside copies.
            min_area: Albumentations minimum retained box area in pixels.
            min_visibility: Minimum retained box visibility fraction in
                ``[0, 1]``.
            allow_lossy: Allow transformed segmentation masks to collapse to
                their largest YOLO-representable polygon.
            seed: Base seed; each source/copy receives a stable derived seed.
            name: Optional virtual-derivative name.
            visualize: Produce an augmentation preview and count report.
            progress: Show export-time progress.
            **compose_args: Ordinary ``albumentations.Compose`` options used
                only when ``transforms`` is not already a Compose/serialized
                pipeline. Annotation processors, additional targets, and seeds
                are reserved so dataset-fixer can preserve synchronization.

        Returns:
            A virtual dataset with deferred augmentation pixels.
        """

        if isinstance(copies, bool) or not isinstance(copies, int) or copies < 1:
            raise ValueError("copies must be at least 1")
        if not isinstance(include_original, bool):
            raise TypeError("include_original must be a bool")
        if min_area < 0:
            raise ValueError("min_area must be non-negative")
        if not 0 <= min_visibility <= 1:
            raise ValueError("min_visibility must be in [0, 1]")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        split_values = tuple(splits) if splits is not None else tuple(self.splits)
        selected = {normalize_split(split) for split in split_values}
        missing = selected - set(self.splits)
        if missing:
            raise ValueError(f"Unknown augmentation splits {sorted(missing)}; available splits are {self.splits}")
        if self.task is Task.SEGMENT and self._projection_exact:
            unsupported = [
                annotation.source_id
                for sample in self._samples
                if sample.split in selected
                for annotation in sample.annotations
                if annotation.rle is not None or not annotation.polygon
            ]
            if unsupported:
                raise DatasetValidationError(
                    ValidationIssue(
                        "Albumentations requires polygon-representable segmentation instances",
                        value=unsupported[:10],
                        expected="one polygon per annotation",
                        suggestion="export the COCO source with allow_lossy=True before planning augmentation",
                    )
                )
        if self.task is Task.POLO and self._projection_exact:
            clipped_circles = [
                annotation.source_id
                for sample in self._samples
                if sample.split in selected
                for annotation in sample.annotations
                if annotation.point is not None
                and annotation.radius is not None
                and (
                    annotation.point[0] - annotation.radius < 0
                    or annotation.point[1] - annotation.radius < 0
                    or annotation.point[0] + annotation.radius > sample.width
                    or annotation.point[1] + annotation.radius > sample.height
                )
            ]
            if clipped_circles:
                raise DatasetValidationError(
                    ValidationIssue(
                        "POLO radius circles must be fully represented before augmentation",
                        value=clipped_circles[:10],
                        expected="every selected point radius entirely inside its source image",
                        suggestion="remove or correct edge-clipped POLO annotations before augmenting",
                    )
                )
        serialized = serialize_pipeline(transforms, dict(compose_args))
        public_settings = {
            "pipeline": serialized,
            "splits": sorted(selected),
            "copies": copies,
            "include_original": include_original,
            "min_area": min_area,
            "min_visibility": min_visibility,
            "allow_lossy": allow_lossy,
            "seed": seed,
            "visualize": visualize,
        }
        operation = PlannedOperation(
            "augment",
            {
                "pipeline": serialized,
                "splits": split_values,
                "copies": copies,
                "include_original": include_original,
                "min_area": min_area,
                "min_visibility": min_visibility,
                "allow_lossy": allow_lossy,
                "seed": seed,
                "visualize": visualize,
            },
            public_settings,
        )
        return self._with_plan(operation, samples=self._samples, name=name, projection_exact=False)

    def export(
        self,
        *,
        destination: str | Path | None = None,
        name: str | None = None,
        splits: Iterable[Literal["train", "val", "test"]] | None = None,
        allow_lossy: bool = False,
        visualize: bool = True,
        progress: bool = True,
        dry_run: bool = False,
    ) -> "Dataset":
        """Materialize the current dataset or virtual pipeline as canonical YOLO.

        Output is built in a private staging directory, completely validated,
        and atomically published. Existing destinations are never overwritten.

        Parameters:
            destination: Final output root. ``None`` derives a sibling path from
                the dataset name, operation, and settings fingerprint.
            name: Optional output dataset name stored in metadata.
            splits: Splits to publish; ``None`` publishes every available split.
            allow_lossy: Permit explicit lossy conversion of COCO RLE/multipart
                masks to one YOLO polygon.
            visualize: Render pending operation audits and final reports.
            progress: Show copying, transformation, and validation progress.
            dry_run: Validate the plan and print destinations/settings without
                writing a dataset.

        Returns:
            The validated materialized dataset, or the unchanged virtual dataset
            for a dry run.
        """

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
        split: Literal["train", "val", "test"] = "val",
        baseline: str | None = None,
        inference: Literal["auto", "native", "sahi"] = "auto",
        protocol: Literal["validation", "locked", "calibrate_then_test"] = "validation",
        calibration_split: Literal["train", "val", "test"] | None = None,
        training_provenance: Literal["required", "warn", "ignore"] = "required",
        confidence_thresholds: tuple[float, ...] = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85),
        postprocess_thresholds: tuple[float, ...] = (0.75, 0.85, 0.95),
        resolution: int = 480,
        comparison_unit: Literal["model", "system"] = "model",
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
        augment_inference: bool = False,
        precision: Literal["full", "half"] = "full",
        sahi_mode: Literal["standard", "sliced", "combined"] = "sliced",
        slice_height: int | None = None,
        slice_width: int | None = None,
        overlap: float = 0.2,
        overlap_height_ratio: float | None = None,
        overlap_width_ratio: float | None = None,
        postprocess_type: Literal["GREEDYNMM", "NMM", "NMS", "LSNMS"] = "GREEDYNMM",
        postprocess_match_metric: Literal["IOU", "IOS"] = "IOS",
        postprocess_class_agnostic: bool = False,
        model_type: str = "ultralytics",
    ) -> "ComparisonResult":
        """Compare model/inference systems on one frozen, verified cohort.

        Parameters:
            models: Model specs accepted by the comparison parser: paths,
                name/path mappings, or detailed configuration dictionaries.
            split: Fixed evaluation split.
            baseline: Model name used for paired deltas; defaults to the first.
            inference: ``"native"``, ``"sahi"``, or automatic availability-
                based selection. Pose supports native inference only.
            protocol: ``"validation"`` tunes and evaluates on one validation
                cohort; ``"locked"`` evaluates fixed thresholds; and
                ``"calibrate_then_test"`` tunes on ``calibration_split`` before
                evaluation on ``split``.
            calibration_split: Distinct tuning split required by
                ``"calibrate_then_test"``.
            training_provenance: Whether unverifiable training/evaluation
                overlap raises, warns, or is ignored.
            confidence_thresholds: Candidate model confidence thresholds.
            postprocess_thresholds: Candidate NMS/NMM match thresholds.
            resolution: Default model input/slice size.
            comparison_unit: ``"model"`` requires one inference backend across
                candidates; ``"system"`` permits backend-specific systems.
            cache: Enable the verified package cache or provide its path.
            notebook_cache: Optional compatible external prediction cache.
            write_notebook_cache: Write predictions to ``notebook_cache``.
            allow_unverified_cache: Permit exploratory unverified cache input.
            visualize: Write rankings, plots, and qualitative audits.
            progress: Show inference and resampling progress.
            destination: Comparison-report output directory.
            device: Ultralytics/SAHI device identifier.
            seed: Deterministic bootstrap and cohort seed.
            bootstrap_resamples: Paired bootstrap sample count.
            augment_inference: Enable native Ultralytics test-time augmentation.
            precision: Native inference precision.
            sahi_mode: Standard whole-image, sliced-only, or combined SAHI.
            slice_height: SAHI slice height; defaults to ``resolution``.
            slice_width: SAHI slice width; defaults to ``resolution``.
            overlap: Default SAHI overlap ratio for both axes.
            overlap_height_ratio: Optional vertical overlap override.
            overlap_width_ratio: Optional horizontal overlap override.
            postprocess_type: SAHI postprocessor name, such as ``"GREEDYNMM"``.
            postprocess_match_metric: SAHI ``"IOU"`` or ``"IOS"`` matching.
            postprocess_class_agnostic: Merge across classes when true.
            model_type: SAHI detection-model adapter name.

        Returns:
            A :class:`ComparisonResult` containing ranking, verification state,
            settings, limitations, and report location.
        """

        if self._plan:
            raise DatasetValidationError(
                "Model comparison requires a fixed on-disk cohort; call dataset.export(...) first"
            )

        from .comparison import compare_models

        inference_settings = {
            "augment": augment_inference,
            "precision": precision,
            "sahi_mode": sahi_mode,
            "slice_height": resolution if slice_height is None else slice_height,
            "slice_width": resolution if slice_width is None else slice_width,
            "overlap": overlap,
            "postprocess_type": postprocess_type,
            "postprocess_match_metric": postprocess_match_metric,
            "postprocess_class_agnostic": postprocess_class_agnostic,
            "model_type": model_type,
        }
        if overlap_height_ratio is not None:
            inference_settings["overlap_height_ratio"] = overlap_height_ratio
        if overlap_width_ratio is not None:
            inference_settings["overlap_width_ratio"] = overlap_width_ratio

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
        split: Literal["train", "val", "test"] | None = "train",
        n: int = 12,
        seed: int = 42,
        columns: int = 3,
        save_to: str | Path | None = None,
        show: bool = True,
    ) -> Any:
        """Render a deterministic contact sheet with task-aware annotations.

        Parameters:
            split: One split to sample, or ``None`` for all splits.
            n: Maximum number of images.
            seed: Deterministic image-sampling seed.
            columns: Contact-sheet column count.
            save_to: Optional PNG/JPEG/PDF output path.
            show: Display in an active notebook or interactive backend.

        Returns:
            The Matplotlib figure. Deferred pixel-generating pipelines must be
            exported before visualization.
        """

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
        required_splits: Iterable[Literal["train", "val", "test"]] = ("train", "val"),
        backend: bool | Literal["auto"] = "auto",
    ) -> None:
        """Raise if structural or optional Ultralytics checks reject the dataset.

        Parameters:
            required_splits: Splits that must exist and contain images.
            backend: ``False`` performs package-level structural checks only,
                ``True`` additionally requires Ultralytics validation, and
                ``"auto"`` runs it only when Ultralytics is installed.

        Virtual pipelines always fail because their final pixels and YAML do not
        exist until :meth:`export`.
        """

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
            errors=self._errors,
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
                elif operation.kind == "rename-classes":
                    current = materialize_rename_classes(current, **kwargs)
                elif operation.kind == "rebalance-empty":
                    current = rebalance_empty_dataset(current, **kwargs)
                elif operation.kind == "tile":
                    tile_settings = kwargs.pop("settings")
                    current = tile_dataset(current, **kwargs, settings=tile_settings)
                elif operation.kind == "augment":
                    current = augment_dataset(current, **kwargs)
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


def _load_provenance(
    root: Path,
    samples: list[Sample],
    *,
    errors: Literal["raise", "skip"],
    warnings: list[str],
) -> dict[str, dict[str, Any]]:
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
        if errors == "raise":
            raise DatasetValidationError(issues)
        warnings.extend(f"Skipped invalid provenance record: {issue.format()}" for issue in issues)
    expected: set[str] = set()
    for sample in samples:
        try:
            expected.add(str(sample.image_path.resolve().relative_to(root.resolve())))
        except ValueError:
            # Virtual projections still point to immutable source images.
            continue
    missing = expected - records.keys()
    if missing:
        issue = ValidationIssue(
            "Derived dataset is missing image provenance records",
            source=str(path),
            value=sorted(missing)[:10],
            expected="one record for every output image",
        )
        if errors == "raise":
            raise DatasetValidationError(issue)
        warnings.append(f"Ignored incomplete provenance: {issue.format()}")
    return records


def _assert_no_orphan_labels(
    root: Path,
    samples: list[Sample],
    *,
    errors: Literal["raise", "skip"],
    warnings: list[str],
) -> None:
    expected = {_label_path_for_image(sample.image_path).resolve() for sample in samples}
    actual = {
        path.resolve()
        for path in root.rglob("*.txt")
        if "labels" in path.parts and path.name not in {"train.txt", "val.txt", "test.txt"}
    }
    orphaned = sorted(actual - expected)
    if orphaned:
        issues = [
            ValidationIssue(
                "Label has no image in the configured dataset splits",
                source=str(path),
                suggestion="add the image to data.yaml, move the label, or remove the orphan",
            )
            for path in orphaned
        ]
        if errors == "raise":
            raise DatasetValidationError(issues)
        warnings.extend(f"Ignored orphan label: {issue.format()}" for issue in issues)
