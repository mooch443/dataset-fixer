from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable, Literal, Mapping, Sequence

from PIL import Image
from tqdm.auto import tqdm

from .artifacts import (
    DATASET_INFO_NAME,
    dataset_info_path,
    lineage_path,
    read_lineage,
)
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
    callback_description,
    derived_name,
    plan_split,
    project_remove_classes,
    resolve_removed_classes,
    resolve_renamed_classes,
    select_empty_images,
)
from .semantic_export import export_semantic_masks
from .tiling import tile_dataset
from .tracing import DatasetTrace, trace_dataset
from .utils import IMAGE_SUFFIXES, ensure_safe_destination, normalize_split, settings_fingerprint, slugify
from .validation import validate_dataset
from .validation_audit import ValidationFailureExample, build_load_validation_audit
from .visualization import visualize_samples, visualize_semantic_masks

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
        validation_audit: dict[str, Any] | None = None,
        validation_audit_visualization: Path | None = None,
        provenance: dict[str, dict[str, Any]] | None = None,
        image_dirs: Mapping[str, Path] | None = None,
        mask_dirs: Mapping[str, Path] | None = None,
        mask_paths: Mapping[Path, Path] | None = None,
        mask_statistics: Mapping[Path, Mapping[str, int]] | None = None,
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
        self._image_dirs = {
            split: path.resolve() for split, path in (image_dirs or {}).items()
        }
        self._mask_dirs = {
            split: path.resolve() for split, path in (mask_dirs or {}).items()
        }
        self._mask_paths = {
            image.resolve(): mask.resolve() for image, mask in (mask_paths or {}).items()
        }
        self._mask_statistics = {
            image.resolve(): dict(statistics)
            for image, statistics in (mask_statistics or {}).items()
        }
        stored_audit = validation_audit or (
            (manifest.get("validation") or {}).get("load_validation")
            if isinstance(manifest.get("validation"), dict)
            else None
        )
        self._validation_audit = dict(
            stored_audit
            or {
                "status": "passed",
                "skipped_count": 0,
                "counts_by_category": {},
                "visualized_count": 0,
                "max_visualized_examples": 4,
                "report": None,
                "visualization": None,
            }
        )
        self._validation_audit_visualization = validation_audit_visualization
        if self._validation_audit_visualization is None and self._validation_audit.get("visualization"):
            candidate = Path(str(self._validation_audit["visualization"]))
            self._validation_audit_visualization = (
                candidate if candidate.is_absolute() else self._location / candidate
            )
        self._provenance = (
            provenance
            if provenance is not None
            else _load_provenance(self._location, samples, errors=errors, warnings=warnings)
        )
        self._warnings = tuple(warnings)
        for sample in self._samples:
            try:
                key = str(sample.image_path.resolve().relative_to(self._location))
            except ValueError:
                key = str(Path("images") / sample.split / sample.relative_path)
            if key in self._provenance:
                sample.provenance = self._provenance[key]

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
        """Load YOLO, COCO, or semantic-mask data and validate it.

        Parameters:
            location: Dataset root, YOLO YAML, COCO JSON, or a
                ``reports/dataset-info.json`` semantic-mask manifest.
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
                retaining an audit trail in :attr:`warnings` and
                :attr:`validation_audit`. Up to four failures are visualized
                outside the source tree. Source files are never changed.
                Errors that make the dataset unusable still raise.
            progress: Show image-loading and validation progress bars.

        Returns:
            A materialized, validated dataset index.
        """

        # ZIP handling belongs to the internal source resolver so this public
        # signature remains identical for directories, manifests, and archives.
        from .sources import resolve_dataset_source

        requested = resolve_dataset_source(location, progress=progress)
        errors = errors.lower()
        if errors not in {"raise", "skip"}:
            raise ValueError("errors must be 'raise' or 'skip'")
        if names is None or isinstance(names, (list, tuple)):
            parsed_names = list(names) if names is not None else None
        else:
            parsed_names = {int(key): str(value) for key, value in names.items()}
        parsed_radii = {int(key): float(value) for key, value in radii.items()} if radii else None
        semantic_manifest = _semantic_mask_manifest(requested)
        if semantic_manifest is not None:
            return cls._open_semantic_masks(
                requested,
                semantic_manifest,
                task=Task.parse(task),
                name=name,
                names=parsed_names,
                deep=deep,
                errors=errors,
                progress=progress,
            )
        if requested.is_file() and requested.name == DATASET_INFO_NAME:
            requested = requested.parent.parent
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
        failure_examples: list[ValidationFailureExample] = []
        warnings.extend(
            validate_dataset(
                samples,
                metadata,
                resolved_task,
                deep=deep,
                progress=progress,
                errors=errors,
                failure_examples=failure_examples,
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
        dataset = cls(
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
        audit, visualization = build_load_validation_audit(
            dataset._warnings,
            failure_examples,
            dataset._samples,
            resolved_task,
            metadata,
            dataset_name=dataset.name,
        )
        if int(audit.get("skipped_count", 0)) > 0 or int(
            dataset._validation_audit.get("skipped_count", 0)
        ) == 0:
            dataset._validation_audit = audit
            dataset._validation_audit_visualization = visualization
        return dataset

    @classmethod
    def _open_semantic_masks(
        cls,
        requested: Path,
        manifest: dict[str, Any],
        *,
        task: Task | None,
        name: str | None,
        names: dict[int, str] | list[str] | None,
        deep: bool,
        errors: Literal["raise", "skip"],
        progress: bool,
    ) -> "Dataset":
        root = requested.parent if requested.is_file() else requested
        if requested.is_file() and requested.name == DATASET_INFO_NAME and root.name == "reports":
            root = root.parent
        if task is not None and task is not Task.SEGMENT:
            raise DatasetValidationError(
                ValidationIssue(
                    "Semantic-mask datasets require the segmentation task",
                    value=task.value,
                    expected="task='segment' or no task override",
                )
            )
        manifest_names = _class_names(manifest.get("classes"))
        if names is None:
            resolved_names = manifest_names or {0: "foreground"}
        elif isinstance(names, list):
            resolved_names = {index: value for index, value in enumerate(names)}
        else:
            resolved_names = dict(names)
        metadata = DatasetMetadata(names=resolved_names)
        (
            samples,
            image_dirs,
            mask_dirs,
            mask_paths,
            mask_statistics,
            load_warnings,
            failure_examples,
        ) = _load_semantic_mask_samples(
            root,
            manifest,
            progress=progress,
            errors=errors,
        )
        load_warnings.extend(
            validate_dataset(
                samples,
                metadata,
                Task.SEGMENT,
                deep=deep,
                progress=progress,
                errors=errors,
                failure_examples=failure_examples,
            )
        )
        inherited_warnings = [str(value) for value in manifest.get("warnings") or []]
        dataset = cls(
            location=root,
            name=name or str(manifest.get("name") or root.name),
            task=Task.SEGMENT,
            metadata=metadata,
            samples=samples,
            manifest=manifest,
            data_yaml=None,
            source_format="semantic_masks",
            warnings=[*inherited_warnings, *load_warnings],
            errors=errors,
            image_dirs=image_dirs,
            mask_dirs=mask_dirs,
            mask_paths=mask_paths,
            mask_statistics=mask_statistics,
        )
        if load_warnings:
            audit, visualization = build_load_validation_audit(
                load_warnings,
                failure_examples,
                samples,
                Task.SEGMENT,
                metadata,
                dataset_name=dataset.name,
            )
            dataset._validation_audit = audit
            dataset._validation_audit_visualization = visualization
        return dataset

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
        """Canonical YOLO training YAML, or ``None`` for masks and virtual plans."""
        return None if self._plan else self._data_yaml

    @property
    def format(self) -> str:
        """Physical annotation format: YOLO, COCO, or binary semantic masks."""
        return self._source_format

    @property
    def manifest(self) -> dict[str, Any]:
        """Copy of the loaded or generated dataset manifest."""
        return dict(self._manifest)

    @property
    def manifest_path(self) -> Path | None:
        """Dataset-fixer manifest path when one exists on disk."""
        candidate = dataset_info_path(self.location)
        if candidate.is_file():
            return candidate
        legacy = self.location / "dataset-fixer.json"
        return legacy if legacy.is_file() else None

    def trace(
        self,
        *,
        search_paths: Iterable[str | Path] = (),
        path_rewrites: Mapping[str | Path, str | Path] | None = None,
    ) -> DatasetTrace:
        """Resolve physical ancestry and exact source mappings for present samples.

        Parameters:
            search_paths: Additional roots in which generated datasets are
                discovered by stable dataset ID.
            path_rewrites: Old absolute path prefixes mapped to their new
                locations after a move or remount.

        Returns:
            A concise dataset chain with exact present-sample and tile mappings.
        """

        return trace_dataset(
            self,
            search_paths=search_paths,
            path_rewrites=path_rewrites,
        )

    def update(
        self,
        dest: str | Path | None = None,
        *,
        progress: bool = True,
    ) -> "Dataset":
        """Upgrade this materialized dataset to the latest artifact schema.

        The upgrade rewrites reports and metadata only. Images, labels, and
        masks are never modified, and any legacy ``path`` key is removed from
        the training YAML so the dataset resolves wherever it is mounted.

        Parameters:
            dest: Optional destination root. A string is expanded and converted
                to :class:`Path`. ``None`` updates reports atomically in place;
                a distinct path creates an upgraded copy without modifying the
                source dataset.
            progress: Show progress for copying, hashing and indexing, report
                generation, and validation. Pass ``False`` to run silently.

        Returns:
            The validated, upgraded materialized dataset.
        """

        if self._plan:
            raise DatasetValidationError(
                "Dataset.update requires a materialized dataset; call export(...) first"
            )
        from .updating import update_dataset

        return update_dataset(self, dest=dest, progress=progress)

    @property
    def image_dirs(self) -> dict[str, Path]:
        """Canonical image directory for each split when known."""
        if self._image_dirs:
            return dict(self._image_dirs)
        directories: dict[str, Path] = {}
        for split in self.splits:
            paths = [sample.image_path.resolve() for sample in self._samples if sample.split == split]
            if paths:
                directories[split] = Path(_common_parent(paths))
        return directories

    @property
    def mask_dirs(self) -> dict[str, Path]:
        """Binary ground-truth mask directory for each semantic-mask split."""
        return dict(self._mask_dirs)

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
    def validation_audit(self) -> dict[str, Any]:
        """Load-time skip totals, categories, and bounded visualization metadata."""
        return dict(self._validation_audit)

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

    def _require_vector_annotations(self, operation: str) -> None:
        if self.format == "semantic_masks":
            raise DatasetValidationError(
                ValidationIssue(
                    f"{operation} requires vector annotations",
                    value=self.format,
                    expected="a YOLO or COCO Dataset with polygon labels",
                    suggestion=(
                        "load the corresponding YOLO/COCO source; binary foreground-union "
                        "masks cannot reconstruct class or instance polygons"
                    ),
                )
            )

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
            ratios: Target weights keyed by ``"train"``, ``"val"``, and/or
                ``"test"``. Values such as ``0.8/0.2`` and ``80/20`` are
                equivalent because finite non-negative weights are normalized
                to sum to one. Every selected input image is assigned to
                exactly one requested output split. Whole groups and explicit
                assignments can make achieved fractions differ from targets.
            name: Optional virtual-derivative name.
            source_splits: Existing splits included in the reassignment corpus.
                ``None`` includes all splits; unselected splits are omitted from
                this operation's output.
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

        self._require_vector_annotations("Dataset.split")
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
        merge_into: str | int | None = None,
        name: str | None = None,
        splits: Iterable[Literal["train", "val", "test"]] | None = None,
        drop_empty_images: bool = False,
        visualize: bool = True,
        progress: bool = True,
    ) -> "Dataset":
        """Plan class removal and compact all surviving class IDs.

        Parameters:
            classes: Class names or integer IDs to remove from the class schema.
            merge_into: Optional surviving class name or integer ID. When set,
                annotations belonging to the removed classes are reassigned to
                this class instead of being discarded.
            name: Optional virtual-derivative name.
            splits: Splits included in the output; ``None`` selects all splits.
                Unselected splits are omitted because class metadata and IDs
                are compacted globally.
            drop_empty_images: Remove every selected output image with no
                remaining annotations, including inputs that were already
                empty. Otherwise they remain negative/background examples.
            visualize: Produce before/after class-count audits at export.
            progress: Show export-time progress.

        Returns:
            A virtual dataset with projected annotations and class metadata.
        """

        self._require_vector_annotations("Dataset.remove_classes")
        selectors = tuple(classes)
        split_values = tuple(splits) if splits is not None else None
        removed, mapping, metadata = resolve_removed_classes(
            self._metadata,
            selectors,
            merge_into=merge_into,
        )
        selected = {normalize_split(split) for split in split_values} if split_values else set(self.splits)
        projected = project_remove_classes(
            self._samples, selected_splits=selected, mapping=mapping, drop_empty_images=drop_empty_images
        )
        settings = {
            "removed_classes": {class_id: self._metadata.names[class_id] for class_id in sorted(removed)},
            "splits": sorted(selected), "drop_empty_images": drop_empty_images,
            "class_mapping": mapping, "visualize": visualize,
        }
        if merge_into is not None:
            output_class_id = mapping[next(iter(removed))]
            settings["merge_into"] = {
                "selector": merge_into,
                "output_class_id": output_class_id,
                "output_class_name": metadata.names[output_class_id],
            }
        operation = PlannedOperation(
            "remove-classes",
            {
                "classes": selectors, "splits": split_values,
                "merge_into": merge_into,
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
        progress: bool = True,
    ) -> "Dataset":
        """Plan class-name changes without changing class IDs or annotations.

        Parameters:
            renames: Mapping from existing class names or integer IDs to new
                non-empty names. Final class names must remain unique.
            name: Optional virtual-derivative name.
            progress: Show export-time copying progress.

        Returns:
            A virtual dataset with projected class metadata. All image pixels,
            annotation geometry, class IDs, POLO radii, and pose metadata remain
            unchanged.
        """

        self._require_vector_annotations("Dataset.rename_classes")
        requested = dict(renames)
        renamed, metadata = resolve_renamed_classes(self._metadata, requested)
        settings = {
            "renamed_classes": renamed,
            "class_ids_changed": False,
        }
        operation = PlannedOperation(
            "rename-classes",
            {
                "renames": requested,
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
            splits: Splits whose complete output composition is rebalanced.
                ``None`` selects all splits. Unselected splits are copied
                unchanged.
            seed: Deterministic negative-image sampling seed.
            name: Optional virtual-derivative name.
            visualize: Produce before/after background-balance reports.
            progress: Show export-time progress.

        Returns:
            A virtual dataset; positive images are never duplicated or removed.
        """

        self._require_vector_annotations("Dataset.rebalance_empty")
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
        max_background_tiles_per_source_image: int | None = None,
        polo_radius_px: float | None = 15.0,
        radius_multiplier: float = 1.0,
        seed: int = 42,
        jpeg_quality: int = 95,
        allow_lossy: bool = False,
        crop_transforms: Any | None = None,
        val_crop_transforms: Any | None = None,
        augment_val: bool = False,
        background_filter: Callable[[Image.Image], bool] | None = None,
        errors: Literal["raise", "skip"] = "raise",
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
            splits: Splits included in the tiled output; defaults to every
                available split. Unselected splits are omitted.
            tile_size: Grid window edge or coverage output edge, in pixels.
            min_area_ratio: Minimum retained fraction of an object's original
                box/mask area. Applies to clipped detection, segmentation, and
                pose annotations in either mode.
            allow_lossy: Permit dropping RLE/multipart masks or keeping the
                largest polygon fragment when one YOLO polygon cannot represent
                the crop exactly. In coverage mode, ``False`` also rejects and
                resamples crops that cut through any source annotation; export
                raises rather than returning fewer requested appearances when
                no lossless replacement can be found.
            crop_transforms: Optional serializable Albumentations pipeline
                applied to a virtual full-source view before each coverage
                crop is selected. This is supported only in coverage mode.
            val_crop_transforms: Optional separate pipeline for validation
                crops, used instead of ``crop_transforms`` when
                ``augment_val`` is set. Selecting a checkpoint against
                appearance changes the deployed model never sees is
                misleading, so validation can keep geometry-only
                augmentation while training keeps the full pipeline.
            augment_val: Apply ``crop_transforms`` -- or
                ``val_crop_transforms`` when given -- to validation crops as
                well as training crops. Test crops are never augmented.
            background_filter: Optional predicate called with each annotation-free
                candidate as an RGB :class:`PIL.Image.Image`. Truthy keeps the
                candidate and falsey discards it. It applies to copied empty
                source images, ordinary background crops, and virtual-camera
                background crops in coverage mode, plus negative grid windows.
                Positive tiles are never passed to the predicate.
            errors: ``"raise"`` aborts export when a crop produces geometry
                that YOLO cannot represent. ``"skip"`` rejects that crop
                candidate, records the detailed reason, and continues sampling
                coverage replacements or omits the affected grid window.
            visualize: Generate previews and audit images during export.
            progress: Show materialization progress during export.

        Grid parameters:
            overlap: Fractional overlap between adjacent windows in ``[0, 1)``.
            negative_tiles: ``"all"`` keeps every empty window, ``"none"``
                removes them, and a float in ``[0, 1)`` targets that final
                background fraction independently in each selected split. The
                calculation includes uncropped small images and uses the
                nearest achievable whole-image count. The operation raises if
                too few empty grid windows exist.

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
            background_ratio: Target fraction of all output images that contain
                no annotations, applied independently to each selected split.
                This includes copied small images and generated crops. Existing
                empty images and object-free regions cropped from populated
                images are sampled in an equal 50/50 mix where possible. If
                one source type cannot supply its half, the other type fills
                the remainder and the export records a warning and exact counts
                in ``coverage_summary/tile_summary.csv``.
                The nearest achievable whole-image count is used, and export
                raises rather than silently returning a different count when
                insufficient object-free crops exist. Must be in ``[0, 1)``.
            large_image_threshold: Images whose largest edge is at or below
                this value are copied once instead of coverage-sampled.
                ``None`` uses ``tile_size``.
            max_attempts_per_target: Random-sampling budget per requested
                object appearance.
            max_background_attempts_per_tile: Rejection-sampling budget for
                each requested background crop.
            max_tiles_per_source_image: Positive-plus-background cap per large
                source image; ``None`` disables the cap.
            max_background_tiles_per_source_image: Cap on background tiles
                taken from any single source image, applied on top of
                ``max_tiles_per_source_image``. ``None`` disables it, so one
                large empty image can supply the whole background quota.
            polo_radius_px: POLO source-space radius used for containment and
                output labels. ``None`` uses each annotation's own radius.
            radius_multiplier: Additional POLO output-radius multiplier.
            seed: Deterministic coverage-sampling seed.
            jpeg_quality: JPEG quality for resized coverage tiles.

        Returns:
            A virtual dataset pipeline with deferred pixel generation.
        """

        self._require_vector_annotations("Dataset.tile")
        mode = mode.lower()
        if mode not in {"grid", "coverage"}:
            raise ValueError("mode must be 'grid' or 'coverage'")
        errors = errors.lower()
        if errors not in {"raise", "skip"}:
            raise ValueError("errors must be 'raise' or 'skip'")
        if not isinstance(augment_val, bool):
            raise TypeError("augment_val must be a bool")
        if crop_transforms is not None and mode != "coverage":
            raise ValueError("crop_transforms is supported only when mode='coverage'")
        if augment_val and crop_transforms is None and val_crop_transforms is None:
            raise ValueError(
                "augment_val=True requires crop_transforms or val_crop_transforms"
            )
        if val_crop_transforms is not None and not augment_val:
            raise ValueError(
                "val_crop_transforms requires augment_val=True; without it "
                "validation crops are never augmented"
            )
        if val_crop_transforms is not None and mode != "coverage":
            raise ValueError(
                "val_crop_transforms is supported only when mode='coverage'"
            )
        if background_filter is not None and not callable(background_filter):
            raise TypeError("background_filter must be callable or None")
        crop_pipeline = serialize_pipeline(crop_transforms, {}) if crop_transforms is not None else None
        val_crop_pipeline = (
            serialize_pipeline(val_crop_transforms, {})
            if val_crop_transforms is not None
            else None
        )
        background_filter_description = callback_description(background_filter)
        if mode == "grid":
            if not (
                negative_tiles in {"all", "none"}
                or (
                    isinstance(negative_tiles, (int, float))
                    and not isinstance(negative_tiles, bool)
                    and math.isfinite(float(negative_tiles))
                    and 0 <= float(negative_tiles) < 1
                )
            ):
                raise ValueError(
                    "negative_tiles must be 'all', 'none', or a finite final background fraction in [0, 1)"
                )
        elif (
            not math.isfinite(float(background_ratio))
            or not 0 <= float(background_ratio) < 1
        ):
            raise ValueError("background_ratio must be a finite final background fraction in [0, 1)")
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
            "max_background_tiles_per_source_image": max_background_tiles_per_source_image,
            "polo_radius_px": polo_radius_px,
            "radius_multiplier": radius_multiplier,
            "seed": seed,
            "jpeg_quality": jpeg_quality,
            "crop_pipeline": crop_pipeline,
            "augment_val": augment_val,
            "val_crop_pipeline": val_crop_pipeline,
        }
        public_settings = {
            "mode": mode, "tile_size": tile_size, "overlap": overlap,
            "min_area_ratio": min_area_ratio, "negative_tiles": negative_tiles,
            "allow_lossy": allow_lossy,
            "background_filter": background_filter_description,
            "errors": errors,
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
                "background_filter": background_filter,
                "background_filter_description": background_filter_description,
                "errors": errors,
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
            splits: Splits to augment; defaults to training only. Unselected
                splits are copied once unchanged. Passing ``None`` selects all
                available splits.
            include_original: Keep each selected source image alongside its
                augmented copies. It does not affect unselected splits.
            min_area: Albumentations minimum retained box area in pixels.
            min_visibility: Per-box minimum fraction of its pre-transform area
                that must remain visible, in ``[0, 1]``. This filters
                annotations; it is not an output-image composition ratio.
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

        self._require_vector_annotations("Dataset.augment")
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
        format: Literal["yolo", "semantic_masks"] = "yolo",
        splits: Iterable[Literal["train", "val", "test"]] | None = None,
        allow_lossy: bool = False,
        visualize: bool = True,
        progress: bool = True,
        dry_run: bool = False,
    ) -> "Dataset":
        """Materialize the current dataset or virtual pipeline in a supported format.

        Output is built in a private staging directory, completely validated,
        and atomically published. Existing destinations are never overwritten.
        When the latest split used ``group_by``, export also verifies physical
        group isolation across every current split and records aggregate group
        statistics without listing individual image paths.

        Parameters:
            destination: Final output root. ``None`` derives a sibling path from
                the dataset name, operation, and settings fingerprint.
            name: Optional output dataset name stored in metadata.
            format: ``"yolo"`` preserves the canonical training layout and
                returns a :class:`Dataset`. ``"semantic_masks"`` writes binary
                foreground-union masks and also returns a :class:`Dataset`.
            splits: Splits included in the published output; ``None`` publishes
                every available split. Unselected splits are omitted.
            allow_lossy: Permit explicit lossy conversion of COCO RLE/multipart
                masks to one YOLO polygon.
            visualize: Render pending operation audits and final reports.
            progress: Show copying, transformation, and validation progress.
            dry_run: Validate the plan and print destinations/settings without
                writing a dataset.

        Returns:
            The validated materialized dataset. Dry runs return the unchanged
            virtual dataset.
        """

        self._require_vector_annotations("Dataset.export")
        format = format.lower()
        if format not in {"yolo", "semantic_masks"}:
            raise ValueError("format must be 'yolo' or 'semantic_masks'")
        if format == "semantic_masks" and allow_lossy:
            raise ValueError("allow_lossy applies only to YOLO export; semantic masks use polygon unions directly")

        if self._plan:
            return self._export_plan(
                destination=destination,
                name=name,
                format=format,
                splits=splits,
                allow_lossy=allow_lossy,
                visualize=visualize,
                progress=progress,
                dry_run=dry_run,
            )
        if format == "semantic_masks":
            return export_semantic_masks(
                self,
                destination=destination,
                name=name,
                splits=splits,
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

    def export_formats(
        self,
        destinations: Mapping[Literal["yolo", "semantic_masks"], str | Path],
        *,
        name: str | None = None,
        splits: Iterable[Literal["train", "val", "test"]] | None = None,
        allow_lossy: bool = False,
        visualize: bool = True,
        progress: bool = True,
        dry_run: bool = False,
    ) -> "dict[str, Dataset]":
        """Export one pipeline to multiple training formats in a safe order.

        YOLO is materialized first whenever requested. A requested semantic-mask
        export is then produced from that validated YOLO result, so a virtual
        transformation pipeline runs only once. Every format needs an explicit,
        separate destination and all destinations are preflighted before any
        output is written.

        Parameters:
            destinations: Mapping from ``"yolo"`` and/or ``"semantic_masks"``
                to their final output roots.
            name: Optional dataset name stored in each output's metadata.
            splits: Splits published in every requested format.
            allow_lossy: Permit lossy conversion for the YOLO output. This is
                rejected when YOLO is not requested.
            visualize: Render pending operation audits and final reports.
            progress: Show export progress and ETA.
            dry_run: Validate and print every export without writing outputs.

        Returns:
            A mapping with one validated output object per requested format.
            Dry runs map each format to the unchanged virtual dataset.
        """

        normalized: dict[str, Path] = {}
        for raw_format, destination in destinations.items():
            if not isinstance(raw_format, str):
                raise TypeError("export format keys must be strings")
            export_format = raw_format.lower()
            if export_format not in {"yolo", "semantic_masks"}:
                raise ValueError("export formats must be 'yolo' and/or 'semantic_masks'")
            normalized[export_format] = Path(destination).expanduser().resolve()
        if not normalized:
            raise ValueError("destinations must request at least one export format")
        if allow_lossy and "yolo" not in normalized:
            raise ValueError("allow_lossy applies only when a YOLO export is requested")
        if "semantic_masks" in normalized and self.task is not Task.SEGMENT:
            raise DatasetValidationError(
                ValidationIssue(
                    "Semantic-mask export requires a segmentation dataset",
                    value=self.task.value,
                    expected="task='segment'",
                )
            )

        resolved_destinations = list(normalized.values())
        if len(set(resolved_destinations)) != len(resolved_destinations):
            raise ValueError("Each export format requires a separate destination")
        for destination in resolved_destinations:
            ensure_safe_destination(self.location, destination)
        for index, first in enumerate(resolved_destinations):
            for second in resolved_destinations[index + 1 :]:
                if first in second.parents or second in first.parents:
                    raise ValueError("Export destinations cannot contain one another")

        selected_splits = tuple(splits) if splits is not None else None
        self._require_vector_annotations("Dataset.export_formats")
        results: dict[str, Dataset] = {}
        semantic_source: Dataset = self
        if "yolo" in normalized:
            yolo = self.export(
                destination=normalized["yolo"],
                name=name,
                format="yolo",
                splits=selected_splits,
                allow_lossy=allow_lossy,
                visualize=visualize,
                progress=progress,
                dry_run=dry_run,
            )
            if not isinstance(yolo, Dataset):
                raise RuntimeError("YOLO export unexpectedly returned a non-Dataset result")
            results["yolo"] = yolo
            semantic_source = yolo
        if "semantic_masks" in normalized:
            semantic = semantic_source.export(
                destination=normalized["semantic_masks"],
                name=name,
                format="semantic_masks",
                splits=selected_splits,
                visualize=visualize,
                progress=progress,
                dry_run=dry_run,
            )
            results["semantic_masks"] = semantic
        return results

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
        if self.format == "semantic_masks":
            return visualize_semantic_masks(
                self._samples,
                self._mask_paths,
                split=normalized,
                n=n,
                seed=seed,
                columns=columns,
                save_to=destination,
                show=show,
            )
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
        available_splits = {sample.split for sample in self._samples}
        for split in required:
            if split not in available_splits:
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
                if self.format == "semantic_masks":
                    raise DatasetValidationError(
                        "Ultralytics backend checks do not apply to binary semantic-mask datasets"
                    )
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
        empty = counts["empty"]
        empty_text = (
            "pending export"
            if empty is None
            else f"{empty} ({counts['empty_fraction']:.1%})"
        )
        state = "virtual pipeline" if self._plan else "materialized"
        lines = [f"Dataset {self.name!r} [{self.task.value}; {state}; {self._source_format}]"]
        if self.format == "semantic_masks":
            lines.extend(
                [
                    f"  images: {counts['images']} | masks: {counts['masks']} | "
                    f"non-empty: {counts['annotated']} | empty: {empty_text}",
                    f"  source classes: {len(self.classes)} | splits: {len(self.splits)} | "
                    "class handling: foreground union",
                ]
            )
        else:
            lines.extend(
                [
                    f"  images: {counts['images']} | annotations: {counts['annotations']} | "
                    f"empty: {empty_text}",
                    f"  classes: {len(self.classes)} | splits: {len(self.splits)}",
                ]
            )
        if counts["image_sizes"] is not None:
            image_sizes = counts["image_sizes"]
            if len(image_sizes) == 1:
                width, height = image_sizes[0]
                lines.append(f"  image size: {width}x{height}")
            elif image_sizes:
                widths = [width for width, _ in image_sizes]
                heights = [height for _, height in image_sizes]
                lines.append(
                    f"  image sizes: {len(image_sizes)} unique | "
                    f"width {min(widths)}-{max(widths)} | "
                    f"height {min(heights)}-{max(heights)}"
                )
        lines.append(f"  location: {self.location}")

        if counts["mask_statistics"] is not None:
            mask_rows = [
                (
                    split,
                    statistics["images"],
                    statistics["nonempty"],
                    statistics["empty"],
                    f"{statistics['foreground_pixels']:,}",
                    f"{statistics['foreground_fraction']:.1%}",
                )
                for split, statistics in counts["mask_statistics"].items()
            ]
            total_pixels = sum(
                statistics["total_pixels"]
                for statistics in counts["mask_statistics"].values()
            )
            foreground_pixels = sum(
                statistics["foreground_pixels"]
                for statistics in counts["mask_statistics"].values()
            )
            mask_rows.append(
                (
                    "total",
                    counts["images"],
                    counts["annotated"],
                    counts["empty"],
                    f"{foreground_pixels:,}",
                    f"{foreground_pixels / total_pixels if total_pixels else 0.0:.1%}",
                )
            )
            lines.extend(
                [
                    "",
                    "  Mask statistics",
                    *_format_table(
                        ("split", "images", "non-empty", "empty", "foreground px", "pixel %"),
                        mask_rows,
                        right_aligned={1, 2, 3, 4, 5},
                    ),
                ]
            )

        if counts["split_statistics"] is not None:
            split_rows = [
                (
                    split,
                    statistics["images"],
                    statistics["annotated"],
                    statistics["empty"],
                    statistics["annotations"],
                )
                for split, statistics in counts["split_statistics"].items()
            ]
            split_rows.append(
                (
                    "total",
                    counts["images"],
                    counts["annotated"],
                    counts["empty"],
                    counts["annotations"],
                )
            )
            lines.extend(
                [
                    "",
                    "  Split statistics",
                    *_format_table(
                        ("split", "images", "annotated", "empty", "annotations"),
                        split_rows,
                        right_aligned={1, 2, 3, 4},
                    ),
                ]
            )

        if counts["class_statistics"] is not None:
            class_rows = [
                (
                    class_id,
                    statistics["name"],
                    statistics["annotations"],
                    statistics["images"],
                    f"{statistics['image_fraction']:.1%}",
                )
                for class_id, statistics in counts["class_statistics"].items()
            ]
            lines.extend(
                [
                    "",
                    "  Class statistics (annotation instances and images containing class)",
                    *_format_table(
                        ("id", "name", "annotations", "images", "image %"),
                        class_rows,
                        right_aligned={0, 2, 3, 4},
                    ),
                ]
            )
        else:
            classes = ", ".join(
                f"{class_id}:{name}" for class_id, name in self.classes.items()
            ) or "none"
            lines.append(f"  class names: {classes}")

        if self._plan:
            lines.append("  pending: " + " → ".join(operation.kind for operation in self._plan))
            lines.append("  export required: data_yaml and training_ready are unavailable until export()")
        elif self.data_yaml is not None:
            lines.append(f"  data_yaml: {self.data_yaml}")
        skipped = int(self._validation_audit.get("skipped_count", 0))
        validation_status = str(self._validation_audit.get("status", "unknown"))
        validation = f"  validation: {validation_status} | warnings: {len(self.warnings)}"
        if skipped:
            validation += f" | skipped: {skipped} (see validation_audit)"
        lines.append(validation)
        return "\n".join(lines)

    def _summary_counts(self) -> dict[str, Any]:
        split_counts = {
            split: sum(sample.split == split for sample in self._samples)
            for split in self.splits
        }
        if self._plan and not self._projection_exact:
            return {
                "splits": split_counts,
                "images": "pending export",
                "annotations": "pending export",
                "annotated": None,
                "empty": None,
                "empty_fraction": 0.0,
                "image_sizes": None,
                "split_statistics": None,
                "class_statistics": None,
                "mask_statistics": None,
            }
        if self.format == "semantic_masks":
            mask_statistics = {
                split: {
                    "images": 0,
                    "nonempty": 0,
                    "empty": 0,
                    "foreground_pixels": 0,
                    "total_pixels": 0,
                    "foreground_fraction": 0.0,
                }
                for split in self.splits
            }
            masks = 0
            for sample in self._samples:
                statistics = self._mask_statistics.get(sample.image_path.resolve())
                if statistics is None:
                    continue
                masks += 1
                split = mask_statistics[sample.split]
                split["images"] += 1
                split["foreground_pixels"] += statistics["foreground_pixels"]
                split["total_pixels"] += statistics["total_pixels"]
                if statistics["foreground_pixels"]:
                    split["nonempty"] += 1
                else:
                    split["empty"] += 1
            for statistics in mask_statistics.values():
                total_pixels = statistics["total_pixels"]
                statistics["foreground_fraction"] = (
                    statistics["foreground_pixels"] / total_pixels if total_pixels else 0.0
                )
            images = len(self._samples)
            empty = sum(statistics["empty"] for statistics in mask_statistics.values())
            return {
                "splits": split_counts,
                "images": images,
                "masks": masks,
                "annotations": None,
                "annotated": images - empty,
                "empty": empty,
                "empty_fraction": empty / images if images else 0.0,
                "image_sizes": sorted({(sample.width, sample.height) for sample in self._samples}),
                "split_statistics": None,
                "class_statistics": None,
                "mask_statistics": mask_statistics,
            }
        images = len(self._samples)
        empty = sum(not sample.annotations for sample in self._samples)
        annotations = sum(len(sample.annotations) for sample in self._samples)
        split_statistics = {
            split: {
                "images": 0,
                "annotated": 0,
                "empty": 0,
                "annotations": 0,
            }
            for split in self.splits
        }
        class_statistics = {
            class_id: {
                "name": name,
                "annotations": 0,
                "images": 0,
                "image_fraction": 0.0,
            }
            for class_id, name in self.classes.items()
        }
        for sample in self._samples:
            split = split_statistics[sample.split]
            split["images"] += 1
            split["annotations"] += len(sample.annotations)
            if sample.annotations:
                split["annotated"] += 1
            else:
                split["empty"] += 1
            present_class_ids: set[int] = set()
            for annotation in sample.annotations:
                class_statistics[annotation.class_id]["annotations"] += 1
                present_class_ids.add(annotation.class_id)
            for class_id in present_class_ids:
                class_statistics[class_id]["images"] += 1
        for statistics in class_statistics.values():
            statistics["image_fraction"] = statistics["images"] / images if images else 0.0
        return {
            "splits": split_counts,
            "images": images,
            "annotations": annotations,
            "annotated": images - empty,
            "empty": empty,
            "empty_fraction": empty / images if images else 0.0,
            "image_sizes": sorted({(sample.width, sample.height) for sample in self._samples}),
            "split_statistics": split_statistics,
            "class_statistics": class_statistics,
            "mask_statistics": None,
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
            validation_audit=self._validation_audit,
            validation_audit_visualization=self._validation_audit_visualization,
            provenance=self._provenance,
        )

    def _export_plan(
        self,
        *,
        destination: str | Path | None,
        name: str | None,
        format: str,
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
            "format": format,
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
            if format == "semantic_masks":
                return export_semantic_masks(
                    current,
                    destination=final_destination,
                    name=final_name,
                    splits=splits,
                    visualize=visualize,
                    progress=progress,
                    dry_run=False,
                )
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


def _semantic_mask_manifest(requested: Path) -> dict[str, Any] | None:
    if requested.is_file() and requested.name in {"dataset-fixer.json", DATASET_INFO_NAME}:
        manifest_path = requested
    else:
        manifest_path = dataset_info_path(requested)
        if not manifest_path.is_file():
            legacy = requested / "dataset-fixer.json"
            manifest_path = legacy if legacy.is_file() else manifest_path
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        root = requested.parent if requested.is_file() else requested
        has_semantic_layout = any(
            (root / split / "masks" / "0").is_dir()
            for split in ("train", "val", "test")
        )
        if not requested.is_file() and not has_semantic_layout:
            return None
        raise DatasetValidationError(
            ValidationIssue(
                f"Unreadable dataset manifest: {exc}",
                source=str(manifest_path),
            )
        ) from exc
    return manifest if manifest.get("format") == "semantic_masks" else None


def _class_names(value: Any) -> dict[int, str]:
    if isinstance(value, list):
        return {index: str(name) for index, name in enumerate(value)}
    if isinstance(value, dict):
        try:
            return {int(class_id): str(name) for class_id, name in value.items()}
        except (TypeError, ValueError):
            return {}
    return {}


def _load_semantic_mask_samples(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    progress: bool,
    errors: Literal["raise", "skip"],
) -> tuple[
    list[Sample],
    dict[str, Path],
    dict[str, Path],
    dict[Path, Path],
    dict[Path, dict[str, int]],
    list[str],
    list[ValidationFailureExample],
]:
    declared_splits = manifest.get("splits")
    if not isinstance(declared_splits, list) or not declared_splits:
        raise DatasetValidationError(
            ValidationIssue(
                "Semantic-mask manifest has no splits",
                source=str(dataset_info_path(root)),
                expected="a non-empty splits list",
            )
        )
    declared = {normalize_split(str(split)) for split in declared_splits}
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

    candidates = [
        (split, image_path, image_path.relative_to(image_dirs[split]))
        for split in splits
        for image_path in sorted(image_dirs[split].rglob("*"))
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES
    ]
    samples: list[Sample] = []
    mask_paths: dict[Path, Path] = {}
    mask_statistics: dict[Path, dict[str, int]] = {}
    warnings: list[str] = []
    failure_examples: list[ValidationFailureExample] = []
    issues: list[ValidationIssue] = []
    expected_masks: set[Path] = set()
    used_masks: dict[Path, Path] = {}
    iterator = tqdm(
        candidates,
        desc="Loading semantic-mask dataset",
        unit="pair",
        disable=not progress,
    )
    for split, image_path, relative_path in iterator:
        mask_path = mask_dirs[split] / relative_path.with_suffix(".png")
        resolved_mask = mask_path.resolve()
        expected_masks.add(resolved_mask)
        previous_image = used_masks.get(resolved_mask)
        issue: ValidationIssue | None = None
        width: int | None = None
        height: int | None = None
        foreground_pixels = 0
        total_pixels = 0
        if previous_image is not None:
            issue = ValidationIssue(
                "Multiple images resolve to the same semantic mask",
                source=str(mask_path),
                value=[str(previous_image), str(image_path)],
                expected="one unique relative image stem per mask",
            )
        elif not mask_path.is_file():
            issue = ValidationIssue(
                "Semantic mask is missing for image",
                source=str(image_path),
                expected=str(mask_path),
            )
        else:
            try:
                with Image.open(image_path) as opened_image, Image.open(mask_path) as opened_mask:
                    opened_image.load()
                    opened_mask.load()
                    width, height = opened_image.size
                    if opened_mask.mode != "L":
                        raise ValueError(
                            f"mask mode is {opened_mask.mode!r}; expected single-channel 'L'"
                        )
                    if opened_mask.size != (width, height):
                        raise ValueError(
                            f"mask dimensions {opened_mask.size} do not match image dimensions "
                            f"{(width, height)}"
                        )
                    histogram = opened_mask.histogram()
                    values = {value for value, count in enumerate(histogram) if count}
                    if not values <= {0, 1, 255}:
                        raise ValueError(
                            f"mask values must be 0/1/255, found {sorted(values)[:10]}"
                        )
                    foreground_pixels = histogram[1] + histogram[255]
                    total_pixels = width * height
            except Exception as exc:
                issue = ValidationIssue(
                    f"Unreadable or invalid semantic image/mask pair: {exc}",
                    source=str(image_path),
                    value=str(mask_path),
                )
        if issue is not None:
            if errors == "raise":
                issues.append(issue)
            else:
                warning = f"Skipped invalid semantic-mask pair: {issue.format()}"
                warnings.append(warning)
                failure_examples.append(
                    ValidationFailureExample(
                        warning=warning,
                        summary=str(issue.message),
                        image_path=image_path if image_path.is_file() else None,
                        relative_path=relative_path,
                        split=split,
                        width=width,
                        height=height,
                    )
                )
            continue
        used_masks[resolved_mask] = image_path
        resolved_image = image_path.resolve()
        samples.append(
            Sample(
                image_path=resolved_image,
                relative_path=relative_path,
                split=split,
                width=int(width),
                height=int(height),
            )
        )
        mask_paths[resolved_image] = resolved_mask
        mask_statistics[resolved_image] = {
            "foreground_pixels": foreground_pixels,
            "total_pixels": total_pixels,
        }

    actual_masks = {
        path.resolve()
        for split in splits
        for path in mask_dirs[split].rglob("*.png")
        if path.is_file()
    }
    for orphan in sorted(actual_masks - expected_masks):
        issue = ValidationIssue(
            "Semantic mask has no matching image",
            source=str(orphan),
        )
        if errors == "raise":
            issues.append(issue)
        else:
            warnings.append(f"Ignored orphan semantic mask: {issue.format()}")
    if issues:
        raise DatasetValidationError(issues)
    if not samples:
        raise DatasetValidationError("Semantic-mask dataset contains no valid image/mask pairs")
    return (
        samples,
        image_dirs,
        mask_dirs,
        mask_paths,
        mask_statistics,
        warnings,
        failure_examples,
    )


def _common_parent(paths: Sequence[Path]) -> str:
    common = Path(os.path.commonpath([str(path) for path in paths]))
    return str(common.parent if common in paths or common.is_file() else common)


def _format_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    right_aligned: set[int] | None = None,
) -> list[str]:
    """Render a small dependency-free table for terminal-friendly summaries."""

    aligned = right_aligned or set()
    rendered_rows = [
        [str(value).replace("\n", "\\n") for value in row]
        for row in rows
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rendered_rows))
        for index, header in enumerate(headers)
    ]

    def render(row: Sequence[str]) -> str:
        cells = [
            value.rjust(widths[index]) if index in aligned else value.ljust(widths[index])
            for index, value in enumerate(row)
        ]
        return "    " + "  ".join(cells)

    return [
        render(headers),
        "    " + "  ".join("-" * width for width in widths),
        *(render(row) for row in rendered_rows),
    ]


def _load_provenance(
    root: Path,
    samples: list[Sample],
    *,
    errors: Literal["raise", "skip"],
    warnings: list[str],
) -> dict[str, dict[str, Any]]:
    path = lineage_path(root)
    legacy = root / "provenance.jsonl"
    if not path.is_file() and not legacy.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    issues: list[ValidationIssue] = []
    try:
        if path.is_file():
            for line_number, record in enumerate(read_lineage(path), start=1):
                try:
                    records[str(Path(record["output_image"]))] = record
                except (KeyError, TypeError) as exc:
                    issues.append(
                        ValidationIssue(
                            "Invalid lineage record",
                            source=str(path),
                            line=line_number,
                            value=str(exc),
                        )
                    )
        else:
            path = legacy
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
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
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        issues.append(
            ValidationIssue(
                f"Unreadable provenance file: {exc}",
                source=str(path),
            )
        )
    if issues:
        if errors == "raise":
            raise DatasetValidationError(issues)
        warnings.extend(f"Skipped invalid provenance record: {issue.format()}" for issue in issues)
    missing_count = 0
    missing_examples: list[str] = []
    for sample in samples:
        try:
            expected = str(sample.image_path.resolve().relative_to(root.resolve()))
        except ValueError:
            # Virtual projections still point to immutable source images.
            continue
        if expected not in records:
            missing_count += 1
            if len(missing_examples) < 10:
                missing_examples.append(expected)
    if missing_count:
        issue = ValidationIssue(
            "Derived dataset is missing image provenance records",
            source=str(path),
            value={"count": missing_count, "examples": sorted(missing_examples)},
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
    expected = {
        _label_path_for_image(sample.image_path, sample.relative_path).resolve()
        for sample in samples
    }
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
