from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from .artifacts import (
    DATASET_INFO_SCHEMA,
    LINEAGE_SCHEMA,
    collect_report_payloads,
    dataset_info_path,
    lineage_path,
    prune_report_directory,
    source_dataset_id,
    source_info_path,
    split_image_summary,
    stable_dataset_id,
    write_json,
    write_lineage,
)
from .dataset_comparison import (
    DatasetReportState,
    report_state_from_samples,
)
from .dataset_comparison import compare_dataset_states
from .io import annotation_to_yolo
from .models import Annotation, DatasetMetadata, Sample, Task
from .utils import environment_snapshot, settings_fingerprint, sha256_file, slugify, to_jsonable


# A generated dataset publishes only the canonical report files, so per-
# operation preview images are pruned before publication. Operations check this
# before rendering one: a preview that cannot be published must not cost time.
PUBLISHES_OPERATION_VISUALS = False


class OutputBuilder:
    """Build a derived dataset privately, then atomically publish it."""

    def __init__(
        self,
        *,
        source_root: Path,
        source_name: str,
        destination: Path,
        name: str,
        task: Task,
        metadata: DatasetMetadata,
        operation: str,
        settings: dict[str, Any],
        parent_manifest: dict[str, Any],
        source_report_state: DatasetReportState | None = None,
    ) -> None:
        self.source_root = source_root.resolve()
        self.source_name = source_name
        self.destination = destination.resolve()
        self.name = slugify(name)
        self.task = task
        self.metadata = metadata
        self.operation = operation
        self.settings = to_jsonable(settings)
        self._settings_fingerprint = settings_fingerprint(self.settings)
        self.parent_manifest = parent_manifest
        self.source_report_state = source_report_state
        self.started = time.time()
        self.staging = Path(tempfile.mkdtemp(prefix=f".{self.destination.name}.tmp-", dir=self.destination.parent))
        self.reports_dir = self.staging / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        self.output_samples: list[Sample] = []
        self.validation_details: dict[str, Any] = {}
        self.visuals: list[str] = []
        self.warnings: list[str] = []
        self.visualize_kwargs: dict[str, Any] = {}
        # Set by operations that can leave part of the source behind, so the
        # published report can state source coverage without re-deriving it.
        self.coverage_summary: dict[str, Any] | None = None

    def cleanup(self) -> None:
        if self.staging.exists():
            shutil.rmtree(self.staging)

    def add_copy(
        self,
        sample: Sample,
        *,
        split: str,
        annotations: list[Annotation] | None = None,
        relative_path: Path | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        relative_path = relative_path or sample.relative_path
        output = self.staging / split / "images" / relative_path
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample.image_path, output)
        self._write_label(output, split, annotations if annotations is not None else sample.annotations, sample.width, sample.height)
        output_annotations = annotations if annotations is not None else sample.annotations
        self._record(sample, output, split, output_annotations, provenance)
        self.output_samples.append(
            Sample(output, relative_path, split, sample.width, sample.height, list(output_annotations))
        )

    def add_image(
        self,
        sample: Sample,
        image: Image.Image,
        *,
        split: str,
        relative_path: Path,
        annotations: list[Annotation],
        provenance: dict[str, Any] | None = None,
        jpeg_quality: int = 95,
    ) -> None:
        output = self.staging / split / "images" / relative_path
        output.parent.mkdir(parents=True, exist_ok=True)
        suffix = output.suffix.lower()
        converted = image.convert("RGB") if suffix in {".jpg", ".jpeg"} else image
        if suffix in {".jpg", ".jpeg"}:
            converted.save(output, quality=jpeg_quality)
        else:
            converted.save(output)
        self._write_label(output, split, annotations, image.width, image.height)
        self._record(sample, output, split, annotations, provenance)
        self.output_samples.append(
            Sample(output, relative_path, split, image.width, image.height, list(annotations))
        )

    def _write_label(
        self, output_image: Path, split: str, annotations: list[Annotation], width: int, height: int
    ) -> None:
        relative = output_image.relative_to(self.staging / split / "images")
        label = self.staging / split / "labels" / relative.with_suffix(".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        rows = [annotation_to_yolo(a, self.task, width, height, self.metadata) for a in annotations]
        label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    def _record(
        self,
        sample: Sample,
        output: Path,
        split: str,
        annotations: list[Annotation],
        provenance: dict[str, Any] | None,
    ) -> None:
        immediate_sha = sample.source_sha256 or sha256_file(sample.image_path)
        parent = sample.provenance or {}
        inherited = {
            key: to_jsonable(parent[key])
            for key in (
                "crop",
                "zoom",
                "scale",
                "tile_index",
                "tile_mode",
                "background_source",
                "background_filter",
                "background_filter_result",
                "class_mapping",
                "class_renames",
                "class_move",
                "group_move",
                "split_group",
                "empty_image",
                "augmentation_index",
                "augmentation",
                "augmentation_seed",
                "albumentations_applied",
                "crop_coordinate_space",
                "source_context",
                "transformed_view_size",
                "crop_transform_seed",
                "crop_transform_attempt",
                "crop_pipeline",
                "crop_albumentations_applied",
                "valid_pixel_fraction",
                "validity_result",
                "crop_transform_warnings",
                "lossy_clipping",
                "lossy_policy",
                "focal_annotation_index",
                "lossy_annotation_indices",
                "lossy_non_focal_indices",
                "coverage_relaxation",
                "letterbox_padding_output_px",
                "letterbox_scale_xy",
                "display_label",
            )
            if key in parent
        }
        operation_provenance = to_jsonable(provenance or {})
        record = {
            "output_image": str(output.relative_to(self.staging)),
            "output_sha256": sha256_file(output),
            "parent_dataset": parent.get("dataset_name") or self.parent_manifest.get("name") or self.source_name,
            "parent_location": parent.get("dataset_location") or str(self.source_root),
            "parent_image": str(sample.image_path),
            "parent_split": sample.split,
            "parent_sha256": immediate_sha,
            "original_dataset": parent.get("original_dataset") or parent.get("parent_dataset") or self.parent_manifest.get("name") or self.source_name,
            "original_image": parent.get("original_image") or parent.get("parent_image") or str(sample.image_path),
            "original_sha256": parent.get("original_sha256") or immediate_sha,
            "output_split": split,
            "output_annotation_count": len(annotations),
            "output_has_labels": bool(annotations),
            "source_annotation_ids": [a.source_id for a in annotations if a.source_id is not None],
            "operation": self.operation,
            "transformation_chain": [
                *(parent.get("transformation_chain") or []),
                {
                    "operation": self.operation,
                    "settings_fingerprint": self._settings_fingerprint,
                    "settings": self.settings,
                },
            ],
            **inherited,
            **operation_provenance,
        }
        label_fn = self.visualize_kwargs.get("label_fn")
        if label_fn is not None:
            display_label = label_fn(output)
            if display_label is not None and not isinstance(display_label, str):
                raise TypeError("label_fn must return a string or None")
            record["display_label"] = display_label
        self.records.append(record)

    def write_yaml(self) -> None:
        available = {record["output_split"] for record in self.records}
        data: dict[str, Any] = {
            # Split entries stay relative to this file, so the dataset resolves
            # identically from the staging root and from its final destination.
            "train": "train/images" if "train" in available else None,
            "val": "val/images" if "val" in available else None,
            "test": "test/images" if "test" in available else None,
            "names": {int(k): v for k, v in sorted(self.metadata.names.items())},
            "channels": self.metadata.channels,
        }
        if self.metadata.radii:
            data["radii"] = {int(k): float(v) for k, v in sorted(self.metadata.radii.items())}
        if self.metadata.kpt_shape:
            data["kpt_shape"] = list(self.metadata.kpt_shape)
        if self.metadata.flip_idx is not None:
            data["flip_idx"] = self.metadata.flip_idx
        if self.metadata.kpt_names:
            data["kpt_names"] = self.metadata.kpt_names
        if self.metadata.kpt_oks_sigmas:
            data["kpt_oks_sigmas"] = self.metadata.kpt_oks_sigmas
        data.update(self.metadata.extra)
        (self.staging / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    def write_reports(self, *, class_mapping: dict[int, int] | None = None) -> dict[str, Any]:
        for record in self.records:
            record.setdefault("class_mapping", class_mapping)
            record.setdefault("warnings", [])
        history = list(self.parent_manifest.get("history") or [])
        operation_record = {
            "operation": self.operation,
            "settings": self.settings,
            "settings_fingerprint": self._settings_fingerprint,
            "class_mapping": class_mapping,
            "duration_seconds": time.time() - self.started,
            "output_images": len(self.records),
            "output_annotations": sum(r["output_annotation_count"] for r in self.records),
            "warnings": self.warnings,
            # Filled in below, once the surviving report files are known, so
            # history never points at a preview that was pruned.
            "visuals": [],
        }
        history.append(operation_record)
        finished = time.time()
        source_fingerprint = settings_fingerprint(
            {
                "location": str(self.source_root),
                "images": sorted((record["parent_image"], record["parent_sha256"]) for record in self.records),
                "parent_manifest": self.parent_manifest,
            }
        )
        environment = environment_snapshot()
        try:
            from . import __version__
        except ImportError:
            __version__ = "unknown"
        source_id = source_dataset_id(
            source_manifest=self.parent_manifest,
            source_name=self.source_name,
            source_path=self.source_root,
            records=self.records,
        )
        dataset_id = stable_dataset_id(
            name=self.name,
            task=self.task.value,
            format_name="yolo",
            classes=self.metadata.names,
            records=self.records,
        )
        for record in self.records:
            record.setdefault("parent_dataset_id", source_id)
            record.setdefault("dataset_id", dataset_id)
        write_lineage(lineage_path(self.staging), self.records)

        audits = dict(self.parent_manifest.get("audits") or {})
        audits.update(collect_report_payloads(self.reports_dir))
        split_validation = self.validation_details.get("split_group_isolation")
        if (
            isinstance(split_validation, dict)
            and split_validation.get("status") == "not_applicable"
        ):
            audits.pop("split_group_audit", None)
        coverage_dir = self.staging / "coverage_summary"
        if coverage_dir.is_dir():
            audits.update(
                {
                    f"coverage.{key}": value
                    for key, value in collect_report_payloads(coverage_dir).items()
                }
            )
        prune_report_directory(self.reports_dir, extra_roots=(coverage_dir,))
        report_visualize_kwargs = dict(self.visualize_kwargs)
        candidate_state = report_state_from_samples(
            root=self.staging,
            name=self.name,
            task=self.task.value,
            format_name="yolo",
            classes=self.metadata.names,
            samples=self.output_samples,
            metadata=self.metadata,
            coverage=self.coverage_summary or audits.get("coverage.source_coverage"),
            details=self.records,
        )
        comparison = compare_dataset_states(
            self.source_report_state,
            candidate_state,
            destination=self.reports_dir / "plots.png",
            visualize_kwargs=report_visualize_kwargs,
        )
        plot = comparison.plot
        self.visuals = ["reports/plots.png"] if plot is not None else []
        operation_record["visuals"] = list(self.visuals)
        load_validation = self.validation_details.get("load_validation")
        if isinstance(load_validation, dict) and load_validation.get("skipped_count", 0):
            load_validation["report"] = (
                "reports/dataset-info.json#audits.load_validation_audit"
            )
            load_validation["visualization"] = None
            if isinstance(audits.get("load_validation_audit"), dict):
                audits["load_validation_audit"]["report"] = load_validation["report"]
                audits["load_validation_audit"]["visualization"] = None
        source_info = {
            "schema": "dataset-fixer-source",
            "schema_version": DATASET_INFO_SCHEMA,
            "id": source_id,
            "path": str(self.source_root),
            "name": self.parent_manifest.get("name") or self.source_name,
            "format": self.parent_manifest.get("format") or (
                "yolo"
                if any((self.source_root / name).is_file() for name in ("data.yaml", "dataset.yaml", "data.yml"))
                else "coco"
            ),
            "task": self.parent_manifest.get("task") or self.task.value,
            "fingerprint": source_fingerprint,
            "classes": self.parent_manifest.get("classes") or self.metadata.names,
            "splits": self.parent_manifest.get("splits") or sorted(
                {str(record.get("parent_split")) for record in self.records}
            ),
            "state": {
                "images": len({str(record.get("parent_image")) for record in self.records}),
                "splits": {
                    split: len(
                        {
                            str(record.get("parent_image"))
                            for record in self.records
                            if record.get("parent_split") == split
                        }
                    )
                    for split in sorted(
                        {str(record.get("parent_split")) for record in self.records}
                    )
                },
            },
        }
        write_json(source_info_path(self.staging), source_info)
        manifest = {
            "schema": "dataset-fixer-dataset-info",
            "schema_version": DATASET_INFO_SCHEMA,
            "dataset_id": dataset_id,
            "name": self.name,
            "location": str(self.destination),
            "task": self.task.value,
            "format": "yolo",
            "splits": sorted({r["output_split"] for r in self.records}),
            "split_summary": split_image_summary(self.records),
            "classes": self.metadata.names,
            "settings": self.settings,
            "settings_fingerprint": self._settings_fingerprint,
            "history": history,
            "parent_manifest_fingerprint": settings_fingerprint(self.parent_manifest) if self.parent_manifest else None,
            "source_dataset": {
                "id": source_id,
                "name": self.parent_manifest.get("name") or self.source_name,
                "path": str(self.source_root),
                "format": source_info["format"],
                "task": self.parent_manifest.get("task") or self.task.value,
                "fingerprint": source_fingerprint,
                "classes": source_info["classes"],
                "splits": source_info["splits"],
                "source_dataset": self.parent_manifest.get("source_dataset"),
            },
            "dataset_fixer": {
                "version": __version__,
                "commit": environment["dataset_fixer_git"]["commit"],
                "dirty": environment["dataset_fixer_git"]["dirty"],
            },
            "environment": environment,
            "class_mapping": class_mapping,
            "warnings": self.warnings,
            "visuals": self.visuals,
            "lineage": {
                "path": "reports/lineage.json.gz",
                "schema_version": LINEAGE_SCHEMA,
                "records": len(self.records),
            },
            "audits": audits,
            "validation": {
                "passed": True,
                "warnings": self.warnings,
                **self.validation_details,
            },
            "training_ready": {"ready": False, "structurally_valid": True, "backend_checked": False},
            "timing": {
                "started_at": datetime.fromtimestamp(self.started, timezone.utc).isoformat(),
                "finished_at": datetime.fromtimestamp(finished, timezone.utc).isoformat(),
                "duration_seconds": finished - self.started,
            },
        }
        write_json(dataset_info_path(self.staging), manifest)
        return manifest

    def publish(
        self,
        *,
        class_mapping: dict[int, int] | None = None,
        progress: bool = True,
        validate: bool = True,
    ) -> dict[str, Any]:
        self.write_yaml()
        manifest = self.write_reports(class_mapping=class_mapping)
        if validate:
            if progress:
                print(
                    "Validating complete staged output before atomic publication "
                    "(streaming, bounded memory)..."
                )
            from .validation import validate_staged_yolo_output

            manifest["training_ready"]["ready"] = validate_staged_yolo_output(
                self.staging,
                self.output_samples,
                self.records,
                self.metadata,
                self.task,
                manifest,
                progress=progress,
            )
            manifest["training_ready"]["backend_checked"] = False
        else:
            manifest["validation"] = {
                "passed": None,
                "deferred_to_final_export": True,
                "warnings": self.warnings,
                **self.validation_details,
            }
        write_json(dataset_info_path(self.staging), manifest)
        os.replace(self.staging, self.destination)
        return manifest

    def result_dataset(self, manifest: dict[str, Any]) -> "Dataset":
        """Build a Dataset from already-known outputs without reopening every image."""

        from .dataset import Dataset

        samples = self.output_samples
        for sample, record in zip(samples, self.records):
            sample.image_path = self.destination / sample.image_path.relative_to(self.staging)
            sample.source_sha256 = record["output_sha256"]
        provenance = {str(record["output_image"]): record for record in self.records}
        self.output_samples = []
        self.records = []
        return Dataset(
            location=self.destination,
            name=self.name,
            task=self.task,
            metadata=self.metadata.copy(),
            samples=samples,
            manifest=manifest,
            data_yaml=self.destination / "data.yaml",
            source_format="yolo",
            warnings=list(self.warnings),
            provenance=provenance,
        )
