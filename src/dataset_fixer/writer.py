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

from .io import annotation_to_yolo
from .models import Annotation, DatasetMetadata, Sample, Task
from .utils import environment_snapshot, settings_fingerprint, sha256_file, slugify, to_jsonable


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
    ) -> None:
        self.source_root = source_root.resolve()
        self.source_name = source_name
        self.destination = destination.resolve()
        self.name = slugify(name)
        self.task = task
        self.metadata = metadata
        self.operation = operation
        self.settings = to_jsonable(settings)
        self.parent_manifest = parent_manifest
        self.started = time.time()
        self.staging = Path(tempfile.mkdtemp(prefix=f".{self.destination.name}.tmp-", dir=self.destination.parent))
        self.reports_dir = self.staging / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        source_reports = self.source_root / "reports"
        if source_reports.is_dir():
            shutil.copytree(source_reports, self.reports_dir, dirs_exist_ok=True)
        source_coverage = self.source_root / "coverage_summary"
        if source_coverage.is_dir():
            shutil.copytree(source_coverage, self.staging / "coverage_summary", dirs_exist_ok=True)
        self.records: list[dict[str, Any]] = []
        self.visuals: list[str] = []
        self.warnings: list[str] = []

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
        self._record(sample, output, split, annotations if annotations is not None else sample.annotations, provenance)

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
            key: parent[key]
            for key in (
                "crop",
                "zoom",
                "scale",
                "tile_index",
                "tile_mode",
                "class_mapping",
                "split_group",
                "empty_image",
            )
            if key in parent
        }
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
            "source_annotation_ids": [a.source_id for a in annotations if a.source_id is not None],
            "operation": self.operation,
            "transformation_chain": [
                *(parent.get("transformation_chain") or []),
                {
                    "operation": self.operation,
                    "settings_fingerprint": settings_fingerprint(self.settings),
                    "settings": self.settings,
                },
            ],
            **inherited,
            **(provenance or {}),
        }
        self.records.append(record)

    def write_yaml(self, *, dataset_root: Path | None = None) -> None:
        available = {record["output_split"] for record in self.records}
        data: dict[str, Any] = {
            # Validate against the private staging root, then rewrite this to
            # the final absolute destination immediately before publication.
            "path": str((dataset_root or self.staging).resolve()),
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
        provenance_path = self.staging / "provenance.jsonl"
        with provenance_path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(to_jsonable(record), sort_keys=True) + "\n")
        history = list(self.parent_manifest.get("history") or [])
        operation_record = {
            "operation": self.operation,
            "settings": self.settings,
            "settings_fingerprint": settings_fingerprint(self.settings),
            "class_mapping": class_mapping,
            "duration_seconds": time.time() - self.started,
            "output_images": len(self.records),
            "output_annotations": sum(r["output_annotation_count"] for r in self.records),
            "warnings": self.warnings,
            "visuals": self.visuals,
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
        manifest = {
            "schema_version": 1,
            "name": self.name,
            "location": str(self.destination),
            "task": self.task.value,
            "format": "yolo",
            "splits": sorted({r["output_split"] for r in self.records}),
            "classes": self.metadata.names,
            "settings": self.settings,
            "settings_fingerprint": settings_fingerprint(self.settings),
            "history": history,
            "parent_manifest_fingerprint": settings_fingerprint(self.parent_manifest) if self.parent_manifest else None,
            "source_dataset": {
                "name": self.parent_manifest.get("name") or self.source_name,
                "location": str(self.source_root),
                "fingerprint": source_fingerprint,
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
            "provenance": "provenance.jsonl",
            "validation": {"passed": True, "warnings": self.warnings},
            "training_ready": {"ready": False, "structurally_valid": True, "backend_checked": False},
            "timing": {
                "started_at": datetime.fromtimestamp(self.started, timezone.utc).isoformat(),
                "finished_at": datetime.fromtimestamp(finished, timezone.utc).isoformat(),
                "duration_seconds": finished - self.started,
            },
        }
        (self.staging / "dataset-fixer.json").write_text(
            json.dumps(to_jsonable(manifest), indent=2, sort_keys=True), encoding="utf-8"
        )
        return manifest

    def publish(
        self,
        *,
        class_mapping: dict[int, int] | None = None,
        progress: bool = True,
    ) -> dict[str, Any]:
        self.write_yaml()
        manifest = self.write_reports(class_mapping=class_mapping)
        # Load and fully validate the private tree before the atomic rename.
        from .dataset import Dataset

        if progress:
            print("Validating complete staged output before atomic publication...")
        candidate = Dataset.open(self.staging, progress=progress)
        manifest["training_ready"]["ready"] = candidate.training_ready
        (self.staging / "dataset-fixer.json").write_text(
            json.dumps(to_jsonable(manifest), indent=2, sort_keys=True), encoding="utf-8"
        )
        self.write_yaml(dataset_root=self.destination)
        os.replace(self.staging, self.destination)
        return manifest
