from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import DatasetValidationError, ValidationIssue
from ..models import Annotation, Sample
from ..utils import normalize_split, sha256_file, to_jsonable
from .types import Cohort, CohortRecord, ModelSpec
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from ..dataset import Dataset


def annotation_dict(annotation: Annotation) -> dict[str, Any]:
    return {
        "class_id": annotation.class_id,
        "bbox": annotation.bbox,
        "polygon": annotation.polygon,
        "rle": annotation.rle,
        "keypoints": annotation.keypoints,
        "point": annotation.point,
        "radius": annotation.radius,
        "source_id": annotation.source_id,
    }


def freeze_cohort(dataset: "Dataset", split: str, *, progress: bool = False) -> Cohort:
    split = normalize_split(split)
    samples = sorted(
        (sample for sample in dataset._samples if sample.split == split),
        key=lambda sample: sample.relative_path.as_posix(),
    )
    if not samples:
        raise DatasetValidationError(
            ValidationIssue("Comparison split is missing or empty", value=split, expected="a non-empty split")
        )
    records: list[CohortRecord] = []
    for sample in tqdm(
        samples,
        desc="Freezing evaluation cohort",
        unit="image",
        disable=not progress,
    ):
        annotations = tuple(to_jsonable(annotation_dict(annotation)) for annotation in sample.annotations)
        encoded = json.dumps(annotations, sort_keys=True, separators=(",", ":")).encode()
        image_hash = sample.source_sha256 or sha256_file(sample.image_path)
        provenance = sample.provenance or {}
        original_hash = provenance.get("original_sha256") or image_hash
        # Combine content with the source-relative logical image name. Content
        # alone would incorrectly cluster distinct constant/blank frames; an
        # absolute path alone would miss the same source copied to a new root.
        original_path = Path(str(provenance.get("original_image") or sample.image_path))
        parts = original_path.parts
        logical_path = sample.relative_path.as_posix()
        if "images" in parts:
            index = len(parts) - 1 - list(reversed(parts)).index("images")
            tail = list(parts[index + 1 :])
            if tail and tail[0].lower() in {"train", "val", "valid", "validation", "test"}:
                tail = tail[1:]
            if tail:
                logical_path = Path(*tail).as_posix()
        original_id = hashlib.sha256(
            f"ultimate-original\0{logical_path}\0{original_hash}".encode()
        ).hexdigest()
        relative = sample.relative_path.as_posix()
        image_id = hashlib.sha256(f"{split}\0{relative}\0{image_hash}".encode()).hexdigest()
        records.append(
            CohortRecord(
                image_id=image_id,
                image_path=sample.image_path.resolve(),
                relative_path=relative,
                split=split,
                width=sample.width,
                height=sample.height,
                image_sha256=image_hash,
                annotation_sha256=hashlib.sha256(encoded).hexdigest(),
                original_id=original_id,
                annotations=annotations,
                provenance=to_jsonable(provenance),
            )
        )
    metadata = {
        "channels": dataset._metadata.channels,
        "radii": dataset._metadata.radii,
        "kpt_shape": dataset._metadata.kpt_shape,
        "flip_idx": dataset._metadata.flip_idx,
        "kpt_names": dataset._metadata.kpt_names,
        "kpt_oks_sigmas": dataset._metadata.kpt_oks_sigmas,
    }
    payload = {
        "split": split,
        "task": dataset.task.value,
        "classes": dataset.classes,
        "metadata": metadata,
        "records": [
            {
                "image_id": record.image_id,
                "relative_path": record.relative_path,
                "width": record.width,
                "height": record.height,
                "image_sha256": record.image_sha256,
                "annotation_sha256": record.annotation_sha256,
                "original_id": record.original_id,
                "physical_group": record.provenance.get("physical_group") or record.original_id,
            }
            for record in records
        ],
    }
    fingerprint = hashlib.sha256(
        json.dumps(to_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return Cohort(split, fingerprint, tuple(records), dataset.task.value, dataset.classes, metadata)


def cohort_record_json(record: CohortRecord) -> dict[str, Any]:
    return {
        "image_id": record.image_id,
        "image_path": str(record.image_path),
        "relative_path": record.relative_path,
        "split": record.split,
        "width": record.width,
        "height": record.height,
        "image_sha256": record.image_sha256,
        "annotation_sha256": record.annotation_sha256,
        "original_id": record.original_id,
        "annotations": record.annotations,
        "provenance": record.provenance,
    }


def write_cohort(cohort: Cohort, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in cohort.records:
            handle.write(json.dumps(to_jsonable(cohort_record_json(record)), sort_keys=True) + "\n")


def check_training_provenance(
    specs: list[ModelSpec], cohort: Cohort, policy: str
) -> tuple[bool, bool, dict[str, Any], list[str]]:
    from ..dataset import Dataset

    if policy not in {"required", "warn", "ignore"}:
        raise ValueError("training_provenance must be 'required', 'warn', or 'ignore'")
    if policy == "ignore":
        return False, False, {spec.name: {"status": "ignored"} for spec in specs}, [
            "Training/evaluation overlap analysis was explicitly disabled."
        ]
    eval_originals = {record.original_id for record in cohort.records}
    complete = True
    overlap_detected = False
    report: dict[str, Any] = {}
    limitations: list[str] = []
    issues: list[ValidationIssue] = []
    for spec in specs:
        if spec.training_dataset is None:
            complete = False
            report[spec.name] = {"status": "unverified", "overlap": []}
            message = f"{spec.name}: training dataset is unknown"
            if policy == "required":
                issues.append(
                    ValidationIssue(
                        "Model training provenance is required",
                        source=str(spec.path),
                        suggestion="add training_dataset to this model specification",
                    )
                )
            else:
                limitations.append(message)
            continue
        training = Dataset.open(spec.training_dataset, progress=False)
        training_cohort = freeze_cohort(training, "train")
        overlaps = sorted(eval_originals & {record.original_id for record in training_cohort.records})
        overlap_detected |= bool(overlaps)
        report[spec.name] = {
            "status": "overlap" if overlaps else "verified",
            "training_dataset": str(training.location),
            "training_fingerprint": training_cohort.fingerprint,
            "overlap_count": len(overlaps),
            "overlap_original_ids": overlaps,
        }
        if overlaps:
            limitations.append(
                f"{spec.name}: {len(overlaps)} evaluation ultimate-original(s) also occur in training."
            )
    if issues:
        raise DatasetValidationError(issues)
    return overlap_detected, complete, report, limitations
