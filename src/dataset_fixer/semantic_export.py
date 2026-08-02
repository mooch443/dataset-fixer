from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from .errors import DatasetValidationError, ValidationIssue
from .models import SemanticMaskExport, Task
from .utils import (
    ensure_safe_destination,
    environment_snapshot,
    normalize_split,
    settings_fingerprint,
    sha256_file,
    slugify,
    to_jsonable,
)

if TYPE_CHECKING:
    from .dataset import Dataset
    from .models import Sample


def export_semantic_masks(
    dataset: "Dataset",
    *,
    destination: str | Path | None,
    name: str | None,
    splits: Iterable[str] | None,
    visualize: bool,
    progress: bool,
    dry_run: bool,
) -> SemanticMaskExport | "Dataset":
    """Publish polygon annotations as paired binary foreground masks."""

    if dataset.task is not Task.SEGMENT:
        raise DatasetValidationError(
            ValidationIssue(
                "Semantic-mask export requires a segmentation dataset",
                value=dataset.task.value,
                expected="task='segment'",
            )
        )
    selected = {normalize_split(split) for split in splits} if splits else set(dataset.splits)
    missing_splits = selected - set(dataset.splits)
    if missing_splits:
        raise ValueError(
            f"Unknown semantic-mask splits {sorted(missing_splits)}; available splits are {dataset.splits}"
        )
    samples = [sample for sample in dataset._samples if sample.split in selected]
    if not samples:
        raise DatasetValidationError("No images selected for semantic-mask export")
    missing_polygons = [
        annotation.source_id
        for sample in samples
        for annotation in sample.annotations
        if not annotation.polygon
    ]
    if missing_polygons:
        raise DatasetValidationError(
            ValidationIssue(
                "Semantic-mask export requires polygon geometry for every annotation",
                value=missing_polygons[:10],
                expected="one polygon per selected segmentation annotation",
                suggestion="convert or remove RLE/multipart annotations before exporting semantic masks",
            )
        )
    _assert_unique_mask_paths(samples)

    settings = {
        "format": "semantic_masks",
        "splits": sorted(selected),
        "mask_encoding": {"background": 0, "foreground": 255},
        "class_handling": "foreground_union",
        "layout": "<split>/images and <split>/masks/0",
        "visualize": visualize,
    }
    fingerprint = settings_fingerprint(settings)
    final_name = slugify(name or f"{dataset.name}__semantic-masks__{fingerprint}")
    final_destination = (
        Path(destination).expanduser().resolve()
        if destination is not None
        else (dataset.location.parent / final_name).resolve()
    )
    ensure_safe_destination(dataset.location, final_destination)
    print(f"\nsemantic-mask export: {final_name}")
    print(f"Destination: {final_destination}")
    print(f"Images: {len(samples)} | splits: {sorted(selected)}")
    if dry_run:
        print("Dry run complete; no semantic-mask dataset was published.")
        return dataset

    final_destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{final_destination.name}.tmp-",
            dir=final_destination.parent,
        )
    )
    records: list[dict[str, Any]] = []
    warnings = list(dataset.warnings)
    try:
        reports = staging / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        source_reports = dataset.location / "reports"
        if source_reports.is_dir():
            shutil.copytree(source_reports, reports, dirs_exist_ok=True)
        source_coverage = dataset.location / "coverage_summary"
        if source_coverage.is_dir():
            shutil.copytree(
                source_coverage,
                staging / "coverage_summary",
                dirs_exist_ok=True,
            )

        iterator = tqdm(samples, desc="Exporting semantic masks", unit="image", disable=not progress)
        first_pair: tuple[Path, Path] | None = None
        for sample in iterator:
            image_output = staging / sample.split / "images" / sample.relative_path
            mask_relative = sample.relative_path.with_suffix(".png")
            mask_output = staging / sample.split / "masks" / "0" / mask_relative
            image_output.parent.mkdir(parents=True, exist_ok=True)
            mask_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sample.image_path, image_output)

            mask = Image.new("L", (sample.width, sample.height), 0)
            draw = ImageDraw.Draw(mask)
            for annotation in sample.annotations:
                assert annotation.polygon is not None
                draw.polygon(annotation.polygon, fill=255)
            mask.save(mask_output, format="PNG", optimize=False)
            if first_pair is None:
                first_pair = image_output, mask_output
            records.append(
                _provenance_record(
                    dataset,
                    sample,
                    image_output,
                    mask_output,
                    staging,
                    settings,
                )
            )

        _validate_semantic_tree(staging, records)
        visuals = [
            str(path)
            for path in dataset._manifest.get("visuals") or []
            if (staging / str(path)).is_file()
        ]
        if visualize and first_pair is not None:
            preview = reports / "semantic_mask_preview.png"
            _write_preview(*first_pair, preview)
            visuals.append(str(preview.relative_to(staging)))
        provenance_path = staging / "provenance.jsonl"
        with provenance_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(to_jsonable(record), sort_keys=True) + "\n")

        environment = environment_snapshot()
        try:
            from . import __version__
        except ImportError:
            __version__ = "unknown"
        history = [
            *(dataset._manifest.get("history") or []),
            {
                "operation": "export-semantic-masks",
                "settings": settings,
                "settings_fingerprint": fingerprint,
                "output_images": len(records),
                "output_masks": len(records),
                "duration_seconds": time.time() - started,
                "warnings": warnings,
                "visuals": visuals,
            },
        ]
        manifest = {
            "schema_version": 1,
            "name": final_name,
            "location": str(final_destination),
            "task": Task.SEGMENT.value,
            "format": "semantic_masks",
            "splits": sorted(selected),
            "classes": dataset.classes,
            "original_classes": dataset.classes,
            "mask_encoding": settings["mask_encoding"],
            "class_handling": "foreground_union",
            "layout": {
                "images": "<split>/images/<relative image path>",
                "masks": "<split>/masks/0/<matching relative stem>.png",
            },
            "settings": settings,
            "settings_fingerprint": fingerprint,
            "history": history,
            "source_dataset": {
                "name": dataset.name,
                "location": str(dataset.location),
            },
            "dataset_fixer": {
                "version": __version__,
                "commit": environment["dataset_fixer_git"]["commit"],
                "dirty": environment["dataset_fixer_git"]["dirty"],
            },
            "environment": environment,
            "warnings": warnings,
            "visuals": visuals,
            "provenance": "provenance.jsonl",
            "validation": {
                "passed": True,
                "images": len(records),
                "masks": len(records),
                "allowed_mask_values": [0, 255],
            },
        }
        (staging / "dataset-fixer.json").write_text(
            json.dumps(to_jsonable(manifest), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(staging, final_destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    split_values = tuple(split for split in ("train", "val", "test") if split in selected)
    result = SemanticMaskExport(
        name=final_name,
        location=final_destination,
        manifest=manifest,
        manifest_path=final_destination / "dataset-fixer.json",
        splits=split_values,
        image_dirs={split: final_destination / split / "images" for split in split_values},
        mask_dirs={split: final_destination / split / "masks" / "0" for split in split_values},
    )
    print(f"\nCreated {result.name}")
    print(f"Location: {result.location}")
    print(f"Images/masks: {len(records)}/{len(records)}")
    return result


def _assert_unique_mask_paths(samples: list["Sample"]) -> None:
    seen: dict[tuple[str, str], Path] = {}
    collisions: list[dict[str, str]] = []
    for sample in samples:
        mask_path = sample.relative_path.with_suffix(".png").as_posix()
        key = sample.split, mask_path.casefold()
        previous = seen.get(key)
        if previous is not None:
            collisions.append(
                {
                    "mask": f"{sample.split}/masks/0/{mask_path}",
                    "first_image": str(previous),
                    "second_image": str(sample.image_path),
                }
            )
        else:
            seen[key] = sample.image_path
    if collisions:
        raise DatasetValidationError(
            ValidationIssue(
                "Multiple images resolve to the same semantic-mask path",
                value=collisions[:10],
                expected="unique relative image stems within each split",
            )
        )


def _provenance_record(
    dataset: "Dataset",
    sample: "Sample",
    image_output: Path,
    mask_output: Path,
    staging: Path,
    settings: dict[str, Any],
) -> dict[str, Any]:
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
            "background_source",
            "source_annotation_indices",
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
        )
        if key in parent
    }
    return {
        "output_image": str(image_output.relative_to(staging)),
        "output_mask": str(mask_output.relative_to(staging)),
        "output_image_sha256": sha256_file(image_output),
        "output_mask_sha256": sha256_file(mask_output),
        "output_split": sample.split,
        "output_annotation_count": len(sample.annotations),
        "source_annotation_ids": [a.source_id for a in sample.annotations if a.source_id is not None],
        "parent_dataset": parent.get("dataset_name") or dataset.name,
        "parent_location": parent.get("dataset_location") or str(dataset.location),
        "parent_image": str(sample.image_path),
        "parent_sha256": immediate_sha,
        "original_dataset": parent.get("original_dataset") or parent.get("parent_dataset") or dataset.name,
        "original_image": parent.get("original_image") or parent.get("parent_image") or str(sample.image_path),
        "original_sha256": parent.get("original_sha256") or immediate_sha,
        "operation": "export-semantic-masks",
        "format": "semantic_masks",
        "mask_encoding": {"background": 0, "foreground": 255},
        "original_classes": dataset.classes,
        **inherited,
        "transformation_chain": [
            *(parent.get("transformation_chain") or []),
            {
                "operation": "export-semantic-masks",
                "settings_fingerprint": settings_fingerprint(settings),
                "settings": settings,
            },
        ],
    }


def _validate_semantic_tree(staging: Path, records: list[dict[str, Any]]) -> None:
    seen_images: set[str] = set()
    seen_masks: set[str] = set()
    for record in records:
        image_path = staging / record["output_image"]
        mask_path = staging / record["output_mask"]
        if record["output_image"] in seen_images:
            raise DatasetValidationError(f"Duplicate output image: {record['output_image']}")
        seen_images.add(record["output_image"])
        if record["output_mask"] in seen_masks:
            raise DatasetValidationError(f"Duplicate output mask: {record['output_mask']}")
        seen_masks.add(record["output_mask"])
        if not image_path.is_file() or not mask_path.is_file():
            raise DatasetValidationError(f"Missing semantic image/mask pair for {image_path}")
        if sha256_file(image_path) != record["parent_sha256"]:
            raise DatasetValidationError(
                f"Semantic export changed source image bytes: {image_path}"
            )
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                raise DatasetValidationError(
                    f"Semantic mask dimensions {mask.size} do not match image dimensions {image.size}: {mask_path}"
                )
            if mask.mode != "L":
                raise DatasetValidationError(f"Semantic mask must be single-channel L mode: {mask_path}")
            values = set(mask.getdata())
            if not values <= {0, 255}:
                raise DatasetValidationError(
                    f"Semantic mask contains values outside 0/255: {mask_path}: {sorted(values)[:10]}"
                )
    actual_masks = {
        str(path.relative_to(staging))
        for path in staging.glob("*/masks/0/**/*.png")
        if path.is_file()
    }
    if actual_masks != seen_masks:
        raise DatasetValidationError(
            ValidationIssue(
                "Semantic-mask output contains orphan or missing masks",
                value={
                    "unexpected": sorted(actual_masks - seen_masks),
                    "missing": sorted(seen_masks - actual_masks),
                },
            )
        )
    actual_images = {
        str(path.relative_to(staging))
        for split_root in staging.iterdir()
        if split_root.is_dir() and (split_root / "images").is_dir()
        for path in (split_root / "images").rglob("*")
        if path.is_file()
    }
    if actual_images != seen_images:
        raise DatasetValidationError(
            ValidationIssue(
                "Semantic-mask output contains orphan or missing images",
                value={
                    "unexpected": sorted(actual_images - seen_images),
                    "missing": sorted(seen_images - actual_images),
                },
            )
        )


def _write_preview(image_path: Path, mask_path: Path, output: Path) -> None:
    with Image.open(image_path) as opened_image, Image.open(mask_path) as opened_mask:
        image = opened_image.convert("RGB")
        mask = opened_mask.convert("L")
    red = Image.new("RGB", image.size, (255, 0, 0))
    overlay = Image.composite(red, image, mask.point(lambda value: 120 if value else 0))
    preview = Image.new("RGB", (image.width * 2, image.height))
    preview.paste(image, (0, 0))
    preview.paste(overlay, (image.width, 0))
    preview.save(output)
