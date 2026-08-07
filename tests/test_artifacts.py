from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

import pytest
import yaml
from PIL import Image, ImageDraw

from dataset_fixer import Dataset, DatasetTrace, DatasetValidationError
from dataset_fixer.artifacts import (
    DATASET_INFO_SCHEMA,
    combine_report_plots,
    split_image_summary,
)
from dataset_fixer.validation_audit import stage_load_validation_audit
from conftest import make_yolo_dataset


def test_compact_dataset_artifacts_and_exact_trace(tmp_path: Path) -> None:
    source = make_yolo_dataset(
        tmp_path / "source",
        task="detect",
        names=["fruit"],
        train_rows=["0 0.5 0.5 0.25 0.25"],
        val_rows=["0 0.5 0.5 0.25 0.25"],
        size=(96, 80),
    )
    exported = (
        Dataset.open(source, task="detect", progress=False)
        .tile(tile_size=64, overlap=0.25, visualize=True)
        .export(destination=tmp_path / "tiles", visualize=True, progress=False)
    )

    reports = exported.location / "reports"
    assert (reports / "dataset-info.json").is_file()
    assert (reports / "source.json").is_file()
    assert (reports / "lineage.json.gz").is_file()
    assert (reports / "plots.png").is_file()
    assert not (reports / "source-operation.png").exists()
    assert not list(exported.location.rglob("*.jsonl"))
    assert not list(exported.location.rglob("*.csv"))
    assert not (exported.location / "dataset-fixer.json").exists()
    assert not (exported.location / "provenance.jsonl").exists()

    info = json.loads((reports / "dataset-info.json").read_text(encoding="utf-8"))
    source_info = json.loads((reports / "source.json").read_text(encoding="utf-8"))
    assert info["dataset_id"]
    assert set(info["split_summary"]) == {"train", "val"}
    for details in info["split_summary"].values():
        assert details["total_images"] == (
            details["labeled_images"] + details["background_images"]
        )
    assert source_info["id"]
    assert source_info["path"]
    with gzip.open(reports / "lineage.json.gz", "rt", encoding="utf-8") as handle:
        lineage = json.load(handle)
    assert len(lineage["records"]) == len(exported._samples)

    trace = exported.trace()
    assert isinstance(trace, DatasetTrace)
    assert trace.datasets[0].dataset_id == info["dataset_id"]
    assert trace.datasets[-1].path == source.resolve()
    assert trace.datasets[-1].present
    assert trace.tiles
    assert trace.tiles[0].resolved_original_image is not None
    assert trace.for_sample(trace.samples[0].output_image) == trace.samples[0]
    assert "tile(s)" in trace.summary()


def test_trace_resolves_moved_source_with_path_rewrite(tmp_path: Path) -> None:
    source = make_yolo_dataset(
        tmp_path / "old" / "source",
        task="detect",
        names=["fruit"],
        train_rows=["0 0.5 0.5 0.25 0.25"],
        val_rows=["0 0.5 0.5 0.25 0.25"],
    )
    exported = Dataset.open(source, task="detect", progress=False).export(
        destination=tmp_path / "derived",
        visualize=False,
        progress=False,
    )
    moved = tmp_path / "new" / "source"
    moved.parent.mkdir()
    shutil.move(str(source), moved)

    trace = exported.trace(path_rewrites={source: moved})
    assert trace.datasets[-1].path == moved.resolve()
    assert trace.datasets[-1].present
    assert all(
        sample.resolved_original_image is not None
        for sample in trace.samples
        if sample.original_image
    )


def test_combined_plots_are_cropped_and_stacked_at_readable_width(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    for index, color in enumerate(("red", "blue"), start=1):
        image = Image.new("RGB", (1200, 800), "white")
        ImageDraw.Draw(image).rectangle((450, 250, 750, 550), fill=color)
        image.save(reports / f"panel_{index}.png")

    output = reports / "plots.png"
    assert combine_report_plots((reports,), output) == output

    with Image.open(output) as combined:
        assert combined.width == 1600
        assert combined.height > combined.width
        # Cropping removes the source whitespace before scaling, so both panels
        # occupy most of the report width instead of becoming thumbnails.
        red_bounds = _color_bounds(combined.convert("RGB"), (255, 0, 0))
        blue_bounds = _color_bounds(combined.convert("RGB"), (0, 0, 255))
        assert red_bounds is not None and red_bounds[2] - red_bounds[0] > 1200
        assert blue_bounds is not None and blue_bounds[2] - blue_bounds[0] > 1200
        assert red_bounds[1] < blue_bounds[1]
    assert not (reports / "panel_1.png").exists()
    assert not (reports / "panel_2.png").exists()


def test_combined_plots_ignore_existing_aggregate_reports(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    Image.new("RGB", (100, 100), "red").save(reports / "direct.png")
    Image.new("RGB", (100, 100), "blue").save(reports / "source-operation.png")
    Image.new("RGB", (100, 100), "green").save(reports / "comparison.png")

    output = reports / "plots.png"
    combine_report_plots((reports,), output)

    with Image.open(output) as combined:
        colors = set(combined.convert("RGB").getdata())
    assert (255, 0, 0) in colors
    assert (0, 0, 255) not in colors
    assert (0, 128, 0) not in colors


def test_load_audit_does_not_embed_a_reopened_aggregate_plot(tmp_path: Path) -> None:
    aggregate = tmp_path / "source" / "reports" / "plots.png"
    aggregate.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), "red").save(aggregate)
    reports = tmp_path / "derived" / "reports"

    detail, staged = stage_load_validation_audit(
        {
            "status": "passed_with_skips",
            "skipped_count": 1,
            "counts_by_category": {"invalid": 1},
        },
        aggregate,
        reports,
    )

    assert staged is None
    assert detail["visualization"] is None
    assert not (reports / "load_validation_examples.png").exists()


def test_split_image_summary_counts_labels_and_background_per_split() -> None:
    assert split_image_summary(
        [
            {"output_split": "train", "output_annotation_count": 2},
            {"output_split": "train", "output_annotation_count": 0},
            {
                "output_split": "val",
                "output_annotation_count": 3,
                "output_has_labels": False,
            },
        ]
    ) == {
        "train": {
            "total_images": 2,
            "labeled_images": 1,
            "background_images": 1,
        },
        "val": {
            "total_images": 1,
            "labeled_images": 0,
            "background_images": 1,
        },
    }


def test_dataset_update_in_place_and_to_string_destination(tmp_path: Path) -> None:
    source = make_yolo_dataset(
        tmp_path / "update-source",
        task="detect",
        names=["fruit"],
        train_rows=["0 0.5 0.5 0.25 0.25", ""],
        val_rows=["0 0.5 0.5 0.25 0.25"],
    )
    exported = Dataset.open(source, task="detect", progress=False).export(
        destination=tmp_path / "outdated",
        visualize=False,
        progress=False,
    )
    info_path = exported.location / "reports" / "dataset-info.json"
    old_info = json.loads(info_path.read_text(encoding="utf-8"))
    dataset_id = old_info["dataset_id"]
    old_info["schema_version"] = 2
    old_info.pop("split_summary")
    info_path.write_text(json.dumps(old_info), encoding="utf-8")
    (exported.location / "reports" / "legacy-audit.json").write_text(
        json.dumps({"status": "passed"}),
        encoding="utf-8",
    )
    (exported.location / "dataset-fixer.json").write_text("{}", encoding="utf-8")
    (exported.location / "provenance.jsonl").write_text("{}\n", encoding="utf-8")
    payload_bytes = {
        path.relative_to(exported.location): path.read_bytes()
        for split in ("train", "val")
        for directory in ("images", "labels")
        for path in (exported.location / split / directory).rglob("*")
        if path.is_file()
    }

    updated = Dataset.open(exported.location, task="detect", progress=False).update()

    assert updated.location == exported.location
    assert updated.manifest["schema_version"] == DATASET_INFO_SCHEMA
    assert updated.manifest["dataset_id"] == dataset_id
    assert updated.manifest["split_summary"] == {
        "train": {
            "total_images": 2,
            "labeled_images": 1,
            "background_images": 1,
        },
        "val": {
            "total_images": 1,
            "labeled_images": 1,
            "background_images": 0,
        },
    }
    assert updated.manifest["audits"]["legacy-audit"] == {"status": "passed"}
    assert not (updated.location / "reports" / "legacy-audit.json").exists()
    assert not (updated.location / "dataset-fixer.json").exists()
    assert not (updated.location / "provenance.jsonl").exists()
    assert all((updated.location / path).read_bytes() == value for path, value in payload_bytes.items())

    destination = tmp_path / "updated-copy"
    copied = updated.update(dest=str(destination))

    assert copied.location == destination.resolve()
    assert copied.manifest["dataset_id"] == dataset_id
    assert copied.manifest["location"] == str(destination.resolve())
    assert yaml.safe_load(copied.data_yaml.read_text(encoding="utf-8"))["path"] == str(
        destination.resolve()
    )
    assert updated.location.is_dir()
    assert all((copied.location / path).read_bytes() == value for path, value in payload_bytes.items())


def test_dataset_update_rejects_raw_and_virtual_datasets(tmp_path: Path) -> None:
    source = make_yolo_dataset(tmp_path / "raw", task="detect")
    dataset = Dataset.open(source, task="detect", progress=False)
    with pytest.raises(DatasetValidationError, match="requires a dataset-fixer manifest"):
        dataset.update()
    virtual = dataset.tile(tile_size=80, visualize=False, progress=False)
    with pytest.raises(DatasetValidationError, match="materialized dataset"):
        virtual.update()


def _color_bounds(image: Image.Image, color: tuple[int, int, int]) -> tuple[int, int, int, int] | None:
    mask = Image.new("1", image.size)
    mask.putdata([pixel == color for pixel in image.getdata()])
    return mask.getbbox()
