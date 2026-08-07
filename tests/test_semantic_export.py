from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from dataset_fixer import Dataset, DatasetValidationError
from conftest import make_image, make_yolo_dataset


def test_semantic_mask_export_writes_foreground_union_and_empty_masks(tmp_path: Path) -> None:
    source = make_yolo_dataset(
        tmp_path / "segments",
        task="segment",
        names=["school", "reef"],
        train_rows=[
            "0 0.1 0.1 0.6 0.1 0.6 0.6 0.1 0.6\n"
            "1 0.4 0.4 0.9 0.4 0.9 0.9 0.4 0.9",
            "",
        ],
        val_rows=["0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"],
        size=(100, 80),
    )
    coverage_summary = source / "coverage_summary"
    coverage_summary.mkdir()
    (coverage_summary / "source_pixel_coverage.csv").write_text(
        "split,source_pixel_coverage_percent\ntrain,50.0\n",
        encoding="utf-8",
    )
    Image.new("RGB", (40, 20), (255, 255, 255)).save(
        coverage_summary / "source_pixel_coverage.jpg"
    )
    exported = Dataset.open(source, task="segment", progress=False).export(
        destination=tmp_path / "semantic",
        format="semantic_masks",
        visualize=True,
        progress=False,
    )

    assert isinstance(exported, Dataset)
    assert exported.format == "semantic_masks"
    assert exported.manifest["format"] == "semantic_masks"
    assert exported.splits == ("train", "val")
    assert not (exported.location / "data.yaml").exists()
    assert not list(exported.location.rglob("labels"))
    train_masks = sorted(exported.mask_dirs["train"].rglob("*.png"))
    assert len(train_masks) == 2
    with Image.open(train_masks[0]) as mask:
        assert mask.mode == "L"
        assert mask.size == (100, 80)
        assert set(mask.getdata()) == {0, 255}
        assert mask.getpixel((50, 40)) == 255  # overlap is still foreground
    with Image.open(train_masks[1]) as mask:
        assert set(mask.getdata()) == {0}
    manifest = json.loads(exported.manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "semantic_masks"
    assert manifest["class_handling"] == "foreground_union"
    assert manifest["validation"]["allowed_mask_values"] == [0, 255]
    assert (exported.location / "reports" / "semantic_mask_preview.png").is_file()
    assert (
        exported.location / "coverage_summary" / "source_pixel_coverage.csv"
    ).is_file()
    assert (
        exported.location / "coverage_summary" / "source_pixel_coverage.jpg"
    ).is_file()
    records = [json.loads(line) for line in (exported.location / "provenance.jsonl").read_text().splitlines()]
    assert all(record["output_mask_sha256"] for record in records)

    reopened = Dataset.open(exported.location, task="segment", progress=False)
    assert reopened.name == exported.name
    assert reopened.splits == exported.splits
    assert reopened.image_dirs == exported.image_dirs
    assert reopened.mask_dirs == exported.mask_dirs
    assert reopened.training_ready
    summary = str(reopened)
    assert "[segment; materialized; semantic_masks]" in summary
    assert "images: 3 | masks: 3 | non-empty: 2 | empty: 1 (33.3%)" in summary
    assert "Mask statistics" in summary
    assert "class handling: foreground union" in summary
    from matplotlib import pyplot as plt

    figure = reopened.visualize(split="train", n=1, show=False)
    assert figure.axes
    plt.close(figure)
    assert exported.manifest_path is not None
    assert Dataset.open(exported.manifest_path, progress=False).location == exported.location
    assert "manifest=" not in repr(reopened)
    with pytest.raises(DatasetValidationError, match="requires vector annotations"):
        reopened.split({"train": 0.8, "val": 0.2})


def test_semantic_mask_export_is_task_restricted_and_virtual_until_export(
    tmp_path: Path,
) -> None:
    detect_source = make_yolo_dataset(tmp_path / "detect", task="detect")
    with pytest.raises(DatasetValidationError, match="requires a segmentation"):
        Dataset.open(detect_source, task="detect", progress=False).export(
            destination=tmp_path / "invalid",
            format="semantic_masks",
            visualize=False,
            progress=False,
        )

    segment_source = make_yolo_dataset(
        tmp_path / "segment",
        task="segment",
        train_rows=["0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"],
        val_rows=["0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"],
    )
    plan = Dataset.open(segment_source, task="segment", progress=False).tile(
        tile_size=80,
        overlap=0,
        negative_tiles="none",
        visualize=False,
        progress=False,
    )
    dry = plan.export(
        destination=tmp_path / "dry-semantic",
        format="semantic_masks",
        dry_run=True,
        visualize=False,
        progress=False,
    )
    assert dry is plan
    assert not (tmp_path / "dry-semantic").exists()

    exported = plan.export(
        destination=tmp_path / "tiled-semantic",
        format="semantic_masks",
        visualize=False,
        progress=False,
    )
    assert isinstance(exported, Dataset)
    assert exported.format == "semantic_masks"
    assert any(path.is_file() for path in exported.mask_dirs["train"].rglob("*.png"))


def test_semantic_mask_dataset_open_skips_invalid_pairs_and_closes_audit_figure(
    tmp_path: Path,
) -> None:
    from matplotlib import pyplot as plt

    source = make_yolo_dataset(
        tmp_path / "invalid-pair-source",
        task="segment",
        train_rows=[
            "0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8",
            "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75",
        ],
        val_rows=["0 0.3 0.3 0.7 0.3 0.7 0.7 0.3 0.7"],
    )
    exported = Dataset.open(source, task="segment", progress=False).export(
        destination=tmp_path / "invalid-pair-masks",
        format="semantic_masks",
        visualize=False,
        progress=False,
    )
    invalid_mask = sorted(exported.mask_dirs["train"].rglob("*.png"))[0]
    with Image.open(invalid_mask) as opened:
        opened.convert("RGB").save(invalid_mask)

    open_figures = set(plt.get_fignums())
    reopened = Dataset.open(exported.location, errors="skip", progress=False)

    assert reopened.format == "semantic_masks"
    assert len(reopened._samples) == 2
    assert reopened.validation_audit["status"] == "passed_with_skips"
    assert reopened.validation_audit["skipped_count"] == 1
    assert any("invalid semantic-mask pair" in warning for warning in reopened.warnings)
    assert set(plt.get_fignums()) == open_figures


def test_export_formats_materializes_yolo_then_semantic_masks(tmp_path: Path) -> None:
    source = make_yolo_dataset(
        tmp_path / "multi_format_source",
        task="segment",
        train_rows=["0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"],
        val_rows=["0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75"],
    )
    pipeline = Dataset.open(source, task="segment", progress=False).tile(
        tile_size=80,
        overlap=0,
        negative_tiles="none",
        visualize=False,
        progress=False,
    )

    outputs = pipeline.export_formats(
        {
            "semantic_masks": tmp_path / "multi_masks",
            "yolo": tmp_path / "multi_yolo",
        },
        visualize=False,
        progress=False,
    )

    assert list(outputs) == ["yolo", "semantic_masks"]
    assert isinstance(outputs["yolo"], Dataset)
    assert isinstance(outputs["semantic_masks"], Dataset)
    assert outputs["semantic_masks"].format == "semantic_masks"
    assert outputs["semantic_masks"].manifest["source_dataset"]["location"] == str(
        outputs["yolo"].location
    )
    assert (outputs["yolo"].location / "data.yaml").is_file()
    assert any(outputs["semantic_masks"].mask_dirs["train"].rglob("*.png"))


def test_export_formats_preflights_all_destinations(tmp_path: Path) -> None:
    source = make_yolo_dataset(
        tmp_path / "multi_format_preflight",
        task="segment",
        train_rows=["0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"],
        val_rows=["0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"],
    )
    dataset = Dataset.open(source, task="segment", progress=False)
    occupied = tmp_path / "occupied_masks"
    occupied.mkdir()

    with pytest.raises(FileExistsError, match="occupied_masks"):
        dataset.export_formats(
            {
                "yolo": tmp_path / "must_not_be_written",
                "semantic_masks": occupied,
            },
            visualize=False,
            progress=False,
        )
    assert not (tmp_path / "must_not_be_written").exists()


def test_semantic_mask_export_includes_split_group_audit(tmp_path: Path) -> None:
    source = make_yolo_dataset(
        tmp_path / "grouped_segments",
        task="segment",
        train_rows=[
            "0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8",
            "0 0.1 0.1 0.6 0.1 0.6 0.6 0.1 0.6",
        ],
        val_rows=["0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75"],
    )
    exported = Dataset.open(source, task="segment", progress=False).split(
        {"train": 0.5, "val": 0.5},
        group_by=lambda path: path.name,
        seed=5,
        visualize=False,
        progress=False,
    ).export(
        destination=tmp_path / "grouped-semantic",
        format="semantic_masks",
        visualize=False,
        progress=False,
    )

    report = json.loads(
        (exported.location / "reports" / "split_group_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "passed"
    assert report["total"] == {"images": 3, "distinct_groups": 3}
    validation = exported.manifest["validation"]["split_group_isolation"]
    assert validation["report"] == "reports/split_group_audit.json"
    records = [
        json.loads(line)
        for line in (exported.location / "provenance.jsonl").read_text().splitlines()
    ]
    assert all("split_group" in record for record in records)


def test_semantic_mask_export_rejects_relative_stem_collisions_before_publication(
    tmp_path: Path,
) -> None:
    root = tmp_path / "colliding_segments"
    for suffix in (".jpg", ".png"):
        image = root / "train" / "images" / "nested" / f"sample{suffix}"
        make_image(image, size=(80, 60))
    label = root / "train" / "labels" / "nested" / "sample.txt"
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text("0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n", encoding="utf-8")
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root.resolve()),
                "train": "train/images",
                "val": None,
                "test": None,
                "names": ["foreground"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "collision_output"

    with pytest.raises(DatasetValidationError, match="same semantic-mask path"):
        Dataset.open(root, task="segment", progress=False).export(
            destination=destination,
            format="semantic_masks",
            visualize=False,
            progress=False,
        )
    assert not destination.exists()
