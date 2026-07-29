from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from dataset_fixer import Dataset, DatasetValidationError, Task
from conftest import make_image, make_yolo_dataset


def test_segment_pose_and_polo_load(tmp_path: Path) -> None:
    segment = make_yolo_dataset(
        tmp_path / "segment",
        task="segment",
        names=["fruit"],
        train_rows=["0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"],
        val_rows=["0 0.25 0.25 0.75 0.25 0.75 0.75"],
    )
    pose = make_yolo_dataset(
        tmp_path / "pose",
        task="pose",
        names=["fruit"],
        train_rows=["0 0.5 0.5 0.5 0.5 0.4 0.4 2 0.6 0.6 1"],
        val_rows=["0 0.5 0.5 0.5 0.5 0.4 0.4 2 0.6 0.6 1"],
        extra={"kpt_shape": [2, 3], "flip_idx": [1, 0], "kpt_names": {0: ["left", "right"]}},
    )
    polo = make_yolo_dataset(
        tmp_path / "polo",
        task="polo",
        names=["fruit"],
        train_rows=["0 15 0.5 0.5"],
        val_rows=["0 15 0.4 0.4"],
        extra={"radii": {0: 15}},
    )
    assert Dataset.open(segment, task="segment", progress=False).task is Task.SEGMENT
    assert Dataset.open(pose, task="pose", progress=False).task is Task.POSE
    assert Dataset.open(polo, task="polo", progress=False).task is Task.POLO


def test_invalid_segmentation_can_be_skipped_virtually(tmp_path: Path) -> None:
    invalid_row = "0 0.2 0.2 0.8 0.8 0.8 0.2 0.2 0.8"
    valid_row = "0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"
    source = make_yolo_dataset(
        tmp_path / "segment_with_invalid_polygon",
        task="segment",
        names=["fruit"],
        train_rows=[f"{invalid_row}\n{valid_row}"],
        val_rows=[valid_row],
    )
    source_label = next((source / "train" / "labels").rglob("*.txt"))
    original_label = source_label.read_text(encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="Invalid or self-intersecting polygon"):
        Dataset.open(source, task="segment", progress=False)

    dataset = Dataset.open(
        source,
        task="segment",
        errors="skip",
        progress=False,
    )

    assert sum(len(sample.annotations) for sample in dataset._samples) == 2
    assert any("Skipped invalid annotation" in warning for warning in dataset.warnings)
    assert any("Invalid or self-intersecting polygon" in warning for warning in dataset.warnings)
    assert source_label.read_text(encoding="utf-8") == original_label

    exported = dataset.export(
        destination=tmp_path / "segment_without_invalid_polygon",
        visualize=False,
        progress=False,
    )
    exported_train_label = next((exported.location / "train" / "labels").rglob("*.txt"))
    assert len(exported_train_label.read_text(encoding="utf-8").splitlines()) == 1
    manifest = json.loads((exported.location / "dataset-fixer.json").read_text(encoding="utf-8"))
    assert any("Skipped invalid annotation" in warning for warning in manifest["warnings"])


def test_grid_tiling_transforms_segment_and_pose(tmp_path: Path) -> None:
    segment = make_yolo_dataset(
        tmp_path / "segment_grid",
        task="segment",
        names=["fruit"],
        train_rows=["0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"],
        val_rows=["0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"],
        size=(160, 120),
    )
    pose = make_yolo_dataset(
        tmp_path / "pose_grid",
        task="pose",
        names=["fruit"],
        train_rows=["0 0.5 0.5 0.5 0.5 0.4 0.4 2 0.6 0.6 1"],
        val_rows=["0 0.5 0.5 0.5 0.5 0.4 0.4 2 0.6 0.6 1"],
        size=(160, 120),
        extra={"kpt_shape": [2, 3], "flip_idx": [1, 0], "kpt_names": {0: ["left", "right"]}},
    )
    segment_tiles = Dataset.open(segment, task="segment", progress=False).tile(
        tile_size=100,
        overlap=0.2,
        visualize=False,
        progress=False,
    ).export(destination=tmp_path / "segment_tiles", visualize=False, progress=False)
    pose_tiles = Dataset.open(pose, task="pose", progress=False).tile(
        tile_size=100,
        overlap=0.2,
        visualize=False,
        progress=False,
    ).export(destination=tmp_path / "pose_tiles", visualize=False, progress=False)
    assert any(annotation.polygon for sample in segment_tiles._samples for annotation in sample.annotations)
    assert any(annotation.keypoints for sample in pose_tiles._samples for annotation in sample.annotations)
    assert all(
        0 <= x <= sample.width and 0 <= y <= sample.height
        for sample in pose_tiles._samples
        for annotation in sample.annotations
        for x, y, _ in annotation.keypoints or []
    )


def test_polo_coverage_reports_and_visual_audit(tmp_path: Path) -> None:
    source = make_yolo_dataset(
        tmp_path / "polo_source",
        task="polo",
        names=["fruit"],
        train_rows=["0 4 0.4 0.4\n0 9 0.6 0.6"],
        val_rows=["0 7 0.5 0.5"],
        size=(300, 260),
        extra={"radii": {0: 15}},
    )
    dataset = Dataset.open(source, task="polo", progress=False)
    tiled = dataset.tile(
        mode="coverage",
        tile_size=100,
        large_image_threshold=200,
        scale_range=(1.0, 1.0),
        polo_radius_px=10,
        target_appearances_per_object=1,
        sparse_appearances_per_object=1,
        max_tiles_per_source_image=5,
        background_ratio=0.5,
        seed=3,
        visualize=True,
        progress=False,
    ).export(destination=tmp_path / "coverage", visualize=False, progress=False)
    summary = tiled.location / "coverage_summary"
    for filename in ("label_coverage.csv", "label_hit_summary.csv", "class_coverage_summary.csv", "tile_summary.csv"):
        assert (summary / filename).is_file()
    assert list((summary / "annotated_originals").rglob("*.jpg"))
    rows = list(csv.DictReader((summary / "label_coverage.csv").open(encoding="utf-8")))
    assert rows and all(float(row["actual_coverages"]) >= 1 for row in rows)
    assert all(annotation.radius == 10 for sample in tiled._samples for annotation in sample.annotations)


@pytest.mark.parametrize("task", ["detect", "segment", "pose", "polo"])
def test_coverage_tiling_supports_every_task(task: str, tmp_path: Path) -> None:
    rows = {
        "detect": "0 0.5 0.5 0.2 0.2",
        "segment": "0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6",
        "pose": "0 0.5 0.5 0.2 0.2 0.47 0.47 2 0.53 0.53 2",
        "polo": "0 10 0.5 0.5",
    }
    extra = {
        "pose": {"kpt_shape": [2, 3], "flip_idx": [1, 0], "kpt_names": {0: ["left", "right"]}},
        "polo": {"radii": {0: 10}},
    }.get(task, {})
    source = make_yolo_dataset(
        tmp_path / f"{task}_coverage_source",
        task=task,
        names=["fruit"],
        train_rows=[rows[task]],
        val_rows=[rows[task]],
        size=(300, 260),
        extra=extra,
    )

    tiled = (
        Dataset.open(source, task=task, progress=False)
        .tile(
            mode="coverage",
            tile_size=100,
            large_image_threshold=200,
            scale_range=(1.0, 1.0),
            target_appearances_per_object=2,
            sparse_appearances_per_object=2,
            background_ratio=0,
            max_tiles_per_source_image=10,
            seed=7,
            visualize=False,
            progress=False,
        )
        .export(
            destination=tmp_path / f"{task}_coverage",
            visualize=False,
            progress=False,
        )
    )

    rows_out = list(csv.DictReader((tiled.location / "coverage_summary" / "label_coverage.csv").open()))
    assert rows_out
    assert all(int(row["requested_coverages"]) == 2 for row in rows_out)
    assert all(int(row["actual_coverages"]) == 2 for row in rows_out)
    assert all(sample.width == 100 and sample.height == 100 for sample in tiled._samples)
    assert tiled.task.value == task


def test_coverage_object_appearance_override_uses_source_id(tmp_path: Path) -> None:
    source = make_yolo_dataset(
        tmp_path / "coverage_override_source",
        task="detect",
        names=["fruit"],
        train_rows=["0 0.5 0.5 0.2 0.2"],
        val_rows=["0 0.5 0.5 0.2 0.2"],
        size=(300, 260),
    )
    dataset = Dataset.open(source, task="detect", progress=False)
    source_id = dataset._samples[0].annotations[0].source_id

    tiled = dataset.tile(
        mode="coverage",
        tile_size=100,
        large_image_threshold=200,
        scale_range=(1.0, 1.0),
        target_appearances_per_object=1,
        sparse_appearances_per_object=1,
        object_appearance_overrides={str(source_id): 3},
        background_ratio=0,
        visualize=False,
        progress=False,
    ).export(
        destination=tmp_path / "coverage_override",
        visualize=False,
        progress=False,
    )

    rows = list(csv.DictReader((tiled.location / "coverage_summary" / "label_coverage.csv").open()))
    overridden = next(row for row in rows if row["source_id"] == str(source_id))
    assert int(overridden["requested_coverages"]) == 3
    assert int(overridden["actual_coverages"]) == 3
    with pytest.raises(ValueError, match="Unknown object_appearance_overrides"):
        dataset.tile(mode="coverage", object_appearance_overrides={"missing": 2})


def test_coco_detection_export_compacts_categories(tmp_path: Path) -> None:
    root = tmp_path / "coco"
    make_image(root / "images" / "train" / "one.jpg", size=(100, 80))
    make_image(root / "images" / "val" / "two.jpg", size=(100, 80))
    categories = [{"id": 4, "name": "apple"}, {"id": 9, "name": "pear"}]
    train = {
        "info": {"name": "coco-fruit"},
        "images": [{"id": 1, "file_name": "one.jpg", "width": 100, "height": 80}],
        "categories": categories,
        "annotations": [{"id": 11, "image_id": 1, "category_id": 9, "bbox": [10, 10, 20, 30], "area": 600}],
    }
    val = {
        "images": [{"id": 2, "file_name": "two.jpg", "width": 100, "height": 80}],
        "categories": categories,
        "annotations": [{"id": 12, "image_id": 2, "category_id": 4, "bbox": [20, 10, 30, 30], "area": 900}],
    }
    (root / "train.json").write_text(json.dumps(train), encoding="utf-8")
    (root / "val.json").write_text(json.dumps(val), encoding="utf-8")
    dataset = Dataset.open(root, task="detect", progress=False)
    assert dataset.classes == {0: "apple", 1: "pear"}
    exported = dataset.export(destination=tmp_path / "coco_yolo", visualize=False, progress=False)
    assert exported.classes == {0: "apple", 1: "pear"}
    data = yaml.safe_load(exported.data_yaml.read_text(encoding="utf-8"))
    assert data["train"] == "train/images" and data["val"] == "val/images"
    labels = "\n".join(path.read_text(encoding="utf-8") for path in exported.location.rglob("*.txt") if "labels" in path.parts)
    assert labels.startswith(("0 ", "1 "))


def test_coco_errors_skip_filters_bad_records_across_loading_stages(tmp_path: Path) -> None:
    root = tmp_path / "coco_with_bad_records"
    make_image(root / "images" / "one.jpg", size=(100, 80))
    data = {
        "images": [
            {"id": 1, "file_name": "one.jpg", "width": 100, "height": 80},
            {"id": 2, "file_name": "missing.jpg", "width": 100, "height": 80},
        ],
        "categories": [{"id": 4, "name": "apple"}],
        "annotations": [
            {"id": 11, "image_id": 1, "category_id": 4, "bbox": [10, 10, 20, 30]},
            {"id": 12, "image_id": 1, "category_id": 4, "bbox": ["bad", 10, 20, 30]},
            {"id": 13, "image_id": 99, "category_id": 4, "bbox": [10, 10, 20, 30]},
            {"id": 11, "image_id": 1, "category_id": 4, "bbox": [10, 10, 20, 30]},
        ],
    }
    (root / "annotations.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "annotations.json").write_text(json.dumps(data), encoding="utf-8")

    dataset = Dataset.open(root, task="detect", errors="skip", progress=False)

    assert len(dataset._samples) == 1
    assert len(dataset._samples[0].annotations) == 1
    assert any("unknown image" in warning for warning in dataset.warnings)
    assert any("IDs must be unique" in warning for warning in dataset.warnings)
    assert any("Malformed COCO annotation" in warning for warning in dataset.warnings)
    assert any("COCO image file not found" in warning for warning in dataset.warnings)
