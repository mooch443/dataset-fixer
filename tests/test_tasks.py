from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml
from PIL import Image

from dataset_fixer import Dataset, Task
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
        fixed_polo_radius_px=10,
        target_coverage_per_label=1,
        sparse_coverage_per_label=1,
        max_total_tiles_per_source_image=5,
        max_bg_ratio=0.5,
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
