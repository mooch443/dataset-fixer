from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml
from PIL import Image

from dataset_fixer import Dataset, DatasetValidationError, Task
from dataset_fixer.models import Annotation, Sample
from dataset_fixer.visualization import save_coverage_annotated_original
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
    assert all(row["coverage_type"] == "sparse" for row in rows)
    assert all(annotation.radius == 10 for sample in tiled._samples for annotation in sample.annotations)
    with Image.open(next((summary / "annotated_originals").rglob("*.jpg"))) as preview:
        assert preview.height > 260


def test_coverage_visual_keeps_legend_off_image_and_boxes_segmentations(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "blue.jpg"
    make_image(image_path, size=(120, 100), color=(20, 90, 150))
    polygon = [(30.0, 20.0), (90.0, 50.0), (45.0, 80.0)]
    sample = Sample(
        image_path=image_path,
        relative_path=Path("blue.jpg"),
        split="val",
        width=120,
        height=100,
        annotations=[
            Annotation(
                class_id=0,
                polygon=polygon,
                bbox=(30.0, 20.0, 90.0, 80.0),
            )
        ],
    )
    output = tmp_path / "coverage_visual.jpg"
    save_coverage_annotated_original(
        sample,
        {0: 3},
        {0: 5},
        {0: "dense"},
        [],
        output,
        {
            "target_appearances_per_object": 5,
            "sparse_appearances_per_object": 1,
            "max_tiles_per_source_image": 100,
            "min_nearby_objects_for_full_coverage": 5,
            "dense_neighbor_radius_px": 50.0,
            "background_ratio": 0.1,
            "polo_radius_px": 15.0,
            "jpeg_quality": 100,
        },
    )

    with Image.open(output) as preview:
        assert preview.size[0] >= 900
        scale = preview.size[0] / 120
        displayed_image_height = round(100 * scale)
        assert preview.size[1] > displayed_image_height
        pixels = preview.convert("RGB")
        # The triangle does not pass through its top-right bounding-box corner;
        # a magenta pixel there therefore comes from the new segmentation box.
        corner_x, corner_y = round(90 * scale), round(20 * scale)
        nearby = [
            pixels.getpixel((x, y))
            for x in range(corner_x - 8, corner_x + 9)
            for y in range(corner_y - 8, corner_y + 9)
        ]
        assert any(red > 170 and blue > 130 and green < 140 for red, green, blue in nearby)


def test_coverage_background_ratio_applies_to_complete_output(tmp_path: Path) -> None:
    positive = "0 0.5 0.5 0.2 0.2"
    rows = [positive, positive, positive, *([""] * 7)]
    source = make_yolo_dataset(
        tmp_path / "coverage_with_existing_backgrounds",
        task="detect",
        names=["fruit"],
        train_rows=rows,
        val_rows=rows,
        size=(160, 120),
    )

    tiled = (
        Dataset.open(source, task="detect", progress=False)
        .tile(
            mode="coverage",
            tile_size=100,
            large_image_threshold=500,
            background_ratio=0.25,
            visualize=False,
            progress=False,
        )
        .export(
            destination=tmp_path / "coverage_with_final_background_ratio",
            visualize=False,
            progress=False,
        )
    )

    for split in ("train", "val"):
        output = [sample for sample in tiled._samples if sample.split == split]
        assert len(output) == 4
        assert sum(not sample.annotations for sample in output) == 1
        assert sum(not sample.annotations for sample in output) / len(output) == 0.25

    summary = {
        row["split"]: row
        for row in csv.DictReader(
            (tiled.location / "coverage_summary" / "tile_summary.csv").open(encoding="utf-8")
        )
    }
    for split in ("train", "val", "all"):
        assert float(summary[split]["background_fraction"]) == 0.25
    assert int(summary["train"]["candidate_background_source_images"]) == 7
    assert int(summary["train"]["dropped_background_source_images"]) == 6
    assert int(summary["train"]["target_background_images"]) == 1
    assert int(summary["train"]["actual_background_images"]) == 1
    class_counts = json.loads(
        (tiled.location / "reports" / "class_counts.json").read_text(encoding="utf-8")
    )
    assert class_counts["operation"] == "tile-coverage"
    assert class_counts["after"]["background"] == 2
    assert class_counts["image_composition"]["after"] == {
        "annotated": 6,
        "background": 2,
        "total": 8,
        "background_fraction": 0.25,
    }
    assert class_counts["annotation_counts"]["after"]["0"] == 6


def test_coverage_balances_background_source_types(tmp_path: Path) -> None:
    positive = "0 0.5 0.5 0.1 0.1"
    source = make_yolo_dataset(
        tmp_path / "coverage_background_source_mix",
        task="detect",
        names=["fruit"],
        train_rows=[positive, positive, "", ""],
        val_rows=[positive, positive, "", ""],
        size=(300, 260),
    )

    tiled = (
        Dataset.open(source, task="detect", progress=False)
        .tile(
            mode="coverage",
            tile_size=100,
            large_image_threshold=200,
            scale_range=(1.0, 1.0),
            target_appearances_per_object=1,
            sparse_appearances_per_object=1,
            background_ratio=0.5,
            seed=13,
            visualize=False,
            progress=False,
        )
        .export(
            destination=tmp_path / "coverage_balanced_background_sources",
            visualize=False,
            progress=False,
        )
    )

    sampling = json.loads(
        (
            tiled.location
            / "coverage_summary"
            / "background_sampling.json"
        ).read_text(encoding="utf-8")
    )
    for split in ("train", "val"):
        details = sampling["splits"][split]
        assert details["status"] == "target and equal source mix met"
        assert details["target_background_images"] == 2
        assert details["actual_background_images"] == 2
        assert details["actual_from_empty_source_images"] == 1
        assert details["actual_from_populated_image_space"] == 1
        assert details["actual_background_fraction"] == 0.5

    background_sources = Counter(
        record.get("background_source")
        for record in tiled.provenance.values()
        if record.get("output_annotation_count") == 0
    )
    assert background_sources == {
        "empty_source_image": 2,
        "populated_image_empty_space": 2,
    }


def test_grid_negative_fraction_applies_to_complete_output(tmp_path: Path) -> None:
    positive = "0 0.5 0.5 0.2 0.2"
    rows = [positive, positive, positive, *([""] * 7)]
    source = make_yolo_dataset(
        tmp_path / "grid_with_existing_backgrounds",
        task="detect",
        names=["fruit"],
        train_rows=rows,
        val_rows=rows,
        size=(160, 120),
    )

    tiled = (
        Dataset.open(source, task="detect", progress=False)
        .tile(
            mode="grid",
            tile_size=200,
            negative_tiles=0.25,
            visualize=False,
            progress=False,
        )
        .export(
            destination=tmp_path / "grid_with_final_background_ratio",
            visualize=False,
            progress=False,
        )
    )

    for split in ("train", "val"):
        output = [sample for sample in tiled._samples if sample.split == split]
        assert len(output) == 4
        assert sum(not sample.annotations for sample in output) == 1
        assert sum(not sample.annotations for sample in output) / len(output) == 0.25
    class_counts = json.loads(
        (tiled.location / "reports" / "class_counts.json").read_text(encoding="utf-8")
    )
    assert class_counts["operation"] == "tile-grid"
    assert class_counts["after"]["background"] == 2


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
