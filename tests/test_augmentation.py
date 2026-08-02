from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image
from shapely.geometry import GeometryCollection, LineString, Polygon

from dataset_fixer import Dataset, DatasetValidationError
from dataset_fixer import tiling as tiling_module
from dataset_fixer.utils import sha256_file
from conftest import make_yolo_dataset

A = pytest.importorskip("albumentations")


def _augmented_sample(dataset: Dataset, split: str = "train"):
    return next(
        sample
        for sample in dataset._samples
        if sample.split == split and "__aug-001" in sample.relative_path.stem
    )


def test_augment_is_virtual_deterministic_and_records_full_provenance(tmp_path: Path) -> None:
    source = make_yolo_dataset(
        tmp_path / "detect_source",
        task="detect",
        names=["fruit"],
        train_rows=["0 0.2 0.5 0.2 0.4"],
        val_rows=["0 0.5 0.5 0.2 0.2"],
    )
    dataset = Dataset.open(source, task="detect", progress=False)
    plan = dataset.augment(
        [A.HorizontalFlip(p=1.0), A.RandomBrightnessContrast(p=1.0)],
        copies=1,
        include_original=False,
        seed=17,
        visualize=True,
    )
    assert plan.data_yaml is None
    assert "augment" in repr(plan)
    assert not (tmp_path / "augmented").exists()

    first = plan.export(destination=tmp_path / "augmented", visualize=False, progress=False)
    second = plan.export(destination=tmp_path / "augmented_again", visualize=False, progress=False)
    augmented = _augmented_sample(first)
    assert augmented.annotations[0].bbox == pytest.approx((112.0, 36.0, 144.0, 84.0), abs=1e-3)
    assert len([sample for sample in first._samples if sample.split == "train"]) == 1
    assert len([sample for sample in first._samples if sample.split == "val"]) == 1
    assert sha256_file(augmented.image_path) == sha256_file(_augmented_sample(second).image_path)
    assert (first.location / "reports" / "augmentation_preview.jpg").is_file()
    report = json.loads((first.location / "reports" / "augmentation.json").read_text(encoding="utf-8"))
    assert report["generated_images"] == 1
    counts = json.loads(
        (first.location / "reports" / "augmentation_class_counts.json").read_text(encoding="utf-8")
    )
    assert counts["before"]["background"] == 0
    assert counts["after"]["background"] == 0
    assert counts["names"]["background"] == "background"
    assert (first.location / "reports" / "augmentation_class_counts.jpg").is_file()
    record = first.provenance[f"train/images/{augmented.relative_path.as_posix()}"]
    assert record["augmentation_index"] == 1
    assert record["augmentation_seed"]
    assert record["albumentations_applied"]
    manifest = json.loads((first.location / "dataset-fixer.json").read_text(encoding="utf-8"))
    assert manifest["environment"]["packages"]["albumentations"] == A.__version__
    assert manifest["history"][0]["settings"]["pipeline"]["__version__"]


def test_augment_transforms_segment_pose_and_polo_annotations(tmp_path: Path) -> None:
    segment_source = make_yolo_dataset(
        tmp_path / "segment_source",
        task="segment",
        names=["fruit"],
        train_rows=["0 0.1 0.2 0.4 0.2 0.4 0.8 0.1 0.8"],
        val_rows=["0 0.2 0.2 0.4 0.2 0.4 0.4 0.2 0.4"],
    )
    pose_source = make_yolo_dataset(
        tmp_path / "pose_source",
        task="pose",
        names=["fruit"],
        train_rows=["0 0.25 0.5 0.2 0.4 0.2 0.4 2 0.3 0.6 1"],
        val_rows=["0 0.5 0.5 0.2 0.2 0.4 0.4 2 0.6 0.6 1"],
        extra={"kpt_shape": [2, 3], "flip_idx": [1, 0], "kpt_names": {0: ["left", "right"]}},
    )
    polo_source = make_yolo_dataset(
        tmp_path / "polo_source",
        task="polo",
        names=["fruit"],
        train_rows=["0 15 0.25 0.5"],
        val_rows=["0 15 0.5 0.5"],
        extra={"radii": {0: 15}},
    )

    segment = (
        Dataset.open(segment_source, task="segment", progress=False)
        .augment([A.HorizontalFlip(p=1.0)], include_original=False, visualize=False)
        .export(destination=tmp_path / "segment_aug", visualize=False, progress=False)
    )
    polygon = _augmented_sample(segment).annotations[0].polygon
    assert polygon is not None
    assert min(x for x, _ in polygon) == pytest.approx(96.0, abs=2.0)
    assert max(x for x, _ in polygon) == pytest.approx(144.0, abs=2.0)

    pose = (
        Dataset.open(pose_source, task="pose", progress=False)
        .augment([A.HorizontalFlip(p=1.0)], include_original=False, visualize=False)
        .export(destination=tmp_path / "pose_aug", visualize=False, progress=False)
    )
    keypoints = _augmented_sample(pose).annotations[0].keypoints
    assert keypoints is not None
    # Keypoints address pixel centers, so horizontal flip is (width - 1) - x.
    assert keypoints[0] == pytest.approx((111.0, 72.0, 1.0), abs=1e-3)
    assert keypoints[1] == pytest.approx((127.0, 48.0, 2.0), abs=1e-3)

    polo = (
        Dataset.open(polo_source, task="polo", progress=False)
        .augment([A.HorizontalFlip(p=1.0)], include_original=False, visualize=False)
        .export(destination=tmp_path / "polo_aug", visualize=False, progress=False)
    )
    point = _augmented_sample(polo).annotations[0]
    assert point.point == pytest.approx((119.0, 60.0), abs=1e-3)
    assert point.radius == pytest.approx(15.0, abs=2.0)


def test_augment_rejects_reserved_compose_annotation_processors(detect_dataset: Path) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    with pytest.raises(TypeError, match="controls bbox_params"):
        dataset.augment([A.HorizontalFlip()], bbox_params={"format": "yolo"})


def test_polo_augmentation_rejects_non_circular_geometry_unless_explicitly_lossy(
    tmp_path: Path,
) -> None:
    source = make_yolo_dataset(
        tmp_path / "polo_distortion_source",
        task="polo",
        names=["fruit"],
        train_rows=["0 15 0.5 0.5"],
        val_rows=["0 15 0.5 0.5"],
        extra={"radii": {0: 15}},
    )
    dataset = Dataset.open(source, task="polo", progress=False)
    strict = dataset.augment(
        A.Resize(height=60, width=160, p=1.0), include_original=False, visualize=False
    )
    with pytest.raises(DatasetValidationError, match="non-circular"):
        strict.export(destination=tmp_path / "strict", visualize=False, progress=False)
    assert not (tmp_path / "strict").exists()

    lossy = dataset.augment(
        A.Resize(height=60, width=160, p=1.0),
        include_original=False,
        allow_lossy=True,
        visualize=False,
    ).export(destination=tmp_path / "lossy", visualize=False, progress=False)
    report = json.loads((lossy.location / "reports" / "augmentation.json").read_text(encoding="utf-8"))
    assert report["lossy_annotations"] == 1


def test_coverage_crop_pipeline_transforms_before_crop_without_border_fill(
    tmp_path: Path,
) -> None:
    source = make_yolo_dataset(
        tmp_path / "virtual_camera",
        task="segment",
        names=["school"],
        train_rows=["0 0.35 0.35 0.65 0.35 0.65 0.65 0.35 0.65"],
        val_rows=["0 0.35 0.35 0.65 0.35 0.65 0.65 0.35 0.65"],
        size=(240, 240),
    )
    dataset = Dataset.open(source, task="segment", progress=False)
    plan = dataset.tile(
        mode="coverage",
        tile_size=96,
        scale_range=(1.0, 1.0),
        large_image_threshold=32,
        target_appearances_per_object=1,
        sparse_appearances_per_object=1,
        background_ratio=0,
        allow_lossy=True,
        crop_transforms=[
            A.Affine(scale=(1.25, 1.25), rotate=(18, 18), fill=0, p=1.0),
        ],
        seed=7,
        visualize=False,
        progress=False,
    )
    exported = plan.export(
        destination=tmp_path / "virtual_tiles", visualize=False, progress=False
    )
    repeated = plan.export(
        destination=tmp_path / "virtual_tiles_repeated", visualize=False, progress=False
    )

    train = next(sample for sample in exported._samples if sample.split == "train")
    repeated_train = next(sample for sample in repeated._samples if sample.split == "train")
    val = next(sample for sample in exported._samples if sample.split == "val")
    with Image.open(train.image_path) as opened:
        pixels = opened.convert("RGB").getdata()
        assert min(pixels, key=sum) != (0, 0, 0)
        assert min(min(pixel) for pixel in pixels) > 5
    assert train.provenance["tile_mode"] == "coverage-augmented"
    assert train.provenance["valid_pixel_fraction"] == 1.0
    assert train.provenance["crop_albumentations_applied"]
    assert train.provenance["crop_pipeline"]
    assert train.provenance["crop_transform_seed"] == repeated_train.provenance["crop_transform_seed"]
    assert train.provenance["crop"] == repeated_train.provenance["crop"]
    assert sha256_file(train.image_path) == sha256_file(repeated_train.image_path)
    assert val.provenance["tile_mode"] == "coverage"
    report = json.loads((exported.location / "reports" / "crop_augmentation.json").read_text())
    assert report["coordinate_order"] == "full-source transform, then coverage crop"
    assert report["sampling_unit"] == (
        "one independently seeded full-source virtual camera per crop candidate"
    )
    assert report["transformed_splits"] == ["train"]
    assert "augment_val=False" in report["unchanged_splits"]["val"]
    assert report["stats"]["accepted_positive_tiles"] >= 1
    assert any("augment_val=False" in warning for warning in exported.warnings)
    coverage_rows = list(
        csv.DictReader(
            (exported.location / "coverage_summary" / "source_pixel_coverage.csv").open()
        )
    )
    train_coverage = next(row for row in coverage_rows if row["split"] == "train")
    assert train_coverage["coverage_status"] == "exact"
    assert float(train_coverage["source_pixel_coverage_percent"]) == pytest.approx(
        10.24,
        rel=1e-3,
    )


def test_each_repeated_positive_crop_samples_a_distinct_virtual_camera(
    tmp_path: Path,
) -> None:
    source = make_yolo_dataset(
        tmp_path / "fresh_camera_per_crop",
        task="detect",
        names=["fruit"],
        train_rows=["0 0.5 0.5 0.08 0.08"],
        val_rows=["0 0.5 0.5 0.08 0.08"],
        size=(320, 320),
    )
    exported = Dataset.open(source, task="detect", progress=False).tile(
        mode="coverage",
        splits=["train"],
        tile_size=96,
        scale_range=(1.0, 1.0),
        large_image_threshold=32,
        target_appearances_per_object=4,
        sparse_appearances_per_object=4,
        background_ratio=0,
        max_attempts_per_target=100,
        allow_lossy=True,
        crop_transforms=A.Compose(
            [A.Affine(scale=(1.05, 1.25), rotate=(-75, 75), p=1.0)]
        ),
        seed=91,
        visualize=False,
        progress=False,
    ).export(
        destination=tmp_path / "fresh_camera_tiles",
        visualize=False,
        progress=False,
    )

    assert len(exported._samples) == 4
    seeds = [sample.provenance["crop_transform_seed"] for sample in exported._samples]
    rotations = [
        sample.provenance["crop_albumentations_applied"][0]["params"]["rotate"]
        for sample in exported._samples
    ]
    matrices = [
        sample.provenance["crop_albumentations_applied"][0]["params"]["matrix"]
        for sample in exported._samples
    ]
    assert len(set(seeds)) == 4
    assert len(set(rotations)) == 4
    assert len(set(matrices)) == 4
    assert all(
        sample.provenance["tile_mode"] == "coverage-augmented"
        for sample in exported._samples
    )
    report = json.loads(
        (exported.location / "reports" / "crop_augmentation.json").read_text()
    )
    assert report["accepted_virtual_camera_crops"] == 4
    assert report["distinct_accepted_virtual_camera_seeds"] == 4
    assert report["fresh_seed_per_accepted_crop"] is True
    assert report["accepted_virtual_camera_crops_by_split"] == {"train": 4}


def test_virtual_coverage_errors_skip_rejects_and_resamples_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_yolo_dataset(
        tmp_path / "skippable_virtual_geometry",
        task="segment",
        names=["school"],
        train_rows=["0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6"],
        val_rows=["0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6"],
        size=(200, 200),
    )
    original = tiling_module._virtual_positive_candidate
    calls = 0

    def fail_first_candidate(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            sample, focus_idx = args[:2]
            annotation = sample.annotations[focus_idx]
            source_polygon = Polygon(annotation.polygon)
            mixed = GeometryCollection(
                [source_polygon, LineString([(0, 0), (1, 1)])]
            )
            raise tiling_module._segmentation_crop_error(
                "Cropped segmentation produced unsupported mixed geometry",
                annotation,
                (20, 20, 100, 100),
                source_image=sample.image_path,
                annotation_index=focus_idx,
                source_geometry=source_polygon,
                result_geometry=mixed,
                detail="Synthetic candidate-local regression fixture.",
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(
        tiling_module,
        "_virtual_positive_candidate",
        fail_first_candidate,
    )
    exported = Dataset.open(source, task="segment", progress=False).tile(
        mode="coverage",
        splits=["train"],
        tile_size=80,
        scale_range=(1.0, 1.0),
        large_image_threshold=20,
        target_appearances_per_object=1,
        sparse_appearances_per_object=1,
        background_ratio=0,
        max_attempts_per_target=5,
        allow_lossy=False,
        crop_transforms=A.NoOp(p=1.0),
        errors="skip",
        visualize=False,
        progress=False,
    ).export(
        destination=tmp_path / "resampled_virtual_geometry",
        visualize=False,
        progress=False,
    )

    assert len(exported._samples) == 1
    report = json.loads(
        (exported.location / "reports" / "tiling_skips.json").read_text()
    )
    assert report["skipped_candidates"] == 1
    assert report["items"][0]["mode"] == "coverage-virtual"
    assert report["items"][0]["details"]["result_geometry"]["type"] == "GeometryCollection"
    crop_report = json.loads(
        (exported.location / "reports" / "crop_augmentation.json").read_text()
    )
    assert crop_report["stats"]["rejected_geometry"] == 1


def test_virtual_background_filter_sees_final_transformed_crop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_yolo_dataset(
        tmp_path / "virtual_background_filter",
        task="detect",
        names=["fruit"],
        train_rows=["0 0.5 0.5 0.1 0.1"],
        val_rows=["0 0.5 0.5 0.1 0.1"],
        size=(200, 100),
    )
    train_image = next((source / "train" / "images").rglob("*.jpg"))
    pixels = Image.new("RGB", (200, 100), (0, 0, 0))
    pixels.paste((90, 130, 170), (100, 0, 200, 100))
    pixels.save(train_image, format="PNG")

    crops = iter([(0, 0, 50, 50), (150, 0, 200, 50)])
    monkeypatch.setattr(
        tiling_module,
        "_random_crop",
        lambda width, height, cfg, rng: next(crops),
    )

    exported = Dataset.open(source, task="detect", progress=False).tile(
        mode="coverage",
        splits=["train"],
        tile_size=50,
        large_image_threshold=20,
        scale_range=(1.0, 1.0),
        target_appearances_per_object=1,
        sparse_appearances_per_object=1,
        background_ratio=0.5,
        max_background_attempts_per_tile=3,
        crop_transforms=A.NoOp(p=1.0),
        background_filter=lambda candidate: candidate.getbbox() is not None,
        seed=9,
        visualize=False,
        progress=False,
    ).export(
        destination=tmp_path / "virtual_filtered_backgrounds",
        visualize=False,
        progress=False,
    )

    background = next(sample for sample in exported._samples if not sample.annotations)
    assert background.provenance["tile_mode"].startswith(
        "coverage-background-augmented"
    )
    assert background.provenance["background_filter_result"] == "accepted"
    report = json.loads(
        (exported.location / "reports" / "background_filter.json").read_text()
    )
    assert report["rejected_candidates"] == 1
    assert report["accepted_candidates"] == 1
    assert report["by_origin"][
        "coverage-virtual-populated_image_empty_space"
    ] == {
        "accepted": 1,
        "accepted_percentage": 50.0,
        "evaluated": 2,
        "rejected": 1,
        "rejected_percentage": 50.0,
    }


def test_coverage_crop_pipeline_can_change_canvas_and_augment_validation(
    tmp_path: Path,
) -> None:
    source = make_yolo_dataset(
        tmp_path / "resized_views",
        task="detect",
        names=["fruit"],
        train_rows=["0 0.5 0.5 0.2 0.2"],
        val_rows=["0 0.5 0.5 0.2 0.2"],
        size=(220, 180),
    )
    test_image = source / "test" / "images" / "test_0.jpg"
    test_image.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (220, 180), (90, 120, 150)).save(test_image)
    test_label = source / "test" / "labels" / "test_0.txt"
    test_label.parent.mkdir(parents=True, exist_ok=True)
    test_label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    data_yaml = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
    data_yaml["test"] = "test/images"
    (source / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8"
    )
    exported = Dataset.open(source, task="detect", progress=False).tile(
        mode="coverage",
        tile_size=80,
        scale_range=(1.0, 1.0),
        large_image_threshold=20,
        target_appearances_per_object=1,
        sparse_appearances_per_object=1,
        background_ratio=0,
        crop_transforms=A.Compose([A.Resize(240, 260), A.HorizontalFlip(p=1.0)]),
        augment_val=True,
        seed=11,
        visualize=False,
        progress=False,
    ).export(destination=tmp_path / "resized_tiles", visualize=False, progress=False)

    assert {sample.split for sample in exported._samples} == {"train", "val", "test"}
    transformed = [sample for sample in exported._samples if sample.split in {"train", "val"}]
    untouched_test = [sample for sample in exported._samples if sample.split == "test"]
    assert all(sample.provenance["tile_mode"] == "coverage-augmented" for sample in transformed)
    assert all(sample.provenance["transformed_view_size"] == [260, 240] for sample in transformed)
    assert all(sample.provenance["tile_mode"] == "coverage" for sample in untouched_test)
    assert all("crop_transform_seed" not in sample.provenance for sample in untouched_test)


def test_crop_pipeline_rejects_grid_and_allows_lossy_final_mask_crop(tmp_path: Path) -> None:
    source = make_yolo_dataset(
        tmp_path / "lossy_crop",
        task="segment",
        names=["school"],
        train_rows=["0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9"],
        val_rows=["0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9"],
        size=(220, 220),
    )
    dataset = Dataset.open(source, task="segment", progress=False)
    with pytest.raises(ValueError, match="coverage"):
        dataset.tile(mode="grid", crop_transforms=A.HorizontalFlip(p=1.0))

    exported = dataset.tile(
        mode="coverage",
        tile_size=80,
        scale_range=(1.0, 1.0),
        large_image_threshold=20,
        target_appearances_per_object=1,
        sparse_appearances_per_object=1,
        background_ratio=0,
        max_attempts_per_target=100,
        allow_lossy=True,
        crop_transforms=A.NoOp(p=1.0),
        visualize=False,
        progress=False,
    ).export(destination=tmp_path / "lossy_tiles", visualize=False, progress=False)
    polygon = exported._samples[0].annotations[0].polygon
    assert polygon is not None
    assert all(0 <= x <= 80 and 0 <= y <= 80 for x, y in polygon)


def test_virtual_camera_reports_insufficient_transformed_view_atomically(tmp_path: Path) -> None:
    source = make_yolo_dataset(
        tmp_path / "too_small_view",
        task="detect",
        names=["fruit"],
        train_rows=["0 0.5 0.5 0.2 0.2"],
        val_rows=["0 0.5 0.5 0.2 0.2"],
        size=(200, 200),
    )
    destination = tmp_path / "insufficient"
    plan = Dataset.open(source, task="detect", progress=False).tile(
        mode="coverage",
        splits=("train",),
        tile_size=80,
        scale_range=(1.0, 1.0),
        large_image_threshold=20,
        target_appearances_per_object=1,
        sparse_appearances_per_object=1,
        background_ratio=0,
        max_attempts_per_target=2,
        allow_lossy=True,
        crop_transforms=A.CenterCrop(height=40, width=40, p=1.0),
        visualize=False,
        progress=False,
    )

    with pytest.raises(DatasetValidationError, match="retry budget"):
        plan.export(destination=destination, visualize=False, progress=False)
    assert not destination.exists()


@pytest.mark.parametrize(
    ("task", "row", "extra"),
    [
        ("detect", "0 0.5 0.5 0.8 0.8", {}),
        (
            "pose",
            "0 0.5 0.5 0.8 0.8 0.5 0.5 2",
            {"kpt_shape": [1, 3], "flip_idx": [0], "kpt_names": {0: ["center"]}},
        ),
        ("polo", "0 70 0.5 0.5", {"radii": {0: 70}}),
    ],
)
def test_virtual_camera_lossy_crops_remain_representable(
    task: str,
    row: str,
    extra: dict,
    tmp_path: Path,
) -> None:
    source = make_yolo_dataset(
        tmp_path / f"lossy_{task}",
        task=task,
        names=["fruit"],
        train_rows=[row],
        val_rows=[row],
        size=(200, 200),
        extra=extra,
    )
    exported = Dataset.open(source, task=task, progress=False).tile(
        mode="coverage",
        splits=("train",),
        tile_size=80,
        scale_range=(1.0, 1.0),
        large_image_threshold=20,
        target_appearances_per_object=1,
        sparse_appearances_per_object=1,
        background_ratio=0,
        max_attempts_per_target=50,
        allow_lossy=True,
        crop_transforms=A.NoOp(p=1.0),
        seed=21,
        visualize=False,
        progress=False,
    ).export(
        destination=tmp_path / f"lossy_{task}_tiles",
        visualize=False,
        progress=False,
    )

    annotation = exported._samples[0].annotations[0]
    if task == "detect":
        assert annotation.bbox is not None
        assert all(0 <= value <= 80 for value in annotation.bbox)
    elif task == "pose":
        assert annotation.keypoints and annotation.keypoints[0][2] == 2
        assert all(0 <= value <= 80 for value in annotation.keypoints[0][:2])
    else:
        assert annotation.point is not None and annotation.radius is not None
        x, y = annotation.point
        assert 0 < annotation.radius <= min(x, y, 80 - x, 80 - y)
