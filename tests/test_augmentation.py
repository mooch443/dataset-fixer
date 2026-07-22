from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_fixer import Dataset, DatasetValidationError
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
