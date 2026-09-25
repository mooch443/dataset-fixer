from __future__ import annotations

import hashlib
import json
import shutil
import threading
from pathlib import Path

import pytest
import yaml
from PIL import Image, ImageOps

import dataset_fixer as df
from dataset_fixer.io import _image_size, _resolve_yolo_path
from conftest import make_image, make_yolo_dataset


@pytest.mark.parametrize("entry", [
    "train/images", "./train/images", "../train/images", "../../old/train/images",
    "/old/export/train/images", "C:\\old\\export\\train\\images", "old/train/images",
])
def test_relocated_split_paths_preserve_labels_and_source(detect_dataset, entry):
    path = detect_dataset / "data.yaml"
    values = yaml.safe_load(path.read_text())
    values["train"] = entry
    path.write_text(yaml.safe_dump(values))
    before = path.read_bytes()
    if entry in {"train/images", "./train/images"}:
        dataset = df.Dataset.open(path, progress=False)
        assert not dataset.warnings
    else:
        with pytest.warns(UserWarning, match="Dataset path fallback"):
            dataset = df.Dataset.open(path, progress=False)
        assert any(entry in message or repr(entry) in message for message in dataset.warnings)
    assert path.read_bytes() == before
    assert dataset.splits == ("train", "val")
    assert len(dataset._samples) == 6
    assert sum(len(sample.annotations) for sample in dataset._samples) == 7


def test_stale_root_and_validation_aliases(detect_dataset):
    path = detect_dataset / "data.yaml"
    values = yaml.safe_load(path.read_text())
    values.update(path="/missing/old/export", train="/old/train/images", val="../valid/images")
    path.write_text(yaml.safe_dump(values))
    with pytest.warns(UserWarning, match="Dataset path fallback") as caught:
        dataset = df.Dataset.open(path, progress=False)
    assert len(caught) == 3
    assert dataset.location == detect_dataset
    assert len(dataset._samples) == 6


def test_existing_external_path_wins_and_ambiguous_fallback_fails(tmp_path):
    root = tmp_path / "dataset"
    external = tmp_path / "external" / "train" / "images"
    external.mkdir(parents=True)
    (root / "train" / "images").mkdir(parents=True)
    warnings = []
    assert _resolve_yolo_path(str(external), root=root, yaml_dir=root, warnings=warnings) == external
    assert not warnings
    for split in ("val", "valid"):
        (root / split / "images").mkdir(parents=True)
    with pytest.raises(df.DatasetValidationError, match="Ambiguous dataset path fallback"):
        _resolve_yolo_path("/old/val/images", root=root, yaml_dir=root, warnings=warnings)


def test_image_lists_recover_relocated_absolute_paths(detect_dataset):
    path = detect_dataset / "data.yaml"
    values = yaml.safe_load(path.read_text())
    images = sorted((detect_dataset / "train" / "images").rglob("*.jpg"))
    listing = detect_dataset / "train.txt"
    listing.write_text("\n".join("/old/export/" + image.relative_to(detect_dataset).as_posix() for image in images))
    values["train"] = "train.txt"
    path.write_text(yaml.safe_dump(values))
    with pytest.warns(UserWarning, match="Dataset path fallback"):
        dataset = df.Dataset.open(path, progress=False)
    assert [s.image_path for s in dataset._samples if s.split == "train"] == images


def test_parallel_reads_overlap_and_preserve_order(detect_dataset, monkeypatch):
    sequential = df.Dataset.open(detect_dataset, workers=1, progress=False)
    barrier = threading.Barrier(2, timeout=5)
    threads = set()
    def read(path):
        threads.add(threading.get_ident())
        barrier.wait()
        return _image_size(path)
    monkeypatch.setattr("dataset_fixer.io._image_size", read)
    parallel = df.Dataset.open(detect_dataset, workers=2, progress=False)
    assert len(threads) == 2
    assert parallel._samples == sequential._samples
    assert parallel.warnings == sequential.warnings


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_dimensions_match_transposed_pixels(tmp_path, orientation):
    root = make_yolo_dataset(tmp_path / "oriented", task="detect", size=(96, 64))
    path = next((root / "train" / "images").rglob("*.jpg"))
    exif = Image.Exif()
    exif[274] = orientation
    Image.new("RGB", (96, 64)).save(path, exif=exif)
    with Image.open(path) as original:
        expected = ImageOps.exif_transpose(original).size
    assert _image_size(path) == expected
    dataset = df.Dataset.open(root, workers=2, progress=False)
    sample = next(sample for sample in dataset._samples if sample.image_path == path)
    assert (sample.width, sample.height) == expected


def test_parallel_loading_still_rejects_truncated_pixels(detect_dataset):
    image = next((detect_dataset / "train" / "images").rglob("*.jpg"))
    image.write_bytes(image.read_bytes()[:-100])
    with pytest.raises(df.DatasetValidationError, match="Unreadable image"):
        df.Dataset.open(detect_dataset, workers=4, progress=False)
    one = df.Dataset.open(detect_dataset, workers=1, errors="skip", progress=False)
    four = df.Dataset.open(detect_dataset, workers=4, errors="skip", progress=False)
    assert one.warnings == four.warnings
    assert one._samples == four._samples


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_reader_count_fails_before_loading(workers):
    with pytest.raises(ValueError, match="workers"):
        df.Dataset.open("/not/read", workers=workers)


@pytest.fixture
def duplicates(tmp_path):
    root = tmp_path / "duplicates"
    # The YAML intentionally lists test first. Priority must not depend on this.
    for split in ("test", "val", "train"):
        for name, color in [("all", (80, 30, 20)), ("own", (20, 30, {"train": 40, "val": 90, "test": 150}[split]))]:
            make_image(root / split / "images" / f"{name}.jpg", color=color)
        if split != "train":
            make_image(root / split / "images" / "evaluation.jpg", color=(10, 100, 10))
        for image in (root / split / "images").glob("*.jpg"):
            label = root / split / "labels" / (image.stem + ".txt")
            label.parent.mkdir(exist_ok=True)
            label.write_text("0 .5 .5 .5 .5\n")
    (root / "data.yaml").write_text(yaml.safe_dump({
        "test": "test/images", "val": "val/images", "train": "train/images", "names": ["object"],
    }, sort_keys=False))
    return root


@pytest.mark.parametrize("priority,all_split,evaluation_split", [
    (("train", "val", "test"), "train", "val"),
    (("train", "test", "valid"), "train", "test"),
    (("test", "val", "train"), "test", "test"),
    (("val", "test", "train"), "val", "val"),
])
def test_duplicate_hierarchy_covers_groups_without_train(duplicates, priority, all_split, evaluation_split):
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in duplicates.rglob("*") if p.is_file()}
    dataset = df.Dataset.open(duplicates, duplicate_splits=priority, progress=False)
    assert {s.split for s in dataset._samples if s.image_path.stem == "all"} == {all_split}
    assert {s.split for s in dataset._samples if s.image_path.stem == "evaluation"} == {evaluation_split}
    assert len(dataset._samples) == 5 and dataset.splits == ("train", "val", "test")
    assert [s.split for s in dataset._samples] == sorted([s.split for s in dataset._samples], key=("test", "val", "train").index)
    assert dataset.validation_audit["fixed_count"] == 3
    assert dataset.validation_audit["counts_by_category"] == {"Resolved cross-split duplicate": 3}
    assert all("kept " in w and "excluded " in w and "priority " in w for w in dataset.warnings)
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in duplicates.rglob("*") if p.is_file()}
    shutil.copyfile(dataset.validation_audit["visualization"], "/tmp/dataset-fixer-duplicate-audit.png")


def test_duplicates_remain_errors_without_explicit_policy(duplicates):
    with pytest.raises(df.DatasetValidationError, match="Byte-identical"):
        df.Dataset.open(duplicates, deep=True, progress=False)
    assert len(df.Dataset.open(duplicates, progress=False)._samples) == 8


@pytest.mark.parametrize("priority", ["train", ("train", "val"), ("train", "valid", "val"), {"train", "val", "test"}, True])
def test_invalid_duplicate_hierarchy_fails_before_loading(priority):
    with pytest.raises(ValueError, match="duplicate_splits"):
        df.Dataset.open("/not/read", duplicate_splits=priority)


def test_duplicate_policy_does_not_skip_other_errors(duplicates):
    label = duplicates / "val" / "labels" / "own.txt"
    label.write_text("0 .5 .5 3 3\n")
    with pytest.raises(df.DatasetValidationError, match="outside"):
        df.Dataset.open(duplicates, duplicate_splits=("train", "val", "test"), progress=False)
    label.write_text("0 .5 .5 .5 .5\n")
    (duplicates / "train" / "labels" / "orphan.txt").write_text("0 .5 .5 .5 .5\n")
    with pytest.raises(df.DatasetValidationError, match="no image"):
        df.Dataset.open(duplicates, duplicate_splits=("train", "val", "test"), progress=False)


def test_resolved_duplicates_stay_excluded_in_export_and_training(duplicates, tmp_path, monkeypatch):
    from dataset_fixer.convert import Kind, prepare
    monkeypatch.setattr("dataset_fixer.convert.cache_root", lambda: tmp_path / "preparation-cache")
    dataset = df.Dataset.open(duplicates, duplicate_splits=("train", "val", "test"), progress=False)
    exported = dataset.export(destination=tmp_path / "export", visualize=False, progress=False)
    assert len(exported._samples) == 5
    assert exported.validation_audit["fixed_count"] == 3
    assert exported.validation_audit["duplicate_splits"] == ["train", "val", "test"]
    prepared = prepare(dataset, Kind.YOLO, workers=2, progress=False)
    assert len(df.Dataset.open(prepared.data_yaml, deep=True, progress=False)._samples) == 5


def test_semantic_duplicate_policy_prunes_masks_and_statistics(duplicates, tmp_path):
    # Binary semantic masks are also filtered by image content, never by mask content.
    for label in duplicates.rglob("labels/*.txt"):
        label.write_text("0 .2 .2 .8 .2 .8 .8 .2 .8\n")
    source = df.Dataset.open(duplicates, progress=False)
    masks = source.export(destination=tmp_path / "masks", format="semantic_masks", visualize=False, progress=False)
    dataset = df.Dataset.open(masks.location, duplicate_splits=("train", "val", "test"), progress=False)
    assert len(dataset._samples) == len(dataset._mask_paths) == len(dataset._mask_statistics) == 5
    assert dataset.validation_audit["fixed_count"] == 3


def test_coco_hierarchy_handles_one_physical_image_in_multiple_splits(tmp_path):
    root = tmp_path / "coco"
    make_image(root / "images" / "shared.jpg")
    for split in ("test", "val", "train"):
        make_image(root / "images" / f"{split}.jpg", color=({"train": 20, "val": 70, "test": 120}[split], 40, 20))
        names = [f"{split}.jpg"] + (["shared.jpg"] if split != "train" else [])
        (root / f"instances_{split}.json").write_text(json.dumps({
            "categories": [{"id": 1, "name": "object"}],
            "images": [{"id": i, "file_name": f"images/{name}", "width": 160, "height": 120} for i, name in enumerate(names)],
            "annotations": [{"id": i, "image_id": i, "category_id": 1, "bbox": [40, 30, 80, 60]} for i in range(len(names))],
        }))
    dataset = df.Dataset.open(root, duplicate_splits=("train", "val", "test"), progress=False)
    assert dataset.format == "coco"
    assert {s.split for s in dataset._samples if s.image_path.name == "shared.jpg"} == {"val"}
    assert len(dataset._samples) == 4
    assert dataset.validation_audit["fixed_count"] == 1


def test_duplicates_within_winning_split_are_retained(duplicates):
    source = duplicates / "train" / "images" / "all.jpg"
    shutil.copyfile(source, source.with_name("second.jpg"))
    label = duplicates / "train" / "labels" / "all.txt"
    shutil.copyfile(label, label.with_name("second.txt"))
    dataset = df.Dataset.open(duplicates, duplicate_splits=("train", "val", "test"), progress=False)
    assert {s.image_path.stem for s in dataset._samples if s.split == "train"} == {"all", "second", "own"}
    assert dataset.validation_audit["fixed_count"] == 3
