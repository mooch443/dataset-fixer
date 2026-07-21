from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dataset_fixer import Dataset, DatasetValidationError, Task


def test_open_identity_and_automatic_validation(detect_dataset: Path) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    assert dataset.name == "orchard"
    assert dataset.location == detect_dataset.resolve()
    assert dataset.data_yaml == (detect_dataset / "data.yaml").resolve()
    assert dataset.task is Task.DETECT
    assert dataset.splits == ("train", "val")
    assert dataset.classes == {0: "fruit", 1: "damaged"}
    assert dataset.training_ready


def test_invalid_label_fails_at_open(detect_dataset: Path) -> None:
    label = next((detect_dataset / "labels").rglob("*.txt"))
    label.write_text("0 nan 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="non-finite"):
        Dataset.open(detect_dataset, task="detect", progress=False)


def test_orphan_label_fails_at_open(detect_dataset: Path) -> None:
    orphan = detect_dataset / "labels" / "train" / "orphan.txt"
    orphan.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="no image"):
        Dataset.open(detect_dataset, task="detect", progress=False)


def test_split_grouping_manifest_and_provenance(detect_dataset: Path, tmp_path: Path) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    result = dataset.split(
        {"train": 0.5, "val": 0.5},
        destination=tmp_path / "resplit",
        group_by=lambda path: path.parent.name,
        seed=7,
        visualize=False,
        progress=False,
    )
    group_targets: dict[str, set[str]] = {}
    for sample in result._samples:
        group_targets.setdefault(sample.relative_path.parent.name, set()).add(sample.split)
    assert all(len(splits) == 1 for splits in group_targets.values())
    assert result.training_ready
    assert len(result.provenance) == len(result._samples)
    manifest = json.loads((result.location / "dataset-fixer.json").read_text(encoding="utf-8"))
    assert manifest["settings"]["seed"] == 7
    assert manifest["environment"]["dataset_fixer_git"]["commit"]
    assert manifest["dataset_fixer"]["version"]
    assert manifest["source_dataset"]["fingerprint"]
    assert manifest["validation"]["passed"] is True
    assert manifest["timing"]["started_at"] and manifest["timing"]["finished_at"]
    assert manifest["settings_fingerprint"] in result.location.name or manifest["settings_fingerprint"]
    assert all(record["transformation_chain"] for record in result.provenance.values())
    assert yaml.safe_load(result.data_yaml.read_text(encoding="utf-8"))["path"] == str(result.location)


def test_remove_classes_compacts_and_chains_original(detect_dataset: Path, tmp_path: Path) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    split = dataset.split(
        {"train": 0.5, "val": 0.5},
        destination=tmp_path / "split",
        visualize=False,
        progress=False,
    )
    clean = split.remove_classes(
        ["fruit"],
        destination=tmp_path / "clean",
        visualize=False,
        progress=False,
    )
    assert clean.classes == {0: "damaged"}
    assert all(annotation.class_id == 0 for sample in clean._samples for annotation in sample.annotations)
    records = list(clean.provenance.values())
    assert records
    assert all(record["original_dataset"] == "orchard" for record in records)
    assert all("class_mapping" in record for record in records)


def test_grid_tile_geometry_and_source_immutability(detect_dataset: Path, tmp_path: Path) -> None:
    before = {path: path.read_bytes() for path in detect_dataset.rglob("*") if path.is_file()}
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    tiled = dataset.tile(
        mode="grid",
        destination=tmp_path / "tiles",
        tile_size=80,
        overlap=0.25,
        visualize=False,
        progress=False,
    )
    assert len(tiled._samples) > len(dataset._samples)
    assert all(sample.width <= 80 and sample.height <= 80 for sample in tiled._samples)
    assert {path: path.read_bytes() for path in detect_dataset.rglob("*") if path.is_file()} == before
    assert all("crop" in record for record in tiled.provenance.values())


def test_existing_destination_is_rejected(detect_dataset: Path, tmp_path: Path) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        dataset.export(destination=destination, visualize=False, progress=False)


def test_dry_run_does_not_publish(detect_dataset: Path, tmp_path: Path) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    destination = tmp_path / "dry"
    returned = dataset.tile(
        destination=destination,
        tile_size=80,
        dry_run=True,
        visualize=False,
        progress=False,
    )
    assert returned is dataset
    assert not destination.exists()


def test_conflicting_group_assignments_fail_before_writes(detect_dataset: Path, tmp_path: Path) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)

    def assign(path: Path) -> str:
        return "train" if path.name.endswith("0.jpg") else "val"

    destination = tmp_path / "conflict"
    with pytest.raises(DatasetValidationError, match="conflicting"):
        dataset.split(
            {"train": 0.5, "val": 0.5},
            destination=destination,
            group_by=lambda path: path.parent.name,
            assign=assign,
            visualize=False,
            progress=False,
        )
    assert not destination.exists()
