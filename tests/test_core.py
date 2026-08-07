from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from dataset_fixer import Dataset, DatasetValidationError, Task
from dataset_fixer.utils import settings_fingerprint, to_jsonable


def test_open_identity_and_automatic_validation(detect_dataset: Path) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    assert dataset.name == "orchard"
    assert dataset.location == detect_dataset.resolve()
    assert dataset.data_yaml == (detect_dataset / "data.yaml").resolve()
    assert dataset.task is Task.DETECT
    assert dataset.splits == ("train", "val")
    assert dataset.classes == {0: "fruit", 1: "damaged"}
    assert dataset.validation_audit["status"] == "passed"
    assert dataset.validation_audit["skipped_count"] == 0
    assert dataset.validation_audit["visualization"] is None
    assert dataset.training_ready


def test_dataset_string_reports_file_derived_statistics(detect_dataset: Path) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)

    summary = str(dataset)

    assert "Dataset 'orchard' [detect; materialized; yolo]" in summary
    assert "images: 6 | annotations: 7 | empty: 0 (0.0%)" in summary
    assert "classes: 2 | splits: 2" in summary
    assert "image size: 160x120" in summary
    assert "Split statistics" in summary
    assert "split  images  annotated  empty  annotations" in summary
    assert "train       4          4      0            5" in summary
    assert "val         2          2      0            2" in summary
    assert "total       6          6      0            7" in summary
    assert "Class statistics (annotation instances and images containing class)" in summary
    assert "fruit              4       4    66.7%" in summary
    assert "damaged            3       3    50.0%" in summary
    assert "validation: passed | warnings: 0" in summary


def test_streaming_settings_fingerprint_preserves_canonical_value(tmp_path: Path) -> None:
    value = {
        "path": tmp_path / "dataset",
        "nested": {2: [Task.SEGMENT, (1.25, None, True)]},
        "text": "reef ü",
    }
    reference = hashlib.sha256(
        json.dumps(
            to_jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:8]
    assert settings_fingerprint(value) == reference


def test_invalid_label_fails_at_open(detect_dataset: Path) -> None:
    label = next(path for path in detect_dataset.rglob("*.txt") if "labels" in path.parts)
    label.write_text("0 nan 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="non-finite"):
        Dataset.open(detect_dataset, task="detect", progress=False)


def test_malformed_label_can_be_skipped_at_open(detect_dataset: Path) -> None:
    label = next(path for path in detect_dataset.rglob("*.txt") if "labels" in path.parts)
    labels = [path for path in detect_dataset.rglob("*.txt") if "labels" in path.parts]
    original_count = sum(len(path.read_text(encoding="utf-8").splitlines()) for path in labels)
    replaced_count = len(label.read_text(encoding="utf-8").splitlines())
    label.write_text("0 0.5 0.5 0.2 0.2\n0 nan 0.5 0.2 0.2\n", encoding="utf-8")

    dataset = Dataset.open(
        detect_dataset,
        task="detect",
        errors="skip",
        progress=False,
    )

    assert any("non-finite" in warning for warning in dataset.warnings)
    assert sum(len(sample.annotations) for sample in dataset._samples) == original_count - replaced_count + 1


def test_orphan_label_fails_at_open(detect_dataset: Path) -> None:
    orphan = detect_dataset / "train" / "labels" / "orphan.txt"
    orphan.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="no image"):
        Dataset.open(detect_dataset, task="detect", progress=False)


def test_errors_skip_applies_across_recoverable_load_errors(detect_dataset: Path) -> None:
    orphan = detect_dataset / "train" / "labels" / "orphan.txt"
    orphan.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    broken_image = next((detect_dataset / "train" / "images").rglob("*.jpg"))
    broken_image.write_bytes(b"not an image")
    data = yaml.safe_load((detect_dataset / "data.yaml").read_text(encoding="utf-8"))
    data["test"] = "missing/images"
    (detect_dataset / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    dataset = Dataset.open(detect_dataset, task="detect", errors="skip", progress=False)

    assert len(dataset._samples) == 5
    assert any("Skipped invalid split entry" in warning for warning in dataset.warnings)
    assert any("Skipped invalid image" in warning for warning in dataset.warnings)
    assert any("Ignored orphan label" in warning for warning in dataset.warnings)


def test_split_grouping_manifest_and_provenance(detect_dataset: Path, tmp_path: Path) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    planned = dataset.split(
        {"train": 0.5, "val": 0.5},
        group_by=lambda path: path.parent.name,
        seed=7,
        visualize=False,
        progress=False,
    )
    assert planned.data_yaml is None
    assert not (tmp_path / "resplit").exists()
    result = planned.export(destination=tmp_path / "resplit", visualize=False, progress=False)
    group_targets: dict[str, set[str]] = {}
    for sample in result._samples:
        group_targets.setdefault(sample.relative_path.parent.name, set()).add(sample.split)
    assert all(len(splits) == 1 for splits in group_targets.values())
    assert result.training_ready
    assert len(result.provenance) == len(result._samples)
    manifest = json.loads((result.location / "dataset-fixer.json").read_text(encoding="utf-8"))
    assert manifest["history"][0]["settings"]["seed"] == 7
    assert manifest["environment"]["dataset_fixer_git"]["commit"]
    assert manifest["dataset_fixer"]["version"]
    assert manifest["source_dataset"]["fingerprint"]
    assert manifest["validation"]["passed"] is True
    assert manifest["timing"]["started_at"] and manifest["timing"]["finished_at"]
    assert manifest["settings_fingerprint"] in result.location.name or manifest["settings_fingerprint"]
    assert all(record["transformation_chain"] for record in result.provenance.values())
    assert yaml.safe_load(result.data_yaml.read_text(encoding="utf-8"))["path"] == str(result.location)


def test_group_aware_export_writes_aggregate_split_audit(
    detect_dataset: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = Dataset.open(detect_dataset, task="detect", progress=False)
    result = source.split(
        {"train": 0.5, "val": 0.5},
        group_by=lambda path: path.parent.name,
        seed=7,
        visualize=False,
        progress=False,
    ).export(
        destination=tmp_path / "audited",
        visualize=False,
        progress=False,
    )

    report = json.loads(
        (result.location / "reports" / "split_group_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "passed"
    assert report["scope"] == "all_current_source_splits"
    assert report["total"] == {"images": 6, "distinct_groups": 3}
    assert report["overlap_count"] == 0
    assert set(report["splits"]) == {"train", "val"}
    assert "groups" not in report
    for details in report["splits"].values():
        histogram = details["group_size"]["histogram"]
        assert sum(histogram.values()) == details["distinct_groups"]
        assert sum(int(size) * count for size, count in histogram.items()) == details["images"]

    manifest = json.loads(
        (result.location / "dataset-fixer.json").read_text(encoding="utf-8")
    )
    validation = manifest["validation"]["split_group_isolation"]
    assert validation["status"] == "passed"
    assert validation["report"] == "reports/split_group_audit.json"
    assert validation["distinct_groups"] == 3
    assert "Split-group audit: passed" in capsys.readouterr().out


def test_group_audit_fails_on_cross_split_overlap_even_for_subset_export(
    detect_dataset: Path,
    tmp_path: Path,
) -> None:
    grouped = Dataset.open(detect_dataset, task="detect", progress=False).split(
        {"train": 0.5, "val": 0.5},
        group_by=lambda path: path.parent.name,
        seed=7,
        visualize=False,
        progress=False,
    ).export(
        destination=tmp_path / "grouped-overlap-source",
        visualize=False,
        progress=False,
    )
    train = next(sample for sample in grouped._samples if sample.split == "train")
    val = next(sample for sample in grouped._samples if sample.split == "val")
    val.provenance["split_group"] = train.provenance["split_group"]
    destination = tmp_path / "overlap-output"

    with pytest.raises(DatasetValidationError, match="appears in multiple") as exc_info:
        grouped.export(
            destination=destination,
            splits=("train",),
            visualize=False,
            progress=False,
        )

    message = str(exc_info.value)
    assert "train" in message and "val" in message
    assert not destination.exists()


def test_group_audit_fails_when_group_identity_is_incomplete(
    detect_dataset: Path,
    tmp_path: Path,
) -> None:
    grouped = Dataset.open(detect_dataset, task="detect", progress=False).split(
        {"train": 0.5, "val": 0.5},
        group_by=lambda path: path.parent.name,
        seed=7,
        visualize=False,
        progress=False,
    ).export(
        destination=tmp_path / "grouped-incomplete-source",
        visualize=False,
        progress=False,
    )
    grouped._samples[0].provenance.pop("split_group")
    destination = tmp_path / "incomplete-output"

    with pytest.raises(DatasetValidationError, match="unverifiable"):
        grouped.export(
            destination=destination,
            visualize=False,
            progress=False,
        )
    assert not destination.exists()


def test_tiling_preserves_groups_and_audits_current_output_population(
    detect_dataset: Path,
    tmp_path: Path,
) -> None:
    source = Dataset.open(detect_dataset, task="detect", progress=False)
    result = source.split(
        {"train": 0.5, "val": 0.5},
        group_by=lambda path: path.parent.name,
        seed=7,
        visualize=False,
        progress=False,
    ).tile(
        tile_size=80,
        overlap=0,
        negative_tiles="all",
        visualize=False,
        progress=False,
    ).export(
        destination=tmp_path / "grouped-tiles",
        visualize=False,
        progress=False,
    )

    report = json.loads(
        (result.location / "reports" / "split_group_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["total"]["images"] == len(result._samples)
    assert report["total"]["images"] > len(source._samples)
    assert report["total"]["distinct_groups"] == 3
    assert all("split_group" in sample.provenance for sample in result._samples)


def test_latest_ungrouped_split_removes_inherited_group_audit(
    detect_dataset: Path,
    tmp_path: Path,
) -> None:
    grouped = Dataset.open(detect_dataset, task="detect", progress=False).split(
        {"train": 0.5, "val": 0.5},
        group_by=lambda path: path.parent.name,
        seed=7,
        visualize=False,
        progress=False,
    ).export(
        destination=tmp_path / "first-grouped",
        visualize=False,
        progress=False,
    )
    assert (grouped.location / "reports" / "split_group_audit.json").is_file()

    result = grouped.split(
        {"train": 0.5, "val": 0.5},
        seed=11,
        visualize=False,
        progress=False,
    ).export(
        destination=tmp_path / "then-ungrouped",
        visualize=False,
        progress=False,
    )

    assert not (result.location / "reports" / "split_group_audit.json").exists()
    manifest = json.loads(
        (result.location / "dataset-fixer.json").read_text(encoding="utf-8")
    )
    validation = manifest["validation"]["split_group_isolation"]
    assert validation["status"] == "not_applicable"
    assert validation["report"] is None
    assert "latest split operation did not use group_by" in validation["reason"]


def test_export_without_group_aware_history_marks_audit_not_applicable(
    detect_dataset: Path,
    tmp_path: Path,
) -> None:
    result = Dataset.open(detect_dataset, task="detect", progress=False).export(
        destination=tmp_path / "ordinary-export",
        visualize=False,
        progress=False,
    )

    assert not (result.location / "reports" / "split_group_audit.json").exists()
    manifest = json.loads(
        (result.location / "dataset-fixer.json").read_text(encoding="utf-8")
    )
    assert manifest["validation"]["split_group_isolation"]["status"] == "not_applicable"


def test_split_ratios_are_normalized_weights(detect_dataset: Path) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)

    planned = dataset.split(
        {"train": 80, "val": 20},
        visualize=False,
        progress=False,
    )

    assert planned.history[-1]["settings"]["ratios"] == {
        "train": pytest.approx(0.8),
        "val": pytest.approx(0.2),
    }
    assert len(planned._samples) == len(dataset._samples)


def test_remove_classes_compacts_and_chains_original(detect_dataset: Path, tmp_path: Path) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    split = dataset.split(
        {"train": 0.5, "val": 0.5},
        visualize=False,
        progress=False,
    )
    clean = split.remove_classes(
        ["fruit"],
        visualize=True,
        progress=False,
    )
    assert clean.classes == {0: "damaged"}
    assert clean.data_yaml is None
    assert not (tmp_path / "split").exists()
    clean = clean.export(destination=tmp_path / "clean", visualize=False, progress=False)
    assert clean.classes == {0: "damaged"}
    assert all(annotation.class_id == 0 for sample in clean._samples for annotation in sample.annotations)
    records = list(clean.provenance.values())
    assert records
    assert all(record["original_dataset"] == "orchard" for record in records)
    assert all("class_mapping" in record for record in records)
    counts = json.loads((clean.location / "reports" / "class_counts.json").read_text(encoding="utf-8"))
    assert counts["before"]["background"] == 0
    assert counts["after"]["background"] == 3
    assert counts["names_before"]["background"] == "background"
    assert counts["names_after"]["background"] == "background"
    assert (clean.location / "reports" / "class_counts.jpg").is_file()


@pytest.mark.parametrize(
    ("removed", "merge_into", "expected_name"),
    [
        (["damaged"], "fruit", "fruit"),
        ([0], 1, "damaged"),
    ],
)
def test_remove_classes_can_merge_annotations_into_surviving_class(
    detect_dataset: Path,
    tmp_path: Path,
    removed: list[str | int],
    merge_into: str | int,
    expected_name: str,
) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    original_annotation_count = sum(len(sample.annotations) for sample in dataset._samples)

    planned = dataset.remove_classes(
        removed,
        merge_into=merge_into,
        visualize=False,
        progress=False,
    )

    assert planned.classes == {0: expected_name}
    assert sum(len(sample.annotations) for sample in planned._samples) == original_annotation_count
    assert {annotation.class_id for sample in planned._samples for annotation in sample.annotations} == {0}
    assert planned.history[-1]["settings"]["merge_into"] == {
        "selector": merge_into,
        "output_class_id": 0,
        "output_class_name": expected_name,
    }

    exported = planned.export(
        destination=tmp_path / f"merged-{expected_name}",
        visualize=False,
        progress=False,
    )

    assert exported.classes == {0: expected_name}
    assert sum(len(sample.annotations) for sample in exported._samples) == original_annotation_count
    counts = json.loads(
        (exported.location / "reports" / "class_counts.json").read_text(encoding="utf-8")
    )
    assert counts["after"]["0"] == original_annotation_count
    assert counts["after"]["background"] == 0
    assert all(
        record["class_mapping"] == {"0": 0, "1": 0}
        for record in exported.provenance.values()
    )


def test_remove_classes_rejects_removed_or_unknown_merge_target(detect_dataset: Path) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)

    with pytest.raises(ValueError, match="not being removed"):
        dataset.remove_classes(["damaged"], merge_into="damaged", visualize=False)
    with pytest.raises(ValueError, match="Unknown merge target class ID"):
        dataset.remove_classes(["damaged"], merge_into=99, visualize=False)


def test_rename_classes_is_virtual_validated_and_exported(
    detect_dataset: Path,
    tmp_path: Path,
) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    original_ids = [
        annotation.class_id
        for sample in dataset._samples
        for annotation in sample.annotations
    ]

    planned = dataset.rename_classes(
        {0: "apple", "damaged": "blemished"},
        progress=False,
    )

    assert dataset.classes == {0: "fruit", 1: "damaged"}
    assert planned.classes == {0: "apple", 1: "blemished"}
    assert planned.history[-1]["operation"] == "rename-classes"
    assert [
        annotation.class_id
        for sample in planned._samples
        for annotation in sample.annotations
    ] == original_ids
    assert planned.remove_classes(["blemished"], visualize=False).classes == {0: "apple"}
    with pytest.raises(ValueError, match="duplicate class names"):
        dataset.rename_classes({"fruit": "damaged"})

    exported = planned.export(
        destination=tmp_path / "renamed",
        visualize=True,
        progress=False,
    )

    assert exported.classes == {0: "apple", 1: "blemished"}
    assert [
        annotation.class_id
        for sample in exported._samples
        for annotation in sample.annotations
    ] == original_ids
    assert (exported.location / "reports" / "class_renames.json").is_file()
    assert not (exported.location / "reports" / "rename_classes_summary.jpg").exists()
    assert all("class_renames" in record for record in exported.provenance.values())
    data = yaml.safe_load(exported.data_yaml.read_text(encoding="utf-8"))
    assert data["names"] == {0: "apple", 1: "blemished"}


def test_virtual_pipeline_repr_and_empty_image_rebalancing(
    tmp_path: Path,
) -> None:
    from conftest import make_yolo_dataset

    source = make_yolo_dataset(
        tmp_path / "empty_source",
        task="detect",
        train_rows=[
            "0 0.5 0.5 0.2 0.2",
            "0 0.5 0.5 0.2 0.2",
            "0 0.5 0.5 0.2 0.2",
            "0 0.5 0.5 0.2 0.2",
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        val_rows=["0 0.5 0.5 0.2 0.2", ""],
    )
    dataset = Dataset.open(source, task="detect", progress=False)
    planned = dataset.remove_classes(["damaged"], visualize=False).rebalance_empty(
        0.25, splits=("train",), seed=9, visualize=False
    )
    assert planned.data_yaml is None
    assert not planned.training_ready
    assert "classes={0: 'fruit'}" in repr(planned)
    assert "remove-classes → rebalance-empty" in str(planned)
    assert "empty: 2 (28.6%)" in str(planned)
    assert not (tmp_path / "balanced").exists()

    exported = planned.export(destination=tmp_path / "balanced", visualize=False, progress=False)
    train = [sample for sample in exported._samples if sample.split == "train"]
    val = [sample for sample in exported._samples if sample.split == "val"]
    assert sum(not sample.annotations for sample in train) == 1
    assert sum(not sample.annotations for sample in val) == 1
    assert (exported.location / "train" / "images").is_dir()
    assert (exported.location / "train" / "labels").is_dir()
    assert not (exported.location / "images").exists()
    assert yaml.safe_load(exported.data_yaml.read_text(encoding="utf-8"))["train"] == "train/images"


def test_empty_balance_report_is_a_distribution_plot(tmp_path: Path) -> None:
    from conftest import make_yolo_dataset

    source = make_yolo_dataset(
        tmp_path / "empty_report_source",
        task="detect",
        train_rows=["0 0.5 0.5 0.2 0.2", "", "", ""],
        val_rows=["0 0.5 0.5 0.2 0.2", ""],
    )
    exported = (
        Dataset.open(source, task="detect", progress=False)
        .rebalance_empty(0.5, splits=("train",), seed=42, visualize=True)
        .export(destination=tmp_path / "empty_report", visualize=False, progress=False)
    )
    report = json.loads(
        (exported.location / "reports" / "empty_image_balance.json").read_text(encoding="utf-8")
    )
    assert report["train"]["before"] == {"annotated": 1, "background": 3}
    assert report["train"]["after"] == {"annotated": 1, "background": 1}
    assert report["val"]["before"] == {"annotated": 1, "background": 1}
    assert (exported.location / "reports" / "empty_image_balance.jpg").is_file()


def test_empty_rebalancing_after_deferred_tiling(tmp_path: Path) -> None:
    from conftest import make_yolo_dataset

    source = make_yolo_dataset(
        tmp_path / "tile_empty_source",
        task="detect",
        train_rows=["0 0.1 0.1 0.08 0.08"],
        val_rows=["0 0.1 0.1 0.08 0.08"],
        size=(160, 120),
    )
    plan = (
        Dataset.open(source, task="detect", progress=False)
        .tile(tile_size=40, overlap=0, negative_tiles="all", visualize=False)
        .rebalance_empty(0.5, splits=("train",), seed=42, visualize=False)
    )
    assert "images: pending export" in str(plan)
    exported = plan.export(destination=tmp_path / "tile_balanced", visualize=False, progress=False)
    train = [sample for sample in exported._samples if sample.split == "train"]
    assert sum(not sample.annotations for sample in train) / len(train) <= 0.5


def test_export_announces_prepublication_and_final_validation_progress(
    detect_dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = Dataset.open(detect_dataset, task="detect", progress=False).remove_classes(
        ["damaged"], visualize=False
    )
    plan.export(destination=tmp_path / "progress", visualize=False, progress=True)
    output = capsys.readouterr().out
    assert "Validating complete staged output before atomic publication" in output
    assert "published dataset is not rescanned" in output


def test_multi_step_export_streams_validation_without_dataset_rescan(
    detect_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    plan = dataset.remove_classes(["damaged"], visualize=False).rebalance_empty(
        0.5, splits=("train",), visualize=False
    )
    def unexpected_open(cls, location, **kwargs):
        raise AssertionError(f"staged validation unexpectedly reopened {location}")

    monkeypatch.setattr(Dataset, "open", classmethod(unexpected_open))
    exported = plan.export(
        destination=tmp_path / "streaming-validation",
        visualize=False,
        progress=False,
    )
    assert exported.training_ready


def test_streaming_staged_validation_rejects_corrupt_labels_atomically(
    detect_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataset_fixer.writer import OutputBuilder

    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    destination = tmp_path / "corrupt-staged-label"
    original_write_reports = OutputBuilder.write_reports

    def write_reports_then_corrupt(self, **kwargs):
        manifest = original_write_reports(self, **kwargs)
        label = next(path for path in self.staging.rglob("*.txt") if "labels" in path.parts)
        label.write_text("0 nan 0.5 0.2 0.2\n", encoding="utf-8")
        return manifest

    monkeypatch.setattr(OutputBuilder, "write_reports", write_reports_then_corrupt)
    with pytest.raises(DatasetValidationError, match="non-finite"):
        dataset.export(destination=destination, visualize=False, progress=False)
    assert not destination.exists()


def test_streaming_staged_validation_rejects_corrupt_provenance_atomically(
    detect_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataset_fixer.writer import OutputBuilder

    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    destination = tmp_path / "corrupt-staged-provenance"
    original_write_reports = OutputBuilder.write_reports

    def write_reports_then_corrupt(self, **kwargs):
        manifest = original_write_reports(self, **kwargs)
        (self.staging / "provenance.jsonl").write_text("{not-json}\n", encoding="utf-8")
        return manifest

    monkeypatch.setattr(OutputBuilder, "write_reports", write_reports_then_corrupt)
    with pytest.raises(DatasetValidationError, match="Invalid provenance record"):
        dataset.export(destination=destination, visualize=False, progress=False)
    assert not destination.exists()


def test_loaded_samples_share_provenance_records_without_per_sample_copies(
    detect_dataset: Path,
    tmp_path: Path,
) -> None:
    exported = Dataset.open(detect_dataset, task="detect", progress=False).export(
        destination=tmp_path / "shared-provenance",
        visualize=False,
        progress=False,
    )
    reopened = Dataset.open(exported.location, task="detect", progress=False)

    for sample in reopened._samples:
        key = str(sample.image_path.relative_to(reopened.location))
        assert sample.provenance is reopened._provenance[key]


def test_grid_tile_geometry_and_source_immutability(detect_dataset: Path, tmp_path: Path) -> None:
    before = {path: path.read_bytes() for path in detect_dataset.rglob("*") if path.is_file()}
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    planned = dataset.tile(
        mode="grid",
        tile_size=80,
        overlap=0.25,
        visualize=False,
        progress=False,
    )
    assert not (tmp_path / "tiles").exists()
    tiled = planned.export(destination=tmp_path / "tiles", visualize=False, progress=False)
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
        tile_size=80,
        visualize=False,
        progress=False,
    )
    dry = returned.export(destination=destination, dry_run=True, visualize=False, progress=False)
    assert dry is returned
    assert not destination.exists()


def test_conflicting_group_assignments_fail_before_writes(detect_dataset: Path, tmp_path: Path) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)

    def assign(path: Path) -> str:
        return "train" if path.name.endswith("0.jpg") else "val"

    destination = tmp_path / "conflict"
    with pytest.raises(DatasetValidationError, match="conflicting"):
        dataset.split(
            {"train": 0.5, "val": 0.5},
            group_by=lambda path: path.parent.name,
            assign=assign,
            visualize=False,
            progress=False,
        )
    assert not destination.exists()
