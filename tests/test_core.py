from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from dataset_fixer import (
    Dataset,
    DatasetValidationError,
    ImagePrediction,
    Model,
    PredictionResult,
    Task,
)
from dataset_fixer.utils import settings_fingerprint, to_jsonable
from conftest import make_yolo_dataset


def _audit(dataset: Dataset, name: str):
    return dataset.manifest["audits"][name]


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


def test_dataset_sample_is_a_deterministic_materialized_view(
    detect_dataset: Path,
) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)

    first = dataset.sample(n=1, split="val", seed=7)
    second = dataset.sample(n=1, split="val", seed=7)

    assert first.location == dataset.location
    assert first.format == dataset.format
    assert first.splits == ("val",)
    assert len(first._samples) == 1
    assert first._samples[0].relative_path == second._samples[0].relative_path
    assert first._samples[0] is not dataset._samples[-1]
    assert first.manifest["view"] == {
        "kind": "sample",
        "n": 1,
        "requested_n": 1,
        "split": "val",
        "seed": 7,
    }
    assert len(dataset._samples) == 6


def test_dataset_sample_validates_size_split_and_seed(detect_dataset: Path) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)

    with pytest.raises(ValueError, match="positive integer"):
        dataset.sample(n=0)
    with pytest.raises(ValueError, match="seed must be an integer"):
        dataset.sample(n=1, seed=True)
    with pytest.raises(ValueError, match="Unknown split"):
        dataset.sample(n=1, split="test")


def test_dataset_presence_filters_sample_and_add_preserve_order(
    tmp_path: Path,
) -> None:
    source = make_yolo_dataset(
        tmp_path / "presence",
        task="detect",
        val_rows=["0 0.5 0.5 0.25 0.25", ""],
    )
    dataset = Dataset.open(source, task="detect", progress=False)
    validation = [sample for sample in dataset._samples if sample.split == "val"]
    result = PredictionResult(
        model_name="presence-model",
        model_kind="ultralytics",
        task="segment",
        backend="native",
        records=(
            ImagePrediction(
                image_id="annotated-miss",
                image_path=validation[0].image_path,
                relative_path=validation[0].relative_path,
                width=validation[0].width,
                height=validation[0].height,
                mask=np.zeros(
                    (validation[0].height, validation[0].width),
                    dtype=bool,
                ),
            ),
            ImagePrediction(
                image_id="empty-false-positive",
                image_path=validation[1].image_path,
                relative_path=validation[1].relative_path,
                width=validation[1].width,
                height=validation[1].height,
                mask=np.ones(
                    (validation[1].height, validation[1].width),
                    dtype=bool,
                ),
            ),
        ),
        inference_seconds=0.0,
    )

    annotated_misses = dataset.filter(
        gt_annotated=True,
        predictions=result,
        has_prediction=False,
    ).sample(n=5, split="val", seed=1335)
    empty_false_positives = dataset.filter(
        gt_annotated=False,
        predictions=result,
        has_prediction=True,
    ).sample(n=5, split="val", seed=42)
    combined = annotated_misses.add(empty_false_positives)

    assert [sample.relative_path for sample in combined._samples] == [
        validation[0].relative_path,
        validation[1].relative_path,
    ]
    assert len(annotated_misses.add(annotated_misses)._samples) == 1
    assert len(dataset._samples) == 4


def test_dataset_prediction_filter_requires_a_result_or_model(
    tmp_path: Path,
) -> None:
    source = make_yolo_dataset(tmp_path / "presence-validation", task="detect")
    dataset = Dataset.open(source, task="detect", progress=False)

    with pytest.raises(ValueError, match="requires predictions=PredictionResult or model=Model"):
        dataset.filter(has_prediction=True)
    with pytest.raises(ValueError, match="at least one filter"):
        dataset.filter()

    image = tmp_path / "unrelated.jpg"
    image.write_bytes(b"not read by the filter")
    unrelated = PredictionResult(
        model_name="unrelated",
        model_kind="ultralytics",
        task="segment",
        backend="native",
        records=(
            ImagePrediction(
                image_id="unrelated",
                image_path=image,
                relative_path=image.name,
                width=1,
                height=1,
                mask=np.zeros((1, 1), dtype=bool),
            ),
        ),
        inference_seconds=0.0,
    )
    with pytest.raises(ValueError, match="does not cover any image"):
        dataset.filter(predictions=unrelated, has_prediction=False)


def test_dataset_prediction_filter_resolves_model_cache_or_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_yolo_dataset(
        tmp_path / "model-backed-filter",
        task="detect",
        val_rows=["0 0.5 0.5 0.25 0.25", ""],
    )
    dataset = Dataset.open(source, task="detect", progress=False)
    validation = [sample for sample in dataset._samples if sample.split == "val"]
    checkpoint = tmp_path / "filter-model.pt"
    checkpoint.write_bytes(b"filter-model")
    model = Model(checkpoint, task="segment")
    calls: list[dict[str, object]] = []
    source_sizes: list[int] = []

    def fake_predict(self, prediction_source, **kwargs):
        assert self is model
        assert prediction_source.location == dataset.location
        source_sizes.append(len(prediction_source._samples))
        calls.append(kwargs)
        return PredictionResult(
            model_name=self.name,
            model_kind="ultralytics",
            task="segment",
            backend="native",
            records=(
                ImagePrediction(
                    image_id="empty",
                    image_path=validation[0].image_path,
                    relative_path=validation[0].relative_path,
                    width=validation[0].width,
                    height=validation[0].height,
                ),
                ImagePrediction(
                    image_id="positive",
                    image_path=validation[1].image_path,
                    relative_path=validation[1].relative_path,
                    width=validation[1].width,
                    height=validation[1].height,
                    objects=(object(),),
                ),
            ),
            inference_seconds=0.0,
        )

    monkeypatch.setattr(Model, "predict", fake_predict)

    filtered = dataset.filter(
        model=model,
        has_prediction=True,
        split="val",
        progress=False,
    )

    assert [sample.relative_path for sample in filtered._samples] == [
        validation[1].relative_path
    ]
    assert calls == [
        {
            "split": "val",
            "progress": False,
            "prediction_cache": True,
        }
    ]
    assert source_sizes == [2]

    calls.clear()
    source_sizes.clear()
    filtered = dataset.filter(
        gt_annotated=False,
        model=model,
        has_prediction=True,
        split="val",
        progress=False,
    )
    assert [sample.relative_path for sample in filtered._samples] == [
        validation[1].relative_path
    ]
    assert len(calls) == 1
    assert source_sizes == [1]

    calls.clear()
    source_sizes.clear()
    annotated = dataset.filter(
        gt_annotated=True,
        model=model,
        split="val",
        progress=False,
    )
    assert [sample.relative_path for sample in annotated._samples] == [
        validation[0].relative_path
    ]
    assert calls == []
    assert source_sizes == []


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
    manifest = result.manifest
    assert manifest["history"][0]["settings"]["seed"] == 7
    assert manifest["environment"]["dataset_fixer_git"]["commit"]
    assert manifest["dataset_fixer"]["version"]
    assert manifest["source_dataset"]["fingerprint"]
    assert manifest["validation"]["passed"] is True
    assert manifest["timing"]["started_at"] and manifest["timing"]["finished_at"]
    assert manifest["settings_fingerprint"] in result.location.name or manifest["settings_fingerprint"]
    assert all(record["transformation_chain"] for record in result.provenance.values())
    generated_yaml = yaml.safe_load(result.data_yaml.read_text(encoding="utf-8"))
    assert "path" not in generated_yaml
    assert generated_yaml["train"] == "train/images"
    assert generated_yaml["val"] == "val/images"


def test_generated_dataset_loads_after_being_moved(
    detect_dataset: Path,
    tmp_path: Path,
) -> None:
    exported = Dataset.open(detect_dataset, task="detect", progress=False).export(
        destination=tmp_path / "portable",
        visualize=False,
        progress=False,
    )
    expected = sorted(
        sample.image_path.relative_to(exported.location).as_posix()
        for sample in exported._samples
    )

    moved = tmp_path / "relocated" / "portable"
    moved.parent.mkdir()
    shutil.move(str(exported.location), moved)

    reopened = Dataset.open(moved, task="detect", progress=False)

    assert reopened.location == moved.resolve()
    assert reopened.data_yaml == moved.resolve() / "data.yaml"
    assert reopened.training_ready
    assert (
        sorted(
            sample.image_path.relative_to(moved.resolve()).as_posix()
            for sample in reopened._samples
        )
        == expected
    )


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

    report = _audit(result, "split_group_audit")
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

    manifest = result.manifest
    validation = manifest["validation"]["split_group_isolation"]
    assert validation["status"] == "passed"
    assert validation["report"] == "reports/dataset-info.json#audits.split_group_audit"
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

    report = _audit(result, "split_group_audit")
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
    assert "split_group_audit" in grouped.manifest["audits"]

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

    assert "split_group_audit" not in result.manifest["audits"]
    manifest = result.manifest
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

    assert "split_group_audit" not in result.manifest["audits"]
    manifest = result.manifest
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
    counts = _audit(clean, "class_counts")
    assert counts["result"]["background"] == 3
    assert counts["names"]["background"] == "background"
    assert (clean.location / "reports" / "plots.png").is_file()


def test_remove_classes_can_drop_every_image_containing_removed_class(
    detect_dataset: Path,
    tmp_path: Path,
) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)

    planned = dataset.remove_classes(
        [1],
        drop_containing_images=True,
        visualize=False,
        progress=False,
    )

    # Class 1 occurs on three images. One of them also contains class 0, and
    # must still be dropped because matching is image-level.
    assert {sample.image_path.name for sample in planned._samples} == {
        "train_1.jpg",
        "train_3.jpg",
        "val_0.jpg",
    }
    assert planned.classes == {0: "fruit"}
    assert planned.history[-1]["settings"]["dropped_containing_images"] == 3
    assert all(
        annotation.class_id == 0
        for sample in planned._samples
        for annotation in sample.annotations
    )

    exported = planned.export(
        destination=tmp_path / "drop-containing",
        visualize=False,
        progress=False,
    )

    assert len(exported._samples) == 3
    counts = _audit(exported, "class_counts")
    assert counts["images"] == {
        "result": 3,
        "dropped_containing_removed_classes": 3,
    }


def test_remove_classes_rejects_drop_containing_with_merge(
    detect_dataset: Path,
) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)

    with pytest.raises(ValueError, match="mutually exclusive"):
        dataset.remove_classes(
            [1],
            merge_into=0,
            drop_containing_images=True,
            visualize=False,
        )


def test_move_images_with_classes_moves_whole_images_and_preserves_labels(
    detect_dataset: Path,
    tmp_path: Path,
) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    original_annotations = {
        sample.image_path.name: [annotation.class_id for annotation in sample.annotations]
        for sample in dataset._samples
    }

    planned = dataset.move_images_with_classes(
        ["damaged"],
        to_split="test",
        group_by=lambda path: path.name,
        visualize=False,
        progress=False,
    )

    assert planned.splits == ("train", "val", "test")
    assert planned.classes == dataset.classes
    assert planned.history[-1]["settings"]["matched_images"] == 3
    assert planned.history[-1]["settings"]["moved_images"] == 3
    for sample in planned._samples:
        if 1 in original_annotations[sample.image_path.name]:
            assert sample.split == "test"
        else:
            assert sample.split in {"train", "val"}
        assert [annotation.class_id for annotation in sample.annotations] == (
            original_annotations[sample.image_path.name]
        )

    exported = planned.export(
        destination=tmp_path / "class-quarantine",
        visualize=False,
        progress=False,
    )

    assert len(exported._samples) == len(dataset._samples)
    moved_records = [
        record
        for record in exported.provenance.values()
        if record.get("class_move", {}).get("moved")
    ]
    assert len(moved_records) == 3
    assert all(record["output_split"] == "test" for record in moved_records)
    summary = _audit(exported, "class_move_summary")
    assert summary["distribution"] == {"test": 3, "train": 2, "val": 1}


def test_move_images_with_classes_honours_source_splits(
    detect_dataset: Path,
) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)

    planned = dataset.move_images_with_classes(
        [1],
        to_split="test",
        group_by=lambda path: path.name,
        source_splits=("train",),
        visualize=False,
        progress=False,
    )

    assert {
        sample.image_path.name: sample.split for sample in planned._samples
    }["val_1.jpg"] == "val"
    assert sum(sample.split == "test" for sample in planned._samples) == 2


def test_move_images_with_classes_expands_to_complete_groups(
    detect_dataset: Path,
) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)

    planned = dataset.move_images_with_classes(
        [1],
        to_split="test",
        source_splits=("train",),
        group_by=lambda path: path.parent.name,
        visualize=False,
        progress=False,
    )

    # Both train groups contain one class-1 trigger, so their class-0 group
    # mates move as well. Validation never triggered a move and stays put.
    assert {sample.split for sample in planned._samples if sample.image_path.name.startswith("train_")} == {"test"}
    assert {sample.split for sample in planned._samples if sample.image_path.name.startswith("val_")} == {"val"}
    settings = planned.history[-1]["settings"]
    assert settings["matched_images"] == 2
    assert settings["matched_groups"] == 2
    assert settings["selected_group_images"] == 4
    assert settings["group_expansion_images"] == 2


def test_move_n_groups_is_deterministic_and_group_atomic(
    detect_dataset: Path,
    tmp_path: Path,
) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)

    def get_group(path: Path) -> str:
        return path.parent.name

    first = dataset.move_n_groups(
        n=1,
        from_split="train",
        to_split="test",
        group_by=get_group,
        seed=17,
        visualize=False,
        progress=False,
    )
    second = dataset.move_n_groups(
        n=1,
        from_split="train",
        to_split="test",
        group_by=get_group,
        seed=17,
        visualize=False,
        progress=False,
    )

    first_test = {sample.image_path.name for sample in first._samples if sample.split == "test"}
    second_test = {sample.image_path.name for sample in second._samples if sample.split == "test"}
    assert first_test == second_test
    assert len(first_test) == 2
    assert first.history[-1]["settings"]["selected_groups"] == 1
    assert first.history[-1]["settings"]["moved_images"] == 2

    exported = first.export(
        destination=tmp_path / "one-group",
        visualize=False,
        progress=False,
    )
    report = _audit(exported, "split_group_audit")
    assert report["status"] == "passed"
    assert report["overlap_count"] == 0
    assert len(
        {
            record["group_move"]["group"]
            for record in exported.provenance.values()
            if "group_move" in record
        }
    ) == 1


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
    counts = _audit(exported, "class_counts")
    assert counts["result"]["0"] == original_annotation_count
    assert counts["result"]["background"] == 0
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
    assert "class_renames" in exported.manifest["audits"]
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
    report = _audit(exported, "empty_image_balance")
    assert report["train"]["result"] == {"annotated": 1, "background": 1}
    assert report["val"]["result"] == {"annotated": 1, "background": 1}
    assert (exported.location / "reports" / "plots.png").is_file()


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
        (self.staging / "reports" / "lineage.json.gz").write_bytes(b"not-gzip")
        return manifest

    monkeypatch.setattr(OutputBuilder, "write_reports", write_reports_then_corrupt)
    with pytest.raises(DatasetValidationError, match="Unreadable staged lineage"):
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


def test_split_ratios_apply_to_annotated_images_as_well_as_totals(
    tmp_path: Path,
) -> None:
    """Background dominates the count, so totals alone can starve val of labels."""

    rows = ["0 0.5 0.5 0.2 0.2" if index % 5 == 0 else "" for index in range(40)]
    source = make_yolo_dataset(
        tmp_path / "sparse_labels",
        task="detect",
        names=["fruit"],
        train_rows=rows,
        val_rows=[""],
        size=(300, 300),
    )

    result = (
        Dataset.open(source, task="detect", progress=False)
        .split({"train": 0.75, "val": 0.25}, seed=7, visualize=False, progress=False)
        .export(destination=tmp_path / "split", visualize=False, progress=False)
    )

    split_record = next(
        record
        for record in result.manifest["history"]
        if record["operation"] == "split"
    )
    distribution = split_record["settings"]["resolved_distribution"]
    assert split_record["settings"]["ratio_targets"] == "total_and_annotated_images"

    annotated = {
        split: value["annotated_images"] for split, value in distribution.items()
    }
    assert sum(annotated.values()) == 8
    # Both dimensions land on the requested ratio, not just the total.
    for split, requested in (("train", 0.75), ("val", 0.25)):
        assert abs(distribution[split]["annotated_fraction"] - requested) <= 0.13
        assert abs(distribution[split]["image_fraction"] - requested) <= 0.13
    assert annotated["val"] > 0


def test_split_still_honours_groups_while_balancing_annotations(
    tmp_path: Path,
) -> None:
    rows = ["0 0.5 0.5 0.2 0.2" if index % 4 == 0 else "" for index in range(24)]
    source = make_yolo_dataset(
        tmp_path / "grouped_labels",
        task="detect",
        names=["fruit"],
        train_rows=rows,
        val_rows=[""],
        size=(300, 300),
    )

    result = (
        Dataset.open(source, task="detect", progress=False)
        .split(
            {"train": 0.5, "val": 0.5},
            group_by=lambda path: path.parent.name,
            seed=3,
            visualize=False,
            progress=False,
        )
        .export(destination=tmp_path / "grouped", visualize=False, progress=False)
    )

    targets: dict[str, set[str]] = {}
    for sample in result._samples:
        targets.setdefault(sample.relative_path.parent.name, set()).add(sample.split)
    assert all(len(splits) == 1 for splits in targets.values())
