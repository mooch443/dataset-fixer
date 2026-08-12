from __future__ import annotations

import json
import math
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from dataset_fixer import (
    ComparisonResult,
    Dataset,
    DatasetValidationError,
    ImagePrediction,
    Model,
    ModelCollection,
    ModelInput,
    PredictionCache,
    PredictionCacheMissError,
    PredictionResult,
    PredictionScoreUnavailableError,
)
from dataset_fixer.geometry import validate_collection_geometry
from dataset_fixer.comparison.cache import (
    load_package_cache,
    save_package_cache,
)
from dataset_fixer.comparison.cohort import freeze_cohort
from dataset_fixer.comparison.grouping import resolve_group_splits
from dataset_fixer.comparison.inference import _adaptive_batches, _run_native, resolve_backend
from dataset_fixer.comparison.metrics import (
    component_filtered_presence_breakdown,
    component_filtered_presence_decisions,
    evaluate_configuration,
    grouped_binary_metric_breakdown,
    grouped_presence_metric_breakdown,
    optimal_match,
    segmentation_binary_metric_breakdown,
    segmentation_binary_metric_rows,
)
from dataset_fixer.comparison.types import Cohort, CohortRecord, ModelSpec, Prediction
from dataset_fixer.prediction_cache import prediction_cache_key
from PIL import Image
from conftest import make_yolo_dataset


def test_cohort_is_ordered_and_content_addressed(detect_dataset: Path) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    first = freeze_cohort(dataset, "val")
    second = freeze_cohort(dataset, "val")
    assert first.fingerprint == second.fingerprint
    assert [record.relative_path for record in first.records] == sorted(record.relative_path for record in first.records)
    label = detect_dataset / "val" / "labels" / "val_0.txt"
    label.write_text("0 0.4 0.5 0.25 0.25\n", encoding="utf-8")
    changed = freeze_cohort(Dataset.open(detect_dataset, task="detect", progress=False), "val")
    assert changed.fingerprint != first.fingerprint


def test_group_split_membership_is_normalized_and_detects_mixed_groups() -> None:
    membership = resolve_group_splits(
        [
            ("train", Path("/dataset/train/aoi-a_01.png")),
            ("validation", Path("/dataset/val/aoi-a_02.png")),
            ("valid", Path("/dataset/val/aoi-b_01.png")),
        ],
        lambda path: path.stem.rsplit("_", 1)[0],
    )

    assert membership == {
        "aoi-a": ("train", "val"),
        "aoi-b": ("val",),
    }


def test_package_cache_is_pickle_free_and_detects_corruption(detect_dataset: Path, tmp_path: Path) -> None:
    cohort = freeze_cohort(Dataset.open(detect_dataset, task="detect", progress=False), "val")
    predictions = {
        0.5: {
            record.image_id: [
                Prediction(
                    0,
                    0.9,
                    bbox=(1, 2, 20, 30),
                    polygon=[(1, 2), (20, 2), (20, 30)],
                    polygons=[
                        [(1, 2), (20, 2), (20, 30)],
                        [(4, 5), (5, 5), (5, 6)],
                    ],
                )
            ]
            for record in cohort.records
        }
    }
    root = tmp_path / "cache"
    save_package_cache(root, cohort, {"setting": 1}, predictions)
    loaded, shards, complete = load_package_cache(root, cohort, (0.5,))
    assert complete and shards == len(cohort.records)
    assert loaded[0.5][cohort.records[0].image_id][0].bbox == pytest.approx((1, 2, 20, 30))
    assert len(loaded[0.5][cohort.records[0].image_id][0].polygons) == 2
    shard = next((root / "images").rglob("*.npz"))
    shard.write_bytes(b"broken")
    loaded, _, complete = load_package_cache(root, cohort, (0.5,))
    assert not complete and 0.5 not in loaded


def test_class_aware_optimal_matching_and_metrics(detect_dataset: Path) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    cohort = freeze_cohort(dataset, "val")
    record = cohort.records[0]
    truth = list(record.annotations)
    wrong_class = Prediction(1, .99, bbox=tuple(truth[0]["bbox"]))
    match = optimal_match(truth, [wrong_class], "detect", cohort.metadata)
    assert not match["matches"]
    predictions = {
        row.image_id: [
            Prediction(int(annotation["class_id"]), .9, bbox=tuple(annotation["bbox"]))
            for annotation in row.annotations
        ]
        for row in cohort.records
    }
    metrics = evaluate_configuration(cohort, predictions, 0.5)
    assert metrics["summary"]["map50_95"] == pytest.approx(1.0)
    assert metrics["summary"]["f1"] == pytest.approx(1.0)


def test_instance_segmentation_breakdown_uses_final_foreground_union_masks(
    tmp_path: Path,
) -> None:
    positive = CohortRecord(
        image_id="positive",
        image_path=tmp_path / "positive.png",
        relative_path="positive.png",
        split="val",
        width=6,
        height=6,
        image_sha256="image-positive",
        annotation_sha256="annotation-positive",
        original_id="original-positive",
        annotations=(
            {
                "class_id": 0,
                "polygon": [(1, 1), (3, 1), (3, 3), (1, 3)],
            },
        ),
    )
    empty = CohortRecord(
        image_id="empty",
        image_path=tmp_path / "empty.png",
        relative_path="empty.png",
        split="val",
        width=6,
        height=6,
        image_sha256="image-empty",
        annotation_sha256="annotation-empty",
        original_id="original-empty",
        annotations=(),
    )
    cohort = Cohort(
        split="val",
        fingerprint="cohort",
        records=(positive, empty),
        task="segment",
        classes={0: "school"},
        metadata={},
    )
    predictions = {
        "positive": [
            Prediction(
                class_id=0,
                score=0.9,
                polygon=[(1, 1), (3, 1), (3, 3), (1, 3)],
            )
        ],
        "empty": [
            Prediction(
                class_id=0,
                score=0.9,
                polygon=[(0, 0), (1, 0), (1, 1), (0, 1)],
            )
        ],
    }

    metrics = segmentation_binary_metric_breakdown(cohort, predictions, 0.5)

    assert metrics["dice"] == pytest.approx(0.5)
    assert metrics["iou"] == pytest.approx(0.5)
    assert metrics["positive_case_dice"] == pytest.approx(1.0)
    assert metrics["positive_case_iou"] == pytest.approx(1.0)
    assert metrics["positive_cases"] == 1
    assert metrics["positive_detected_cases"] == 1
    assert metrics["empty_cases"] == 1
    assert metrics["empty_false_positive_cases"] == 1
    assert metrics["empty_image_specificity"] == pytest.approx(0.0)
    assert metrics["empty_false_positive_pixels"] == 4
    assert metrics["empty_mean_false_positive_pixels"] == pytest.approx(4.0)
    assert metrics["raw_presence_precision"] == pytest.approx(0.5)

    rows = segmentation_binary_metric_rows(cohort, predictions, 0.5)
    component_areas = {
        str(row["case_id"]): row["prediction_component_areas"] for row in rows
    }
    filtered = component_filtered_presence_breakdown(rows, component_areas, 5)
    assert filtered["component_filtered_positive_image_recall"] == pytest.approx(1.0)
    assert filtered["component_filtered_empty_image_specificity"] == pytest.approx(1.0)
    assert filtered["component_filtered_presence_precision"] == pytest.approx(1.0)
    decisions = component_filtered_presence_decisions(rows, component_areas, 5)
    assert decisions == {"positive": True, "empty": False}

    grouped = grouped_binary_metric_breakdown(
        rows,
        {"positive": "island-a", "empty": "island-b"},
    )
    assert grouped["group_count"] == 2
    assert grouped["group_macro_dice"] == pytest.approx(0.5)

    grouped_presence = grouped_presence_metric_breakdown(
        rows,
        {"positive": "island-a", "empty": "island-b"},
        decisions,
    )
    assert grouped_presence["group_macro_presence_precision"] == pytest.approx(1.0)
    assert grouped_presence["group_macro_presence_recall"] == pytest.approx(1.0)
    assert grouped_presence["group_macro_presence_f1"] == pytest.approx(1.0)
    assert grouped_presence["group_defined_presence_f1_count"] == 1
    assert math.isnan(grouped_presence["per_group"][1]["presence_f1"])


def test_inference_is_explicit_and_pose_supports_sahi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dataset_fixer.comparison.inference.sahi_available", lambda: True)
    with pytest.raises(ValueError, match="auto.*removed"):
        resolve_backend("auto", "pose")
    assert resolve_backend("sahi", "pose") == "sahi"


def test_native_inference_verifies_each_image_identity(
    detect_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohort = freeze_cohort(Dataset.open(detect_dataset, task="detect", progress=False), "val")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    calls: list[list[str]] = []

    class FakeYOLO:
        def __init__(self, path: str) -> None:
            assert path == str(checkpoint)

        def predict(self, *, source, **kwargs):
            sources = [source] if isinstance(source, str) else list(source)
            calls.append(sources)
            boxes = types.SimpleNamespace(xyxy=[], conf=[], cls=[])
            return [
                types.SimpleNamespace(path=value, boxes=boxes, masks=None, keypoints=None)
                for value in sources
            ]

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    predictions = _run_native(
        ModelSpec("model", checkpoint), cohort, 0.5, 0.1, None, False, {}
    )
    assert calls == [[str(record.image_path) for record in cohort.records]]
    assert list(predictions) == [record.image_id for record in cohort.records]


def test_native_inference_batches_honors_device_backs_off_and_reuses_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    images = []
    for index in range(5):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (16, 16), "white").save(path)
        images.append(path)
    initializations = 0
    calls: list[int] = []

    class FakeYOLO:
        task = "detect"

        def __init__(self, path: str) -> None:
            nonlocal initializations
            assert path == str(checkpoint)
            initializations += 1

        def predict(self, *, source, **kwargs):
            assert kwargs["device"] == "mps"
            sources = [source] if isinstance(source, str) else list(source)
            calls.append(len(sources))
            if len(sources) > 2:
                raise RuntimeError("MPS backend out of memory")
            boxes = types.SimpleNamespace(xyxy=[], conf=[], cls=[])
            return [
                types.SimpleNamespace(path=value, boxes=boxes, masks=None, keypoints=None)
                for value in sources
            ]

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    model = Model(
        checkpoint,
        task="detect",
        resolution=128,
        device="mps",
        batch_size=-1,
    )
    first = model.predict(images, progress=False)
    second = model.predict(images, progress=False)

    assert initializations == 1
    assert calls == [5, 2, 2, 1, 2, 2, 1]
    assert first.settings["requested_batch_size"] == -1
    assert first.settings["resolved_batch_size"] == 2
    assert first.settings["oom_retries"] == 1
    assert first.settings["runtime_reused"] is False
    assert second.settings["runtime_reused"] is True
    assert second.settings["oom_retries"] == 0


def test_prediction_size_policy_retains_smaller_and_skips_only_oversized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "size-policy.pt"
    checkpoint.write_bytes(b"checkpoint")
    images: list[Path] = []
    for name, size in (("smaller", (64, 80)), ("exact", (128, 128)), ("larger", (129, 128))):
        path = tmp_path / f"{name}.png"
        Image.new("RGB", size, "white").save(path)
        images.append(path)
    calls: list[list[str]] = []

    class FakeYOLO:
        task = "detect"

        def __init__(self, _path: str) -> None:
            pass

        def predict(self, *, source, **_kwargs):
            sources = [source] if isinstance(source, str) else list(source)
            calls.append(sources)
            boxes = types.SimpleNamespace(xyxy=[], conf=[], cls=[])
            return [
                types.SimpleNamespace(path=value, boxes=boxes, masks=None, keypoints=None)
                for value in sources
            ]

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    model = Model(
        checkpoint,
        task="detect",
        native_tile_size=128,
        upscale_factor=1,
        batch_size=8,
    )

    with pytest.raises(DatasetValidationError, match="exceeds native_tile_size"):
        model.predict(images, progress=False)
    assert not model.loaded

    result = model.predict(images, errors="skip", progress=False)

    assert [record.image_path for record in result.records] == [
        images[0].resolve(),
        images[1].resolve(),
    ]
    assert calls == [[str(images[0].resolve()), str(images[1].resolve())]]
    policy = result.settings["source_size_policy"]
    assert policy["smaller_or_equal"] == "retain"
    assert policy["oversized"] == "skip"
    assert policy["skipped_inputs"][0]["source"] == str(images[2].resolve())


def test_sahi_accepts_oversized_full_images_but_native_still_rejects_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "sahi-size-policy.pt"
    checkpoint.write_bytes(b"checkpoint")
    image = tmp_path / "full-image.png"
    Image.new("RGB", (300, 200), "white").save(image)
    model = Model(
        checkpoint,
        task="detect",
        native_tile_size=128,
        upscale_factor=1,
        inference="sahi",
    )

    monkeypatch.setattr(
        "dataset_fixer.comparison.inference.sahi_available", lambda: True
    )

    def fake_predict(_model, inputs, **options):
        assert options["backend"] == "sahi"
        return (
            {value.image_id: [] for value in inputs},
            "detect",
            {"reconstructed_source_images": len(inputs)},
        )

    monkeypatch.setattr(
        "dataset_fixer.comparison.inference.predict_model_inputs", fake_predict
    )

    with pytest.raises(DatasetValidationError, match="exceeds native_tile_size"):
        model.predict(image, inference="native", progress=False)
    result = model.predict(image, progress=False)

    assert len(result.records) == 1
    assert (result.records[0].width, result.records[0].height) == (300, 200)
    assert result.settings["source_size_policy"]["maximum_size"] is None
    assert result.settings["source_size_policy"]["oversized"] == (
        "retain-for-sahi-slicing"
    )
    assert result.settings["source_size_policy"]["skipped_inputs"] == []


def test_mixed_native_and_sahi_collection_keeps_native_shared_size_limit(
    detect_dataset: Path,
    tmp_path: Path,
) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    val_samples = [sample for sample in dataset._samples if sample.split == "val"]
    val_sample = val_samples[0]
    with Image.open(val_sample.image_path) as opened:
        opened.resize((256, 256)).save(val_sample.image_path)
    val_sample.width = val_sample.height = 256
    for smaller in val_samples[1:]:
        with Image.open(smaller.image_path) as opened:
            opened.resize((100, 100)).save(smaller.image_path)
        smaller.width = smaller.height = 100
    dataset.manifest["geometry"] = {
        "native_tile_size": 128,
        "tiled": False,
    }
    checkpoints = [tmp_path / "native.pt", tmp_path / "sahi.pt"]
    for checkpoint in checkpoints:
        checkpoint.write_bytes(checkpoint.stem.encode())
    sahi_only = Model.load_many(
        {
            "sahi": {
                "path": checkpoints[1],
                "task": "detect",
                "native_tile_size": 128,
                "upscale_factor": 1,
                "inference": "sahi",
            }
        }
    )
    mixed = Model.load_many(
        {
            "native": {
                "path": checkpoints[0],
                "task": "detect",
                "native_tile_size": 128,
                "upscale_factor": 1,
                "inference": "native",
            },
            "sahi": {
                "path": checkpoints[1],
                "task": "detect",
                "native_tile_size": 128,
                "upscale_factor": 1,
                "inference": "sahi",
            },
        }
    )

    sahi_active = validate_collection_geometry(
        dataset, sahi_only, split="val", errors="skip"
    )
    mixed_active = validate_collection_geometry(
        dataset, mixed, split="val", errors="skip"
    )

    assert not sahi_active._geometry_skip_audit
    assert len(mixed_active._geometry_skip_audit) == 1
    assert mixed_active._geometry_maximum_size == (128, 128)


def test_comparison_size_policy_uses_one_fair_filtered_cohort(
    detect_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    val_samples = [sample for sample in dataset._samples if sample.split == "val"]
    with Image.open(val_samples[0].image_path) as opened:
        opened.resize((64, 64)).save(val_samples[0].image_path)
    val_samples[0].width = val_samples[0].height = 64
    checkpoint = tmp_path / "comparison-size-policy.pt"
    checkpoint.write_bytes(b"checkpoint")
    models = Model.load_many(
        {
            "candidate": {
                "path": checkpoint,
                "task": "detect",
                "native_tile_size": 128,
                "upscale_factor": 1,
            }
        }
    )
    captured: dict[str, object] = {}

    def fake_compare(active: Dataset, _models: ModelCollection, **kwargs: object) -> str:
        captured["samples"] = tuple(
            sample.image_path for sample in active._samples if sample.split == "val"
        )
        captured["audit"] = tuple(active._geometry_skip_audit)
        captured["errors"] = kwargs["errors"]
        return "filtered"

    monkeypatch.setattr("dataset_fixer.comparison.engine._compare_models", fake_compare)

    with pytest.raises(DatasetValidationError, match="exceeds native_tile_size"):
        models.compare(dataset, progress=False)
    assert models.compare(dataset, errors="skip", progress=False) == "filtered"
    assert captured["samples"] == (val_samples[0].image_path,)
    assert len(captured["audit"]) == 1
    assert captured["errors"] == "skip"


def test_collection_predict_uses_each_models_shared_batching_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoints = [tmp_path / "first.pt", tmp_path / "second.pt"]
    for checkpoint in checkpoints:
        checkpoint.write_bytes(checkpoint.stem.encode())
    images = []
    for index in range(5):
        path = tmp_path / f"manual-{index}.png"
        Image.new("RGB", (16, 16), "white").save(path)
        images.append(path)
    initializations: dict[str, int] = {}
    calls: dict[str, list[int]] = {}

    class FakeYOLO:
        task = "detect"

        def __init__(self, path: str) -> None:
            self.name = Path(path).stem
            initializations[self.name] = initializations.get(self.name, 0) + 1

        def predict(self, *, source, **_kwargs):
            sources = [source] if isinstance(source, str) else list(source)
            calls.setdefault(self.name, []).append(len(sources))
            boxes = types.SimpleNamespace(xyxy=[], conf=[], cls=[])
            return [
                types.SimpleNamespace(path=value, boxes=boxes, masks=None, keypoints=None)
                for value in sources
            ]

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    models = Model.load_many(
        {
            "first": {"path": checkpoints[0], "task": "detect", "batch_size": 2},
            "second": {"path": checkpoints[1], "task": "detect", "batch_size": 3},
        }
    )

    results = models.predict(images, progress=False)

    assert tuple(results) == models.names
    assert calls == {"first": [2, 2, 1], "second": [3, 2]}
    assert initializations == {"first": 1, "second": 1}
    assert all(len(result.records) == len(images) for result in results.values())


def test_adaptive_inference_never_exceeds_128_items() -> None:
    sizes: list[int] = []
    consumed: list[int] = []

    class Progress:
        def update(self, count: int) -> None:
            consumed.append(count)

    telemetry = _adaptive_batches(
        list(range(300)),
        lambda batch: sizes.append(len(batch)) or list(batch),
        lambda _batch, _results: None,
        requested=-1,
        device="cuda",
        resolution=128,
        progress_bar=Progress(),
        source="test",
    )

    assert sizes == [128, 128, 44]
    assert consumed == sizes
    assert telemetry["initial_batch_size"] == 128
    assert telemetry["resolved_batch_size"] == 128


def test_model_rejects_batch_sizes_over_128(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="1 through 128"):
        Model(checkpoint, task="detect", batch_size=129)


def test_generic_model_auto_detects_and_predicts_supported_images(
    detect_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "detect-model.pt"
    checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "args.yaml").write_text(
        "task: detect\ndata: training/data.yaml\n",
        encoding="utf-8",
    )
    image = detect_dataset / "val" / "images" / "val_0.jpg"

    class FakeYOLO:
        task = "detect"

        def __init__(self, path: str) -> None:
            assert path == str(checkpoint)

        def predict(self, *, source, **kwargs):
            assert source == str(image.resolve())
            assert kwargs["imgsz"] == 320
            boxes = types.SimpleNamespace(
                xyxy=[[1, 2, 20, 30]],
                conf=[0.9],
                cls=[0],
            )
            return [
                types.SimpleNamespace(
                    path=source,
                    boxes=boxes,
                    masks=None,
                    keypoints=None,
                )
            ]

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    model = Model(checkpoint, name="detector", resolution=320)

    assert model.kind == "ultralytics"
    assert model.task == "detect"
    assert not model.loaded
    assert model.describe()["digest"] == model.digest
    result = model.predict(image, confidence=0.2, progress=False)

    assert isinstance(result, PredictionResult)
    assert result.task == "detect"
    assert model.loaded
    assert result.backend == "native"
    assert result["image_000000"].objects[0].bbox == pytest.approx((1, 2, 20, 30))
    assert result.summary()["predictions"] == 1
    saved = result.save(tmp_path / "saved-predictions")
    assert (saved / "prediction-manifest.json").is_file()
    assert (saved / "predictions.json").is_file()
    model.unload()
    assert not model.loaded


def test_model_load_many_returns_reusable_unbound_collection(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")

    models = Model.load_many(
        {
            "baseline": {
                "path": checkpoint,
                "resolution": 384,
                "confidence": 0.5,
                "postprocess": 0.5,
                "inference": "native",
                "device": "mps",
            }
        }
    )

    assert isinstance(models, ModelCollection)
    assert not hasattr(models, "source")
    assert models.names == ("baseline",)
    assert models["baseline"].resolution == 384
    assert models["baseline"].device == "mps"
    assert models["baseline"].confidence == pytest.approx(0.5)
    assert models["baseline"].postprocess == pytest.approx(0.5)


def test_model_predict_opt_in_cache_round_trips_without_loading_runtime(
    detect_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "cached-model.pt"
    checkpoint.write_bytes(b"checkpoint")
    image = detect_dataset / "val" / "images" / "val_0.jpg"

    class FakeYOLO:
        task = "detect"
        calls = 0

        def __init__(self, _path: str) -> None:
            pass

        def predict(self, *, source, **_kwargs):
            FakeYOLO.calls += 1
            boxes = types.SimpleNamespace(
                xyxy=[[1, 2, 8, 9]],
                conf=[0.9],
                cls=[0],
            )
            return [
                types.SimpleNamespace(
                    path=source,
                    boxes=boxes,
                    masks=None,
                    keypoints=None,
                )
            ]

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    cache = PredictionCache(tmp_path / "shared-cache")
    first_model = Model(checkpoint, task="detect", resolution=64)
    first = first_model.predict(
        image,
        progress=False,
        prediction_cache=cache,
    )
    assert first.cache_info["status"] == "fresh"
    assert first.cache_info["location"].startswith(
        str(cache.location / "predictions")
    )
    assert FakeYOLO.calls == 1

    second_model = Model(checkpoint, name="renamed", task="detect", resolution=64)
    second = second_model.predict(
        image,
        progress=False,
        prediction_cache=cache,
        destination=tmp_path / "cached-output",
    )
    assert second.cache_info["status"] == "hit"
    assert second.model_name == "renamed"
    assert second.records[0].objects[0].bbox == pytest.approx((1, 2, 8, 9))
    assert FakeYOLO.calls == 1
    assert (tmp_path / "cached-output" / "predictions.json").is_file()


def test_prediction_cache_round_trips_semantic_and_native_masks(
    detect_dataset: Path,
    tmp_path: Path,
) -> None:
    image = (detect_dataset / "val" / "images" / "val_0.jpg").resolve()
    with Image.open(image) as opened:
        width, height = opened.size
    model_input = ModelInput(
        image_id="case",
        image_path=image,
        width=width,
        height=height,
        relative_path="val_0.jpg",
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[1:3, 2:4] = 1
    native_mask = np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1)
    result = PredictionResult(
        model_name="semantic",
        model_kind="nnunet",
        task="semantic_segment",
        backend="native",
        records=(
            ImagePrediction(
                image_id="case",
                image_path=image,
                relative_path="val_0.jpg",
                width=width,
                height=height,
                mask=mask,
                native_mask=native_mask,
                metadata={"fold": 0},
            ),
        ),
        inference_seconds=1.5,
    )
    identity = {"schema": 1, "test": "native-mask"}
    key = prediction_cache_key(identity)
    cache = PredictionCache(tmp_path / "mask-cache")
    cached = cache.save(
        key,
        result,
        namespace="semantic",
        identity=identity,
        inputs=(model_input,),
    )
    loaded = cache.load(
        key,
        namespace="semantic",
        identity=identity,
        inputs=(model_input,),
    )

    assert cached.cache_info["status"] == "fresh"
    assert loaded is not None
    assert loaded.cache_info["status"] == "hit"
    assert np.array_equal(loaded.records[0].mask, mask)
    assert np.array_equal(loaded.records[0].native_mask, native_mask)
    assert loaded.records[0].metadata == {"fold": 0}


def test_prediction_cache_round_trips_foreground_probabilities(
    detect_dataset: Path,
    tmp_path: Path,
) -> None:
    image = (detect_dataset / "val" / "images" / "val_0.jpg").resolve()
    with Image.open(image) as opened:
        width, height = opened.size
    model_input = ModelInput("case", image, width, height, "val_0.jpg")
    probability = np.linspace(0, 1, width * height, dtype=np.float32).reshape(
        height, width
    )
    result = PredictionResult(
        model_name="semantic",
        model_kind="ultralytics",
        task="semantic_segment",
        backend="native",
        records=(
            ImagePrediction(
                "case",
                image,
                "val_0.jpg",
                width,
                height,
                mask=(probability >= 0.5).astype(np.uint8),
                foreground_probability=probability,
            ),
        ),
        inference_seconds=0.1,
    )
    identity = {"schema": 1, "test": "probability-map"}
    key = prediction_cache_key(identity)
    cache = PredictionCache(tmp_path / "probability-cache")
    cache.save(
        key,
        result,
        namespace="semantic",
        identity=identity,
        inputs=(model_input,),
    )
    loaded = cache.load(
        key,
        namespace="semantic",
        identity=identity,
        inputs=(model_input,),
    )

    assert loaded is not None
    assert loaded.records[0].foreground_probability is not None
    assert np.allclose(
        loaded.records[0].foreground_probability,
        probability,
        atol=5e-4,
    )
    compact = loaded.save(tmp_path / "compact-probability-result")
    assert not (compact / "foreground-probabilities").exists()
    complete = loaded.save(
        tmp_path / "complete-probability-result",
        include_probabilities=True,
    )
    assert len(list((complete / "foreground-probabilities").glob("*.npy"))) == 1


def test_prediction_threshold_is_task_aware_and_source_configurable(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    semantic = Model(
        checkpoint,
        task="semantic",
        prediction_threshold=0.61,
    )
    instance = Model(
        checkpoint,
        task="segment",
        prediction_threshold=0.17,
    )
    hinted_checkpoint = tmp_path / "yolo26x-sem.pt"
    hinted_checkpoint.write_bytes(b"semantic-checkpoint")
    hinted_semantic = Model(
        hinted_checkpoint,
        prediction_threshold=0.66,
    )

    assert semantic.prediction_threshold == pytest.approx(0.61)
    assert semantic.foreground_probability_threshold == pytest.approx(0.61)
    assert instance.prediction_threshold == pytest.approx(0.17)
    assert instance.confidence == pytest.approx(0.17)
    assert hinted_semantic.task is None
    assert hinted_semantic.prediction_threshold == pytest.approx(0.66)
    assert hinted_semantic.foreground_probability_threshold == pytest.approx(0.66)
    assert hinted_semantic.confidence == pytest.approx(0.25)

    reconfigured = semantic._configured_copy(
        overrides={"prediction_threshold": 0.73}
    )
    assert reconfigured.prediction_threshold == pytest.approx(0.73)
    with pytest.raises(ValueError, match="both threshold aliases"):
        Model(
            checkpoint,
            task="semantic",
            prediction_threshold=0.6,
            foreground_probability_threshold=0.5,
        )


def test_cache_only_probability_requirement_rejects_hard_mask_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(image)
    checkpoint = tmp_path / "semantic.pt"
    checkpoint.write_bytes(b"semantic")
    model = Model(checkpoint, task="semantic", resolution=8)
    cache = PredictionCache(tmp_path / "cache")
    calls = 0

    def hard_mask_only(_model: object, inputs: object, **_kwargs: object):
        nonlocal calls
        calls += 1
        image_id = tuple(inputs)[0].image_id  # type: ignore[arg-type]
        return {image_id: np.ones((8, 8), dtype=np.uint8)}, "semantic_segment", {}

    monkeypatch.setattr(
        "dataset_fixer.comparison.inference.predict_model_inputs",
        hard_mask_only,
    )
    model.predict(image, prediction_cache=cache, progress=False)
    assert calls == 1
    with pytest.raises(PredictionCacheMissError) as error:
        model.predict(
            image,
            prediction_cache=cache,
            cache_only=True,
            require_probability_maps=True,
            progress=False,
        )
    assert error.value.reason == "missing-probability-maps"
    assert calls == 1


def test_model_prediction_threshold_reaches_semantic_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataset_fixer.comparison.inference import SemanticOutput

    image = tmp_path / "semantic-image.png"
    Image.new("RGB", (8, 8)).save(image)
    checkpoint = tmp_path / "semantic-threshold.pt"
    checkpoint.write_bytes(b"semantic")
    model = Model(
        checkpoint,
        task="semantic",
        resolution=8,
        prediction_threshold=0.7,
    )

    def semantic_output(_model: object, inputs: object, **options: object):
        assert options["foreground_probability_threshold"] == pytest.approx(0.7)
        image_id = tuple(inputs)[0].image_id  # type: ignore[arg-type]
        probability = np.full((8, 8), 0.6, dtype=np.float32)
        return {
            image_id: SemanticOutput(
                class_map=np.zeros((8, 8), dtype=np.uint8),
                foreground_probability=probability,
                probability_source="model-probabilities",
            )
        }, "semantic_segment", {}

    monkeypatch.setattr(
        "dataset_fixer.comparison.inference.predict_model_inputs",
        semantic_output,
    )
    result = model.predict(image, progress=False)

    assert result.settings["prediction_threshold"] == pytest.approx(0.7)
    assert not np.any(result.records[0].mask)
    assert np.allclose(result.records[0].foreground_probability, 0.6)


def test_semantic_threshold_errors_without_scores_but_keeps_hard_mask_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "hard-map-image.png"
    Image.new("RGB", (8, 8)).save(image)
    checkpoint = tmp_path / "hard-map-semantic.pt"
    checkpoint.write_bytes(b"hard-map-semantic")
    thresholded = Model(
        checkpoint,
        task="semantic",
        resolution=8,
        prediction_threshold=0.7,
    )
    cache = PredictionCache(tmp_path / "hard-map-cache")
    calls = 0

    def hard_map_only(_model: object, inputs: object, **_options: object):
        nonlocal calls
        calls += 1
        image_id = tuple(inputs)[0].image_id  # type: ignore[arg-type]
        return {image_id: np.ones((8, 8), dtype=np.uint8)}, "semantic_segment", {}

    monkeypatch.setattr(
        "dataset_fixer.comparison.inference.predict_model_inputs",
        hard_map_only,
    )
    with pytest.raises(PredictionScoreUnavailableError) as error:
        thresholded.predict(
            image,
            prediction_cache=cache,
            progress=False,
        )
    assert error.value.reason == "backend-returned-no-semantic-probabilities"
    assert calls == 1

    # The scoreless artifact is not destroyed: an unthresholded semantic model
    # with the same checkpoint/image identity can still reuse its hard mask.
    unthresholded = Model(checkpoint, task="semantic", resolution=8)
    recovered = unthresholded.predict(
        image,
        prediction_cache=cache,
        cache_only=True,
        progress=False,
    )
    assert recovered.cache_info["status"] == "hit"
    assert np.all(recovered.records[0].mask == 1)
    assert calls == 1


def test_semantic_probability_cache_is_shared_across_reference_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_yolo_dataset(
        tmp_path / "segments-image-identity",
        task="segment",
        names=["school"],
        train_rows=[""],
        val_rows=["0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"],
        size=(12, 12),
    )
    first = Dataset.open(source, task="segment", progress=False).export(
        destination=tmp_path / "semantic-reference-a",
        format="semantic_masks",
        visualize=False,
        progress=False,
    )
    second_root = tmp_path / "semantic-reference-b"
    shutil.copytree(first.location, second_root)
    second = Dataset.open(second_root, progress=False)
    second_mask = next(second.mask_dirs["val"].glob("*.png"))
    Image.new("L", (12, 12), 0).save(second_mask)

    checkpoint = tmp_path / "shared-semantic.pt"
    checkpoint.write_bytes(b"semantic")
    model = Model(checkpoint, task="semantic", resolution=12)
    cache = PredictionCache(tmp_path / "shared-cache")
    calls = 0

    def semantic_output(_model: object, inputs: object, **_options: object):
        nonlocal calls
        calls += 1
        values = {}
        for value in inputs:  # type: ignore[union-attr]
            probability = np.full((value.height, value.width), 0.7, dtype=np.float32)
            values[value.image_id] = types.SimpleNamespace(
                class_map=np.ones((value.height, value.width), dtype=np.uint8),
                foreground_probability=probability,
                probability_source="model-probabilities",
            )
        return values, "semantic_segment", {}

    monkeypatch.setattr(
        "dataset_fixer.comparison.inference.predict_model_inputs",
        semantic_output,
    )
    fresh = model.predict(
        first,
        split="val",
        prediction_cache=cache,
        progress=False,
    )
    reused = model.predict(
        second,
        split="val",
        prediction_cache=cache,
        require_probability_maps=True,
        progress=False,
    )

    assert calls == 1
    assert fresh.cache_info["key"] == reused.cache_info["key"]
    assert reused.cache_info["status"] == "hit"
    assert reused.records[0].image_path.is_relative_to(second.location)


def test_shared_semantic_cache_rebases_generated_ids_across_dataset_formats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_yolo_dataset(
        tmp_path / "cross-format-image-identity",
        task="segment",
        names=["school"],
        train_rows=[""],
        val_rows=["0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"],
        size=(12, 12),
    )
    vector = Dataset.open(source, task="segment", progress=False)
    semantic = vector.export(
        destination=tmp_path / "cross-format-semantic",
        format="semantic_masks",
        visualize=False,
        progress=False,
    )
    checkpoint = tmp_path / "cross-format-semantic.pt"
    checkpoint.write_bytes(b"cross-format-semantic")
    model = Model(checkpoint, task="semantic", resolution=12)
    cache = PredictionCache(tmp_path / "shared-cross-format-cache")
    calls = 0

    def semantic_output(_model: object, inputs: object, **_options: object):
        nonlocal calls
        calls += 1
        return {
            value.image_id: types.SimpleNamespace(
                class_map=np.ones((value.height, value.width), dtype=np.uint8),
                foreground_probability=np.full(
                    (value.height, value.width), 0.7, dtype=np.float32
                ),
                probability_source="model-probabilities",
            )
            for value in inputs  # type: ignore[union-attr]
        }, "semantic_segment", {}

    monkeypatch.setattr(
        "dataset_fixer.comparison.inference.predict_model_inputs",
        semantic_output,
    )
    fresh = model.predict(
        semantic,
        split="val",
        prediction_cache=cache,
        progress=False,
    )
    reused = model.predict(
        vector,
        split="val",
        prediction_cache=cache,
        cache_only=True,
        require_probability_maps=True,
        progress=False,
    )

    assert calls == 1
    assert fresh.records[0].image_id != reused.records[0].image_id
    assert reused.cache_info["status"] == "image-compatible-hit"
    assert reused.cache_info["key"] == fresh.cache_info["key"]
    assert reused.cache_info["requested_key"] != fresh.cache_info["key"]
    assert reused.records[0].image_path.is_relative_to(vector.location)
    assert reused.records[0].foreground_probability is not None


def test_shared_instance_cache_rebases_generated_ids_across_dataset_formats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_yolo_dataset(
        tmp_path / "cross-format-instance-identity",
        task="segment",
        names=["school"],
        train_rows=[""],
        val_rows=["0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"],
        size=(12, 12),
    )
    vector = Dataset.open(source, task="segment", progress=False)
    semantic = vector.export(
        destination=tmp_path / "cross-format-instance-semantic",
        format="semantic_masks",
        visualize=False,
        progress=False,
    )
    checkpoint = tmp_path / "cross-format-instance.pt"
    checkpoint.write_bytes(b"cross-format-instance")
    model = Model(checkpoint, task="segment", resolution=12)
    cache = PredictionCache(tmp_path / "shared-cross-format-instance-cache")
    calls = 0

    def instance_output(_model: object, inputs: object, **_options: object):
        nonlocal calls
        calls += 1
        prediction = Prediction(
            class_id=0,
            score=0.9,
            bbox=(2.0, 2.0, 9.0, 9.0),
            polygon=[(2.0, 2.0), (9.0, 2.0), (9.0, 9.0), (2.0, 9.0)],
        )
        return {
            value.image_id: [prediction]
            for value in inputs  # type: ignore[union-attr]
        }, "segment", {}

    monkeypatch.setattr(
        "dataset_fixer.comparison.inference.predict_model_inputs",
        instance_output,
    )
    fresh = model.predict(
        semantic,
        split="val",
        prediction_cache=cache,
        progress=False,
    )
    reused = model.predict(
        vector,
        split="val",
        prediction_cache=cache,
        cache_only=True,
        progress=False,
    )

    assert calls == 1
    assert fresh.records[0].image_id != reused.records[0].image_id
    assert reused.cache_info["status"] == "image-compatible-hit"
    assert reused.cache_info["key"] == fresh.cache_info["key"]
    assert reused.cache_info["requested_key"] != fresh.cache_info["key"]
    assert reused.records[0].image_path.is_relative_to(vector.location)
    assert len(reused.records[0].objects) == 1
    assert reused.records[0].objects[0].score == pytest.approx(0.9)


def test_semantic_predict_cache_on_vector_dataset_reuses_complete_masks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_yolo_dataset(
        tmp_path / "segments-for-semantic-predict",
        task="segment",
        names=["school"],
        train_rows=[""],
        val_rows=["0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"],
        size=(16, 16),
    )
    dataset = Dataset.open(source, task="segment", progress=False)
    checkpoint = tmp_path / "semantic.pt"
    checkpoint.write_bytes(b"semantic-checkpoint")
    model = Model(checkpoint, task="semantic", resolution=32)
    calls = 0

    def fake_predict_inputs(_model, inputs, **_kwargs):
        nonlocal calls
        calls += 1
        return (
            {
                value.image_id: np.pad(
                    np.ones((8, 8), dtype=np.uint8),
                    ((4, 4), (4, 4)),
                )
                for value in inputs
            },
            "semantic_segment",
            {"synthetic": True},
        )

    monkeypatch.setattr(
        "dataset_fixer.comparison.inference.predict_model_inputs",
        fake_predict_inputs,
    )
    first = model.predict(
        dataset,
        split="val",
        progress=False,
        prediction_cache=True,
    )
    second = model.predict(
        dataset,
        split="val",
        progress=False,
        prediction_cache=True,
    )

    assert calls == 1
    assert first.cache_info["status"] == "fresh"
    assert second.cache_info["status"] == "hit"
    assert np.array_equal(first.records[0].mask, second.records[0].mask)
    assert Path(second.cache_info["location"]).is_relative_to(
        dataset.location / ".cache" / "evaluations" / "predictions"
    )
    broken_mask = next(
        (Path(second.cache_info["location"]) / "raw-result" / "masks").glob("*.png")
    )
    broken_mask.unlink()
    repaired = model.predict(
        dataset,
        split="val",
        progress=False,
        prediction_cache=True,
        batch_size=2,
    )
    assert repaired.cache_info["status"] == "fresh"
    assert calls == 2
    execution_only_change = model.predict(
        dataset,
        split="val",
        progress=False,
        prediction_cache=True,
        batch_size=1,
    )
    assert execution_only_change.cache_info["status"] == "hit"
    assert calls == 2
    changed_threshold = model.predict(
        dataset,
        split="val",
        progress=False,
        prediction_cache=True,
        confidence=0.6,
    )
    assert changed_threshold.cache_info["status"] == "fresh"
    assert calls == 3


def test_predict_and_native_compare_share_the_existing_package_cache(
    detect_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    checkpoint = tmp_path / "shared-native.pt"
    checkpoint.write_bytes(b"shared-native")
    model = Model(checkpoint, task="detect", resolution=480)
    cache = PredictionCache(tmp_path / "explicit-comparison-cache")
    direct_calls = 0

    def fake_predict_inputs(_model, inputs, **_kwargs):
        nonlocal direct_calls
        direct_calls += 1
        return ({value.image_id: [] for value in inputs}, "detect", {})

    monkeypatch.setattr(
        "dataset_fixer.comparison.inference.predict_model_inputs",
        fake_predict_inputs,
    )
    direct = model.predict(
        dataset,
        split="val",
        progress=False,
        prediction_cache=cache,
    )
    assert direct_calls == 1
    assert direct.cache_info["status"] == "fresh"

    def cached_only_comparison_inference(
        _spec,
        _cohort,
        *,
        thresholds,
        existing=None,
        **_kwargs,
    ):
        assert existing is not None
        assert set(existing) == {float(value) for value in thresholds}
        return dict(existing), {"synthetic": 0.0}

    monkeypatch.setattr(
        "dataset_fixer.comparison.engine.run_inference",
        cached_only_comparison_inference,
    )
    comparison = model.compare(
        dataset,
        split="val",
        progress=False,
        prediction_cache=cache,
        destination=tmp_path / "predict-then-compare",
    )
    assert comparison.cache_statistics["prediction_hits"] == 1
    assert Path(direct.cache_info["location"]).is_relative_to(
        cache.location / "predictions"
    )


def test_native_compare_cache_is_reused_by_predict(
    detect_dataset: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    checkpoint = tmp_path / "comparison-first.pt"
    checkpoint.write_bytes(b"comparison-first")
    model = Model(checkpoint, task="detect", resolution=480)
    cache = PredictionCache(tmp_path / "comparison-first-cache")
    comparison_calls = 0

    def fake_inference(
        _spec,
        cohort,
        *,
        thresholds,
        existing=None,
        on_threshold=None,
        **_kwargs,
    ):
        nonlocal comparison_calls
        values = dict(existing or {})
        for threshold in thresholds:
            threshold = float(threshold)
            if threshold in values:
                continue
            comparison_calls += 1
            values[threshold] = {record.image_id: [] for record in cohort.records}
            if on_threshold is not None:
                on_threshold(threshold, values[threshold])
        return values, {"synthetic": 0.0}

    monkeypatch.setattr(
        "dataset_fixer.comparison.engine.run_inference",
        fake_inference,
    )
    model.compare(
        dataset,
        split="val",
        progress=False,
        prediction_cache=cache,
        destination=tmp_path / "comparison-first-report",
    )
    assert comparison_calls == 1

    def forbidden_predict(*_args, **_kwargs):
        raise AssertionError("direct inference ran despite a compatible comparison cache")

    monkeypatch.setattr(
        "dataset_fixer.comparison.inference.predict_model_inputs",
        forbidden_predict,
    )
    result = model.predict(
        dataset,
        split="val",
        progress=False,
        prediction_cache=cache,
    )
    assert result.cache_info["status"] == "hit"
    assert comparison_calls == 1


def test_model_collection_rejects_removed_shared_configuration(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    models = Model.load_many({"candidate": {"path": checkpoint, "task": "detect"}})

    with pytest.raises(TypeError, match="unexpected keyword argument 'resolution'"):
        models.compare(object(), resolution=640)
    with pytest.raises(TypeError, match="unexpected keyword argument 'baseline'"):
        models.compare(object(), baseline="candidate")
    with pytest.raises(TypeError, match="unexpected keyword argument 'inference'"):
        Model.load_many({"candidate": checkpoint}, inference="sahi")
    with pytest.raises(DatasetValidationError, match="missing source/path"):
        Model.load_many({"candidate": {"model_folder": checkpoint}})


def test_model_owns_explicit_comparison_and_sahi_settings(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    model = Model(
        checkpoint,
        task="detect",
        resolution=640,
        inference="sahi",
        confidence=0.4,
        postprocess=0.8,
        sahi_slice_height=320,
        sahi_slice_width=256,
        sahi_overlap=0.1,
    )

    assert model.resolution == 640
    assert model.inference == "sahi"
    assert model.confidence == pytest.approx(0.4)
    assert model.postprocess == pytest.approx(0.8)
    assert model.settings["sahi_slice_height"] == 320
    assert model.settings["sahi_slice_width"] == 256
    assert model.settings["sahi_overlap"] == pytest.approx(0.1)


def test_model_collection_compare_atomic_result(
    detect_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")

    def fake_inference(spec, cohort, *, thresholds, existing=None, on_threshold=None, **kwargs):
        values = dict(existing or {})
        for threshold in thresholds:
            threshold = float(threshold)
            if threshold in values:
                continue
            fake_inference.calls += 1
            values[threshold] = {
                record.image_id: [
                    Prediction(int(annotation["class_id"]), .95, bbox=tuple(annotation["bbox"]))
                    for annotation in record.annotations
                ]
                for record in cohort.records
            }
            if on_threshold is not None:
                on_threshold(threshold, values[threshold])
        return values, {"fake": 0.01}
    fake_inference.calls = 0

    monkeypatch.setattr("dataset_fixer.comparison.engine.run_inference", fake_inference)
    destination = tmp_path / "comparison"
    models = Model.load_many(
        {
            "baseline": {
                "path": checkpoint,
                "task": "detect",
                "confidence": 0.5,
                "postprocess": 0.5,
            }
        }
    )
    result = models.compare(
        dataset,
        split="val",
        progress=False,
        destination=destination,
    )
    assert isinstance(result, ComparisonResult)
    assert result.cohort_verified
    assert result.ranking[0]["score"] == pytest.approx(1.0)
    assert (destination / "reports" / "result.json").is_file()
    assert (destination / "reports" / "plots.png").is_file()
    assert (destination / "reports" / "comparison.png").is_file()
    assert not list(destination.rglob("*.csv"))
    assert not list(destination.rglob("*.jsonl"))
    assert (dataset.location / ".cache" / "evaluations").is_dir()
    prediction_cache = dataset.location / ".cache" / "evaluations" / "predictions"
    assert list(prediction_cache.glob("*/cache-manifest.json"))
    assert not list(tmp_path.glob(".comparison.building-*"))

    old_manifest_path = destination / "reports" / "result.json"
    old_manifest = json.loads(old_manifest_path.read_text())
    old_manifest["schema"] = 8
    old_manifest_path.write_text(json.dumps(old_manifest), encoding="utf-8")
    regenerated = models.compare(
        dataset,
        split="val",
        progress=False,
        destination=destination,
    )
    assert fake_inference.calls == 1
    assert json.loads(old_manifest_path.read_text())["schema"] == 12
    assert regenerated.cache_statistics["prediction_hits"] == 1

    models.compare(
        dataset,
        split="val",
        progress=False,
        destination=destination,
    )
    assert fake_inference.calls == 1

    cached_elsewhere = models.compare(
        dataset,
        split="val",
        progress=False,
        destination=tmp_path / "comparison-from-dataset-cache",
    )
    assert (
        cached_elsewhere.cache_statistics["models"]["baseline"]["root"]
        == result.cache_statistics["models"]["baseline"]["root"]
    )
    assert fake_inference.calls == 1
    assert cached_elsewhere.cache_statistics["prediction_hits"] == 1

    direct = Model(
        checkpoint,
        name="direct",
        task="detect",
        confidence=0.5,
        postprocess=0.5,
    )
    direct_result = direct.compare(
        dataset,
        split="val",
        progress=False,
        destination=tmp_path / "direct-comparison",
    )
    assert direct_result.ranking[0]["model"] == "direct"
    assert direct_result.settings["models"]["direct"]["backend"] == "native"
    assert fake_inference.calls == 1, "renaming identical model bytes invalidated the cache"

    changed_execution = models.configure(
        {"baseline": {"batch_size": 2, "device": "cpu", "workers": 3}}
    )
    execution_result = changed_execution.compare(
        dataset,
        split="val",
        progress=False,
        destination=tmp_path / "comparison-changed-batch",
    )
    assert fake_inference.calls == 1, "execution-only settings invalidated predictions"
    assert execution_result.cache_statistics["prediction_hits"] == 1


def test_segment_comparison_adds_postprocessed_binary_breakdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_yolo_dataset(
        tmp_path / "segments",
        task="segment",
        names=["school"],
        train_rows=["0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"],
        val_rows=["0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8", ""],
        size=(16, 16),
    )
    dataset = Dataset.open(source, task="segment", progress=False)
    dataset.manifest["geometry"] = {
        "native_tile_size": 16,
        "tiled": True,
    }
    checkpoint = tmp_path / "segment.pt"
    checkpoint.write_bytes(b"segment checkpoint")

    def fake_inference(
        spec,
        cohort,
        *,
        thresholds,
        existing=None,
        on_threshold=None,
        **kwargs,
    ):
        values = dict(existing or {})
        for threshold in thresholds:
            threshold = float(threshold)
            if threshold in values:
                continue
            values[threshold] = {
                record.image_id: [
                    Prediction(
                        class_id=int(annotation["class_id"]),
                        score=0.9,
                        bbox=tuple(annotation["bbox"]),
                        polygon=[tuple(point) for point in annotation["polygon"]],
                    )
                    for annotation in record.annotations
                ]
                for record in cohort.records
            }
            if on_threshold is not None:
                on_threshold(threshold, values[threshold])
        return values, {"fake": 0.01}

    monkeypatch.setattr("dataset_fixer.comparison.engine.run_inference", fake_inference)
    models = Model.load_many(
        {
            "segmenter": {
                "path": checkpoint,
                "task": "segment",
                "native_tile_size": 16,
                "upscale_factor": 1,
                "confidence": 0.5,
                "postprocess": 0.5,
            }
        }
    )
    destination = tmp_path / "segment-comparison"

    result = models.compare(
        dataset,
        progress=False,
        destination=destination,
        min_connected_component_area=2,
        group_by=lambda path: path.stem.split("_")[0],
    )

    row = result.ranking[0]
    assert row["heldout_projection"] == "instance-polygon-foreground-union"
    assert row["dice"] == pytest.approx(1.0)
    assert row["positive_case_dice"] == pytest.approx(1.0)
    assert row["positive_cases"] == 1
    assert row["empty_cases"] == 1
    assert row["empty_image_specificity"] == pytest.approx(1.0)
    manifest = json.loads((destination / "reports" / "result.json").read_text())
    assert manifest["schema"] == 12
    assert manifest["ranking"][0]["model_type"] == models[0].model_type
    assert manifest["ranking"][0]["positive_micro_iou"] == pytest.approx(1.0)
    assert manifest["ranking"][0]["small_object_dice"] == pytest.approx(1.0)
    assert manifest["ranking"][0]["raw_presence_precision"] == pytest.approx(1.0)
    assert manifest["ranking"][0][
        "component_filtered_presence_precision"
    ] == pytest.approx(1.0)
    assert manifest["presence_analysis"]["threshold_source"] == "explicit"
    assert manifest["presence_analysis"][
        "resolved_min_connected_component_area_px"
    ] == pytest.approx(2)
    assert manifest["grouped_analysis"]["status"] == "complete"
    assert manifest["grouped_analysis"]["primary_ranking_unchanged"] is True
    assert manifest["settings"]["grouping"]["group_splits"] == {
        "train": ["train"],
        "val": ["val"],
    }
    assert {
        row["group"]: row["dataset_splits"]
        for row in manifest["grouped_analysis"]["models"]["segmenter"]["per_group"]
    } == {"val": ["val"]}
    assert manifest["object_size_analysis"]["status"] == "complete"
    assert not any(
        "unavailable for tiled" in limitation
        for limitation in manifest["limitations"]
    )
    assert manifest["object_size_analysis"]["reference_object_extraction"] == (
        "native-instance-annotations"
    )
    assert manifest["object_size_analysis"]["matching_class_policy"] == "class-aware"
    assert manifest["object_size_analysis"]["connectivity"] is None
    assert manifest["reports"]["metric_breakdown"] == "reports/metric-breakdown.png"
    assert manifest["reports"]["grouped_metric_breakdown"] == (
        "reports/grouped-metric-breakdown.png"
    )
    assert (destination / "reports" / "grouped-metric-breakdown.png").is_file()
    assert manifest["grouped_analysis"]["presence"]["status"] == "complete"
    for metric in ("precision", "recall", "f1"):
        assert manifest["reports"][f"grouped_presence_{metric}"] == (
            f"reports/grouped-presence-{metric}.png"
        )
        assert (destination / "reports" / f"grouped-presence-{metric}.png").is_file()
    assert (destination / "reports" / "object-size-breakdown.png").is_file()


def test_comparison_visuals_have_data_and_metadata_sidecars(
    detect_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    checkpoint = tmp_path / "visual.pt"
    checkpoint.write_bytes(b"visual checkpoint")

    def fake_inference(spec, cohort, *, thresholds, **kwargs):
        return {
            float(threshold): {
                record.image_id: [
                    Prediction(int(annotation["class_id"]), .9, bbox=tuple(annotation["bbox"]))
                    for annotation in record.annotations
                ]
                for record in cohort.records
            }
            for threshold in thresholds
        }, {"fake": .01}

    monkeypatch.setattr("dataset_fixer.comparison.engine.run_inference", fake_inference)
    destination = tmp_path / "visual-comparison"
    models = Model.load_many(
        {
            "visual": {
                "path": checkpoint,
                "task": "detect",
                "confidence": .5,
                "postprocess": .5,
            }
        }
    )
    models.compare(
        dataset,
        save_prediction_plots=True,
        progress=False,
        destination=destination,
    )
    assert (destination / "reports" / "plots.png").is_file()
    assert (destination / "reports" / "comparison.png").is_file()
    assert len(list((destination / "predictions").rglob("*.png"))) == 2
    assert not (destination / "figures").exists()
    assert not (destination / "qualitative").exists()
