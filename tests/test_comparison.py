from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from dataset_fixer import (
    ComparisonResult,
    Dataset,
    DatasetValidationError,
    Model,
    ModelCollection,
    PredictionResult,
)
from dataset_fixer.comparison.cache import (
    load_package_cache,
    save_package_cache,
)
from dataset_fixer.comparison.cohort import freeze_cohort
from dataset_fixer.comparison.inference import _adaptive_batches, _run_native, resolve_backend
from dataset_fixer.comparison.metrics import evaluate_configuration, optimal_match
from dataset_fixer.comparison.types import ModelSpec, Prediction
from PIL import Image


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
