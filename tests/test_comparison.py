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
from dataset_fixer.comparison.inference import _run_native, resolve_backend
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
    calls: list[str] = []

    class FakeYOLO:
        def __init__(self, path: str) -> None:
            assert path == str(checkpoint)

        def predict(self, *, source, **kwargs):
            assert isinstance(source, str), "cohort images must be inferred independently"
            calls.append(source)
            boxes = types.SimpleNamespace(xyxy=[], conf=[], cls=[])
            return [types.SimpleNamespace(path=source, boxes=boxes, masks=None, keypoints=None)]

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    predictions = _run_native(
        ModelSpec("model", checkpoint), cohort, 0.5, 0.1, None, False, {}
    )
    assert calls == [str(record.image_path) for record in cohort.records]
    assert list(predictions) == [record.image_id for record in cohort.records]


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
    with pytest.raises(DatasetValidationError, match="missing path"):
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

    def fake_inference(spec, cohort, *, thresholds, **kwargs):
        fake_inference.calls += 1
        values = {}
        for threshold in thresholds:
            values[float(threshold)] = {
                record.image_id: [
                    Prediction(int(annotation["class_id"]), .95, bbox=tuple(annotation["bbox"]))
                    for annotation in record.annotations
                ]
                for record in cohort.records
            }
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
    assert not list(tmp_path.glob(".comparison.building-*"))

    models.compare(
        dataset,
        split="val",
        progress=False,
        destination=destination,
    )
    assert fake_inference.calls == 1

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
