from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from dataset_fixer import ComparisonResult, Dataset
from dataset_fixer.comparison.cache import (
    import_notebook_cache,
    load_package_cache,
    model_hash,
    notebook_cache_basename,
    notebook_dataset_hash,
    restricted_pickle_load,
    save_package_cache,
    write_notebook_numpy_cache,
)
from dataset_fixer.comparison.cohort import freeze_cohort
from dataset_fixer.comparison.inference import resolve_backend
from dataset_fixer.comparison.metrics import evaluate_configuration, optimal_match
from dataset_fixer.comparison.types import Prediction
from PIL import Image


def test_cohort_is_ordered_and_content_addressed(detect_dataset: Path) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    first = freeze_cohort(dataset, "val")
    second = freeze_cohort(dataset, "val")
    assert first.fingerprint == second.fingerprint
    assert [record.relative_path for record in first.records] == sorted(record.relative_path for record in first.records)
    label = detect_dataset / "labels" / "val" / "val_0.txt"
    label.write_text("0 0.4 0.5 0.25 0.25\n", encoding="utf-8")
    changed = freeze_cohort(Dataset.open(detect_dataset, task="detect", progress=False), "val")
    assert changed.fingerprint != first.fingerprint


def test_package_cache_is_pickle_free_and_detects_corruption(detect_dataset: Path, tmp_path: Path) -> None:
    cohort = freeze_cohort(Dataset.open(detect_dataset, task="detect", progress=False), "val")
    predictions = {
        0.5: {
            record.image_id: [Prediction(0, 0.9, bbox=(1, 2, 20, 30))]
            for record in cohort.records
        }
    }
    root = tmp_path / "cache"
    save_package_cache(root, cohort, {"setting": 1}, predictions)
    loaded, shards, complete = load_package_cache(root, cohort, (0.5,))
    assert complete and shards == len(cohort.records)
    assert loaded[0.5][cohort.records[0].image_id][0].bbox == pytest.approx((1, 2, 20, 30))
    shard = next((root / "images").rglob("*.npz"))
    shard.write_bytes(b"broken")
    loaded, _, complete = load_package_cache(root, cohort, (0.5,))
    assert not complete and 0.5 not in loaded


def test_restricted_pickle_rejects_globals(tmp_path: Path) -> None:
    class Evil:
        def __reduce__(self):
            return os.system, ("echo forbidden",)

    path = tmp_path / "malicious.gridcache.pkl"
    path.write_bytes(pickle.dumps(Evil()))
    with pytest.raises(pickle.UnpicklingError, match="forbidden"):
        restricted_pickle_load(path)


def test_notebook_v2_round_trip_is_content_verified(tmp_path: Path) -> None:
    split_root = tmp_path / "val"
    (split_root / "images").mkdir(parents=True)
    (split_root / "labels").mkdir()
    image = split_root / "images" / "fruit.jpg"
    Image.new("RGB", (100, 80), (20, 30, 40)).save(image)
    (split_root / "labels" / "fruit.txt").write_text("0 15 0.5 0.5\n", encoding="utf-8")
    dataset = Dataset.open(split_root, task="polo", names=["fruit"], radii={0: 15}, progress=False)
    cohort = freeze_cohort(dataset, "train")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    predictions = {
        .75: {cohort.records[0].image_id: [Prediction(0, .9, bbox=(35, 25, 65, 55), point=(50, 40))]}
    }
    key = {
        "cache_version": 2,
        "cache_format": "gridcache_v2_numpy_sharded",
        "model_path": str(checkpoint.resolve()),
        "model_hash": model_hash(checkpoint),
        "dataset_root": str(split_root.resolve()),
        "dataset_hash": notebook_dataset_hash(split_root),
        "resolution": 480,
        "min_conf": .35,
        "iou_list": [.75],
        "device": "None",
        "overlap_height_ratio": .2,
        "overlap_width_ratio": .2,
        "postprocess_class_agnostic": False,
        "model_type": "ultralytics",
    }
    cache_dir = tmp_path / "legacy"
    target = cache_dir / notebook_cache_basename(checkpoint, key, numpy=True)
    write_notebook_numpy_cache(target, key=key, cohort=cohort, predictions=predictions)
    restored, audit = import_notebook_cache(
        [cache_dir], model_sha256=model_hash(checkpoint), resolution=480,
        confidence_floor=.35, thresholds=(.75,), cohort=cohort,
        expected_key={
            "device": "None", "overlap_height_ratio": .2, "overlap_width_ratio": .2,
            "postprocess_class_agnostic": False, "model_type": "ultralytics",
        },
    )
    assert audit and audit["verified"] is True
    assert restored[.75][cohort.records[0].image_id][0].point == pytest.approx((50, 40))


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


def test_pose_never_resolves_to_sahi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dataset_fixer.comparison.inference.sahi_available", lambda: True)
    assert resolve_backend("auto", "pose") == "native"
    with pytest.raises(ValueError, match="SAHI"):
        resolve_backend("sahi", "pose")


def test_compare_models_facade_atomic_result(
    detect_dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")

    def fake_inference(spec, cohort, *, thresholds, **kwargs):
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

    monkeypatch.setattr("dataset_fixer.comparison.engine.run_inference", fake_inference)
    destination = tmp_path / "comparison"
    result = dataset.compare_models(
        {"baseline": checkpoint},
        split="val",
        baseline="baseline",
        inference="native",
        training_provenance="ignore",
        confidence_thresholds=(0.5,),
        postprocess_thresholds=(0.5,),
        cache=False,
        visualize=False,
        progress=False,
        destination=destination,
        bootstrap_resamples=20,
    )
    assert isinstance(result, ComparisonResult)
    assert result.cohort_verified
    assert result.ranking[0]["score"] == pytest.approx(1.0)
    assert (destination / "model-comparison.json").is_file()
    assert (destination / "metrics" / "ranking.csv").is_file()
    assert not list(tmp_path.glob(".comparison.building-*"))


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
    dataset.compare_models(
        {"visual": checkpoint}, inference="native", training_provenance="ignore",
        confidence_thresholds=(.5,), postprocess_thresholds=(.5,), cache=False,
        visualize=True, progress=False, destination=destination, bootstrap_resamples=20,
    )
    for name in ("ranking_forest", "precision_recall", "f1_confidence", "cohort_composition"):
        assert (destination / "figures" / f"{name}.pdf").is_file()
        assert (destination / "figures" / f"{name}.svg").is_file()
        assert (destination / "figures" / f"{name}.png").is_file()
        assert (destination / "figures" / "data" / f"{name}.csv").is_file()
        assert (destination / "figures" / "metadata" / f"{name}.json").is_file()
