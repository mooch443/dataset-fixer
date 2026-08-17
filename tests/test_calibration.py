from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from dataset_fixer import (
    Dataset,
    ImagePrediction,
    Model,
    PredictionCacheMissError,
    PredictionResult,
    calibrate_prediction_thresholds,
)
from dataset_fixer.calibration import _case_row, _summarize_rows
from conftest import make_yolo_dataset


def test_area_filtered_component_f1_requires_one_to_one_overlap() -> None:
    reference = np.zeros((10, 10), dtype=bool)
    reference[2, 1:7] = True  # retained reference touched by two predictions
    reference[7:9, 7:9] = True  # retained and missed
    reference[0, 9] = True  # excluded tiny reference

    prediction = np.zeros_like(reference)
    prediction[1:3, 1:3] = True  # retained; touches first reference
    prediction[1:3, 5:7] = True  # retained duplicate on first reference
    prediction[5:7, 0:2] = True  # retained false positive
    prediction[9, 0] = True  # excluded tiny prediction

    row = _case_row(reference, prediction, minimum_component_area=4)
    metrics = _summarize_rows((row,), minimum_component_area=4)

    assert row["component_reference_count"] == 2
    assert row["component_prediction_count"] == 3
    assert row["component_match_count"] == 1
    assert metrics["area_filtered_component_precision"] == pytest.approx(1 / 3)
    assert metrics["area_filtered_component_recall"] == pytest.approx(1 / 2)
    assert metrics["area_filtered_component_f1"] == pytest.approx(0.4)
    assert metrics["foreground_precision"] == pytest.approx(
        row["tp"] / (row["tp"] + row["fp"])
    )
    # Image presence is still reported separately and does not require overlap.
    assert metrics["area_filtered_image_presence_f1"] == pytest.approx(1.0)


def test_grouped_threshold_calibration_uses_cached_probabilities_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_yolo_dataset(
        tmp_path / "segments",
        task="segment",
        names=["school"],
        train_rows=["0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"],
        val_rows=[
            "0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8",
            "0 0.3 0.3 0.7 0.3 0.7 0.7 0.3 0.7",
        ],
        size=(20, 20),
    )
    dataset = Dataset.open(source, task="segment", progress=False).export(
        destination=tmp_path / "semantic",
        format="semantic_masks",
        visualize=False,
        progress=False,
    )
    checkpoint = tmp_path / "semantic.pt"
    checkpoint.write_bytes(b"semantic")
    model = Model(checkpoint, task="semantic", prediction_threshold=0.4)
    calls: list[dict[str, object]] = []

    def cached_predict(_source: object, **options: object) -> PredictionResult:
        calls.append(options)
        records = []
        for sample in dataset._samples:
            if sample.split != "val":
                continue
            mask_path = dataset._mask_paths[sample.image_path.resolve()]
            with Image.open(mask_path) as opened:
                reference = np.asarray(opened.copy()) > 0
            probability = np.where(reference, 0.7, 0.45).astype(np.float32)
            records.append(
                ImagePrediction(
                    image_id=sample.image_path.stem,
                    image_path=sample.image_path.resolve(),
                    relative_path=sample.image_path.name,
                    width=20,
                    height=20,
                    mask=(probability >= 0.4).astype(np.uint8),
                    foreground_probability=probability,
                )
            )
        # Use the exact normalized IDs/order expected by calibration.
        from dataset_fixer.model import normalize_model_inputs

        inputs, _, _ = normalize_model_inputs(dataset, split="val", progress=False)
        by_path = {record.image_path: record for record in records}
        normalized = tuple(
            ImagePrediction(
                image_id=value.image_id,
                image_path=value.image_path,
                relative_path=value.relative_path,
                width=value.width,
                height=value.height,
                mask=by_path[value.image_path].mask,
                foreground_probability=by_path[
                    value.image_path
                ].foreground_probability,
            )
            for value in inputs
        )
        return PredictionResult(
            model_name=model.name,
            model_kind="ultralytics",
            task="semantic_segment",
            backend="sahi",
            records=normalized,
            inference_seconds=0.0,
            cache_info={"status": "hit", "location": str(tmp_path / "cache")},
        )

    monkeypatch.setattr(model, "predict", cached_predict)
    result = calibrate_prediction_thresholds(
        (model,),
        dataset,
        split="val",
        group_by=lambda path: path.stem,
        thresholds=(0.4, 0.5, 0.8),
        folds=2,
        destination=tmp_path / "calibration",
        progress=False,
    )

    assert result.recommendations[model.name] == pytest.approx(0.5)
    for table in (
        result.cache_audit,
        result.threshold_scores,
        result.fold_scores,
        result.improvements,
    ):
        assert isinstance(table, pd.DataFrame)
        assert isinstance(table.index, pd.RangeIndex)
    assert calls == [
        {
            "split": "val",
            "prediction_cache": True,
            "cache_only": True,
            "require_probability_maps": True,
            "progress": False,
        }
    ]
    assert (result.location / "recommendations.json").is_file()
    assert (result.location / "threshold-curves.png").is_file()
    assert (result.location / "calibration-improvements.csv").is_file()
    improvement = result.improvements.iloc[0]
    assert improvement["baseline_threshold"] == pytest.approx(0.4)
    assert improvement["recommended_threshold"] == pytest.approx(0.5)
    assert improvement["cv_macro_dice_gain"] > 0
    assert improvement["cv_macro_dice_relative_gain_pct"] > 0

    repeated = calibrate_prediction_thresholds(
        (model,),
        dataset,
        split="val",
        group_by=lambda path: path.stem,
        thresholds=(0.4, 0.5, 0.8),
        folds=2,
        destination=tmp_path / "calibration-repeated",
        progress=False,
    )
    assert len(calls) == 1
    assert repeated.cache_audit.iloc[0]["cache"] == "score-hit"
    assert repeated.cache_audit.iloc[0]["cached_thresholds"] == 3
    assert repeated.cache_audit.iloc[0]["scored_thresholds"] == 0

    extended = calibrate_prediction_thresholds(
        (model,),
        dataset,
        split="val",
        group_by=lambda path: path.stem,
        thresholds=(0.4, 0.5, 0.6, 0.8),
        folds=2,
        destination=tmp_path / "calibration-extended",
        progress=False,
    )
    assert len(calls) == 2
    assert extended.cache_audit.iloc[0]["cached_thresholds"] == 3
    assert extended.cache_audit.iloc[0]["scored_thresholds"] == 1


def test_probability_map_rerun_requires_explicit_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_yolo_dataset(
        tmp_path / "segments-rerun",
        task="segment",
        names=["school"],
        train_rows=[""],
        val_rows=[
            "0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8",
            "0 0.3 0.3 0.7 0.3 0.7 0.7 0.3 0.7",
        ],
        size=(16, 16),
    )
    dataset = Dataset.open(source, task="segment", progress=False).export(
        destination=tmp_path / "semantic-rerun",
        format="semantic_masks",
        visualize=False,
        progress=False,
    )
    checkpoint = tmp_path / "semantic-rerun.pt"
    checkpoint.write_bytes(b"semantic")
    model = Model(checkpoint, task="semantic", prediction_threshold=0.5)
    from dataset_fixer.model import normalize_model_inputs

    inputs, _, _ = normalize_model_inputs(dataset, split="val", progress=False)
    calls: list[bool] = []

    def missing_then_fresh(_source: object, **options: object) -> PredictionResult:
        cache_only = bool(options["cache_only"])
        calls.append(cache_only)
        if cache_only:
            raise PredictionCacheMissError(
                "probabilities absent",
                reason="missing-probability-maps",
            )
        records = []
        for value in inputs:
            assert value.mask_path is not None
            with Image.open(value.mask_path) as opened:
                reference = np.asarray(opened.copy()) > 0
            probability = np.where(reference, 0.8, 0.2).astype(np.float32)
            records.append(
                ImagePrediction(
                    image_id=value.image_id,
                    image_path=value.image_path,
                    relative_path=value.relative_path,
                    width=value.width,
                    height=value.height,
                    mask=(probability >= 0.5).astype(np.uint8),
                    foreground_probability=probability,
                )
            )
        return PredictionResult(
            model_name=model.name,
            model_kind="ultralytics",
            task="semantic_segment",
            backend="sahi",
            records=tuple(records),
            inference_seconds=1.0,
            cache_info={"status": "fresh", "location": str(tmp_path / "cache")},
        )

    monkeypatch.setattr(model, "predict", missing_then_fresh)
    unavailable = calibrate_prediction_thresholds(
        (model,),
        dataset,
        group_by=lambda path: path.stem,
        thresholds=(0.4, 0.5),
        folds=2,
        destination=tmp_path / "no-rerun",
        progress=False,
    )
    assert unavailable.recommendations == {}
    assert calls == [True]

    calls.clear()
    rerun = calibrate_prediction_thresholds(
        (model,),
        dataset,
        group_by=lambda path: path.stem,
        thresholds=(0.4, 0.5),
        folds=2,
        rerun_missing_probability_maps=True,
        destination=tmp_path / "with-rerun",
        progress=False,
    )
    assert rerun.recommendations[model.name] == pytest.approx(0.5)
    assert calls == [True, False]
    assert rerun.cache_audit.iloc[0]["cache"] == "rerun"
