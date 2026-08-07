from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dataset_fixer import Model
from dataset_fixer.comparison.inference import (
    _postprocess_payload_predictions,
    _sahi_prediction,
    _semantic_probabilities,
    resolve_backend,
)
from dataset_fixer.comparison.types import Prediction
from dataset_fixer.sahi_support import (
    SahiSettings,
    build_tile_manifest,
    class_map_probabilities,
    resolve_sahi_settings,
    stitch_probability_tiles,
)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("slice_height", "sahi_slice_height"),
        ("slice_width", "sahi_slice_width"),
        ("overlap", "sahi_overlap"),
        ("overlap_height_ratio", "sahi_overlap_height_ratio"),
        ("overlap_width_ratio", "sahi_overlap_width_ratio"),
        ("postprocess_type", "sahi_postprocess_type"),
        ("postprocess_match_metric", "sahi_postprocess_match_metric"),
        ("postprocess_class_agnostic", "sahi_postprocess_class_agnostic"),
        ("model_type", "sahi_model_type"),
    ],
)
def test_explicit_api_rejects_auto_and_unprefixed_sahi_settings(
    tmp_path: Path,
    old: str,
    new: str,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    with pytest.raises(ValueError, match="auto.*removed"):
        Model(checkpoint, inference="auto")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=f"{old}->{new}"):
        Model(checkpoint, settings={old: 1})


def test_removed_sahi_mode_is_rejected(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    with pytest.raises(ValueError, match="sahi_mode was removed"):
        Model(checkpoint, settings={"sahi_mode": "combined"})


def test_sahi_settings_are_prefixed_and_axis_overrides_win() -> None:
    resolved = resolve_sahi_settings(
        {
            "sahi_slice_height": 320,
            "sahi_slice_width": 512,
            "sahi_overlap": 0.1,
            "sahi_overlap_height_ratio": 0.25,
            "sahi_postprocess_type": "nmm",
        },
        resolution=480,
    )
    assert resolved.slice_height == 320
    assert resolved.slice_width == 512
    assert resolved.overlap_height_ratio == pytest.approx(0.25)
    assert resolved.overlap_width_ratio == pytest.approx(0.1)
    assert resolved.postprocess_type == "NMM"
    assert all(key.startswith("sahi_") for key in resolved.as_dict())


def test_tile_manifest_and_feathered_stitching_cover_the_full_image() -> None:
    pytest.importorskip("sahi")
    settings = SahiSettings(4, 4, 0.5, 0.5, "GREEDYNMM", "IOS", False, "ultralytics")
    manifest = build_tile_manifest(width=6, height=4, settings=settings)
    assert [tile.box for tile in manifest] == [(0, 0, 4, 4), (2, 0, 6, 4)]
    tiles = []
    for index, tile in enumerate(manifest):
        probabilities = np.zeros((2, tile.height, tile.width), dtype=np.float32)
        probabilities[index] = 1.0
        tiles.append((tile, probabilities))
    stitched = stitch_probability_tiles(width=6, height=4, tiles=tiles)
    assert stitched.shape == (2, 4, 6)
    assert np.all(np.isfinite(stitched))
    assert np.allclose(stitched.sum(axis=0), 1.0)
    assert np.all(np.argmax(stitched, axis=0)[:, :2] == 0)
    assert np.all(np.argmax(stitched, axis=0)[:, -2:] == 1)


def test_class_map_fallback_preserves_multiclass_ids() -> None:
    class_map = np.asarray([[0, 2], [1, 2]], dtype=np.uint8)
    probabilities = class_map_probabilities(class_map, num_classes=4)
    assert probabilities.shape == (4, 2, 2)
    assert np.array_equal(np.argmax(probabilities, axis=0), class_map)


def test_single_channel_semantic_logits_become_binary_probabilities() -> None:
    result = types.SimpleNamespace(
        semantic_logits=np.asarray([[[[-2.0, 2.0], [-1.0, 1.0]]]], dtype=np.float32)
    )
    probabilities = _semantic_probabilities(
        result,
        expected_shape=(2, 2),
        num_classes=2,
        source="test",
    )
    assert probabilities.shape == (2, 2, 2)
    assert np.allclose(probabilities.sum(axis=0), 1.0)
    assert np.array_equal(
        np.argmax(probabilities, axis=0),
        np.asarray([[0, 1], [0, 1]]),
    )


def test_pose_nmm_confidence_weights_keypoints() -> None:
    settings = SahiSettings(32, 32, 0.2, 0.2, "GREEDYNMM", "IOU", False, "ultralytics")
    predictions = [
        Prediction(0, 0.9, bbox=(0, 0, 10, 10), keypoints=[(2, 2, 1.0)]),
        Prediction(0, 0.5, bbox=(1, 1, 11, 11), keypoints=[(6, 6, 0.5)]),
    ]
    merged = _postprocess_payload_predictions(
        predictions, task="pose", threshold=0.5, settings=settings
    )
    assert len(merged) == 1
    assert merged[0].bbox == (0, 0, 11, 11)
    assert merged[0].score == pytest.approx(0.9)
    expected = (2 * 0.9 + 6 * 0.25) / 1.15
    assert merged[0].keypoints[0][0] == pytest.approx(expected)
    assert merged[0].keypoints[0][1] == pytest.approx(expected)


def test_polo_nmm_preserves_and_averages_radius() -> None:
    settings = SahiSettings(32, 32, 0.2, 0.2, "NMM", "IOS", False, "ultralytics")
    predictions = [
        Prediction(0, 0.75, bbox=(0, 0, 10, 10), point=(5, 5), radius=5),
        Prediction(0, 0.25, bbox=(2, 2, 12, 12), point=(7, 7), radius=7),
    ]
    merged = _postprocess_payload_predictions(
        predictions, task="polo", threshold=0.5, settings=settings
    )
    assert len(merged) == 1
    assert merged[0].point == pytest.approx((5.5, 5.5))
    assert merged[0].radius == pytest.approx(5.5)


def test_sahi_segmentation_preserves_all_contours() -> None:
    value = types.SimpleNamespace(
        bbox=types.SimpleNamespace(to_xyxy=lambda: [0, 0, 10, 10]),
        score=types.SimpleNamespace(value=0.8),
        category=types.SimpleNamespace(id=0),
        mask=types.SimpleNamespace(
            segmentation=[
                [0, 0, 4, 0, 4, 4, 0, 4],
                [6, 6, 8, 6, 8, 8, 6, 8],
            ]
        ),
    )
    prediction = _sahi_prediction(value, "segment", "NMM", "IOU", 0.5)
    assert len(prediction.polygons) == 2
    assert prediction.polygon == prediction.polygons[0]


def test_sahi_ultralytics_detection_uses_source_coordinate_tiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("sahi")
    checkpoint = tmp_path / "detect.pt"
    checkpoint.write_bytes(b"model")
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 4), "white").save(image)

    class FakeYOLO:
        task = "detect"
        names = {0: "object"}

        def __init__(self, _: str) -> None:
            pass

        def predict(self, **kwargs: object):
            assert np.asarray(kwargs["source"]).shape == (4, 4, 3)
            boxes = types.SimpleNamespace(xyxy=[[1, 1, 3, 3]], conf=[0.9], cls=[0])
            return [types.SimpleNamespace(boxes=boxes, masks=None, keypoints=None)]

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    result = Model(checkpoint, task="detect").predict(
        image,
        inference="sahi",
        sahi_slice_height=4,
        sahi_slice_width=4,
        sahi_overlap=0,
        sahi_postprocess_type="NMS",
        progress=False,
    )
    assert result.backend == "sahi"
    assert [value.bbox for value in result.records[0].objects] == [
        (1.0, 1.0, 3.0, 3.0),
        (5.0, 1.0, 7.0, 3.0),
    ]


def test_sahi_ultralytics_semantic_stitches_class_maps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("sahi")
    checkpoint = tmp_path / "semantic.pt"
    checkpoint.write_bytes(b"model")
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 4), "white").save(image)
    calls = 0

    class FakeYOLO:
        task = "semantic"
        names = {0: "background", 1: "foreground"}

        def __init__(self, _: str) -> None:
            pass

        def predict(self, **_: object):
            nonlocal calls
            class_id = calls
            calls += 1
            semantic = types.SimpleNamespace(
                data=np.full((4, 4), class_id, dtype=np.uint8)
            )
            return [types.SimpleNamespace(semantic_mask=semantic)]

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    result = Model(checkpoint, task="semantic").predict(
        image,
        inference="sahi",
        sahi_slice_height=4,
        sahi_slice_width=4,
        sahi_overlap=0,
        progress=False,
    )
    mask = result.records[0].mask
    assert result.task == "semantic_segment"
    assert np.all(mask[:, :4] == 0)
    assert np.all(mask[:, 4:] == 1)


def test_native_ultralytics_semantic_preserves_multiclass_masks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "semantic.pt"
    checkpoint.write_bytes(b"model")
    image = tmp_path / "image.png"
    Image.new("RGB", (4, 3), "white").save(image)
    class_map = np.asarray(
        [[0, 1, 2, 2], [0, 1, 2, 2], [0, 0, 1, 2]], dtype=np.uint8
    )

    class FakeYOLO:
        task = "semantic"

        def __init__(self, _: str) -> None:
            pass

        def predict(self, **kwargs: object):
            return [
                types.SimpleNamespace(
                    path=kwargs["source"],
                    semantic_mask=types.SimpleNamespace(data=class_map),
                )
            ]

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    result = Model(checkpoint, task="semantic").predict(image, progress=False)
    assert result.backend == "native"
    assert result.task == "semantic_segment"
    assert np.array_equal(result.records[0].mask, class_map)
    saved = result.save(tmp_path / "saved")
    with Image.open(saved / "masks" / f"{result.records[0].image_id}.png") as opened:
        assert set(np.asarray(opened).reshape(-1)) == {0, 1, 2}


def test_missing_sahi_fails_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dataset_fixer.comparison.inference.sahi_available", lambda: False)
    with pytest.raises(ImportError, match=r"dataset-fixer\[sahi\]"):
        resolve_backend("sahi", "detect")
