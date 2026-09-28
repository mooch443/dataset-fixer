"""Decoded videos must use the same prediction/cache/rendering path as images."""
from __future__ import annotations

import csv
import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
import torch
from PIL import Image

import dataset_fixer as df
from dataset_fixer.comparison.types import Prediction


def write_video(path, *, timestamps=(0, 200, 400, 600, 800), audio=False, size=(128, 96), rotation=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as output:
        video = output.add_stream("libx264", rate=5)
        video.width, video.height = size
        video.pix_fmt = "yuv420p" if size[0] % 2 == size[1] % 2 == 0 else "yuv444p"
        video.time_base = video.codec_context.time_base = Fraction(1, 1000)
        if rotation:
            video.set_display_rotation(rotation)
        sound = output.add_stream("aac", rate=48000) if audio else None
        if sound:
            sound.layout = "stereo"
        for index, timestamp in enumerate(timestamps):
            pixels = np.full((size[1], size[0], 3), (30 + index * 10, 40, 80), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = timestamp, Fraction(1, 1000)
            for packet in video.encode(frame):
                output.mux(packet)
        for packet in video.encode():
            output.mux(packet)
        if sound:
            samples = np.sin(np.arange(48000) * 2 * np.pi * 440 / 48000).astype(np.float32) * .1
            frame = av.AudioFrame.from_ndarray(np.stack([samples, samples]), format="fltp", layout="stereo")
            frame.sample_rate, frame.pts, frame.time_base = 48000, 0, Fraction(1, 48000)
            for packet in sound.encode(frame):
                output.mux(packet)
            for packet in sound.encode():
                output.mux(packet)
    return path


@pytest.fixture
def source(tmp_path):
    return write_video(tmp_path / "input.mp4")


def make_model(tmp_path, task="pose", family="ultralytics"):
    root = tmp_path / family
    root.mkdir(exist_ok=True)
    if family == "nnunet":
        (root / "fold_0").mkdir()
        (root / "dataset.json").write_text(json.dumps({"channel_names": {"0": "R", "1": "G", "2": "B"},
                                                      "labels": {"background": 0, "foreground": 1}, "file_ending": ".png"}))
        (root / "plans.json").write_text(json.dumps({"experiment_planner_used": "nnUNetPlannerResEncM"}))
        torch.save({"network_weights": {}}, root / "fold_0/checkpoint_final.pth")
        return df.Model(root, device="cpu")
    path = root / "model.pt"
    if family == "rfdetr":
        config = {"resolution": 96, "patch_size": 16, "num_windows": 2,
                  "use_grouppose_keypoints": task == "pose", "segmentation_head": task == "segment"}
        torch.save({"model_name": "RFDETRNano", "model_config": config}, path)
    else:
        torch.save({"train_args": {"task": task, "imgsz": 96}}, path)
    path.with_suffix(".json").write_text(json.dumps({"dataset_schema": {
        "classes": {"0": "wolf"}, "kpt_names": {"0": ["nose", "tail", "ear"]}, "skeleton": [[1, 2]]}}))
    return df.Model(path, device="cpu")


@pytest.fixture
def adapter(monkeypatch):
    calls = []
    def predict(model, inputs, **kwargs):
        calls.append((inputs, kwargs))
        result = {}
        for item in inputs:
            assert item.image_path.is_file()
            index = int(item.image_id.removeprefix("frame_"))
            points = [(30., 30., .8), (80., 60., .9)] + ([(50., 30., .5)] if index >= 2 else [])
            kwargs_obj = dict(bbox=(20., 15., 100., 80.))
            if model.task == "pose":
                kwargs_obj["keypoints"] = points
            elif model.task == "segment":
                kwargs_obj["polygons"] = [[(20., 15.), (100., 15.), (100., 80.), (20., 80.)]]
            elif model.task == "polo":
                kwargs_obj.update(point=(60., 45.), radius=10.)
            result[item.image_id] = [] if index == 1 else [Prediction(0, .95, **kwargs_obj)]
        return result, model.task, {}
    monkeypatch.setattr("dataset_fixer.comparison.inference.predict_model_inputs", predict)
    return calls


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize(("family", "task"), [
    ("ultralytics", "detect"), ("ultralytics", "pose"), ("ultralytics", "segment"), ("ultralytics", "polo"),
    ("rfdetr", "detect"), ("rfdetr", "pose"), ("rfdetr", "segment"),
])
def test_video_uses_canonical_inference_and_exports_all_frames(tmp_path, source, adapter, family, task):
    model = make_model(tmp_path, task, family)
    style = df.MarkerStyle(radius=6, shape="diamond", color="magenta", show_labels=False)
    result = model.predict(source, destination=tmp_path / "predictions", markers=style,
                           batch_size=2, confidence=.12, postprocess=.6, resolution=96,
                           prediction_cache=False, progress=False)
    assert isinstance(result, df.VideoPredictionResult) and len(result) == 1
    record = result[0]
    assert record.frames == 5 and record.objects == 4 and record.fps == 5
    assert result.task == task and record.source == source
    assert [len(inputs) for inputs, _ in adapter] == [2, 2, 1]
    assert all(options["confidence"] == .12 and options["postprocess"] == .6 for _, options in adapter)
    assert not any(item.image_path.exists() for inputs, _ in adapter for item in inputs)
    rows = read_csv(record.csv_path)
    assert [int(row["frame"]) for row in rows] == list(range(5))
    assert [float(row["timestamp_seconds"]) for row in rows] == pytest.approx([0, .2, .4, .6, .8])
    assert rows[1]["record_type"] == "empty" and rows[1]["object_index"] == ""
    assert rows[0]["class_name"] == "wolf" and float(rows[0]["x_min"]) == 20
    if task == "pose":
        assert float(rows[0]["keypoint_0_x"]) == 30
        assert rows[0]["keypoint_2_x"] == "" and float(rows[2]["keypoint_2_x"]) == 50
    if task == "segment":
        assert json.loads(rows[0]["polygons_json"])[0][0] == [20, 15]
    with av.open(str(record.video_path)) as video:
        frames = list(video.decode(video=0))
        assert len(frames) == 5 and (frames[0].width, frames[0].height) == (128, 96)
        assert [float(frame.time) for frame in frames] == pytest.approx([0, .2, .4, .6, .8])
        frames[0].to_image().save(tmp_path / f"review-{family}-{task}.png")
    manifest = json.loads((result.location / "manifest.json").read_text())
    assert manifest["markers"]["shape"] == "diamond"
    assert result.summary()["frames"] == 5


@pytest.mark.parametrize("suffix", [".mp4", ".mov", ".mkv"])
def test_video_preserves_audio_and_variable_frame_timing(tmp_path, adapter, suffix):
    source = write_video(tmp_path / f"variable{suffix}", timestamps=(0, 100, 400, 600, 900), audio=True)
    model = make_model(tmp_path)
    result = model.predict(source, batch_size=2, prediction_cache=False, progress=False)
    assert result[0].audio and result.location == tmp_path / "variable-predictions"
    rows = read_csv(result[0].csv_path)
    assert [float(row["timestamp_seconds"]) for row in rows] == pytest.approx([0, .1, .4, .6, .9])
    with av.open(str(result[0].video_path)) as output:
        assert output.streams.audio[0].codec_context.name == "aac"
        frames = list(output.decode(video=0))
        assert [float(frame.time) for frame in frames] == pytest.approx([0, .1, .4, .6, .9], abs=.001)
        with av.open(str(source)) as original:
            stream = original.streams.video[0]
            if stream.duration is not None:
                expected_duration = stream.duration * stream.time_base
            else:
                original_frames = list(original.decode(video=0))
                expected_duration = (original_frames[-1].pts + original_frames[-1].duration - original_frames[0].pts) * original_frames[-1].time_base
        assert float(output.streams.video[0].duration * output.streams.video[0].time_base) == pytest.approx(float(expected_duration), abs=.001)
    with av.open(str(result[0].video_path)) as output:
        sound = list(output.decode(audio=0))
        assert sum(frame.samples for frame in sound) >= 48000
        assert max(float(np.max(np.abs(frame.to_ndarray()))) for frame in sound) > .05


def test_video_reuses_verified_frame_cache(tmp_path, source, adapter):
    model = make_model(tmp_path)
    options = dict(batch_size=2, prediction_cache=tmp_path / "cache", progress=False)
    first = model.predict(source, destination=tmp_path / "first", **options)
    second = model.predict(source, destination=tmp_path / "second", cache_only=True, **options)
    assert len(adapter) == 3
    assert first[0].csv_path.read_text() == second[0].csv_path.read_text()
    with pytest.raises(df.PredictionCacheMissError):
        model.predict(source, destination=tmp_path / "miss", cache_only=True, confidence=.9, **options)
    assert not (tmp_path / "miss").exists()


def test_rfdetr_pose_cache_retains_nonprobability_scores(tmp_path, source, monkeypatch):
    model = make_model(tmp_path, family="rfdetr")
    def predict(model, inputs, **kwargs):
        objects = [Prediction(0, 2.73, bbox=(20., 15., 100., 80.), keypoints=[(30., 30., .9)],
                              metadata={"backend": "rfdetr", "score_domain": "nonnegative"})]
        return {item.image_id: objects for item in inputs}, "pose", {}
    monkeypatch.setattr("dataset_fixer.comparison.inference.predict_model_inputs", predict)
    options = dict(batch_size=2, prediction_cache=tmp_path / "cache", progress=False)
    original = model.predict(source, destination=tmp_path / "original", **options)
    monkeypatch.setattr("dataset_fixer.comparison.inference.predict_model_inputs",
                        lambda *a, **kw: pytest.fail("cache replay ran inference"))
    cached = model.predict(source, destination=tmp_path / "replayed", cache_only=True, **options)
    assert original[0].csv_path.read_text() == cached[0].csv_path.read_text()
    assert all(float(row["confidence"]) == 2.73 for row in read_csv(cached[0].csv_path))


@pytest.mark.parametrize(("score", "metadata"), [(1.1, {}), (-1, {"score_domain": "nonnegative"}),
                                                (float("inf"), {"score_domain": "nonnegative"}),
                                                (float("nan"), {"score_domain": "nonnegative"})])
def test_cached_scores_still_reject_invalid_domains(score, metadata):
    from dataset_fixer.prediction_cache import _prediction_from_json
    with pytest.raises(ValueError, match="score domain"):
        _prediction_from_json({"class_id": 0, "score": score, "metadata": metadata})


def test_multiple_videos_names_overwrite_protection_and_collection(tmp_path, adapter):
    source = tmp_path / "clips"
    paths = [write_video(source / name / "same.clip.mp4", timestamps=(0,)) for name in ("a", "b")]
    model = make_model(tmp_path)
    output = tmp_path / "outputs"
    result = model.predict(source, destination=output, prediction_cache=False, progress=False)
    assert [record.source for record in result] == paths
    assert len({record.video_path.name for record in result}) == 2
    assert all("same.clip-" in record.video_path.name for record in result)
    with pytest.raises(FileExistsError):
        model.predict(source, destination=output, progress=False)
    collection = df.Model.load_many({"first": model, "second": model})
    results = collection.predict(paths, destination=tmp_path / "collection", progress=False, prediction_cache=False,
                                 markers=df.MarkerStyle(show_boxes=False))
    assert list(results) == ["first", "second"]
    assert results["first"][0].video_path != results["second"][0].video_path


def test_failed_video_never_publishes_partial_outputs(tmp_path, source, adapter, monkeypatch):
    model = make_model(tmp_path)
    original = model.predict
    def fail(source, **kwargs):
        if not isinstance(source, Path) and source[0].image_id == "frame_000000002":
            raise RuntimeError("inference failed")
        return original(source, **kwargs)
    monkeypatch.setattr(model, "predict", fail)
    with pytest.raises(RuntimeError, match="inference failed"):
        model.predict(source, destination=tmp_path / "failed", batch_size=2, prediction_cache=False, progress=False)
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".video-predictions-*"))
    with pytest.raises(ValueError, match="mixed sources"):
        original([source, tmp_path / "image.png"], progress=False)


def test_semantic_video_exports_foreground_counts(tmp_path, source, monkeypatch):
    model = make_model(tmp_path, family="nnunet")
    def predict(model, inputs, **kwargs):
        mask = np.zeros((96, 128), dtype=np.uint8)
        mask[20:60, 30:80] = 1
        return tuple(df.ImagePrediction(item.image_id, item.image_path, item.relative_path, item.width, item.height,
                                        mask=mask, foreground_probability=mask.astype(np.float16)) for item in inputs)
    monkeypatch.setattr("dataset_fixer.semantic_comparison.predict_nnunet_model", predict)
    result = model.predict(source, destination=tmp_path / "semantic", batch_size=2,
                           prediction_cache=False, progress=False, prediction_threshold=.5)
    assert result.task == "semantic_segment"
    assert all(row["foreground_pixels"] == "2000" for row in read_csv(result[0].csv_path))
    with av.open(str(result[0].video_path)) as video:
        next(video.decode(video=0)).to_image().save(tmp_path / "review-semantic.png")


def test_odd_frame_dimensions_and_invalid_video(tmp_path, adapter):
    source = write_video(tmp_path / "odd.mp4", size=(129, 97), timestamps=(0,))
    model = make_model(tmp_path)
    result = model.predict(source, prediction_cache=False, progress=False)
    with av.open(str(result[0].video_path)) as video:
        frame = next(video.decode(video=0))
        assert (frame.width, frame.height) == (129, 97)
    invalid = tmp_path / "invalid.mp4"
    invalid.write_bytes(b"not a video")
    with pytest.raises(av.error.InvalidDataError):
        model.predict(invalid, prediction_cache=False, progress=False)
    assert not (tmp_path / "invalid-predictions").exists()


def test_rotated_video_uses_display_coordinates(tmp_path, adapter):
    if not hasattr(av.video.stream.VideoStream, "set_display_rotation"):
        pytest.skip("PyAV cannot create a display-matrix fixture")
    source = write_video(tmp_path / "portrait.mp4", rotation=90, timestamps=(0,))
    model = make_model(tmp_path)
    result = model.predict(source, prediction_cache=False, progress=False)
    assert (adapter[0][0][0].width, adapter[0][0][0].height) == (96, 128)
    with av.open(str(result[0].video_path)) as output:
        frame = next(output.decode(video=0))
        assert frame.rotation == 0 and (frame.width, frame.height) == (96, 128)


def test_multiple_objects_and_semantic_projection_use_same_predictions(tmp_path, source, monkeypatch):
    model = make_model(tmp_path, task="segment")
    objects = [Prediction(0, .9, polygons=[[(10, 10), (30, 10), (30, 30), (10, 30)]]),
               Prediction(0, .8, polygons=[[(70, 50), (90, 50), (90, 70), (70, 70)]])]
    def predict(model, inputs, **kwargs):
        assert kwargs["backend"] == "sahi"
        return {item.image_id: objects for item in inputs}, "segment", {}
    monkeypatch.setattr("dataset_fixer.comparison.inference.predict_model_inputs", predict)
    options = dict(inference="sahi", prediction_cache=False, progress=False)
    native = model.predict(source, destination=tmp_path / "objects", **options)
    rows = read_csv(native[0].csv_path)
    assert len(rows) == 10 and [row["object_index"] for row in rows] == ["0", "1"] * 5
    output = tmp_path / "projected"
    output.mkdir()  # Existing empty destinations are supported.
    semantic = model.predict(source, destination=output, output_task="semantic_segment", **options)
    assert semantic.task == "semantic_segment"
    assert all(int(row["foreground_pixels"]) > 0 for row in read_csv(semantic[0].csv_path))
