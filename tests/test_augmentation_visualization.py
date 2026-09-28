from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import dataset_fixer as df
from dataset_fixer.models import Annotation, DatasetMetadata, Sample, Task
from dataset_fixer.visualization import render_annotated_sample
from conftest import make_yolo_dataset


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.setattr("dataset_fixer.convert.cache_root", lambda: tmp_path / "prepared-cache")
    root = make_yolo_dataset(
        tmp_path / "pose", task="pose", names=["animal"], size=(96, 96),
        train_rows=["0 .5 .5 .6 .6 .3 .4 2 .7 .6 2"] * 4,
        val_rows=["0 .5 .5 .6 .6 .3 .4 2 .7 .6 2"] * 2,
        extra={"kpt_shape": [2, 3], "flip_idx": [1, 0],
               "kpt_names": {0: ["nose", "tail"]}, "skeleton": [[1, 2]]},
    )
    return df.Dataset.open(root, progress=False)


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "base.pt"
    torch.save({"train_args": {"task": "pose", "model": "yolo26n-pose.pt", "imgsz": 96}}, path)
    return path


@pytest.mark.parametrize("source", ["path", "wandb", "model"])
def test_preview_uses_shared_weight_resolution(dataset, checkpoint, tmp_path, monkeypatch, source):
    import dataset_fixer.model_sources as sources
    from dataset_fixer.training import preview

    downloaded = []
    def download(reference, **kwargs):
        downloaded.append(reference)
        return checkpoint, None
    monkeypatch.setattr(sources, "_download_wandb", download)
    weights = {"path": checkpoint, "wandb": "wandb:team/project/run123",
               "model": df.Model(checkpoint, task="pose", resolution=64)}[source]
    seen = []
    original = preview._native_samples
    def capture(result, *args):
        seen.append((result.selection, result.config))
        return original(result, *args)
    monkeypatch.setattr(preview, "_native_samples", capture)
    output = tmp_path / f"{source}.png"
    result = dataset.visualize(weights=weights, samples=1, columns=1, panel_size=1.5,
                               destination=output, show=False)
    assert result is None and output.stat().st_size > 1000
    selection, config = seen[0]
    assert selection.weights == checkpoint and selection.family == df.ModelTypes.YOLO
    assert config.resolution == (64 if source == "model" else 96)
    assert downloaded == (["wandb:team/project/run123"] if source == "wandb" else [])
    if source == "model":
        assert not weights.loaded


def test_native_preview_preserves_options_sampling_paths_and_rng(dataset, tmp_path, monkeypatch):
    import dataset_fixer.visualization as visualization
    charts, options_seen, labels = [], [], []
    def finish(chart, options):
        charts.append(chart.to_dict())
        options_seen.append(options)
    monkeypatch.setattr(visualization, "finish_visualization", finish)
    style = df.MarkerStyle(radius=5, shape="diamond", show_boxes=False, color="cyan",
                           keypoint_colors={"nose": "yellow"}, show_labels=False)
    config = df.TrainingConfig(resolution=96, seed=99, workers=0)
    kwargs = dict(
        augmentations={"mosaic": 0, "augmentations": []}, config=config, split="val", samples=1,
        seed=15, columns=2, panel_size=2, zoom=True, context_fraction=.1, minimum_context=20,
        markers=style, line_width=2, outline_width=4, outline_alpha=.5, label_mode="wrap",
        label_fn=lambda path: labels.append(path) or path.stem, show=False,
        destination=tmp_path / "unused.svg",
    )
    # Initialize the native modules before checking their per-call random state.
    dataset.visualize(**kwargs)
    charts.clear()
    options_seen.clear()
    labels.clear()
    random.seed(109)
    np.random.seed(109)
    torch.manual_seed(109)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    for _ in range(2):
        assert dataset.visualize(**kwargs) is None
    assert charts[0] == charts[1]
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    expected = random.Random(15).sample([s for s in dataset._samples if s.split == "val"], 1)[0]
    assert labels == [expected.image_path, expected.image_path]
    assert config.seed == 99
    assert options_seen[0].markers is style
    for name in ("columns", "panel_size", "zoom", "context_fraction", "minimum_context", "line_width",
                 "outline_width", "outline_alpha", "label_mode", "destination", "show"):
        assert getattr(options_seen[0], name) == kwargs[name]
    assert len(charts[0]["vconcat"][0]["hconcat"]) == 1


def test_none_weights_keeps_original_view_and_none_samples_selects_all(dataset, monkeypatch):
    import dataset_fixer.visualization as visualization
    calls = []
    monkeypatch.setattr("dataset_fixer.training.selection.resolve_model_source", lambda *a, **kw: pytest.fail("resolved weights=None"))
    monkeypatch.setattr(visualization, "finish_visualization", lambda *args: None)
    dataset.visualize(weights=None, split=None, samples=None, show=False,
                      label_fn=lambda path: calls.append(path) or None)
    assert calls == [sample.image_path for sample in dataset._samples]
    calls.clear()
    dataset.visualize(augmentations={}, config=df.TrainingConfig(resolution=96, workers=0),
                      split=None, samples=None, show=False, label_fn=lambda path: calls.append(path) or None)
    assert calls == [sample.image_path for sample in dataset._samples]
    assert not hasattr(df, "preview_augmentations")


@pytest.mark.parametrize("shape", ["circle", "square", "diamond", "cross"])
def test_marker_style_pixels_visibility_and_shared_prediction_renderer(tmp_path, shape):
    from dataset_fixer.comparison.reporting import _draw_panel
    image = Image.new("RGB", (120, 100), "#303030")
    annotation = Annotation(0, bbox=(10, 10, 110, 90), keypoints=[(35, 50, 2), (85, 50, 2)])
    sample = Sample(tmp_path / "unwritten.png", Path("example.png"), "train", 120, 100, [annotation])
    metadata = DatasetMetadata(names={0: "animal"}, kpt_names={0: ["nose", "tail"]}, extra={"skeleton": [[1, 2]]})
    style = df.MarkerStyle(radius=7, shape=shape, color="#ff00ff", show_boxes=False,
                           show_skeleton=False, show_labels=False, keypoint_colors={0: "cyan"})
    rendered = render_annotated_sample(sample, Task.POSE, metadata, image=image, markers=style, line_width=2)
    assert rendered.getpixel((35, 50)) == (0, 255, 255)
    assert rendered.getpixel((85, 50)) == (255, 0, 255)
    assert rendered.getpixel((10, 10)) == image.getpixel((10, 10))
    assert rendered.getpixel((60, 50)) == image.getpixel((60, 50))
    comparison = _draw_panel(image, [annotation], truth=True, markers=style, linewidth=2)
    assert np.array_equal(np.asarray(comparison), np.asarray(rendered))
    invisible = render_annotated_sample(sample, Task.POSE, metadata, image=image,
                                        markers=replace(style, show_keypoints=False))
    assert np.array_equal(np.asarray(invisible), np.asarray(image))
    skeleton = render_annotated_sample(sample, Task.POSE, metadata, image=image,
                                      markers=replace(style, show_keypoints=False, show_skeleton=True))
    assert skeleton.getpixel((60, 50)) == (255, 0, 255)
    if shape != "cross":
        hollow = render_annotated_sample(sample, Task.POSE, metadata, image=image,
                                         markers=replace(style, fill=False))
        assert hollow.getpixel((35, 50)) == image.getpixel((35, 50))


@pytest.mark.parametrize("options", [dict(radius=-1), dict(radius=float("nan")), dict(shape="triangle"),
                                     dict(show_boxes=1), dict(color="not-a-color"), dict(label_size=0)])
def test_invalid_marker_styles_fail_before_rendering(options):
    with pytest.raises((ValueError, TypeError)):
        df.MarkerStyle(**options)
