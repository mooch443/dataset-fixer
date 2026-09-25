"""RF-DETR optimization must preserve predictions, cached runtimes and weights."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import dataset_fixer as df
from dataset_fixer.model import ModelInput
from dataset_fixer.rfdetr_engine import predict_inputs


@pytest.mark.parametrize(("device", "cuda_available", "resolved", "dtype"), [
    (None, False, "cpu", torch.float32),
    (None, True, "cuda", torch.float16),
    ("cuda:1", True, "cuda:1", torch.float16),
    ("cpu", True, "cpu", torch.float32),
    ("mps", False, "mps", torch.float32),
])
def test_optimization_device_cache_and_inference_mode(tmp_path, monkeypatch, device, cuda_available, resolved, dtype):
    import rfdetr

    checkpoint = tmp_path / "model.pth"
    torch.save({"model_name": "RFDETRNano", "model_config": {"patch_size": 16, "num_windows": 2}}, checkpoint)
    original = checkpoint.read_bytes()
    model = df.Model(checkpoint, kind="rfdetr", task="detect")
    image = ModelInput("image", tmp_path / "image.jpg", 96, 96, "image.jpg")
    loaded, optimized = [], []

    def load(path, **overrides):
        assert path == str(checkpoint)
        assert overrides["device"] == resolved
        native = SimpleNamespace(model=SimpleNamespace(), model_config=SimpleNamespace(use_grouppose_keypoints=False))

        def optimize(**kwargs):
            assert torch.is_inference_mode_enabled()
            assert native.model.model_config is native.model_config
            optimized.append(kwargs)

        def predict(path, **kwargs):
            assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
            assert optimized
            assert kwargs == {"threshold": .2, "include_source_image": False}
            return SimpleNamespace(xyxy=np.array([[1, 2, 30, 40]]), confidence=np.array([.9]), class_id=np.array([0]))

        native.optimize_for_inference, native.predict = optimize, predict
        loaded.append(native)
        return native

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(rfdetr, "from_checkpoint", load)
    options = dict(resolution=96, confidence=.2, device=device, progress=False, backend="native")
    for _ in range(2):
        predictions, task, settings = predict_inputs(model, (image,), **options)
        assert task == "detect" and predictions["image"][0].bbox == (1, 2, 30, 40)
        assert settings["device"] == resolved
        assert settings["optimized_for_inference"]
        assert settings["inference_dtype"] == str(dtype).removeprefix("torch.")
    assert optimized == [{"compile": False, "dtype": dtype}] and len(loaded) == 1
    predict_inputs(model, (image,), **{**options, "resolution": 192})
    assert len(optimized) == len(loaded) == 2
    model.unload()
    predict_inputs(model, (image,), **options)
    assert len(optimized) == len(loaded) == 3
    assert checkpoint.read_bytes() == original
    assert not torch.is_inference_mode_enabled() and torch.is_grad_enabled()


@pytest.mark.parametrize(("variant", "task"), [
    ("RFDETRNano", "detect"),
    ("RFDETRSegNano", "segment"),
    ("RFDETRKeypointPreview", "pose"),
])
def test_native_optimized_predictions_match_original_weights(tmp_path, monkeypatch, variant, task):
    import rfdetr

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    # Small real native networks with random weights: no downloads or training.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        extra = {"num_keypoints_per_class": [8]} if task == "pose" else {}
        original = getattr(rfdetr, variant)(pretrain_weights=None, device="cpu", resolution=96,
            num_classes=1, num_queries=5, num_select=5, group_detr=1, **extra)
    checkpoint = tmp_path / "native.pth"
    state = original.model.model.state_dict()
    torch.save({"model": state, "model_name": variant, "model_config": original.model_config.model_dump(),
                "args": {"class_names": ["animal"]}}, checkpoint)
    source_bytes = checkpoint.read_bytes()
    image_path = tmp_path / "image.png"
    Image.fromarray(np.random.default_rng(7).integers(0, 256, (96, 96, 3), dtype=np.uint8)).save(image_path)
    with torch.inference_mode():
        expected = original.predict(str(image_path), threshold=0, include_source_image=False)

    model = df.Model(checkpoint, kind="rfdetr", task=task)
    item = ModelInput("image", image_path, 96, 96, "image.png")
    output, actual_task, settings = predict_inputs(model, (item,), resolution=96, confidence=0,
                                                  device="cpu", progress=False, backend="native")
    native = next(iter(model._runtime.values()))
    assert native._is_optimized_for_inference and settings["inference_dtype"] == "float32"
    assert not native.is_optimized_inplace
    assert native.model.inference_model is not native.model.model
    assert not native.model.inference_model.training
    assert native.model.model._export is False
    assert all(torch.equal(value, native.model.model.state_dict()[key]) for key, value in state.items())
    assert checkpoint.read_bytes() == source_bytes
    assert actual_task == task and len(output["image"]) == len(expected.class_id) > 0
    boxes = expected.data["xyxy"] if task == "pose" else expected.xyxy
    scores = expected.detection_confidence if task == "pose" else expected.confidence
    np.testing.assert_allclose([p.bbox for p in output["image"]], boxes, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose([p.score for p in output["image"]], scores, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal([p.class_id for p in output["image"]], expected.class_id)
    with torch.inference_mode():
        actual = native.predict(str(image_path), threshold=0, include_source_image=False)
    if task == "pose":
        np.testing.assert_allclose(np.array([p.keypoints for p in output["image"]])[..., :2], expected.xy, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(actual.keypoint_confidence, expected.keypoint_confidence, rtol=1e-5, atol=1e-6)
    elif task == "segment":
        np.testing.assert_array_equal(actual.mask, expected.mask)


def test_failed_optimization_is_not_cached(tmp_path, monkeypatch):
    import rfdetr

    checkpoint = tmp_path / "model.pth"
    torch.save({"model_name": "RFDETRNano", "model_config": {"patch_size": 16, "num_windows": 2}}, checkpoint)
    model = df.Model(checkpoint, kind="rfdetr", task="detect")
    error = RuntimeError("optimization failed")

    def fail(**kwargs):
        raise error

    monkeypatch.setattr(rfdetr, "from_checkpoint", lambda *a, **kw: SimpleNamespace(
        model=SimpleNamespace(), model_config=SimpleNamespace(), optimize_for_inference=fail))
    with pytest.raises(RuntimeError, match="optimization failed") as raised:
        predict_inputs(model, (), resolution=96, confidence=0, device="cpu", progress=False, backend="native")
    assert raised.value is error and not model.loaded
    assert not torch.is_inference_mode_enabled() and torch.is_grad_enabled()
