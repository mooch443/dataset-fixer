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


def test_native_pose_fusion_scores_survive_evaluation_and_cache(tmp_path):
    from rfdetr.models.postprocess import PostProcess
    from dataset_fixer.comparison.cache import load_package_cache, save_package_cache
    from dataset_fixer.comparison.inference import _assert_exact_predictions
    from dataset_fixer.comparison.metrics import evaluate_configuration
    from dataset_fixer.comparison.types import Cohort, CohortRecord, Prediction

    # Native learned localization precision can amplify object scores above one.
    points = torch.zeros(1, 2, 8, 8)
    points[..., :2] = .5
    points[..., 2:4] = 5
    points[..., 4] = points[..., 6] = 5
    outputs = {"pred_logits": torch.tensor([[[4.], [3.]]]),
               "pred_boxes": torch.tensor([[[.5, .5, .8, .8], [.05, .05, .1, .1]]]),
               "pred_keypoints": points}
    native = PostProcess(num_select=2, num_keypoints_per_class=[8], trace_alpha=.2)(outputs, torch.tensor([[96, 96]]))[0]
    scores = native["scores"].tolist()
    assert scores[0] > scores[1] > 1
    predictions = [Prediction(0, score, bbox=tuple(native["boxes"][i].tolist()),
                              keypoints=[(48., 48., 1.)] * 8,
                              metadata={"backend": "rfdetr", "score_domain": "nonnegative"})
                   for i, score in enumerate(scores)]
    truth = {"class_id": 0, "bbox": predictions[0].bbox, "keypoints": [(48., 48., 2)] * 8}
    record = CohortRecord("image", tmp_path / "image.jpg", "image.jpg", "val", 96, 96,
                          "image-sha", "annotation-sha", "original", (truth,))
    cohort = Cohort("val", "fingerprint", (record,), "pose", {0: "animal"}, {"kpt_shape": [8, 3]})
    by_image = {"image": predictions}
    _assert_exact_predictions(cohort, by_image, "rfdetr")
    metrics = evaluate_configuration(cohort, by_image, .5)
    assert metrics["summary"]["map50_95"] == pytest.approx(1.0)
    save_package_cache(tmp_path / "cache", cohort, {}, {.7: by_image})
    loaded, _, complete = load_package_cache(tmp_path / "cache", cohort, (.7,))
    assert complete
    np.testing.assert_allclose([p.score for p in loaded[.7]["image"]], scores)
    _assert_exact_predictions(cohort, loaded[.7], "rfdetr")


@pytest.mark.parametrize("domain,score", [
    ("nonnegative", -.1), ("nonnegative", float("nan")), ("nonnegative", float("inf")),
    ("probability", 2.73), (None, 2.73), ("invalid", .5),
])
def test_invalid_prediction_scores_are_rejected_in_fresh_and_cached_results(tmp_path, domain, score):
    from dataset_fixer.comparison.cache import _validate_cached_predictions
    from dataset_fixer.comparison.inference import _assert_exact_predictions
    from dataset_fixer.comparison.types import Cohort, CohortRecord, Prediction

    metadata = {} if domain is None else {"score_domain": domain}
    prediction = Prediction(0, score, bbox=(0, 0, 10, 10), metadata=metadata)
    record = CohortRecord("image", tmp_path / "image.jpg", "image.jpg", "val", 96, 96,
                          "image-sha", "annotation-sha", "original", ())
    cohort = Cohort("val", "fingerprint", (record,), "detect", {0: "animal"}, {})
    with pytest.raises(df.DatasetValidationError, match="Prediction score"):
        _assert_exact_predictions(cohort, {"image": [prediction]}, "model")
    with pytest.raises(df.DatasetValidationError, match="Cached prediction score"):
        _validate_cached_predictions([prediction], record, cohort, tmp_path / "cache.npz")
