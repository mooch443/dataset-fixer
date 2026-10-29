import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from dataset_fixer import id
from dataset_fixer.id.data import SampleIndex, as_dataset


@pytest.fixture
def attention_case(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("captum")
    from dataset_fixer.id import model as identity_model

    class ColorEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.identity = dict(preprocessing=dict(resolution=16, mean=[0]*3, std=[1]*3))

        def forward(self, images):
            return torch.nn.functional.normalize(images[:, :2].mean(dim=(2, 3)), dim=1)

    module = ColorEncoder().train()
    monkeypatch.setattr(identity_model, "runtime", lambda model: module)
    rows = []
    for name, color, truth in [("correct", (220, 0, 0), "wolf-a"),
                                ("incorrect", (0, 220, 0), "wolf-a"),
                                ("unlabeled", (220, 0, 0), None),
                                ("tied", (220, 220, 0), "wolf-a")]:
        path = tmp_path / f"{name}.png"
        Image.new("RGB", (16, 16), color).save(path)
        rows.append(dict(path=str(path), width=16, height=16, identity_id=truth, clip_id=name))
    data = as_dataset(SampleIndex(rows), location=tmp_path)
    gallery = SimpleNamespace(identities=("wolf-a", "wolf-b"), labels=("wolf-a", "wolf-b"),
                              embeddings=np.eye(2, dtype=np.float32), validate=lambda model: None,
                              model_fingerprint="test-model", fingerprint="test-gallery")
    return SimpleNamespace(_device="cpu"), module, data, gallery


@pytest.mark.parametrize("target", [None, "wolf-b"])
def test_attention_reports_ground_truth_prediction_and_signed_target_margin(attention_case, tmp_path, target):
    model, module, data, gallery = attention_case
    destination = tmp_path / "attention.png"
    id.visualize_attention(model, data, gallery=gallery, identity_id=target,
                           samples=4, columns=3, patch_fraction=1, destination=destination, show=False)
    records = {json.loads(r["source_ref"])[0].split("/")[-1]: r
               for r in json.loads(destination.with_suffix(".json").read_text())}
    assert records["correct.png"]["correct"] is True
    assert records["incorrect.png"]["correct"] is False
    assert records["unlabeled.png"]["correct"] is None
    assert records["tied.png"]["correct"] is False
    assert records["tied.png"]["margin"] == 0
    assert records["correct.png"]["prediction"] == "wolf-a"
    assert records["incorrect.png"]["prediction"] == "wolf-b"
    assert records["correct.png"]["margin"] == (-1 if target else 1)
    assert records["incorrect.png"]["margin"] == 1
    assert records["correct.png"]["competitor"] == ("wolf-a" if target else "wolf-b")
    assert module.training and next(module.parameters()).device.type == "cpu"


def test_attention_single_identity_has_no_margin(attention_case, tmp_path):
    model, _, data, gallery = attention_case
    gallery.identities = gallery.labels = ("wolf-a",)
    gallery.embeddings = gallery.embeddings[:1]
    id.visualize_attention(model, data, gallery=gallery, contrast=False, samples=1, columns=1,
                           patch_fraction=1, destination=tmp_path / "single.png", show=False)
    record = json.loads((tmp_path / "single.json").read_text())[0]
    assert record["competitor"] is None and record["margin"] is None


@pytest.mark.parametrize("columns", [0, -1, 1.5, True])
def test_attention_rejects_invalid_columns_before_loading_model(columns):
    with pytest.raises(ValueError, match="columns must be a positive integer"):
        id.visualize_attention(None, None, gallery=None, columns=columns, show=False)


def test_attention_overlay_strength_uses_shared_scale():
    from dataset_fixer.id.inspection import _attention_item
    from dataset_fixer.visualization import VisualizationOptions
    from pathlib import Path
    sample = SimpleNamespace(image_path=Path("test.png"), clip_id="test")
    record = dict(correct=True, ground_truth="wolf", prediction="wolf", identity_id="wolf",
                  competitor=None, similarity=1.)
    rgb = np.full((4, 4, 3), .5)
    weak, strong = [_attention_item(sample, rgb, np.full((4, 4), value), record, 1., VisualizationOptions())
                    for value in (.01, 1.)]
    assert len(weak.panels) == len(strong.panels) == 1
    assert np.abs(weak.panels[0].image.astype(float) - rgb*255).max() < 2
    assert np.abs(strong.panels[0].image.astype(float) - rgb*255).max() > 50
