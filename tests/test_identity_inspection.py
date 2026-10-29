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


@pytest.fixture
def prediction_case(tmp_path):
    from dataset_fixer import ImagePrediction, PredictionResult
    from dataset_fixer.id.data import ImageRef
    cases = [
        ("correct", "wolf-a", "wolf-a", False, [("wolf-a", .98), ("wolf-b", .4)]),
        ("incorrect", "wolf-b", "wolf-a", False, [("wolf-a", .986), ("wolf-b", .985)]),
        ("tied", "wolf-a", "wolf-a", False, [("wolf-a", .9), ("wolf-b", .9)]),
        ("unlabeled", None, "wolf-a", False, [("wolf-a", .8), ("wolf-b", .3)]),
        ("unknown", "new-wolf", None, True, [("wolf-a", .2), ("wolf-b", .1)]),
        ("rejected_known", "wolf-a", None, True, [("wolf-a", .2), ("wolf-b", .1)]),
        ("single", "wolf-a", "wolf-a", False, [("wolf-a", .8)]),
        ("embedding", "wolf-a", None, None, []),
    ]
    records = []
    for number, (name, truth, prediction, rejected, ranking) in enumerate(cases):
        path = tmp_path / f"{name}.png"
        Image.new("RGB", (20, 16), (50, 80 + number*15, 100)).save(path)
        records.append(ImagePrediction(name, path, path.name, 20, 16, metadata=dict(
            source_ref=ImageRef(str(path)).key, reference_identity=truth, identity_id=prediction,
            rejected=rejected, similarity=ranking[0][1] if ranking else None, ranking=ranking,
            clip_id="0909_126_PM3_Zz_SH")))
    return PredictionResult("test-model", "id", "id", "torch", tuple(records), 0.,
        model_metadata=dict(name="test-model", model_type="resnet18", resolution=128,
                            hash="1fbe0718", source_created_at="2026-10-29T15:08:17"))


def test_prediction_context_distinguishes_errors_rejections_and_missing_labels(prediction_case):
    from dataset_fixer.id.evaluation import _prediction_context
    contexts = {record.image_id: _prediction_context(record) for record in prediction_case}
    assert {key: value["correct"] for key, value in contexts.items()} == dict(
        correct=True, incorrect=False, tied=False, unlabeled=None, unknown=True,
        rejected_known=False, single=True, embedding=None)
    assert contexts["incorrect"]["margin"] == pytest.approx(.001)
    assert contexts["tied"]["margin"] == 0
    assert contexts["single"]["margin"] is None
    assert contexts["embedding"]["similarity"] is None
    assert contexts["unknown"]["prediction"] == "unknown (rejected)"
    assert contexts["unknown"]["identity_id"] == "wolf-a"
    assert contexts["unknown"]["competitor"] == "wolf-b"
    assert contexts["unknown"]["margin"] == pytest.approx(.1)


def test_prediction_grid_shares_identity_cards_and_model_heading(prediction_case, tmp_path, monkeypatch):
    from dataset_fixer.id import inspection
    panels, native = [], inspection._identity_panel

    def capture(*args):
        panel = native(*args)
        panels.append(panel)
        return panel

    monkeypatch.setattr(inspection, "_identity_panel", capture)
    chart = prediction_case.visualize(samples=8, columns=3, destination=tmp_path / "predictions.png", show=False)
    spec = chart.to_dict()
    assert [len(row["hconcat"]) for row in spec["vconcat"]] == [3, 3, 2]
    assert len(panels) == 8
    assert {panel.heading.vconcat[1].mark.color for panel in panels} == {"#218739", "#C53030", "#6B7280"}
    assert any("Margin: +0.0010" in panel.footer for panel in panels)
    assert any(panel.footer.startswith("Best: wolf-a") for panel in panels)
    assert any(panel.footer == "Embedding only" for panel in panels)
    text = json.dumps(spec)
    assert text.count("1fbe0718") == text.count("resnet18") == text.count("128px") == 1
    assert "Reference:" not in text
    assert len([row for rows in spec["datasets"].values() for row in rows if "url" in row]) == 8
    hidden = prediction_case.visualize(samples=1, columns=1, show_model_slugs=False, show=False).to_dict()
    assert "1fbe0718" not in json.dumps(hidden)
    with pytest.raises(TypeError, match="show_model_slugs must be a boolean"):
        prediction_case.visualize(show_model_slugs="yes", show=False)
