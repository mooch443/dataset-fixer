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
@pytest.mark.parametrize("color_scale", ["shared", "per_image"])
def test_attention_reports_ground_truth_prediction_and_signed_target_margin(attention_case, tmp_path, target, color_scale):
    model, module, data, gallery = attention_case
    destination = tmp_path / "attention.png"
    id.visualize_attention(model, data, gallery=gallery, identity_id=target,
                           samples=4, columns=3, patch_fraction=1, color_scale=color_scale, destination=destination, show=False)
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
    bound = max(r['max_abs_effect'] for r in records.values())
    assert all(r['color_scale'] == color_scale and r['color_bound'] ==
               (r['max_abs_effect'] if color_scale == 'per_image' else bound) for r in records.values())
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


@pytest.mark.parametrize("contrast", [False, True])
@pytest.mark.parametrize("count", [1, 4])
def test_reference_attention_averages_raw_maps_and_excludes_self_matches(attention_case, tmp_path, monkeypatch, contrast, count):
    from dataset_fixer.id import model as identity_model
    from dataset_fixer.id.data import ImageRef
    model, module, _, _ = attention_case
    monkeypatch.setattr(identity_model, 'fingerprint', lambda model: 'test-model')
    monkeypatch.setattr('dataset_fixer.id.gallery.fingerprint', lambda model: 'test-model')
    labels, vectors, refs = [], [], []
    colors = [(220, 20, 0), (190, 45, 0), (150, 65, 0),
              (20, 220, 0), (45, 190, 0), (65, 150, 0), (220, 20, 0),
              (170, 140, 0), (140, 170, 0)]
    for i, color in enumerate(colors):
        path = tmp_path / f'ref-{i}.png'
        pixels = np.full((16, 16, 3), 40, dtype=np.uint8)
        pixels[2:14, 3 + i % 3:11 + i % 3] = color
        Image.fromarray(pixels).save(path)
        labels.append('wolf-a' if i < 3 or i == 6 else 'wolf-b' if i < 6 else 'wolf-c')
        vector = pixels[:, :, :2].mean(axis=(0, 1)).astype(np.float32)
        vectors.append(vector / np.linalg.norm(vector))
        refs.append(dict(source_ref=ImageRef(str(path)).key, clip_id=f'clip-{i % 2}'))
    gallery = id.Gallery(np.asarray(vectors), tuple(labels), tuple(refs), 'test-model')
    fingerprint = gallery.fingerprint
    scores = gallery.scores(np.eye(2, dtype=np.float32))
    path = tmp_path / 'reference-attention.png'
    gallery.visualize_attention(model, samples_per_identity=count, contrast=contrast, patch_fraction=.5,
                                color_scale='per_image', destination=path, show=False)
    records = json.loads(path.with_suffix('.json').read_text())
    assert [r['identity_id'] for r in records] == ['wolf-a', 'wolf-b', 'wolf-c']
    with np.load(path.with_suffix('.npz')) as maps:
        means = {r['identity_id']: np.mean([maps[s['map_key']] for s in r['sources']], axis=0)
                 for r in records}
        for i, record in enumerate(records):
            expected = means[record['identity_id']].copy()
            np.testing.assert_allclose(maps[record['mean_map_key']], expected)
            if contrast:
                expected -= np.mean([value for label, value in means.items() if label != record['identity_id']], axis=0)
                assert len(record['other_identity_means']) == 2
                if count == 4:
                    pooled = np.mean([maps[s['map_key']] for other in records
                        if other['identity_id'] != record['identity_id'] for s in other['sources']], axis=0)
                    assert not np.allclose(expected, means[record['identity_id']] - pooled)
            assert np.abs(expected).max() > .001
            np.testing.assert_allclose(maps[f'attribution_{i}'], expected)
            assert record['aggregation_count'] == min(count, gallery.labels.count(record['identity_id']))
            assert record['color_bound'] == pytest.approx(np.abs(expected).max())
            assert record['representative'] == 'embedding_medoid'
            assert record['source_ref'] in [s['source_ref'] for s in record['sources']]
            for source in record['sources']:
                assert source['reference_row'] in source['excluded_reference_rows']
                if source['reference_row'] in (0, 6):
                    assert set(source['excluded_reference_rows']) == {0, 6}
    assert module.training and gallery.fingerprint == fingerprint
    np.testing.assert_array_equal(gallery.scores(np.eye(2, dtype=np.float32)), scores)
    singleton = id.Gallery(np.asarray(vectors)[[0, 3]], ('wolf-a', 'wolf-b'), (refs[0], refs[3]), 'test-model')
    with pytest.raises(ValueError, match='at least two distinct'):
        singleton.visualize_attention(model, show=False)
    assert module.training


def test_per_image_scale_preserves_zero_maps_and_displays_actual_effect(tmp_path):
    from dataset_fixer.id.inspection import _attention_options, _render_attention
    from pathlib import Path
    observations = []
    for effect in (1., .002, 0.):
        heat = np.zeros((32, 32), dtype=np.float32)
        heat[8:24, 8:24] = effect
        heat[4:8, 8:24] = -effect / 2
        metadata = dict(correct=True, ground_truth='wolf', prediction='wolf', identity_id='wolf',
                        competitor='other', margin=.8, similarity=.95, max_abs_effect=float(effect))
        sample = SimpleNamespace(image_path=Path('wolf.png'), clip_id=f'effect {effect}')
        observations.append((sample, np.full((32, 32, 3), .5), heat, metadata))
    for scale in ('shared', 'per_image'):
        path = tmp_path / f'{scale}.png'
        options = _attention_options(3, 3, .15, scale, path, False)
        _render_attention(observations, options, color_scale=scale, contrast=True)
        records = json.loads(path.with_suffix('.json').read_text())
        assert [r['max_abs_effect'] for r in records] == [1., .002, 0.]
        assert [r['color_bound'] for r in records] == ([1., 1., 1.] if scale == 'shared' else [1., .002, 0.])


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
