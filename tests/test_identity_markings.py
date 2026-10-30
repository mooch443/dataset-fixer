import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from dataset_fixer import id
from dataset_fixer.id.data import SampleIndex


@pytest.fixture
def marking_case(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("captum")
    from dataset_fixer.id import model as identity_model

    class TextureEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.identity = dict(preprocessing=dict(resolution=96, mean=[.2]*3, std=[.5]*3))

        def forward(self, images):
            rgb = images * .5 + .2
            features = rgb[:, :2].square().mean(dim=(2, 3)) + .03
            return torch.nn.functional.normalize(features, dim=1)

    module = TextureEncoder().train()
    model = SimpleNamespace(_device="cpu")
    monkeypatch.setattr(identity_model, "runtime", lambda model: module)
    monkeypatch.setattr("dataset_fixer.id.gallery.fingerprint", lambda model: "markings-test")
    rng = np.random.default_rng(71)
    texture = cv2.GaussianBlur(rng.integers(0, 256, (64, 64), dtype=np.uint8), (3, 3), .5)
    # Repeatable local texture, translated and illuminated differently per clip.
    rows, references, embeddings = [], [], []
    for identity, channel in (("individual-a", 0), ("individual-b", 1)):
        for clip in range(3):
            pixels = np.full((96, 96, 3), 30 + clip, np.uint8)
            start = 14 + clip
            pixels[start:start+64, start:start+64, channel] = texture
            path = tmp_path / f"{identity}-{clip}.png"
            Image.fromarray(pixels).save(path)
            rows.append(dict(path=str(path), width=96, height=96, identity_id=identity,
                             clip_id=f"clip-{clip}", tracklet_id=f"{identity}-{clip}"))
    index = SampleIndex(rows)
    for sample in index:
        pixels = sample.image_ref.read()
        with torch.no_grad():
            embeddings.append(module(identity_model.preprocess(pixels, module.identity["preprocessing"])[None])[0].numpy())
        references.append(dict(source_ref=sample.image_ref.key, clip_id=sample.clip_id,
            pixel_sha256=hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()))
    gallery = id.Gallery(np.stack(embeddings), tuple(index.frame.identity_id), tuple(references), "markings-test")
    return model, module, index, gallery


def test_captum_group_effect_equals_full_image_margin_change():
    torch = pytest.importorskip("torch")
    pytest.importorskip("captum")
    from dataset_fixer.id.markings import _ablate
    image = torch.arange(3*16*16, dtype=torch.float32).reshape(1, 3, 16, 16) % 7
    def objective(batch):
        return batch.square().mean(dim=(1, 2, 3))
    box = (4, 5, 10, 11)
    effect, altered = _ablate(objective, image, box)
    assert effect == pytest.approx(float(objective(image) - objective(altered[None])), abs=1e-6)
    assert effect > 0
    outside = np.ones((16, 16), bool)
    outside[5:11, 4:10] = False
    np.testing.assert_array_equal(image[0, :, outside], altered[:, outside])


def test_markings_match_across_clips_and_save_measured_evidence(marking_case, tmp_path, monkeypatch):
    model, module, index, gallery = marking_case
    from dataset_fixer import visualization
    from dataset_fixer.id import markings
    render = visualization.visualize_records
    load = markings._load_candidates
    rows_rendered = []
    def render_rows(records, *, options, prepare, **kwargs):
        rows_rendered.append(records)
        assert len(records) <= options.columns == 2
        assert all(len(prepare(record).panels) == 1 for record in records)
        return render(records, options=options, prepare=prepare, **kwargs)
    def load_focal(index, selected, *args):
        assert set(index.frame.identity_id) == {"individual-a"}
        assert len(selected) <= 3
        return load(index, selected, *args)
    monkeypatch.setattr(visualization, "visualize_records", render_rows)
    monkeypatch.setattr(markings, "_load_candidates", load_focal)
    destination = tmp_path / "markings.png"
    fingerprint = gallery.fingerprint
    table = id.explain_markings(model, gallery=gallery, identity_id="individual-a", samples_per_identity=3,
                               patches=2, patch_fraction=.2, destination=destination, show=False)
    assert not table.empty
    assert set(table.role) == {"anchor", "cross_clip"}
    assert set(table.identity_id) == {"individual-a"}
    assert table.patch.nunique() <= 2
    assert len(rows_rendered) == table.patch.nunique()
    assert all(len({row["patch"] for row in rows}) == 1 for rows in rows_rendered)
    assert np.isfinite(table.effect).all()
    np.testing.assert_allclose(table.margin_before - table.margin_after, table.effect, atol=1e-6)
    np.testing.assert_allclose(table.effect - table.control_effect, table.effect_minus_control, atol=1e-6)
    for _, group in table.groupby("patch"):
        assert group.clip_id.nunique() == len(group) >= 2
        assert group.pixel_sha256.nunique() == len(group)
        assert (group[group.role == "cross_clip"].match_ratio < .75).all()
    for row in table.itertuples():
        assert any(gallery.references[i]["source_ref"] == row.source_ref for i in row.excluded_reference_rows)
    assert all(summary["clips"] >= 2 for summary in table.attrs["clip_summary"])
    assert module.training and next(module.parameters()).device.type == "cpu"
    assert gallery.fingerprint == fingerprint
    saved = json.loads(destination.with_suffix(".json").read_text())
    assert len(saved["observations"]) == len(table)
    with np.load(destination.with_suffix(".npz")) as arrays:
        for row in table.itertuples():
            original, altered = (arrays[f"{row.array_key}_{suffix}"] for suffix in ("original", "altered"))
            mask = np.ones(original.shape[:2], bool)
            a, b, c, d = row.box
            mask[b:d, a:c] = False
            np.testing.assert_array_equal(original[mask], altered[mask])
            # Reproduce the reported margin change from saved exact inputs and
            # the same remaining gallery, without Captum or display rounding.
            import torch
            keep = np.ones(len(gallery.labels), bool)
            keep[row.excluded_reference_rows] = False
            from dataset_fixer.id.gallery import identity_scores
            labels = tuple(label for i, label in enumerate(gallery.labels) if keep[i])
            inputs = np.stack([arrays[f"{row.array_key}_{suffix}"] for suffix in ("input", "ablated_input")])
            with torch.no_grad():
                scores = identity_scores(module(torch.from_numpy(inputs)).numpy(), gallery.embeddings[keep], labels)
            margins = scores[:, 0] - scores[:, 1]
            np.testing.assert_allclose(margins, [row.margin_before, row.margin_after], atol=1e-6)
    assert Image.open(destination).width > 500


def test_no_claim_of_cross_clip_matches_for_duplicates_or_flat_images(marking_case):
    model, module, index, gallery = marking_case
    from dataset_fixer.id.markings import _load_candidates, _proposals
    duplicated = index.frame.iloc[[0, 0]].copy()
    duplicated["clip_id"] = ["clip-a", "clip-b"]
    images = _load_candidates(SampleIndex(duplicated), [0, 1], module.identity["preprocessing"], .2)
    assert len(images) == 1
    assert _proposals(images, "individual-a", 2) == []
    flat_path = index[0].image_path.parent / "flat.png"
    Image.new("RGB", (96, 96), (80, 80, 80)).save(flat_path)
    flat = SampleIndex([dict(path=str(flat_path), width=96, height=96, identity_id="individual-a", clip_id="clip-flat")])
    assert _load_candidates(flat, [0], module.identity["preprocessing"], .2) == []
    # Same-clip images cannot become cross-clip confirmations.
    same_clip = index.frame.copy()
    same_clip["clip_id"] = "single"
    with pytest.raises(ValueError, match="at least two clips"):
        id.explain_markings(model, SampleIndex(same_clip), gallery=gallery, identity_id="individual-a", show=False)


def test_marking_model_state_restored_after_score_failure(marking_case, monkeypatch):
    model, module, _, gallery = marking_case
    def fail(*args, **kwargs):
        raise RuntimeError("model scoring failed")
    monkeypatch.setattr("dataset_fixer.id.markings._measure", fail)
    with pytest.raises(RuntimeError, match="model scoring failed"):
        id.explain_markings(model, gallery=gallery, identity_id="individual-a", samples_per_identity=3, show=False)
    assert module.training and next(module.parameters()).device.type == "cpu"
