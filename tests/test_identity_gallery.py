"""Opt-in reference selection and compatibility with existing enrollment."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from dataset_fixer import id, Model
from dataset_fixer.id import gallery as implementation
from dataset_fixer.id.data import SampleIndex, as_dataset, balanced_rows, index_of
from test_identity import identity_data


@pytest.fixture
def encoder(monkeypatch):
    calls = []
    def embed(model, index, **kwargs):
        calls.append(index)
        result = []
        for sample in index:
            pixels = sample.image_ref.read()
            vector = pixels.mean(axis=(0, 1)).astype(np.float32)
            vector /= np.linalg.norm(vector)
            result.append(SimpleNamespace(embedding=vector, metadata=dict(
                source_ref=sample.image_ref.key, pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                reference_identity=sample.identity_id, original_id=sample.provenance['original_id'],
                clip_id=sample.clip_id, tracklet_id=sample.provenance['tracklet_id'])))
        return result
    monkeypatch.setattr(implementation, 'embed', embed)
    monkeypatch.setattr(implementation, 'fingerprint', lambda model: 'encoder')
    model = SimpleNamespace(kind='id')
    model.enroll = lambda source, **kwargs: Model.enroll(model, source, **kwargs)
    return model, calls


def test_default_protocol_enrollment_matches_previous_selection(identity_data, encoder):
    model, calls = encoder
    protocol = id.Protocol(identity_data, gallery_samples=3, seed=29)
    index = index_of(identity_data)
    original_rows = []
    for label in sorted(index.frame.identity_id.unique()):
        rows = np.flatnonzero((index.frame.identity_id == label) & (index.frame.split == 'train'))
        original_rows.extend(rows[balanced_rows(index.take(rows), 3, 29)])
    assert index_of(protocol.references).frame.equals(index.take(original_rows).frame)
    assert protocol.reference_candidates is protocol.references
    before = model.enroll(protocol.references, samples_per_identity=3)
    after = model.enroll(protocol)
    assert before.references == after.references
    np.testing.assert_array_equal(before.embeddings, after.embeddings)
    assert len(calls[-1]) == 6


def test_medoid_enrollment_deduplicates_candidates_and_persists_report(identity_data, encoder, tmp_path):
    pytest.importorskip('kmedoids')
    model, calls = encoder
    index = index_of(identity_data)
    # Five candidates per identity, with one copied image per identity.
    for label in index.names.values():
        rows = index.where(split='train', identity_id=label)
        rows[1].image_path.write_bytes(rows[0].image_path.read_bytes())
    protocol = id.Protocol(identity_data, gallery_samples=3, gallery_selection='medoids', gallery_candidates=5)
    assert len(index_of(protocol.reference_candidates)) == 10
    gallery = model.enroll(protocol)
    assert len(calls[-1]) == 10
    assert gallery.summary.candidates.tolist() == [5, 5]
    assert gallery.summary.unique_candidates.tolist() == [4, 4]
    assert gallery.summary.references.tolist() == [3, 3]
    assert gallery.summary.shortfall.tolist() == [0, 0]
    assert gallery.summary.tracklets.tolist() == [3, 3]
    assert all(ref['tracklet_id'] is not None for ref in gallery.references)
    assert gallery.references == model.enroll(protocol).references
    restored = id.Gallery.load(gallery.save(tmp_path/'gallery.npz'))
    assert restored.metadata == gallery.metadata
    assert restored.fingerprint == gallery.fingerprint
    restored.visualize(samples=6, columns=3, destination=tmp_path/'gallery.png', show=False)
    assert (tmp_path/'gallery.png').is_file()
    # An exhausted candidate pool is reported, rather than filled with duplicates.
    short = model.enroll(protocol, samples_per_identity=5)
    assert short.summary.shortfall.tolist() == [1, 1]


def test_medoid_selection_matches_library_with_clip_tracklet_quotas():
    fasterpam = pytest.importorskip('kmedoids').fasterpam
    from sklearn.metrics.pairwise import cosine_distances
    records, vectors = [], []
    for clip in ('a', 'b'):
        for track in ('0', '1'):
            for angle in (0., .1, .2, 2.):
                records.append(dict(path=str(len(records)), width=1, height=1,
                                    identity_id='wolf', clip_id=clip, tracklet_id=track))
                vectors.append([np.cos(angle), np.sin(angle)])
    index = SampleIndex(records)
    features = np.asarray(vectors, dtype=np.float32)
    state = np.random.get_state()
    selected = implementation._medoid_rows(index, features, 4, 7)
    assert index.take(selected).frame.groupby(['clip_id', 'tracklet_id'], observed=True).size().tolist() == [1]*4
    for group in range(4):
        distance = cosine_distances(features[group*4:group*4+4]).astype(np.float32)
        expected = fasterpam(distance, 1, random_state=7, n_cpu=1).medoids[0] + group*4
        assert expected in selected
    np.testing.assert_array_equal(state[1], np.random.get_state()[1])


def test_protocol_candidate_roles_remain_disjoint(identity_data, encoder, tmp_path):
    model, _ = encoder
    frame = index_of(identity_data).frame.astype(object)
    additions = []
    for role in ('novel', 'unknown'):
        for clip in ('0', '1'):
            additions.append(dict(path=f'{role}-{clip}', identity_id=role, clip_id=f'{role}-{clip}',
                                  split='val', width=1, height=1))
    import pandas as pd
    data = as_dataset(SampleIndex(pd.concat([frame, pd.DataFrame(additions)])), location=tmp_path)
    protocol = id.Protocol.open_set(data, unknown_identities={'val': ['unknown']},
        novel_identities={'val': ['novel']}, gallery_selection='medoids', gallery_samples=2, gallery_candidates=5)
    candidates = index_of(protocol.reference_candidates).frame
    assert set(candidates.identity_id) == {'wolf-a', 'wolf-b', 'novel'}
    assert set(candidates.clip_id).isdisjoint(index_of(protocol.queries['val']).frame.clip_id)
    assert set(index_of(protocol.training).frame.identity_id) == {'wolf-a', 'wolf-b'}
    with pytest.raises(ValueError, match='reserve candidate'):
        model.enroll(id.Protocol(identity_data), selection='medoids')


def test_shared_scores_match_numpy_and_differentiable_torch_and_legacy_save(tmp_path):
    torch = pytest.importorskip('torch')
    rng = np.random.default_rng(8)
    references = rng.normal(size=(7, 3)).astype(np.float32)
    references /= np.linalg.norm(references, axis=1, keepdims=True)
    queries = rng.normal(size=(263, 3)).astype(np.float32)
    labels = ('b', 'a', 'c', 'a', 'b', 'c', 'a')
    expected = np.stack([(queries @ references.T)[:, np.asarray(labels) == label].max(axis=1)
                         for label in sorted(set(labels))], axis=1)
    gallery = id.Gallery(references, labels, tuple({} for _ in labels), 'encoder')
    np.testing.assert_allclose(gallery.scores(queries), expected, atol=1e-6)
    inputs = torch.tensor(queries, requires_grad=True)
    scores = implementation.identity_scores(inputs, torch.tensor(references), labels)
    np.testing.assert_allclose(scores.detach().numpy(), expected, atol=1e-6)
    scores.sum().backward()
    assert torch.isfinite(inputs.grad).all() and inputs.grad.abs().sum() > 0
    assert gallery.scores(queries[:0]).shape == (0, 3)
    path = tmp_path/'legacy.npz'
    np.savez(path, embeddings=gallery.embeddings, metadata=json.dumps(dict(
        labels=gallery.labels, references=gallery.references, model='encoder')))
    loaded = id.Gallery.load(path)
    assert loaded.fingerprint == gallery.fingerprint
    assert loaded.metadata == {}


def test_training_result_uses_protocol_and_bundles_local_report(identity_data, encoder, monkeypatch, tmp_path):
    pytest.importorskip('kmedoids')
    from dataset_fixer.training.session import TrainingResult
    monkeypatch.setattr('dataset_fixer.id.model.fingerprint', lambda model: 'encoder')
    model, calls = encoder
    result = SimpleNamespace(model=model, output_dir=tmp_path, auxiliary_files={},
        _identity_protocol=id.Protocol(identity_data, gallery_samples=2, gallery_selection='medoids', gallery_candidates=5))
    gallery = TrainingResult._enrollment(result)
    assert gallery.metadata['selection'] == 'medoids'
    assert len(calls) == 1
    assert TrainingResult._enrollment(result) is gallery
    assert len(calls) == 1
    assert set(result.auxiliary_files) == {'identity/gallery.npz', 'identity/enrollment.csv'}


def test_monitor_uses_opt_in_selector_without_new_metrics(identity_data, tmp_path, monkeypatch):
    pytest.importorskip('kmedoids')
    from dataset_fixer.id import monitoring
    from test_identity_monitoring import monitor_result, monitor_module
    monkeypatch.setattr('dataset_fixer.static_rendering.save_chart', lambda *a, **k: None)
    module = monitor_module()
    result = monitor_result(identity_data, tmp_path)
    monitoring.prepare_monitor(result)
    baseline = monitoring.monitor_epoch(result, module, 0)
    result._identity_protocol = id.Protocol(identity_data, gallery_samples=2,
        gallery_selection='medoids', gallery_candidates=5)
    original, selected = implementation._medoid_rows, []
    def observe(*args):
        rows = original(*args)
        selected.append(rows)
        return rows
    monkeypatch.setattr(implementation, '_medoid_rows', observe)
    medoids = monitoring.monitor_epoch(result, module, 1)
    assert len(selected) == 1 and len(selected[0]) == 4
    assert baseline.keys() == medoids.keys()
    assert baseline['embeddings/samples'] == medoids['embeddings/samples'] == 12


@pytest.mark.parametrize('kwargs', [dict(selection='other'), dict(samples_per_identity=0),
    dict(selection='medoids', samples_per_identity=20, candidates_per_identity=10)])
def test_invalid_selection_fails_before_pixels(identity_data, encoder, kwargs):
    model, calls = encoder
    with pytest.raises(ValueError):
        model.enroll(identity_data, **kwargs)
    assert not calls
