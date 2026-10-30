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
                clip_id=sample.clip_id, tracklet_id=sample.provenance['tracklet_id'],
                frame_index=sample.frame_index, local_id=sample.local_id)))
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


def test_portable_gallery_preserves_lazy_pixels_and_provenance(encoder, tmp_path):
    import shutil
    from PIL import Image
    model, _ = encoder
    source = tmp_path / 'source'
    source.mkdir()
    records = []
    for row in range(4):
        pixels = np.random.default_rng(row).integers(1, 255, (18, 22, 3), dtype=np.uint8)
        pixels[:2] = 0
        path = source / f'{row}.png'
        record = dict(path=str(path), width=22, height=18, identity_id=f'wolf-{row // 2}',
                      clip_id='clip', tracklet_id=str(row), frame_index=row, local_id='local',
                      trim_black_borders=True, crop=(1, 1, 20, 16))
        if row % 2:
            path = path.with_suffix('.npz')
            np.savez(path, images=pixels[None])
            record.update(path=str(path), array='images', row=0)
        else:
            Image.fromarray(pixels).save(path)
        records.append(record)
    gallery = model.enroll(as_dataset(SampleIndex(records), location=source), samples_per_identity=2)
    original = gallery._index()
    expected = [s.image_ref.read() for s in original]
    gallery.save(tmp_path / 'saved' / 'gallery.npz', copy_images=True)
    shutil.move(tmp_path / 'saved', tmp_path / 'moved')
    shutil.rmtree(source)
    restored = id.Gallery.load(tmp_path / 'moved' / 'gallery.npz')
    assert restored.fingerprint == gallery.fingerprint
    assert restored.references == gallery.references
    assert restored.metadata == gallery.metadata
    np.testing.assert_array_equal(restored.embeddings, gallery.embeddings)
    fields = ['identity_id', 'clip_id', 'tracklet_id', 'frame_index', 'local_id', 'original_id']
    assert restored._index().frame[fields].equals(original.frame[fields])
    for pixels, sample in zip(expected, restored._index()):
        np.testing.assert_array_equal(pixels, sample.image_ref.read())
        assert sample.clip_id == 'clip'
        assert sample.image_ref.array is None and sample.image_ref.crop is None
    # A second portable export must read the relocated copies, not old sources.
    copied = id.Gallery.load(restored.save(tmp_path / 'copied' / 'gallery.npz', copy_images=True))
    assert copied.fingerprint == gallery.fingerprint
    copied.visualize(samples=4, columns=2, destination=tmp_path / 'gallery.png', show=False)


def test_training_bundle_gallery_survives_source_removal(identity_data, tmp_path, monkeypatch):
    import shutil
    from dataclasses import replace
    from dataset_fixer import TrainingConfig
    from dataset_fixer.id.evaluation import RejectionCalibration
    from test_identity import tiny_model
    id.register_backbone('portable-tiny', tiny_model, feature_dim=4,
                         preprocessing=dict(resolution=16, mean=[0]*3, std=[1]*3))
    result = id.train(id.Protocol(identity_data, gallery_samples=3), backbone='portable-tiny',
        augmentations=[], config=TrainingConfig(epochs=1, batch_size=4, workers=0,
            augmentation_samples=0, output_dir=tmp_path/'run', backend_options=dict(
                loss='cross_entropy', pretrained=False, embedding_dim=8, embedding_samples=0,
                reference_attention=False, trainer=dict(enable_progress_bar=False,
                    enable_model_summary=False, num_sanity_val_steps=0))))
    # No evaluation or attention call is required to publish the selected gallery.
    gallery = Model(result.bundle.path).gallery
    assert len(gallery.labels) == 6
    assert gallery.fingerprint == result.model.gallery.fingerprint
    assert Model(result.best_weights).gallery is None
    assert Model(result.bundle.path).rejection is None
    query = identity_data.sample(2, split='test')
    before = result.model.predict(query, gallery=gallery, prediction_cache=False)
    pixels = [s.image_ref.read() for s in gallery._index()]
    calibration = RejectionCalibration(.5, .1, 0., gallery.model_fingerprint, gallery.fingerprint,
                                       'fixture', 2, 2)
    result.auxiliary_files['identity/rejection.json'] = calibration.save(tmp_path/'rejection.json')
    assert result._publish(final=True)
    moved = tmp_path/'portable.zip'
    shutil.copy2(result.bundle.path, moved)
    # A stale calibration must never ship alongside a different gallery.
    replace(calibration, gallery_fingerprint='outdated').save(tmp_path/'rejection.json')
    assert result._publish(final=True)
    assert 'identity/rejection.json' not in result.bundle.files
    for folder in ('train', 'val', 'run', 'package-cache'):
        shutil.rmtree(tmp_path/folder)
    monkeypatch.setattr('dataset_fixer.model_sources._download_wandb', lambda *a, **k: (moved, None))
    restored = Model('wandb:team/project/runs/portable')
    assert restored.gallery.fingerprint == gallery.fingerprint
    restored.rejection.validate(restored, restored.gallery)
    for expected, sample in zip(pixels, restored.gallery._index()):
        np.testing.assert_array_equal(expected, sample.image_ref.read())
    after = restored.predict(query, gallery=restored.gallery, prediction_cache=False)
    assert [r.identity_id for r in after] == [r.identity_id for r in before]
    np.testing.assert_allclose([r.similarity for r in after], [r.similarity for r in before])
    restored.unload()
    assert restored.gallery.fingerprint == gallery.fingerprint


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
    import torch
    from dataset_fixer.training.session import TrainingResult
    monkeypatch.setattr('dataset_fixer.id.model.fingerprint', lambda model: 'encoder')
    model, calls = encoder
    model._runtime = {}
    module = torch.nn.Linear(3, 3)
    monkeypatch.setattr('dataset_fixer.id.model.runtime', lambda model: module)
    result = SimpleNamespace(model=model, output_dir=tmp_path, auxiliary_files={}, config=SimpleNamespace(seed=7),
        _identity_protocol=id.Protocol(identity_data, gallery_samples=2, gallery_selection='medoids', gallery_candidates=5))
    state = torch.get_rng_state()
    gallery = TrainingResult._enrollment(result)
    torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
    assert gallery.metadata['selection'] == 'medoids'
    assert len(calls) == 1
    assert TrainingResult._enrollment(result) is gallery
    assert len(calls) == 1
    assert set(result.auxiliary_files) == {'identity/gallery.npz', 'identity/enrollment.csv', 'identity/gallery.references'}
    assert len(list(result.auxiliary_files['identity/gallery.references'].glob('images/*.jxl'))) == 4


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
