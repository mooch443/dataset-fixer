from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dataset_fixer import id, Model, ImagePrediction, PredictionResult, TrainingConfig, TrainingSession, WandbConfig
from dataset_fixer.id.data import as_dataset, index_of
from test_identity import identity_data, tiny_model, pytestmark
from test_training import run


@pytest.mark.parametrize('tied', [False, True])
def test_batched_retrieval_matches_scalar_library(identity_data, tmp_path, monkeypatch, tied):
    import torch
    from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
    from pytorch_metric_learning.utils.inference import CustomKNN
    from pytorch_metric_learning.distances import CosineSimilarity
    from dataset_fixer.id import evaluation
    rng = np.random.default_rng(31)
    references = rng.normal(size=(18, 8)).astype('float32')
    vectors = rng.normal(size=(263, 8)).astype('float32')
    if tied:
        references[1:4] = references[0]
        vectors[:3] = references[0]
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    reference_labels = ('wolf-a', 'wolf-b', 'novel')*6
    gallery = id.Gallery(references, reference_labels,
        tuple(dict(pixel_sha256=f'ref-{i}') for i in range(18)), 'encoder')
    monkeypatch.setattr('dataset_fixer.id.gallery.fingerprint', lambda model: 'encoder')
    monkeypatch.setattr('dataset_fixer.id.model.runtime', lambda model: SimpleNamespace(identity={}))
    labels = [('wolf-b', None, 'unknown', 'novel', 'wolf-a')[i % 5] for i in range(len(vectors))]
    sample = identity_data._samples[0]
    records = tuple(ImagePrediction(str(i), sample.image_path, str(i), 24, 20,
        metadata=dict(embedding=v.tolist(), reference_identity=label, clip_id=f'clip-{i % 3}',
                      pixel_sha256=f'query-{i}')) for i, (v, label) in enumerate(zip(vectors, labels)))
    # An enrollment duplicate is excluded before scoring, as before batching.
    duplicate = ImagePrediction('duplicate', sample.image_path, 'duplicate', 24, 20,
        metadata=dict(embedding=vectors[0].tolist(), reference_identity='wolf-a', pixel_sha256='ref-0'))
    monkeypatch.setattr(evaluation, 'embed', lambda *args, **kwargs:
                        PredictionResult('model', 'id', 'id', 'torch', (*records, duplicate), 0.))
    # Keep real report/cohort handling, with a small existing image fixture.
    data = as_dataset(index_of(identity_data).where(split='test'), location=tmp_path)
    expected = []
    codes = {v: i for i, v in enumerate(gallery.identities)}
    ref_codes = np.array([codes[v] for v in gallery.labels])
    calculator = AccuracyCalculator(include=('mean_average_precision',), k=len(gallery.labels),
        device=torch.device('cpu'), knn_func=CustomKNN(CosineSimilarity(), batch_size=128))
    for label, vector in zip(labels, vectors):
        if label in codes:
            expected.append(calculator.get_accuracy(vector[None], np.array([codes[label]]),
                gallery.embeddings.copy(), ref_codes)['mean_average_precision'])
    original, calls = AccuracyCalculator.get_accuracy, []
    def observe(self, query, query_labels, reference, *args, **kwargs):
        calls.append((len(query), reference))
        return original(self, query, query_labels, reference, *args, **kwargs)
    monkeypatch.setattr(AccuracyCalculator, 'get_accuracy', observe)
    events = []
    report = evaluation.evaluate(SimpleNamespace(name='model'), data, gallery=gallery,
        novel_identities=['novel'], unknown_identities=['unknown'], rejection=None,
        destination=tmp_path/'report', progress=False, _on_progress=lambda *event: events.append(event))
    table = pd.read_csv(report.location/'queries.csv')
    np.testing.assert_allclose(table.loc[table.role != 'unknown', 'average_precision'], expected, rtol=1e-12, atol=1e-12)
    assert table.loc[table.role == 'unknown', 'average_precision'].isna().all()
    assert table.identity_id.tolist() == [v for v in labels if v is not None]
    assert len(calls) == 3 and all(n <= 128 for n, _ in calls)
    assert all(reference is calls[0][1] for _, reference in calls)
    assert report.settings['excluded_duplicates'] == 1
    assert events == [('metrics', 0, 263), ('metrics', 263, 263), ('report', 0, 1), ('report', 1, 1)]
    scores = gallery.scores(vectors)
    expected_ranks = [float(1 + np.sum(np.delete(score, codes[label]) >= score[codes[label]]) == 1)
                      for score, label in zip(scores, labels) if label in codes]
    np.testing.assert_array_equal(table.loc[table.role != 'unknown', 'rank1'].to_numpy(), expected_ranks)
    for role in ('known', 'novel'):
        group = table[table.role == role]
        summary = report.ranking[report.ranking.role == role].set_index('weighting')
        assert summary.loc['crop', 'average_precision'] == pytest.approx(group.average_precision.mean())
        balanced = group.groupby(['identity_id', 'clip_id']).average_precision.mean().groupby(level=0).mean().mean()
        assert summary.loc['identity_clip', 'average_precision'] == pytest.approx(balanced)


def test_embedding_progress_cached_quiet_and_failure(identity_data, tmp_path, monkeypatch):
    import torch
    from dataset_fixer.id.training import _IdentityModule
    from dataset_fixer.id.data import ImageRef
    id.register_backbone('progress-tiny', tiny_model, feature_dim=4,
                         preprocessing=dict(resolution=16, mean=[0]*3, std=[1]*3))
    module = _IdentityModule(dict(framework='id', backbone='progress-tiny', identities=['wolf-a','wolf-b'], embedding_dim=4), pretrained=False)
    checkpoint = tmp_path/'model.ckpt'
    torch.save(dict(identity=module.identity, state_dict=module.state_dict()), checkpoint)
    model = Model(checkpoint)
    data = identity_data.sample(7, seed=3)
    events = []
    options = dict(batch_size=3, prediction_cache=tmp_path/'cache', _on_progress=lambda *event: events.append(event))
    fresh = model.embed(data, **options)
    assert events == [('embeddings', 0, 7), ('embeddings', 7, 7)]
    events.clear()
    cached = model.embed(data, progress=False, cache_only=True, **options)
    assert cached.cache_info['hits'] == 7
    np.testing.assert_array_equal([r.embedding for r in fresh], [r.embedding for r in cached])
    assert events == [('embeddings', 0, 7), ('embeddings', 7, 7)]
    events.clear()
    def broken(*args):
        raise OSError('read failed')
    monkeypatch.setattr(ImageRef, 'read', broken)
    with pytest.raises(OSError, match='read failed'):
        model.embed(data, progress=False, **options)
    assert events == [('embeddings', 0, 7)]


@pytest.mark.parametrize('settings', [None, {'console': 'off'}, {'console': 'wrap', 'quiet': True}])
def test_id_console_capture_defaults_preserve_explicit_settings(identity_data, tmp_path, monkeypatch, run, settings):
    import wandb
    from dataset_fixer.id.training import select_identity
    options = {} if settings is None else {'settings': settings}
    captured = []
    monkeypatch.setattr(wandb, 'run', None)
    monkeypatch.setattr(wandb, 'init', lambda **kwargs: captured.append(kwargs) or run)
    config = TrainingConfig(output_dir=tmp_path/'run')
    selection = select_identity(identity_data, backbone='resnet18', config=config)
    TrainingSession()._register(identity_data, selection, config, wandb=WandbConfig(project='wolves', init=options))
    if settings is None:
        assert 'settings' not in captured[0]
    else:
        assert captured[0]['settings'] == settings
    assert 'mode' not in captured[0]
    assert options == ({} if settings is None else {'settings': settings})


def test_periodic_progress_reports_completed_batches_and_failures(monkeypatch, run):
    from dataset_fixer.id import _progress, monitoring
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(_progress, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(monitoring, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    logged, definitions = [], []
    monkeypatch.setattr(run, 'log', lambda values: logged.append(values))
    monkeypatch.setattr(run, 'define_metric', lambda *args, **kwargs: definitions.append((args, kwargs)), raising=False)
    with pytest.raises(OSError, match='broken batch'):
        with monitoring.evaluation_progress(SimpleNamespace(wandb_run=run), 'test') as callback:
            work = _progress.batches(12, 3, stage='embeddings', description='ID embeddings', on_progress=callback)
            assert next(work) == (0, 3)
            clock.now = 12.
            assert next(work) == (3, 6)
            assert logged[-1]['evaluation/test/embeddings_completed'] == 3
            work.close()  # Interrupted work must not claim the second batch completed.
            raise OSError('broken batch')
    completed = [v['evaluation/test/embeddings_completed'] for v in logged if 'evaluation/test/embeddings_completed' in v]
    assert completed == [0, 3]
    assert run.summary['evaluation/test/stage'] == 'failed'
    assert any(kwargs.get('step_metric') == 'evaluation/test/elapsed_seconds' for args, kwargs in definitions)
