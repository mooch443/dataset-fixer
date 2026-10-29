"""Optional embedding diagnostics must not interrupt model training/publication."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('lightning')

from dataset_fixer import id, TrainingConfig
from dataset_fixer.id import monitoring
from dataset_fixer.id.training import _IdentityModule
from test_identity import identity_data, tiny_model, pytestmark


def monitor_result(data, path, count=12):
    return SimpleNamespace(dataset=data, config=TrainingConfig(batch_size=4,
        backend_options=dict(embedding_samples=count, embedding_method='pca')),
        output_dir=path, auxiliary_files={}, wandb_run=None)


def monitor_module():
    id.register_backbone('safe-monitor-tiny', tiny_model, feature_dim=4,
                         preprocessing=dict(resolution=16, mean=[0]*3, std=[1]*3))
    return _IdentityModule(dict(backbone='safe-monitor-tiny', identities=['wolf-a', 'wolf-b'],
                                embedding_dim=4, loss='cross_entropy'), pretrained=False)


def test_large_monitor_request_is_bounded(identity_data, tmp_path, monkeypatch):
    monkeypatch.setattr(monitoring, '_MAX_EMBEDDING_SAMPLES', 12)
    result = monitor_result(identity_data, tmp_path, 10**20)
    monitoring.prepare_monitor(result)
    assert len(result._embedding_cohort) == 12
    assert result._embedding_cohort.frame.split.value_counts().loc[['train', 'val']].tolist() == [6, 6]
    monitoring.prepare_monitor(result)
    result = monitor_result(identity_data, tmp_path/'disabled', 0)
    monkeypatch.setattr(monitoring, 'index_of', lambda *a: pytest.fail('Disabled monitor must not select data'))
    monitoring.prepare_monitor(result)
    assert result._embedding_cohort is None


def test_monitor_memory_failure_reduces_next_attempt_and_preserves_rng(identity_data, tmp_path, monkeypatch):
    import random
    module = monitor_module().train()
    result = monitor_result(identity_data, tmp_path)
    monitoring.prepare_monitor(result)
    native = module.forward
    def oom(images):
        random.random()
        np.random.rand()
        torch.rand(1)
        raise torch.OutOfMemoryError('test out of memory')
    monkeypatch.setattr(module, 'forward', oom)
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    assert monitoring.monitor_epoch(result, module, 0)['embeddings/skipped'] == 1
    assert module.training
    assert random.getstate() == before[0]
    np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])
    assert len(result._embedding_cohort) == 6 and result._embedding_batch_size == 2
    monkeypatch.setattr(module, 'forward', native)
    metrics = monitoring.monitor_epoch(result, module, 1)
    assert metrics['embeddings/samples'] == 6 and metrics['embeddings/skipped'] == 0
    cohort = result._embedding_cohort.frame.copy()
    monitoring.monitor_epoch(result, module, 2)
    assert result._embedding_cohort.frame.equals(cohort)


def test_cohort_preparation_failure_is_optional(identity_data, tmp_path, monkeypatch):
    result = monitor_result(identity_data, tmp_path)
    def fail(*args):
        raise OSError('unavailable diagnostic storage')
    monkeypatch.setattr(monitoring, '_save_monitor_cohort', fail)
    monitoring.prepare_monitor(result)
    for epoch in range(2):
        assert monitoring.monitor_epoch(result, None, epoch)['embeddings/skipped'] == 1


def test_failed_monitor_does_not_block_training_or_checkpoint_bundle(identity_data, tmp_path, monkeypatch):
    monitor_module()
    def fail(*args, **kwargs):
        raise OSError('diagnostic renderer unavailable')
    monkeypatch.setattr('dataset_fixer.static_rendering.save_chart', fail)
    result = id.train(identity_data.with_sampling(samples_per_epoch=8), backbone='safe-monitor-tiny',
        augmentations=[], wandb=False, config=TrainingConfig(epochs=2, batch_size=4, workers=0,
            augmentation_samples=0, output_dir=tmp_path/'run', backend_options=dict(
                pretrained=False, loss='cross_entropy', embedding_dim=4, embedding_samples=12,
                trainer=dict(enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0))))
    assert result.best_weights.is_file() and result.resumable_checkpoint.is_file() and result.bundle.path.is_file()
    checkpoint = torch.load(result.resumable_checkpoint, weights_only=False)
    assert checkpoint['epoch'] == 1 and checkpoint['global_step'] == 4
    assert result._embedding_disabled


def test_monitor_does_not_swallow_user_interrupt(identity_data, tmp_path, monkeypatch):
    result = monitor_result(identity_data, tmp_path)
    monitoring.prepare_monitor(result)
    def interrupt(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(monitoring, '_monitor_epoch', interrupt)
    with pytest.raises(KeyboardInterrupt):
        monitoring.monitor_epoch(result, None, 0)
