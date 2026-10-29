"""Coverage measures completed physical observations, not sampler/prefetch draws."""
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('lightning')

from dataset_fixer import id, TrainingConfig, WandbConfig
from dataset_fixer.id.data import SampleIndex, as_dataset, index_of
from dataset_fixer.id.training import _IdentityDataModule, _IdentityModule
from test_identity import identity_data, tiny_model, pytestmark
from test_training import run


def test_physical_sources_deduplicate_and_only_consumed_rows_count(identity_data):
    frame = index_of(identity_data).frame.copy()
    duplicate = frame.iloc[[0]].copy()
    duplicate['crop'] = [(0, 0, 10, 10)]
    data = as_dataset(SampleIndex(pd.concat([frame, duplicate], ignore_index=True)), parent=identity_data)
    module = SimpleNamespace(identity=dict(identities=['wolf-a', 'wolf-b'], validation_samples=4,
        preprocessing=dict(resolution=16, mean=[0]*3, std=[1]*3)))
    config = TrainingConfig(batch_size=4, workers=0)
    dm = _IdentityDataModule(data, module, config, [])
    assert len(dm.coverage['train']) == len(dm.coverage['val']) == 10
    assert len(dm.val_data) == 4
    first, duplicate = dm.train_data[0], dm.train_data[10]
    assert first[2] == duplicate[2]
    assert not dm.coverage['train'].any()  # Pixel loading/prefetch is not consumption.
    dm.mark_seen('train', torch.tensor([first[2], duplicate[2], first[2]]))
    dm.mark_seen('val', torch.tensor(dm.val_data.source_ids))
    assert dm.coverage['train'].sum() == 1 and dm.coverage['val'].sum() == 4
    restored = _IdentityDataModule(data, module, config, [])
    restored.load_state_dict(dm.state_dict())
    for split in ('train', 'val'):
        np.testing.assert_array_equal(restored.coverage[split], dm.coverage[split])


@pytest.mark.parametrize('validation_sampling', ['fixed', 'cycle'])
def test_completed_batch_coverage_logs_and_survives_lightning_resume(identity_data, tmp_path, monkeypatch, run, validation_sampling):
    from lightning.pytorch.callbacks import Callback
    id.register_backbone('coverage-tiny', tiny_model, feature_dim=4,
                         preprocessing=dict(resolution=16, mean=[0]*3, std=[1]*3))
    completed, logged = {'train': set(), 'val': set()}, []
    original_step = _IdentityModule._step
    def step(self, batch, stage):
        if not self.trainer.sanity_checking:
            completed[stage].update(batch[2].tolist())
        assert batch[2].device.type == 'cpu'
        return original_step(self, batch, stage)
    class CheckSanityCoverage(Callback):
        def on_train_start(self, trainer, module):
            assert not trainer.datamodule.coverage['val'].any()
    monkeypatch.setattr(_IdentityModule, '_step', step)
    monkeypatch.setattr(run, 'log', lambda metrics, **kw: logged.append(metrics))
    monkeypatch.setattr(run, 'define_metric', lambda *a, **kw: None, raising=False)
    config = TrainingConfig(epochs=1, batch_size=4, workers=0, augmentation_samples=0,
        output_dir=tmp_path/'first', backend_options=dict(loss='cross_entropy', pretrained=False,
            embedding_dim=4, embedding_samples=0, validation_samples=4, validation_sampling=validation_sampling, trainer=dict(
                callbacks=[CheckSanityCoverage()], enable_progress_bar=False, enable_model_summary=False)))
    data = identity_data.with_sampling(samples_per_epoch=5)
    first = id.train(data, backbone='coverage-tiny', augmentations=[], config=config, wandb=WandbConfig(run=run))
    assert first.metadata['training']['effective_samples_per_epoch'] == 8
    assert first.metadata['training']['requested_samples_per_epoch'] == 5
    for split in ('train', 'val'):
        counts = next(row for row in first.metadata['dataset_coverage'] if row['split'] == split)
        assert counts['seen'] == len(completed[split]) and counts['total'] == 10
    assert first.metrics['dataset_coverage/percent'] == 5 * sum(map(len, completed.values()))
    assert {key for entry in logged for key in entry if key.startswith('dataset_coverage/')} == {
        'dataset_coverage/chart', 'dataset_coverage/percent'}
    assert any('dataset_coverage/chart' in entry for entry in logged)
    assert first.auxiliary_files['monitoring/coverage.csv'].is_file()
    options = {**config.backend_options, 'trainer': dict(enable_progress_bar=False, enable_model_summary=False)}
    resumed = id.train(data, resume=first.resumable_checkpoint, augmentations=[], wandb=False,
        config=replace(config, epochs=2, output_dir=tmp_path/'resumed', backend_options=options))
    assert resumed.metrics['dataset_coverage/percent'] == 5 * sum(map(len, completed.values()))
    val_counts = next(row for row in resumed.metadata['dataset_coverage'] if row['split'] == 'val')
    assert val_counts['percent'] == (40. if validation_sampling == 'fixed' else 80.)
    stored = torch.load(resumed.resumable_checkpoint, weights_only=False)['_IdentityDataModule']
    assert np.unpackbits(stored['train']['seen'].numpy(), count=10).sum() == len(completed['train'])


def test_cycling_validation_visits_all_sources_with_balanced_identity_shares(identity_data):
    frame = index_of(identity_data).frame
    # Unequal populations; a small identity must not disappear near cycle end.
    val = frame.index[(frame.split == 'val') & (frame.identity_id == 'wolf-a')]
    data = as_dataset(SampleIndex(frame.drop(val[2:]).reset_index(drop=True)), parent=identity_data)
    module = SimpleNamespace(identity=dict(identities=['wolf-a', 'wolf-b'], validation_samples=3,
        validation_sampling='cycle', preprocessing=dict(resolution=16, mean=[0]*3, std=[1]*3)))
    config = TrainingConfig(batch_size=4, workers=0)
    dm = _IdentityDataModule(data, module, config, [])
    original = index_of(data).frame.copy()
    consumed = []
    for epoch in range(4):
        dm._cycle_validation(epoch)
        counts = dm.val_data.index.frame.identity_id.value_counts()
        assert set(counts.index) == {'wolf-a', 'wolf-b'} and counts.max() - counts.min() <= 1
        dm.mark_seen('val', torch.tensor(dm.val_data.source_ids))
        consumed.append(dm.val_data.source_ids.copy())
    assert dm.coverage['val'].all()
    recreated = _IdentityDataModule(data, module, config, [])
    recreated._cycle_validation(3)
    np.testing.assert_array_equal(recreated.val_data.source_ids, consumed[3])
    pd.testing.assert_frame_equal(index_of(data).frame, original)


def test_combined_coverage_deduplicates_sources_shared_between_splits(tmp_path):
    from dataset_fixer.id.monitoring import coverage_epoch
    records = [dict(path=path, split=split, identity_id='a', width=16, height=16)
        for split, paths in [('train', ['common.png', 'train.png']), ('val', ['common.png', 'val.png'])]
        for path in paths]
    module = SimpleNamespace(identity=dict(identities=['a'], preprocessing=dict(resolution=16, mean=[0]*3, std=[1]*3)))
    dm = _IdentityDataModule(as_dataset(SampleIndex(records), location=tmp_path), module,
        TrainingConfig(batch_size=2, workers=0), [])
    result = SimpleNamespace(metadata={}, _coverage_plot_failed=True)
    assert dm.coverage_total == 3
    dm.mark_seen('train', torch.tensor([0]))
    dm.mark_seen('val', torch.tensor([0]))
    assert coverage_epoch(result, dm, 0) == {'dataset_coverage/percent': 100 / 3}
    dm.mark_seen('train', torch.tensor([1]))
    assert coverage_epoch(result, dm, 1) == {'dataset_coverage/percent': 200 / 3}
    dm.mark_seen('val', torch.tensor([1]))
    assert coverage_epoch(result, dm, 2) == {'dataset_coverage/percent': 100.}
