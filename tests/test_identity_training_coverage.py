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


def test_completed_batch_coverage_logs_and_survives_lightning_resume(identity_data, tmp_path, monkeypatch, run):
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
            embedding_dim=4, embedding_samples=0, validation_samples=4, trainer=dict(
                callbacks=[CheckSanityCoverage()], enable_progress_bar=False, enable_model_summary=False)))
    data = identity_data.with_sampling(samples_per_epoch=5)
    first = id.train(data, backbone='coverage-tiny', augmentations=[], config=config, wandb=WandbConfig(run=run))
    assert first.metadata['training']['effective_samples_per_epoch'] == 8
    assert first.metadata['training']['requested_samples_per_epoch'] == 5
    for split in ('train', 'val'):
        assert first.metrics[f'dataset_coverage/{split}_seen'] == len(completed[split])
        assert first.metrics[f'dataset_coverage/{split}_total'] == 10
        assert first.metrics[f'dataset_coverage/{split}_percent'] == 10 * len(completed[split])
    assert any('dataset_coverage/chart' in entry for entry in logged)
    assert first.auxiliary_files['monitoring/coverage.csv'].is_file()
    options = {**config.backend_options, 'trainer': dict(enable_progress_bar=False, enable_model_summary=False)}
    resumed = id.train(data, resume=first.resumable_checkpoint, augmentations=[], wandb=False,
        config=replace(config, epochs=2, output_dir=tmp_path/'resumed', backend_options=options))
    assert resumed.metrics['dataset_coverage/train_seen'] == len(completed['train'])
    assert resumed.metrics['dataset_coverage/val_percent'] == 40.
    stored = torch.load(resumed.resumable_checkpoint, weights_only=False)['_IdentityDataModule']
    assert np.unpackbits(stored['train']['seen'].numpy(), count=10).sum() == len(completed['train'])
