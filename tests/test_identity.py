from pathlib import Path
from importlib.util import find_spec
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from dataset_fixer import Dataset, id, TrainingConfig
from dataset_fixer.id.data import ImageRef, SampleIndex, index_of, as_dataset

pytestmark = pytest.mark.skipif(
    any(find_spec(package) is None for package in ('lightning', 'timm', 'pytorch_metric_learning', 'sklearn', 'mmnpz', 'captum')),
    reason="Install the optional dataset-fixer[id] extra to run identification integration tests",
)


@pytest.fixture
def identity_data(tmp_path):
    records = []
    rng = np.random.default_rng(11)
    for split in ('train', 'val', 'test'):
        for label in ('wolf-a', 'wolf-b'):
            for row in range(5):
                path = tmp_path / split / label / f'{split}-clip' / f'{row}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                pixels = rng.integers(0, 60, (20, 24, 3), dtype=np.uint8)
                pixels[..., 0 if label == 'wolf-a' else 1] += 150
                Image.fromarray(pixels).save(path)
                records.append(dict(path=str(path), identity_id=label, clip_id=f'{split}-clip',
                                    tracklet_id=str(row//2), width=24, height=20, split=split))
    return as_dataset(SampleIndex(records), location=tmp_path, name='wolves')


def test_folders_manifest_equivalence(identity_data, tmp_path):
    folders = Dataset.open(tmp_path, format='identity_folders', layout='{split}/{identity_id}/{clip_id}/{image}')
    rows = index_of(folders).frame.drop(columns=['crop']).astype(object).where(pd.notna(index_of(folders).frame.drop(columns=['crop'])), None).to_dict('records')
    path = tmp_path / 'samples.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in rows))
    manifest = Dataset.open(path, format='identity_manifest')
    assert folders.classes == manifest.classes
    assert len(folders._samples) == len(manifest._samples) == 30
    for a, b in zip(folders._samples, manifest._samples):
        np.testing.assert_array_equal(a.image_ref.read(), b.image_ref.read())
        assert (a.identity_id, a.clip_id, a.split) == (b.identity_id, b.clip_id, b.split)


def test_clip_views_and_sampling(identity_data):
    data = identity_data
    held = data.holdout(clip_id='train-clip', identity_ids=['wolf-a'])
    assert 'share frames/backgrounds' in held.warnings[-1]
    assert (index_of(data).frame.split == 'train').sum() == 10
    assert (index_of(held).frame.split == 'train').sum() == 5
    split = data.split({'train': .5, 'val': .5}, group_by='clip_id')
    assert index_of(split).frame.groupby('clip_id', observed=True).split.nunique().max() == 1
    balanced = data.with_sampling(samples_per_epoch=101, balance_by=('identity_id', 'clip_id', 'tracklet_id'), max_per_batch={'tracklet_id': 1})
    from dataset_fixer.id.training import _DiverseBatches
    train = index_of(balanced).where(split='train')
    sampler = _DiverseBatches(train, 4, 7)
    first = list(sampler)
    assert sum(map(len, first)) == 101
    for batch in first:
        keys = [(train.frame.iloc[i].clip_id, train.frame.iloc[i].tracklet_id) for i, _, _ in batch]
        assert len(keys) == len(set(keys))
    assert first == list(sampler)
    sampler.epoch = 1
    assert first != list(sampler)
    assert len(set(i for batch in first for i, _, _ in batch)) == len(train)


def test_npz_parity_lazy(tmp_path, monkeypatch):
    rng = np.random.default_rng(9)
    pixels = rng.integers(0, 256, (12, 8, 10, 3), dtype=np.uint8)
    path = tmp_path / 'clip' / 'images.npz'
    path.parent.mkdir()
    np.savez(path, images=pixels, ids=np.repeat([2, 3], 6), frames=np.tile(np.arange(6), 2),
             frame_segment_indexes=np.repeat([0, 1], 6), tracklets=np.array([[0, 5], [0, 5]]))
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('NPZ importer must not load full arrays'))
    data = Dataset.open(path, format='trex_npz')
    assert index_of(data).frame.identity_id.isna().all()
    assert index_of(data).frame.tracklet_id.nunique() == 2
    for row in (0, 7, 11):
        np.testing.assert_array_equal(data._samples[row].image_ref.read(), pixels[row])


def test_crop_context_and_custom_pipeline(tmp_path):
    from dataset_fixer.models import Sample, Annotation, Task, DatasetMetadata
    image = tmp_path / 'animal.png'
    Image.fromarray(np.zeros((100, 150, 3), dtype=np.uint8)).save(image)
    source = Dataset(location=tmp_path, name='animal', task=Task.DETECT, metadata=DatasetMetadata({0:'a'}),
        samples=[Sample(image, Path('animal.png'), 'train', 150, 100, [Annotation(0, bbox=(20, 10, 120, 90))])],
        manifest={}, data_yaml=None, source_format='test', warnings=[])
    cropped = id.from_dataset(source, identity='class', context_fraction=.1)
    assert cropped._samples[0].image_ref.crop == (10, 2, 130, 98)
    import albumentations as A
    from dataset_fixer.id.model import augmentation_pipeline, preprocess
    aug, _ = augmentation_pipeline([A.Crop(x_min=0, y_min=0, x_max=30, y_max=40)])
    raw = aug(image=cropped._samples[0].image_ref.read())['image']
    assert raw.shape == (40, 30, 3)
    assert preprocess(raw, dict(resolution=16, mean=[0]*3, std=[1]*3)).shape == (3, 16, 16)


def tiny_model():
    import torch
    return torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())


def test_train_reload_transfer_resume(identity_data, tmp_path, monkeypatch):
    import torch
    monkeypatch.setenv('DATASET_FIXER_CACHE', str(tmp_path/'cache'))
    id.register_backbone('test-tiny', tiny_model, feature_dim=4,
                         preprocessing=dict(resolution=16, mean=[0]*3, std=[1]*3))
    options = dict(embedding_dim=8, loss='cross_entropy', pretrained=False, validation_samples=6,
                   trainer=dict(enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0))
    config = TrainingConfig(epochs=1, batch_size=4, workers=0, augmentation_samples=4,
                            output_dir=tmp_path/'run', backend_options=options)
    result = id.train(identity_data, backbone='test-tiny', augmentations=[], config=config, wandb=False)
    assert result.best_weights.is_file() and result.resumable_checkpoint.is_file()
    assert result.bundle.path.is_file()
    assert result.auxiliary_files['training/augmentations.png'].is_file()
    assert result.metadata['training']['validation_samples'] == 6
    assert result.metadata['training']['validation_total'] == 10
    from dataset_fixer import Model
    local, bundle = Model(result.best_weights), Model(result.bundle.path)
    assert local.kind == bundle.kind == 'id'
    gallery = local.enroll(result._identity_protocol.references)
    a = local.predict(result._identity_protocol.queries['test'], gallery=gallery, prediction_cache=False)
    b = bundle.predict(result._identity_protocol.queries['test'], gallery=gallery, prediction_cache=False)
    np.testing.assert_allclose([r.embedding for r in a], [r.embedding for r in b])
    assert all(r.identity_id in gallery.identities for r in a)
    report = result.evaluate(split='test', plots=0)
    assert 'rank1' in report.ranking
    assert len(index_of(result._identity_protocol.queries['test'])) == 10
    with pytest.raises(ValueError, match='same validation_samples'):
        id.train(identity_data, resume=result.resumable_checkpoint, config=TrainingConfig(
            epochs=2, batch_size=4, workers=0, augmentation_samples=0, output_dir=tmp_path/'changed-validation',
            backend_options={**options, 'validation_samples': 4}))
    legacy = torch.load(result.resumable_checkpoint, weights_only=False)
    legacy['identity'].pop('augmentation_rng')
    legacy_path = tmp_path/'legacy.ckpt'
    torch.save(legacy, legacy_path)
    with pytest.raises(ValueError, match='same augmentation_rng'):
        id.train(identity_data, resume=legacy_path, config=TrainingConfig(
            epochs=2, batch_size=4, workers=0, augmentation_samples=0, output_dir=tmp_path/'legacy-resume',
            backend_options=options))
    resumed = id.train(identity_data, resume=result.resumable_checkpoint, augmentations=[],
        config=TrainingConfig(epochs=2, batch_size=4, workers=0, augmentation_samples=0, output_dir=tmp_path/'resumed', backend_options=options))
    assert torch.load(resumed.resumable_checkpoint, weights_only=False)['epoch'] == 1
    uninterrupted = id.train(identity_data, backbone='test-tiny', augmentations=[],
        config=TrainingConfig(epochs=2, batch_size=4, workers=0, augmentation_samples=0,
                              output_dir=tmp_path/'uninterrupted', backend_options=options))
    continued_state = torch.load(resumed.resumable_checkpoint, weights_only=False)['state_dict']
    full_state = torch.load(uninterrupted.resumable_checkpoint, weights_only=False)['state_dict']
    for key in full_state:
        torch.testing.assert_close(full_state[key], continued_state[key], rtol=0, atol=0)
    frame = index_of(identity_data).frame.copy()
    frame['identity_id'] = frame.identity_id.astype(str) + '-new'
    transferred = id.train(as_dataset(SampleIndex(frame), location=tmp_path), weights=local, augmentations=[],
        config=TrainingConfig(epochs=1, batch_size=4, workers=0, augmentation_samples=0, output_dir=tmp_path/'transferred', backend_options=options))
    assert all(v.endswith('-new') for v in torch.load(transferred.best_weights, weights_only=False)['identity']['identities'])


def test_lightning_leafspec_warning_filter_is_scoped(identity_data, tmp_path, monkeypatch):
    import warnings
    from lightning.pytorch import Trainer
    from lightning.pytorch.utilities import _pytree
    message = ('`isinstance(treespec, LeafSpec)` is deprecated, '
               'use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.')
    native_fit = Trainer.fit
    def fit(self, *args, **kwargs):
        # Probe the exact upstream warning even after Lightning eventually fixes it.
        warnings.warn_explicit(message, FutureWarning, _pytree.__file__, 21,
                               module=_pytree.__name__)
        warnings.warn_explicit('Another pytree deprecation', FutureWarning, _pytree.__file__, 22,
                               module=_pytree.__name__)
        warnings.warn(message, FutureWarning)  # Same message outside Lightning must survive.
        return native_fit(self, *args, **kwargs)
    monkeypatch.setattr(Trainer, 'fit', fit)
    id.register_backbone('warning-tiny', tiny_model, feature_dim=4,
                         preprocessing=dict(resolution=16, mean=[0]*3, std=[1]*3))
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        result = id.train(identity_data, backbone='warning-tiny', augmentations=[], wandb=False,
            config=TrainingConfig(epochs=1, batch_size=4, workers=0, augmentation_samples=0,
                output_dir=tmp_path/'run', backend_options=dict(embedding_dim=4, embedding_samples=0,
                    loss='cross_entropy', pretrained=False,
                    trainer=dict(enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0))))
        leaf_warnings = [w for w in seen if str(w.message) == message]
        assert len(leaf_warnings) == 1 and leaf_warnings[0].filename == __file__
        assert any(str(w.message) == 'Another pytree deprecation' for w in seen)
        warnings.warn_explicit(message, FutureWarning, _pytree.__file__, 21, module=_pytree.__name__)
        assert str(seen[-1].message) == message and seen[-1].filename == _pytree.__file__
    assert result.best_weights.is_file() and result.resumable_checkpoint.is_file()


def test_threshold_ties_library():
    from dataset_fixer.id.evaluation import threshold_at_fpir
    scores = np.array([.2, .3, .5, .5, .8])
    for target in (0, .01, .2, .4, 1):
        threshold, achieved = threshold_at_fpir(scores, target)
        assert achieved == pytest.approx(np.mean(scores >= threshold))
        assert achieved <= target


def test_open_protocol_excludes_roles_and_calibrates(identity_data, tmp_path):
    from dataset_fixer.id.evaluation import calibrate, evaluate
    from dataset_fixer import Model
    import torch
    frame = index_of(identity_data).frame.astype(object).copy()
    additions = []
    rng = np.random.default_rng(51)
    for split in ('val', 'test'):
        for role in ('unknown', 'novel'):
            label = f'{split}-{role}'
            for clip in range(2):
                for row in range(2):
                    path = tmp_path/f'{label}-{clip}-{row}.png'
                    Image.fromarray(rng.integers(0, 256, (20,24,3), dtype=np.uint8)).save(path)
                    additions.append(dict(path=str(path), identity_id=label, clip_id=f'{label}-{clip}',
                                          split=split, width=24, height=20))
    frame = pd.concat([frame, pd.DataFrame(additions)], ignore_index=True)
    data = as_dataset(SampleIndex(frame), location=tmp_path)
    protocol = id.Protocol(data, unknown_identities={'val':['val-unknown'], 'test':['test-unknown']},
                           novel_identities={'val':['val-novel'], 'test':['test-novel']}, gallery_samples=3)
    assert set(index_of(protocol.training).frame.identity_id.dropna()) == {'wolf-a','wolf-b'}
    assert set(index_of(protocol.references).frame.identity_id.dropna()) == {'wolf-a','wolf-b','val-novel','test-novel'}
    for split in ('val','test'):
        label = f'{split}-novel'
        reference_clips = set(index_of(protocol.references).where(identity_id=label).frame.clip_id)
        query_clips = set(index_of(protocol.queries[split]).where(identity_id=label).frame.clip_id)
        assert reference_clips.isdisjoint(query_clips)
    id.register_backbone('protocol-tiny', tiny_model, feature_dim=4,
                         preprocessing=dict(resolution=16, mean=[0]*3, std=[1]*3))
    from dataset_fixer.id.training import _IdentityModule
    module = _IdentityModule(dict(framework='id', backbone='protocol-tiny', identities=['wolf-a','wolf-b'], embedding_dim=8), pretrained=False)
    checkpoint = tmp_path/'model.ckpt'
    torch.save({'identity':module.identity, 'state_dict':module.state_dict()},checkpoint)
    model = Model(checkpoint)
    gallery = model.enroll(protocol.references, samples_per_identity=3)
    calibration = calibrate(model, protocol.queries['val'], gallery=gallery,
                            unknown_identities=protocol.unknown_identities['val'], target_fpir=.01)
    assert calibration.achieved_fpir <= .01
    assert calibration.limitations
    validation = model.predict(protocol.queries['val'], gallery=gallery, mode='open', rejection=calibration)
    assert all(r.rejected for r in validation if r.metadata['reference_identity']=='val-unknown')
    predictions = model.predict(protocol.queries['test'], gallery=gallery, mode='open', rejection=calibration)
    assert all(isinstance(r.rejected, bool) for r in predictions)
    report = evaluate(model, protocol.queries['test'], gallery=gallery, rejection=calibration,
        unknown_identities=protocol.unknown_identities['test'], novel_identities=protocol.novel_identities['test'], destination=tmp_path/'report', plots=3)
    assert (report.location/'predictions.png').is_file()
    assert set(report.ranking.role) == {'known','novel','unknown'}
    assert report.ranking.loc[report.ranking.role == 'unknown', 'false_positive'].notna().all()
    with pytest.raises(ValueError, match='changed'):
        from dataclasses import replace
        replace(calibration, gallery_fingerprint='different').validate(model,gallery)
    with pytest.raises(ValueError, match='disjoint'):
        id.Protocol(data, unknown_identities={'val':['val-unknown'],'test':['val-unknown']})
    saved = gallery.save(tmp_path/'gallery.npz')
    assert id.Gallery.load(saved).fingerprint == gallery.fingerprint
    predictions.visualize(samples=4, destination=tmp_path/'predictions.png', show=False)


def test_manifest_export_and_preview_rng(identity_data, tmp_path):
    import random
    import torch
    import albumentations as A
    from dataset_fixer.training.preview import visualize_augmentations
    from dataset_fixer.visualization import VisualizationOptions
    exported = identity_data.export(destination=tmp_path/'manifest', format='identity_manifest', visualize=False)
    assert len(index_of(exported)) == len(index_of(identity_data))
    assert len(index_of(identity_data.sample(4))) == 4
    transform = A.Compose([A.RandomCrop(10, 10)], seed=13)
    before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state(), transform.random_generator.bit_generator.state)
    chart = visualize_augmentations(identity_data, augmentations=transform, weights=None, type='id', version=None, s=None,
        model_type='id', config=TrainingConfig(resolution=16), split='train', options=VisualizationOptions(samples=4, show=False))
    after = (random.getstate(), np.random.get_state(), torch.random.get_rng_state(), transform.random_generator.bit_generator.state)
    assert before[0] == after[0] and before[3] == after[3]
    np.testing.assert_array_equal(before[1][1],after[1][1])
    assert torch.equal(before[2],after[2])
    assert chart is None


# Reuse the shared session's W&B test double, including confirmation/retention rules.
from test_training import run


def test_identity_wandb_lifecycle(identity_data, tmp_path, monkeypatch, run):
    import io
    import zipfile
    import torch
    import dataset_fixer as df
    from dataset_fixer.id.training import _IdentityModule
    id.register_backbone('wandb-tiny', tiny_model, feature_dim=4,
                         preprocessing=dict(resolution=16, mean=[0]*3, std=[1]*3))
    logged = []
    definitions = []
    monkeypatch.setattr(run, 'log', lambda value, **kw: logged.append(value))
    monkeypatch.setattr(run, 'define_metric', lambda *args, **kwargs: definitions.append((args, kwargs)), raising=False)
    def validation(self, batch, index):
        self.log('val_loss', torch.tensor([3.,2.,2.,1.][self.current_epoch]), on_epoch=True, batch_size=len(batch[1]))
    monkeypatch.setattr(_IdentityModule, 'validation_step', validation)
    with df.TrainingSession(wandb=df.WandbConfig(run=run)) as session:
        result = id.train(identity_data, backbone='wandb-tiny', augmentations=[], session=session,
            config=TrainingConfig(epochs=4,batch_size=4,workers=0,augmentation_samples=3,output_dir=tmp_path/'wandb-run',
                backend_options=dict(pretrained=False,embedding_dim=8,trainer=dict(enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0))))
        assert result.checkpoint_config.wandb_upload == 'best'
        assert len(run.artifacts) == 3  # first best plus two improvements, no plateau upload
        result.evaluate(split='test',plots=0)
    assert any('training/augmentations' in entry for entry in logged)
    batches = [entry for entry in logged if 'progress/batch' in entry]
    assert [entry['progress/epoch'] for entry in batches] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [entry['progress/batch'] for entry in batches] == [1, 3]*4
    assert all(np.isfinite(entry['train/batch_loss']) for entry in batches)
    assert all(a['progress/global_step'] < b['progress/global_step'] for a, b in zip(batches, batches[1:]))
    assert any(args == ('train/batch_loss',) and kwargs['step_metric'] == 'progress/global_step' for args, kwargs in definitions)
    projections = [entry for entry in logged if 'embeddings/tsne' in entry]
    assert [entry['epoch'] for entry in projections] == [0,1,2,3]
    assert any(args == ('val/loss',) and kwargs['summary']=='min' for args,kwargs in definitions)
    assert any('optim/lr' in entry and 'train/loss' in entry and 'val/loss' in entry for entry in logged)
    assert not any('train_loss' in entry or 'val_loss' in entry for entry in logged)
    evaluation = [entry for entry in logged if 'evaluation/test/elapsed_seconds' in entry]
    assert evaluation and run.summary['evaluation/test/stage'] == 'complete'
    assert evaluation[-1]['evaluation/test/report_completed'] == 1
    assert any(entry.get('evaluation/test/metrics_completed') == 10 for entry in evaluation)
    assert len(list(run.logged_artifacts())) == 1
    assert all(artifact.waited for artifact in run.artifacts)
    assert not run.closed  # caller-owned run
    with zipfile.ZipFile(io.BytesIO(next(iter(run.artifacts[-1].files.values())))) as archive:
        assert 'training/augmentations.png' in archive.namelist()
        assert 'evaluation/test/metrics.json' in archive.namelist()
        assert 'monitoring/embeddings.png' in archive.namelist()
        assert 'monitoring/cohort.json' in archive.namelist()
        checkpoint = next(name for name in archive.namelist() if name.startswith('weights/') and name.endswith('.ckpt'))
        assert 'optimizer_states' not in torch.load(io.BytesIO(archive.read(checkpoint)),weights_only=False)
    monkeypatch.setattr('dataset_fixer.model_sources._download_wandb', lambda *a, **k: (result.wandb_bundle.path,run))
    remote = df.Model('wandb:team/project/runs/test')
    assert remote.kind == 'id'
    result._enrollment().validate(remote)
    np.testing.assert_allclose(remote.embed(identity_data.sample(3),prediction_cache=False)[0].embedding,
                               result.model.embed(identity_data.sample(3),prediction_cache=False)[0].embedding)


def test_resume_preserves_best_when_validation_does_not_improve(identity_data, tmp_path, monkeypatch):
    import torch
    from dataset_fixer.id.training import _IdentityModule
    id.register_backbone('resume-tiny', tiny_model, feature_dim=4,
                         preprocessing=dict(resolution=16,mean=[0]*3,std=[1]*3))
    def validate(self,batch,index):
        self.log('val_loss',torch.tensor(1. if self.current_epoch == 0 else 2.),batch_size=len(batch[1]))
    monkeypatch.setattr(_IdentityModule,'validation_step',validate)
    options=dict(embedding_dim=8,loss='cross_entropy',pretrained=False,
                 trainer=dict(enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0))
    first=id.train(identity_data,backbone='resume-tiny',augmentations=[],config=TrainingConfig(
        epochs=1,batch_size=4,workers=0,augmentation_samples=0,output_dir=tmp_path/'first',backend_options=options))
    resumed=id.train(identity_data,resume=first.bundle.path,config=TrainingConfig(
        epochs=2,batch_size=4,workers=0,augmentation_samples=0,output_dir=tmp_path/'second',backend_options=options))
    original=torch.load(first.best_weights,weights_only=False)
    selected=torch.load(resumed.best_weights,weights_only=False)
    assert selected['epoch']==original['epoch']==0
    assert resumed.best_value == 1. and resumed.best_epoch == 0
    for key,value in original['state_dict'].items():
        assert torch.equal(value,selected['state_dict'][key])


@pytest.mark.parametrize('accumulation', [1, 2])
def test_native_lightning_display_and_batch_telemetry(identity_data, tmp_path, monkeypatch, capsys, run, accumulation):
    import dataset_fixer as df
    from lightning.pytorch.callbacks import ModelSummary, TQDMProgressBar, RichProgressBar, RichModelSummary
    from lightning.pytorch.trainer.connectors.callback_connector import _RICH_AVAILABLE
    from dataset_fixer.id import training
    id.register_backbone('display-tiny', tiny_model, feature_dim=4,
                         preprocessing=dict(resolution=16, mean=[0]*3, std=[1]*3))
    logged, trainers, losses = [], [], []
    monkeypatch.setattr(run, 'log', lambda value, **kw: logged.append(value))
    monkeypatch.setattr(run, 'define_metric', lambda *args, **kwargs: None, raising=False)
    original_step = training._IdentityModule._step
    def step(self, batch, stage):
        loss = original_step(self, batch, stage)
        if stage == 'train':
            losses.append(float(loss.detach()))
        return loss
    monkeypatch.setattr(training._IdentityModule, '_step', step)
    native = training.Trainer
    def build(**kwargs):
        trainer = native(**kwargs)
        trainers.append(trainer)
        return trainer
    monkeypatch.setattr(training, 'Trainer', build)
    result = id.train(identity_data.with_sampling(samples_per_epoch=8), backbone='display-tiny',
        augmentations=[], wandb=df.WandbConfig(run=run), config=TrainingConfig(
            epochs=1, batch_size=4, workers=0, augmentation_samples=0, output_dir=tmp_path/'display',
            backend_options=dict(pretrained=False, embedding_dim=4, embedding_samples=0,
                                 trainer=dict(precision='bf16-mixed', log_every_n_steps=2,
                                              accumulate_grad_batches=accumulation))))
    bar_type = RichProgressBar if _RICH_AVAILABLE else TQDMProgressBar
    summary_type = RichModelSummary if _RICH_AVAILABLE else ModelSummary
    assert any(type(cb) is bar_type for cb in trainers[0].callbacks)
    assert any(type(cb) is summary_type for cb in trainers[0].callbacks)
    progress = [entry for entry in logged if 'progress/batch' in entry]
    assert [entry['progress/global_step'] for entry in progress] == ([1, 2] if accumulation == 1 else [0, 1])
    assert all(np.isfinite(entry['train/batch_loss']) for entry in progress)
    assert [entry['train/batch_loss'] for entry in progress] == pytest.approx(losses)
    assert result.best_weights.is_file()
    stdout = capsys.readouterr().out
    assert 'validation sanity check complete' in stdout
    assert 'first training batch completed' in stdout
    assert 'training batch 2/2' in stdout


def test_custom_instance_freeze_and_cache(identity_data,tmp_path):
    import torch
    from dataset_fixer import Model
    from dataset_fixer.id.training import _IdentityModule, select_identity
    custom=tiny_model()
    id.register_backbone('instance-tiny',tiny_model,model=custom,feature_dim=4,
        preprocessing=dict(resolution=16,mean=[0]*3,std=[1]*3))
    selection=select_identity(identity_data,backbone=custom)
    assert selection.name=='instance-tiny' and selection.metadata['instance'] is custom
    module=_IdentityModule(dict(framework='id',backbone='instance-tiny',identities=['wolf-a','wolf-b'],
                                freeze_backbone=True,embedding_dim=0),pretrained=False,instance=custom)
    module.train()
    assert not module.encoder.training and all(not p.requires_grad for p in module.encoder.parameters())
    path=tmp_path/'custom.ckpt'
    torch.save({'identity':module.identity,'state_dict':module.state_dict()},path)
    model=Model(path)
    subset=identity_data.sample(2)
    first=model.embed(subset)
    second=model.embed(subset,cache_only=True)
    assert first.cache_info['misses']==second.cache_info['hits']==2
    np.testing.assert_allclose(first[0].embedding,second[0].embedding)
    assert first[0].embedding.shape==(4,)
    first.save(tmp_path/'predictions')
    assert 'embedding' in json.loads((tmp_path/'predictions/predictions.json').read_text())[0]['metadata']


def test_protocol_rejects_reused_source_observation(identity_data,tmp_path):
    frame=index_of(identity_data).frame.copy().astype(object)
    train=frame.iloc[0].copy()
    train['split']='val'
    train['clip_id']='different-clip-name'
    duplicated=as_dataset(SampleIndex(pd.concat([frame,pd.DataFrame([train])],ignore_index=True)),location=tmp_path)
    with pytest.raises(ValueError,match='share source observations'):
        id.Protocol(duplicated)


def test_timm_vit_resolution_and_hierarchical_weights():
    import torch
    from dataset_fixer.id.model import build_backbone
    encoder,dim,prep=build_backbone('timm:vit_tiny_patch16_224',pretrained=False,resolution=32)
    with torch.inference_mode():
        assert encoder(torch.zeros(1,3,32,32)).shape==(1,dim)
    records=[]
    for identity,clip,count in [('a','long',20),('a','short',2),('b','only',3)]:
        records.extend(dict(path='unused',width=1,height=1,identity_id=identity,clip_id=clip) for _ in range(count))
    index=SampleIndex(records)
    weights=index.weights(('identity_id','clip_id'))
    assert weights[:20].sum()==pytest.approx(weights[20:22].sum())
    assert weights[:22].sum()==pytest.approx(weights[22:].sum())


def test_workbook_notes_unmapped_and_npz_part_groups(tmp_path):
    from openpyxl import Workbook
    workbook=Workbook();sheet=workbook.active;sheet.title='Global ID Summary'
    clip='2509_202_BM3_GCP4_SH'
    sheet.append(['global_id',clip,'note'])
    sheet.append(['wolf-a','202_ID03 -> 80-100%','qualified'])
    sheet.append(['wolf-b','202_ID04',''])
    workbook.save(tmp_path/'identities.xlsx')
    folder=tmp_path/clip;folder.mkdir()
    for part in range(2):
        np.savez(folder/f'part{part}.npz',images=np.full((3,8,8,3),part,dtype=np.uint8),
            ids=np.array([3,4,9]),frames=np.arange(3)+part*3,
            frame_segment_indexes=np.array([100,101,102]),tracklets=np.array([[0,5]]))
    data=Dataset.open(folder,format='trex_npz',identities=tmp_path/'identities.xlsx')
    index=index_of(data)
    assert index.frame.identity_id.notna().sum()==4
    assert '80-100%' in index.frame.notes.dropna().iloc[0]
    assert index.frame.tracklet_id.nunique()==3
    assert data._metadata.extra['tracklets']
    assert any('global tracklet indexes' in warning for warning in data.warnings)
    assert data._samples[0].local_id==3 and isinstance(data._samples[0].local_id,int)


def test_metadata_views_and_prediction_filter_keep_npz_rows_lazy(tmp_path, monkeypatch):
    from dataset_fixer import ImagePrediction, PredictionResult
    archive = tmp_path/'crops.npz'
    pixels = np.arange(4*8*8*3, dtype=np.uint8).reshape(4,8,8,3)
    np.savez(archive, images=pixels)
    index = SampleIndex([dict(path=str(archive), array='images', row=row, width=8, height=8,
                             identity_id='wolf-a' if row<3 else None, clip_id='clip') for row in range(4)])
    data = as_dataset(index, location=tmp_path)
    records = tuple(ImagePrediction(str(row), archive, str(row), 8, 8,
        metadata=dict(embedding=[1.,0.], identity_id='wolf-a' if row==2 else None, rejected=row!=2,
                      source_ref=ImageRef(str(archive), 'images', row).key)) for row in (1,2))
    predictions = PredictionResult('test', 'id', 'id', 'torch', records, 0.)
    monkeypatch.setattr(SampleIndex, '__getitem__', lambda *args: pytest.fail('Metadata views must not construct image samples'))
    assert 'wolf-a' in str(data)
    assert data._summary_counts()['annotations']==3
    assert len(index_of(data.filter(gt_annotated=False)))==1
    positive = data.filter(predictions=predictions, has_prediction=True)
    negative = data.filter(predictions=predictions, has_prediction=False)
    assert index_of(positive).frame.row.tolist()==[2]
    assert index_of(negative).frame.row.tolist()==[1]
    assert index_of(data.filter(identity_ids=['wolf-a'], predictions=predictions,
                                has_prediction=True)).frame.row.tolist() == [2]


def test_identity_selection_is_lazy_and_combines_filters(identity_data, monkeypatch):
    before = index_of(identity_data).frame.copy()
    monkeypatch.setattr(SampleIndex, '__getitem__', lambda *args: pytest.fail('Filtering must read metadata only'))
    selected = identity_data.filter(identity_ids=['wolf-b'], gt_annotated=True, split='val')
    assert id.stats(selected).identity_id.tolist() == ['wolf-b']
    assert len(index_of(selected)) == 5
    assert selected.splits == ('val',)
    assert len(index_of(identity_data.filter(identity_ids=['wolf-a']))) == 15
    assert len(index_of(identity_data.filter(identity_ids=[]))) == 0
    pd.testing.assert_frame_equal(index_of(identity_data).frame, before)
    with pytest.raises(ValueError, match='Unknown identity_ids'):
        identity_data.filter(identity_ids=['typo'])
    with pytest.raises(TypeError, match='sequence'):
        identity_data.filter(identity_ids='wolf-a')
    with pytest.raises(TypeError, match='string labels'):
        identity_data.filter(identity_ids=[1])


def test_arcface_custom_training_advances_prefetched_sampling(identity_data, tmp_path, monkeypatch):
    from dataset_fixer.id.training import _DiverseBatches
    custom = tiny_model()
    id.register_backbone('prefetch-tiny', tiny_model, model=custom, feature_dim=4,
                         preprocessing=dict(resolution=16, mean=[0]*3, std=[1]*3))
    observed = []
    native = _DiverseBatches.__iter__
    def trace(self):
        draws = []
        observed.append(draws)
        for batch in native(self):
            draws.extend(batch)
            yield batch
    monkeypatch.setattr(_DiverseBatches, '__iter__', trace)
    data = identity_data.with_sampling(samples_per_epoch=8)
    result = id.train(data, backbone=custom, augmentations=[], config=TrainingConfig(
        epochs=2, batch_size=4, workers=1, augmentation_samples=0, output_dir=tmp_path/'prefetch',
        backend_options=dict(embedding_dim=4, freeze_backbone=True, pretrained=False,
            trainer=dict(enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0))))
    assert result.metadata['identity']['loss']=='subcenter_arcface'
    assert {draw[2] for draws in observed for draw in draws}=={0,1}
    train = index_of(data).where(split='train')
    expected = _DiverseBatches(train, 4, result.config.seed)
    # Lightning may prime then recreate an iterator. Every prefetched sequence
    # must still use its own correct epoch; unused priming draws do not train.
    for draws in observed:
        assert len(draws)==8
        expected.epoch = draws[0][2]
        assert draws==[draw for batch in native(expected) for draw in batch]
    import torch
    assert torch.load(result.resumable_checkpoint, weights_only=False)['global_step']==4


def test_identity_stats_without_pixels_and_report(identity_data, tmp_path, monkeypatch):
    monkeypatch.setattr(ImageRef, 'read', lambda *a: pytest.fail('Stats/report must not read image pixels'))
    individual = id.stats(identity_data)
    assert individual.samples.tolist()==[15,15]
    assert id.stats(identity_data, by=('identity_id','clip_id')).samples.tolist()==[5]*6
    clips = id.stats(identity_data, by='clip_id')
    assert clips.tracklets.tolist()==[3]*3
    total = id.stats(identity_data, by=()).iloc[0]
    assert total.samples==30 and total.individuals==2 and total.tracklets==9
    assert identity_data.report(destination=tmp_path/'stats.png', show=False).is_file()


def test_wolf_default_augmentations_are_serializable():
    import albumentations as A
    from dataset_fixer.id.model import augmentation_pipeline
    pipeline, serialized = augmentation_pipeline(None, seed=7)
    restored, again = augmentation_pipeline(serialized, seed=7)
    assert again==serialized
    affine = next(t for t in pipeline.transforms if isinstance(t, A.Affine))
    assert isinstance(affine, A.Affine) and affine.keep_ratio
    assert affine.rotate==(-5.,5.) and affine.scale['x']==(.8,1.6)
    assert sum(isinstance(t,A.Erasing) for t in pipeline.transforms)==2
    assert all(t.fill == 'random' for t in pipeline.transforms if isinstance(t, A.Erasing))
    image = np.full((128,128,3), 130, dtype=np.uint8)
    np.testing.assert_array_equal(pipeline(image=image)['image'], restored(image=image)['image'])


def test_per_image_augmentation_decisions_are_independent_and_reproducible():
    from collections import Counter
    import random
    import albumentations as A
    from dataset_fixer.id.model import augmentation_pipeline, seed_augmentations
    # Previously both flips always happened together because each got the same seed.
    pipeline, _ = augmentation_pipeline([A.HorizontalFlip(p=.5), A.VerticalFlip(p=.5)])
    pipeline.save_applied_params = True
    image = np.random.default_rng(1).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    before = random.getstate(), np.random.get_state()
    outcomes = Counter()
    for seed in range(1000):
        seed_augmentations(pipeline, seed)
        result = pipeline(image=image)
        outcomes[tuple(name for name, _ in result['applied_transforms'])] += 1
        seed_augmentations(pipeline, seed)
        np.testing.assert_array_equal(result['image'], pipeline(image=image)['image'])
    assert len(outcomes) == 4 and all(180 < count < 320 for count in outcomes.values())
    assert before[0] == random.getstate()
    np.testing.assert_array_equal(before[1][1], np.random.get_state()[1])


def test_validation_sample_is_fixed_balanced_and_does_not_change_dataset(identity_data):
    from types import SimpleNamespace
    from dataset_fixer.id.training import _IdentityDataModule
    module = SimpleNamespace(identity=dict(identities=['wolf-a', 'wolf-b'], validation_samples=6,
        preprocessing=dict(resolution=16, mean=[0]*3, std=[1]*3)))
    config = TrainingConfig(batch_size=4, workers=0)
    first = _IdentityDataModule(identity_data, module, config, [])
    second = _IdentityDataModule(identity_data, module, config, [])
    pd.testing.assert_frame_equal(first.val_data.index.frame, second.val_data.index.frame)
    assert first.val_data.index.frame.identity_id.value_counts().tolist() == [3, 3]
    assert len(index_of(identity_data).where(split='val')) == first.validation_total == 10
    module.identity['validation_samples'] = None
    assert len(_IdentityDataModule(identity_data, module, config, []).val_data) == 10
    for invalid in (0, -1, 1, True, 1.5):
        module.identity['validation_samples'] = invalid
        with pytest.raises(ValueError, match='validation_samples'):
            _IdentityDataModule(identity_data, module, config, [])


def test_augmentation_preview_matches_training_pixels_and_reports_effects(identity_data, monkeypatch):
    import albumentations as A
    from dataset_fixer.id.evaluation import augmentation_preview
    from dataset_fixer.id.training import _Pixels
    from dataset_fixer.visualization import VisualizationOptions
    prep = dict(resolution=16, mean=[0]*3, std=[1]*3)
    pipeline = [A.RandomCropFromBorders(p=1), A.RandomBrightnessContrast(p=1),
                A.Erasing(fill='random', p=1), A.ToGray(p=1)]
    index = index_of(identity_data).where(split='train')
    pixels = _Pixels(index, prep, {'wolf-a': 0, 'wolf-b': 1}, pipeline, training=True, seed=7)
    seen = []
    def capture(records, *, prepare, **kwargs):
        for position in records:
            item = prepare(position)
            expected = np.clip(pixels[position][0].numpy().transpose(1, 2, 0)*255, 0, 255).astype(np.uint8)
            np.testing.assert_array_equal(item.panels[1].image, expected)
            caption = item.panels[1].footer
            assert all(name in caption for name in ('crop', 'bright', 'contrast', 'erase', 'gray'))
            seen.append(item)
    monkeypatch.setattr('dataset_fixer.visualization.visualize_records', capture)
    monkeypatch.setattr('dataset_fixer.visualization.finish_visualization', lambda chart, options: chart)
    augmentation_preview(identity_data, pipeline, TrainingConfig(resolution=16, backend_options={'preprocessing': prep}),
                         'train', VisualizationOptions(samples=6, seed=7, show=False))
    assert len(seen) == 6


def test_monitoring_keeps_rng_and_cohort_fixed(identity_data, tmp_path):
    import random
    import torch
    from types import SimpleNamespace
    from dataset_fixer.id.training import _IdentityModule
    from dataset_fixer.id.monitoring import prepare_monitor, monitor_epoch
    id.register_backbone('monitor-tiny', tiny_model, feature_dim=4,
                         preprocessing=dict(resolution=16,mean=[0]*3,std=[1]*3))
    module = _IdentityModule(dict(framework='id',backbone='monitor-tiny',identities=['wolf-a','wolf-b'],embedding_dim=4),pretrained=False)
    module.train()
    result = SimpleNamespace(dataset=identity_data, config=TrainingConfig(batch_size=4,
        backend_options=dict(embedding_samples=12)), output_dir=tmp_path, auxiliary_files={}, wandb_run=None)
    prepare_monitor(result)
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    metrics = monitor_epoch(result, module, 0)
    assert module.training and metrics['embeddings/samples']==12
    assert 'val/monitor_rank1' in metrics
    assert random.getstate()==before[0]
    np.testing.assert_array_equal(np.random.get_state()[1],before[1][1])
    assert torch.equal(torch.get_rng_state(),before[2])
    original = pd.read_csv(tmp_path/'monitoring/embeddings.csv')
    monitor_epoch(result,module,1)
    pd.testing.assert_frame_equal(original,pd.read_csv(tmp_path/'monitoring/embeddings.csv'))


def test_attention_uses_captum_and_preserves_model(identity_data, tmp_path, monkeypatch):
    import torch
    from captum.attr import Occlusion
    from dataset_fixer import Model
    from dataset_fixer.id.training import _IdentityModule
    id.register_backbone('attention-tiny',tiny_model,feature_dim=4,
                         preprocessing=dict(resolution=16,mean=[.5]*3,std=[.5]*3))
    module = _IdentityModule(dict(framework='id',backbone='attention-tiny',identities=['wolf-a','wolf-b'],embedding_dim=4),pretrained=False)
    path=tmp_path/'attention.ckpt'
    torch.save({'identity':module.identity,'state_dict':module.state_dict()},path)
    model=Model(path)
    gallery=model.enroll(identity_data,split='train',samples_per_identity=3)
    calls=[]
    native=Occlusion.attribute
    def observe(self,*args,**kwargs):
        calls.append(kwargs)
        return native(self,*args,**kwargs)
    monkeypatch.setattr(Occlusion,'attribute',observe)
    before = model.embed(identity_data.sample(1),prediction_cache=False)[0].embedding.copy()
    id.visualize_attention(model,identity_data.sample(2,split='test'),gallery=gallery,
        patch_fraction=.5,samples=2,destination=tmp_path/'attention.png',show=False)
    assert len(calls)==2 and calls[0]['sliding_window_shapes']==(3,8,8)
    np.testing.assert_array_equal(before,model.embed(identity_data.sample(1),prediction_cache=False)[0].embedding)
    with np.load(tmp_path/'attention.npz') as maps:
        assert maps['attribution_0'].shape==(16,16) and np.isfinite(maps['attribution_0']).all()
    assert json.loads((tmp_path/'attention.json').read_text())[0]['contrast'] is True


def test_black_border_option_for_png_npz_manifest_and_custom_augmentation(tmp_path):
    import albumentations as A
    from dataset_fixer.id.model import augmentation_pipeline, input_index
    pixels = np.zeros((20,24,3), dtype=np.uint8)
    pixels[4:16,5:19] = (90,100,110)
    pixels[6:8,7:9] = 0  # interior black regions do not change the outer crop
    folder = tmp_path/'wolf-a'/'clip'
    folder.mkdir(parents=True)
    image = folder/'crop.png'
    Image.fromarray(pixels).save(image)
    trimmed = Dataset.open(tmp_path, format='identity_folders', layout='{identity_id}/{clip_id}/{image}')
    raw = Dataset.open(tmp_path, format='identity_folders', layout='{identity_id}/{clip_id}/{image}', trim_black_borders=False)
    assert trimmed._samples[0].image_ref.read().shape == (12,14,3)
    np.testing.assert_array_equal(raw._samples[0].image_ref.read(), pixels)
    ref = trimmed._samples[0].image_ref
    assert ImageRef(*json.loads(ref.key)).trim_black_borders
    assert input_index([ref])[0].image_ref.trim_black_borders
    transform, _ = augmentation_pipeline([A.CenterCrop(8,8)])
    assert transform(image=ref.read())['image'].shape == (8,8,3)
    # A single nonblack border pixel counts as real background and is preserved.
    pixels[0,0,1] = 1
    Image.fromarray(pixels).save(image)
    assert ref.read().shape == (16,19,3)
    archive=tmp_path/'tracklets.npz'
    np.savez(archive,images=np.stack([pixels,np.zeros_like(pixels)]),ids=[3,3],frames=[0,1])
    npz=Dataset.open(archive,format='trex_npz')
    assert npz._samples[0].image_ref.read().shape==(16,19,3)
    assert npz._samples[1].image_ref.read().shape==(20,24,3)  # entirely black stays valid
    manifest=tmp_path/'references.jsonl'
    manifest.write_text(json.dumps(dict(path=str(image),identity_id='wolf-a',trim_black_borders=False))+'\n')
    stored=Dataset.open(manifest,format='identity_manifest')
    assert not stored._samples[0].image_ref.trim_black_borders
    override=Dataset.open(manifest,format='identity_manifest',trim_black_borders=True)
    assert override._samples[0].image_ref.trim_black_borders
