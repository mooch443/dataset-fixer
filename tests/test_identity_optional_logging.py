"""Local evaluation and checkpoint results must outlive optional telemetry."""
from types import SimpleNamespace

import pandas as pd
import pytest

from dataset_fixer import id, TrainingConfig, TrainingSession, WandbConfig
from dataset_fixer.training.config import ModelTypes
from dataset_fixer.training.session import TrainingResult
from test_identity import identity_data


@pytest.mark.parametrize('state', ['absent', 'finished', 'broken', 'active'])
def test_evaluation_retains_reports_with_optional_logger(identity_data, tmp_path, monkeypatch, state):
    calls = []
    def write(*args, **kwargs):
        calls.append(args)
        if state in ('finished', 'broken'):
            raise RuntimeError('logger unavailable')
    class Summary(dict):
        def update(self, values):
            write(values)
            super().update(values)
    run = None if state == 'absent' else SimpleNamespace(
        _is_finished=state == 'finished', define_metric=write, log=write, summary=Summary())
    result = TrainingResult(identity_data, SimpleNamespace(family=ModelTypes.ID), TrainingConfig(),
                            tmp_path, TrainingSession(), wandb_run=run, best_weights=tmp_path/'best.ckpt',
                            _model=object(), finished=True)
    result._identity_protocol = id.Protocol(identity_data, gallery_samples=2)
    monkeypatch.setattr(result, '_enrollment', lambda **kwargs: object())
    def evaluate(model, data, *, destination, _on_progress, **kwargs):
        if _on_progress:
            _on_progress('embeddings', 2, 2)
        destination.mkdir(parents=True)
        ranking = pd.DataFrame([dict(rank1=.9)])
        ranking.to_csv(destination/'metrics.csv', index=False)
        return SimpleNamespace(location=destination, ranking=ranking)
    monkeypatch.setattr('dataset_fixer.id.evaluation.evaluate', evaluate)
    report = result.evaluate(split='test', plots=0)
    assert report.ranking.rank1.tolist() == [.9]
    assert result.metrics['evaluation/test'] == [dict(rank1=.9)]
    assert result.auxiliary_files['evaluation/test/metrics.csv'].is_file()
    assert result.wandb_run is run
    if state in ('absent', 'finished'):
        assert not calls
    if state == 'active':
        assert run.summary['evaluation/test'] == [dict(rank1=.9)]


def test_failed_logger_initialization_still_registers_local_training(identity_data, tmp_path, monkeypatch):
    import wandb
    def unavailable(**kwargs):
        raise RuntimeError('no credentials')
    monkeypatch.setattr(wandb, 'run', None)
    monkeypatch.setattr(wandb, 'init', unavailable)
    session = TrainingSession(wandb=WandbConfig(project='test'), disconnect='after_safe')
    result = session._register(identity_data, SimpleNamespace(family=ModelTypes.ID, name='resnet18', task='id'),
                               TrainingConfig(output_dir=tmp_path/'local'))
    assert result.wandb_run is None
    assert session.results == [result]
    assert result.output_dir.is_dir()
    assert not session.disconnect
