from __future__ import annotations

import io
import json
import shutil
import sys
import warnings
import zipfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import pytest
import torch
import yaml

import dataset_fixer as df
from dataset_fixer.training.selection import select
from dataset_fixer.training.backends import prepare_data, rfdetr_configs, rfdetr_data_module
from dataset_fixer.training.session import verified_copy
from conftest import make_yolo_dataset


@pytest.fixture(autouse=True)
def training_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("dataset_fixer.convert.cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr("dataset_fixer.training.session.cache_root", lambda: tmp_path / "cache")


@pytest.fixture
def pose(tmp_path):
    keypoints = " ".join(f"{0.25 + i * .06} 0.5 2" for i in range(8))
    root = make_yolo_dataset(tmp_path / "pose", task="pose", names=["wolf"],
        train_rows=[f"0 0.5 0.5 0.8 0.8 {keypoints}"], val_rows=[f"0 0.5 0.5 0.8 0.8 {keypoints}"],
        extra={"kpt_shape": [8, 3], "flip_idx": [0, 3, 2, 1, 4, 5, 6, 7]})
    from PIL import Image, ImageDraw
    for path in root.rglob("*.jpg"):
        with Image.open(path) as image:
            drawing = ImageDraw.Draw(image)
            drawing.ellipse((20, 35, 140, 85), fill=(170, 150, 80))
            drawing.line([(40, 60), (112, 60)], fill="white", width=2)
            image.save(path)
    return df.Dataset.open(root, progress=False)


def test_yolo_inference_uses_installed_catalogue(pose, monkeypatch):
    from ultralytics.utils import downloads
    monkeypatch.setattr(downloads, "GITHUB_ASSETS_NAMES", {"yolo26s-pose.pt", "yolo99z-pose.pt", "yolo26s.pt"})
    assert select(pose, type=df.ModelTypes.YOLO, version=26, s="s").name == "yolo26s-pose.pt"
    assert select(pose, type=df.ModelTypes.YOLO, version=99, s="z").name == "yolo99z-pose.pt"
    with pytest.raises(ValueError, match="no pretrained"):
        select(pose, type=df.ModelTypes.YOLO, version=88)
    with pytest.raises(ValueError, match="incompatible"):
        select(pose, model_type="detect")
    with pytest.raises(ValueError, match="does not support"):
        select(pose, type=df.ModelTypes.NNUNET)


def test_polygon_and_mask_tasks_remain_distinct(tmp_path):
    root = make_yolo_dataset(tmp_path / "polygons", task="segment", names=["island"],
        train_rows=["0 .2 .2 .8 .2 .8 .8 .2 .8"], val_rows=["0 .2 .2 .8 .2 .8 .8 .2 .8"])
    polygons = df.Dataset.open(root, progress=False)
    assert select(polygons, version=26, s="n").name == "yolo26n-seg.pt"
    assert select(polygons, type=df.ModelTypes.NNUNET).task == "semantic_segment"
    masks = polygons.export(destination=tmp_path / "masks", format="semantic_masks", visualize=False, progress=False)
    assert select(masks, version=26, s="n").name == "yolo26n-sem.pt"
    with pytest.raises(ValueError, match="does not support"):
        select(masks, type=df.ModelTypes.RFDETR)
    with pytest.raises(ValueError, match="incompatible"):
        select(masks, model_type="segment")


class Artifact:
    def __init__(self, name, type, metadata):
        self.name, self.type, self.metadata = name, type, metadata
        self.files, self.waited, self.fail = {}, False, False
        self.qualified_name = f"team/project/{name}:v3"
        self.url = "https://wandb.ai/team/project/artifacts/model/test/v3"
        self.digest = "digest"
        self.state, self.aliases, self.deleted, self.owner = "PENDING", [], False, None

    def add_file(self, path, name):
        self.files[name] = Path(path).read_bytes()

    def wait(self, timeout):
        if self.fail:
            raise TimeoutError("upload timeout")
        if self.owner:
            for artifact in self.owner.artifacts:
                if (artifact is not self and "latest" in artifact.aliases
                    and artifact.qualified_name.rsplit(":", 1)[0] == self.qualified_name.rsplit(":", 1)[0]):
                    artifact.aliases.remove("latest")
            self.owner.events.append(("confirmed", self.qualified_name))
        self.state, self.aliases = "COMMITTED", ["latest"]
        self.waited = True
        return self

    def delete(self, delete_aliases=False):
        assert not delete_aliases
        if self.owner.fail_delete or self.aliases:
            raise PermissionError("artifact deletion refused")
        self.owner.events.append(("deleted", self.qualified_name))
        self.deleted = True


class Config(dict):
    def update(self, values, **_):
        super().update(values)


class Run:
    id, project, entity = "test", "project", "team"
    def __init__(self):
        self.summary, self.config, self.tags, self.artifacts = {}, Config(), (), []
        self.settings = SimpleNamespace(mode="online")
        self.fail, self.closed = False, False
        self.fail_delete, self.events = False, []
    def log_artifact(self, artifact, aliases):
        artifact.fail = self.fail
        artifact.owner = self
        artifact.qualified_name = f"{self.entity}/{self.project}/{artifact.name}:v{len(self.artifacts) + 3}"
        self.artifacts.append(artifact)
        return artifact
    def logged_artifacts(self):
        return (artifact for artifact in self.artifacts if not artifact.deleted)
    def finish(self, **kwargs):
        self.closed = True
    def log(self, *args, **kwargs):
        pass


@pytest.fixture
def run(monkeypatch):
    value = Run()
    import wandb
    monkeypatch.setattr(wandb, "Artifact", Artifact)
    def get_artifact(reference, type):
        return next(artifact for artifact in value.artifacts
                    if artifact.qualified_name == reference and artifact.type == type and not artifact.deleted)
    monkeypatch.setattr(wandb, "Api", lambda **_: SimpleNamespace(run=lambda _: value, artifact=get_artifact))
    return value


def job(pose, tmp_path, run=None, **session_options):
    session = df.TrainingSession(**session_options)
    config = df.TrainingConfig(output_dir=tmp_path / "output", resolution=96, epochs=2, workers=0)
    selection = select(pose, type=df.ModelTypes.RFDETR, config=config)
    result = session._register(pose, selection, config, wandb=df.WandbConfig(run=run) if run else False)
    return session, result


def checkpoints(tmp_path, epoch=0):
    from rfdetr.config import RFDETRKeypointPreviewConfig
    mc = RFDETRKeypointPreviewConfig(resolution=96, model_name="RFDETRKeypointPreview", num_classes=1, num_keypoints_per_class=[8])
    best, latest = tmp_path / "best.pth", tmp_path / "last.ckpt"
    payload = {"model": {"weight": torch.ones(1)}, "epoch": epoch, "model_name": "RFDETRKeypointPreview", "model_config": mc.model_dump(), "args": {"class_names": ["wolf"]}}
    torch.save(payload, best)
    torch.save({**payload, "optimizer_states": [{"state": {"lr": .01}}]}, latest)
    return df.Checkpoints(best, latest, epoch, "val/keypoint_map_50_95", .7)


def test_confirmed_upload_backup_bundle_and_warm_start(pose, tmp_path, run):
    session, result = job(pose, tmp_path, run, checkpointing=df.CheckpointConfig(backup_dir=tmp_path / "drive"))
    result._capture(None, checkpoints(tmp_path))
    assert result.uploaded_digest == result.wandb_bundle.sha256
    assert result.copied_digest == result.bundle.sha256 != result.uploaded_digest
    assert run.artifacts[-1].waited
    assert run.summary["best_model_artifact"].endswith(":v3")
    with zipfile.ZipFile(result.wandb_bundle.path) as archive:
        assert "weights/last.ckpt" not in archive.namelist()
        saved = torch.load(io.BytesIO(archive.read("weights/best.pth")), weights_only=False)
        assert torch.equal(saved["model"]["weight"], torch.ones(1))
    assert result.wandb_bundle.size < result.bundle.size
    assert select(pose, weights=result.wandb_bundle.path, config=df.TrainingConfig(resolution=192)).weights
    with pytest.raises(ValueError, match="optimizer state"):
        select(pose, resume=result.wandb_bundle.path)
    backup = tmp_path / "drive" / result.output_dir.name / "model.zip"
    assert backup.read_bytes() == result.bundle.path.read_bytes()
    loaded = df.Model.load_many(result.bundle.path)[0]
    assert loaded.kind == "rfdetr" and loaded.task == "pose"
    selected = select(pose, weights=result.bundle.path, config=df.TrainingConfig(resolution=192))
    assert selected.weights and selected.resume is None
    assert selected.provenance["sha256"]
    resumed = select(pose, resume=result.bundle.path, config=df.TrainingConfig(resolution=96))
    assert resumed.resume.name == "last.ckpt"
    from dataset_fixer.model_sources import _validate_checkpoint
    resumed.resume.write_bytes(b"corrupt latest checkpoint")
    with pytest.raises(df.DatasetValidationError, match="SHA-256"):
        _validate_checkpoint(resumed.resume, resumed.metadata, progress=False)
    with pytest.raises(ValueError, match="optimizer"):
        select(pose, resume=result.best_weights)


def test_timeout_keeps_backup_and_runtime_then_retry(pose, tmp_path, run, monkeypatch):
    session, result = job(pose, tmp_path, run, checkpointing=df.CheckpointConfig(backup_dir=tmp_path / "drive", final_attempts=1), disconnect="after_safe", disconnect_delay=0)
    calls = []
    monkeypatch.setattr("dataset_fixer.training.session.in_colab", lambda: True)
    monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace(runtime=SimpleNamespace(unassign=lambda: calls.append(True))))
    run.fail = True
    with pytest.warns(RuntimeWarning):
        result._capture(None, checkpoints(tmp_path))
    assert result.copied_digest and result.uploaded_digest is None
    with pytest.warns(RuntimeWarning), pytest.raises(RuntimeError, match="Publication incomplete"):
        session.finish()
    assert calls == [] and not run.closed
    run.fail = False
    assert session.finish()
    assert calls == [True] and run.closed


@pytest.mark.parametrize("keep,remaining", [(1, 1), (2, 2), (None, 3)])
def test_wandb_retention_keeps_best_and_resume_after_confirmation(pose, tmp_path, run, keep, remaining):
    _, result = job(pose, tmp_path, run, checkpointing=df.CheckpointConfig(keep_wandb_versions=keep, wandb_contents="full"))
    result._capture(None, checkpoints(tmp_path))
    best = result.best_weights
    for epoch in (1, 2):
        result._capture(None, replace(checkpoints(tmp_path, epoch), best=best))
    assert len(list(run.logged_artifacts())) == remaining
    assert not result.retention_pending
    for index, (event, _) in enumerate(run.events):
        if event == "deleted":
            assert sum(name == "confirmed" for name, _ in run.events[:index]) >= 2
    with zipfile.ZipFile(io.BytesIO(next(iter(run.artifacts[-1].files.values())))) as archive:
        best = torch.load(io.BytesIO(archive.read("weights/best.pth")), weights_only=False)
        latest = torch.load(io.BytesIO(archive.read("weights/last.ckpt")), weights_only=False)
        assert best["epoch"] == 0 and latest["epoch"] == 2
        assert latest["optimizer_states"]


def test_wandb_retention_waits_for_upload_and_verified_backup(pose, tmp_path, run, monkeypatch):
    session, result = job(pose, tmp_path, run, checkpointing=df.CheckpointConfig(backup_dir=tmp_path / "drive"))
    result._capture(None, checkpoints(tmp_path))
    first = run.artifacts[-1]
    run.fail = True
    with pytest.warns(RuntimeWarning, match="upload failed"):
        result._capture(None, checkpoints(tmp_path, 1))
    assert not first.deleted
    run.fail = False
    def fail_backup(source, destination):
        if tmp_path / "drive" in destination.parents:
            raise OSError("Drive unavailable")
        verified_copy(source, destination)
    monkeypatch.setattr("dataset_fixer.training.session.verified_copy", fail_backup)
    with pytest.warns(UserWarning, match="backup failed"):
        result._capture(None, checkpoints(tmp_path, 2))
    assert not first.deleted and result.uploaded_digest == result.wandb_bundle.sha256
    monkeypatch.setattr("dataset_fixer.training.session.verified_copy", verified_copy)
    assert session.finish()
    assert first.deleted and not result.retention_pending


def test_wandb_cleanup_retry_preserves_confirmed_upload_and_disconnect(pose, tmp_path, run, monkeypatch):
    session, result = job(pose, tmp_path, run, checkpointing=df.CheckpointConfig(final_attempts=1),
                          disconnect="after_safe", disconnect_delay=0)
    calls = []
    monkeypatch.setattr("dataset_fixer.training.session.in_colab", lambda: True)
    monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace(runtime=SimpleNamespace(unassign=lambda: calls.append(True))))
    result._capture(None, checkpoints(tmp_path))
    first = run.artifacts[-1]
    run.fail_delete = True
    with pytest.warns(RuntimeWarning, match="cleanup failed"):
        result._capture(None, checkpoints(tmp_path, 1))
    assert result.uploaded_digest == result.wandb_bundle.sha256 and result.retention_pending and not first.deleted
    with pytest.warns(RuntimeWarning, match="cleanup failed"):
        assert session.finish()
    assert calls == [True] and run.closed and result.finished
    run.fail_delete = False
    assert session.finish()
    assert first.deleted and not result.retention_pending and len(run.artifacts) == 2 and calls == [True]
    assert result.wandb_bundle.uploaded


def test_wandb_retention_is_scoped_numeric_and_checksum_verified(run):
    from dataset_fixer.wandb import _prune_checkpoint_artifacts
    def add(reference, *, type="model", metadata=None, state="COMMITTED"):
        artifact = Artifact("unused", type, metadata or {"sha256": "verified", "bundle_file": "model.zip"})
        artifact.qualified_name, artifact.state, artifact.owner = reference, state, run
        run.artifacts.append(artifact)
        return artifact
    old = [add(f"team/project/model-test:v{version}") for version in (2, 9, 3, 10)]
    untouched = [add("team/project/model-test:v11"), add("team/project/model-other:v1"),
                 add("elsewhere/project/model-test:v1"), add("team/project/model-test:v1", type="dataset"),
                 add("team/project/model-test:v4", metadata={"bundle_file": "custom.zip"}),
                 add("team/project/model-test:v5", state="PENDING")]
    run.summary["checkpoint_artifact"] = old[-1].qualified_name
    with pytest.raises(RuntimeError, match="checksum"):
        _prune_checkpoint_artifacts(run, sha256="incorrect", keep_versions=2, timeout=10)
    assert not any(a.deleted for a in run.artifacts)
    _prune_checkpoint_artifacts(run, sha256="verified", keep_versions=2, timeout=10)
    assert [a.deleted for a in old] == [True, False, True, False]
    assert not any(a.deleted for a in untouched)
    run.summary["checkpoint_artifact"] = "team/project/model-other:v1"
    with pytest.raises(ValueError, match="belong"):
        _prune_checkpoint_artifacts(run, sha256="verified", keep_versions=1, timeout=10)


def test_wandb_retention_preserves_manual_aliases(pose, tmp_path, run):
    session, result = job(pose, tmp_path, run)
    result._capture(None, checkpoints(tmp_path))
    first = run.artifacts[-1]
    first.aliases.append("pinned")
    with pytest.warns(RuntimeWarning, match="cleanup failed"):
        result._capture(None, checkpoints(tmp_path, 1))
    assert not first.deleted and first.aliases == ["pinned"]
    first.aliases.clear()
    assert session.finish() and first.deleted
    assert len(run.artifacts) == 2


@pytest.mark.parametrize("keep", [0, -1, True, 1.5, "1"])
def test_checkpoint_retention_rejects_invalid_counts(keep):
    with pytest.raises(ValueError, match="keep_wandb_versions"):
        df.CheckpointConfig(keep_wandb_versions=keep)


@pytest.mark.parametrize("contents", ["latest", "none", False, None])
def test_checkpoint_rejects_invalid_wandb_contents(contents):
    with pytest.raises(ValueError, match="wandb_contents"):
        df.CheckpointConfig(wandb_contents=contents)


@pytest.mark.parametrize("mode", ["epoch", "all", False, None])
def test_checkpoint_rejects_invalid_wandb_upload_mode(mode):
    with pytest.raises(ValueError, match="wandb_upload"):
        df.CheckpointConfig(wandb_upload=mode)


@pytest.mark.parametrize("contents", ["best", "full"])
def test_best_upload_skips_plateaus_keeps_backups_and_final_reports(pose, tmp_path, run, contents):
    session, result = job(pose, tmp_path, run, checkpointing=df.CheckpointConfig(
        backup_dir=tmp_path / "drive", wandb_upload="best", wandb_contents=contents))
    result._capture(None, checkpoints(tmp_path))
    selected = result.best_weights
    first_upload, first_backup = result.uploaded_digest, result.copied_digest
    result._capture(None, replace(checkpoints(tmp_path, 1), best=selected), {"train/loss": 0.4})
    assert len(run.artifacts) == 1 and result.uploaded_digest == first_upload
    assert result.copied_digest != first_backup and result.copied_digest == result.bundle.sha256
    backup = tmp_path / "drive" / result.output_dir.name / "model.zip"
    with zipfile.ZipFile(backup) as archive:
        best = torch.load(io.BytesIO(archive.read("weights/best.pth")), weights_only=False)
        latest = torch.load(io.BytesIO(archive.read("weights/last.ckpt")), weights_only=False)
        assert best["epoch"] == 0 and latest["epoch"] == 1
    improved = replace(checkpoints(tmp_path, 2), value=0.8)
    result._capture(None, improved)
    assert len(run.artifacts) == 2 and run.artifacts[0].deleted
    renamed = tmp_path / "final-best.pth"
    torch.save(torch.load(result.best_weights, weights_only=False), renamed)
    result._capture(None, replace(checkpoints(tmp_path, 3), best=renamed, value=0.8), final=True)
    assert len(run.artifacts) == 2  # Renaming/stripping does not improve the metric.
    report = tmp_path / "evaluation.json"
    report.write_text('{"map": 0.8}')
    result.auxiliary_files["evaluation/metrics.json"] = report
    assert session.finish()
    assert len(run.artifacts) == 3 and run.artifacts[1].deleted
    with zipfile.ZipFile(result.wandb_bundle.path) as archive:
        assert archive.read("evaluation/metrics.json") == report.read_bytes()
    assert session.finish() and len(run.artifacts) == 3


def test_best_upload_is_immediate_and_retries_between_intervals(pose, tmp_path, run):
    _, result = job(pose, tmp_path, run, checkpointing=df.CheckpointConfig(
        every_n_epochs=5, wandb_upload="best", backup_dir=tmp_path / "drive"))
    run.fail = True
    with pytest.warns(RuntimeWarning, match="upload failed"):
        result._capture(None, checkpoints(tmp_path))
    selected = result.best_weights
    assert len(run.artifacts) == 1 and result.uploaded_digest is None
    run.fail = False
    result._capture(None, replace(checkpoints(tmp_path, 1), best=selected))
    assert len(run.artifacts) == 2 and result.wandb_bundle.uploaded
    result._capture(None, replace(checkpoints(tmp_path, 2), best=selected))
    assert len(run.artifacts) == 2
    result._capture(None, replace(checkpoints(tmp_path, 3), value=0.8))
    assert len(run.artifacts) == 3


def test_best_upload_without_selected_best_waits_until_final_recovery(pose, tmp_path, run):
    session, result = job(pose, tmp_path, run, checkpointing=df.CheckpointConfig(
        wandb_upload="best", backup_dir=tmp_path / "drive"))
    result._capture(None, replace(checkpoints(tmp_path), best=None))
    assert not run.artifacts and result.copied_digest == result.bundle.sha256
    result._record_error(RuntimeError("failed before best selection"))
    assert session.finish() and len(run.artifacts) == 1
    with zipfile.ZipFile(result.wandb_bundle.path) as archive:
        assert "failure.txt" in archive.namelist() and "weights/last.ckpt" in archive.namelist()


def test_best_upload_custom_provider_without_metric_uses_snapshot(pose, tmp_path, run):
    _, result = job(pose, tmp_path, run, checkpointing=df.CheckpointConfig(wandb_upload="best"))
    result._capture(None, replace(checkpoints(tmp_path), value=None))
    selected = result.best_weights
    result._capture(None, replace(checkpoints(tmp_path, 1), best=selected, value=None))
    assert len(run.artifacts) == 1
    result._capture(None, replace(checkpoints(tmp_path, 2), value=None))
    assert len(run.artifacts) == 2 and result.best_epoch == 2


@pytest.mark.parametrize("family,model_key,state_keys", [
    (df.ModelTypes.YOLO, "model", ("optimizer", "scaler")),
    (df.ModelTypes.RFDETR, "model", ("optimizer_states", "lr_schedulers")),
    (df.ModelTypes.NNUNET, "network_weights", ("optimizer_state", "grad_scaler_state")),
])
def test_weights_checkpoint_preserves_parameters_and_original_state(tmp_path, family, model_key, state_keys):
    from dataset_fixer.training.backends import weights_checkpoint
    source, destination = tmp_path / "original.pt", tmp_path / "upload" / "best.pt"
    weight = torch.tensor([0.123456789, 1.23456789], dtype=torch.float64)
    saved = {model_key: {"weight": weight}, "epoch": 17, "model_config": {"resolution": 1536},
             **{key: {"state": torch.ones(1024)} for key in state_keys}}
    torch.save(saved, source)
    original = source.read_bytes()
    assert weights_checkpoint(source, family, destination) == destination
    stripped = torch.load(destination, weights_only=False)
    assert all(key not in stripped for key in state_keys)
    assert stripped["epoch"] == 17 and stripped["model_config"] == saved["model_config"]
    assert stripped[model_key]["weight"].dtype == weight.dtype
    assert torch.equal(stripped[model_key]["weight"], weight)
    assert source.read_bytes() == original and destination.stat().st_size < source.stat().st_size
    assert weights_checkpoint(destination, family, tmp_path / "unnecessary.pt") == destination


def test_weights_bundle_failure_preserves_previous_upload_and_retry(pose, tmp_path, run, monkeypatch):
    from dataset_fixer.training.backends import weights_checkpoint
    session, result = job(pose, tmp_path, run, checkpointing=df.CheckpointConfig(backup_dir=tmp_path / "drive", final_attempts=1))
    result._capture(None, checkpoints(tmp_path))
    first = run.artifacts[-1]
    def fail(*args):
        raise OSError("cannot create upload weights")
    monkeypatch.setattr("dataset_fixer.training.backends.weights_checkpoint", fail)
    with pytest.warns(UserWarning, match="upload failed"):
        result._capture(None, checkpoints(tmp_path, 1))
    assert result.copied_digest == result.bundle.sha256
    assert result.wandb_bundle is None and not first.deleted
    with pytest.warns(UserWarning), pytest.raises(RuntimeError, match="Publication incomplete"):
        session.finish()
    monkeypatch.setattr("dataset_fixer.training.backends.weights_checkpoint", weights_checkpoint)
    assert session.finish() and first.deleted


def test_before_best_weights_available_uploads_recovery_checkpoint(pose, tmp_path, run):
    _, result = job(pose, tmp_path, run)
    result._capture(None, replace(checkpoints(tmp_path), best=None))
    assert result.wandb_bundle.sha256 == result.bundle.sha256 == result.uploaded_digest
    with zipfile.ZipFile(result.wandb_bundle.path) as archive:
        saved = torch.load(io.BytesIO(archive.read("weights/last.ckpt")), weights_only=False)
        assert saved["optimizer_states"]
    assert "best_model_artifact" not in run.summary


def test_verified_copy_retains_previous_on_corruption(tmp_path, monkeypatch):
    source, destination = tmp_path / "new", tmp_path / "old"
    source.write_bytes(b"new weights")
    destination.write_bytes(b"old weights")
    monkeypatch.setattr(shutil, "copyfile", lambda _, target: Path(target).write_bytes(b"corrupt"))
    with pytest.raises(OSError, match="verification"):
        verified_copy(source, destination)
    assert destination.read_bytes() == b"old weights"


@pytest.mark.parametrize("failure", [ValueError, KeyboardInterrupt])
def test_session_finalizes_on_error_and_preserves_original(pose, tmp_path, run, failure, monkeypatch, capsys):
    session, result = job(pose, tmp_path, run, disconnect="after_safe", disconnect_delay=0)
    events = []
    monkeypatch.setattr("dataset_fixer.training.session.in_colab", lambda: True)

    def close(exit_code):
        assert run.summary["failure_type"] == failure.__name__
        assert "evaluation failed" in capsys.readouterr().err
        events.append(("finished", exit_code))
        run.closed = True

    def disconnect():
        assert events == [("finished", 1)]
        assert "Session failed; checkpoints and failure reports confirmed safe" in capsys.readouterr().out
        events.append(("disconnected", None))

    monkeypatch.setattr(run, "finish", close)
    monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace(runtime=SimpleNamespace(unassign=disconnect)))
    with pytest.raises(failure, match="evaluation failed"):
        with session:
            result._capture(None, checkpoints(tmp_path))
            raise failure("evaluation failed")
    assert result.finished and session.disconnected
    assert events == [("finished", 1), ("disconnected", None)]
    assert run.summary["session_failed"] and run.summary["failure_message"] == "evaluation failed"
    with zipfile.ZipFile(result.bundle.path) as archive:
        assert "failure.txt" in archive.namelist()
        assert "evaluation failed" in archive.read("failure.txt").decode()
    original = result.error
    result._record_error(RuntimeError("later cleanup failure"))
    assert result.error == original
    assert "later cleanup failure" not in capsys.readouterr().err


def test_resolution_validation_and_pe_resize():
    from dataset_fixer.rfdetr_engine import resolution_overrides
    config = {"patch_size": 12, "num_windows": 2, "resolution": 576, "positional_encoding_size": 48}
    assert resolution_overrides(config, 1296) == {"resolution": 1296, "positional_encoding_size": 108}
    with pytest.raises(ValueError, match="divisible"):
        resolution_overrides(config, 1280)
    assert resolution_overrides({**config, "positional_encoding_size": 37}, 1296) == {"resolution": 1296}


def test_real_rfdetr_data_and_augmentation(pose, tmp_path):
    from rfdetr import RFDETRDataModule
    session, result = job(pose, tmp_path)
    prepared = prepare_data(result)
    variant, mc, options = rfdetr_configs(result.selection, result.config, prepared, {})
    tc = variant._train_config_class(dataset_dir=str(prepared.location), dataset_file="yolo", **options)
    assert tc.keypoint_flip_pairs == [1, 3]
    data = RFDETRDataModule(mc, tc)
    data.setup("fit")
    images, targets = next(iter(data.val_dataloader()))
    assert targets[0]["keypoints"].shape == (1, 8, 3)
    assert images.tensors.shape[-1] == 96
    output = tmp_path / "augmentation.png"
    df.preview_augmentations(pose, {"HorizontalFlip": {"p": 1}}, type=df.ModelTypes.RFDETR, config=replace(result.config, resolution=384),
                             samples=1, destination=output, show=False)
    assert output.stat().st_size > 1000
    shutil.copyfile(output, "/tmp/dataset-fixer-native-augmentation.png")


@pytest.mark.parametrize("annotation_ids", [(0, 1), (9, 17), ()])
def test_rfdetr_evaluation_ids_leave_source_data_unchanged(pose, tmp_path, monkeypatch, annotation_ids):
    from rfdetr.datasets import yolo
    original_samples, original_metadata = deepcopy((pose._samples, pose._metadata))
    original_files = {p: p.read_bytes() for p in pose.location.rglob("*") if p.is_file()}
    _, result = job(pose, tmp_path)
    prepared = prepare_data(result)
    prepared_files = {p: p.read_bytes() for p in prepared.location.rglob("*") if p.is_file()}
    variant, mc, options = rfdetr_configs(result.selection, result.config, prepared, {})
    tc = variant._train_config_class(dataset_dir=str(prepared.location), dataset_file="yolo", **options)
    build_coco = yolo._build_coco_api_from_samples
    originals = []
    def capture(*args, **kwargs):
        coco = build_coco(*args, **kwargs)
        annotation = coco.dataset["annotations"][0]
        coco.dataset["annotations"] = [{**annotation, "id": i} for i in annotation_ids]
        coco.createIndex()
        originals.append((coco, deepcopy(coco.dataset)))
        return coco
    monkeypatch.setattr(yolo, "_build_coco_api_from_samples", capture)
    data = rfdetr_data_module(mc, tc)
    for stage in ("fit", "validate", "test", "predict", "fit", "test"):
        data.setup(stage)
    assert len(originals) == 3  # Repeated setup neither reloads nor renumbers.
    for native, (original, before) in zip((data._dataset_train, data._dataset_val, data._dataset_test), originals):
        assert original.dataset == before
        expected_ids = list(range(1, len(annotation_ids) + 1)) if 0 in annotation_ids else list(annotation_ids)
        assert list(native.coco.anns) == expected_ids
        expected = {**before, "annotations": [
            {**ann, "id": i} for ann, i in zip(before["annotations"], expected_ids)
        ]}
        assert native.coco.dataset == expected
        assert all(native.coco.anns[ann["id"]] is ann for ann in native.coco.imgToAnns[0])
        assert (native.coco is original) == (0 not in annotation_ids)
    assert pose._samples == original_samples and pose._metadata == original_metadata
    assert {p: p.read_bytes() for p in pose.location.rglob("*") if p.is_file()} == original_files
    assert {p: p.read_bytes() for p in prepared.location.rglob("*") if p.is_file()} == prepared_files


def test_rfdetr_keypoint_metric_counts_first_annotation(pose, tmp_path):
    from rfdetr.training.callbacks.coco_eval import COCOEvalCallback
    _, result = job(pose, tmp_path)
    prepared = prepare_data(result)
    variant, mc, options = rfdetr_configs(result.selection, result.config, prepared, {})
    tc = variant._train_config_class(dataset_dir=str(prepared.location), dataset_file="yolo", **options)
    data = rfdetr_data_module(mc, tc)
    data.setup("fit")
    data.setup("test")
    callback = COCOEvalCallback(keypoint_oks_sigmas=[0.1] * 8)
    for split in ("train", "val", "val_ema", "test"):
        native = getattr(data, f"_dataset_{split.removesuffix('_ema')}")
        annotation, = native.coco.dataset["annotations"]
        assert annotation["id"] > 0
        assert annotation["category_id"] == annotation["image_id"] == 0
        x, y, w, h = annotation["bbox"]
        predictions = {annotation["image_id"]: {
            "boxes": torch.tensor([[x, y, x + w, y + h]]),
            "labels": torch.tensor([annotation["category_id"]]),
            "scores": torch.ones(1),
            "keypoints": torch.tensor(annotation["keypoints"]).reshape(1, 8, 3),
        }}
        metric = callback._get_or_create_keypoint_oks_metric(SimpleNamespace(datamodule=data), split)
        metric.update(predictions)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="Found annotation id 0.*")
            scores = metric.compute()
        assert scores["map"] == pytest.approx(1.0)
        assert scores["mar"] == pytest.approx(1.0)


def test_rfdetr_native_augmentation_options_override_defaults(pose, tmp_path):
    _, result = job(pose, tmp_path)
    prepared = prepare_data(result)
    flags = {"multi_scale": True, "expanded_scales": True, "do_random_resize_via_padding": True, "lr_encoder": 5e-5}
    config = replace(result.config, backend_options=flags)
    variant, _, options = rfdetr_configs(result.selection, config, prepared, {})
    native = variant._train_config_class(dataset_dir=str(prepared.location), dataset_file="yolo", **options)
    assert all(getattr(native, key) == value for key, value in flags.items())
    with pytest.raises(ValueError, match="Unknown RF-DETR training options.*mosaic"):
        rfdetr_configs(result.selection, replace(config, backend_options={"mosaic": 1.0}), prepared, {})


@pytest.mark.parametrize("augmentations", [
    {"mosaic": 1.0},
    {"Mosaik": {}},
    {"HorizontalFlip": {"p": 2.0}},
    {"HorizontalFlip": {"probability": 0.5}},
    {"OneOf": {"transforms": [{"Blur": {"typo": 3}}]}},
    {"OneOf": {"p": 0.2, "transforms": [{"Blur": {"p": 1}}]}},
    {"SomeOf": {"transforms": ["Blur"]}},
])
@pytest.mark.parametrize("preview", [False, True])
def test_rfdetr_invalid_augmentations_fail_before_preparation(pose, monkeypatch, augmentations, preview):
    monkeypatch.setattr("dataset_fixer.training.backends.prepare_data", lambda *_: pytest.fail("prepared invalid options"))
    command = df.preview_augmentations if preview else df.train
    with pytest.raises(ValueError):
        command(pose, type=df.ModelTypes.RFDETR, augmentations=augmentations,
                config=df.TrainingConfig(resolution=96))


def test_augmentation_validation_covers_native_option_routes(pose, monkeypatch):
    from dataset_fixer.training.augmentations import validate_augmentations
    monkeypatch.setattr("dataset_fixer.training.backends.prepare_data", lambda *_: pytest.fail("prepared invalid options"))
    with pytest.raises(ValueError, match="mosaic"):
        df.train(pose, type=df.ModelTypes.RFDETR,
                 config=df.TrainingConfig(backend_options={"aug_config": {"mosaic": 0}}))
    with pytest.raises(ValueError, match="nnU-Net augmentations"):
        validate_augmentations(df.ModelTypes.NNUNET, df.TrainingConfig(), {"mosaic": 0})
    with pytest.raises(ValueError, match="mosaic"):
        df.train(pose, augmentations={"mosaic": 2.0})
    with pytest.raises(ValueError, match="Albumentations"):
        df.train(pose, augmentations={"augmentations": ["Blur"]})
    with pytest.raises(SyntaxError, match="mosaik"):
        df.train(pose, config=df.TrainingConfig(backend_options={"mosaik": 0}))


def test_rfdetr_nested_valid_config_and_missing_flip_metadata(pose, tmp_path):
    _, result = job(pose, tmp_path)
    prepared = prepare_data(result)
    aug = {"HorizontalFlip": {"p": 0.5}, "OneOf": [{"Blur": {"p": 0.3}}, {"GaussianBlur": {"p": 0.7}}]}
    _, _, options = rfdetr_configs(result.selection, result.config, prepared, aug)
    assert options["aug_config"] == aug
    from dataset_fixer.training.augmentations import validate_rfdetr_augmentations
    with pytest.raises(ValueError, match="no keypoint flip pairs"):
        validate_rfdetr_augmentations(aug, flip_pairs=[])
    with pytest.raises(ValueError, match="augmentation_backend='cpu'"):
        rfdetr_configs(result.selection, replace(result.config, backend_options={"augmentation_backend": "gpu"}), prepared, aug)


def test_yolo_rejects_already_constructed_transform_with_ignored_parameters(pose):
    import albumentations as A
    with pytest.warns(UserWarning, match="not valid"):
        transform = A.Blur(typo=3)
    with pytest.raises(ValueError, match="typo"):
        df.train(pose, augmentations={"mosaic": 0, "augmentations": [transform]})


def test_roboflow_download_is_cached_and_source_layout_normalized(pose, tmp_path, monkeypatch):
    calls = []
    class Version:
        def download(self, format, location):
            calls.append(location)
            assert not Path(location).exists()
            shutil.copytree(pose.location, location)
            yaml_path = Path(location) / "data.yaml"
            content = yaml.safe_load(yaml_path.read_text())
            content.pop("path", None)
            content.update(train="../train/images", val="../val/images")
            yaml_path.write_text(yaml.safe_dump(content))
    client = SimpleNamespace(workspace=lambda _: SimpleNamespace(project=lambda _: SimpleNamespace(version=lambda _: Version())))
    monkeypatch.setitem(sys.modules, "roboflow", SimpleNamespace(Roboflow=lambda **_: client))
    first = df.Dataset.open("roboflow:workspace/project/5", progress=False)
    second = df.Dataset.open("roboflow:workspace/project/5", progress=False)
    assert first.classes == second.classes == {0: "wolf"}
    assert len(calls) == 1
    image = next((first.location / "train").rglob("*.jpg"))
    image.unlink()
    df.Dataset.open("roboflow:workspace/project/5", progress=False)
    assert len(calls) == 2
    image = next((first.location / "train").rglob("*.jpg"))
    image.write_bytes(b"x" * image.stat().st_size)
    recovered = df.Dataset.open("roboflow:workspace/project/5", progress=False)
    assert len(calls) == 3 and recovered._metadata.flip_idx == pose._metadata.flip_idx


def test_train_announces_model_and_callbacks(pose, tmp_path, monkeypatch, capsys):
    from dataset_fixer.training.backends import ADAPTERS
    events = []
    def fake(result, prepared, augmentations):
        result._announce(result.config.resolution)
        result._emit("train_start")
        result._capture(None, checkpoints(tmp_path))
        result._emit("train_end")
    monkeypatch.setitem(ADAPTERS, df.ModelTypes.RFDETR, fake)
    result = df.train(pose, type=df.ModelTypes.RFDETR, config=df.TrainingConfig(output_dir=tmp_path / "training", resolution=96),
                      callbacks=[lambda event: events.append(event.name)])
    assert result.finished and result.bundle
    assert events == ["train_start", "checkpoint_saved", "train_end"]
    assert "RFDETRKeypointPreview: task=pose, resolution=96" in capsys.readouterr().out


@pytest.mark.parametrize("interval", [1, 10])
@pytest.mark.parametrize("upload", ["interval", "best"])
def test_real_lightning_adapter_keeps_resumable_current_epoch(pose, tmp_path, monkeypatch, run, interval, upload):
    import pytorch_lightning as pl
    import rfdetr
    from pytorch_lightning.callbacks import ModelCheckpoint
    from rfdetr.training.callbacks.best_model import BestModelCallback
    scores = (.5, .5, .6) if upload == "best" else (.5, .6)
    class Tiny(pl.LightningModule):
        def __init__(self, mc, tc):
            super().__init__()
            self.model_config, self.train_config = mc, tc
            self.model = torch.nn.Linear(1, 1)
        def training_step(self, batch, batch_idx):
            return self.model(batch).square().mean()
        def validation_step(self, batch, batch_idx):
            self.log("val/keypoint_map_50_95", torch.tensor(scores[self.current_epoch]))
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=.01)
    class Data(pl.LightningDataModule):
        def __init__(self, *args):
            super().__init__()
        def train_dataloader(self):
            return torch.utils.data.DataLoader(torch.ones(2, 1), batch_size=1)
        def val_dataloader(self):
            return self.train_dataloader()
    def build(tc, mc, **kwargs):
        last = ModelCheckpoint(dirpath=tc.output_dir, filename="last", save_top_k=1, enable_version_counter=False)
        best = BestModelCallback(tc.output_dir, monitor_regular="val/keypoint_map_50_95", run_test=False)
        return pl.Trainer(accelerator="cpu", devices=1, max_epochs=len(scores), callbacks=[last, best], logger=False,
                          enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0)
    monkeypatch.setattr(rfdetr, "RFDETRModelModule", Tiny)
    monkeypatch.setattr(rfdetr, "RFDETRDataModule", Data)
    monkeypatch.setattr(rfdetr, "build_trainer", build)
    cfg = df.TrainingConfig(output_dir=tmp_path / "native-training", epochs=len(scores), resolution=96, workers=0,
                            backend_options={"model": {"pretrain_weights": None}, "checkpoint_interval": interval})
    result = df.train(pose, type=df.ModelTypes.RFDETR, config=cfg, wandb=df.WandbConfig(run=run),
                      checkpointing=df.CheckpointConfig(wandb_contents="full", wandb_upload=upload))
    assert len(run.artifacts) >= 2
    for epoch, artifact in zip((0, len(scores) - 1), run.artifacts[:2]):
        with zipfile.ZipFile(io.BytesIO(next(iter(artifact.files.values())))) as zipped:
            latest = torch.load(io.BytesIO(zipped.read("weights/last.ckpt")), weights_only=False)
            best = torch.load(io.BytesIO(zipped.read("weights/checkpoint_best_regular.pth")), weights_only=False)
            assert latest["epoch"] == best["epoch"] == epoch
            assert latest["optimizer_states"]
            assert latest["dataset_schema"]["kpt_shape"] == [8, 3]
    assert result.best_weights.name == "checkpoint_best_total.pth"


def test_native_yolo_one_epoch(pose, tmp_path, run):
    import albumentations as A
    from ultralytics import YOLO
    from ultralytics.data.augment import Albumentations
    torch.set_num_threads(1)
    initial = tmp_path / "initial.pt"
    native = YOLO("yolo26n-pose.yaml")
    native.save(initial)
    augmentations = {"mosaic": 0.0, "mixup": 0.0, "augmentations": [A.Blur(p=1.0)]}
    applied = []
    def observe(event):
        if event.name == "train_start":
            custom = next(t for t in event.trainer.train_loader.dataset.transforms.transforms
                          if isinstance(t, Albumentations))
            applied.append((event.trainer.args.mosaic, event.trainer.args.mixup,
                            custom.transform.transforms[0].p))
    result = df.train(pose, weights=initial, augmentations=augmentations, callbacks=[observe], wandb=df.WandbConfig(run=run),
        config=df.TrainingConfig(output_dir=tmp_path / "yolo-training",
        epochs=1, resolution=64, batch_size=1, workers=0, device="cpu", backend_options={"amp": False, "plots": False, "val": True, "nbs": 1, "warmup_epochs": 0}))
    assert applied == [(0.0, 0.0, 1.0)]
    assert result.best_weights.is_file() and result.resumable_checkpoint.is_file()
    assert torch.load(result.resumable_checkpoint, weights_only=False)["optimizer"] is not None
    assert result.model.kind == "ultralytics"
    starts = []
    def started(event):
        if event.name == "train_start":
            starts.append((event.trainer.start_epoch, len(event.trainer.optimizer.state)))
    with pytest.raises(ValueError, match="optimizer state"):
        select(pose, resume=result.wandb_bundle.path)
    warmed = df.train(pose, weights=result.wandb_bundle.path, callbacks=[started], config=replace(result.config,
                      output_dir=tmp_path / "warm-start", resolution=96))
    assert starts == [(0, 0)] and warmed.best_weights.is_file()
    assert warmed.selection.provenance["sha256"]
    with pytest.raises(ValueError, match="Change resolution"):
        select(pose, resume=result.bundle.path, config=df.TrainingConfig(resolution=96))
    starts.clear()
    resumed = df.train(pose, resume=result.bundle.path, callbacks=[started], config=replace(result.config,
                      output_dir=tmp_path / "resumed", epochs=2))
    assert starts[0][0] == 1 and starts[0][1] > 0 and resumed.best_weights.is_file()
    df.preview_augmentations(pose, {"mosaic": 0.0}, type=df.ModelTypes.YOLO, version=26, s="n",
                            config=df.TrainingConfig(resolution=64, workers=0), samples=1,
                            destination=tmp_path / "yolo-augmentation.png", show=False)
    shutil.copyfile(tmp_path / "yolo-augmentation.png", "/tmp/dataset-fixer-yolo-augmentation.png")


@pytest.mark.parametrize("mosaic,rect,effective", [(1.0, False, 1.0), (0.0, False, 0.0), (1.0, True, 0.0)])
def test_yolo_preview_combines_native_and_custom_transforms(pose, tmp_path, monkeypatch, mosaic, rect, effective):
    import albumentations as A
    from ultralytics.data import build
    from ultralytics.data.augment import Albumentations, Mosaic
    original = build.build_yolo_dataset
    seen = []
    def capture(*args, **kwargs):
        native = original(*args, **kwargs)
        seen.append(native)
        return native
    monkeypatch.setattr(build, "build_yolo_dataset", capture)
    output = tmp_path / "native-augmentations.png"
    df.preview_augmentations(pose, {"mosaic": mosaic, "augmentations": [A.Blur(p=1.0)]},
        type=df.ModelTypes.YOLO, version=26, s="n", samples=1, show=False, destination=output,
        config=df.TrainingConfig(workers=0, backend_options={"rect": rect, "imgsz": 96, "batch": 1}))
    native = seen[0]
    assert native.imgsz == 96 and native.rect == rect
    stages = native.transforms.transforms
    assert next(t for t in stages[0].transforms if isinstance(t, Mosaic)).p == effective
    transforms = next(t for t in stages if isinstance(t, Albumentations)).transform.transforms
    assert len(transforms) == 1 and isinstance(transforms[0], A.Blur) and transforms[0].p == 1.0
    assert output.stat().st_size > 1000
    shutil.copyfile(output, f"/tmp/dataset-fixer-yolo-native-mosaic-{mosaic}-rect-{rect}.png")


def test_yolo_augmentation_defaults_and_explicit_disable():
    from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
    from ultralytics.data.augment import Albumentations
    from dataset_fixer.training.backends import _yolo_options
    defaults = get_cfg(overrides=_yolo_options(df.TrainingConfig(), {}))
    assert defaults.mosaic == DEFAULT_CFG_DICT["mosaic"]
    assert Albumentations().transform.transforms
    disabled = get_cfg(overrides=_yolo_options(df.TrainingConfig(), {"mosaic": 0.0, "augmentations": []}))
    assert disabled.mosaic == 0.0 and Albumentations(transforms=disabled.augmentations).transform.transforms == []
    with pytest.raises(ValueError, match="Conflicting resolution"):
        _yolo_options(df.TrainingConfig(resolution=64, backend_options={"imgsz": 96}), {})
    with pytest.raises(ValueError, match="TrainingConfig"):
        _yolo_options(df.TrainingConfig(), {"seed": 42})
    with pytest.raises(SyntaxError, match="not a valid YOLO argument"):
        _yolo_options(df.TrainingConfig(), {"mosaik": 1.0})


@pytest.mark.parametrize("semantic", [False, True])
def test_yolo_augmented_masks_use_native_dataset_builder(tmp_path, semantic):
    import albumentations as A
    root = make_yolo_dataset(tmp_path / "polygons", task="segment", names=["island"], size=(64, 64),
        train_rows=["0 .2 .2 .8 .2 .8 .8 .2 .8"], val_rows=["0 .2 .2 .8 .2 .8 .8 .2 .8"])
    data = df.Dataset.open(root, progress=False)
    if semantic:
        data = data.export(destination=tmp_path / "masks", format="semantic_masks", visualize=False, progress=False)
    output = tmp_path / "mask-augmentation.png"
    df.preview_augmentations(data, {"mosaic": 1.0, "augmentations": [A.Blur(p=1.0)]},
        type=df.ModelTypes.YOLO, version=26, s="n", config=df.TrainingConfig(resolution=64, workers=0),
        samples=1, show=False, destination=output)
    assert output.stat().st_size > 1000
    shutil.copyfile(output, f"/tmp/dataset-fixer-yolo-mask-semantic-{semantic}.png")


def test_native_nnunet_one_epoch(tmp_path, monkeypatch, run):
    import os
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("nnUNet_compile", "false")
    torch.set_num_threads(1)
    root = make_yolo_dataset(tmp_path / "segmentation", task="segment", names=["island"], size=(64, 64),
        train_rows=["0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"], val_rows=["0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"])
    data = df.Dataset.open(root, task="segment", progress=False).export(destination=tmp_path / "semantic", format="semantic_masks", visualize=False, progress=False)
    result = df.train(data, type=df.ModelTypes.NNUNET, wandb=df.WandbConfig(run=run), config=df.TrainingConfig(output_dir=tmp_path / "nnunet-training",
        epochs=1, resolution=64, batch_size=1, workers=0, device="cpu",
        backend_options={"num_iterations_per_epoch": 1, "num_val_iterations_per_epoch": 1}))
    assert result.best_weights.is_file() and result.resumable_checkpoint.is_file()
    saved = torch.load(result.resumable_checkpoint, weights_only=False)
    assert saved["optimizer_state"] and saved["trainer_name"] == "nnUNetTrainer"
    assert result.model.kind == "nnunet"
    assert json.loads((result.output_dir / "native/plans.json").read_text())["configurations"]["2d"]["batch_size"] == 1
    loaded = df.Model.load_many(result.bundle.path)[0]
    assert loaded.kind == "nnunet"
    assert select(data, weights=result.bundle.path).family == df.ModelTypes.NNUNET
    assert select(data, weights=result.wandb_bundle.path).family == df.ModelTypes.NNUNET
    with pytest.raises(ValueError, match="optimizer state"):
        select(data, resume=result.wandb_bundle.path)
    assert select(data, weights=result.best_weights).name == "nnUNetPlannerResEncM"
    df.preview_augmentations(data, type=df.ModelTypes.NNUNET, config=result.config, samples=1,
                             destination=tmp_path / "nnunet-augmentation.png", show=False)
    shutil.copyfile(tmp_path / "nnunet-augmentation.png", "/tmp/dataset-fixer-nnunet-augmentation.png")


def test_rfdetr_predictions_use_existing_eval_and_renderer(pose, tmp_path, monkeypatch):
    import numpy as np
    import rfdetr
    session, result = job(pose, tmp_path)
    result._capture(None, checkpoints(tmp_path))
    def predict(path, **kwargs):
        assert torch.is_inference_mode_enabled()
        return SimpleNamespace(data={"xyxy": np.array([[16, 12, 144, 108]])},
            detection_confidence=np.array([2.73]), class_id=np.array([0]),
            xy=np.array([[[160 * (.25 + i * .06), 60] for i in range(8)]]),
            keypoint_confidence=np.ones((1, 8)))
    optimized = []
    native = SimpleNamespace(model=SimpleNamespace(), model_config=SimpleNamespace(use_grouppose_keypoints=True),
                             predict=predict, optimize_for_inference=lambda **kwargs: optimized.append(kwargs))
    monkeypatch.setattr(rfdetr, "from_checkpoint", lambda *args, **kwargs: native)
    evaluated = result.evaluate(samples=1, plots=1)
    assert len(optimized) == 1
    assert native.model.model_config is native.model_config
    assert len(evaluated.ranking) == 1
    assert result.metrics["evaluation"]
    assert (result.output_dir / "predictions.png").is_file()
    shutil.copyfile(result.output_dir / "predictions.png", "/tmp/dataset-fixer-training-predictions.png")


def test_wandb_artifact_source_round_trip(pose, tmp_path, run, monkeypatch):
    import wandb
    session, result = job(pose, tmp_path, run)
    result._capture(None, checkpoints(tmp_path))
    artifact = run.artifacts[-1]
    def download(root):
        Path(root).mkdir(parents=True, exist_ok=True)
        for name, content in artifact.files.items():
            (Path(root) / name).write_bytes(content)
    artifact.download = download
    verified = []
    artifact.verify = lambda root: verified.append(root)
    monkeypatch.setattr(wandb, "Api", lambda: SimpleNamespace(run=lambda _: run, artifact=lambda _: artifact))
    resolved = select(pose, weights="wandb:team/project/test", config=df.TrainingConfig(resolution=192))
    assert resolved.provenance["name"] == artifact.qualified_name
    assert resolved.provenance["digest"] == artifact.digest and verified


def test_custom_checkpoint_provider_and_no_checkpoint_disconnect(pose, tmp_path, monkeypatch):
    calls = []
    class Provider:
        config = df.CheckpointConfig()
        def checkpoints(self, trainer, defaults):
            calls.append(defaults)
            return defaults
    session, result = job(pose, tmp_path, checkpointing=Provider())
    assert not session.finish()
    result._capture(None, checkpoints(tmp_path))
    assert len(calls) == 1
    assert session.finish()


def test_nnunet_rejects_incompatible_plans_before_loading(tmp_path, monkeypatch):
    from dataset_fixer.training.backends import train_nnunet
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "plans.json").write_text(json.dumps({"configurations": {"2d": {"architecture": {"name": "old"}}}}))
    preprocessed = tmp_path / "preprocessed" / "Dataset001_test"
    preprocessed.mkdir(parents=True)
    (preprocessed / "testPlans.json").write_text(json.dumps({"configurations": {"2d": {"architecture": {"name": "new"}}}}))
    dataset_json = tmp_path / "dataset.json"
    dataset_json.write_text('{}')
    prepared = SimpleNamespace(backend={"environment": {"nnUNet_preprocessed": str(preprocessed.parent)}, "dataset_name": preprocessed.name},
                               paths={"dataset_json": dataset_json})
    selection = SimpleNamespace(weights=parent / "best.pth", resume=None, metadata={"model_folder": str(parent)})
    result = SimpleNamespace(config=df.TrainingConfig(), selection=selection)
    with pytest.raises(ValueError, match="architecture"):
        train_nnunet(result, prepared, None)


def test_backup_failure_does_not_block_upload_or_other_runs(pose, tmp_path, run, monkeypatch):
    session, result = job(pose, tmp_path, run, checkpointing=df.CheckpointConfig(backup_dir=tmp_path / "drive", final_attempts=1))
    def fail_backup(source, destination):
        if tmp_path / "drive" in destination.parents:
            raise OSError("Drive unavailable")
        verified_copy(source, destination)
    monkeypatch.setattr("dataset_fixer.training.session.verified_copy", fail_backup)
    with pytest.warns(UserWarning, match="backup failed"):
        result._capture(None, checkpoints(tmp_path))
    assert result.uploaded_digest and result.copied_digest is None
    other = session._register(pose, result.selection, replace(result.config, output_dir=tmp_path / "other"),
                             checkpointing=df.CheckpointConfig())
    other._capture(None, checkpoints(tmp_path))
    with pytest.warns(UserWarning), pytest.raises(RuntimeError, match="Publication incomplete"):
        session.finish()
    assert other.finished and not result.finished
    monkeypatch.setattr("dataset_fixer.training.session.verified_copy", verified_copy)
    assert session.finish()
    result._capture(None, checkpoints(tmp_path, epoch=1))
    assert len(list((tmp_path / "drive" / result.output_dir.name).glob("*.zip"))) == 1
    assert set((result.output_dir / "bundles").glob("*.zip")) == {result.bundle.path, result.wandb_bundle.path}


def test_legacy_rf_metadata_and_native_positional_interpolation(pose, tmp_path):
    from dataset_fixer.rfdetr_engine import checkpoint_metadata
    from rfdetr.models.weights import interpolate_position_embeddings
    path = tmp_path / "checkpoint_best_total.pth"
    torch.save({"model": {"class_embed.weight": torch.zeros(2, 256), "_kp_active_mask": torch.ones(1, 8)},
                "args": SimpleNamespace(pretrain_weights="rf-detr-keypoint-preview.pth", class_names=["wolf"], resolution=96)}, path)
    selected = select(pose, weights=path)
    assert selected.task == "pose" and selected.name == "RFDETRKeypointPreview"
    metadata = checkpoint_metadata(path)
    assert metadata["model_config"]["num_keypoints_per_class"] == [8]
    key = "backbone.embeddings.position_embeddings"
    state = {key: torch.arange(65 * 4, dtype=torch.float32).reshape(1, 65, 4), "trained_head": torch.ones(2, 4)}
    class_token = state[key][:, :1].clone()
    interpolate_position_embeddings(state, 16)
    assert state[key].shape == (1, 257, 4)
    assert torch.equal(state[key][:, :1], class_token) and torch.equal(state["trained_head"], torch.ones(2, 4))


def test_failure_before_first_checkpoint_uploads_report_without_disconnect(pose, tmp_path, run, monkeypatch):
    session, result = job(pose, tmp_path, run, disconnect="after_safe", disconnect_delay=0)
    calls = []
    monkeypatch.setattr("dataset_fixer.training.session.in_colab", lambda: True)
    monkeypatch.setitem(sys.modules, "google.colab", SimpleNamespace(runtime=SimpleNamespace(unassign=lambda: calls.append(True))))
    with pytest.raises(RuntimeError, match="failed before epoch"):
        with session:
            raise RuntimeError("failed before epoch")
    assert not calls and result.best_weights is None and run.artifacts[-1].waited
    with zipfile.ZipFile(result.bundle.path) as archive:
        assert "failed before epoch" in archive.read("failure.txt").decode()
