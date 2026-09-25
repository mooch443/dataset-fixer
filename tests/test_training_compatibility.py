"""Existing loading and installation contracts remain valid with training APIs."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import tomllib
import subprocess
import sys

import pytest
import torch
import yaml

from dataset_fixer import Dataset, Model
from dataset_fixer.model_sources import _download_wandb
from dataset_fixer.model import ModelInput


def test_standard_install_retains_existing_backends():
    from packaging.requirements import Requirement

    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"]
    dependencies = {Requirement(value).name for value in project["dependencies"]}
    assert {"ultralytics", "nnunetv2", "batchgeneratorsv2"} <= dependencies
    assert not {"rfdetr", "roboflow"} & dependencies


def test_existing_api_imports_without_new_optional_backends():
    subprocess.run([sys.executable, "-c", """
import sys
from importlib.abc import MetaPathFinder
class WithoutNewBackends(MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'rfdetr', 'roboflow', 'pytorch_lightning'}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, WithoutNewBackends())
import dataset_fixer as df
assert df.Dataset.open and df.Model.load_many and df.train
"""], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("suffix", [".pth", ".ckpt"])
def test_model_does_not_infer_rfdetr_from_extension_alone(tmp_path, suffix):
    path = tmp_path / f"legacy-model{suffix}"
    torch.save({"model": torch.nn.Linear(2, 1), "train_args": {"task": "detect"}}, path)
    assert Model(path, task="detect").kind == "ultralytics"
    # These generic metadata names are also used by unrelated Lightning models.
    torch.save({"model_name": "OtherNetwork", "model_config": {"num_classes": 2}, "args": {"pretrain_weights": "base.pth"}}, path)
    assert Model(path, task="detect").kind == "ultralytics"
    # Existing exported files may use formats other than torch serialization.
    path.write_bytes(b"exported-model")
    assert Model(path, task="detect").kind == "ultralytics"


def test_rfdetr_auto_detection_uses_checkpoint_metadata(tmp_path):
    path = tmp_path / "model.PTH"
    torch.save({"model_name": "RFDETRKeypointPreview", "model_config": {"use_grouppose_keypoints": True}}, path)
    model = Model(path)
    assert model.kind == "rfdetr" and model.task == "pose"


def test_visualization_annotations_preserve_prediction_input_identity(tmp_path):
    original = ModelInput("image", tmp_path / "image.png", 64, 64, "image.png")
    annotated = replace(original, reference_annotations=({"class_id": 0, "keypoints": [[4, 5, 2]]},))
    cached = {original: "cached prediction"}
    assert annotated == original and cached[annotated] == "cached prediction"


def test_local_dataset_path_recovery_is_explicit_and_read_only(detect_dataset):
    yaml_path = detect_dataset / "data.yaml"
    values = yaml.safe_load(yaml_path.read_text())
    values["train"] = "../train/images"
    yaml_path.write_text(yaml.safe_dump(values))
    original = yaml_path.read_bytes()
    with pytest.warns(UserWarning, match="Dataset path fallback.*../train/images"):
        dataset = Dataset.open(detect_dataset, progress=False)
    assert len(dataset._samples) == 6
    assert any("../train/images" in warning for warning in dataset.warnings)
    assert yaml_path.read_bytes() == original


def test_legacy_wandb_file_survives_optional_artifact_discovery_failure(tmp_path, monkeypatch):
    import wandb

    def unavailable():
        raise ConnectionError("Artifact API unavailable")

    payload = b"legacy-checkpoint"
    remote = SimpleNamespace(name="weights/best.pt", size=len(payload))
    run = SimpleNamespace(id="old-run", summary={}, logged_artifacts=unavailable, files=lambda: [remote])
    monkeypatch.setattr(wandb, "Api", lambda: SimpleNamespace(run=lambda _: run))

    def download(_, root, **kwargs):
        path = root / "best.pt"
        path.write_bytes(payload)
        return path

    monkeypatch.setattr("dataset_fixer.model_sources._download_wandb_file", download)
    with pytest.warns(RuntimeWarning, match="trying legacy run files"):
        path, resolved_run = _download_wandb("wandb:team/project/old-run", requested=None, progress=False)
    assert path.read_bytes() == payload and resolved_run is run


def test_recorded_wandb_artifact_failure_cannot_fall_back_to_other_weights(monkeypatch):
    import wandb

    def unavailable(_):
        raise ConnectionError("Recorded artifact unavailable")

    run = SimpleNamespace(summary={"best_model_artifact": "team/project/model:v3"},
                          files=lambda: pytest.fail("Must not replace the recorded best weights"))
    monkeypatch.setattr(wandb, "Api", lambda: SimpleNamespace(run=lambda _: run, artifact=unavailable))
    with pytest.raises(ConnectionError, match="Recorded artifact"):
        _download_wandb("wandb:team/project/run", requested=None, progress=False)
