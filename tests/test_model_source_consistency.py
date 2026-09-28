"""Every public model entry point uses the same backend/source resolution."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import torch

import dataset_fixer as df
from dataset_fixer.training.selection import select
from conftest import make_yolo_dataset


@pytest.fixture(params=["ultralytics", "rfdetr", "nnunet"])
def source_files(request, tmp_path):
    family = request.param
    root = tmp_path / "model"
    root.mkdir()
    if family == "nnunet":
        (root / "fold_0").mkdir()
        (root / "dataset.json").write_text(json.dumps({"channel_names": {"0": "R", "1": "G", "2": "B"},
                                                      "labels": {"background": 0, "foreground": 1}, "file_ending": ".png"}))
        (root / "plans.json").write_text(json.dumps({"experiment_planner_used": "nnUNetPlannerResEncM"}))
        path = root / "fold_0" / "checkpoint_final.pth"
        torch.save({"network_weights": {}}, path)
        local = root
        task, name = "semantic_segment", "nnUNetPlannerResEncM"
    elif family == "rfdetr":
        from rfdetr import RFDETRKeypointPreview
        config = RFDETRKeypointPreview._model_config_class(resolution=96, num_classes=1, num_keypoints_per_class=[2])
        path = root / "best.pth"
        torch.save({"model": {}, "model_name": "RFDETRKeypointPreview", "model_config": config.model_dump(),
                    "args": {"class_names": ["animal"], "keypoint_flip_pairs": [0, 1]}}, path)
        local, task, name = path, "pose", "RFDETRKeypointPreview"
    else:
        path = root / "best.pt"
        torch.save({"train_args": {"task": "pose", "model": "yolo26n-pose.pt", "imgsz": 96}}, path)
        local, task, name = path, "pose", "yolo26n-pose"
    bundle = tmp_path / "model.zip"
    manifest = {"format": "model-bundle", "framework": family, "geometry": {"input_size": [96, 96]},
                "model": {"framework": family, "task": task, "model_name": name, "checkpoint": path.name,
                          "folds": [0], "planner": "nnUNetPlannerResEncM"}}
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("bundle_manifest.json", json.dumps(manifest))
        for file in root.rglob("*"):
            if file.is_file():
                archive.write(file, str(file.relative_to(root)))
    return family, task, local, bundle


@pytest.mark.parametrize("source_kind", ["local", "bundle", "wandb"])
def test_direct_and_collection_loading_resolve_identically(source_files, source_kind, monkeypatch):
    family, task, local, bundle = source_files
    downloads = []
    def download(reference, **kwargs):
        downloads.append(reference)
        return bundle, None
    monkeypatch.setattr("dataset_fixer.model_sources._download_wandb", download)
    reference = {"local": local, "bundle": bundle, "wandb": "wandb:team/project/runs/run123"}[source_kind]
    direct = df.Model(reference, device="mps")
    collection = df.Model.load_many({"source": reference, "device": "mps"})[0]
    assert direct.path == collection.path
    assert direct.kind == collection.kind == family
    assert direct.task == collection.task == task
    assert direct.name == collection.name
    assert direct.resolution == collection.resolution
    assert direct.geometry == collection.geometry
    assert direct.device == collection.device == "mps"
    assert not direct.loaded and not collection.loaded
    renamed = df.Model.load_many({"renamed": direct})[0]
    assert renamed.name == "renamed" and renamed.path == direct.path
    assert renamed.kind == direct.kind and renamed.geometry == direct.geometry
    assert renamed._resolved_source.manifest == direct._resolved_source.manifest
    if source_kind == "wandb":
        assert downloads == ["wandb:team/project/run123"] * 2
        assert direct.source_key == collection.source_key == "wandb:team/project/run123"
        assert direct.wandb == "https://wandb.ai/team/project/runs/run123"
    configured = df.Model(reference, name="explicit", resolution=192, device="cpu")
    assert configured.name == "explicit" and configured.resolution == 192
    assert configured.geometry.input_size == (192, 192)


@pytest.mark.parametrize("source_files", ["rfdetr"], indirect=True)
def test_loaded_rfdetr_sources_work_for_training_and_preview(source_files, tmp_path, monkeypatch):
    family, task, local, bundle = source_files
    monkeypatch.setattr("dataset_fixer.convert.cache_root", lambda: tmp_path / "prepared-cache")
    monkeypatch.setattr("dataset_fixer.model_sources._download_wandb", lambda *a, **kw: (bundle, None))
    root = make_yolo_dataset(tmp_path / "data", task="pose", names=["animal"], size=(96, 96),
                            train_rows=["0 .5 .5 .6 .6 .3 .4 2 .7 .6 2"],
                            extra={"kpt_shape": [2, 3], "flip_idx": [1, 0]})
    # The fixture helper defaults validation to detection rows; preview needs only train.
    (root / "val" / "labels" / "val_0.txt").write_text("0 .5 .5 .6 .6 .3 .4 2 .7 .6 2\n")
    dataset = df.Dataset.open(root, progress=False)
    model = df.Model("wandb:team/project/runs/run123", device="mps")
    selected = select(dataset, weights=model)
    assert selected.family == df.ModelTypes.RFDETR and selected.task == "pose"
    assert selected.resolution == 96 and selected.metadata["model"]["model_name"] == "RFDETRKeypointPreview"
    output = tmp_path / "rfdetr.svg"
    assert dataset.visualize(weights=model, augmentations={"HorizontalFlip": {"p": 1}},
                             samples=None, split="val", columns=2, zoom=True, minimum_context=20,
                             markers=df.MarkerStyle(shape="cross", radius=5, show_boxes=False),
                             destination=output, show=False) is None
    assert output.stat().st_size > 1000 and not model.loaded


@pytest.mark.parametrize("source_files", ["rfdetr"], indirect=True)
def test_rfdetr_metadata_detection_does_not_depend_on_checkpoint_suffix(source_files):
    family, _, local, _ = source_files
    renamed = Path(local).with_suffix(".pt")
    renamed.write_bytes(Path(local).read_bytes())
    model = df.Model(renamed)
    assert model.kind == "rfdetr" and model.task == "pose" and model.resolution == 96
