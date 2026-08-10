from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

from dataset_fixer import Dataset, Geometry
from dataset_fixer.bundle import Config, _dataset
from dataset_fixer.convert import Kind, prepare
from dataset_fixer.wandb import configure
from conftest import make_yolo_dataset


class _RunConfig(dict):
    def update(self, values: dict, allow_val_change: bool = False) -> None:
        assert allow_val_change
        super().update(values)


def _zip_dataset(tmp_path: Path) -> Path:
    root = make_yolo_dataset(
        tmp_path / "source-folder",
        task="segment",
        names=["island"],
        train_rows=["0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9"],
        val_rows=["0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"],
    )
    archive = tmp_path / "islands-128-08.08.2026-merged-3class_masks.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for path in root.rglob("*"):
            if path.is_file():
                zipped.write(path, Path("dataset") / path.relative_to(root))
    return archive


def test_zip_source_basename_reaches_preparation_bundle_and_wandb(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = _zip_dataset(tmp_path)
    monkeypatch.setattr("dataset_fixer.sources.cache_root", lambda: tmp_path / "cache")
    dataset = Dataset.open(archive, task="segment", progress=False)
    prepared = prepare(
        dataset,
        Kind.YOLO_SEG,
        destination=tmp_path / "prepared",
        name="model",
        native_tile_size=(120, 160),
        progress=False,
    )

    assert dataset._source_name == archive.name
    assert prepared.source_name == archive.name
    assert _dataset(prepared)["source_dataset_zip"] == archive.name

    run = SimpleNamespace(
        config=_RunConfig(),
        tags=("existing", f"source-zip-{archive.stem}"),
        update=lambda: None,
    )
    configure(run, prepared.config)
    assert run.config["dataset_source"] == archive.name
    assert run.config["source_dataset_zip"] == archive.name
    assert archive.name in run.tags
    assert f"source-zip-{archive.stem}" not in run.tags

    old_manifest = json.loads(prepared.manifest.read_text(encoding="utf-8"))
    old_manifest["source"].pop("basename")
    prepared.manifest.write_text(json.dumps(old_manifest), encoding="utf-8")
    reused = prepare(
        dataset,
        Kind.YOLO_SEG,
        destination=tmp_path / "prepared",
        name="model",
        native_tile_size=(120, 160),
        progress=False,
    )
    assert reused.reused
    assert reused.source_name == archive.name
    reused_run = SimpleNamespace(config=_RunConfig(), tags=(), update=lambda: None)
    configure(reused_run, reused.config)
    assert archive.name in reused_run.tags


def test_folder_source_basename_is_tagged_without_absolute_path(tmp_path: Path) -> None:
    folder = make_yolo_dataset(tmp_path / "training-folder", task="detect")
    dataset = Dataset.open(folder, progress=False)
    config = Config(
        name="model",
        framework="ultralytics",
        task="detect",
        geometry=Geometry.create(native_tile_size=64),
        dataset={"dataset_source": dataset._source_name},
    )
    run = SimpleNamespace(config=_RunConfig(), tags=(), update=lambda: None)

    configure(run, config)

    assert dataset._source_name == "training-folder"
    assert run.config["dataset_source"] == "training-folder"
    assert "source_dataset_zip" not in run.config
    assert "training-folder" in run.tags
    assert str(tmp_path) not in repr((run.config, run.tags))
