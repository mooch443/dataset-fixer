from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from dataset_fixer import Dataset, DatasetValidationError, Geometry, Model
from dataset_fixer.bundle import Config, Outcome, create
from dataset_fixer.convert import Kind, prepare
from dataset_fixer.model_sources import _download_wandb_file
from dataset_fixer.sources import extract_archive
from dataset_fixer.wandb import configure, upload
from conftest import make_yolo_dataset


def _semantic_dataset(tmp_path: Path, *, size: tuple[int, int] = (32, 32)) -> Dataset:
    source = make_yolo_dataset(
        tmp_path / "polygons",
        task="segment",
        names=["island"],
        train_rows=["0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9"],
        val_rows=["0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8"],
        size=size,
    )
    return Dataset.open(source, task="segment", progress=False).export(
        destination=tmp_path / "semantic",
        format="semantic_masks",
        visualize=False,
        progress=False,
    )


def test_dataset_open_infers_and_reuses_zip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = _semantic_dataset(tmp_path)
    archive = tmp_path / "semantic.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for path in dataset.location.rglob("*"):
            if path.is_file():
                zipped.write(path, Path("one-root") / path.relative_to(dataset.location))

    first = Dataset.open(archive, progress=False)
    second = Dataset.open(archive, progress=True)

    assert first.splits == dataset.splits
    assert second.location == first.location
    assert "Cache hit: extracted semantic.zip" in capsys.readouterr().out


def test_safe_zip_rejects_traversal_and_does_not_publish_partial_cache(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("../escape.txt", "no")

    with pytest.raises(DatasetValidationError, match="Unsafe ZIP member path"):
        extract_archive(archive, category="test-unsafe", progress=False)
    assert not (tmp_path / "escape.txt").exists()


def test_wandb_signed_url_download_reports_byte_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"wandb-model-bytes" * 1024
    source = tmp_path / "remote.pt"
    source.write_bytes(payload)
    seen = {"total": None, "bytes": 0}

    class RecordingBar:
        def __init__(self, **kwargs: object) -> None:
            seen["total"] = kwargs.get("total")

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def update(self, count: int) -> None:
            seen["bytes"] += count

    monkeypatch.setattr("dataset_fixer.model_sources.tqdm", RecordingBar)
    remote = SimpleNamespace(
        name="best.pt",
        url=source.as_uri(),
        download=lambda **_: pytest.fail("SDK fallback should not be used"),
    )
    root = tmp_path / "downloads"
    root.mkdir()

    downloaded = _download_wandb_file(
        remote,
        root,
        expected_size=len(payload),
        progress=True,
    )

    assert downloaded.read_bytes() == payload
    assert seen == {"total": len(payload), "bytes": len(payload)}


def test_model_load_many_standalone_and_immutable_configure(tmp_path: Path) -> None:
    checkpoint = tmp_path / "standalone-yolox.pt"
    checkpoint.write_bytes(b"checkpoint")

    original = Model.load_many([checkpoint])
    configured = original.configure(
        {
            "standalone-yolox": {
                "task": "semantic",
                "native_tile_size": 128,
                "upscale_factor": 2,
                "inference": "sahi",
                "device": "cuda",
                "batch_size": 8,
                "workers": 4,
            }
        }
    )

    assert original[0].geometry == Geometry()
    assert original[0].device is None
    assert configured[0] is not original[0]
    assert configured[0].geometry == Geometry((128, 128), 2, (256, 256))
    assert configured[0].settings["sahi_slice_height"] == 128
    assert configured[0].device == "cuda"
    assert original[0].batch_size == -1
    assert configured[0].batch_size == 8
    with pytest.raises(KeyError, match="Unknown model"):
        original.configure({"missing": {"device": "cpu"}})
    with pytest.raises(DatasetValidationError, match="Contradictory model geometry"):
        original.configure(
            {
                "standalone-yolox": {
                    "native_tile_size": 128,
                    "upscale_factor": 2,
                    "input_size": 512,
                }
            }
        )


def test_tiled_geometry_mismatch_fails_before_runtime(tmp_path: Path) -> None:
    dataset = _semantic_dataset(tmp_path, size=(128, 128))
    dataset._manifest.setdefault("history", []).append(
        {"operation": "tile-grid", "settings": {"tile_size": 128}}
    )
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    models = Model.load_many(
        {
            "trained-on-256": {
                "source": checkpoint,
                "task": "semantic",
                "native_tile_size": 256,
                "upscale_factor": 2,
                "input_size": 512,
            }
        }
    )

    with pytest.raises(DatasetValidationError, match="inference was not started"):
        models.compare(dataset, progress=False)
    assert not models[0].loaded


def test_prepare_thresholds_jpeg_resizes_and_reuses(tmp_path: Path) -> None:
    dataset = _semantic_dataset(tmp_path, size=(32, 32))
    sample = next(value for value in dataset._samples if value.split == "train")
    original_mask = dataset._mask_paths[sample.image_path.resolve()]
    values = np.asarray(Image.open(original_mask).convert("L"), dtype=np.uint8)
    jpeg = original_mask.with_suffix(".jpg")
    Image.fromarray(values).save(jpeg, quality=70)
    dataset._mask_paths[sample.image_path.resolve()] = jpeg.resolve()

    first = prepare(
        dataset,
        Kind.YOLO_SEM,
        name="semantic-run-a",
        native_tile_size=32,
        upscale_factor=2,
        destination=tmp_path / "prepared",
        workers=1,
        mask_threshold=128,
        base_model="yolo26m-sem.pt",
        epochs=5,
        device="cpu",
        progress=False,
    )
    second = prepare(
        dataset,
        Kind.YOLO_SEM,
        name="semantic-run-b",
        native_tile_size=32,
        upscale_factor=2,
        destination=tmp_path / "prepared",
        workers=1,
        mask_threshold=128,
        base_model="yolo26m-sem.pt",
        epochs=10,
        device="cuda",
        progress=False,
    )

    assert first.geometry.input_size == (64, 64)
    assert second.reused
    assert first.config is not None
    assert first.config.framework == "ultralytics"
    assert first.config.task == "semantic"
    assert first.config.model["base_model"] == "yolo26m-sem.pt"
    assert first.config.training["imgsz"] == 64
    assert first.config.training["epochs"] == 5
    assert second.config is not None
    assert second.config.name == "semantic-run-b"
    assert second.config.training["epochs"] == 10
    assert second.config.training["device"] == "cuda"
    prepared_masks = sorted((first.location / "masks").rglob("*.png"))
    assert prepared_masks
    with Image.open(prepared_masks[0]) as opened:
        assert opened.size == (64, 64)
        assert set(np.unique(np.asarray(opened)).tolist()) <= {0, 1}
    cases = json.loads(first.paths["cases"].read_text(encoding="utf-8"))
    jpeg_record = next(value for value in cases if value["source_mask"].endswith(".jpg"))
    assert jpeg_record["mask_source"]["threshold"] == 128


def test_prepare_returns_complete_nnunet_bundle_config(tmp_path: Path) -> None:
    dataset = _semantic_dataset(tmp_path, size=(32, 32))

    data_only = prepare(
        dataset,
        Kind.NNUNET,
        native_tile_size=32,
        upscale_factor=2,
        destination=tmp_path / "prepared-nnunet",
        workers=1,
        preprocess=False,
        planner="nnUNetPlannerResEncM",
        progress=False,
    )
    plan_directory = (
        data_only.paths["preprocessed_root"] / str(data_only.backend["dataset_name"])
    )
    plan_directory.mkdir(parents=True, exist_ok=True)
    (plan_directory / "nnUNetResEncUNetMPlans.json").write_text(
        json.dumps({"plans_name": "nnUNetResEncUNetMPlans"}),
        encoding="utf-8",
    )

    prepared = prepare(
        dataset,
        Kind.NNUNET,
        name="nnunet-resenc-m-64px-2x",
        native_tile_size=32,
        upscale_factor=2,
        destination=tmp_path / "prepared-nnunet",
        workers=1,
        preprocess=False,
        planner="nnUNetPlannerResEncM",
        trainer="nnUNetTrainer_100epochs",
        configuration="2d",
        fold=1,
        epochs=100,
        checkpoint_name="checkpoint_final.pth",
        device="cpu",
        progress=False,
    )

    assert prepared.reused
    assert prepared.config is not None
    assert prepared.config.framework == "nnunetv2"
    assert prepared.config.task == "semantic"
    assert prepared.config.geometry.input_size == (64, 64)
    assert prepared.config.model == {
        "name": "nnunet-resenc-m-64px-2x",
        "planner": "nnUNetPlannerResEncM",
        "plans": "nnUNetResEncUNetMPlans",
        "trainer": "nnUNetTrainer_100epochs",
        "configuration": "2d",
        "fold": 1,
        "checkpoint": "checkpoint_final.pth",
    }
    assert prepared.config.training["epochs"] == 100
    assert prepared.config.training["device"] == "cpu"


def test_prepare_requires_name_for_run_specific_configuration(tmp_path: Path) -> None:
    dataset = _semantic_dataset(tmp_path, size=(32, 32))

    with pytest.raises(ValueError, match="requires name"):
        prepare(
            dataset,
            Kind.YOLO_SEM,
            destination=tmp_path / "prepared-without-name",
            base_model="yolo26m-sem.pt",
            progress=False,
        )


def test_yolo_seg_never_converts_semantic_masks_to_polygons(tmp_path: Path) -> None:
    dataset = _semantic_dataset(tmp_path)
    with pytest.raises(DatasetValidationError, match="never converted to polygons"):
        prepare(
            dataset,
            Kind.YOLO_SEG,
            destination=tmp_path / "bad-seg",
            progress=False,
        )


def test_new_bundle_round_trip_loads_geometry(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"selected-weights")
    config = Config(
        name="island-sem",
        framework="ultralytics",
        task="semantic",
        geometry=Geometry.create(native_tile_size=128, upscale_factor=2),
        dataset={"content_sha256": "abc", "preparation_kind": "yolo-sem"},
        model={"name": "island-sem"},
        training={"epochs": 5},
    )
    outcome = Outcome(checkpoint=checkpoint, selection_metric="dice", selection_value=0.8)

    bundle = create(config, outcome, destination=tmp_path / "bundles", progress=False)
    loaded = Model.load_many([bundle.path])

    assert bundle.path.is_file()
    assert bundle.size == bundle.path.stat().st_size
    assert len(bundle.sha256) == 64
    assert loaded.names == ("island-sem",)
    assert loaded[0].geometry == Geometry((128, 128), 2, (256, 256))


class _Config(dict):
    def update(self, values: dict, allow_val_change: bool = False) -> None:
        assert allow_val_change
        super().update(values)


def test_wandb_helpers_configure_upload_and_preserve_local_failures(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"weights")
    config = Config(
        name="school-sem",
        framework="ultralytics",
        task="semantic",
        geometry=Geometry.create(native_tile_size=128, upscale_factor=2),
        dataset={"content_sha256": "abc", "preparation_kind": "yolo-sem"},
    )
    bundle = create(config, Outcome(checkpoint=checkpoint), destination=tmp_path / "bundle", progress=False)
    run = SimpleNamespace(id="run", config=_Config(), tags=("existing",), summary={})
    run.update = lambda: None
    upload_call: dict[str, str] = {}

    def upload_file(path: str, root: str = ".") -> SimpleNamespace:
        upload_call.update(path=path, root=root)
        return SimpleNamespace(url="https://example.invalid/file")

    run.upload_file = upload_file

    assert configure(run, config) is run
    assert run.config["native_tile_size"] == [128, 128]
    assert "native-128x128" in run.tags
    assert all("dataset-fixer" not in str(value) for value in [*run.tags, *run.config])
    uploaded = upload(run, bundle)
    assert uploaded.uploaded
    assert uploaded.path == bundle.path and uploaded.path.is_file()
    assert upload_call == {"path": str(bundle.path), "root": str(bundle.path.parent)}
    assert run.summary["evaluation_bundle"] == bundle.path.name

    failed_run = SimpleNamespace(id="run", summary={})
    failed_run.upload_file = lambda _path: (_ for _ in ()).throw(ConnectionError("offline"))
    with pytest.warns(RuntimeWarning, match="local bundle remains"):
        failed = upload(failed_run, bundle)
    assert not failed.uploaded
    assert failed.path.is_file()
    assert "evaluation_bundle" not in failed_run.summary

    missing = upload(None, bundle)
    assert not missing.uploaded
    assert missing.path.is_file()
