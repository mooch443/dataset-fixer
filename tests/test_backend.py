from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dataset_fixer import Dataset
from conftest import make_yolo_dataset


def test_ultralytics_detection_preflight(detect_dataset: Path) -> None:
    pytest.importorskip("ultralytics")
    dataset = Dataset.open(detect_dataset, task="detect", progress=False)
    dataset.assert_trainable(backend=True)


def test_modified_ultralytics_locate_loader(tmp_path: Path) -> None:
    pytest.importorskip("ultralytics")
    from ultralytics.cfg import TASKS
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset

    if "locate" not in TASKS:
        pytest.skip("installed Ultralytics does not include the POLO locate task")
    source = make_yolo_dataset(
        tmp_path / "polo_backend",
        task="polo",
        names=["fruit"],
        train_rows=["0 15 0.4 0.4", "0 15 0.6 0.6"],
        val_rows=["0 15 0.5 0.5"],
        extra={"radii": {0: 15}},
    )
    dataset = Dataset.open(source, task="polo", progress=False)
    dataset.assert_trainable(backend=True)
    checked = check_det_dataset(str(dataset.data_yaml), autodownload=False)
    loader = YOLODataset(
        img_path=checked["train"],
        data=checked,
        task="locate",
        imgsz=64,
        augment=False,
        cache=False,
        rect=False,
        batch_size=1,
        stride=32,
        prefix="dataset-fixer integration: ",
    )
    assert loader.use_locations
    assert len(loader) == 2
    assert loader.labels[0]["locations"].shape[1] == 2
    assert loader.labels[0]["radii"].shape[1] == 1
