from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from PIL import Image


def make_image(path: Path, size: tuple[int, int] = (160, 120), color=(70, 110, 150)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def make_yolo_dataset(
    root: Path,
    *,
    task: str,
    names: list[str] | None = None,
    train_rows: list[str] | None = None,
    val_rows: list[str] | None = None,
    size: tuple[int, int] = (160, 120),
    extra: dict | None = None,
) -> Path:
    names = names or ["fruit", "damaged"]
    train_rows = train_rows or ["0 0.5 0.5 0.25 0.25", "1 0.25 0.25 0.2 0.2"]
    val_rows = val_rows if val_rows is not None else ["0 0.5 0.5 0.25 0.25"]
    for index, row in enumerate(train_rows):
        image = root / "train" / "images" / f"group_{index // 2}" / f"train_{index}.jpg"
        make_image(image, size=size)
        label = root / "train" / "labels" / f"group_{index // 2}" / f"train_{index}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(row + "\n", encoding="utf-8")
    for index, row in enumerate(val_rows):
        image = root / "val" / "images" / f"val_{index}.jpg"
        make_image(image, size=size, color=(110, 80, 130))
        label = root / "val" / "labels" / f"val_{index}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(row + "\n", encoding="utf-8")
    data = {
        "path": str(root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": None,
        "names": names,
        "name": root.name,
    }
    data.update(extra or {})
    (root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return root


@pytest.fixture
def detect_dataset(tmp_path: Path) -> Path:
    return make_yolo_dataset(
        tmp_path / "orchard",
        task="detect",
        train_rows=[
            "0 0.5 0.5 0.25 0.25\n1 0.2 0.2 0.1 0.1",
            "0 0.45 0.5 0.2 0.3",
            "1 0.6 0.5 0.2 0.2",
            "0 0.5 0.4 0.3 0.2",
        ],
        val_rows=["0 0.5 0.5 0.25 0.25", "1 0.3 0.4 0.15 0.2"],
    )
