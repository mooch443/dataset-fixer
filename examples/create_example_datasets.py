"""Create deterministic MIT-licensed example datasets for the documentation."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml
from PIL import Image, ImageDraw


def create_example_datasets(root: str | Path, *, seed: int = 42) -> dict[str, Path]:
    """Create small YOLO detection and POLO datasets and return their roots."""

    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "raw_sequences": _create_detection(root / "raw_sequences", train=18, val=0, seed=seed),
        "detection": _create_detection(root / "detection", train=12, val=6, seed=seed + 1),
        "polo": _create_polo(root / "polo", train=8, val=4, seed=seed + 2),
    }
    (root / "DATA_LICENSE.txt").write_text(
        "The example images and annotations generated here are released under the MIT License.\n"
        "Copyright (c) 2026 Tristan Walter. See the repository LICENSE file.\n",
        encoding="utf-8",
    )
    return outputs


def _create_detection(root: Path, *, train: int, val: int, seed: int) -> Path:
    names = ["fruit", "damaged"]
    for split, count in (("train", train), ("val", val)):
        for index in range(count):
            rng = random.Random(seed * 10_000 + index + (1000 if split == "val" else 0))
            width, height = 640, 420
            image = Image.new("RGB", (width, height), _background(index, split))
            draw = ImageDraw.Draw(image)
            rows: list[str] = []
            sequence = f"row-{index // 3:02d}"
            for fruit_index in range(2 + index % 4):
                radius = rng.randint(22, 38)
                x = rng.randint(radius + 5, width - radius - 5)
                y = rng.randint(radius + 5, height - radius - 5)
                class_id = 1 if (index + fruit_index) % 5 == 0 else 0
                color = (196, 71, 62) if class_id else (242, 178, 52)
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=3)
                rows.append(_yolo_box(class_id, x, y, radius * 2, radius * 2, width, height))
            stem = f"{sequence}__frame-{index:03d}"
            image_path = root / "images" / split / sequence / f"{stem}.jpg"
            label_path = root / "labels" / split / sequence / f"{stem}.txt"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(image_path, quality=95)
            label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    _write_yaml(root, names=names, train=train > 0, val=val > 0)
    return root


def _create_polo(root: Path, *, train: int, val: int, seed: int) -> Path:
    radius = 18
    for split, count in (("train", train), ("val", val)):
        for index in range(count):
            rng = random.Random(seed * 10_000 + index + (1000 if split == "val" else 0))
            width, height = 1280, 900
            image = Image.new("RGB", (width, height), _background(index, split))
            draw = ImageDraw.Draw(image)
            rows: list[str] = []
            point_count = 5 + (index * 3) % 13
            for point_index in range(point_count):
                x = rng.randint(45, width - 45)
                y = rng.randint(45, height - 45)
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(244, 177, 51), outline="white", width=3)
                rows.append(f"0 {radius} {x / width:.8f} {y / height:.8f}")
            stem = f"tree-{split}-{index:03d}"
            image_path = root / "images" / split / f"{stem}.jpg"
            label_path = root / "labels" / split / f"{stem}.txt"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(image_path, quality=95)
            label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    _write_yaml(root, names=["fruit"], train=train > 0, val=val > 0, radii={0: radius})
    return root


def _write_yaml(
    root: Path,
    *,
    names: list[str],
    train: bool,
    val: bool,
    radii: dict[int, int] | None = None,
) -> None:
    payload = {
        "path": str(root),
        "train": "images/train" if train else None,
        "val": "images/val" if val else None,
        "test": None,
        "names": names,
        "name": root.name,
    }
    if radii:
        payload["radii"] = radii
    (root / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _yolo_box(class_id: int, x: float, y: float, width: float, height: float, image_width: int, image_height: int) -> str:
    return f"{class_id} {x / image_width:.8f} {y / image_height:.8f} {width / image_width:.8f} {height / image_height:.8f}"


def _background(index: int, split: str) -> tuple[int, int, int]:
    shift = 17 if split == "val" else 0
    return 34 + (index * 7 + shift) % 30, 78 + (index * 11 + shift) % 35, 49 + (index * 5 + shift) % 25


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", default="example-datasets")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name, path in create_example_datasets(args.destination, seed=args.seed).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
