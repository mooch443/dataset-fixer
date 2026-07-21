"""Download a small, pinned subset of the public MIT-licensed SAWIT dataset."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import yaml

SAWIT_REPOSITORY = "https://github.com/dtnguyen0304/sawit"
SAWIT_COMMIT = "8dd9e2a0b08879f78b293d50e9b59bb24bcdfcc0"
SAWIT_LICENSE = "MIT"
CLASS_NAMES = ["frog", "lizard", "bird", "small_mammal", "large_mammal", "spider", "scorpion"]
PREFIXES = ("IMG-Frog-", "IMG-Liz-", "IMG-Bird-")


def download_sawit_examples(root: str | Path, *, images_per_class: int = 8) -> dict[str, Path]:
    """Download official SAWIT images/labels and create unsplit and fixed-split views.

    The files are selected deterministically from a pinned upstream commit. No
    annotations or pixels are generated or altered.
    """

    if images_per_class < 4:
        raise ValueError("images_per_class must be at least 4")
    root = Path(root).expanduser().resolve()
    cache = root / "upstream"
    unsplit = root / "sawit_unsplit"
    fixed = root / "sawit_fixed"
    expected = root / "SOURCE.json"
    if expected.is_file() and unsplit.is_dir() and fixed.is_dir():
        source = json.loads(expected.read_text(encoding="utf-8"))
        if source.get("commit") == SAWIT_COMMIT and source.get("images_per_class") == images_per_class:
            return {"unsplit": unsplit, "fixed": fixed}

    root.mkdir(parents=True, exist_ok=True)
    image_entries = _github_directory("data/images/test")
    label_entries = {entry["name"]: entry for entry in _github_directory("data/labels/YOLO_format/test")}
    selected: list[dict[str, Any]] = []
    for prefix in PREFIXES:
        candidates = sorted(
            (entry for entry in image_entries if entry["name"].startswith(prefix)),
            key=lambda entry: entry["name"],
        )
        if len(candidates) < images_per_class:
            raise RuntimeError(f"SAWIT has only {len(candidates)} matching files for {prefix!r}")
        selected.extend(candidates[:images_per_class])

    records: list[dict[str, Any]] = []
    for image_entry in selected:
        image_name = image_entry["name"]
        label_name = f"{Path(image_name).stem}.txt"
        if label_name not in label_entries:
            raise RuntimeError(f"Official SAWIT label is missing for {image_name}")
        category = _category_from_name(image_name)
        image_cache = cache / "images" / image_name
        label_cache = cache / "labels" / label_name
        _download(_raw_url(f"data/images/test/{image_name}"), image_cache)
        _download(_raw_url(f"data/labels/YOLO_format/test/{label_name}"), label_cache)
        records.append(
            {
                "image": image_name,
                "label": label_name,
                "category": category,
                "image_sha256": _sha256(image_cache),
                "label_sha256": _sha256(label_cache),
            }
        )

    _materialize_unsplit(unsplit, records, cache)
    _materialize_fixed(fixed, records, cache, images_per_class)
    license_path = root / "UPSTREAM_LICENSE"
    _download(_raw_url("LICENSE"), license_path)
    source = {
        "dataset": "SAWIT: A Small-Sized Animal Wild Image Dataset with Annotations",
        "repository": SAWIT_REPOSITORY,
        "commit": SAWIT_COMMIT,
        "license": SAWIT_LICENSE,
        "license_file": str(license_path),
        "images_per_class": images_per_class,
        "selection": "lexicographically first files for IMG-Frog-, IMG-Liz-, and IMG-Bird-",
        "files": records,
    }
    expected.write_text(json.dumps(source, indent=2, sort_keys=True), encoding="utf-8")
    return {"unsplit": unsplit, "fixed": fixed}


def _github_directory(path: str) -> list[dict[str, Any]]:
    url = f"https://api.github.com/repos/dtnguyen0304/sawit/contents/{path}?ref={SAWIT_COMMIT}"
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "dataset-fixer"})
    with urllib.request.urlopen(request, timeout=60) as response:
        value = json.load(response)
    if not isinstance(value, list):
        raise RuntimeError(f"Unexpected GitHub response for {path}: {value}")
    return value


def _raw_url(path: str) -> str:
    return f"https://raw.githubusercontent.com/dtnguyen0304/sawit/{SAWIT_COMMIT}/{path}"


def _download(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "dataset-fixer"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(destination)


def _materialize_unsplit(root: Path, records: list[dict[str, Any]], cache: Path) -> None:
    for record in records:
        _copy_pair(root, "train", record, cache)
    _write_yaml(root, train=True, val=False)


def _materialize_fixed(root: Path, records: list[dict[str, Any]], cache: Path, images_per_class: int) -> None:
    counters: dict[str, int] = {}
    validation_per_class = max(2, images_per_class // 4)
    for record in records:
        category = record["category"]
        index = counters.get(category, 0)
        counters[category] = index + 1
        split = "val" if index >= images_per_class - validation_per_class else "train"
        _copy_pair(root, split, record, cache)
    _write_yaml(root, train=True, val=True)


def _copy_pair(root: Path, split: str, record: dict[str, Any], cache: Path) -> None:
    category = record["category"]
    image_output = root / "images" / split / category / record["image"]
    label_output = root / "labels" / split / category / record["label"]
    image_output.parent.mkdir(parents=True, exist_ok=True)
    label_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache / "images" / record["image"], image_output)
    shutil.copy2(cache / "labels" / record["label"], label_output)


def _write_yaml(root: Path, *, train: bool, val: bool) -> None:
    value = {
        "path": str(root.resolve()),
        "train": "images/train" if train else None,
        "val": "images/val" if val else None,
        "test": None,
        "names": CLASS_NAMES,
        "name": root.name,
        "source": SAWIT_REPOSITORY,
        "source_commit": SAWIT_COMMIT,
        "source_license": SAWIT_LICENSE,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _category_from_name(name: str) -> str:
    if name.startswith("IMG-Frog-"):
        return "frog"
    if name.startswith("IMG-Liz-"):
        return "lizard"
    if name.startswith("IMG-Bird-"):
        return "bird"
    raise ValueError(f"Unsupported tutorial category: {name}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", default="public-example-datasets")
    parser.add_argument("--images-per-class", type=int, default=8)
    arguments = parser.parse_args()
    for key, value in download_sawit_examples(arguments.destination, images_per_class=arguments.images_per_class).items():
        print(f"{key}: {value}")
