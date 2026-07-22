from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from enum import Enum
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
STANDARD_SPLITS = ("train", "val", "test")
SPLIT_ALIASES = {"validation": "val", "valid": "val"}


def normalize_split(value: str) -> str:
    split = SPLIT_ALIASES.get(value.lower(), value.lower())
    if split not in STANDARD_SPLITS:
        raise ValueError(f"Unsupported split {value!r}; expected train, val, or test")
    return split


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return value or "dataset"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def settings_fingerprint(settings: dict[str, Any]) -> str:
    encoded = json.dumps(to_jsonable(settings), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:8]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    if callable(value):
        return {"module": getattr(value, "__module__", None), "qualname": getattr(value, "__qualname__", None), "repr": repr(value)}
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


def package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in (
        "dataset-fixer", "numpy", "Pillow", "PyYAML", "matplotlib", "shapely", "tqdm",
        "albumentations", "ultralytics", "sahi",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return result


def git_state(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", "-C", str(path), *args], capture_output=True, text=True, check=True, timeout=3
            )
            return proc.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit or "uncommitted", "dirty": bool(status) if status is not None else None}


def environment_snapshot(project_root: Path | None = None) -> dict[str, Any]:
    dataset_fixer_state = git_state(Path(__file__).resolve().parents[2])
    try:
        from ._version import __version__, commit_id
    except ImportError:
        __version__, commit_id = "0.1.0", None
    if commit_id:
        dataset_fixer_state["commit"] = commit_id
        dataset_fixer_state["dirty"] = bool(re.search(r"\.d\d{8}(?:$|\+)", __version__))
    dataset_fixer_state["version"] = __version__
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": package_versions(),
        "dataset_fixer_git": dataset_fixer_state,
        "caller_git": git_state(project_root or Path.cwd()),
    }


def ensure_safe_destination(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError(f"Destination must be separate from source: source={source}, destination={destination}")


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def unique_preserving_order(values: Iterable[Any]) -> list[Any]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
