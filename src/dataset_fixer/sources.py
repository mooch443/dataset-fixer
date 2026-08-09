"""Internal, content-addressed source caching and archive extraction.

This module deliberately has no public cache knobs.  Dataset and model APIs
use the same resolver so notebooks do not need to copy Drive files or unpack
archives themselves.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from tqdm.auto import tqdm

from .artifacts import DATASET_INFO_NAME
from .errors import DatasetValidationError, ValidationIssue


_COPY_CHUNK = 8 * 1024 * 1024


def in_colab() -> bool:
    """Return whether the current interpreter is a Google Colab kernel."""

    try:
        import google.colab  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def cache_root() -> Path:
    """Return the package-managed cache root for the current environment."""

    root = (
        Path("/content/dataset-fixer-cache")
        if in_colab()
        else Path("~/.cache/dataset-fixer").expanduser()
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _source_identity(path: Path) -> dict[str, Any]:
    stat_result = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
    }


def _same_source_identity(value: Any, expected: dict[str, Any]) -> bool:
    """Compare stable source fields while accepting legacy inode metadata."""

    return isinstance(value, dict) and all(
        value.get(key) == raw for key, raw in expected.items()
    )


def _is_colab_drive_file(path: Path) -> bool:
    return (
        in_colab()
        and path.is_file()
        and (path == Path("/content/drive") or Path("/content/drive") in path.parents)
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def sha256_progress(
    path: str | Path,
    *,
    progress: bool = True,
    description: str | None = None,
) -> str:
    """Hash one file while reporting byte progress."""

    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle, tqdm(
        total=source.stat().st_size,
        unit="B",
        unit_scale=True,
        desc=description or f"Hashing {source.name}",
        disable=not progress,
        leave=False,
    ) as bar:
        for chunk in iter(lambda: handle.read(_COPY_CHUNK), b""):
            digest.update(chunk)
            bar.update(len(chunk))
    return digest.hexdigest()


def local_source(path: str | Path, *, progress: bool = True) -> Path:
    """Copy a Google Drive input to local Colab storage when necessary."""

    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if not _is_colab_drive_file(source):
        return source

    identity = _source_identity(source)
    identity_hash = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    destination = cache_root() / "drive" / identity_hash / source.name
    metadata_path = destination.with_suffix(destination.suffix + ".source.json")
    candidates = [destination]
    candidates.extend(
        directory / source.name
        for directory in sorted((cache_root() / "drive").glob("*"))
        if directory.is_dir()
        if directory / source.name != destination
    )
    for candidate in candidates:
        candidate_metadata = candidate.with_suffix(candidate.suffix + ".source.json")
        if not candidate.is_file() or not candidate_metadata.is_file():
            continue
        try:
            metadata = json.loads(candidate_metadata.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            _same_source_identity(metadata, identity)
            and candidate.stat().st_size == identity["size"]
        ):
            if progress:
                print(f"Cache hit: local copy of {source.name}")
            return candidate

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{source.name}.", suffix=".partial", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle, tqdm(
            total=identity["size"],
            unit="B",
            unit_scale=True,
            desc=f"Copying {source.name} from Drive",
            disable=not progress,
        ) as bar:
            for chunk in iter(lambda: input_handle.read(_COPY_CHUNK), b""):
                output_handle.write(chunk)
                bar.update(len(chunk))
            output_handle.flush()
            os.fsync(output_handle.fileno())
        temporary.replace(destination)
        _atomic_json(metadata_path, identity)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _safe_member_path(destination: Path, member: zipfile.ZipInfo) -> Path:
    raw = PurePosixPath(member.filename)
    if raw.is_absolute() or ".." in raw.parts or not raw.parts:
        raise DatasetValidationError(
            ValidationIssue(
                "Unsafe ZIP member path",
                value=member.filename,
                expected="a relative path without '..' components",
            )
        )
    unix_mode = (member.external_attr >> 16) & 0xFFFF
    if stat.S_ISLNK(unix_mode):
        raise DatasetValidationError(
            ValidationIssue(
                "ZIP archives containing symbolic links are not supported",
                value=member.filename,
            )
        )
    target = destination.joinpath(*raw.parts)
    try:
        target.resolve().relative_to(destination.resolve())
    except ValueError as exc:
        raise DatasetValidationError(
            ValidationIssue("Unsafe ZIP member path", value=member.filename)
        ) from exc
    return target


@dataclass(frozen=True)
class ExtractedArchive:
    archive: Path
    root: Path
    sha256: str


def _source_index_path(source: Path, category: str) -> Path:
    source_key = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()
    return cache_root() / category / "sources" / f"{source_key}.json"


def _indexed_extraction(
    source: Path,
    category: str,
    *,
    progress: bool,
) -> ExtractedArchive | None:
    index = _source_index_path(source, category)
    if not index.is_file():
        return None
    try:
        value = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    identity = _source_identity(source)
    if not _same_source_identity(value.get("source"), identity):
        return None
    digest = str(value.get("sha256") or "")
    if len(digest) != 64:
        return None
    destination = cache_root() / category / "extracted" / digest
    marker = destination / ".complete.json"
    if not marker.is_file():
        return None
    try:
        complete = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if complete.get("sha256") != digest:
        return None
    if progress:
        print(f"Cache hit: extracted {source.name} (source metadata unchanged)")
    return ExtractedArchive(source, destination, digest)


def _record_extraction(source: Path, category: str, digest: str) -> None:
    _atomic_json(
        _source_index_path(source, category),
        {"source": _source_identity(source), "sha256": digest},
    )


def extract_archive(
    archive: str | Path,
    *,
    category: str,
    progress: bool = True,
) -> ExtractedArchive:
    """Safely and atomically extract a ZIP into the automatic cache."""

    requested = Path(archive).expanduser().resolve()
    if not requested.is_file() or requested.suffix.lower() != ".zip":
        raise DatasetValidationError(
            ValidationIssue("Source is not a ZIP archive", value=str(requested))
        )
    indexed = _indexed_extraction(requested, category, progress=progress)
    if indexed is not None:
        return indexed
    source = local_source(requested, progress=progress)
    if not source.is_file() or source.suffix.lower() != ".zip":
        raise DatasetValidationError(
            ValidationIssue("Source is not a ZIP archive", value=str(source))
        )
    digest = sha256_progress(source, progress=progress)
    destination = cache_root() / category / "extracted" / digest
    marker = destination / ".complete.json"
    if marker.is_file():
        try:
            value = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            value = {}
        if value.get("sha256") == digest:
            _record_extraction(requested, category, digest)
            if progress:
                print(f"Cache hit: extracted {source.name}")
            return ExtractedArchive(source, destination, digest)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{digest[:12]}.", dir=destination.parent)
    )
    try:
        with zipfile.ZipFile(source) as zipped:
            members = zipped.infolist()
            if not members:
                raise DatasetValidationError("ZIP archive is empty")
            total = sum(member.file_size for member in members if not member.is_dir())
            with tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=f"Extracting {source.name}",
                disable=not progress,
            ) as bar:
                for member in members:
                    target = _safe_member_path(temporary, member)
                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zipped.open(member) as input_handle, target.open("wb") as output_handle:
                        for chunk in iter(lambda: input_handle.read(_COPY_CHUNK), b""):
                            output_handle.write(chunk)
                            bar.update(len(chunk))
        _atomic_json(temporary / ".complete.json", {"sha256": digest})
        if destination.exists():
            # An interrupted/stale extraction is safe to replace because this
            # directory is owned by the content-addressed cache.
            shutil.rmtree(destination)
        temporary.replace(destination)
        _record_extraction(requested, category, digest)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return ExtractedArchive(source, destination, digest)


def _dataset_indicators(path: Path) -> bool:
    if (path / "reports" / DATASET_INFO_NAME).is_file():
        return True
    if (path / DATASET_INFO_NAME).is_file():
        return True
    if any((path / name).is_file() for name in ("data.yaml", "dataset.yaml", "data.yml")):
        return True
    if any(
        candidate.is_file() and not candidate.name.startswith(".")
        for candidate in path.glob("*.json")
    ):
        return True
    for split in ("train", "val", "test"):
        if (path / "images" / split).is_dir() or (path / split / "images").is_dir():
            return True
    return False


def _non_cache_children(path: Path) -> list[Path]:
    return sorted(
        child for child in path.iterdir() if child.name not in {".complete.json", "__MACOSX"}
    )


def find_dataset_root(extracted: Path) -> Path:
    """Find exactly one dataset root inside an extracted archive."""

    current = extracted
    while True:
        if _dataset_indicators(current):
            break
        children = _non_cache_children(current)
        directories = [child for child in children if child.is_dir()]
        files = [child for child in children if child.is_file()]
        if len(directories) == 1 and not files:
            current = directories[0]
            continue
        break

    candidates: list[Path] = []
    if _dataset_indicators(current):
        candidates.append(current)
    for candidate in current.rglob("*"):
        if candidate.is_dir() and _dataset_indicators(candidate):
            if not any(parent in candidates for parent in candidate.parents):
                candidates.append(candidate)
    unique = sorted({candidate.resolve() for candidate in candidates})
    if len(unique) != 1:
        raise DatasetValidationError(
            ValidationIssue(
                "Dataset ZIP must contain exactly one unambiguous dataset root",
                value=[str(path.relative_to(extracted)) for path in unique],
                expected="one dataset directory/YAML/JSON root",
            )
        )
    return unique[0]


def resolve_dataset_source(location: str | Path, *, progress: bool = True) -> Path:
    """Resolve a normal dataset source or transparently unpack a dataset ZIP."""

    requested = Path(location).expanduser().resolve()
    if requested.suffix.lower() != ".zip":
        return requested
    extracted = extract_archive(requested, category="datasets", progress=progress)
    return find_dataset_root(extracted.root)


def fingerprint_files(
    files: Iterable[Path],
    *,
    progress: bool = True,
    description: str = "Hashing source files",
) -> str:
    """Hash ordered file paths and bytes with one concise progress bar."""

    paths = sorted({Path(path).expanduser().resolve() for path in files})
    digest = hashlib.sha256()
    total = sum(path.stat().st_size for path in paths)
    common = Path(os.path.commonpath([str(path.parent) for path in paths])) if paths else Path(".")
    with tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=description,
        disable=not progress,
        leave=False,
    ) as bar:
        for path in paths:
            digest.update(path.relative_to(common).as_posix().encode("utf-8"))
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(_COPY_CHUNK), b""):
                    digest.update(chunk)
                    bar.update(len(chunk))
    return digest.hexdigest()
