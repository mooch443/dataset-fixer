from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from .artifacts import dataset_info_path, lineage_path, read_lineage
from .utils import sha256_file

if TYPE_CHECKING:
    from .dataset import Dataset


@dataclass(frozen=True)
class DatasetTraceNode:
    """One physical dataset in a current-to-source lineage chain."""

    dataset_id: str | None
    name: str
    path: Path
    present: bool
    generated: bool
    samples: int | None
    resolved_by: str


@dataclass(frozen=True)
class SampleTrace:
    """Exact source mapping for one physically present output sample."""

    output_image: str
    parent_image: str | None
    original_image: str | None
    resolved_parent_image: Path | None
    resolved_original_image: Path | None
    parent_sha256: str | None
    original_sha256: str | None
    tile_index: int | None
    crop: Any
    transformation_chain: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class DatasetTrace:
    """Resolved dataset ancestry plus exact sample and tile mappings."""

    datasets: tuple[DatasetTraceNode, ...]
    samples: tuple[SampleTrace, ...]

    @property
    def tiles(self) -> tuple[SampleTrace, ...]:
        return tuple(sample for sample in self.samples if sample.tile_index is not None or sample.crop)

    @property
    def complete(self) -> bool:
        return all(node.present for node in self.datasets)

    def for_sample(self, relative_path: str | Path) -> SampleTrace:
        requested = Path(relative_path).as_posix()
        matches = [
            sample
            for sample in self.samples
            if sample.output_image == requested
            or Path(sample.output_image).name == Path(requested).name
            or sample.output_image.endswith("/" + requested)
        ]
        if len(matches) != 1:
            raise KeyError(
                f"Expected exactly one lineage record for {requested!r}; found {len(matches)}"
            )
        return matches[0]

    def summary(self) -> str:
        lines = [
            f"Dataset trace: {len(self.datasets)} dataset(s), "
            f"{len(self.samples)} present sample(s), {len(self.tiles)} tile(s)"
        ]
        for index, node in enumerate(self.datasets):
            marker = "present" if node.present else "missing"
            identity = (node.dataset_id or "unknown")[:12]
            count = "?" if node.samples is None else str(node.samples)
            lines.append(
                f"  {index}. {node.name} [{identity}] {marker}; "
                f"samples={count}; path={node.path}"
            )
        unresolved = sum(
            1
            for sample in self.samples
            if sample.original_image and sample.resolved_original_image is None
        )
        if unresolved:
            lines.append(f"  unresolved original image paths: {unresolved}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def trace_dataset(
    dataset: "Dataset",
    *,
    search_paths: Iterable[str | Path] = (),
    path_rewrites: Mapping[str | Path, str | Path] | None = None,
) -> DatasetTrace:
    """Trace a materialized dataset through every discoverable physical parent."""

    rewrites = tuple(
        (Path(old).expanduser(), Path(new).expanduser())
        for old, new in (path_rewrites or {}).items()
    )
    roots = _unique_paths(
        [dataset.location.parent, *(Path(path).expanduser() for path in search_paths)]
    )
    nodes: list[DatasetTraceNode] = []
    seen_ids: set[str] = set()
    current_path = dataset.location
    current_manifest = dataset.manifest
    current_name = dataset.name

    while True:
        dataset_id = _dataset_id(current_manifest)
        if dataset_id and dataset_id in seen_ids:
            break
        if dataset_id:
            seen_ids.add(dataset_id)
        records = _read_records(current_path)
        first_node = not nodes
        nodes.append(
            DatasetTraceNode(
                dataset_id=dataset_id,
                name=current_name,
                path=current_path,
                present=current_path.is_dir(),
                generated=dataset_info_path(current_path).is_file(),
                samples=len(records) if records else None,
                resolved_by="current" if first_node else "dataset-info",
            )
        )
        source = current_manifest.get("source_dataset")
        if not isinstance(source, dict) or not source:
            break
        source_id = str(source.get("id")) if source.get("id") else None
        stored = Path(str(source.get("path") or source.get("location") or "")).expanduser()
        resolved, resolved_by = _resolve_dataset_path(
            stored,
            source_id=source_id,
            roots=roots,
            rewrites=rewrites,
        )
        source_manifest = _read_manifest(resolved) if resolved.is_dir() else {}
        embedded_source = source.get("source_dataset")
        if not source_manifest and isinstance(embedded_source, dict) and embedded_source:
            source_manifest = {
                "name": source.get("name"),
                "dataset_id": source_id,
                "source_dataset": embedded_source,
            }
        nodes.append(
            DatasetTraceNode(
                dataset_id=source_id or _dataset_id(source_manifest),
                name=str(source.get("name") or resolved.name or "source"),
                path=resolved,
                present=resolved.is_dir(),
                generated=dataset_info_path(resolved).is_file(),
                samples=_manifest_sample_count(source_manifest),
                resolved_by=resolved_by,
            )
        )
        if not source_manifest or not isinstance(source_manifest.get("source_dataset"), dict):
            break
        # Replace the source placeholder with the richer generated-dataset node
        # on the next loop iteration.
        nodes.pop()
        current_path = resolved
        current_manifest = source_manifest
        current_name = str(source_manifest.get("name") or source.get("name") or resolved.name)

    current_records = _read_records(dataset.location)
    sample_traces = tuple(
        _sample_trace(record, roots=roots, rewrites=rewrites)
        for record in current_records
    )
    return DatasetTrace(datasets=tuple(nodes), samples=sample_traces)


def _sample_trace(
    record: Mapping[str, Any],
    *,
    roots: tuple[Path, ...],
    rewrites: tuple[tuple[Path, Path], ...],
) -> SampleTrace:
    parent = str(record.get("parent_image")) if record.get("parent_image") else None
    original = str(record.get("original_image")) if record.get("original_image") else None
    return SampleTrace(
        output_image=str(record.get("output_image") or ""),
        parent_image=parent,
        original_image=original,
        resolved_parent_image=_resolve_image(
            parent,
            expected_sha=str(record.get("parent_sha256") or "") or None,
            roots=roots,
            rewrites=rewrites,
        ),
        resolved_original_image=_resolve_image(
            original,
            expected_sha=str(record.get("original_sha256") or "") or None,
            roots=roots,
            rewrites=rewrites,
        ),
        parent_sha256=str(record.get("parent_sha256")) if record.get("parent_sha256") else None,
        original_sha256=str(record.get("original_sha256")) if record.get("original_sha256") else None,
        tile_index=int(record["tile_index"]) if record.get("tile_index") is not None else None,
        crop=record.get("crop"),
        transformation_chain=tuple(
            dict(item)
            for item in record.get("transformation_chain") or []
            if isinstance(item, dict)
        ),
    )


def _resolve_dataset_path(
    stored: Path,
    *,
    source_id: str | None,
    roots: tuple[Path, ...],
    rewrites: tuple[tuple[Path, Path], ...],
) -> tuple[Path, str]:
    for candidate, method in _path_candidates(stored, rewrites):
        if candidate.is_dir() and _path_matches_id(candidate, source_id):
            return candidate.resolve(), method
    if source_id:
        for root in roots:
            if not root.is_dir():
                continue
            for info in root.rglob("dataset-info.json"):
                if any(part in {".cache", "evaluations"} for part in info.parts):
                    continue
                manifest = _read_json(info)
                if _dataset_id(manifest) == source_id:
                    return info.parent.parent.resolve(), "id-search"
    return (_rewrite_path(stored, rewrites) or stored).resolve(), "unresolved"


def _resolve_image(
    stored: str | None,
    *,
    expected_sha: str | None,
    roots: tuple[Path, ...],
    rewrites: tuple[tuple[Path, Path], ...],
) -> Path | None:
    if not stored:
        return None
    path = Path(stored).expanduser()
    for candidate, _ in _path_candidates(path, rewrites):
        if candidate.is_file() and _hash_matches(candidate, expected_sha):
            return candidate.resolve()
    for root in roots:
        if not root.is_dir():
            continue
        for candidate in root.rglob(path.name):
            if candidate.is_file() and _hash_matches(candidate, expected_sha):
                return candidate.resolve()
    return None


def _path_candidates(
    path: Path,
    rewrites: tuple[tuple[Path, Path], ...],
) -> tuple[tuple[Path, str], ...]:
    rewritten = _rewrite_path(path, rewrites)
    candidates: list[tuple[Path, str]] = []
    if rewritten is not None:
        candidates.append((rewritten, "path-rewrite"))
    candidates.append((path, "stored-path"))
    return tuple(candidates)


def _rewrite_path(
    path: Path,
    rewrites: tuple[tuple[Path, Path], ...],
) -> Path | None:
    for old, new in rewrites:
        try:
            relative = path.relative_to(old)
        except ValueError:
            continue
        return new / relative
    return None


def _path_matches_id(path: Path, expected: str | None) -> bool:
    if expected is None:
        return True
    manifest = _read_manifest(path)
    return not manifest or _dataset_id(manifest) == expected


def _read_records(path: Path) -> list[dict[str, Any]]:
    candidate = lineage_path(path)
    if candidate.is_file():
        try:
            return read_lineage(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            return []
    legacy = path / "provenance.jsonl"
    if not legacy.is_file():
        return []
    records: list[dict[str, Any]] = []
    try:
        for line in legacy.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if isinstance(record, dict):
                    records.append(record)
    except (OSError, json.JSONDecodeError):
        return []
    return records


def _read_manifest(path: Path) -> dict[str, Any]:
    modern = dataset_info_path(path)
    if modern.is_file():
        return _read_json(modern)
    legacy = path / "dataset-fixer.json"
    return _read_json(legacy) if legacy.is_file() else {}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _dataset_id(manifest: Mapping[str, Any]) -> str | None:
    value = manifest.get("dataset_id") or manifest.get("id")
    return str(value) if value else None


def _manifest_sample_count(manifest: Mapping[str, Any]) -> int | None:
    lineage = manifest.get("lineage")
    if isinstance(lineage, dict) and lineage.get("records") is not None:
        return int(lineage["records"])
    return None


def _hash_matches(path: Path, expected: str | None) -> bool:
    return expected is None or sha256_file(path) == expected


def _unique_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in result:
            result.append(resolved)
    return tuple(result)
