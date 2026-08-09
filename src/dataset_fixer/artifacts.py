from __future__ import annotations

import csv
import gzip
import json
import re
import shutil
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Iterable

from .utils import settings_fingerprint, to_jsonable


REPORTS_DIR = "reports"
DATASET_INFO_NAME = "dataset-info.json"
SOURCE_INFO_NAME = "source.json"
LINEAGE_NAME = "lineage.json.gz"
DATASET_INFO_SCHEMA = 4
LINEAGE_SCHEMA = 3
SUPPORTED_LINEAGE_SCHEMAS = (2, LINEAGE_SCHEMA)
_LINEAGE_RECORDS = re.compile(r'"records"\s*:\s*\[')

# Files a generated dataset's reports/ directory is allowed to contain. Anything
# else is a transient operation artifact that belongs in dataset-info.json.
CANONICAL_REPORT_FILES = (
    DATASET_INFO_NAME,
    SOURCE_INFO_NAME,
    LINEAGE_NAME,
    "plots.png",
)


def dataset_info_path(root: Path) -> Path:
    return root / REPORTS_DIR / DATASET_INFO_NAME


def source_info_path(root: Path) -> Path:
    return root / REPORTS_DIR / SOURCE_INFO_NAME


def lineage_path(root: Path) -> Path:
    return root / REPORTS_DIR / LINEAGE_NAME


def stable_dataset_id(
    *,
    name: str,
    task: str,
    format_name: str,
    classes: Any,
    records: Iterable[dict[str, Any]],
) -> str:
    samples = [
        {
            "output": str(record.get("output_image")),
            "image_sha256": record.get("output_sha256")
            or record.get("output_image_sha256"),
            "mask_sha256": record.get("output_mask_sha256"),
            "annotations": record.get("output_annotation_count"),
        }
        for record in records
    ]
    return settings_fingerprint(
        {
            "schema": DATASET_INFO_SCHEMA,
            "name": name,
            "task": task,
            "format": format_name,
            "classes": classes,
            "samples": samples,
        }
    )


def source_dataset_id(
    *,
    source_manifest: dict[str, Any],
    source_name: str,
    source_path: Path,
    records: Iterable[dict[str, Any]],
) -> str:
    existing = source_manifest.get("dataset_id") or source_manifest.get("id")
    if existing:
        return str(existing)
    # Paths are deliberately excluded from the identity. A dataset may be moved
    # or mounted elsewhere without becoming a different physical dataset.
    samples = sorted(
        (
            str(record.get("parent_sha256")),
            str(record.get("original_sha256") or ""),
        )
        for record in records
    )
    return settings_fingerprint(
        {
            "schema": DATASET_INFO_SCHEMA,
            "name": source_manifest.get("name") or source_name,
            "samples": samples,
        }
    )


def split_image_summary(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Count labeled and background output images for every dataset split."""

    counts: dict[str, dict[str, int]] = {}
    for record in records:
        split = str(record.get("output_split") or "unknown")
        details = counts.setdefault(
            split,
            {
                "total_images": 0,
                "labeled_images": 0,
                "background_images": 0,
            },
        )
        has_labels = record.get("output_has_labels")
        if has_labels is None:
            has_labels = int(record.get("output_annotation_count") or 0) > 0
        details["total_images"] += 1
        details["labeled_images" if bool(has_labels) else "background_images"] += 1
    return {split: counts[split] for split in sorted(counts)}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_lineage(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write compact lineage with shared transformation definitions.

    Transformation settings can be large (for example a resolved split-group
    assignment) and are usually identical for every output record. Schema 3
    stores each unique transformation once and references it from records.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    record_values = records if isinstance(records, Sequence) else list(records)
    transformations: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for record in record_values:
        for step in record.get("transformation_chain") or ():
            if not isinstance(step, dict):
                continue
            identity = _lineage_step_identity(step)
            if identity not in transformations:
                transformations[identity] = (
                    f"t{len(transformations)}",
                    to_jsonable(step),
                )
    transformation_payload = {
        identifier: step for identifier, step in transformations.values()
    }
    header = {
        "schema": "dataset-fixer-lineage",
        "schema_version": LINEAGE_SCHEMA,
        "transformations": transformation_payload,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=9) as handle:
        encoded_header = json.dumps(
            header, ensure_ascii=False, separators=(",", ":")
        )
        handle.write(encoded_header[:-1])
        handle.write(',"records":[')
        for index, record in enumerate(record_values):
            value = dict(record)
            chain = []
            for step in record.get("transformation_chain") or ():
                if not isinstance(step, dict):
                    continue
                identity = _lineage_step_identity(step)
                chain.append(transformations[identity][0])
            value["transformation_chain"] = chain
            if index:
                handle.write(",")
            json.dump(
                to_jsonable(value),
                handle,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        handle.write("]}")
    temporary.replace(path)


def read_lineage(path: Path) -> list[dict[str, Any]]:
    """Read lineage records with memory bounded to one expanded record."""

    return list(iter_lineage(path))


def iter_lineage(path: Path) -> Iterator[dict[str, Any]]:
    """Stream schema-2/3 lineage without inflating the whole gzip document.

    Schema 2 repeated complete transformation settings in every record. The
    parser also interns those repeated steps while reading an older artifact,
    so callers retaining all records do not retain thousands of copies.
    """

    decoder = json.JSONDecoder()
    shared_steps: dict[tuple[str, str], dict[str, Any]] = {}
    chunk_size = 1024 * 1024
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        buffer = ""
        match = None
        while match is None:
            chunk = handle.read(chunk_size)
            if not chunk:
                raise ValueError(f"Lineage records are missing: {path}")
            buffer += chunk
            match = _LINEAGE_RECORDS.search(buffer)

        header_text = buffer[: match.start()] + '"records":[]}'
        try:
            header = json.loads(header_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed lineage header: {path}: {exc}") from exc
        if header.get("schema") != "dataset-fixer-lineage":
            raise ValueError(f"Unsupported lineage document: {path}")
        schema_version = int(header.get("schema_version", -1))
        if schema_version not in SUPPORTED_LINEAGE_SCHEMAS:
            expected = ", ".join(str(value) for value in SUPPORTED_LINEAGE_SCHEMAS)
            raise ValueError(
                f"Unsupported lineage schema {header.get('schema_version')!r}; "
                f"expected one of {expected}"
            )
        transformations = header.get("transformations") or {}
        if schema_version >= 3 and not isinstance(transformations, dict):
            raise ValueError(f"Lineage transformations must be an object: {path}")

        buffer = buffer[match.end() :]
        position = 0
        expect_record = True
        finished = False
        while not finished:
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    break
                chunk = handle.read(chunk_size)
                if not chunk:
                    raise ValueError(f"Truncated lineage records: {path}")
                buffer = buffer[position:] + chunk
                position = 0

            if buffer[position] == "]":
                position += 1
                finished = True
                break
            if not expect_record:
                if buffer[position] != ",":
                    raise ValueError(f"Malformed lineage record separator: {path}")
                position += 1
                expect_record = True
                continue

            while True:
                try:
                    record, end = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError as exc:
                    chunk = handle.read(chunk_size)
                    if not chunk:
                        raise ValueError(
                            f"Malformed or truncated lineage record: {path}: {exc}"
                        ) from exc
                    buffer += chunk
            position = end
            expect_record = False
            if isinstance(record, dict):
                if schema_version >= 3:
                    identifiers = record.get("transformation_chain") or ()
                    try:
                        record["transformation_chain"] = [
                            transformations[str(identifier)]
                            for identifier in identifiers
                        ]
                    except (KeyError, TypeError) as exc:
                        raise ValueError(
                            f"Invalid lineage transformation reference: {path}: {exc}"
                        ) from exc
                _intern_lineage_steps(record, shared_steps)
                yield record

            # Keep at most the unconsumed suffix plus one record in memory.
            if position >= chunk_size:
                buffer = buffer[position:]
                position = 0

        remainder = buffer[position:] + handle.read()
        if remainder.strip() != "}":
            raise ValueError(f"Malformed lineage document ending: {path}")


def _intern_lineage_steps(
    record: dict[str, Any],
    shared_steps: dict[tuple[str, str], dict[str, Any]],
) -> None:
    chain = record.get("transformation_chain")
    if not isinstance(chain, list):
        return
    interned: list[Any] = []
    for step in chain:
        if not isinstance(step, dict):
            interned.append(step)
            continue
        identity = _lineage_step_identity(step)
        shared = shared_steps.setdefault(identity, step)
        interned.append(shared)
    record["transformation_chain"] = interned


def _lineage_step_identity(step: dict[str, Any]) -> tuple[str, str]:
    fingerprint = step.get("settings_fingerprint")
    if not fingerprint:
        fingerprint = settings_fingerprint(step.get("settings") or {})
    return str(step.get("operation") or ""), str(fingerprint)


def collect_report_payloads(reports_dir: Path) -> dict[str, Any]:
    """Merge operation report tables into one manifest section and remove them."""

    payloads: dict[str, Any] = {}
    for path in sorted(reports_dir.rglob("*.json")):
        if path.name in {DATASET_INFO_NAME, SOURCE_INFO_NAME}:
            continue
        key = path.relative_to(reports_dir).with_suffix("").as_posix().replace("/", ".")
        try:
            payloads[key] = _normalize_audit_payload(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError):
            payloads[key] = {"status": "unreadable"}
        path.unlink(missing_ok=True)
    for path in sorted(reports_dir.rglob("*.csv")):
        key = path.relative_to(reports_dir).with_suffix("").as_posix().replace("/", ".")
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except OSError:
            rows = []
        if key in payloads:
            payloads[key] = {"summary": payloads[key], "rows": rows}
        else:
            payloads[key] = rows
        path.unlink(missing_ok=True)
    return payloads


def _normalize_audit_payload(value: Any) -> Any:
    """Remove legacy before/after pairs from newly written dataset reports."""

    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name == "before" or name.endswith("_before"):
                continue
            if name == "after":
                continue
            if name.endswith("_after"):
                name = name.removesuffix("_after")
            output[name] = _normalize_audit_payload(item)
        if "after" in value:
            output["result"] = _normalize_audit_payload(value["after"])
        return output
    if isinstance(value, list):
        return [_normalize_audit_payload(item) for item in value]
    return value


def prune_report_directory(reports_dir: Path, *, extra_roots: Iterable[Path] = ()) -> None:
    """Reduce a generated ``reports/`` tree to the canonical artifact files.

    Operation audits already live inside ``dataset-info.json``; the transient
    preview images and nested legacy report directories that produced them are
    removed so a derived dataset never inherits another dataset's history.
    """

    for root in extra_roots:
        if root.is_dir():
            shutil.rmtree(root, ignore_errors=True)
    if not reports_dir.is_dir():
        return
    for path in sorted(reports_dir.iterdir(), reverse=True):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.name not in CANONICAL_REPORT_FILES:
            path.unlink(missing_ok=True)
