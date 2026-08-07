from __future__ import annotations

import csv
import gzip
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageChops, ImageDraw, ImageFont

from .utils import settings_fingerprint, to_jsonable


REPORTS_DIR = "reports"
DATASET_INFO_NAME = "dataset-info.json"
SOURCE_INFO_NAME = "source.json"
LINEAGE_NAME = "lineage.json.gz"
DATASET_INFO_SCHEMA = 3
LINEAGE_SCHEMA = 2


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
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "dataset-fixer-lineage",
        "schema_version": LINEAGE_SCHEMA,
        "records": [to_jsonable(record) for record in records],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    temporary.replace(path)


def read_lineage(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "dataset-fixer-lineage":
        raise ValueError(f"Unsupported lineage document: {path}")
    if int(payload.get("schema_version", -1)) != LINEAGE_SCHEMA:
        raise ValueError(
            f"Unsupported lineage schema {payload.get('schema_version')!r}; "
            f"expected {LINEAGE_SCHEMA}"
        )
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Lineage records must be a list: {path}")
    return [dict(record) for record in records if isinstance(record, dict)]


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


def combine_report_plots(
    roots: Iterable[Path],
    output: Path,
    *,
    limit: int = 16,
) -> Path | None:
    """Collapse direct report images into one readable, vertically stacked sheet.

    Each panel is cropped to its visible content and scaled independently.  The
    compositor deliberately does not include an existing aggregate report: doing
    so would recursively embed ``plots.png`` in every derived dataset.
    """

    candidates: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        candidates.extend(
            path
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and path != output
            and path.name not in {"plots.png", "comparison.png", "source-operation.png"}
            and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
    if not candidates:
        return None
    chosen = candidates[:limit]
    canvas_width = 1600
    outer_padding = 32
    panel_padding = 24
    label_height = 52
    content_width = canvas_width - 2 * (outer_padding + panel_padding)
    prepared: list[tuple[Path, Image.Image]] = []
    for path in chosen:
        with Image.open(path) as opened:
            image = _crop_plot_whitespace(opened.convert("RGB"))
        scale = min(content_width / max(1, image.width), 4.0)
        if abs(scale - 1.0) > 1e-6:
            image = image.resize(
                (
                    max(1, round(image.width * scale)),
                    max(1, round(image.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )
        prepared.append((path, image))

    panel_heights = [label_height + image.height + 2 * panel_padding for _, image in prepared]
    canvas_height = outer_padding + sum(panel_heights) + outer_padding * len(prepared)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "#f3f4f6")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=28)
    except TypeError:  # Pillow 10.0
        font = ImageFont.load_default()
    y = outer_padding
    for (path, image), panel_height in zip(prepared, panel_heights):
        panel_left = outer_padding
        panel_right = canvas_width - outer_padding
        draw.rounded_rectangle(
            (panel_left, y, panel_right, y + panel_height),
            radius=14,
            fill="white",
            outline="#d1d5db",
            width=2,
        )
        title = path.stem.replace("_", " ")[:90]
        draw.text(
            (panel_left + panel_padding, y + 16),
            title,
            fill="#111827",
            font=font,
        )
        image_x = (canvas_width - image.width) // 2
        image_y = y + label_height + panel_padding
        canvas.paste(image, (image_x, image_y))
        y += panel_height + outer_padding
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    for path in candidates:
        path.unlink(missing_ok=True)
    for root in roots:
        if root != output.parent and root.is_dir():
            shutil.rmtree(root, ignore_errors=True)
    return output


def _crop_plot_whitespace(image: Image.Image, *, padding: int = 12) -> Image.Image:
    """Trim near-white outer margins without cutting into plotted content."""

    background = Image.new("RGB", image.size, "white")
    difference = ImageChops.difference(image, background).convert("L")
    visible = difference.point(lambda value: 255 if value > 8 else 0)
    bounds = visible.getbbox()
    if bounds is None:
        return image
    left, top, right, bottom = bounds
    return image.crop(
        (
            max(0, left - padding),
            max(0, top - padding),
            min(image.width, right + padding),
            min(image.height, bottom + padding),
        )
    )
