"""Render one readable current-state summary for a physical dataset.

The report describes what is on disk right now: identity, per-split annotated
and background composition, and a small deterministic example row per split.
It deliberately never embeds a previous ``plots.png`` or an operation preview,
so a dataset derived from a dataset does not accumulate nested history sheets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image, ImageDraw, ImageFont

from .models import DatasetMetadata, Task


REPORT_WIDTH = 2400
EXAMPLES_PER_SPLIT = 4

_BACKDROP = "#eef1f5"
_PANEL = "#ffffff"
_BORDER = "#d3d9e0"
_TEXT = "#111827"
_MUTED = "#5b6572"
_ANNOTATED = "#2f9e5f"
_BACKGROUND = "#b9c2cd"
_LETTERBOX = "#f4f6f9"
_MASK_OVERLAY = (255, 45, 45)
_ANNOTATION_COLORS = (
    "#ff00ff",
    "#7fff00",
    "#ff5f00",
    "#ffff00",
    "#ff1493",
)

_OUTER = 48
_INNER = 32
_GAP = 24
_COVERAGE_DETAIL_HEIGHT = 260


def render_dataset_report(
    root: Path,
    records: Iterable[Mapping[str, Any]],
    *,
    name: str,
    task: str,
    format_name: str,
    classes: Mapping[int, str],
    output: Path,
    metadata: DatasetMetadata | None = None,
    coverage: Mapping[str, Any] | None = None,
    width: int = REPORT_WIDTH,
) -> Path | None:
    """Draw ``output`` from the images and labels physically present in ``root``.

    Parameters:
        coverage: Optional source-coverage statistic from an operation that can
            leave part of its source behind, rendered as its own panel.

    Returns the written path, or ``None`` when no split holds a readable image.
    """

    splits = _split_views(root, records)
    if not splits:
        return None

    content = width - 2 * (_OUTER + _INNER)
    cell_width = (content - (EXAMPLES_PER_SPLIT - 1) * _GAP) // EXAMPLES_PER_SPLIT
    cell_height = round(cell_width * 0.72)
    header = _render_header(width, name=name, task=task, format_name=format_name, classes=classes)
    sections = []
    if coverage:
        sections.append(_render_coverage(coverage, width=width))
    sections += [
        _render_split(
            view,
            width=width,
            cell_width=cell_width,
            cell_height=cell_height,
            task=task,
            format_name=format_name,
            classes=classes,
            metadata=metadata,
        )
        for view in splits
    ]

    height = _OUTER + header.height + sum(section.height + _GAP for section in sections) + _OUTER - _GAP
    canvas = Image.new("RGB", (width, height), _BACKDROP)
    canvas.paste(header, (0, _OUTER))
    y = _OUTER + header.height + _GAP
    for section in sections:
        canvas.paste(section, (0, y))
        y += section.height + _GAP
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)
    return output


class _SplitView:
    __slots__ = ("split", "total", "annotated", "background", "examples")

    def __init__(self, split: str, rows: list[dict[str, Any]]) -> None:
        self.split = split
        self.total = len(rows)
        self.annotated = sum(1 for row in rows if row["annotated"])
        self.background = self.total - self.annotated
        self.examples = _deterministic_examples(rows)


def _split_views(root: Path, records: Iterable[Mapping[str, Any]]) -> list[_SplitView]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        relative = str(record.get("output_image") or "")
        if not relative:
            continue
        image_path = root / relative
        if not image_path.is_file():
            continue
        annotated = record.get("output_has_labels")
        if annotated is None:
            annotated = int(record.get("output_annotation_count") or 0) > 0
        mask = record.get("output_mask")
        grouped.setdefault(str(record.get("output_split") or "unknown"), []).append(
            {
                "relative": relative,
                "image_path": image_path,
                "mask_path": (root / str(mask)) if mask else None,
                "annotated": bool(annotated),
            }
        )
    order = ("train", "val", "test")
    ranked = sorted(
        grouped,
        key=lambda split: (order.index(split) if split in order else len(order), split),
    )
    return [_SplitView(split, grouped[split]) for split in ranked if grouped[split]]


def _deterministic_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer annotated images in stable path order, then fill from the rest."""

    ordered = sorted(rows, key=lambda row: row["relative"])
    chosen = [row for row in ordered if row["annotated"]][:EXAMPLES_PER_SPLIT]
    if len(chosen) < EXAMPLES_PER_SPLIT:
        chosen += [row for row in ordered if not row["annotated"]][
            : EXAMPLES_PER_SPLIT - len(chosen)
        ]
    return chosen


def _render_header(
    width: int,
    *,
    name: str,
    task: str,
    format_name: str,
    classes: Mapping[int, str],
) -> Image.Image:
    title_font = _font(52)
    body_font = _font(30)
    class_text = ", ".join(
        f"{index}: {classes[index]}" for index in sorted(classes)
    ) or "no classes declared"
    class_lines = _wrap(class_text, body_font, width - 2 * (_OUTER + _INNER))
    height = 40 + 62 + 44 + len(class_lines) * 38 + 32
    panel = Image.new("RGB", (width, height), _BACKDROP)
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle(
        (_OUTER, 0, width - _OUTER, height - 1),
        radius=18,
        fill=_PANEL,
        outline=_BORDER,
        width=2,
    )
    left = _OUTER + _INNER
    draw.text(
        (left, 34),
        _fit(name, title_font, width - 2 * (_OUTER + _INNER)),
        fill=_TEXT,
        font=title_font,
    )
    draw.text(
        (left, 34 + 62),
        f"task: {task}   ·   format: {format_name}   ·   {len(classes)} class(es)",
        fill=_MUTED,
        font=body_font,
    )
    y = 34 + 62 + 44
    for line in class_lines:
        draw.text((left, y), line, fill=_MUTED, font=body_font)
        y += 38
    return panel


def _render_coverage(coverage: Mapping[str, Any], *, width: int) -> Image.Image:
    """Draw how much of the source dataset reached this dataset."""

    title_font = _font(38)
    body_font = _font(28)
    caption_font = _font(22)
    bar_height = 46
    caption_gap = 38
    row_gap = 26
    height = (
        26 + 46 + row_gap
        + caption_gap + bar_height
        + row_gap
        + caption_gap + bar_height
        + row_gap + 8 + _COVERAGE_DETAIL_HEIGHT
        + 30
    )
    panel = Image.new("RGB", (width, height), _BACKDROP)
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle(
        (_OUTER, 0, width - _OUTER, height - 1),
        radius=18,
        fill=_PANEL,
        outline=_BORDER,
        width=2,
    )
    left = _OUTER + _INNER
    right = width - _OUTER - _INNER
    draw.text((left, 26), "source coverage", fill=_TEXT, font=title_font)

    labels_total = int(coverage.get("source_labels") or 0)
    labels_hit = int(coverage.get("source_labels_covered_at_least_once") or 0)
    never = int(coverage.get("source_labels_never_covered") or 0)
    label_pct = float(coverage.get("source_label_coverage_percent") or 0.0)
    space_pct = float(coverage.get("source_image_space_coverage_percent") or 0.0)

    y = 26 + 46 + row_gap
    draw.text(
        (left, y),
        f"{labels_hit:,} of {labels_total:,} source labels represented at least once"
        + (f"   ·   {never:,} never covered" if never else "   ·   none lost"),
        fill="#b00020" if never else _MUTED,
        font=body_font,
    )
    y += caption_gap
    _draw_ratio_bar(
        draw,
        (left, y, right, y + bar_height),
        fraction=label_pct / 100.0,
        filled_label=f"labels covered {label_pct:.1f}%",
        empty_label=f"never covered {100 - label_pct:.1f}%",
        font=caption_font,
        fill=_ANNOTATED if not never else "#d98324",
    )

    y += bar_height + row_gap
    draw.text(
        (left, y),
        f"{space_pct:.1f}% of the source image area is covered by output tiles",
        fill=_MUTED,
        font=body_font,
    )
    y += caption_gap
    _draw_ratio_bar(
        draw,
        (left, y, right, y + bar_height),
        fraction=space_pct / 100.0,
        filled_label=f"image space covered {space_pct:.1f}%",
        empty_label=f"not sampled {100 - space_pct:.1f}%",
        font=caption_font,
        fill="#2f6fb0",
    )

    y += bar_height + row_gap + 8
    column = (right - left - _GAP) // 2
    _draw_split_distribution(
        draw,
        (left, y, left + column, y + _COVERAGE_DETAIL_HEIGHT),
        coverage,
        body_font=body_font,
        caption_font=caption_font,
    )
    _draw_position_heatmap(
        panel,
        draw,
        (right - column, y, right, y + _COVERAGE_DETAIL_HEIGHT),
        coverage.get("label_positions"),
        body_font=body_font,
        caption_font=caption_font,
    )
    return panel


def _draw_split_distribution(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    coverage: Mapping[str, Any],
    *,
    body_font: ImageFont.ImageFont,
    caption_font: ImageFont.ImageFont,
) -> None:
    """Show where the source labels ended up, including the ones that did not."""

    left, top, right, bottom = box
    draw.text((left, top), "source labels by destination split", fill=_TEXT, font=body_font)
    splits = coverage.get("splits") or {}
    segments = [
        (split, int((value or {}).get("source_labels_covered_at_least_once") or 0))
        for split, value in sorted(splits.items())
    ]
    never = int(coverage.get("source_labels_never_covered") or 0)
    if never:
        segments.append(("never covered", never))
    total = sum(count for _, count in segments)
    bar_top = top + 44
    bar_bottom = bar_top + 40
    if total <= 0:
        draw.rounded_rectangle((left, bar_top, right, bar_bottom), radius=8, fill=_BACKGROUND)
        return
    palette = {"train": "#2f9e5f", "val": "#2f6fb0", "test": "#8a5cd6"}
    x = left
    for index, (name, count) in enumerate(segments):
        width = (
            right - x
            if index == len(segments) - 1
            else max(2, round((right - left) * count / total))
        )
        color = "#b00020" if name == "never covered" else palette.get(name, "#7a8595")
        draw.rounded_rectangle((x, bar_top, x + width, bar_bottom), radius=8, fill=color)
        _bar_label(
            draw,
            f"{name} {count:,}",
            (x, bar_top, x + width, bar_bottom),
            font=caption_font,
            fill="#ffffff",
        )
        x += width
    legend = "   ·   ".join(
        f"{name}: {count:,} ({100.0 * count / total:.1f}%)" for name, count in segments
    )
    draw.text(
        (left, bar_bottom + 12),
        _fit(legend, caption_font, right - left),
        fill=_MUTED,
        font=caption_font,
    )

    images = int(coverage.get("source_images") or 0)
    represented = int(coverage.get("source_images_represented") or 0)
    lines = [
        f"source images represented: {represented:,} of {images:,} "
        f"({float(coverage.get('source_image_representation_percent') or 0.0):.1f}%)"
    ]
    missing_area = int(coverage.get("source_images_without_exact_area") or 0)
    if missing_area:
        lines.append(
            f"{missing_area:,} source image(s) have no exact area measurement"
        )
    y = bar_bottom + 52
    for line in lines:
        draw.text((left, y), _fit(line, caption_font, right - left), fill=_MUTED, font=caption_font)
        y += 30


def _draw_position_heatmap(
    panel: Image.Image,
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    histogram: Mapping[str, Any] | None,
    *,
    body_font: ImageFont.ImageFont,
    caption_font: ImageFont.ImageFont,
) -> None:
    """Show where labels sit in the source frame, flagging uncovered cells."""

    left, top, right, bottom = box
    draw.text(
        (left, top),
        "source label positions",
        fill=_TEXT,
        font=body_font,
    )
    if not histogram:
        return
    grid = histogram.get("labels") or []
    uncovered = histogram.get("uncovered") or []
    rows = len(grid)
    columns = len(grid[0]) if rows else 0
    if not rows or not columns:
        return
    peak = max((max(line) for line in grid), default=0)
    map_top = top + 44
    map_bottom = bottom - 26
    cell_width = (right - left) / columns
    cell_height = (map_bottom - map_top) / rows
    for row_index in range(rows):
        for column_index in range(columns):
            count = grid[row_index][column_index]
            missing = uncovered[row_index][column_index] if uncovered else 0
            x0 = left + column_index * cell_width
            y0 = map_top + row_index * cell_height
            cell = (x0, y0, x0 + cell_width - 1, y0 + cell_height - 1)
            if missing:
                colour = "#b00020"
            elif count:
                weight = count / peak if peak else 0.0
                # Light to saturated blue, so density reads without a legend.
                colour = (
                    round(226 - 179 * weight),
                    round(236 - 125 * weight),
                    round(247 - 71 * weight),
                )
            else:
                colour = "#f1f4f8"
            draw.rectangle(cell, fill=colour)
    draw.rectangle((left, map_top, right, map_bottom), outline=_BORDER, width=1)
    draw.text(
        (left, map_bottom + 6),
        f"densest cell: {peak:,} label(s)"
        + ("   ·   red cells were never covered" if any(any(line) for line in uncovered) else ""),
        fill=_MUTED,
        font=caption_font,
    )


def _draw_ratio_bar(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fraction: float,
    filled_label: str,
    empty_label: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    left, top, right, bottom = box
    span = right - left
    fraction = max(0.0, min(1.0, fraction))
    draw.rounded_rectangle(box, radius=8, fill=_BACKGROUND)
    if fraction > 0:
        filled = left + max(2, round(span * fraction))
        draw.rounded_rectangle((left, top, filled, bottom), radius=8, fill=fill)
        _bar_label(draw, filled_label, (left, top, filled, bottom), font=font, fill="#ffffff")
    if fraction < 1:
        start = left + round(span * fraction)
        _bar_label(draw, empty_label, (start, top, right, bottom), font=font, fill="#33404f")


def _render_split(
    view: _SplitView,
    *,
    width: int,
    cell_width: int,
    cell_height: int,
    task: str,
    format_name: str,
    classes: Mapping[int, str],
    metadata: DatasetMetadata | None,
) -> Image.Image:
    title_font = _font(38)
    body_font = _font(28)
    caption_font = _font(22)

    bar_height = 46
    caption_height = 34
    height = (
        32
        + 46
        + 40
        + bar_height
        + 34
        + cell_height
        + caption_height
        + 28
    )
    panel = Image.new("RGB", (width, height), _BACKDROP)
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle(
        (_OUTER, 0, width - _OUTER, height - 1),
        radius=18,
        fill=_PANEL,
        outline=_BORDER,
        width=2,
    )
    left = _OUTER + _INNER
    right = width - _OUTER - _INNER
    draw.text((left, 26), f"{view.split}", fill=_TEXT, font=title_font)
    draw.text(
        (left, 26 + 46),
        (
            f"{view.total:,} images   ·   "
            f"{view.annotated:,} annotated ({_percent(view.annotated, view.total)})   ·   "
            f"{view.background:,} background ({_percent(view.background, view.total)})"
        ),
        fill=_MUTED,
        font=body_font,
    )

    bar_top = 26 + 46 + 40
    _draw_composition_bar(
        draw,
        (left, bar_top, right, bar_top + bar_height),
        annotated=view.annotated,
        total=view.total,
        font=caption_font,
    )

    row_top = bar_top + bar_height + 34
    for index in range(EXAMPLES_PER_SPLIT):
        cell_left = left + index * (cell_width + _GAP)
        if index >= len(view.examples):
            draw.rounded_rectangle(
                (cell_left, row_top, cell_left + cell_width, row_top + cell_height),
                radius=10,
                fill=_LETTERBOX,
                outline=_BORDER,
                width=1,
            )
            continue
        example = view.examples[index]
        thumbnail = _example_thumbnail(
            example,
            size=(cell_width, cell_height),
            task=task,
            format_name=format_name,
            metadata=metadata,
        )
        panel.paste(thumbnail, (cell_left, row_top))
        draw.rectangle(
            (cell_left, row_top, cell_left + cell_width, row_top + cell_height),
            outline=_BORDER,
            width=1,
        )
        label = "annotated" if example["annotated"] else "background"
        draw.text(
            (cell_left, row_top + cell_height + 8),
            _fit_middle(
                f"{example['relative']}  ·  {label}",
                caption_font,
                cell_width,
                keep_suffix=len(label) + 5,
            ),
            fill=_MUTED,
            font=caption_font,
        )
    return panel


def _draw_composition_bar(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    annotated: int,
    total: int,
    font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = box
    span = right - left
    draw.rounded_rectangle(box, radius=8, fill=_BACKGROUND)
    if total > 0 and annotated > 0:
        filled = left + max(2, round(span * annotated / total))
        draw.rounded_rectangle((left, top, filled, bottom), radius=8, fill=_ANNOTATED)
        _bar_label(
            draw,
            f"annotated {_percent(annotated, total)}",
            (left, top, filled, bottom),
            font=font,
            fill="#ffffff",
        )
    if annotated < total:
        start = left + round(span * annotated / total)
        _bar_label(
            draw,
            f"background {_percent(total - annotated, total)}",
            (start, top, right, bottom),
            font=font,
            fill="#33404f",
        )


def _bar_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    *,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    left, top, right, bottom = box
    text_width = _text_width(text, font)
    if right - left < text_width + 24:
        return
    draw.text(
        (left + 12, top + (bottom - top - _text_height(text, font)) // 2),
        text,
        fill=fill,
        font=font,
    )


def _example_thumbnail(
    example: Mapping[str, Any],
    *,
    size: tuple[int, int],
    task: str,
    format_name: str,
    metadata: DatasetMetadata | None,
) -> Image.Image:
    cell_width, cell_height = size
    canvas = Image.new("RGB", size, _LETTERBOX)
    try:
        with Image.open(example["image_path"]) as opened:
            image = opened.convert("RGB")
            source_size = image.size
    except OSError:
        return canvas
    scale = min(cell_width / max(1, source_size[0]), cell_height / max(1, source_size[1]))
    target = (
        max(1, round(source_size[0] * scale)),
        max(1, round(source_size[1] * scale)),
    )
    image = image.resize(target, Image.Resampling.LANCZOS)
    if format_name == "semantic_masks":
        image = _apply_mask_overlay(image, example.get("mask_path"), target)
    else:
        _draw_annotations(
            image,
            example["image_path"],
            source_size=source_size,
            scale=scale,
            task=task,
            metadata=metadata,
        )
    canvas.paste(image, ((cell_width - target[0]) // 2, (cell_height - target[1]) // 2))
    return canvas


def _apply_mask_overlay(
    image: Image.Image,
    mask_path: Any,
    target: tuple[int, int],
) -> Image.Image:
    if not mask_path or not Path(mask_path).is_file():
        return image
    try:
        with Image.open(mask_path) as opened:
            mask = opened.convert("L").resize(target, Image.Resampling.NEAREST)
    except OSError:
        return image
    tinted = Image.composite(Image.new("RGB", target, _MASK_OVERLAY), image, mask)
    return Image.blend(image, tinted, 0.55)


def _draw_annotations(
    image: Image.Image,
    image_path: Path,
    *,
    source_size: tuple[int, int],
    scale: float,
    task: str,
    metadata: DatasetMetadata | None,
) -> None:
    annotations = _read_annotations(
        image_path,
        source_size=source_size,
        task=task,
        metadata=metadata,
    )
    if not annotations:
        return
    draw = ImageDraw.Draw(image)
    for annotation in annotations:
        color = _ANNOTATION_COLORS[annotation.class_id % len(_ANNOTATION_COLORS)]
        if annotation.polygon and len(annotation.polygon) >= 3:
            points = [(x * scale, y * scale) for x, y in annotation.polygon]
            draw.line([*points, points[0]], fill="#000000", width=5, joint="curve")
            draw.line([*points, points[0]], fill=color, width=2, joint="curve")
        elif annotation.bbox is not None:
            x1, y1, x2, y2 = (value * scale for value in annotation.bbox)
            draw.rectangle((x1, y1, x2, y2), outline="#000000", width=5)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        if annotation.point is not None:
            x, y = annotation.point[0] * scale, annotation.point[1] * scale
            radius = max(3.0, (annotation.radius or 0.0) * scale)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="#000000", width=5)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
        for keypoint in annotation.keypoints or ():
            x, y, visibility = keypoint
            if visibility == 0:
                continue
            px, py = x * scale, y * scale
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color, outline="#000000", width=2)


def _read_annotations(
    image_path: Path,
    *,
    source_size: tuple[int, int],
    task: str,
    metadata: DatasetMetadata | None,
) -> list[Any]:
    label_path = _label_path(image_path)
    if label_path is None or not label_path.is_file():
        return []
    try:
        parsed_task = Task.parse(task)
    except Exception:
        return []
    if parsed_task is None:
        return []
    from .io import _parse_yolo_line

    effective = metadata or DatasetMetadata(names={})
    annotations: list[Any] = []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            annotations.append(
                _parse_yolo_line(line, parsed_task, effective, *source_size)
            )
        except (ValueError, TypeError):
            continue
    return annotations


def _label_path(image_path: Path) -> Path | None:
    parts = list(image_path.parts)
    indices = [index for index, part in enumerate(parts) if part == "images"]
    if not indices:
        return None
    parts[indices[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def _percent(value: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{100.0 * value / total:.1f}%"


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _text_width(text: str, font: ImageFont.ImageFont) -> int:
    left, _, right, _ = font.getbbox(text)
    return int(right - left)


def _text_height(text: str, font: ImageFont.ImageFont) -> int:
    _, top, _, bottom = font.getbbox(text)
    return int(bottom - top)


def _wrap(text: str, font: ImageFont.ImageFont, maximum: int) -> list[str]:
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and _text_width(candidate, font) > maximum:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines[:3] or [""]


def _fit(value: str, font: ImageFont.ImageFont, maximum: int) -> str:
    """Trim the tail of ``value`` until it is drawn no wider than ``maximum``."""

    if _text_width(value, font) <= maximum:
        return value
    trimmed = value
    while trimmed and _text_width(trimmed + "…", font) > maximum:
        trimmed = trimmed[:-1]
    return trimmed + "…"


def _fit_middle(
    value: str,
    font: ImageFont.ImageFont,
    maximum: int,
    *,
    keep_suffix: int,
) -> str:
    """Elide the middle of ``value`` so it fits, keeping its final characters.

    Captions end in the detail that distinguishes them, so the tail is
    preserved and the long common prefix is what gives way.
    """

    if _text_width(value, font) <= maximum:
        return value
    suffix = value[-keep_suffix:] if keep_suffix < len(value) else value
    head = value[: len(value) - len(suffix)]
    while head and _text_width(f"{head}…{suffix}", font) > maximum:
        head = head[:-1]
    if not head:
        return _fit(suffix, font, maximum)
    return f"{head}…{suffix}"
