from __future__ import annotations

import math
import random
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence, TypeVar

import cv2
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps
from shapely.geometry import Polygon
from shapely.validation import explain_validity

from .models import Annotation, DatasetMetadata, Sample, Task
from .static_rendering import (
    LabelMode,
    card,
    concat_grid,
    display_image,
    finish_chart,
    format_label,
    save_chart,
    text_region,
)

if TYPE_CHECKING:
    from .validation_audit import ValidationFailureExample

SPLIT_COLORS = {"train": "#2ca02c", "val": "#ff7f0e", "test": "#1f77b4"}
ANNOTATION_COLORS = (
    "#ff00ff",  # magenta
    "#7fff00",  # chartreuse
    "#ff5f00",  # vivid orange
    "#ffff00",  # yellow
    "#ff1493",  # deep pink
)

_VISUALIZE_KWARGS = {"label_fn", "label_mode", "line_width", "outline_width"}

COMMON_VISUALIZE_PARAMETERS = frozenset(
    "samples columns seed panel_size zoom context_fraction minimum_context "
    "label_fn label_mode line_width outline_width outline_alpha destination show".split()
)


@dataclass(frozen=True)
class VisualizationOptions:
    samples: int | None = None
    columns: int = 1
    seed: int = 42
    panel_size: float = 3.0
    zoom: bool = False
    context_fraction: float = 0.35
    minimum_context: int = 100
    label_fn: Callable[[Path], str | None] | None = None
    label_mode: LabelMode = "middle"
    line_width: float | None = None
    outline_width: float = 1.0
    outline_alpha: float = 1.0
    destination: Path | None = None
    show: bool = True


@dataclass(frozen=True)
class VisualizationPanel:
    title: str
    image: np.ndarray
    mask: np.ndarray | None = None
    color: str = "#C86552"
    footer: str | None = None


@dataclass(frozen=True)
class VisualizationItem:
    image_path: Path
    label: str
    panels: tuple[VisualizationPanel, ...]
    foreground: np.ndarray


_VisualRecord = TypeVar("_VisualRecord")


def visualize_records(
    records: Sequence[_VisualRecord],
    *,
    options: VisualizationOptions,
    prepare: Callable[[_VisualRecord], VisualizationItem],
    title: str | Callable[[int], str] | None = None,
) -> Any:
    if not records:
        raise ValueError("At least one image is required for visualization")
    count = len(records) if options.samples is None else min(options.samples, len(records))
    if count == len(records):
        selected = list(records)
    else:
        selected = random.Random(options.seed).sample(list(records), count)
    items = [prepare(record) for record in selected]
    panel_count = len(items[0].panels)
    if panel_count == 0 or any(len(item.panels) != panel_count for item in items):
        raise ValueError("Every visualization item must have the same non-zero panel count")

    prepared: list[tuple[VisualizationItem, tuple[int, int, int, int]]] = []
    for item in items:
        foreground = np.asarray(item.foreground, dtype=bool)
        height, width = foreground.shape
        bounds = (
            foreground_crop_bounds(
                foreground,
                context_fraction=options.context_fraction,
                minimum_context=options.minimum_context,
            )
            if options.zoom and np.any(foreground)
            else (0, 0, width, height)
        )
        prepared.append((item, bounds))

    import altair as alt

    viewport = max(144, round(options.panel_size * 96))
    item_charts: list[Any] = []
    for item, bounds in prepared:
        x0, y0, x1, y1 = bounds
        label = (
            options.label_fn(item.image_path)
            if options.label_fn is not None
            else item.label
        )
        if label is not None and not isinstance(label, str):
            raise TypeError("label_fn must return a string or None")
        panel_charts: list[Any] = []
        for panel in item.panels:
            image = np.asarray(panel.image)[y0:y1, x0:x1]
            mask_crop = (
                None
                if panel.mask is None
                else np.asarray(panel.mask, dtype=bool)[y0:y1, x0:x1]
            )
            if mask_crop is not None:
                image = draw_mask_outline(
                    image,
                    mask_crop,
                    color=panel.color,
                    line_width=options.line_width,
                    outline_width=options.outline_width,
                    alpha=options.outline_alpha,
                )
            panel_charts.append(
                card(
                    image,
                    width=viewport,
                    height=viewport,
                    heading=format_label(
                        panel.title,
                        mode=options.label_mode,
                        maximum=max(18, viewport // 7),
                    ),
                    footer=format_label(
                        panel.footer or "",
                        mode=options.label_mode,
                        maximum=max(18, viewport // 7),
                    ),
                )
            )
        panel_row = alt.hconcat(*panel_charts, spacing=8)
        label_lines = format_label(
            label or "",
            mode=options.label_mode,
            maximum=max(24, (viewport * panel_count) // 7),
        )
        regions: list[Any] = []
        if label_lines:
            regions.append(
                text_region(
                    label_lines,
                    width=viewport * panel_count + 8 * (panel_count - 1),
                    font_size=12,
                )
            )
        regions.append(panel_row)
        item_charts.append(alt.vconcat(*regions, spacing=5))
    resolved_title = title(len(items)) if callable(title) else title
    return concat_grid(
        item_charts,
        columns=options.columns,
        spacing=20,
        title=resolved_title,
    )


def visualization_options(**overrides: Any) -> VisualizationOptions:
    unknown = sorted(set(overrides) - COMMON_VISUALIZE_PARAMETERS)
    if unknown:
        raise TypeError("Unknown visualization options: " + ", ".join(unknown))
    defaults = VisualizationOptions()
    values = {
        name: getattr(defaults, name)
        for name in COMMON_VISUALIZE_PARAMETERS
    }
    values.update(overrides)
    for name in ("columns", "minimum_context"):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    samples = values["samples"]
    if samples is not None and (
        isinstance(samples, bool) or not isinstance(samples, int) or samples < 1
    ):
        raise ValueError("samples must be a positive integer or None")
    seed = values["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not isinstance(values["zoom"], bool) or not isinstance(values["show"], bool):
        raise TypeError("zoom and show must be booleans")
    if values["label_mode"] not in {"middle", "wrap"}:
        raise ValueError("label_mode must be 'middle' or 'wrap'")
    label_fn = values["label_fn"]
    if label_fn is not None and not callable(label_fn):
        raise TypeError("label_fn must be callable or None")
    numeric = {
        "panel_size": (0.0, None),
        "context_fraction": (0.0, None),
        "outline_alpha": (0.0, 1.0),
        "line_width": (0.0, None),
        "outline_width": (0.0, None),
    }
    for name, (minimum, maximum) in numeric.items():
        value = values[name]
        if value is None and name in {"line_width", "outline_width"}:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < minimum
            or (
                name in {"panel_size", "line_width", "outline_width"}
                and float(value) == 0
            )
            or (maximum is not None and float(value) > maximum)
        ):
            raise ValueError(f"{name} must be a valid finite value")
        values[name] = float(value)
    values["outline_width"] = values["outline_width"] or 1.0
    values["line_width"] = values["line_width"] or values["outline_width"]
    destination = values["destination"]
    values["destination"] = (
        Path(destination).expanduser().resolve() if destination is not None else None
    )
    return VisualizationOptions(**values)


def foreground_crop_bounds(
    foreground: np.ndarray,
    *,
    context_fraction: float,
    minimum_context: int,
) -> tuple[int, int, int, int]:
    mask = np.asarray(foreground, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("visualization foreground must be a 2D mask")
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0, width, height
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    crop_width = min(
        width,
        max(minimum_context, math.ceil((right - left) * (1 + 2 * context_fraction))),
    )
    crop_height = min(
        height,
        max(minimum_context, math.ceil((bottom - top) * (1 + 2 * context_fraction))),
    )
    x0 = max(0, min(width - crop_width, round((left + right - crop_width) / 2)))
    y0 = max(0, min(height - crop_height, round((top + bottom - crop_height) / 2)))
    return x0, y0, x0 + crop_width, y0 + crop_height


def draw_mask_outline(
    image: Image.Image | np.ndarray,
    mask: np.ndarray,
    *,
    color: str,
    line_width: float | None,
    outline_width: float,
    alpha: float,
) -> np.ndarray:
    """Burn a double-stroked boundary into a copy of an RGB image."""

    base = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
    if base.ndim == 2:
        base = np.repeat(base[..., None], 3, axis=2)
    base = np.ascontiguousarray(base[..., :3], dtype=np.uint8).copy()
    values = np.asarray(mask, dtype=bool)
    if not np.any(values) or alpha <= 0:
        return base
    if values.shape != base.shape[:2]:
        raise ValueError("mask and image dimensions must match")
    contours, _ = cv2.findContours(
        values.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return base
    resolved_line_width = max(1, round(outline_width if line_width is None else line_width))
    halo_width = max(resolved_line_width + 2, round(outline_width))
    overlay = base.copy()
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), halo_width)
    cv2.drawContours(
        overlay,
        contours,
        -1,
        ImageColor.getrgb(color),
        resolved_line_width,
    )
    if alpha >= 1:
        return overlay
    return np.uint8(np.round(base * (1 - alpha) + overlay * alpha))


def finish_visualization(chart: Any, options: VisualizationOptions) -> None:
    finish_chart(
        chart,
        destination=options.destination,
        show=options.show,
        overwrite=False,
    )


def normalize_visualize_kwargs(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate options shared by direct, operation, and report visualization."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("visualize_kwargs must be a mapping or None")
    unknown = sorted(set(value) - _VISUALIZE_KWARGS)
    if unknown:
        raise TypeError(
            "Unknown visualize_kwargs: "
            + ", ".join(unknown)
            + "; supported keys are "
            + ", ".join(sorted(_VISUALIZE_KWARGS))
        )
    normalized = dict(value)
    label_fn = normalized.get("label_fn")
    if label_fn is not None and not callable(label_fn):
        raise TypeError("visualize_kwargs['label_fn'] must be callable or None")
    label_mode = normalized.get("label_mode")
    if label_mode is not None and label_mode not in {"middle", "wrap"}:
        raise ValueError("visualize_kwargs['label_mode'] must be 'middle' or 'wrap'")
    for key in ("line_width", "outline_width"):
        width = normalized.get(key)
        if width is None:
            continue
        if (
            not isinstance(width, (int, float))
            or isinstance(width, bool)
            or not math.isfinite(float(width))
            or float(width) <= 0
        ):
            raise ValueError(f"visualize_kwargs[{key!r}] must be a positive number")
        normalized[key] = float(width)
    return normalized


def draw_label_position_heatmap(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    histogram: Mapping[str, Any],
    *,
    border: str = "#d3d9e0",
) -> tuple[int, bool, tuple[float, float, float, float]]:
    """Draw one density grid without distorting its row/column aspect ratio."""

    left, top, right, bottom = box
    grid = histogram.get("labels") or []
    uncovered = histogram.get("uncovered") or []
    rows = len(grid)
    columns = len(grid[0]) if rows else 0
    if not rows or not columns:
        return 0, False, (left, top, right, bottom)
    peak = max((max(line) for line in grid), default=0)
    available_width = right - left
    available_height = bottom - top
    grid_aspect = columns / rows
    if available_width / available_height > grid_aspect:
        map_width = available_height * grid_aspect
        map_height = available_height
    else:
        map_width = available_width
        map_height = available_width / grid_aspect
    map_left = left + (available_width - map_width) / 2
    map_top = top
    map_right = map_left + map_width
    map_bottom = map_top + map_height
    cell_width = map_width / columns
    cell_height = map_height / rows
    for row_index in range(rows):
        for column_index in range(columns):
            count = grid[row_index][column_index]
            missing = uncovered[row_index][column_index] if uncovered else 0
            x0 = map_left + column_index * cell_width
            y0 = map_top + row_index * cell_height
            cell = (x0, y0, x0 + cell_width - 1, y0 + cell_height - 1)
            if missing:
                colour: Any = "#b00020"
            elif count:
                weight = count / peak if peak else 0.0
                colour = (
                    round(226 - 179 * weight),
                    round(236 - 125 * weight),
                    round(247 - 71 * weight),
                )
            else:
                colour = "#f1f4f8"
            draw.rectangle(cell, fill=colour)
    draw.rectangle((map_left, map_top, map_right, map_bottom), outline=border, width=1)
    has_uncovered = bool(uncovered and any(any(line) for line in uncovered))
    return peak, has_uncovered, (map_left, map_top, map_right, map_bottom)


def save_label_position_summary(
    coverage: Mapping[str, Any],
    output: Path,
) -> Path:
    """Save the source/output position comparison used by the dataset report."""

    image = Image.new("RGB", (1400, 520), "white")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 28)
        caption_font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        title_font = ImageFont.load_default()
        caption_font = ImageFont.load_default()
    panels = (
        ("source label positions", coverage.get("label_positions"), (40, 70, 850, 440)),
        ("output label positions", coverage.get("output_label_positions"), (900, 70, 1360, 440)),
    )
    for title, histogram, box in panels:
        draw.text((box[0], 24), title, fill="#111827", font=title_font)
        if not histogram:
            continue
        peak, has_uncovered, map_box = draw_label_position_heatmap(draw, box, histogram)
        caption = f"densest cell: {peak:,} label(s)"
        if has_uncovered:
            caption += " · red cells were never covered"
        draw.text((box[0], map_box[3] + 10), caption, fill="#5b6572", font=caption_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=95)
    return output


def visualize_validation_failures(
    examples: list["ValidationFailureExample"],
    task: Task,
    metadata: DatasetMetadata,
    *,
    total_count: int,
    dataset_name: str,
    save_to: Path,
    show: bool = True,
) -> None:
    """Render a bounded grid of load-time validation failures."""

    columns = 1 if len(examples) == 1 else 2

    def prepare(example: "ValidationFailureExample") -> VisualizationItem:
        annotations = (
            [example.annotation]
            if example.annotation is not None and example.annotation.polygon is None
            else []
        )
        if (
            example.image_path is not None
            and example.image_path.is_file()
            and example.width is not None
            and example.height is not None
        ):
            sample = Sample(
                image_path=example.image_path,
                relative_path=example.relative_path or Path(example.image_path.name),
                split=example.split or "unknown",
                width=example.width,
                height=example.height,
                annotations=annotations,
            )
            try:
                rendered = render_annotated_sample(sample, task, metadata)
                if example.annotation is not None:
                    rendered = _highlight_invalid_annotation(
                        rendered,
                        example.annotation,
                        width=example.width,
                        height=example.height,
                    )
            except Exception:
                rendered = _draw_failure_placeholder()
        else:
            rendered = _draw_failure_placeholder()
        source = (
            str(example.relative_path)
            if example.relative_path is not None
            else "no readable image available"
        )
        message = textwrap.fill(
            example.summary or example.warning,
            width=58,
            max_lines=2,
            placeholder=" …",
        )
        foreground = np.ones(np.asarray(rendered).shape[:2], dtype=bool)
        if example.annotation is not None and example.annotation.polygon:
            foreground = _annotation_focus_mask(
                example.annotation,
                width=rendered.width,
                height=rendered.height,
            )
        return VisualizationItem(
            image_path=example.image_path or Path("unavailable"),
            label=(
                f"Skipped · {example.split or 'unknown split'} · "
                f"{textwrap.shorten(source, width=78, placeholder='…')}\n{message}"
            ),
            panels=(VisualizationPanel(title="Validation failure", image=np.asarray(rendered)),),
            foreground=foreground,
        )

    options = VisualizationOptions(
        samples=None,
        columns=columns,
        panel_size=4.6,
        zoom=True,
        label_mode="wrap",
        destination=save_to,
        show=show,
    )
    chart = visualize_records(
        examples,
        options=options,
        prepare=prepare,
        title=(
            f"Load validation skips — {dataset_name} — {total_count} failed item(s); "
            f"showing {len(examples)}"
        ),
    )
    finish_chart(chart, destination=save_to, show=show, overwrite=True)


def _draw_failure_placeholder() -> Image.Image:
    image = Image.new("RGB", (720, 480), "#242830")
    draw = ImageDraw.Draw(image)
    message = "No readable image\navailable for this failure"
    draw.multiline_text(
        (image.width / 2, image.height / 2),
        message,
        fill="white",
        font=_font(24),
        anchor="mm",
        align="center",
    )
    return image


def _polygon_invalidity_details(
    points: list[tuple[float, float]],
    *,
    width: int,
    height: int,
) -> tuple[list[str], list[tuple[float, float, str]]]:
    """Return human-readable polygon defects and locations worth marking."""

    reasons: list[str] = []
    markers: list[tuple[float, float, str]] = []
    if len(points) < 3:
        reasons.append(f"only {len(points)} point(s); at least 3 required")

    finite_points = [
        (index, x, y)
        for index, (x, y) in enumerate(points)
        if math.isfinite(x) and math.isfinite(y)
    ]
    if len(finite_points) != len(points):
        reasons.append("non-finite vertex coordinate")

    for index, x, y in finite_points:
        if (
            x < -0.01 * width
            or y < -0.01 * height
            or x > 1.01 * width
            or y > 1.01 * height
        ):
            reasons.append(f"vertex {index} lies outside the image")
            markers.append(
                (
                    min(max(x, 0.0), float(width)),
                    min(max(y, 0.0), float(height)),
                    f"vertex {index} outside image\n({x:.1f}, {y:.1f})",
                )
            )

    if len(points) >= 3 and len(finite_points) == len(points):
        try:
            polygon = Polygon(points)
            validity = explain_validity(polygon)
            if polygon.is_empty:
                reasons.append("empty polygon")
            if polygon.area <= 0:
                reasons.append("zero-area polygon")
            if not polygon.is_valid:
                reason = validity.split("[", 1)[0].strip() or "invalid polygon"
                reasons.append(reason.lower())
                coordinate = re.search(r"\[\s*([^\]]+)\s*\]", validity)
                if coordinate:
                    values = coordinate.group(1).replace(",", " ").split()
                    if len(values) >= 2:
                        try:
                            x, y = float(values[0]), float(values[1])
                        except ValueError:
                            pass
                        else:
                            markers.append((x, y, reason.lower()))
        except Exception as error:
            reasons.append(f"geometry could not be constructed ({type(error).__name__})")

    return list(dict.fromkeys(reasons)), list(dict.fromkeys(markers))


def _highlight_invalid_annotation(
    image: Image.Image,
    annotation: Annotation,
    *,
    width: int,
    height: int,
) -> Image.Image:
    """Overlay ordered vertices and explicit defects on a rejected polygon."""

    if annotation.polygon is None:
        return image
    reasons, markers = _polygon_invalidity_details(
        annotation.polygon,
        width=width,
        height=height,
    )
    if not reasons:
        return image

    rendered = image.convert("RGB").copy()
    draw = ImageDraw.Draw(rendered)

    finite_polygon = [
        (x, y)
        for x, y in annotation.polygon
        if math.isfinite(x) and math.isfinite(y)
    ]
    if len(finite_polygon) >= 2:
        closed_polygon = finite_polygon + (
            [finite_polygon[0]] if len(finite_polygon) >= 3 else []
        )
        draw.line(closed_polygon, fill="#dc2626", width=4, joint="curve")
    if finite_polygon:
        for index, (x, y) in enumerate(finite_polygon):
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="white", outline="#dc2626", width=2)
            draw.text((x + 7, y + 7), str(index), fill="#111827", font=_font(13))
    for x, y, label in markers:
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), outline="#f59e0b", width=4)
        draw.text((x + 12, y + 12), label.replace("\n", " "), fill="#111827", font=_font(13), stroke_width=3, stroke_fill="white")
    message = "Invalid polygon: " + "; ".join(reasons)
    box = draw.textbbox((14, rendered.height - 28), message, font=_font(14), stroke_width=2)
    draw.rectangle((8, box[1] - 5, min(rendered.width - 8, box[2] + 8), rendered.height - 6), fill="#fff7ed", outline="#fca5a5", width=2)
    draw.text((14, rendered.height - 28), message, fill="#7f1d1d", font=_font(14))
    return rendered


def _focus_invalid_annotation(
    annotation: Annotation,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Zoom to the finite polygon vertices while retaining nearby image context."""

    if annotation.polygon is None:
        return (0, 0, width, height)
    points = [
        (float(x), float(y))
        for x, y in annotation.polygon
        if math.isfinite(x) and math.isfinite(y)
    ]
    if not points or width <= 0 or height <= 0:
        return (0, 0, width, height)

    clipped_x = [min(max(x, 0.0), float(width)) for x, _ in points]
    clipped_y = [min(max(y, 0.0), float(height)) for _, y in points]
    center_x = (min(clipped_x) + max(clipped_x)) / 2
    center_y = (min(clipped_y) + max(clipped_y)) / 2
    polygon_width = max(clipped_x) - min(clipped_x)
    polygon_height = max(clipped_y) - min(clipped_y)
    minimum_window = min(
        float(min(width, height)),
        max(48.0, min(256.0, 0.15 * min(width, height))),
    )
    window_width = min(float(width), max(minimum_window, polygon_width * 1.7))
    window_height = min(float(height), max(minimum_window, polygon_height * 1.7))

    def bounded_window(center: float, span: float, limit: float) -> tuple[float, float]:
        if span >= limit:
            return 0.0, limit
        lower = center - span / 2
        upper = center + span / 2
        if lower < 0:
            upper -= lower
            lower = 0.0
        if upper > limit:
            lower -= upper - limit
            upper = limit
        return max(0.0, lower), min(limit, upper)

    left, right = bounded_window(center_x, window_width, float(width))
    top, bottom = bounded_window(center_y, window_height, float(height))
    return round(left), round(top), round(right), round(bottom)


def _annotation_focus_mask(
    annotation: Annotation,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    mask = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    if annotation.polygon:
        points = [
            (min(max(float(x), 0.0), width - 1), min(max(float(y), 0.0), height - 1))
            for x, y in annotation.polygon
            if math.isfinite(x) and math.isfinite(y)
        ]
        if len(points) >= 2:
            draw.line(points, fill=1, width=3, joint="curve")
        elif points:
            x, y = points[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=1)
    return np.asarray(mask, dtype=bool)


def visualize_samples(
    samples: list[Sample],
    task: Task,
    metadata: DatasetMetadata,
    *,
    split: str | None,
    options: VisualizationOptions,
):
    candidates = [s for s in samples if split is None or s.split == split]
    title_split = split or "all splits"

    def prepare(sample: Sample) -> VisualizationItem:
        rendered = render_annotated_sample(
            sample,
            task,
            metadata,
            line_width=options.line_width,
            outline_width=options.outline_width,
        )
        image = np.asarray(rendered)
        if options.outline_alpha < 1:
            with Image.open(sample.image_path) as opened:
                source = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
            image = np.uint8(
                source * (1 - options.outline_alpha) + image * options.outline_alpha
            )
        return VisualizationItem(
            image_path=sample.image_path,
            label=_sample_title(
                sample,
                None,
                default=(
                    f"{sample.relative_path}\n{sample.width}×{sample.height} | "
                    f"{len(sample.annotations)} annotations | {sample.split}"
                ),
            )
            or "",
            panels=(VisualizationPanel(title="Annotation", image=image),),
            foreground=_sample_foreground_mask(sample, task),
        )

    figure = visualize_records(
        candidates,
        options=options,
        prepare=prepare,
        title=lambda count: f"Annotation check — {title_split} ({count} images)",
    )
    return finish_visualization(figure, options)


def visualize_semantic_masks(
    samples: list[Sample],
    mask_paths: dict[Path, Path],
    *,
    split: str | None,
    options: VisualizationOptions,
):
    """Render paired source images and binary-mask overlays."""

    candidates = [sample for sample in samples if split is None or sample.split == split]

    def prepare(sample: Sample) -> VisualizationItem:
        mask_path = mask_paths[sample.image_path.resolve()]
        with Image.open(sample.image_path) as opened_image, Image.open(mask_path) as opened_mask:
            image = np.asarray(opened_image.convert("RGB"))
            mask = opened_mask.convert("L")
        mask_array = np.asarray(mask) > 0
        foreground = sum(mask.histogram()[1:])
        fraction = foreground / (mask.width * mask.height) if mask.width and mask.height else 0.0
        return VisualizationItem(
            image_path=sample.image_path,
            label=_sample_title(
                sample,
                None,
                default=(
                    f"{sample.split} · {sample.relative_path}\n"
                    f"foreground: {fraction:.1%}"
                ),
            )
            or "",
            panels=(
                VisualizationPanel(
                    title="Annotation",
                    image=image,
                    mask=mask_array,
                    color="#ff2020",
                ),
            ),
            foreground=mask_array,
        )

    title_split = split or "all splits"
    figure = visualize_records(
        candidates,
        options=options,
        prepare=prepare,
        title=lambda count: f"Semantic-mask check — {title_split} ({count} images)",
    )
    return finish_visualization(figure, options)


def _sample_foreground_mask(sample: Sample, task: Task) -> np.ndarray:
    """Rasterize annotation geometry only for shared visualization cropping."""

    canvas = Image.new("1", (sample.width, sample.height), 0)
    draw = ImageDraw.Draw(canvas)
    for annotation in sample.annotations:
        if annotation.bbox is not None and task in {Task.DETECT, Task.POSE}:
            draw.rectangle(annotation.bbox, fill=1)
        if annotation.polygon:
            draw.polygon(annotation.polygon, fill=1)
        if annotation.point is not None:
            x, y = annotation.point
            radius = max(1.0, float(annotation.radius or 1.0))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=1)
        if annotation.keypoints:
            for x, y, visibility in annotation.keypoints:
                if visibility != 0:
                    draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=1)
    return np.asarray(canvas, dtype=bool)


def _sample_title(
    sample: Sample,
    label_fn: Callable[[Path], str | None] | None,
    *,
    default: str,
) -> str | None:
    if label_fn is not None:
        label = label_fn(sample.image_path)
    elif sample.provenance is not None and "display_label" in sample.provenance:
        label = sample.provenance["display_label"]
    else:
        return default
    if label is not None and not isinstance(label, str):
        raise TypeError("label_fn must return a string or None")
    return label


def render_annotated_sample(
    sample: Sample,
    task: Task,
    metadata: DatasetMetadata,
    *,
    resize_to: tuple[int, int] | None = None,
    show_names: bool = True,
    line_width: float | None = None,
    outline_width: float | None = None,
) -> Image.Image:
    """Render one sample for both public visualization and report previews."""

    with Image.open(sample.image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    if resize_to is not None and image.size != resize_to:
        image = image.resize(resize_to, Image.Resampling.LANCZOS)
    scale_x = image.width / max(1, sample.width)
    scale_y = image.height / max(1, sample.height)
    draw = ImageDraw.Draw(image)
    short_side = min(image.size)
    colour_width = (
        max(1, round(float(line_width)))
        if line_width is not None
        else max(2, round(max(3, short_side / 140) * 0.45))
    )
    resolved_outline_width = (
        max(1, round(float(outline_width)))
        if outline_width is not None
        else max(colour_width + 1, round(short_side / 140))
    )
    marker_radius = max(3, round(short_side / 120))
    font = ImageFont.load_default()

    for annotation in sample.annotations:
        color = ANNOTATION_COLORS[annotation.class_id % len(ANNOTATION_COLORS)]
        name = metadata.names.get(annotation.class_id, str(annotation.class_id))
        if annotation.bbox is not None and task in {Task.DETECT, Task.POSE}:
            x1, y1, x2, y2 = annotation.bbox
            box = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
            draw.rectangle(box, outline="white", width=resolved_outline_width)
            draw.rectangle(box, outline=color, width=colour_width)
            if show_names:
                text_box = draw.textbbox((box[0], box[1]), name, font=font)
                text_height = text_box[3] - text_box[1]
                text_width = text_box[2] - text_box[0]
                text_top = max(0, box[1] - text_height - 4)
                draw.rectangle(
                    (box[0], text_top, box[0] + text_width + 4, text_top + text_height + 4),
                    fill=color,
                )
                draw.text(
                    (box[0] + 2, text_top + 2),
                    name,
                    fill="white",
                    font=font,
                )
        if annotation.polygon:
            points = [(x * scale_x, y * scale_y) for x, y in annotation.polygon]
            if len(points) >= 2:
                closed = [*points, points[0]]
                draw.line(
                    closed,
                    fill="white",
                    width=resolved_outline_width,
                    joint="curve",
                )
                draw.line(
                    closed,
                    fill=color,
                    width=colour_width,
                    joint="curve",
                )
        if annotation.keypoints:
            names = metadata.kpt_names.get(annotation.class_id, [])
            skeleton_value = metadata.extra.get("skeleton", [])
            if isinstance(skeleton_value, dict):
                skeleton_value = skeleton_value.get(annotation.class_id, skeleton_value.get(str(annotation.class_id), []))
            for edge in skeleton_value or []:
                if len(edge) != 2:
                    continue
                # COCO and Ultralytics skeletons are conventionally 1-based.
                first, second = int(edge[0]) - 1, int(edge[1]) - 1
                if not (0 <= first < len(annotation.keypoints) and 0 <= second < len(annotation.keypoints)):
                    continue
                x1, y1, v1 = annotation.keypoints[first]
                x2, y2, v2 = annotation.keypoints[second]
                if v1 != 0 and v2 != 0:
                    draw.line(
                        (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y),
                        fill=color,
                        width=colour_width,
                    )
            for idx, (x, y, visibility) in enumerate(annotation.keypoints):
                if visibility == 0:
                    continue
                px, py = x * scale_x, y * scale_y
                draw.ellipse(
                    (px - marker_radius, py - marker_radius, px + marker_radius, py + marker_radius),
                    fill=color,
                    outline="white",
                    width=1,
                )
                if show_names and idx < len(names):
                    draw.text(
                        (px + marker_radius + 1, py + 1),
                        names[idx],
                        fill="white",
                        font=font,
                    )
        if annotation.point and annotation.radius is not None:
            x, y = annotation.point[0] * scale_x, annotation.point[1] * scale_y
            radius_x = annotation.radius * scale_x
            radius_y = annotation.radius * scale_y
            draw.ellipse(
                (x - radius_x, y - radius_y, x + radius_x, y + radius_y),
                outline="red",
                width=colour_width,
            )
            draw.ellipse(
                (x - marker_radius, y - marker_radius, x + marker_radius, y + marker_radius),
                fill="red",
            )
    return image


def save_split_preview(
    samples: list[Sample],
    assignments: dict[str, str],
    group_lookup: dict[str, str],
    task: Task,
    metadata: DatasetMetadata,
    output: Path,
) -> Path:
    splits = [split for split in ("train", "val", "test") if split in assignments.values()]
    chosen: list[tuple[str, Sample]] = []
    for split in splits:
        candidates = sorted(
            (
                sample
                for sample in samples
                if assignments.get(str(sample.image_path)) == split
            ),
            key=lambda sample: (
                -bool(sample.annotations),
                group_lookup.get(str(sample.image_path), ""),
                str(sample.relative_path),
            ),
        )
        seen_groups: set[str] = set()
        selected: list[Sample] = []
        for sample in candidates:
            group = group_lookup.get(str(sample.image_path), str(sample.image_path))
            if group in seen_groups:
                continue
            selected.append(sample)
            seen_groups.add(group)
            if len(selected) == 2:
                break
        if len(selected) < 2:
            remaining = [sample for sample in candidates if sample not in selected]
            selected.extend(remaining[: 2 - len(selected)])
        chosen.extend((split, sample) for sample in selected)

    def prepare(value: tuple[str, Sample]) -> VisualizationItem:
        target, sample = value
        group = group_lookup.get(str(sample.image_path), "ungrouped")
        if len(group) > 70:
            group = f"…{group[-69:]}"
        return VisualizationItem(
            image_path=sample.image_path,
            label=(
                f"{target.upper()} · group {group}\n"
                f"{sample.relative_path} · {len(sample.annotations)} annotations"
            ),
            panels=(
                VisualizationPanel(
                    title="Annotation",
                    image=np.asarray(render_annotated_sample(sample, task, metadata)),
                ),
            ),
            foreground=np.ones((sample.height, sample.width), dtype=bool),
        )

    details = []
    for split in splits:
        paths = [path for path, target in assignments.items() if target == split]
        groups = {group_lookup.get(path, path) for path in paths}
        details.append(f"{split}: {len(paths)} images / {len(groups)} groups")
    options = VisualizationOptions(samples=None, columns=2, panel_size=4.0, show=False)
    chart = visualize_records(
        chosen,
        options=options,
        prepare=prepare,
        title="Split assignment audit — " + " · ".join(details),
    )
    save_chart(chart, output)
    return output


def save_split_summary(samples: list[Sample], assignments: dict[str, str], output: Path) -> Path:
    import altair as alt

    splits = [s for s in ("train", "val", "test") if s in assignments.values()]
    rows = [
        {
            "split": split,
            "images": sum(v == split for v in assignments.values()),
            "annotations": sum(
                len(sample.annotations)
                for sample in samples
                if assignments.get(str(sample.image_path)) == split
            ),
        }
        for split in splits
    ]
    colors = alt.Scale(domain=splits, range=[SPLIT_COLORS[split] for split in splits])

    def panel(field: str, title: str) -> Any:
        return (
            alt.Chart(alt.Data(values=rows))
            .mark_bar()
            .encode(
                x=alt.X("split:N", sort=splits, title="split"),
                y=alt.Y(f"{field}:Q", title=field),
                color=alt.Color("split:N", scale=colors, legend=None),
                tooltip=["split:N", alt.Tooltip(f"{field}:Q", format=",")],
            )
            .properties(width=330, height=260, title=title)
        )

    save_chart(alt.hconcat(panel("images", "Images per split"), panel("annotations", "Annotations per split"), spacing=34), output)
    return output


def save_class_count_summary(
    before: dict[str, int],
    after: dict[str, int],
    output: Path,
    *,
    title: str = "Class counts before and after removal",
) -> Path:
    import altair as alt

    labels = sorted((set(before) | set(after)) - {"background"}) + ["background"]
    rows = [
        {"class": label, "phase": phase, "count": values.get(label, 0)}
        for label in labels
        for phase, values in (("before", before), ("after", after))
    ]
    chart = (
        alt.Chart(alt.Data(values=rows))
        .mark_bar()
        .encode(
            x=alt.X("class:N", sort=labels, title="class / background"),
            xOffset=alt.XOffset("phase:N", sort=["before", "after"]),
            y=alt.Y("count:Q", title="count"),
            color=alt.Color("phase:N", sort=["before", "after"]),
            tooltip=["class:N", "phase:N", alt.Tooltip("count:Q", format=",")],
        )
        .properties(width=max(520, 72 * len(labels)), height=300, title=title)
    )
    save_chart(chart, output)
    return output


def save_tiling_count_summary(
    before_annotations: dict[str, int],
    after_annotations: dict[str, int],
    image_composition: dict[str, dict[str, int]],
    output: Path,
) -> Path:
    """Plot annotation counts and image composition without mixing their units."""

    import altair as alt

    labels = sorted(set(before_annotations) | set(after_annotations))
    annotation_rows = [
        {"class": label, "phase": phase, "count": values.get(label, 0)}
        for label in labels
        for phase, values in (("before", before_annotations), ("after", after_annotations))
    ]
    composition_rows = [
        {"phase": phase, "kind": kind, "images": image_composition[phase][kind]}
        for phase in ("before", "after")
        for kind in ("annotated", "background")
    ]
    annotations = (
        alt.Chart(alt.Data(values=annotation_rows))
        .mark_bar()
        .encode(
            x=alt.X("class:N", sort=labels),
            xOffset=alt.XOffset("phase:N", sort=["before", "after"]),
            y=alt.Y("count:Q", title="annotation instances"),
            color=alt.Color("phase:N", sort=["before", "after"]),
        )
        .properties(width=max(420, 62 * len(labels)), height=280, title="Annotations by class")
    )
    composition = (
        alt.Chart(alt.Data(values=composition_rows))
        .mark_bar()
        .encode(
            x=alt.X("phase:N", sort=["before", "after"]),
            xOffset=alt.XOffset("kind:N", sort=["annotated", "background"]),
            y=alt.Y("images:Q", title="images"),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(domain=["annotated", "background"], range=["#4477AA", "#CC6677"]),
            ),
        )
        .properties(width=320, height=280, title="Image composition used for background ratio")
    )
    save_chart(alt.hconcat(annotations, composition, spacing=34), output)
    return output


def save_source_pixel_coverage_summary(
    rows: list[dict[str, Any]],
    output: Path,
) -> Path:
    """Plot exact source-area coverage aggregates and per-image distributions."""

    import altair as alt

    exact = [row for row in rows if row.get("coverage_status") == "exact"]
    available = {str(row["split"]) for row in rows}
    splits = [split for split in ("train", "val", "test") if split in available]
    splits.extend(sorted(available - set(splits)))
    aggregate_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    unsupported: list[str] = []
    for split in splits:
        split_exact = [row for row in exact if row["split"] == split]
        values = sorted(float(row["source_pixel_coverage_percent"]) for row in split_exact)
        covered = sum(float(row["covered_source_area_px"]) for row in split_exact)
        total = sum(float(row["source_area_px"]) for row in split_exact)
        metrics = {
            "pixel-weighted": 100.0 * covered / total if total else 0.0,
            "mean per source": sum(values) / len(values) if values else 0.0,
            "median per source": (
            values[len(values) // 2]
            if len(values) % 2 == 1
            else ((values[len(values) // 2 - 1] + values[len(values) // 2]) / 2 if values else 0.0)
            ),
        }
        aggregate_rows.extend(
            {"split": split, "metric": metric, "percent": value}
            for metric, value in metrics.items()
        )
        distribution_rows.extend(
            {"split": split, "percent": value}
            for value in values
        )
        count = sum(
            row["split"] == split and row.get("coverage_status") != "exact"
            for row in rows
        )
        if count:
            unsupported.append(f"{split}={count}")

    bars = (
        alt.Chart(alt.Data(values=aggregate_rows))
        .mark_bar()
        .encode(
            x=alt.X("split:N", sort=splits),
            xOffset=alt.XOffset("metric:N"),
            y=alt.Y("percent:Q", scale=alt.Scale(domain=[0, 105]), title="source area (%)"),
            color=alt.Color(
                "metric:N",
                scale=alt.Scale(
                    domain=["pixel-weighted", "mean per source", "median per source"],
                    range=["#4477AA", "#66CCEE", "#228833"],
                ),
            ),
            tooltip=["split:N", "metric:N", alt.Tooltip("percent:Q", format=".1f")],
        )
        .properties(width=max(360, 105 * len(splits)), height=300, title="Spatial coverage by split")
    )
    distribution = (
        alt.Chart(alt.Data(values=distribution_rows))
        .mark_boxplot(size=42)
        .encode(
            x=alt.X("split:N", sort=splits),
            y=alt.Y("percent:Q", scale=alt.Scale(domain=[0, 105]), title="source pixel coverage (%)"),
            color=alt.Color("split:N", legend=None),
        )
        .properties(
            width=max(320, 90 * len(splits)),
            height=300,
            title=alt.TitleParams(
                text="Per-source coverage distribution",
                subtitle=("Unsupported source(s): " + ", ".join(unsupported)) if unsupported else None,
            ),
        )
    )
    chart = alt.hconcat(bars, distribution, spacing=36).properties(
        title="Original source-pixel coverage from unioned tile footprints"
    )
    save_chart(chart, output)
    return output


def save_label_coverage_summary(
    rows: list[dict[str, Any]],
    output: Path,
) -> Path:
    """Plot label-hit and requested-appearance coverage by split."""

    import altair as alt

    available = {str(row["split"]) for row in rows}
    splits = [split for split in ("train", "val", "test") if split in available]
    splits.extend(sorted(available - set(splits)))
    percent_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    for split in splits:
        selected = [row for row in rows if row["split"] == split]
        total = sum(int(row["total_labels"]) for row in selected)
        covered = sum(int(row["labels_covered_at_least_once"]) for row in selected)
        requested = sum(int(row["requested_coverages"]) for row in selected)
        actual = sum(int(row["actual_coverages"]) for row in selected)
        percent_rows.extend(
            (
                {"split": split, "metric": "labels hit at least once", "percent": 100.0 * covered / total if total else 0.0},
                {"split": split, "metric": "requested appearances produced", "percent": 100.0 * actual / requested if requested else 0.0},
            )
        )
        count_rows.extend(
            (
                {"split": split, "status": "covered", "count": covered},
                {"split": split, "status": "never covered", "count": total - covered},
            )
        )

    percentages = (
        alt.Chart(alt.Data(values=percent_rows))
        .mark_bar()
        .encode(
            x=alt.X("split:N", sort=splits),
            xOffset="metric:N",
            y=alt.Y("percent:Q", scale=alt.Scale(domain=[0, 105]), title="coverage (%)"),
            color=alt.Color("metric:N", scale=alt.Scale(range=["#228833", "#4477AA"])),
        )
        .properties(width=max(340, 100 * len(splits)), height=290, title="Annotation sampling coverage")
    )
    counts = (
        alt.Chart(alt.Data(values=count_rows))
        .mark_bar()
        .encode(
            x=alt.X("split:N", sort=splits),
            y=alt.Y("count:Q", title="source annotations", stack="zero"),
            color=alt.Color(
                "status:N",
                sort=["covered", "never covered"],
                scale=alt.Scale(domain=["covered", "never covered"], range=["#228833", "#CC6677"]),
            ),
        )
        .properties(width=max(300, 86 * len(splits)), height=290, title="Labels represented at least once")
    )
    save_chart(alt.hconcat(percentages, counts, spacing=36).properties(title="Coverage-tiling annotation audit"), output)
    return output


def save_empty_image_balance_summary(summary: dict[str, dict[str, Any]], output: Path) -> Path:
    """Plot annotated/background image distributions before and after balancing."""

    import altair as alt

    splits = list(summary)
    rows = [
        {
            "split": split,
            "phase": phase,
            "kind": kind,
            "images": int(summary[split][phase][kind]),
        }
        for split in splits
        for phase in ("before", "after")
        for kind in ("annotated", "background")
    ]

    def panel(phase: str) -> Any:
        return (
            alt.Chart(alt.Data(values=[row for row in rows if row["phase"] == phase]))
            .mark_bar()
            .encode(
                x=alt.X("split:N", sort=splits),
                xOffset=alt.XOffset("kind:N"),
                y=alt.Y("images:Q", title="images"),
                color=alt.Color(
                    "kind:N",
                    scale=alt.Scale(domain=["annotated", "background"], range=["#4477AA", "#CC6677"]),
                ),
                tooltip=["split:N", "kind:N", alt.Tooltip("images:Q", format=",")],
            )
            .properties(width=max(300, 88 * len(splits)), height=280, title=phase.capitalize())
        )

    save_chart(alt.hconcat(panel("before"), panel("after"), spacing=36).properties(title="Annotated and background image distribution"), output)
    return output


def save_class_removal_preview(
    sample: Sample,
    class_mapping: dict[int, int],
    task: Task,
    before_metadata: DatasetMetadata,
    after_metadata: DatasetMetadata,
    output: Path,
) -> Path:
    after = Sample(
        sample.image_path,
        sample.relative_path,
        sample.split,
        sample.width,
        sample.height,
        [
            annotation.clone(class_id=class_mapping[annotation.class_id])
            for annotation in sample.annotations
            if annotation.class_id in class_mapping
        ],
    )

    def prepare(_: Sample) -> VisualizationItem:
        return VisualizationItem(
            image_path=sample.image_path,
            label=str(sample.relative_path),
            panels=(
                VisualizationPanel(
                    title="Before",
                    image=np.asarray(render_annotated_sample(sample, task, before_metadata)),
                ),
                VisualizationPanel(
                    title="After",
                    image=np.asarray(render_annotated_sample(after, task, after_metadata)),
                ),
            ),
            foreground=np.ones((sample.height, sample.width), dtype=bool),
        )

    chart = visualize_records(
        [sample],
        options=VisualizationOptions(samples=None, columns=1, panel_size=4.0, show=False),
        prepare=prepare,
        title="Class removal preview",
    )
    save_chart(chart, output)
    return output


def save_tiling_preview(
    items: list[tuple[Sample, list[tuple[int, int, int, int]], str]],
    task: Task,
    metadata: DatasetMetadata,
    output: Path,
    *,
    mode: str,
    visualize_kwargs: Mapping[str, Any] | None = None,
) -> Path:
    """Show one small pass-through source and up to three tiled sources."""

    render_options = normalize_visualize_kwargs(visualize_kwargs)

    def prepare(value: tuple[Sample, list[tuple[int, int, int, int]], str]) -> VisualizationItem:
        sample, boxes, status = value
        rendered = render_annotated_sample(
            sample,
            task,
            metadata,
            line_width=render_options.get("line_width"),
            outline_width=render_options.get("outline_width"),
        ).convert("RGBA")
        draw = ImageDraw.Draw(rendered)
        preview_scale = max(rendered.size) / 400
        window_width = max(2, round(1.5 * preview_scale))
        index_font = _font(max(13, round(11 * preview_scale)))
        for index, (left, top, right, bottom) in enumerate(boxes):
            draw.rectangle(
                (left, top, right, bottom),
                outline="#00d4ff",
                width=window_width,
            )
            if len(boxes) <= 20:
                _text_box(
                    draw,
                    (left + window_width * 2, top + window_width * 2),
                    str(index),
                    index_font,
                    (255, 255, 255, 255),
                    pad=max(3, window_width),
                )
        label = _sample_title(
            sample,
            render_options.get("label_fn"),
            default=(
                f"{status} · {sample.split} · {sample.width}×{sample.height}\n"
                f"{sample.relative_path} · {len(sample.annotations)} annotations"
            ),
        )
        return VisualizationItem(
            image_path=sample.image_path,
            label=label or "",
            panels=(
                VisualizationPanel(
                    title="Crop windows",
                    image=np.asarray(rendered),
                    footer="PASS-THROUGH · NO CROP" if not boxes else None,
                ),
            ),
            foreground=np.ones((sample.height, sample.width), dtype=bool),
        )

    pass_through = sum(not boxes for _, boxes, _ in items)
    tiled = len(items) - pass_through
    options = VisualizationOptions(
        samples=None,
        columns=2,
        panel_size=4.2,
        label_mode=render_options.get("label_mode", "middle"),
        show=False,
    )
    chart = visualize_records(
        items,
        options=options,
        prepare=prepare,
        title=(
            f"{mode.capitalize()} tiling preview — "
            f"{pass_through} small pass-through source(s), {tiled} tiled source(s) — "
            "cyan rectangles = output crop windows"
        ),
    )
    save_chart(chart, output)
    return output


def coverage_color(percent: float) -> tuple[int, int, int]:
    percent = max(0, min(100, percent))
    if percent >= 100:
        return 0, 220, 90
    if percent >= 67:
        return 255, 210, 0
    if percent >= 34:
        return 255, 130, 0
    return 255, 40, 40


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in ("Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    *,
    box_fill: tuple[int, int, int, int] = (0, 0, 0, 190),
    pad: int = 6,
) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rounded_rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        radius=max(4, pad),
        fill=box_fill,
    )
    draw.text((x, y), text, fill=fill, font=font)


def _wrap_legend_lines(
    draw: ImageDraw.ImageDraw, lines: list[str], font: ImageFont.ImageFont, max_width: int
) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        words: list[str] = []
        for word in line.split():
            bounds = draw.textbbox((0, 0), word, font=font)
            if bounds[2] - bounds[0] <= max_width:
                words.append(word)
                continue
            chunk = ""
            for character in word:
                candidate = f"{chunk}{character}"
                chunk_bounds = draw.textbbox((0, 0), candidate, font=font)
                if chunk and chunk_bounds[2] - chunk_bounds[0] > max_width:
                    words.append(chunk)
                    chunk = character
                else:
                    chunk = candidate
            if chunk:
                words.append(chunk)
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            bounds = draw.textbbox((0, 0), candidate, font=font)
            if current and bounds[2] - bounds[0] > max_width:
                wrapped.append(current)
                current = word
            else:
                current = candidate
        wrapped.append(current)
    return wrapped


def save_coverage_annotated_original(
    sample: Sample,
    coverage_counts: dict[int, int],
    coverage_targets: dict[int, int],
    coverage_types: dict[int, str],
    background_boxes: list[tuple[int, int, int, int]],
    output: Path,
    settings: dict[str, Any],
) -> Path:
    with Image.open(sample.image_path) as opened:
        annotated = ImageOps.exif_transpose(opened).convert("RGBA")
    source_width, source_height = annotated.size
    preview_scale = max(
        1.0,
        900.0 / max(1, source_width, source_height),
    )
    if preview_scale > 1:
        annotated = annotated.resize(
            (
                int(round(source_width * preview_scale)),
                int(round(source_height * preview_scale)),
            ),
            Image.Resampling.BILINEAR,
        )
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = annotated.size
    min_dim = min(width, height)
    marker_radius = max(12, min(40, int(min_dim * 0.022)))
    center_radius = max(4, int(marker_radius * 0.28))
    line_width = max(3, min(10, int(min_dim * 0.005)))
    underlay = line_width + 4
    font_size = max(15, min(30, int(min_dim * 0.026)))
    bg_size = max(14, min(26, int(min_dim * 0.022)))
    font, bg_font = _font(font_size), _font(bg_size)

    cyan = (0, 190, 255)
    for index, box in enumerate(background_boxes):
        left, top, right, bottom = (
            int(round(value * preview_scale))
            for value in box
        )
        display_box = (left, top, right, bottom)
        draw.rectangle(display_box, fill=cyan + (35,))
        draw.rectangle(display_box, outline=(255, 255, 255, 235), width=underlay)
        draw.rectangle(display_box, outline=cyan + (230,), width=line_width)
        text = f"bg {index}"
        tx, ty = left + max(6, line_width * 2), top + max(6, line_width * 2)
        bounds = draw.textbbox((0, 0), text, font=bg_font)
        tx = min(max(tx, 8), max(8, width - (bounds[2] - bounds[0]) - 20))
        ty = min(max(ty, 8), max(8, height - (bounds[3] - bounds[1]) - 20))
        _text_box(draw, (tx, ty), text, bg_font, cyan + (230,), box_fill=(0, 0, 0, 135), pad=max(5, bg_size // 4))

    for label_idx, annotation in enumerate(sample.annotations):
        geometry_color = (
            (255, 0, 255),
            (127, 255, 0),
            (255, 95, 0),
            (255, 255, 0),
            (255, 20, 147),
        )[annotation.class_id % len(ANNOTATION_COLORS)]
        geometry_rgba = geometry_color + (235,)
        if annotation.polygon:
            polygon = [
                (
                    int(round(x * preview_scale)),
                    int(round(y * preview_scale)),
                )
                for x, y in annotation.polygon
            ]
            if len(polygon) >= 3:
                closed = [*polygon, polygon[0]]
                draw.polygon(polygon, fill=geometry_color + (38,))
                draw.line(
                    closed,
                    fill=(255, 255, 255, 245),
                    width=underlay,
                    joint="curve",
                )
                draw.line(
                    closed,
                    fill=geometry_rgba,
                    width=line_width,
                    joint="curve",
                )
                xs, ys = zip(*polygon)
                bounds = (min(xs), min(ys), max(xs), max(ys))
                draw.rectangle(
                    bounds,
                    outline=(255, 255, 255, 245),
                    width=underlay,
                )
                draw.rectangle(
                    bounds,
                    outline=geometry_rgba,
                    width=line_width,
                )
        elif annotation.bbox is not None:
            bounds = tuple(
                int(round(value * preview_scale))
                for value in annotation.bbox
            )
            draw.rectangle(
                bounds,
                outline=(255, 255, 255, 245),
                width=underlay,
            )
            draw.rectangle(
                bounds,
                outline=geometry_rgba,
                width=line_width,
            )

        if annotation.point is not None:
            anchor = annotation.point
        elif annotation.bbox is not None:
            x1, y1, x2, y2 = annotation.bbox
            anchor = ((x1 + x2) / 2, (y1 + y2) / 2)
        elif annotation.polygon:
            xs, ys = zip(*annotation.polygon)
            anchor = (sum(xs) / len(xs), sum(ys) / len(ys))
        else:
            continue
        x, y = (
            int(round(anchor[0] * preview_scale)),
            int(round(anchor[1] * preview_scale)),
        )
        x, y = min(max(x, 0), width - 1), min(max(y, 0), height - 1)
        count = coverage_counts.get(label_idx, 0)
        target = coverage_targets.get(label_idx, settings["target_appearances_per_object"])
        coverage_type = coverage_types.get(label_idx, "sparse")
        color = coverage_color(100 * count / target if target else 0)
        rgba = color + (200,)
        marker = (x - marker_radius, y - marker_radius, x + marker_radius, y + marker_radius)
        if sample.split == "train":
            draw.ellipse(marker, outline=(255, 255, 255, 235), width=underlay)
            draw.ellipse(marker, outline=rgba, width=line_width)
        else:
            draw.rectangle(marker, outline=(255, 255, 255, 235), width=underlay)
            draw.rectangle(marker, outline=rgba, width=line_width)
        draw.ellipse((x - center_radius, y - center_radius, x + center_radius, y + center_radius), fill=(255, 255, 255, 255))
        inner = max(1, center_radius // 2)
        draw.ellipse((x - inner, y - inner, x + inner, y + inner), fill=rgba)
        configured_radius = settings["polo_radius_px"]
        radius = int(
            round(
                (configured_radius if configured_radius is not None else annotation.radius or 0)
                * preview_scale
            )
        )
        if annotation.point is not None and radius >= 3:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color + (130,), width=max(2, line_width // 2))
        type_label = {
            "dense": "D",
            "sparse": "S",
            "override": "O",
        }.get(coverage_type, "?")
        text = f"{type_label} {count}/{target}"
        bounds = draw.textbbox((0, 0), text, font=font)
        tw, th = bounds[2] - bounds[0], bounds[3] - bounds[1]
        tx, ty = x + marker_radius + 12, y - marker_radius - 6
        if tx + tw + 20 > width:
            tx = x - marker_radius - tw - 24
        if ty < 0:
            ty = y + marker_radius + 12
        tx = min(max(tx, 8), max(8, width - tw - 20))
        ty = min(max(ty, 8), max(8, height - th - 20))
        _text_box(draw, (tx, ty), text, font, rgba, box_fill=(0, 0, 0, 125), pad=max(6, font_size // 4))

    dense = sum(value == "dense" for value in coverage_types.values())
    sparse = sum(value == "sparse" for value in coverage_types.values())
    overrides = sum(value == "override" for value in coverage_types.values())
    hit = sum(coverage_counts.get(i, 0) >= 1 for i in range(len(sample.annotations)))
    complete = sum(
        coverage_counts.get(i, 0) >= coverage_targets.get(i, 1)
        for i in range(len(sample.annotations))
    )
    percent_hit = 100 * hit / len(sample.annotations) if sample.annotations else 0
    cap = settings["max_tiles_per_source_image"]
    filename = sample.image_path.name
    if len(filename) > 72:
        filename = f"{filename[:35]}…{filename[-36:]}"
    lines = [f"{sample.split.upper()} coverage | {filename}"]
    if sample.annotations:
        lines.extend(
            [
                (
                    f"Objects: {len(sample.annotations)} | dense: {dense} × "
                    f"{settings['target_appearances_per_object']} | sparse: {sparse} × "
                    f"{settings['sparse_appearances_per_object']} | overrides: {overrides}"
                ),
                (
                    f"Coverage: hit at least once {hit}/{len(sample.annotations)} "
                    f"({percent_hit:.1f}%) | completed target {complete}/{len(sample.annotations)}"
                ),
                (
                    f"Dense rule: at least {settings['min_nearby_objects_for_full_coverage']} "
                    f"other objects within {settings['dense_neighbor_radius_px']:.0f}px"
                ),
            ]
        )
    else:
        lines.append(
            "No source annotations; this image is useful only as background imagery."
        )
    configured_background_ratio = settings["background_ratio"]
    if isinstance(configured_background_ratio, Mapping):
        configured_background_ratio = configured_background_ratio[sample.split]
    background_policy = settings.get("background_ratio_policy", {}).get(sample.split)
    if background_policy and background_policy["mode"] == "best_effort_source_fraction":
        background_target = (
            f"best effort at input {float(background_policy['target_fraction']):.1%}"
        )
    elif isinstance(configured_background_ratio, (list, tuple)):
        low, high = map(float, configured_background_ratio)
        background_target = f"{low:.1%}–{high:.1%} (aim {high:.1%})"
    elif configured_background_ratio is None:
        background_target = "best effort at input fraction"
    else:
        background_target = f"{float(configured_background_ratio):.1%}"
    lines.extend(
        [
            (
                f"Background crops from this source: {len(background_boxes)} | "
                f"dataset target: {background_target} | "
                f"max tiles/source: {cap if cap is not None else 'disabled'}"
            ),
            (
                "Markers: D=dense, S=sparse, O=override, followed by actual/target; "
                "green=complete, yellow/orange=partial, red=low"
            ),
            (
                "Magenta/lime/orange geometry with white under-stroke = annotations; "
                "segmentation rectangles = mask bounds; cyan = background crop"
            ),
        ]
    )

    # Put explanatory text below the raster so it cannot hide annotations on
    # small source images.
    composited = Image.alpha_composite(annotated, overlay).convert("RGB")
    panel_font_size = max(15, min(24, int(min_dim * 0.028)))
    panel_font = _font(panel_font_size)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    fitted_lines = _wrap_legend_lines(
        measure,
        lines,
        panel_font,
        max(120, width - 32),
    )
    gap = max(5, panel_font_size // 3)
    line_heights = [
        measure.textbbox((0, 0), line, font=panel_font)[3]
        - measure.textbbox((0, 0), line, font=panel_font)[1]
        for line in fitted_lines
    ]
    panel_height = 28 + sum(line_heights) + gap * max(0, len(fitted_lines) - 1)
    canvas = Image.new("RGB", (width, height + panel_height), (12, 12, 16))
    canvas.paste(composited, (0, 0))
    panel_draw = ImageDraw.Draw(canvas)
    panel_draw.rectangle(
        (0, height, width, height + max(3, line_width)),
        fill=(255, 0, 255),
    )
    y_cursor = height + 16
    for line_index, line in enumerate(fitted_lines):
        color = (255, 255, 255) if line_index == 0 else (225, 225, 232)
        panel_draw.text((16, y_cursor), line, fill=color, font=panel_font)
        bounds = panel_draw.textbbox((16, y_cursor), line, font=panel_font)
        y_cursor += bounds[3] - bounds[1] + gap
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=int(settings["jpeg_quality"]))
    return output


def display_report(path: Path) -> None:
    """Display the canonical report PNG inline without redrawing its contents."""

    display_image(path)
