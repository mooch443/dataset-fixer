from __future__ import annotations

import math
import random
import re
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps
from shapely.geometry import Polygon
from shapely.validation import explain_validity

from .models import Annotation, DatasetMetadata, Sample, Task

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

_VISUALIZE_KWARGS = {"label_fn", "line_width", "outline_width"}


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
):
    """Render a bounded grid of load-time validation failures."""

    columns = 1 if len(examples) == 1 else 2
    rows = max(1, math.ceil(max(1, len(examples)) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(7 * columns, 5.6 * rows), squeeze=False)
    flat = axes.flatten()
    for ax, example in zip(flat, examples):
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
                _draw_sample(ax, sample, task, metadata)
                if example.annotation is not None:
                    _focus_invalid_annotation(
                        ax,
                        example.annotation,
                        width=example.width,
                        height=example.height,
                    )
                    _highlight_invalid_annotation(
                        ax,
                        example.annotation,
                        width=example.width,
                        height=example.height,
                    )
            except Exception:
                _draw_failure_placeholder(ax)
        else:
            _draw_failure_placeholder(ax)
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
        ax.set_title(
            f"Skipped · {example.split or 'unknown split'} · "
            f"{textwrap.shorten(source, width=78, placeholder='…')}\n{message}",
            color="#7f1d1d",
            fontsize=9,
            pad=8,
        )
    for ax in flat[len(examples) :]:
        ax.axis("off")
    fig.suptitle(
        f"Load validation skips — {dataset_name} — {total_count} failed item(s); "
        f"showing {len(examples)}",
        fontsize=13,
    )
    fig.subplots_adjust(top=0.88, hspace=0.38, wspace=0.08)
    save_to.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_to, bbox_inches="tight", dpi=140)
    if show:
        _display_or_print(fig, save_to)
    return fig


def _draw_failure_placeholder(ax) -> None:
    ax.set_facecolor("#242830")
    ax.text(
        0.5,
        0.5,
        "No readable image\navailable for this failure",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="white",
        fontsize=11,
    )
    ax.set_xticks([])
    ax.set_yticks([])


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
    ax,
    annotation: Annotation,
    *,
    width: int,
    height: int,
) -> None:
    """Overlay ordered vertices and explicit defects on a rejected polygon."""

    if annotation.polygon is None:
        return
    reasons, markers = _polygon_invalidity_details(
        annotation.polygon,
        width=width,
        height=height,
    )
    if not reasons:
        return

    finite_polygon = [
        (x, y)
        for x, y in annotation.polygon
        if math.isfinite(x) and math.isfinite(y)
    ]
    if len(finite_polygon) >= 2:
        closed_polygon = finite_polygon + (
            [finite_polygon[0]] if len(finite_polygon) >= 3 else []
        )
        ax.plot(
            [point[0] for point in closed_polygon],
            [point[1] for point in closed_polygon],
            color="#dc2626",
            linewidth=2.0,
            zorder=8,
        )
    if finite_polygon:
        ax.scatter(
            [point[0] for point in finite_polygon],
            [point[1] for point in finite_polygon],
            marker="o",
            s=42,
            facecolors="white",
            edgecolors="#dc2626",
            linewidths=1.5,
            zorder=9,
        )
        for index, (x, y) in enumerate(finite_polygon):
            ax.annotate(
                str(index),
                xy=(x, y),
                xytext=(5, 5),
                textcoords="offset points",
                color="#111827",
                fontsize=7,
                bbox={
                    "boxstyle": "round,pad=0.16",
                    "facecolor": "white",
                    "edgecolor": "#d1d5db",
                    "alpha": 0.92,
                },
                zorder=10,
            )
    for x, y, label in markers:
        ax.scatter(
            [x],
            [y],
            marker="o",
            s=110,
            facecolors="none",
            edgecolors="#f59e0b",
            linewidths=2.2,
            zorder=11,
            clip_on=False,
        )
        ax.scatter([x], [y], marker="o", s=18, c="#f59e0b", zorder=12)
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(9, 11),
            textcoords="offset points",
            color="#111827",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.24",
                "facecolor": "white",
                "edgecolor": "#f59e0b",
                "alpha": 0.94,
            },
            arrowprops={"arrowstyle": "->", "color": "#f59e0b", "linewidth": 1.2},
            zorder=13,
        )
    ax.text(
        0.02,
        0.02,
        "Invalid polygon\n" + "\n".join(reasons),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#7f1d1d",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "#fff7ed",
            "edgecolor": "#fca5a5",
            "alpha": 0.94,
        },
        zorder=14,
    )


def _focus_invalid_annotation(
    ax,
    annotation: Annotation,
    *,
    width: int,
    height: int,
) -> None:
    """Zoom to the finite polygon vertices while retaining nearby image context."""

    if annotation.polygon is None:
        return
    points = [
        (float(x), float(y))
        for x, y in annotation.polygon
        if math.isfinite(x) and math.isfinite(y)
    ]
    if not points or width <= 0 or height <= 0:
        return

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
    ax.set_xlim(left, right)
    # imshow uses an upper-left origin, so increasing data y runs downward.
    ax.set_ylim(bottom, top)


def visualize_samples(
    samples: list[Sample],
    task: Task,
    metadata: DatasetMetadata,
    *,
    split: str | None,
    n: int,
    seed: int,
    columns: int,
    label_fn: Callable[[Path], str | None] | None = None,
    line_width: float | None = None,
    outline_width: float | None = None,
    save_to: Path | None = None,
    show: bool = True,
):
    candidates = [s for s in samples if split is None or s.split == split]
    rng = random.Random(seed)
    chosen = rng.sample(candidates, min(n, len(candidates))) if candidates else []
    rows = max(1, math.ceil(max(1, len(chosen)) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 5 * rows), squeeze=False)
    flat = axes.flatten()
    for ax, sample in zip(flat, chosen):
        _draw_sample(
            ax,
            sample,
            task,
            metadata,
            label_fn=label_fn,
            line_width=line_width,
            outline_width=outline_width,
        )
    for ax in flat[len(chosen) :]:
        ax.axis("off")
    title_split = split or "all splits"
    fig.suptitle(f"Annotation check — {title_split} ({len(chosen)} images)", fontsize=13)
    fig.tight_layout()
    if save_to:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight", dpi=140)
    if show:
        _display_or_print(fig, save_to)
    return fig


def visualize_semantic_masks(
    samples: list[Sample],
    mask_paths: dict[Path, Path],
    *,
    split: str | None,
    n: int,
    seed: int,
    columns: int,
    label_fn: Callable[[Path], str | None] | None = None,
    save_to: Path | None = None,
    show: bool = True,
):
    """Render paired source images and binary-mask overlays."""

    candidates = [sample for sample in samples if split is None or sample.split == split]
    rng = random.Random(seed)
    chosen = rng.sample(candidates, min(n, len(candidates))) if candidates else []
    rows = max(1, math.ceil(max(1, len(chosen)) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 5 * rows), squeeze=False)
    flat = axes.flatten()
    for ax, sample in zip(flat, chosen):
        mask_path = mask_paths[sample.image_path.resolve()]
        with Image.open(sample.image_path) as opened_image, Image.open(mask_path) as opened_mask:
            image = opened_image.convert("RGB")
            mask = opened_mask.convert("L")
        overlay = Image.blend(
            image,
            Image.composite(Image.new("RGB", image.size, (255, 32, 32)), image, mask),
            0.45,
        )
        foreground = sum(mask.histogram()[1:])
        fraction = foreground / (mask.width * mask.height) if mask.width and mask.height else 0.0
        ax.imshow(overlay)
        title = _sample_title(
            sample,
            label_fn,
            default=(
                f"{sample.split} · {sample.relative_path}\n"
                f"foreground: {fraction:.1%}"
            ),
        )
        if title is not None:
            ax.set_title(title, fontsize=8)
        ax.axis("off")
    for ax in flat[len(chosen) :]:
        ax.axis("off")
    title_split = split or "all splits"
    fig.suptitle(f"Semantic-mask check — {title_split} ({len(chosen)} images)", fontsize=13)
    fig.tight_layout()
    if save_to:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight", dpi=140)
    if show:
        _display_or_print(fig, save_to)
    return fig


def _draw_sample(
    ax,
    sample: Sample,
    task: Task,
    metadata: DatasetMetadata,
    *,
    label_fn: Callable[[Path], str | None] | None = None,
    line_width: float | None = None,
    outline_width: float | None = None,
) -> None:
    ax.imshow(
        render_annotated_sample(
            sample,
            task,
            metadata,
            line_width=line_width,
            outline_width=outline_width,
        )
    )
    title = _sample_title(
        sample,
        label_fn,
        default=(
            f"{sample.relative_path}\n{sample.width}×{sample.height} | "
            f"{len(sample.annotations)} annotations | {sample.split}"
        ),
    )
    if title is not None:
        ax.set_title(title, fontsize=8)
    ax.axis("off")


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
                draw.text((box[0] + 2, text_top + 2), name, fill="white", font=font)
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
                draw.line(closed, fill=color, width=colour_width, joint="curve")
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
                    draw.text((px + marker_radius + 1, py + 1), names[idx], fill="white", font=font)
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

    columns = 2
    rows = max(1, len(splits))
    fig, axes = plt.subplots(rows, columns, figsize=(12, 5 * rows), squeeze=False)
    flat = axes.flatten()
    for ax, (target, sample) in zip(flat, chosen):
        _draw_sample(ax, sample, task, metadata)
        group = group_lookup.get(str(sample.image_path), "ungrouped")
        if len(group) > 70:
            group = f"…{group[-69:]}"
        ax.set_title(
            f"{target.upper()} · group {group}\n"
            f"{sample.relative_path} · {len(sample.annotations)} annotations",
            color=SPLIT_COLORS[target],
            fontsize=8,
        )
    for ax in flat[len(chosen) :]:
        ax.axis("off")
    details = []
    for split in splits:
        paths = [path for path, target in assignments.items() if target == split]
        groups = {group_lookup.get(path, path) for path in paths}
        details.append(f"{split}: {len(paths)} images / {len(groups)} groups")
    fig.suptitle("Split assignment audit — " + " · ".join(details), fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", dpi=140)
    plt.close(fig)
    return output


def save_split_summary(samples: list[Sample], assignments: dict[str, str], output: Path) -> Path:
    splits = [s for s in ("train", "val", "test") if s in assignments.values()]
    image_counts = [sum(v == split for v in assignments.values()) for split in splits]
    annotation_counts = [
        sum(len(sample.annotations) for sample in samples if assignments.get(str(sample.image_path)) == split)
        for split in splits
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(splits, image_counts, color=[SPLIT_COLORS[s] for s in splits])
    axes[0].set_title("Images per split")
    axes[1].bar(splits, annotation_counts, color=[SPLIT_COLORS[s] for s in splits])
    axes[1].set_title("Annotations per split")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", dpi=140)
    plt.close(fig)
    return output


def save_class_count_summary(
    before: dict[str, int],
    after: dict[str, int],
    output: Path,
    *,
    title: str = "Class counts before and after removal",
) -> Path:
    labels = sorted((set(before) | set(after)) - {"background"}) + ["background"]
    before_values = [before.get(label, 0) for label in labels]
    after_values = [after.get(label, 0) for label in labels]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.8), 4.5))
    ax.bar([value - 0.2 for value in x], before_values, width=0.4, label="before")
    ax.bar([value + 0.2 for value in x], after_values, width=0.4, label="after")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.text(
        0.99,
        0.98,
        "class bars = annotations\nbackground = empty images",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#555555",
    )
    ax.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", dpi=140)
    plt.close(fig)
    return output


def save_tiling_count_summary(
    before_annotations: dict[str, int],
    after_annotations: dict[str, int],
    image_composition: dict[str, dict[str, int]],
    output: Path,
) -> Path:
    """Plot annotation counts and image composition without mixing their units."""

    labels = sorted(set(before_annotations) | set(after_annotations))
    x = list(range(len(labels)))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(11, len(labels) * 0.8 + 6), 4.5),
    )

    axes[0].bar(
        [value - 0.2 for value in x],
        [before_annotations.get(label, 0) for label in labels],
        width=0.4,
        label="before",
    )
    axes[0].bar(
        [value + 0.2 for value in x],
        [after_annotations.get(label, 0) for label in labels],
        width=0.4,
        label="after",
    )
    axes[0].set_xticks(x, labels, rotation=30, ha="right")
    axes[0].set_ylabel("annotation instances")
    axes[0].set_title("Annotations by class")
    axes[0].legend()

    phases = ["before", "after"]
    categories = ["annotated", "background"]
    colors = {"annotated": "#4477AA", "background": "#CC6677"}
    phase_x = list(range(len(phases)))
    width = 0.36
    for offset, category in zip((-width / 2, width / 2), categories):
        axes[1].bar(
            [value + offset for value in phase_x],
            [image_composition[phase][category] for phase in phases],
            width=width,
            label=category,
            color=colors[category],
        )
    axes[1].set_xticks(phase_x, phases)
    axes[1].set_ylabel("images")
    axes[1].set_title("Image composition used for background ratio")
    axes[1].legend()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", dpi=140)
    plt.close(fig)
    return output


def save_source_pixel_coverage_summary(
    rows: list[dict[str, Any]],
    output: Path,
) -> Path:
    """Plot exact source-area coverage aggregates and per-image distributions."""

    exact = [row for row in rows if row.get("coverage_status") == "exact"]
    available = {str(row["split"]) for row in rows}
    splits = [split for split in ("train", "val", "test") if split in available]
    splits.extend(sorted(available - set(splits)))
    fig, axes = plt.subplots(1, 2, figsize=(max(11, len(splits) * 2.6 + 6), 4.8))

    weighted: list[float] = []
    means: list[float] = []
    medians: list[float] = []
    distributions: list[list[float]] = []
    unsupported_counts: list[int] = []
    for split in splits:
        split_exact = [row for row in exact if row["split"] == split]
        values = sorted(float(row["source_pixel_coverage_percent"]) for row in split_exact)
        covered = sum(float(row["covered_source_area_px"]) for row in split_exact)
        total = sum(float(row["source_area_px"]) for row in split_exact)
        weighted.append(100.0 * covered / total if total else 0.0)
        means.append(sum(values) / len(values) if values else 0.0)
        medians.append(
            values[len(values) // 2]
            if len(values) % 2 == 1
            else ((values[len(values) // 2 - 1] + values[len(values) // 2]) / 2 if values else 0.0)
        )
        distributions.append(values)
        unsupported_counts.append(
            sum(
                row["split"] == split and row.get("coverage_status") != "exact"
                for row in rows
            )
        )

    x = list(range(len(splits)))
    width = 0.24
    for offset, values, label, color in (
        (-width, weighted, "pixel-weighted", "#4477AA"),
        (0.0, means, "mean per source", "#66CCEE"),
        (width, medians, "median per source", "#228833"),
    ):
        bars = axes[0].bar(
            [position + offset for position in x],
            values,
            width=width,
            label=label,
            color=color,
        )
        axes[0].bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8)
    axes[0].set_xticks(x, splits)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("union of original source area (%)")
    axes[0].set_title("Spatial coverage by split")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)

    nonempty = [(split, values) for split, values in zip(splits, distributions) if values]
    if nonempty:
        labels, values = zip(*nonempty)
        axes[1].boxplot(values, tick_labels=labels, showfliers=True)
        axes[1].set_ylim(0, 105)
        axes[1].set_ylabel("source pixel coverage (%)")
        axes[1].set_title("Per-source coverage distribution")
        axes[1].grid(axis="y", alpha=0.2)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "No exact source footprints", ha="center", va="center")
    if any(unsupported_counts):
        axes[1].text(
            0.02,
            0.02,
            "Unsupported source(s): "
            + ", ".join(
                f"{split}={count}"
                for split, count in zip(splits, unsupported_counts)
                if count
            ),
            transform=axes[1].transAxes,
            fontsize=8,
            color="#AA2222",
        )

    fig.suptitle("Original source-pixel coverage from unioned tile footprints")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return output


def save_label_coverage_summary(
    rows: list[dict[str, Any]],
    output: Path,
) -> Path:
    """Plot label-hit and requested-appearance coverage by split."""

    available = {str(row["split"]) for row in rows}
    splits = [split for split in ("train", "val", "test") if split in available]
    splits.extend(sorted(available - set(splits)))
    label_hit: list[float] = []
    requested_hit: list[float] = []
    covered_counts: list[int] = []
    missed_counts: list[int] = []
    for split in splits:
        selected = [row for row in rows if row["split"] == split]
        total = sum(int(row["total_labels"]) for row in selected)
        covered = sum(int(row["labels_covered_at_least_once"]) for row in selected)
        requested = sum(int(row["requested_coverages"]) for row in selected)
        actual = sum(int(row["actual_coverages"]) for row in selected)
        label_hit.append(100.0 * covered / total if total else 0.0)
        requested_hit.append(100.0 * actual / requested if requested else 0.0)
        covered_counts.append(covered)
        missed_counts.append(total - covered)

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(splits) * 2.5 + 5), 4.6))
    x = list(range(len(splits)))
    width = 0.36
    for offset, values, label, color in (
        (-width / 2, label_hit, "labels hit at least once", "#228833"),
        (width / 2, requested_hit, "requested appearances produced", "#4477AA"),
    ):
        bars = axes[0].bar(
            [position + offset for position in x],
            values,
            width=width,
            label=label,
            color=color,
        )
        axes[0].bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8)
    axes[0].set_xticks(x, splits)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("coverage (%)")
    axes[0].set_title("Annotation sampling coverage")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(x, covered_counts, label="covered", color="#228833")
    axes[1].bar(
        x,
        missed_counts,
        bottom=covered_counts,
        label="never covered",
        color="#CC6677",
    )
    axes[1].set_xticks(x, splits)
    axes[1].set_ylabel("source annotations")
    axes[1].set_title("Labels represented at least once")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle("Coverage-tiling annotation audit")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return output


def save_empty_image_balance_summary(summary: dict[str, dict[str, Any]], output: Path) -> Path:
    """Plot annotated/background image distributions before and after balancing."""

    splits = list(summary)
    categories = ["annotated", "background"]
    colors = {"annotated": "#4477AA", "background": "#CC6677"}
    fig, axes = plt.subplots(1, 2, figsize=(max(9, len(splits) * 2.4), 4.5), sharey=True)
    for ax, phase in zip(axes, ("before", "after")):
        x = list(range(len(splits)))
        width = 0.36
        for offset, category in zip((-width / 2, width / 2), categories):
            values = [int(summary[split][phase][category]) for split in splits]
            bars = ax.bar(
                [value + offset for value in x],
                values,
                width=width,
                label=category,
                color=colors[category],
            )
            ax.bar_label(bars, padding=2, fontsize=8)
        ax.set_xticks(x, splits)
        ax.set_title(phase.capitalize())
        ax.set_xlabel("split")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("images")
    axes[1].legend(frameon=False)
    fig.suptitle("Annotated and background image distribution")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return output


def save_class_removal_preview(
    sample: Sample,
    class_mapping: dict[int, int],
    task: Task,
    before_metadata: DatasetMetadata,
    after_metadata: DatasetMetadata,
    output: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _draw_sample(axes[0], sample, task, before_metadata)
    axes[0].set_title("Before")
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
    _draw_sample(axes[1], after, task, after_metadata)
    axes[1].set_title("After")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", dpi=140)
    plt.close(fig)
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

    columns = 2
    rows = max(1, math.ceil(len(items) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(14, 6 * rows), squeeze=False)
    flat = axes.flatten()
    visualization_options = normalize_visualize_kwargs(visualize_kwargs)
    for ax, (sample, boxes, status) in zip(flat, items):
        _draw_sample(
            ax,
            sample,
            task,
            metadata,
            **visualization_options,
        )
        for index, (left, top, right, bottom) in enumerate(boxes):
            ax.add_patch(
                patches.Rectangle(
                    (left, top),
                    right - left,
                    bottom - top,
                    fill=False,
                    edgecolor="#00d4ff",
                    linewidth=1.4,
                )
            )
            if len(boxes) <= 20:
                ax.text(
                    left + 3,
                    top + 12,
                    str(index),
                    color="white",
                    fontsize=7,
                    bbox={"facecolor": "black", "alpha": 0.65, "pad": 1},
                )
        if not boxes:
            ax.text(
                0.02,
                0.04,
                "PASS-THROUGH · NO CROP",
                transform=ax.transAxes,
                color="white",
                fontsize=9,
                bbox={"facecolor": "#167c3a", "alpha": 0.9, "pad": 4},
            )
        title = _sample_title(
            sample,
            visualization_options.get("label_fn"),
            default=(
                f"{status} · {sample.split} · {sample.width}×{sample.height}\n"
                f"{sample.relative_path} · {len(sample.annotations)} annotations"
            ),
        )
        if title is None:
            ax.set_title("")
        else:
            ax.set_title(title, fontsize=8)
    for ax in flat[len(items) :]:
        ax.axis("off")
    pass_through = sum(not boxes for _, boxes, _ in items)
    tiled = len(items) - pass_through
    fig.suptitle(
        f"{mode.capitalize()} tiling preview — "
        f"{pass_through} small pass-through source(s), {tiled} tiled source(s)\n"
        "cyan rectangles = output crop windows",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", dpi=140)
    plt.close(fig)
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


def _display_or_print(fig, save_to: Path | None) -> None:
    displayed = False
    try:
        from IPython import get_ipython
        from IPython.display import display

        if get_ipython() is not None:
            display(fig)
            displayed = True
    except Exception:
        pass
    if save_to and not displayed:
        print(f"Visualization: {save_to}")
    plt.close(fig)


def display_report(path: Path) -> None:
    """Display the canonical report PNG inline without redrawing its contents."""

    try:
        from IPython import get_ipython

        if get_ipython() is None:
            print(f"Visualization: {path}")
            return
    except Exception:
        print(f"Visualization: {path}")
        return
    with Image.open(path) as opened:
        report = opened.convert("RGB").copy()
    aspect = report.height / max(1, report.width)
    fig = plt.figure(figsize=(16, 16 * aspect), frameon=False)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(report)
    ax.axis("off")
    _display_or_print(fig, path)
