from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .models import Annotation, DatasetMetadata, Sample, Task

SPLIT_COLORS = {"train": "#2ca02c", "val": "#ff7f0e", "test": "#1f77b4"}
ANNOTATION_COLORS = (
    "#ff00ff",  # magenta
    "#7fff00",  # chartreuse
    "#ff5f00",  # vivid orange
    "#ffff00",  # yellow
    "#ff1493",  # deep pink
)


def visualize_samples(
    samples: list[Sample],
    task: Task,
    metadata: DatasetMetadata,
    *,
    split: str | None,
    n: int,
    seed: int,
    columns: int,
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
        _draw_sample(ax, sample, task, metadata)
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


def _draw_sample(ax, sample: Sample, task: Task, metadata: DatasetMetadata) -> None:
    with Image.open(sample.image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        ax.imshow(image)
    for annotation in sample.annotations:
        color = ANNOTATION_COLORS[annotation.class_id % len(ANNOTATION_COLORS)]
        name = metadata.names.get(annotation.class_id, str(annotation.class_id))
        if annotation.bbox is not None and task in {Task.DETECT, Task.POSE}:
            x1, y1, x2, y2 = annotation.bbox
            ax.add_patch(
                patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="white",
                    linewidth=3.5,
                )
            )
            ax.add_patch(
                patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.8,
                )
            )
            ax.text(x1, max(0, y1 - 3), name, color="white", fontsize=7, bbox={"facecolor": color, "alpha": 0.8, "pad": 2})
        if annotation.polygon:
            ax.add_patch(
                patches.Polygon(
                    annotation.polygon,
                    closed=True,
                    fill=False,
                    edgecolor="white",
                    linewidth=3.5,
                )
            )
            ax.add_patch(
                patches.Polygon(
                    annotation.polygon,
                    closed=True,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.8,
                )
            )
            xs, ys = zip(*annotation.polygon)
            left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
            for outline, line_width in (("white", 3.2), (color, 1.6)):
                ax.add_patch(
                    patches.Rectangle(
                        (left, top),
                        right - left,
                        bottom - top,
                        fill=False,
                        edgecolor=outline,
                        linewidth=line_width,
                        linestyle="--",
                    )
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
                    ax.plot([x1, x2], [y1, y2], color=color, linewidth=1.0, alpha=0.8)
            for idx, (x, y, visibility) in enumerate(annotation.keypoints):
                if visibility == 0:
                    continue
                ax.scatter([x], [y], s=18, c=[color], edgecolors="white", linewidths=0.5)
                if idx < len(names):
                    ax.text(x + 2, y + 2, names[idx], fontsize=5, color="white", bbox={"facecolor": "black", "alpha": 0.5, "pad": 1})
        if annotation.point and annotation.radius is not None:
            ax.add_patch(
                patches.Circle(annotation.point, radius=annotation.radius, linewidth=1.0, edgecolor="red", facecolor="none")
            )
            ax.scatter([annotation.point[0]], [annotation.point[1]], s=8, c="red")
    ax.set_title(
        f"{sample.relative_path}\n{sample.width}×{sample.height} | {len(sample.annotations)} annotations | {sample.split}",
        fontsize=8,
    )
    ax.axis("off")


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
    sample: Sample, removed: set[int], task: Task, metadata: DatasetMetadata, output: Path
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _draw_sample(axes[0], sample, task, metadata)
    axes[0].set_title("Before")
    after = Sample(
        sample.image_path,
        sample.relative_path,
        sample.split,
        sample.width,
        sample.height,
        [a for a in sample.annotations if a.class_id not in removed],
    )
    _draw_sample(axes[1], after, task, metadata)
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
) -> Path:
    """Show one small pass-through source and up to three tiled sources."""

    columns = 2
    rows = max(1, math.ceil(len(items) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(14, 6 * rows), squeeze=False)
    flat = axes.flatten()
    for ax, (sample, boxes, status) in zip(flat, items):
        _draw_sample(ax, sample, task, metadata)
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
        ax.set_title(
            f"{status} · {sample.split} · {sample.width}×{sample.height}\n"
            f"{sample.relative_path} · {len(sample.annotations)} annotations",
            fontsize=8,
        )
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
    lines.extend(
        [
            (
                f"Background crops from this source: {len(background_boxes)} | "
                f"dataset target: {float(settings['background_ratio']):.1%} | "
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
    try:
        from IPython import get_ipython
        from IPython.display import display

        if get_ipython() is not None:
            display(fig)
            return
    except Exception:
        pass
    if save_to:
        print(f"Visualization: {save_to}")
    plt.close(fig)
