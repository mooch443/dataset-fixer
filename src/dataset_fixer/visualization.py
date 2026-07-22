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
    colors = plt.cm.tab20.colors
    for annotation in sample.annotations:
        color = colors[annotation.class_id % len(colors)]
        name = metadata.names.get(annotation.class_id, str(annotation.class_id))
        if annotation.bbox is not None and task in {Task.DETECT, Task.POSE}:
            x1, y1, x2, y2 = annotation.bbox
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.5))
            ax.text(x1, max(0, y1 - 3), name, color="white", fontsize=7, bbox={"facecolor": color, "alpha": 0.8, "pad": 2})
        if annotation.polygon:
            ax.add_patch(patches.Polygon(annotation.polygon, closed=True, fill=False, edgecolor=color, linewidth=1.5))
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


def save_split_preview(samples: list[Sample], assignments: dict[str, str], output: Path) -> Path:
    chosen: list[Sample] = []
    for split in ("train", "val", "test"):
        match = next((s for s in samples if assignments.get(str(s.image_path)) == split), None)
        if match:
            chosen.append(match)
    columns = max(1, len(chosen))
    fig, axes = plt.subplots(1, columns, figsize=(6 * columns, 5), squeeze=False)
    for ax, sample in zip(axes.flatten(), chosen):
        with Image.open(sample.image_path) as opened:
            ax.imshow(ImageOps.exif_transpose(opened))
        target = assignments[str(sample.image_path)]
        ax.set_title(f"{sample.relative_path}\n→ {target}", color=SPLIT_COLORS[target])
        ax.axis("off")
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


def save_grid_preview(sample: Sample, boxes: list[tuple[int, int, int, int]], output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 8))
    with Image.open(sample.image_path) as opened:
        ax.imshow(ImageOps.exif_transpose(opened))
    for idx, (left, top, right, bottom) in enumerate(boxes):
        ax.add_patch(
            patches.Rectangle(
                (left, top), right - left, bottom - top, fill=False, edgecolor="#00d4ff", linewidth=1.2
            )
        )
        ax.text(left + 3, top + 12, str(idx), color="white", fontsize=7, bbox={"facecolor": "black", "alpha": 0.6})
    ax.set_title(f"Grid tiling preview — {sample.relative_path} ({len(boxes)} tiles)")
    ax.axis("off")
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
        words = line.split()
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
    background_boxes: list[tuple[int, int, int, int]],
    output: Path,
    settings: dict[str, Any],
) -> Path:
    with Image.open(sample.image_path) as opened:
        annotated = ImageOps.exif_transpose(opened).convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = annotated.size
    min_dim = min(width, height)
    marker_radius = max(14, int(min_dim * 0.014))
    center_radius = max(4, int(marker_radius * 0.28))
    line_width = max(4, int(min_dim * 0.0035))
    underlay = line_width + 4
    font_size = max(18, int(min_dim * 0.015))
    legend_size = max(18, int(min_dim * 0.016))
    bg_size = max(16, int(min_dim * 0.012))
    font, legend_font, bg_font = _font(font_size), _font(legend_size), _font(bg_size)

    cyan = (0, 190, 255)
    for index, box in enumerate(background_boxes):
        left, top, right, bottom = box
        draw.rectangle(box, fill=cyan + (35,))
        draw.rectangle(box, outline=(255, 255, 255, 235), width=underlay)
        draw.rectangle(box, outline=cyan + (230,), width=line_width)
        text = f"bg {index}"
        tx, ty = left + max(6, line_width * 2), top + max(6, line_width * 2)
        bounds = draw.textbbox((0, 0), text, font=bg_font)
        tx = min(max(tx, 8), max(8, width - (bounds[2] - bounds[0]) - 20))
        ty = min(max(ty, 8), max(8, height - (bounds[3] - bounds[1]) - 20))
        _text_box(draw, (tx, ty), text, bg_font, cyan + (230,), box_fill=(0, 0, 0, 135), pad=max(5, bg_size // 4))

    for label_idx, annotation in enumerate(sample.annotations):
        if annotation.point is None:
            continue
        x, y = map(lambda v: int(round(v)), annotation.point)
        x, y = min(max(x, 0), width - 1), min(max(y, 0), height - 1)
        count = coverage_counts.get(label_idx, 0)
        target = coverage_targets.get(label_idx, settings["target_coverage_per_label"])
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
        radius = int(settings["fixed_polo_radius_px"])
        if radius >= 3:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color + (130,), width=max(2, line_width // 2))
        text = f"{count}/{target}"
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

    dense = sum(coverage_targets.get(i) == settings["target_coverage_per_label"] for i in range(len(sample.annotations)))
    sparse = len(sample.annotations) - dense
    hit = sum(coverage_counts.get(i, 0) >= 1 for i in range(len(sample.annotations)))
    percent_hit = 100 * hit / len(sample.annotations) if sample.annotations else 0
    cap = settings["max_total_tiles_per_source_image"]
    lines = [
        f"{sample.split.upper()} coverage: {sample.image_path.stem}",
        f"dense target: {settings['target_coverage_per_label']}, sparse target: {settings['sparse_coverage_per_label']}",
        f"dense labels: {dense}, sparse labels: {sparse}",
        f"labels hit at least once: {hit}/{len(sample.annotations)} ({percent_hit:.1f}%)",
        f"background tiles: {len(background_boxes)}",
        f"background ratio: {settings['max_bg_ratio']}",
        f"fixed label radius: {settings['fixed_polo_radius_px']}px",
        f"dense radius: {settings['dense_neighbor_radius_px']}px, min neighbors: {settings['min_nearby_labels_for_full_coverage']}",
        f"max total tiles/source: {cap if cap is not None else 'disabled'}",
        "green=complete, yellow/orange=partial, red=low",
        "cyan rectangles=background tiles",
        "circle=train, square=val/test",
    ]
    # Shrink and wrap the legend as needed so it never leaves the raster,
    # including unusually wide-but-short orchard imagery.
    fitted_lines = lines
    for candidate_size in range(legend_size, 6, -1):
        candidate_font = _font(candidate_size)
        candidate_lines = _wrap_legend_lines(draw, lines, candidate_font, max(40, width - 40))
        heights = [draw.textbbox((0, 0), line, font=candidate_font)[3] for line in candidate_lines]
        candidate_gap = max(2, candidate_size // 4)
        if sum(heights) + len(heights) * (candidate_gap + 8) <= height - 24:
            legend_font = candidate_font
            legend_size = candidate_size
            fitted_lines = candidate_lines
            break
    y_cursor = 12
    gap = max(2, legend_size // 4)
    for line in fitted_lines:
        _text_box(draw, (12, y_cursor), line, legend_font, (255, 255, 255, 255), pad=max(6, legend_size // 4))
        bounds = draw.textbbox((12, y_cursor), line, font=legend_font)
        y_cursor += bounds[3] - bounds[1] + gap + 8
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(annotated, overlay).convert("RGB").save(output, quality=int(settings["jpeg_quality"]))
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
