"""Canonical pandas-backed dataset comparison and report rendering."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import altair as alt
import numpy as np
import pandas as pd
from PIL import Image

from .comparison.reporting import combine_report_plots
from .models import DatasetMetadata, Sample, Task
from .static_rendering import format_label, save_chart
from .tabular import chart_data, frame, stable_sort
from .utils import settings_fingerprint, sha256_file
from .visualization import (
    VisualizationItem,
    VisualizationOptions,
    VisualizationPanel,
    display_report,
    normalize_visualize_kwargs,
    render_annotated_sample,
    visualize_records,
)

if TYPE_CHECKING:
    from .dataset import Dataset


@dataclass(frozen=True)
class DatasetReportState:
    """One physical dataset represented once for tables and visualization."""

    root: Path
    name: str
    task: str
    format_name: str
    classes: dict[int, str]
    rows: tuple[dict[str, Any], ...]
    metadata: DatasetMetadata | None = None
    coverage: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class DatasetComparisonResult:
    """Lineage-aware dataset differences and their shared overview image.

    Parameters:
        baseline: Baseline dataset name.
        candidate: Candidate dataset name.
        overview: Aggregate before/after counts and deltas.
        splits: Per-split counts and deltas.
        classes: Per-class annotation counts and deltas.
        images: Aligned image rows classified by change status.
        plot: Saved comparison image, when a destination was supplied.
    """

    baseline: str
    candidate: str
    overview: pd.DataFrame
    splits: pd.DataFrame
    classes: pd.DataFrame
    images: pd.DataFrame
    plot: Path | None = None

    def __post_init__(self) -> None:
        for name in ("overview", "splits", "classes", "images"):
            object.__setattr__(self, name, frame(getattr(self, name)))

    def __repr__(self) -> str:
        changed = int(self.images["status"].ne("unchanged").sum()) if len(self.images) else 0
        return (
            f"DatasetComparisonResult(baseline={self.baseline!r}, "
            f"candidate={self.candidate!r}, changed_images={changed})"
        )


def dataset_report_state(dataset: "Dataset") -> DatasetReportState:
    """Build the canonical comparison state for a physical dataset."""

    if dataset._plan and not dataset._projection_exact:
        raise ValueError("Deferred dataset operations must be exported before comparison")
    return report_state_from_samples(
        root=dataset.location,
        name=dataset.name,
        task=dataset.task.value,
        format_name=dataset.format,
        classes=dataset.classes,
        samples=dataset._samples,
        metadata=dataset._metadata,
        mask_paths=dataset._mask_paths,
        details=dataset.provenance.values(),
        coverage=(dataset.manifest.get("audits") or {}).get("coverage.source_coverage"),
    )


def report_state_from_samples(
    *,
    root: Path,
    name: str,
    task: str,
    format_name: str,
    classes: Mapping[int, str],
    samples: Iterable[Sample],
    metadata: DatasetMetadata | None = None,
    mask_paths: Mapping[Path, Path] | None = None,
    details: Iterable[Mapping[str, Any]] = (),
    coverage: Mapping[str, Any] | None = None,
) -> DatasetReportState:
    """Construct one independently owned state at a serialization boundary."""

    root = root.resolve()
    by_path = {
        _path(root, row["output_image"]): dict(row)
        for row in details
        if row.get("output_image")
    }
    masks = {image.resolve(): mask.resolve() for image, mask in (mask_paths or {}).items()}
    rows = []
    for sample in samples:
        image = sample.image_path.resolve()
        row = by_path.get(image, {}).copy()
        counts = pd.Series(
            [annotation.class_id for annotation in sample.annotations], dtype="Int64"
        ).value_counts(sort=False)
        mask = masks.get(image)
        row.update(
            sample=sample,
            image_path=image,
            mask_path=mask,
            width=sample.width,
            height=sample.height,
            pixels=sample.width * sample.height,
            output_image=_relative(image, root),
            relative_path=sample.relative_path.as_posix(),
            output_split=sample.split,
            output_annotation_count=len(sample.annotations),
            output_has_labels=bool(sample.annotations),
            output_sha256=row.get("output_sha256")
            or row.get("output_image_sha256")
            or sample.source_sha256,
            output_mask=_relative(mask, root) if mask else row.get("output_mask"),
            output_mask_sha256=row.get("output_mask_sha256")
            or (sha256_file(mask) if mask else None),
            annotation_fingerprint=settings_fingerprint([asdict(value) for value in sample.annotations]),
            class_counts={int(key): int(value) for key, value in counts.items()},
        )
        rows.append(row)
    return DatasetReportState(
        root, name, task, format_name, dict(classes), tuple(rows), metadata, coverage
    )


def compare_dataset_states(
    baseline: DatasetReportState,
    candidate: DatasetReportState,
    *,
    destination: str | Path | None = None,
    visualize_kwargs: Mapping[str, Any] | None = None,
    show: bool = False,
    width: int = 2400,
) -> DatasetComparisonResult:
    """Compare and optionally render states through the one canonical path."""

    before, after = _tables(baseline, candidate)
    images = before.merge(
        after, on=["key", "occurrence"], how="outer", suffixes=("_before", "_after"), indicator=True
    )
    both = images["_merge"].eq("both")
    image_changed = (
        _different(images, "image_sha256")
        & images["image_sha256_before"].notna()
        & images["image_sha256_after"].notna()
    )
    changed = both & (
        image_changed
        | _different(images, "mask_sha256")
        | _different(images, "annotation_fingerprint")
        | _different(images, "annotation_count")
    )
    moved = both & ~changed & (
        _different(images, "split") | _different(images, "relative_path")
    )
    images["status"] = "unchanged"
    images.loc[images["_merge"].eq("left_only"), "status"] = "removed"
    images.loc[images["_merge"].eq("right_only"), "status"] = "added"
    images.loc[changed, "status"] = "modified"
    images.loc[moved, "status"] = "moved"
    columns = [
        "key", "occurrence", "status", "relative_path_before", "relative_path_after",
        "split_before", "split_after", "annotation_count_before", "annotation_count_after",
    ]
    result = DatasetComparisonResult(
        baseline.name,
        candidate.name,
        _overview(before, after, baseline, candidate),
        _splits(before, after),
        _classes(before, after, baseline.classes, candidate.classes),
        stable_sort(frame(images.reindex(columns=columns)), ["status", "key", "occurrence"]),
    )
    output = _destination(destination)
    if output is not None:
        render_dataset_states(
            candidate,
            output=output,
            baseline=baseline,
            visualize_kwargs=visualize_kwargs,
            width=width,
        )
        result = replace(result, plot=output)
        if show:
            display_report(output)
    return result


def compare_datasets(
    baseline: "Dataset",
    candidate: "Dataset",
    *,
    destination: str | Path | None = None,
    visualize_kwargs: Mapping[str, Any] | None = None,
    show: bool = True,
    width: int = 2400,
) -> DatasetComparisonResult:
    """Public Dataset adapter around :func:`compare_dataset_states`."""

    temporary = tempfile.TemporaryDirectory(prefix="dataset-fixer-comparison-") if destination is None and show else None
    output = Path(temporary.name) / "plots.png" if temporary else destination
    result = compare_dataset_states(
        dataset_report_state(baseline),
        dataset_report_state(candidate),
        destination=output,
        visualize_kwargs=visualize_kwargs,
        show=show,
        width=width,
    )
    if temporary:
        temporary.cleanup()
        result = replace(result, plot=None)
    return result


def render_dataset_overview(
    dataset: "Dataset",
    *,
    destination: str | Path | None = None,
    visualize_kwargs: Mapping[str, Any] | None = None,
    show: bool = True,
    width: int = 2400,
) -> Path | None:
    """Render a one-dataset view through the comparison report pipeline."""

    options = normalize_visualize_kwargs(visualize_kwargs)
    existing = dataset.location / "reports" / "plots.png"
    if not options and existing.is_file():
        output = existing
        if destination is not None:
            output = Path(destination).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(existing, output)
        if show:
            display_report(output)
        return output
    temporary = tempfile.TemporaryDirectory(prefix="dataset-fixer-report-") if destination is None else None
    output = Path(temporary.name) / "plots.png" if temporary else _destination(destination)
    rendered = render_dataset_states(
        dataset_report_state(dataset), output=output, visualize_kwargs=options, width=width
    )
    if rendered and show:
        display_report(rendered)
    if temporary:
        temporary.cleanup()
        return None
    return rendered


def render_dataset_states(
    candidate: DatasetReportState,
    *,
    output: Path,
    baseline: DatasetReportState | None = None,
    visualize_kwargs: Mapping[str, Any] | None = None,
    width: int = 2400,
) -> Path:
    """Render one or two states using shared pandas charts and image cards."""

    states = (baseline, candidate) if baseline is not None else (candidate,)
    options = normalize_visualize_kwargs(visualize_kwargs)
    with tempfile.TemporaryDirectory(prefix="dataset-fixer-report-parts-") as temporary:
        parts = Path(temporary)
        save_chart(_profile_chart(states), parts / "dataset statistics.png")
        coverage = _coverage_chart(states)
        if coverage is not None:
            save_chart(coverage, parts / "source coverage.png")
        examples = _example_chart(states, options)
        if examples is not None:
            save_chart(examples, parts / "annotated examples.png")
        rendered = combine_report_plots((parts,), output, width=width)
    if rendered is None:
        raise RuntimeError("Dataset comparison produced no report panels")
    return rendered


def _profile_chart(states: tuple[DatasetReportState, ...]) -> alt.TopLevelMixin:
    rows = pd.concat(
        [frame(state.rows).assign(dataset=state.name) for state in states], ignore_index=True
    )
    rows["dataset_label"] = rows["dataset"].map(_axis_label)
    split_order = [
        split for split in ("train", "val", "test") if split in set(rows["output_split"])
    ] + sorted(set(rows["output_split"]) - {"train", "val", "test"})
    pixels = rows.assign(megapixels=rows["pixels"].astype(float) / 1_000_000)
    grouped = rows.groupby(["dataset", "dataset_label", "output_split"], sort=False).agg(
        images=("output_image", "size"), annotations=("output_annotation_count", "sum")
    ).reset_index().rename(columns={"output_split": "split"})
    grouped["split_label"] = grouped["split"].map(_axis_label)
    composition = (
        rows.assign(kind=np.where(rows["output_has_labels"].astype(bool), "annotated", "background"))
        .groupby(["dataset", "dataset_label", "output_split", "kind"], sort=False).size().rename("images")
        .reset_index().rename(columns={"output_split": "split"})
    )
    composition["split_label"] = composition["split"].map(_axis_label)
    classes = _class_profile(states)
    dataset_order = [state.name for state in states]
    dataset_label_order = [_axis_label(value) for value in dataset_order]
    split_label_order = [_axis_label(value) for value in split_order]
    color = alt.Color("dataset:N", sort=dataset_order, title="dataset")
    pixels_chart = alt.Chart(chart_data(pixels[["dataset", "dataset_label", "megapixels"]])).mark_boxplot(size=44).encode(
        x=alt.X("dataset_label:N", sort=dataset_label_order, title=None, axis=_horizontal_axis()),
        y=alt.Y("megapixels:Q", title="megapixels per image"),
        color=color,
        tooltip=["dataset:N", alt.Tooltip("megapixels:Q", format=".3f")],
    ).properties(width=440, height=280, title="Image-pixel distribution")

    def split_panel(metric: str, title: str) -> alt.Chart:
        return alt.Chart(chart_data(grouped)).mark_bar().encode(
            x=alt.X("split_label:N", sort=split_label_order, title="split", axis=_horizontal_axis()),
            xOffset=alt.XOffset("dataset:N", sort=dataset_order),
            y=alt.Y(f"{metric}:Q", title=metric),
            color=color,
            tooltip=["dataset:N", "split:N", alt.Tooltip(f"{metric}:Q", format=",")],
        ).properties(width=440, height=280, title=title)

    composition_chart = alt.Chart(chart_data(composition)).mark_bar().encode(
        x=alt.X("split_label:N", sort=split_label_order, title="split", axis=_horizontal_axis()),
        xOffset=alt.XOffset("dataset:N", sort=dataset_order),
        y=alt.Y("images:Q", title="images", stack="zero"),
        color=alt.Color(
            "kind:N", sort=["annotated", "background"],
            scale=alt.Scale(domain=["annotated", "background"], range=["#2f9e5f", "#b9c2cd"]),
        ),
        column=alt.Column(
            "dataset_label:N",
            sort=dataset_label_order,
            title=None,
            header=alt.Header(labelExpr="split(datum.label, '\\n')", labelLineHeight=13),
        ),
    ).properties(width=220, height=260, title="Annotated/background images per split")
    lower: list[alt.TopLevelMixin] = [composition_chart]
    if len(classes):
        classes["class_label"] = classes["class_name"].map(_axis_label)
        class_order = list(dict.fromkeys(classes["class_label"]))
        lower.append(
            alt.Chart(chart_data(classes)).mark_bar().encode(
                x=alt.X("class_label:N", sort=class_order, title="class", axis=_horizontal_axis()),
                xOffset=alt.XOffset("dataset:N", sort=dataset_order),
                y=alt.Y("annotations:Q", title="annotated objects"),
                color=color,
                tooltip=["dataset:N", "class_name:N", alt.Tooltip("annotations:Q", format=",")],
            ).properties(width=max(700, 70 * classes["class_name"].nunique()), height=280, title="Annotated objects per class")
        )
    charts: list[alt.TopLevelMixin] = [
        alt.hconcat(pixels_chart, split_panel("images", "Images per split"), split_panel("annotations", "Annotated objects per split"), spacing=34),
        alt.hconcat(*lower, spacing=34),
    ]
    title = " vs ".join(dataset_order)
    return alt.vconcat(*charts, spacing=38).properties(
        title=alt.TitleParams(text=title, subtitle=[
            f"{state.name}: task={state.task}, format={state.format_name}, classes={len(state.classes)}"
            for state in states
        ])
    ).resolve_scale(color="independent")


def _class_profile(states: tuple[DatasetReportState, ...]) -> pd.DataFrame:
    rows = []
    for state in states:
        totals: dict[int, int] = {}
        for counts in frame(state.rows)["class_counts"]:
            for key, value in counts.items():
                totals[int(key)] = totals.get(int(key), 0) + int(value)
        rows.extend(
            {"dataset": state.name, "class_id": key, "class_name": state.classes.get(key, str(key)), "annotations": totals.get(key, 0)}
            for key in sorted(state.classes)
        )
    return frame(rows)


def _coverage_chart(states: tuple[DatasetReportState, ...]) -> alt.TopLevelMixin | None:
    summaries = []
    dataset_order = [state.name for state in states]
    for state in states:
        coverage = state.coverage
        if not coverage:
            continue
        summaries.extend(
            {"dataset": state.name, "metric": metric, "percent": float(coverage.get(key) or 0)}
            for metric, key in (
                ("labels represented", "source_label_coverage_percent"),
                ("source pixels represented", "source_image_space_coverage_percent"),
                ("source images represented", "source_image_representation_percent"),
            )
        )
    if not summaries:
        return None
    summary_data = frame(summaries)
    summary_data["metric_label"] = summary_data["metric"].map(_axis_label)
    bars = alt.Chart(chart_data(summary_data)).mark_bar().encode(
        x=alt.X("metric_label:N", title=None, axis=_horizontal_axis()),
        xOffset="dataset:N",
        y=alt.Y("percent:Q", scale=alt.Scale(domain=[0, 100]), title="percent"),
        color=alt.Color(
            "dataset:N",
            sort=dataset_order,
            scale=alt.Scale(
                domain=dataset_order,
                range=["#2f6fb0", "#ef8a17", "#2f9e5f", "#8e63ce"],
            ),
        ),
        tooltip=["dataset:N", "metric:N", alt.Tooltip("percent:Q", format=".1f")],
    ).properties(width=720, height=270, title="Source coverage")
    heatmap_rows = [
        row
        for state in states
        if state.coverage
        for row in [_coverage_heatmap_row(state)]
        if row is not None
    ]
    if not heatmap_rows:
        return bars
    heatmaps = alt.vconcat(*heatmap_rows, spacing=28).properties(
        title="Annotation-position distributions"
    )
    return alt.vconcat(bars, heatmaps, spacing=34)


def _coverage_heatmap_row(state: DatasetReportState) -> alt.TopLevelMixin | None:
    panels = []
    for title, key in (
        ("Source", "label_positions"),
        ("Output", "output_label_positions"),
    ):
        histogram = (state.coverage or {}).get(key) or {}
        grid = histogram.get("labels") or []
        if not grid or not grid[0]:
            continue
        missing = histogram.get("uncovered") or []
        rows, columns = len(grid), len(grid[0])
        cells = [
            {
                "dataset": state.name,
                "coordinate": title.lower(),
                "x": x,
                "y": y,
                "labels": int(value),
                "uncovered": int(missing[y][x]) if missing else 0,
            }
            for y, line in enumerate(grid)
            for x, value in enumerate(line)
        ]
        cell_size = 42
        panels.append(
            alt.Chart(chart_data(cells)).mark_rect().encode(
                x=alt.X(
                    "x:O",
                    sort=list(range(columns)),
                    axis=None,
                    scale=alt.Scale(paddingInner=0, paddingOuter=0),
                ),
                y=alt.Y(
                    "y:O",
                    sort=list(range(rows)),
                    axis=None,
                    scale=alt.Scale(paddingInner=0, paddingOuter=0),
                ),
                color=alt.Color("labels:Q", scale=alt.Scale(scheme="blues")),
                stroke=alt.condition(
                    alt.datum.uncovered > 0,
                    alt.value("#b00020"),
                    alt.value("white"),
                ),
                tooltip=["dataset:N", "coordinate:N", "labels:Q", "uncovered:Q"],
            ).properties(
                width=columns * cell_size,
                height=rows * cell_size,
                title=title,
            )
        )
    if not panels:
        return None
    return alt.hconcat(*panels, spacing=30).properties(
        title=alt.TitleParams(text=state.name, anchor="middle")
    )


def _axis_label(value: Any) -> str:
    return "\n".join(
        format_label(
            str(value),
            mode="wrap",
            maximum=18,
            wrap_width=18,
            maximum_lines=3,
        )
    )


def _horizontal_axis() -> alt.Axis:
    return alt.Axis(
        labelAngle=0,
        labelAlign="center",
        labelBaseline="top",
        labelExpr="split(datum.label, '\\n')",
        labelLineHeight=13,
        labelLimit=180,
    )


def _example_chart(
    states: tuple[DatasetReportState, ...], options: Mapping[str, Any]
) -> alt.TopLevelMixin | None:
    selected: list[tuple[DatasetReportState, dict[str, Any]]] = []
    for state in states:
        data = frame(state.rows).sort_values(["output_split", "relative_path"], kind="stable")
        for _, split in data.groupby("output_split", sort=False):
            annotated = split[split["output_has_labels"].astype(bool)]
            background = split[~split["output_has_labels"].astype(bool)]
            chosen = pd.concat([annotated, background], ignore_index=True).head(4)
            selected.extend((state, row) for row in chosen.to_dict(orient="records"))
    if not selected:
        return None
    label_fn = options.get("label_fn")

    def prepare(value: tuple[DatasetReportState, dict[str, Any]]) -> VisualizationItem:
        state, row = value
        sample = row["sample"]
        with Image.open(sample.image_path) as opened:
            source = np.asarray(opened.convert("RGB"))
        mask = None
        if row.get("mask_path") and Path(row["mask_path"]).is_file():
            with Image.open(row["mask_path"]) as opened:
                mask = np.asarray(opened.convert("L")) > 0
        rendered = (
            source
            if mask is not None
            else np.asarray(render_annotated_sample(
                sample,
                Task.parse(state.task) or Task.DETECT,
                state.metadata or DatasetMetadata(names=state.classes),
                show_names=False,
                line_width=options.get("line_width"),
                outline_width=options.get("outline_width"),
            ))
        )
        label = label_fn(sample.image_path) if label_fn else (
            f"{state.name} · {sample.split} · {sample.relative_path} · "
            f"{len(sample.annotations)} object(s)"
        )
        if label is not None and not isinstance(label, str):
            raise TypeError("label_fn must return a string or None")
        return VisualizationItem(
            image_path=sample.image_path,
            label=label or "",
            panels=(VisualizationPanel(title=state.name, image=rendered, mask=mask),),
            foreground=np.ones(source.shape[:2], dtype=bool),
        )

    return visualize_records(
        selected,
        options=VisualizationOptions(
            samples=None, columns=4, panel_size=3.2, show=False,
            label_mode=options.get("label_mode", "middle"),
            line_width=options.get("line_width"),
            outline_width=options.get("outline_width", 1.0),
        ),
        prepare=prepare,
        title="Deterministic annotated examples by dataset and split",
    )


def _tables(
    baseline: DatasetReportState, candidate: DatasetReportState
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = []
    for state in (baseline, candidate):
        data = frame(state.rows)
        output = data.get("output_image", pd.Series(dtype="string"))
        relative = data.get("relative_path", output).astype(str)
        physical = output.map(lambda value: str(_path(state.root, value)))
        original = data.get("original_sha256", pd.Series(pd.NA, index=data.index, dtype="string"))
        image_hash = data.get("output_sha256", pd.Series(pd.NA, index=data.index, dtype="string"))
        key = ("original:" + original.astype("string")).where(original.notna(), "path:" + relative)
        values.append(frame(pd.DataFrame({
            "key": key,
            "physical_path": physical,
            "parent_path": data.get("parent_image", pd.Series(pd.NA, index=data.index)),
            "relative_path": relative,
            "split": data.get("output_split", pd.Series("unknown", index=data.index)).fillna("unknown"),
            "image_sha256": image_hash,
            "mask_sha256": data.get("output_mask_sha256", pd.Series(pd.NA, index=data.index)),
            "annotation_fingerprint": data.get("annotation_fingerprint", pd.Series(pd.NA, index=data.index)),
            "annotation_count": data.get("output_annotation_count", pd.Series(0, index=data.index)).fillna(0),
            "annotated": data.get("output_has_labels", pd.Series(False, index=data.index)).fillna(False),
            "class_counts": data.get("class_counts", pd.Series([{} for _ in data.index], index=data.index)),
        })))
    parent_paths = set(values[1]["parent_path"].dropna()) & set(values[0]["physical_path"])
    values[0].loc[values[0]["physical_path"].isin(parent_paths), "key"] = (
        "parent:" + values[0].loc[values[0]["physical_path"].isin(parent_paths), "physical_path"]
    )
    values[1].loc[values[1]["parent_path"].isin(parent_paths), "key"] = (
        "parent:" + values[1].loc[values[1]["parent_path"].isin(parent_paths), "parent_path"].astype(str)
    )
    common = set(values[0]["relative_path"]) & set(values[1]["relative_path"])
    for data in values:
        shared = data["relative_path"].isin(common)
        data.loc[shared, "key"] = "path:" + data.loc[shared, "relative_path"].astype(str)
        data.sort_values(["key", "relative_path", "split"], kind="stable", inplace=True)
        data["occurrence"] = data.groupby("key", sort=False).cumcount()
    return values[0], values[1]


def _different(data: pd.DataFrame, column: str) -> pd.Series:
    before, after = data[f"{column}_before"], data[f"{column}_after"]
    sentinel = "\0dataset-fixer-missing"
    return ~before.astype("string").fillna(sentinel).eq(after.astype("string").fillna(sentinel))


def _overview(
    before: pd.DataFrame,
    after: pd.DataFrame,
    baseline: DatasetReportState,
    candidate: DatasetReportState,
) -> pd.DataFrame:
    def totals(data: pd.DataFrame, classes: int) -> dict[str, int]:
        annotated = int(data["annotated"].sum())
        return {
            "images": len(data), "annotated_images": annotated,
            "background_images": len(data) - annotated,
            "annotations": int(data["annotation_count"].sum()), "classes": classes,
            "splits": int(data["split"].nunique()),
        }

    left, right = totals(before, len(baseline.classes)), totals(after, len(candidate.classes))
    return frame(
        {"metric": key, "baseline": value, "candidate": right[key], "delta": right[key] - value}
        for key, value in left.items()
    )


def _splits(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    def grouped(data: pd.DataFrame, suffix: str) -> pd.DataFrame:
        return data.groupby("split", sort=True).agg(
            **{f"images_{suffix}": ("key", "size"), f"annotated_{suffix}": ("annotated", "sum"),
               f"annotations_{suffix}": ("annotation_count", "sum")}
        )

    result = grouped(before, "before").join(grouped(after, "after"), how="outer").fillna(0).reset_index()
    for metric in ("images", "annotated", "annotations"):
        result[f"{metric}_delta"] = result[f"{metric}_after"] - result[f"{metric}_before"]
    return frame(result)


def _classes(
    before: pd.DataFrame,
    after: pd.DataFrame,
    before_names: Mapping[int, str],
    after_names: Mapping[int, str],
) -> pd.DataFrame:
    def totals(data: pd.DataFrame) -> dict[int, int]:
        result: dict[int, int] = {}
        for counts in data["class_counts"]:
            for key, value in counts.items():
                result[int(key)] = result.get(int(key), 0) + int(value)
        return result

    left, right = totals(before), totals(after)
    return frame(
        {"class_id": key, "name_before": before_names.get(key), "name_after": after_names.get(key),
         "annotations_before": left.get(key, 0), "annotations_after": right.get(key, 0),
         "annotations_delta": right.get(key, 0) - left.get(key, 0)}
        for key in sorted(set(before_names) | set(after_names))
    )


def _destination(destination: str | Path | None) -> Path | None:
    if destination is None:
        return None
    path = Path(destination).expanduser().resolve()
    if path.suffix.lower() == ".png":
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    return path / "plots.png"


def _path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return (path if path.is_absolute() else root / path).resolve()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
