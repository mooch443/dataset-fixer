from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import altair as alt
import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFont

from ..static_rendering import save_chart
from ..tabular import TableLike, chart_data, frame
from ..utils import to_jsonable
from ..visualization import (
    VisualizationItem,
    VisualizationOptions,
    VisualizationPanel,
    visualize_records,
)
from .plot_labels import (
    model_identity_card,
    model_identity_chart,
    model_identity_row_height,
    with_model_identities,
)
from .types import Cohort, Prediction

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def write_csv(path: Path, rows: TableLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _csv_frame(rows).to_csv(path, index=False, lineterminator="\r\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), indent=2, sort_keys=True), encoding="utf-8")


def write_tables(root: Path, ranking: TableLike) -> None:
    data = frame(ranking)
    serialized = _csv_frame(ranking)
    write_csv(root / "tables" / "model_comparison.csv", serialized)
    columns = [
        key
        for key in (
            "rank", "model", "backend", "score", "confidence", "postprocess",
            "support_images", "support_annotations", "support_clusters",
        )
        if key in data
    ]
    lines = [
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        " & ".join(columns) + " \\\\ \\hline",
        *(
            " & ".join(_latex(value) for value in row) + " \\\\"
        for row in serialized[columns].itertuples(index=False, name=None)
        ),
        "\\end{tabular}",
    ]
    (root / "tables" / "model_comparison.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    try:
        data.to_excel(root / "tables" / "model_comparison.xlsx", index=False)
    except ImportError:
        pass


def render_figures(
    root: Path,
    *,
    cohort: Cohort,
    ranking: list[dict[str, Any]],
    grid: list[dict[str, Any]],
    per_class: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    per_image: list[dict[str, Any]],
    pr_data: dict[str, Any],
    cache_audit: dict[str, Any],
    leakage_audit: dict[str, Any],
    metadata: dict[str, Any],
) -> list[str]:
    metadata = {**metadata, "_model_presentations": ranking}
    paths: list[str] = []
    model_order = [row["model"] for row in ranking]
    paths += _ranking_figure(root, ranking, metadata)
    paths += _pr_figure(root, pr_data, model_order, metadata)
    paths += _f1_figure(root, grid, model_order, metadata)
    paths += _class_heatmap(root, per_class, model_order, metadata)
    paths += _paired_figure(root, paired, metadata)
    paths += _grid_heatmap(root, grid, model_order, metadata)
    paths += _calibration_figure(root, grid, model_order, metadata)
    paths += _error_figure(root, ranking, metadata)
    paths += _pareto_figure(root, ranking, metadata)
    paths += _cohort_figure(root, cohort, metadata)
    paths += _leakage_figure(root, leakage_audit, model_order, metadata)
    paths += _cache_figure(root, cache_audit, model_order, metadata)
    if cohort.task == "polo":
        paths += _count_figures(root, per_image, model_order, metadata)
    return paths


def render_qualitative(
    root: Path,
    cohort: Cohort,
    predictions: dict[str, dict[str, list[Prediction]]],
    confidences: dict[str, float],
    ranking: list[dict[str, Any]],
    *,
    n: int = 6,
    seed: int = 42,
) -> list[str]:
    if not cohort.records:
        return []
    records = sorted(cohort.records, key=lambda record: record.relative_path)
    count = min(n, len(records))
    if count == len(records):
        chosen = records
    else:
        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(len(records), size=count, replace=False).tolist())
        chosen = [records[index] for index in indices]
    output_dir = root / "qualitative"
    output_dir.mkdir(parents=True, exist_ok=True)
    result: list[str] = []
    selection_rows = []
    for index, record in enumerate(chosen, start=1):
        with Image.open(record.image_path) as opened:
            image = opened.convert("RGB")
        panels = [
            VisualizationPanel(
                title="Ground truth",
                image=np.asarray(_draw_panel(image, list(record.annotations), truth=True)),
            )
        ]
        for model in ranking:
            name = str(model["model"])
            selected = [p for p in predictions[name][record.image_id] if p.score >= confidences[name]]
            rendered, heading = _draw_matched_panel(
                image,
                list(record.annotations),
                selected,
                cohort.task,
                cohort.metadata,
            )
            panels.append(VisualizationPanel(
                title=name,
                image=np.asarray(rendered),
                footer=heading,
                heading=model_identity_card(model, width=346, maximum=48),
            ))

        def prepare(_: Any) -> VisualizationItem:
            return VisualizationItem(
                image_path=record.image_path,
                label=str(record.relative_path),
                panels=tuple(panels),
                foreground=np.ones((image.height, image.width), dtype=bool),
            )

        chart = visualize_records(
            [record],
            options=VisualizationOptions(samples=None, columns=1, panel_size=3.6, show=False),
            prepare=prepare,
            title=f"Random sample (seed={seed})",
        )
        path = output_dir / f"comparison_{index:02d}.png"
        save_chart(chart, path)
        result.append(str(path.relative_to(root)))
        selection_rows.append({"index": index, "seed": seed, "image_id": record.image_id, "relative_path": record.relative_path})
    write_json(output_dir / "selection.json", selection_rows)
    return result


def render_prediction_grids(
    root: Path,
    cohort: Cohort,
    predictions: dict[str, dict[str, list[Prediction]]],
    confidences: dict[str, float],
    ranking: list[dict[str, Any]],
) -> list[str]:
    """Render one annotated grid per image, with at most two models per row."""

    output_root = root / "predictions"
    rendered: list[str] = []
    for record in cohort.records:
        columns = min(2, len(ranking))
        with Image.open(record.image_path) as opened:
            image = opened.convert("RGB")

        def prepare(model: dict[str, Any]) -> VisualizationItem:
            name = str(model["model"])
            selected = [
                prediction
                for prediction in predictions[name][record.image_id]
                if prediction.score >= confidences[name]
            ]
            panel, heading = _draw_matched_panel(
                image,
                list(record.annotations),
                selected,
                cohort.task,
                cohort.metadata,
            )
            return VisualizationItem(
                image_path=record.image_path,
                label="",
                panels=(VisualizationPanel(
                    title=name,
                    image=np.asarray(panel),
                    footer=heading,
                    heading=model_identity_card(model, width=403, maximum=48),
                ),),
                foreground=np.ones((image.height, image.width), dtype=bool),
            )

        chart = visualize_records(
            ranking,
            options=VisualizationOptions(samples=None, columns=columns, panel_size=4.2, show=False),
            prepare=prepare,
            title=str(record.relative_path),
        )
        relative = Path(record.relative_path).with_suffix(".png")
        path = output_root / relative
        save_chart(chart, path)
        rendered.append(str(path.relative_to(root)))
    return rendered


def _save_figure(root: Path, name: str, chart: Any, rows: TableLike, metadata: dict[str, Any]) -> list[str]:
    figure_dir = root / "figures"
    data_path = figure_dir / "data" / f"{name}.csv"
    meta_path = figure_dir / "metadata" / f"{name}.json"
    write_csv(data_path, rows)
    _assert_csv_roundtrip(data_path, rows)
    public_metadata = {
        key: value for key, value in metadata.items() if not key.startswith("_")
    }
    write_json(meta_path, {**public_metadata, "figure": name, "rows": len(rows)})
    columns = set(frame(rows).columns)
    model_presentations = metadata.get("_model_presentations")
    if (
        name != "ranking_forest"
        and isinstance(model_presentations, list)
        and {"model", "model_a", "model_b"} & columns
    ):
        chart = with_model_identities(
            chart,
            model_presentations,
            series_colors=(
                COLORS
                if name in {
                    "precision_recall",
                    "f1_confidence",
                    "calibration_reliability",
                    "throughput_performance_pareto",
                    "polo_count_agreement",
                    "polo_bland_altman",
                    "polo_count_residuals",
                }
                else None
            ),
        )
    outputs = []
    for suffix in ("pdf", "svg", "png"):
        path = figure_dir / f"{name}.{suffix}"
        save_chart(chart, path, scale_factor=2.5)
        outputs.append(str(path.relative_to(root)))
    return outputs


def _ranking_figure(root: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> list[str]:
    row_height = model_identity_row_height(rows)
    data = frame(rows)
    data["model_key"] = data.index.astype(str)
    data["value"] = (
        data["uncertainty_score"].fillna(data["score"])
        if "uncertainty_score" in data else data["score"]
    )
    data["low"] = data["ci_low"].fillna(data["value"]) if "ci_low" in data else data["value"]
    data["high"] = data["ci_high"].fillna(data["value"]) if "ci_high" in data else data["value"]
    base = alt.Chart(chart_data(data)).encode(
        y=alt.Y("model_key:N", sort=list(data["model_key"]), axis=None),
    )
    intervals = base.mark_rule(color=COLORS[0], strokeWidth=2).encode(x="low:Q", x2="high:Q")
    points = base.mark_point(color=COLORS[0], filled=True, size=75).encode(
        x=alt.X("value:Q", title=rows[0].get("uncertainty_metric", "score") if rows else "score"),
        tooltip=["model:N", "model_hash:N", alt.Tooltip("value:Q", format=".3f"), alt.Tooltip("low:Q", format=".3f"), alt.Tooltip("high:Q", format=".3f")],
    )
    metrics = (intervals + points).properties(
        width=560,
        height=alt.Step(row_height),
        title="Ultimate-original cluster performance with 95% bootstrap intervals",
    )
    chart = alt.hconcat(
        model_identity_chart(rows, row_height=row_height), metrics, spacing=18
    ).resolve_scale(y="shared")
    return _save_figure(root, "ranking_forest", chart, rows, meta)


def _pr_figure(root: Path, data: dict[str, Any], order: list[str], meta: dict[str, Any]) -> list[str]:
    rows = []
    for name in order:
        curves = list((data.get(name) or {}).values())
        if not curves: continue
        grid = np.linspace(0, 1, 101)
        vals = []
        for curve in curves:
            recall, precision = np.asarray(curve["recall"]), np.asarray(curve["precision"])
            vals.append([np.max(precision[recall >= point]) if np.any(recall >= point) else 0 for point in grid])
        mean = np.mean(vals, axis=0)
        rows.extend({"model": name, "recall": float(x), "precision": float(y)} for x, y in zip(grid, mean))
    chart = (
        alt.Chart(chart_data(rows))
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("recall:Q", scale=alt.Scale(domain=[0, 1]), title="Recall"),
            y=alt.Y("precision:Q", scale=alt.Scale(domain=[0, 1]), title="Precision"),
            color=alt.Color("model:N", sort=order, scale=alt.Scale(range=COLORS), legend=None),
            tooltip=["model:N", alt.Tooltip("recall:Q", format=".2f"), alt.Tooltip("precision:Q", format=".3f")],
        )
        .properties(width=500, height=420, title="Precision–recall curves")
    )
    return _save_figure(root, "precision_recall", chart, rows, meta)


def _f1_figure(root: Path, grid: list[dict[str, Any]], order: list[str], meta: dict[str, Any]) -> list[str]:
    rows = (
        frame(grid).groupby(["model", "confidence"], sort=True, as_index=False)["f1"].max()
        .assign(model=lambda data: pd.Categorical(data["model"], order, ordered=True))
        .sort_values(["model", "confidence"], kind="stable").reset_index(drop=True)
    )
    chart = (
        alt.Chart(chart_data(rows))
        .mark_line(point=True)
        .encode(
            x=alt.X("confidence:Q", title="Confidence threshold"),
            y=alt.Y("f1:Q", scale=alt.Scale(domain=[0, 1]), title="F1"),
            color=alt.Color("model:N", sort=order, scale=alt.Scale(range=COLORS), legend=None),
        )
        .properties(width=500, height=360, title="F1–confidence curves")
    )
    return _save_figure(root, "f1_confidence", chart, rows, meta)


def _class_heatmap(root: Path, rows: list[dict[str, Any]], order: list[str], meta: dict[str, Any]) -> list[str]:
    classes = sorted({str(row["class_name"]) for row in rows})
    chart = (
        alt.Chart(chart_data(rows))
        .mark_rect()
        .encode(
            x=alt.X("class_name:N", sort=classes, title="class"),
            y=alt.Y("model:N", sort=order, title=None),
            color=alt.Color("ap:Q", scale=alt.Scale(domain=[0, 1], scheme="viridis"), title="AP"),
            tooltip=["model:N", "class_name:N", alt.Tooltip("ap:Q", format=".3f")],
        )
        .properties(width=max(360, 54 * len(classes)), height=max(180, 42 * len(order)), title="Per-class average precision")
    )
    return _save_figure(root, "per_class_heatmap", chart, rows, meta)


def _paired_figure(root: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> list[str]:
    data = [{**row, "pair": f"{row['model_b']} − {row['model_a']}"} for row in rows]
    order = [row["pair"] for row in data]
    base = alt.Chart(chart_data(data)).encode(y=alt.Y("pair:N", sort=order, title=None))
    axis_title = "Model B − model A in ultimate-original macro F1"
    intervals = base.mark_rule(color=COLORS[1], strokeWidth=2).encode(x=alt.X("ci_low:Q", title=axis_title), x2="ci_high:Q")
    points = base.mark_point(color=COLORS[1], filled=True, size=70).encode(x=alt.X("difference:Q", title=axis_title))
    zero = alt.Chart(chart_data([{"zero": 0}])).mark_rule(color="black").encode(x="zero:Q")
    chart = (intervals + points + zero).properties(
        width=520,
        height=max(180, 40 * len(rows)),
        title="All paired model differences",
    )
    return _save_figure(root, "paired_differences", chart, rows, meta)


def _grid_heatmap(root: Path, rows: list[dict[str, Any]], order: list[str], meta: dict[str, Any]) -> list[str]:
    chart = (
        alt.Chart(chart_data(rows))
        .mark_rect()
        .encode(
            x=alt.X("confidence:O", title="Confidence"),
            y=alt.Y("postprocess:O", title="Postprocess"),
            color=alt.Color("score:Q", scale=alt.Scale(domain=[0, 1], scheme="viridis")),
            tooltip=["model:N", "confidence:Q", "postprocess:Q", alt.Tooltip("score:Q", format=".3f")],
        )
        .properties(width=240, height=240)
        .facet(column=alt.Column("model:N", sort=order, title=None))
        .properties(title="Confidence × postprocessing sweep")
    )
    return _save_figure(root, "threshold_heatmaps", chart, rows, meta)


def _calibration_figure(root: Path, rows: list[dict[str, Any]], order: list[str], meta: dict[str, Any]) -> list[str]:
    data = frame(rows)[["model", "confidence", "precision"]]
    data["model"] = pd.Categorical(data["model"], order, ordered=True)
    data = data.sort_values(["model", "confidence"], kind="stable").reset_index(drop=True)
    lines = alt.Chart(chart_data(data)).mark_line(point=True).encode(
        x=alt.X("confidence:Q", scale=alt.Scale(domain=[0, 1]), title="Confidence threshold"),
        y=alt.Y("precision:Q", scale=alt.Scale(domain=[0, 1]), title="Observed precision"),
        color=alt.Color("model:N", sort=order, scale=alt.Scale(range=COLORS), legend=None),
    )
    identity = alt.Chart(chart_data([{"x": 0, "y": 0, "x2": 1, "y2": 1}])).mark_rule(color="gray", strokeDash=[6, 4]).encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
    chart = (lines + identity).properties(width=500, height=360, title="Confidence reliability")
    return _save_figure(root,"calibration_reliability",chart,data,meta)


def _error_figure(root: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> list[str]:
    order = [str(row["model"]) for row in rows]
    data = frame(rows).reindex(columns=["model", "fp", "fn"], fill_value=0).melt(
        id_vars="model", value_vars=["fp", "fn"], var_name="error", value_name="count"
    )
    data["error"] = data["error"].str.upper()
    data["model"] = pd.Categorical(data["model"], order, ordered=True)
    data["error"] = pd.Categorical(data["error"], ["FP", "FN"], ordered=True)
    data = data.sort_values(["model", "error"], kind="stable").reset_index(drop=True)
    chart = (
        alt.Chart(chart_data(data))
        .mark_bar()
        .encode(
            x=alt.X("model:N", sort=order),
            xOffset=alt.XOffset("error:N", sort=["FP", "FN"]),
            y=alt.Y("count:Q", title="Count"),
            color=alt.Color("error:N", scale=alt.Scale(domain=["FP", "FN"], range=[COLORS[1], COLORS[4]])),
        )
        .properties(width=max(420, 90 * len(rows)), height=280, title="Error decomposition")
    )
    return _save_figure(root,"error_decomposition",chart,rows,meta)


def _pareto_figure(root: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> list[str]:
    data = frame(rows).assign(
        throughput=lambda value: value["throughput_images_per_second"].fillna(0)
    )
    points = alt.Chart(chart_data(data)).mark_point(filled=True, size=95).encode(
        x=alt.X("throughput:Q", title="Throughput (images/s)"),
        y=alt.Y("score:Q", title="Ranking metric"),
        color=alt.Color("model:N", scale=alt.Scale(range=COLORS), legend=None),
        tooltip=["model:N", alt.Tooltip("throughput:Q", format=".2f"), alt.Tooltip("score:Q", format=".3f")],
    )
    labels = alt.Chart(chart_data(data)).mark_text(align="left", dx=7, dy=-6).encode(
        x="throughput:Q", y="score:Q", text="model:N"
    )
    return _save_figure(root,"throughput_performance_pareto",(points+labels).properties(width=500,height=340,title="Performance–throughput trade-off"),rows,meta)


def _cohort_figure(root: Path, cohort: Cohort, meta: dict[str, Any]) -> list[str]:
    counts={name:0 for name in cohort.classes.values()}
    for record in cohort.records:
        for ann in record.annotations: counts[cohort.classes[int(ann["class_id"])]]+=1
    rows=[{"class_name":name,"count":value,"unit":"annotations"} for name,value in counts.items()]
    rows.append({
        "class_name": "background",
        "count": sum(not record.annotations for record in cohort.records),
        "unit": "empty images",
    })
    chart = (
        alt.Chart(chart_data(rows))
        .mark_bar()
        .encode(
            x=alt.X("class_name:N", sort=[row["class_name"] for row in rows], title="class / background"),
            y=alt.Y("count:Q", title="Count"),
            color=alt.condition(alt.datum.class_name == "background", alt.value(COLORS[4]), alt.value(COLORS[0])),
            tooltip=["class_name:N", "unit:N", alt.Tooltip("count:Q", format=",")],
        )
        .properties(width=max(420, 65 * len(rows)), height=280, title=f"Frozen cohort composition — {len(cohort.records)} images")
    )
    return _save_figure(root,"cohort_composition",chart,rows,meta)


def _cache_figure(root: Path, audit: dict[str, Any], order: list[str], meta: dict[str, Any]) -> list[str]:
    rows = frame({"model": name, **audit.get(name, {})} for name in order)
    source = rows["source"] if "source" in rows else pd.Series("fresh", index=rows.index)
    data = rows.assign(source=source.fillna("fresh").astype(str), value=1)
    chart=(alt.Chart(chart_data(data)).mark_bar().encode(
        y=alt.Y("model:N",sort=order,title=None),x=alt.X("value:Q",axis=None,title=None),
        color=alt.Color("source:N",scale=alt.Scale(range=COLORS)),tooltip=["model:N","source:N"]
    ).properties(width=440,height=max(180,38*len(rows)),title="Prediction cache source audit"))
    return _save_figure(root,"cache_source_audit",chart,rows,meta)


def _leakage_figure(root: Path, audit: dict[str, Any], order: list[str], meta: dict[str, Any]) -> list[str]:
    rows = frame(
        {"model": name, "status": audit.get(name, {}).get("status", "unknown"), "overlap_count": int(audit.get(name, {}).get("overlap_count", 0))}
        for name in order
    )
    data = rows.assign(
        state=np.select(
            [rows["overlap_count"].ne(0), rows["status"].eq("verified")],
            ["overlap", "verified"],
            default="unknown",
        )
    )
    chart=(alt.Chart(chart_data(data)).mark_bar().encode(
        y=alt.Y("model:N",sort=order,title=None),
        x=alt.X("overlap_count:Q",title="Overlapping ultimate originals"),
        color=alt.Color("state:N",scale=alt.Scale(domain=["overlap","verified","unknown"],range=["#D55E00","#009E73","#999999"]),legend=None),
        tooltip=["model:N","status:N","overlap_count:Q"],
    ).properties(width=480,height=max(180,38*len(rows)),title="Training/evaluation leakage audit"))
    return _save_figure(root, "cohort_leakage_audit", chart, rows, meta)


def _count_figures(root: Path, rows: list[dict[str, Any]], order: list[str], meta: dict[str, Any]) -> list[str]:
    paths=[]
    maximum=max([float(row["gt"]) for row in rows]+[float(row["pred"]) for row in rows]+[1.0])
    points=alt.Chart(chart_data(rows)).mark_point(filled=True,opacity=.65).encode(
        x=alt.X("gt:Q",title="Ground-truth count"),y=alt.Y("pred:Q",title="Predicted count"),
        color=alt.Color("model:N",sort=order,scale=alt.Scale(range=COLORS),legend=None)
    )
    identity=alt.Chart(chart_data([{"x":0,"y":0,"x2":maximum,"y2":maximum}])).mark_rule(color="gray",strokeDash=[6,4]).encode(x="x:Q",y="y:Q",x2="x2:Q",y2="y2:Q")
    paths += _save_figure(root,"polo_count_agreement",(points+identity).properties(width=420,height=420,title="POLO count agreement"),rows,meta)
    derived = frame(rows).assign(
        mean_count=lambda value: (value["gt"] + value["pred"]) / 2,
        residual=lambda value: value["pred"] - value["gt"],
    )
    bland=alt.Chart(chart_data(derived)).mark_point(filled=True,opacity=.65).encode(
        x=alt.X("mean_count:Q",title="Mean count"),y=alt.Y("residual:Q",title="Prediction − truth"),
        color=alt.Color("model:N",sort=order,scale=alt.Scale(range=COLORS),legend=None)
    )
    zero=alt.Chart(chart_data([{"zero":0}])).mark_rule(color="gray",strokeDash=[6,4]).encode(y="zero:Q")
    paths += _save_figure(root,"polo_bland_altman",(bland+zero).properties(width=500,height=340,title="POLO Bland–Altman view"),rows,meta)
    histogram=alt.Chart(chart_data(derived)).mark_bar(opacity=.5).encode(
        x=alt.X("residual:Q",bin=alt.Bin(maxbins=24),title="Count residual"),y=alt.Y("count():Q",title="Images"),
        color=alt.Color("model:N",sort=order,scale=alt.Scale(range=COLORS),legend=None)
    ).properties(width=500,height=340,title="POLO count-residual distribution")
    paths += _save_figure(root,"polo_count_residuals",histogram,rows,meta)
    return paths


def _draw_panel(image: Image.Image, values: list[Any], *, truth: bool) -> Image.Image:
    rendered = image.convert("RGBA")
    overlay = Image.new("RGBA", rendered.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for value in values:
        color = COLORS[2] if truth else COLORS[1]
        _draw_value(draw, value, truth=truth, color=color, alpha=.9, linewidth=2)
    return Image.alpha_composite(rendered, overlay).convert("RGB")


def _draw_matched_panel(
    image: Image.Image,
    truth: list[dict[str, Any]],
    predictions: list[Prediction],
    task: str,
    metadata: dict[str, Any],
) -> tuple[Image.Image, str]:
    from .metrics import optimal_match

    matched = optimal_match(truth, predictions, task, metadata)
    rendered = image.convert("RGBA")
    overlay = Image.new("RGBA", rendered.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    heading = (
        f"TP {len(matched['matches'])}  FP {len(matched['unmatched_pred'])}  "
        f"FN {len(matched['unmatched_gt'])}"
    )
    for truth_index, prediction_index, _ in matched["matches"]:
        _draw_value(draw, truth[truth_index], truth=True, color="#FFFFFF", alpha=.75, linewidth=3)
        _draw_value(draw, predictions[prediction_index], truth=False, color="#009E73", alpha=1, linewidth=2)
        first, second = _center(truth[truth_index]), _center(predictions[prediction_index])
        if first and second:
            draw.line((first, second), fill=(*ImageColor.getrgb("#009E73"), 205), width=1)
    for index in matched["unmatched_pred"]:
        _draw_value(draw, predictions[index], truth=False, color="#E69F00", alpha=1, linewidth=2)
    for index in matched["unmatched_gt"]:
        _draw_value(draw, truth[index], truth=True, color="#D55E00", alpha=1, linewidth=2)
    return Image.alpha_composite(rendered, overlay).convert("RGB"), heading


def _draw_value(draw: ImageDraw.ImageDraw, value: Any, *, truth: bool, color: str, alpha: float, linewidth: float) -> None:
    get = value.get if truth else lambda key, default=None: getattr(value, key, default)
    rgba = (*ImageColor.getrgb(color), round(255 * alpha))
    width = max(1, round(linewidth))
    box = get("bbox")
    if box:
        draw.rectangle(tuple(map(float, box)), outline=rgba, width=width)
    polygons = get("polygons") or ([get("polygon")] if get("polygon") else [])
    for polygon in polygons:
        points = [tuple(map(float, point[:2])) for point in polygon]
        if len(points) >= 2:
            draw.line([*points, points[0]], fill=rgba, width=width, joint="curve")
    point = get("point")
    if point:
        x, y = map(float, point)
        radius_marker = max(3, width + 2)
        draw.ellipse((x-radius_marker,y-radius_marker,x+radius_marker,y+radius_marker),fill=rgba,outline=(0,0,0,180),width=1)
        radius = get("radius", None)
        if radius:
            draw.ellipse((x-radius,y-radius,x+radius,y+radius),outline=rgba,width=width)
    keypoints = get("keypoints")
    if keypoints:
        for keypoint in keypoints:
            if len(keypoint) >= 3 and keypoint[2] is not None and keypoint[2] <= 0:
                continue
            x, y = float(keypoint[0]), float(keypoint[1])
            draw.ellipse((x-3,y-3,x+3,y+3),fill=rgba,outline=(0,0,0,180),width=1)


def _center(value: Any) -> tuple[float, float] | None:
    get = value.get if isinstance(value, dict) else lambda key, default=None: getattr(value, key, default)
    point = get("point")
    if point:
        return float(point[0]), float(point[1])
    box = get("bbox")
    if box:
        return (float(box[0]) + float(box[2])) / 2, (float(box[1]) + float(box[3])) / 2
    return None


def _cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(to_jsonable(value), sort_keys=True)
    return value


def _csv_cell(value: Any) -> str:
    cell = _cell(value)
    return "" if cell is None or cell is pd.NA else str(cell)


def _csv_frame(rows: TableLike) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows.astype(object).map(_csv_cell)
    return pd.DataFrame.from_records(
        ({key: _csv_cell(value) for key, value in row.items()} for row in rows)
    )


def _assert_csv_roundtrip(path: Path, rows: TableLike) -> None:
    source = _csv_frame(rows).astype(str)
    if not len(source.columns):
        return
    restored = pd.read_csv(path, dtype=str, keep_default_na=False)
    if len(restored) != len(source):
        raise AssertionError(f"Figure data row count changed while writing {path}")
    if not restored.equals(source):
        raise AssertionError(f"Figure values changed while writing {path}")


def _latex(value: Any) -> str:
    text = str(_cell(value))
    return text.replace("_", "\\_").replace("%", "\\%")


def combine_report_plots(
    roots: Iterable[Path],
    output: Path,
    *,
    limit: int = 16,
    width: int = 1600,
) -> Path | None:
    """Collapse comparison figures into one readable, vertically stacked sheet.

    Each panel is cropped to its visible content and scaled independently. The
    compositor deliberately does not include an existing aggregate report:
    doing so would recursively embed ``plots.png`` in every rebuilt comparison.
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
    canvas_width = width
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
