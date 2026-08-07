from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ..utils import to_jsonable
from .types import Cohort, Prediction

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in fields})


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), indent=2, sort_keys=True), encoding="utf-8")


def write_tables(root: Path, ranking: list[dict[str, Any]]) -> None:
    write_csv(root / "tables" / "model_comparison.csv", ranking)
    columns = [key for key in ("rank", "model", "backend", "score", "confidence", "postprocess", "support_images", "support_annotations", "support_clusters") if ranking and key in ranking[0]]
    lines = ["\\begin{tabular}{" + "l" * len(columns) + "}", " & ".join(columns) + " \\\\ \\hline"]
    for row in ranking:
        lines.append(" & ".join(_latex(row.get(key)) for key in columns) + " \\\\")
    lines.append("\\end{tabular}")
    (root / "tables" / "model_comparison.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        import pandas as pd

        pd.DataFrame(ranking).to_excel(root / "tables" / "model_comparison.xlsx", index=False)
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
    model_order: list[str],
    *,
    baseline: str | None = None,
    n: int = 6,
) -> list[str]:
    if not cohort.records:
        return []
    from .metrics import optimal_match

    scored = []
    for record in cohort.records:
        counts = [sum(p.score >= confidences[name] for p in predictions[name][record.image_id]) for name in model_order]
        errors = {}
        for name in model_order:
            selected = [p for p in predictions[name][record.image_id] if p.score >= confidences[name]]
            match = optimal_match(list(record.annotations), selected, cohort.task, cohort.metadata)
            errors[name] = len(match["unmatched_gt"]) + len(match["unmatched_pred"])
        scored.append({"disagreement": max(counts)-min(counts), "density": len(record.annotations), "errors": errors, "record": record})
    by_path = lambda value: value["record"].relative_path
    candidates = [
        ("largest_inter_model_disagreement", max(scored, key=lambda v: (v["disagreement"], v["density"], by_path(v)))),
        ("highest_ranked_model_failure", max(scored, key=lambda v: (v["errors"][model_order[0]], v["density"], by_path(v)))),
        ("densest", max(scored, key=lambda v: (v["density"], by_path(v)))),
        ("sparsest", min(scored, key=lambda v: (v["density"], by_path(v)))),
    ]
    if baseline in model_order:
        candidates.insert(1, ("baseline_failure", max(scored, key=lambda v: (v["errors"][baseline], v["density"], by_path(v)))))
    ordered_density = sorted(scored, key=lambda v: (v["density"], by_path(v)))
    candidates.append(("representative_median", ordered_density[len(ordered_density)//2]))
    chosen: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for reason, value in candidates + [("additional_disagreement", v) for v in sorted(scored, key=lambda x: (-x["disagreement"], by_path(x)))]:
        image_id = value["record"].image_id
        if image_id not in seen:
            chosen.append((reason, value["record"])); seen.add(image_id)
        if len(chosen) >= min(n, len(scored)):
            break
    output_dir = root / "qualitative"
    output_dir.mkdir(parents=True, exist_ok=True)
    result: list[str] = []
    selection_rows = []
    for index, (reason, record) in enumerate(chosen, start=1):
        fig, axes = plt.subplots(1, len(model_order) + 1, figsize=(4 * (len(model_order) + 1), 4), squeeze=False)
        image = Image.open(record.image_path).convert("RGB")
        _draw_panel(axes[0, 0], image, list(record.annotations), cohort.task, "Ground truth", truth=True)
        for column, name in enumerate(model_order, start=1):
            selected = [p for p in predictions[name][record.image_id] if p.score >= confidences[name]]
            _draw_matched_panel(
                axes[0, column], image, list(record.annotations), selected,
                cohort.task, cohort.metadata, name,
            )
        fig.suptitle(record.relative_path)
        fig.tight_layout()
        path = output_dir / f"comparison_{index:02d}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        result.append(str(path.relative_to(root)))
        selection_rows.append({"index": index, "reason": reason, "image_id": record.image_id, "relative_path": record.relative_path})
    write_json(output_dir / "selection.json", selection_rows)
    return result


def _save_figure(root: Path, name: str, fig: Any, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> list[str]:
    figure_dir = root / "figures"
    data_path = figure_dir / "data" / f"{name}.csv"
    meta_path = figure_dir / "metadata" / f"{name}.json"
    write_csv(data_path, rows)
    _assert_csv_roundtrip(data_path, rows)
    write_json(meta_path, {**metadata, "figure": name, "rows": len(rows)})
    outputs = []
    for suffix, dpi in (("pdf", 300), ("svg", 300), ("png", 600)):
        path = figure_dir / f"{name}.{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        outputs.append(str(path.relative_to(root)))
    plt.close(fig)
    return outputs


def _ranking_figure(root: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> list[str]:
    fig, ax = plt.subplots(figsize=(8, max(3, .6 * len(rows) + 1)))
    y = np.arange(len(rows))
    values = [row.get("uncertainty_score", row["score"]) for row in rows]
    low = [row.get("ci_low", value) for row, value in zip(rows, values)]
    high = [row.get("ci_high", value) for row, value in zip(rows, values)]
    errors = np.vstack((np.asarray(values) - np.asarray(low), np.asarray(high) - np.asarray(values)))
    ax.errorbar(values, y, xerr=errors, fmt="o", color=COLORS[0], capsize=4)
    ax.set_yticks(y, [row["model"] for row in rows]); ax.invert_yaxis(); ax.set_xlabel(rows[0].get("uncertainty_metric", "score") if rows else "score")
    ax.set_title("Ultimate-original cluster performance with 95% bootstrap intervals"); ax.grid(axis="x", alpha=.25)
    return _save_figure(root, "ranking_forest", fig, rows, meta)


def _pr_figure(root: Path, data: dict[str, Any], order: list[str], meta: dict[str, Any]) -> list[str]:
    fig, ax = plt.subplots(figsize=(7, 6)); rows = []
    for index, name in enumerate(order):
        curves = list((data.get(name) or {}).values())
        if not curves: continue
        grid = np.linspace(0, 1, 101)
        vals = []
        for curve in curves:
            recall, precision = np.asarray(curve["recall"]), np.asarray(curve["precision"])
            vals.append([np.max(precision[recall >= point]) if np.any(recall >= point) else 0 for point in grid])
        mean = np.mean(vals, axis=0)
        ax.plot(grid, mean, label=name, color=COLORS[index % len(COLORS)])
        rows.extend({"model": name, "recall": float(x), "precision": float(y)} for x, y in zip(grid, mean))
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1), title="Precision–recall curves"); ax.grid(alpha=.2); ax.legend(frameon=False)
    return _save_figure(root, "precision_recall", fig, rows, meta)


def _f1_figure(root: Path, grid: list[dict[str, Any]], order: list[str], meta: dict[str, Any]) -> list[str]:
    fig, ax = plt.subplots(figsize=(7, 5)); rows: list[dict[str, Any]] = []
    for index, name in enumerate(order):
        selected = [row for row in grid if row["model"] == name]
        by_confidence: dict[float, list[float]] = {}
        for row in selected:
            by_confidence.setdefault(float(row["confidence"]), []).append(float(row["f1"]))
        x = sorted(by_confidence); y = [max(by_confidence[value]) for value in x]
        ax.plot(x, y, "o-", color=COLORS[index % len(COLORS)], label=name)
        rows.extend({"model": name, "confidence": a, "f1": b} for a, b in zip(x, y))
    ax.set(xlabel="Confidence threshold", ylabel="F1", ylim=(0, 1), title="F1–confidence curves"); ax.grid(alpha=.2); ax.legend(frameon=False)
    return _save_figure(root, "f1_confidence", fig, rows, meta)


def _class_heatmap(root: Path, rows: list[dict[str, Any]], order: list[str], meta: dict[str, Any]) -> list[str]:
    classes = sorted({str(row["class_name"]) for row in rows})
    lookup = {(row["model"], str(row["class_name"])): float(row.get("ap", math.nan)) for row in rows}
    values = np.asarray([[lookup.get((name, cls), math.nan) for cls in classes] for name in order])
    fig, ax = plt.subplots(figsize=(max(6, .7 * len(classes)), max(3, .55 * len(order))))
    im = ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto"); fig.colorbar(im, ax=ax, label="AP")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right"); ax.set_yticks(range(len(order)), order); ax.set_title("Per-class average precision")
    return _save_figure(root, "per_class_heatmap", fig, rows, meta)


def _paired_figure(root: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> list[str]:
    fig, ax = plt.subplots(figsize=(7, max(3, .55 * len(rows) + 1)))
    if rows:
        y = np.arange(len(rows)); values = [r["difference"] for r in rows]
        errors = np.vstack((np.asarray(values)-np.asarray([r["ci_low"] for r in rows]), np.asarray([r["ci_high"] for r in rows])-np.asarray(values)))
        ax.errorbar(values, y, xerr=errors, fmt="o", color=COLORS[1], capsize=4)
        ax.set_yticks(y, [r["model"] for r in rows]); ax.axvline(0, color="black", lw=1)
    ax.set_title("Paired differences from baseline"); ax.set_xlabel("Difference in ultimate-original macro F1"); ax.grid(axis="x", alpha=.2)
    return _save_figure(root, "paired_differences", fig, rows, meta)


def _grid_heatmap(root: Path, rows: list[dict[str, Any]], order: list[str], meta: dict[str, Any]) -> list[str]:
    fig, axes = plt.subplots(1, max(1, len(order)), figsize=(5 * max(1, len(order)), 4), squeeze=False)
    for ax, name in zip(axes[0], order):
        selected = [row for row in rows if row["model"] == name]
        xs = sorted({row["confidence"] for row in selected}); ys = sorted({row["postprocess"] for row in selected})
        matrix = np.asarray([[next((r["score"] for r in selected if r["confidence"] == x and r["postprocess"] == y), math.nan) for x in xs] for y in ys])
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto"); ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs]); ax.set_yticks(range(len(ys)), [f"{y:g}" for y in ys]); ax.set(xlabel="Confidence", ylabel="Postprocess", title=name); fig.colorbar(im, ax=ax)
    fig.suptitle("Confidence × postprocessing sweep")
    return _save_figure(root, "threshold_heatmaps", fig, rows, meta)


def _calibration_figure(root: Path, rows: list[dict[str, Any]], order: list[str], meta: dict[str, Any]) -> list[str]:
    fig, ax = plt.subplots(figsize=(7, 5)); data=[]
    for i,name in enumerate(order):
        values=sorted((r for r in rows if r["model"]==name), key=lambda r:r["confidence"])
        x=[r["confidence"] for r in values]; y=[r.get("precision",0) for r in values]
        ax.plot(x,y,"o-",label=name,color=COLORS[i%len(COLORS)]); data += [{"model":name,"confidence":a,"precision":b} for a,b in zip(x,y)]
    ax.plot([0,1],[0,1],"--",color="gray",label="identity"); ax.set(xlabel="Confidence threshold",ylabel="Observed precision",title="Confidence reliability"); ax.legend(frameon=False); ax.grid(alpha=.2)
    return _save_figure(root,"calibration_reliability",fig,data,meta)


def _error_figure(root: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> list[str]:
    fig,ax=plt.subplots(figsize=(8,4)); names=[r["model"] for r in rows]; x=np.arange(len(rows)); ax.bar(x-.18,[r.get("fp",0) for r in rows],.36,label="FP",color=COLORS[1]); ax.bar(x+.18,[r.get("fn",0) for r in rows],.36,label="FN",color=COLORS[4]); ax.set_xticks(x,names,rotation=20,ha="right"); ax.set(title="Error decomposition",ylabel="Count"); ax.legend(frameon=False)
    return _save_figure(root,"error_decomposition",fig,rows,meta)


def _pareto_figure(root: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> list[str]:
    fig,ax=plt.subplots(figsize=(7,5))
    for i,row in enumerate(rows):
        throughput = row.get("throughput_images_per_second") or 0
        ax.scatter(throughput,row["score"],color=COLORS[i%len(COLORS)],s=55); ax.annotate(row["model"],(throughput,row["score"]),xytext=(5,4),textcoords="offset points")
    ax.set(xlabel="Throughput (images/s)",ylabel="Ranking metric",title="Performance–throughput trade-off"); ax.grid(alpha=.2)
    return _save_figure(root,"throughput_performance_pareto",fig,rows,meta)


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
    fig,ax=plt.subplots(figsize=(max(6,.6*len(rows)),4)); ax.bar([r["class_name"] for r in rows],[r["count"] for r in rows],color=[*([COLORS[0]]*(len(rows)-1)),COLORS[4]]); ax.tick_params(axis="x",rotation=35); ax.set(title=f"Frozen cohort composition — {len(cohort.records)} images",ylabel="Count (annotations; background = empty images)")
    return _save_figure(root,"cohort_composition",fig,rows,meta)


def _cache_figure(root: Path, audit: dict[str, Any], order: list[str], meta: dict[str, Any]) -> list[str]:
    rows=[{"model":name,**audit.get(name,{})} for name in order]; sources=[str(r.get("source","fresh")) for r in rows]; labels=sorted(set(sources)); fig,ax=plt.subplots(figsize=(7,max(3,.5*len(rows)))); ax.barh(range(len(rows)),[labels.index(v)+1 for v in sources],color=[COLORS[labels.index(v)%len(COLORS)] for v in sources]); ax.set_yticks(range(len(rows)),order); ax.set_xticks(range(1,len(labels)+1),labels,rotation=20,ha="right"); ax.set_title("Prediction cache source audit")
    return _save_figure(root,"cache_source_audit",fig,rows,meta)


def _leakage_figure(root: Path, audit: dict[str, Any], order: list[str], meta: dict[str, Any]) -> list[str]:
    rows = [
        {"model": name, "status": audit.get(name, {}).get("status", "unknown"), "overlap_count": int(audit.get(name, {}).get("overlap_count", 0))}
        for name in order
    ]
    fig, ax = plt.subplots(figsize=(7, max(3, .5 * len(rows))))
    colors = ["#D55E00" if row["overlap_count"] else "#009E73" if row["status"] == "verified" else "#999999" for row in rows]
    ax.barh(range(len(rows)), [row["overlap_count"] for row in rows], color=colors)
    ax.set_yticks(range(len(rows)), order); ax.set(xlabel="Overlapping ultimate originals", title="Training/evaluation leakage audit")
    return _save_figure(root, "cohort_leakage_audit", fig, rows, meta)


def _count_figures(root: Path, rows: list[dict[str, Any]], order: list[str], meta: dict[str, Any]) -> list[str]:
    paths=[]
    fig,ax=plt.subplots(figsize=(6,6))
    for i,name in enumerate(order):
        selected=[r for r in rows if r["model"]==name]; ax.scatter([r["gt"] for r in selected],[r["pred"] for r in selected],alpha=.6,label=name,color=COLORS[i%len(COLORS)])
    limits=ax.get_xlim(); ax.plot(limits,limits,"--",color="gray"); ax.set(xlabel="Ground-truth count",ylabel="Predicted count",title="POLO count agreement"); ax.legend(frameon=False); paths += _save_figure(root,"polo_count_agreement",fig,rows,meta)
    fig,ax=plt.subplots(figsize=(7,5))
    for i,name in enumerate(order):
        selected=[r for r in rows if r["model"]==name]; means=[(r["gt"]+r["pred"])/2 for r in selected]; errors=[r["pred"]-r["gt"] for r in selected]; ax.scatter(means,errors,alpha=.6,label=name,color=COLORS[i%len(COLORS)])
    ax.axhline(0,color="gray",ls="--"); ax.set(xlabel="Mean count",ylabel="Prediction − truth",title="POLO Bland–Altman view"); ax.legend(frameon=False); paths += _save_figure(root,"polo_bland_altman",fig,rows,meta)
    fig,ax=plt.subplots(figsize=(7,5))
    for i,name in enumerate(order):
        errors=[r["pred"]-r["gt"] for r in rows if r["model"]==name]; ax.hist(errors,bins="auto",alpha=.45,label=name,color=COLORS[i%len(COLORS)])
    ax.axvline(0,color="gray",ls="--"); ax.set(xlabel="Count residual",ylabel="Images",title="POLO count-residual distribution"); ax.legend(frameon=False); paths += _save_figure(root,"polo_count_residuals",fig,rows,meta)
    return paths


def _draw_panel(ax: Any, image: Image.Image, values: list[Any], task: str, title: str, *, truth: bool) -> None:
    ax.imshow(image); ax.set_title(title); ax.axis("off")
    for value in values:
        get = value.get if truth else lambda key, default=None: getattr(value, key, default)
        color = COLORS[2] if truth else COLORS[1]
        box=get("bbox")
        if box:
            ax.add_patch(plt.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,color=color,lw=1.5))
        polygons=get("polygons") or ([get("polygon")] if get("polygon") else [])
        for polygon in polygons:
            xy=np.asarray(polygon); ax.plot(np.r_[xy[:,0],xy[0,0]],np.r_[xy[:,1],xy[0,1]],color=color,lw=1.5)
        point=get("point")
        if point:
            ax.scatter(*point,s=28,color=color,edgecolor="white",linewidth=.7)
            radius=get("radius",None)
            if radius: ax.add_patch(plt.Circle(point,radius,fill=False,color=color,lw=1))
        keypoints=get("keypoints")
        if keypoints:
            xy=np.asarray([[p[0],p[1]] for p in keypoints if len(p)<3 or p[2] is None or p[2]>0]);
            if len(xy): ax.scatter(xy[:,0],xy[:,1],s=15,color=color)


def _draw_matched_panel(
    ax: Any,
    image: Image.Image,
    truth: list[dict[str, Any]],
    predictions: list[Prediction],
    task: str,
    metadata: dict[str, Any],
    title: str,
) -> None:
    from .metrics import optimal_match

    matched = optimal_match(truth, predictions, task, metadata)
    ax.imshow(image); ax.axis("off")
    ax.set_title(
        f"{title}\nTP {len(matched['matches'])}  FP {len(matched['unmatched_pred'])}  FN {len(matched['unmatched_gt'])}"
    )
    for truth_index, prediction_index, _ in matched["matches"]:
        _draw_value(ax, truth[truth_index], truth=True, color="#FFFFFF", alpha=.75, linewidth=2.5)
        _draw_value(ax, predictions[prediction_index], truth=False, color="#009E73", alpha=1, linewidth=1.7)
        first, second = _center(truth[truth_index]), _center(predictions[prediction_index])
        if first and second:
            ax.plot([first[0], second[0]], [first[1], second[1]], color="#009E73", lw=.8, alpha=.8)
    for index in matched["unmatched_pred"]:
        _draw_value(ax, predictions[index], truth=False, color="#E69F00", alpha=1, linewidth=1.8)
    for index in matched["unmatched_gt"]:
        _draw_value(ax, truth[index], truth=True, color="#D55E00", alpha=1, linewidth=2)


def _draw_value(ax: Any, value: Any, *, truth: bool, color: str, alpha: float, linewidth: float) -> None:
    get = value.get if truth else lambda key, default=None: getattr(value, key, default)
    box = get("bbox")
    if box:
        ax.add_patch(
            plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False,
                          color=color, lw=linewidth, alpha=alpha)
        )
    polygons = get("polygons") or ([get("polygon")] if get("polygon") else [])
    for polygon in polygons:
        points = np.asarray(polygon)
        ax.plot(np.r_[points[:, 0], points[0, 0]], np.r_[points[:, 1], points[0, 1]], color=color, lw=linewidth, alpha=alpha)
    point = get("point")
    if point:
        ax.scatter(*point, s=30, color=color, edgecolor="black", linewidth=.4, alpha=alpha)
        radius = get("radius", None)
        if radius:
            ax.add_patch(plt.Circle(point, radius, fill=False, color=color, lw=linewidth, alpha=alpha))
    keypoints = get("keypoints")
    if keypoints:
        visible = np.asarray([[p[0], p[1]] for p in keypoints if len(p) < 3 or p[2] is None or p[2] > 0])
        if len(visible):
            ax.scatter(visible[:, 0], visible[:, 1], s=18, color=color, edgecolor="black", linewidth=.3, alpha=alpha)


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
    if isinstance(value, (dict, list, tuple)): return json.dumps(to_jsonable(value), sort_keys=True)
    return value


def _assert_csv_roundtrip(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        restored = list(csv.DictReader(handle))
    if len(restored) != len(rows):
        raise AssertionError(f"Figure data row count changed while writing {path}")
    for source, target in zip(rows, restored):
        for key, value in source.items():
            cell = _cell(value)
            expected = "" if cell is None else str(cell)
            if expected != target.get(key, ""):
                raise AssertionError(f"Figure value {key!r} changed while writing {path}")


def _latex(value: Any) -> str:
    text=str(_cell(value)); return text.replace("_","\\_").replace("%","\\%")
