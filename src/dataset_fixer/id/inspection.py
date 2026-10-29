"""Identity inspection prepared for the existing tabular and image renderers."""
from pathlib import Path

import numpy as np

from .data import stats, balanced_rows


def dataset_report(data, *, destination=None, show=True, width=2400):
    import altair as alt
    from ..tabular import chart_data
    from ..static_rendering import finish_chart
    totals = stats(data, by=()).iloc[0]
    identities = stats(data).fillna({"identity_id": "unlabeled"})
    clips = stats(data, by="clip_id").fillna({"clip_id": "no clip"})
    identities = identities.nlargest(50, "samples")
    clips = clips.nlargest(50, "samples")
    panel_width = max(280, width // 4)
    def bars(table, field, metric, title):
        return alt.Chart(chart_data(table.reset_index(drop=True))).mark_bar().encode(
            y=alt.Y(f"{field}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X(f"{metric}:Q", title=metric.title()), tooltip=list(table.columns),
        ).properties(width=panel_width, height=alt.Step(22), title=title)
    chart = alt.vconcat(
        bars(identities, "identity_id", "samples", "Samples per individual (including unlabeled)"),
        alt.hconcat(bars(clips, "clip_id", "samples", "Samples per clip"),
                    bars(clips, "clip_id", "tracklets", "Distinct tracklets per clip")),
        title=f"{data.name}: {int(totals.samples):,} crops · {int(totals.individuals)} individuals · {int(totals.unlabeled):,} unlabeled",
    )
    output = Path(destination) if destination is not None else None
    finish_chart(chart, destination=output, show=show, overwrite=True)
    return output


def embedding_chart(table, *, identities, method, epoch):
    import altair as alt
    from ..tabular import chart_data
    return alt.Chart(chart_data(table)).mark_point(filled=True, size=48, opacity=.75).encode(
        x=alt.X("x:Q", title=f"{method} 1", axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("y:Q", title=f"{method} 2", axis=alt.Axis(labels=False, ticks=False)),
        color=alt.Color("identity_id:N", title="Individual", scale=alt.Scale(domain=identities, scheme="tableau20")),
        shape=alt.Shape("split:N", title="Split", scale=alt.Scale(domain=["train", "val"], range=["circle", "triangle"])),
        tooltip=["identity_id:N", "clip_id:N", "tracklet_id:N", "split:N", "source:N"],
    ).properties(width=620, height=470, title=alt.TitleParams(
        text=f"Embedding space · epoch {epoch + 1}",
        subtitle=[f"{len(table)} fixed, balanced crops · current encoder · deterministic preprocessing",
                  "Projection axes are refitted each epoch; distances between epochs are not comparable."],
    ))


def coverage_chart(rows, *, epoch):
    import altair as alt
    from ..tabular import chart_data
    base = alt.Chart(chart_data(rows)).encode(
        x=alt.X('split:N', sort=['train', 'val'], title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y('percent:Q', title='Unique source images used (%)', scale=alt.Scale(domain=[0, 100])),
        tooltip=['split:N', 'seen:Q', 'total:Q', alt.Tooltip('percent:Q', format='.2f')],
    )
    bars = base.mark_bar(size=90).encode(color=alt.Color('split:N', legend=None,
        scale=alt.Scale(domain=['train', 'val'], range=['#4c78a8', '#f58518'])))
    labels = base.mark_text(dy=-10, fontSize=14).encode(text=alt.Text('percent:Q', format='.2f'))
    return (bars + labels).properties(width=360, height=230, padding=16, title=alt.TitleParams(
        text=f'Dataset coverage · epoch {epoch+1}', offset=25, subtitle=[
            'Cumulative completed batches; repeats count once',
            ' · '.join(f"{r['split']}: {r['seen']:,} / {r['total']:,}" for r in rows),
        ]))


def _attention_item(sample, rgb, heat, metadata, bound, options):
    import altair as alt
    from matplotlib import colormaps
    from ..static_rendering import format_label, text_region
    from ..visualization import VisualizationItem, VisualizationPanel

    scaled = heat / max(bound, 1e-12)
    colors = colormaps["RdBu_r"]((scaled + 1) / 2)[..., :3]
    alpha = np.abs(scaled)[..., None] * .65
    overlay = ((rgb * (1 - alpha) + colors * alpha) * 255).astype(np.uint8)
    width = round(options.panel_size * 96)

    def caption(text, **style):
        return text_region(format_label(text, mode="middle", maximum=width // 7), width=width, **style)

    color = "#6B7280" if metadata["correct"] is None else "#218739" if metadata["correct"] else "#C53030"
    headings = [caption(f"GT: {metadata['ground_truth'] or 'unlabeled'}"),
                caption(f"Pred: {metadata['prediction']}", color=color, font_weight="bold")]
    if metadata.get("target_override", False):
        headings.append(caption(f"Target: {metadata['identity_id']}"))
    if metadata["competitor"] is None:
        footer = f"Similarity: {metadata['similarity']:.4f}\nNo competing identity"
    else:
        footer = (f"vs: {metadata['competitor']}\n"
                  f"Margin: {metadata['margin']:+.4f} · sim: {metadata['similarity']:.4f}")
    panel = VisualizationPanel("", overlay, footer=footer, heading=alt.vconcat(*headings, spacing=0))
    return VisualizationItem(sample.image_path, sample.clip_id or sample.image_path.name,
                             (panel,), np.ones(heat.shape, bool))


def visualize_attention(model, source, *, gallery, identity_id=None, contrast=True,
                        samples=4, columns=2, patch_fraction=0.15, device=None, destination=None, show=True):
    """Visualize Captum occlusion attribution for identity similarity.

    Parameters:
        model: Loaded identity Model.
        source: Query Dataset, image paths, or image references.
        gallery: Compatible enrollment gallery.
        identity_id: Identity to explain; None uses each query's predicted individual.
        contrast: Explain target similarity minus the strongest competing identity.
        samples: Maximum number of balanced query crops to inspect.
        columns: Number of overlay cards per grid row, default 2.
        patch_fraction: Occlusion window fraction of model input width, in (0, 1].
        device: Optional cpu, cuda:0, or mps override; otherwise use the Model device.
        destination: Optional output image path, with raw attribution NPZ/JSON sidecars.
        show: Display through the existing visualization renderer.

    Red indicates regions supporting the target score, blue indicates opposing
    regions. This is sensitivity to mean-color occlusion, not anatomical proof or
    a transformer attention map. All overlays share one symmetric color scale.
    Prediction text is green for correct, red for incorrect (including tied
    top scores), and gray when ground truth is unavailable. Margins compare the
    explained target with its strongest competitor on the original input.
    """
    import json
    import altair as alt
    import torch
    from captum.attr import Occlusion
    from .model import input_index, runtime, preprocess
    from .gallery import identity_scores
    from ..visualization import visualization_options, visualize_records, finish_visualization
    options = visualization_options(samples=samples, columns=columns, destination=destination, show=show)
    if samples is None or not 0 < patch_fraction <= 1:
        raise ValueError("samples must be positive and patch_fraction must be in (0, 1]")
    gallery.validate(model)
    if contrast and len(gallery.identities) < 2:
        raise ValueError("Contrast attribution needs at least two enrolled individuals")
    if identity_id is not None and str(identity_id) not in gallery.identities:
        raise ValueError(f"Identity {identity_id!r} is not enrolled")
    index = input_index(source)
    selected = balanced_rows(index, samples)
    module = runtime(model)
    previous_device, was_training = next(module.parameters()).device, module.training
    target_device = device or model._device or str(previous_device)
    observations, maps, metadata = [], {}, []
    try:
        module.to(target_device).eval()
        references = torch.as_tensor(gallery.embeddings.copy(), device=target_device)
        def scores(images):
            return identity_scores(module(images), references, gallery.labels)
        with torch.no_grad():
            for number, row in enumerate(selected):
                sample = index[row]
                tensor = preprocess(sample.image_ref.read(), module.identity["preprocessing"])[None].to(target_device)
                initial = scores(tensor)[0]
                predicted = int(initial.argmax())
                target = gallery.identities.index(str(identity_id)) if identity_id is not None else predicted
                others = [i for i in range(len(gallery.identities)) if i != target]
                competitor = max(others, key=lambda i: float(initial[i])) if others else None
                def objective(images):
                    values = scores(images)
                    return values[:, target] - values[:, others].max(dim=1).values if contrast else values[:, target]
                size = tensor.shape[-1]
                patch = max(1, round(size*patch_fraction))
                attribution = Occlusion(objective).attribute(tensor, sliding_window_shapes=(3, patch, patch),
                    strides=(3, max(1, patch//2), max(1, patch//2)), baselines=0., perturbations_per_eval=16)
                heat = attribution[0].mean(dim=0).cpu().numpy()
                prep = module.identity["preprocessing"]
                rgb = tensor[0].cpu().numpy().transpose(1,2,0)*np.array(prep["std"])+np.array(prep["mean"])
                rgb = np.clip(rgb, 0, 1)
                bound = float(np.abs(heat).max())
                target_label = gallery.identities[target]
                prediction = gallery.identities[predicted]
                truth = sample.identity_id
                maps[f"attribution_{number}"] = heat
                record = dict(source_ref=sample.image_ref.key, identity_id=target_label,
                    target_override=identity_id is not None,
                    ground_truth=truth, prediction=prediction,
                    correct=None if truth is None else bool(truth == prediction and int((initial == initial[predicted]).sum()) == 1),
                    competitor=None if competitor is None else gallery.identities[competitor],
                    margin=None if competitor is None else float(initial[target] - initial[competitor]),
                    similarity=float(initial[target]), contrast=contrast, baseline="preprocessing channel mean",
                    patch_pixels=patch, max_abs_effect=bound, model_fingerprint=gallery.model_fingerprint,
                    gallery_fingerprint=gallery.fingerprint)
                metadata.append(record)
                observations.append((sample, rgb, heat, record))
    finally:
        module.to(previous_device).train(was_training)
    bound = max((record["max_abs_effect"] for record in metadata), default=0.)
    for record in metadata:
        record.update(color_scale="shared", color_bound=bound)
    chart = visualize_records(observations, options=options,
        prepare=lambda item: _attention_item(*item, bound, options))
    chart = chart.properties(title=alt.TitleParams(text="Identity evidence · occlusion sensitivity",
        anchor="middle", fontSize=16, offset=14, subtitle=[
            f"Overlay: red supports · blue opposes {'margin' if contrast else 'similarity'}",
            f"Shared effect scale: ±{bound:.4g}",
            "Prediction: green correct · red incorrect",
            "Gray prediction: no ground truth",
        ]))
    finish_visualization(chart, options)
    if destination:
        path = Path(destination)
        np.savez_compressed(path.with_suffix(".npz"), **maps)
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return chart
