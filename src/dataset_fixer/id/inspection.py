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


def visualize_attention(model, source, *, gallery, identity_id=None, contrast=True,
                        samples=4, patch_fraction=0.15, device=None, destination=None, show=True):
    """Visualize Captum occlusion attribution for identity similarity.

    Parameters:
        model: Loaded identity Model.
        source: Query Dataset, image paths, or image references.
        gallery: Compatible enrollment gallery.
        identity_id: Identity to explain; None uses each query's predicted individual.
        contrast: Explain target similarity minus the strongest competing identity.
        samples: Maximum number of balanced query crops to inspect.
        patch_fraction: Occlusion window fraction of model input width, in (0, 1].
        device: Optional cpu, cuda:0, or mps override; otherwise use the Model device.
        destination: Optional output image path, with raw attribution NPZ/JSON sidecars.
        show: Display through the existing visualization renderer.

    Red indicates regions supporting the target score, blue indicates opposing
    regions. This is sensitivity to mean-color occlusion, not anatomical proof or
    a transformer attention map. Each image has its own symmetric color scale.
    """
    import json
    import torch
    from captum.attr import Occlusion
    from matplotlib import colormaps
    from .model import input_index, runtime, preprocess
    from ..visualization import VisualizationItem, VisualizationPanel, VisualizationOptions, visualize_records, finish_visualization
    if not isinstance(samples, int) or samples < 1 or not 0 < patch_fraction <= 1:
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
    items, maps, metadata = [], {}, []
    try:
        module.to(target_device).eval()
        references = torch.as_tensor(gallery.embeddings.copy(), device=target_device)
        masks = [torch.tensor(np.asarray(gallery.labels) == label, device=target_device) for label in gallery.identities]
        def scores(images):
            similarity = module(images) @ references.T
            return torch.stack([similarity[:, mask].max(dim=1).values for mask in masks], dim=1)
        with torch.no_grad():
            for number, row in enumerate(selected):
                sample = index[row]
                tensor = preprocess(sample.image_ref.read(), module.identity["preprocessing"])[None].to(target_device)
                initial = scores(tensor)[0]
                target = gallery.identities.index(str(identity_id)) if identity_id is not None else int(initial.argmax())
                others = [i for i in range(len(gallery.identities)) if i != target]
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
                scaled = heat/max(bound, 1e-12)
                colors = colormaps["RdBu_r"]((scaled+1)/2)[..., :3]
                alpha = np.abs(scaled)[..., None]*.65
                overlay = ((rgb*(1-alpha)+colors*alpha)*255).astype(np.uint8)
                target_label = gallery.identities[target]
                label = f"{target_label} · {sample.clip_id or sample.image_path.name}"
                footer = f"Red: supports · Blue: opposes\nMax |effect|: {bound:.4f}"
                items.append(VisualizationItem(sample.image_path, label, (
                    VisualizationPanel("Model input", (rgb*255).astype(np.uint8)),
                    VisualizationPanel("Target vs strongest competitor" if contrast else "Target similarity", overlay, footer=footer),
                ), np.ones((size,size), bool)))
                maps[f"attribution_{number}"] = heat
                metadata.append(dict(source_ref=sample.image_ref.key, identity_id=target_label,
                    similarity=float(initial[target]), contrast=contrast, baseline="preprocessing channel mean",
                    patch_pixels=patch, max_abs_effect=bound, model_fingerprint=gallery.model_fingerprint,
                    gallery_fingerprint=gallery.fingerprint))
    finally:
        module.to(previous_device).train(was_training)
    options = VisualizationOptions(samples=len(items), columns=1, destination=Path(destination) if destination else None, show=show)
    chart = visualize_records(items, options=options, prepare=lambda item: item, title="Identity evidence · occlusion sensitivity")
    finish_visualization(chart, options)
    if destination:
        path = Path(destination)
        np.savez_compressed(path.with_suffix(".npz"), **maps)
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return chart
