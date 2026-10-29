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


_PREDICTION_LEGEND = ("Prediction: green correct · red incorrect", "Gray prediction: correctness unavailable")


def _identity_panel(pixels, metadata, options):
    """Shared ground truth, decision colors and score captions for ID images."""
    import altair as alt
    from ..static_rendering import format_label, text_region
    from ..visualization import VisualizationPanel

    width = max(144, round(options.panel_size * 96))

    def caption(text, **style):
        return text_region(format_label(text, mode=options.label_mode, maximum=max(18, width // 7)), width=width, **style)

    color = "#6B7280" if metadata["correct"] is None else "#218739" if metadata["correct"] else "#C53030"
    prediction_prefix = "Rep. pred" if metadata.get("aggregation_count") else "Pred"
    headings = [caption(f"GT: {metadata['ground_truth'] or 'unlabeled'}"),
                caption(f"{prediction_prefix}: {metadata['prediction']}", color=color, font_weight="bold")]
    if metadata.get("target_override", False):
        headings.append(caption(f"Target: {metadata['identity_id']}"))
    if metadata["similarity"] is None:
        footer = "Embedding only"
    elif metadata["competitor"] is None:
        footer = f"Similarity: {metadata['similarity']:.4f}"
    else:
        footer = (f"vs: {metadata['competitor']}\n"
                  f"Margin: {metadata['margin']:+.4f} · sim: {metadata['similarity']:.4f}")
    if metadata.get("rejected") and metadata.get("identity_id"):
        footer = f"Best: {metadata['identity_id']} · {footer}"
    if metadata.get("aggregation_count"):
        footer = footer.replace("Margin:", "Rep. margin:").replace("Similarity:", "Rep. similarity:")
        footer += f"\nMean evidence: {metadata['aggregation_count']} references"
    if "max_abs_effect" in metadata:
        effect = "mean Δ" if metadata.get("aggregation_count") else "Δ"
        footer += f"\nMax |{effect}|: {metadata['max_abs_effect']:.4g}"
    return VisualizationPanel("", pixels, footer=footer, heading=alt.vconcat(*headings, spacing=0))


def _attention_item(sample, rgb, heat, metadata, bound, options):
    from matplotlib import colormaps
    from ..visualization import VisualizationItem

    scaled = heat / max(bound, 1e-12)
    colors = colormaps["RdBu_r"]((scaled + 1) / 2)[..., :3]
    alpha = np.abs(scaled)[..., None] * .65
    overlay = ((rgb * (1 - alpha) + colors * alpha) * 255).astype(np.uint8)
    panel = _identity_panel(overlay, metadata, options)
    return VisualizationItem(sample.image_path, sample.clip_id or sample.image_path.name,
                             (panel,), np.ones(heat.shape, bool))


def visualize_attention(model, source, *, gallery, identity_id=None, contrast=True,
                        samples=4, columns=2, patch_fraction=0.15, color_scale="shared",
                        device=None, destination=None, show=True):
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
        color_scale: "shared" compares effect magnitudes across images; "per_image"
            reveals each map's pattern. Each card reports its actual maximum effect.
        device: Optional cpu, cuda:0, or mps override; otherwise use the Model device.
        destination: Optional output image path, with raw attribution NPZ/JSON sidecars.
        show: Display through the existing visualization renderer.

    Red indicates regions supporting the target score, blue indicates opposing
    regions. This is sensitivity to mean-color occlusion, not anatomical proof or
    a transformer attention map. Shared scaling is the default.
    Prediction text is green for correct, red for incorrect (including tied
    top scores), and gray when ground truth is unavailable. Margins compare the
    explained target with its strongest competitor on the original input.
    """
    from .model import input_index
    options = _attention_options(samples, columns, patch_fraction, color_scale, destination, show)
    if samples is None:
        raise ValueError("samples must be positive")
    index = input_index(source)
    observations = _attention_observations(model, index, balanced_rows(index, samples), gallery,
        identity_id=identity_id, contrast=contrast, patch_fraction=patch_fraction, device=device)
    return _render_attention(observations, options, color_scale=color_scale, contrast=contrast)


def _attention_options(samples, columns, patch_fraction, color_scale, destination, show):
    from ..visualization import visualization_options
    options = visualization_options(samples=samples, columns=columns, destination=destination, show=show)
    if not 0 < patch_fraction <= 1:
        raise ValueError("patch_fraction must be in (0, 1]")
    if color_scale not in {"shared", "per_image"}:
        raise ValueError("color_scale must be 'shared' or 'per_image'")
    return options


def _attention_observations(model, index, selected, gallery, *, identity_id=None, contrast=True,
                            patch_fraction, device, reference_mode=False):
    import torch
    from captum.attr import Occlusion
    from .model import runtime, preprocess
    from .gallery import identity_scores
    gallery.validate(model)
    if contrast and len(gallery.identities) < 2:
        raise ValueError("Contrast attribution needs at least two enrolled individuals")
    if identity_id is not None and str(identity_id) not in gallery.identities:
        raise ValueError(f"Identity {identity_id!r} is not enrolled")
    module = runtime(model)
    previous_device, was_training = next(module.parameters()).device, module.training
    target_device = device or model._device or str(previous_device)
    observations = []
    hashes = []
    if reference_mode:
        import hashlib
        for row, ref in enumerate(gallery.references):
            digest = ref.get("pixel_sha256")
            if digest is None:
                pixels = index[row].image_ref.read()
                digest = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
            hashes.append(digest)
    try:
        module.to(target_device).eval()
        references = torch.as_tensor(gallery.embeddings.copy(), device=target_device)
        with torch.no_grad():
            for number, row in enumerate(selected):
                sample = index[row]
                keep = np.ones(len(gallery.labels), dtype=bool)
                if reference_mode:
                    keep = np.array([digest != hashes[row] and ref["source_ref"] != sample.image_ref.key
                                     for digest, ref in zip(hashes, gallery.references)])
                    if not any(keep[i] and label == sample.identity_id for i, label in enumerate(gallery.labels)):
                        raise ValueError(f"Reference attention needs at least two distinct images for {sample.identity_id!r}")
                labels = tuple(label for label, use in zip(gallery.labels, keep) if use)
                identities = tuple(sorted(set(labels)))
                if contrast and len(identities) < 2:
                    raise ValueError("Reference attention needs a distinct competing identity after duplicate exclusion")
                vectors = references[torch.as_tensor(np.flatnonzero(keep), device=target_device)]
                def scores(images):
                    return identity_scores(module(images), vectors, labels)
                tensor = preprocess(sample.image_ref.read(), module.identity["preprocessing"])[None].to(target_device)
                initial = scores(tensor)[0]
                predicted = int(initial.argmax())
                target_label = sample.identity_id if reference_mode else str(identity_id) if identity_id is not None else identities[predicted]
                target = identities.index(target_label)
                others = [i for i in range(len(identities)) if i != target]
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
                prediction = identities[predicted]
                truth = sample.identity_id
                record = dict(source_ref=sample.image_ref.key, identity_id=target_label,
                    target_override=identity_id is not None or reference_mode,
                    ground_truth=truth, prediction=prediction,
                    correct=None if truth is None else bool(truth == prediction and int((initial == initial[predicted]).sum()) == 1),
                    competitor=None if competitor is None else identities[competitor],
                    margin=None if competitor is None else float(initial[target] - initial[competitor]),
                    similarity=float(initial[target]), contrast=contrast, baseline="preprocessing channel mean",
                    patch_pixels=patch, max_abs_effect=bound, model_fingerprint=gallery.model_fingerprint,
                    gallery_fingerprint=gallery.fingerprint)
                if reference_mode:
                    record.update(reference_row=int(row), map_key=f"reference_{number}",
                                  excluded_reference_rows=np.flatnonzero(~keep).tolist())
                observations.append((sample, rgb, heat, record))
    finally:
        module.to(previous_device).train(was_training)
    return observations


def _render_attention(observations, options, *, color_scale, contrast, reference_maps=None):
    import json
    import altair as alt
    from ..visualization import visualize_records, finish_visualization
    metadata = [item[3] for item in observations]
    bound = max((record["max_abs_effect"] for record in metadata), default=0.)
    for number, record in enumerate(metadata):
        record.update(map_key=f"attribution_{number}", color_scale=color_scale,
                      color_bound=record["max_abs_effect"] if color_scale == "per_image" else bound)
    chart = visualize_records(observations, options=options,
        prepare=lambda item: _attention_item(*item, item[3]["color_bound"], options))
    title = "Reference identity evidence · mean occlusion sensitivity" if reference_maps is not None else "Identity evidence · occlusion sensitivity"
    chart = chart.properties(title=alt.TitleParams(text=title,
        anchor="middle", fontSize=16, offset=14, subtitle=[
            f"Overlay: red supports · blue opposes {'margin' if contrast else 'similarity'}",
            f"Shared effect scale: ±{bound:.4g}" if color_scale == "shared" else "Per-image scale · effect magnitudes shown on each card",
            *(["Mean maps in input coordinates · real medoid image · self-matches excluded"] if reference_maps is not None else []),
            *_PREDICTION_LEGEND,
        ]))
    finish_visualization(chart, options)
    if options.destination:
        path = Path(options.destination)
        maps = {f"attribution_{i}": item[2] for i, item in enumerate(observations)}
        np.savez_compressed(path.with_suffix(".npz"), **maps, **(reference_maps or {}))
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return chart


def _visualize_reference_attention(model, gallery, *, samples_per_identity=20, samples=None,
                                  columns=3, patch_fraction=0.15, color_scale="shared", contrast=True,
                                  device=None, destination=None, show=True):
    from .gallery import _candidate_rows, _embedding_medoids
    options = _attention_options(samples, columns, patch_fraction, color_scale, destination, show)
    if isinstance(samples_per_identity, bool) or not isinstance(samples_per_identity, int) or samples_per_identity < 1:
        raise ValueError("samples_per_identity must be a positive integer")
    gallery.validate(model)
    index = gallery._index()
    identities = gallery.identities if samples is None else gallery.identities[:samples]
    selected = [row for row in _candidate_rows(index, samples_per_identity, 7)
                if index.frame.identity_id.iloc[row] in identities]
    observations = _attention_observations(model, index, selected, gallery, contrast=contrast,
        patch_fraction=patch_fraction, device=device, reference_mode=True)
    combined = []
    for label in identities:
        members = [item for item in observations if item[3]["identity_id"] == label]
        rows = [item[3]["reference_row"] for item in members]
        representative = int(_embedding_medoids(gallery.embeddings[rows], 1, 7)[0])
        sample, rgb, _, record = members[representative]
        heat = np.mean([item[2] for item in members], axis=0)
        metadata = dict(record, aggregation="mean", aggregation_count=len(members), representative="embedding_medoid",
                        max_abs_effect=float(np.abs(heat).max()), sources=[item[3] for item in members])
        combined.append((sample, rgb, heat, metadata))
    return _render_attention(combined, options, color_scale=color_scale, contrast=contrast,
                             reference_maps={f"reference_{i}": item[2] for i, item in enumerate(observations)})
