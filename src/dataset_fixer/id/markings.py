"""Cross-clip texture correspondences and measured identity-margin effects.

OpenCV proposes local correspondences; Captum measures model sensitivity. Neither
is treated as anatomical correspondence or evidence of statistical significance.
"""
from pathlib import Path
import hashlib
import json
import warnings

import numpy as np


def _box(point, size, shape):
    height, width = shape[:2]
    size = max(2, min(round(size), height, width))
    x = min(max(round(point[0] - size / 2), 0), width - size)
    y = min(max(round(point[1] - size / 2), 0), height - size)
    return (x, y, x + size, y + size)


def _overlaps(first, second):
    return (max(first[0], second[0]) < min(first[2], second[2]) and
            max(first[1], second[1]) < min(first[3], second[3]))


def _load_candidates(index, selected, preprocessing, patch_fraction):
    import cv2
    from .model import preprocess
    detector = cv2.SIFT_create(nfeatures=256)
    images, seen = [], set()
    for row in selected:
        sample = index[row]
        pixels = sample.image_ref.read()
        digest = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        tensor = preprocess(pixels, preprocessing)
        rgb = tensor.numpy().transpose(1, 2, 0) * preprocessing["std"] + preprocessing["mean"]
        rgb = np.uint8(np.rint(np.clip(rgb, 0, 1) * 255))
        size = max(2, round(rgb.shape[1] * patch_fraction))
        # Avoid matching the artificial black canvas boundary. This is not an
        # animal segmentation mask: real background can still be a candidate.
        foreground = np.any(rgb > 0, axis=2).astype(np.uint8)
        mask = cv2.erode(foreground, np.ones((size, size), np.uint8)) * 255
        points, descriptors = detector.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), mask)
        if descriptors is None or len(points) < 2:
            continue
        images.append(dict(sample=sample, digest=digest, tensor=tensor, rgb=rgb,
                           source_shape=pixels.shape[:2], points=points, descriptors=descriptors,
                           patch_size=size))
    return images


def _matches(first, second):
    """SIFT ratio test plus mutual nearest neighbors, all supplied by OpenCV."""
    import cv2
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    reverse = {m.queryIdx: m.trainIdx for m in matcher.match(second, first)}
    return [(m.queryIdx, m.trainIdx, float(m.distance), float(m.distance / n.distance))
            for m, n in matcher.knnMatch(first, second, k=2)
            if m.distance < .75 * n.distance and reverse.get(m.trainIdx) == m.queryIdx]


def _proposals(images, identity_id, patches):
    focal = [i for i, image in enumerate(images) if image["sample"].identity_id == identity_id]
    proposals = []
    for anchor in focal:
        image = images[anchor]
        groups = {}
        for other in focal:
            match_image = images[other]
            if image["sample"].clip_id == match_image["sample"].clip_id:
                continue
            for key, matched, distance, ratio in _matches(image["descriptors"], match_image["descriptors"]):
                clip = match_image["sample"].clip_id
                group = groups.setdefault(key, {})
                if clip not in group or ratio < group[clip][3]:
                    group[clip] = (other, matched, distance, ratio)
        for key, group in groups.items():
            box = _box(image["points"][key].pt, image["patch_size"], image["rgb"].shape)
            observations = [(anchor, box, 0., 0., "anchor")]
            for other, matched, distance, ratio in sorted(group.values(), key=lambda value: value[3]):
                target = images[other]
                scale = target["points"][matched].size / image["points"][key].size
                # Keep the physical neighborhood approximately comparable under
                # the scale change measured by SIFT; reject extreme changes.
                if .5 <= scale <= 2:
                    target_box = _box(target["points"][matched].pt, image["patch_size"] * scale, target["rgb"].shape)
                    observations.append((other, target_box, distance, ratio, "cross_clip"))
            if len(observations) > 1:
                proposals.append(dict(anchor=anchor, key=key, observations=observations))
    proposals.sort(key=lambda p: (-len(p["observations"]), np.mean([o[3] for o in p["observations"][1:]])))
    selected = []
    for proposal in proposals:
        # Avoid multiple SIFT orientations / nearby keypoints rediscovering the
        # same patch, including proposals anchored in another matched image.
        if any(i == j and _overlaps(box, other_box)
               for previous in selected for i, box, *_ in previous["observations"]
               for j, other_box, *_ in proposal["observations"]):
            continue
        selected.append(proposal)
        if len(selected) >= max(patches, min(12, 4 * patches)):
            break
    return selected


def _lookalike(proposal, images, identity_id):
    """Nearest other-identity descriptor, explicitly not an accepted match."""
    import cv2
    anchor = images[proposal["anchor"]]
    descriptor = anchor["descriptors"][proposal["key"]:proposal["key"] + 1]
    candidates = []
    for i, image in enumerate(images):
        if image["sample"].identity_id == identity_id:
            continue
        match = cv2.BFMatcher(cv2.NORM_L2).match(descriptor, image["descriptors"])[0]
        scale = image["points"][match.trainIdx].size / anchor["points"][proposal["key"]].size
        if .5 <= scale <= 2:
            box = _box(image["points"][match.trainIdx].pt, anchor["patch_size"] * scale, image["rgb"].shape)
            candidates.append((i, box, float(match.distance), None, "other_identity"))
    return min(candidates, key=lambda item: item[2]) if candidates else None


def _control_box(box, rgb):
    x0, y0, x1, y1 = box
    size = x1 - x0
    for center in ((x1 + size / 2, (y0 + y1) / 2), (x0 - size / 2, (y0 + y1) / 2),
                   ((x0 + x1) / 2, y1 + size / 2), ((x0 + x1) / 2, y0 - size / 2)):
        other = _box(center, size, rgb.shape)
        a, b, c, d = other
        if not _overlaps(box, other) and np.any(rgb[b:d, a:c] > 0, axis=2).mean() >= .95:
            return other
    return None


def _ablate(objective, tensor, box):
    import cv2
    import torch
    from captum.attr import FeatureAblation
    x0, y0, x1, y1 = box
    sigma = max(1., (x1 - x0) / 4)
    # Blur only within the region: no pixels from its surroundings enter the
    # baseline, and every pixel outside it is unchanged in Captum's input.
    region = tensor[0, :, y0:y1, x0:x1].cpu().numpy().transpose(1, 2, 0)
    blurred = cv2.GaussianBlur(region, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT_101)
    baseline = tensor.clone()
    baseline[0, :, y0:y1, x0:x1] = torch.as_tensor(blurred.transpose(2, 0, 1).copy(), device=tensor.device)
    mask = torch.zeros((1, 1, *tensor.shape[-2:]), dtype=torch.long, device=tensor.device)
    mask[..., y0:y1, x0:x1] = 1
    attribution = FeatureAblation(objective).attribute(tensor, baselines=baseline,
                                                      feature_mask=mask, perturbations_per_eval=2)
    # Captum repeats the group effect at each feature; summing would multiply
    # the measured score change by patch area and channel count.
    effect = float(attribution[0, :, y0:y1, x0:x1].mean())
    return effect, baseline[0].cpu()


def _measure(module, image, box, gallery, hashes, identity_id, device):
    import torch
    from .gallery import identity_scores, _reference_mask
    keep = _reference_mask(gallery, hashes, image["sample"].image_ref.key, image["digest"])
    labels = tuple(label for label, use in zip(gallery.labels, keep) if use)
    identities = sorted(set(labels))
    if identity_id not in identities or len(identities) < 2:
        raise ValueError("Marking effects need focal and competing gallery references after excluding "
                         "the inspected image and exact duplicates; enroll more distinct references")
    references = torch.as_tensor(gallery.embeddings[keep].copy(), device=device)
    target = identities.index(identity_id)
    others = [i for i in range(len(identities)) if i != target]
    def scores(batch):
        return identity_scores(module(batch), references, labels)
    def objective(batch):
        values = scores(batch)
        return values[:, target] - values[:, others].max(dim=1).values
    tensor = image["tensor"][None].to(device)
    with torch.no_grad():
        initial = scores(tensor)[0]
        competitor = max(others, key=lambda i: float(initial[i]))
        before = float(initial[target] - initial[competitor])
        effect, altered = _ablate(objective, tensor, box)
        control = _control_box(box, image["rgb"])
        control_effect = _ablate(objective, tensor, control)[0] if control else None
    if not np.isfinite([before, effect, *([] if control_effect is None else [control_effect])]).all():
        raise ValueError("The model produced non-finite marking scores")
    return dict(margin_before=before, margin_after=before - effect, effect=effect,
                control_box=control, control_effect=control_effect,
                effect_minus_control=None if control_effect is None else effect - control_effect,
                prediction=identities[int(initial.argmax())], competitor=identities[competitor],
                excluded_reference_rows=np.flatnonzero(~keep).tolist()), altered


def explain_markings(model, source=None, *, gallery, identity_id, samples_per_identity=20,
                     patches=3, patch_fraction=.15, context_fraction=1., columns=2,
                     device=None, destination=None, show=True):
    """Show cross-clip patch matches and their Captum-measured effects.

    source defaults to the gallery's references; otherwise pass a labeled ID
    Dataset with clip metadata. Inspect at most samples_per_identity images per
    enrolled identity, balanced across clips/tracklets. patches bounds reported
    candidates; patch_fraction sets their width at the model's input resolution.
    Each identity gets one figure, grouped by marking, showing all matched clips.
    context_fraction adds display context on each side of the outlined patch
    (default 1 gives three times its width); columns counts image pairs per row.
    These display settings never change the measured ablation regions.

    Returns a DataFrame with one measurement per patch/image. Its attrs contain
    settings, limitations and equally weighted per-clip summaries. Positive
    effect means blurring the patch reduces the focal-vs-strongest-rival margin.
    This same focal objective is used on the other wolves' images too.
    destination saves the grid plus CSV/JSON and original/altered input NPZs.
    No training, enrollment changes, or W&B metrics are performed.
    """
    import pandas as pd
    from .model import input_index, runtime
    from .gallery import _candidate_rows, _reference_hashes
    from .inspection import _render_markings
    from ..visualization import visualization_options
    options = visualization_options(context_fraction=context_fraction, columns=columns,
                                    destination=destination, show=show, panel_size=2.)
    for name, value in (("samples_per_identity", samples_per_identity), ("patches", patches)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not 0 < patch_fraction <= .5:
        raise ValueError("patch_fraction must be in (0, 0.5]")
    identity_id = str(identity_id)
    gallery.validate(model)
    if identity_id not in gallery.identities or len(gallery.identities) < 2:
        raise ValueError("Choose an enrolled identity from a gallery with at least two individuals")
    index = gallery._index() if source is None else input_index(source)
    eligible = index.frame.identity_id.isin(gallery.identities) & index.frame.clip_id.notna()
    omitted = int((~eligible).sum())
    index = index.take(np.flatnonzero(eligible))
    if index.where(identity_id=identity_id).frame.clip_id.nunique() < 2:
        raise ValueError("Cross-clip marking evidence requires focal images from at least two clips")
    module = runtime(model)
    preprocessing = module.identity["preprocessing"]
    images = _load_candidates(index, _candidate_rows(index, samples_per_identity, 7), preprocessing, patch_fraction)
    proposals = _proposals(images, identity_id, patches)
    metadata = dict(identity_id=identity_id, samples_per_identity=samples_per_identity, patches=patches,
        patch_fraction=patch_fraction, model_fingerprint=gallery.model_fingerprint,
        gallery_fingerprint=gallery.fingerprint, preprocessing=preprocessing, seed=7,
        omitted_unlabeled_or_missing_clip=omitted,
        images_with_descriptors=len(images), candidates_measured=len(proposals),
        objective="cosine(focal) - max(cosine(other enrolled identities))",
        effect="margin(original) - margin(patch blurred)", baseline="within-patch Gaussian blur; sigma=width/4, min 1",
        matcher="OpenCV SIFT; mutual nearest neighbor and ratio < 0.75",
        display=dict(context_fraction=context_fraction, columns=columns),
        coordinates="xyxy exclusive, model input; source_box refers to decoded ImageRef crop",
        limitations=["Candidate texture correspondence, not verified anatomical alignment or fur segmentation.",
                     "Discovery and effect ranking use these same images; this is not held-out confirmation or statistical significance.",
                     "Nearby controls may cover different anatomy or background; blurring can also change coarse structure.",
                     "Scoring excludes self and exact pixel duplicates, but can include other images from the same clip."])
    records, arrays = [], {}
    previous_device, training = next(module.parameters()).device, module.training
    measured = []
    try:
        if proposals:
            target_device = device or model._device or str(previous_device)
            hashes = _reference_hashes(gallery)
            module.to(target_device).eval()
            for proposal in proposals:
                results = []
                for image_row, box, distance, ratio, role in proposal["observations"]:
                    values, altered = _measure(module, images[image_row], box, gallery, hashes, identity_id, target_device)
                    results.append((image_row, box, distance, ratio, role, values, altered))
                measured.append((float(np.mean([item[5]["effect"] for item in results])), proposal, results))
            measured.sort(key=lambda item: item[0], reverse=True)
            for number, (_, proposal, results) in enumerate(measured[:patches], 1):
                other = _lookalike(proposal, images, identity_id)
                if other is not None:
                    row, box, distance, ratio, role = other
                    values, altered = _measure(module, images[row], box, gallery, hashes, identity_id, target_device)
                    results.append((*other, values, altered))
                for row, box, distance, ratio, role, values, altered in results:
                    image = images[row]
                    sample = image["sample"]
                    key = f"observation_{len(records)}"
                    rgb = altered.numpy().transpose(1, 2, 0) * preprocessing["std"] + preprocessing["mean"]
                    arrays[f"{key}_original"] = image["rgb"]
                    arrays[f"{key}_altered"] = np.uint8(np.rint(np.clip(rgb, 0, 1) * 255))
                    arrays[f"{key}_input"] = image["tensor"].numpy()
                    arrays[f"{key}_ablated_input"] = altered.numpy()
                    scale = np.array([image["source_shape"][1], image["source_shape"][0]] * 2) / image["rgb"].shape[0]
                    records.append(dict(patch=number, identity_id=sample.identity_id, focal_identity=identity_id,
                        clip_id=sample.clip_id, source_ref=sample.image_ref.key, pixel_sha256=image["digest"],
                        role=role, box=box, source_box=(np.array(box) * scale).tolist(), array_key=key,
                        descriptor_distance=distance, match_ratio=ratio, **values))
    finally:
        module.to(previous_device).train(training)
    columns = ["patch", "identity_id", "focal_identity", "clip_id", "source_ref", "pixel_sha256", "role",
               "box", "source_box", "array_key", "descriptor_distance", "match_ratio", "margin_before",
               "margin_after", "effect", "control_box", "control_effect", "effect_minus_control",
               "prediction", "competitor", "excluded_reference_rows"]
    table = pd.DataFrame(records, columns=columns)
    metadata["clip_summary"] = []
    for patch, group in table[table.role != "other_identity"].groupby("patch"):
        effects = group.groupby("clip_id").effect.mean()
        metadata["clip_summary"].append(dict(patch=int(patch), clips=len(effects),
            mean_effect=float(effects.mean()), min_effect=float(effects.min()), max_effect=float(effects.max()),
            positive_clip_fraction=float((effects > 0).mean())))
    table.attrs.update(metadata)
    if records:
        _render_markings(records, arrays, metadata, options=options)
    else:
        warnings.warn("No cross-clip texture matches passed the checks in the inspected images; no marking evidence was inferred.",
                      RuntimeWarning, stacklevel=2)
    if destination is not None:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path.with_suffix(".csv"), index=False)
        path.with_suffix(".json").write_text(json.dumps(dict(settings=metadata, observations=records), indent=2))
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
    return table
