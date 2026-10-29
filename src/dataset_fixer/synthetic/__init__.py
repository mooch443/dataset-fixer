"""Small, reproducible multi-individual simulations returning ordinary Datasets.

``generate`` makes annotated scenes; ``identity`` extracts imperfect detections.
Neither function imports optional identity-training dependencies.
"""
from __future__ import annotations

import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import Event, Lock
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from ..artifacts import dataset_info_path, lineage_path, write_json, write_lineage
from ..augmentation import _albumentations, serialize_pipeline
from ..dataset import Dataset
from ..models import Task
from ..sources import cache_root
from ..utils import settings_fingerprint, sha256_file
from ..writer import staged_directory
from ._simulation import (GENES, DNA_LENGTHS, SIMULATION_VERSION, SceneSettings, WeatherSettings, Tracklets, Trajectory,
                          fraction, population, positive_int, rng_for)
from ._rendering import (Background, Weather, SceneEffects, crop_transform, individual,
                         segmentation, visibility)

__all__ = ["generate", "identity"]


def _destination(destination, kind, settings):
    cached = destination is None
    path = cache_root() / "synthetic" / f"{kind}-{settings_fingerprint(settings)}" if cached else Path(destination)
    return path.expanduser().resolve(), cached


def _default_effects():
    A = _albumentations()
    return [A.ChromaticAberration(primary_distortion_limit=0.01, secondary_distortion_limit=0.02, p=0.8),
            A.OpticalDistortion(distort_limit=0.03, p=0.8),
            A.GaussianBlur(sigma_limit=(0.2, 0.6), p=0.3)]


def generate(*, individuals: int | Mapping[str, Sequence[float]] = 12,
             clips: int = 3, frames_per_clip: int = 100, difference: float | None = None,
             variation: float = 0.03, size: tuple[int, int] = (960, 640),
             individual_scale: float = 1.0,
             patterns: Sequence[str] | None = None, neck_focus: float | None = None,
             texture_noise: Mapping[str, float] | None = None,
             backgrounds: Sequence[str] = ("grass", "water"), augmentations: Any = None,
             weather: Mapping[str, float] | None = None,
             task: Literal["segment", "detect", "pose", "polo"] | Task = "segment",
             seed: int = 42, workers: int = 1, progress: bool = True,
             destination: str | Path | None = None) -> Dataset:
    """Simulate shared scenes with stable numeric DNA and consecutive motion.

    Parameters:
        individuals: Count, or identity-string to DNA-vector mapping. Explicit
            vectors use the gene order in ``manifest['synthetic']['gene_names']``.
        clips: Number of independently initialized clips containing the population.
        frames_per_clip: Consecutive scene frames, not an identity-crop budget.
        difference: Population spread in [0, 1], default .25 for a count; omit
            when supplying explicit DNA. Zero gives identical baseline appearances.
        variation: Body wiggle amplitude in [0, 1], default .03. Walking remains
            enabled at zero; limb attachment coordinates and bone lengths stay fixed.
        size: Scene (width, height), each at least 64 pixels.
        individual_scale: Positive size multiplier for bodies, limbs, and markings,
            default 1.0. Does not resize the frame or change initial positions.
        patterns: Allowed body motifs: stripes, dots, stars; None selects all three
            for a generated population. Omit when reusing explicit DNA.
        neck_focus: Concentrate generated identity differences in the band behind
            the eyes, in [0, 1], default 0. At 1 the rest of the appearance is shared.
            Omit for explicit DNA, which already encodes the focus and pattern genes.
        texture_noise: Separate genetic pattern/fur noise strengths in [0, 1].
            None uses pattern=.25, fur=.18; {} disables both. Omitted keys are zero.
            Difference controls noise-field diversity. Omit when reusing explicit DNA.
        backgrounds: Nonempty sequence drawn from 'grass' and 'water'.
        augmentations: Existing Albumentations input; None uses mild camera effects,
            [] disables them. Parameters are sampled once and replayed per clip.
        weather: Moving cloud shadows/fog: strengths clouds/fog in [0, 1] and
            speed multiplier in [0, 10]. None uses clouds=.25, fog=.12, speed=1;
            {} disables weather. In a mapping, omitted strengths are zero.
        task: Existing scene geometry: segment, detect, pose, or polo. Pose exports
            a bounding box and head/tail keypoints, with one connecting skeleton edge.
        seed: Nonnegative integer seed. Shorter runs preserve longer runs' prefixes.
        workers: Positive number of rendering threads, default 1. Independent
            clips run concurrently; frames within a clip remain sequential.
            Does not change generated pixels, metadata, or cache identity.
        progress: Show one notebook/terminal progress bar for completed scene
            frames across all workers, default True. Cached datasets load directly.
        destination: New output directory, or None for a reusable managed cache.

    All scenes initially belong to train. Overlap/visibility losses are recorded
    independently of visible scene annotations; use identity() for valid tracklets.
    """
    clips, frames_per_clip = positive_int(clips, "clips"), positive_int(frames_per_clip, "frames_per_clip")
    workers = min(positive_int(workers, "workers"), clips)
    if not isinstance(progress, bool):
        raise TypeError("progress must be a boolean")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if len(size) != 2:
        raise ValueError("size must be (width, height)")
    size = tuple(positive_int(v, "size") for v in size)
    if min(size) < 64:
        raise ValueError("Scene width and height must be at least 64")
    if isinstance(backgrounds, str):
        backgrounds = (backgrounds,)
    backgrounds = tuple(backgrounds)
    if not backgrounds or set(backgrounds) - {"grass", "water"}:
        raise ValueError("backgrounds must contain grass and/or water")
    task = Task.parse(task)
    if task not in {Task.SEGMENT, Task.DETECT, Task.POSE, Task.POLO}:
        raise ValueError("Generate a scene task, then call synthetic.identity for ID crops")
    if isinstance(individual_scale, bool) or not isinstance(individual_scale, (int, float)) or not np.isfinite(individual_scale) or individual_scale <= 0:
        raise ValueError("individual_scale must be finite and positive")
    dna = population(individuals, difference, seed, patterns=patterns, neck_focus=neck_focus, texture_noise=texture_noise)
    gene_count = len(next(iter(dna.values())))
    config = SceneSettings(size, fraction(variation, "variation"), seed, float(individual_scale))
    atmosphere = WeatherSettings.parse(weather)
    pipeline = serialize_pipeline(_default_effects() if augmentations is None else augmentations, {})
    settings = dict(version=DNA_LENGTHS.index(gene_count) + 1, simulation_version=SIMULATION_VERSION,
                    individuals=dna, clips=clips, frames_per_clip=frames_per_clip,
                    variation=config.variation, size=size, backgrounds=backgrounds, augmentations=pipeline,
                    weather=asdict(atmosphere),
                    individual_scale=config.individual_scale,
                    seed=seed, task=task.value, gene_names=GENES[:gene_count], gene_range=[0, 1])
    world = {k: v for k, v in settings.items() if k not in {"clips", "frames_per_clip", "task"}}
    world_id = hashlib.sha256(json.dumps(world, sort_keys=True).encode()).hexdigest()
    destination, cached = _destination(destination, "scenes", settings)
    if cached and destination.exists():
        return Dataset.open(destination, progress=False)
    with staged_directory(destination) as staging, tqdm(
        total=clips * frames_per_clip, desc="Generating scenes", unit="frame", disable=not progress,
    ) as bar:
        stop = Event()
        progress_lock = Lock()

        def render_clip(clip_index):
            images, annotations, records = [], [], []
            tracks = Tracklets()
            clip = f"clip-{clip_index:04d}"
            background_kind = backgrounds[int(rng_for(seed, "environment", clip).integers(len(backgrounds)))]
            background = Background(size, background_kind, seed, clip)
            weather_effect = Weather(size, atmosphere, seed, clip)
            states = {label: Trajectory(config, clip, label) for label in dna}
            effects = SceneEffects(pipeline, seed, clip)
            for frame in range(frames_per_clip):
                if stop.is_set():
                    break
                pixels = background.frame(frame)
                masks, landmarks, truncated = [], [], []
                for label, genes in dna.items():
                    foreground, mask, points, clipped = individual(genes, states[label], size)
                    pixels[mask > 0] = foreground[mask > 0]
                    masks.append(mask)
                    landmarks.append(points)
                    truncated.append(clipped)
                    states[label].advance()
                pixels = weather_effect.apply(pixels, frame)
                pixels, masks, landmarks = effects.apply(pixels, masks, landmarks)
                visible, observations = visibility(masks, truncated)
                height, width = pixels.shape[:2]
                path = Path("train/images") / clip / f"{frame:06d}.png"
                target = staging / path
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels).save(target)
                image_id = len(images)
                images.append(dict(id=image_id, file_name=path.as_posix(), width=width, height=height))
                frame_annotations = []
                for local_id, (label, observation, shape, points) in enumerate(zip(dna, observations, visible, landmarks)):
                    observation.update(identity_id=label, local_id=local_id, head_tail=points.tolist(),
                                       tracklet_id=tracks.observe(clip, label, frame, observation["usable"]))
                    area = int(shape.sum())
                    observation["visible_fraction"] = area / max(1, int(masks[local_id].sum()))
                    if area < 3:
                        continue
                    ys, xs = np.nonzero(shape)
                    bbox = [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1),
                            float(ys.max() - ys.min() + 1)]
                    annotation = dict(id=len(annotations), image_id=image_id, category_id=0, bbox=bbox,
                                      identity_id=label, area=area, iscrowd=0)
                    if task is Task.SEGMENT:
                        annotation["segmentation"] = segmentation(shape)
                    elif task is Task.POSE:
                        points_in_frame = (points >= 0).all(axis=1) & (points < [width, height]).all(axis=1)
                        points = points.copy()
                        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
                        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
                        visible_points = points_in_frame & shape[np.round(points[:, 1]).astype(int), np.round(points[:, 0]).astype(int)]
                        annotation["keypoints"] = np.column_stack((points, np.where(visible_points, 2, np.where(points_in_frame, 1, 0)))).reshape(-1).tolist()
                        annotation["num_keypoints"] = int(points_in_frame.sum())
                    elif task is Task.POLO:
                        annotation["point"] = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
                        annotation["radius"] = min(bbox[2:]) / 2
                    annotations.append(annotation)
                    frame_annotations.append(dict(identity_id=label, source_id=annotation["id"]))
                # A source observation ID stays stable across exports and crop variants.
                scene_id = hashlib.sha256(f"{world_id}/{clip}/{frame}".encode()).hexdigest()
                digest = sha256_file(target)
                records.append(dict(output_image=path.as_posix(), output_sha256=digest, original_id=scene_id,
                                    sample_metadata=dict(clip_id=clip, frame_index=frame),
                                    annotation_metadata=frame_annotations,
                                    synthetic=dict(observations=observations, background=background_kind,
                                                   scene_id=scene_id, scene_sha256=digest)))
                with progress_lock:
                    bar.update(1)
            return images, annotations, records

        images, annotations, records = [], [], []
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="synthetic") if workers > 1 else None
        try:
            batches = executor.map(render_clip, range(clips)) if executor else map(render_clip, range(clips))
            # Merge in clip order regardless of completion order, keeping COCO
            # IDs and provenance identical to the sequential renderer.
            for clip_images, clip_annotations, clip_records in batches:
                image_offset, annotation_offset = len(images), len(annotations)
                for image in clip_images:
                    image["id"] += image_offset
                for annotation in clip_annotations:
                    annotation["id"] += annotation_offset
                    annotation["image_id"] += image_offset
                for record in clip_records:
                    for annotation in record["annotation_metadata"]:
                        annotation["source_id"] += annotation_offset
                images.extend(clip_images)
                annotations.extend(clip_annotations)
                records.extend(clip_records)
        finally:
            stop.set()
            if executor:
                # Finish active frame writes before staged-directory cleanup.
                executor.shutdown(wait=True, cancel_futures=True)
        category = dict(id=0, name="individual")
        if task is Task.POSE:
            category.update(keypoints=["head", "tail"], skeleton=[[1, 2]])
        elif task is Task.POLO:
            category["radius"] = min(size) * 0.04
        write_json(staging / "annotations.json", dict(info=dict(name=destination.name, task=task.value),
                   images=images, annotations=annotations, categories=[category]))
        write_json(dataset_info_path(staging), dict(name=destination.name, task=task.value, format="coco", synthetic=settings))
        write_lineage(lineage_path(staging), records)
        Dataset.open(staging, progress=False)  # Validate before publishing.
    return Dataset.open(destination, progress=False)


def identity(scenes: Dataset, *, normalization: Literal["posture", "none"] = "posture",
             bbox_jitter: float = 0.05, context_fraction: float = 0.15,
             orientation_jitter: float = 5.0, size: int = 128,
             destination: str | Path | None = None, seed: int | None = None) -> Dataset:
    """Extract valid identity tracklets from generated scenes, with detection error.

    Parameters:
        scenes: Generated Dataset or its ordinary, geometry-preserving export.
        normalization: 'posture' points head lower-right/tail upper-left; 'none'
            retains observed orientation. Rotation includes scene context.
        bbox_jitter: Maximum displacement of each box edge relative to its width/
            height, in [0, .45]. Jitter is temporally correlated within tracklets.
        context_fraction: Nonnegative additional box fraction on each side.
        orientation_jitter: Maximum posture error in degrees, default 5.
        size: Square output resolution, default 128, preserving aspect ratio.
        destination: New repository directory, or None for a reusable managed cache.
        seed: Override randomness; None inherits the scene seed.

    Overlaps and truncated bodies produce no crop and break tracklets. Source
    scene labels remain intact. The result loads with format='identity_manifest'.
    """
    from ..id.data import open_dataset

    synthetic = scenes.manifest.get("synthetic")
    if not synthetic:
        raise ValueError("identity requires a Dataset generated by synthetic.generate")
    if scenes._plan:
        raise ValueError("Export pending dataset operations before extracting synthetic identities")
    if normalization not in {"posture", "none"}:
        raise ValueError("normalization must be 'posture' or 'none'")
    bbox_jitter = fraction(bbox_jitter, "bbox_jitter", maximum=0.45)
    if not np.isfinite(context_fraction) or context_fraction < 0:
        raise ValueError("context_fraction must be finite and nonnegative")
    orientation_jitter = fraction(orientation_jitter, "orientation_jitter", maximum=180)
    size = positive_int(size, "size")
    if size < 2:
        raise ValueError("Crop size must be at least 2")
    seed = synthetic["seed"] if seed is None else seed
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    scenes._ensure_provenance(progress=False)
    samples = sorted(scenes._samples, key=lambda s: (s.clip_id or "", s.frame_index if s.frame_index is not None else -1))
    settings = dict(normalization=normalization, bbox_jitter=bbox_jitter, context_fraction=context_fraction,
                    orientation_jitter=orientation_jitter, size=size, seed=seed,
                    source=[(s.provenance.get("synthetic", {}).get("scene_id"), s.split) for s in samples])
    destination, cached = _destination(destination, "identities", settings)
    if cached and destination.exists():
        return open_dataset(destination / "samples.jsonl", format="identity_manifest")
    tracks = Tracklets()
    jitter_states = {}
    count = 0
    with staged_directory(destination, source=scenes.location) as staging:
        with (staging / "samples.jsonl").open("w") as handle:
            for sample in samples:
                info = sample.provenance.get("synthetic")
                if not info or sha256_file(sample.image_path) != info["scene_sha256"]:
                    raise ValueError("Scene pixels changed since simulation; generate with augmentations= instead of transforming before identity()")
                with Image.open(sample.image_path) as image:
                    pixels = np.asarray(image.convert("RGB"))
                for observation in info["observations"]:
                    label = observation["identity_id"]
                    track = tracks.observe(sample.clip_id, label, sample.frame_index, observation["usable"])
                    if track is None:
                        continue
                    if track not in jitter_states:
                        rng = rng_for(seed, "crop", track)
                        jitter_states[track] = (rng, rng.uniform(-1, 1, 5))
                    rng, jitter = jitter_states[track]
                    jitter = 0.92 * jitter + 0.08 * rng.uniform(-1, 1, 5)
                    jitter_states[track] = (rng, jitter)
                    true = np.asarray(observation["bbox"], dtype=float)
                    widths = np.tile(true[2:] - true[:2], 2)
                    detected = true + jitter[:4] * bbox_jitter * widths
                    matrix = crop_transform(detected, observation["head_tail"], normalization=normalization,
                                            context=context_fraction, angular_jitter=jitter[4] * orientation_jitter, size=size)
                    crop = cv2.warpAffine(pixels, matrix, (size, size), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REFLECT_101)
                    # Paths never incorporate arbitrary user identity strings.
                    relative = Path("images") / sample.clip_id / f"{observation['local_id']:04d}-{sample.frame_index:06d}.png"
                    target = staging / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(crop).save(target)
                    details = dict(source_scene=str(sample.image_path.relative_to(scenes.location)),
                                   scene_id=info["scene_id"], true_bbox=true.tolist(), detected_bbox=detected.tolist(),
                                   head_tail=observation["head_tail"], crop_transform=matrix.tolist(),
                                   visible_fraction=observation["visible_fraction"], normalization=normalization)
                    record = dict(path=relative.as_posix(), width=size, height=size, identity_id=label,
                                  clip_id=sample.clip_id, frame_index=sample.frame_index,
                                  local_id=observation["local_id"], tracklet_id=track, split=sample.split,
                                  original_id=info["scene_id"], synthetic=json.dumps(details, sort_keys=True))
                    handle.write(json.dumps(record) + "\n")
                    count += 1
        if not count:
            raise ValueError("No usable identity crops: increase scene size/frames or reduce the population")
        write_json(dataset_info_path(staging), dict(name=destination.name, task="id", format="identity_manifest",
                   synthetic=synthetic, settings=settings, samples=count))
        open_dataset(staging / "samples.jsonl", format="identity_manifest")
    return open_dataset(destination / "samples.jsonl", format="identity_manifest")
