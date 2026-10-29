"""Streaming video I/O and exports around the canonical image prediction API."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from .models import DatasetMetadata, Sample, Task
from .utils import IMAGE_SUFFIXES, VIDEO_SUFFIXES, sha256_file, slugify, to_jsonable
from .visualization import MarkerStyle, VisualizationOptions, annotation_item, annotations_from_values

if TYPE_CHECKING:
    from .dataset import Dataset
    from .model import Model


def _h264_stream(writer, *, width, height, rate, time_base):
    """Shared encoder settings for scene and prediction videos."""
    stream = writer.add_stream("libx264", rate=rate)
    stream.width, stream.height = width, height
    stream.pix_fmt = "yuv420p" if width % 2 == height % 2 == 0 else "yuv444p"
    stream.time_base = stream.codec_context.time_base = time_base
    stream.codec_context.max_b_frames = 0
    stream.options = {"crf": "18", "preset": "fast"}
    return stream


def _display_video(path):
    from IPython import get_ipython
    from IPython.display import Video, display

    if get_ipython() is not None:
        display(Video(filename=str(path), embed=True, html_attributes="controls loop", width=800))
    else:
        print(f"Video: {path}")


def export_video(dataset: Dataset, *, clip_id, fps, destination, annotations, markers, show, overwrite):
    """Stream one indexed scene clip to an atomic, notebook-playable MP4."""
    import av
    from .sources import cache_root

    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean")
    if dataset.task is Task.ID:
        raise ValueError("Export the scene Dataset as video, not its individual identity crops")
    if dataset._plan:
        raise ValueError("Export pending dataset operations before exporting video")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    dataset._ensure_provenance(progress=False)
    if any(s.clip_id is None or s.frame_index is None for s in dataset._samples):
        raise ValueError("Video export requires clip_id and frame_index metadata on every scene")
    clips = sorted({s.clip_id for s in dataset._samples})
    if clip_id is None:
        if len(clips) != 1:
            raise ValueError(f"Select clip_id from the available clips: {clips}")
        clip_id = clips[0]
    samples = [s for s in dataset._samples if s.clip_id == clip_id]
    if not samples:
        raise ValueError(f"Unknown clip_id {clip_id!r}; available clips: {clips}")
    if any(isinstance(s.frame_index, bool) or not isinstance(s.frame_index, (int, np.integer)) or s.frame_index < 0 for s in samples):
        raise ValueError("frame_index values must be nonnegative integers")
    samples.sort(key=lambda s: s.frame_index)
    indexes = [s.frame_index for s in samples]
    if len(set(indexes)) != len(indexes):
        raise ValueError("A scene clip must have exactly one image per frame_index")
    width, height = samples[0].width, samples[0].height
    if any((s.width, s.height) != (width, height) for s in samples):
        raise ValueError("All frames in a clip must have identical dimensions")
    rate = Fraction(str(fps)).limit_denominator(1_000_000)
    if rate <= 0:
        raise ValueError("fps is too small to represent as a video frame rate")
    time_base = 1 / rate
    cached = destination is None
    if cached:
        # Annotation renderer revision; retain reusable raw-video cache entries.
        content = dict(version=2 if annotations else 1, fps=str(rate), annotations=annotations, markers=asdict(markers) if markers else None,
                       task=dataset.task.value, metadata=asdict(dataset._metadata),
                       frames=[(int(s.frame_index), sha256_file(s.image_path),
                                [asdict(a) for a in s.annotations] if annotations else None,
                                sha256_file(dataset._mask_paths[s.image_path])
                                if annotations and dataset.format == "semantic_masks" else None) for s in samples])
        digest = hashlib.sha256(json.dumps(to_jsonable(content), sort_keys=True).encode()).hexdigest()[:24]
        destination = cache_root() / "videos" / f"{slugify(clip_id)}-{digest}.mp4"
    destination = Path(destination).expanduser().resolve()
    if destination.suffix.lower() != ".mp4":
        raise ValueError("Video destination must be an .mp4 file")
    if destination.exists() and not cached and not overwrite:
        raise FileExistsError(f"Destination already exists: {destination}")
    if overwrite or not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        options = VisualizationOptions(markers=markers or MarkerStyle(show_labels=False))
        # H.264 4:2:0 is widely playable in macOS browsers. Pad odd dimensions by
        # one pixel at the right/bottom rather than changing scene coordinates.
        encoded_size = (width + width % 2, height + height % 2)
        with tempfile.TemporaryDirectory(prefix=".video-", dir=destination.parent) as temporary:
            staged = Path(temporary) / "clip.mp4"
            with av.open(str(staged), "w", options={"movflags": "+faststart"}) as writer:
                stream = _h264_stream(writer, width=encoded_size[0], height=encoded_size[1], rate=rate, time_base=time_base)
                durations = {}

                def mux(packet):
                    duration = durations.pop(packet.pts * packet.time_base)
                    packet.duration = max(1, round(duration / packet.time_base))
                    writer.mux(packet)

                for number, sample in enumerate(samples):
                    with Image.open(sample.image_path) as source:
                        image = source.convert("RGB")
                    if image.size != (width, height):
                        raise ValueError(f"Frame dimensions changed: {sample.image_path}")
                    if annotations:
                        mask = None
                        if dataset.format == "semantic_masks":
                            with Image.open(dataset._mask_paths[sample.image_path]) as opened:
                                mask = np.asarray(opened.convert("L")) > 0
                        panel = annotation_item(sample, dataset.task, dataset._metadata, options, image=image, mask=mask)
                        image = panel.panels[0].render((0, 0, width, height), (width, height))
                    if image.size != encoded_size:
                        canvas = Image.new("RGB", encoded_size)
                        canvas.paste(image, (0, 0))
                        image = canvas
                    frame = av.VideoFrame.from_image(image)
                    frame.pts, frame.time_base = int(sample.frame_index - indexes[0]), time_base
                    frame.duration = int(indexes[number + 1] - indexes[number]) if number + 1 < len(samples) else 1
                    durations[frame.pts * time_base] = frame.duration * time_base
                    for packet in stream.encode(frame):
                        mux(packet)
                for packet in stream.encode():
                    mux(packet)
            # Confirm the published file decodes completely, streaming one frame
            # at a time, before making it visible to a notebook or video player.
            with av.open(str(staged)) as probe:
                count = sum(1 for _ in probe.decode(video=0))
            if count != len(samples):
                raise ValueError(f"Video encoded {count} of {len(samples)} frames")
            if overwrite:
                os.replace(staged, destination)  # Old output survives any encoding failure.
            else:
                os.link(staged, destination)  # Atomic publication; refuses replacement.
    if show:
        _display_video(destination)
    return destination


@dataclass(frozen=True)
class VideoPrediction:
    """Saved predictions for one video, without retaining decoded frames.

    Parameters:
        source: Original input video.
        video_path: Rendered prediction video.
        csv_path: Per-frame prediction table.
        frames: Number of processed frames.
        width: Output frame width in pixels.
        height: Output frame height in pixels.
        fps: Output frame rate.
        objects: Total object predictions.
        inference_seconds: Time spent predicting frames.
        audio: Whether the rendered video retains audio.
    """

    source: Path
    video_path: Path
    csv_path: Path
    frames: int
    width: int
    height: int
    fps: float
    objects: int
    inference_seconds: float
    audio: bool


@dataclass(frozen=True)
class VideoPredictionResult:
    """Video artifacts returned by Model.predict, in input order.

    ``result[0].video_path`` and ``result[0].csv_path`` locate the first pair.
    The manifest includes model identity, effective settings, and class and
    keypoint names. Frames and predictions are streamed instead of retained.

    Parameters:
        model_name: Predicting model's display name.
        task: Prediction task.
        location: Output artifact directory.
        records: Saved artifacts in input-video order.
    """

    model_name: str
    task: str
    location: Path
    records: tuple[VideoPrediction, ...]

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, index: int) -> VideoPrediction:
        return self.records[index]

    def summary(self) -> dict[str, Any]:
        return {"model": self.model_name, "task": self.task, "videos": len(self),
                "frames": sum(record.frames for record in self),
                "objects": sum(record.objects for record in self),
                "inference_seconds": sum(record.inference_seconds for record in self),
                "location": str(self.location)}


def video_sources(source: Any) -> tuple[Path, ...] | None:
    """Recognize homogeneous video requests; leave image normalization intact."""
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve()
        paths = sorted(p for p in path.rglob("*") if p.is_file()
                       and p.suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES) if path.is_dir() else [path]
    elif isinstance(source, Sequence) and source and all(isinstance(p, (str, Path)) for p in source):
        paths = [Path(p).expanduser().resolve() for p in source]
    else:
        return None
    if not any(p.suffix.lower() in VIDEO_SUFFIXES for p in paths):
        return None
    if any(p.suffix.lower() not in VIDEO_SUFFIXES for p in paths):
        raise ValueError("Predict images and videos in separate calls; mixed sources are not supported")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if len(set(paths)) != len(paths):
        raise ValueError("Video source contains duplicate paths")
    return tuple(paths)


def video_destination(source: Any, paths: tuple[Path, ...]) -> Path:
    anchor = Path(source).expanduser().resolve() if isinstance(source, (str, Path)) else paths[0]
    return anchor.with_name(f"{anchor.stem}-predictions")


def _model_metadata(model: Model) -> DatasetMetadata:
    manifest = model._resolved_source.manifest
    schema = (manifest.get("dataset_schema") or dict(manifest.get("model") or {}).get("dataset_schema")
              or manifest.get("checkpoint_schema") or {})
    names = schema.get("classes") or manifest.get("class_names") or {}
    names = dict(enumerate(names)) if isinstance(names, list) else names
    return DatasetMetadata(names={int(k): str(v) for k, v in names.items()},
                           kpt_names={int(k): list(v) for k, v in (schema.get("kpt_names") or {}).items()},
                           kpt_shape=tuple(schema["kpt_shape"]) if schema.get("kpt_shape") else None,
                           extra={"skeleton": schema.get("skeleton", [])})


_CSV_FIELDS = ["video", "frame", "timestamp_seconds", "record_type", "object_index",
               "class_id", "class_name", "confidence", "x_min", "y_min", "x_max", "y_max",
               "point_x", "point_y", "radius", "polygons_json", "foreground_pixels", "foreground_fraction"]


def _prediction_rows(record, *, source, frame_index, timestamp, metadata):
    base = {"video": str(source), "frame": frame_index, "timestamp_seconds": timestamp}
    if record.mask is not None:
        count = int(np.count_nonzero(record.mask))
        yield {**base, "record_type": "semantic", "foreground_pixels": count,
               "foreground_fraction": count / (record.width * record.height)}
    elif not record.objects:
        yield {**base, "record_type": "empty"}
    else:
        for index, obj in enumerate(record.objects):
            row = {**base, "record_type": "object", "object_index": index,
                   "class_id": obj.class_id, "class_name": metadata.names.get(obj.class_id, str(obj.class_id)),
                   "confidence": obj.score}
            if obj.bbox is not None:
                row.update(zip(("x_min", "y_min", "x_max", "y_max"), obj.bbox))
            if obj.point is not None:
                row.update(point_x=obj.point[0], point_y=obj.point[1], radius=obj.radius)
            polygons = obj.polygons or ([obj.polygon] if obj.polygon else [])
            if polygons:
                row["polygons_json"] = json.dumps(polygons, separators=(",", ":"))
            for keypoint, (x, y, confidence) in enumerate(obj.keypoints or []):
                row.update({f"keypoint_{keypoint}_x": x, f"keypoint_{keypoint}_y": y,
                            f"keypoint_{keypoint}_confidence": confidence})
            yield row


def predict_videos(model, source, paths, *, destination, markers, predict_kwargs):
    """Stage complete video/CSV pairs atomically using bounded image batches."""
    if markers is not None and not isinstance(markers, MarkerStyle):
        raise TypeError("markers must be a MarkerStyle or None")
    root = Path(destination).expanduser().resolve() if destination is not None else video_destination(source, paths)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError(f"Prediction destination is not empty: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    metadata = _model_metadata(model)
    names = [slugify(path.stem) for path in paths]
    outputs, settings = [], []
    with tempfile.TemporaryDirectory(prefix=".video-predictions-", dir=root.parent) as temporary:
        stage = Path(temporary) / "output"
        stage.mkdir()
        for path, name in zip(paths, names):
            if sum(value.casefold() == name.casefold() for value in names) > 1:
                name += "-" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
            record, task, effective = _predict_video(model, path, stage / name, metadata, markers, predict_kwargs)
            outputs.append(replace(record, video_path=root / record.video_path.name, csv_path=root / record.csv_path.name))
            settings.append(effective)
        result = VideoPredictionResult(model.name, task, root, tuple(outputs))
        manifest = {"schema": 1, "kind": "video-predictions", "model": model.describe(),
                    "summary": result.summary(), "videos": [asdict(row) for row in outputs],
                    "settings": settings, "annotation_metadata": asdict(metadata),
                    "markers": asdict(markers) if markers else None,
                    "csv": {"indices": "zero-based", "coordinates": "displayed-frame pixels after rotation",
                            "timestamps": "seconds from first video frame", "object_index": "per-frame, not a track ID"}}
        (stage / "manifest.json").write_text(json.dumps(to_jsonable(manifest), indent=2), encoding="utf-8")
        stage.rename(root)
    return result


def _predict_video(model, source, output, metadata, markers, predict_kwargs):
    import av
    from tqdm.auto import tqdm
    from .model import ModelInput

    video_path, csv_path = output.parent / f"{output.name}.mp4", output.parent / f"{output.name}.csv"
    batch_size = predict_kwargs.get("batch_size") or model.batch_size
    chunk_size = min(128, batch_size) if batch_size > 0 else 16
    options = VisualizationOptions(markers=markers or MarkerStyle(show_labels=False))
    kwargs = {**predict_kwargs, "progress": False}
    fields = list(_CSV_FIELDS)
    for index in range(metadata.kpt_shape[0] if metadata.kpt_shape else 0):
        fields.extend(f"keypoint_{index}_{axis}" for axis in ("x", "y", "confidence"))
    pending = []
    frame_count = object_count = 0
    inference_seconds = 0.0
    first_time = None
    last_time = None
    task, effective = None, {}

    with tempfile.TemporaryDirectory(prefix="frames-", dir=output.parent) as temporary, av.open(str(source)) as reader:
        if not reader.streams.video:
            raise ValueError(f"Video contains no video stream: {source}")
        incoming = reader.streams.video[0]
        rate = incoming.average_rate or incoming.guessed_rate
        if rate is None or not math.isfinite(float(rate)) or rate <= 0:
            raise ValueError(f"Video has no valid frame rate: {source}")
        # Probe orientation separately so the streaming decoder still sees all
        # audio packets, including those preceding the first video frame.
        with av.open(str(source)) as probe:
            first = next(probe.decode(video=0), None)
            if first is None:
                raise ValueError(f"Video contains no decodable frames: {source}")
            rotation = first.rotation
            width, height = first.to_image().rotate(rotation, expand=True).size
        spool = Path(temporary) / "rows.jsonl"
        with av.open(str(video_path), "w", options={"movflags": "+faststart"}) as writer, spool.open("w", encoding="utf-8") as rows:
            outgoing = _h264_stream(writer, width=width, height=height, rate=rate, time_base=incoming.time_base)
            durations = {}

            def mux_video(packet):
                duration = durations.pop(packet.pts * packet.time_base)
                packet.duration = max(1, round(duration / packet.time_base))
                writer.mux(packet)
            audio_source = next(iter(reader.streams.audio), None)
            audio = writer.add_stream("aac", rate=48000) if audio_source else None
            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=48000) if audio else None
            if audio:
                audio.layout = "stereo"

            def consume():
                nonlocal task, effective, inference_seconds, object_count
                if not pending:
                    return
                result = model.predict(tuple(item[0] for item in pending), **kwargs)
                task, effective = result.task, result.settings
                inference_seconds += result.inference_seconds
                if len(result) != len(pending):
                    raise ValueError("Video inference must return one ordered prediction per frame")
                for (item, pts, time_base, duration, timestamp), record in zip(pending, result):
                    if record.image_id != item.image_id or (record.width, record.height) != (width, height):
                        raise ValueError("Video predictions changed frame order or dimensions")
                    for row in _prediction_rows(record, source=source, frame_index=int(item.image_id.removeprefix("frame_")),
                                                timestamp=timestamp, metadata=metadata):
                        fields.extend(key for key in row if key not in fields)
                        rows.write(json.dumps(to_jsonable(row)) + "\n")
                    object_count += len(record.objects)
                    with Image.open(item.image_path) as opened:
                        sample = Sample(item.image_path, Path(item.relative_path), "train", width, height,
                                        annotations_from_values(record.objects))
                        rendered = annotation_item(sample, Task.SEGMENT if record.mask is not None else Task.parse(task),
                                                   metadata, options, image=opened, mask=record.mask)
                        pixels = rendered.panels[0].render((0, 0, width, height), (width, height))
                    frame = av.VideoFrame.from_image(pixels)
                    frame.pts, frame.time_base = pts, time_base
                    frame.duration = duration
                    durations[pts * time_base] = duration * time_base
                    for packet in outgoing.encode(frame):
                        mux_video(packet)
                    item.image_path.unlink()
                pending.clear()

            streams = [incoming, *([audio_source] if audio_source else [])]
            with tqdm(total=incoming.frames or None, desc=f"{model.name}: {source.name}", unit="frame",
                      disable=not predict_kwargs.get("progress", True)) as progress:
                for packet in reader.demux(streams):
                    for frame in packet.decode():
                        if isinstance(frame, av.AudioFrame):
                            for converted in resampler.resample(frame):
                                for encoded in audio.encode(converted):
                                    writer.mux(encoded)
                            continue
                        pixels = frame.to_image()
                        if rotation:
                            pixels = pixels.rotate(rotation, expand=True)
                        if pixels.size != (width, height) or frame.rotation != rotation:
                            raise ValueError("Video changes resolution or orientation during playback")
                        time_base = frame.time_base or incoming.time_base or Fraction(1, 90000)
                        pts = frame.pts if frame.pts is not None else round(Fraction(frame_count, 1) / rate / time_base)
                        timestamp = pts * time_base
                        if last_time is not None and timestamp <= last_time:
                            raise ValueError("Video frame timestamps must increase monotonically")
                        first_time = timestamp if first_time is None else first_time
                        last_time = timestamp
                        duration = frame.duration or max(1, round(1 / rate / time_base))
                        if incoming.duration is not None:
                            end = ((incoming.start_time or 0) + incoming.duration) * incoming.time_base
                            if end > timestamp:
                                duration = min(duration, max(1, round((end - timestamp) / time_base)))
                        image_path = Path(temporary) / f"frame_{frame_count:09d}.png"
                        pixels.save(image_path, compress_level=1)
                        item = ModelInput(image_path.stem, image_path, width, height,
                                          f"{source.name}/{image_path.name}")
                        pending.append((item, pts, time_base, duration, float(timestamp - first_time)))
                        frame_count += 1
                        if len(pending) >= chunk_size:
                            consume()
                        progress.update()
                consume()
            if not frame_count:
                raise ValueError(f"Video contains no decodable frames: {source}")
            for packet in outgoing.encode():
                mux_video(packet)
            if audio:
                for converted in resampler.resample(None):
                    for packet in audio.encode(converted):
                        writer.mux(packet)
                for packet in audio.encode():
                    writer.mux(packet)
        # Discover keypoint columns over the entire clip, including classes
        # appearing late, while keeping only one batch of predictions in RAM.
        with spool.open(encoding="utf-8") as rows, csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv_writer = csv.DictWriter(handle, fieldnames=fields)
            csv_writer.writeheader()
            csv_writer.writerows(json.loads(row) for row in rows)
    return VideoPrediction(source, video_path, csv_path, frame_count, width, height, float(rate),
                           object_count, inference_seconds, audio_source is not None), task, effective
