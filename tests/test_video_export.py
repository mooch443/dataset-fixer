"""Scene video exports must play in notebooks and retain the source timeline."""
from dataclasses import replace

import av
import numpy as np
import pytest
from PIL import Image

from dataset_fixer import Dataset, synthetic


@pytest.fixture
def scenes(tmp_path):
    return synthetic.generate(individuals=2, clips=2, frames_per_clip=4,
                              size=(161, 121), augmentations=[], destination=tmp_path / "scenes")


def decode(path):
    with av.open(str(path)) as video:
        stream = video.streams.video[0]
        frames = list(video.decode(video=0))
        return frames, float(stream.duration * stream.time_base), stream.codec_context.name


def test_video_retains_pixels_order_timing_and_playable_format(scenes, tmp_path):
    # Reopening must recover frame metadata; sample iteration order is irrelevant.
    reopened = Dataset.open(scenes.location)
    reopened._samples.reverse()
    output = reopened.export_video(clip_id="clip-0001", fps=10, destination=tmp_path / "clip.mp4", show=False)
    frames, duration, codec = decode(output)
    assert codec == "h264" and all(f.format.name == "yuv420p" for f in frames)
    assert len(frames) == 4 and frames[0].to_image().size == (162, 122)
    assert [float(f.time) for f in frames] == pytest.approx([0, .1, .2, .3])
    assert duration == pytest.approx(.4)
    for frame, sample in zip(frames, scenes._samples[4:]):
        original = np.asarray(Image.open(sample.image_path)).astype(float)
        rendered = frame.to_ndarray(format="rgb24")[:121, :161].astype(float)
        assert np.abs(original - rendered).mean() < 6  # H.264 is lossy.
    contents = output.read_bytes()
    assert contents.index(b"moov") < contents.index(b"mdat")  # Streaming/inline fast-start.
    with pytest.raises(FileExistsError):
        reopened.export_video(clip_id="clip-0001", destination=output, show=False)
    assert output.read_bytes() == contents
    replaced = reopened.export_video(clip_id="clip-0001", destination=output, fps=5, show=False, overwrite=True)
    assert replaced == output and output.read_bytes() != contents
    frames, duration, _ = decode(output)
    assert len(frames) == 4 and duration == pytest.approx(.8)


def test_video_preserves_gaps(scenes, tmp_path):
    scenes._samples = [replace(s, frame_index=i) for s, i in zip(scenes._samples[:3], [10, 12, 15])]
    output = scenes.export_video(fps=10, destination=tmp_path / "gaps.mp4", show=False)
    frames, duration, _ = decode(output)
    assert [float(f.time) for f in frames] == pytest.approx([0, .2, .5])
    assert duration == pytest.approx(.6)


def test_inline_player_cache_and_overlays(scenes, tmp_path, monkeypatch):
    import IPython
    import IPython.display

    monkeypatch.setattr("dataset_fixer.sources.cache_root", lambda: tmp_path / "cache")
    monkeypatch.setattr(IPython, "get_ipython", lambda: object())
    displayed = []
    monkeypatch.setattr(IPython.display, "display", displayed.append)
    output = scenes.export_video(clip_id="clip-0000")
    html = displayed[0]._repr_html_()
    assert "data:video/mp4;base64," in html and "controls loop" in html
    modified = output.stat().st_mtime_ns
    assert scenes.export_video(clip_id="clip-0000", show=False) == output
    assert output.stat().st_mtime_ns == modified
    output.write_bytes(b"stale cached video")
    assert scenes.export_video(clip_id="clip-0000", overwrite=True, show=False) == output
    assert len(decode(output)[0]) == 4
    annotated = scenes.export_video(clip_id="clip-0000", annotations=True, show=False)
    assert annotated != output
    raw_frames, _, _ = decode(output)
    overlay_frames, _, _ = decode(annotated)
    assert len(overlay_frames) == len(raw_frames)
    assert any(np.any(a.to_ndarray() != b.to_ndarray()) for a, b in zip(raw_frames, overlay_frames))


def test_video_validates_selection_and_cleans_failed_encoding(scenes, tmp_path):
    with pytest.raises(ValueError, match="Select clip_id"):
        scenes.export_video(show=False)
    with pytest.raises(ValueError, match="Unknown clip_id"):
        scenes.export_video(clip_id="missing", show=False)
    for fps in (0, -2, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="fps"):
            scenes.export_video(clip_id="clip-0000", fps=fps, show=False)
    with pytest.raises(ValueError, match=".mp4"):
        scenes.export_video(clip_id="clip-0000", destination=tmp_path / "wrong.avi", show=False)
    scenes._samples = scenes._samples[:4]
    previous = scenes.export_video(destination=tmp_path / "existing.mp4", show=False)
    saved = previous.read_bytes()
    output = tmp_path / "failed.mp4"
    Image.new("RGB", (10, 10)).save(scenes._samples[2].image_path)
    with pytest.raises(ValueError, match="Frame dimensions changed"):
        scenes.export_video(destination=output, show=False)
    assert not output.exists() and not list(tmp_path.glob(".video-*"))
    with pytest.raises(ValueError, match="Frame dimensions changed"):
        scenes.export_video(destination=previous, overwrite=True, show=False)
    assert previous.read_bytes() == saved and not list(tmp_path.glob(".video-*"))


def test_semantic_video_cache_tracks_mask_changes(scenes, tmp_path, monkeypatch):
    monkeypatch.setattr("dataset_fixer.sources.cache_root", lambda: tmp_path / "cache")
    masks = scenes.export(destination=tmp_path / "masks", format="semantic_masks", visualize=False, progress=False)
    first = masks.export_video(clip_id="clip-0000", annotations=True, show=False)
    mask_path = masks._mask_paths[masks._samples[0].image_path]
    with Image.open(mask_path) as mask:
        Image.new("L", mask.size).save(mask_path)
    second = masks.export_video(clip_id="clip-0000", annotations=True, show=False)
    assert first != second
    before, _, _ = decode(first)
    after, _, _ = decode(second)
    assert np.any(before[0].to_ndarray() != after[0].to_ndarray())


def test_video_rejects_ambiguous_frame_metadata(scenes):
    scenes._samples = [scenes._samples[0], scenes._samples[0]]
    with pytest.raises(ValueError, match="exactly one image"):
        scenes.export_video(show=False)
    scenes._samples = [replace(scenes._samples[0], frame_index=None)]
    with pytest.raises(ValueError, match="frame_index metadata"):
        scenes.export_video(show=False)
