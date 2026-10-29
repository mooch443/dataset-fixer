import json
import random
from threading import Barrier, get_ident

import albumentations as A
import numpy as np
import pytest

from dataset_fixer import synthetic
from dataset_fixer.artifacts import iter_lineage, lineage_path
from dataset_fixer.id.data import index_of


def generate(tmp_path, name, *, workers=1, **options):
    settings = dict(individuals=3, clips=3, frames_per_clip=3, size=(256, 192),
                    augmentations=[A.ChromaticAberration(p=1), A.OpticalDistortion(distort_limit=.03, p=1)])
    settings.update(options)
    return synthetic.generate(**settings, workers=workers, destination=tmp_path / name)


@pytest.mark.parametrize("task", ["segment", "detect", "pose", "polo"])
def test_workers_preserve_pixels_annotations_provenance_and_crops(tmp_path, task):
    before_numpy, before_python = np.random.get_state(), random.getstate()
    serial = generate(tmp_path, "serial", task=task)
    parallel = generate(tmp_path, "parallel", task=task, workers=3, progress=False)
    assert serial.manifest["synthetic"] == parallel.manifest["synthetic"]
    for left, right in zip(serial._samples, parallel._samples, strict=True):
        assert left.image_path.read_bytes() == right.image_path.read_bytes()
        assert left.provenance == right.provenance
    a, b = [json.loads((data.location / "annotations.json").read_text()) for data in (serial, parallel)]
    for field in ("images", "annotations", "categories"):
        assert a[field] == b[field]
    assert list(iter_lineage(lineage_path(serial.location))) == list(iter_lineage(lineage_path(parallel.location)))
    crops = [synthetic.identity(data, workers=workers, destination=tmp_path / f"crops-{name}")
             for name, data, workers in (("serial", serial, 1), ("parallel", parallel, 3))]
    assert (crops[0].location / "samples.jsonl").read_bytes() == (crops[1].location / "samples.jsonl").read_bytes()
    for left, right in zip(crops[0]._samples, crops[1]._samples, strict=True):
        assert left.image_path.read_bytes() == right.image_path.read_bytes()
    after = np.random.get_state()
    assert before_numpy[0] == after[0] and before_numpy[2:] == after[2:]
    np.testing.assert_array_equal(before_numpy[1], after[1])
    assert before_python == random.getstate()


def test_parallel_prefix_and_cache_reuse(tmp_path, monkeypatch):
    full = generate(tmp_path, "full", workers=3)
    short = generate(tmp_path, "short", workers=2, clips=2, frames_per_clip=2)
    lookup = {(s.clip_id, s.frame_index): s for s in full._samples}
    for sample in short._samples:
        other = lookup[sample.clip_id, sample.frame_index]
        assert sample.image_path.read_bytes() == other.image_path.read_bytes()
        assert sample.provenance["synthetic"] == other.provenance["synthetic"]
    monkeypatch.setattr(synthetic, "cache_root", lambda: tmp_path / "cache")
    options = dict(individuals=2, clips=2, frames_per_clip=1, size=(128, 96), augmentations=[])
    cached = synthetic.generate(**options)
    monkeypatch.setattr(synthetic, "individual", lambda *args: pytest.fail("Worker count must not invalidate cache"))
    monkeypatch.setattr(synthetic, "tqdm", lambda **kwargs: pytest.fail("Cached datasets must not show rendering progress"))
    assert synthetic.generate(**options, workers=8, progress=False).location == cached.location


def test_clips_run_on_distinct_threads(tmp_path, monkeypatch):
    barrier, threads = Barrier(2), set()
    base = synthetic.Background

    class ConcurrentBackground(base):
        def frame(self, frame):
            threads.add(get_ident())
            barrier.wait(timeout=10)
            return super().frame(frame)

    monkeypatch.setattr(synthetic, "Background", ConcurrentBackground)
    data = generate(tmp_path, "concurrent", clips=2, frames_per_clip=1, workers=8)
    assert len(threads) == 2
    assert [s.clip_id for s in data._samples] == ["clip-0000", "clip-0001"]


def test_worker_failure_cleans_staging_after_other_workers_stop(tmp_path, monkeypatch):
    base = synthetic.Trajectory
    barrier = Barrier(2)

    class FailingTrajectory(base):
        def __init__(self, settings, clip, label):
            if clip in {"clip-0000", "clip-0001"} and label == "individual-0000":
                barrier.wait(timeout=10)
                if clip == "clip-0001":
                    raise RuntimeError("worker failed")
            super().__init__(settings, clip, label)

    monkeypatch.setattr(synthetic, "Trajectory", FailingTrajectory)
    with pytest.raises(RuntimeError, match="worker failed"):
        generate(tmp_path, "failed", workers=2)
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".failed.tmp-*"))


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_worker_counts_fail_before_writing(tmp_path, workers):
    with pytest.raises(ValueError, match="workers must be a positive integer"):
        generate(tmp_path, "bad", workers=workers)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("normalization", ["posture", "none"])
def test_identity_workers_preserve_jitter_and_tracklet_breaks(tmp_path, normalization):
    scenes = generate(tmp_path, "scenes", clips=2, frames_per_clip=5, individual_scale=.35, augmentations=[])
    for sample in scenes._samples:
        if sample.frame_index == 2:
            sample.provenance["synthetic"]["observations"][0]["usable"] = False
    serial = synthetic.identity(scenes, normalization=normalization, destination=tmp_path / "serial")
    parallel = synthetic.identity(scenes, normalization=normalization, workers=8, destination=tmp_path / "parallel")
    assert serial.manifest["settings"] == parallel.manifest["settings"]
    assert (serial.location / "samples.jsonl").read_bytes() == (parallel.location / "samples.jsonl").read_bytes()
    for a, b in zip(serial._samples, parallel._samples, strict=True):
        assert a.image_path.read_bytes() == b.image_path.read_bytes()
    frame = index_of(parallel).frame
    selected = frame.loc[frame.identity_id == "individual-0000"]
    assert not (selected.frame_index == 2).any()
    assert selected.groupby("clip_id").tracklet_id.nunique().tolist() == [2, 2]


def test_identity_clips_run_on_distinct_threads_and_reuse_cache(tmp_path, monkeypatch):
    scenes = generate(tmp_path, "scenes", clips=2, frames_per_clip=1, individual_scale=.35, augmentations=[])
    barrier, threads = Barrier(2), set()
    original = synthetic.sha256_file

    def concurrent_hash(path):
        threads.add(get_ident())
        barrier.wait(timeout=10)
        return original(path)

    monkeypatch.setattr(synthetic, "sha256_file", concurrent_hash)
    monkeypatch.setattr(synthetic, "cache_root", lambda: tmp_path / "cache")
    data = synthetic.identity(scenes, workers=8)
    assert len(threads) == 2
    monkeypatch.setattr(synthetic, "sha256_file", lambda *args: pytest.fail("Worker count must not invalidate crop cache"))
    assert synthetic.identity(scenes, workers=1).location == data.location


def test_identity_worker_failure_cleans_staging_after_other_workers_stop(tmp_path, monkeypatch):
    scenes = generate(tmp_path, "scenes", clips=2, frames_per_clip=2, individual_scale=.35, augmentations=[])
    barrier = Barrier(2)
    original = synthetic.sha256_file

    def failing_hash(path):
        if path.stem == "000000":
            barrier.wait(timeout=10)
            if path.parent.name == "clip-0001":
                raise RuntimeError("crop worker failed")
        return original(path)

    monkeypatch.setattr(synthetic, "sha256_file", failing_hash)
    with pytest.raises(RuntimeError, match="crop worker failed"):
        synthetic.identity(scenes, workers=2, destination=tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".failed.tmp-*"))


def test_identity_invalid_worker_counts_fail_before_writing(tmp_path):
    scenes = generate(tmp_path, "scenes", clips=1, frames_per_clip=1, augmentations=[])
    for workers in [0, -1, True, 1.5]:
        with pytest.raises(ValueError, match="workers must be a positive integer"):
            synthetic.identity(scenes, workers=workers, destination=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
