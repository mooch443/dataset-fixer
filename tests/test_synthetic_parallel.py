import json
import random
from threading import Barrier, get_ident

import albumentations as A
import numpy as np
import pytest

from dataset_fixer import synthetic
from dataset_fixer.artifacts import lineage_path


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
    assert lineage_path(serial.location).read_bytes() == lineage_path(parallel.location).read_bytes()
    crops = [synthetic.identity(data, destination=tmp_path / f"crops-{name}")
             for name, data in (("serial", serial), ("parallel", parallel))]
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
