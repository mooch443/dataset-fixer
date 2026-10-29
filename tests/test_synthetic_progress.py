import pytest

from dataset_fixer import synthetic


@pytest.fixture
def bars(monkeypatch):
    created = []

    class Progress:
        def __init__(self, **kwargs):
            self.options, self.updates, self.closed = kwargs, [], False
            created.append(self)

        def __enter__(self):
            return self

        def update(self, count):
            self.updates.append(count)

        def __exit__(self, *args):
            self.closed = True

    monkeypatch.setattr(synthetic, "tqdm", Progress)
    return created


@pytest.mark.parametrize("workers", [1, 3])
def test_progress_counts_completed_frames_across_workers(tmp_path, bars, workers):
    data = synthetic.generate(individuals=2, clips=3, frames_per_clip=4, workers=workers,
                              size=(128, 96), augmentations=[], destination=tmp_path / "scenes")
    assert len(bars) == 1
    bar = bars[0]
    assert bar.options["total"] == len(data._samples) == 12
    assert bar.options["unit"] == "frame" and not bar.options["disable"]
    assert bar.updates == [1] * 12 and bar.closed


def test_progress_is_closed_on_worker_failure(tmp_path, monkeypatch, bars):
    def fail(*args):
        raise RuntimeError("render failed")

    monkeypatch.setattr(synthetic, "individual", fail)
    with pytest.raises(RuntimeError, match="render failed"):
        synthetic.generate(individuals=2, clips=3, frames_per_clip=4, workers=3,
                           size=(128, 96), augmentations=[], destination=tmp_path / "failed")
    assert len(bars) == 1 and bars[0].closed and not bars[0].updates
    assert not list(tmp_path.iterdir())


def test_progress_can_be_disabled(tmp_path, capsys):
    synthetic.generate(individuals=2, clips=2, frames_per_clip=2, workers=2, size=(128, 96),
                       augmentations=[], progress=False, destination=tmp_path / "silent")
    output = capsys.readouterr()
    assert not output.out and not output.err


def test_progress_requires_boolean(tmp_path, bars):
    with pytest.raises(TypeError, match="progress must be a boolean"):
        synthetic.generate(progress="yes", destination=tmp_path / "invalid")
    assert not bars and not list(tmp_path.iterdir())
