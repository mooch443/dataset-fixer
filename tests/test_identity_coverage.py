import copy

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from dataset_fixer import Dataset, id
from dataset_fixer.id.data import ImageRef, SampleIndex, as_dataset, index_of


@pytest.fixture
def coverage_data(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    records = []
    for split, count in (("train", 5), ("val", 2), ("test", 2)):
        for clip in range(count):
            for label in ("a", "b", None):
                for track in range(3 if label else 1):
                    for frame in range(track + 1):
                        key = f"{split}-{clip}-{label}-{track}-{frame}"
                        records.append(dict(path=str(source / f"{key}.png"), width=8, height=8,
                            identity_id=label, clip_id=f"{split}-{clip}" if label else None,
                            tracklet_id=str(track) if label else None, frame_index=track * 10 + frame,
                            original_id=key, split=split, synthetic='{"visible_fraction": 1.0}'))
    data = as_dataset(SampleIndex(records), location=source)
    data._manifest["synthetic"] = {"individuals": {"a": [0.2], "b": [0.4]}}
    return data


def paths(data, label=None):
    frame = index_of(data).frame
    return set(frame.loc[frame.identity_id == label, "path"] if label else frame.path)


def test_limits_keep_whole_groups_and_held_out_data_without_reading_pixels(coverage_data, monkeypatch):
    data = coverage_data.with_sampling(samples_per_epoch=100, balance_by=("identity_id", "clip_id", "tracklet_id"))
    source = index_of(data).frame.copy(deep=True)
    manifest = copy.deepcopy(data.manifest)
    monkeypatch.setattr(ImageRef, "read", lambda self: pytest.fail("Coverage must not read pixels"))
    limits = {"a": {"clips": 1, "tracklets_per_clip": 1}}
    result = data.limit_coverage(limits)
    frame = index_of(result).frame
    selected = frame.loc[(frame.identity_id == "a") & (frame.split == "train")]
    assert selected.clip_id.nunique() == selected.tracklet_id.nunique() == 1
    expected = source.loc[(source.split != "train") | (source.identity_id != "a") |
                          (source.clip_id.isin(selected.clip_id) & source.tracklet_id.isin(selected.tracklet_id))]
    pd.testing.assert_frame_equal(frame, expected.reset_index(drop=True))
    pd.testing.assert_frame_equal(index_of(data).frame, source)
    assert data.manifest == manifest
    assert index_of(result).sampling == index_of(data).sampling
    assert result.manifest["synthetic"] == manifest["synthetic"]
    assert result.manifest["coverage_limits"] == [dict(limits=limits, split="train", seed=42)]
    limits["a"]["clips"] = 5
    assert result.manifest["coverage_limits"][0]["limits"]["a"]["clips"] == 1
    # The scarce identity remains trainable, with independent held-out queries.
    protocol = id.Protocol(result)
    assert set(index_of(protocol.training).frame.identity_id.dropna()) == {"a", "b"}
    assert set(index_of(protocol.references).frame.clip_id).isdisjoint(
        index_of(protocol.queries["test"]).frame.clip_id)


def test_selection_is_repeatable_nested_and_independent_of_other_limits(coverage_data):
    data = coverage_data
    small = {"a": {"clips": 1, "tracklets_per_clip": 1}}
    large = {"a": {"clips": 3, "tracklets_per_clip": 2}}
    result = data.limit_coverage(small, seed=7)
    assert paths(result) == paths(data.limit_coverage(small, seed=7))
    assert paths(result) <= paths(data.limit_coverage(large, seed=7))
    assert paths(result) == paths(data.limit_coverage(large, seed=7).limit_coverage(small, seed=7))
    together = data.limit_coverage({"b": {"clips": 2}, **small}, seed=7)
    assert paths(together, "a") == paths(result, "a")
    reordered = as_dataset(SampleIndex(index_of(data).frame.sample(frac=1, random_state=99)), parent=data)
    assert paths(reordered.limit_coverage(small, seed=7)) == paths(result)
    assert len({tuple(sorted(paths(data.limit_coverage(small, seed=seed)))) for seed in range(8)}) > 1


def test_clip_and_tracklet_caps_work_independently_and_across_splits(coverage_data):
    data = coverage_data
    clipped = data.limit_coverage({"a": {"clips": 1}}, split="validation")
    stats = id.stats(clipped, by=("identity_id", "split"))
    assert stats.loc[(stats.identity_id == "a") & (stats.split == "val"), "clips"].item() == 1
    assert paths(clipped.filter(identity_ids=["a"], split="train")) == paths(data.filter(identity_ids=["a"], split="train"))
    tracked = data.limit_coverage({"a": {"tracklets_per_clip": 1}})
    train = index_of(tracked).frame.query("identity_id == 'a' and split == 'train'")
    assert train.clip_id.nunique() == 5
    assert train.groupby("clip_id", observed=True).tracklet_id.nunique().eq(1).all()
    global_view = data.limit_coverage({"a": {"clips": 1, "tracklets_per_clip": 1}}, split=None)
    assert index_of(global_view).frame.loc[lambda f: f.identity_id == "a", "clip_id"].nunique() == 1
    for limits in ({}, {"a": {}}, {"a": {"clips": 999, "tracklets_per_clip": 999}}):
        pd.testing.assert_frame_equal(index_of(data.limit_coverage(limits)).frame, index_of(data).frame)


@pytest.mark.parametrize("limits, kwargs, error", [
    ([], {}, "limits must map"),
    ({1: {"clips": 1}}, {}, "identity strings"),
    ({"a": 1}, {}, "cap mappings"),
    ({"a": {"tracklets": 1}}, {}, "support only"),
    ({"a": {"clips": 0}}, {}, "positive integers"),
    ({"a": {"clips": True}}, {}, "positive integers"),
    ({"a": {"clips": 1.5}}, {}, "positive integers"),
    ({"a": {"tracklets_per_clip": -1}}, {}, "positive integers"),
    ({"missing": {"clips": 1}}, {}, "No observations"),
    ({"a": {"clips": 1}}, {"seed": True}, "nonnegative integer"),
    ({"a": {"clips": 1}}, {"seed": -1}, "nonnegative integer"),
    ({"a": {"clips": 1}}, {"seed": 1.5}, "nonnegative integer"),
    ({"a": {"clips": 1}}, {"split": "invalid"}, "split"),
])
def test_invalid_coverage_policies_fail_early(coverage_data, limits, kwargs, error):
    with pytest.raises((TypeError, ValueError), match=error):
        coverage_data.limit_coverage(limits, **kwargs)


def test_only_required_metadata_in_selected_scope_is_validated(coverage_data):
    frame = index_of(coverage_data).frame.copy()
    frame["tracklet_id"] = None
    data = as_dataset(SampleIndex(frame), parent=coverage_data)
    assert len(index_of(data.limit_coverage({"a": {"clips": 1}}))) < len(frame)
    with pytest.raises(ValueError, match="non-missing.*tracklet_id"):
        data.limit_coverage({"a": {"tracklets_per_clip": 1}})
    frame.loc[frame.split == "train", "clip_id"] = None
    data = as_dataset(SampleIndex(frame), parent=coverage_data)
    with pytest.raises(ValueError, match="non-missing.*clip_id"):
        data.limit_coverage({"a": {"clips": 1}})
    assert data.limit_coverage({"a": {"clips": 1}}, split="val")
    only_train = as_dataset(index_of(coverage_data).where(split="train"), parent=coverage_data)
    with pytest.raises(ValueError, match="Unknown split"):
        only_train.limit_coverage({"a": {"clips": 1}}, split="val")
    frame = index_of(coverage_data).frame.loc[lambda f: (f.identity_id != "a") | (f.split == "test")]
    with pytest.raises(ValueError, match="No observations"):
        as_dataset(SampleIndex(frame), parent=coverage_data).limit_coverage({"a": {"clips": 1}})


def test_coverage_requires_an_identity_dataset(detect_dataset):
    with pytest.raises(ValueError, match="id.from_dataset"):
        Dataset.open(detect_dataset).limit_coverage({"a": {"clips": 1}})


@pytest.mark.parametrize("copy_images", [False, True])
def test_limited_view_exports_and_reopens_with_metadata(coverage_data, tmp_path, copy_images):
    data = coverage_data.limit_coverage({"a": {"clips": 1, "tracklets_per_clip": 1}})
    for sample in data._samples:
        Image.fromarray(np.full((8, 8, 3), sample.frame_index, np.uint8)).save(sample.image_path)
    exported = data.export(destination=tmp_path / "export", format="identity_manifest", copy_images=copy_images)
    loaded = Dataset.open(exported.location / "samples.jsonl", format="identity_manifest")
    assert loaded.manifest["coverage_limits"] == data.manifest["coverage_limits"]
    columns = ["identity_id", "clip_id", "tracklet_id", "split", "frame_index", "original_id", "synthetic"]
    pd.testing.assert_frame_equal(index_of(loaded).frame[columns], index_of(data).frame[columns], check_categorical=False)
    for before, after in zip(data._samples, loaded._samples):
        np.testing.assert_array_equal(before.image_ref.read(), after.image_ref.read())
