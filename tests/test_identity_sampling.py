from collections import Counter

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('lightning')

from dataset_fixer.id.data import SampleIndex
from dataset_fixer.id.training import _DiverseBatches


def sample_index():
    return SampleIndex([
        dict(path=f'/unused/{clip}/{tracklet}.png', width=16, height=16,
             identity_id=f'wolf-{tracklet % 2}', clip_id=f'clip-{clip}', tracklet_id=str(tracklet))
        for clip in range(4) for tracklet in range(3)
    ], sampling=dict(balance_by=('identity_id', 'clip_id', 'tracklet_id'), samples_per_epoch=101))


def test_diverse_batches_preserve_weighted_draws_and_full_batch_sizes():
    index = sample_index()
    sampler = _DiverseBatches(index, 8, 7)
    batches = list(sampler)
    assert len(sampler) == len(batches) == 13
    assert [len(batch) for batch in batches] == [8]*12 + [5]
    draws = [key for batch in batches for key in batch]
    expected = list(torch.utils.data.WeightedRandomSampler(sampler.weights, 101, replacement=True,
                                                         generator=torch.Generator().manual_seed(7)))
    assert [row for row, _, _ in sorted(draws, key=lambda item: item[1])] == expected
    assert Counter(row for row, _, _ in draws) == Counter(expected)
    assert batches == list(sampler)
    sampler.epoch = 1
    later = list(sampler)
    assert later != batches and {key[2] for batch in later for key in batch} == {1}


def test_clumped_draws_are_spread_across_clips_and_tracklets(monkeypatch):
    index = sample_index()
    clumped = [row for row in range(12) for _ in range(3)]
    monkeypatch.setattr('dataset_fixer.id.training.WeightedRandomSampler', lambda *a, **k: iter(clumped))
    batches = list(_DiverseBatches(index, 8, 7))
    for batch in batches:
        rows = index.frame.iloc[[key[0] for key in batch]]
        assert rows.clip_id.nunique() == 4
        assert len(rows[['clip_id', 'tracklet_id']].drop_duplicates()) == len(batch)
    assert Counter(key[0] for batch in batches for key in batch) == Counter(clumped)


def test_sparse_diversity_fills_batches_and_explicit_caps_remain_strict():
    index = SampleIndex([
        dict(path=f'/unused/{i}.png', width=16, height=16, identity_id='a', clip_id='one', tracklet_id='one')
        for i in range(3)
    ], sampling=dict(samples_per_epoch=17))
    assert [len(batch) for batch in _DiverseBatches(index, 8, 1)] == [8, 8, 1]
    index.sampling['max_per_batch'] = {'tracklet_id': 2}
    with pytest.raises(TypeError, match='max_per_batch'):
        len(_DiverseBatches(index, 8, 1))
    assert [len(batch) for batch in _DiverseBatches(index, 8, 1)] == [2]*8 + [1]
    index.frame['clip_id'] = None
    index.frame['tracklet_id'] = None
    index.sampling.pop('max_per_batch')
    assert [len(batch) for batch in _DiverseBatches(index, 8, 1)] == [8, 8, 1]


def test_clip_coverage_cannot_defer_an_identity_to_epoch_tail(monkeypatch):
    index = SampleIndex([
        dict(path=f'/unused/{label}-{i}.png', width=16, height=16, identity_id=label,
             clip_id='one-clip' if label == 'a' else f'clip-{i}', tracklet_id=str(i))
        for label in ('a', 'b') for i in range(8)
    ], sampling=dict(balance_by=('identity_id', 'clip_id', 'tracklet_id'), samples_per_epoch=64))
    draws = list(range(16))*4
    monkeypatch.setattr('dataset_fixer.id.training.WeightedRandomSampler', lambda *a, **k: iter(draws))
    batches = list(_DiverseBatches(index, 8, 7))
    assert Counter(row for batch in batches for row, _, _ in batch) == Counter(draws)
    for batch in batches:
        frame = index.frame.iloc[[row for row, _, _ in batch]]
        assert frame.identity_id.value_counts().to_dict() == {'a': 4, 'b': 4}
        assert frame.clip_id.nunique() == 5  # maximum under those identity shares
        assert len(frame[['clip_id', 'tracklet_id']].drop_duplicates()) == 8
