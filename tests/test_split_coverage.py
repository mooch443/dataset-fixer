from itertools import product
import pytest

from dataset_fixer import Dataset
from dataset_fixer.errors import DatasetValidationError
from dataset_fixer.id.data import ImageRef, SampleIndex, as_dataset, index_of
from dataset_fixer.planning import allocate_groups


def identity_clips(tmp_path, clips):
    return as_dataset(SampleIndex([
        dict(path=str(tmp_path / clip / f'{row}.png'), width=16, height=16,
             identity_id=label, clip_id=clip, split='train')
        for clip, labels in clips.items() for row, label in enumerate(labels)
    ]), location=tmp_path)


def test_identity_coverage_preserves_groups_and_is_lazy(tmp_path, monkeypatch):
    data = identity_clips(tmp_path, {'only-a': ['a'], 'b1': ['b'], 'b2': ['b'], 'b3': ['b', None]})
    monkeypatch.setattr(ImageRef, 'read', lambda self: pytest.fail('Splitting must not decode pixels'))
    kwargs = dict(group_by='clip_id', classes_must_appear_in=('train',), seed=7)
    ratios = {'train': .5, 'val': .25, 'test': .25}
    result = data.split(ratios, **kwargs)
    frame = index_of(result).frame
    assert set(frame.loc[frame.split == 'train', 'identity_id'].dropna()) == {'a', 'b'}
    assert frame.groupby('clip_id', observed=True).split.nunique().max() == 1
    assert len(frame) == len(index_of(data))
    assert set(index_of(data).frame.split) == {'train'}
    assert any("lacks coverage" in warning and 'a' in warning for warning in result.warnings)
    assert result.manifest['split_policy']['classes_must_appear_in'] == ['train']
    assert frame.equals(index_of(data.split(ratios, **kwargs)).frame)
    assert index_of(data.split(ratios, group_by='clip_id')).frame.equals(
        index_of(data.split(ratios, group_by='clip_id', classes_must_appear_in=())).frame)


def test_identity_all_required_splits_and_source_selection(tmp_path):
    data = identity_clips(tmp_path, {f'clip-{i}': ['a', 'b', None] for i in range(3)})
    data = data.holdout(clip_id='clip-0', to_split='test')
    result = data.split({'train': 1, 'val': 1}, source_splits=('train',), group_by='clip_id',
                        classes_must_appear_in=('train', 'validation', 'train'))
    frame = index_of(result).frame
    assert set(frame.clip_id) == {'clip-1', 'clip-2'}
    for split in ('train', 'val'):
        assert set(frame.loc[frame.split == split, 'identity_id'].dropna()) == {'a', 'b'}
    assert result.manifest['split_policy']['classes_must_appear_in'] == ['train', 'val']
    labeled = data.filter(gt_annotated=True)
    result = labeled.split({'train': 1, 'val': 1, 'test': 1},
                           group_by=('identity_id', 'clip_id'),
                           classes_must_appear_in=('train', 'val', 'test'))
    assert index_of(result).frame.groupby('identity_id', observed=True).split.nunique().eq(3).all()


def test_coverage_rejects_impossible_clips_and_honors_assignments(tmp_path):
    data = identity_clips(tmp_path, {'only-a': ['a'], 'b1': ['b'], 'b2': ['b']})
    with pytest.raises(DatasetValidationError, match="class 'a'.*distinct unassigned groups"):
        data.split({'train': 1, 'val': 1}, group_by='clip_id', classes_must_appear_in=('train', 'val'))
    with pytest.raises(DatasetValidationError, match="class 'a'"):
        data.split({'train': 1, 'val': 1}, group_by='clip_id', classes_must_appear_in=('train',),
                   assign=lambda p: 'val' if p.parent.name == 'only-a' else None)
    result = data.split({'train': 1, 'val': 1}, group_by='clip_id', classes_must_appear_in=('val',),
                        assign=lambda p: 'val' if p.parent.name == 'only-a' else None)
    assert set(index_of(result).frame.loc[lambda f: f.split == 'val', 'identity_id']) == {'a', 'b'}
    # Each class has two clips, but their overlapping memberships form an odd
    # cycle: no partition can put every class on both sides.
    cycle = identity_clips(tmp_path, {'ab': ['a', 'b'], 'bc': ['b', 'c'], 'ca': ['c', 'a']})
    with pytest.raises(DatasetValidationError, match='Class overlaps'):
        cycle.split({'train': 1, 'val': 1}, group_by='clip_id', classes_must_appear_in=('train', 'val'))


@pytest.mark.parametrize('required,ratios,error', [
    ('train', {'train': 1}, TypeError),
    (('test',), {'train': 1, 'val': 1}, ValueError),
    (('val',), {'train': 1, 'val': 0}, ValueError),
    (('typo',), {'train': 1}, ValueError),
])
def test_coverage_validates_required_splits(tmp_path, required, ratios, error):
    data = identity_clips(tmp_path, {'one': ['a'], 'two': ['a']})
    with pytest.raises(error):
        data.split(ratios, classes_must_appear_in=required)


def test_detection_coverage_survives_export(detect_dataset, tmp_path):
    data = Dataset.open(detect_dataset, task='detect', progress=False)
    planned = data.split({'train': 1, 'val': 1, 'test': 1},
                         group_by=lambda path: path.parent.name,
                         classes_must_appear_in=('train', 'val', 'test'), visualize=False)
    exported = planned.export(destination=tmp_path/'covered', visualize=False, progress=False)
    for result in (planned, exported):
        for split in ('train', 'val', 'test'):
            assert {a.class_id for s in result._samples if s.split == split for a in s.annotations} == {0, 1}
    assert exported.manifest['history'][0]['settings']['classes_must_appear_in'] == ['train', 'val', 'test']


def test_solver_feasibility_matches_exhaustive_small_group_partitions():
    # Compare the library constraint adapter with every possible partition,
    # including shared classes, repeated class-sets, and explicit locks.
    memberships = ({'a'}, {'b'}, {'a', 'b'})
    for labels in product(memberships, repeat=3):
        classes = set().union(*labels)
        for locked in ({}, {0: 'val'}):
            possible = any(
                all(assignment[group] == split for group, split in locked.items()) and
                all(any(assignment[g] == split and label in labels[g] for g in range(3))
                    for label in classes for split in ('train', 'val'))
                for assignment in product(('train', 'val'), repeat=3)
            )
            try:
                actual = allocate_groups({g: 1 for g in range(3)}, {'train': .5, 'val': .5}, locked, 7,
                                         group_classes=dict(enumerate(labels)), classes_must_appear_in=('train', 'val'))
            except DatasetValidationError:
                assert not possible
            else:
                assert possible
                assert all(actual[g] == split for g, split in locked.items())
                assert all(any(actual[g] == split and label in labels[g] for g in range(3))
                           for label in classes for split in ('train', 'val'))
