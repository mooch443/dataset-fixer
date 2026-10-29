"""Enrollment follows supplied dataset splits, retaining observation exclusions."""
import pandas as pd
import pytest

from dataset_fixer import id
from dataset_fixer.id.data import SampleIndex, as_dataset, index_of
from test_identity import identity_data


@pytest.mark.parametrize('selection', ['balanced', 'medoids'])
def test_protocol_accepts_tracklet_split_with_shared_clips(identity_data, selection):
    diagnostic = identity_data.split(
        {'train': .8, 'val': .2}, source_splits=('train',),
        group_by='tracklet_id', classes_must_appear_in=('train', 'val'), seed=7,
    )
    protocol = id.Protocol(diagnostic, gallery_samples=2, gallery_selection=selection)
    training = index_of(protocol.training).where(split='train').frame
    queries = index_of(protocol.queries['val']).frame
    references = index_of(protocol.reference_candidates).frame
    assert set(references.path) <= set(training.path)
    assert set(references.clip_id) == set(queries.clip_id) == {'train-clip'}
    assert set(references.identity_id) == set(queries.identity_id) == {'wolf-a', 'wolf-b'}
    assert set(references.tracklet_id).isdisjoint(queries.tracklet_id)
    assert set(references.path).isdisjoint(queries.path)
    assert len(index_of(protocol.queries['test'])) == 0
    # The source's independent val/test partitions are untouched.
    assert index_of(identity_data).frame.split.value_counts().to_dict() == {'train': 10, 'val': 10, 'test': 10}


def test_shared_clips_do_not_allow_reused_observations(identity_data, tmp_path):
    frame = index_of(identity_data).frame.astype(object)
    copied = frame.iloc[0].copy()
    copied['split'] = 'val'
    data = as_dataset(SampleIndex(pd.concat([frame, pd.DataFrame([copied])])), location=tmp_path)
    with pytest.raises(ValueError, match='share source observations'):
        id.Protocol(data)
