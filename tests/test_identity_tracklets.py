"""A tracklet has one identity from import through every derived dataset."""
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from dataset_fixer import Dataset, id
from dataset_fixer.id.data import SampleIndex, index_of
from dataset_fixer.utils import qualified_tracklet_id


def test_tracklet_imports_and_derived_views_preserve_source_ids(tmp_path):
    records = []
    for clip in ('clip-a', 'clip-b'):
        for track in ('001', '002'):
            for frame in range(2):
                path = tmp_path / 'wolf' / clip / track / f'{frame}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (8, 8), (100, 120, 140)).save(path)
                records.append(dict(path=str(path), identity_id='wolf', clip_id=clip, tracklet_id=track))
    folders = Dataset.open(tmp_path, format='identity_folders', layout='{identity_id}/{clip_id}/{tracklet_id}/{image}')
    original = index_of(folders).frame
    assert original.tracklet_id.nunique() == 4
    expected = dict(zip(original.path, original.tracklet_id))
    for suffix in ('csv', 'jsonl'):
        path = tmp_path / f'samples.{suffix}'
        if suffix == 'csv':
            pd.DataFrame(records).to_csv(path, index=False)
        else:
            path.write_text('\n'.join(json.dumps(r) for r in records))
        imported = Dataset.open(path, format='identity_manifest')
        assert dict(zip(index_of(imported).frame.path, index_of(imported).frame.tracklet_id)) == expected

    split = folders.filter(identity_ids=['wolf']).split(
        {'train': .5, 'val': .25, 'test': .25}, group_by='tracklet_id',
        classes_must_appear_in=('train', 'val', 'test'), seed=7,
    )
    sampled = split.with_sampling(balance_by=('identity_id', 'tracklet_id'), samples_per_epoch=16)
    derived = id.from_dataset(sampled, identity='identity_id', context_fraction=.1)
    exported = derived.export(destination=tmp_path / 'derived', format='identity_manifest')
    for view in (split, sampled, derived, exported,
                 Dataset.open(exported.location / 'samples.jsonl', format='identity_manifest'),
                 split.holdout(clip_id='clip-a', to_split='test')):
        frame = index_of(view).frame
        assert frame.tracklet_id.tolist() == [expected[p] for p in frame.path]
    assert index_of(split).frame.groupby('tracklet_id', observed=True).split.nunique().max() == 1
    np.testing.assert_allclose(index_of(sampled).weights(), original.groupby(
        ['identity_id', 'tracklet_id'], observed=True).identity_id.transform('size').rdiv(1 / 4))
    # IDs remain source identities, even if a derived view relabels its clip or animal.
    relabeled = SampleIndex(original.assign(clip_id='derived', identity_id='renamed'))
    assert relabeled.frame.tracklet_id.tolist() == original.tracklet_id.tolist()
    assert index_of(folders).take([7, 0, 2]).frame.tracklet_id.tolist() == original.tracklet_id.iloc[[7, 0, 2]].tolist()


def test_npz_tracklets_span_parts_but_not_clips(tmp_path):
    pytest.importorskip('mmnpz')
    for clip in ('a', 'b'):
        folder = tmp_path / clip
        folder.mkdir()
        for part in range(2):
            np.savez(folder / f'part-{part}.npz', images=np.ones((2, 8, 8, 3), dtype=np.uint8),
                     ids=np.array([3, 3]), frames=np.array([part * 2, part * 2 + 1]),
                     frame_segment_indexes=np.array([100, 100]), tracklets=np.array([[0, 3]]))
    data = Dataset.open(tmp_path, format='trex_npz')
    frame = index_of(data).frame
    assert frame.tracklet_id.nunique() == 2
    assert frame.groupby('clip_id', observed=True).tracklet_id.nunique().tolist() == [1, 1]
    assert id.stats(data, by=()).tracklets.tolist() == [2]
    assert frame.local_id.tolist() == [3] * 8
    assert frame.frame_segment_index.tolist() == [100] * 8


def test_tracklet_qualification_handles_missing_metadata_and_separators():
    pairs = [('a:b', 'c'), ('a', 'b:c'), ('a"/b', '001'), (None, '1')]
    ids = [qualified_tracklet_id(*pair) for pair in pairs]
    assert len(set(ids)) == len(pairs)
    assert [qualified_tracklet_id('changed-clip', value) for value in ids] == ids
    assert qualified_tracklet_id(None, pd.NA) is None
    frame = SampleIndex([dict(path=f'{i}.png', width=8, height=8, clip_id=clip, tracklet_id=track)
                         for i, (clip, track) in enumerate([*pairs, ('a', None)])]).frame
    assert frame.tracklet_id.iloc[:-1].tolist() == ids
    assert pd.isna(frame.tracklet_id.iloc[-1])


def test_gallery_retains_qualified_ids_on_reload(tmp_path):
    source = qualified_tracklet_id('clip', '001')
    gallery = id.Gallery(np.eye(2), ('wolf-a', 'wolf-b'),
                        ({'clip_id': 'derived', 'tracklet_id': source},
                         {'clip_id': 'other', 'tracklet_id': '001'}), 'model')
    assert gallery.references[0]['tracklet_id'] == source
    assert gallery.references[1]['tracklet_id'] == qualified_tracklet_id('other', '001')
    restored = id.Gallery.load(gallery.save(tmp_path / 'gallery.npz'))
    assert restored.references == gallery.references
    assert restored.fingerprint == gallery.fingerprint
