from pathlib import Path
import json
import shutil

import albumentations as A
import cv2
import numpy as np
import pytest
from PIL import Image

from dataset_fixer import Dataset, synthetic, id
from dataset_fixer.id.data import ImageRef, SampleIndex, as_dataset, index_of
from dataset_fixer.segmentation import rasterize_annotations
from dataset_fixer.synthetic._simulation import GENES, LIMB_END, SLOTS, SceneSettings, Trajectory, population
from dataset_fixer.synthetic._rendering import SceneEffects, individual, visibility
from dataset_fixer.augmentation import serialize_pipeline


def generate(tmp_path, name="scenes", **kwargs):
    options = dict(individuals=3, clips=1, frames_per_clip=4, size=(256, 192), augmentations=[])
    options.update(kwargs)
    return synthetic.generate(destination=tmp_path / name, **options)


def pixels(sample):
    return np.asarray(Image.open(sample.image_path))


def test_reproducible_prefix_and_independent_population(tmp_path):
    short = generate(tmp_path, "short", augmentations=None)
    full = generate(tmp_path, "full", clips=2, frames_per_clip=7, augmentations=None)
    assert short.manifest["synthetic"]["individuals"] == full.manifest["synthetic"]["individuals"]
    for a, b in zip(short._samples, full._samples):
        np.testing.assert_array_equal(pixels(a), pixels(b))
        assert a.provenance["synthetic"] == b.provenance["synthetic"]
        assert a.clip_id == b.clip_id
    first = synthetic.identity(short, destination=tmp_path / "short-id")
    second = synthetic.identity(full, destination=tmp_path / "full-id")
    other = {(s.clip_id, s.frame_index, s.identity_id): s for s in second._samples}
    for sample in first._samples:
        np.testing.assert_array_equal(sample.image_ref.read(), other[sample.clip_id, sample.frame_index, sample.identity_id].image_ref.read())
    again = generate(tmp_path, "again", individuals=short.manifest["synthetic"]["individuals"], seed=99)
    assert again.manifest["synthetic"]["individuals"] == short.manifest["synthetic"]["individuals"]
    assert not np.array_equal(pixels(again._samples[0]), pixels(short._samples[0]))


def test_dna_similarity_and_fixed_topology():
    identical = list(population(4, 0, 7).values())
    np.testing.assert_array_equal(identical[0], identical[-1])
    nearby = np.asarray(list(population(4, .02, 7).values()))
    distant = np.asarray(list(population(4, .8, 7).values()))
    assert np.linalg.norm(nearby[0] - nearby[1]) < np.linalg.norm(distant[0] - distant[1])
    state = Trajectory(SceneSettings((320, 240), .2, 7), "clip", "animal")
    state.position = np.array([160., 120.])
    genes = np.asarray(identical[0])
    before = genes.copy()
    image, mask, _, _ = individual(genes, state, (320, 240))
    changed = genes.copy()
    changed[3] += .01
    next_image, next_mask, _, _ = individual(changed, state, (320, 240))
    np.testing.assert_array_equal(mask, next_mask)
    assert np.abs(image.astype(float) - next_image).mean() < 1
    absent = genes.copy()
    absent[14:LIMB_END:6] = 0
    emerging = absent.copy()
    emerging[14] = .401
    # A tiny chromosome change at a topology threshold cannot create a full leg.
    absent[14] = .399
    a = individual(absent, state, (320, 240))[1]
    b = individual(emerging, state, (320, 240))[1]
    assert np.count_nonzero(a != b) < 5
    for _ in range(30):
        state.advance()
        individual(genes, state, (320, 240))
    np.testing.assert_array_equal(genes, before)
    assert tuple(GENES[14:LIMB_END:6]) == tuple(f"limb_{i}_presence" for i in range(SLOTS))


def test_motion_is_continuous():
    state = Trajectory(SceneSettings((960, 640), .03, 42), "clip", "animal")
    positions, headings = [], []
    for _ in range(120):
        positions.append(state.position.copy())
        headings.append(state.heading)
        state.advance()
    velocities = np.diff(positions, axis=0)
    alignment = (velocities[1:] * velocities[:-1]).sum(axis=1) / np.linalg.norm(velocities[1:], axis=1) / np.linalg.norm(velocities[:-1], axis=1)
    assert np.median(alignment) > .99
    assert np.max(np.abs(np.diff(headings))) < .2


def test_overlap_discards_both_crops_and_restarts_tracklets(tmp_path, monkeypatch):
    class Crossing(Trajectory):
        def __init__(self, settings, clip, label):
            super().__init__(settings, clip, label)
            self.number = int(label[-4:])
            self.heading = 0.
            self.position = self.crossing_position()

        def crossing_position(self):
            return np.array(([80, 80] if self.number == 0 else [160, 80])
                            if self.frame != 1 and self.number < 2 else
                            ([120, 80] if self.number < 2 else [220, 150]), dtype=float)

        def advance(self):
            self.frame += 1
            self.position = self.crossing_position()

    monkeypatch.setattr(synthetic, "Trajectory", Crossing)
    scenes = generate(tmp_path, frames_per_clip=3)
    crops = synthetic.identity(scenes, destination=tmp_path / "id")
    rows = index_of(crops).frame
    assert len(rows) == 7
    for label in ("individual-0000", "individual-0001"):
        subset = rows[rows.identity_id == label]
        assert list(subset.frame_index) == [0, 2]
        assert subset.tracklet_id.nunique() == 2
    assert rows[rows.identity_id == "individual-0002"].tracklet_id.nunique() == 1
    observations = scenes._samples[1].provenance["synthetic"]["observations"]
    assert [o["overlap"] for o in observations] == [True, True, False]
    assert len(scenes._samples) == 3
    assert scenes._samples[1].annotations  # Scene annotations survive.


def test_mask_overlap_not_box_overlap_and_border_loss():
    masks = np.zeros((3, 20, 20), bool)
    masks[0, 2:10, 2] = True
    masks[0, 2, 2:10] = True
    masks[1, 5:13, 7] = True
    masks[1, 12, 5:13] = True
    masks[2, :4, 15:18] = True
    _, observations = visibility(masks, [False] * 3)
    assert not observations[0]["overlap"] and not observations[1]["overlap"]
    assert observations[2]["truncated"] and not observations[2]["usable"]


def test_posture_and_detection_jitter(tmp_path):
    scenes = generate(tmp_path, frames_per_clip=12)
    crops = synthetic.identity(scenes, destination=tmp_path / "posture", orientation_jitter=0)
    unaligned = synthetic.identity(scenes, destination=tmp_path / "none", normalization="none", bbox_jitter=0)
    for sample in crops._samples:
        meta = sample.provenance["synthetic"]
        points = np.column_stack((meta["head_tail"], np.ones(2))) @ np.asarray(meta["crop_transform"]).T
        direction = points[0] - points[1]
        assert direction[0] > 0 and direction[1] > 0
        np.testing.assert_allclose(direction[0], direction[1], atol=1e-4)
        true = np.asarray(meta["true_bbox"])
        error = np.abs(np.asarray(meta["detected_bbox"]) - true)
        assert np.all(error <= .05 * np.tile(true[2:] - true[:2], 2) + 1e-9)
        assert sample.image_ref.read().shape == (128, 128, 3)
    for sample in unaligned._samples:
        meta = sample.provenance["synthetic"]
        matrix = np.asarray(meta["crop_transform"])
        assert matrix[0, 1] == matrix[1, 0] == 0
        assert meta["detected_bbox"] == meta["true_bbox"]


def test_scene_effect_replay_transforms_geometry():
    image = np.zeros((120, 160, 3), np.uint8)
    image[30:70, 40:80] = 200
    mask = image[..., 0] > 0
    pipeline = serialize_pipeline([A.HorizontalFlip(p=1)], {})
    effects = SceneEffects(pipeline, 42, "clip")
    args = (image, [mask], [np.array([[50., 50.], [70., 50.]])])
    result, masks, points = effects.apply(*args)
    other, other_masks, other_points = effects.apply(*args)
    np.testing.assert_array_equal(result, image[:, ::-1])
    np.testing.assert_array_equal(masks[0], mask[:, ::-1])
    np.testing.assert_allclose(points[0], [[109, 50], [89, 50]])
    np.testing.assert_array_equal(result, other)
    np.testing.assert_array_equal(masks, other_masks)
    np.testing.assert_array_equal(points, other_points)
    lens = SceneEffects(serialize_pipeline([A.OpticalDistortion(distort_limit=.03, p=1)], {}), 42, "clip")
    result, masks, points = lens.apply(*args)
    assert masks[0, round(points[0, 0, 1]), round(points[0, 0, 0])]
    # Image interpolation is linear; label interpolation is nearest-neighbor.
    mask = masks[0].astype(np.uint8)
    boundary = cv2.dilate(mask, np.ones((3, 3), np.uint8)) != cv2.erode(mask, np.ones((3, 3), np.uint8))
    assert not np.any(((result[..., 0] > 100) != masks[0]) & ~boundary)


def test_portable_identity_move_and_metadata(tmp_path):
    scenes = generate(tmp_path)
    crops = synthetic.identity(scenes, destination=tmp_path / "id")
    expected = [s.image_ref.read() for s in crops._samples]
    exported = crops.export(destination=tmp_path / "portable", format="identity_manifest", copy_images=True)
    records = [json.loads(line) for line in (exported.location / "samples.jsonl").read_text().splitlines()]
    assert all(not Path(row["path"]).is_absolute() and row["crop"] is None for row in records)
    shutil.rmtree(scenes.location)
    shutil.rmtree(crops.location)
    shutil.move(exported.location, tmp_path / "moved")
    loaded = Dataset.open(tmp_path / "moved/samples.jsonl", format="identity_manifest")
    assert loaded.manifest["synthetic"]["gene_names"] == list(GENES)
    for sample, reference in zip(loaded._samples, expected):
        np.testing.assert_array_equal(sample.image_ref.read(), reference)
        assert sample.provenance["synthetic"]["crop_transform"]
    assert loaded.with_sampling(samples_per_epoch=20)._samples.fingerprint


def test_portable_export_materializes_lazy_crops(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    image = np.arange(30 * 40 * 3, dtype=np.uint8).reshape(30, 40, 3)
    Image.fromarray(image).save(source / "frame.png")
    data = as_dataset(SampleIndex([dict(path=str(source / "frame.png"), width=20, height=10,
                                        crop=(10, 5, 30, 15), identity_id="a", clip_id="c", tracklet_id="t")]), location=source)
    exported = data.export(destination=tmp_path / "copy", format="identity_manifest", copy_images=True)
    shutil.rmtree(source)
    np.testing.assert_array_equal(exported._samples[0].image_ref.read(), image[5:15, 10:30])
    assert exported._samples[0].provenance["tracklet_id"] == "t"


def test_portable_export_materializes_npz_rows(tmp_path):
    pytest.importorskip("mmnpz")
    source = tmp_path / "source"
    source.mkdir()
    images = np.arange(2 * 20 * 30 * 3, dtype=np.uint8).reshape(2, 20, 30, 3)
    padded = np.pad(images, ((0, 0), (2, 2), (3, 3), (0, 0)))
    np.savez(source / "frames.npz", images=padded)
    data = as_dataset(SampleIndex([dict(path=str(source / "frames.npz"), array="images", row=1,
                                        width=36, height=24, identity_id="a", trim_black_borders=True)]), location=source)
    copied = data.export(destination=tmp_path / "copy", format="identity_manifest", copy_images=True)
    shutil.rmtree(source)
    assert copied._samples[0].image_ref.array is None
    assert not copied._samples[0].image_ref.trim_black_borders
    np.testing.assert_array_equal(copied._samples[0].image_ref.read(), images[1])


@pytest.mark.parametrize("task", ["detect", "pose", "polo"])
def test_scene_tasks_and_yolo_roundtrip(tmp_path, task):
    scenes = generate(tmp_path, task=task)
    exported = scenes.export(destination=tmp_path / "yolo", visualize=False, progress=False)
    loaded = Dataset.open(exported.location, progress=False)
    assert loaded.task.value == task
    assert loaded.manifest["synthetic"]["individuals"] == scenes.manifest["synthetic"]["individuals"]
    assert loaded._samples[0].clip_id == "clip-0000"
    assert [a.identity_id for a in loaded._samples[0].annotations] == [a.identity_id for a in scenes._samples[0].annotations]
    original = synthetic.identity(scenes, destination=tmp_path / "original-id")
    derived = synthetic.identity(loaded, destination=tmp_path / "exported-id")
    for a, b in zip(original._samples, derived._samples):
        np.testing.assert_array_equal(a.image_ref.read(), b.image_ref.read())


def test_segmentation_exports_preserve_exact_masks(tmp_path):
    scenes = generate(tmp_path)
    expected = [rasterize_annotations(s.annotations, width=s.width, height=s.height) for s in scenes._samples]
    exported = scenes.export(destination=tmp_path / "masks", format="semantic_masks", visualize=False, progress=False)
    loaded = Dataset.open(exported.location, progress=False)
    for sample, mask in zip(loaded._samples, expected):
        np.testing.assert_array_equal(np.asarray(Image.open(loaded._mask_paths[sample.image_path])) > 0, mask > 0)
    assert loaded.manifest["synthetic"] == scenes.manifest["synthetic"]
    assert loaded.provenance  # Semantic repositories deliberately defer lineage loading.
    assert loaded._samples[0].clip_id == "clip-0000"
    scenes.export(destination=tmp_path / "polygons", allow_lossy=True, visualize=False, progress=False)


def test_closed_open_protocols_and_split_isolation(tmp_path):
    scenes = generate(tmp_path, individuals=8, clips=6, frames_per_clip=20, size=(480, 320))
    data = synthetic.identity(scenes, destination=tmp_path / "id")
    split = data.split({"train": .5, "val": .25, "test": .25}, group_by="clip_id")
    assert index_of(split).frame.groupby("clip_id", observed=True).split.nunique().max() == 1
    closed = id.Protocol(split, gallery_samples=2)
    assert len(closed.training._samples) > 0
    label = index_of(closed.training).frame.identity_id.dropna().iloc[0]
    limited = split.limit_coverage({label: {"clips": 1, "tracklets_per_clip": 1}})
    constrained = id.Protocol(limited, gallery_samples=2)
    sparse = index_of(constrained.training).frame
    sparse = sparse.loc[(sparse.identity_id == label) & (sparse.split == "train")]
    assert sparse.clip_id.nunique() == sparse.tracklet_id.nunique() == 1
    for held_out in ("val", "test"):
        assert index_of(constrained.queries[held_out]).frame.equals(index_of(closed.queries[held_out]).frame)
    # Lost observations are not replaced: choose roles from actual clip coverage.
    frame = index_of(split).frame
    coverage = [set(frame.loc[frame.split == s, "identity_id"]) for s in ("train", "val", "test")]
    labels = sorted(set.intersection(*coverage))
    assert len(labels) >= 6
    unknown = {"val": [labels[-1]], "test": [labels[-2]]}
    novel = {"val": [labels[-3]], "test": [labels[-4]]}
    protocol = id.Protocol.open_set(split, gallery_samples=2, unknown_identities=unknown, novel_identities=novel)
    assert set(index_of(protocol.training).frame.identity_id).isdisjoint(labels[-4:])
    assert set(index_of(protocol.references).frame.identity_id).isdisjoint(labels[-2:])


def test_atomic_errors_and_validation(tmp_path):
    with pytest.raises(ValueError, match="DNA"):
        generate(tmp_path, individuals={"invalid": [0, 1]})
    with pytest.raises(ValueError, match="difference"):
        generate(tmp_path, individuals=population(1, 0, 7), difference=.3)
    with pytest.raises(ValueError, match="positive integer"):
        generate(tmp_path, clips=0)
    with pytest.raises(ValueError, match="HWC RGB"):
        generate(tmp_path, augmentations=[A.Normalize(p=1)])
    assert not (tmp_path / "scenes").exists()
    assert not list(tmp_path.glob(".scenes.tmp-*"))
    scenes = generate(tmp_path)
    with pytest.raises(FileExistsError):
        generate(tmp_path)
    with pytest.raises(ValueError, match="normalization"):
        synthetic.identity(scenes, normalization="bad")


def test_cache_and_no_global_rng_side_effects(tmp_path, monkeypatch):
    monkeypatch.setattr(synthetic, "cache_root", lambda: tmp_path)
    np.random.seed(71)
    before = np.random.get_state()
    options = dict(individuals=2, clips=1, frames_per_clip=2, size=(128, 96))
    a = synthetic.generate(**options)
    b = synthetic.generate(**options)
    assert a.location == b.location
    c = synthetic.identity(a)
    assert synthetic.identity(b).location == c.location
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


def test_shared_renderer_outlines_disconnected_masks(tmp_path):
    from dataset_fixer.models import Sample, Annotation, DatasetMetadata, Task
    from dataset_fixer.synthetic._rendering import segmentation
    from dataset_fixer.visualization import render_annotated_sample

    mask = np.zeros((100, 100), np.uint8)
    mask[10:40, 10:40] = 1
    mask[60:90, 60:90] = 1
    encoded = segmentation(mask)
    image = Image.fromarray(np.zeros((100, 100, 3), np.uint8))
    sample = Sample(tmp_path / "frame.png", Path("frame.png"), "train", 100, 100,
                    [Annotation(0, rle=encoded)])
    rendered = np.asarray(render_annotated_sample(sample, Task.SEGMENT, DatasetMetadata({0: "individual"}), image=image))
    assert rendered[10, 20].any() and rendered[60, 70].any()
    assert not rendered[50, 50].any()
