"""Appearance locality, DNA compatibility, scene scaling, and pose targets."""
import json

import albumentations as A
import numpy as np
import pytest

from dataset_fixer import synthetic
from dataset_fixer.synthetic._appearance import body_colors, neck_weight
from dataset_fixer.synthetic._rendering import individual
from dataset_fixer.synthetic._simulation import GENES, GENE_INDEX, LIMB_END, PATTERNS, VERSION, SceneSettings, Trajectory, population


def render(dna, scale=1):
    state = Trajectory(SceneSettings((640, 480), 0, 42, scale), "clip", "same-pose")
    state.position = np.array([320., 240.])
    state.heading = 0.
    return individual(dna, state, (640, 480)), state


def test_motifs_change_pixels_without_changing_anatomy():
    results = []
    for pattern in PATTERNS:
        dna = next(iter(population(1, .25, 42, patterns=(pattern,)).values()))
        for other in set(PATTERNS) - {pattern}:
            assert dna[GENE_INDEX["mark_" + other]] == dna[GENE_INDEX["neck_mark_" + other]] == 0
        (pixels, mask, points, _), _ = render(dna, 2)
        results.append(pixels)
        if pattern == PATTERNS[0]:
            reference_mask, reference_points = mask, points
        else:
            np.testing.assert_array_equal(mask, reference_mask)
            np.testing.assert_array_equal(points, reference_points)
    assert all(np.count_nonzero(results[i] != results[j]) > 100 for i, j in ((0, 1), (0, 2), (1, 2)))


def test_neck_focus_restricts_identity_differences_to_body_attached_band():
    dna = list(population(4, .25, 42, neck_focus=1).values())
    first, state = render(dna[0], 2)
    yy, xx = np.mgrid[:480, :640]
    local = state.pose(dna[0]).material(np.column_stack((xx.ravel(), yy.ravel())))
    neck = neck_weight(local, state._rig.length).reshape(480, 640) > 0
    for genes in dna[1:]:
        other, _ = render(genes, 2)
        np.testing.assert_array_equal(first[1], other[1])
        changed = np.any(first[0] != other[0], axis=2)
        assert changed.sum() > 50
        assert not np.any(changed & ~neck)
    identical = list(population(2, 0, 42, neck_focus=1).values())
    np.testing.assert_array_equal(identical[0], identical[1])


def test_material_pattern_is_stable_under_body_deformation_and_small_dna_edits():
    genes = np.asarray(next(iter(population(1, .25, 42, neck_focus=.8).values())))
    _, state = render(genes)
    state.settings = SceneSettings((640, 480), .15, 42)
    points = np.array([[-20., 3.], [10., -3.], [15., 3.]])
    initial = body_colors(genes, points, state._rig.scale, state._rig.length)
    for _ in range(10):
        state.advance()
        pose = state.pose(genes)
        recovered = pose.material(pose.world(points))
        np.testing.assert_allclose(recovered, points, atol=1e-10)
        np.testing.assert_allclose(body_colors(genes, recovered, state._rig.scale, state._rig.length), initial, atol=1)
    changed = genes.copy()
    changed[GENE_INDEX["mark_dots"]] += .001
    yy, xx = np.mgrid[-20:21, -40:41]
    points = np.column_stack((xx.ravel(), yy.ravel()))
    a = body_colors(genes, points, 32, 35)
    b = body_colors(changed, points, 32, 35)
    assert np.abs(a.astype(float) - b).mean() < 1


def test_neck_motifs_remain_visible_for_random_offsets():
    yy, xx = np.mgrid[-15:16, -40:41]
    points = np.column_stack((xx.ravel(), yy.ravel()))
    band = neck_weight(points, 40) > 0
    for pattern in PATTERNS:
        for dna in population(10, .25, 42, neck_focus=1, patterns=(pattern,), texture_noise={}).values():
            genes = np.asarray(dna)
            genes[3:6] = genes[GENE_INDEX["neck_red"]:GENE_INDEX["neck_blue"] + 1] = .2
            genes[13] = 0
            genes[GENE_INDEX["neck_mark_red"]:GENE_INDEX["neck_mark_blue"] + 1] = 1
            genes[GENE_INDEX["neck_mark_strength"]] = 1
            colors = body_colors(genes, points, 35, 40)
            marked = np.any(colors > 80, axis=1)
            assert marked.sum() > 20
            assert not np.any(marked & ~band)


def test_individual_scale_doubles_rig_and_silhouette():
    genes = next(iter(population(1, .25, 42).values()))
    small, a = render(genes)
    large, b = render(genes, 2)
    np.testing.assert_allclose(b.pose(genes).joints - b.position, 2 * (a.pose(genes).joints - a.position))
    np.testing.assert_allclose(large[2] - b.position, 2 * (small[2] - a.position))
    assert large[1].sum() / small[1].sum() == pytest.approx(4, rel=.1)


def test_legacy_dna_remains_unchanged_and_uses_legacy_stripes(tmp_path):
    genes = np.asarray(next(iter(population(1, .25, 42).values())))[:LIMB_END]
    dna = {"legacy": genes.tolist()}
    scenes = synthetic.generate(individuals=dna, clips=1, frames_per_clip=1, size=(240, 160),
                                augmentations=[], weather={}, destination=tmp_path / "legacy")
    assert scenes.manifest["synthetic"]["individuals"] == dna
    assert scenes.manifest["synthetic"]["version"] == 1
    assert scenes.manifest["synthetic"]["gene_names"] == list(GENES[:LIMB_END])
    local = np.column_stack((np.arange(-60, 61), np.zeros(121)))
    colors = body_colors(genes, local, 30, 35)
    base, mark = 35 + genes[3:6] * 200, 35 + genes[9:12] * 200
    expected = np.tile(base, (len(local), 1))
    stripe = np.cos(local[:, 0] / 30 * (3 + 12 * genes[12])) > .55
    expected[stripe] = base * (1 - genes[13] * .8) + mark * genes[13] * .8
    np.testing.assert_array_equal(colors, expected.astype(np.uint8))


def test_pose_exports_head_tail_with_scaled_and_augmented_coordinates(tmp_path):
    scenes = synthetic.generate(individuals=1, clips=1, frames_per_clip=2, task="pose", size=(320, 240),
                                individual_scale=1.5, patterns=("stars",), neck_focus=.8,
                                augmentations=[A.HorizontalFlip(p=1)], destination=tmp_path / "pose")
    assert scenes.manifest["synthetic"]["version"] == VERSION
    assert scenes.manifest["synthetic"]["individual_scale"] == 1.5
    document = json.loads((scenes.location / "annotations.json").read_text())
    assert document["categories"][0]["keypoints"] == ["head", "tail"]
    assert document["categories"][0]["skeleton"] == [[1, 2]]
    for sample in scenes._samples:
        observation = sample.provenance["synthetic"]["observations"][0]
        points = np.asarray(sample.annotations[0].keypoints)
        assert points.shape == (2, 3) and set(points[:, 2]) == {2}
        np.testing.assert_allclose(points[:, :2], observation["head_tail"])
    reused = synthetic.generate(individuals=scenes.manifest["synthetic"]["individuals"], clips=1,
                                frames_per_clip=1, size=(320, 240), seed=123, destination=tmp_path / "reused")
    assert reused.manifest["synthetic"]["individuals"] == scenes.manifest["synthetic"]["individuals"]


@pytest.mark.parametrize("options", [dict(patterns=[]), dict(patterns=["triangles"]), dict(neck_focus=1.1),
                                    dict(individual_scale=0), dict(individual_scale=float("inf")), dict(individual_scale=True)])
def test_appearance_validation_before_writes(tmp_path, options):
    with pytest.raises(ValueError):
        synthetic.generate(destination=tmp_path / "invalid", **options)
    assert not (tmp_path / "invalid").exists()


def test_explicit_dna_does_not_silently_apply_population_controls():
    dna = population(1, .25, 42)
    for options in (dict(patterns=["dots"]), dict(neck_focus=.5)):
        with pytest.raises(ValueError, match="explicit DNA"):
            population(dna, None, 42, **options)
