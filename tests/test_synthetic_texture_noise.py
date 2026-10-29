"""Texture noise must be hereditary, continuous, independent, and body attached."""
import numpy as np
import pytest

from dataset_fixer import synthetic
from dataset_fixer.synthetic._appearance import _marking, _texture_noise, body_colors
from dataset_fixer.synthetic._simulation import (GENES, GENE_INDEX, NOISE_COMPONENTS, NOISE_PREFIXES,
                                                 PATTERN_END, SceneSettings, Trajectory, population)


def points():
    y, x = np.mgrid[-18:19, -40:41]
    return np.column_stack((x.ravel(), y.ravel()))


def test_pattern_and_fur_noise_are_independent_genetic_layers():
    genes = np.asarray(next(iter(population(1, .25, 42).values())))
    local = points()
    coverage = _marking(genes, local, 32, 36)
    coat = _texture_noise(genes, local, 32, "fur")
    altered = genes.copy()
    altered[GENE_INDEX["pattern_noise_coefficient_0"]] += .1
    assert np.any(_marking(altered, local, 32, 36) != coverage)
    np.testing.assert_array_equal(_texture_noise(altered, local, 32, "fur"), coat)
    altered = genes.copy()
    altered[GENE_INDEX["fur_noise_coefficient_0"]] += .1
    assert np.any(_texture_noise(altered, local, 32, "fur") != coat)
    np.testing.assert_array_equal(_marking(altered, local, 32, 36), coverage)
    # Pattern noise cannot change an unmarked coat.
    genes[13] = 0
    altered = genes.copy()
    altered[GENE_INDEX["pattern_noise_coefficient_0"]] += .1
    np.testing.assert_array_equal(body_colors(genes, local, 32, 36), body_colors(altered, local, 32, 36))


@pytest.mark.parametrize("kind", ["pattern", "fur"])
def test_noise_diversity_grows_with_population_difference(kind):
    distances = []
    for difference in (0, .03, .25, .75):
        fields = [_texture_noise(np.asarray(dna), points(), 32, kind).ravel()
                  for dna in population(12, difference, 42).values()]
        distances.append(np.mean([np.mean((a - b) ** 2) for i, a in enumerate(fields) for b in fields[i + 1:]]))
    assert distances[0] == 0
    assert all(a < b for a, b in zip(distances, distances[1:])), distances


def test_small_noise_gene_changes_produce_related_fields():
    genes = np.asarray(next(iter(population(1, .25, 42).values())))
    for kind in ("pattern", "fur"):
        base = _texture_noise(genes, points(), 32, kind)
        small, large = genes.copy(), genes.copy()
        index = GENE_INDEX[kind + "_noise_coefficient_0"]
        small[index] += .001
        large[index] += .1
        near = np.abs(_texture_noise(small, points(), 32, kind) - base).mean()
        far = np.abs(_texture_noise(large, points(), 32, kind) - base).mean()
        assert 0 < near < far / 20


def test_noise_reuses_dna_across_labels_clips_seeds_and_deformation():
    genes = np.asarray(next(iter(population(1, .25, 42, neck_focus=.8).values())))
    expected = body_colors(genes, points(), 32, 36)
    for seed, clip, label in ((42, "one", "animal"), (99, "two", "renamed")):
        state = Trajectory(SceneSettings((960, 640), .08, seed), clip, label)
        for _ in range(8):
            pose = state.pose(genes)
            local = pose.material(pose.world(points()))
            np.testing.assert_allclose(body_colors(genes, local, 32, 36), expected, atol=1)
            state.advance()


def test_texture_controls_are_encoded_and_v2_pixels_are_preserved(tmp_path):
    dna = population(1, .25, 42, texture_noise={})
    genes = np.asarray(next(iter(dna.values())))
    for prefix in NOISE_PREFIXES:
        assert genes[GENE_INDEX[prefix + "_strength"]] == 0
    np.testing.assert_array_equal(body_colors(genes, points(), 32, 36),
                                  body_colors(genes[:PATTERN_END], points(), 32, 36))
    legacy = {"v2": genes[:PATTERN_END].tolist()}
    scenes = synthetic.generate(individuals=legacy, clips=1, frames_per_clip=1, size=(160, 120),
                                augmentations=[], weather={}, destination=tmp_path / "legacy-v2")
    assert scenes.manifest["synthetic"]["version"] == 2
    assert scenes.manifest["synthetic"]["gene_names"] == list(GENES[:PATTERN_END])
    assert scenes.manifest["synthetic"]["individuals"] == legacy
    explicit = population(legacy, None, 99)
    assert explicit == legacy
    selected = next(iter(population(1, .25, 42, texture_noise={"fur": .4}).values()))
    assert selected[GENE_INDEX["fur_noise_strength"]] == selected[GENE_INDEX["neck_fur_noise_strength"]] == .4
    assert selected[GENE_INDEX["pattern_noise_strength"]] == selected[GENE_INDEX["neck_pattern_noise_strength"]] == 0


def test_neck_focus_keeps_global_noise_shared():
    genes = np.asarray(list(population(5, .25, 42, neck_focus=1).values()))
    for kind in ("pattern", "fur"):
        start = GENE_INDEX[kind + "_noise_strength"]
        np.testing.assert_array_equal(genes[:, start:start + 2 + NOISE_COMPONENTS],
                                      np.repeat(genes[:1, start:start + 2 + NOISE_COMPONENTS], 5, axis=0))
        local = GENE_INDEX["neck_" + kind + "_noise_coefficient_0"]
        assert np.ptp(genes[:, local]) > 0


@pytest.mark.parametrize("value", [{"fur": -1}, {"pattern": 1.1}, {"fur": float("nan")}, {"noise": .3}, []])
def test_noise_validation_before_writes(tmp_path, value):
    with pytest.raises(ValueError, match="texture_noise"):
        synthetic.generate(texture_noise=value, destination=tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_explicit_dna_cannot_silently_override_noise():
    dna = population(1, .25, 42)
    with pytest.raises(ValueError, match="explicit DNA"):
        population(dna, None, 42, texture_noise={"fur": .5})
