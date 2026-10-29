"""Check world-space foot planting, body attachments, and coherent weather."""
import numpy as np
import pytest
from PIL import Image

from dataset_fixer import synthetic
from dataset_fixer.synthetic._simulation import LIMB_END, SIMULATION_VERSION, SceneSettings, Trajectory, WeatherSettings, population
from dataset_fixer.synthetic._rendering import Weather, individual


def test_walking_feet_plant_bones_keep_length_and_roots_follow_body():
    dna = next(iter(population(1, 0, 42).values()))
    state = Trajectory(SceneSettings((960, 640), .08, 42), "clip", "animal")
    state.position = np.array([400., 320.])
    state.heading = 0.
    state.speed = 1.3
    previous = state.pose(dna)
    rig = state._rig
    roots = rig.roots.copy()
    planted_pairs, moving_feet = 0, 0
    changes = []
    for _ in range(100):
        state.advance()
        pose = state.pose(dna)
        np.testing.assert_allclose(pose.material(pose.joints[:, 0]), roots, atol=1e-10)
        lengths = np.linalg.norm(np.diff(pose.joints, axis=1), axis=2)
        np.testing.assert_allclose(lengths[rig.active], np.repeat(rig.bones[rig.active, None], 2, axis=1), atol=1e-10)
        holding = previous.planted & pose.planted
        np.testing.assert_allclose(pose.joints[holding, 2], previous.joints[holding, 2], atol=1e-10)
        planted_pairs += holding.sum()
        moving_feet += (np.linalg.norm(pose.joints[:, 2] - previous.joints[:, 2], axis=1)[rig.active] > .2).sum()
        changes.append(np.max(abs(pose.deform(rig.outline) - previous.deform(rig.outline))))
        previous = pose
    assert planted_pairs > 150 and moving_feet > 100
    assert 0 < np.median(changes) < 2
    np.testing.assert_array_equal(rig.roots, roots)
    np.testing.assert_array_equal(rig.active, np.asarray(dna)[14:LIMB_END:6] > .4)


def test_stationary_body_does_not_walk_and_zero_variation_still_steps():
    dna = next(iter(population(1, 0, 42).values()))
    state = Trajectory(SceneSettings((960, 640), 0, 42), "clip", "animal")
    initial = state.pose(dna)
    for _ in range(10):
        state.frame += 1
        np.testing.assert_array_equal(state.pose(dna).joints, initial.joints)
    rig = state._rig
    moving = []
    for _ in range(40):
        state.advance()
        pose = state.pose(dna)
        np.testing.assert_array_equal(pose.deform(rig.outline), rig.outline)
        moving.append(pose.material(pose.joints[:, 2]))
    assert np.max(np.ptp(moving, axis=0)) > 1


def test_extreme_dna_retains_finite_skeletons_and_attached_landmarks():
    size = (320, 240)
    for label, dna in population(12, 1, 51).items():
        state = Trajectory(SceneSettings(size, 1, 42), "clip", label)
        state.position = np.array([160., 120.])
        for _ in range(3):
            _, mask, points, _ = individual(dna, state, size)
            pose = state.pose(dna)
            assert np.isfinite(pose.joints).all()
            x, y = np.round(points).astype(int).T
            assert mask[y, x].all()
            state.advance()


def test_weather_is_bounded_local_and_temporally_coherent():
    size = (240, 160)
    image = np.full((160, 240, 3), 100, np.uint8)
    clouds = Weather(size, WeatherSettings(clouds=.6, fog=0), 42, "clip")
    a, b, c = [clouds.apply(image, frame) for frame in (0, 1, 60)]
    assert a.min() >= 40 and a.max() <= 100 and a.std() > 5
    assert np.abs(a.astype(float) - b).mean() < 1
    assert np.abs(a.astype(float) - c).mean() > 3
    np.testing.assert_array_equal(a, clouds.apply(image, 0))
    np.testing.assert_array_equal(a, Weather(size, clouds.settings, 42, "clip").apply(image, 0))
    assert not np.array_equal(a, Weather(size, clouds.settings, 42, "other").apply(image, 0))
    fog = Weather(size, WeatherSettings(clouds=0, fog=.5), 42, "clip").apply(image, 20)
    assert fog.min() >= 100 and fog.max() <= 168 and fog.std() > 5
    frozen = Weather(size, WeatherSettings(speed=0), 42, "clip")
    np.testing.assert_array_equal(frozen.apply(image, 0), frozen.apply(image, 100))
    np.testing.assert_array_equal(Weather(size, WeatherSettings.parse({}), 42, "clip").apply(image, 0), image)


def test_weather_preserves_scene_geometry_and_is_persisted(tmp_path):
    settings = dict(individuals=2, clips=1, frames_per_clip=3, size=(240, 160), augmentations=[])
    clear = synthetic.generate(**settings, weather={}, destination=tmp_path / "clear")
    cloudy = synthetic.generate(**settings, weather={"clouds": .6, "fog": .3, "speed": 2}, destination=tmp_path / "cloudy")
    assert cloudy.manifest["synthetic"]["weather"] == dict(clouds=.6, fog=.3, speed=2)
    assert cloudy.manifest["synthetic"]["simulation_version"] == SIMULATION_VERSION
    assert cloudy.manifest["synthetic"]["individuals"] == clear.manifest["synthetic"]["individuals"]
    for a, b in zip(clear._samples, cloudy._samples):
        assert a.provenance["synthetic"]["observations"] == b.provenance["synthetic"]["observations"]
        assert not np.array_equal(np.asarray(Image.open(a.image_path)), np.asarray(Image.open(b.image_path)))


@pytest.mark.parametrize("weather", [{"clouds": -1}, {"fog": 2}, {"speed": float("nan")}, {"rain": .2}, []])
def test_weather_validation_happens_before_writes(tmp_path, weather):
    with pytest.raises(ValueError, match="weather"):
        synthetic.generate(weather=weather, destination=tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
