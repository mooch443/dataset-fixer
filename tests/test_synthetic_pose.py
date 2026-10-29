import albumentations as A
import numpy as np
import pytest

from dataset_fixer import synthetic
from dataset_fixer.augmentation import serialize_pipeline, _pose_after, _keypoint_tag
from dataset_fixer.id.data import index_of
from dataset_fixer.models import Annotation, Sample
from dataset_fixer.synthetic._rendering import SceneEffects
from dataset_fixer.synthetic._simulation import Trajectory


def test_dropped_landmarks_remain_missing_without_shifting_other_individuals():
    image = np.zeros((80, 120, 3), np.uint8)
    masks = np.zeros((3, 80, 120), np.uint8)
    landmarks = np.array([[[-10., 30.], [10., 30.]],
                          [[50., 30.], [60., 30.]],
                          [[130., 90.], [140., 90.]]])
    effects = SceneEffects(serialize_pipeline([A.OpticalDistortion(distort_limit=(0, 0), p=1)], {}), 42, "clip")
    for _ in range(2):  # Initial transform and its replay must handle missing IDs.
        _, _, points = effects.apply(image, masks, landmarks)
        np.testing.assert_array_equal(points[0, 0], [-1., -1.])
        np.testing.assert_array_equal(points[2], np.full((2, 2), -1.))
        np.testing.assert_allclose(points[0, 1], landmarks[0, 1], atol=1)
        np.testing.assert_allclose(points[1], landmarks[1], atol=1)
        assert np.isfinite(points).all()
    landmarks[0, 0] = [20., 30.]
    _, _, restored = effects.apply(image, masks, landmarks)
    np.testing.assert_allclose(restored[0], landmarks[0], atol=1)


@pytest.mark.parametrize("drop_inside", [False, True])
def test_partial_pose_hides_missing_head_and_restarts_identity_tracklet(tmp_path, monkeypatch, drop_inside):
    class Leaving(Trajectory):
        def __init__(self, settings, clip, label):
            super().__init__(settings, clip, label)
            self.number = int(label[-4:])
            self.heading = np.pi / 2
            self.position = self.location()

        def location(self):
            return np.array([40, 95 if self.frame == 1 and not drop_inside else 55]
                            if self.number == 0 else [145, 40], dtype=float)

        def advance(self):
            self.frame += 1
            self.position = self.location()

    monkeypatch.setattr(synthetic, "Trajectory", Leaving)
    if drop_inside:
        apply = SceneEffects.apply
        calls = 0

        def drop_point(self, *args):
            nonlocal calls
            image, masks, points = apply(self, *args)
            calls += 1
            if calls == 2:
                points[0, 0] = [-1., -1.]  # Remapping lost a point, despite a usable mask.
            return image, masks, points

        monkeypatch.setattr(SceneEffects, "apply", drop_point)
    scenes = synthetic.generate(individuals=2, clips=1, frames_per_clip=3, size=(192, 96), task="pose",
                                augmentations=[A.OpticalDistortion(distort_limit=(0, 0), p=1)],
                                weather={}, variation=0, progress=False, destination=tmp_path / "scenes")
    sample = scenes._samples[1]
    annotation = next(a for a in sample.annotations if a.identity_id == "individual-0000")
    assert annotation.keypoints[0] == (0., 0., 0.)
    assert annotation.keypoints[1][2] == 2
    assert next(a for a in sample.annotations if a.identity_id == "individual-0001").keypoints[0][2] == 2
    observations = sample.provenance["synthetic"]["observations"]
    assert observations[0]["head_tail"][0] == [-1., -1.]
    assert observations[0]["truncated"] == (not drop_inside)
    assert not observations[0]["usable"] and observations[1]["usable"]
    crops = synthetic.identity(scenes, destination=tmp_path / "crops")
    frame = index_of(crops).frame
    scarce = frame.loc[frame.identity_id == "individual-0000"]
    assert scarce.frame_index.tolist() == [0, 2]
    assert scarce.tracklet_id.nunique() == 2


def test_regular_pose_augmentation_does_not_restore_dropped_points_at_origin(tmp_path):
    sample = Sample(tmp_path / "unused.png", tmp_path / "unused.png", "train", 120, 80,
                    [Annotation(0, bbox=(0, 0, 60, 60), keypoints=[(20, 20, 2), (40, 40, 2)])])
    result = dict(bboxes=[(0, 0, 60, 60)], bbox_ids=[0], keypoints=[(40., 40.)],
                  keypoint_ids=[_keypoint_tag(0, 1)])
    annotations = _pose_after(sample, result, 120, 80, flip_idx=None, horizontal_flip=False)
    assert annotations[0].keypoints == [(0., 0., 0.), (40., 40., 2)]
