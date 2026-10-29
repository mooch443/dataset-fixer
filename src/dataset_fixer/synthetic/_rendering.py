"""Procedural pixels and geometry; no dataset lifecycle or plotting machinery."""
from __future__ import annotations

import cv2
import numpy as np
from shapely.geometry import Polygon

from ..augmentation import _albumentations, _target_replay_pipeline
from ._simulation import rng_for, random_seed
from ._appearance import body_colors


class Background:
    def __init__(self, size, kind, seed, clip):
        width, height = size
        rng = rng_for(seed, "background", clip)
        self.kind = kind
        self.y, self.x = np.mgrid[:height, :width].astype(np.float32)
        coarse = cv2.resize(rng.normal(0, 1, (24, 32)).astype(np.float32), size,
                            interpolation=cv2.INTER_CUBIC)
        detail = rng.normal(0, 3, (height, width)).astype(np.float32)
        base = np.array([55, 86, 33] if kind == "grass" else [30, 79, 100], dtype=float)
        base += rng.uniform(-15, 15, 3)
        self.texture = np.clip(base + (coarse * 11 + detail)[..., None], 0, 255).astype(np.uint8)
        self.phase = rng.uniform(0, 2 * np.pi)
        if kind == "grass":
            for _ in range(width * height // 75):
                x, y = rng.integers([0, 0], [width, height])
                end = (int(x + rng.integers(-4, 5)), int(y - rng.integers(3, 12)))
                color = tuple(map(int, np.clip(base + rng.uniform(-20, 35), 0, 255)))
                cv2.line(self.texture, (int(x), int(y)), end, color, 1, cv2.LINE_AA)

    def frame(self, index):
        t = index * 0.045 + self.phase
        if self.kind == "grass":
            return cv2.remap(self.texture, self.x + 0.8 * np.sin(self.y / 35 + t), self.y,
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        ripple = 3 * np.sin(self.x / 19 + self.y / 9 - t) + 2 * np.sin(self.y / 7 + t * 1.4)
        return np.clip(self.texture.astype(float) + ripple[..., None], 0, 255).astype(np.uint8)


class Weather:
    """Advected multiscale fields shared by every object and the background.

    OpenCV interpolates the seeded fields; no randomness is drawn per frame.
    Fields are sampled on a wrapped, padded noise lattice to avoid edge seams.
    """
    def __init__(self, size, settings, seed, clip):
        self.settings = settings
        width, height = size
        self.y, self.x = np.mgrid[:height, :width].astype(np.float32) / min(size)
        self.layers = []
        for name in ("clouds", "fog"):
            rng = rng_for(seed, "weather", clip, name)
            coarse = rng.uniform(0, 1, (24, 24)).astype(np.float32)
            # Periodic padding lets cubic interpolation cross the lattice seam.
            field = np.pad(coarse, 2, mode="wrap")
            direction = rng.uniform(0, 2 * np.pi)
            velocity = np.array([np.cos(direction), np.sin(direction)]) * rng.uniform(.002, .004)
            origin = rng.uniform(0, 24, 2)
            self.layers.append((field, velocity, origin))

    def field(self, layer, frame):
        field, velocity, origin = self.layers[layer]
        drift = velocity * frame * self.settings.speed
        frequency = 3.0 if layer == 0 else 5.0
        result = np.zeros_like(self.x)
        for octave, weight in ((1, .7), (2.3, .3)):
            x = (self.x * frequency * octave - drift[0] * frequency * octave + origin[0]) % 24 + 2
            y = (self.y * frequency * octave - drift[1] * frequency * octave + origin[1]) % 24 + 2
            result += weight * cv2.remap(field, x.astype(np.float32), y.astype(np.float32), cv2.INTER_CUBIC)
        # Soft local patches with genuinely clear areas between them.
        density = np.clip((result - .3) / .5, 0, 1)
        return density * density * (3 - 2 * density)

    def apply(self, pixels, frame):
        if not (self.settings.clouds or self.settings.fog):
            return pixels
        image = pixels.astype(np.float32)
        if self.settings.clouds:
            image *= 1 - self.settings.clouds * self.field(0, frame)[..., None]
        if self.settings.fog:
            alpha = self.settings.fog * self.field(1, frame)[..., None]
            image = image * (1 - alpha) + np.array([225, 232, 235], np.float32) * alpha
        return np.rint(np.clip(image, 0, 255)).astype(np.uint8)


def individual(dna, state, size):
    """Draw a connected body with immutable limb presence, returning its full mask."""
    genes = np.asarray(dna)
    width, height = size
    pose = state.pose(dna)
    rig = state._rig
    scale, length, breadth = rig.scale, rig.length, rig.breadth
    mask = np.zeros((height, width), np.uint8)
    pixels = np.zeros((height, width, 3), np.uint8)
    extents = [pose.world(rig.outline)]
    limb_color = tuple(map(int, 35 + genes[6:9] * 200))
    for slot in np.flatnonzero(rig.active):
        joints = pose.joints[slot]
        thickness = max(1, round(2 * rig.widths[slot]))
        extents.extend((joints - thickness / 2, joints + thickness / 2))
        path = np.round(joints).astype(np.int32)
        cv2.polylines(mask, [path], False, 1, thickness)
        cv2.polylines(pixels, [path], False, limb_color, thickness)
    body_mask = np.zeros_like(mask)
    cv2.fillPoly(body_mask, [np.round(pose.world(rig.outline)).astype(np.int32)], 1)
    mask |= body_mask
    ys, xs = np.nonzero(body_mask)
    local = pose.material(np.column_stack((xs, ys)))
    pixels[ys, xs] = body_colors(genes, local, scale, length)
    # Two small eyes make the head direction visually inspectable.
    for side in (-1, 1):
        eye = pose.world([[length * 0.65, side * breadth * 0.43]])[0]
        cv2.circle(pixels, tuple(np.round(eye).astype(int)), max(1, round(scale * 0.045)), (12, 16, 18), -1)
    points = np.concatenate(extents)
    truncated = bool(np.any((points < 0) | (points >= np.array(size) - 1)))
    head_tail = pose.world([[length * 0.9, 0], [-length * 0.9, 0]])
    return pixels, mask, head_tail, truncated


class SceneEffects:
    def __init__(self, pipeline, seed, clip):
        self.pipeline = (_target_replay_pipeline(_albumentations(), pipeline, random_seed(seed, "lens", clip))
                         if pipeline["transform"].get("transforms") else None)
        self.replay = None

    def apply(self, pixels, masks, landmarks):
        if self.pipeline is None:
            return pixels, np.asarray(masks, dtype=bool), np.asarray(landmarks)
        inputs = dict(image=pixels, masks=np.asarray(masks, dtype=np.uint8), keypoints=np.asarray(landmarks).reshape(-1, 2),
                      keypoint_ids=np.arange(len(landmarks) * 2), bboxes=np.empty((0, 4)), bbox_ids=np.empty(0))
        if self.replay is None:
            result = self.pipeline(**inputs)
            self.replay = result["replay"]
        else:
            result = _albumentations().ReplayCompose.replay(self.replay, **inputs)
        image = result["image"]
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Scene augmentations must produce HWC RGB uint8 pixels; remove Normalize/ToTensor")
        points = np.empty((len(landmarks) * 2, 2))
        points[np.asarray(result["keypoint_ids"], dtype=int)] = result["keypoints"]
        return image, np.asarray(result["masks"], dtype=bool), points.reshape(-1, 2, 2)


def visibility(masks, truncated):
    """Flag both sides of every overlap; retain front-to-back visible silhouettes."""
    overlap = masks.sum(axis=0) > 1
    covered = np.zeros(masks.shape[1:], dtype=bool)
    visible = np.zeros_like(masks)
    observations = []
    for i in reversed(range(len(masks))):
        visible[i] = masks[i] & ~covered
        covered |= masks[i]
    for mask, prior_truncation in zip(masks, truncated):
        ys, xs = np.nonzero(mask)
        intersects = bool(np.any(mask & overlap))
        boundary = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
        clipped = bool(prior_truncation or boundary)
        box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None
        observations.append(dict(bbox=box, overlap=intersects, truncated=clipped,
                                 usable=bool(len(xs) >= 9 and not intersects and not clipped)))
    return visible, observations


def segmentation(mask):
    """Use an exact single polygon when possible, otherwise lossless COCO RLE."""
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 1 and len(contours[0]) >= 3 and Polygon(contours[0].reshape(-1, 2)).is_valid:
        candidate = np.zeros_like(mask, dtype=np.uint8)
        cv2.fillPoly(candidate, contours, 1)
        if np.array_equal(candidate, mask):
            return [contours[0].reshape(-1).astype(float).tolist()]
    from pycocotools import mask as mask_utils
    encoded = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
    return dict(size=encoded["size"], counts=encoded["counts"].decode("ascii"))


def crop_transform(box, head_tail, *, normalization, context, angular_jitter, size):
    x0, y0, x1, y1 = box
    center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    delta = np.asarray(head_tail[0]) - head_tail[1]
    angle = np.pi / 4 - np.arctan2(delta[1], delta[0]) + np.deg2rad(angular_jitter) if normalization == "posture" else 0.0
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    # Fit the rotated detection rectangle; sample context directly from the scene.
    extent = max(abs(c) * (x1 - x0) + abs(s) * (y1 - y0),
                 abs(s) * (x1 - x0) + abs(c) * (y1 - y0)) * (1 + 2 * context)
    linear = rotation * ((size - 1) / max(extent, 1))
    return np.column_stack((linear, (size - 1) / 2 - linear @ center))
