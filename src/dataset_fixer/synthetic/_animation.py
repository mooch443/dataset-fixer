"""Planar planted-foot gait and analytic two-bone IK, evaluated before rendering.

The support/swing cycle is driven by distance travelled, not wall-clock time.
Material coordinates keep limb roots, eyes, and markings attached to the body.
"""
from dataclasses import dataclass

import numpy as np

from ._simulation import SLOTS, LIMB_END


@dataclass
class BodyPose:
    position: np.ndarray
    rotation: np.ndarray
    length: float
    breadth: float
    phase: float
    variation: float
    joints: np.ndarray
    planted: np.ndarray

    def deform(self, points, *, inverse=False):
        points = np.asarray(points, dtype=float).copy()
        stretch = 1 + self.variation * 0.12 * np.sin(2 * self.phase)
        breath = 1 + self.variation * 0.25 * np.cos(2 * self.phase)
        if inverse:
            points[..., 0] /= stretch
        bend = self.variation * self.length * 0.7 * np.sin(points[..., 0] / self.length * 1.5 + self.phase)
        sway = self.variation * self.breadth * np.sin(2 * self.phase)
        if inverse:
            points[..., 1] = (points[..., 1] - bend - sway) / breath
        else:
            points[..., 0] *= stretch
            points[..., 1] = points[..., 1] * breath + bend + sway
        return points

    def world(self, points):
        return self.deform(points) @ self.rotation.T + self.position

    def material(self, points):
        return self.deform((np.asarray(points) - self.position) @ self.rotation, inverse=True)


def two_bone(hips, feet, lengths, sides):
    """Equal-length two-bone chains: cosine-rule knee with a stable bend side."""
    delta = feet - hips
    distance = np.linalg.norm(delta, axis=1)
    direction = delta / np.maximum(distance[:, None], 1e-12)
    direction[distance < 1e-12] = [1, 0]
    reach = np.minimum(distance, lengths * 2 * .999)
    tips = hips + direction * reach[:, None]
    perpendicular = np.column_stack((-direction[:, 1], direction[:, 0])) * sides[:, None]
    height = np.sqrt(np.maximum(0, lengths ** 2 - (reach / 2) ** 2))
    knees = (hips + tips) / 2 + perpendicular * height[:, None]
    return np.stack((hips, knees, tips), axis=1), distance <= lengths * 2 * .999


class WalkingRig:
    duty = .62

    def __init__(self, dna, size, individual_scale=1.0):
        self.dna = np.asarray(dna).copy()
        self.scale = min(size) * .065 * individual_scale
        self.length = self.scale * (.75 + self.dna[0] * .65)
        self.breadth = self.scale * (.25 + self.dna[1] * .5)
        power = 2 + self.dna[2] * 8
        angle = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        self.outline = np.column_stack((self.length * np.sign(np.cos(angle)) * abs(np.cos(angle)) ** (2 / power),
                                       self.breadth * np.sign(np.sin(angle)) * abs(np.sin(angle)) ** (2 / power)))
        presence, attachment, offset, extent, tilt, thickness = self.dna[14:LIMB_END].reshape(SLOTS, 6).T
        self.active = presence > .4
        self.sides = np.where(np.arange(SLOTS) < SLOTS // 2, -1., 1.)
        growth = np.clip((presence - .4) / .2, 0, 1)
        x = (attachment * 1.6 - .8) * self.length
        edge = self.breadth * np.maximum(0, 1 - abs(x / self.length) ** power) ** (1 / power)
        self.roots = np.column_stack((x, self.sides * edge * .85))
        reach = self.scale * (.20 + .65 * extent) * growth
        self.rest = self.roots + np.column_stack((reach * (tilt - .5), self.sides * reach * (.65 + .35 * offset)))
        self.bones = reach * .78
        self.widths = self.scale * (.035 + .07 * thickness) * growth
        self.strides = np.maximum(reach * 1.05, .1)
        # Alternating contralateral groups; DNA determines roots, never gait phase.
        self.offsets = ((np.arange(SLOTS) % 2) + (self.sides > 0)) % 2 * .5
        self.feet = None
        self.starts = None
        self.was_support = np.zeros(SLOTS, bool)
        self.last_frame = None
        self.last_pose = None
        self.phase = None

    def evaluate(self, state):
        if self.last_frame == state.frame:
            return self.last_pose
        c, s = np.cos(state.heading), np.sin(state.heading)
        rotation = np.array([[c, -s], [s, c]])
        if self.phase is None:
            self.phase = state.gait_phase
        stride = np.median(self.strides[self.active]) if self.active.any() else self.scale
        phase = state.distance / stride * 2 * np.pi + self.phase + .25 * state.deformation[0]
        pose = BodyPose(state.position.copy(), rotation, self.length, self.breadth, phase,
                        state.settings.variation, np.empty((SLOTS, 3, 2)), np.zeros(SLOTS, bool))
        hips, rest = pose.world(self.roots), pose.world(self.rest)
        cycles = state.distance / self.strides + self.offsets + self.phase / (2 * np.pi)
        progress = cycles % 1
        support = progress < self.duty
        forward = rotation[:, 0]
        lead = self.strides[:, None] * (self.duty / 2) * forward
        targets = rest + lead
        if self.feet is None:
            self.feet = rest + self.strides[:, None] * (self.duty / 2 - np.minimum(progress, self.duty))[:, None] * forward
            self.starts = self.feet.copy()
        else:
            lifting = self.was_support & ~support
            self.starts[lifting] = self.feet[lifting]
            landing = ~self.was_support & support
            self.feet[landing] = targets[landing]
        # A sharp turn can exhaust reach. Lift early instead of sliding a planted
        # foot or stretching a bone. Normal steps keep exact world-space anchors.
        early = support & (np.linalg.norm(self.feet - hips, axis=1) > self.bones * 1.9) & self.active
        self.offsets[early] += self.duty - progress[early]
        progress[early] = self.duty
        support[early] = False
        self.starts[early] = self.feet[early]
        t = np.clip((progress - self.duty) / (1 - self.duty), 0, 1)
        eased = t * t * (3 - 2 * t)
        swinging = self.starts * (1 - eased[:, None]) + targets * eased[:, None]
        # A small outward swing arc is the top-down projection of a lifted foot.
        swinging += (np.sin(np.pi * t) ** 2 * self.bones * .12 * self.sides)[:, None] * rotation[:, 1]
        self.feet[~support] = swinging[~support]
        pose.joints, reachable = two_bone(hips, self.feet, self.bones, self.sides)
        pose.planted = support & reachable & self.active
        self.was_support = support
        self.last_frame, self.last_pose = state.frame, pose
        return pose
