"""Seeded population and trajectory state, independent of rendering and storage."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math

import numpy as np


VERSION = 3
SIMULATION_VERSION = 4
SLOTS = 8
LIMB_END = 14 + SLOTS * 6
PATTERNS = ("stripes", "dots", "stars")
GENES = (
    "body_length", "body_width", "roundness", "body_red", "body_green", "body_blue",
    "limb_red", "limb_green", "limb_blue", "mark_red", "mark_green", "mark_blue",
    "mark_frequency", "mark_strength",
    *(f"limb_{i}_{field}" for i in range(SLOTS)
      for field in ("presence", "attachment", "offset", "length", "angle", "width")),
    *(f"mark_{pattern}" for pattern in PATTERNS),
    "mark_size", "mark_offset_x", "mark_offset_y", "neck_focus",
    "neck_red", "neck_green", "neck_blue", "neck_mark_red", "neck_mark_green", "neck_mark_blue",
    *(f"neck_mark_{pattern}" for pattern in PATTERNS),
    "neck_mark_size", "neck_mark_offset_x", "neck_mark_offset_y", "neck_mark_frequency", "neck_mark_strength",
)
PATTERN_END = len(GENES)
NOISE_COMPONENTS = 8
NOISE_PREFIXES = ("pattern_noise", "fur_noise", "neck_pattern_noise", "neck_fur_noise")
GENES += tuple(f"{prefix}_{field}" for prefix in NOISE_PREFIXES
               for field in ("strength", "frequency", *(f"coefficient_{i}" for i in range(NOISE_COMPONENTS))))
DNA_LENGTHS = (LIMB_END, PATTERN_END, len(GENES))
GENE_INDEX = {name: i for i, name in enumerate(GENES)}


def random_seed(seed, *keys):
    # The random-stream schema is independent of DNA and renderer revisions.
    token = "\0".join(map(str, (1, seed, *keys))).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "little")


def rng_for(seed, *keys):
    return np.random.default_rng(random_seed(seed, *keys))


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def fraction(value, name, *, maximum=1.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be finite and in [0, {maximum}]")
    return float(value)


def population(individuals, difference, seed, *, patterns=None, neck_focus=None, texture_noise=None):
    if isinstance(individuals, Mapping):
        if difference is not None:
            raise ValueError("difference applies only when individuals is a count")
        if patterns is not None or neck_focus is not None or texture_noise is not None:
            raise ValueError("patterns, neck_focus, and texture_noise apply only when individuals is a count; edit explicit DNA instead")
        if not individuals or any(not isinstance(k, str) or not k for k in individuals):
            raise ValueError("individuals must map non-empty identity strings to DNA vectors")
        result = {}
        for label, dna in sorted(individuals.items()):
            genes = np.asarray(dna, dtype=float)
            if genes.shape not in {(n,) for n in DNA_LENGTHS} or not np.isfinite(genes).all() or np.any((genes < 0) | (genes > 1)):
                raise ValueError(f"DNA for {label!r} must contain one of {DNA_LENGTHS} finite genes in [0, 1]")
            result[label] = genes.tolist()
        if len({len(v) for v in result.values()}) != 1:
            raise ValueError("All individuals must use the same DNA version")
        return result
    count = positive_int(individuals, "individuals")
    difference = fraction(0.25 if difference is None else difference, "difference")
    neck_focus = fraction(0.0 if neck_focus is None else neck_focus, "neck_focus")
    if patterns is None:
        patterns = PATTERNS
    if isinstance(patterns, str):
        patterns = (patterns,)
    if not isinstance(patterns, (list, tuple)) or not patterns or any(p not in PATTERNS for p in patterns):
        raise ValueError("patterns must contain stripes, dots, and/or stars")
    if texture_noise is None:
        texture_noise = dict(pattern=.25, fur=.18)
    if not isinstance(texture_noise, Mapping) or set(texture_noise) - {"pattern", "fur"}:
        raise ValueError("texture_noise must map pattern and/or fur to strengths in [0, 1]")
    noise_strengths = {kind: fraction(texture_noise.get(kind, 0), f"texture_noise.{kind}") for kind in ("pattern", "fur")}
    base = rng_for(seed, "population").uniform(0.25, 0.75, len(GENES))
    # Keep low-spread populations clear of topology thresholds.
    base[14:LIMB_END:6] = [0.8, 0.8, 0.8, 0.2, 0.8, 0.8, 0.8, 0.2]
    # Balanced pattern weights allow every requested motif to appear in a
    # population. Rendering blends them continuously, without categorical jumps.
    for prefix in ("mark_", "neck_mark_"):
        for pattern in PATTERNS:
            base[GENE_INDEX[prefix + pattern]] = .35 if pattern in patterns else 0
    focus_index = GENE_INDEX["neck_focus"]
    spread = np.full(len(GENES), difference)
    spread[:focus_index] *= 1 - neck_focus
    for prefix in NOISE_PREFIXES:
        start = GENE_INDEX[prefix + "_strength"]
        if not prefix.startswith("neck_"):
            spread[start:start + 2 + NOISE_COMPONENTS] *= 1 - neck_focus
        # Strengths are independent user controls; identity differences live in
        # continuous field coefficients and frequencies, not random seed hashes.
        base[start] = noise_strengths["fur" if "fur" in prefix else "pattern"]
        spread[start] = 0
    result = {}
    for i in range(count):
        genes = np.clip(base + spread * rng_for(seed, "dna", i).uniform(-1, 1, len(GENES)), 0, 1)
        genes[focus_index] = neck_focus
        for prefix in ("mark_", "neck_mark_"):
            for pattern in PATTERNS:
                if pattern not in patterns:
                    genes[GENE_INDEX[prefix + pattern]] = 0
        result[f"individual-{i:04d}"] = genes.tolist()
    return result


@dataclass(frozen=True)
class SceneSettings:
    size: tuple[int, int]
    variation: float
    seed: int
    individual_scale: float = 1.0


@dataclass(frozen=True)
class WeatherSettings:
    clouds: float = 0.25
    fog: float = 0.12
    speed: float = 1.0

    @classmethod
    def parse(cls, value):
        if value is None:
            return cls()
        if not isinstance(value, Mapping) or set(value) - {"clouds", "fog", "speed"}:
            raise ValueError("weather must map clouds, fog, and/or speed to numeric strengths")
        return cls(fraction(value.get("clouds", 0), "weather.clouds"),
                   fraction(value.get("fog", 0), "weather.fog"),
                   fraction(value.get("speed", 1), "weather.speed", maximum=10))


class Trajectory:
    def __init__(self, settings, clip, label):
        self.settings = settings
        self.rng = rng_for(settings.seed, "motion", clip, label)
        self.position = self.rng.uniform(0.12, 0.88, 2) * settings.size
        self.heading = self.rng.uniform(-np.pi, np.pi)
        self.turn = 0.0
        self.speed = min(settings.size) * self.rng.uniform(0.0015, 0.004)
        # Retain the trajectory RNG sequence across the animation upgrade.
        self.gait_phase = float(self.rng.uniform(0, 2 * np.pi, SLOTS)[0])
        self.deformation = np.zeros(SLOTS)
        self.distance = 0.0
        self.frame = 0
        self._rig = None

    def pose(self, dna):
        from ._animation import WalkingRig

        if self._rig is None or not np.array_equal(self._rig.dna, dna):
            self._rig = WalkingRig(dna, self.settings.size, self.settings.individual_scale)
        return self._rig.evaluate(self)

    def advance(self):
        previous = self.position.copy()
        self.turn = 0.94 * self.turn + self.rng.normal(0, 0.007)
        self.heading += self.turn
        direction = np.array([np.cos(self.heading), np.sin(self.heading)])
        self.position += self.speed * direction
        margin = min(self.settings.size) * 0.12
        # Reflect outside the camera field, allowing genuine exit/re-entry gaps.
        for axis, extent in enumerate(self.settings.size):
            if self.position[axis] < -margin or self.position[axis] > extent + margin:
                self.position[axis] = np.clip(self.position[axis], -margin, extent + margin)
                self.heading = np.pi - self.heading if axis == 0 else -self.heading
        self.distance += np.linalg.norm(self.position - previous)
        self.frame += 1
        self.deformation = 0.95 * self.deformation + 0.05 * self.rng.uniform(-1, 1, SLOTS)


class Tracklets:
    """Every lost observation terminates a track, including an isolated loss."""
    def __init__(self):
        self.state = {}

    def observe(self, clip, label, frame, usable):
        key = (clip, label)
        previous, serial = self.state.get(key, (None, -1))
        if not usable:
            self.state[key] = (None, serial)
            return None
        if previous is None or frame != previous + 1:
            serial += 1
        self.state[key] = (frame, serial)
        return f"{clip}/{label}/{serial:04d}"
