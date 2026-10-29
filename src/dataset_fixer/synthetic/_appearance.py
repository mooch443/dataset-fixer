"""Body-attached pattern tiles rendered with OpenCV and SciPy interpolation."""
from functools import lru_cache

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from ._simulation import GENE_INDEX, LIMB_END, PATTERN_END, PATTERNS, NOISE_COMPONENTS, rng_for


@lru_cache(maxsize=2)
def _noise_basis(kind):
    """Fixed, independent smooth bases; genomes mix them without reseeding."""
    rng = rng_for(0, "texture-noise-basis", kind)
    bases = rng.normal(size=(NOISE_COMPONENTS, 3 if kind == "pattern" else 1, 64, 64))
    sigma = (0, 0, .6, 2.0) if kind == "fur" else (0, 0, 2.0, 2.0)
    bases = gaussian_filter(bases, sigma, mode="wrap")
    bases -= np.mean(bases, axis=(-2, -1), keepdims=True)
    bases /= np.std(bases, axis=(-2, -1), keepdims=True)
    return bases.astype(np.float32)


@lru_cache(maxsize=128)
def _noise_tile(kind, coefficients, footprint):
    weights = (np.asarray(coefficients) * 2 - 1) / np.sqrt(NOISE_COMPONENTS)
    field = np.tanh(np.tensordot(weights, _noise_basis(kind), axes=1))
    # Prefilter for the output pixel footprint so fine coat grain does not
    # shimmer when the animal moves by a fraction of a pixel.
    return gaussian_filter(field, (0, .35 * footprint, .35 * footprint), mode="wrap")


def _texture_noise(genes, local, scale, kind, *, neck=False):
    """Material-space fields: independent of identity label, clip, and frame."""
    if len(genes) <= PATTERN_END:
        return None
    prefix = ("neck_" if neck else "") + kind + "_noise_"
    strength = genes[GENE_INDEX[prefix + "strength"]]
    if not strength:
        return None
    start = GENE_INDEX[prefix + "coefficient_0"]
    frequency = genes[GENE_INDEX[prefix + "frequency"]]
    # Fur has fine directional grain; pattern noise has broader irregularities.
    frequency = (.35 + .45 * frequency) if kind == "fur" else (.5 + frequency)
    tile = _noise_tile(kind, tuple(genes[start:start + NOISE_COMPONENTS]), frequency * 64 / scale)
    coordinates = (np.asarray(local)[:, ::-1].T / scale * frequency + 2) * tile.shape[-1]
    values = np.column_stack([map_coordinates(channel, coordinates, order=1, mode="grid-wrap", prefilter=False)
                              for channel in tile])
    return strength * values


@lru_cache(maxsize=256)
def _tile(weights, size):
    side = 128
    radius = side * (.13 + .18 * size)
    layers = np.zeros((3, side, side), np.uint8)
    cv2.rectangle(layers[0], (round(side / 2 - radius), 0), (round(side / 2 + radius), side - 1), 255, -1)
    cv2.circle(layers[1], (side // 2, side // 2), round(radius), 255, -1, cv2.LINE_AA)
    angles = np.arange(10) * np.pi / 5
    radii = np.where(np.arange(10) % 2, radius * .5, radius * 1.25)
    vertices = np.column_stack((np.cos(angles), np.sin(angles))) * radii[:, None] + side / 2
    cv2.fillPoly(layers[2], [np.round(vertices).astype(np.int32)], 255, lineType=cv2.LINE_AA)
    weights = np.asarray(weights) ** 4
    # Vanishing pattern genes fade out instead of normalizing tiny values into
    # a suddenly opaque motif.
    weights /= max(weights.sum(), .1 ** 4)
    return np.tensordot(weights, layers.astype(float) / 255, axes=1)


def _marking(genes, local, scale, length, *, neck=False):
    prefix = "neck_mark_" if neck else "mark_"
    weights = tuple(genes[GENE_INDEX[prefix + pattern]] for pattern in PATTERNS)
    tile = _tile(weights, genes[GENE_INDEX[prefix + "size"]])
    frequency = genes[GENE_INDEX[prefix + "frequency"]]
    period = scale * (.6 + 1 - frequency)
    offset = np.array([genes[GENE_INDEX[prefix + "offset_x"]], genes[GENE_INDEX[prefix + "offset_y"]]])
    noise = _texture_noise(genes, local, scale, "pattern", neck=neck)
    if neck:
        # Anchor a column inside the narrow neck band. A global body lattice can
        # otherwise put every dot/star outside that band for some chromosomes.
        local = local - [length * .375, 0]
        period = length * .45 * (1 - .35 * frequency)
        offset = .5 + (offset - .5) * [.2, .8]
    coordinates = (local / period + offset) * tile.shape[0]
    if noise is not None:
        coordinates += noise[:, :2] * tile.shape[0] * .22
    coverage = map_coordinates(tile, coordinates[:, ::-1].T, order=1, mode="grid-wrap", prefilter=False)
    if noise is not None:
        coverage = np.clip(coverage * (1 + .35 * noise[:, 2]), 0, 1)
    return coverage


def neck_weight(local, length):
    """Forward band just behind the eyes: x/half-length in [.15, .60]."""
    x = np.asarray(local)[..., 0] / length
    edge = np.clip(np.minimum((x - .15) / .08, (.60 - x) / .08), 0, 1)
    return edge * edge * (3 - 2 * edge)


def body_colors(genes, local, scale, length):
    base = 35 + genes[3:6] * 200
    mark = 35 + genes[9:12] * 200
    if len(genes) == LIMB_END:
        # Preserve appearances when reusing the original 62-gene chromosomes.
        coverage = np.cos(local[:, 0] / max(scale, 1) * (3 + 12 * genes[12])) > .55
    else:
        coverage = _marking(genes, local, scale, length)
    alpha = coverage[:, None] * genes[13] * .8
    colors = base * (1 - alpha) + mark * alpha
    fur = _texture_noise(genes, local, scale, "fur")
    if fur is not None:
        colors += fur * 45
    if len(genes) > LIMB_END:
        focus = genes[GENE_INDEX["neck_focus"]]
        if focus:
            neck_base = 35 + genes[GENE_INDEX["neck_red"]:GENE_INDEX["neck_blue"] + 1] * 200
            neck_mark = 35 + genes[GENE_INDEX["neck_mark_red"]:GENE_INDEX["neck_mark_blue"] + 1] * 200
            alpha = _marking(genes, local, scale, length, neck=True)[:, None] * genes[GENE_INDEX["neck_mark_strength"]] * .8
            neck = neck_base * (1 - alpha) + neck_mark * alpha
            fur = _texture_noise(genes, local, scale, "fur", neck=True)
            if fur is not None:
                neck += fur * 45
            blend = focus * neck_weight(local, length)[:, None]
            colors = colors * (1 - blend) + neck * blend
    return np.clip(colors, 0, 255).astype(np.uint8)
