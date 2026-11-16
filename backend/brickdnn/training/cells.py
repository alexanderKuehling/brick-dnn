"""The built-in segmentation dataset "Cells": generated, nothing is downloaded (needs NumPy).

Every image looks like a fluorescence micrograph: two to seven cells on a dark, slightly
tinted background with sensor noise and vignetting. A cell is an ellipse of random size,
aspect and rotation (its outline slightly wavy), stained pink to magenta; inside it, a little
off its centre, lies its nucleus, a smaller ellipse stained blue to violet. The mask says for
every pixel whether it is background (0), cell (1, the cytoplasm) or nucleus (2).

Cells are placed mostly apart; some touch a neighbour and some are cut by the border of the
image. Images differ in brightness and contrast, cells in how strongly they are stained, and
the colours of the classes come closer in some images, so a network needs the outlines (a
nucleus is a smaller ellipse inside a larger one) and not only the colour. The nucleus
covers only about 3 % of the pixels: a small class, which shows what the weighted
cross-entropy and the Dice loss are good for.

As for Shapes (shapes.py), image ``index`` only depends on ``index`` (a seed of its own), so
the dataset is the same on every machine and for every run, and any part of it can be
generated on its own.
"""

from __future__ import annotations

import colorsys
import math
from typing import Any

CELL_CLASSES = ("background", "cell", "nucleus")
CELLS_SIZE = 96  # pixels (square images)
CELLS_SEED = 20_261_116
# A cell's centre keeps at least this distance (in units of the two cells' radii) from the
# centres of the cells placed before it, unless it is placed to touch one of them.
_MIN_DISTANCE = 1.05
# The share of cells (after the first) that are placed right next to an earlier one.
_TOUCHING = 0.2
# The share of cells that may reach beyond the border of the image.
_AT_BORDER = 0.3


def _numpy() -> Any:
    import numpy

    return numpy


def _interpolation(size: int, grid: int) -> Any:
    """The weights (size × grid) that interpolate grid values linearly to size pixels."""
    np = _numpy()
    steps = np.linspace(0, grid - 1, size)
    low = np.clip(np.floor(steps).astype(int), 0, grid - 2)
    weights = np.zeros((size, grid), dtype=np.float32)
    weights[np.arange(size), low] = 1 - (steps - low)
    weights[np.arange(size), low + 1] = steps - low
    return weights


def _smooth_noise(rng: Any, size: int, grid: int) -> Any:
    """Soft blotches: a grid × grid of random values (mean 0, deviation 1), interpolated
    bilinearly to size × size."""
    np = _numpy()
    values = rng.normal(0.0, 1.0, size=(grid, grid)).astype(np.float32)
    weights = _interpolation(size, grid)
    return weights @ values @ weights.T


def _ellipse_distance(
    xx: Any, yy: Any, cx: float, cy: float, a: float, b: float, angle: float, wave: Any
) -> Any:
    """Approximately the signed distance in pixels (negative inside) to the outline of an
    ellipse with the half axes ``a`` and ``b``, turned by ``angle``. ``wave`` (amplitudes and
    phases of harmonics 2, 3 … of the angle) makes the outline slightly wavy.

    The level f of (u/a)² + (v/b)² = f² is 1 on the outline, and (f - 1) / |∇f| is its
    distance to first order: exact enough for an edge one pixel wide."""
    np = _numpy()
    cos, sin = math.cos(angle), math.sin(angle)
    dx, dy = xx - cx, yy - cy
    u = dx * cos + dy * sin
    v = -dx * sin + dy * cos
    level = np.sqrt((u / a) ** 2 + (v / b) ** 2)
    gradient = np.sqrt((u / a**2) ** 2 + (v / b**2) ** 2) / np.maximum(level, 1e-6)
    if len(wave):
        theta = np.arctan2(v / b, u / a)
        radius = 1.0
        for harmonic, (amplitude, phase) in enumerate(wave, start=2):
            radius = radius + amplitude * np.cos(harmonic * theta + phase)
        level = level / radius
        gradient = gradient / radius
    return (level - 1.0) / np.maximum(gradient, 1e-6)


def _blur_axis(image: Any, kernel: Any, axis: int) -> Any:
    """Blur ``image`` along ``axis`` with a symmetric kernel (edges repeated)."""
    np = _numpy()
    radius = len(kernel) // 2
    size = image.shape[axis]

    def part(start: int, stop: int) -> Any:
        return image[(slice(None),) * axis + (slice(start, stop),)]

    first, last = part(0, 1), part(size - 1, size)
    padded = np.concatenate([*([first] * radius), image, *([last] * radius)], axis=axis)

    def shifted(offset: int) -> Any:
        start = radius + offset
        return padded[(slice(None),) * axis + (slice(start, start + size),)]

    blurred = kernel[radius] * image
    for offset in range(1, radius + 1):
        blurred += kernel[radius + offset] * (shifted(-offset) + shifted(offset))
    return blurred


def _gaussian_blur(image: Any, sigma: float) -> Any:
    """Blur height and width of ``image`` (H × W × C) with a Gaussian."""
    np = _numpy()
    radius = max(1, int(math.ceil(2.5 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return _blur_axis(_blur_axis(image, kernel, 0), kernel, 1)


def _colour(hue: float, saturation: float) -> Any:
    np = _numpy()
    return np.array(colorsys.hsv_to_rgb(hue % 1.0, saturation, 1.0), dtype=np.float32)


def _place(
    rng: Any, size: int, radius: float, placed: list[tuple[float, float, float]]
) -> tuple[float, float]:
    """A centre for a cell of about ``radius``: apart from the cells placed so far, or
    (sometimes) touching one of them. The centre may lie near the border, so that the
    image cuts the cell."""
    if placed and float(rng.uniform()) < _TOUCHING:
        px, py, pr = placed[int(rng.integers(len(placed)))]
        direction = float(rng.uniform(0, 2 * math.pi))
        # The radii are rough for ellipses: a little closer than their sum touches.
        reach = (pr + radius) * float(rng.uniform(0.82, 0.95))
        cx, cy = px + reach * math.cos(direction), py + reach * math.sin(direction)
        if -0.2 * radius < cx < size + 0.2 * radius and -0.2 * radius < cy < size + 0.2 * radius:
            return cx, cy
    # Most cells lie inside the image, some are cut by its border.
    margin = (0.25 if float(rng.uniform()) < _AT_BORDER else 1.0) * radius
    best, best_gap = (size / 2, size / 2), -math.inf
    for _attempt in range(30):
        cx = float(rng.uniform(margin, size - margin))
        cy = float(rng.uniform(margin, size - margin))
        gap = min(
            (
                math.hypot(cx - px, cy - py) / (_MIN_DISTANCE * (radius + pr))
                for px, py, pr in placed
            ),
            default=math.inf,
        )
        if gap > 1:
            return cx, cy
        if gap > best_gap:
            best, best_gap = (cx, cy), gap
    return best  # a crowded image: where it overlaps least


def cells_sample(index: int, size: int = CELLS_SIZE) -> tuple[Any, Any]:
    """Image ``index`` as uint8 arrays: the picture (size × size × 3, RGB) and its mask
    (size × size, class indices, see CELL_CLASSES)."""
    np = _numpy()
    rng = np.random.default_rng([CELLS_SEED, int(index)])
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) + 0.5
    scale = size / CELLS_SIZE  # the sizes below are for CELLS_SIZE

    # The image: how bright it is, how much the stains glow against the background.
    gain = float(rng.uniform(0.45, 1.0))
    # The stains: cytoplasm pink to magenta, nuclei blue to violet; in some images the two
    # hues come closer (a violet nucleus in a purple cell).
    closer = float(rng.uniform(0.0, 0.06))
    cytoplasm_hue = float(rng.uniform(0.86, 0.97)) - closer
    nucleus_hue = float(rng.uniform(0.62, 0.72)) + closer
    cytoplasm_saturation = float(rng.uniform(0.5, 0.85))
    nucleus_saturation = float(rng.uniform(0.55, 0.85))

    mask = np.zeros((size, size), dtype=np.uint8)
    signal = np.zeros((size, size, 3), dtype=np.float32)
    placed: list[tuple[float, float, float]] = []
    nuclei = []
    for _ in range(int(rng.integers(2, 8))):
        a = float(rng.uniform(9.0, 17.0)) * scale  # the longer half axis
        b = a * float(rng.uniform(0.55, 0.95))
        radius = (a + b) / 2
        cx, cy = _place(rng, size, radius, placed)
        placed.append((cx, cy, radius))
        angle = float(rng.uniform(0, math.pi))
        wave = [
            (float(rng.uniform(0.0, 0.05)), float(rng.uniform(0, 2 * math.pi)))
            for _harmonic in range(3)
        ]
        # Only the pixels around the cell are computed.
        reach = a * 1.15 + 2
        box = (
            slice(max(0, int(cy - reach)), min(size, int(cy + reach) + 1)),
            slice(max(0, int(cx - reach)), min(size, int(cx + reach) + 1)),
        )
        bx, by = xx[box], yy[box]
        distance = _ellipse_distance(bx, by, cx, cy, a, b, angle, wave)
        coverage = np.clip(0.5 - distance, 0.0, 1.0)
        # Stained unevenly: brighter towards the nucleus, with a grain (below).
        strength = float(rng.uniform(0.35, 1.0))
        hue = cytoplasm_hue + float(rng.uniform(-0.03, 0.03))
        falloff = 1.0 - 0.35 * np.clip(1.0 + distance / radius, 0.0, 1.0) ** 2
        colour = _colour(hue, cytoplasm_saturation + float(rng.uniform(-0.1, 0.1)))
        signal[box] += (strength * coverage * falloff)[..., None] * colour
        mask[box][distance < 0] = 1

        # The nucleus: a smaller ellipse, a little off the centre of the cell.
        share = float(rng.uniform(0.33, 0.46))
        na = a * share * float(rng.uniform(0.9, 1.1))
        nb = b * share * float(rng.uniform(0.9, 1.15))
        room = (1.0 - share) * 0.45
        shift_u = float(rng.uniform(-room, room)) * a
        shift_v = float(rng.uniform(-room, room)) * b
        cos, sin = math.cos(angle), math.sin(angle)
        ncx, ncy = cx + shift_u * cos - shift_v * sin, cy + shift_u * sin + shift_v * cos
        nangle = angle + float(rng.uniform(-0.5, 0.5))
        nwave = [(float(rng.uniform(0.0, 0.04)), float(rng.uniform(0, 2 * math.pi)))]
        ndistance = _ellipse_distance(bx, by, ncx, ncy, na, nb, nangle, nwave)
        # A nucleus stays inside its cell, at least a pixel and a half from its outline.
        nuclei.append((box, np.maximum(ndistance, distance + 1.5 * scale), strength))

    # The nuclei come last, so that a touching neighbour never covers one.
    for box, ndistance, strength in nuclei:
        coverage = np.clip(0.5 - ndistance, 0.0, 1.0)
        glow = strength * float(rng.uniform(0.9, 1.4))
        colour = _colour(nucleus_hue + float(rng.uniform(-0.03, 0.03)), nucleus_saturation)
        # The cytoplasm stain fades where the nucleus is.
        signal[box] *= 1.0 - 0.7 * coverage[..., None]
        signal[box] += (glow * coverage)[..., None] * colour
        mask[box][ndistance < 0] = 2

    # Chromatin and grain: fine texture in the stains, coarse unevenness over the image.
    texture = 1.0 + 0.18 * _smooth_noise(rng, size, 24) + 0.08 * _smooth_noise(rng, size, 6)
    signal *= np.clip(texture, 0.3, None)[..., None]

    # The background: dark and slightly tinted, unevenly lit.
    tint = _colour(float(rng.uniform(0, 1)), float(rng.uniform(0.2, 0.7)))
    background = float(rng.uniform(0.03, 0.1)) * tint + float(rng.uniform(0.01, 0.04))
    unevenly = 1.0 + 0.25 * _smooth_noise(rng, size, 3)
    image = gain * signal + background * unevenly[..., None]

    # A light blur (the optics), vignetting, then the noise of the sensor (it grows with the
    # light), and a little gamma to vary the contrast.
    image = _gaussian_blur(image, float(rng.uniform(0.6, 1.0)) * scale)
    ox, oy = rng.uniform(-0.15, 0.15, size=2) * size
    r2 = ((xx - size / 2 - ox) ** 2 + (yy - size / 2 - oy) ** 2) / (size / 2) ** 2
    image *= (1.0 - float(rng.uniform(0.15, 0.4)) * np.clip(r2 / 2, 0.0, 1.0))[..., None]
    noise = float(rng.uniform(0.008, 0.025))
    spread = np.sqrt(noise**2 + 0.012 * np.clip(image, 0.0, None))
    image += rng.standard_normal(size=image.shape, dtype=np.float32) * spread
    levels = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    gamma = float(rng.uniform(0.8, 1.2))
    curve = np.rint(255.0 * (np.arange(256) / 255.0) ** gamma).astype(np.uint8)
    return curve[levels], mask


def cells_arrays(start: int, count: int, size: int = CELLS_SIZE) -> tuple[Any, Any]:
    """Images ``start`` … ``start + count - 1`` as uint8 arrays: pictures (count × 3 × size
    × size, channels first) and masks (count × size × size)."""
    np = _numpy()
    images = np.empty((count, 3, size, size), dtype=np.uint8)
    masks = np.empty((count, size, size), dtype=np.uint8)
    for i in range(count):
        picture, mask = cells_sample(start + i, size)
        images[i] = picture.transpose(2, 0, 1)
        masks[i] = mask
    return images, masks
