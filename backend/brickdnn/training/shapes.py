"""The built-in segmentation dataset "Shapes": generated, nothing is downloaded (needs NumPy).

Every image shows one to three shapes (circles, squares, triangles) of random size and
rotation on a noisy greyish background with a gradient; the mask says for every pixel
whether it is background (0), circle (1), square (2) or triangle (3). Later shapes cover
earlier ones.

The colour of a shape hints at its class: circles are reddish, triangles greenish, squares
bluish. The hues of the classes overlap (an orange shape can be a circle or a triangle), so
a network gets most pixels right after a few epochs by colour alone and needs the outlines
for the rest. That makes it a quick first dataset on a CPU.

Image ``index`` only depends on ``index`` (a seed of its own), so the dataset is the same
on every machine and for every run, and any part of it can be generated on its own.
"""

from __future__ import annotations

import colorsys
import math
from typing import Any

SHAPE_CLASSES = ("background", "circle", "square", "triangle")
SHAPES_SIZE = 64  # pixels (square images)
SHAPES_SEED = 20_240_611
# Shapes may overlap, but a shape whose centre is closer than this to an earlier one (in
# units of their radii) is placed again, so that no shape hides another completely.
_MIN_DISTANCE = 0.9
# Hue (0-1) of each class and how far a shape may be off it. Neighbouring classes are 0.27
# to 0.43 apart, so with this spread some hues fit two classes.
_HUES = {1: 0.03, 2: 0.6, 3: 0.3}
_HUE_SPREAD = 0.18
# A shape differs at least this much from the background (largest difference of a channel).
_MIN_CONTRAST = 0.22


def _numpy() -> Any:
    import numpy

    return numpy


def _polygon_distance(xx: Any, yy: Any, points: Any) -> Any:
    """Signed distance (negative inside) to the edges of a convex polygon whose corners are
    given in the order of increasing angle."""
    np = _numpy()
    distance = np.full(xx.shape, -np.inf, dtype=np.float32)
    count = len(points)
    for i in range(count):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % count]
        ex, ey = x1 - x0, y1 - y0
        length = math.hypot(ex, ey)
        nx, ny = ey / length, -ex / length  # points away from the centre
        distance = np.maximum(distance, (xx - x0) * nx + (yy - y0) * ny)
    return distance


def _regular_polygon(cx: float, cy: float, radius: float, corners: int, angle: float) -> Any:
    np = _numpy()
    angles = angle + np.arange(corners) * (2 * math.pi / corners)
    return np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1)


def _smooth_noise(rng: Any, size: int) -> Any:
    """Soft blotches: a 4 × 4 grid of random values, interpolated to size × size."""
    np = _numpy()
    values = rng.normal(0.0, 1.0, size=(4, 4))
    grid = np.linspace(0, 3, size)
    low = np.clip(np.floor(grid).astype(int), 0, 2)
    t = grid - low
    rows = values[low] * (1 - t[:, None]) + values[low + 1] * t[:, None]
    return rows[:, low] * (1 - t[None, :]) + rows[:, low + 1] * t[None, :]


def shapes_sample(index: int, size: int = SHAPES_SIZE) -> tuple[Any, Any]:
    """Image ``index`` as uint8 arrays: the picture (size × size × 3, RGB) and its mask
    (size × size, class indices, see SHAPE_CLASSES)."""
    np = _numpy()
    rng = np.random.default_rng([SHAPES_SEED, int(index)])
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) + 0.5

    # Background: greyish with a tint, a gradient across the image and soft blotches.
    base = float(rng.uniform(0.35, 0.8)) + rng.uniform(-0.15, 0.15, size=3)
    direction = float(rng.uniform(0, 2 * math.pi))
    ramp = ((xx - size / 2) * math.cos(direction) + (yy - size / 2) * math.sin(direction)) / size
    image = base + ramp[..., None] * (
        float(rng.uniform(-0.25, 0.25)) + rng.uniform(-0.08, 0.08, size=3)
    )
    image = image + 0.06 * _smooth_noise(rng, size)[..., None]

    mask = np.zeros((size, size), dtype=np.uint8)
    placed: list[tuple[float, float, float]] = []
    for _ in range(int(rng.integers(1, 4))):
        for _attempt in range(20):
            radius = float(rng.uniform(0.13, 0.24)) * size
            cx = float(rng.uniform(radius * 0.8, size - radius * 0.8))
            cy = float(rng.uniform(radius * 0.8, size - radius * 0.8))
            if all(
                math.hypot(cx - px, cy - py) > _MIN_DISTANCE * (radius + pr)
                for px, py, pr in placed
            ):
                break
        placed.append((cx, cy, radius))
        kind = int(rng.integers(1, 4))
        angle = float(rng.uniform(0, 2 * math.pi))
        if kind == 1:
            distance = np.hypot(xx - cx, yy - cy) - radius * 0.9
        elif kind == 2:
            distance = _polygon_distance(xx, yy, _regular_polygon(cx, cy, radius, 4, angle))
        else:
            distance = _polygon_distance(xx, yy, _regular_polygon(cx, cy, radius * 1.15, 3, angle))
        for _attempt in range(20):
            hue = (_HUES[kind] + float(rng.uniform(-_HUE_SPREAD, _HUE_SPREAD))) % 1.0
            colour = np.array(
                colorsys.hsv_to_rgb(
                    hue, float(rng.uniform(0.65, 1.0)), float(rng.uniform(0.55, 1.0))
                )
            )
            if float(np.abs(colour - base).max()) >= _MIN_CONTRAST:
                break
        coverage = np.clip(0.5 - distance, 0.0, 1.0)[..., None]  # a soft edge, one pixel wide
        shade = 1.0 + 0.12 * ((xx - cx) + (yy - cy))[..., None] / (2 * radius)
        image = image * (1 - coverage) + colour * shade * coverage
        mask[distance < 0] = kind

    image = image + rng.normal(0.0, 0.035, size=image.shape)
    pixels = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    return pixels, mask


def shapes_arrays(start: int, count: int, size: int = SHAPES_SIZE) -> tuple[Any, Any]:
    """Images ``start`` … ``start + count - 1`` as uint8 arrays: pictures (count × 3 × size
    × size, channels first) and masks (count × size × size)."""
    np = _numpy()
    images = np.empty((count, 3, size, size), dtype=np.uint8)
    masks = np.empty((count, size, size), dtype=np.uint8)
    for i in range(count):
        picture, mask = shapes_sample(start + i, size)
        images[i] = picture.transpose(2, 0, 1)
        masks[i] = mask
    return images, masks
