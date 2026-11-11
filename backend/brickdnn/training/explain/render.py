"""Small pictures and numbers for explanations (NumPy only, no Pillow needed).

Everything an explanation shows as a picture (an input image, a heatmap, feature maps,
filters, the background of a decision boundary) is turned into a PNG data URL here, and
large arrays into short lists of rounded numbers.
"""

from __future__ import annotations

import base64
import math
import struct
import zlib
from typing import Any

import numpy as np

# Heatmaps: dark purple -> red -> yellow (like "inferno"), readable in light and dark mode.
HEAT_STOPS = np.array(
    [
        [0, 0, 4],
        [40, 11, 84],
        [101, 21, 110],
        [159, 42, 99],
        [212, 72, 66],
        [245, 125, 21],
        [250, 193, 39],
        [252, 255, 164],
    ],
    dtype=np.float64,
)
# Signed values (weights, attributions): blue -> white -> red.
DIVERGING_STOPS = np.array(
    [[33, 102, 172], [103, 169, 207], [247, 247, 247], [239, 138, 98], [178, 24, 43]],
    dtype=np.float64,
)
# Classes of a decision boundary (and the points on it), the same in the UI legend.
CLASS_COLORS = [
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#b279a2",
    "#eeca3b",
    "#9d755d",
    "#ff9da6",
    "#bab0ac",
]


def png(pixels: np.ndarray) -> bytes:
    """Encode ``pixels`` (H×W gray, H×W×3 RGB or H×W×4 RGBA, uint8) as a PNG file."""
    pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
    if pixels.ndim == 2:
        pixels = pixels[:, :, None]
    height, width, channels = pixels.shape
    color_type = {1: 0, 3: 2, 4: 6}[channels]
    rows = np.concatenate(
        [np.zeros((height, 1), dtype=np.uint8), pixels.reshape(height, width * channels)], axis=1
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows.tobytes(), 9))
        + chunk(b"IEND", b"")
    )


def data_url(pixels: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(png(pixels)).decode("ascii")


def _interpolate(values: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Values in [0, 1] -> RGB floats along the colour stops."""
    position = np.clip(values, 0.0, 1.0) * (len(stops) - 1)
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, len(stops) - 1)
    share = (position - low)[..., None]
    return stops[low] * (1 - share) + stops[high] * share


def heat_rgba(values: np.ndarray, alpha: bool = True) -> np.ndarray:
    """A heatmap of ``values`` in [0, 1]: RGBA whose opacity grows with the value (to lay
    it over an image), or opaque RGB."""
    rgb = _interpolate(values, HEAT_STOPS)
    if not alpha:
        return rgb.round().astype(np.uint8)
    opacity = np.clip(values, 0.0, 1.0) ** 0.7 * 235 + 20
    return np.concatenate([rgb, opacity[..., None]], axis=-1).round().astype(np.uint8)


def diverging_rgb(values: np.ndarray) -> np.ndarray:
    """Values in [-1, 1] -> blue (negative), white (0), red (positive)."""
    return _interpolate((np.asarray(values) + 1) / 2, DIVERGING_STOPS).round().astype(np.uint8)


def hex_rgb(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def normalise(values: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Non-negative ``values`` scaled to [0, 1] by a high percentile (one hot pixel does
    not make everything else dark); all zero stays zero."""
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    top = float(np.percentile(values, percentile)) if values.size else 0.0
    if top <= 1e-12:
        top = float(values.max()) if values.size else 0.0
    if top <= 1e-12:
        return np.zeros_like(values)
    return np.clip(values / top, 0.0, 1.0)


def unit_to_uint8(image: np.ndarray) -> np.ndarray:
    """An image C×H×W with values in [0, 1] -> H×W (gray) or H×W×3 uint8."""
    pixels = (np.clip(image, 0.0, 1.0) * 255).round().astype(np.uint8)
    if pixels.shape[0] == 1:
        return pixels[0]
    return np.transpose(pixels[:3], (1, 2, 0))


def stretch(values: np.ndarray) -> np.ndarray:
    """Any 2-D array scaled to 0-255 (min -> black, max -> white)."""
    values = np.asarray(values, dtype=np.float64)
    low, high = float(values.min()), float(values.max())
    if high - low <= 1e-12:
        return np.full(values.shape, 128 if high != 0 else 0, dtype=np.uint8)
    return ((values - low) / (high - low) * 255).round().astype(np.uint8)


def shrink(pixels: np.ndarray, largest: int) -> np.ndarray:
    """Nearest-neighbour downsampling so that neither side is larger than ``largest``."""
    height, width = pixels.shape[:2]
    factor = max(height, width) / largest
    if factor <= 1:
        return pixels
    rows = (np.arange(max(1, round(height / factor))) * factor).astype(int)
    cols = (np.arange(max(1, round(width / factor))) * factor).astype(int)
    return pixels[rows][:, cols]


def resize_map(values: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinear resize of a 2-D map (e.g. a coarse Grad-CAM or occlusion grid)."""
    values = np.asarray(values, dtype=np.float64)
    h, w = values.shape
    if (h, w) == (height, width):
        return values
    ys = np.clip((np.arange(height) + 0.5) * h / height - 0.5, 0, h - 1)
    xs = np.clip((np.arange(width) + 0.5) * w / width - 0.5, 0, w - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    wy, wx = (ys - y0)[:, None], (xs - x0)[None, :]
    top = values[y0][:, x0] * (1 - wx) + values[y0][:, x1] * wx
    bottom = values[y1][:, x0] * (1 - wx) + values[y1][:, x1] * wx
    return top * (1 - wy) + bottom * wy


def grid(values: np.ndarray, largest: int = 14, digits: int = 2) -> list[list[float]]:
    """A coarse copy of a 2-D map (block means, at most ``largest`` per side) as rounded
    numbers: small enough for an AI assistant to read."""
    values = np.asarray(values, dtype=np.float64)
    height, width = values.shape
    rows = min(height, largest)
    cols = min(width, largest)
    row_edges = np.linspace(0, height, rows + 1).round().astype(int)
    col_edges = np.linspace(0, width, cols + 1).round().astype(int)
    out = [
        [
            round(float(values[r0:r1, c0:c1].mean()), digits)
            for c0, c1 in zip(col_edges[:-1], col_edges[1:], strict=True)
        ]
        for r0, r1 in zip(row_edges[:-1], row_edges[1:], strict=True)
    ]
    return out


def tiles(images: list[np.ndarray], columns: int, pad: int = 1, background: int = 0) -> np.ndarray:
    """Equal-sized tiles (H×W or H×W×3 uint8) in a grid with ``pad`` pixels between them."""
    if not images:
        return np.zeros((1, 1), dtype=np.uint8)
    first = images[0]
    height, width = first.shape[:2]
    rows = math.ceil(len(images) / columns)
    shape = (rows * (height + pad) - pad, columns * (width + pad) - pad, *first.shape[2:])
    sheet = np.full(shape, background, dtype=np.uint8)
    for index, image in enumerate(images):
        r, c = divmod(index, columns)
        top, left = r * (height + pad), c * (width + pad)
        sheet[top : top + height, left : left + width] = image
    return sheet


def histogram(values: np.ndarray, bins: int = 24) -> dict[str, Any]:
    """Counts of ``values`` in ``bins`` equal bins, with the edges and summary numbers."""
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"edges": [], "counts": [], "count": 0}
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-12:
        low, high = low - 0.5, high + 0.5
    counts, edges = np.histogram(values, bins=bins, range=(low, high))
    return {
        "edges": [sig(e) for e in edges],
        "counts": [int(c) for c in counts],
        "count": int(values.size),
        "mean": sig(values.mean()),
        "std": sig(values.std()),
        "min": sig(values.min()),
        "max": sig(values.max()),
    }


def sig(value: Any, digits: int = 4) -> float | None:
    """A number rounded to ``digits`` significant digits (None for NaN and infinity)."""
    value = float(value)
    if not math.isfinite(value):
        return None
    if value == 0:
        return 0.0
    return round(value, digits - 1 - int(math.floor(math.log10(abs(value)))))


def sig_list(values: Any, digits: int = 4) -> list[float | None]:
    return [sig(v, digits) for v in np.asarray(values, dtype=np.float64).ravel()]
