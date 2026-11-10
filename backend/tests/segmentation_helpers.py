"""Tiny segmentation datasets and networks for the tests (needs Pillow and NumPy)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .helpers import chain

SIZE = 16


def tiny_unet(
    classes: int = 4, size: int = SIZE, channels: int = 3, width: int = 8
) -> dict[str, Any]:
    """A U-Net with one level: conv, pool, conv, upsample, skip connection, conv, 1×1 conv."""
    nodes = [
        ("input", "Input", {"shape": [channels, size, size]}),
        ("enc", "Conv2d", {"out_channels": width}),
        ("enc_relu", "Activation", {"function": "relu"}),
        ("pool", "MaxPool2d", {}),
        ("mid", "Conv2d", {"out_channels": 2 * width}),
        ("mid_relu", "Activation", {"function": "relu"}),
        ("up", "ConvTranspose2d", {"out_channels": width}),
        ("skip", "Concat", {"dim": 0}),
        ("dec", "Conv2d", {"out_channels": width}),
        ("dec_relu", "Activation", {"function": "relu"}),
        ("mask", "Conv2d", {"out_channels": classes, "kernel_size": [1, 1]}),
    ]
    edges = [
        ("input", "enc"),
        ("enc", "enc_relu"),
        ("enc_relu", "pool"),
        ("pool", "mid"),
        ("mid", "mid_relu"),
        ("mid_relu", "up"),
        ("up", "skip"),
        ("enc_relu", "skip"),
        ("skip", "dec"),
        ("dec", "dec_relu"),
        ("dec_relu", "mask"),
    ]
    return {
        "version": 1,
        "nodes": [{"id": i, "type": t, "params": p} for i, t, p in nodes],
        "edges": [{"source": s, "target": t} for s, t in edges],
    }


def flat_net(classes: int = 2, size: int = SIZE, channels: int = 3) -> dict[str, Any]:
    """The smallest segmentation network: two convolutions that keep the image size."""
    return chain(
        ("Input", {"shape": [channels, size, size]}),
        ("Conv2d", {"out_channels": 8}),
        ("Activation", {"function": "relu"}),
        ("Conv2d", {"out_channels": classes, "kernel_size": [1, 1]}),
    )


def sample(index: int, size: int = SIZE, classes: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """An easy picture: a grey background (class 0) with one bright square per further class,
    red for class 1 and blue for class 2; returns (RGB picture, class map)."""
    rng = np.random.default_rng(index)
    picture = np.full((size, size, 3), 90, dtype=np.uint8)
    picture += rng.integers(0, 20, size=picture.shape, dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    colours = {1: (230, 40, 40), 2: (40, 60, 230), 3: (40, 200, 60)}
    half = size // 2
    for label in range(1, classes):
        side = int(rng.integers(size // 4, half))
        x = int(rng.integers(0, half - side + 1)) + (half if label % 2 == 0 else 0)
        y = int(rng.integers(0, size - side + 1))
        picture[y : y + side, x : x + side] = colours[label]
        mask[y : y + side, x : x + side] = label
    return picture, mask


def png(array: np.ndarray, mode: str | None = None) -> bytes:
    buffer = io.BytesIO()
    image = Image.fromarray(array)
    if mode is not None:
        image = image.convert(mode)
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def encode_mask(mask: np.ndarray, kind: str = "indices") -> bytes:
    """A class map as a mask file: ``indices`` (grey values 0, 1, 2 …), ``binary`` (0 / 255),
    ``palette`` (a palette PNG), ``levels`` (0, 128, 255 …), ``colors`` (one colour per
    class) or ``jpeg`` (a lossy file with in-between values)."""
    if kind == "indices":
        return png(mask)
    if kind == "binary":
        return png(np.where(mask > 0, 255, 0).astype(np.uint8))
    if kind == "palette":
        image = Image.frombytes("P", (mask.shape[1], mask.shape[0]), mask.tobytes())
        image.putpalette([0, 0, 0, 255, 0, 0, 0, 0, 255, 0, 255, 0] + [0] * (252 * 3))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    if kind == "levels":
        return png((mask.astype(np.uint16) * 127).clip(0, 255).astype(np.uint8))
    if kind == "colors":
        colours = np.array([[0, 0, 0], [255, 0, 0], [0, 0, 255], [0, 255, 0]], dtype=np.uint8)
        return png(colours[mask])
    if kind == "jpeg":
        buffer = io.BytesIO()
        smooth = Image.fromarray((mask * 80).astype(np.uint8)).resize(
            (mask.shape[1] * 4, mask.shape[0] * 4), Image.BILINEAR
        )
        smooth.save(buffer, format="JPEG", quality=60)
        return buffer.getvalue()
    raise ValueError(kind)


def dataset_files(
    count: int = 12,
    *,
    size: int = SIZE,
    classes: int = 3,
    masks: str = "indices",
    layout: str = "flat",
    prefix: str = "",
    mask_suffix: str = "",
    images: str = "images",
    mask_dir: str = "masks",
) -> dict[str, bytes]:
    """The files of a segmentation dataset as ``{path: content}``. ``layout``: ``flat``
    (images/ and masks/), ``split`` (train/images, train/masks, val/…, test/…) or
    ``split_inside`` (images/train, masks/train, …)."""
    files: dict[str, bytes] = {}
    for index in range(count):
        picture, mask = sample(index, size, classes)
        name = f"{index:03d}"
        if layout == "flat":
            image_path, mask_path = f"{images}/{name}", f"{mask_dir}/{name}"
        else:
            part = "train" if index % 4 < 2 else "val" if index % 4 == 2 else "test"
            if layout == "split":
                image_path, mask_path = f"{part}/{images}/{name}", f"{part}/{mask_dir}/{name}"
            else:
                image_path, mask_path = f"{images}/{part}/{name}", f"{mask_dir}/{part}/{name}"
        files[f"{prefix}{image_path}.png"] = png(picture)
        ending = ".jpg" if masks == "jpeg" else ".png"
        files[f"{prefix}{mask_path}{mask_suffix}{ending}"] = encode_mask(mask, masks)
    return files


def write_files(root: Path, files: dict[str, bytes]) -> Path:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


def zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()
