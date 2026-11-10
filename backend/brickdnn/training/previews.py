"""Preview images of a segmentation run: what the trained network predicts for a few
validation images (needs PyTorch and Pillow).

They are written into a folder next to the weights (``previews/`` in a run directory)::

    index.json            what there is, see :func:`save_previews`
    0_image.png           the image as the network got it (resized to the Input)
    0_truth.png           its mask: one colour per class
    0_prediction.png      the mask the network predicts
    0_overlay.png         the prediction drawn over the image
    1_image.png …

The number of samples and the size of the images are limited, so the folder stays small (at
most about 2 MB, usually far less): it counts as storage of the run.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch

from .segmentation import IGNORE_COLOR, IGNORE_INDEX, class_colors, rgb
from .segmetrics import confusion, confusion_metrics
from .state import PREVIEW_DIR, PREVIEW_INDEX_FILE, write_json_atomic

__all__ = ["PREVIEW_DIR", "PREVIEW_SAMPLES", "PREVIEW_SIDE", "save_previews"]
INDEX_FILE = PREVIEW_INDEX_FILE
# How many validation images are shown, and their longest side in pixels (larger ones are
# scaled down).
PREVIEW_SAMPLES = 8
PREVIEW_SIDE = 192
# How strongly the colour of the predicted class covers the image in the overlay.
OVERLAY_ALPHA = 0.55
PARTS = ("image", "truth", "prediction", "overlay")


def _palette(num_classes: int) -> list[int]:
    colours = [rgb(c) for c in class_colors(num_classes)]
    colours += [(0, 0, 0)] * (256 - len(colours))
    colours[IGNORE_INDEX] = rgb(IGNORE_COLOR)
    return [channel for colour in colours for channel in colour]


def save_previews(
    directory: str | Path,
    images: torch.Tensor,
    truth: torch.Tensor,
    predicted: torch.Tensor,
    class_names: list[str],
    epoch: int | None,
) -> dict[str, Any] | None:
    """Write the preview images of some samples and their index; returns the index, or None
    if Pillow is missing.

    ``images``: uint8 (N, C, H, W) with 1 or 3 channels, ``truth`` and ``predicted``: class
    maps (N, H, W), ``truth`` with IGNORE_INDEX for ignored pixels. The index is::

        {"epoch": 7, "split": "validation", "width": 64, "height": 64,
         "input_size": [64, 64], "class_names": [...], "colors": ["#20242c", ...],
         "ignore_color": "#8a8f98",
         "samples": [{"index": 0, "image": "0_image.png", "truth": "0_truth.png",
                      "prediction": "0_prediction.png", "overlay": "0_overlay.png",
                      "iou": 0.81, "pixel_accuracy": 0.97, "classes": [0, 2]}]}

    ``iou`` is the mean IoU of the sample over the classes in its mask or prediction,
    ``classes`` are those classes.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    directory = Path(directory)
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)
    count = min(len(images), PREVIEW_SAMPLES)
    num_classes = len(class_names)
    height, width = int(images.shape[-2]), int(images.shape[-1])
    scale = min(1.0, PREVIEW_SIDE / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    palette = _palette(num_classes)
    colours = torch.tensor([rgb(c) for c in class_colors(num_classes)], dtype=torch.float32)
    samples = []
    for i in range(count):
        pixels = images[i].permute(1, 2, 0).contiguous()
        if pixels.shape[-1] == 1:
            pixels = pixels.expand(-1, -1, 3)
        picture = Image.fromarray(pixels.numpy().copy())
        true_map = truth[i].to(torch.uint8)
        predicted_map = predicted[i].to(torch.uint8)
        # The prediction over the image: the background (class 0) stays as it is.
        tint = colours[predicted_map.long().clamp(max=num_classes - 1)]
        alpha = (predicted_map > 0).float().unsqueeze(-1) * OVERLAY_ALPHA
        blended = (pixels.float() * (1 - alpha) + tint * alpha).round().clamp(0, 255)
        overlay = Image.fromarray(blended.to(torch.uint8).numpy())
        maps = {}
        for name, classes in (("truth", true_map), ("prediction", predicted_map)):
            mask = Image.frombytes("P", (width, height), classes.contiguous().numpy().tobytes())
            mask.putpalette(palette)
            maps[name] = mask
        if scale < 1.0:
            picture = picture.resize(size, Image.BILINEAR)
            overlay = overlay.resize(size, Image.BILINEAR)
            maps = {name: mask.resize(size, Image.NEAREST) for name, mask in maps.items()}
        files = {part: f"{i}_{part}.png" for part in PARTS}
        picture.save(directory / files["image"], optimize=True)
        overlay.save(directory / files["overlay"], optimize=True)
        for name, mask in maps.items():
            mask.save(directory / files[name], optimize=True)
        metrics = confusion_metrics(confusion(true_map, predicted_map, num_classes))
        samples.append(
            {
                "index": i,
                **files,
                "iou": metrics["iou"],
                "pixel_accuracy": metrics["pixel_accuracy"],
                "classes": [c for c, v in enumerate(metrics["class_iou"]) if v is not None],
            }
        )
    index = {
        "epoch": epoch,
        "split": "validation",
        "width": size[0],
        "height": size[1],
        "input_size": [height, width],
        "class_names": list(class_names),
        "colors": class_colors(num_classes),
        "ignore_color": IGNORE_COLOR,
        "samples": samples,
    }
    write_json_atomic(directory / INDEX_FILE, index)
    return index
