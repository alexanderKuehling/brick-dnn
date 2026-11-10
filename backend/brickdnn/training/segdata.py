"""Data of a segmentation training: images and one mask per image (needs PyTorch).

Like the other sources (see data.py), a segmentation source returns ready-to-use batches:
the images as normalised floats (N, C, H, W) and the masks as class indices (N, H, W,
``long``; ``IGNORE_INDEX`` for ignored pixels). Everything is stored as ``uint8`` at the size
of the Input, so a dataset needs (C + 1)·H·W bytes per image; datasets that would need more
than :data:`PRELOAD_BYTES` are read from disk batch by batch instead.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .compat import TrainingPlan
from .config import TrainingConfig
from .data import (
    BUILTIN_TEST_SHARE,
    Batch,
    ImageTransform,
    PreparedData,
    Subset,
    _image_preprocessing,
    _pil,
    load_image,
    predefined_split_indices,
    split_indices,
)
from .datasets import BUILTIN_DATASETS, DatasetInfo
from .reporting import Reporter, StopTraining, TrainingError
from .segmentation import IGNORE_INDEX, MaskDecoder, Pair, find_pairs
from .tabular import val_size_for

# Segmentation folders up to this size (uint8, at the size of the Input) are loaded into
# memory before training. A training process has about 1.8 GB for everything.
PRELOAD_BYTES = 256 * 1024**2


def resize_masks(masks: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Nearest-neighbour resize of class maps (N, H, W): no classes are mixed."""
    if tuple(masks.shape[-2:]) == (height, width):
        return masks
    resized = F.interpolate(masks.unsqueeze(1).float(), size=(height, width), mode="nearest-exact")
    return resized.squeeze(1).to(masks.dtype)


class MaskTensorSource:
    """Images and masks held in memory as uint8."""

    def __init__(self, images: torch.Tensor, masks: torch.Tensor, transform: ImageTransform):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.size = (transform.image["height"], transform.image["width"])

    def __len__(self) -> int:
        return len(self.masks)

    def batch(self, indices: torch.Tensor) -> Batch:
        masks = resize_masks(self.masks[indices], *self.size)
        return [self.transform(self.images[indices])], [masks.long()]


class MaskFileSource:
    """Images and masks read from disk for every batch (datasets that do not fit into
    memory)."""

    def __init__(
        self,
        pairs: list[Pair],
        root: Path,
        image: dict[str, Any],
        decoder: MaskDecoder,
        transform: ImageTransform,
    ):
        self.pairs = pairs
        self.root = root
        self.image = image
        self.decoder = decoder
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def batch(self, indices: torch.Tensor) -> Batch:
        loaded = [
            load_pair(self.pairs[i], self.image, self.decoder, self.root) for i in indices.tolist()
        ]
        images = torch.stack([image for image, _ in loaded])
        masks = torch.stack([mask for _, mask in loaded])
        return [self.transform(images)], [masks.long()]


def load_mask(pair: Pair, image: dict[str, Any], decoder: MaskDecoder) -> torch.Tensor:
    """The class map of one mask file as uint8 (H, W) at the planned size."""
    Image = _pil()
    height, width = image["height"], image["width"]
    try:
        classes = decoder.decode(pair.mask)
    except ValueError as exc:  # DatasetError
        raise TrainingError(str(exc)) from exc
    if classes.shape != (height, width):
        resized = Image.fromarray(classes).resize((width, height), Image.NEAREST)
        return torch.frombuffer(bytearray(resized.tobytes()), dtype=torch.uint8).view(height, width)
    return torch.from_numpy(classes.copy())


def load_pair(
    pair: Pair, image: dict[str, Any], decoder: MaskDecoder, root: Path
) -> tuple[torch.Tensor, torch.Tensor]:
    """One image (uint8 C×H×W, converted and resized) and its class map (uint8 H×W)."""
    return load_image(pair.image, image, root), load_mask(pair, image, decoder)


def _preprocessing(
    plan: TrainingPlan, mean: list[float], std: list[float], ignore: bool
) -> dict[str, Any]:
    assert plan.image is not None
    preprocessing = _image_preprocessing(plan.image, mean, std, list(plan.class_names or []))
    preprocessing["task"] = "segmentation"
    preprocessing["mask"] = {
        "resize": "nearest",
        "binary": bool((plan.segmentation or {}).get("binary")),
        "ignore_index": IGNORE_INDEX if ignore else None,
    }
    return preprocessing


def segmentation_data(
    plan: TrainingPlan,
    info: DatasetInfo,
    path: str | Path | None,
    config: TrainingConfig,
    reporter: Reporter,
    generator: torch.Generator,
) -> PreparedData:
    """The data of a segmentation plan: the built-in Shapes or a folder of images and masks."""
    if plan.image is None:
        raise TrainingError("The network does not take images.")
    if info.kind == "builtin":
        return shapes_data(plan, info, config, reporter, generator)
    if path is None:
        raise TrainingError("The dataset has no files. Upload it again.")
    return folder_data(plan, info, Path(path), config, reporter, generator)


def shapes_data(
    plan: TrainingPlan,
    info: DatasetInfo,
    config: TrainingConfig,
    reporter: Reporter,
    generator: torch.Generator,
) -> PreparedData:
    """Shapes: generated here. Validation data comes from its training images; its test
    images are the test set (like the official test split of the other built-in datasets)."""
    spec = BUILTIN_DATASETS.get(info.builtin or "")
    if spec is None or not spec.generated or plan.image is None:
        raise TrainingError(f"Unknown built-in dataset '{info.builtin}'.")
    reporter.status("preparing", f"Generating {spec.name} …")
    try:
        from .shapes import shapes_arrays

        pictures, classes = shapes_arrays(0, spec.train_size + spec.test_size, spec.shape[1])
    except ImportError as exc:
        raise TrainingError(f"{spec.name} needs NumPy. Install it with: pip install numpy") from exc
    images, masks = torch.from_numpy(pictures), torch.from_numpy(classes)
    train_rows, val_rows, _ = split_indices(
        spec.train_size, config.val_split, config.max_samples, generator
    )
    test_rows = torch.arange(spec.train_size, spec.train_size + spec.test_size)
    if config.max_samples is not None:
        n_test = val_size_for(config.max_samples, BUILTIN_TEST_SHARE)
        if n_test < len(test_rows):
            keep = torch.randperm(len(test_rows), generator=generator)[:n_test]
            test_rows = test_rows[keep.sort().values]
    mean, std = list(spec.mean), list(spec.std)
    if plan.image["channels"] != spec.shape[0]:
        mean, std = [0.5] * plan.image["channels"], [0.25] * plan.image["channels"]
    source = MaskTensorSource(images, masks, ImageTransform(plan.image, mean, std))
    preprocessing = _preprocessing(plan, mean, std, ignore=False)
    preprocessing["dataset"] = spec.id
    return PreparedData(
        Subset(source, train_rows),
        Subset(source, val_rows),
        preprocessing,
        test=Subset(source, test_rows),
    )


def folder_data(
    plan: TrainingPlan,
    info: DatasetInfo,
    path: Path,
    config: TrainingConfig,
    reporter: Reporter,
    generator: torch.Generator,
) -> PreparedData:
    image = plan.image
    assert image is not None
    if not path.is_dir():
        raise TrainingError("The folder of this dataset was not found. Upload it again.")
    reporter.status("preparing", "Looking for images and masks …")
    changed = TrainingError(
        "The dataset has changed since it was checked. Upload or select it again."
    )
    try:
        pairs = find_pairs(path).pairs
        decoder = MaskDecoder(info.segmentation or {}, len(info.class_names or []), path)
    except ValueError as exc:  # DatasetError
        raise TrainingError(str(exc)) from exc
    if len(pairs) != (info.num_samples or 0):
        raise changed
    if info.predefined_split is not None:
        split = torch.tensor([-1 if p.split is None else p.split for p in pairs])
        if [int((split == label).sum()) for label in range(3)] != list(info.predefined_split):
            raise changed
        train, val, test = predefined_split_indices(split, config.max_samples, generator)
    else:
        train, val, test = split_indices(
            len(pairs), config.val_split, config.max_samples, generator, config.test_split
        )
    channels = image["channels"]
    mean, std = [0.5] * channels, [0.5] * channels
    transform = ImageTransform(image, mean, std)
    ignore = (info.segmentation or {}).get("ignore_index") is not None
    preprocessing = _preprocessing(plan, mean, std, ignore)
    if info.predefined_split is not None:
        sizes = [len(train), len(val), len(test) if test is not None else 0]
        preprocessing["split"] = {"source": "folders", "counts": sizes}

    parts = [train, val] if test is None else [train, val, test]
    used = torch.cat(parts)
    height, width = image["height"], image["width"]
    if len(used) * (channels + 1) * height * width > PRELOAD_BYTES:
        reporter.log(
            "The images do not fit into memory at this Input size and are read from disk "
            "during training."
        )
        source = MaskFileSource(pairs, path, image, decoder, transform)
        return PreparedData(
            Subset(source, train),
            Subset(source, val),
            preprocessing,
            test=Subset(source, test) if test is not None else None,
        )

    images = torch.empty((len(used), channels, height, width), dtype=torch.uint8)
    masks = torch.empty((len(used), height, width), dtype=torch.uint8)
    total = len(used)
    last = 0.0
    for i, index in enumerate(used.tolist()):
        now = time.monotonic()
        if now - last >= 0.5:
            last = now
            if reporter.should_stop():
                raise StopTraining
            reporter.status("preparing", f"Loading images {i:,} / {total:,}")
        images[i], masks[i] = load_pair(pairs[index], image, decoder, path)
    reporter.status("preparing", f"Loading images {total:,} / {total:,}")
    sources = []
    start = 0
    for rows in parts:
        stop = start + len(rows)
        sources.append(MaskTensorSource(images[start:stop], masks[start:stop], transform))
        start = stop
    return PreparedData(
        sources[0], sources[1], preprocessing, test=sources[2] if test is not None else None
    )
