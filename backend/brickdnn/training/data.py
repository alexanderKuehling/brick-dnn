"""Turn a dataset into tensors the way the :class:`TrainingPlan` says.

Every dataset becomes two *sources* (training and validation). A source knows its length
and returns ready-to-use batches for a tensor of sample indices; images are kept as
``uint8`` and only converted to normalised floats batch by batch, which keeps memory low
(CIFAR-10 as float32 would need about 600 MB).
"""

from __future__ import annotations

import math
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn.functional as F

from .compat import TrainingPlan
from .config import TrainingConfig
from .datasets import (
    BUILTIN_DATASETS,
    BuiltinDataset,
    DatasetError,
    DatasetInfo,
    list_image_files,
    open_csv,
    parse_number,
)
from .reporting import Reporter, StopTraining, TrainingError, short_error

Batch = tuple[list[torch.Tensor], list[torch.Tensor]]

# Demo data: this many samples in total, split into training and validation, but fewer
# for large inputs so that all demo tensors together stay below DEMO_BYTES.
DEMO_SAMPLES = 2560
DEMO_BYTES = 500 * 1024**2
# Token inputs of demo data only use the first ids of the vocabulary, so that every id
# is seen often enough to be learned within a few epochs.
DEMO_VOCAB = 256
# Image datasets up to this size (uint8) are loaded into memory before training.
PRELOAD_BYTES = 2 * 1024**3
# Mean/std of converted built-in images are estimated from at most this many images,
# converted in chunks of about this size (float32).
STATS_SAMPLES = 10_000
STATS_CHUNK_BYTES = 64 * 1024**2
# Built-in datasets ignore val_split (they have an official test split). With max_samples,
# the test split is shrunk as if val_split had its default value (20 % of all samples).
BUILTIN_VAL_SPLIT = 0.2
# CSV files are parsed in blocks of this many rows.
CSV_BLOCK_ROWS = 4096
# Rows per chunk when computing statistics of tabular data.
STATS_CHUNK_ROWS = 65_536
LUMINANCE = (0.299, 0.587, 0.114)


class Source(Protocol):
    def __len__(self) -> int: ...

    def batch(self, indices: torch.Tensor) -> Batch: ...


@dataclass
class PreparedData:
    train: Source
    val: Source
    preprocessing: dict[str, Any]
    # Regression on CSV data: standard deviation of the target, to report the
    # validation RMSE in the units of the CSV file.
    target_std: float | None = None


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #


class TensorSource:
    """Samples held in memory. ``transforms`` convert stored inputs (e.g. uint8 images)."""

    def __init__(
        self,
        inputs: list[torch.Tensor],
        targets: list[torch.Tensor],
        transforms: list[Callable[[torch.Tensor], torch.Tensor] | None] | None = None,
    ):
        self.inputs = inputs
        self.targets = targets
        self.transforms = transforms or [None] * len(inputs)

    def __len__(self) -> int:
        return len(self.targets[0])

    def batch(self, indices: torch.Tensor) -> Batch:
        inputs = [
            t(x[indices]) if t is not None else x[indices]
            for x, t in zip(self.inputs, self.transforms, strict=True)
        ]
        return inputs, [y[indices] for y in self.targets]


class Subset:
    """A view of some samples of another source (no copy)."""

    def __init__(self, source: Source, indices: torch.Tensor):
        self.source = source
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def batch(self, indices: torch.Tensor) -> Batch:
        return self.source.batch(self.indices[indices])


class ImageFileSource:
    """Images read from disk for every batch (datasets that do not fit into memory)."""

    def __init__(
        self,
        files: list[Path],
        labels: torch.Tensor,
        root: Path,
        image: dict[str, Any],
        transform: Callable[[torch.Tensor], torch.Tensor],
    ):
        self.files = files
        self.labels = labels
        self.root = root
        self.image = image
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def batch(self, indices: torch.Tensor) -> Batch:
        images = torch.stack(
            [load_image(self.files[i], self.image, self.root) for i in indices.tolist()]
        )
        return [self.transform(images)], [self.labels[indices]]


def iterate_batches(
    source: Source,
    batch_size: int,
    *,
    shuffle: bool = False,
    drop_last: bool = False,
    generator: torch.Generator | None = None,
):
    n = len(source)
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for indices in order.split(batch_size):
        if drop_last and len(indices) < batch_size:
            break
        yield source.batch(indices)


def val_size_for(max_samples: int, val_split: float) -> int:
    """Validation samples that go with ``max_samples`` training samples (same proportion)."""
    return max(1, round(max_samples * val_split / (1 - val_split)))


def split_indices(
    n: int, val_split: float, max_samples: int | None, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random (seeded) training/validation split.

    ``max_samples`` caps the training part and shrinks the validation part in proportion,
    so a quick experiment on a large dataset does not validate on all of it.
    """
    if n < 2:
        raise TrainingError("The dataset needs at least 2 samples (training + validation).")
    order = torch.randperm(n, generator=generator)
    n_val = min(max(1, round(n * val_split)), n - 1)
    train, val = order[n_val:], order[:n_val]
    if max_samples is not None:
        train = train[:max_samples]
        val = val[: val_size_for(max_samples, val_split)]
    return train, val


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #


def convert_channels(x: torch.Tensor, channels: int) -> torch.Tensor:
    """(N, C, H, W) floats: RGB -> grayscale (luminance) or grayscale -> RGB."""
    have = x.shape[1]
    if have == channels:
        return x
    if channels == 1 and have == 3:
        weights = torch.tensor(LUMINANCE, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        return (x * weights).sum(dim=1, keepdim=True)
    if channels == 1:
        return x.mean(dim=1, keepdim=True)
    if have == 1:
        return x.expand(-1, channels, -1, -1)
    raise TrainingError(f"Cannot convert images with {have} channels to {channels}.")


def resize(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Bilinear resize of (N, C, H, W) floats, antialiased when shrinking."""
    h, w = x.shape[-2:]
    if (h, w) == (height, width):
        return x
    shrink = height < h or width < w
    return F.interpolate(
        x, size=(height, width), mode="bilinear", align_corners=False, antialias=shrink
    )


def apply_layout(x: torch.Tensor, layout: str) -> torch.Tensor:
    """``chw`` keeps (N, C, H, W), ``hw`` gives (N, H, W), ``flat`` gives (N, C·H·W)."""
    if layout == "hw":
        return x[:, 0]
    if layout == "flat":
        return x.flatten(1)
    return x


class ImageTransform:
    """uint8 images (N, C, H, W) of any size -> normalised floats as the network expects."""

    def __init__(self, image: dict[str, Any], mean: list[float], std: list[float]):
        self.image = image
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)

    def to_unit(self, images: torch.Tensor) -> torch.Tensor:
        """Converted and resized, values in [0, 1] (not normalised)."""
        x = images.float().div_(255.0)
        x = convert_channels(x, self.image["channels"])
        return resize(x, self.image["height"], self.image["width"])

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        x = (self.to_unit(images) - self.mean) / self.std
        return apply_layout(x, self.image["layout"])


def _pil():
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow comes with torchvision
        raise TrainingError(
            "Reading images needs Pillow. Install it with: pip install pillow"
        ) from None
    return Image


def load_image(path: Path, image: dict[str, Any], root: Path) -> torch.Tensor:
    """Read one image file as uint8 (C, H, W) with the planned channels and size."""
    Image = _pil()
    channels, height, width = image["channels"], image["height"], image["width"]
    try:
        with Image.open(path) as img:
            img = img.convert("L" if channels == 1 else "RGB")
            if img.size != (width, height):
                img = img.resize((width, height), Image.BILINEAR)
            data = bytearray(img.tobytes())
    except Exception as exc:
        try:
            name = path.relative_to(root).as_posix()
        except ValueError:
            name = path.name
        raise TrainingError(
            f"Cannot read the image '{name}'. Remove it from the dataset or replace it."
        ) from exc
    pixels = torch.frombuffer(data, dtype=torch.uint8).view(height, width, channels)
    return pixels.permute(2, 0, 1).contiguous()


def _image_preprocessing(
    image: dict[str, Any], mean: list[float], std: list[float], class_names: list[str]
) -> dict[str, Any]:
    return {
        "kind": "image",
        "channels": image["channels"],
        "height": image["height"],
        "width": image["width"],
        "layout": image["layout"],
        "resize": "bilinear",
        "scale": "pixel / 255",
        "mean": [round(m, 6) for m in mean],
        "std": [round(s, 6) for s in std],
        "class_names": class_names,
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def prepare_data(
    plan: TrainingPlan,
    info: DatasetInfo,
    path: str | Path | None,
    config: TrainingConfig,
    data_root: str | Path | None,
    reporter: Reporter,
    generator: torch.Generator,
) -> PreparedData:
    """Load (or generate) the data for ``plan``. Raises :class:`TrainingError`."""
    try:
        if info.kind == "random":
            return demo_data(plan, config, generator, reporter)
        if info.kind == "builtin":
            if data_root is None:
                raise TrainingError("No folder for downloaded datasets was configured.")
            return builtin_data(plan, info, Path(data_root), config, reporter, generator)
        if path is None:
            raise TrainingError("The dataset has no file. Upload it again.")
        if info.kind == "csv":
            return csv_data(plan, info, Path(path), config, reporter, generator)
        if info.kind == "images":
            return image_folder_data(plan, info, Path(path), config, reporter, generator)
    except DatasetError as exc:
        raise TrainingError(str(exc)) from exc
    raise TrainingError(f"Unknown dataset kind '{info.kind}'.")


# --------------------------------------------------------------------------- #
# Demo data
# --------------------------------------------------------------------------- #


def _pattern(shape: list[int], generator: torch.Generator) -> torch.Tensor:
    """A fixed pattern that survives pooling: for rank >= 2 it is the sum of a profile
    along the first axis (channels) and one along the last axis (features / width)."""
    if len(shape) <= 1:
        return torch.randn(shape, generator=generator)
    first = torch.randn(shape[0], generator=generator).view(-1, *[1] * (len(shape) - 1))
    last = torch.randn(shape[-1], generator=generator)
    return ((first + last) / math.sqrt(2)).expand(shape)


def demo_samples(
    plan: TrainingPlan, n: int, generator: torch.Generator, rule: torch.Generator
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Generate ``n`` samples that follow a hidden rule.

    Every input carries a hidden 2-D direction ``u`` (radius 0.5-1, random angle): float
    inputs are noise plus ``u`` times two fixed patterns, token inputs draw their ids from
    four groups with probabilities that depend on ``u``. The directions of all inputs are
    added up; a class target is the sector (one of K equal slices of the circle) that the
    sum points to, a value target a fixed linear function of it (unit variance).
    The ``rule`` generator fixes the patterns, ``generator`` draws the samples.
    """
    total = torch.zeros(n, 2)
    inputs = []
    for spec in plan.inputs:
        shape = list(spec["shape"])
        angle = torch.rand(n, generator=generator) * (2 * math.pi)
        radius = 0.5 + 0.5 * torch.rand(n, generator=generator)
        u = torch.stack([radius * angle.cos(), radius * angle.sin()], dim=1)
        total += u
        if spec["dtype"] == "long":
            vocab = int(spec.get("vocab") or 2)
            group_size = max(1, min(vocab, DEMO_VOCAB) // 4)
            probs = torch.stack([1 + u[:, 0], 1 - u[:, 0], 1 + u[:, 1], 1 - u[:, 1]], 1) / 4
            length = math.prod(shape)
            groups = torch.multinomial(probs, length, replacement=True, generator=generator)
            offsets = torch.randint(0, group_size, (n, length), generator=generator)
            tokens = (groups * group_size + offsets).clamp(max=vocab - 1)
            inputs.append(tokens.view(n, *shape))
        else:
            p1, p2 = _pattern(shape, rule), _pattern(shape, rule)
            x = torch.randn(n, *shape, generator=generator)
            view = (-1, *[1] * len(shape))
            step = max(1, 2**22 // max(1, math.prod(shape)))  # add the signal in chunks
            for start in range(0, n, step):
                part = u[start : start + step]
                signal = part[:, 0].view(view) * p1 + part[:, 1].view(view) * p2
                x[start : start + step] += 1.5 * signal
            inputs.append(x)

    # E[radius²] = 7/12, so each coordinate of one direction has variance 7/24.
    scale = math.sqrt(len(plan.inputs) * 7 / 24)
    targets = []
    for target in plan.targets:
        if target["kind"] == "class":
            k = int(target["num_classes"])
            offset = float(torch.rand(1, generator=rule)) * 2 * math.pi
            angle = (torch.atan2(total[:, 1], total[:, 0]) - offset) % (2 * math.pi)
            targets.append((angle / (2 * math.pi / k)).long().clamp(0, k - 1))
        else:
            shape = list(target["shape"])
            weights = torch.randn(math.prod(shape), 2, generator=rule)
            weights = weights / weights.norm(dim=1, keepdim=True)
            targets.append((total @ weights.T / scale).view(n, *shape))
    return inputs, targets


def _demo_sample_bytes(plan: TrainingPlan) -> int:
    size = 0
    for spec in plan.inputs:
        size += math.prod(spec["shape"]) * (8 if spec["dtype"] == "long" else 4)
    for target in plan.targets:
        size += 8 if target["kind"] == "class" else 4 * math.prod(target["shape"])
    return max(size, 1)


def demo_data(
    plan: TrainingPlan,
    config: TrainingConfig,
    generator: torch.Generator,
    reporter: Reporter | None = None,
) -> PreparedData:
    total = min(DEMO_SAMPLES, max(2, DEMO_BYTES // _demo_sample_bytes(plan)))
    n_val = min(max(1, round(total * config.val_split)), total - 1)
    n_train = total - n_val
    if config.max_samples is not None and config.max_samples < n_train:
        n_train = config.max_samples
        n_val = min(n_val, val_size_for(config.max_samples, config.val_split))
    if total < DEMO_SAMPLES and reporter is not None:
        reporter.log(
            f"The inputs are large, so the demo data has only {n_train + n_val:,} samples."
        )
    rule = torch.Generator().manual_seed(int(torch.randint(0, 2**31, (1,), generator=generator)))
    inputs, targets = demo_samples(plan, n_train + n_val, generator, rule)
    train = TensorSource([x[:n_train] for x in inputs], [y[:n_train] for y in targets])
    val = TensorSource([x[n_train:] for x in inputs], [y[n_train:] for y in targets])
    return PreparedData(train, val, {"kind": "demo"})


# --------------------------------------------------------------------------- #
# Built-in datasets (torchvision)
# --------------------------------------------------------------------------- #


def open_torchvision(dataset_id: str, root: Path, train: bool, download: bool) -> Any:
    """Open a torchvision dataset (replaced by fake data in the tests)."""
    try:
        from torchvision import datasets
    except ImportError as exc:
        raise TrainingError(
            "The built-in datasets need torchvision. Install it with: pip install torchvision"
        ) from exc
    except Exception as exc:  # e.g. a torchvision build that does not match PyTorch
        traceback.print_exc()
        raise TrainingError(
            f"torchvision cannot be loaded ({short_error(exc)}). Install the torchvision "
            f"version that matches PyTorch {torch.__version__}."
        ) from exc

    classes = {
        "mnist": datasets.MNIST,
        "fashion_mnist": datasets.FashionMNIST,
        "cifar10": datasets.CIFAR10,
    }
    return classes[dataset_id](root=str(root), train=train, download=download)


def _as_nchw(data: Any) -> torch.Tensor:
    """torchvision ``.data`` (MNIST: tensor N×H×W, CIFAR: numpy N×H×W×C) -> uint8 N×C×H×W."""
    images = data if isinstance(data, torch.Tensor) else torch.as_tensor(data)
    images = images.to(torch.uint8)
    if images.dim() == 3:
        return images.unsqueeze(1)
    if images.dim() == 4 and images.shape[-1] in (1, 3) and images.shape[1] not in (1, 3):
        return images.permute(0, 3, 1, 2).contiguous()
    return images


def _channel_stats(
    images: torch.Tensor, transform: ImageTransform, generator: torch.Generator
) -> tuple[list[float], list[float]]:
    """Per-channel mean and std of converted images (estimated on a random subset)."""
    if len(images) > STATS_SAMPLES:
        images = images[torch.randperm(len(images), generator=generator)[:STATS_SAMPLES]]
    image = transform.image
    channels = image["channels"]
    largest = max(
        math.prod(images.shape[1:]),
        max(channels, images.shape[1]) * image["height"] * image["width"],
    )
    chunk = max(1, STATS_CHUNK_BYTES // (4 * largest))
    total = torch.zeros(channels, dtype=torch.float64)
    total_sq = torch.zeros(channels, dtype=torch.float64)
    count = 0
    for part in images.split(chunk):
        x = transform.to_unit(part)
        total += x.sum(dim=(0, 2, 3), dtype=torch.float64)
        total_sq += x.square().sum(dim=(0, 2, 3), dtype=torch.float64)
        count += x.shape[0] * x.shape[2] * x.shape[3]
    mean = total / count
    std = (total_sq / count - mean * mean).clamp_min(1e-12).sqrt()
    return mean.tolist(), [max(s, 1e-3) for s in std.tolist()]


def _open_splits(spec: BuiltinDataset, root: Path, download: bool) -> tuple[Any, Any]:
    return (
        open_torchvision(spec.id, root, True, download),
        open_torchvision(spec.id, root, False, download),
    )


def _open_builtin(spec: BuiltinDataset, root: Path, reporter: Reporter) -> tuple[Any, Any]:
    """Open the training and test split of a built-in dataset, downloading it if needed."""
    reporter.status("preparing", f"Loading {spec.name} …")
    try:
        return _open_splits(spec, root, download=False)
    except TrainingError:
        raise
    except Exception as exc:
        # torchvision: "Dataset not found [or corrupted]. You can use download=True …"
        if not (isinstance(exc, RuntimeError) and "not found" in str(exc).lower()):
            traceback.print_exc()
            raise TrainingError(
                f"Could not load {spec.name} ({short_error(exc)}). The downloaded files may be "
                f"damaged: delete the folder {root} and try again."
            ) from exc
    reporter.status("preparing", f"Downloading {spec.name} …")
    reporter.log(f"Downloading {spec.name} (about {spec.download_mb} MB) …")
    root.mkdir(parents=True, exist_ok=True)
    try:
        splits = _open_splits(spec, root, download=True)
    except TrainingError:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise TrainingError(
            f"Could not download {spec.name}. Check the internet connection of the server."
        ) from exc
    reporter.log(f"Downloaded {spec.name}.")
    return splits


def builtin_data(
    plan: TrainingPlan,
    info: DatasetInfo,
    data_root: Path,
    config: TrainingConfig,
    reporter: Reporter,
    generator: torch.Generator,
) -> PreparedData:
    spec = BUILTIN_DATASETS.get(info.builtin or "")
    if spec is None or plan.image is None:
        raise TrainingError(f"Unknown built-in dataset '{info.builtin}'.")
    train_set, val_set = _open_builtin(spec, data_root / spec.id, reporter)
    train_images = _as_nchw(train_set.data)
    train_labels = torch.as_tensor(train_set.targets, dtype=torch.long)
    val_images = _as_nchw(val_set.data)
    val_labels = torch.as_tensor(val_set.targets, dtype=torch.long)
    if config.max_samples is not None:
        if config.max_samples < len(train_images):
            keep = torch.randperm(len(train_images), generator=generator)[: config.max_samples]
            train_images, train_labels = train_images[keep], train_labels[keep]
        n_val = val_size_for(config.max_samples, BUILTIN_VAL_SPLIT)
        if n_val < len(val_images):
            keep = torch.randperm(len(val_images), generator=generator)[:n_val]
            val_images, val_labels = val_images[keep], val_labels[keep]

    image = plan.image
    native = (spec.shape[1], spec.shape[2]) == (image["height"], image["width"])
    if native and spec.shape[0] == image["channels"]:
        mean, std = list(spec.mean), list(spec.std)
    else:
        reporter.status("preparing", f"Converting {spec.name} images …")
        mean, std = _channel_stats(train_images, ImageTransform(image, [0.0], [1.0]), generator)
    transform = ImageTransform(image, mean, std)
    class_names = list(plan.class_names or spec.class_names)
    preprocessing = _image_preprocessing(image, mean, std, class_names)
    preprocessing["dataset"] = spec.id
    return PreparedData(
        TensorSource([train_images], [train_labels], [transform]),
        TensorSource([val_images], [val_labels], [transform]),
        preprocessing,
    )


# --------------------------------------------------------------------------- #
# CSV files
# --------------------------------------------------------------------------- #


TOO_LARGE = "The CSV file contains values that are too large to train on."


def _standardise(values: torch.Tensor, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and std (float64, population, 1 where constant) of ``values[rows]``."""
    total = torch.zeros(values.shape[1:], dtype=torch.float64)
    for index in rows.split(STATS_CHUNK_ROWS):
        total += values[index].sum(dim=0, dtype=torch.float64)
    mean = total / max(len(rows), 1)
    squares = torch.zeros_like(mean)
    for index in rows.split(STATS_CHUNK_ROWS):
        squares += (values[index].double() - mean).square().sum(dim=0)
    std = (squares / max(len(rows), 1)).sqrt()
    std = torch.where(std > 1e-8, std, torch.ones_like(std))
    if not (torch.isfinite(mean).all() and torch.isfinite(std).all()):
        raise TrainingError(TOO_LARGE)
    return mean, std


def _read_csv_tensors(
    path: Path, info: DatasetInfo, plan: TrainingPlan
) -> tuple[torch.Tensor, torch.Tensor, list[str], str]:
    """Stream the CSV file into preallocated tensors: features (float32) and targets
    (float64: class indices or values). Returns them with the column names and delimiter."""
    rows_expected = info.num_samples or 0
    width = (info.num_features or 0) + 1
    changed = TrainingError("The CSV file has changed since it was checked. Upload it again.")
    if rows_expected < 2 or width < 2:
        raise changed
    classification = plan.targets[0]["kind"] == "class"
    num_classes = int(plan.targets[0].get("num_classes") or 0)
    label_index = {name: i for i, name in enumerate(info.class_names or [])}
    features = torch.empty((rows_expected, width - 1), dtype=torch.float32)
    targets = torch.empty(rows_expected, dtype=torch.float64)

    with open_csv(path) as (header, rows, delimiter):
        decimal_comma = delimiter != ","
        columns = list(info.columns)
        if len(columns) != width:
            columns = [f"column_{i + 1}" for i in range(width - 1)] + ["target"]
        offset = 2 if header is not None else 1  # line number of the first data row
        block: list[list[float]] = []
        block_targets: list[float] = []
        filled = 0

        def store() -> None:
            nonlocal filled
            values = torch.tensor(block, dtype=torch.float32)
            bad = (~torch.isfinite(values)).nonzero()
            if len(bad):
                row, col = (int(v) for v in bad[0])
                raise TrainingError(
                    f"Line {filled + row + offset}, column '{columns[col]}': the value is too "
                    "large (numbers up to about 3.4e38 are supported)."
                )
            features[filled : filled + len(block)] = values
            targets[filled : filled + len(block)] = torch.tensor(block_targets, dtype=torch.float64)
            filled += len(block)
            block.clear()
            block_targets.clear()

        for index, row in enumerate(rows):
            line = index + offset
            if index >= rows_expected:
                raise changed
            if len(row) != width:
                raise TrainingError(f"Line {line} has {len(row)} values, expected {width}.")
            values = []
            for col, cell in enumerate(row[:-1]):
                value = parse_number(cell, decimal_comma)
                if value is None:
                    shown = cell.strip()[:30] or "(empty)"
                    raise TrainingError(
                        f"Line {line}, column '{columns[col]}': '{shown}' is not a number."
                    )
                values.append(value)
            cell = row[-1].strip()
            number = parse_number(cell, decimal_comma)
            if classification and info.target_kind == "text":
                if cell not in label_index:
                    raise TrainingError(f"Line {line}: unknown class '{cell[:30]}'.")
                number = float(label_index[cell])
            elif number is None:
                raise TrainingError(f"Line {line}: the target '{cell[:30]}' is not a number.")
            elif classification and not (number.is_integer() and 0 <= number < num_classes):
                raise TrainingError(
                    f"Line {line}: the target {cell[:30]} is not a class index between 0 and "
                    f"{num_classes - 1}."
                )
            block.append(values)
            block_targets.append(number)
            if len(block) == CSV_BLOCK_ROWS:
                store()
        if block:
            store()
    if filled != rows_expected:
        raise changed
    return features, targets, columns, delimiter


def csv_data(
    plan: TrainingPlan,
    info: DatasetInfo,
    path: Path,
    config: TrainingConfig,
    reporter: Reporter,
    generator: torch.Generator,
) -> PreparedData:
    reporter.status("preparing", "Reading the CSV file …")
    if not path.is_file():
        raise TrainingError("The CSV file was not found. Upload it again.")
    x, targets, columns, delimiter = _read_csv_tensors(path, info, plan)
    train, val = split_indices(len(x), config.val_split, config.max_samples, generator)
    mean, std = _standardise(x, train)
    x.sub_(mean.float()).div_(std.float())  # in place: the CSV may be large
    if not torch.isfinite(x).all():
        raise TrainingError(TOO_LARGE)
    preprocessing: dict[str, Any] = {
        "kind": "tabular",
        "columns": columns[:-1],
        "target": columns[-1],
        "delimiter": delimiter,
        "decimal_comma": delimiter != ",",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "target_mean": None,
        "target_std": None,
        "class_names": None,
    }
    target_std = None
    if plan.targets[0]["kind"] == "class":
        y = targets.long()
        preprocessing["class_names"] = list(plan.class_names or [])
    else:
        y_mean, y_std = _standardise(targets.unsqueeze(1), train)
        y = ((targets.unsqueeze(1) - y_mean) / y_std).float()
        if not torch.isfinite(y).all():
            raise TrainingError(TOO_LARGE)
        target_std = float(y_std[0])
        preprocessing["target_mean"] = float(y_mean[0])
        preprocessing["target_std"] = target_std
    data = TensorSource([x], [y])
    return PreparedData(Subset(data, train), Subset(data, val), preprocessing, target_std)


# --------------------------------------------------------------------------- #
# Image folders
# --------------------------------------------------------------------------- #


def image_folder_data(
    plan: TrainingPlan,
    info: DatasetInfo,
    path: Path,
    config: TrainingConfig,
    reporter: Reporter,
    generator: torch.Generator,
) -> PreparedData:
    image = plan.image
    if image is None:
        raise TrainingError("The network does not take images.")
    root = path / (info.subdir or "")
    if not root.is_dir():
        raise TrainingError("The image folder was not found. Upload it again.")
    reporter.status("preparing", "Looking for images …")
    classes = list_image_files(root)
    if list(classes) != list(info.class_names or []):
        raise TrainingError(
            "The image folder has changed since it was checked. Upload or select it again."
        )
    files = [f for names in classes.values() for f in names]
    labels = torch.tensor(
        [label for label, names in enumerate(classes.values()) for _ in names], dtype=torch.long
    )
    train, val = split_indices(len(files), config.val_split, config.max_samples, generator)
    channels = image["channels"]
    mean, std = [0.5] * channels, [0.5] * channels
    transform = ImageTransform(image, mean, std)
    preprocessing = _image_preprocessing(image, mean, std, list(plan.class_names or []))

    used = torch.cat([train, val])
    sample_bytes = channels * image["height"] * image["width"]
    if len(used) * sample_bytes > PRELOAD_BYTES:
        reporter.log("The images do not fit into memory and are read from disk during training.")
        source = ImageFileSource(files, labels, root, image, transform)
        return PreparedData(Subset(source, train), Subset(source, val), preprocessing)

    images = torch.empty((len(used), channels, image["height"], image["width"]), dtype=torch.uint8)
    total = len(used)
    last = 0.0
    for i, index in enumerate(used.tolist()):
        now = time.monotonic()
        if now - last >= 0.5:
            last = now
            if reporter.should_stop():
                raise StopTraining
            reporter.status("preparing", f"Loading images {i:,} / {total:,}")
        images[i] = load_image(files[index], image, root)
    reporter.status("preparing", f"Loading images {total:,} / {total:,}")
    n_train = len(train)
    return PreparedData(
        TensorSource([images[:n_train]], [labels[train]], [transform]),
        TensorSource([images[n_train:]], [labels[val]], [transform]),
        preprocessing,
    )
