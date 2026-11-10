"""Segmentation datasets: images with one mask each, a class for every pixel (no PyTorch
needed; reading the files needs Pillow and NumPy).

A dataset is a folder (or ZIP file) with the images in ``images/`` and their masks in
``masks/``; an image and its mask have the same file name (the ending may differ, and the
mask may end in ``_mask``)::

    images/001.jpg   masks/001.png
    images/002.jpg   masks/002_mask.png

With ``train/images``, ``train/masks``, ``val/images``, ``val/masks`` (and optionally
``test/…``; also ``images/train``, ``masks/train``, …) the folders are the split.

How the masks are read is decided once for the whole dataset, from the values that occur:

``binary``   only black (0) and white (255): background and foreground
``indices``  grey values or palette indices below 32: the class of each pixel (0, 1, 2 …);
             255 marks pixels that are ignored (they count neither for the loss nor for the
             metrics)
``levels``   at most 32 other grey values: classes in ascending order (0, 128, 255 → 0, 1, 2)
``colors``   colour masks with at most 32 different colours: one class per colour, black
             first, then sorted by red, green, blue (or in the order of ``classes.json``)

Anything else is refused, e.g. masks saved as JPEG or with smoothed edges, which hold many
in-between values. ``classes.txt`` (one name per line) or ``classes.json`` (a list of names, or
of ``{"name": "road", "color": [128, 64, 128]}``) name the classes.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .datasets import IMAGE_EXTENSIONS, MIN_SAMPLES, DatasetError, DatasetInfo, _hidden

# The most classes a segmentation dataset can have, and the mask value of ignored pixels.
MAX_SEGMENTATION_CLASSES = 32
IGNORE_INDEX = 255
# The most image/mask pairs of a dataset and the largest image or mask (pixels).
MAX_PAIRS = 20_000
MAX_SIDE = 4096
IMAGE_DIRS = ("images", "image", "imgs", "img")
MASK_DIRS = ("masks", "mask", "labels", "label", "annotations", "annotation")
# Folder names of a predefined split: training (0), validation (1), test (2).
SPLIT_DIRS = {
    "train": 0,
    "training": 0,
    "val": 1,
    "valid": 1,
    "validation": 1,
    "test": 2,
    "testing": 2,
}
SPLIT_NAMES = ("train", "val", "test")
# Endings of a mask's file name that are not part of the image's name: 001_mask.png ↔ 001.jpg
MASK_SUFFIXES = ("_mask", "-mask", "_label", "_seg", "_gt")
CLASS_FILES = ("classes.json", "classes.txt")
LAYOUT_HINT = (
    "Put the images into a folder images/ and their masks into a folder masks/, with the "
    "same file names (images/001.jpg and masks/001.png)."
)
# Colours of the classes in previews and predicted masks. The background (class 0) is dark;
# classes 1 to 8 get eight hues in a fixed order that stay apart for people with a colour
# vision deficiency (checked on a dark surface: neighbours in this order differ by at least
# 8 in OKLab under simulated protanopia and deuteranopia). Further classes reuse the hues in
# lighter and darker steps: with that many, the legend and the table name the classes.
BACKGROUND_COLOR = "#1a1a19"
_BASE_COLORS = (
    "#3987e5",  # blue
    "#d95926",  # orange
    "#199e70",  # aqua
    "#c98500",  # yellow
    "#d55181",  # magenta
    "#008300",  # green
    "#9085e9",  # violet
    "#e66767",  # red
)
IGNORE_COLOR = "#8a8f98"


def _mixed(color: str, other: int, share: float) -> str:
    """``color`` mixed with white (``other`` 255) or black (0)."""
    parts = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(c + (other - c) * share):02x}" for c in parts)


CLASS_COLORS = (
    BACKGROUND_COLOR,
    *_BASE_COLORS,
    *(_mixed(color, 255, 0.45) for color in _BASE_COLORS),
    *(_mixed(color, 0, 0.4) for color in _BASE_COLORS),
    *(_mixed(color, 255, 0.72) for color in _BASE_COLORS[:7]),
)


def class_colors(count: int) -> list[str]:
    """The colours of ``count`` classes (they repeat after 32)."""
    return [CLASS_COLORS[i % len(CLASS_COLORS)] for i in range(count)]


def rgb(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def _imaging() -> tuple[Any, Any]:
    """(numpy, PIL.Image), or a :class:`DatasetError` that says what to install."""
    try:
        import numpy
        from PIL import Image
    except ImportError:
        raise DatasetError(
            "Reading segmentation masks needs Pillow and NumPy. Install them with: "
            "pip install pillow numpy"
        ) from None
    return numpy, Image


# --------------------------------------------------------------------------- #
# Pairs of images and masks
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Pair:
    key: str  # what image and mask have in common, e.g. "train/001"
    image: Path
    mask: Path
    split: int | None  # 0 training, 1 validation, 2 test (None: not in a split folder)


@dataclass
class Pairing:
    pairs: list[Pair]
    unmatched_images: list[str]
    unmatched_masks: list[str]

    @property
    def has_split(self) -> bool:
        """True if the folders are a split: every pair is in a training, validation or test
        folder, and there are training and validation pairs."""
        splits = {pair.split for pair in self.pairs}
        return None not in splits and {0, 1} <= splits

    def split_counts(self) -> list[int]:
        return [sum(1 for pair in self.pairs if pair.split == label) for label in range(3)]


def _role(parts: tuple[str, ...]) -> tuple[str | None, tuple[str, ...]]:
    """("image" | "mask" | None, the folders without the one that decided it)."""
    for i, part in enumerate(parts):
        lower = part.lower()
        if lower in IMAGE_DIRS:
            return "image", parts[:i] + parts[i + 1 :]
        if lower in MASK_DIRS:
            return "mask", parts[:i] + parts[i + 1 :]
    return None, parts


def _mask_stem(stem: str) -> str:
    lower = stem.lower()
    for suffix in MASK_SUFFIXES:
        if lower.endswith(suffix) and len(stem) > len(suffix):
            return stem[: -len(suffix)]
    return stem


def _image_files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not any(_hidden(part) for part in p.relative_to(root).parts)
    )


def find_pairs(root: str | Path) -> Pairing:
    """Match the images below ``root`` with their masks (sorted by name)."""
    root = Path(root)
    images: dict[str, Path] = {}
    masks: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in _image_files(root):
        relative = path.relative_to(root)
        role, folders = _role(relative.parts[:-1])
        if role is None:
            continue
        stem = _mask_stem(relative.stem) if role == "mask" else relative.stem
        key = "/".join([*folders, stem])
        found = images if role == "image" else masks
        if key in found:
            duplicates.append(relative.as_posix())
            continue
        found[key] = path
    if duplicates:
        raise DatasetError(
            f"'{duplicates[0]}' has the same name as another file in its folder (only the "
            "ending differs), so it is not clear which mask belongs to which image."
        )
    pairs = []
    for key in sorted(images.keys() & masks.keys()):
        split = next(
            (SPLIT_DIRS[p.lower()] for p in key.split("/")[:-1] if p.lower() in SPLIT_DIRS), None
        )
        pairs.append(Pair(key, images[key], masks[key], split))
    return Pairing(
        pairs,
        sorted(images[k].relative_to(root).as_posix() for k in images.keys() - masks.keys()),
        sorted(masks[k].relative_to(root).as_posix() for k in masks.keys() - images.keys()),
    )


def is_segmentation_folder(path: str | Path) -> bool:
    """True if ``path`` holds a folder of images and a folder of masks (see the module)."""
    root = Path(path)
    if not root.is_dir():
        return False
    roles = set()
    for file in _image_files(root):
        role, _ = _role(file.relative_to(root).parts[:-1])
        if role is not None:
            roles.add(role)
            if len(roles) == 2:
                return True
    return False


# --------------------------------------------------------------------------- #
# Class names
# --------------------------------------------------------------------------- #


def _find_class_file(root: Path) -> Path | None:
    """classes.json or classes.txt, the one closest to the top of the dataset."""
    found = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.name.lower() in CLASS_FILES
        and not any(_hidden(part) for part in p.relative_to(root).parts)
    ]
    found.sort(key=lambda p: (len(p.relative_to(root).parts), CLASS_FILES.index(p.name.lower())))
    return found[0] if found else None


def read_class_file(root: Path) -> tuple[list[str], list[list[int]] | None] | None:
    """``(names, colours or None)`` from classes.json / classes.txt, or None without one."""
    path = _find_class_file(root)
    if path is None:
        return None
    name = path.name
    if path.stat().st_size > 256 * 1024:
        raise DatasetError(f"{name} is too large to be a list of class names.")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        raise DatasetError(f"{name} cannot be read as UTF-8 text.") from None
    colors: list[list[int]] | None = None
    if name.lower().endswith(".json"):
        wrong = DatasetError(
            f'{name} must be a list of class names, e.g. ["background", "cat"], or of '
            '{"name": "cat", "color": [255, 0, 0]} for colour masks.'
        )
        try:
            data = json.loads(text)
        except ValueError:
            raise wrong from None
        if isinstance(data, dict):
            data = data.get("classes", data.get("names"))
        if not isinstance(data, list) or not data:
            raise wrong
        if all(isinstance(item, str) for item in data):
            names = [item.strip() for item in data]
        elif all(isinstance(item, dict) and isinstance(item.get("name"), str) for item in data):
            names = [item["name"].strip() for item in data]
            given = [item.get("color") for item in data]
            if any(color is not None for color in given):
                colors = []
                for color in given:
                    if (
                        not isinstance(color, list)
                        or len(color) != 3
                        or not all(
                            isinstance(c, int) and not isinstance(c, bool) and 0 <= c <= 255
                            for c in color
                        )
                    ):
                        raise DatasetError(
                            f"{name}: every class needs a color [red, green, blue] with "
                            "values from 0 to 255 (or none of them has one)."
                        )
                    colors.append(list(color))
                if len({tuple(c) for c in colors}) != len(colors):
                    raise DatasetError(f"{name} lists the same color for two classes.")
        else:
            raise wrong
    else:
        names = [line.strip() for line in text.splitlines() if line.strip()]
    if not names or any(not n for n in names):
        raise DatasetError(f"{name} contains no class names.")
    if len(names) > MAX_SEGMENTATION_CLASSES:
        raise DatasetError(
            f"{name} lists {len(names)} classes; segmentation supports at most "
            f"{MAX_SEGMENTATION_CLASSES}."
        )
    return [n[:60] for n in names], colors


# --------------------------------------------------------------------------- #
# Reading masks
# --------------------------------------------------------------------------- #


def _open(Image: Any, path: Path, root: Path, what: str) -> Any:
    try:
        img = Image.open(path)
        width, height = img.size
    except Exception:
        raise DatasetError(
            f"The {what} '{_shown(path, root)}' cannot be read. Remove it or save it again."
        ) from None
    if max(width, height) > MAX_SIDE:
        img.close()
        raise DatasetError(
            f"The {what} '{_shown(path, root)}' has {width}×{height} pixels; at most "
            f"{MAX_SIDE}×{MAX_SIDE} are supported. Scale the dataset down first."
        )
    return img


def _shown(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def raw_mask(path: Path, root: Path) -> Any:
    """The pixels of a mask file as they are stored: a 2-D array of grey values or palette
    indices (uint8 or uint16), or a 3-D array height × width × 3 for colour masks."""
    np, Image = _imaging()
    with _open(Image, path, root, "mask") as img:
        try:
            if img.mode == "P":
                return np.asarray(img)  # the palette indices are the classes
            if img.mode in ("1", "L", "LA", "La"):
                return np.asarray(img.convert("L"))
            if img.mode.startswith("I") or img.mode == "F":
                values = np.asarray(img)
                return np.clip(np.rint(values), 0, 65535).astype(np.uint16)
            pixels = np.asarray(img.convert("RGB"))
        except DatasetError:
            raise
        except Exception:
            raise DatasetError(
                f"The mask '{_shown(path, root)}' cannot be read. Remove it or save it again."
            ) from None
    if (pixels[..., 0] == pixels[..., 1]).all() and (pixels[..., 1] == pixels[..., 2]).all():
        return np.ascontiguousarray(pixels[..., 0])  # grey values stored as RGB
    return pixels


def _pack(pixels: Any) -> Any:
    """RGB pixels (… × 3, uint8) as one number per pixel."""
    np, _ = _imaging()
    wide = pixels.astype(np.uint32)
    return (wide[..., 0] << 16) | (wide[..., 1] << 8) | wide[..., 2]


def _unpack(value: int) -> list[int]:
    return [(value >> 16) & 255, (value >> 8) & 255, value & 255]


class MaskDecoder:
    """Turns mask files into class maps the way the dataset was inspected: uint8 arrays with
    the class index of every pixel (IGNORE_INDEX for ignored pixels)."""

    def __init__(self, segmentation: dict[str, Any], num_classes: int, root: Path):
        np, _ = _imaging()
        self.np = np
        self.root = root
        self.format = segmentation.get("mask_format")
        self.num_classes = num_classes
        self.ignore = segmentation.get("ignore_index") is not None
        values = segmentation.get("mask_values") or []
        if self.format in ("binary", "levels"):
            if self.format == "binary":
                values = [0, 255]
            self.lut = np.full(256, 254, dtype=np.uint8)  # 254: a value that was not there
            for index, value in enumerate(values):
                self.lut[int(value)] = index
        elif self.format == "colors":
            packed = np.array([(c[0] << 16) | (c[1] << 8) | c[2] for c in values], dtype=np.uint32)
            self.order = np.argsort(packed)
            self.sorted = packed[self.order]
        elif self.format != "indices":
            raise DatasetError("How the masks of this dataset are read is not known.")

    def _changed(self, path: Path) -> DatasetError:
        return DatasetError(
            f"The mask '{_shown(path, self.root)}' holds values that were not there when the "
            "dataset was checked. Upload the dataset again."
        )

    def decode(self, path: Path) -> Any:
        np = self.np
        raw = raw_mask(path, self.root)
        if self.format == "colors":
            if raw.ndim == 2:
                raw = np.stack([raw.astype(np.uint8)] * 3, axis=-1)
            packed = _pack(raw)
            position = np.searchsorted(self.sorted, packed).clip(0, len(self.sorted) - 1)
            if not (self.sorted[position] == packed).all():
                raise self._changed(path)
            return self.order[position].astype(np.uint8)
        if raw.ndim != 2 or int(raw.max(initial=0)) > 255:
            raise self._changed(path)
        raw = raw.astype(np.uint8)
        if self.format == "indices":
            bad = raw >= self.num_classes
            if self.ignore:
                bad &= raw != IGNORE_INDEX
            if bad.any():
                raise self._changed(path)
            return raw
        classes = self.lut[raw]
        if (classes == 254).any():
            raise self._changed(path)
        return classes


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #


def _refuse_values(count: int, examples: list[int], jpeg: bool) -> DatasetError:
    shown = ", ".join(str(v) for v in examples[:8])
    more = "more than " if count > 256 else ""
    text = (
        f"The masks hold {more}{min(count, 256)} different grey values ({shown} …), but a mask "
        "needs one value per class: the class indices 0, 1, 2 … (at most "
        f"{MAX_SEGMENTATION_CLASSES} classes) or black and white. "
    )
    if jpeg:
        text += (
            "Some masks are JPEG files: JPEG compression changes the pixel values, so the "
            "classes cannot be read from them. Save the masks as PNG."
        )
    else:
        text += (
            "Smoothed (anti-aliased) edges or resized masks add in-between values; save the "
            "masks as PNG without smoothing."
        )
    return DatasetError(text)


def inspect_segmentation_folder(path: str | Path, name: str | None = None) -> DatasetInfo:
    """Check a folder with images and masks and describe it (see the module docstring).

    Every mask is read once (to find the classes and how many pixels each has); of the
    images only the size is read. Raises :class:`DatasetError`.
    """
    np, Image = _imaging()
    path = Path(path)
    if not path.is_dir():
        raise DatasetError(f"'{path}' is not a folder.")
    pairing = find_pairs(path)
    pairs = pairing.pairs
    if not pairs:
        if not pairing.unmatched_images and not pairing.unmatched_masks:
            raise DatasetError(f"No folders with images and masks were found. {LAYOUT_HINT}")
        if not pairing.unmatched_masks:
            raise DatasetError(f"The dataset has images but no masks. {LAYOUT_HINT}")
        if not pairing.unmatched_images:
            raise DatasetError(f"The dataset has masks but no images. {LAYOUT_HINT}")
        raise DatasetError(
            "No image has a mask with the same file name: e.g. "
            f"'{pairing.unmatched_images[0]}' and '{pairing.unmatched_masks[0]}' do not match. "
            f"{LAYOUT_HINT}"
        )
    if len(pairs) < MIN_SAMPLES:
        raise DatasetError(f"The dataset needs at least {MIN_SAMPLES} images with a mask.")
    if len(pairs) > MAX_PAIRS:
        raise DatasetError(
            f"The dataset has {len(pairs):,} images with masks; at most {MAX_PAIRS:,} are "
            "supported."
        )
    listed = read_class_file(path)

    grey = np.zeros(65536, dtype=np.int64)  # pixels per grey value / palette index
    colours: Counter[int] = Counter()  # pixels per colour (packed), colour masks only
    sizes: Counter[tuple[int, int]] = Counter()
    colour_images = 0
    mismatches: list[str] = []
    jpeg = False
    for pair in pairs:
        with _open(Image, pair.image, path, "image") as img:
            size = img.size
            if img.mode not in ("1", "L", "LA", "La", "F") and not img.mode.startswith("I"):
                colour_images += 1
        sizes[size] += 1
        raw = raw_mask(pair.mask, path)
        if (raw.shape[1], raw.shape[0]) != size:
            mismatches.append(_shown(pair.mask, path))
        jpeg = jpeg or pair.mask.suffix.lower() in (".jpg", ".jpeg")
        if raw.ndim == 3:
            values, counts = np.unique(_pack(raw).ravel(), return_counts=True)
            for value, count in zip(values.tolist(), counts.tolist(), strict=True):
                colours[value] += count
            if len(colours) > 256:
                break
        else:
            grey += np.bincount(raw.ravel(), minlength=65536)

    used = np.flatnonzero(grey).tolist()
    mask_values: list[Any] | None = None
    ignore_index: int | None = None
    if colours:
        for value in used:  # grey masks among colour masks: grey is a colour too
            if value > 255:
                raise _refuse_values(len(used), used, jpeg)
            colours[(value << 16) | (value << 8) | value] += int(grey[value])
        if listed is not None and listed[1] is not None:
            order = [(c[0] << 16) | (c[1] << 8) | c[2] for c in listed[1]]
            unknown = [c for c in colours if c not in set(order)]
            if unknown:
                shown = ", ".join(str(tuple(_unpack(c))) for c in sorted(unknown)[:3])
                raise DatasetError(
                    f"The masks use colours that classes.json does not list, e.g. {shown}. "
                    "List every colour of the masks, or save the masks without smoothed edges."
                )
        else:
            order = sorted(colours)  # black first
        if len(order) > MAX_SEGMENTATION_CLASSES:
            more = "more than " if len(colours) > 256 else ""
            raise DatasetError(
                f"The masks hold {more}{min(len(colours), 256)} different colours, but a colour "
                f"mask needs one colour per class (at most {MAX_SEGMENTATION_CLASSES} classes). "
                + (
                    "Some masks are JPEG files: JPEG compression changes the colours. Save "
                    "the masks as PNG."
                    if jpeg
                    else "Smoothed (anti-aliased) edges add in-between colours; save the masks "
                    "as PNG without smoothing."
                )
            )
        mask_format = "colors"
        mask_values = [_unpack(c) for c in order]
        counts = [int(colours.get(c, 0)) for c in order]
        ignored = 0
    else:
        if listed is not None and listed[1] is not None:
            raise DatasetError(
                "classes.json gives a color for every class, but the masks are not colour "
                "images. Leave the colors out: grey masks hold the class index of each pixel."
            )
        indices = [v for v in used if v != IGNORE_INDEX]
        if set(used) <= {0, 255} and 255 in used:
            mask_format = "binary"
            counts = [int(grey[0]), int(grey[255])]
            ignored = 0
        elif max(indices) < MAX_SEGMENTATION_CLASSES:
            mask_format = "indices"
            counts = [int(c) for c in grey[: max(indices) + 1]]
            ignored = int(grey[IGNORE_INDEX])
            ignore_index = IGNORE_INDEX if ignored else None
        elif len(used) <= MAX_SEGMENTATION_CLASSES and max(used) <= 255:
            mask_format = "levels"
            mask_values = used
            counts = [int(grey[v]) for v in used]
            ignored = 0
        else:
            raise _refuse_values(len(used), used, jpeg)

    names = list(listed[0]) if listed is not None else []
    if len(names) > len(counts) and mask_format == "indices":
        counts += [0] * (len(names) - len(counts))  # listed classes that no mask shows
    if len(counts) < 2:
        raise DatasetError(
            "The masks are all background (the value 0 everywhere), so there is nothing to "
            "learn. Mark the objects with 1, 2 … (or white in black-and-white masks)."
        )
    if len(names) > len(counts):
        raise DatasetError(
            f"The class file lists {len(names)} classes, but the masks only hold "
            f"{len(counts)} different values."
        )
    if not names:
        names = ["background", "foreground"] if len(counts) == 2 else ["background"]
    names += [f"class {i}" for i in range(len(names), len(counts))]

    total = sum(counts) or 1
    width, height = sizes.most_common(1)[0][0]
    widths = [size[0] for size in sizes]
    heights = [size[1] for size in sizes]
    info = DatasetInfo(
        kind="segmentation",
        name=name or path.name,
        task="segmentation",
        num_samples=len(pairs),
        sample_shape=[3 if colour_images else 1, height, width],
        class_names=names,
    )
    if pairing.has_split:
        info.predefined_split = pairing.split_counts()
    info.segmentation = {
        "mask_format": mask_format,
        "mask_values": mask_values,
        "ignore_index": ignore_index,
        "ignored_share": round(ignored / (total + ignored), 6),
        "class_pixel_share": [round(count / total, 6) for count in counts],
        "image_sizes": {
            "most_common": [width, height],
            "min": [min(widths), min(heights)],
            "max": [max(widths), max(heights)],
            "different": len(sizes),
        },
        "unmatched_images": len(pairing.unmatched_images),
        "unmatched_masks": len(pairing.unmatched_masks),
        "unmatched_examples": [*pairing.unmatched_images[:3], *pairing.unmatched_masks[:3]],
        "mask_size_mismatches": len(mismatches),
        "mask_size_examples": mismatches[:3],
    }
    return info


def describe(info: DatasetInfo) -> list[str]:
    """Notes on a segmentation dataset for the user: what was left out or looks odd."""
    details = info.segmentation or {}
    notes = []
    images, masks = details.get("unmatched_images") or 0, details.get("unmatched_masks") or 0
    if images or masks:
        parts = []
        if images:
            parts.append(f"{images:,} image{'s' if images != 1 else ''} without a mask")
        if masks:
            parts.append(f"{masks:,} mask{'s' if masks != 1 else ''} without an image")
        examples = ", ".join(details.get("unmatched_examples") or [])
        notes.append(
            f"Not used: {' and '.join(parts)}" + (f" (e.g. {examples})." if examples else ".")
        )
    mismatches = details.get("mask_size_mismatches") or 0
    if mismatches:
        notes.append(
            f"{mismatches:,} mask{'s have' if mismatches != 1 else ' has'} another size than "
            "the image; masks are resized to fit (nearest neighbour)."
        )
    return notes
