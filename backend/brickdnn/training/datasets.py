"""Dataset descriptions and file inspection (no PyTorch needed).

Four kinds of data can be used for training:

``random``   demo data generated to fit the network, with a hidden rule it can learn
``builtin``  MNIST, Fashion-MNIST or CIFAR-10 (downloaded on first use, needs torchvision)
``csv``      a CSV file: every column but the last is a feature, the last one is the target
``images``   a folder (or ZIP file) with one sub-folder of images per class

The functions here only read metadata (columns, classes, counts) so that the API can
check a dataset against a network before any training starts. Loading the actual
tensors happens in :mod:`brickdnn.training.engine`.
"""

from __future__ import annotations

import contextlib
import csv
import itertools
import math
import shutil
import stat
import zipfile
import zlib
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path, PurePosixPath
from typing import Any

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
MAX_CLASSES = 1000
MIN_SAMPLES = 4
# Where built-in datasets are downloaded when no other folder is configured (CLI).
DEFAULT_DATA_ROOT = Path.home() / ".cache" / "brickdnn" / "datasets"


class DatasetError(ValueError):
    """Raised for unusable data. The message is shown to the user."""


# --------------------------------------------------------------------------- #
# Built-in datasets
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BuiltinDataset:
    id: str
    name: str
    description: str
    shape: tuple[int, int, int]  # (channels, height, width)
    class_names: tuple[str, ...]
    train_size: int
    val_size: int
    download_mb: int
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "sample_shape": list(self.shape),
            "num_classes": len(self.class_names),
            "class_names": list(self.class_names),
            "train_size": self.train_size,
            "val_size": self.val_size,
            "download_mb": self.download_mb,
        }

    def info(self) -> DatasetInfo:
        return DatasetInfo(
            kind="builtin",
            name=self.name,
            builtin=self.id,
            num_samples=self.train_size + self.val_size,
            sample_shape=list(self.shape),
            class_names=list(self.class_names),
            train_size=self.train_size,
            val_size=self.val_size,
        )


BUILTIN_DATASETS: dict[str, BuiltinDataset] = {
    d.id: d
    for d in (
        BuiltinDataset(
            id="mnist",
            name="MNIST",
            description="Handwritten digits 0-9, grayscale 28×28.",
            shape=(1, 28, 28),
            class_names=tuple(str(i) for i in range(10)),
            train_size=60_000,
            val_size=10_000,
            download_mb=12,
            mean=(0.1307,),
            std=(0.3081,),
        ),
        BuiltinDataset(
            id="fashion_mnist",
            name="Fashion-MNIST",
            description="Clothing items in 10 classes, grayscale 28×28.",
            shape=(1, 28, 28),
            class_names=(
                "T-shirt/top",
                "Trouser",
                "Pullover",
                "Dress",
                "Coat",
                "Sandal",
                "Shirt",
                "Sneaker",
                "Bag",
                "Ankle boot",
            ),
            train_size=60_000,
            val_size=10_000,
            download_mb=30,
            mean=(0.2860,),
            std=(0.3530,),
        ),
        BuiltinDataset(
            id="cifar10",
            name="CIFAR-10",
            description="Small colour photos in 10 classes (planes, cars, animals …), RGB 32×32.",
            shape=(3, 32, 32),
            class_names=(
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ),
            train_size=50_000,
            val_size=10_000,
            download_mb=170,
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    )
}


def builtin_info(dataset_id: str) -> DatasetInfo:
    try:
        return BUILTIN_DATASETS[dataset_id].info()
    except KeyError:
        known = ", ".join(BUILTIN_DATASETS)
        raise DatasetError(f"Unknown dataset '{dataset_id}'. Available: {known}.") from None


# --------------------------------------------------------------------------- #
# Dataset metadata
# --------------------------------------------------------------------------- #


@dataclass
class DatasetInfo:
    """What is known about a dataset without loading it.

    Stored as JSON by the API (``to_dict``/``from_dict``) and written into every job file.
    """

    kind: str  # random | builtin | csv | images
    name: str
    num_samples: int | None = None
    # builtin
    builtin: str | None = None
    sample_shape: list[int] | None = None  # (channels, height, width)
    train_size: int | None = None
    val_size: int | None = None
    # csv
    num_features: int | None = None
    columns: list[str] = field(default_factory=list)  # feature columns + target column (last)
    has_header: bool | None = None
    delimiter: str | None = None
    target_kind: str | None = None  # integer | number | text
    target_range: list[float] | None = None  # [min, max] of numeric targets
    # builtin, images, csv with integer or text targets
    class_names: list[str] | None = None
    # images
    class_counts: list[int] | None = None
    subdir: str | None = None  # folder (relative to the dataset dir) holding the class folders

    @property
    def num_classes(self) -> int | None:
        return len(self.class_names) if self.class_names is not None else None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["num_classes"] = self.num_classes
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetInfo:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


def random_info() -> DatasetInfo:
    return DatasetInfo(kind="random", name="Demo data")


# --------------------------------------------------------------------------- #
# CSV files
# --------------------------------------------------------------------------- #


def parse_number(cell: str, decimal_comma: bool = False) -> float | None:
    """Parse a CSV cell as a finite number, or return ``None``."""
    text = cell.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        if not decimal_comma or "," not in text:
            return None
        try:
            value = float(text.replace(".", "").replace(",", "."))
        except ValueError:
            return None
    return value if math.isfinite(value) else None


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        return ","


@contextlib.contextmanager
def open_csv(path: str | Path) -> Iterator[tuple[list[str] | None, Iterator[list[str]], str]]:
    """Open a CSV file for streaming: yields ``(header or None, rows, delimiter)``.

    The delimiter (``,``, ``;`` or tab) is detected automatically. The first row is
    treated as a header when one of its feature cells is not a number. Empty lines are
    skipped. Reading errors become :class:`DatasetError`.
    """
    try:
        f = Path(path).open(newline="", encoding="utf-8-sig")  # noqa: SIM115 (closed below)
    except OSError as exc:
        raise DatasetError(f"The CSV file cannot be opened: {exc.strerror}.") from exc
    with f:
        try:
            sample = f.read(64 * 1024)
            f.seek(0)
            delimiter = _sniff_delimiter(sample)
            rows = (
                row for row in csv.reader(f, delimiter=delimiter) if any(c.strip() for c in row)
            )
            first = next(rows, None)
            if first is None:
                raise DatasetError("The CSV file is empty.")
            decimal_comma = delimiter != ","
            is_header = len(first) < 2 or any(
                parse_number(cell, decimal_comma) is None for cell in first[:-1]
            )
            if is_header and all(parse_number(cell, decimal_comma) is not None for cell in first):
                is_header = False
            header = [c.strip() for c in first] if is_header else None
            yield header, rows if is_header else itertools.chain([first], rows), delimiter
        except UnicodeDecodeError as exc:
            raise DatasetError("The CSV file is not UTF-8 encoded text.") from exc
        except csv.Error as exc:
            raise DatasetError(f"The CSV file cannot be read: {exc}.") from exc


def read_csv(path: str | Path) -> tuple[list[str] | None, list[list[str]], str]:
    """Read a whole CSV file into ``(header or None, rows, delimiter)``, see :func:`open_csv`."""
    with open_csv(path) as (header, rows, delimiter):
        return header, list(rows), delimiter


def inspect_csv(path: str | Path, name: str | None = None) -> DatasetInfo:
    """Check a CSV file and describe it. Raises :class:`DatasetError` with a helpful message.

    The file is streamed, so memory use does not grow with its size.
    """
    with open_csv(path) as (header, rows, delimiter):
        decimal_comma = delimiter != ","
        offset = 2 if header is not None else 1  # line number of the first data row
        width = len(header) if header is not None else None
        columns = header
        count = 0
        numeric, integer = True, True
        low, high = math.inf, -math.inf
        labels: set[str] = set()  # distinct targets, up to MAX_CLASSES + 1

        for index, row in enumerate(rows):
            line = index + offset
            if width is None:
                width = len(row)
            if columns is None:
                columns = [*(f"column_{i + 1}" for i in range(width - 1)), "target"]
            if width < 2:
                break
            if len(row) != width:
                raise DatasetError(f"Line {line} has {len(row)} values, expected {width}.")
            for col, cell in enumerate(row[:-1]):
                if parse_number(cell, decimal_comma) is None:
                    shown = cell.strip()[:30] or "(empty)"
                    raise DatasetError(
                        f"Line {line}, column '{columns[col]}': '{shown}' is not a number. "
                        "All columns except the last (the target) must be numeric."
                    )
            target = row[-1].strip()
            if not target:
                raise DatasetError(f"Line {line}: the target (last column) is empty.")
            count += 1
            if len(labels) <= MAX_CLASSES:
                labels.add(target)
            if numeric:
                value = parse_number(target, decimal_comma)
                if value is None:
                    numeric = False
                else:
                    low, high = min(low, value), max(high, value)
                    integer = integer and value.is_integer()

    if width is None or width < 2:
        raise DatasetError(
            "The CSV file needs at least two columns: one or more features and the target "
            "in the last column."
        )
    if count < MIN_SAMPLES:
        raise DatasetError(f"The CSV file needs at least {MIN_SAMPLES} data rows.")

    info = DatasetInfo(
        kind="csv",
        name=name or Path(path).stem,
        num_samples=count,
        num_features=width - 1,
        columns=list(columns or []),
        has_header=header is not None,
        delimiter=delimiter,
    )
    if numeric:
        info.target_range = [low, high]
        if integer:
            info.target_kind = "integer"
            unique = sorted({int(parse_number(t, decimal_comma) or 0) for t in labels})
            if len(labels) <= MAX_CLASSES:
                info.class_names = [str(v) for v in unique]
        else:
            info.target_kind = "number"
    else:
        info.target_kind = "text"
        if len(labels) > MAX_CLASSES:
            raise DatasetError(
                f"The target column has more than {MAX_CLASSES:,} different text values. "
                f"Classification supports at most {MAX_CLASSES} classes."
            )
        info.class_names = sorted(labels)
    return info


# --------------------------------------------------------------------------- #
# Image folders
# --------------------------------------------------------------------------- #


def _hidden(name: str) -> bool:
    return name.startswith(".") or name == "__MACOSX"


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS and not _hidden(path.name)


def _has_images(folder: Path) -> bool:
    return any(c.is_file() and _is_image(c) for c in folder.iterdir())


def find_image_root(path: str | Path) -> Path:
    """Skip wrapper folders: ``data.zip -> data/ -> cats/, dogs/`` gives ``data/``.

    A single folder that directly contains images is a class folder, not a wrapper.
    """
    root = Path(path)
    for _ in range(5):
        children = [c for c in root.iterdir() if not _hidden(c.name)]
        dirs = [c for c in children if c.is_dir()]
        if len(dirs) == 1 and not _has_images(root) and not _has_images(dirs[0]):
            root = dirs[0]
        else:
            break
    return root


def list_image_files(root: str | Path) -> dict[str, list[Path]]:
    """Map every class folder (sorted by name) to its image files (sorted, recursive)."""
    root = Path(root)
    result: dict[str, list[Path]] = {}
    for class_dir in sorted(c for c in root.iterdir() if c.is_dir() and not _hidden(c.name)):
        files = sorted(
            p
            for p in class_dir.rglob("*")
            if p.is_file()
            and _is_image(p)
            and not any(_hidden(part) for part in p.relative_to(class_dir).parts)
        )
        if files:
            result[class_dir.name] = files
    return result


def inspect_image_folder(path: str | Path, name: str | None = None) -> DatasetInfo:
    """Check a folder with one sub-folder of images per class."""
    path = Path(path)
    if not path.is_dir():
        raise DatasetError(f"'{path}' is not a folder.")
    root = find_image_root(path)
    classes = list_image_files(root)
    if not classes:
        if any(_is_image(p) for p in root.iterdir()):
            raise DatasetError(
                "The images must be sorted into one folder per class, "
                "e.g. cats/001.jpg and dogs/001.jpg."
            )
        raise DatasetError(
            "No images found. Supported formats: " + ", ".join(IMAGE_EXTENSIONS) + "."
        )
    if len(classes) < 2:
        only = next(iter(classes))
        raise DatasetError(
            f"Found only one class folder ('{only}'). Classification needs at least two "
            "classes, one folder per class."
        )
    if len(classes) > MAX_CLASSES:
        raise DatasetError(f"Found {len(classes)} class folders; at most {MAX_CLASSES} allowed.")
    counts = [len(files) for files in classes.values()]
    if sum(counts) < MIN_SAMPLES:
        raise DatasetError(f"The dataset needs at least {MIN_SAMPLES} images.")
    subdir = root.relative_to(path).as_posix()
    return DatasetInfo(
        kind="images",
        name=name or path.name,
        num_samples=sum(counts),
        class_names=list(classes),
        class_counts=counts,
        subdir="" if subdir == "." else subdir,
    )


def extract_image_zip(
    zip_path: str | Path,
    dest: str | Path,
    *,
    max_files: int = 200_000,
    max_bytes: int = 4 * 1024**3,
) -> int:
    """Safely extract the images of a ZIP file into ``dest`` and return their number.

    Only image files are extracted. Absolute paths, ``..`` components and symbolic links
    are rejected, hidden files and ``__MACOSX`` folders are skipped, and the number of
    files and the uncompressed size are limited (also when the ZIP lies about sizes).
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    total = 0
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                name = member.filename.replace("\\", "/")
                parts = PurePosixPath(name).parts
                if name.startswith("/") or ".." in parts or (parts and ":" in parts[0]):
                    raise DatasetError(f"The ZIP file contains an unsafe path: '{name[:80]}'.")
                if any(_hidden(part) for part in parts):
                    continue
                if PurePosixPath(name).suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise DatasetError("The ZIP file contains symbolic links.")
                count += 1
                if count > max_files:
                    raise DatasetError(f"The ZIP file contains more than {max_files:,} images.")
                target = dest.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, target.open("wb") as out:
                    while chunk := src.read(1024 * 1024):
                        total += len(chunk)
                        if total > max_bytes:
                            raise DatasetError(
                                "The images in the ZIP file are larger than "
                                f"{max_bytes / 1024**3:.1f} GB when unpacked."
                            )
                        out.write(chunk)
    except (zipfile.BadZipFile, zlib.error, EOFError) as exc:
        shutil.rmtree(dest, ignore_errors=True)
        raise DatasetError("This is not a valid ZIP file.") from exc
    except (DatasetError, OSError, RuntimeError, NotImplementedError) as exc:
        shutil.rmtree(dest, ignore_errors=True)
        if isinstance(exc, DatasetError):
            raise
        raise DatasetError(f"The ZIP file cannot be extracted: {exc}") from exc
    if count == 0:
        shutil.rmtree(dest, ignore_errors=True)
        raise DatasetError(
            "The ZIP file contains no images. Supported formats: "
            + ", ".join(IMAGE_EXTENSIONS)
            + "."
        )
    return count
