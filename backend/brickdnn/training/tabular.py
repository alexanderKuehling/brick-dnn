"""Tabular data for both engines (no PyTorch or NumPy needed): reading the rows of a
checked CSV file, the stored split of prepared datasets and the sizes of a random split.

The PyTorch engine (:mod:`.data`) and the scikit-learn engine (:mod:`.classical_engine`)
store the rows in their own arrays, but read and split them by the same rules.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path

from .datasets import SPLIT_FILE, DatasetInfo, open_csv, parse_number
from .reporting import TrainingError

CHANGED = "The CSV file has changed since it was checked. Upload it again."


def csv_columns(info: DatasetInfo) -> list[str]:
    """Feature columns + target column (named ``column_1`` … ``target`` without header)."""
    width = (info.num_features or 0) + 1
    columns = list(info.columns)
    if len(columns) != width:
        columns = [f"column_{i + 1}" for i in range(width - 1)] + ["target"]
    return columns


Row = tuple[int, list[float], str, float | None]


@contextlib.contextmanager
def csv_rows(path: Path, info: DatasetInfo) -> Iterator[tuple[str, int, Iterator[Row]]]:
    """Read the data rows of a CSV file checked by :func:`.datasets.inspect_csv`.

    Yields ``(delimiter, line number of the first data row, rows)``; every row is
    ``(line number, feature values, target cell, target as a number or None)``. The rows
    raise :class:`TrainingError` (with the line and column) when they do not fit the
    file's description, e.g. because it was changed.
    """
    rows_expected = info.num_samples or 0
    width = (info.num_features or 0) + 1
    if rows_expected < 2 or width < 2:
        raise TrainingError(CHANGED)
    columns = csv_columns(info)
    with open_csv(path) as (header, rows, delimiter):
        decimal_comma = delimiter != ","
        offset = 2 if header is not None else 1

        def parse() -> Iterator[Row]:
            for index, row in enumerate(rows):
                line = index + offset
                if index >= rows_expected:
                    raise TrainingError(CHANGED)
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
                yield line, values, cell, parse_number(cell, decimal_comma)

        yield delimiter, offset, parse()


def held_out_size(max_samples: int, share: float, train_share: float) -> int:
    """Validation or test samples that go with ``max_samples`` training samples, so that
    they keep the proportion ``share : train_share`` (none for a share of 0)."""
    if share <= 0:
        return 0
    return max(1, round(max_samples * share / train_share))


def has_validation(info: DatasetInfo, val_split: float) -> bool:
    """Whether a training on ``info`` gets validation data: the validation rows of the split
    of a prepared dataset (or of the folders of a segmentation), otherwise ``val_split``."""
    if info.predefined_split is not None:
        counts = [*info.predefined_split, 0, 0][:3]
        return int(counts[1]) > 0
    return val_split > 0


def val_size_for(max_samples: int, val_split: float) -> int:
    """Validation samples that go with ``max_samples`` training samples (same proportion)."""
    return held_out_size(max_samples, val_split, 1 - val_split)


def split_sizes(n: int, val_split: float, test_split: float) -> tuple[int, int, int]:
    """Numbers of training, validation and test samples: every part with a share above 0
    gets at least one (a share of 0 leaves its part empty)."""
    parts = ["training"]
    parts += ["validation"] if val_split > 0 else []
    parts += ["test"] if test_split > 0 else []
    if n < len(parts):
        raise TrainingError(
            f"The dataset needs at least {len(parts)} samples ({' + '.join(parts)})."
        )
    with_val = int(val_split > 0)
    n_test = min(max(1, round(n * test_split)), n - 1 - with_val) if test_split > 0 else 0
    n_val = min(max(1, round(n * val_split)), n - n_test - 1) if with_val else 0
    return n - n_val - n_test, n_val, n_test


def read_split(path: Path, n: int, expected: list[int] | None = None) -> bytes:
    """The split stored with a prepared CSV file (``split.bin`` next to it): one byte per
    data row, 0 training, 1 validation, 2 test. ``expected`` are the [train, val, test]
    counts from the dataset info. Validation and test rows are optional, training rows are
    not."""
    changed = TrainingError(
        "The split of this dataset (split.bin) does not match its rows. "
        "Create the dataset again in the Data tab."
    )
    try:
        raw = (path.parent / SPLIT_FILE).read_bytes()
    except OSError:
        raise TrainingError(
            "The split of this dataset (split.bin) is missing. Create the dataset again in the "
            "Data tab."
        ) from None
    if len(raw) != n or n == 0:
        raise changed
    counts = [raw.count(label) for label in (0, 1, 2)]
    if sum(counts) != n:  # other values than 0, 1 and 2
        raise changed
    if expected is not None and [int(c) for c in expected] != counts:
        raise changed
    if counts[0] == 0:
        raise TrainingError(
            "The split of this dataset has no training rows. "
            "Change the split in the Data tab and create the dataset again."
        )
    return raw
