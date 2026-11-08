"""The in-memory dataset that every preparation step reads and returns.

A :class:`Table` is a pandas DataFrame plus two things pandas does not know:

* the **kind** of every column (``numeric``, ``boolean``, ``datetime``, ``categorical`` or
  ``text``), which decides what the UI offers and how a column is analysed, and
* the **split**: for every row whether it belongs to the training (0), validation (1) or test
  (2) data. Steps that learn something (imputation values, clipping bounds, categories ...)
  learn it from the training rows only, so validation and test data stay unseen.

Column storage is normalised, whatever pandas version is installed:

=============  =====================================================================
kind           dtype of the column
=============  =====================================================================
numeric        ``float64``, ``NaN`` for missing values
boolean        ``float64`` with 0.0 / 1.0, ``NaN`` for missing values
datetime       ``datetime64[ns]`` without time zone, ``NaT`` for missing values
categorical    ``object`` holding ``str`` values, ``None`` for missing values
text           ``object`` holding ``str`` values, ``None`` for missing values
=============  =====================================================================

Tables are treated as immutable: a step never changes the table it gets, it returns a new
one (pandas copy-on-write makes the copies cheap).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Copy-on-write: shallow copies are safe and cheap, and pandas 3 behaves the same way.
pd.set_option("mode.copy_on_write", True)

KINDS = ("numeric", "boolean", "datetime", "categorical", "text")
NUMERIC_KINDS = ("numeric", "boolean")
STRING_KINDS = ("categorical", "text")

TRAIN, VAL, TEST = 0, 1, 2

# A single step may add at most this many columns (e.g. one-hot encoding an ID column by
# mistake): every preview runs all steps again, so such a step would make the editor slow.
# Tables themselves may be wider (imports have no column limit by default).
MAX_NEW_COLUMNS = 2000
SPLIT_NAMES = ("train", "val", "test")

# Without a better hint, string columns with more distinct values than this (and mostly
# unique values) are free text rather than categories.
TEXT_MIN_UNIQUE = 50
TEXT_UNIQUE_SHARE = 0.5
TEXT_MEAN_LENGTH = 30


class DataError(ValueError):
    """Unusable data or step settings. The message is shown to the user as is."""


def _as_str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if value is pd.NaT or value is pd.NA:
        return None
    return value if isinstance(value, str) else str(value)


def normalize(series: pd.Series, kind: str) -> pd.Series:
    """Convert ``series`` to the storage format of ``kind`` (see the module docstring).

    Values that cannot be converted become missing. Raises :class:`DataError` for an
    unknown kind.
    """
    if kind not in KINDS:
        raise DataError(f"Unknown column type '{kind}'.")
    name = series.name
    if kind in NUMERIC_KINDS:
        if pd.api.types.is_bool_dtype(series):
            result = series.astype("float64")
        else:
            result = pd.to_numeric(series, errors="coerce").astype("float64")
        result = result.where(np.isfinite(result), np.nan)
    elif kind == "datetime":
        if pd.api.types.is_datetime64_any_dtype(series):
            result = series
            if getattr(result.dt, "tz", None) is not None:
                result = result.dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            result = pd.to_datetime(series, errors="coerce")
        result = result.astype("datetime64[ns]")
    else:
        values = [_as_str_or_none(v) for v in series.tolist()]
        result = pd.Series(np.array(values, dtype=object), index=series.index)
    result.name = name
    return result


def infer_kind(series: pd.Series) -> str:
    """Kind of an already typed column (from its dtype and, for strings, its values).

    This does not parse strings: the CSV reader decides about numbers and dates in string
    columns itself (see :mod:`brickdnn.data.io`).
    """
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    present = series.dropna()
    if len(present) == 0:
        return "categorical"
    strings = present.astype(str)
    unique = strings.nunique()
    mean_length = float(strings.str.len().mean())
    if mean_length > TEXT_MEAN_LENGTH or (
        unique > TEXT_MIN_UNIQUE and unique / len(strings) > TEXT_UNIQUE_SHARE and mean_length > 15
    ):
        return "text"
    return "categorical"


@dataclass
class Table:
    """A dataset: a DataFrame, the kind of every column and the split of every row."""

    df: pd.DataFrame
    kinds: dict[str, str] = field(default_factory=dict)
    # int8 per row: TRAIN, VAL or TEST. None before the split is assigned.
    split: np.ndarray | None = None

    def __post_init__(self) -> None:
        columns = [str(c) for c in self.df.columns]
        if len(set(columns)) != len(columns):
            dupes = sorted({c for c in columns if columns.count(c) > 1})
            raise DataError("Column names must be unique: " + ", ".join(dupes[:5]) + ".")
        if list(self.df.columns) != columns:
            self.df = self.df.set_axis(columns, axis=1)
        if not isinstance(self.df.index, pd.RangeIndex) or self.df.index.start != 0:
            self.df = self.df.reset_index(drop=True)
        missing = [c for c in columns if c not in self.kinds]
        for name in missing:
            self.kinds[name] = infer_kind(self.df[name])
        self.kinds = {c: self.kinds[c] for c in columns}
        if self.split is not None:
            self.split = np.asarray(self.split, dtype=np.int8)
            if len(self.split) != len(self.df):
                raise DataError("Internal error: the split does not match the number of rows.")

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        kinds: dict[str, str] | None = None,
        split: np.ndarray | None = None,
    ) -> Table:
        """Build a table and normalise every column to the storage format of its kind."""
        kinds = dict(kinds or {})
        df = df.reset_index(drop=True)
        df.columns = [str(c) for c in df.columns]
        columns = {}
        final_kinds = {}
        for name in df.columns:
            kind = kinds.get(name) or infer_kind(df[name])
            columns[name] = normalize(df[name], kind)
            final_kinds[name] = kind
        frame = pd.DataFrame(columns, index=pd.RangeIndex(len(df)))
        return cls(frame, final_kinds, split)

    # ------------------------------------------------------------------ basics

    @property
    def n_rows(self) -> int:
        return len(self.df)

    @property
    def columns(self) -> list[str]:
        return list(self.df.columns)

    def kind(self, column: str) -> str:
        try:
            return self.kinds[column]
        except KeyError:
            raise DataError(f"There is no column '{column}'.") from None

    def columns_of(self, kinds: Iterable[str], exclude: Iterable[str] = ()) -> list[str]:
        """Names of the columns with one of ``kinds``, in table order, without ``exclude``."""
        wanted, skip = set(kinds), set(exclude)
        return [c for c in self.columns if self.kinds[c] in wanted and c not in skip]

    def column_info(self) -> list[dict[str, str]]:
        """``[{"name", "kind"}, ...]`` as sent to the UI."""
        return [{"name": c, "kind": self.kinds[c]} for c in self.columns]

    def train_mask(self) -> np.ndarray:
        """Boolean mask of the training rows (all rows while no split is assigned)."""
        if self.split is None:
            return np.ones(self.n_rows, dtype=bool)
        return self.split == TRAIN

    def train_df(self) -> pd.DataFrame:
        """The training rows (all rows while no split is assigned)."""
        return self.df[self.train_mask()] if self.split is not None else self.df

    def split_counts(self) -> list[int] | None:
        """``[train, val, test]`` row counts, or None while no split is assigned."""
        if self.split is None:
            return None
        counts = np.bincount(self.split.astype(np.int64), minlength=3)
        return [int(v) for v in counts[:3]]

    # ------------------------------------------------------------------ derived tables

    def copy(self) -> Table:
        """A copy that can be changed freely (cheap thanks to copy-on-write)."""
        split = self.split.copy() if self.split is not None else None
        return Table(self.df.copy(deep=False), dict(self.kinds), split)

    def take(self, rows: np.ndarray | Sequence[int] | pd.Series) -> Table:
        """The rows selected by a boolean mask or by positions, renumbered from 0."""
        rows = np.asarray(rows)
        if rows.dtype == bool:
            if len(rows) != self.n_rows:
                raise DataError("Internal error: the row mask does not match the table.")
            positions = np.flatnonzero(rows)
        else:
            positions = rows.astype(np.int64)
        df = self.df.iloc[positions].reset_index(drop=True)
        split = self.split[positions] if self.split is not None else None
        return Table(df, dict(self.kinds), split)

    def with_column(
        self,
        name: str,
        values: pd.Series | np.ndarray | Sequence[Any],
        kind: str | None = None,
        *,
        after: str | None = None,
    ) -> Table:
        """Add or replace a column. A new column goes to the end, or right after ``after``.

        ``values`` are normalised to ``kind`` (inferred from the values if omitted).
        """
        series = (
            pd.Series(values, index=self.df.index) if not isinstance(values, pd.Series) else values
        )
        series = series.reset_index(drop=True)
        if len(series) != self.n_rows:
            raise DataError("Internal error: the new column does not match the number of rows.")
        series.name = name
        kind = kind or infer_kind(series)
        series = normalize(series, kind)
        df = self.df.copy(deep=False)
        kinds = dict(self.kinds)
        if name in df.columns:
            df[name] = series
        else:
            position = df.columns.get_loc(after) + 1 if after in df.columns else len(df.columns)
            df.insert(position, name, series)
        kinds[name] = kind
        return Table(df, kinds, self.split.copy() if self.split is not None else None)

    def drop(self, columns: Iterable[str]) -> Table:
        drop = [c for c in columns if c in self.kinds]
        df = self.df.drop(columns=drop)
        kinds = {c: k for c, k in self.kinds.items() if c not in drop}
        return Table(df, kinds, self.split.copy() if self.split is not None else None)

    def rename(self, mapping: dict[str, str]) -> Table:
        df = self.df.rename(columns=mapping)
        kinds = {mapping.get(c, c): k for c, k in self.kinds.items()}
        return Table(df, kinds, self.split.copy() if self.split is not None else None)

    def with_split(self, split: np.ndarray | None) -> Table:
        return Table(self.df, dict(self.kinds), split)
