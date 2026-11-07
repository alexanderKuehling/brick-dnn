"""Saving imported tables to disk and loading them again, fast and without pickle.

Layout of a table directory::

    table.json         {"version": 1, "rows": n, "columns": [{"name", "kind", "file", ...}]}
    col_0000.npy       numeric / boolean: float64; datetime: int64 nanoseconds (NaT = min int)
    col_0001.npy       categorical / text: int32 codes, -1 for missing ...
    col_0001.json      ... plus the list of distinct strings the codes refer to
    split.npy          int8 split per row (only if the table has one)

Everything is loaded with ``numpy.load(..., allow_pickle=False)``, so a tampered file can
never execute code. Imported tables have no split (it is assigned by the pipeline).
``table.json`` is written last (atomically), so a half-written table is never loaded.
"""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .table import KINDS, NUMERIC_KINDS, DataError, Table

FORMAT_VERSION = 1
INDEX_FILE = "table.json"
SPLIT_FILE = "split.npy"
# Tables kept in memory by load_table_cached.
CACHE_SIZE = 4

DAMAGED = "The stored data is damaged. Import the file again."
NOT_FOUND = "The stored data was not found. Import the file again."

_cache: OrderedDict[tuple[str, int, int, int], Table] = OrderedDict()
_cache_lock = threading.Lock()


def save_table(table: Table, directory: str | Path) -> int:
    """Write ``table`` into ``directory`` (created if needed). Returns the bytes written."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = 0
    entries: list[dict[str, Any]] = []
    files: set[str] = set()
    for position, name in enumerate(table.columns):
        kind = table.kinds[name]
        stem = f"col_{position:04d}"
        entry: dict[str, Any] = {"name": name, "kind": kind, "file": f"{stem}.npy"}
        series = table.df[name]
        if kind in NUMERIC_KINDS:
            data = series.to_numpy(dtype=np.float64)
        elif kind == "datetime":
            data = series.to_numpy(dtype="datetime64[ns]").view(np.int64)
        else:
            codes, uniques = pd.factorize(series.to_numpy(dtype=object), use_na_sentinel=True)
            data = codes.astype(np.int32)
            strings = [str(v) for v in uniques]
            entry["strings"] = f"{stem}.json"
            # ASCII escapes keep every Python string intact (even lone surrogates).
            written += _write_bytes(directory / entry["strings"], json.dumps(strings).encode())
            files.add(entry["strings"])
        written += _write_array(directory / entry["file"], data)
        files.add(entry["file"])
        entries.append(entry)

    index: dict[str, Any] = {"version": FORMAT_VERSION, "rows": table.n_rows, "columns": entries}
    if table.split is not None:
        written += _write_array(directory / SPLIT_FILE, np.asarray(table.split, dtype=np.int8))
        files.add(SPLIT_FILE)
        index["split"] = SPLIT_FILE

    # Remove the files of an older, wider table in the same place.
    for path in directory.glob("col_*"):
        if path.name not in files:
            path.unlink(missing_ok=True)
    if SPLIT_FILE not in files:
        (directory / SPLIT_FILE).unlink(missing_ok=True)

    payload = json.dumps(index, indent=1).encode()  # ASCII: safe for any column name
    temporary = directory / f".{INDEX_FILE}.tmp"
    written += _write_bytes(temporary, payload)
    os.replace(temporary, directory / INDEX_FILE)
    return written


def _write_bytes(path: Path, data: bytes) -> int:
    path.write_bytes(data)
    return len(data)


def _write_array(path: Path, array: np.ndarray) -> int:
    with path.open("wb") as f:
        np.save(f, np.ascontiguousarray(array), allow_pickle=False)
        return f.tell()


def load_table(directory: str | Path) -> Table:
    """Load a table written by :func:`save_table`. Raises :class:`DataError` if the files
    are missing or damaged ("The stored data is damaged. Import the file again.")."""
    directory = Path(directory)
    try:
        index = json.loads((directory / INDEX_FILE).read_bytes())
    except FileNotFoundError:
        raise DataError(NOT_FOUND) from None
    except (OSError, ValueError):
        raise DataError(DAMAGED) from None
    try:
        return _load(directory, index)
    except DataError:
        raise
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        raise DataError(DAMAGED) from None


def _load(directory: Path, index: Any) -> Table:
    if not isinstance(index, dict) or index.get("version") != FORMAT_VERSION:
        raise DataError(DAMAGED)
    rows = index["rows"]
    if not isinstance(rows, int) or rows < 0 or not isinstance(index["columns"], list):
        raise DataError(DAMAGED)
    columns: dict[str, Any] = {}
    kinds: dict[str, str] = {}
    for entry in index["columns"]:
        name, kind = entry["name"], entry["kind"]
        if not isinstance(name, str) or kind not in KINDS or name in columns:
            raise DataError(DAMAGED)
        data = _read_array(directory, entry["file"], rows)
        if kind in NUMERIC_KINDS:
            if data.dtype != np.float64:
                raise DataError(DAMAGED)
            columns[name] = data
        elif kind == "datetime":
            if data.dtype != np.int64:
                raise DataError(DAMAGED)
            columns[name] = data.view("datetime64[ns]")
        else:
            strings = json.loads(_safe_path(directory, entry["strings"]).read_bytes())
            if data.dtype != np.int32 or not isinstance(strings, list):
                raise DataError(DAMAGED)
            if not all(isinstance(s, str) for s in strings):
                raise DataError(DAMAGED)
            if len(data) and (data.min() < -1 or data.max() >= len(strings)):
                raise DataError(DAMAGED)
            # The last slot is None, so code -1 picks it.
            lookup = np.empty(len(strings) + 1, dtype=object)
            lookup[: len(strings)] = strings
            lookup[-1] = None
            columns[name] = lookup[data]
        kinds[name] = kind

    split = None
    if index.get("split"):
        split = _read_array(directory, index["split"], rows)
        if split.dtype != np.int8 or (len(split) and (split.min() < 0 or split.max() > 2)):
            raise DataError(DAMAGED)
    df = pd.DataFrame(columns, index=pd.RangeIndex(rows))
    return Table(df, kinds, split)


def _safe_path(directory: Path, name: Any) -> Path:
    """A file inside ``directory``; names from table.json must not point elsewhere."""
    if not isinstance(name, str) or not name or "/" in name or "\\" in name or name[0] == ".":
        raise DataError(DAMAGED)
    return directory / name


def _read_array(directory: Path, name: Any, rows: int) -> np.ndarray:
    data = np.load(_safe_path(directory, name), allow_pickle=False)
    if data.ndim != 1 or len(data) != rows:
        raise DataError(DAMAGED)
    return data


def load_table_cached(directory: str | Path) -> Table:
    """:func:`load_table` with a small in-process LRU cache (a few tables), keyed by the
    directory and the modification time of ``table.json``. Callers must not change the
    returned table (tables are treated as immutable anyway)."""
    path = Path(directory).resolve()
    try:
        info = (path / INDEX_FILE).stat()
    except FileNotFoundError:
        raise DataError(NOT_FOUND) from None
    except OSError:
        raise DataError(DAMAGED) from None
    # table.json is replaced on every save, so its inode changes even within one mtime tick.
    key = (str(path), info.st_mtime_ns, info.st_ino, info.st_size)
    with _cache_lock:
        table = _cache.get(key)
        if table is not None:
            _cache.move_to_end(key)
            return table.copy()
    # Load outside the lock: other tables stay available meanwhile.
    table = load_table(path)
    with _cache_lock:
        for old in [k for k in _cache if k[0] == key[0] and k != key]:
            del _cache[old]
        _cache[key] = table
        _cache.move_to_end(key)
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return table.copy()


def clear_cache(directory: str | Path | None = None) -> None:
    """Forget cached tables: all of them, or those loaded from ``directory``."""
    with _cache_lock:
        if directory is None:
            _cache.clear()
            return
        path = str(Path(directory).resolve())
        for key in [k for k in _cache if k[0] == path]:
            del _cache[key]
