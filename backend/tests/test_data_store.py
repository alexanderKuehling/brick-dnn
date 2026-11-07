"""Column-wise storage of tables (no pickle) and the in-process LRU cache."""

from __future__ import annotations

import json
import os
import threading

import numpy as np
import pandas as pd
import pytest

from brickdnn.data import store
from brickdnn.data.store import (
    CACHE_SIZE,
    clear_cache,
    load_table,
    load_table_cached,
    save_table,
)
from brickdnn.data.table import DataError, Table


@pytest.fixture(autouse=True)
def _empty_cache():
    clear_cache()
    yield
    clear_cache()


def sample_table() -> Table:
    df = pd.DataFrame(
        {
            "x": [1.5, np.nan, -3.0, 1e300],
            "flag": [1.0, 0.0, np.nan, 1.0],
            "when": [
                pd.Timestamp("2024-01-31 12:00:00.123456789"),
                pd.NaT,
                pd.Timestamp("1970-01-01"),
                pd.Timestamp("1900-05-06"),
            ],
            "city": ["Köln", None, "Köln", "München 🏙"],
            "notes": ["", 'line\nbreak "quoted"', None, "\ud800 lone surrogate"],
        }
    )
    kinds = {
        "x": "numeric",
        "flag": "boolean",
        "when": "datetime",
        "city": "categorical",
        "notes": "text",
    }
    return Table.from_frame(df, kinds)


def assert_same(a: Table, b: Table) -> None:
    assert a.kinds == b.kinds and a.columns == b.columns
    pd.testing.assert_frame_equal(a.df, b.df)
    for name in ("city", "notes"):
        if name in a.kinds:
            assert a.df[name].tolist() == b.df[name].tolist()  # None stays None, "" stays ""


def test_round_trip_all_kinds(tmp_path):
    table = sample_table()
    written = save_table(table, tmp_path / "t")
    files = sorted(p.name for p in (tmp_path / "t").iterdir())
    assert files == [
        "col_0000.npy",
        "col_0001.npy",
        "col_0002.npy",
        "col_0003.json",
        "col_0003.npy",
        "col_0004.json",
        "col_0004.npy",
        "table.json",
    ]
    assert written == sum(p.stat().st_size for p in (tmp_path / "t").iterdir())
    loaded = load_table(tmp_path / "t")
    assert_same(table, loaded)
    assert loaded.split is None
    assert loaded.df["when"].dtype == "datetime64[ns]"
    assert loaded.df["city"].dtype == object
    index = json.loads((tmp_path / "t" / "table.json").read_text())
    assert index["version"] == 1 and index["rows"] == 4
    assert index["columns"][3] == {
        "name": "city",
        "kind": "categorical",
        "file": "col_0003.npy",
        "strings": "col_0003.json",
    }
    codes = np.load(tmp_path / "t" / "col_0003.npy", allow_pickle=False)
    assert codes.dtype == np.int32 and codes.tolist() == [0, -1, 0, 1]


def test_round_trip_with_split_and_unicode_names(tmp_path):
    table = Table.from_frame(pd.DataFrame({"Größe (cm)": [1.0, 2.0, 3.0]}))
    table = table.with_split(np.array([0, 1, 2], dtype=np.int8))
    save_table(table, tmp_path)
    loaded = load_table(tmp_path)
    assert loaded.columns == ["Größe (cm)"]
    assert loaded.split.tolist() == [0, 1, 2] and loaded.split.dtype == np.int8


@pytest.mark.parametrize(
    ("frame", "kinds"),
    [
        (
            pd.DataFrame({"a": pd.Series([], dtype=float), "b": pd.Series([], dtype=object)}),
            {"a": "numeric", "b": "categorical"},
        ),
        (pd.DataFrame(index=pd.RangeIndex(5)), {}),
        (pd.DataFrame(index=pd.RangeIndex(0)), {}),
        (pd.DataFrame({"a": [None, None], "d": [None, None]}), {"a": "text", "d": "datetime"}),
    ],
)
def test_round_trip_empty_tables(tmp_path, frame, kinds):
    table = Table.from_frame(frame, kinds)
    save_table(table, tmp_path)
    loaded = load_table(tmp_path)
    assert loaded.n_rows == table.n_rows and loaded.kinds == table.kinds
    pd.testing.assert_frame_equal(loaded.df, table.df)


def test_saving_again_replaces_old_files(tmp_path):
    save_table(sample_table().with_split(np.zeros(4, dtype=np.int8)), tmp_path)
    smaller = Table.from_frame(pd.DataFrame({"only": ["a", "b"]}))
    save_table(smaller, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "col_0000.json",
        "col_0000.npy",
        "table.json",
    ]
    assert_same(smaller, load_table(tmp_path))


def test_missing_and_damaged_files(tmp_path):
    with pytest.raises(DataError, match="was not found. Import the file again."):
        load_table(tmp_path / "nothing")
    with pytest.raises(DataError, match="was not found"):
        load_table_cached(tmp_path / "nothing")

    save_table(sample_table(), tmp_path)
    index_path = tmp_path / "table.json"
    original = index_path.read_text()

    def damage(change) -> None:
        index = json.loads(original)
        change(index)
        index_path.write_text(json.dumps(index))
        with pytest.raises(DataError, match="The stored data is damaged. Import the file again."):
            load_table(tmp_path)

    damage(lambda i: i.update(version=99))
    damage(lambda i: i.update(rows=5))
    damage(lambda i: i["columns"][0].update(kind="weird"))
    damage(lambda i: i["columns"][0].update(file="../secret.npy"))
    damage(lambda i: i["columns"][3].update(strings="/etc/passwd"))
    damage(lambda i: i["columns"][1].update(name=i["columns"][0]["name"]))
    damage(lambda i: i["columns"][0].update(file="missing.npy"))
    index_path.write_text("{not json")
    with pytest.raises(DataError, match="damaged"):
        load_table(tmp_path)

    # Codes pointing outside the string list.
    index_path.write_text(original)
    np.save(tmp_path / "col_0003.npy", np.array([0, 5, 0, 0], dtype=np.int32))
    with pytest.raises(DataError, match="damaged"):
        load_table(tmp_path)


def test_never_unpickles(tmp_path):
    save_table(sample_table(), tmp_path)
    # A pickled object array in place of a column file must not be loaded.
    with (tmp_path / "col_0000.npy").open("wb") as f:
        np.save(f, np.array([object(), 1, 2, 3], dtype=object), allow_pickle=True)
    with pytest.raises(DataError, match="damaged"):
        load_table(tmp_path)


def test_wrong_dtype_is_damage(tmp_path):
    save_table(sample_table(), tmp_path)
    np.save(tmp_path / "col_0000.npy", np.arange(4, dtype=np.int64))
    with pytest.raises(DataError, match="damaged"):
        load_table(tmp_path)


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


def count_loads(monkeypatch) -> list[str]:
    calls: list[str] = []
    real = store.load_table

    def counting(directory):
        calls.append(str(directory))
        return real(directory)

    monkeypatch.setattr(store, "load_table", counting)
    return calls


def test_cache_reuses_tables_and_sees_changes(tmp_path, monkeypatch):
    calls = count_loads(monkeypatch)
    save_table(sample_table(), tmp_path)
    first = load_table_cached(tmp_path)
    second = load_table_cached(tmp_path / ".." / tmp_path.name)  # same directory, resolved
    assert len(calls) == 1
    assert_same(first, second)
    second.kinds["x"] = "categorical"  # a copy: the cached table is not affected
    assert load_table_cached(tmp_path).kinds["x"] == "numeric"

    changed = Table.from_frame(pd.DataFrame({"y": [1.0, 2.0]}))
    save_table(changed, tmp_path)
    assert_same(load_table_cached(tmp_path), changed)
    assert len(calls) == 2
    # Only one entry per directory is kept.
    assert sum(1 for key in store._cache if key[0] == str(tmp_path.resolve())) == 1


def test_cache_invalidates_when_table_json_changes(tmp_path):
    save_table(Table.from_frame(pd.DataFrame({"y": [1.0]})), tmp_path)
    load_table_cached(tmp_path)
    # Rewritten in the same nanosecond tick is still noticed (inode and size change).
    stat = (tmp_path / "table.json").stat()
    save_table(Table.from_frame(pd.DataFrame({"z": [1.0, 2.0, 3.0]})), tmp_path)
    os.utime(tmp_path / "table.json", ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert load_table_cached(tmp_path).columns == ["z"]


def test_cache_is_lru(tmp_path, monkeypatch):
    calls = count_loads(monkeypatch)
    directories = [tmp_path / f"t{i}" for i in range(CACHE_SIZE + 1)]
    for i, directory in enumerate(directories):
        save_table(Table.from_frame(pd.DataFrame({"v": [float(i)]})), directory)
    for directory in directories[:CACHE_SIZE]:
        load_table_cached(directory)
    load_table_cached(directories[0])  # most recently used now
    load_table_cached(directories[-1])  # evicts directories[1]
    assert len(calls) == CACHE_SIZE + 1
    load_table_cached(directories[0])
    assert len(calls) == CACHE_SIZE + 1
    load_table_cached(directories[1])
    assert len(calls) == CACHE_SIZE + 2
    clear_cache(directories[1])
    load_table_cached(directories[1])
    assert len(calls) == CACHE_SIZE + 3


def test_cache_is_thread_safe(tmp_path):
    for i in range(3):
        save_table(Table.from_frame(pd.DataFrame({"v": np.arange(1000.0) + i})), tmp_path / str(i))
    errors: list[BaseException] = []

    def work(worker: int) -> None:
        try:
            for step in range(30):
                i = (worker + step) % 3
                table = load_table_cached(tmp_path / str(i))
                assert table.df["v"].iloc[0] == i
        except BaseException as exc:  # pragma: no cover - only on failure
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(w,)) for w in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(store._cache) <= CACHE_SIZE
