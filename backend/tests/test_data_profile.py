"""Profiles (statistics, histograms, correlations, target analysis, warnings with fixes) and
row samples for the analysis views."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest

from brickdnn.data import profile as profile_module
from brickdnn.data.profile import MAX_BINS, MIN_BINS, profile_table, sample_table
from brickdnn.data.steps import StepContext, get_step
from brickdnn.data.table import Table

DAY_MS = 86_400_000


def ms(text: str) -> int:
    return int(pd.Timestamp(text).value // 1_000_000)


def small_table() -> Table:
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 100, np.nan],
            "f": [1.0, 0.0, 1.0, 1.0, np.nan, 0.0],
            "c": ["a", "b", "a", None, "a", "c"],
            "d": pd.to_datetime(
                ["2024-01-01", "2024-01-02", None, "2024-01-03", "2024-01-04", "2024-01-05"]
            ),
            "y": [1.0, 2.0, 2.0, 3.0, 3.0, 3.0],
        }
    )
    return Table.from_frame(
        df, {"x": "numeric", "f": "boolean", "c": "categorical", "d": "datetime", "y": "numeric"}
    )


def by_name(profile: dict, name: str) -> dict:
    return next(c for c in profile["columns"] if c["name"] == name)


def codes(profile: dict) -> set[tuple[str, str | None]]:
    return {(w["code"], w["column"]) for w in profile["warnings"]}


def warning(profile: dict, code: str, column: str | None) -> dict:
    return next(w for w in profile["warnings"] if w["code"] == code and w["column"] == column)


def json_safe(value) -> None:
    json.dumps(value, allow_nan=False)


# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #


def test_overview_numbers():
    table = small_table()
    profile = profile_table(table)
    json_safe(profile)
    assert profile["n_rows"] == 6 and profile["n_columns"] == 5
    assert profile["missing_cells"] == 4
    assert profile["duplicate_rows"] == 0 and profile["analysis_rows"] == 6
    assert profile["split_counts"] is None and profile["target"] is None
    assert profile["memory_bytes"] == int(table.df.memory_usage(deep=True).sum())
    assert [c["name"] for c in profile["columns"]] == ["x", "f", "c", "d", "y"]


def test_numeric_column_stats():
    x = by_name(profile_table(small_table()), "x")
    assert (x["kind"], x["integer"], x["count"], x["missing"], x["unique"]) == (
        "numeric",
        True,
        5,
        1,
        5,
    )
    stats = x["stats"]
    values = pd.Series([1, 2, 3, 4, 100.0])
    assert stats["mean"] == 22.0
    assert stats["std"] == pytest.approx(values.std())
    assert (stats["min"], stats["q25"], stats["median"], stats["q75"], stats["max"]) == (
        1.0,
        2.0,
        3.0,
        4.0,
        100.0,
    )
    assert stats["q01"] == pytest.approx(1.04) and stats["q99"] == pytest.approx(96.16)
    assert stats["q05"] == pytest.approx(values.quantile(0.05))
    assert stats["q95"] == pytest.approx(values.quantile(0.95))
    assert stats["skew"] == pytest.approx(values.skew())
    assert stats["kurtosis"] == pytest.approx(values.kurt())
    assert (stats["zeros"], stats["negative"]) == (0, 0)
    assert stats["iqr_outliers"] == 1  # 100 > q75 + 1.5 × IQR = 7
    assert stats["z_outliers"] == 0
    assert x["top"] is None and x["mean_length"] is None and x["target_association"] is None
    histogram = x["histogram"]
    assert sum(histogram["counts"]) == 5
    assert len(histogram["edges"]) == len(histogram["counts"]) + 1
    assert MIN_BINS <= len(histogram["counts"]) <= MAX_BINS
    assert histogram["edges"][0] <= 1 and histogram["edges"][-1] >= 100
    widths = np.diff(histogram["edges"])
    assert np.allclose(widths, widths[0]) and float(widths[0]).is_integer()  # whole-number bins


def test_small_integer_histogram_has_one_bin_per_value():
    y = by_name(profile_table(small_table()), "y")
    assert y["histogram"] == {"edges": [0.5, 1.5, 2.5, 3.5], "counts": [1, 2, 3]}


def test_freedman_diaconis_bins():
    rng = np.random.default_rng(1)
    table = Table.from_frame(
        pd.DataFrame({"a": rng.normal(size=5000), "b": rng.normal(size=5000) * 1e-6})
    )
    for name in ("a", "b"):
        histogram = by_name(profile_table(table), name)["histogram"]
        assert MIN_BINS <= len(histogram["counts"]) <= MAX_BINS
        assert sum(histogram["counts"]) == 5000
        assert histogram["edges"][0] == table.df[name].min()
        assert histogram["edges"][-1] == table.df[name].max()


def test_constant_numeric_column():
    table = Table.from_frame(pd.DataFrame({"k": [7.0, 7.0, 7.0]}))
    k = by_name(profile_table(table), "k")
    assert k["histogram"] == {"edges": [6.5, 7.5], "counts": [3]}
    assert k["stats"]["std"] == 0 and k["stats"]["skew"] is None and k["stats"]["kurtosis"] is None


def test_boolean_column():
    f = by_name(profile_table(small_table()), "f")
    assert (f["kind"], f["integer"], f["count"], f["missing"], f["unique"]) == (
        "boolean",
        True,
        5,
        1,
        2,
    )
    assert f["top"] == [{"value": "1", "count": 3}, {"value": "0", "count": 2}]
    assert f["other_count"] == 0 and f["histogram"] is None
    assert f["stats"]["mean"] == pytest.approx(0.6) and f["stats"]["zeros"] == 2


def test_categorical_column():
    c = by_name(profile_table(small_table()), "c")
    assert (c["count"], c["missing"], c["unique"], c["integer"]) == (5, 1, 3, False)
    # Ties keep the order of first appearance.
    assert c["top"] == [
        {"value": "a", "count": 3},
        {"value": "b", "count": 1},
        {"value": "c", "count": 1},
    ]
    assert c["other_count"] == 0 and c["mean_length"] == 1.0
    assert c["stats"] is None and c["histogram"] is None


def test_top_values_are_limited():
    values = [f"v{i}" for i in range(30) for _ in range(i + 1)]
    table = Table.from_frame(pd.DataFrame({"c": values}), {"c": "categorical"})
    c = by_name(profile_table(table), "c")
    assert len(c["top"]) == profile_module.TOP_VALUES
    assert c["top"][0] == {"value": "v29", "count": 30}
    assert c["other_count"] == len(values) - sum(t["count"] for t in c["top"])


def test_datetime_column_in_milliseconds():
    d = by_name(profile_table(small_table()), "d")
    assert (d["count"], d["missing"], d["unique"], d["integer"]) == (5, 1, 5, False)
    stats = d["stats"]
    assert stats["min"] == ms("2024-01-01") and stats["max"] == ms("2024-01-05")
    assert stats["median"] == ms("2024-01-03") and stats["mean"] == ms("2024-01-03")
    assert stats["std"] == pytest.approx(np.std(np.arange(5) * DAY_MS, ddof=1))
    assert (stats["zeros"], stats["negative"]) == (0, 0)
    histogram = d["histogram"]
    assert histogram["edges"][0] == ms("2024-01-01") and histogram["edges"][-1] == ms("2024-01-05")
    assert sum(histogram["counts"]) == 5 and d["top"] is None


def test_text_column():
    notes = [f"A long customer note number {i} with some free text." for i in range(60)]
    table = Table.from_frame(pd.DataFrame({"notes": notes}), {"notes": "text"})
    profile = profile_table(table)
    notes_profile = by_name(profile, "notes")
    assert notes_profile["unique"] == 60 and len(notes_profile["top"]) == 20
    assert notes_profile["other_count"] == 40
    assert notes_profile["mean_length"] == pytest.approx(np.mean([len(n) for n in notes]))
    assert warning(profile, "text", "notes")["fix"]["type"] == "drop_columns"


def test_empty_tables_and_columns():
    empty = Table.from_frame(
        pd.DataFrame({"a": pd.Series([], dtype=float), "b": pd.Series([], dtype=object)}),
        {"a": "numeric", "b": "categorical"},
    )
    for table in (empty, Table(pd.DataFrame(index=pd.RangeIndex(4)))):
        profile = profile_table(table, target="a")
        json_safe(profile)
        json_safe(sample_table(table, 10))
        assert profile["warnings"] == [] and profile["correlations"] == []
    table = Table.from_frame(
        pd.DataFrame({"a": [np.nan, np.nan], "d": [pd.NaT, pd.NaT], "c": [None, None]}),
        {"a": "numeric", "d": "datetime", "c": "categorical"},
    )
    profile = profile_table(table)
    json_safe(profile)
    for column in profile["columns"]:
        assert column["count"] == 0 and column["missing"] == 2 and column["stats"] is None
        assert column["histogram"] is None
    assert codes(profile) >= {("high_missing", "a"), ("high_missing", "d"), ("high_missing", "c")}


def test_json_safety_with_extreme_values():
    table = Table.from_frame(
        pd.DataFrame({"x": [1e308, -1e308, 1e308, 0.0], "c": ["a", "b", "a", "b"]})
    )
    profile = profile_table(table, target="c", task="classification")
    json_safe(profile)
    json_safe(sample_table(table, 3))
    assert by_name(profile, "x")["stats"]["std"] is None  # overflows to infinity


# --------------------------------------------------------------------------- #
# Correlations and target
# --------------------------------------------------------------------------- #


def test_correlation_matrices():
    a = np.arange(1.0, 51.0)
    table = Table.from_frame(
        pd.DataFrame({"a": a, "double": 2 * a, "cube": (a - 25) ** 3, "k": 1.0, "s": "x"})
    )
    profile = profile_table(table)
    pearson, spearman = profile["correlations"]
    assert pearson["method"] == "pearson" and spearman["method"] == "spearman"
    assert pearson["columns"] == ["a", "double", "cube", "k"]
    matrix = np.array(pearson["matrix"], dtype=object)
    assert matrix[0, 1] == 1.0 and matrix[1, 0] == 1.0 and matrix[0, 0] == 1.0
    assert 0.8 < matrix[0, 2] < 1.0
    assert spearman["matrix"][0][2] == 1.0  # monotonic
    assert all(v is None for v in pearson["matrix"][3]) and pearson["matrix"][0][3] is None
    assert (matrix == matrix.T).all()


def test_correlations_need_two_columns_and_limit_columns(monkeypatch):
    assert profile_table(Table.from_frame(pd.DataFrame({"a": [1.0, 2.0]})))["correlations"] == []
    monkeypatch.setattr(profile_module, "MAX_CORRELATION_COLUMNS", 2)
    df = pd.DataFrame(
        {"a": [1.0, np.nan, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 5.0], "c": [2.0, 1.0, 0.0, 3.0]}
    )
    names = profile_table(Table.from_frame(df))["correlations"][0]["columns"]
    assert names == ["b", "c"]  # the most complete columns, in table order


def test_regression_target():
    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    df = pd.DataFrame(
        {
            "copy": y,
            "noise": rng.normal(size=300),
            "group": np.where(y > 0, "high", "low"),
            "random_group": rng.choice(["a", "b", "c"], 300),
            "when": pd.Timestamp("2024-01-01") + pd.to_timedelta(np.arange(300), "D"),
            "y": y,
        }
    )
    profile = profile_table(Table.from_frame(df), target="y", task="regression")
    json_safe(profile)
    assert by_name(profile, "copy")["target_association"] == {"method": "pearson", "value": 1.0}
    assert by_name(profile, "noise")["target_association"]["value"] < 0.2
    assert by_name(profile, "group")["target_association"]["method"] == "eta"
    assert by_name(profile, "group")["target_association"]["value"] > 0.7
    assert by_name(profile, "random_group")["target_association"]["value"] < 0.3
    assert by_name(profile, "when")["target_association"] is None
    assert by_name(profile, "y")["target_association"] is None
    target = profile["target"]
    assert target["column"] == "y" and target["task"] == "regression"
    assert target["classes"] is None and target["imbalance_ratio"] is None
    assert sum(target["histogram"]["counts"]) == 300
    assert target["stats"]["mean"] == pytest.approx(y.mean())


def test_classification_target():
    rng = np.random.default_rng(0)
    labels = rng.choice(["yes", "no"], 400, p=[0.2, 0.8])
    df = pd.DataFrame(
        {
            "copy": labels,
            "flag": (labels == "yes").astype(float),
            "score": (labels == "yes") * 10.0 + rng.normal(size=400) * 0.01,
            "noise": rng.normal(size=400),
            "random_group": rng.choice(["a", "b", "c"], 400),
            "label": labels,
        }
    )
    table = Table.from_frame(df, {"copy": "categorical", "flag": "boolean", "label": "categorical"})
    profile = profile_table(table, target="label", task="classification")
    assert by_name(profile, "copy")["target_association"] == {"method": "cramers_v", "value": 1.0}
    assert by_name(profile, "flag")["target_association"] == {"method": "cramers_v", "value": 1.0}
    score = by_name(profile, "score")["target_association"]
    assert score["method"] == "eta" and score["value"] == pytest.approx(1.0, abs=1e-3)
    assert by_name(profile, "noise")["target_association"]["value"] < 0.2
    assert by_name(profile, "random_group")["target_association"]["value"] < 0.2
    target = profile["target"]
    yes = int((labels == "yes").sum())
    assert target["classes"] == [
        {"value": "no", "count": 400 - yes},
        {"value": "yes", "count": yes},
    ]
    assert target["imbalance_ratio"] == pytest.approx((400 - yes) / yes)
    assert target["histogram"] is None and target["stats"] is None
    assert warning(profile, "imbalanced", "label")["fix"] == {
        "type": "balance_classes",
        "params": {"method": "oversample"},
        "label": "Balance classes (oversample)",
    }


def test_numeric_class_labels_are_sorted_numerically():
    table = Table.from_frame(pd.DataFrame({"x": [1.0, 2, 3, 4, 5], "y": [10.0, 2, 2, 1, 10]}))
    target = profile_table(table, target="y", task="classification")["target"]
    assert target["classes"] == [
        {"value": "1", "count": 1},
        {"value": "2", "count": 2},
        {"value": "10", "count": 2},
    ]
    assert target["imbalance_ratio"] == 2.0


def test_task_is_resolved_and_unknown_targets_ignored():
    table = Table.from_frame(pd.DataFrame({"x": [1.0, 2, 3, 4], "y": ["a", "b", "a", "b"]}))
    assert profile_table(table, target="y")["target"]["task"] == "classification"
    profile = profile_table(table, target="gone", task="regression")
    assert profile["target"] is None
    assert all(c["target_association"] is None for c in profile["columns"])


# --------------------------------------------------------------------------- #
# Warnings
# --------------------------------------------------------------------------- #


def messy_table(n: int = 200) -> Table:
    rng = np.random.default_rng(3)
    numbers = rng.integers(1, 1000, n).astype(str).astype(object)
    numbers[:6] = "n.a."
    zeros = np.where(rng.random(n) < 0.7, 0.0, rng.random(n) * 10)
    sparse = rng.normal(size=n)
    sparse[: int(n * 0.5)] = np.nan
    some_missing = rng.normal(size=n)
    some_missing[:20] = np.nan
    city = rng.choice(["Berlin", "Hamburg"], n).astype(object)
    city[:30] = None
    base = rng.normal(size=n)
    outliers = rng.normal(size=n)
    outliers[:8] = 50.0
    df = pd.DataFrame(
        {
            "customer_id": np.arange(1000, 1000 + n),
            "points": np.arange(n) * 7 + 3,
            "constant": "same",
            "sparse": sparse,
            "some_missing": some_missing,
            "city": city,
            "income": np.exp(rng.normal(10, 1.5, n)),
            "debt": -np.exp(rng.normal(5, 1.5, n)) + 10,
            "outliers": outliers,
            "zeros": zeros,
            "mixed": numbers,
            "many": [f"c{i % 80}" for i in range(n)],
            "base": base,
            "twin": base * 2 + 0.001 * rng.normal(size=n),
            "label": rng.choice(["a", "b"], n),
        }
    )
    kinds = {"constant": "categorical", "mixed": "categorical", "many": "categorical"}
    return Table.from_frame(df, kinds)


def test_warnings_and_fixes():
    table = messy_table()
    profile = profile_table(table, target="label", task="classification")
    json_safe(profile)
    found = codes(profile)
    assert found >= {
        ("id_like", "customer_id"),
        ("constant", "constant"),
        ("high_missing", "sparse"),
        ("high_missing", "some_missing"),
        ("high_missing", "city"),
        ("skewed", "income"),
        ("skewed", "debt"),
        ("outliers", "outliers"),
        ("many_zeros", "zeros"),
        ("mixed_types", "mixed"),
        ("high_cardinality", "many"),
        ("high_correlation", "twin"),
    }
    # "points" is unique but not consecutive and not named like an ID.
    assert ("id_like", "points") not in found
    assert ("high_missing", "label") not in found

    assert warning(profile, "id_like", "customer_id") == {
        "code": "id_like",
        "severity": "warning",
        "column": "customer_id",
        "message": "customer_id has a different value in every row; it looks like an ID.",
        "fix": {
            "type": "drop_columns",
            "params": {"columns": ["customer_id"]},
            "label": "Drop customer_id",
        },
    }
    assert warning(profile, "constant", "constant")["message"] == (
        "constant has the same value in every row (same)."
    )
    sparse = warning(profile, "high_missing", "sparse")
    assert sparse["severity"] == "warning" and sparse["fix"]["type"] == "drop_columns"
    assert sparse["message"] == "sparse is missing in 50 % of the rows."
    some = warning(profile, "high_missing", "some_missing")
    assert some["severity"] == "info"
    assert some["fix"]["params"] == {"columns": ["some_missing"], "strategy": "median"}
    assert some["message"] == "some_missing is missing in 10 % of the rows (20 rows)."
    city = warning(profile, "high_missing", "city")["fix"]
    assert city["params"]["strategy"] == "most_frequent"
    income = warning(profile, "skewed", "income")
    assert income["fix"]["params"] == {"columns": ["income"], "function": "log1p"}
    assert income["message"].startswith("income is strongly skewed (skewness ")
    assert warning(profile, "skewed", "debt")["fix"]["params"]["function"] == "yeo_johnson"
    assert warning(profile, "outliers", "outliers")["fix"] == {
        "type": "clip",
        "params": {"columns": ["outliers"], "mode": "quantiles", "lower_q": 0.01, "upper_q": 0.99},
        "label": "Clip outliers to the 1st–99th percentile",
    }
    assert warning(profile, "many_zeros", "zeros")["fix"] is None
    mixed = warning(profile, "mixed_types", "mixed")
    assert mixed["fix"]["params"] == {"columns": ["mixed"], "to": "numeric", "decimal_comma": False}
    assert "97 % of its values are numbers (others: 'n.a.')" in mixed["message"]
    assert warning(profile, "high_cardinality", "many")["fix"]["params"] == {
        "columns": ["many"],
        "min_share": 0.01,
        "label": "Other",
    }
    assert "r = 1.00" in warning(profile, "high_correlation", "twin")["message"]
    # Warnings come before infos.
    severities = [w["severity"] for w in profile["warnings"]]
    assert severities == sorted(severities, key=lambda s: s != "warning")


def test_id_like_columns():
    n = 30
    df = pd.DataFrame(
        {
            "Unnamed: 0": np.arange(n),
            "number": np.arange(1, n + 1),
            "year": np.arange(1990, 1990 + n),
            "customerId": np.arange(5000, 5000 + n) * 3,
            "uuid": [f"3f2c8a5e-0b7d-4c1e-9a6f-{i:012d}" for i in range(n)],
            "name": [f"P{i}" for i in range(n)],
            "y": np.random.default_rng(0).normal(size=n),
        }
    )
    found = {c for code, c in codes(profile_table(Table.from_frame(df))) if code == "id_like"}
    assert found == {"Unnamed: 0", "number", "customerId", "uuid", "name"}


def test_every_fix_is_a_valid_step():
    table = messy_table()
    profile = profile_table(table, target="label", task="classification")
    ctx = StepContext(target="label", task="classification")
    fixes = [w["fix"] for w in profile["warnings"] if w["fix"]]
    assert len(fixes) >= 10
    for fix in fixes:
        step = get_step(fix["type"])(fix["params"])
        step.clean(table, ctx)  # raises DataError for unknown or invalid settings
        assert isinstance(fix["label"], str) and fix["label"]


def test_one_warning_family_per_column():
    profile = profile_table(messy_table(), target="label", task="classification")
    drops = [
        w["column"] for w in profile["warnings"] if w["fix"] and w["fix"]["type"] == "drop_columns"
    ]
    assert len(drops) == len(set(drops))
    # A column to drop gets no further advice.
    assert not any(
        w["column"] == "sparse" and w["code"] != "high_missing" for w in profile["warnings"]
    )


def test_target_missing_duplicates_and_decimal_comma_mixed_types():
    values = ["1,5", "2,25", "3,0", "4,75"] * 10 + ["?"] * 2
    df = pd.DataFrame({"m": values, "y": [1.0, 2.0] * 20 + [np.nan, np.nan]})
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)
    profile = profile_table(
        Table.from_frame(df, {"m": "categorical"}), target="y", task="regression"
    )
    assert warning(profile, "mixed_types", "m")["fix"]["params"]["decimal_comma"] is True
    target_missing = warning(profile, "target_missing", "y")
    assert target_missing["fix"] == {
        "type": "drop_missing",
        "params": {"columns": ["y"], "how": "any"},
        "label": "Drop rows without y",
    }
    assert "missing in 2 rows" in target_missing["message"]
    assert profile["duplicate_rows"] == 40  # 45 rows, only 5 different ones
    duplicates = warning(profile, "duplicates", None)
    assert duplicates["message"] == "40 rows are exact duplicates of other rows."
    assert duplicates["fix"]["params"] == {"columns": [], "keep": "first"}


def test_leading_zero_codes_are_no_mixed_types():
    table = Table.from_frame(
        pd.DataFrame({"zip": ["01067", "10115", "04109"] * 5}), {"zip": "categorical"}
    )
    assert ("mixed_types", "zip") not in codes(profile_table(table))


def test_analysis_sample_for_large_tables(monkeypatch):
    monkeypatch.setattr(profile_module, "ANALYSIS_ROWS", 50)
    df = pd.DataFrame({"a": np.arange(200.0) % 10, "b": np.arange(200.0) % 10})
    profile = profile_table(Table.from_frame(df))
    assert profile["analysis_rows"] == 50 and profile["n_rows"] == 200
    assert profile["columns"][0]["count"] == 200  # statistics stay exact
    assert profile["duplicate_rows"] > 0
    assert warning(profile, "duplicates", None)["message"].startswith("In a sample of 50 rows, ")
    assert profile_table(Table.from_frame(df)) == profile  # the sample is seeded


# --------------------------------------------------------------------------- #
# Samples
# --------------------------------------------------------------------------- #


def test_sample_values_and_types():
    table = small_table()
    sample = sample_table(table, 100)
    json_safe(sample)
    assert (sample["n_total"], sample["rows"], sample["seed"]) == (6, 6, 0)
    assert sample["index"] == [0, 1, 2, 3, 4, 5] and sample["split"] is None
    values = {c["name"]: c["values"] for c in sample["columns"]}
    assert [c["kind"] for c in sample["columns"]] == [
        "numeric",
        "boolean",
        "categorical",
        "datetime",
        "numeric",
    ]
    assert values["x"] == [1, 2, 3, 4, 100, None]
    assert all(type(v) is int for v in values["x"][:5])
    assert values["f"] == [1, 0, 1, 1, None, 0]
    assert values["c"] == ["a", "b", "a", None, "a", "c"]
    assert values["d"][:2] == [ms("2024-01-01"), ms("2024-01-02")] and values["d"][2] is None
    floats = sample_table(Table.from_frame(pd.DataFrame({"z": [0.5, 2.0, np.nan]})), 5)
    assert floats["columns"][0]["values"] == [0.5, 2.0, None]


def test_sample_is_seeded_sorted_and_has_split():
    n = 1000
    table = Table.from_frame(pd.DataFrame({"i": np.arange(n, dtype=float)}))
    table = table.with_split(np.arange(n) % 3)
    first = sample_table(table, 50, seed=7)
    assert first == sample_table(table, 50, seed=7)
    assert first["index"] != sample_table(table, 50, seed=8)["index"]
    assert first["rows"] == 50 and first["n_total"] == n
    assert first["index"] == sorted(first["index"]) and len(set(first["index"])) == 50
    assert first["columns"][0]["values"] == first["index"]
    assert first["split"] == [("train", "val", "test")[i % 3] for i in first["index"]]
    assert sample_table(table, 0)["index"] == []


def test_profile_speed():
    """A 200,000 × 12 table profiles in a few seconds."""
    n = 200_000
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            **{f"x{i}": rng.normal(size=n) for i in range(5)},
            "count": rng.integers(0, 20, n).astype(float),
            "flag": rng.integers(0, 2, n).astype(float),
            "city": rng.choice(["Berlin", "Hamburg", "Köln", "Leipzig"], n),
            "code": [f"c{i}" for i in rng.integers(0, 500, n)],
            "day": pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 900, n), "D"),
            "note": [f"note {i} with a little free text in it, long" for i in range(n)],
            "label": rng.choice(["a", "b", "c"], n),
        }
    )
    table = Table.from_frame(df, {"flag": "boolean", "note": "text", "code": "categorical"})
    start = time.perf_counter()
    profile = profile_table(table, target="label", task="classification")
    elapsed = time.perf_counter() - start
    json_safe(profile)
    assert profile["analysis_rows"] == profile_module.ANALYSIS_ROWS
    assert elapsed < 8, f"profile took {elapsed:.1f} s"
