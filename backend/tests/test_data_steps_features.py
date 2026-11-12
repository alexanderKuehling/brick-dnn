"""The feature engineering steps (brickdnn.data.steps.features): combining columns row-wise,
pairwise interactions and group statistics learned from the training rows."""

from __future__ import annotations

import json

import numpy as np
import pytest

from brickdnn.data.steps import StepContext, catalog
from brickdnn.data.steps import features as features_module
from brickdnn.data.table import TEST, TRAIN, VAL, DataError

from .data_helpers import apply, check_code, run, table, values

NAN = float("nan")


def fraud_table(split=None):
    """A small fraud-like table: amounts, balances, flags and categories."""
    return table(
        {
            "amount_usd": [10.0, 250.0, NAN, 40.0, 1200.0, 15.0],
            "account_balance_usd": [500.0, 0.0, 300.0, NAN, 100.0, 1500.0],
            "used_vpn": [True, False, True, False, True, False],
            "is_new_merchant": [1.0, 0.0, NAN, 0.0, 1.0, 0.0],
            "ip_country_mismatch": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "merchant_category": ["food", "travel", "food", None, "travel", "food"],
            "channel": ["online", "pos", "online", "pos", "online", "pos"],
            "is_fraud": ["no", "no", "no", "no", "yes", "no"],
        },
        kinds={"used_vpn": "boolean", "is_new_merchant": "boolean"},
        split=split,
    )


FLAGS = ["used_vpn", "is_new_merchant", "ip_country_mismatch"]
FRAUD_CTX = StepContext(target="is_fraud", task="classification")


def error_of(source, step_type, params, ctx=None) -> str:
    with pytest.raises(DataError) as info:
        apply(source, step_type, params, ctx)
    return str(info.value)


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #


def test_catalog_lists_the_feature_steps():
    data = catalog()
    json.dumps(data, allow_nan=False)
    groups = [g["id"] for g in data["groups"]]
    assert {"id": "features", "label": "Feature engineering"} in data["groups"]
    assert groups.index("transform") < groups.index("features") < groups.index("encode")
    specs = {s["type"]: s for s in data["steps"]}
    for step_type in ("combine_columns", "interactions", "group_features"):
        spec = specs[step_type]
        assert spec["group"] == "features" and spec["description"].endswith(".")
        for param in spec["params"]:
            if param["kind"] in ("select", "multiselect"):
                allowed = [o["value"] for o in param["options"]]
                defaults = (
                    param["default"] if param["kind"] == "multiselect" else [param["default"]]
                )
                assert set(defaults) <= set(allowed)
    assert specs["group_features"]["fitted"] is True
    assert specs["combine_columns"]["fitted"] is specs["interactions"]["fitted"] is False
    params = {p["name"]: p for p in specs["combine_columns"]["params"]}
    assert params["columns"]["column_kinds"] == ["numeric", "boolean"]
    assert params["operations"]["kind"] == "multiselect"
    params = {p["name"]: p for p in specs["group_features"]["params"]}
    assert params["group_by"]["column_kinds"] == ["categorical", "boolean"]
    assert params["value"]["kind"] == "column" and params["value"]["optional"] is True


# --------------------------------------------------------------------------- #
# Combine columns
# --------------------------------------------------------------------------- #


def test_combine_columns_counts_flags_and_sums():
    source = fraud_table()
    result = check_code(
        source,
        "combine_columns",
        {"columns": FLAGS, "operations": ["nonzero", "sum", "count"], "name": "risk"},
    )
    assert values(result.table, "risk_nonzero") == [3, 0, 1, 0, 3, 0]
    assert values(result.table, "risk_sum") == [3, 0, 1, 0, 3, 0]
    assert values(result.table, "risk_count") == [3, 3, 2, 3, 3, 3]
    assert result.table.kinds["risk_nonzero"] == "numeric"
    assert result.summary == (
        "Added 'risk_sum', 'risk_count' and 'risk_nonzero' (row-wise sum, count of values, "
        "count of non-zero values of 3 columns); 1 row with missing inputs uses the values "
        "present."
    )
    assert result.fitted is None and not result.warnings


def test_combine_columns_statistics_and_missing_values():
    source = table({"a": [1.0, NAN, NAN, -2.0], "b": [3.0, 4.0, NAN, 0.0], "c": [5.0, 1, NAN, 2]})
    ops = ["sum", "mean", "min", "max", "std", "count", "nonzero", "product"]
    result = check_code(source, "combine_columns", {"columns": ["a", "b", "c"], "operations": ops})
    t = result.table
    assert values(t, "a_b_c_sum") == [9, 5, None, 0]
    assert values(t, "a_b_c_mean") == [3, 2.5, None, 0]
    assert values(t, "a_b_c_min") == [1, 1, None, -2]
    assert values(t, "a_b_c_max") == [5, 4, None, 2]
    assert values(t, "a_b_c_std")[:3] == [pytest.approx(np.std([1, 3, 5])), 1.5, None]
    assert values(t, "a_b_c_count") == [3, 2, 0, 3]
    assert values(t, "a_b_c_nonzero") == [3, 2, 0, 2]
    assert values(t, "a_b_c_product") == [15, 4, None, 0]
    assert "(1 without any value)" in result.summary

    strict = check_code(
        source, "combine_columns", {"columns": ["a", "b"], "operations": ["sum"], "missing": "any"}
    )
    assert values(strict.table, "a_b_sum") == [4, None, None, -2]
    assert "2 rows with a missing input get missing results" in strict.summary


def test_combine_columns_names_overflow_and_warnings():
    source = table({"a": [1e200, 2.0, 3.0], "b": [1e200, 2.0, 3.0], "c": [1.0, 1.0, 1.0]})
    result = check_code(
        source, "combine_columns", {"columns": ["a", "b"], "operations": ["product"], "name": "ab"}
    )
    assert values(result.table, "ab") == [None, 4, 9]
    assert "1 value too large for a number, set missing" in result.summary
    many = table({c: [1.0, 1.0] for c in "abcd"})
    result, _, _ = apply(many, "combine_columns", {"columns": list("abcd"), "operations": ["max"]})
    assert result.table.columns[-1] == "row_max"
    assert result.warnings == ["Every value of 'row_max' is 1: it tells the model nothing."]


def test_combine_columns_errors():
    source = fraud_table()
    assert "at least two columns" in error_of(
        source, "combine_columns", {"columns": ["amount_usd"]}
    )
    assert "numeric or boolean columns needed" in error_of(
        source, "combine_columns", {"columns": ["amount_usd", "channel"]}
    )
    assert "at least one operation" in error_of(
        source, "combine_columns", {"columns": FLAGS, "operations": []}
    )
    text = error_of(
        source,
        "combine_columns",
        {"columns": ["amount_usd", "is_fraud"]},
        StepContext(target="is_fraud"),
    )
    assert "numeric or boolean" in text  # a category target is not even offered
    numeric_target = table({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    text = error_of(
        numeric_target, "combine_columns", {"columns": ["x", "y"]}, StepContext(target="y")
    )
    assert "'y' is the target" in text and "target leakage" in text
    clash = error_of(
        source, "combine_columns", {"columns": FLAGS, "operations": ["sum"], "name": "channel"}
    )
    assert "There is already a column 'channel'" in clash


# --------------------------------------------------------------------------- #
# Pairwise interactions
# --------------------------------------------------------------------------- #


def test_interactions_values_and_safe_division():
    source = fraud_table()
    result = check_code(
        source,
        "interactions",
        {
            "columns": ["amount_usd", "account_balance_usd"],
            "operations": ["multiply", "divide", "subtract", "add"],
        },
    )
    t = result.table
    new = ["amount_usd*account_balance_usd", "amount_usd/account_balance_usd"]
    new += ["amount_usd-account_balance_usd", "amount_usd+account_balance_usd"]
    assert t.columns[-4:] == new
    assert values(t, new[0]) == [5000, 0, None, None, 120000, 22500]
    # 250 / 0 and missing values give missing values
    assert values(t, new[1]) == [0.02, None, None, None, 12, 0.01]
    assert values(t, new[2]) == [-490, 250, None, None, 1100, -1485]
    assert values(t, new[3]) == [510, 250, None, None, 1300, 1515]
    assert result.summary == (
        "Added 4 columns from 1 pair of columns (a × b, a / b, a − b, a + b); 8 values missing "
        "because an input is missing; 1 division by zero gave missing values."
    )


def test_interactions_pairs_directions_and_second_list():
    source = table({"a": [1.0, 2.0], "b": [4.0, 8.0], "c": [0.5, 0.25], "flag": [True, False]})
    result = check_code(
        source,
        "interactions",
        {"columns": ["a", "b", "c"], "operations": ["divide"], "both_directions": True},
    )
    assert result.table.columns[4:] == ["a/b", "b/a", "a/c", "c/a", "b/c", "c/b"]
    assert values(result.table, "c/b") == [0.125, 0.03125]
    # Each column of the first list with each of the second; a pair only once.
    result = check_code(
        source,
        "interactions",
        {"columns": ["a", "b"], "columns_b": ["b", "a", "flag"], "operations": ["multiply"]},
    )
    assert result.table.columns[4:] == ["a*b", "a*flag", "b*flag"]
    assert values(result.table, "b*flag") == [4, 0]


def test_interactions_cap_and_errors():
    source = table({f"x{i}": [float(i), 1.0] for i in range(25)})
    columns = source.columns
    text = error_of(source, "interactions", {"columns": columns})
    assert "300 columns (300 pairs of columns)" in text
    assert text.startswith("Most new columns: ")
    assert "more than the limit of 200" in text and "at most 2,000" in text
    result, _, _ = apply(source, "interactions", {"columns": columns, "max_columns": 300})
    assert len(result.table.columns) == 25 + 300
    with pytest.raises(DataError, match="at most 2000"):
        apply(source, "interactions", {"columns": columns, "max_columns": 5000})
    assert "at least two columns" in error_of(source, "interactions", {"columns": ["x1"]})
    assert "at least two columns" in error_of(
        source, "interactions", {"columns": ["x1"], "columns_b": ["x1"]}
    )
    fraud = fraud_table()
    assert "categorical column" in error_of(
        fraud, "interactions", {"columns": ["amount_usd", "merchant_category"]}
    )
    numbers = table({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    text = error_of(
        numbers, "interactions", {"columns": ["x"], "columns_b": ["y"]}, StepContext(target="y")
    )
    assert text.startswith("Combine with: 'y' is the target") and "target leakage" in text


# --------------------------------------------------------------------------- #
# Group statistics
# --------------------------------------------------------------------------- #

GROUP_SPLIT = [TRAIN, TRAIN, TRAIN, TRAIN, VAL, TEST, TRAIN, VAL]


def group_table(val_amounts=(1000.0, 7.0)):
    return table(
        {
            "merchant_category": ["food", "food", "travel", "travel", "food", "game", None, "x"],
            "amount_usd": [10.0, 30.0, 100.0, 300.0, val_amounts[0], val_amounts[1], 50.0, 0.0],
            "is_fraud": ["no", "no", "no", "yes", "no", "no", "no", "yes"],
        },
        split=np.array(GROUP_SPLIT),
    )


def test_group_statistics_are_learned_from_the_training_rows_only():
    params = {
        "group_by": ["merchant_category"],
        "value": "amount_usd",
        "stats": ["mean", "median", "std", "min", "max", "count", "rows"],
    }
    result = check_code(group_table(), "group_features", params, FRAUD_CTX)
    t = result.table
    # Training rows: food 10, 30; travel 100, 300; the row without a category (50).
    assert values(t, "amount_usd_mean_by_merchant_category") == [20, 20, 200, 200, 20, 98, 98, 98]
    assert values(t, "amount_usd_median_by_merchant_category")[:4] == [20, 20, 200, 200]
    assert values(t, "amount_usd_std_by_merchant_category")[:4] == [10, 10, 100, 100]
    assert values(t, "amount_usd_min_by_merchant_category")[:4] == [10, 10, 100, 100]
    assert values(t, "amount_usd_max_by_merchant_category")[:4] == [30, 30, 300, 300]
    # Counts of unseen groups (and rows without a group) are 0.
    assert values(t, "amount_usd_count_by_merchant_category") == [2, 2, 2, 2, 2, 0, 0, 0]
    assert values(t, "rows_by_merchant_category") == [2, 2, 2, 2, 2, 0, 0, 0]
    assert result.fitted["groups"] == 2
    assert result.fitted["_groups"] == [["food"], ["travel"]]
    assert result.fitted["overall"]["mean"] == 98  # (10 + 30 + 100 + 300 + 50) / 5
    assert "2 validation/test rows with a group that is not in the training rows" in result.summary
    assert "1 row without a group value gets the overall training value" in result.summary

    # Other validation/test values change nothing that was learned.
    other = check_code(group_table((5.0, 99999.0)), "group_features", params, FRAUD_CTX)
    assert other.fitted == result.fitted
    assert values(other.table, "amount_usd_mean_by_merchant_category") == values(
        t, "amount_usd_mean_by_merchant_category"
    )


def test_difference_and_ratio_to_the_group_mean():
    params = {
        "group_by": ["merchant_category"],
        "value": "amount_usd",
        "stats": ["diff_mean", "ratio_mean"],
        "unseen": "missing",
    }
    result = check_code(group_table(), "group_features", params, FRAUD_CTX)
    t = result.table
    assert t.columns[-2:] == [
        "amount_usd_diff_mean_by_merchant_category",
        "amount_usd_ratio_mean_by_merchant_category",
    ]
    assert values(t, t.columns[-2]) == [-10, 10, -100, 100, 980, None, None, None]
    assert values(t, t.columns[-1]) == [0.5, 1.5, 0.5, 1.5, 50, None, None, None]
    assert "gets missing values" in result.summary and "6 values missing" in result.summary

    # A group mean of 0 gives a missing ratio; with "overall" unseen rows use the overall mean.
    zero = table({"g": ["a", "a", "b"], "x": [-1.0, 1.0, 4.0]})
    result = check_code(
        zero, "group_features", {"group_by": ["g"], "value": "x", "stats": ["ratio_mean"]}
    )
    assert values(result.table, "x_ratio_mean_by_g") == [None, None, 1]


def test_group_statistics_min_rows_and_several_group_columns():
    source = table(
        {
            "category": ["a", "a", "a", "b", "b", "c"],
            "online": [True, True, False, True, True, False],
            "amount": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
        }
    )
    result = check_code(
        source,
        "group_features",
        {
            "group_by": ["category", "online"],
            "value": "amount",
            "stats": ["mean", "rows"],
            "min_rows": 2,
        },
    )
    t = result.table
    # (a, 1) and (b, 1) have two rows; (a, 0) and (c, 0) one: they count as unseen.
    overall = np.mean([1, 3, 5, 7, 9, 11])
    assert values(t, "amount_mean_by_category_online") == [2, 2, overall, 8, 8, overall]
    assert values(t, "rows_by_category_online") == [2, 2, 1, 2, 2, 1]
    assert result.fitted["_groups"] == [["a", "0"], ["a", "1"], ["b", "1"], ["c", "0"]]
    assert result.fitted["booleans"] == ["online"]
    assert "2 groups with fewer than 2 training values count as unseen" in result.summary


def test_rows_per_group_needs_no_value_column():
    source = group_table()
    result = check_code(
        source, "group_features", {"group_by": ["merchant_category"], "stats": ["rows"]}
    )
    assert values(result.table, "rows_by_merchant_category") == [2, 2, 2, 2, 2, 0, 0, 0]
    assert result.fitted["overall"] == {}
    text = error_of(
        source, "group_features", {"group_by": ["merchant_category"], "stats": ["mean"]}
    )
    assert "Value column: choose the number column for the mean" in text


def test_group_statistics_forbid_the_target():
    source = table({"g": ["a", "b", "a"], "x": [1.0, 2.0, 3.0], "y": [0.0, 1.0, 0.5]})
    ctx = StepContext(target="y", task="regression")
    text = error_of(source, "group_features", {"group_by": ["g"], "value": "y"}, ctx)
    assert "'y' is the target" in text and "target leakage" in text
    assert "Target mean" in text  # where to go instead
    fraud = group_table()
    text = error_of(
        fraud, "group_features", {"group_by": ["is_fraud"], "value": "amount_usd"}, FRAUD_CTX
    )
    assert "Group by: 'is_fraud' is the target" in text
    # In a pipeline the error is the step's report.
    result = run(
        source, ("group_features", {"group_by": ["g"], "value": "y"}), settings={"target": "y"}
    )
    assert result.failed and "target leakage" in result.reports[0]["error"]


def test_group_statistics_errors_and_warnings(monkeypatch):
    source = group_table()
    assert "Group by: choose a category column" in error_of(
        source, "group_features", {"value": "amount_usd"}
    )
    assert "numeric or boolean columns needed" in error_of(
        source,
        "group_features",
        {"group_by": ["merchant_category"], "value": "merchant_category"},
    )
    assert "categorical or boolean columns needed" in error_of(
        source, "group_features", {"group_by": ["amount_usd"], "value": "amount_usd"}
    )
    monkeypatch.setattr(features_module, "MAX_GROUPS", 1)
    assert "2 groups; at most 1" in error_of(
        source, "group_features", {"group_by": ["merchant_category"], "value": "amount_usd"}
    )
    monkeypatch.undo()
    # An ID-like group column: every group has a single row.
    ids = table({"card": [f"c{i}" for i in range(6)], "amount": [float(i) for i in range(6)]})
    result, _, _ = apply(
        ids, "group_features", {"group_by": ["card"], "value": "amount", "stats": ["mean"]}
    )
    assert any("only one training row" in w for w in result.warnings)
    empty = table({"g": [None, None], "x": [1.0, 2.0]}, kinds={"g": "categorical"})
    assert "no values in the training rows" in error_of(
        empty, "group_features", {"group_by": ["g"], "value": "x"}
    )
