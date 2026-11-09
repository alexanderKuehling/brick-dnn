"""The sanity checks of the Readiness view (brickdnn.data.checks): every check finds what it
should, stays quiet on ordinary data, and its fix is a step that runs and solves it."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from brickdnn.data import checks
from brickdnn.data.pipeline import (
    assign_split,
    clean_settings,
    readiness,
    resolve_task,
)
from brickdnn.data.table import TEST, TRAIN, VAL, Table

from .data_helpers import run, table, values

ISSUE_KEYS = {"code", "severity", "column", "columns", "message", "fix"}


def check(
    t: Table, target: str, *, parts: np.ndarray | None = None, **options: Any
) -> dict[str, Any]:
    """Readiness of ``t`` with a split (assigned like a pipeline does, unless given)."""
    settings = clean_settings({"target": target, **options})
    task = resolve_task(t, settings)
    t = t.with_split(parts if parts is not None else assign_split(t, settings, task))
    result = readiness(t, settings, task)
    json.dumps(result, allow_nan=False)
    assert all(set(i) == ISSUE_KEYS for i in result["issues"])
    return result


def found(result: dict[str, Any], code: str) -> list[dict[str, Any]]:
    return [i for i in result["issues"] if i["code"] == code]


def one(result: dict[str, Any], code: str) -> dict[str, Any]:
    issues = found(result, code)
    assert len(issues) == 1, [i["message"] for i in result["issues"]]
    return issues[0]


def codes(result: dict[str, Any]) -> set[str]:
    return {i["code"] for i in result["issues"]}


def after_fixes(source: Table, target: str, *fixes: dict[str, Any], **options: Any):
    """Run the fixes as steps; return the readiness of the result and the result."""
    settings = {"target": target, **options}
    result = run(source, *[(f["type"], f["params"]) for f in fixes], settings=settings)
    assert not result.failed, [r["error"] for r in result.reports]
    check = readiness(result.table, result.settings, result.task)
    return check, result


@pytest.fixture
def rng():
    return np.random.default_rng(7)


# --------------------------------------------------------------------------- #
# Target leakage
# --------------------------------------------------------------------------- #


def test_leakage_of_a_copy_and_of_a_perfectly_correlated_feature(rng):
    n = 300
    y = rng.normal(100, 20, n)
    source = table(
        {
            "x": rng.normal(size=n),
            "y_copy": y.copy(),
            "cents": y * 100 + rng.normal(0, 1, n),
            "related": y + rng.normal(0, 15, n),  # r ≈ 0.8: a good feature, no leak
            "y": y,
        }
    )
    result = check(source, "y")
    leaks = {i["column"]: i for i in found(result, "leakage")}
    assert set(leaks) == {"y_copy", "cents"}
    assert "y_copy equals the target y in almost every row" in leaks["y_copy"]["message"]
    assert "cents correlates almost perfectly with the target" in leaks["cents"]["message"]
    assert leaks["cents"]["severity"] == "warning" and leaks["cents"]["columns"] == ["cents"]
    assert leaks["cents"]["fix"] == {
        "type": "drop_columns",
        "params": {"columns": ["cents"]},
        "label": "Drop cents",
    }
    fixed, _ = after_fixes(source, "y", leaks["y_copy"]["fix"], leaks["cents"]["fix"])
    assert "leakage" not in codes(fixed)


def test_leakage_of_class_codes_and_categories(rng):
    n = 400
    animal = rng.choice(np.array(["bird", "cat", "dog"], dtype=object), n)
    source = table(
        {
            "size": rng.normal(size=n) + (animal == "dog"),
            "label_code": pd.Series(animal).map({"bird": 0.0, "cat": 1.0, "dog": 2.0}),
            "group": pd.Series(animal).map({"bird": "A", "cat": "B", "dog": "C"}),
            "colour": rng.choice(np.array(["red", "blue"], dtype=object), n),
            "animal": animal,
        }
    )
    result = check(source, "animal")
    leaks = {i["column"]: i["message"] for i in found(result, "leakage")}
    assert set(leaks) == {"label_code", "group"}
    assert (
        "each value of group belongs to one class in 100 % of the training rows" in (leaks["group"])
    )


def test_leakage_of_a_threshold_for_a_binary_target(rng):
    n = 300
    score = rng.uniform(0, 100, n)
    noisy = score + rng.normal(0, 15, n)
    source = table({"score": score, "noisy": noisy, "passed": np.where(score >= 50, "yes", "no")})
    leaks = found(check(source, "passed"), "leakage")
    assert [i["column"] for i in leaks] == ["score"]
    assert "a single threshold on score separates the classes in 100 %" in leaks[0]["message"]


def test_no_leakage_for_ordinary_or_rare_patterns(rng):
    n = 400
    labels = rng.integers(0, 3, n)
    centers = np.array([[5.0, 3.4, 1.5], [5.9, 2.8, 4.3], [6.6, 3.0, 5.5]])
    x = centers[labels] + rng.normal(0, 0.6, (n, 3))
    source = table(
        {
            "a": x[:, 0],
            "b": x[:, 1],
            "c": x[:, 2],
            # unique per row: "pure" values prove nothing
            "name": [f"n{i}" for i in range(n)],
            "y": np.array(["p", "q", "r"])[labels],
        }
    )
    assert "leakage" not in codes(check(source, "y"))
    # a 99 : 1 target: a random feature is "right" for 99 % of the rows, but no leak
    rare = table({"x": rng.normal(size=n), "y": np.where(np.arange(n) < 4, "b", "a")})
    assert "leakage" not in codes(check(rare, "y"))


def test_leakage_by_name(rng):
    n = 200
    source = table(
        {
            "tenure": rng.normal(size=n),
            "churn_flag": rng.integers(0, 2, n).astype(float),
            "Is Churn": rng.normal(size=n),
            "churn_rate_region": rng.normal(size=n),
            "churn": rng.choice(np.array(["yes", "no"], dtype=object), n),
        }
    )
    leaks = found(check(source, "churn"), "leakage")
    assert sorted(i["column"] for i in leaks) == ["Is Churn", "churn_flag"]
    assert "by its name, churn_flag looks like another form of the target churn" in (
        next(i["message"] for i in leaks if i["column"] == "churn_flag")
    )
    assert checks._leak_name("price_log", "price") and checks._leak_name("label_of_y2", "y2")
    assert not checks._leak_name("price_per_room", "price")
    assert not checks._leak_name("x_id", "x")  # one-letter targets: too many false hits


def test_leakage_of_one_hot_columns_is_one_issue(rng):
    n = 300
    churn = rng.choice(np.array(["yes", "no"], dtype=object), n)
    source = table(
        {
            "age": rng.normal(40, 10, n),
            "status": np.where(churn == "yes", "churned", "stayed"),
            "churn": churn,
        }
    )
    before = one(check(source, "churn"), "leakage")
    assert before["columns"] == ["status"]
    encoded, _ = after_fixes(source, "churn", {"type": "encode", "params": {"columns": ["status"]}})
    leak = one(encoded, "leakage")
    assert leak["columns"] == ["status=churned", "status=stayed"]
    assert "(and the same for status=stayed)" in leak["message"]
    assert leak["fix"]["label"] == "Drop 2 columns"


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #


def test_rows_in_training_and_validation_or_test(rng):
    n = 200
    base = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    base["y"] = base["a"] * 2 + rng.normal(size=n)
    source = table(pd.concat([base, base.iloc[:60]], ignore_index=True).to_dict("list"))
    result = check(source, "y")
    overlap = one(result, "split_overlap")
    assert overlap["severity"] == "warning"
    assert "are exact copies of training rows" in overlap["message"]
    assert "(60 duplicate rows in all)" in overlap["message"]
    assert overlap["fix"] == {
        "type": "drop_duplicates",
        "params": {"columns": [], "keep": "first"},
        "label": "Remove duplicates",
    }
    assert "duplicate_rows" not in codes(result)  # the overlap says it all
    fixed, _ = after_fixes(source, "y", overlap["fix"])
    assert not codes(fixed) & {"split_overlap", "duplicate_rows"}


def test_duplicate_rows_within_the_training_rows(rng):
    n = 200
    base = pd.DataFrame({"a": rng.normal(size=n), "y": rng.normal(size=n)})
    source = table(pd.concat([base, base.iloc[:4]], ignore_index=True).to_dict("list"))
    split = np.array([TRAIN] * 140 + [VAL] * 30 + [TEST] * 30 + [TRAIN] * 4, dtype=np.int8)
    duplicates = one(check(source, "y", parts=split), "duplicate_rows")
    assert duplicates["severity"] == "info"  # 2 % of the rows
    assert duplicates["message"] == "4 rows (2 %) are exact copies of other rows."
    assert duplicates["fix"]["type"] == "drop_duplicates"
    many = table(pd.concat([base, base.iloc[:40]], ignore_index=True).to_dict("list"))
    split = np.array([TRAIN] * 140 + [VAL] * 30 + [TEST] * 30 + [TRAIN] * 40, dtype=np.int8)
    assert one(check(many, "y", parts=split), "duplicate_rows")["severity"] == "warning"
    assert not codes(check(table(base.to_dict("list")), "y")) & {"duplicate_rows"}


def test_duplicates_in_a_sample_of_large_tables(rng, monkeypatch):
    monkeypatch.setattr(checks, "DUPLICATE_ROWS", 150)
    n = 200
    base = pd.DataFrame({"a": rng.normal(size=n), "y": rng.normal(size=n)})
    source = table(pd.concat([base, base], ignore_index=True).to_dict("list"))
    overlap = one(check(source, "y"), "split_overlap")
    assert overlap["message"].startswith("In a sample of 150 rows, ")


def test_conflicting_labels(rng):
    n = 300
    source = table(
        {
            "a": rng.integers(0, 3, n).astype(float),
            "b": rng.integers(0, 2, n).astype(float),
            "y": rng.choice(np.array(["x", "z"], dtype=object), n),
        }
    )
    conflict = one(check(source, "y"), "conflicting_labels")
    assert conflict["severity"] == "info" and conflict["fix"] is None
    assert "have the same features as another row but a different class" in conflict["message"]
    continuous = table(
        {"a": rng.normal(size=n), "y": rng.choice(np.array(["x", "z"], dtype=object), n)}
    )
    assert "conflicting_labels" not in codes(check(continuous, "y"))


def test_few_rows(rng):
    small = table({"a": rng.normal(size=40), "y": rng.normal(size=40)})
    assert one(check(small, "y"), "few_rows")["message"].startswith("Only 28 training rows")
    n, k = 200, 40
    wide = table({**{f"f{i}": rng.normal(size=n) for i in range(k)}, "y": rng.normal(size=n)})
    message = one(check(wide, "y"), "few_rows")["message"]
    assert message.startswith("140 training rows for 40 features (3.5 per feature)")
    enough = table({"a": rng.normal(size=200), "y": rng.normal(size=200)})
    assert "few_rows" not in codes(check(enough, "y"))


# --------------------------------------------------------------------------- #
# Feature columns
# --------------------------------------------------------------------------- #


def test_near_constant_features(rng):
    n = 300
    rare = np.zeros(n)
    rare[:2] = 1.0
    mostly = np.where(rng.random(n) < 0.95, 0.0, 1.0)
    source = table(
        {"rare": rare, "mostly": mostly, "x": rng.normal(size=n), "y": rng.normal(size=n)}
    )
    near = one(check(source, "y"), "near_constant")
    assert near["columns"] == ["rare"] and near["severity"] == "info"
    assert "of the training rows; it carries almost no information" in near["message"]
    fixed, _ = after_fixes(source, "y", near["fix"])
    assert "near_constant" not in codes(fixed)
    # too few rows to say
    small = table({"rare": rare[:60], "x": rng.normal(size=60), "y": rng.normal(size=60)})
    assert "near_constant" not in codes(check(small, "y"))


def test_redundant_features_keep_the_one_closer_to_the_target(rng):
    n = 300
    x = rng.normal(size=n)
    source = table(
        {
            # a less exact measurement of x: less associated with the target
            "x_scaled": 3 * (x + rng.normal(0, 0.12, n)) + 1,
            "x": x,
            "similar": x + rng.normal(0, 0.5, n),  # r ≈ 0.9: kept
            "y": x + rng.normal(0, 0.3, n),
        }
    )
    redundant = one(check(source, "y"), "redundant")
    assert redundant["columns"] == ["x_scaled"]
    assert redundant["message"].startswith("x_scaled is almost a copy of x (r = 0.99")
    assert redundant["message"].endswith("; one of them is enough.")
    fixed, _ = after_fixes(source, "y", redundant["fix"])
    assert "redundant" not in codes(fixed)
    # without a target, the later column of a pair goes
    pair = table({"a": x, "b": -2 * x, "c": rng.normal(size=n)})
    issues = checks.feature_issues(pair, None, None, None)
    assert [i["message"] for i in issues if i["code"] == "redundant"] == [
        "b is almost a copy of a (r ≈ -1); one of them is enough."
    ]


def test_numeric_ids(rng):
    n = 200
    source = table(
        {
            "row": np.arange(n, dtype=float),
            "customerId": rng.permutation(np.arange(5000, 5000 + n)).astype(float),
            "year": np.arange(1900, 1900 + n, dtype=float),  # consecutive, but no ID
            "y": rng.normal(size=n),
        }
    )
    ids = one(check(source, "y"), "id_like")
    assert ids["columns"] == ["row", "customerId"] and ids["severity"] == "warning"
    assert ids["fix"]["params"] == {"columns": ["row", "customerId"]}


def test_skewed_features_and_target(rng):
    n = 400
    source = table(
        {
            "income": rng.lognormal(10, 1, n),
            "debt": -rng.lognormal(5, 1, n),
            "normal": rng.normal(size=n),
            "price": rng.lognormal(12, 1, n),
        }
    )
    result = check(source, "price")
    skewed = {i["fix"]["params"]["function"]: i for i in found(result, "skewed")}
    assert skewed["log1p"]["columns"] == ["income"]
    assert skewed["log1p"]["fix"]["label"] == "Log-transform income (log1p)"
    assert skewed["yeo_johnson"]["columns"] == ["debt"]
    target = one(result, "skewed_target")
    assert target["fix"] == {
        "type": "transform",
        "params": {"columns": ["price"], "function": "log1p"},
        "label": "Log-transform price (log1p)",
    }
    fixed, _ = after_fixes(
        source, "price", skewed["log1p"]["fix"], skewed["yeo_johnson"]["fix"], target["fix"]
    )
    assert not codes(fixed) & {"skewed", "skewed_target"}


def test_very_different_scales(rng):
    n = 300
    source = table(
        {
            "income": rng.normal(50_000, 20_000, n),
            "ratio": rng.normal(0.5, 0.01, n),
            "flag": rng.integers(0, 2, n).astype(float),
            "y": rng.normal(size=n),
        }
    )
    scales = one(check(source, "y"), "scales")
    assert scales["severity"] == "info" and scales["columns"] == ["income", "ratio"]
    assert scales["fix"]["params"] == {"columns": ["income", "ratio"], "method": "standard"}
    fixed, _ = after_fixes(source, "y", scales["fix"])
    assert "scales" not in codes(fixed)
    similar = table({"a": rng.normal(0, 5, n), "b": rng.normal(0, 1, n), "y": rng.normal(size=n)})
    assert "scales" not in codes(check(similar, "y"))


# --------------------------------------------------------------------------- #
# String feature columns
# --------------------------------------------------------------------------- #


def test_numbers_stored_as_text(rng):
    n = 200
    price = rng.uniform(5, 5000, n).round(2)
    source = table(
        {
            "price": [f"${v:,.2f}" for v in price],
            "weight": [f"{v:.1f} kg".replace(".", ",") for v in rng.uniform(50, 99, n)],
            "y": rng.normal(size=n),
        },
        kinds={"price": "categorical", "weight": "categorical"},
    )
    result = check(source, "y")
    issues = {tuple(i["columns"]): i for i in found(result, "numbers_as_text")}
    price_issue = issues[("price",)]
    assert price_issue["severity"] == "error"
    assert price_issue["message"].startswith("price holds numbers as text ('$")
    assert price_issue["fix"] == {
        "type": "cast",
        "params": {"columns": ["price"], "to": "numeric", "strip_units": True},
        "label": "Convert price to numbers",
    }
    assert issues[("weight",)]["fix"]["params"] == {
        "columns": ["weight"],
        "to": "numeric",
        "decimal_comma": True,
        "strip_units": True,
    }
    assert "categorical_features" not in codes(result)  # no one-hot encoding of numbers
    fixed, prepared = after_fixes(source, "y", price_issue["fix"], issues[("weight",)]["fix"])
    assert fixed["ready"], fixed["issues"]
    assert values(prepared.table, "price") == price.tolist()


@pytest.mark.parametrize(
    "texts",
    [
        [f"{v:05d}" for v in range(1000, 1200)],  # zip codes with leading zeros
        [f"Z{v}" for v in range(10000, 10200)],  # codes
        [f"{v % 30}{'ABC'[v % 3]}" for v in range(200)],  # flat numbers: 12A, 12B
        [str(v % 7) for v in range(200)],  # numbers that are categories on purpose
    ],
)
def test_codes_are_no_numbers_stored_as_text(texts):
    source = table({"code": texts, "y": np.arange(200.0) % 3}, kinds={"code": "categorical"})
    assert "numbers_as_text" not in codes(check(source, "y", task="regression"))


def test_dates_stored_as_text(rng):
    n = 200
    days = [f"2023-{rng.integers(1, 13):02d}-{rng.integers(1, 29):02d}" for _ in range(n)]
    source = table({"day": days, "y": rng.normal(size=n)}, kinds={"day": "categorical"})
    dates = one(check(source, "y"), "dates_as_text")
    assert dates["message"] == (
        "day holds dates as text. Convert it to dates, then split it into parts."
    )
    fixed, prepared = after_fixes(source, "y", dates["fix"])
    assert prepared.table.kinds["day"] == "datetime"
    assert "date_features" in codes(fixed)


def test_categories_with_a_value_per_row(rng):
    n = 100
    source = table(
        {
            "name": [f"Person {i}" for i in range(n)],
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
        },
        kinds={"name": "categorical"},
    )
    ids = one(check(source, "y"), "id_like")
    assert ids["severity"] == "error" and ids["columns"] == ["name"]
    assert "categorical_features" not in codes(check(source, "y"))
    fixed, _ = after_fixes(source, "y", ids["fix"])
    assert fixed["ready"], fixed["issues"]


def test_spellings_that_differ_only_in_case_or_spaces(rng):
    n = 200
    city = rng.choice(np.array(["Berlin", "Hamburg", "Munich"], dtype=object), n)
    city[:10] = "berlin"
    city[10:15] = "BERLIN"
    city[15:20] = "Hamburg  "
    source = table({"city": city, "y": rng.normal(size=n)})
    variants = one(check(source, "y"), "inconsistent_categories")
    assert variants["severity"] == "error"
    assert variants["message"].startswith("city has values that differ only in upper/lower")
    assert variants["fix"]["type"] == "replace_values"
    assert variants["fix"]["params"]["mapping"] == {
        "berlin": "Berlin",
        "BERLIN": "Berlin",
        "Hamburg  ": "Hamburg",
    }
    fixed, prepared = after_fixes(source, "y", variants["fix"])
    assert set(prepared.table.df["city"]) == {"Berlin", "Hamburg", "Munich"}
    assert one(fixed, "categorical_features")["columns"] == ["city"]


def test_target_classes_that_differ_only_in_case(rng):
    n = 200
    y = rng.choice(np.array(["yes", "no"], dtype=object), n)
    y[:12] = "Yes"
    source = table({"x": rng.normal(size=n), "y": y})
    result = check(source, "y")
    assert result["num_classes"] == 3
    variants = one(result, "inconsistent_categories")
    assert variants["severity"] == "warning"
    assert variants["fix"]["params"] == {"columns": ["y"], "mapping": {"Yes": "yes"}}
    fixed, _ = after_fixes(source, "y", variants["fix"])
    assert fixed["num_classes"] == 2


@pytest.mark.parametrize(("task", "method"), [("regression", "target"), ("auto", "frequency")])
def test_many_categories(rng, task, method):
    n = 600
    source = table(
        {
            "shop": [f"shop {i % 80}" for i in rng.permutation(n)],
            "kind": rng.choice(np.array(["a", "b"], dtype=object), n),
            "y": rng.integers(0, 2, n).astype(float),
        }
    )
    result = check(source, "y", task=task)
    many = one(result, "high_cardinality")
    assert many["columns"] == ["shop"]
    assert many["message"].startswith("shop has 80 categories.")
    assert many["fix"]["params"] == {"columns": ["shop"], "method": method}
    assert one(result, "categorical_features")["columns"] == ["kind"]
    fixed, prepared = after_fixes(source, "y", many["fix"], task=task)
    assert prepared.table.kinds["shop"] == "numeric"


# --------------------------------------------------------------------------- #
# Classes
# --------------------------------------------------------------------------- #


def test_rare_classes_with_a_fix_that_removes_their_rows(rng):
    n = 300
    y = np.array(["a"] * 150 + ["b"] * 144 + ["c"] * 4 + ["d"] * 2, dtype=object)
    source = table({"x": rng.normal(size=n), "y": y})
    rare = one(check(source, "y"), "rare_classes")
    assert rare["severity"] == "warning"
    assert rare["message"].startswith("2 classes have fewer than 5 training rows (c: ")
    assert rare["fix"] == {
        "type": "filter_rows",
        "params": {"column": "y", "operator": "not_in", "values": ["c", "d"]},
        "label": "Remove the rows of c, d",
    }
    fixed, _ = after_fixes(source, "y", rare["fix"])
    assert "rare_classes" not in codes(fixed) and fixed["num_classes"] == 2
    codes_ = pd.Series(y).map({"a": 1.0, "b": 2.0, "c": 2.0, "d": 7.0}).to_numpy()
    numbers = table({"x": rng.normal(size=n), "y": codes_})
    numeric = one(check(numbers, "y"), "rare_classes")
    assert numeric["fix"]["params"]["values"] == [7.0]
    assert after_fixes(numbers, "y", numeric["fix"])[0]["num_classes"] == 2
    # a class that would leave a single one: no fix
    two = table({"x": rng.normal(size=n), "y": np.where(y == "d", "d", "a")})
    assert one(check(two, "y"), "rare_classes")["fix"] is None


def test_classes_missing_from_validation_or_test_rows(rng):
    n = 100
    y = np.array(["a"] * 80 + ["b"] * 20, dtype=object)
    source = table({"x": rng.normal(size=n), "y": y})
    split = np.array([TRAIN] * 60 + [VAL] * 10 + [TEST] * 10 + [TRAIN] * 20, dtype=np.int8)
    missing = one(
        check(source, "y", parts=split, split={"stratify": False}), "classes_not_in_split"
    )
    assert missing["message"] == (
        "1 class has no validation and test rows (b), so the validation and test scores say "
        "nothing about it. Turn on “Stratified” under Target & split."
    )
    assert "classes_not_in_split" not in codes(check(source, "y"))  # stratified split


# --------------------------------------------------------------------------- #
# The whole list
# --------------------------------------------------------------------------- #


def test_checks_never_decide_whether_data_is_ready(rng):
    n = 300
    churn = rng.choice(np.array(["yes", "no"], dtype=object), n)
    data = {
        "id": np.arange(n, dtype=float),
        "income": rng.lognormal(10, 1, n),
        "copy": np.where(churn == "yes", 1.0, 0.0),
        "price": [f"{v:.0f} €" for v in rng.uniform(1, 500, n)],
        "city": rng.choice(np.array(["A", "a", "B"], dtype=object), n),
        "churn": churn,
    }
    for source in (
        table(data),
        table({k: v for k, v in data.items() if k not in ("price", "city")}),
    ):
        settings = clean_settings({"target": "churn"})
        task = resolve_task(source, settings)
        source = source.with_split(assign_split(source, settings, task))
        full = readiness(source, settings, task)
        plain = readiness(source, settings, task, checks=False)
        assert full["ready"] == plain["ready"]
        errors = [i["code"] for i in full["issues"] if i["severity"] == "error"]
        assert bool(errors) == bool([i for i in plain["issues"] if i["severity"] == "error"])
        severities = [i["severity"] for i in full["issues"]]
        assert severities == sorted(severities, key=["error", "warning", "info"].index)
    assert full["ready"]
    assert [i["code"] for i in full["issues"]][:2] == ["leakage", "id_like"]
