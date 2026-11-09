"""Running preparations: settings and step validation, the train/validation/test split,
step reports, error isolation and the training readiness check."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from brickdnn.data import pipeline
from brickdnn.data.pipeline import (
    MAX_STEPS,
    assign_split,
    class_labels,
    clean_settings,
    clean_steps,
    readiness,
    resolve_task,
    run_pipeline,
    split_sizes,
)
from brickdnn.data.steps import REGISTRY, Step, StepResult
from brickdnn.data.table import TEST, TRAIN, VAL, DataError, Table

from .data_helpers import run, steps, table

REPORT_KEYS = {
    "id",
    "type",
    "status",
    "summary",
    "error",
    "warnings",
    "details",
    "rows_before",
    "rows_after",
    "columns",
    "added",
    "removed",
    "fitted",
    "duration_ms",
}


def churn(n: int = 200, seed: int = 0) -> Table:
    rng = np.random.default_rng(seed)
    return table(
        {
            "id": np.arange(n, dtype=float),
            "age": np.where(rng.random(n) < 0.1, np.nan, rng.integers(18, 80, n)).astype(float),
            "income": rng.lognormal(10, 0.5, n),
            "city": rng.choice(np.array(["Berlin", "Hamburg", "Munich", None], dtype=object), n),
            "notes": [f"A long free text note about customer number {i}." for i in range(n)],
            "signup": pd.to_datetime("2021-01-01") + pd.to_timedelta(rng.integers(0, 900, n), "D"),
            "churn": rng.choice(np.array(["yes", "no"], dtype=object), n, p=[0.25, 0.75]),
        },
        kinds={"notes": "text"},
    )


# --------------------------------------------------------------------------- #
# Settings and steps
# --------------------------------------------------------------------------- #


def test_clean_settings_fills_in_defaults():
    assert clean_settings(None) == {
        "target": None,
        "task": "auto",
        "split": {"val": 0.15, "test": 0.15, "seed": 42, "stratify": True},
    }
    cleaned = clean_settings(
        {"target": " price ", "task": "regression", "split": {"val": 0.2, "test": 0, "seed": 7.0}}
    )
    assert cleaned == {
        "target": "price",
        "task": "regression",
        "split": {"val": 0.2, "test": 0.0, "seed": 7, "stratify": True},
    }
    assert clean_settings({"target": "", "unknown": 1})["target"] is None


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ([], "must be an object"),
        ({"target": 3}, "Target: choose a column"),
        ({"task": "clustering"}, "Task: choose auto"),
        ({"split": "70/30"}, "Split: expected"),
        ({"split": {"val": 0}}, "must be more than 0"),
        ({"split": {"val": -0.1}}, "Validation share"),
        ({"split": {"val": "0.2"}}, "Validation share"),
        ({"split": {"test": True}}, "Test share"),
        ({"split": {"val": 0.5, "test": 0.5}}, "at most 0.9"),
        ({"split": {"val": 0.6, "test": 0.31}}, "at most 0.9"),
        ({"split": {"seed": -1}}, "Seed"),
        ({"split": {"seed": 2**31}}, "Seed"),
        ({"split": {"seed": 1.5}}, "Seed"),
        ({"split": {"stratify": "yes"}}, "Stratify"),
    ],
)
def test_clean_settings_errors(options, message):
    with pytest.raises(DataError, match=message):
        clean_settings(options)


def test_clean_steps_normalises_steps():
    cleaned = clean_steps(
        [
            {"id": "a", "type": "join", "params": {"source": 2}},
            {"id": "b", "type": "impute", "enabled": False, "params": None, "extra": 1},
        ]
    )
    assert cleaned == [
        {"id": "a", "type": "join", "enabled": True, "params": {"source": 2}},
        {"id": "b", "type": "impute", "enabled": False, "params": {}},
    ]
    assert clean_steps(None) == []


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"id": "a"}, "must be a list"),
        ([{"id": f"s{i}", "type": "impute"} for i in range(MAX_STEPS + 1)], "at most 100 steps"),
        (["impute"], "Step 1: expected an object"),
        ([{"type": "impute"}], "Step 1: the step needs an id"),
        ([{"id": " ", "type": "impute"}], "needs an id"),
        ([{"id": "x" * 65, "type": "impute"}], "longer than 64"),
        (
            [{"id": "a", "type": "impute"}, {"id": "a", "type": "impute"}],
            "Step 2: the id 'a' is used",
        ),
        ([{"id": "a", "type": "magic"}], "unknown step type 'magic'"),
        ([{"id": "a", "type": "impute", "enabled": "yes"}], "'enabled' must be true or false"),
        ([{"id": "a", "type": "impute", "params": []}], "settings must be an object"),
        (
            [{"id": "a", "type": "impute"}, {"id": "b", "type": "append", "params": {}}],
            "Step 2 \\(Append rows\\) combines tables",
        ),
    ],
)
def test_clean_steps_errors(data, message):
    with pytest.raises(DataError, match=message):
        clean_steps(data)


# --------------------------------------------------------------------------- #
# Task and split
# --------------------------------------------------------------------------- #


def test_resolve_task():
    t = table(
        {
            "label": ["a", "b", "a"],
            "flag": [1.0, 0.0, 1.0],
            "rating": [1.0, 5.0, 3.0],
            "price": [1.5, 2.0, 3.25],
            "day": pd.to_datetime(["2024-01-01"] * 3),
        },
        kinds={"flag": "boolean"},
    )
    auto = {"task": "auto"}
    assert resolve_task(t, {**auto, "target": "label"}) == "classification"
    assert resolve_task(t, {**auto, "target": "flag"}) == "classification"
    assert resolve_task(t, {**auto, "target": "rating"}) == "classification"
    assert resolve_task(t, {**auto, "target": "price"}) == "regression"
    assert resolve_task(t, {**auto, "target": "day"}) is None
    assert resolve_task(t, {**auto, "target": "missing"}) is None
    assert resolve_task(t, {**auto, "target": None}) is None
    assert resolve_task(t, {"task": "regression", "target": "rating"}) == "regression"
    many = table({"y": np.arange(21, dtype=float)})
    assert resolve_task(many, {**auto, "target": "y"}) == "regression"


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 10, 99, 1000])
def test_split_sizes_match_the_training_engine(n):
    train, val, test = split_sizes(n, 0.15, 0.15)
    assert train + val + test == n
    if n >= 3:
        # brickdnn.training.data.split_sizes (copied here: it needs torch)
        n_test = min(max(1, round(n * 0.15)), n - 2)
        n_val = min(max(1, round(n * 0.15)), n - n_test - 1)
        assert (val, test) == (n_val, n_test)
    assert train >= min(n, 1)


def test_assign_split_is_seeded_and_has_the_right_sizes():
    t = churn(1000)
    settings = clean_settings({"target": "churn", "split": {"stratify": False}})
    first = assign_split(t, settings, "classification")
    assert first.dtype == np.int8
    assert np.bincount(first, minlength=3).tolist() == [700, 150, 150]
    assert np.array_equal(first, assign_split(t, settings, "classification"))
    other = clean_settings({"target": "churn", "split": {"stratify": False, "seed": 1}})
    assert not np.array_equal(first, assign_split(t, other, "classification"))


def test_stratified_split_keeps_class_shares():
    labels = np.array(["a"] * 800 + ["b"] * 150 + ["c"] * 50, dtype=object)
    t = table({"y": labels, "x": np.arange(1000.0)})
    settings = clean_settings({"target": "y", "split": {"val": 0.2, "test": 0.1}})
    split = assign_split(t, settings, "classification")
    assert np.bincount(split, minlength=3).tolist() == [700, 200, 100]
    for label, total in (("a", 800), ("b", 150), ("c", 50)):
        part = split[labels == label]
        assert abs((part == VAL).sum() - total * 0.2) <= 1
        assert abs((part == TEST).sum() - total * 0.1) <= 1


def test_stratified_split_with_tiny_classes_and_missing_targets():
    labels = np.array(["a"] * 20 + ["rare"] + [None] * 9, dtype=object)
    t = table({"y": labels})
    settings = clean_settings({"target": "y"})
    split = assign_split(t, settings, "classification")
    assert split[20] == TRAIN  # a class with one row stays in the training data
    assert np.bincount(split, minlength=3).tolist() == [22, 4, 4]
    assert (split[21:] != TRAIN).any()  # rows without target are split too


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_split_of_tiny_tables(n):
    t = (
        table({"y": ["a", "b", "a"][:n]})
        if n
        else Table(pd.DataFrame({"y": []}), {"y": "categorical"})
    )
    split = assign_split(t, clean_settings({"target": "y"}), "classification")
    assert len(split) == n
    if n:
        assert (split == TRAIN).sum() >= 1


def test_class_labels_order():
    names, codes = class_labels(pd.Series([10.0, 2.0, np.nan, 1.0, 2.0]), "numeric")
    assert names == ["1", "2", "10"]
    assert codes.tolist() == [2, 1, -1, 0, 1]
    names, codes = class_labels(pd.Series(["b", None, "B", "a"], dtype=object), "categorical")
    assert names == ["B", "a", "b"]
    assert codes.tolist() == [2, -1, 0, 1]
    assert class_labels(pd.Series([2.5, 1.0]), "numeric")[0] == ["1", "2.5"]


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def test_reports_have_the_step_report_shape():
    result = run(
        churn(),
        ("drop_columns", {"columns": ["id", "notes"]}),
        ("impute", {"strategy": "median"}),
        ("formula", {"name": "rich", "expression": "income > 30000"}),
        settings={"target": "churn"},
    )
    assert not result.failed
    assert result.task == "classification"
    for report in result.reports:
        assert set(report) == REPORT_KEYS
        assert report["status"] == "ok"
        assert report["error"] is None
        assert isinstance(report["duration_ms"], float)
        json.dumps(report, allow_nan=False)
    drop, impute, formula = result.reports
    assert drop["removed"] == ["id", "notes"] and drop["added"] == []
    assert drop["rows_before"] == drop["rows_after"] == 200
    assert impute["fitted"]["values"]["age"] == float(
        np.median(result.before.train_df()["age"].dropna())
    )
    assert formula["added"] == ["rich"]
    assert formula["columns"][-1] == {"name": "rich", "kind": "boolean"}
    assert result.table.split_counts() == [140, 30, 30]
    assert result.before.columns == churn().columns
    assert result.before.split is not None
    assert result.params["s2"]["strategy"] == "median"


def test_a_failing_step_skips_the_rest_and_keeps_the_last_good_table():
    result = run(
        churn(),
        ("drop_columns", {"columns": ["id"]}),
        ("filter_rows", {"column": "age", "operator": ">", "value": 1000}),
        ("drop_columns", {"columns": ["notes"]}),
        settings={"target": "churn"},
    )
    assert result.failed
    assert [r["status"] for r in result.reports] == ["ok", "error", "skipped"]
    assert "would remove all 200 rows" in result.reports[1]["error"]
    assert result.reports[1]["summary"] == ""
    assert "notes" in result.table.columns and "id" not in result.table.columns
    assert result.reports[2]["columns"] == result.table.column_info()


def test_unexpected_errors_are_reported_not_raised(monkeypatch, caplog):
    class Broken(Step):
        type = "broken"
        label = "Broken"
        group = "columns"
        description = "Raises."

        def run(self, table, params, ctx):
            raise ZeroDivisionError("division by zero")

    class LosesSplit(Broken):
        type = "loses_split"

        def run(self, table, params, ctx):
            return StepResult(table.with_split(None), "Oops.")

    monkeypatch.setitem(REGISTRY, "broken", Broken)
    monkeypatch.setitem(REGISTRY, "loses_split", LosesSplit)
    result = run(churn(), ("broken", {}))
    assert result.reports[0]["error"] == (
        "Unexpected error in this step: ZeroDivisionError: division by zero"
    )
    assert "broken" in caplog.text
    result = run(churn(), ("loses_split", {}))
    assert "lost the train/validation/test split" in result.reports[0]["error"]
    result = run(churn(), ("impute", {"bogus": 1}))
    assert result.reports[0]["error"] == "Unknown setting 'bogus' for 'Fill missing values'."


def test_disabled_steps_and_upto():
    items = steps(
        ("drop_columns", {"columns": ["id"]}),
        ("drop_columns", {"columns": ["notes"]}),
        ("drop_columns", {"columns": ["signup"]}),
    )
    items[0]["enabled"] = False
    result = run_pipeline(churn(), items, {}, upto="s2")
    assert [r["status"] for r in result.reports] == ["disabled", "ok", "not_run"]
    assert result.table.columns == ["id", "age", "income", "city", "signup", "churn"]
    assert result.reports[2]["rows_before"] == 200
    assert not result.failed
    unknown = run_pipeline(churn(), items, {}, upto="nope")
    assert [r["status"] for r in unknown.reports] == ["disabled", "ok", "ok"]


def test_combine_steps_run_before_the_split():
    main = table({"k": [1.0, 2.0, 3.0, 4.0] * 5, "y": ["a", "b"] * 10})
    extra = table({"k": [1.0, 2.0, 3.0, 4.0], "v": [10.0, 20.0, 30.0, 40.0]})
    more = table({"k": [9.0] * 10, "y": ["a"] * 10, "v": [0.0] * 10})
    loader = {2: extra, 3: more}.__getitem__
    result = run(
        main,
        ("join", {"source": 2, "left_on": ["k"]}),
        ("append", {"source": 3, "add_source_column": True}),
        ("drop_columns", {"columns": ["k"]}),
        settings={"target": "y"},
        load_source=loader,
        source_name={2: "extra", 3: "more"}.get,
        source_id=1,
    )
    assert not result.failed, result.reports
    assert result.before.n_rows == 30 and result.before.split is not None
    assert result.before.columns == ["k", "y", "v", "source"]
    assert result.table.split_counts() == [22, 4, 4]
    assert result.reports[0]["rows_after"] == 20 and result.reports[1]["rows_after"] == 30
    assert result.fitted["s1"] == {"_plan": result.fitted["s1"]["_plan"]}
    assert result.reports[0]["fitted"] is None  # private plan is not sent to the UI


def test_upto_a_combine_step_still_assigns_the_split():
    main = table({"k": [1.0, 2.0, 3.0, 4.0], "y": ["a", "b", "a", "b"]})
    extra = table({"k": [1.0], "v": [1.0]})
    result = run_pipeline(
        main,
        clean_steps(
            steps(("join", {"source": 2, "left_on": ["k"]}), ("drop_columns", {"columns": ["v"]}))
        ),
        {"target": "y"},
        load_source=lambda i: extra,
        upto="s1",
    )
    assert [r["status"] for r in result.reports] == ["ok", "not_run"]
    assert result.table.split is not None and "v" in result.table.columns


def test_split_stays_aligned_and_fitting_uses_training_rows():
    source = churn(300)
    result = run(
        source,
        ("sort_rows", {"column": "income", "descending": True}),
        ("filter_rows", {"column": "age", "operator": "not_missing"}),
        ("impute", {"columns": ["city"], "strategy": "most_frequent"}),
        settings={"target": "churn", "split": {"seed": 3}},
    )
    before = result.before
    # every remaining row keeps its split label: look the rows up by id
    by_id = dict(zip(before.df["id"].tolist(), before.split.tolist(), strict=True))
    assert [by_id[i] for i in result.table.df["id"]] == result.table.split.tolist()
    train_cities = before.df["city"][before.train_mask() & before.df["age"].notna().to_numpy()]
    assert result.fitted["s3"]["values"]["city"] == train_cities.value_counts().idxmax()


def test_task_is_resolved_on_the_result_when_a_step_creates_the_target():
    result = run(
        table({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}),
        ("formula", {"name": "y", "expression": "x * 1.5"}),
        settings={"target": "y"},
    )
    assert result.task == "regression"


def test_stratify_warning_for_tiny_classes():
    t = table({"x": np.arange(30.0), "y": ["a"] * 28 + ["b", "c"]})
    result = run(t, settings={"target": "y"})
    assert result.warnings == [
        "Stratified split: 2 classes (b, c) have fewer than 3 rows, so not every part contains them."
    ]
    assert run(t, settings={"target": "y", "split": {"stratify": False}}).warnings == []


def test_invalid_settings_raise():
    with pytest.raises(DataError, match="Task"):
        run(churn(), settings={"task": "x"})


# --------------------------------------------------------------------------- #
# Readiness
# --------------------------------------------------------------------------- #


def issues(result: dict) -> dict[str, dict]:
    return {i["message"]: i for i in result["issues"]}


def test_readiness_of_raw_data_lists_errors_with_fixes():
    result = run(churn(), settings={"target": "churn"})
    check = readiness(result.table, result.settings, result.task)
    assert not check["ready"]
    assert check["task"] == "classification"
    assert check["class_names"] == ["no", "yes"] and check["num_classes"] == 2
    assert check["num_features"] == 6 and check["rows"] == 200
    assert check["split_counts"] == [140, 30, 30]
    found = issues(check)
    assert found["1 feature column is categorical (city). Networks need numbers."]["fix"] == {
        "type": "encode",
        "params": {"columns": ["city"], "method": "onehot"},
        "label": "One-hot encode city",
    }
    assert (
        found["notes is free text and cannot be used as a feature."]["fix"]["type"]
        == "drop_columns"
    )
    assert found["signup is a date. Split it into numbers first."]["fix"] == {
        "type": "datetime_features",
        "params": {
            "columns": ["signup"],
            "parts": ["year", "month", "weekday"],
            "drop_original": True,
        },
        "label": "Split signup into year, month, weekday",
    }
    missing = found["1 feature column has missing values (age)."]
    assert missing["fix"]["label"] == "Impute age (median)" and missing["column"] == "age"
    assert [i["severity"] for i in check["issues"]][:4] == ["error"] * 4
    json.dumps(check, allow_nan=False)


def test_readiness_becomes_ready_after_preparation():
    result = run(
        churn(),
        ("drop_columns", {"columns": ["notes", "signup"]}),
        ("impute", {"strategy": "median"}),
        ("impute", {"strategy": "most_frequent"}),
        (
            "replace_values",
            {"columns": ["city"], "mapping": {"Berlin": 1, "Hamburg": 2, "Munich": 3}},
        ),
        ("cast", {"columns": ["city"], "to": "numeric"}),
        settings={"target": "churn"},
    )
    assert not result.failed, result.reports
    check = readiness(result.table, result.settings, result.task)
    assert check["ready"], check["issues"]
    assert check["num_features"] == 4


@pytest.mark.parametrize(
    ("data", "options", "message"),
    [
        ({"x": [1.0, 2.0, 3.0, 4.0]}, {}, "Choose the target column"),
        ({"x": [1.0, 2.0, 3.0, 4.0]}, {"target": "y"}, "'y' does not exist"),
        (
            {"x": [1.0] * 4, "y": ["a", None, "b", "a"]},
            {"target": "y"},
            "1 row has no target value",
        ),
        ({"x": [1.0] * 4, "y": ["a"] * 4}, {"target": "y"}, "has only 1 class ('a')"),
        (
            {"x": [1.0] * 4, "y": ["a", "b", "c", "d"]},
            {"target": "y", "task": "regression"},
            "cannot be used for regression",
        ),
        ({"x": [1.0] * 4, "y": pd.to_datetime(["2024-01-01"] * 4)}, {"target": "y"}, "is a date"),
        ({"y": [1.0, 2.0, 3.5, 4.0]}, {"target": "y"}, "no feature columns"),
        (
            {"1": [1.0, 2.0, 3.0, 4.0], "2": [1.0] * 4, "y": [0.0, 1.0, 0.0, 1.0]},
            {"target": "y"},
            "All feature names are numbers",
        ),
        ({"x": [1.0, 2.0], "y": [0.5, 1.5]}, {"target": "y"}, "at least 4 are needed"),
    ],
)
def test_readiness_errors(data, options, message):
    t = table(data)
    settings = clean_settings(options)
    t = t.with_split(assign_split(t, settings, resolve_task(t, settings)))
    check = readiness(t, settings, resolve_task(t, settings))
    assert not check["ready"]
    assert any(message in i["message"] for i in check["issues"] if i["severity"] == "error"), check


def test_readiness_fixes_for_targets():
    t = table({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": ["1", "2", None, "3.5", "4"]})
    settings = clean_settings({"target": "y", "task": "regression"})
    t = t.with_split(assign_split(t, settings, "regression"))
    found = issues(readiness(t, settings, "regression"))
    assert found["1 row has no target value (y)."]["fix"] == {
        "type": "drop_missing",
        "params": {"columns": ["y"], "how": "any"},
        "label": "Drop 1 row without a target",
    }
    regression = next(i for m, i in found.items() if "regression" in m)
    assert regression["fix"]["type"] == "cast"


def test_readiness_warnings():
    n = 100
    t = table(
        {
            "x": np.arange(n, dtype=float),
            "same": [3.0] * n,
            "y": ["a"] * 90 + ["b"] * 10,
        }
    )
    settings = clean_settings({"target": "y"})
    t = t.with_split(assign_split(t, settings, "classification"))
    check = readiness(t, settings, "classification")
    assert check["ready"]
    messages = issues(check)
    assert (
        messages["1 feature column is constant (same) and carry no information."]["fix"]["type"]
        == "drop_columns"
    )
    imbalance = next(i for m, i in messages.items() if m.startswith("The classes are imbalanced"))
    assert imbalance["fix"]["type"] == "balance_classes"
    train = t.df["y"][t.train_mask()].value_counts()
    ratio = train["a"] / train["b"]
    assert (
        imbalance["message"]
        == f"The classes are imbalanced ({ratio:.1f} : 1 in the training rows)."
    )
    rare = table({"x": np.arange(10.0), "y": ["a"] * 9 + ["b"]})
    rare = rare.with_split(np.array([TRAIN] * 8 + [VAL, TEST], dtype=np.int8))
    check = readiness(rare, settings, "classification")
    assert any("fewer than 5 training rows (b: 0)" in i["message"] for i in check["issues"])


def test_readiness_without_validation_rows():
    t = table({"x": np.arange(6.0), "y": np.arange(6.0) * 1.5})
    t = t.with_split(np.array([TRAIN] * 5 + [TEST], dtype=np.int8))
    check = readiness(t, clean_settings({"target": "y"}), "regression")
    assert any("no validation rows" in i["message"] for i in check["issues"])


def test_json_safe_turns_numpy_values_into_plain_python():
    value = pipeline.json_safe(
        {"a": np.float64(1.5), "b": [np.int64(2), np.nan], "c": (np.bool_(True),)}
    )
    assert value == {"a": 1.5, "b": [2, None], "c": [True]}
    assert type(value["a"]) is float and type(value["b"][0]) is int
