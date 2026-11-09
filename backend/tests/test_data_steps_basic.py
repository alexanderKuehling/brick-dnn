"""The basic preparation steps: combine (join, append), columns, rows and missing values.

Every step is also checked against its own pandas code (``to_pandas``): running the code on
the same data must give the same table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from brickdnn.data.steps import StepContext
from brickdnn.data.steps.columns import FormulaCompiler
from brickdnn.data.steps.missing import knn_fill
from brickdnn.data.table import TEST, TRAIN, VAL, DataError, Table

from .data_helpers import apply, check_code, run_code, script_frame, table, values

TARGET_CTX = StepContext(target="y")


@pytest.fixture
def people() -> Table:
    return table(
        {
            "name": ["Ann", "Bob", "Cem", "Dana", None, "Fay"],
            "age": [34.0, np.nan, 51.0, 28.0, 45.0, -999.0],
            "income": [52_000.0, 38_500.5, np.nan, 61_000.0, 44_250.0, 39_000.0],
            "city": ["Berlin", "Hamburg", None, "Berlin", "?", "Munich"],
            "member": [1.0, 0.0, np.nan, 1.0, 0.0, 1.0],
            "joined": pd.to_datetime(
                ["2021-03-01", "2022-07-15", None, "2020-01-31", "2023-11-02", "2021-12-24"]
            ),
            "y": ["yes", "no", "no", "yes", "no", "yes"],
        },
        kinds={"member": "boolean", "name": "categorical"},
    )


# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #


def test_drop_columns(people):
    result = check_code(people, "drop_columns", {"columns": ["name", "joined"]})
    assert result.table.columns == ["age", "income", "city", "member", "y"]
    assert result.summary == "Dropped 2 columns: 'name' and 'joined'."
    assert result.warnings == []


def test_drop_columns_needs_columns_and_warns_about_the_target(people):
    with pytest.raises(DataError, match="choose at least one column"):
        apply(people, "drop_columns", {"columns": []})
    with pytest.raises(DataError, match="no column 'nope'"):
        apply(people, "drop_columns", {"columns": ["nope"]})
    result, _, _ = apply(people, "drop_columns", {"columns": ["y"]}, TARGET_CTX)
    assert "target column 'y'" in result.warnings[0]


def test_keep_columns_always_keeps_the_target(people):
    result = check_code(people, "keep_columns", {"columns": ["income", "age"]}, TARGET_CTX)
    assert result.table.columns == ["age", "income", "y"]
    assert result.summary == "Kept 2 columns and the target 'y'; dropped 4."


def test_rename_columns(people):
    result = check_code(
        people, "rename_columns", {"mapping": {"age": "Age", "y": "label"}}, TARGET_CTX
    )
    assert result.table.columns[:2] == ["name", "Age"]
    assert result.table.kinds["label"] == "categorical"
    assert "target 'y' is now called 'label'" in result.warnings[0]
    with pytest.raises(DataError, match="no column 'nope'"):
        apply(people, "rename_columns", {"mapping": {"nope": "x"}})
    with pytest.raises(DataError, match="two columns would be called 'income'"):
        apply(people, "rename_columns", {"mapping": {"age": "income"}})
    with pytest.raises(DataError, match="enter a new name"):
        apply(people, "rename_columns", {"mapping": {"age": "  "}})
    swapped = check_code(people, "rename_columns", {"mapping": {"age": "income", "income": "age"}})
    assert values(swapped.table, "age") == values(people, "income")


def test_cast_text_to_numbers_reports_lost_values():
    source = table(
        {"price": ["1.234,5", "12,25", "free", None, "7"]}, kinds={"price": "categorical"}
    )
    result = check_code(
        source, "cast", {"columns": ["price"], "to": "numeric", "decimal_comma": True}
    )
    assert values(result.table, "price") == [1234.5, 12.25, None, None, 7.0]
    assert result.table.kinds["price"] == "numeric"
    assert "1 value could not be converted" in result.summary
    assert "'free'" in result.summary


def test_cast_text_with_units_and_thousands_separators_to_numbers():
    source = table(
        {
            "price": ["$1,234.50", "$12", "-5 °C", "45 %", "ca. 3", "N/A", None, "12.5.2024"],
            "euro": ["1.234,50 €", "12 €", "0,5 €", "€ 7", None, "3-4 €", "kg", "-2,25"],
        },
        kinds={"price": "categorical", "euro": "categorical"},
    )
    result = check_code(
        source, "cast", {"columns": ["price"], "to": "numeric", "strip_units": True}
    )
    assert values(result.table, "price") == [1234.5, 12.0, -5.0, 45.0, 3.0, None, None, None]
    assert "2 values could not be converted" in result.summary
    comma = check_code(
        source,
        "cast",
        {"columns": ["euro"], "to": "numeric", "decimal_comma": True, "strip_units": True},
    )
    assert values(comma.table, "euro") == [1234.5, 12.0, 0.5, 7.0, None, None, None, -2.25]
    # without the option, units make the values missing as before
    plain = check_code(source, "cast", {"columns": ["price"], "to": "numeric"})
    assert values(plain.table, "price") == [None] * 8


def test_cast_numbers_are_exact():
    text = ["19014.449957082357", "0.1", "1e-05", "-3"]
    source = table({"x": text}, kinds={"x": "categorical"})
    result = check_code(source, "cast", {"columns": ["x"], "to": "numeric"})
    assert values(result.table, "x") == [float(v) for v in text]


def test_cast_numbers_to_categories_and_back(people):
    result = check_code(
        people, "cast", {"columns": ["age", "member", "joined"], "to": "categorical"}
    )
    assert values(result.table, "age") == ["34", None, "51", "28", "45", "-999"]
    assert values(result.table, "member") == ["1", "0", None, "1", "0", "1"]
    assert values(result.table, "joined")[:2] == ["2021-03-01", "2022-07-15"]
    floats = table({"x": [2.5, 1e20, 3.0]})
    cast = check_code(floats, "cast", {"columns": ["x"], "to": "text"})
    assert values(cast.table, "x") == ["2.5", "1e+20", "3"]


def test_cast_to_boolean_and_dates():
    source = table(
        {
            "flag": ["Yes", "no", "maybe", None],
            "n": [0.0, 1.0, 2.0, np.nan],
            "day": ["31.01.2024", "01.02.2024", "x", None],
            "iso": ["2024-01-31", "2024-02-01T10:30:00", None, "2024-03-01"],
            "ts": [0.0, 86_400.0, 1e300, np.nan],
        },
        kinds={"flag": "categorical", "day": "categorical", "iso": "categorical"},
    )
    flags = check_code(source, "cast", {"columns": ["flag", "n"], "to": "boolean"})
    assert values(flags.table, "flag") == [1.0, 0.0, None, None]
    assert values(flags.table, "n") == [0.0, 1.0, None, None]
    dates = check_code(source, "cast", {"columns": ["day", "iso", "ts"], "to": "datetime"})
    assert dates.table.kinds["day"] == "datetime"
    assert values(dates.table, "day")[:2] == [
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-02-01"),
    ]
    assert values(dates.table, "iso")[1] == pd.Timestamp("2024-02-01 10:30")
    assert values(dates.table, "ts")[:3] == [
        pd.Timestamp("1970-01-01"),
        pd.Timestamp("1970-01-02"),
        None,
    ]
    assert dates.fitted["formats"] == {"day": "%d.%m.%Y", "iso": "ISO8601"}
    fixed = check_code(
        source, "cast", {"columns": ["day"], "to": "datetime", "datetime_format": "%m.%d.%Y"}
    )
    assert values(fixed.table, "day")[0] is None  # month 31 does not exist
    back = check_code(dates.table, "cast", {"columns": ["day"], "to": "numeric"})
    assert values(back.table, "day")[0] == pd.Timestamp("2024-01-31").timestamp()


def test_cast_errors(people):
    with pytest.raises(DataError, match="cannot be turned into true/false"):
        apply(people, "cast", {"columns": ["joined"], "to": "boolean"})
    with pytest.raises(DataError, match="Date format"):
        apply(people, "cast", {"columns": ["city"], "to": "datetime", "datetime_format": "dd.mm"})
    with pytest.raises(DataError, match="choose at least one column"):
        apply(people, "cast", {"columns": [], "to": "numeric"})


# --------------------------------------------------------------------------- #
# Formula
# --------------------------------------------------------------------------- #


@pytest.fixture
def numbers() -> Table:
    return table(
        {
            "a": [1.0, 2.0, np.nan, 4.0, 0.0],
            "b": [10.0, 0.0, 3.0, -2.0, 5.0],
            "my col": [1.0, 1.0, 1.0, 1.0, 1.0],
            "city": ["Berlin", None, "Rome", "Berlin", "Oslo"],
            "flag": [1.0, 0.0, 1.0, np.nan, 0.0],
        },
        kinds={"flag": "boolean"},
    )


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("a + b * 2", [21.0, 2.0, None, 0.0, 10.0]),
        ("a / b", [0.1, None, None, -2.0, 0.0]),  # division by zero -> missing
        ("-a ** 2 + `my col`", [0.0, -3.0, None, -15.0, 1.0]),
        ("b // 3 + b % 3", [4.0, 0.0, 1.0, 0.0, 3.0]),
        ("log1p(a) - log1p(a)", [0.0, 0.0, None, 0.0, 0.0]),
        ("sqrt(b)", [10**0.5, 0.0, 3**0.5, None, 5**0.5]),
        ("round(b / 4) + floor(2.7) + ceil(0.2) + abs(-1)", [6.0, 4.0, 5.0, 4.0, 5.0]),
        ("round(a / 3, 2)", [0.33, 0.67, None, 1.33, 0.0]),
        ("min(a, b) + max(a, 2)", [3.0, 2.0, None, 2.0, 2.0]),
        ("clip(b, 0, 4)", [4.0, 0.0, 3.0, 0.0, 4.0]),
        ("where(a > 1, a, 0)", [0.0, 2.0, None, 4.0, 0.0]),
        ("isnull(a) + 2 ** 3", [8.0, 8.0, 9.0, 8.0, 8.0]),
        ("exp(0) * 1 / 0 + a", [None, None, None, None, None]),  # 1 / 0 is inf -> missing
        ("flag * 3", [3.0, 0.0, 3.0, None, 0.0]),
        ("10", [10.0] * 5),
    ],
)
def test_formula_numbers(numbers, expression, expected):
    result = check_code(numbers, "formula", {"name": "out", "expression": expression})
    got = values(result.table, "out")
    assert result.table.kinds["out"] == "numeric"
    assert len(got) == len(expected)
    for g, e in zip(got, expected, strict=True):
        assert (g is None and e is None) or g == pytest.approx(e), (expression, got)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("a > 1", [0.0, 1.0, None, 1.0, 0.0]),
        ("city == 'Berlin'", [1.0, None, 0.0, 1.0, 0.0]),
        ('city != "Berlin" and a >= 0', [0.0, None, None, 0.0, 1.0]),
        ("not flag or b < 0", [0.0, 1.0, 0.0, None, 1.0]),
        ("0 < b <= 5", [0.0, 0.0, 1.0, 0.0, 1.0]),
        ("1 < 2", [1.0] * 5),
    ],
)
def test_formula_conditions_give_booleans(numbers, expression, expected):
    result = check_code(numbers, "formula", {"name": "c", "expression": expression})
    assert result.table.kinds["c"] == "boolean"
    assert values(result.table, "c") == expected


def test_formula_text_result_and_replacing_a_column(numbers):
    result = check_code(
        numbers, "formula", {"name": "size", "expression": "where(b > 4, 'big', city)"}
    )
    assert result.table.kinds["size"] == "categorical"
    assert values(result.table, "size") == ["big", None, "Rome", "Berlin", "big"]
    replaced = check_code(numbers, "formula", {"name": "a", "expression": "a * 2"})
    assert replaced.table.columns == numbers.columns
    assert replaced.summary.startswith("Replaced 'a' (numbers)")


def test_formula_summary_counts_missing_and_invalid(numbers):
    result, _, _ = apply(numbers, "formula", {"name": "q", "expression": "a / b"})
    assert result.summary == (
        "Added 'q' (numbers): 1 value missing because an input is missing; 1 value could not "
        "be computed (e.g. division by zero or log of a negative number)."
    )


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("__import__('os').system('echo hacked')", "not allowed"),
        ("a.__class__", "Attribute access"),
        ("df.drop", "Attribute access"),
        ("open('/etc/passwd')", "open(…) is not allowed"),
        ("eval('1')", "eval(…) is not allowed"),
        ("(lambda: 1)()", "not allowed"),
        ("[a for a in b]", "not allowed"),
        ("a[0]", "Indexing"),
        ("a if b else 1", "where(condition, a, b)"),
        ("{1: 2}", "not allowed"),
        ("f'{a}'", "f-strings"),
        ("'x' * 100000000", "text"),
        ("city + 1", "text"),
        ("city > 'B'", "only be compared with == and !="),
        ("city == 1", "cannot compare text with numbers"),
        ("log(a, b)", "takes 1 argument"),
        ("log(x=a)", "plain arguments"),
        ("log", "is a function"),
        ("nope + 1", "no column 'nope'"),
        ("`no col`", "no column 'no col'"),
        ("`a", "not closed"),
        ("a +", "not a valid formula"),
        ("a in b", "'in' and 'is' are not allowed"),
        ("round(a, b)", "digits from 0 to 15"),
        ("__col0__", "not allowed"),
        pytest.param(" + ".join(["a"] * 200), "too long", id="many-nodes"),
        pytest.param("(" * 400 + "1" + ")" * 400, "formula", id="deep-nesting"),
        pytest.param("1" * 1001, "at most 1,000 characters", id="long-text"),
    ],
)
def test_formula_rejects_unsafe_or_invalid_expressions(numbers, expression, message):
    with pytest.raises(DataError, match=message.replace("(", r"\(").replace(")", r"\)")):
        apply(numbers, "formula", {"name": "out", "expression": expression})


def test_formula_rejects_dates(people):
    with pytest.raises(DataError, match="holds dates"):
        apply(people, "formula", {"name": "x", "expression": "joined + 1"})


def test_formula_code_contains_only_whitelisted_names(numbers):
    compiler = FormulaCompiler(numbers.kinds)
    expr = compiler.compile("where(isnull(a), `my col`, log(b) + clip(a, 0, 1))")
    for token in ("__", "import", "eval", "exec", "open", "getattr", "lambda"):
        assert token not in expr.code
    assert compiler.columns == ["a", "my col", "b"]


# --------------------------------------------------------------------------- #
# Replace values, mark missing
# --------------------------------------------------------------------------- #


def test_replace_values_in_text_and_number_columns(people):
    result = check_code(
        people,
        "replace_values",
        {"columns": ["city", "age"], "mapping": {"?": None, "Berlin": "BER", "-999": None, "x": 1}},
    )
    assert values(result.table, "city") == ["BER", "Hamburg", None, "BER", None, "Munich"]
    assert values(result.table, "age")[-1] is None
    assert result.summary == "Replaced 4 values in 2 columns."
    assert "Not found: 'x'." in result.warnings


def test_replace_values_hints_at_numbers_and_checks_types(people):
    sizes = table({"size": ["S", "M", "L", "M"]})
    result = check_code(
        sizes, "replace_values", {"columns": ["size"], "mapping": {"S": 1, "M": 2, "L": 3}}
    )
    assert values(result.table, "size") == ["1", "2", "3", "2"]
    assert "use “Change type”" in result.warnings[-1]
    with pytest.raises(DataError, match="none of the old values can occur in 'age'"):
        apply(people, "replace_values", {"columns": ["age"], "mapping": {"abc": 1}})
    with pytest.raises(DataError, match="'many' is not a number"):
        apply(people, "replace_values", {"columns": ["age"], "mapping": {"1": "many"}})
    with pytest.raises(DataError, match="add at least one value"):
        apply(people, "replace_values", {"columns": ["age"], "mapping": {}})
    dates = check_code(
        people, "replace_values", {"columns": ["joined"], "mapping": {"2020-01-31": "2020-02-01"}}
    )
    assert values(dates.table, "joined")[3] == pd.Timestamp("2020-02-01")


def test_mark_missing_compares_numbers_as_numbers_and_skips_the_target(people):
    source = people.with_column("y", ["yes", "?", "no", "no", "yes", "?"], "categorical")
    result = check_code(source, "mark_missing", {"columns": [], "values": [-999, "?"]}, TARGET_CTX)
    assert values(result.table, "age")[-1] is None
    assert values(result.table, "city")[4] is None
    assert values(result.table, "y") == ["yes", "?", "no", "no", "yes", "?"]  # target untouched
    assert result.summary == "Marked 2 values as missing in 2 columns."
    explicit = check_code(source, "mark_missing", {"columns": ["y"], "values": ["?"]}, TARGET_CTX)
    assert values(explicit.table, "y")[1] is None
    with pytest.raises(DataError, match="add at least one value"):
        apply(source, "mark_missing", {"values": []})


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #


def test_drop_duplicates_keeps_the_split_aligned():
    source = table(
        {"a": [1.0, 1.0, 2.0, 1.0, np.nan, np.nan], "b": ["x", "x", "y", "z", None, None]},
        split=np.array([TRAIN, VAL, TRAIN, TEST, VAL, TRAIN]),
    )
    result = check_code(source, "drop_duplicates", {})
    assert values(result.table, "b") == ["x", "y", "z", None]
    assert result.table.split.tolist() == [TRAIN, TRAIN, TEST, VAL]
    assert result.summary == "Removed 2 duplicate rows, 33.3 % of the rows."
    last = check_code(source, "drop_duplicates", {"columns": ["a"], "keep": "last"})
    assert last.table.split.tolist() == [TRAIN, TEST, TRAIN]
    none = check_code(table({"a": [1.0, 2.0]}), "drop_duplicates", {})
    assert none.summary == "No duplicate rows found."


@pytest.mark.parametrize(
    ("column", "operator", "extra", "kept"),
    [
        ("age", ">=", {"value": 34}, [0, 2, 4]),
        ("age", "<", {"value": "34"}, [3, 5]),
        ("age", "==", {"value": 51}, [2]),
        ("age", "!=", {"value": 51}, [0, 3, 4, 5]),
        ("age", "between", {"value": 28, "value2": 45}, [0, 3, 4]),
        ("age", "in", {"values": [28, "45"]}, [3, 4]),
        ("age", "not_in", {"values": [28]}, [0, 2, 4, 5]),
        ("age", "is_missing", {}, [1]),
        ("age", "not_missing", {}, [0, 2, 3, 4, 5]),
        ("city", "==", {"value": "Berlin"}, [0, 3]),
        ("city", "!=", {"value": "Berlin"}, [1, 4, 5]),
        ("city", "contains", {"value": "BUR"}, [1]),
        ("city", "in", {"values": ["Munich", "?"]}, [4, 5]),
        ("city", "not_in", {"values": ["Munich", "?"]}, [0, 1, 3]),
        ("member", "==", {"value": True}, [0, 3, 5]),
        ("member", "==", {"value": "no"}, [1, 4]),
        ("joined", ">", {"value": "2021-12-24"}, [1, 4]),
        ("joined", "between", {"value": "2021-01-01", "value2": "2021-12-31"}, [0, 5]),
    ],
)
def test_filter_rows(people, column, operator, extra, kept):
    source = people.with_split(np.array([TRAIN, VAL, TEST, TRAIN, VAL, TEST]))
    result = check_code(source, "filter_rows", {"column": column, "operator": operator, **extra})
    assert values(result.table, "name") == [values(people, "name")[i] for i in kept]
    assert result.table.split.tolist() == [source.split[i] for i in kept]


def test_filter_rows_errors(people):
    with pytest.raises(DataError, match="“contains” works on text"):
        apply(people, "filter_rows", {"column": "age", "operator": "contains", "value": "3"})
    with pytest.raises(DataError, match="holds text"):
        apply(people, "filter_rows", {"column": "city", "operator": ">", "value": "B"})
    with pytest.raises(DataError, match="enter a value"):
        apply(people, "filter_rows", {"column": "age", "operator": ">", "value": None})
    with pytest.raises(DataError, match="'old' is not a number"):
        apply(people, "filter_rows", {"column": "age", "operator": ">", "value": "old"})
    with pytest.raises(DataError, match="not a date"):
        apply(people, "filter_rows", {"column": "joined", "operator": ">", "value": "soon"})
    with pytest.raises(DataError, match="must not be larger"):
        apply(
            people, "filter_rows", {"column": "age", "operator": "between", "value": 5, "value2": 1}
        )
    with pytest.raises(DataError, match="add at least one value"):
        apply(people, "filter_rows", {"column": "age", "operator": "in", "values": []})
    with pytest.raises(DataError, match="would remove all 6 rows"):
        apply(people, "filter_rows", {"column": "age", "operator": ">", "value": 1000})


def test_filter_summary(people):
    result, _, _ = apply(people, "filter_rows", {"column": "age", "operator": ">=", "value": 30})
    assert result.summary == "Kept 3 of 6 rows where age ≥ 30 (removed 3)."


def test_drop_missing(people):
    any_missing = check_code(people, "drop_missing", {"columns": [], "how": "any"})
    assert values(any_missing.table, "name") == ["Ann", "Dana", "Fay"]
    one = check_code(people, "drop_missing", {"columns": ["age"]})
    assert one.summary == "Removed 1 row (16.7 %) with a missing age."
    both = check_code(people, "drop_missing", {"columns": ["age", "income"], "how": "all"})
    assert both.table.n_rows == 6
    empty = table({"a": [np.nan, 1.0], "b": [2.0, np.nan]})
    with pytest.raises(DataError, match="no row would be left"):
        apply(empty, "drop_missing", {})


def test_sort_rows_is_stable_with_missing_values_last(people):
    source = people.with_split(np.array([TRAIN, VAL, TEST, TRAIN, VAL, TEST]))
    result = check_code(source, "sort_rows", {"column": "member"})
    assert values(result.table, "name") == ["Bob", None, "Ann", "Dana", "Fay", "Cem"]
    assert result.table.split.tolist() == [VAL, VAL, TRAIN, TRAIN, TEST, TEST]
    down = check_code(source, "sort_rows", {"column": "joined", "descending": True})
    assert values(down.table, "name")[0] is None  # 2023-11-02
    assert "newest first" in down.summary
    assert check_code(source, "sort_rows", {"column": "city"}).table.n_rows == 6


# --------------------------------------------------------------------------- #
# Missing values
# --------------------------------------------------------------------------- #


def test_drop_sparse_columns_learns_from_training_rows_only():
    split = np.array([TRAIN] * 4 + [VAL] * 4)
    source = table(
        {
            "train_sparse": [np.nan, np.nan, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0],
            "val_sparse": [1.0, 1.0, 1.0, 1.0, np.nan, np.nan, np.nan, np.nan],
            "y": [np.nan, np.nan, np.nan, 1.0, 0.0, 1.0, 0.0, 1.0],
        },
        split=split,
    )
    result = check_code(source, "drop_sparse_columns", {"threshold": 0.5}, TARGET_CTX)
    assert result.table.columns == ["val_sparse", "y"]  # the target is never dropped
    assert result.fitted == {"columns": ["train_sparse"], "missing": {"train_sparse": 0.75}}
    assert "'train_sparse'" in result.summary


def test_impute_median_learns_from_training_rows_only():
    split = np.array([TRAIN, TRAIN, TRAIN, VAL, TEST, VAL])
    source = table({"x": [1.0, 2.0, 30.0, 1e9, np.nan, np.nan]}, split=split)
    result = check_code(source, "impute", {"strategy": "median"})
    assert values(result.table, "x")[-2:] == [2.0, 2.0]  # not influenced by 1e9 (validation)
    assert result.fitted["values"] == {"x": 2.0}
    assert result.summary == "Filled 2 missing values in 1 column with the median."
    assert result.details == [{"column": "x", "text": "median 2"}]
    mean = check_code(source, "impute", {"strategy": "mean"})
    assert mean.fitted["values"] == {"x": 11.0}


def test_impute_empty_columns_skips_unsuitable_ones_and_the_target(people):
    result = check_code(people, "impute", {"columns": [], "strategy": "median"}, TARGET_CTX)
    assert set(result.fitted["kinds"]) == {"age", "income", "member", "joined"}
    assert "Skipped 2 columns with categories ('name' and 'city')" in result.warnings[0]
    assert values(result.table, "member")[2] in (0.0, 1.0)  # booleans stay 0/1
    assert values(result.table, "joined")[2] == pd.Timestamp("2021-12-24")
    with pytest.raises(DataError, match="'city' holds categories; the median only works"):
        apply(people, "impute", {"columns": ["city"], "strategy": "median"})


def test_impute_most_frequent_constant_and_indicator(people):
    frequent = check_code(
        people, "impute", {"columns": ["city", "name"], "strategy": "most_frequent"}
    )
    assert values(frequent.table, "city")[2] == "Berlin"
    assert values(frequent.table, "name")[4] == "Ann"  # ties: the smallest value
    constant = check_code(
        people,
        "impute",
        {"columns": [], "strategy": "constant", "fill_value": "unknown", "add_indicator": True},
        TARGET_CTX,
    )
    assert values(constant.table, "city")[2] == "unknown"
    assert constant.table.columns[:3] == ["name", "name_missing", "age"]
    assert values(constant.table, "name_missing") == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert constant.table.kinds["name_missing"] == "boolean"
    assert "Skipped" in constant.warnings[0]  # 'unknown' is not a number
    zero = check_code(
        people, "impute", {"columns": ["age", "joined"], "strategy": "constant", "fill_value": 0}
    )
    assert values(zero.table, "age")[1] == 0.0
    with pytest.raises(DataError, match="enter the value"):
        apply(people, "impute", {"strategy": "constant"})
    with pytest.raises(DataError, match="'x' is not a number"):
        apply(people, "impute", {"columns": ["age"], "strategy": "constant", "fill_value": "x"})


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        ("forward_fill", [None, 1.0, 1.0, 4.0, 4.0]),
        ("backward_fill", [1.0, 1.0, 4.0, 4.0, None]),
        ("interpolate", [1.0, 1.0, 2.5, 4.0, 4.0]),
    ],
)
def test_impute_by_row_order(strategy, expected):
    source = table({"x": [np.nan, 1.0, np.nan, 4.0, np.nan]})
    result = check_code(source, "impute", {"strategy": strategy})
    assert values(result.table, "x") == expected


def test_knn_matches_a_brute_force_computation():
    rng = np.random.default_rng(3)
    n = 60
    frame = pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(size=n) * 10, "c": rng.normal(size=n)}
    )
    frame.loc[rng.choice(n, 10, replace=False), "a"] = np.nan
    frame.loc[rng.choice(n, 8, replace=False), "b"] = np.nan
    split = np.where(np.arange(n) < 40, TRAIN, VAL)
    source = table(frame.to_dict("list"), split=split)
    result = check_code(source, "impute", {"strategy": "knn", "k": 3})
    filled = result.table.df

    train = frame[split == TRAIN]
    mean, std = train.mean(), train.std(ddof=0)
    z = (frame - mean) / std
    for i in np.flatnonzero(frame["a"].isna()):
        candidates = [j for j in np.flatnonzero(split == TRAIN) if not np.isnan(frame["a"][j])]
        distances = []
        for j in candidates:
            both = ~(z.iloc[i].isna() | z.iloc[j].isna())
            d = ((z.iloc[i][both] - z.iloc[j][both]) ** 2).sum() * 3 / both.sum()
            distances.append((d, j))
        nearest = [j for _, j in sorted(distances)[:3]]
        assert filled["a"][i] == pytest.approx(frame["a"][nearest].mean())
    assert filled["a"].notna().all() and filled["b"].notna().all()
    assert result.fitted["reference_rows"] == 40
    assert "_reference" in result.fitted


def test_knn_fill_falls_back_to_the_mean_and_rounds_booleans():
    df = pd.DataFrame({"x": [np.nan, 1.0, 0.0, 1.0], "f": [np.nan, 2.0, 2.0, 2.0]})
    reference = {"x": [1.0, 0.0, 1.0], "f": [2.0, 2.0, 2.0]}
    out = knn_fill(
        df, ["x"], ["x", "f"], {"x": 0.7, "f": 2.0}, {"x": 1.0, "f": 1.0}, reference, 2, ["x"]
    )
    assert out["x"][0] == 1.0  # no coordinates in common: mean 0.7, rounded


# --------------------------------------------------------------------------- #
# Combine
# --------------------------------------------------------------------------- #


@pytest.fixture
def regions() -> Table:
    return table(
        {
            "town": ["Berlin", "Hamburg", "Rome", None],
            "region": ["East", "North", "South", "Nowhere"],
            "age": [1.0, 2.0, 3.0, 4.0],
        }
    )


def join_ctx(other: Table, name: str = "regions") -> StepContext:
    return StepContext(
        load_source=lambda i: (
            other
            if i == 2
            else (_ for _ in ()).throw(DataError("This table does not exist (any more)."))
        ),
        source_name=lambda i: name,
        source_info=lambda i: {"filename": f"{name}.csv"},
    )


@pytest.mark.parametrize(
    ("how", "cities", "regions_"),
    [
        (
            "left",
            ["Berlin", "Hamburg", None, "Berlin", "?", "Munich"],
            ["East", "North", None, "East", None, None],
        ),
        ("inner", ["Berlin", "Hamburg", "Berlin"], ["East", "North", "East"]),
        (
            "right",
            ["Berlin", "Berlin", "Hamburg", "Rome", None],
            ["East", "East", "North", "South", "Nowhere"],
        ),
    ],
)
def test_join(people, regions, how, cities, regions_):
    other = regions
    ctx = join_ctx(other)
    params = {"source": 2, "how": how, "left_on": ["city"], "right_on": ["town"]}
    result = check_code(people, "join", params, ctx, load_source=lambda i: script_frame(other))
    assert values(result.table, "city") == cities
    assert values(result.table, "region") == regions_
    assert "age_right" in result.table.columns  # name conflict: suffix
    assert "town" not in result.table.columns  # the key is merged into 'city'
    assert result.table.kinds["region"] == "categorical"
    assert result.table.split is None


def test_outer_join_keeps_everything_and_fills_keys(people, regions):
    ctx = join_ctx(regions)
    params = {"source": 2, "how": "outer", "left_on": ["city"], "right_on": ["town"]}
    result = check_code(people, "join", params, ctx, load_source=lambda i: script_frame(regions))
    cities = values(result.table, "city")
    assert result.table.n_rows == 8  # 6 here + Rome + the row without a key
    assert "Rome" in cities and cities.count(None) == 2
    assert "2 rows only in 'regions' were added" in result.summary


def test_join_summary_and_warnings(people):
    other = table({"city": ["Berlin", "Berlin", "Munich"], "score": [1.0, 2.0, 3.0]})
    result, _, _ = apply(
        people, "join", {"source": 2, "left_on": ["city"]}, join_ctx(other, "scores")
    )
    assert result.summary == (
        "Joined 'scores' (left join on city): 3 of 6 rows found a match; 1 column added."
    )
    assert result.table.n_rows == 8
    assert "1 key appear more than once in 'scores'" in result.warnings[0]


def test_join_errors(people, regions):
    ctx = join_ctx(regions)
    with pytest.raises(DataError, match="as many columns"):
        apply(people, "join", {"source": 2, "left_on": ["city", "age"], "right_on": ["town"]}, ctx)
    with pytest.raises(DataError, match="'regions' has no column 'city'"):
        apply(people, "join", {"source": 2, "left_on": ["city"]}, ctx)
    with pytest.raises(DataError, match="holds numbers here, but 'town' holds categories"):
        apply(people, "join", {"source": 2, "left_on": ["age"], "right_on": ["town"]}, ctx)
    with pytest.raises(DataError, match="choose at least one key"):
        apply(people, "join", {"source": 2, "left_on": []}, ctx)
    with pytest.raises(DataError, match="does not exist"):
        apply(people, "join", {"source": 9, "left_on": ["city"]}, ctx)
    with pytest.raises(DataError, match="not available"):
        apply(people, "join", {"source": 2, "left_on": ["city"]}, StepContext())
    many = table({"k": [1.0] * 30, "v": list(range(30))})
    with pytest.raises(DataError, match="would create 900 rows because 1 key appear"):
        apply(many, "join", {"source": 2, "left_on": ["k"]}, join_ctx(many))
    with pytest.raises(DataError, match="no rows"):
        apply(
            people,
            "join",
            {"source": 2, "how": "inner", "left_on": ["name"], "right_on": ["town"]},
            ctx,
        )


def test_join_never_matches_missing_keys():
    left = table({"k": [1.0, np.nan], "a": ["x", "y"]})
    right = table({"k": [np.nan, 1.0], "b": ["missing", "one"]})
    result = check_code(
        left,
        "join",
        {"source": 2, "left_on": ["k"]},
        join_ctx(right),
        load_source=lambda i: script_frame(right),
    )
    assert values(result.table, "b") == ["one", None]


def test_append_unions_columns_and_resolves_kind_conflicts():
    first = table(
        {
            "id": [1.0, 2.0],
            "when": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "only_here": [1.0, 0.0],
        },
        kinds={"only_here": "boolean"},
    )
    second = table(
        {"id": ["a3", "4"], "when": pd.to_datetime(["2024-02-01", None]), "extra": ["x", "y"]}
    )
    ctx = StepContext(
        load_source=lambda i: second, source_name=lambda i: {1: "jan", 2: "feb"}[i], source_id=1
    )
    params = {"source": 2, "add_source_column": True}
    result = check_code(first, "append", params, ctx, load_source=lambda i: script_frame(second))
    t = result.table
    assert t.columns == ["id", "when", "only_here", "extra", "source"]
    assert t.kinds == {
        "id": "categorical",
        "when": "datetime",
        "only_here": "boolean",
        "extra": "categorical",
        "source": "categorical",
    }
    assert values(t, "id") == ["1", "2", "a3", "4"]
    assert values(t, "only_here") == [1.0, 0.0, None, None]
    assert values(t, "source") == ["jan", "jan", "feb", "feb"]
    assert any("'id' holds numbers here but categories" in w for w in result.warnings)
    assert result.summary == "Appended 2 rows of 'feb'; the data has 4 rows now."
    with pytest.raises(DataError, match="has a column 'extra' already"):
        apply(
            first, "append", {"source": 2, "add_source_column": True, "source_column": "extra"}, ctx
        )


def test_combine_steps_refuse_tables_with_a_split(people, regions):
    split = people.with_split(np.zeros(people.n_rows, dtype=np.int8))
    with pytest.raises(DataError, match="before all other steps"):
        apply(
            split,
            "join",
            {"source": 2, "left_on": ["city"], "right_on": ["town"]},
            join_ctx(regions),
        )


def test_step_code_runs_on_new_data_with_nan_strings(people):
    """Generated code must not depend on None vs NaN in string columns."""
    result, cleaned, ctx = apply(
        people, "impute", {"columns": ["city"], "strategy": "most_frequent"}
    )
    from brickdnn.data.steps import get_step

    lines = get_step("impute")({}).to_pandas(cleaned, result.fitted, ctx)
    new = pd.DataFrame({"city": [np.nan, "Rome", None]})
    assert run_code(lines, new)["city"].tolist() == ["Berlin", "Rome", "Berlin"]


@pytest.mark.parametrize("strategy", ["forward_fill", "backward_fill", "interpolate"])
def test_ordered_fills_stay_within_their_part_of_the_split(strategy):
    """Rows of the split alternate; a fill must never take a value from another part."""
    from brickdnn.data.steps import get_step

    values = [1.0, None, 100.0, None, 3.0, None, 200.0, None, 5.0]
    split = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int8)
    table = Table.from_frame(pd.DataFrame({"a": values}), split=split)
    ctx = StepContext()
    step = get_step("impute")({"columns": ["a"], "strategy": strategy})
    filled = step.run(table, step.clean(table, ctx), ctx).table.df["a"]
    train = filled[split == 0].dropna()
    val = filled[split == 1].dropna()
    assert set(train) <= {1.0, 2.0, 3.0, 4.0, 5.0}  # interpolation may create values in between
    assert set(val) <= {100.0, 150.0, 200.0}
