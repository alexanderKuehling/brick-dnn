"""The advanced preparation steps: outliers, transforms, encoding, dates, balancing and
feature selection (brickdnn.data.steps.{outliers,transform,encode,dates,target,selection})."""

from __future__ import annotations

import json
import math
import statistics

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from brickdnn.data.steps import StepContext, catalog, get_step  # noqa: E402
from brickdnn.data.steps.transform import (  # noqa: E402
    box_cox_llf,
    fit_lambda,
    norm_ppf,
    yeo_johnson_llf,
)
from brickdnn.data.table import TEST, TRAIN, VAL, DataError, Table  # noqa: E402

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_table(columns: dict, split=None, kinds=None) -> Table:
    df = pd.DataFrame(columns)
    return Table.from_frame(df, kinds=kinds, split=None if split is None else np.array(split))


def run(table: Table, step_type: str, params: dict, ctx: StepContext | None = None):
    ctx = ctx or StepContext()
    step = get_step(step_type)(params)
    cleaned = step.clean(table, ctx)
    result = step.run(table, cleaned, ctx)
    # Fitted values must be plain JSON (no numpy types, no NaN / inf).
    json.dumps(result.fitted, allow_nan=False)
    assert all(isinstance(d["text"], str) for d in result.details)
    if result.table.split is not None:
        assert len(result.table.split) == result.table.n_rows
    return step, cleaned, result


def _plain(values) -> list:
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NaT:
            out.append(None)
        else:
            out.append(v)
    return out


def assert_same_frame(actual: pd.DataFrame, expected: Table) -> None:
    assert list(actual.columns) == expected.columns
    assert len(actual) == expected.n_rows
    for column in expected.columns:
        kind = expected.kinds[column]
        a, e = actual[column], expected.df[column]
        if kind in ("numeric", "boolean"):
            np.testing.assert_allclose(
                a.to_numpy(dtype=float), e.to_numpy(dtype=float), rtol=1e-12, atol=1e-12
            )
        elif kind == "datetime":
            assert a.equals(e), column
        else:
            assert _plain(a.tolist()) == _plain(e.tolist()), column


def assert_pandas_matches(
    step, params, result, table: Table, ctx: StepContext, rows: np.ndarray | None = None
) -> list[str]:
    """Run the generated pandas code on the input frame and compare with the step result
    (only the ``rows`` mask if given)."""
    lines = step.to_pandas(params, result.fitted, ctx)
    assert lines and all(isinstance(line, str) for line in lines)
    namespace = {"pd": pd, "np": np, "df": table.df.copy()}
    exec("\n".join(lines), namespace)
    if rows is None:
        assert_same_frame(namespace["df"], result.table)
    else:
        assert_same_frame(namespace["df"][rows].reset_index(drop=True), result.table.take(rows))
    return lines


def held_out_rows(table: Table) -> np.ndarray:
    """Target encoding gives training rows out-of-fold values; the script (new data) and the
    validation/test rows use the encoding learned from all training rows."""
    return table.split != TRAIN if table.split is not None else np.zeros(table.n_rows, bool)


def values(result, column: str) -> list:
    return _plain(result.table.df[column].tolist())


# A churn-like table used by the equivalence tests.
def churn_table(n: int = 300, seed: int = 0) -> Table:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "income": np.exp(rng.normal(10, 1, n)),
            "age": rng.integers(18, 80, n).astype(float),
            "score": rng.normal(0, 1, n),
            "city": rng.choice(
                ["Berlin", "Hamburg", "Munich", "Cologne", "Leipzig"],
                n,
                p=[0.4, 0.3, 0.2, 0.08, 0.02],
            ),
            "flag": rng.choice([True, False], n),
            "signup": pd.Timestamp("2024-01-01")
            + pd.to_timedelta(rng.integers(0, 700 * 24, n), unit="h"),
            "plan": rng.choice(["basic", "plus", "pro"], n),
            "churn": rng.choice(["no", "yes"], n, p=[0.75, 0.25]),
        }
    )
    df.loc[3, "income"] = np.nan
    df.loc[5, "city"] = None
    df.loc[7, "signup"] = pd.NaT
    split = rng.choice([TRAIN, VAL, TEST], n, p=[0.7, 0.15, 0.15])
    return Table.from_frame(df, split=split)


CHURN_CTX = StepContext(target="churn", task="classification", seed=3)

# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #

ADVANCED = (
    "outliers",
    "clip",
    "scale",
    "transform",
    "quantile_transform",
    "bin",
    "encode",
    "rare_categories",
    "datetime_features",
    "balance_classes",
    "feature_selection",
)


def test_catalog_lists_the_advanced_steps_with_consistent_params():
    specs = {s["type"]: s for s in catalog()["steps"]}
    json.dumps(catalog(), allow_nan=False)
    for step_type in ADVANCED:
        spec = specs[step_type]
        assert spec["label"] and spec["description"].endswith(".")
        names = [p["name"] for p in spec["params"]]
        assert len(names) == len(set(names))
        for param in spec["params"]:
            if param.get("visible_if"):
                assert param["visible_if"]["param"] in names
            if param["kind"] == "select":
                assert param["default"] in [o["value"] for o in param["options"]]
    assert specs["balance_classes"]["training_only"] is True
    assert all(specs[t]["fitted"] for t in ("outliers", "clip", "scale", "encode", "bin"))
    assert specs["datetime_features"]["fitted"] is False


def test_catalog_matches_the_frontend_fixtures():
    defaults = {t: get_step(t).defaults() for t in ("clip", "encode", "datetime_features")}
    clip = defaults["clip"]
    assert (clip["columns"], clip["mode"], clip["lower_q"], clip["upper_q"]) == (
        [],
        "quantiles",
        0.01,
        0.99,
    )
    assert defaults["encode"]["columns"] == [] and defaults["encode"]["method"] == "onehot"
    dates = defaults["datetime_features"]
    assert dates["parts"] == ["year", "month", "weekday"] and dates["drop_original"] is True
    specs = {s["type"]: s for s in catalog()["steps"]}
    clip_params = {p["name"]: p for p in specs["clip"]["params"]}
    assert clip_params["lower_q"]["visible_if"] == {"param": "mode", "values": ["quantiles"]}
    assert clip_params["columns"]["empty_means"] == "All numeric columns"


# --------------------------------------------------------------------------- #
# Outliers
# --------------------------------------------------------------------------- #

# Training values 1 … 10 and 100: Q1 = 3.5, Q3 = 8.5, IQR = 5 -> fences -4 and 16.
IQR_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000, -50]
IQR_SPLIT = [TRAIN] * 11 + [VAL, TEST]


def test_outliers_iqr_learns_bounds_from_training_rows_only():
    table = make_table({"x": IQR_VALUES}, IQR_SPLIT)
    step, params, result = run(table, "outliers", {"method": "iqr"})
    assert result.fitted == {"bounds": {"x": [-4.0, 16.0]}}
    assert values(result, "x") == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 16, -4]
    assert (
        result.summary
        == "Clipped 3 outliers (23.1 %) in 1 column to the bounds (IQR rule, k = 1.5)."
    )
    assert result.details == [{"column": "x", "text": "bounds -4 … 16 · 3 outliers"}]
    assert_pandas_matches(step, params, result, table, StepContext())


def test_outliers_zscore_mad_and_quantile_bounds():
    # mean 5, population std 2 -> z = 1.5 gives 2 … 8.
    table = make_table({"x": [2, 4, 4, 4, 5, 5, 7, 9]})
    _, _, result = run(table, "outliers", {"method": "zscore", "z": 1.5})
    assert result.fitted["bounds"]["x"] == [2.0, 8.0]
    assert values(result, "x") == [2, 4, 4, 4, 5, 5, 7, 8]

    # median 3, MAD 1 -> 3 ± 3.5 / 0.6745.
    table = make_table({"x": [1, 2, 3, 4, 100]})
    _, _, result = run(table, "outliers", {"method": "mad", "action": "set_missing"})
    lower, upper = result.fitted["bounds"]["x"]
    assert lower == pytest.approx(3 - 3.5 / 0.6745) and upper == pytest.approx(3 + 3.5 / 0.6745)
    assert values(result, "x") == [1, 2, 3, 4, None]
    assert "to missing" in result.summary

    table = make_table({"x": [float(v) for v in range(101)]})
    _, _, result = run(table, "outliers", {"method": "quantile", "lower_q": 0.05, "upper_q": 0.9})
    assert result.fitted["bounds"]["x"] == [5.0, 90.0]
    assert "outside the 5th–90th percentile" in result.summary


def test_outliers_mad_falls_back_to_the_mean_absolute_deviation():
    table = make_table({"x": [5, 5, 5, 5, 5, 6, 20]})
    _, _, result = run(table, "outliers", {"method": "mad"})
    # MAD = 0 -> scale 1.253314 · mean |x − 5| = 1.253314 · 16 / 7.
    scale = 1.253314 * 16 / 7
    assert result.fitted["bounds"]["x"] == pytest.approx([5 - 3.5 * scale, 5 + 3.5 * scale])
    assert any("mean absolute deviation" in w for w in result.warnings)


def test_outliers_remove_only_training_rows_by_default():
    table = make_table({"x": IQR_VALUES, "y": list(range(13))}, IQR_SPLIT)
    step, params, result = run(table, "outliers", {"action": "remove", "columns": ["x"]})
    # The training outlier (100) goes, the validation/test outliers stay.
    assert values(result, "y") == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
    assert result.table.split.tolist() == [TRAIN] * 10 + [VAL, TEST]
    assert result.summary.startswith("Removed 1 training row (9.1 %) with outliers in 1 column")
    assert step.to_pandas(params, result.fitted, StepContext())[0].startswith("#")

    step, params, result = run(
        table, "outliers", {"action": "remove", "remove_from": "all", "columns": ["x"]}
    )
    assert values(result, "y") == list(range(10))
    assert result.table.split.tolist() == [TRAIN] * 10
    assert result.warnings
    assert_pandas_matches(step, params, result, table, StepContext())


def test_outliers_flag_adds_a_boolean_column_after_the_column():
    table = make_table({"x": IQR_VALUES, "y": list(range(13))}, IQR_SPLIT)
    step, params, result = run(table, "outliers", {"action": "flag", "columns": ["x"]})
    assert result.table.columns == ["x", "x_outlier", "y"]
    assert result.table.kinds["x_outlier"] == "boolean"
    assert values(result, "x_outlier") == [0] * 10 + [1, 1, 1]
    assert values(result, "x") == IQR_VALUES
    assert_pandas_matches(step, params, result, table, StepContext())

    taken = make_table({"x": IQR_VALUES, "x_outlier": IQR_VALUES}, IQR_SPLIT)
    with pytest.raises(DataError, match="already a column 'x_outlier'"):
        run(taken, "outliers", {"action": "flag", "columns": ["x"]})


def test_outliers_leave_the_target_and_other_kinds_alone():
    table = make_table({"x": [1, 2, 3, 4, 100], "price": [1, 2, 3, 4, 1000], "c": list("abcde")})
    ctx = StepContext(target="price", task="regression")
    _, _, result = run(table, "outliers", {}, ctx)
    assert values(result, "price") == [1, 2, 3, 4, 1000]
    assert list(result.fitted["bounds"]) == ["x"]
    # Listed explicitly, the target is handled too.
    _, _, result = run(table, "outliers", {"columns": ["price"]}, ctx)
    assert values(result, "price")[-1] < 1000
    with pytest.raises(DataError, match="'c' is a categorical column; numeric columns needed"):
        run(table, "outliers", {"columns": ["c"]})


def test_outliers_quantiles_must_be_ordered():
    table = make_table({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(DataError, match="lower quantile must be smaller"):
        run(table, "outliers", {"method": "quantile", "lower_q": 0.5, "upper_q": 0.5})


# --------------------------------------------------------------------------- #
# Clip
# --------------------------------------------------------------------------- #


def test_clip_quantiles_from_training_rows():
    # 0 … 100 in training: 10th percentile 10, 90th 90. The test row 500 is clipped to 90
    # but does not move the bounds.
    x = [float(v) for v in range(101)] + [500.0]
    table = make_table({"x": x}, [TRAIN] * 101 + [TEST])
    step, params, result = run(table, "clip", {"lower_q": 0.1, "upper_q": 0.9})
    assert result.fitted == {"bounds": {"x": [10.0, 90.0]}}
    assert values(result, "x")[:12] == [10.0] * 11 + [11.0]
    assert values(result, "x")[-1] == 90.0
    assert result.summary == "Clipped 21 values (20.6 %) in 1 column to the 10th–90th percentile."
    assert result.details == [{"column": "x", "text": "10 … 90 · 21 values clipped"}]
    assert_pandas_matches(step, params, result, table, StepContext())


def test_clip_fixed_values_and_std():
    table = make_table({"a": [-5.0, 0.0, 5.0, 50.0], "b": [1.0, 2.0, 3.0, 4.0]})
    step, params, result = run(table, "clip", {"mode": "values", "lower": 0, "upper": 10})
    assert values(result, "a") == [0, 0, 5, 10]
    assert result.summary == "Clipped 2 values (25 %) in 2 columns to 0 … 10."
    assert_pandas_matches(step, params, result, table, StepContext())

    _, _, result = run(table, "clip", {"mode": "values", "upper": 3})
    assert values(result, "b") == [1, 2, 3, 3]
    assert "to at most 3" in result.summary

    # mean 2.5, std sqrt(1.25)
    _, _, result = run(table, "clip", {"mode": "std", "k": 1, "columns": ["b"]})
    std = math.sqrt(1.25)
    assert result.fitted["bounds"]["b"] == pytest.approx([2.5 - std, 2.5 + std])

    with pytest.raises(DataError, match="Enter a lower bound, an upper bound or both"):
        run(table, "clip", {"mode": "values"})
    with pytest.raises(DataError, match="must not be larger"):
        run(table, "clip", {"mode": "values", "lower": 5, "upper": 1})
    with pytest.raises(DataError, match="Upper quantile: must be at least 0.5"):
        run(table, "clip", {"upper_q": 0.2})


def test_clip_nothing_to_clip():
    table = make_table({"x": [1.0, 2.0, 3.0]})
    _, _, result = run(table, "clip", {"mode": "values", "lower": 0, "upper": 5})
    assert result.summary == "No values to clip in 1 column (0 … 5)."


# --------------------------------------------------------------------------- #
# Scale
# --------------------------------------------------------------------------- #

SCALE_X = [1.0, 2.0, 3.0, 4.0, 100.0]
SCALE_SPLIT = [TRAIN, TRAIN, TRAIN, TRAIN, VAL]


@pytest.mark.parametrize(
    ("params", "center", "scale"),
    [
        ({"method": "standard"}, 2.5, math.sqrt(1.25)),
        ({"method": "minmax"}, 1.0, 3.0),
        ({"method": "minmax", "range_min": -1, "range_max": 1}, 2.5, 1.5),
        ({"method": "robust"}, 2.5, 1.5),
        ({"method": "maxabs"}, 0.0, 4.0),
    ],
)
def test_scale_learns_from_training_rows(params, center, scale):
    table = make_table({"x": SCALE_X}, SCALE_SPLIT)
    step, cleaned, result = run(table, "scale", params)
    assert result.fitted["center"]["x"] == pytest.approx(center)
    assert result.fitted["scale"]["x"] == pytest.approx(scale)
    expected = [(v - center) / scale for v in SCALE_X]
    assert values(result, "x") == pytest.approx(expected)
    assert_pandas_matches(step, cleaned, result, table, StepContext())


def test_scale_minmax_range_and_constant_columns():
    table = make_table({"x": [1.0, 2.0, 3.0], "c": [7.0, 7.0, 7.0]})
    _, _, result = run(table, "scale", {"method": "minmax", "range_min": -1, "range_max": 1})
    assert values(result, "x") == pytest.approx([-1, 0, 1])
    assert values(result, "c") == [-1, -1, -1]
    assert result.summary == (
        "Scaled 2 columns to -1 … 1 (minimum and maximum of the training rows)."
    )
    assert result.warnings
    with pytest.raises(DataError, match="lower end must be smaller"):
        run(table, "scale", {"method": "minmax", "range_min": 1, "range_max": 1})


# --------------------------------------------------------------------------- #
# Transform
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        ("log", np.log),
        ("log1p", np.log1p),
        ("sqrt", np.sqrt),
        ("square", np.square),
        ("reciprocal", lambda x: 1 / x),
        ("abs", np.abs),
        ("exp", np.exp),
    ],
)
def test_transform_functions(function, expected):
    x = [0.5, 1.0, 2.0, 8.0, None]
    table = make_table({"x": x, "y": [1, 2, 3, 4, 5]})
    step, params, result = run(table, "transform", {"columns": ["x"], "function": function})
    assert values(result, "x")[:4] == pytest.approx(expected(np.array(x[:4], dtype=float)))
    assert values(result, "x")[4] is None
    assert values(result, "y") == [1, 2, 3, 4, 5]
    assert result.fitted is None
    assert_pandas_matches(step, params, result, table, StepContext())


def test_transform_domain_errors_name_the_count_and_suggest_alternatives():
    table = make_table({"income": [0.0, -3.0, 5.0, 10.0], "z": [-2.0, -1.0, 0.0, 1.0]})
    with pytest.raises(DataError) as error:
        run(table, "transform", {"columns": ["income"], "function": "log"})
    assert str(error.value) == (
        "log is not defined for 2 values ≤ 0 in 'income'. Use log1p or Yeo-Johnson instead."
    )
    with pytest.raises(DataError, match="log1p is not defined for 2 values ≤ -1 in 'z'"):
        run(table, "transform", {"columns": ["z"], "function": "log1p"})
    with pytest.raises(DataError, match="square root is not defined for 1 value < 0"):
        run(table, "transform", {"columns": ["income"], "function": "sqrt"})
    with pytest.raises(DataError, match="1 / x is not defined for 1 value equal to 0"):
        run(table, "transform", {"columns": ["z"], "function": "reciprocal"})
    with pytest.raises(DataError, match="Box-Cox needs positive values, but 'income' has 2"):
        run(table, "transform", {"columns": ["income"], "function": "box_cox"})
    with pytest.raises(DataError, match="Columns: choose at least one column"):
        run(table, "transform", {"columns": []})
    big = make_table({"x": [1.0, 800.0]})
    with pytest.raises(DataError, match="exp overflows for 1 value above 709"):
        run(big, "transform", {"columns": ["x"], "function": "exp"})
    # λ is learned on the training rows; a huge test value would overflow.
    rng = np.random.default_rng(0)
    skewed = make_table({"x": [*(-rng.gamma(1, 1, 50)), 1e300]}, [TRAIN] * 50 + [TEST])
    with pytest.raises(DataError, match="gives numbers too large for 1 value in 'x'"):
        run(skewed, "transform", {"columns": ["x"], "function": "yeo_johnson"})


def test_transform_into_a_new_column():
    table = make_table({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    step, params, result = run(
        table, "transform", {"columns": ["x"], "function": "log1p", "replace": False}
    )
    assert result.table.columns == ["x", "x_log", "y"]
    assert values(result, "x") == [1, 2]
    assert result.summary == "Added 1 column with log(1 + x): 'x_log'."
    assert_pandas_matches(step, params, result, table, StepContext())
    taken = make_table({"x": [1.0, 2.0], "x_log": [3.0, 4.0]})
    with pytest.raises(DataError, match="already a column 'x_log'"):
        run(taken, "transform", {"columns": ["x"], "replace": False})


def _direct_yeo_johnson(x: np.ndarray, lmbda: float) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    if lmbda == 0:
        out[pos] = np.log1p(x[pos])
    else:
        out[pos] = ((x[pos] + 1) ** lmbda - 1) / lmbda
    if lmbda == 2:
        out[~pos] = -np.log1p(-x[~pos])
    else:
        out[~pos] = -((-x[~pos] + 1) ** (2 - lmbda) - 1) / (2 - lmbda)
    return out


def test_power_log_likelihoods_match_the_textbook_formulas():
    rng = np.random.default_rng(1)
    x = rng.gamma(2.0, 3.0, 200)
    mixed = rng.normal(0.5, 2.0, 200)
    for lmbda in (-1.3, -0.5, 0.0, 0.4, 1.0, 2.0, 2.7):
        y = (x**lmbda - 1) / lmbda if lmbda else np.log(x)
        direct = -len(x) / 2 * np.log(y.var()) + (lmbda - 1) * np.log(x).sum()
        assert box_cox_llf(lmbda, x) == pytest.approx(direct, rel=1e-9)
        y = _direct_yeo_johnson(mixed, lmbda)
        direct = -len(mixed) / 2 * np.log(y.var()) + (lmbda - 1) * np.sum(
            np.sign(mixed) * np.log1p(np.abs(mixed))
        )
        assert yeo_johnson_llf(lmbda, mixed) == pytest.approx(direct, rel=1e-9)


def test_power_lambda_is_the_maximum_likelihood_estimate():
    rng = np.random.default_rng(2)
    x = rng.gamma(2.0, 3.0, 500)
    mixed = rng.normal(0, 1, 500) ** 3
    grid = np.linspace(-5, 5, 2001)
    for data, function, llf in (
        (x, "box_cox", box_cox_llf),
        (mixed, "yeo_johnson", yeo_johnson_llf),
    ):
        scores = [llf(v, data) for v in grid]
        best = fit_lambda(data, function)
        # The maximum likelihood: at least as good as every grid point, and next to the best.
        assert llf(best, data) >= max(scores) - 1e-9
        assert best == pytest.approx(grid[int(np.argmax(scores))], abs=0.005)
    # Log-normal data: Box-Cox λ ≈ 0; normal data: Yeo-Johnson λ ≈ 1.
    assert abs(fit_lambda(np.exp(rng.normal(0, 1, 5000)), "box_cox")) < 0.05
    assert abs(fit_lambda(rng.normal(10, 1, 5000), "yeo_johnson") - 1) < 0.15
    # The search is bounded to [-5, 5].
    assert -5 <= fit_lambda(np.array([1.0, 1.0, 1.0, 1e6]), "box_cox") <= 5


@pytest.mark.parametrize("function", ["yeo_johnson", "box_cox"])
def test_power_transform_fits_on_training_rows_and_matches_pandas(function):
    rng = np.random.default_rng(4)
    x = list(rng.gamma(1.5, 2.0, 80)) + [1e4, 2e4]
    split = [TRAIN] * 80 + [VAL, TEST]
    table = make_table({"x": x}, split)
    step, params, result = run(table, "transform", {"columns": ["x"], "function": function})
    lmbda = result.fitted["lambdas"]["x"]
    assert lmbda == pytest.approx(fit_lambda(np.array(x[:80]), function), abs=1e-9)
    assert result.details[0]["text"].startswith("λ = ")
    skew_before = pd.Series(x[:80]).skew()
    skew_after = pd.Series(values(result, "x")[:80]).skew()
    assert abs(skew_after) < abs(skew_before) / 3
    assert_pandas_matches(step, params, result, table, StepContext())


def test_yeo_johnson_handles_negative_values_and_special_lambdas():
    table = make_table({"x": [-3.0, -1.0, 0.0, 1.0, 4.0, None]})
    step = get_step("transform")({"columns": ["x"], "function": "yeo_johnson"})
    params = step.clean(table, StepContext())
    for lmbda in (0.0, 2.0, 0.5):
        fitted = {"lambdas": {"x": lmbda}}
        namespace = {"pd": pd, "np": np, "df": table.df.copy()}
        exec("\n".join(step.to_pandas(params, fitted, StepContext())), namespace)
        got = namespace["df"]["x"].to_numpy()
        expected = _direct_yeo_johnson(np.array([-3.0, -1.0, 0.0, 1.0, 4.0]), lmbda)
        np.testing.assert_allclose(got[:5], expected, rtol=1e-12)
        assert np.isnan(got[5])


# --------------------------------------------------------------------------- #
# Quantile transform
# --------------------------------------------------------------------------- #


def test_norm_ppf_matches_the_exact_inverse_normal_cdf():
    exact = statistics.NormalDist()
    p = np.concatenate([[1e-7, 1e-5, 0.001, 0.02, 0.02425, 0.3], np.linspace(0.01, 0.99, 99)])
    p = np.concatenate([p, 1 - p])
    expected = np.array([exact.inv_cdf(v) for v in p])
    np.testing.assert_allclose(norm_ppf(p), expected, rtol=2e-9, atol=1e-9)
    assert np.isnan(norm_ppf(np.array([np.nan]))[0])


def test_quantile_transform_uniform_and_normal():
    # Training values 1 … 5 -> quantiles 1 … 5 at 0, 0.25 … 1.
    table = make_table(
        {"x": [1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 10.0, -5.0, None]},
        [TRAIN] * 5 + [VAL, VAL, TEST, TEST],
    )
    step, params, result = run(table, "quantile_transform", {})
    assert result.fitted == {"quantiles": {"x": [1.0, 2.0, 3.0, 4.0, 5.0]}}
    assert values(result, "x") == pytest.approx([0, 0.25, 0.5, 0.75, 1, 0.375, 1, 0, None])
    assert_pandas_matches(step, params, result, table, StepContext())

    step, params, result = run(table, "quantile_transform", {"output": "normal"})
    got = values(result, "x")
    ppf = statistics.NormalDist().inv_cdf
    assert got[2] == pytest.approx(0, abs=1e-9)
    assert got[1] == pytest.approx(ppf(0.25), abs=1e-8)
    assert got[6] == pytest.approx(ppf(1 - 1e-7), abs=1e-8)
    assert got[7] == pytest.approx(ppf(1e-7), abs=1e-8)
    assert got[8] is None
    assert "normal distribution" in result.summary
    assert_pandas_matches(step, params, result, table, StepContext())


def test_quantile_transform_stores_at_most_n_quantiles():
    table = make_table({"x": [float(v) for v in range(1000)]})
    _, _, result = run(table, "quantile_transform", {"n_quantiles": 11})
    assert result.fitted["quantiles"]["x"] == pytest.approx(np.linspace(0, 999, 11))
    with pytest.raises(DataError, match="Quantiles: must be at least 10"):
        run(table, "quantile_transform", {"n_quantiles": 5})


# --------------------------------------------------------------------------- #
# Bin
# --------------------------------------------------------------------------- #


def test_bin_quantiles_from_training_rows():
    # Training values 1 … 8: quartile cuts 2.75, 4.5, 6.25. The test value 100 lands in the
    # last bin, the validation value -10 in the first.
    x = [1.0, 2, 3, 4, 5, 6, 7, 8, 100, -10, None]
    table = make_table({"x": x}, [TRAIN] * 8 + [TEST, VAL, VAL])
    step, params, result = run(table, "bin", {"column": "x", "bins": 4})
    assert result.fitted == {"cuts": [2.75, 4.5, 6.25]}
    assert result.table.columns == ["x", "x_bin"]
    assert values(result, "x_bin") == [0, 0, 1, 1, 2, 2, 3, 3, 3, 0, None]
    assert result.summary == "Binned 'x' into 4 bins (equal counts) in the new column 'x_bin'."
    assert result.details[0] == {"column": "x_bin", "text": "≤ 2.75: 3 rows"}
    assert_pandas_matches(step, params, result, table, StepContext())


def test_bin_equal_width_custom_and_categorical():
    table = make_table({"age": [10.0, 20, 30, 40, 50], "y": [1, 2, 3, 4, 5]})
    step, params, result = run(
        table, "bin", {"column": "age", "method": "equal_width", "bins": 2, "name": "decade"}
    )
    assert result.fitted == {"cuts": [30.0]}
    assert values(result, "decade") == [0, 0, 0, 1, 1]
    assert_pandas_matches(step, params, result, table, StepContext())

    step, params, result = run(
        table,
        "bin",
        {
            "column": "age",
            "method": "custom",
            "edges": ["18", 40],
            "output": "categorical",
            "replace": True,
        },
    )
    assert result.table.columns == ["age", "y"]
    assert result.table.kinds["age"] == "categorical"
    assert values(result, "age") == ["≤ 18", "(18, 40]", "(18, 40]", "(18, 40]", "> 40"]
    assert result.fitted is None
    assert_pandas_matches(step, params, result, table, StepContext())

    with pytest.raises(DataError, match="Cut points: 'abc' is not a number"):
        run(table, "bin", {"column": "age", "method": "custom", "edges": ["abc"]})
    with pytest.raises(DataError, match="Cut points: enter at least one"):
        run(table, "bin", {"column": "age", "method": "custom", "edges": []})


def test_bin_warns_about_fewer_bins_for_tied_values():
    table = make_table({"x": [0.0] * 8 + [1.0, 2.0]})
    _, _, result = run(table, "bin", {"column": "x", "bins": 5})
    # Quantiles at 0.2 … 0.8 of the training values: 0, 0, 0 and 0.2.
    assert result.fitted["cuts"] == pytest.approx([0.0, 0.2])
    assert result.warnings and result.warnings[0].startswith("Only 3 bins instead of 5")


# --------------------------------------------------------------------------- #
# Encode
# --------------------------------------------------------------------------- #

CITY = ["b", "a", "b", "c", None, "b", "d", "a"]
CITY_SPLIT = [TRAIN, TRAIN, TRAIN, TRAIN, TRAIN, TRAIN, VAL, TEST]


def test_onehot_uses_training_categories_and_replaces_the_column():
    table = make_table({"id": range(8), "city": CITY, "z": [0.0] * 8}, CITY_SPLIT)
    step, params, result = run(table, "encode", {})
    # 'd' only appears in a validation row: no column, all zeros.
    assert result.table.columns == ["id", "city=a", "city=b", "city=c", "z"]
    assert all(result.table.kinds[c] == "boolean" for c in ("city=a", "city=b", "city=c"))
    assert values(result, "city=a") == [0, 1, 0, 0, 0, 0, 0, 1]
    assert values(result, "city=b") == [1, 0, 1, 0, 0, 1, 0, 0]
    assert result.summary == "One-hot encoded 1 column into 3 new columns."
    assert "1 validation/test value with unseen categories" in result.details[0]["text"]
    assert_pandas_matches(step, params, result, table, StepContext())


def test_onehot_max_categories_drop_first_and_booleans():
    table = make_table(
        {"city": CITY, "flag": [True, False, True, True, None, False, True, False]},
        CITY_SPLIT,
        kinds={"flag": "boolean"},
    )
    step, params, result = run(
        table,
        "encode",
        {"columns": ["city", "flag"], "max_categories": 2, "drop_first": True},
    )
    # Most frequent in training: b (3), a (1) / c (1) -> a wins the tie alphabetically.
    assert result.table.columns == ["city=b", "city_other", "flag=1"]
    assert values(result, "city_other") == [0, 0, 0, 1, 0, 0, 1, 0]
    assert values(result, "flag=1") == [1, 0, 1, 1, 0, 0, 1, 0]
    assert result.fitted["columns"]["city"]["dropped"] == "a"
    assert_pandas_matches(step, params, result, table, StepContext())


def test_ordinal_and_frequency_encoding():
    table = make_table({"city": CITY}, CITY_SPLIT)
    step, params, result = run(table, "encode", {"method": "ordinal"})
    assert values(result, "city") == [1, 0, 1, 2, None, 1, -1, 0]
    assert result.table.kinds["city"] == "numeric"
    assert_pandas_matches(step, params, result, table, StepContext())

    step, params, result = run(table, "encode", {"method": "frequency"})
    # Training categories: b 3, a 1, c 1 of 5 non-missing values.
    assert values(result, "city") == pytest.approx([0.6, 0.2, 0.6, 0.2, None, 0.6, 0, 0.2])
    assert_pandas_matches(step, params, result, table, StepContext())


def test_target_encoding_binary_with_smoothing():
    table = make_table(
        {
            "city": ["a", "a", "a", "b", "b", "c", "a", "z"],
            "churn": ["yes", "yes", "no", "no", "no", "yes", "yes", "no"],
        },
        [TRAIN] * 6 + [VAL, TEST],
    )
    ctx = StepContext(target="churn", task="classification")
    step, params, result = run(table, "encode", {"method": "target", "smoothing": 2}, ctx)
    # prior = 3/6 = 0.5; a: (2 + 2·0.5) / (3 + 2) = 0.6; b: (0 + 1) / 4 = 0.25;
    # c: (1 + 1) / 3; unseen z: the prior. Validation/test targets are not used.
    mapping = result.fitted["columns"]["city"]["mapping"]
    assert mapping == pytest.approx({"a": 0.6, "b": 0.25, "c": 2 / 3})
    assert result.fitted["columns"]["city"]["prior"] == pytest.approx(0.5)
    # Validation and test rows use that mapping; training rows get out-of-fold values.
    assert values(result, "city")[6:] == pytest.approx([0.6, 0.5])
    assert values(result, "churn") == ["yes", "yes", "no", "no", "no", "yes", "yes", "no"]
    assert result.summary == (
        "Replaced the categories of 1 column by the smoothed share of 'churn' = 'yes'; "
        "training rows use out-of-fold values (5 folds)."
    )
    assert_pandas_matches(step, params, result, table, ctx, rows=held_out_rows(table))


def test_target_encoding_does_not_hand_training_rows_their_own_label():
    """An ID-like column: every category occurs once. In-sample means would reproduce the
    label exactly; out-of-fold values cannot know it."""
    n = 400
    rng = np.random.default_rng(1)
    churn = rng.choice(["no", "yes"], n)
    split = np.where(np.arange(n) % 5 == 0, VAL, TRAIN)
    table = make_table({"id": [f"c{i}" for i in range(n)], "churn": churn}, split)
    ctx = StepContext(target="churn", task="classification")
    _, _, result = run(
        table, "encode", {"method": "target", "smoothing": 0, "columns": ["id"]}, ctx
    )
    encoded = np.array(values(result, "id"), dtype=float)
    train = split == TRAIN
    label = (churn == "yes").astype(float)
    # Unique categories: every training row falls back to the prior of the other folds.
    assert np.corrcoef(encoded[train], label[train])[0, 1] < 0.2
    assert not np.allclose(encoded[train], label[train])


def test_target_encoding_multiclass_and_regression():
    table = make_table(
        {"city": ["a", "a", "b", "b", "c"], "kind": ["x", "y", "z", "z", "x"]},
    )
    ctx = StepContext(target="kind", task="classification")
    step, params, result = run(
        table, "encode", {"method": "target", "smoothing": 0, "columns": ["city"]}, ctx
    )
    assert result.table.columns == ["city_te_x", "city_te_y", "city_te_z", "kind"]
    info = result.fitted["columns"]["city"]
    assert info["mapping"]["x"] == pytest.approx({"a": 0.5, "b": 0, "c": 1})
    assert info["mapping"]["z"] == pytest.approx({"a": 0, "b": 1, "c": 0})

    table = make_table(
        {"city": ["a", "a", "b", None, "a", "b"], "price": [10.0, 20.0, 40.0, 50.0, 0, 0]},
        [TRAIN, TRAIN, TRAIN, TRAIN, VAL, TEST],
    )
    ctx = StepContext(target="price", task="regression")
    step, params, result = run(table, "encode", {"method": "target", "smoothing": 1}, ctx)
    # prior 30; a: (30 + 30) / 3 = 20; b: (40 + 30) / 2 = 35; missing stays missing.
    assert result.fitted["columns"]["city"]["mapping"] == pytest.approx({"a": 20, "b": 35})
    assert values(result, "city")[3:] == pytest.approx([None, 20, 35])
    assert_pandas_matches(step, params, result, table, ctx, rows=held_out_rows(table))


def test_target_encoding_errors():
    table = make_table({"city": ["a", "b"], "churn": ["no", "no"]})
    with pytest.raises(DataError, match="needs a target. Choose the target column"):
        run(table, "encode", {"method": "target"})
    ctx = StepContext(target="churn", task="classification")
    with pytest.raises(DataError, match="at least two classes"):
        run(table, "encode", {"method": "target"}, ctx)
    with pytest.raises(DataError, match="cannot be encoded with itself"):
        run(table, "encode", {"method": "target", "columns": ["churn"]}, ctx)


def test_rare_categories_from_training_rows():
    city = ["a"] * 6 + ["b"] * 3 + ["c", None, "c", "new"]
    split = [TRAIN] * 11 + [VAL, TEST]
    table = make_table({"city": city}, split)
    step, params, result = run(table, "rare_categories", {"min_share": 0.2})
    # Training shares: a 0.6, b 0.3, c 0.1 -> c is rare; unseen 'new' too.
    assert result.fitted == {"keep": {"city": ["a", "b"]}}
    assert values(result, "city") == ["a"] * 6 + ["b"] * 3 + ["Other", None, "Other", "Other"]
    assert result.summary == (
        "Grouped 2 rare categories (3 values, 25 %) into 'Other' in 1 column."
    )
    assert_pandas_matches(step, params, result, table, StepContext())


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #


def test_datetime_parts():
    dates = ["2024-03-02 14:30", "2023-12-31 08:05", None]
    table = make_table({"when": pd.to_datetime(dates), "y": [1, 2, 3]}, kinds={"when": "datetime"})
    parts = ["year", "quarter", "month", "day", "weekday", "dayofyear", "hour", "minute"]
    parts += ["is_weekend", "timestamp"]
    step, params, result = run(table, "datetime_features", {"parts": parts})
    assert result.table.columns == [f"when_{p}" for p in parts] + ["y"]
    assert values(result, "when_year") == [2024, 2023, None]
    assert values(result, "when_quarter") == [1, 4, None]
    assert values(result, "when_weekday") == [5, 6, None]
    assert values(result, "when_dayofyear") == [62, 365, None]
    assert values(result, "when_hour") == [14, 8, None]
    assert values(result, "when_minute") == [30, 5, None]
    assert values(result, "when_is_weekend") == [1, 1, None]
    assert result.table.kinds["when_is_weekend"] == "boolean"
    assert values(result, "when_timestamp")[0] == pd.Timestamp("2024-03-02 14:30").timestamp()
    assert "1 missing date" in result.details[0]["text"]
    assert_pandas_matches(step, params, result, table, StepContext())


def test_datetime_cyclical_and_keep_original():
    table = make_table({"when": pd.to_datetime(["2024-01-15", "2024-07-15"])})
    step, params, result = run(
        table,
        "datetime_features",
        {"parts": ["year", "month"], "cyclical": True, "drop_original": False},
    )
    assert result.table.columns == ["when", "when_year", "when_month_sin", "when_month_cos"]
    assert values(result, "when_month_sin") == pytest.approx([0, math.sin(math.pi)], abs=1e-12)
    assert values(result, "when_month_cos") == pytest.approx([1, -1])
    assert result.summary == (
        "Added 3 columns (year, month) from 1 date column ('when') and kept the original."
    )
    assert_pandas_matches(step, params, result, table, StepContext())


def test_datetime_all_date_columns_except_the_target():
    table = make_table(
        {
            "a": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "b": pd.to_datetime(["2020-05-05", "2021-06-06"]),
            "due": pd.to_datetime(["2025-01-01", "2025-01-02"]),
        }
    )
    ctx = StepContext(target="due")
    step, params, result = run(table, "datetime_features", {"parts": ["year"]}, ctx)
    assert result.table.columns == ["a_year", "b_year", "due"]
    lines = assert_pandas_matches(step, params, result, table, ctx)
    assert lines[0].startswith("for name in")
    with pytest.raises(DataError, match="already a column 'a_year'"):
        run(table.with_column("a_year", [1, 2], "numeric"), "datetime_features", {}, ctx)
    with pytest.raises(DataError, match="choose at least one part"):
        run(table, "datetime_features", {"parts": []}, ctx)


# --------------------------------------------------------------------------- #
# Balance classes
# --------------------------------------------------------------------------- #


def balance_table() -> Table:
    y = ["a"] * 6 + ["b"] * 2 + ["c"] * 1 + ["a", "b", "c"]
    split = [TRAIN] * 9 + [VAL, TEST, VAL]
    return make_table({"x": list(range(12)), "y": y}, split)


def test_oversample_training_rows_only():
    table = balance_table()
    ctx = StepContext(target="y", task="classification", seed=7)
    step, params, result = run(table, "balance_classes", {}, ctx)
    out = result.table
    train = out.split == TRAIN
    assert out.df["y"][train].value_counts().to_dict() == {"a": 6, "b": 6, "c": 6}
    # Validation and test rows are unchanged; every added row is a training row copy.
    assert out.df[~train]["x"].tolist() == [9, 10, 11]
    assert out.n_rows == 12 + 9
    assert out.df["x"].tolist()[:12] == list(range(12))
    assert set(out.df["x"][12:]) <= {6, 7, 8}
    assert result.summary == (
        "Oversampled the training rows to 6 per class: added 9 copies of rows of 2 classes."
    )
    assert step.training_only
    assert step.to_pandas(params, result.fitted, ctx)[0].startswith("# ")
    # Seeded: the same seed gives the same rows.
    _, _, again = run(table, "balance_classes", {}, ctx)
    assert again.table.df["x"].tolist() == out.df["x"].tolist()


def test_undersample_training_rows_only():
    table = balance_table()
    ctx = StepContext(target="y", task="classification", seed=1)
    _, _, result = run(table, "balance_classes", {"method": "undersample"}, ctx)
    out = result.table
    train = out.split == TRAIN
    assert out.df["y"][train].value_counts().to_dict() == {"a": 1, "b": 1, "c": 1}
    assert out.df[~train]["x"].tolist() == [9, 10, 11]
    assert result.summary.startswith("Undersampled the training rows to 1 per class: removed 6")
    seeds = {
        tuple(
            run(
                table,
                "balance_classes",
                {"method": "undersample"},
                StepContext(target="y", task="classification", seed=s),
            )[2].table.df["x"]
        )
        for s in range(10)
    }
    assert len(seeds) > 1


def test_balance_classes_errors_and_balanced_input():
    table = balance_table()
    with pytest.raises(DataError, match="Balancing needs a target"):
        run(table, "balance_classes", {})
    with pytest.raises(DataError, match="only works for classification; 'x' is a regression"):
        run(table, "balance_classes", {}, StepContext(target="x", task="regression"))
    balanced = make_table({"y": ["a", "b", "a", "b"]})
    _, _, result = run(balanced, "balance_classes", {}, StepContext(target="y"))
    assert result.summary == (
        "The classes are already balanced (2 training rows each); nothing changed."
    )
    assert result.table is balanced


# --------------------------------------------------------------------------- #
# Feature selection
# --------------------------------------------------------------------------- #


def selection_table() -> Table:
    rng = np.random.default_rng(5)
    n = 2000
    a = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    return make_table(
        {
            "a": a,
            # |r| with a ≈ 0.98, but less related to the target than a.
            "b": a + 0.2 * rng.normal(0, 1, n),
            "noise": noise,
            "copy": noise * 2 + 0.001 * rng.normal(0, 1, n),
            "const": [1.0] * n,
            "holes": [None] * 1500 + list(range(500)),
            "y": a + 0.1 * rng.normal(0, 1, n),
        }
    )


def test_feature_selection_correlation_keeps_the_feature_closer_to_the_target():
    table = selection_table()
    ctx = StepContext(target="y", task="regression")
    step, params, result = run(table, "feature_selection", {}, ctx)
    dropped = result.fitted["drop"]
    assert "b" in dropped and "a" not in dropped
    assert len(dropped) == 2 and {"noise", "copy"} & set(dropped)
    assert result.summary.startswith("Dropped 2 of 6 features correlated with another feature")
    assert result.details[0] == {"column": "b", "text": "|r| = 0.9791 with 'a'"}
    assert_pandas_matches(step, params, result, table, ctx)
    # Without a target, y is a feature too, and the later feature of each pair goes.
    _, _, result = run(table, "feature_selection", {}, StepContext())
    assert result.fitted["drop"] == ["b", "copy", "y"]
    _, _, result = run(table, "feature_selection", {"max_correlation": 1}, ctx)
    assert result.summary == "Kept all 6 features: no pair of features has |r| > 1."


def test_feature_selection_variance_missing_and_target():
    table = selection_table()
    ctx = StepContext(target="y", task="regression")
    _, _, result = run(table, "feature_selection", {"method": "variance"}, ctx)
    assert result.fitted["drop"] == ["const"]
    assert result.details == [{"column": "const", "text": "variance 0"}]

    _, _, result = run(table, "feature_selection", {"method": "missing", "max_missing": 0.5}, ctx)
    assert result.fitted["drop"] == ["holes"]
    assert result.details[0]["text"] == "75 % missing"

    _, _, result = run(
        table, "feature_selection", {"method": "target_correlation", "min_association": 0.3}, ctx
    )
    assert result.fitted["drop"] == ["noise", "copy", "const", "holes"]
    assert {"column": "const", "text": "constant"} in result.details
    with pytest.raises(DataError, match="needs a target"):
        run(table, "feature_selection", {"method": "target_correlation"})


def test_feature_selection_class_target_associations():
    table = make_table(
        {
            "num": [1.0, 1.1, 0.9, 5.0, 5.1, 4.9],
            "cat": ["a", "a", "a", "b", "b", "b"],
            "junk": ["x", "y", "x", "y", "x", "y"],
            "label": ["no", "no", "no", "yes", "yes", "yes"],
        }
    )
    ctx = StepContext(target="label", task="classification")
    _, _, result = run(
        table, "feature_selection", {"method": "target_correlation", "min_association": 0.5}, ctx
    )
    assert result.fitted["drop"] == ["junk"]
    assert result.details[0]["text"].startswith("Cramér's V 0.33")


# --------------------------------------------------------------------------- #
# Generated pandas code equals the step result, on a realistic table
# --------------------------------------------------------------------------- #

EQUIVALENCE_CASES = [
    ("outliers", {"method": method, "action": action})
    for method in ("iqr", "zscore", "mad", "quantile")
    for action in ("clip", "set_missing", "flag")
] + [
    ("outliers", {"action": "remove", "remove_from": "all", "method": "zscore", "z": 2}),
    ("clip", {}),
    ("clip", {"mode": "std", "k": 2}),
    ("clip", {"mode": "values", "lower": 20, "upper": 60}),
    ("scale", {"method": "standard"}),
    ("scale", {"method": "robust", "columns": ["income"]}),
    ("transform", {"columns": ["income", "age"], "function": "box_cox"}),
    ("transform", {"columns": ["score"], "function": "yeo_johnson", "replace": False}),
    ("transform", {"columns": ["income"], "function": "log", "replace": False, "suffix": "_ln"}),
    ("quantile_transform", {"output": "normal", "n_quantiles": 100}),
    ("bin", {"column": "income", "bins": 10, "output": "categorical"}),
    ("encode", {}),
    ("encode", {"columns": ["city", "plan", "flag"], "max_categories": 2}),
    ("encode", {"method": "ordinal"}),
    ("encode", {"method": "frequency"}),
    ("encode", {"method": "target", "smoothing": 5}),
    ("rare_categories", {"min_share": 0.05, "label": "rare"}),
    ("datetime_features", {}),
    ("datetime_features", {"parts": ["month", "weekday", "hour"], "cyclical": True}),
    ("feature_selection", {"method": "target_correlation", "min_association": 0.1}),
    ("feature_selection", {"method": "no_partner"}),
    ("feature_selection", {"method": "no_partner", "partner_method": "pearson"}),
]


@pytest.mark.parametrize(("step_type", "params"), EQUIVALENCE_CASES)
def test_generated_pandas_code_reproduces_the_step(step_type, params):
    table = churn_table()
    step, cleaned, result = run(table, step_type, params, CHURN_CTX)
    assert result.summary.endswith(".")
    assert "churn" in result.table.columns
    rows = held_out_rows(table) if params.get("method") == "target" else None
    assert_pandas_matches(step, cleaned, result, table, CHURN_CTX, rows=rows)


@pytest.mark.parametrize(("step_type", "params"), EQUIVALENCE_CASES)
def test_fitted_values_ignore_validation_and_test_rows(step_type, params):
    """Changing validation/test rows must not change anything the step learns."""
    table = churn_table()
    _, _, result = run(table, step_type, params, CHURN_CTX)
    other = table.df.copy()
    held_out = table.split != TRAIN
    rng = np.random.default_rng(9)
    other.loc[held_out, "income"] = rng.uniform(1e6, 1e7, held_out.sum())
    other.loc[held_out, "age"] = 200.0
    other.loc[held_out, "city"] = "Paris"
    other.loc[held_out, "churn"] = "yes"
    changed = Table(other, dict(table.kinds), table.split.copy())
    _, _, again = run(changed, step_type, params, CHURN_CTX)
    assert again.fitted == result.fitted
