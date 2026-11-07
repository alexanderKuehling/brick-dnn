"""Preparation steps: scaling, mathematical and power transforms, quantile transform, binning.

The power transforms (Yeo-Johnson, Box-Cox) find their λ by maximum likelihood on the
training rows like ``scipy.stats``, without scipy: the profile log-likelihood is evaluated on
a grid over [-5, 5] and refined with a golden-section search around the best grid point.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ..table import DataError, Table
from .base import Param, Step, StepContext, StepResult, register
from .outliers import (
    COLUMNS_PARAM,
    NUMERIC,
    check_new_names,
    chosen_columns,
    code_float,
    code_insert_after,
    fmt_count,
    fmt_number,
    insert_after,
    name_list,
    no_columns_result,
    plain,
    plural,
    train_values,
)

# --------------------------------------------------------------------------- #
# Scale
# --------------------------------------------------------------------------- #

SCALE_TEXT = {
    "standard": "Standardised {cols} (mean 0, standard deviation 1 on the training rows).",
    "minmax": "Scaled {cols} to {low} … {high} (minimum and maximum of the training rows).",
    "robust": "Scaled {cols} robustly (median 0, IQR 1 on the training rows).",
    "maxabs": "Scaled {cols} to -1 … 1 by the largest absolute value in the training rows.",
}


@register
class ScaleStep(Step):
    type = "scale"
    label = "Scale numbers"
    group = "transform"
    description = (
        "Bring numbers to a common scale: standard, min–max, robust or max-abs. Training "
        "standardises all features anyway, so this matters for robust or min–max scaling and "
        "for the exported data."
    )
    fitted = True
    params = (
        COLUMNS_PARAM,
        Param(
            "method",
            "select",
            "Method",
            default="standard",
            options=(
                ("standard", "Standard (mean 0, std 1)"),
                ("minmax", "Min–max (to a range)"),
                ("robust", "Robust (median 0, IQR 1)"),
                ("maxabs", "Max-abs (to -1 … 1)"),
            ),
            help=(
                "Standard: (x − mean) / std. Min–max: the training minimum and maximum become "
                "the ends of the range. Robust: (x − median) / IQR, little affected by "
                "outliers. Max-abs: x / max |x|, keeps zeros and signs."
            ),
        ),
        Param(
            "range_min",
            "number",
            "Range from",
            default=0,
            visible_if={"param": "method", "values": ["minmax"]},
        ),
        Param(
            "range_max",
            "number",
            "Range to",
            default=1,
            visible_if={"param": "method", "values": ["minmax"]},
        ),
    )

    def clean(self, table: Table, ctx: StepContext) -> dict[str, Any]:
        params = super().clean(table, ctx)
        if params["method"] == "minmax" and params["range_min"] >= params["range_max"]:
            raise DataError("Range: the lower end must be smaller than the upper end.")
        return params

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        columns = chosen_columns(table, params["columns"], ctx, NUMERIC)
        if not columns:
            return no_columns_result(table, "numeric columns")
        method = params["method"]
        centers: dict[str, float] = {}
        scales: dict[str, float] = {}
        details: list[dict[str, Any]] = []
        warnings: list[str] = []
        constant: list[str] = []
        result = table
        for column in columns:
            values = train_values(table, column)
            if len(values) == 0:
                warnings.append(f"'{column}' has no values in the training rows; left unchanged.")
                continue
            if method == "standard":
                center, spread = values.mean(), values.std()
                text = f"mean {fmt_number(center)}, std {fmt_number(spread)}"
            elif method == "minmax":
                low, high = values.min(), values.max()
                text = f"min {fmt_number(low)}, max {fmt_number(high)}"
                spread = high - low
            elif method == "robust":
                q1, center, q3 = np.quantile(values, [0.25, 0.5, 0.75])
                spread = q3 - q1
                text = f"median {fmt_number(center)}, IQR {fmt_number(spread)}"
            else:
                center, spread = 0.0, np.abs(values).max()
                text = f"max |x| {fmt_number(spread)}"
            if spread == 0 or not math.isfinite(spread):
                spread = 1.0
                constant.append(column)
            if method == "minmax":
                scale = spread / (params["range_max"] - params["range_min"])
                center = low - params["range_min"] * scale
            else:
                scale = spread
            centers[column], scales[column] = plain(center), plain(scale)
            scaled = (table.df[column] - centers[column]) / scales[column]
            result = result.with_column(column, scaled, "numeric")
            details.append({"column": column, "text": text})
        if constant:
            warnings.append(
                f"No spread in the training rows of {name_list(constant)}; only shifted."
            )
        summary = SCALE_TEXT[method].format(
            cols=plural(len(centers), "column"),
            low=fmt_number(params["range_min"]),
            high=fmt_number(params["range_max"]),
        )
        return StepResult(result, summary, details, {"center": centers, "scale": scales}, warnings)

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        fitted = fitted or {}
        lines = []
        for column, center in fitted.get("center", {}).items():
            ref = f"df[{column!r}]"
            scale = code_float(fitted["scale"][column])
            lines.append(f"{ref} = ({ref} - {code_float(center)}) / {scale}")
        return lines or ["# Nothing to scale."]


# --------------------------------------------------------------------------- #
# Power transforms (Yeo-Johnson, Box-Cox): maximum likelihood estimate of λ
# --------------------------------------------------------------------------- #

LAMBDA_RANGE = (-5.0, 5.0)
LAMBDA_GRID = 41
LAMBDA_TOL = 1e-6
# λ is fitted on at most this many training values (a seeded random sample of them).
MAX_FIT_VALUES = 100_000
GOLDEN = (math.sqrt(5) - 1) / 2


def _log_variance(y: np.ndarray) -> float:
    var = float(np.var(y))
    return math.log(var) if var > 0 and math.isfinite(var) else math.inf


def yeo_johnson_llf(lmbda: float, x: np.ndarray, log1p_abs: np.ndarray | None = None) -> float:
    """Profile log-likelihood of the Yeo-Johnson transform (as ``scipy.stats.yeojohnson_llf``)."""
    if log1p_abs is None:
        log1p_abs = np.log1p(np.abs(x))
    positive = x >= 0
    y = np.empty_like(log1p_abs)
    with np.errstate(over="ignore", invalid="ignore"):
        if abs(lmbda) < 1e-12:
            y[positive] = log1p_abs[positive]
        else:
            y[positive] = np.expm1(lmbda * log1p_abs[positive]) / lmbda
        if abs(2 - lmbda) < 1e-12:
            y[~positive] = -log1p_abs[~positive]
        else:
            y[~positive] = -np.expm1((2 - lmbda) * log1p_abs[~positive]) / (2 - lmbda)
        log_var = _log_variance(y)
    if not math.isfinite(log_var):
        return -math.inf
    return -len(x) / 2 * log_var + (lmbda - 1) * float(np.sum(np.sign(x) * log1p_abs))


def box_cox_llf(lmbda: float, x: np.ndarray, log_x: np.ndarray | None = None) -> float:
    """Profile log-likelihood of the Box-Cox transform (as ``scipy.stats.boxcox_llf``)."""
    if log_x is None:
        log_x = np.log(x)
    # Dividing x by its geometric mean only shifts the log-likelihood by a constant, and
    # keeps x**λ in range for large values.
    centred = log_x - log_x.mean()
    with np.errstate(over="ignore", invalid="ignore"):
        y = centred if abs(lmbda) < 1e-12 else np.expm1(lmbda * centred) / lmbda
        log_var = _log_variance(y)
    if not math.isfinite(log_var):
        return -math.inf
    return (lmbda - 1) * float(log_x.sum()) - len(x) / 2 * (log_var + 2 * lmbda * log_x.mean())


def maximise(func: Callable[[float], float], low: float, high: float) -> float:
    """The argmax of ``func`` on [low, high]: grid search, then golden-section refinement."""
    grid = np.linspace(low, high, LAMBDA_GRID)
    scores = [func(float(v)) for v in grid]
    best = int(np.argmax(scores))
    a = float(grid[max(best - 1, 0)])
    b = float(grid[min(best + 1, len(grid) - 1)])
    c, d = b - GOLDEN * (b - a), a + GOLDEN * (b - a)
    fc, fd = func(c), func(d)
    while b - a > LAMBDA_TOL:
        if fc >= fd:
            b, d, fd = d, c, fc
            c = b - GOLDEN * (b - a)
            fc = func(c)
        else:
            a, c, fc = c, d, fd
            d = a + GOLDEN * (b - a)
            fd = func(d)
    candidate = (a + b) / 2
    return candidate if func(candidate) >= scores[best] else float(grid[best])


def fit_lambda(values: np.ndarray, function: str, seed: int = 0) -> float:
    """Maximum likelihood λ of ``function`` ("yeo_johnson" or "box_cox") for ``values``."""
    if len(values) > MAX_FIT_VALUES:
        rng = np.random.default_rng(seed)
        values = values[rng.choice(len(values), MAX_FIT_VALUES, replace=False)]
    if function == "box_cox":
        log_x = np.log(values)
        return maximise(lambda v: box_cox_llf(v, values, log_x), *LAMBDA_RANGE)
    log1p_abs = np.log1p(np.abs(values))
    return maximise(lambda v: yeo_johnson_llf(v, values, log1p_abs), *LAMBDA_RANGE)


def _is_zero(x: float) -> bool:
    return abs(x) < 1e-12


def yeo_johnson(series: pd.Series, lmbda: float) -> pd.Series:
    """Yeo-Johnson transform; the same operations as the generated code."""
    a = series.abs() + 1
    pos = np.log(a) if _is_zero(lmbda) else (a**lmbda - 1) / lmbda
    neg = -np.log(a) if _is_zero(2 - lmbda) else -((a ** (2 - lmbda) - 1) / (2 - lmbda))
    return pos.where(series >= 0, neg)


def _code_yeo_johnson(ref: str, lmbda: float) -> str:
    a = f"({ref}.abs() + 1)"
    pos = f"np.log{a}" if _is_zero(lmbda) else f"({a} ** {lmbda!r} - 1) / {lmbda!r}"
    lam2 = 2 - lmbda
    neg = f"-np.log{a}" if _is_zero(lam2) else f"-(({a} ** {lam2!r} - 1) / {lam2!r})"
    return f"({pos}).where({ref} >= 0, {neg})"


def box_cox(series: pd.Series, lmbda: float) -> pd.Series:
    return np.log(series) if _is_zero(lmbda) else (series**lmbda - 1) / lmbda


def _code_box_cox(ref: str, lmbda: float) -> str:
    return f"np.log({ref})" if _is_zero(lmbda) else f"({ref} ** {lmbda!r} - 1) / {lmbda!r}"


# --------------------------------------------------------------------------- #
# Transform
# --------------------------------------------------------------------------- #

# function -> (text for the summary, apply, code template, invalid values, error message)
FUNCTIONS: dict[str, dict[str, Any]] = {
    "log": {
        "text": "log",
        "apply": np.log,
        "code": "np.log({ref})",
        "invalid": lambda s: s <= 0,
        "error": "log is not defined for {count} ≤ 0 in '{column}'. Use log1p or "
        "Yeo-Johnson instead.",
    },
    "log1p": {
        "text": "log(1 + x)",
        "apply": np.log1p,
        "code": "np.log1p({ref})",
        "invalid": lambda s: s <= -1,
        "error": "log1p is not defined for {count} ≤ -1 in '{column}'. Use Yeo-Johnson instead.",
    },
    "sqrt": {
        "text": "the square root",
        "apply": np.sqrt,
        "code": "np.sqrt({ref})",
        "invalid": lambda s: s < 0,
        "error": "The square root is not defined for {count} < 0 in '{column}'. Use "
        "Yeo-Johnson instead.",
    },
    "square": {
        "text": "x²",
        "apply": lambda s: s**2,
        "code": "{ref} ** 2",
        "invalid": lambda s: s.abs() > 1e154,
        "error": "Squaring overflows for {count} in '{column}'. Scale the column first.",
    },
    "reciprocal": {
        "text": "1 / x",
        "apply": lambda s: 1 / s,
        "code": "1 / {ref}",
        "invalid": lambda s: s == 0,
        "error": "1 / x is not defined for {count} equal to 0 in '{column}'.",
    },
    "abs": {
        "text": "the absolute value",
        "apply": lambda s: s.abs(),
        "code": "{ref}.abs()",
        "invalid": None,
        "error": "",
    },
    "exp": {
        "text": "exp",
        "apply": np.exp,
        "code": "np.exp({ref})",
        "invalid": lambda s: s > 709,
        "error": "exp overflows for {count} above 709 in '{column}'. Scale the column first.",
    },
    "yeo_johnson": {
        "text": "Yeo-Johnson (λ from the training rows)",
        "invalid": None,
        "error": "",
    },
    "box_cox": {
        "text": "Box-Cox (λ from the training rows)",
        "invalid": lambda s: s <= 0,
        "error": "Box-Cox needs positive values, but '{column}' has {count} ≤ 0. Use "
        "Yeo-Johnson instead.",
    },
}
POWER = ("yeo_johnson", "box_cox")


def _skew(series: pd.Series) -> float | None:
    skew = series.dropna().skew() if series.notna().sum() >= 3 else None
    return float(skew) if skew is not None and math.isfinite(skew) else None


@register
class TransformStep(Step):
    type = "transform"
    label = "Transform numbers"
    group = "transform"
    description = (
        "Apply log, square root, a power transform (Yeo-Johnson, Box-Cox) or another "
        "function, e.g. to make skewed numbers more symmetric."
    )
    fitted = True
    params = (
        Param("columns", "columns", "Columns", default=[], column_kinds=NUMERIC),
        Param(
            "function",
            "select",
            "Function",
            default="log1p",
            options=(
                ("log", "Logarithm, ln(x)"),
                ("log1p", "log(1 + x)"),
                ("sqrt", "Square root"),
                ("square", "Square, x²"),
                ("reciprocal", "Reciprocal, 1 / x"),
                ("abs", "Absolute value"),
                ("exp", "Exponential, eˣ"),
                ("yeo_johnson", "Yeo-Johnson (power, any values)"),
                ("box_cox", "Box-Cox (power, positive values)"),
            ),
            help=(
                "log(1 + x) works for counts and amounts with zeros. Yeo-Johnson and Box-Cox "
                "pick the power λ that makes the training values most normal (maximum "
                "likelihood); Box-Cox needs values > 0, Yeo-Johnson works for all values."
            ),
        ),
        Param("replace", "bool", "Replace the column", default=True),
        Param(
            "suffix",
            "string",
            "Suffix of the new column",
            default="_log",
            visible_if={"param": "replace", "values": [False]},
            help="The new column is called <column><suffix>, e.g. income_log.",
        ),
    )

    def clean(self, table: Table, ctx: StepContext) -> dict[str, Any]:
        params = super().clean(table, ctx)
        if not params["columns"]:
            raise DataError("Columns: choose at least one column.")
        return params

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        function = params["function"]
        spec = FUNCTIONS[function]
        columns = params["columns"]
        replace = params["replace"]
        names = {c: c if replace else c + params["suffix"] for c in columns}
        if not replace:
            check_new_names(table, names.values())
        for column in columns:
            series = table.df[column]
            if spec["invalid"] is not None:
                count = int(spec["invalid"](series).sum())
                if count:
                    raise DataError(
                        spec["error"].format(count=plural(count, "value"), column=column)
                    )
        train = table.train_mask()
        lambdas: dict[str, float] = {}
        details: list[dict[str, Any]] = []
        warnings: list[str] = []
        result = table
        for column in columns:
            series = table.df[column]
            if function in POWER:
                values = train_values(table, column)
                if len(values) < 2 or np.all(values == values[0]):
                    warnings.append(
                        f"'{column}' has fewer than two different training values; left unchanged."
                    )
                    continue
                lmbda = plain(fit_lambda(values, function, ctx.seed))
                lambdas[column] = lmbda
                transformed = (yeo_johnson if function == "yeo_johnson" else box_cox)(series, lmbda)
                prefix = f"λ = {fmt_number(lmbda)} · "
            else:
                with np.errstate(all="ignore"):
                    transformed = spec["apply"](series)
                prefix = ""
            overflow = int((series.notna() & ~np.isfinite(transformed)).sum())
            if overflow:
                raise DataError(
                    f"{spec['text'].capitalize()} gives numbers too large for "
                    f"{plural(overflow, 'value')} in '{column}'. Scale the column first."
                )
            before, after = _skew(series[train]), _skew(transformed[train])
            skew = (
                f"skewness {fmt_number(before)} → {fmt_number(after)}"
                if before is not None and after is not None
                else "too few values for the skewness"
            )
            details.append({"column": names[column], "text": prefix + skew})
            if replace:
                result = result.with_column(column, transformed, "numeric")
            else:
                result = insert_after(result, column, [(names[column], transformed, "numeric")])
        done = [names[c] for c in columns if function not in POWER or c in lambdas]
        if replace:
            summary = f"Applied {spec['text']} to {plural(len(done), 'column')}."
        else:
            summary = f"Added {plural(len(done), 'column')} with {spec['text']}: {name_list(done)}."
        fitted = {"lambdas": lambdas} if function in POWER else None
        return StepResult(result, summary, details, fitted, warnings)

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        params = {**self.defaults(), **params}
        function = params["function"]
        lines = []
        if function in POWER:
            lambdas = (fitted or {}).get("lambdas", {})
            items = [(c, lambdas[c]) for c in params["columns"] if c in lambdas]
        else:
            items = [(c, None) for c in params["columns"]]
        for column, lmbda in items:
            ref = f"df[{column!r}]"
            if function == "yeo_johnson":
                expr = _code_yeo_johnson(ref, lmbda)
            elif function == "box_cox":
                expr = _code_box_cox(ref, lmbda)
            else:
                expr = FUNCTIONS[function]["code"].format(ref=ref)
            if params["replace"]:
                lines.append(f"{ref} = {expr}")
            else:
                lines += code_insert_after(column, [(column + params["suffix"], expr)])
        return lines or ["# Nothing to transform."]


# --------------------------------------------------------------------------- #
# Quantile transform
# --------------------------------------------------------------------------- #

# Probabilities are clipped to [P_CLIP, 1 - P_CLIP] before the inverse normal CDF.
P_CLIP = 1e-7
# Acklam's rational approximation of the inverse normal CDF (relative error < 1.2e-9).
ACKLAM_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
ACKLAM_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
ACKLAM_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
ACKLAM_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
ACKLAM_LOW = 0.02425


def _polynomial(coefficients: tuple[float, ...], x: np.ndarray) -> np.ndarray:
    result = np.full_like(x, coefficients[0])
    for c in coefficients[1:]:
        result = result * x + c
    return result


def norm_ppf(p: np.ndarray | pd.Series) -> np.ndarray:
    """Inverse CDF of the standard normal distribution for p in (0, 1) (NaN stays NaN)."""
    p = np.asarray(p, dtype=float)
    q = p - 0.5
    r = q * q
    central = _polynomial(ACKLAM_A, r) * q / (_polynomial(ACKLAM_B, r) * r + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.sqrt(-2 * np.log(np.minimum(p, 1 - p)))
    tail = _polynomial(ACKLAM_C, t) / (_polynomial(ACKLAM_D, t) * t + 1)
    return np.where(np.abs(q) <= 0.5 - ACKLAM_LOW, central, np.where(q < 0, tail, -tail))


# The same function in the generated script.
NORM_PPF_CODE = f"""\
def norm_ppf(p):
    # Inverse normal CDF (Acklam's approximation, relative error < 1.2e-9).
    def polynomial(coefficients, x):
        result = np.full_like(x, coefficients[0])
        for c in coefficients[1:]:
            result = result * x + c
        return result

    p = np.asarray(p, dtype=float)
    q = p - 0.5
    r = q * q
    central = polynomial({ACKLAM_A!r}, r) * q / (
        polynomial({ACKLAM_B!r}, r) * r + 1
    )
    t = np.sqrt(-2 * np.log(np.minimum(p, 1 - p)))
    tail = polynomial({ACKLAM_C!r}, t) / (
        polynomial({ACKLAM_D!r}, t) * t + 1
    )
    return np.where(np.abs(q) <= 0.5 - {ACKLAM_LOW!r}, central, np.where(q < 0, tail, -tail))
"""


def quantile_map(series: pd.Series, quantiles: np.ndarray) -> np.ndarray:
    """Position of every value among the learned quantiles, 0 … 1 (like scikit-learn's
    ``QuantileTransformer``: the average of the forward and backward interpolation, so
    repeated quantiles map to their middle)."""
    references = np.linspace(0, 1, len(quantiles))
    x = series.to_numpy(dtype=float)
    return 0.5 * (
        np.interp(x, quantiles, references) - np.interp(-x, -quantiles[::-1], -references[::-1])
    )


@register
class QuantileTransformStep(Step):
    type = "quantile_transform"
    label = "Quantile transform"
    group = "transform"
    description = (
        "Replace numbers by their rank among the training values, mapped to a uniform or "
        "normal distribution. Robust against outliers and skew."
    )
    fitted = True
    params = (
        COLUMNS_PARAM,
        Param(
            "output",
            "select",
            "Output distribution",
            default="uniform",
            options=(("uniform", "Uniform (0 … 1)"), ("normal", "Normal (mean 0, std 1)")),
            help=(
                "Uniform: the share of training values below the value. Normal: that share "
                "mapped through the inverse normal CDF."
            ),
        ),
        Param(
            "n_quantiles",
            "int",
            "Quantiles",
            default=1000,
            min=10,
            max=10000,
            help="How many quantiles of the training values are stored; values in between are "
            "interpolated. At most the number of training values is used.",
        ),
    )

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        columns = chosen_columns(table, params["columns"], ctx, NUMERIC)
        if not columns:
            return no_columns_result(table, "numeric columns")
        normal = params["output"] == "normal"
        learned: dict[str, list[float]] = {}
        details: list[dict[str, Any]] = []
        warnings: list[str] = []
        result = table
        for column in columns:
            values = train_values(table, column)
            if len(values) == 0:
                warnings.append(f"'{column}' has no values in the training rows; left unchanged.")
                continue
            n = min(params["n_quantiles"], len(values))
            quantiles = np.quantile(values, np.linspace(0, 1, n))
            learned[column] = [plain(v) for v in quantiles]
            mapped = quantile_map(table.df[column], np.asarray(learned[column]))
            if normal:
                mapped = norm_ppf(np.clip(mapped, P_CLIP, 1 - P_CLIP))
            result = result.with_column(column, mapped, "numeric")
            details.append(
                {
                    "column": column,
                    "text": f"{fmt_count(n)} quantiles, median {fmt_number(np.median(values))}",
                }
            )
        target = "a normal distribution" if normal else "a uniform distribution (0 … 1)"
        summary = (
            f"Mapped {plural(len(learned), 'column')} to {target} by their rank in the "
            "training rows."
        )
        return StepResult(result, summary, details, {"quantiles": learned}, warnings)

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        params = {**self.defaults(), **params}
        learned = (fitted or {}).get("quantiles", {})
        if not learned:
            return ["# Nothing to transform."]
        normal = params["output"] == "normal"
        lines = NORM_PPF_CODE.splitlines() + [""] if normal else []
        for column, quantiles in learned.items():
            ref = f"df[{column!r}]"
            lines += [
                f"q = np.array({quantiles!r})",
                "r = np.linspace(0, 1, len(q))",
                f"u = 0.5 * (np.interp({ref}, q, r) - np.interp(-{ref}, -q[::-1], -r[::-1]))",
            ]
            if normal:
                lines.append(f"{ref} = norm_ppf(np.clip(u, {P_CLIP!r}, 1 - {P_CLIP!r}))")
            else:
                lines.append(f"{ref} = u")
        return lines


# --------------------------------------------------------------------------- #
# Bin
# --------------------------------------------------------------------------- #


def bin_labels(cuts: list[float]) -> list[str]:
    """``["≤ 18", "(18, 30]", "> 30"]`` for cuts ``[18, 30]`` (``["all"]`` without cuts)."""
    if not cuts:
        return ["all"]
    for fmt in (fmt_number, lambda v: f"{v:.10g}", repr):
        texts = [fmt(c) for c in cuts]
        if len(set(texts)) == len(texts):
            break
    labels = [f"≤ {texts[0]}"]
    labels += [f"({a}, {b}]" for a, b in zip(texts[:-1], texts[1:], strict=True)]
    labels.append(f"> {texts[-1]}")
    return labels


def apply_bins(series: pd.Series, cuts: list[float], output: str) -> pd.Series:
    edges = [-np.inf, *cuts, np.inf]
    if output == "ordinal":
        return pd.cut(series, edges, labels=False).astype(float)
    return pd.cut(series, edges, labels=bin_labels(cuts)).astype(object)


def _code_bins(ref: str, cuts: list[float], output: str) -> str:
    edges = "[-np.inf, " + "".join(f"{c!r}, " for c in cuts) + "np.inf]"
    if output == "ordinal":
        return f"pd.cut({ref}, {edges}, labels=False).astype(float)"
    return f"pd.cut({ref}, {edges}, labels={bin_labels(cuts)!r}).astype(object)"


def _custom_cuts(edges: list[Any]) -> list[float]:
    cuts = []
    for value in edges:
        try:
            number = float(str(value).strip().replace(",", ".")) if value != "" else math.nan
        except ValueError:
            number = math.nan
        if not math.isfinite(number):
            raise DataError(f"Cut points: '{value}' is not a number.")
        cuts.append(number)
    if not cuts:
        raise DataError("Cut points: enter at least one, e.g. 18, 30, 65.")
    return sorted(set(cuts))


@register
class BinStep(Step):
    type = "bin"
    label = "Bin numbers"
    group = "transform"
    description = "Group a number column into ranges (bins), e.g. age groups."
    fitted = True
    params = (
        Param("column", "column", "Column", column_kinds=NUMERIC),
        Param(
            "method",
            "select",
            "Bins",
            default="quantile",
            options=(
                ("quantile", "Equal counts (quantiles)"),
                ("equal_width", "Equal width"),
                ("custom", "Own cut points"),
            ),
            help=(
                "Equal counts: about the same number of training rows per bin. Equal width: "
                "the training range split into equally wide bins. Every bin includes its "
                "upper edge; values beyond the training range go to the first or last bin."
            ),
        ),
        Param(
            "bins",
            "int",
            "Number of bins",
            default=5,
            min=2,
            max=100,
            visible_if={"param": "method", "values": ["quantile", "equal_width"]},
        ),
        Param(
            "edges",
            "list",
            "Cut points",
            default=[],
            visible_if={"param": "method", "values": ["custom"]},
            help="e.g. 18, 30, 65 → 4 bins: up to 18, 18–30, 30–65 and above 65.",
        ),
        Param(
            "output",
            "select",
            "Result",
            default="ordinal",
            options=(
                ("ordinal", "Bin number (0, 1, 2 …)"),
                ("categorical", "Category (range as text)"),
            ),
            help="Bin numbers keep the order; categories can be one-hot encoded afterwards.",
        ),
        Param("replace", "bool", "Replace the column", default=False),
        Param(
            "name",
            "string",
            "New column name",
            default=None,
            optional=True,
            visible_if={"param": "replace", "values": [False]},
            help="Default: <column>_bin.",
        ),
    )

    def clean(self, table: Table, ctx: StepContext) -> dict[str, Any]:
        params = super().clean(table, ctx)
        if params["method"] == "custom":
            params["edges"] = _custom_cuts(params["edges"])
        return params

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        column = params["column"]
        method = params["method"]
        name = column if params["replace"] else (params["name"] or f"{column}_bin")
        if not params["replace"]:
            check_new_names(table, [name])
        warnings: list[str] = []
        if method == "custom":
            cuts = list(params["edges"])
            how = "own cut points"
        else:
            values = train_values(table, column)
            if len(values) == 0:
                raise DataError(f"'{column}' has no values in the training rows.")
            low, high = values.min(), values.max()
            if method == "equal_width":
                cuts = np.linspace(low, high, params["bins"] + 1)[1:-1]
                how = "equal width"
            else:
                cuts = np.quantile(values, np.linspace(0, 1, params["bins"] + 1)[1:-1])
                how = "equal counts"
            cuts = [plain(c) for c in np.unique(cuts) if c < high]
            if len(cuts) + 1 < params["bins"]:
                warnings.append(
                    f"Only {len(cuts) + 1} bins instead of {params['bins']}: '{column}' has "
                    "too few different values in the training rows."
                )
        binned = apply_bins(table.df[column], cuts, params["output"])
        kind = "numeric" if params["output"] == "ordinal" else "categorical"
        if params["replace"]:
            result = table.with_column(column, binned, kind)
        else:
            result = insert_after(table, column, [(name, binned, kind)])
        labels = bin_labels(cuts)
        codes = pd.cut(table.df[column], [-np.inf, *cuts, np.inf], labels=False)
        counts = np.bincount(codes.dropna().astype(int), minlength=len(labels))
        details = [
            {"column": name, "text": f"{label}: {plural(int(count), 'row')}"}
            for label, count in list(zip(labels, counts, strict=True))[:20]
        ]
        target = "in place" if params["replace"] else f"in the new column '{name}'"
        summary = f"Binned '{column}' into {len(labels)} bins ({how}) {target}."
        fitted = None if method == "custom" else {"cuts": cuts}
        return StepResult(result, summary, details, fitted, warnings)

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        params = {**self.defaults(), **params}
        column = params["column"]
        if params["method"] == "custom":
            cuts = _custom_cuts(params["edges"])
        else:
            cuts = (fitted or {}).get("cuts", [])
        expr = _code_bins(f"df[{column!r}]", list(cuts), params["output"])
        if params["replace"]:
            return [f"df[{column!r}] = {expr}"]
        return code_insert_after(column, [(params["name"] or f"{column}_bin", expr)])
