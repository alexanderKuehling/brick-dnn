"""Statistics for the analysis views: per-column profiles, correlations, target analysis,
warnings with suggested fixes, and row samples for the charts.

Counts and statistics are exact (computed on all rows). Correlations, target associations and
duplicate checks on very large tables use a seeded random sample of rows (``ANALYSIS_ROWS``);
the profile says how many rows they are based on. Every number in the output is a plain
Python ``float``/``int`` or ``None`` (never NaN or infinity), so it can be sent as JSON.
Datetimes (statistics, histogram edges, sample values) are milliseconds since the epoch.
How the number columns relate to each other (``relations``: strongest partner of every column)
comes from :mod:`brickdnn.data.relations`, computed on the training rows.

The JSON shapes are documented in ``frontend/src/api/types.ts`` (``DataProfile``,
``ColumnProfile``, ``DataSample`` ...); keep both in sync.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from .io import _LEADING_ZERO, LEADING_ZERO_SHARE, parse_numbers
from .relations import Relations, column_relations
from .relations import to_json as relations_json
from .table import NUMERIC_KINDS, SPLIT_NAMES, Table

# Correlations / associations / duplicate checks look at most at this many rows.
ANALYSIS_ROWS = 100_000
# Correlation matrices include at most this many numeric columns (most complete first).
MAX_CORRELATION_COLUMNS = 40
# Number of most frequent values listed for categorical, text and boolean columns.
TOP_VALUES = 20
# Histograms: bins chosen with the Freedman–Diaconis rule, clamped to this range.
MIN_BINS, MAX_BINS = 10, 50

# Seed of the analysis sample (the same rows on every preview).
ANALYSIS_SEED = 0
# Listed text values are shortened to this many characters.
MAX_VALUE_LENGTH = 200
# Classes listed in the target profile.
MAX_TARGET_CLASSES = 1000
# Cramér's V merges rarer categories into one group beyond this many.
MAX_ASSOCIATION_GROUPS = 200

# Warning thresholds.
HIGH_MISSING = 0.4  # drop the column
SOME_MISSING = 0.05  # impute it
HIGH_CARDINALITY = 50
SKEWNESS = 2.0
OUTLIER_SHARE = 0.01
# At least this share of the values on both the minimum and the maximum: already clipped.
CLIPPED_SHARE = 0.005
ZERO_SHARE = 0.5
IMBALANCE = 3.0
NUMBER_SHARE = 0.9
HIGH_CORRELATION = 0.95
MAX_CORRELATION_WARNINGS = 10
ID_MIN_ROWS = 10

_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
_QUANTILE_KEYS = ("q01", "q05", "q25", "median", "q75", "q95", "q99")
_ID_NAME = re.compile(
    r"(?:^|[_\W])(?:id|uuid|guid|idx|key|nr|no)$"
    r"|^(?:id|uuid|guid|idx|index)(?:[_\W]|$)"
    r"|^(?:row|row_?(?:id|num|number|nr)|unnamed: ?\d+)$",
    re.IGNORECASE,
)
_CAMEL_ID = re.compile(r"[a-z](?:Id|ID)$")


def profile_table(
    table: Table,
    *,
    target: str | None = None,
    task: str | None = None,
    relations: Relations | bool = True,
) -> dict[str, Any]:
    """Profile of the whole table, see ``DataProfile`` in ``frontend/src/api/types.ts``.

    ``target`` / ``task`` ("classification" | "regression") add the target analysis and the
    association of every column with the target. Warnings carry a suggested fix where one
    exists (``{"type": "impute", "params": {...}}``, using the step types of
    :mod:`brickdnn.data.steps`). ``relations``: the column relations of
    :func:`brickdnn.data.relations.column_relations` if already computed, True to compute
    them, False to leave them out.
    """
    # Overflows (values near ±1e308) give infinite statistics, which become None.
    with np.errstate(all="ignore"):
        return _profile(table, target, task, relations)


def _profile(
    table: Table, target: str | None, task: str | None, relations: Relations | bool
) -> dict[str, Any]:
    df = table.df
    n = table.n_rows
    if target not in table.kinds:
        target = None
    if target is not None and task not in ("classification", "regression"):
        task = _resolve_task(table, target)
    if target is None or task is None:
        target, task = None, None

    positions = _analysis_positions(n)
    analysis = df if positions is None else df.iloc[positions]
    columns = [_column_profile(df[name], table.kinds[name]) for name in table.columns]
    target_profile = None
    if target is not None and task is not None:
        target_profile = _target_profile(table, target, task)
        _add_associations(table, analysis, columns, target, task)
    correlations = _correlations(table, analysis, columns)
    if relations is True:
        relations = column_relations(table, target)
    duplicates = _duplicate_rows(analysis) if n and table.columns else 0

    profile = {
        "n_rows": n,
        "n_columns": len(table.columns),
        "missing_cells": sum(c["missing"] for c in columns),
        "duplicate_rows": duplicates,
        "memory_bytes": _memory_bytes(table, analysis),
        "split_counts": table.split_counts(),
        "analysis_rows": len(analysis),
        "columns": columns,
        "correlations": correlations,
        "target": target_profile,
        "warnings": [],
        "relations": relations_json(relations) if isinstance(relations, Relations) else None,
    }
    profile["warnings"] = _warnings(table, profile, target, task)
    return _clean_json(profile)


def sample_table(table: Table, rows: int, *, seed: int = 0) -> dict[str, Any]:
    """A seeded random sample of rows for the charts and the data preview, see
    ``DataSample`` in ``frontend/src/api/types.ts``. Rows keep their table order.

    Values: numbers as numbers (booleans as 0/1), datetimes as milliseconds since the epoch,
    strings as strings, missing values as ``None``.
    """
    n = table.n_rows
    rows = max(int(rows), 0)
    if n <= rows:
        positions = np.arange(n, dtype=np.int64)
    else:
        positions = np.sort(np.random.default_rng(seed).choice(n, rows, replace=False))
    columns = []
    for name in table.columns:
        kind = table.kinds[name]
        values = _json_values(table.df[name].to_numpy()[positions], kind)
        columns.append({"name": name, "kind": kind, "values": values})
    split = None
    if table.split is not None:
        split = [SPLIT_NAMES[int(s)] for s in table.split[positions]]
    return {
        "n_total": n,
        "rows": len(positions),
        "seed": int(seed),
        "index": positions.tolist(),
        "split": split,
        "columns": columns,
    }


# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #


def _analysis_positions(n: int) -> np.ndarray | None:
    """Sorted positions of the rows used for correlations etc.; None means all rows."""
    if n <= ANALYSIS_ROWS:
        return None
    rng = np.random.default_rng(ANALYSIS_SEED)
    return np.sort(rng.choice(n, ANALYSIS_ROWS, replace=False))


def _floats(series: pd.Series, kind: str) -> np.ndarray:
    """Numeric and boolean values as float64, datetimes as float ms; NaN where missing."""
    if kind == "datetime":
        ns = series.to_numpy(dtype="datetime64[ns]").view(np.int64)
        result = np.floor_divide(ns, 1_000_000).astype(np.float64)
        result[ns == np.iinfo(np.int64).min] = np.nan
        return result
    result = series.to_numpy(dtype=np.float64, na_value=np.nan).copy()
    result[~np.isfinite(result)] = np.nan
    return result


def _duplicate_rows(df: pd.DataFrame) -> int:
    """Rows equal to an earlier row. Hashing whole rows once is much faster than comparing
    column by column, which matters for wide tables (hundreds of pixel columns)."""
    hashes = pd.util.hash_pandas_object(df, index=False)
    candidates = hashes.duplicated(keep=False).to_numpy()
    if not candidates.any():
        return 0
    # Confirm on the rows that share a hash, so a hash collision never counts.
    return int(df[candidates].duplicated(keep="first").sum())


def _column_profile(series: pd.Series, kind: str) -> dict[str, Any]:
    n = len(series)
    profile: dict[str, Any] = {
        "name": str(series.name),
        "kind": kind,
        "integer": False,
        "count": 0,
        "missing": 0,
        "unique": 0,
        "stats": None,
        "histogram": None,
        "top": None,
        "other_count": None,
        "mean_length": None,
        "target_association": None,
    }
    if kind in NUMERIC_KINDS or kind == "datetime":
        values = _floats(series, kind)
        present = values[~np.isnan(values)]
        count = len(present)
        integer = bool(count) and bool(np.all(np.floor(present) == present))
        profile.update(count=count, missing=n - count, unique=int(pd.unique(present).size))
        if count:
            profile["stats"] = _numeric_stats(present, dates=kind == "datetime")
        if kind == "boolean":
            profile["integer"] = True
            ones = int((present == 1).sum())
            top = [{"value": "0", "count": count - ones}, {"value": "1", "count": ones}]
            top = [t for t in top if t["count"]]
            profile["top"] = sorted(top, key=lambda t: -t["count"])
            profile["other_count"] = 0
        else:
            profile["integer"] = integer and kind == "numeric"
            if count:
                profile["histogram"] = _histogram(present, profile["integer"], profile["stats"])
        return profile

    counts = series.value_counts(sort=False, dropna=True)
    counts = counts.sort_values(ascending=False, kind="stable")
    count = int(counts.sum())
    profile.update(count=count, missing=n - count, unique=len(counts))
    top = counts.iloc[:TOP_VALUES]
    profile["top"] = [{"value": _short(str(value)), "count": int(c)} for value, c in top.items()]
    profile["other_count"] = count - int(top.sum())
    if count:
        lengths = counts.index.astype(str).str.len().to_numpy(dtype=np.float64)
        profile["mean_length"] = float(np.dot(lengths, counts.to_numpy(dtype=np.float64)) / count)
    return profile


def _memory_bytes(table: Table, analysis: pd.DataFrame) -> int:
    """Memory of the table, like ``df.memory_usage(deep=True)``; the strings of large tables
    are measured on the analysis rows and scaled up."""
    df = table.df
    total = int(df.memory_usage(deep=False).sum())
    if not len(analysis):
        return total
    scale = len(df) / len(analysis)
    for name in table.columns:
        if df[name].dtype == object:
            column = analysis[name]
            strings = column.memory_usage(deep=True, index=False) - column.memory_usage(
                deep=False, index=False
            )
            total += int(strings * scale)
    return total


def _short(text: str) -> str:
    return text if len(text) <= MAX_VALUE_LENGTH else text[: MAX_VALUE_LENGTH - 1] + "…"


# Values the skewness after the suggested transform is estimated on (seeded sample).
TRANSFORM_SAMPLE = 20_000
# The transform is suggested only if it brings the skewness below SKEWNESS or by a quarter.
TRANSFORM_GAIN = 0.75


def _skew_after_transform(x: np.ndarray, log: bool) -> float | None:
    """The skewness after the transform the warning would suggest (log1p for values ≥ 0,
    else Yeo-Johnson with its fitted λ). Columns with many equal values (counts that are
    mostly 0, say) stay skewed whatever the transform; then it is not suggested again and
    again."""
    from brickdnn.data.steps.transform import fit_lambda, yeo_johnson

    values = x
    if len(values) > TRANSFORM_SAMPLE:
        rng = np.random.default_rng(0)
        values = values[rng.choice(len(values), TRANSFORM_SAMPLE, replace=False)]
    series = pd.Series(values.astype(float))
    try:
        if log:
            done = np.log1p(series)
        else:
            done = yeo_johnson(series, fit_lambda(series.to_numpy(), "yeo_johnson"))
        skew = float(done.skew())
    except (ValueError, FloatingPointError, OverflowError):
        return None
    return skew if math.isfinite(skew) else None


def _numeric_stats(x: np.ndarray, *, dates: bool = False) -> dict[str, Any]:
    """``NumericStats`` of the non-missing values ``x`` (at least one)."""
    count = len(x)
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if count > 1 else None
    quantiles = np.quantile(x, _QUANTILES)
    stats: dict[str, Any] = {
        "mean": mean,
        "std": std,
        "min": float(x.min()),
        "max": float(x.max()),
        **{key: float(q) for key, q in zip(_QUANTILE_KEYS, quantiles, strict=True)},
        "skew": None,
        "kurtosis": None,
        "zeros": 0 if dates else int((x == 0).sum()),
        "negative": 0 if dates else int((x < 0).sum()),
    }
    constant = stats["min"] == stats["max"]
    if count >= 3 and not constant:
        series = pd.Series(x)
        stats["skew"] = float(series.skew())
        stats["kurtosis"] = float(series.kurt()) if count >= 4 else None
        if not dates and abs(stats["skew"]) > SKEWNESS:
            stats["skew_transformed"] = _skew_after_transform(x, stats["min"] >= 0)
    q25, q75 = stats["q25"], stats["q75"]
    iqr = q75 - q25
    stats["iqr_outliers"] = int(((x < q25 - 1.5 * iqr) | (x > q75 + 1.5 * iqr)).sum())
    # How many values sit exactly on the minimum and the maximum: clipping (winsorising)
    # piles values up at both ends.
    stats["at_min"] = int((x == stats["min"]).sum())
    stats["at_max"] = int((x == stats["max"]).sum())
    if std and math.isfinite(std):
        stats["z_outliers"] = int((np.abs(x - mean) > 3 * std).sum())
    else:
        stats["z_outliers"] = 0
    return stats


def _histogram(x: np.ndarray, integer: bool, stats: dict[str, Any]) -> dict[str, list] | None:
    """Bins by the Freedman–Diaconis rule; whole numbers get whole-number bins."""
    low, high = stats["min"], stats["max"]
    count = len(x)
    if not math.isfinite(high - low):
        return None  # a range beyond float64 (values near ±1e308)
    if low == high:
        edges = np.array([low - 0.5, low + 0.5])
    elif integer and high - low + 1 <= MAX_BINS:
        edges = np.arange(low - 0.5, high + 1.0, 1.0)
    else:
        iqr = stats["q75"] - stats["q25"]
        if iqr > 0:
            width = 2 * iqr / count ** (1 / 3)
            bins = math.ceil((high - low) / width)
        else:
            bins = math.ceil(math.log2(count) + 1)  # Sturges
        bins = min(max(bins, MIN_BINS), MAX_BINS)
        if integer:
            step = math.ceil((high - low + 1) / bins)
            bins = math.ceil((high - low + 1) / step)
            edges = low - 0.5 + step * np.arange(bins + 1, dtype=np.float64)
        else:
            edges = np.linspace(low, high, bins + 1)
    # Equal-width bins: numpy counts them without sorting the values.
    counts, _ = np.histogram(x, bins=len(edges) - 1, range=(edges[0], edges[-1]))
    return {"edges": edges.tolist(), "counts": counts.tolist()}


# --------------------------------------------------------------------------- #
# Target
# --------------------------------------------------------------------------- #


def _resolve_task(table: Table, target: str) -> str | None:
    from .pipeline import resolve_task  # pipeline imports the steps; import it lazily

    return resolve_task(table, {"target": target, "task": "auto"})


def _class_codes(series: pd.Series, kind: str) -> tuple[list[str], np.ndarray]:
    from .pipeline import class_labels  # the class order of exported datasets

    return class_labels(series, kind)


def _target_profile(table: Table, target: str, task: str) -> dict[str, Any]:
    kind = table.kinds[target]
    series = table.df[target]
    profile: dict[str, Any] = {
        "column": target,
        "task": task,
        "classes": None,
        "imbalance_ratio": None,
        "histogram": None,
        "stats": None,
    }
    if task == "classification":
        names, codes = _class_codes(series, kind)
        counts = np.bincount(codes[codes >= 0], minlength=len(names))
        profile["classes"] = [
            {"value": _short(name), "count": int(c)}
            for name, c in zip(names[:MAX_TARGET_CLASSES], counts, strict=False)
        ]
        if len(counts) and counts.min() > 0:
            profile["imbalance_ratio"] = float(counts.max() / counts.min())
    elif kind in NUMERIC_KINDS:
        values = _floats(series, kind)
        present = values[~np.isnan(values)]
        if len(present):
            profile["stats"] = _numeric_stats(present)
            integer = bool(np.all(np.floor(present) == present))
            profile["histogram"] = _histogram(present, integer, profile["stats"])
    return profile


def _codes(series: pd.Series, kind: str) -> np.ndarray:
    """Group codes (-1 = missing) of a categorical or boolean column."""
    if kind in NUMERIC_KINDS:
        values = _floats(series, kind)
        codes = np.full(len(values), -1, dtype=np.int64)
        present = ~np.isnan(values)
        codes[present] = pd.factorize(values[present])[0]
        return codes
    return np.asarray(pd.factorize(series.to_numpy(dtype=object))[0], dtype=np.int64)


def _add_associations(
    table: Table, analysis: pd.DataFrame, columns: list[dict[str, Any]], target: str, task: str
) -> None:
    """Set ``target_association`` of every column but the target (on the analysis rows)."""
    target_kind = table.kinds[target]
    y = analysis[target]
    if task == "classification":
        y_codes = _class_codes(y, target_kind)[1]
        y_values = None
    else:
        if target_kind not in NUMERIC_KINDS:
            return
        y_codes = None
        y_values = _floats(y, target_kind)
    for profile in columns:
        name, kind = profile["name"], profile["kind"]
        if name == target or kind in ("datetime", "text"):
            continue
        x = analysis[name]
        value: float | None
        if y_values is not None:  # regression
            if kind in NUMERIC_KINDS:
                method, value = "pearson", _pearson(_floats(x, kind), y_values)
            else:
                method, value = "eta", _eta(y_values, _codes(x, kind))
        else:
            assert y_codes is not None
            if kind == "numeric":
                method, value = "eta", _eta(_floats(x, kind), y_codes)
            else:
                method, value = "cramers_v", _cramers_v(_codes(x, kind), y_codes)
        if value is not None:
            profile["target_association"] = {"method": method, "value": round(value, 4)}


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    """|Pearson r| over the rows where both are present."""
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 3:
        return None
    a, b = a[ok], b[ok]
    a = a - a.mean()
    b = b - b.mean()
    denominator = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    if not denominator or not math.isfinite(denominator):
        return None
    return min(abs(float(np.dot(a, b)) / denominator), 1.0)


def _eta(values: np.ndarray, groups: np.ndarray) -> float | None:
    """Correlation ratio: how much of the variance of ``values`` the groups explain."""
    ok = ~np.isnan(values) & (groups >= 0)
    if ok.sum() < 3:
        return None
    v, g = values[ok], groups[ok]
    mean = v.mean()
    total = float(((v - mean) ** 2).sum())
    if not total or not math.isfinite(total):
        return None
    counts = np.bincount(g)
    sums = np.bincount(g, weights=v)
    used = counts > 0
    means = sums[used] / counts[used]
    between = float((counts[used] * (means - mean) ** 2).sum())
    return math.sqrt(min(max(between / total, 0.0), 1.0))


def _limit_groups(codes: np.ndarray) -> np.ndarray:
    """Codes 0 … k-1 with at most MAX_ASSOCIATION_GROUPS groups (rare ones merged)."""
    counts = np.bincount(codes)
    order = np.argsort(-counts, kind="stable")
    remap = np.full(len(counts), min(len(counts), MAX_ASSOCIATION_GROUPS) - 1, dtype=np.int64)
    keep = order[: MAX_ASSOCIATION_GROUPS - 1] if len(counts) > MAX_ASSOCIATION_GROUPS else order
    remap[keep] = np.arange(len(keep))
    return remap[codes]


def _cramers_v(a: np.ndarray, b: np.ndarray) -> float | None:
    """Cramér's V of two categorical variables (codes, -1 = missing)."""
    ok = (a >= 0) & (b >= 0)
    n = int(ok.sum())
    if n < 3:
        return None
    a, b = _limit_groups(a[ok]), _limit_groups(b[ok])
    rows, cols = int(a.max()) + 1, int(b.max()) + 1
    observed = np.bincount(a * cols + b, minlength=rows * cols).reshape(rows, cols)
    observed = observed[observed.sum(axis=1) > 0][:, observed.sum(axis=0) > 0]
    k = min(observed.shape)
    if k < 2:
        return None
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    return math.sqrt(min(chi2 / n / (k - 1), 1.0))


# --------------------------------------------------------------------------- #
# Correlations
# --------------------------------------------------------------------------- #


def _correlations(
    table: Table, analysis: pd.DataFrame, columns: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Pearson and Spearman matrices of the numeric and boolean columns."""
    candidates = [c for c in columns if c["kind"] in NUMERIC_KINDS]
    if len(candidates) > MAX_CORRELATION_COLUMNS:
        chosen = {
            c["name"]
            for c in sorted(candidates, key=lambda c: c["missing"])[:MAX_CORRELATION_COLUMNS]
        }
        candidates = [c for c in candidates if c["name"] in chosen]
    names = [c["name"] for c in candidates]
    if len(names) < 2:
        return []
    values = pd.DataFrame(
        {name: _floats(analysis[name], table.kinds[name]) for name in names},
        index=pd.RangeIndex(len(analysis)),
    )
    constant = {c["name"] for c in candidates if c["unique"] < 2}
    result = []
    ranks = pd.DataFrame({name: _ranks(values[name].to_numpy()) for name in names})
    for method, frame in (("pearson", values), ("spearman", ranks)):
        matrix = frame.corr(method="pearson", min_periods=3).to_numpy()
        rows = []
        for i, a in enumerate(names):
            row = []
            for j, b in enumerate(names):
                value = matrix[i, j]
                if a in constant or b in constant or not math.isfinite(value):
                    row.append(None)
                elif i == j:
                    row.append(1.0)
                else:
                    row.append(round(max(min(float(value), 1.0), -1.0), 4))
            rows.append(row)
        result.append({"method": method, "columns": names, "matrix": rows})
    return result


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank); NaN stays NaN."""
    result = np.full(len(x), np.nan)
    present = ~np.isnan(x)
    _, inverse, counts = np.unique(x[present], return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    result[present] = (ends - (counts - 1) / 2.0)[inverse]
    return result


# --------------------------------------------------------------------------- #
# Warnings
# --------------------------------------------------------------------------- #


def _percent(share: float) -> str:
    value = share * 100
    if 0 < value < 0.1:
        return "< 0.1 %"
    text = f"{value:.0f}" if value >= 10 else f"{value:.1f}".removesuffix(".0")
    return f"{text} %"


def _number(value: float) -> str:
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:,.4g}"


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}" + ("" if n == 1 else "s")


def _warning(
    code: str,
    severity: str,
    column: str | None,
    message: str,
    fix_type: str | None = None,
    params: dict[str, Any] | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    fix = None
    if fix_type is not None:
        fix = {"type": fix_type, "params": params or {}, "label": label or fix_type}
    return {"code": code, "severity": severity, "column": column, "message": message, "fix": fix}


def _drop(code: str, severity: str, column: str, message: str) -> dict[str, Any]:
    return _warning(
        code, severity, column, message, "drop_columns", {"columns": [column]}, f"Drop {column}"
    )


def _looks_like_id(profile: dict[str, Any]) -> bool:
    """A different value in every row, and either an ID-like name or a typical ID shape."""
    name, kind, count = profile["name"], profile["kind"], profile["count"]
    if count < 2 or profile["unique"] != count:
        return False
    named = bool(_ID_NAME.search(name) or _CAMEL_ID.search(name))
    if kind == "numeric":
        if not profile["integer"]:
            return False
        # Unnamed: only row numbers 0/1 … n (not e.g. consecutive years).
        stats = profile["stats"]
        row_numbers = stats["min"] in (0, 1) and stats["max"] - stats["min"] + 1 == count
        return named or (count >= ID_MIN_ROWS and row_numbers)
    if kind == "categorical":
        return named or count >= ID_MIN_ROWS
    return kind == "text" and named


def _value_text(profile: dict[str, Any]) -> str:
    """The single value of a constant column, for messages."""
    if profile["kind"] == "datetime":
        stamp = pd.Timestamp(int(profile["stats"]["min"]), unit="ms")
        return stamp.strftime("%Y-%m-%d") if stamp == stamp.normalize() else str(stamp)
    if profile["kind"] == "numeric":
        return _number(profile["stats"]["min"])
    return profile["top"][0]["value"] if profile["top"] else ""


def _number_share(series: pd.Series) -> tuple[float, bool, str]:
    """Share of the values of a string column that are numbers, whether they use a
    decimal comma, and examples of the other values."""
    counts = series.value_counts(dropna=True)
    total = counts.sum()
    if not total:
        return 0.0, False, ""
    values = counts.index.astype(str).tolist()
    weights = counts.to_numpy(dtype=np.float64)
    head = values[:200]
    if np.isnan(parse_numbers(head, ".")).all() and np.isnan(parse_numbers(head, ",")).all():
        return 0.0, False, ""
    best = (0.0, False, np.zeros(len(values), dtype=bool))
    for comma in (False, True):
        ok = ~np.isnan(parse_numbers(values, "," if comma else "."))
        share = float(weights[ok].sum() / total)
        if share > best[0]:
            best = (share, comma, ok)
    share, comma, ok = best
    others = [v for v, good in zip(values, ok, strict=True) if not good][:3]
    examples = ", ".join("'" + (v if len(v) <= 20 else v[:19] + "…") + "'" for v in others)
    if share >= NUMBER_SHARE:
        numbers = [v for v, good in zip(values, ok, strict=True) if good]
        text = "\n".join(numbers)
        if len(_LEADING_ZERO.findall(text)) >= LEADING_ZERO_SHARE * len(numbers):
            return 0.0, False, ""  # codes with leading zeros: text on purpose
    return share, comma, examples


def _column_warnings(
    table: Table, profile: dict[str, Any], n_rows: int, target: str | None
) -> list[dict[str, Any]]:
    name, kind = profile["name"], profile["kind"]
    count, missing = profile["count"], profile["missing"]
    missing_share = missing / n_rows

    if name == target:
        if not missing:
            return []
        return [
            _warning(
                "target_missing",
                "warning",
                name,
                f"The target {name} is missing in {_plural(missing, 'row')}; rows without a "
                "target cannot be used for training.",
                "drop_missing",
                {"columns": [name], "how": "any"},
                f"Drop rows without {name}",
            )
        ]
    if count == 0:
        return [_drop("high_missing", "warning", name, f"{name} has no values (all missing).")]
    if profile["unique"] == 1:
        value = _value_text(profile)
        message = (
            f"{name} has the same value in every row ({value})."
            if not missing
            else f"{name} has only one value ({value}) besides missing values."
        )
        return [_drop("constant", "warning", name, message)]
    if missing_share > HIGH_MISSING:
        message = f"{name} is missing in {_percent(missing_share)} of the rows."
        return [_drop("high_missing", "warning", name, message)]
    if _looks_like_id(profile):
        message = f"{name} has a different value in every row; it looks like an ID."
        return [_drop("id_like", "warning", name, message)]
    if kind == "text":
        length = profile["mean_length"] or 0
        message = (
            f"{name} is free text ({length:.0f} characters on average); networks cannot use "
            "it directly."
        )
        return [_drop("text", "info", name, message)]

    warnings: list[dict[str, Any]] = []
    if missing_share >= SOME_MISSING:
        strategy = "median" if kind in ("numeric", "datetime") else "most_frequent"
        label = "median" if strategy == "median" else "most frequent value"
        warnings.append(
            _warning(
                "high_missing",
                "info",
                name,
                f"{name} is missing in {_percent(missing_share)} of the rows "
                f"({_plural(missing, 'row')}).",
                "impute",
                {"columns": [name], "strategy": strategy},
                f"Impute {name} ({label})",
            )
        )

    stats = profile["stats"]
    if kind == "numeric" and stats is not None and profile["unique"] > 2:
        skew = stats["skew"]
        if skew is not None and abs(skew) > SKEWNESS:
            log = stats["min"] >= 0
            after = stats.get("skew_transformed")
            helps = after is None or abs(after) <= max(SKEWNESS, TRANSFORM_GAIN * abs(skew))
            if helps:
                warnings.append(
                    _warning(
                        "skewed",
                        "warning",
                        name,
                        f"{name} is strongly skewed (skewness {skew:.1f}).",
                        "transform",
                        {"columns": [name], "function": "log1p" if log else "yeo_johnson"},
                        f"Log-transform {name} (log1p)" if log else f"Yeo-Johnson transform {name}",
                    )
                )
            else:
                transform = "log(1 + x)" if log else "a Yeo-Johnson transform"
                warnings.append(
                    _warning(
                        "skewed",
                        "info",
                        name,
                        f"{name} is strongly skewed (skewness {skew:.1f}); {transform} would "
                        f"hardly change that ({after:.1f}), as many of its values are equal. "
                        "Trees and networks cope; a yes/no column (value > 0) or bins can "
                        "describe such a column better.",
                    )
                )
        iqr = stats["q75"] - stats["q25"]
        outliers = stats["iqr_outliers"]
        # A column clipped to its 1st–99th percentile can still have values outside 1.5 × IQR
        # (skewed columns): clipping again would change nothing, so no warning then.
        pile = max(2, CLIPPED_SHARE * count)
        clipped = stats.get("at_min", 0) >= pile and stats.get("at_max", 0) >= pile
        if iqr > 0 and outliers > OUTLIER_SHARE * count and not clipped:
            warnings.append(
                _warning(
                    "outliers",
                    "info",
                    name,
                    f"{name} has {_plural(outliers, 'outlier')} ({_percent(outliers / count)} "
                    "of the values lie outside 1.5 × IQR of the quartiles).",
                    "clip",
                    {"columns": [name], "mode": "quantiles", "lower_q": 0.01, "upper_q": 0.99},
                    f"Clip {name} to the 1st–99th percentile",
                )
            )
        if stats["zeros"] > ZERO_SHARE * count:
            warnings.append(
                _warning(
                    "many_zeros",
                    "info",
                    name,
                    f"{name} is 0 in {_percent(stats['zeros'] / count)} of the rows with a value.",
                )
            )

    if kind == "categorical":
        share, comma, examples = _number_share(table.df[name])
        if NUMBER_SHARE <= share < 1:
            warnings.append(
                _warning(
                    "mixed_types",
                    "warning",
                    name,
                    f"{name} is categorical, but {_percent(share)} of its values are numbers "
                    f"(others: {examples}). Converted, the other values become missing.",
                    "cast",
                    {"columns": [name], "to": "numeric", "decimal_comma": comma},
                    f"Convert {name} to numbers",
                )
            )
        elif profile["unique"] > HIGH_CARDINALITY:
            unique = profile["unique"]
            warnings.append(
                _warning(
                    "high_cardinality",
                    "warning",
                    name,
                    f"{name} has {unique:,} different values; one-hot encoding would create "
                    f"{unique:,} columns.",
                    "rare_categories",
                    {"columns": [name], "min_share": 0.01, "label": "Other"},
                    f"Group rare categories of {name}",
                )
            )
    return warnings


def _warnings(
    table: Table, profile: dict[str, Any], target: str | None, task: str | None
) -> list[dict[str, Any]]:
    n_rows = profile["n_rows"]
    warnings: list[dict[str, Any]] = []
    if n_rows == 0:
        return warnings
    dropped: set[str] = set()
    for column in profile["columns"]:
        found = _column_warnings(table, column, n_rows, target)
        dropped.update(
            w["column"] for w in found if w["fix"] and w["fix"]["type"] == "drop_columns"
        )
        warnings.extend(found)

    target_profile = profile["target"]
    if target_profile is not None and task == "classification":
        ratio = target_profile["imbalance_ratio"]
        if ratio is not None and ratio > IMBALANCE and len(target_profile["classes"]) >= 2:
            warnings.append(
                _warning(
                    "imbalanced",
                    "warning",
                    target,
                    f"The classes of {target} are imbalanced ({ratio:.1f} : 1 between the "
                    "largest and the smallest class).",
                    "balance_classes",
                    {"method": "oversample"},
                    "Balance classes (oversample)",
                )
            )

    duplicates = profile["duplicate_rows"]
    if duplicates:
        sampled = profile["analysis_rows"] < n_rows
        rows = "row is an exact duplicate" if duplicates == 1 else "rows are exact duplicates"
        others = "another row" if duplicates == 1 else "other rows"
        message = f"{duplicates:,} {rows} of {others}."
        if sampled:
            message = (
                f"In a sample of {profile['analysis_rows']:,} rows, "
                + message[0].lower()
                + message[1:]
            )
        warnings.append(
            _warning(
                "duplicates",
                "info",
                None,
                message,
                "drop_duplicates",
                {"columns": [], "keep": "first"},
                "Remove duplicates",
            )
        )

    warnings.extend(_correlation_warnings(profile, target, dropped))
    order = {"warning": 0, "info": 1}
    return sorted(warnings, key=lambda w: order[w["severity"]])


def _correlation_warnings(
    profile: dict[str, Any], target: str | None, dropped: set[str]
) -> list[dict[str, Any]]:
    pearson = next((m for m in profile["correlations"] if m["method"] == "pearson"), None)
    if pearson is None:
        return []
    names = pearson["columns"]
    association = {
        c["name"]: (c["target_association"] or {}).get("value", 0.0) for c in profile["columns"]
    }
    pairs = []
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            r = pearson["matrix"][i][j]
            if r is not None and abs(r) > HIGH_CORRELATION and target not in (a, b):
                pairs.append((-abs(r), i, j, r))
    warnings: list[dict[str, Any]] = []
    for _, i, j, r in sorted(pairs):
        a, b = names[i], names[j]
        if a in dropped or b in dropped:
            continue
        drop = a if target is not None and association[a] < association[b] else b
        dropped.add(drop)
        warnings.append(
            _drop(
                "high_correlation",
                "info",
                drop,
                f"{a} and {b} are almost perfectly correlated (r = {r:.2f}); one of them is "
                "enough.",
            )
        )
        if len(warnings) >= MAX_CORRELATION_WARNINGS:
            break
    return warnings


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #


def _json_values(values: np.ndarray, kind: str) -> list[Any]:
    """Column values as JSON-ready Python values (vectorised; see ``sample_table``)."""
    if kind in NUMERIC_KINDS:
        x = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(x)
        whole = finite & (np.floor(x) == x) & (np.abs(x) < 2**53)
        if kind == "boolean" or whole.sum() == finite.sum():
            result = np.where(finite, x, 0).astype(np.int64).astype(object)
        else:
            result = x.astype(object)
        result[~finite] = None
        return result.tolist()
    if kind == "datetime":
        ns = np.asarray(values, dtype="datetime64[ns]").view(np.int64)
        result = np.floor_divide(ns, 1_000_000).astype(object)
        result[ns == np.iinfo(np.int64).min] = None
        return result.tolist()
    result = np.asarray(values, dtype=object).copy()
    result[pd.isna(result)] = None
    return result.tolist()


def _clean_json(value: Any) -> Any:
    """``value`` with plain Python types only; NaN and infinity become None."""
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean_json(value.tolist())
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return str(value)
