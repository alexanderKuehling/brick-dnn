"""Preparation steps for missing values: mark placeholder values as missing, drop columns
that are mostly empty and fill missing values (imputation).

Imputation values (mean, median, most frequent value, KNN reference rows) are learned from
the training rows only and then used for all rows.
"""

from __future__ import annotations

import inspect
import textwrap
from typing import Any

import numpy as np
import pandas as pd

from ..table import NUMERIC_KINDS, STRING_KINDS, DataError, Table
from .base import Param, Step, StepContext, StepResult, register
from .columns import (
    KIND_NAMES,
    as_label,
    as_number,
    as_timestamp,
    cell_value,
    chosen,
    code_value,
    fmt_count,
    fmt_number,
    fmt_share,
    from_fitted,
    plural,
    quoted,
    set_columns,
    storage,
    to_fitted,
)

# --------------------------------------------------------------------------- #
# Mark placeholders as missing
# --------------------------------------------------------------------------- #


def _sentinels(values: list[Any], kind: str) -> list[Any]:
    """The values of ``values`` that can occur in a ``kind`` column, in storage format."""
    result: list[Any] = []
    for value in values:
        if kind in NUMERIC_KINDS:
            converted: Any = as_number(value)
        elif kind == "datetime":
            converted = as_timestamp(value) if isinstance(value, str) else None
        else:
            converted = as_label(value)
        if converted is not None and converted not in result:
            result.append(converted)
    return result


@register
class MarkMissing(Step):
    type = "mark_missing"
    label = "Mark values as missing"
    group = "missing"
    description = "Treat placeholder values like -999, ? or unknown as missing values."
    params = (
        Param(
            "columns",
            "columns",
            "Columns",
            default=[],
            optional=True,
            empty_means="All columns",
        ),
        Param(
            "values",
            "list",
            "Placeholder values",
            default=[],
            help="e.g. -999, ?, unknown. Numbers are compared as numbers in numeric columns.",
        ),
    )

    def clean(self, table: Table, ctx: StepContext) -> dict[str, Any]:
        params = super().clean(table, ctx)
        if not params["values"]:
            raise DataError("Placeholder values: add at least one value, e.g. -999 or ?.")
        return params

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        columns = chosen(table, params["columns"], ctx)
        updates: dict[str, tuple[pd.Series, str]] = {}
        applied: dict[str, list[Any]] = {}
        details = []
        total = 0
        for column in columns:
            kind = table.kinds[column]
            values = _sentinels(params["values"], kind)
            if not values:
                continue
            applied[column] = [to_fitted(v) for v in values]
            series = table.df[column]
            hits = series.isin(values)
            count = int(hits.sum())
            if count:
                updates[column] = (storage(series.mask(hits), kind), kind)
                total += count
                details.append({"column": column, "text": plural(count, "value")})
        result = set_columns(table, updates) if updates else table
        summary = (
            f"Marked {plural(total, 'value')} as missing in {plural(len(updates), 'column')}."
            if total
            else "None of the placeholder values were found; nothing changed."
        )
        fitted = {"values": applied, "kinds": {c: table.kinds[c] for c in applied}}
        return StepResult(result, summary, details=details, fitted=fitted)

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        fitted = fitted or {}
        lines = []
        for column, values in fitted.get("values", {}).items():
            kind = fitted["kinds"][column]
            literal = "[" + ", ".join(code_value(from_fitted(v, kind)) for v in values) + "]"
            ref = f"df[{column!r}]"
            lines.append(f"{ref} = {ref}.mask({ref}.isin({literal}))")
        return lines or ["# No column can hold these values."]


# --------------------------------------------------------------------------- #
# Drop mostly empty columns
# --------------------------------------------------------------------------- #


@register
class DropSparseColumns(Step):
    type = "drop_sparse_columns"
    label = "Drop mostly empty columns"
    group = "missing"
    description = "Remove columns with many missing values in the training rows."
    fitted = True
    params = (
        Param(
            "threshold",
            "number",
            "Drop if more missing than",
            default=0.5,
            min=0,
            max=1,
            step=0.05,
            help="Share of missing values in the training rows, e.g. 0.5 = more than half. "
            "The target is never dropped.",
        ),
    )

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        threshold = params["threshold"]
        train = table.train_df()
        candidates = [c for c in table.columns if c != ctx.target]
        if len(train) == 0:
            return StepResult(
                table,
                "There are no training rows; nothing dropped.",
                fitted={"columns": [], "missing": {}},
            )
        shares = train[candidates].isna().mean() if candidates else pd.Series(dtype=float)
        dropped = [c for c in candidates if float(shares[c]) > threshold]
        missing = {c: round(float(shares[c]), 6) for c in dropped}
        details = [{"column": c, "text": f"{fmt_share(shares[c], 1)} missing"} for c in dropped]
        limit = fmt_share(threshold, 1)
        if not dropped:
            summary = f"No column has more than {limit} missing values in the training rows."
        else:
            summary = (
                f"Dropped {plural(len(dropped), 'column')} with more than {limit} missing values "
                f"in the training rows: {quoted(dropped)}."
            )
        return StepResult(
            table.drop(dropped),
            summary,
            details=details,
            fitted={"columns": dropped, "missing": missing},
        )

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        columns = list((fitted or {}).get("columns", []))
        if not columns:
            return ["# No column was mostly empty in the training rows."]
        return [f"df = df.drop(columns={columns!r}, errors='ignore')"]


# --------------------------------------------------------------------------- #
# Impute
# --------------------------------------------------------------------------- #

STRATEGIES = (
    ("mean", "Mean"),
    ("median", "Median"),
    ("most_frequent", "Most frequent value"),
    ("constant", "A fixed value"),
    ("knn", "Nearest neighbours (KNN)"),
    ("forward_fill", "Previous value (forward fill)"),
    ("backward_fill", "Next value (backward fill)"),
    ("interpolate", "Linear interpolation"),
)
# Strategies and the column kinds they work on.
STRATEGY_KINDS = {
    "mean": ("numeric", "boolean", "datetime"),
    "median": ("numeric", "boolean", "datetime"),
    "most_frequent": ("numeric", "boolean", "datetime", "categorical", "text"),
    "constant": ("numeric", "boolean", "datetime", "categorical", "text"),
    "knn": ("numeric", "boolean"),
    "forward_fill": ("numeric", "boolean", "datetime", "categorical", "text"),
    "backward_fill": ("numeric", "boolean", "datetime", "categorical", "text"),
    "interpolate": ("numeric", "boolean"),
}
FILLED_WITH = {
    "mean": "with the mean",
    "median": "with the median",
    "most_frequent": "with the most frequent value",
    "forward_fill": "with the previous value",
    "backward_fill": "with the next value",
    "interpolate": "by linear interpolation",
}
# KNN: reference rows (a seeded sample of the training rows), fewer when very many values
# are missing (rows to fill × reference rows ≤ the budget, at least the minimum), and rows
# per distance block.
KNN_REFERENCE_ROWS = 5000
KNN_MIN_REFERENCE_ROWS = 1000
KNN_PAIR_BUDGET = 50_000_000
KNN_CHUNK_ROWS = 1024


def knn_fill(df, columns, features, mean, std, reference, k, booleans=(), chunk=1024):
    """Fill the missing values of ``columns`` with the mean of the ``k`` nearest reference
    rows that have a value. Distance: Euclidean on the standardised ``features`` (float32);
    missing coordinates are skipped and the sum is scaled up (like scikit-learn's
    nan_euclidean). Without any usable neighbour the training mean is used. ``booleans``
    are rounded to 0/1. Returns {column: filled float array}.
    """
    f32 = np.float32
    query = np.column_stack([(df[f].to_numpy(dtype=float) - mean[f]) / std[f] for f in features])
    ref = np.column_stack(
        [(np.asarray(reference[f], dtype=float) - mean[f]) / std[f] for f in features]
    )
    query, ref = query.astype(f32), ref.astype(f32)
    ref_zero = np.nan_to_num(ref, nan=0.0)
    ref_mask_t = np.ascontiguousarray((~np.isnan(ref)).astype(f32).T)
    ref_zero_t = np.ascontiguousarray(ref_zero.T)
    ref_square_t = np.ascontiguousarray((ref_zero**2).T)
    values = {c: df[c].to_numpy(dtype=float).copy() for c in columns}
    targets = {c: np.asarray(reference[c], dtype=float) for c in columns}
    rows = np.flatnonzero(np.any([np.isnan(values[c]) for c in columns], axis=0))
    for start in range(0, len(rows), chunk):
        block = rows[start : start + chunk]
        q_mask = (~np.isnan(query[block])).astype(f32)
        q_zero = np.nan_to_num(query[block], nan=0.0)
        # squared distance over the coordinates both rows have, scaled to all features
        distance = (q_zero**2) @ ref_mask_t
        distance -= 2 * (q_zero @ ref_zero_t)
        distance += q_mask @ ref_square_t
        common = q_mask @ ref_mask_t
        np.maximum(distance, 0, out=distance)
        distance *= len(features)
        np.divide(distance, common, out=distance, where=common > 0)
        distance[common == 0] = np.inf
        for c in columns:
            missing = np.isnan(values[c][block])
            if not missing.any():
                continue
            has_value = ~np.isnan(targets[c])
            n = min(int(k), int(has_value.sum()))
            fallback = mean[c] if c in mean else np.nanmean(targets[c])
            if n == 0:
                values[c][block[missing]] = fallback
                continue
            dist = distance[missing]
            if not has_value.all():
                dist[:, ~has_value] = np.inf
            if n <= 8:  # a few passes of argmin are faster than argpartition
                index = np.arange(len(dist))
                nearest = np.empty((len(dist), n), dtype=np.int64)
                near = np.empty((len(dist), n), dtype=f32)
                for i in range(n):
                    nearest[:, i] = dist.argmin(axis=1)
                    near[:, i] = dist[index, nearest[:, i]]
                    dist[index, nearest[:, i]] = np.inf
            else:
                nearest = np.argpartition(dist, n - 1, axis=1)[:, :n]
                near = np.take_along_axis(dist, nearest, axis=1)
            usable = np.isfinite(near)
            picked = np.where(usable, np.nan_to_num(targets[c][nearest]), 0.0)
            counts = usable.sum(axis=1)
            filled = np.where(counts > 0, picked.sum(axis=1) / np.maximum(counts, 1), fallback)
            if c in booleans:
                filled = np.where(filled >= 0.5, 1.0, 0.0)
            values[c][block[missing]] = filled
    return values


def _knn_source() -> str:
    return textwrap.dedent(inspect.getsource(knn_fill))


def _most_frequent(values: pd.Series) -> Any:
    counts = values.dropna().value_counts()
    if counts.empty:
        return None
    best = counts[counts == counts.max()].index
    return sorted(best)[0]


def _round_bool(value: float) -> float:
    return 1.0 if value >= 0.5 else 0.0


STRATEGY_NAMES = {
    "mean": "the mean",
    "median": "the median",
    "most_frequent": "the most frequent value",
    "constant": "a fixed value",
    "knn": "KNN",
    "forward_fill": "forward fill",
    "backward_fill": "backward fill",
    "interpolate": "interpolation",
}


def _strategy_check(column: str, kind: str, strategy: str) -> str | None:
    """Why ``strategy`` cannot fill ``column`` (None if it can)."""
    if kind in STRATEGY_KINDS[strategy]:
        return None
    label = STRATEGY_NAMES[strategy]
    if kind in STRING_KINDS:
        return (
            f"'{column}' holds {KIND_NAMES[kind]}; {label} only works for numbers. Use "
            "“Most frequent value” or “A fixed value” for categories."
        )
    return f"'{column}' holds {KIND_NAMES[kind]}; {label} only works for numbers."


@register
class Impute(Step):
    type = "impute"
    label = "Fill missing values"
    group = "missing"
    description = "Replace missing values, e.g. with the median of the training rows."
    fitted = True
    params = (
        Param(
            "columns",
            "columns",
            "Columns",
            default=[],
            optional=True,
            empty_means="All columns with missing values",
        ),
        Param(
            "strategy",
            "select",
            "Fill with",
            default="median",
            options=STRATEGIES,
            help="Mean, median, most frequent value and KNN are learned from the training rows. "
            "KNN uses the average of the k most similar training rows (standardised numeric "
            "columns). Previous / next value and interpolation follow the row order (sort "
            "the rows first) and fill training, validation and test rows separately, so no "
            "value crosses the split.",
        ),
        Param(
            "fill_value",
            "value",
            "Value",
            default=None,
            optional=True,
            visible_if={"param": "strategy", "values": ["constant"]},
        ),
        Param(
            "k",
            "int",
            "Neighbours",
            default=5,
            min=1,
            max=50,
            visible_if={"param": "strategy", "values": ["knn"]},
        ),
        Param(
            "add_indicator",
            "bool",
            "Add a column that marks filled values",
            default=False,
            help="Adds <column>_missing (1 where a value was missing), so the network can "
            "still see where values were filled in.",
        ),
    )

    def clean(self, table: Table, ctx: StepContext) -> dict[str, Any]:
        params = super().clean(table, ctx)
        strategy = params["strategy"]
        if strategy == "constant" and (params["fill_value"] is None or params["fill_value"] == ""):
            raise DataError("Value: enter the value to fill in.")
        for column in params["columns"]:
            kind = table.kinds[column]
            problem = _strategy_check(column, kind, strategy)
            if problem:
                raise DataError(f"Columns: {problem}")
            if strategy == "constant":
                cell_value(params["fill_value"], kind, column, "Value")
        return params

    # ---------------------------------------------------------------- run

    def _columns(
        self, table: Table, params: dict[str, Any], ctx: StepContext
    ) -> tuple[list[str], list[str]]:
        """The columns to fill and (empty selection only) the skipped unsuitable ones."""
        if params["columns"]:
            return list(params["columns"]), []
        with_missing = [c for c in chosen(table, [], ctx) if bool(table.df[c].isna().any())]
        strategy = params["strategy"]
        usable, skipped = [], []
        for column in with_missing:
            kind = table.kinds[column]
            ok = _strategy_check(column, kind, strategy) is None
            if ok and strategy == "constant":
                try:
                    cell_value(params["fill_value"], kind, column, "Value")
                except DataError:
                    ok = False
            (usable if ok else skipped).append(column)
        return usable, skipped

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        strategy = params["strategy"]
        columns, skipped = self._columns(table, params, ctx)
        warnings = []
        if skipped:
            kinds = sorted({KIND_NAMES[table.kinds[c]] for c in skipped})
            hint = (
                " Use “Most frequent value” or “A fixed value” for categories."
                if strategy not in ("most_frequent", "constant", "forward_fill", "backward_fill")
                else ""
            )
            warnings.append(
                f"Skipped {plural(len(skipped), 'column')} with {' and '.join(kinds)} "
                f"({quoted(skipped)}): {STRATEGY_NAMES[strategy]} does not work "
                f"for {'it' if len(skipped) == 1 else 'them'}.{hint}"
            )
        before_missing = {c: table.df[c].isna().to_numpy() for c in columns}
        fitted: dict[str, Any] = {"strategy": strategy, "kinds": {}, "values": {}}
        updates: dict[str, tuple[pd.Series, str]] = {}
        details = []
        if strategy == "knn":
            updates, knn_fitted, knn_details = self._knn(table, columns, params, ctx)
            fitted.update(knn_fitted)
            details += knn_details
        else:
            for column in columns:
                kind = table.kinds[column]
                filled, value, note = self._fill(table, column, kind, params)
                if filled is None:
                    warnings.append(note)
                    continue
                fitted["kinds"][column] = kind
                if value is not None:
                    fitted["values"][column] = to_fitted(value)
                updates[column] = (filled, kind)
                if note:
                    details.append({"column": column, "text": note})

        filled_total = 0
        unfilled = 0
        for column, (series, _) in updates.items():
            was = before_missing[column]
            now = series.isna().to_numpy()
            filled_total += int((was & ~now).sum())
            unfilled += int((was & now).sum())
        result = set_columns(table, updates) if updates else table

        indicators: dict[str, str] = {}
        if params["add_indicator"]:
            used = set(result.columns)
            for column in updates:
                if not before_missing[column].any():
                    continue
                name, n = f"{column}_missing", 2
                while name in used:
                    name, n = f"{column}_missing_{n}", n + 1
                used.add(name)
                indicators[column] = name
            for column, name in indicators.items():
                flag = pd.Series(before_missing[column].astype(float))
                result = set_columns(result, {name: (flag, "boolean")}, after={name: column})
        fitted["indicators"] = indicators

        if unfilled:
            same_part = " in the same part of the split" if table.split is not None else ""
            where = {
                "forward_fill": f"at the start (no earlier value{same_part})",
                "backward_fill": f"at the end (no later value{same_part})",
                "interpolate": "(no neighbouring value" + same_part + ")" if same_part else "",
            }.get(strategy, "")
            warnings.append(
                f"{plural(unfilled, 'value')} could not be filled{' ' + where if where else ''}."
            )
        if strategy == "constant" and updates:
            how = f"with {_shown_value(params['fill_value'])}"
        elif strategy == "knn":
            how = f"with the mean of the {plural(params['k'], 'nearest training row')} (KNN)"
        else:
            how = FILLED_WITH[strategy]
        if not columns:
            summary = "No column with missing values to fill."
        elif filled_total:
            summary = (
                f"Filled {plural(filled_total, 'missing value')} in "
                f"{plural(sum(1 for c in updates if before_missing[c].any()), 'column')} {how}."
            )
        else:
            summary = f"No missing values to fill; new data will be filled {how}."
        if indicators:
            summary = summary[:-1] + f", and added {plural(len(indicators), 'indicator column')}."
        return StepResult(result, summary, details=details, fitted=fitted, warnings=warnings)

    def _fill(
        self, table: Table, column: str, kind: str, params: dict[str, Any]
    ) -> tuple[pd.Series | None, Any, str]:
        """``(filled column, learned value, detail text)``; column None if not possible."""
        strategy = params["strategy"]
        series = table.df[column]
        if strategy in ("forward_fill", "backward_fill", "interpolate"):
            # Within each part of the split only: filling a training row from a neighbouring
            # validation or test row (or the other way round) would leak between them.
            source = (
                series.groupby(pd.Series(table.split, index=series.index), sort=False)
                if table.split is not None
                else None
            )
            if strategy == "forward_fill":
                filled = source.ffill() if source is not None else series.ffill()
            elif strategy == "backward_fill":
                filled = source.bfill() if source is not None else series.bfill()
            else:

                def interpolate(values: pd.Series) -> pd.Series:
                    return values.interpolate(method="linear", limit_direction="both")

                filled = (
                    source.transform(interpolate) if source is not None else interpolate(series)
                )
                if kind == "boolean":
                    filled = filled.round().clip(0, 1)
                return filled, None, ""
            return storage(filled, kind), None, ""
        if strategy == "constant":
            value = cell_value(params["fill_value"], kind, column, "Value")
        else:
            train = series[table.train_mask()] if table.split is not None else series
            value = _learn(train, kind, strategy)
            if value is None:
                return (
                    None,
                    None,
                    f"'{column}' has no values in the training rows, so it was not filled.",
                )
        filled = (
            series.where(series.notna(), value) if kind in STRING_KINDS else series.fillna(value)
        )
        label = {"mean": "mean", "median": "median", "most_frequent": "most frequent"}.get(
            strategy, "value"
        )
        return storage(filled, kind), value, f"{label} {_shown_value(value)}"

    def _knn(
        self, table: Table, columns: list[str], params: dict[str, Any], ctx: StepContext
    ) -> tuple[dict[str, tuple[pd.Series, str]], dict[str, Any], list[dict[str, Any]]]:
        if not columns:
            return {}, {}, []
        features = table.columns_of(NUMERIC_KINDS, exclude=[ctx.target] if ctx.target else [])
        for column in columns:
            if column not in features:
                features.append(column)
        train_rows = np.flatnonzero(table.train_mask())
        if len(train_rows) == 0:
            raise DataError("KNN needs training rows to compare with; there are none.")
        to_fill = int(table.df[columns].isna().any(axis=1).sum())
        limit = KNN_REFERENCE_ROWS
        if to_fill:
            limit = min(limit, max(KNN_MIN_REFERENCE_ROWS, KNN_PAIR_BUDGET // to_fill))
        if len(train_rows) > limit:
            rng = np.random.default_rng(ctx.seed)
            train_rows = np.sort(rng.choice(train_rows, limit, replace=False))
        train = table.df.iloc[np.flatnonzero(table.train_mask())]
        mean: dict[str, float] = {}
        std: dict[str, float] = {}
        for f in features:
            values = train[f].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            m = float(values.mean()) if len(values) else 0.0
            s = float(values.std()) if len(values) > 1 else 0.0
            mean[f], std[f] = m, (s if s > 1e-12 else 1.0)
        reference_frame = table.df.iloc[train_rows]
        reference = {
            f: [
                None if np.isnan(v) else v
                for v in reference_frame[f].to_numpy(dtype=float).tolist()
            ]
            for f in features
        }
        booleans = [c for c in columns if table.kinds[c] == "boolean"]
        filled = knn_fill(
            table.df, columns, features, mean, std, reference, params["k"], booleans, KNN_CHUNK_ROWS
        )
        updates = {c: (pd.Series(filled[c], name=c), table.kinds[c]) for c in columns}
        fitted = {
            "kinds": {c: table.kinds[c] for c in columns},
            "k": params["k"],
            "columns": columns,
            "features": features,
            "mean": mean,
            "std": std,
            "reference_rows": len(train_rows),
            "_reference": reference,
            "_booleans": booleans,
        }
        sampled = " sampled" if len(train_rows) < int(table.train_mask().sum()) else ""
        details = [
            {
                "text": f"compared on {plural(len(features), 'column')} with "
                f"{fmt_count(len(train_rows))}{sampled} training rows"
            }
        ]
        return updates, fitted, details

    # ---------------------------------------------------------------- code

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        fitted = fitted or {}
        strategy = fitted.get("strategy", params["strategy"])
        kinds = fitted.get("kinds", {})
        indicators = fitted.get("indicators", {})
        flags = {column: f"_was_missing_{i}" for i, column in enumerate(indicators, start=1)}
        lines = [f"{flags[c]} = df[{c!r}].isna().astype(float)" for c in indicators]
        if strategy == "knn" and fitted.get("columns"):
            lines += _knn_source().rstrip().splitlines()
            lines += [
                f"_knn_mean = {fitted['mean']!r}",
                f"_knn_std = {fitted['std']!r}",
                f"_knn_reference = {fitted['_reference']!r}",
                f"_filled = knn_fill(df, {fitted['columns']!r}, {fitted['features']!r}, "
                f"_knn_mean, _knn_std, _knn_reference, {fitted['k']!r}, "
                f"{fitted.get('_booleans', [])!r})",
            ]
            lines += [f"df[{c!r}] = _filled[{c!r}]" for c in fitted["columns"]]
        else:
            for column, kind in kinds.items():
                ref = f"df[{column!r}]"
                if strategy == "forward_fill":
                    lines.append(f"{ref} = {ref}.ffill()")
                elif strategy == "backward_fill":
                    lines.append(f"{ref} = {ref}.bfill()")
                elif strategy == "interpolate":
                    line = f"{ref} = {ref}.interpolate(method='linear', limit_direction='both')"
                    if kind == "boolean":
                        line += ".round().clip(0, 1)"
                    lines.append(line)
                elif column in fitted.get("values", {}):
                    value = code_value(from_fitted(fitted["values"][column], kind))
                    if kind in STRING_KINDS:
                        lines.append(f"{ref} = {ref}.where({ref}.notna(), {value})")
                    else:
                        lines.append(f"{ref} = {ref}.fillna({value})")
        for column, name in indicators.items():
            lines.append(
                f"df.insert(df.columns.get_loc({column!r}) + 1, {name!r}, {flags[column]})"
            )
        return lines or ["# Nothing to fill."]


def _learn(train: pd.Series, kind: str, strategy: str) -> Any:
    """The fill value learned from the training values (None without values)."""
    present = train.dropna()
    if present.empty:
        return None
    if strategy == "most_frequent":
        value = _most_frequent(present)
        return float(value) if kind in NUMERIC_KINDS else value
    if kind == "datetime":
        stamp = present.mean() if strategy == "mean" else present.median()
        return pd.Timestamp(stamp).floor("s") if not pd.isna(stamp) else None
    values = present.to_numpy(dtype=float)
    value = float(values.mean() if strategy == "mean" else np.median(values))
    return _round_bool(value) if kind == "boolean" else value


def _shown_value(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        text = value.strftime("%Y-%m-%d %H:%M:%S")
        return text[:-9] if text.endswith(" 00:00:00") else text
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return fmt_number(value)
    if value is None:
        return "–"
    return f"'{value}'"
