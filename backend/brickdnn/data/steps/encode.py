"""Preparation steps: turning categories into numbers and grouping rare categories.

Categories are always learned from the training rows. Boolean columns can be encoded too;
their categories are "0" and "1".
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..table import MAX_NEW_COLUMNS, DataError, Table
from .base import Param, Step, StepContext, StepResult, register
from .outliers import (
    check_new_names,
    chosen_columns,
    code_insert_after,
    fmt_count,
    fmt_number,
    fmt_share,
    insert_after,
    no_columns_result,
    plain,
    plural,
)

CATEGORY_KINDS = ("categorical", "boolean", "text")
BOOLEAN_KEYS = {0: "0", 1: "1"}


def category_keys(series: pd.Series, boolean: bool) -> pd.Series:
    """The values of a column as category strings (missing stays missing)."""
    return series.map(BOOLEAN_KEYS) if boolean else series


def _code_keys(column: str, boolean: bool) -> str:
    ref = f"df[{column!r}]"
    return f"{ref}.map({BOOLEAN_KEYS!r})" if boolean else ref


def class_labels(values: pd.Series) -> list[Any]:
    """The distinct non-missing values of a target, sorted like the export orders classes
    (numbers numerically, everything else alphabetically as text)."""
    present = values.dropna().unique().tolist()
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present):
        return sorted(present)
    return sorted(present, key=str)


def class_name(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _unseen_note(keys: pd.Series, known: set[str], train: np.ndarray) -> str:
    unseen = keys.notna() & ~keys.isin(known) & ~train
    count = int(unseen.sum())
    if not count:
        return ""
    return f" · {plural(count, 'validation/test value')} with unseen categories"


@register
class EncodeStep(Step):
    type = "encode"
    label = "Encode categories"
    group = "encode"
    description = "Turn categories into numbers, e.g. one column per category (one-hot)."
    fitted = True
    params = (
        Param(
            "columns",
            "columns",
            "Columns",
            default=[],
            optional=True,
            column_kinds=CATEGORY_KINDS,
            empty_means="All categorical columns",
        ),
        Param(
            "method",
            "select",
            "Method",
            default="onehot",
            options=(
                ("onehot", "One-hot"),
                ("ordinal", "Ordinal (category index)"),
                ("frequency", "Frequency"),
                ("target", "Target mean"),
            ),
            help=(
                "One-hot: a 0/1 column '<column>=<category>' per category, the usual choice. "
                "Ordinal: the category's number in alphabetical order (unseen: -1). "
                "Frequency: the category's share of the training rows (unseen: 0). Target "
                "mean: the smoothed mean of the target per category (unseen: the overall "
                "mean). Categories come from the training rows; missing values stay missing "
                "(one-hot: all 0)."
            ),
        ),
        Param(
            "drop_first",
            "bool",
            "Drop the first category",
            default=False,
            visible_if={"param": "method", "values": ["onehot"]},
            help=(
                "Leaves out the column of the first category (alphabetically), which the "
                "others imply. Only needed for linear models without regularisation."
            ),
        ),
        Param(
            "max_categories",
            "int",
            "Max. categories per column",
            default=30,
            min=1,
            max=1000,
            visible_if={"param": "method", "values": ["onehot"]},
            help=(
                "Only the most frequent categories get their own column; all others "
                "(and unseen ones) are marked in '<column>_other'."
            ),
        ),
        Param(
            "smoothing",
            "number",
            "Smoothing",
            default=10,
            min=0,
            max=10000,
            visible_if={"param": "method", "values": ["target"]},
            help=(
                "(sum of the target in the category + smoothing · overall mean) / (rows of "
                "the category + smoothing). Pulls rare categories towards the overall mean, "
                "so the model does not learn the noise of a few rows. Training rows get "
                "out-of-fold values (learned from the other training rows, 5 folds), so a "
                "row never sees its own target."
            ),
        ),
    )

    def clean(self, table: Table, ctx: StepContext) -> dict[str, Any]:
        params = super().clean(table, ctx)
        if params["method"] == "target":
            if not ctx.target:
                raise DataError(
                    "Target mean encoding needs a target. Choose the target column under "
                    "'Target & split' first."
                )
            if ctx.target not in table.kinds:
                raise DataError(f"The target column '{ctx.target}' does not exist (any more).")
            if ctx.target in params["columns"]:
                raise DataError(f"The target '{ctx.target}' cannot be encoded with itself.")
        return params

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        columns = chosen_columns(table, params["columns"], ctx, ("categorical",))
        if not columns:
            return no_columns_result(table, "categorical columns")
        method = params["method"]
        train = table.train_mask()
        learned: dict[str, dict[str, Any]] = {}
        details: list[dict[str, Any]] = []
        warnings: list[str] = []
        result = table
        target_info = _target_info(table, ctx, train) if method == "target" else None
        added = 0
        for column in columns:
            boolean = table.kinds[column] == "boolean"
            keys = category_keys(table.df[column], boolean)
            counts = keys[train].value_counts()
            if counts.empty:
                warnings.append(f"'{column}' has no values in the training rows; left unchanged.")
                continue
            info: dict[str, Any] = {"boolean": boolean}
            categories = sorted(str(c) for c in counts.index)
            note = _unseen_note(keys, set(categories), train)
            if method == "onehot":
                planned = min(len(counts), params["max_categories"]) + (
                    1 if len(counts) > params["max_categories"] else 0
                )
                if added + planned > MAX_NEW_COLUMNS:
                    raise DataError(
                        f"One-hot encoding '{column}' would add more than "
                        f"{MAX_NEW_COLUMNS:,} columns in this step. Lower “Most categories”, "
                        "group rare categories first or use another method."
                    )
                info, new = _onehot(column, keys, counts, params, boolean)
                check_new_names(table, [name for name, _, _ in new], replaced=[column])
                result = insert_after(result, column, new, drop_after=True)
                added += len(new)
                text = f"{plural(len(info['categories']), 'category', 'categories')}"
                if info["other"]:
                    text += f" of {fmt_count(len(counts))}, the rest in '{column}_other'"
                if info["dropped"] is not None:
                    text += f", '{info['dropped']}' dropped"
            elif method in ("ordinal", "frequency"):
                if method == "ordinal":
                    mapping = {c: i for i, c in enumerate(categories)}
                    unseen = -1
                else:
                    total = int(counts.sum())
                    mapping = {str(c): plain(n / total) for c, n in counts.items()}
                    mapping = {c: mapping[c] for c in categories}
                    unseen = 0
                info.update(mapping=mapping, unseen=unseen)
                values = _map_values(keys, mapping, unseen)
                result = result.with_column(column, values, "numeric")
                text = f"{plural(len(categories), 'category', 'categories')}"
            else:
                info.update(_target_encoding(keys, target_info, params["smoothing"]))
                oof_rows, oof = _out_of_fold(keys, target_info, params["smoothing"], ctx.seed)
                if "classes" in info:
                    new = [
                        (
                            f"{column}_te_{name}",
                            _with_out_of_fold(
                                _map_values(keys, info["mapping"][name], info["prior"][name]),
                                oof_rows,
                                oof[name],
                            ),
                            "numeric",
                        )
                        for name in info["classes"]
                    ]
                    check_new_names(table, [n for n, _, _ in new], replaced=[column])
                    result = insert_after(result, column, new, drop_after=True)
                    added += len(new)
                else:
                    (name,) = target_info["y"]
                    values = _with_out_of_fold(
                        _map_values(keys, info["mapping"], info["prior"]), oof_rows, oof[name]
                    )
                    result = result.with_column(column, values, "numeric")
                text = f"{plural(len(categories), 'category', 'categories')}"
            learned[column] = info
            details.append({"column": column, "text": text + note})
        n_cols = plural(len(learned), "column")
        if method == "onehot":
            summary = f"One-hot encoded {n_cols} into {plural(added, 'new column')}."
        elif method == "ordinal":
            summary = f"Replaced the categories of {n_cols} by their index (alphabetical order)."
        elif method == "frequency":
            summary = f"Replaced the categories of {n_cols} by their share of the training rows."
        else:
            summary = _target_summary(n_cols, target_info, added)
            summary = (
                summary[:-1] + f"; training rows use out-of-fold values ({TARGET_FOLDS} folds)."
            )
        return StepResult(result, summary, details, {"columns": learned}, warnings)

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        params = {**self.defaults(), **params}
        lines: list[str] = []
        for column, info in (fitted or {}).get("columns", {}).items():
            keys = _code_keys(column, info["boolean"])
            ref = f"df[{column!r}]"
            if params["method"] == "onehot":
                new = [(f"{column}={c}", f"({keys} == {c!r}).astype(float)") for c in info["kept"]]
                if info["other"]:
                    known = info["categories"]
                    expr = f"({ref}.notna() & ~{keys}.isin({known!r})).astype(float)"
                    new.append((f"{column}_other", expr))
                lines += code_insert_after(column, new)
                lines.append(f"df = df.drop(columns=[{column!r}])")
            elif "classes" in info:
                new = [
                    (
                        f"{column}_te_{name}",
                        _code_map(keys, ref, info["mapping"][name], info["prior"][name]),
                    )
                    for name in info["classes"]
                ]
                lines += code_insert_after(column, new)
                lines.append(f"df = df.drop(columns=[{column!r}])")
            else:
                unseen = info["prior"] if "prior" in info else info["unseen"]
                lines.append(f"{ref} = {_code_map(keys, ref, info['mapping'], unseen)}")
        return lines or ["# Nothing to encode."]


def _map_values(keys: pd.Series, mapping: dict[str, float], unseen: float) -> pd.Series:
    """Category -> number; categories not in ``mapping`` -> ``unseen``; missing stays."""
    return keys.map(mapping).astype(float).fillna(unseen).where(keys.notna())


def _code_map(keys: str, ref: str, mapping: dict[str, float], unseen: float) -> str:
    return f"{keys}.map({mapping!r}).astype(float).fillna({unseen!r}).where({ref}.notna())"


def _onehot(
    column: str, keys: pd.Series, counts: pd.Series, params: dict[str, Any], boolean: bool
) -> tuple[dict[str, Any], list[tuple[str, pd.Series, str]]]:
    # Most frequent first, ties alphabetically; the kept ones in alphabetical order.
    ranked = sorted(((-int(n), str(c)) for c, n in counts.items()))
    limit = params["max_categories"]
    categories = sorted(c for _, c in ranked[:limit])
    other = len(ranked) > limit
    kept = categories[1:] if params["drop_first"] else categories
    # Category codes (-1: missing or not a kept category) are much faster than comparing
    # strings once per category; the generated code compares the strings.
    codes = pd.Categorical(keys, categories=categories).codes
    new = [(f"{column}={c}", (codes == categories.index(c)).astype(float), "boolean") for c in kept]
    if other:
        is_other = ((codes == -1) & keys.notna().to_numpy()).astype(float)
        new.append((f"{column}_other", is_other, "boolean"))
    info = {
        "boolean": boolean,
        "categories": categories,
        "kept": kept,
        "dropped": categories[0] if params["drop_first"] else None,
        "other": other,
    }
    return info, new


def _target_info(table: Table, ctx: StepContext, train: np.ndarray) -> dict[str, Any]:
    target = ctx.target
    y = table.df[target]
    kind = table.kinds[target]
    task = ctx.task or ("regression" if kind == "numeric" else "classification")
    rows = train & y.notna().to_numpy()
    if not rows.any():
        raise DataError(f"The target '{target}' has no values in the training rows.")
    if task == "regression":
        if kind not in ("numeric", "boolean"):
            raise DataError(f"The regression target '{target}' must be numeric.")
        return {"target": target, "task": task, "y": {None: y.astype(float)}, "rows": rows}
    classes = class_labels(y[rows])
    if len(classes) < 2:
        raise DataError(
            f"Target mean encoding needs at least two classes of '{target}' in the training rows."
        )
    names = [class_name(c) for c in classes]
    if len(classes) == 2:
        indicators = {names[1]: (y == classes[1]).astype(float)}
    else:
        indicators = {name: (y == c).astype(float) for name, c in zip(names, classes, strict=True)}
    return {"target": target, "task": task, "y": indicators, "rows": rows, "classes": names}


def _target_encoding(keys: pd.Series, info: dict[str, Any], smoothing: float) -> dict[str, Any]:
    rows = info["rows"] & keys.notna().to_numpy()
    mappings: dict[Any, dict[str, float]] = {}
    priors: dict[Any, float] = {}
    for name, y in info["y"].items():
        prior = float(y[info["rows"]].mean())
        stats = y[rows].groupby(keys[rows]).agg(["sum", "count"])
        encoded = (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)
        mappings[name] = {str(c): plain(v) for c, v in sorted(encoded.items())}
        priors[name] = plain(prior)
    if len(info["y"]) == 1:
        (name,) = info["y"]
        return {"mapping": mappings[name], "prior": priors[name]}
    return {"classes": list(info["y"]), "mapping": mappings, "prior": priors}


# Training rows get target encodings learned from the other folds (like scikit-learn's
# TargetEncoder): with in-sample means, rare categories would hand each row its own label.
TARGET_FOLDS = 5


def _out_of_fold(
    keys: pd.Series, info: dict[str, Any], smoothing: float, seed: int
) -> tuple[np.ndarray, dict[Any, np.ndarray]]:
    """Positions of the training rows that taught the encoding, and for every encoded value
    (class or target) their encodings learned from the other folds."""
    rows = np.flatnonzero(info["rows"] & keys.notna().to_numpy())
    folds = min(TARGET_FOLDS, len(rows))
    if folds < 2:
        return rows[:0], {name: np.empty(0) for name in info["y"]}
    fold = np.random.default_rng(seed).permutation(len(rows)) % folds
    row_keys = keys.to_numpy(dtype=object)[rows]
    out: dict[Any, np.ndarray] = {}
    for name, y in info["y"].items():
        target = y.to_numpy(dtype=float)[rows]
        encoded = np.empty(len(rows))
        for f in range(folds):
            fit, apply = fold != f, fold == f
            prior = float(target[fit].mean())
            stats = pd.Series(target[fit]).groupby(row_keys[fit]).agg(["sum", "count"])
            means = (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)
            mapped = pd.Series(row_keys[apply]).map(means).astype(float).fillna(prior)
            encoded[apply] = mapped.to_numpy()
        out[name] = encoded
    return rows, out


def _with_out_of_fold(values: pd.Series, rows: np.ndarray, encoded: np.ndarray) -> pd.Series:
    result = values.to_numpy(dtype=float, copy=True)
    result[rows] = encoded
    return pd.Series(result, index=values.index)


def _target_summary(n_cols: str, info: dict[str, Any], added: int) -> str:
    target = info["target"]
    if info["task"] == "regression":
        return f"Replaced the categories of {n_cols} by the smoothed mean of '{target}'."
    if len(info["y"]) == 1:
        (positive,) = info["y"]
        return (
            f"Replaced the categories of {n_cols} by the smoothed share of '{target}' = "
            f"'{positive}'."
        )
    return (
        f"Encoded {n_cols} into {plural(added, 'column')}, one per class of '{target}' "
        "(smoothed class shares)."
    )


# --------------------------------------------------------------------------- #
# Rare categories
# --------------------------------------------------------------------------- #


@register
class RareCategoriesStep(Step):
    type = "rare_categories"
    label = "Group rare categories"
    group = "encode"
    description = (
        "Merge categories that are rare in the training rows into one label such as 'Other'."
    )
    fitted = True
    params = (
        Param(
            "columns",
            "columns",
            "Columns",
            default=[],
            optional=True,
            column_kinds=("categorical", "text"),
            empty_means="All categorical columns",
        ),
        Param(
            "min_share",
            "number",
            "Minimum share",
            default=0.01,
            min=0,
            max=0.5,
            step=0.005,
            help=(
                "Categories with a smaller share of the training rows (0.01 = 1 %) are "
                "merged, as are categories that only appear in validation or test rows."
            ),
        ),
        Param("label", "string", "New label", default="Other"),
    )

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        columns = chosen_columns(table, params["columns"], ctx, ("categorical",))
        if not columns:
            return no_columns_result(table, "categorical columns")
        label = params["label"]
        train = table.train_mask()
        keep: dict[str, list[str]] = {}
        details: list[dict[str, Any]] = []
        warnings: list[str] = []
        result = table
        merged_categories = merged_values = cells = 0
        for column in columns:
            series = table.df[column]
            counts = series[train].value_counts()
            if counts.empty:
                warnings.append(f"'{column}' has no values in the training rows; left unchanged.")
                continue
            shares = counts / counts.sum()
            kept = sorted(str(c) for c, s in shares.items() if s >= params["min_share"])
            rare = series.notna() & ~series.isin(kept)
            merged = sorted(set(series[rare].tolist()) - {label})
            keep[column] = kept
            if not kept:
                warnings.append(
                    f"Every category of '{column}' is rarer than {fmt_share(params['min_share'], 1)}"
                    f", so all became '{label}'. Lower the minimum share."
                )
            merged_categories += len(merged)
            merged_values += int((rare & (series != label)).sum())
            cells += int(series.notna().sum())
            result = result.with_column(column, series.where(~rare, label), table.kinds[column])
            details.append(
                {
                    "column": column,
                    "text": f"kept {fmt_count(len(kept))} of "
                    f"{plural(len(counts), 'category', 'categories')}",
                }
            )
        n_cols = plural(len(keep), "column")
        if merged_categories:
            summary = (
                f"Grouped {plural(merged_categories, 'rare category', 'rare categories')} "
                f"({plural(merged_values, 'value')}, {fmt_share(merged_values, cells)}) into "
                f"'{label}' in {n_cols}."
            )
        else:
            summary = (
                f"No category below {fmt_number(params['min_share'] * 100)} % in {n_cols}; "
                "nothing grouped."
            )
        return StepResult(result, summary, details, {"keep": keep}, warnings)

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        params = {**self.defaults(), **params}
        lines = []
        for column, kept in (fitted or {}).get("keep", {}).items():
            ref = f"df[{column!r}]"
            lines.append(
                f"{ref} = {ref}.where({ref}.isin({kept!r}) | {ref}.isna(), {params['label']!r})"
            )
        return lines or ["# Nothing to group."]
