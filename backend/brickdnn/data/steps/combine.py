"""Preparation steps that combine imported tables (data integration): join the columns of
another table by key columns, or append its rows.

Combine steps run first, before the train/validation/test split is assigned. Their pandas
code (see :meth:`Step.to_pandas`) calls ``load_source(<id>)``, which the exported script
defines: it reads that data source like the import did.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..table import NUMERIC_KINDS, STRING_KINDS, DataError, Table
from .base import Param, Step, StepContext, StepResult, register
from .columns import (
    KIND_NAMES,
    comment,
    fmt_count,
    plural,
    quoted,
    storage,
    text_code,
    values_as_text,
)

# Refuse joins that would multiply the rows (repeated keys) beyond this factor …
MAX_JOIN_FACTOR = 5
# … or beyond this many rows (the default import limit of the server).
MAX_RESULT_ROWS = 2_000_000

JOIN_HOW = (
    ("left", "All rows of this table (left join)"),
    ("inner", "Only matching rows (inner join)"),
    ("outer", "All rows of both tables (outer join)"),
    ("right", "All rows of the other table (right join)"),
)


def _source_comment(ctx: StepContext, source_id: int) -> str:
    name = ctx.name_of(source_id)
    filename = None
    if ctx.source_info is not None:
        try:
            filename = (ctx.source_info(source_id) or {}).get("filename")
        except DataError:
            filename = None
    text = f"{name} ({filename})" if filename and filename != name else name
    return comment(text)


def _missing_series(index: pd.Index, kind: str) -> pd.Series:
    """An all-missing column in the storage format of ``kind``."""
    if kind == "datetime":
        return pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")
    if kind in STRING_KINDS:
        return pd.Series(None, index=index, dtype=object)
    return pd.Series(np.nan, index=index, dtype="float64")


def _missing_code(frame: str, kind: str) -> str:
    if kind == "datetime":
        return f"pd.Series(pd.NaT, index={frame}.index, dtype='datetime64[ns]')"
    if kind in STRING_KINDS:
        return f"pd.Series(None, index={frame}.index, dtype=object)"
    return f"pd.Series(np.nan, index={frame}.index, dtype='float64')"


def _finish(df: pd.DataFrame, kinds: dict[str, str]) -> Table:
    """A combined frame as a table: string columns get ``None`` for missing values again."""
    df = df.reset_index(drop=True)
    for name, kind in kinds.items():
        if kind in STRING_KINDS or df[name].dtype == object:
            df[name] = storage(df[name], kind)
    return Table(df, kinds)


# --------------------------------------------------------------------------- #
# Join
# --------------------------------------------------------------------------- #


def _compatible(left: str, right: str) -> bool:
    return (
        (left in NUMERIC_KINDS and right in NUMERIC_KINDS)
        or (left in STRING_KINDS and right in STRING_KINDS)
        or left == right == "datetime"
    )


@register
class Join(Step):
    type = "join"
    label = "Join another table"
    group = "combine"
    description = "Add the columns of another imported table by matching key columns."
    combine = True
    params = (
        Param("source", "source", "Table", default=None),
        Param(
            "how",
            "select",
            "Keep",
            default="left",
            options=JOIN_HOW,
            help="Which rows to keep when a key has no match.",
        ),
        Param("left_on", "columns", "Key columns here", default=[]),
        Param(
            "right_on",
            "source_columns",
            "Key columns there",
            default=[],
            optional=True,
            source_param="source",
            empty_means="The same names as here",
            help="Same number and order as the key columns here.",
        ),
        Param(
            "suffix",
            "string",
            "Suffix for duplicate names",
            default="_right",
            help="Added to columns of the other table whose names exist here already.",
        ),
    )

    def clean(self, table: Table, ctx: StepContext) -> dict[str, Any]:
        params = super().clean(table, ctx)
        if not params["left_on"]:
            raise DataError("Key columns here: choose at least one key column.")
        right_on = [str(c) for c in params["right_on"]] or list(params["left_on"])
        if len(right_on) != len(params["left_on"]):
            raise DataError(
                f"Key columns there: choose as many columns ({len(right_on)} chosen) as key "
                f"columns here ({len(params['left_on'])}), in the same order."
            )
        if len(set(right_on)) != len(right_on):
            raise DataError("Key columns there: a column is chosen twice.")
        params["right_on"] = right_on
        return params

    def _plan(self, table: Table, other: Table, params: dict[str, Any], name: str) -> dict:
        left_on, right_on = params["left_on"], params["right_on"]
        unknown = [c for c in right_on if c not in other.kinds]
        if unknown:
            raise DataError(f"Key columns there: '{name}' has no column {quoted(unknown)}.")
        for left, right in zip(left_on, right_on, strict=True):
            lk, rk = table.kinds[left], other.kinds[right]
            if not _compatible(lk, rk):
                raise DataError(
                    f"The key '{left}' holds {KIND_NAMES[lk]} here, but '{right}' holds "
                    f"{KIND_NAMES[rk]} in '{name}'. Change the type of one of them first."
                )
        used = set(table.columns)
        rename: dict[str, str] = {}
        for column in other.columns:
            if column in right_on:
                continue
            new, n = column, 2
            if new in used:
                new = f"{column}{params['suffix']}"
                while new in used:
                    new, n = f"{column}{params['suffix']}{n}", n + 1
            used.add(new)
            rename[column] = new
        temps = []
        for i in range(len(right_on)):
            temp = f"__key_{i}__"
            while temp in used:
                temp = "_" + temp
            used.add(temp)
            temps.append(temp)
        rename.update(dict(zip(right_on, temps, strict=True)))
        return {"rename": rename, "temps": temps}

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        if table.split is not None:
            raise DataError("Joins must come before all other steps.")
        source = params["source"]
        other = ctx.source(source)
        name = ctx.name_of(source)
        left_on, how = params["left_on"], params["how"]
        plan = self._plan(table, other, params, name)
        temps = plan["temps"]
        right = other.df.rename(columns=plan["rename"])
        complete = right[temps].notna().all(axis=1)

        # Count the result rows before joining: repeated keys can multiply them.
        counts_left = table.df[left_on].groupby(left_on, dropna=False, sort=False).size()
        counts_left = counts_left.rename("__left_n__").reset_index()
        keys_right = right.loc[complete, temps].set_axis(left_on, axis=1)
        counts_right = keys_right.groupby(left_on, sort=False).size()
        counts_right = counts_right.rename("__right_n__").reset_index()
        both = counts_left.merge(counts_right, on=left_on, how="outer")
        n_left = both["__left_n__"].fillna(0).to_numpy(dtype=float)
        n_right = both["__right_n__"].fillna(0).to_numpy(dtype=float)
        matched = float((n_left * n_right).sum())
        left_only = float(n_left[n_right == 0].sum())
        right_only = float(n_right[n_left == 0].sum()) + float((~complete).sum())
        rows = {
            "left": matched + left_only,
            "inner": matched,
            "right": matched + right_only,
            "outer": matched + left_only + right_only,
        }[how]
        limit = MAX_JOIN_FACTOR * max(table.n_rows, other.n_rows, 1)
        repeated = int((counts_right["__right_n__"] > 1).sum())
        if rows > min(limit, MAX_RESULT_ROWS):
            why = (
                f" because {plural(repeated, 'key')} appear more than once in '{name}'"
                if repeated
                else ""
            )
            raise DataError(
                f"The join would create {fmt_count(rows)} rows{why}. At most "
                f"{fmt_count(min(limit, MAX_RESULT_ROWS))} are allowed; check the key columns."
            )
        if rows == 0 and table.n_rows:
            raise DataError(
                f"No row of '{name}' matches, so the join would leave no rows. Check that the "
                "key columns hold the same values (spelling, numbers vs. text)."
            )

        merged = table.df.merge(
            right[complete], how=how, left_on=left_on, right_on=temps, sort=False
        )
        if how in ("right", "outer"):
            for left, temp in zip(left_on, temps, strict=True):
                merged[left] = merged[left].where(merged[left].notna(), merged[temp])
        merged = merged.drop(columns=temps)
        if how in ("right", "outer") and not complete.all():
            extra = right[~complete].rename(columns=dict(zip(temps, left_on, strict=True)))
            for column in merged.columns:
                if column not in extra.columns:
                    extra[column] = pd.Series(None, index=extra.index, dtype=merged[column].dtype)
            merged = pd.concat([merged, extra[list(merged.columns)]], ignore_index=True)

        kinds = dict(table.kinds)
        for left, right_key in zip(left_on, params["right_on"], strict=True):
            if kinds[left] != other.kinds[right_key] and kinds[left] in NUMERIC_KINDS:
                kinds[left] = "numeric"
        for column, new in plan["rename"].items():
            if new not in temps:
                kinds[new] = other.kinds[column]
        result = _finish(merged, {c: kinds[c] for c in merged.columns})

        keys = " + ".join(
            left if left == right_key else f"{left} = {right_key}"
            for left, right_key in zip(left_on, params["right_on"], strict=True)
        )
        matched_rows = int(n_left[(n_right > 0)].sum())
        added = [c for c in result.columns if c not in table.kinds]
        summary = (
            f"Joined '{name}' ({how} join on {keys}): {fmt_count(matched_rows)} of "
            f"{fmt_count(table.n_rows)} rows found a match"
        )
        if how in ("right", "outer") and right_only:
            summary += f", {plural(int(right_only), 'row')} only in '{name}' were added"
        summary += f"; {plural(len(added), 'column')} added."
        warnings = []
        if repeated:
            warnings.append(
                f"{plural(repeated, 'key')} appear more than once in '{name}', so matching rows "
                f"were repeated ({fmt_count(table.n_rows)} → {fmt_count(result.n_rows)} rows)."
            )
        if matched_rows == 0:
            warnings.append(
                "No row found a match. Check that the key columns hold the same values "
                "(spelling, numbers vs. text)."
            )
        details = [
            {
                "text": f"'{name}': {plural(other.n_rows, 'row')}, {plural(other.df.shape[1], 'column')}"
            }
        ]
        return StepResult(
            result,
            summary,
            details=details,
            warnings=warnings,
            fitted={"_plan": plan},
        )

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        plan = (fitted or {}).get("_plan")
        if not plan:
            return ["# Join: did not run."]
        temps, left_on, how = plan["temps"], list(params["left_on"]), params["how"]
        lines = [
            f"other = load_source({params['source']!r})  # {_source_comment(ctx, params['source'])}",
            f"other = other.rename(columns={plan['rename']!r})",
            f"complete = other[{temps!r}].notna().all(axis=1)",
            f"df = df.merge(other[complete], how={how!r}, left_on={left_on!r}, "
            f"right_on={temps!r}, sort=False)",
        ]
        if how in ("right", "outer"):
            for left, temp in zip(left_on, temps, strict=True):
                lines.append(
                    f"df[{left!r}] = df[{left!r}].where(df[{left!r}].notna(), df[{temp!r}])"
                )
        lines.append(f"df = df.drop(columns={temps!r})")
        if how in ("right", "outer"):
            renamed = dict(zip(temps, left_on, strict=True))
            lines += [
                "if not complete.all():  # rows of the other table without a key",
                f"    extra = other[~complete].rename(columns={renamed!r})",
                "    for column in df.columns:",
                "        if column not in extra.columns:",
                "            extra[column] = pd.Series(None, index=extra.index, dtype=df[column].dtype)",
                "    df = pd.concat([df, extra[list(df.columns)]], ignore_index=True)",
            ]
        return lines


# --------------------------------------------------------------------------- #
# Append
# --------------------------------------------------------------------------- #


@register
class Append(Step):
    type = "append"
    label = "Append rows"
    group = "combine"
    description = "Add the rows of another imported table with the same columns below."
    combine = True
    params = (
        Param("source", "source", "Table", default=None),
        Param(
            "add_source_column",
            "bool",
            "Add a column with the table name",
            default=False,
            help="Marks every row with the name of the table it comes from.",
        ),
        Param(
            "source_column",
            "string",
            "Column name",
            default="source",
            visible_if={"param": "add_source_column", "values": [True]},
        ),
    )

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        if table.split is not None:
            raise DataError("Appending rows must come before all other steps.")
        source = params["source"]
        other = ctx.source(source)
        name = ctx.name_of(source)
        if table.n_rows + other.n_rows > MAX_RESULT_ROWS:
            raise DataError(
                f"Together the tables have {fmt_count(table.n_rows + other.n_rows)} rows; at "
                f"most {fmt_count(MAX_RESULT_ROWS)} are supported."
            )
        column = params["source_column"] if params["add_source_column"] else None
        if column and column in other.kinds:
            raise DataError(
                f"Column name: '{name}' has a column '{column}' already; choose another name."
            )
        if column and column in table.kinds and table.kinds[column] not in STRING_KINDS:
            raise DataError(
                f"Column name: there is a column '{column}' already; choose another name."
            )

        here, there = table.kinds, other.kinds
        final: dict[str, str] = {}
        conflicts = []
        for c in [*table.columns, *(c for c in other.columns if c not in here)]:
            a, b = here.get(c), there.get(c)
            if a and b and a != b:
                if a in NUMERIC_KINDS and b in NUMERIC_KINDS:
                    final[c] = "numeric"
                elif a in STRING_KINDS and b in STRING_KINDS:
                    final[c] = "text"
                else:
                    final[c] = "categorical"
                    conflicts.append(c)
            else:
                final[c] = a or b  # type: ignore[assignment]
        plan = {
            "text_here": {c: here[c] for c in conflicts},
            "text_there": {c: there[c] for c in conflicts},
            "add_here": {c: final[c] for c in other.columns if c not in here},
            "add_there": {c: final[c] for c in table.columns if c not in there},
            "column": column,
            "label_here": (
                ctx.name_of(ctx.source_id)
                if getattr(ctx, "source_id", None) is not None
                else "original"
            ),
            "label_there": name,
        }
        df, add = table.df.copy(deep=False), other.df.copy(deep=False)
        for c, kind in plan["text_here"].items():
            df[c] = values_as_text(df[c], kind)
        for c, kind in plan["text_there"].items():
            add[c] = values_as_text(add[c], kind)
        for c, kind in plan["add_here"].items():
            df[c] = _missing_series(df.index, kind)
        for c, kind in plan["add_there"].items():
            add[c] = _missing_series(add.index, kind)
        if column:
            if column not in df.columns:
                df[column] = plan["label_here"]
            add[column] = name
            final[column] = "categorical"
        combined = pd.concat([df, add[list(df.columns)]], ignore_index=True)
        result = _finish(combined, {c: final[c] for c in combined.columns})

        warnings = []
        for c in conflicts:
            warnings.append(
                f"'{c}' holds {KIND_NAMES[here[c]]} here but {KIND_NAMES[there[c]]} in "
                f"'{name}'; it is a category column now."
            )
        if plan["add_here"]:
            warnings.append(
                f"{plural(len(plan['add_here']), 'column')} only exist in '{name}' "
                f"({quoted(list(plan['add_here']))}); they are missing in the other rows."
            )
        if plan["add_there"]:
            warnings.append(
                f"{plural(len(plan['add_there']), 'column')} do not exist in '{name}' "
                f"({quoted(list(plan['add_there']))}); they are missing in its rows."
            )
        summary = (
            f"Appended {plural(other.n_rows, 'row')} of '{name}'; the data has "
            f"{plural(result.n_rows, 'row')} now."
        )
        return StepResult(result, summary, warnings=warnings, fitted={"_plan": plan})

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        plan = (fitted or {}).get("_plan")
        if not plan:
            return ["# Append: did not run."]
        lines = [
            f"other = load_source({params['source']!r})  # {_source_comment(ctx, params['source'])}"
        ]
        for c, kind in plan["text_here"].items():
            lines.append(f"df[{c!r}] = {text_code(f'df[{c!r}]', kind)}")
        for c, kind in plan["text_there"].items():
            lines.append(f"other[{c!r}] = {text_code(f'other[{c!r}]', kind)}")
        for c, kind in plan["add_here"].items():
            lines.append(f"df[{c!r}] = {_missing_code('df', kind)}")
        for c, kind in plan["add_there"].items():
            lines.append(f"other[{c!r}] = {_missing_code('other', kind)}")
        column = plan.get("column")
        if column:
            lines += [
                f"if {column!r} not in df.columns:",
                f"    df[{column!r}] = {plan['label_here']!r}",
                f"other[{column!r}] = {plan['label_there']!r}",
            ]
        lines.append("df = pd.concat([df, other[list(df.columns)]], ignore_index=True)")
        return lines
