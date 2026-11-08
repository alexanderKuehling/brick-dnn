"""Helpers for the data preparation tests: small tables, pipelines, and running the pandas
code of steps (``Step.to_pandas``) to compare it with ``Step.run``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from brickdnn.data.pipeline import clean_steps, json_safe, run_pipeline
from brickdnn.data.steps import StepContext, get_step
from brickdnn.data.table import STRING_KINDS, Table


def table(data: dict[str, Any], kinds: dict[str, str] | None = None, split=None) -> Table:
    """A table from column lists; kinds are inferred unless given."""
    return Table.from_frame(pd.DataFrame(data), kinds=kinds, split=split)


def steps(*items: tuple[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """``steps(("impute", {...}), ...)`` -> pipeline steps with ids s1, s2, ..."""
    return [
        {"id": f"s{i}", "type": step_type, "enabled": True, "params": params}
        for i, (step_type, params) in enumerate(items, start=1)
    ]


def run(source: Table, *items: tuple[str, dict[str, Any]], settings=None, **kwargs):
    return run_pipeline(source, clean_steps(steps(*items)), settings or {}, **kwargs)


def apply(source: Table, step_type: str, params: dict[str, Any], ctx: StepContext | None = None):
    """Clean and run one step: ``(StepResult, cleaned params, ctx)``."""
    ctx = ctx or StepContext()
    step = get_step(step_type)(params)
    cleaned = step.clean(source, ctx)
    return step.run(source, cleaned, ctx), cleaned, ctx


def script_frame(source: Table) -> pd.DataFrame:
    """The DataFrame the exported script works on: like the table, but missing strings
    are NaN (as pandas reads them)."""
    df = source.df.copy()
    for name, kind in source.kinds.items():
        if kind in STRING_KINDS:
            df[name] = df[name].where(df[name].notna(), np.nan)
    return df


def run_code(lines: list[str], df: pd.DataFrame, **names: Any) -> pd.DataFrame:
    """Execute generated step code on a copy of ``df`` and return the resulting ``df``."""
    namespace = {"np": np, "pd": pd, "df": df.copy(), **names}
    exec(compile("\n".join(lines), "<step code>", "exec"), namespace)
    return namespace["df"]


def assert_frame_matches(df: pd.DataFrame, expected: Table) -> None:
    """``df`` (from generated code) holds the same columns and values as ``expected``;
    missing values may be None or NaN."""
    assert list(df.columns) == expected.columns
    assert len(df) == expected.n_rows
    for name, kind in expected.kinds.items():
        got, want = df[name], expected.df[name]
        if kind in ("numeric", "boolean"):
            np.testing.assert_array_equal(
                got.to_numpy(dtype=float, na_value=np.nan), want.to_numpy(dtype=float), err_msg=name
            )
        elif kind == "datetime":
            assert pd.api.types.is_datetime64_any_dtype(got), name
            np.testing.assert_array_equal(
                got.to_numpy(dtype="datetime64[ns]"), want.to_numpy(dtype="datetime64[ns]"), name
            )
        else:
            as_list = [None if pd.isna(v) else v for v in got.tolist()]
            assert as_list == want.tolist(), name


def check_code(
    source: Table,
    step_type: str,
    params: dict[str, Any],
    ctx: StepContext | None = None,
    **names: Any,
):
    """Run a step and its pandas code on ``source`` and assert both give the same table.
    Returns the step result."""
    result, cleaned, ctx = apply(source, step_type, params, ctx)
    fitted = json_safe(result.fitted) if result.fitted is not None else None
    lines = get_step(step_type)(params).to_pandas(json_safe(cleaned), fitted, ctx)
    assert_frame_matches(run_code(lines, script_frame(source), **names), result.table)
    return result


def values(t: Table, column: str) -> list[Any]:
    """Column values with None for missing (numbers as floats)."""
    return [None if pd.isna(v) else v for v in t.df[column].tolist()]
