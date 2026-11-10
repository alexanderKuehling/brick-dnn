"""Sanity checks of prepared data for the Readiness view (see
:func:`brickdnn.data.pipeline.readiness`).

They find what makes a dataset technically usable but a network trained on it poor or its
scores misleading: target leakage, rows shared by the training and the validation/test part,
ID columns, near-constant and redundant features, numbers and dates stored as text,
spellings of one category that differ only in case, too many categories for one-hot encoding,
rare classes, skewed values, very different scales and too few rows.

Every issue is ``{"code", "severity", "column", "columns", "message", "fix"}`` (``Readiness``
in ``frontend/src/api/types.ts``); the fix, if any, is a step that solves it.

Like the steps, the checks learn from the **training rows** only (statistics, correlations,
leakage); a seeded random sample of them on large tables. Row duplicates compare all rows
(at most ``DUPLICATE_ROWS``). Thresholds are set so that ordinary data stays quiet: a check
needs a minimum number of rows before it says anything.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import rows as rowstats
from .io import _LEADING_ZERO, LEADING_ZERO_SHARE, parse_numbers
from .profile import _CAMEL_ID, _ID_NAME, HIGH_CARDINALITY, SKEWNESS, _eta
from .steps.columns import (
    NUMBER_PATTERN,
    UNIT_NUMBER,
    exact_numbers,
    number_text,
    parse_dates,
)
from .steps.columns import _date_format as date_format
from .steps.text import fmt_number, fmt_share, plural
from .table import NUMERIC_KINDS, TEST, TRAIN, VAL, DataError, Table

# Training rows the value checks look at at most (a seeded random sample)...
CHECK_ROWS = 100_000
CHECK_SEED = 0
# ... and cells (rows × numeric features) at most: wide tables use fewer rows.
CHECK_CELLS = 5_000_000
MIN_CHECK_ROWS = 2_000
# Rows the duplicate checks compare at most.
DUPLICATE_ROWS = 1_000_000
# Values of a string column the number and date detection looks at.
TEXT_SAMPLE = 5_000

# Target leakage: a feature (almost) determines the target in the training rows.
LEAK_MIN_ROWS = 30
LEAK_CORRELATION = 0.98  # |r| with a numeric target
LEAK_ETA = 0.99  # correlation ratio: the feature is almost constant per class (or vice versa)
LEAK_MAX_ERROR = 0.01  # a feature value / threshold predicts the class with ≤ 1 % errors …
LEAK_ERROR_SHARE = 0.1  # … and ≤ 10 % of the errors of always guessing the largest class
LEAK_VALUE_ROWS = 5  # feature values with fewer training rows prove nothing …
LEAK_COVERAGE = 0.8  # … so values with enough rows must cover this share of the rows
LEAK_MAX_VALUES = 50  # numbers with at most this many values are treated as categories
LEAK_SUFFIXES = (
    "id|label|labels|code|codes|encoded|enc|name|class|cls|category|cat|idx|index|num|number"
    "|level|group|bin|binned|flag|str|text|value|log|norm|scaled|pct|percent|true|actual"
)
# Near-constant: one value in at least this share of the training rows.
NEAR_CONSTANT = 0.99
NEAR_CONSTANT_MIN_ROWS = 100
# Redundant features: |r| at least this.
REDUNDANT = 0.98
MAX_PAIR_COLUMNS = 1_000
# Skewed numbers (|skewness| > SKEWNESS) with at least this many different values.
SKEW_MIN_VALUES = 10
# Scales: the largest standard deviation is at least this many times the smallest.
SCALE_RATIO = 1_000
# String columns: number / date detection, ID-like values.
NUMBER_SHARE = 0.9
UNIT_SHARE = 0.8  # numbers with a unit: the same unit in this share of the values
DATE_SHARE = 0.9
MIN_TEXT_VALUES = 5
ID_MIN_ROWS = 20
ID_UNIQUE_SHARE = 0.95
MAX_REPLACEMENTS = 500
# One-hot encoding keeps this many categories by default (the encode step's max_categories).
ONEHOT_CATEGORIES = 30
# Classes.
MIN_CLASS_ROWS = 5
# Rows.
MIN_TRAIN_ROWS = 50
ROWS_PER_FEATURE = 5
DUPLICATE_WARNING = 0.05  # share of duplicate rows that is more than a note
OVERLAP_WARNING = 0.01  # share of validation/test rows that are copies of training rows
CONFLICT_SHARE = 0.01
# Row structure (atypical rows, near twins): rows checked at most, rows needed, and the share
# of rows with a twin / of validation and test rows with a twin in training that is worth a
# warning.
STRUCTURE_ROWS = 4_000
STRUCTURE_MIN_ROWS = 50
STRUCTURE_TWIN_SHARE = 0.05
STRUCTURE_HELD_SHARE = 0.05

# Order of the warnings and notes in the list (errors keep the order they are found in).
PRIORITY = (
    "leakage",
    "split_overlap",
    "near_twins",
    "atypical_rows",
    "group_column_feature",
    "rare_classes",
    "classes_not_in_split",
    "inconsistent_categories",
    "id_like",
    "few_rows",
    "imbalanced",
    "duplicate_rows",
    "conflicting_labels",
    "constant",
    "near_constant",
    "redundant",
    "skewed_target",
    "skewed",
    "scales",
    "many_features",
)


# --------------------------------------------------------------------------- #
# Issues and text
# --------------------------------------------------------------------------- #


def issue(
    code: str,
    severity: str,
    message: str,
    columns: Sequence[str] = (),
    fix: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A readiness issue; ``column`` is the first of ``columns`` (or None)."""
    names = [str(c) for c in columns]
    return {
        "code": code,
        "severity": severity,
        "column": names[0] if names else None,
        "columns": names,
        "message": message,
        "fix": fix,
    }


def step_fix(step_type: str, params: dict[str, Any], label: str) -> dict[str, Any]:
    return {"type": step_type, "params": params, "label": label}


def names_text(names: Sequence[str], limit: int = 5) -> str:
    """``age, income`` or ``a, b, c, d, e and 3 more``."""
    names = [str(n) for n in names]
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + f" and {len(names) - limit:,} more"


def _columns_text(columns: Sequence[str]) -> str:
    return columns[0] if len(columns) == 1 else plural(len(columns), "column")


def _quoted(values: Sequence[str], limit: int = 3) -> str:
    shown = [f"'{v if len(v) <= 24 else v[:23] + '…'}'" for v in values[:limit]]
    return ", ".join(shown)


def sort_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Errors first (in the order found), then warnings and notes by ``PRIORITY``."""
    severity = {"error": 0, "warning": 1, "info": 2}
    rank = {code: i for i, code in enumerate(PRIORITY)}
    return sorted(
        issues,
        key=lambda i: (
            severity.get(i["severity"], 3),
            0 if i["severity"] == "error" else rank.get(i.get("code"), len(rank)),
        ),
    )


# --------------------------------------------------------------------------- #
# Samples and statistics
# --------------------------------------------------------------------------- #


def train_positions(table: Table, limit: int = CHECK_ROWS) -> np.ndarray:
    """Sorted positions of the training rows; a seeded random sample of ``limit`` of them."""
    positions = np.flatnonzero(table.train_mask())
    if len(positions) > limit:
        rng = np.random.default_rng(CHECK_SEED)
        positions = np.sort(rng.choice(positions, limit, replace=False))
    return positions


def _floats(series: pd.Series) -> np.ndarray:
    values = series.to_numpy(dtype=np.float64, na_value=np.nan).copy()
    values[~np.isfinite(values)] = np.nan
    return values


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    """|Pearson r| over the rows where both are present (like ``profile._pearson``, with
    element-wise sums, which are much faster than ``np.dot`` on some BLAS builds)."""
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 3:
        return None
    a, b = a[ok], b[ok]
    a = a - a.mean()
    b = b - b.mean()
    denominator = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    if not denominator or not math.isfinite(denominator):
        return None
    return min(abs(float((a * b).sum())) / denominator, 1.0)


def _skewness(x: np.ndarray) -> float | None:
    """Sample skewness like pandas (adjusted Fisher–Pearson) of finite values."""
    n = len(x)
    if n < 3:
        return None
    centered = x - x.mean()
    squares = centered * centered
    m2 = float(squares.mean())
    if m2 <= 0 or not math.isfinite(m2):
        return None
    m3 = float((squares * centered).mean())
    value = m3 / m2**1.5 * math.sqrt(n * (n - 1)) / (n - 2)
    return value if math.isfinite(value) else None


@dataclass
class _Stats:
    """Statistics of the numeric features on the training sample (one entry per column)."""

    names: list[str]
    values: np.ndarray  # rows × columns, NaN where missing
    count: np.ndarray
    unique: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    std: np.ndarray
    dominant: np.ndarray  # share of the most frequent value (exact when ≥ 50 %)
    integer: np.ndarray


def _numeric_stats(table: Table, columns: list[str], positions: np.ndarray) -> _Stats:
    rows = len(positions)
    # column by column in memory: the checks below work on single columns
    values = np.empty((rows, len(columns)), dtype=np.float64, order="F")
    for j, name in enumerate(columns):
        values[:, j] = _floats(table.df[name].iloc[positions])
    count = (~np.isnan(values)).sum(axis=0)
    ordered = np.sort(values, axis=0)  # NaN last
    if rows > 1:
        changes = ordered[1:] != ordered[:-1]
        inside = np.arange(1, rows)[:, None] < count[None, :]
        unique = (changes & inside).sum(axis=0) + (count > 0)
    else:
        unique = (count > 0).astype(np.int64)
    last = np.maximum(count - 1, 0)
    cols = np.arange(len(columns))
    minimum = np.where(count > 0, ordered[0, cols] if rows else np.nan, np.nan)
    maximum = np.where(count > 0, ordered[last, cols] if rows else np.nan, np.nan)
    median = ordered[last // 2, cols] if rows else np.full(len(columns), np.nan)
    with np.errstate(all="ignore"):
        std = np.nanstd(values, axis=0) if rows else np.zeros(len(columns))
        dominant = np.where(count > 0, (values == median).sum(axis=0) / np.maximum(count, 1), 0)
        whole = np.isnan(values) | (np.floor(values) == values)
    return _Stats(
        names=list(columns),
        values=values,
        count=count,
        unique=unique,
        minimum=minimum,
        maximum=maximum,
        std=np.nan_to_num(std),
        dominant=dominant,
        integer=whole.all(axis=0) & (count > 0),
    )


def _group_codes(values: np.ndarray | pd.Series) -> np.ndarray:
    """Codes 0 … k-1 of the values (-1 = missing)."""
    if isinstance(values, np.ndarray) and values.dtype.kind == "f":
        codes = np.full(len(values), -1, dtype=np.int64)
        present = ~np.isnan(values)
        codes[present] = pd.factorize(values[present])[0]
        return codes
    return np.asarray(pd.factorize(pd.Series(values, dtype=object))[0], dtype=np.int64)


# --------------------------------------------------------------------------- #
# String feature columns (errors: they must become numbers anyway)
# --------------------------------------------------------------------------- #


def _text_sample(table: Table, column: str, positions: np.ndarray) -> pd.Series:
    """Non-missing training values of a string column, at most ``TEXT_SAMPLE`` (seeded)."""
    values = table.df[column].iloc[positions].dropna()
    if len(values) > TEXT_SAMPLE:
        values = values.sample(TEXT_SAMPLE, random_state=CHECK_SEED)
    return values.astype(str).str.strip()


_AFFIXED = re.compile(rf"^([^\d+\-]*?)({NUMBER_PATTERN})(\D*)$")
_UNIT_PREFIX = r"[^\w\s]*|[A-Z]{3}|(?i:ca|approx|about|circa)\.?"


def _unit_like(text: pd.Series, weights: np.ndarray) -> bool:
    """Whether the text around the numbers looks like a unit or currency: the same text
    (e.g. "$", " kg", " %") in at least ``UNIT_SHARE`` of the values, and no letter right
    before the number (codes such as "Z98334" or "SKU12")."""
    parts = text.str.extract(_AFFIXED)
    prefix = parts[0].fillna("")
    suffix = parts[2].fillna("")
    total = weights.sum()
    # Before the number only symbols ("$", "~"), currency codes ("EUR ") or "ca.": a word
    # before a number is a name ("shop 17", "Room 12") or a code ("Z98334").
    named = ~prefix.str.strip().str.fullmatch(_UNIT_PREFIX, na=False).to_numpy()
    if weights[named].sum() > (1 - UNIT_SHARE) * total:
        return False
    keys = (
        prefix.str.strip().str.casefold()
        + "\x00"
        + suffix.str.strip().str.casefold().str.removesuffix("s")
    )
    shares = pd.Series(weights, index=keys.to_numpy()).groupby(level=0).sum()
    return bool(shares.max() >= UNIT_SHARE * total)


def numbers_in_text(values: pd.Series) -> dict[str, Any] | None:
    """How a string column (its stripped values) can be read as numbers with the *Change
    type* step: ``{"decimal_comma", "strip_units", "share", "examples", "others"}``, or None
    if fewer than ``NUMBER_SHARE`` of the values are numbers, if all of them are plain
    numbers already (categories on purpose) or if they are codes with leading zeros."""
    counts = values.value_counts()
    if len(counts) < MIN_TEXT_VALUES:
        return None
    text = pd.Series(counts.index.astype(str), dtype=object)
    weights = counts.to_numpy(dtype=np.float64)
    total = weights.sum()
    if weights[text.str.contains(r"\d", regex=True).to_numpy()].sum() < NUMBER_SHARE * total:
        return None
    if float(text.str.len().median()) > 40:
        return None
    extracted = text.str.extract(UNIT_NUMBER, expand=False).str.replace("'", "", regex=False)
    extracted = extracted.fillna("")
    plain = ~np.isnan(exact_numbers(text).to_numpy())
    if plain.all():
        return None  # numbers that are categories on purpose (e.g. changed by a step)
    best: tuple[float, bool, bool, np.ndarray] | None = None
    # Ties: a decimal point ("12,345" is 12345), then without "strip_units".
    for decimal_comma in (False, True):
        for strip_units in (False, True):
            numbers = exact_numbers(number_text(text, decimal_comma, strip_units)).to_numpy()
            # Only what the import reads as numbers counts: "1,5" is no number with a
            # decimal point ("Change type" would read it as 15).
            ok = ~np.isnan(numbers) & ~np.isnan(
                parse_numbers((extracted if strip_units else text).tolist(), ".,"[decimal_comma])
            )
            share = float(weights[ok].sum() / total)
            if best is None or share > best[0] + 1e-12:
                best = (share, decimal_comma, strip_units, ok)
    assert best is not None
    share, decimal_comma, strip_units, ok = best
    if share < NUMBER_SHARE:
        return None
    if strip_units and not _unit_like(text[ok], weights[ok]):
        return None
    numbers = extracted[ok] if strip_units else text[ok]
    joined = "\n".join(numbers.tolist())
    if len(_LEADING_ZERO.findall(joined)) >= max(1, LEADING_ZERO_SHARE * len(numbers)):
        return None  # codes such as zip codes
    return {
        "decimal_comma": decimal_comma,
        "strip_units": strip_units,
        "share": share,
        "examples": text[ok].tolist()[:3],
        "others": text[~ok].tolist()[:3],
    }


_DATE_LIKE = re.compile(r"\d.*[-/.:]|[-/.:].*\d")


def dates_in_text(values: pd.Series) -> bool:
    """Whether at least ``DATE_SHARE`` of the values are dates (in one of the formats the
    *Change type* step reads)."""
    if len(values) < MIN_TEXT_VALUES or values.nunique() < MIN_TEXT_VALUES:
        return False
    if values.str.match(_DATE_LIKE).mean() < DATE_SHARE:
        return False
    parsed = parse_dates(values, date_format(values))
    return bool(parsed.notna().mean() >= DATE_SHARE)


def _variant_key(value: str) -> str:
    return " ".join(value.split()).casefold()


def category_variants(table: Table, column: str, train_counts: pd.Series) -> dict[str, Any] | None:
    """Values of a string column that differ only in upper/lower case or spaces.

    Returns ``{"groups": [[spellings …], …], "mapping": {variant: most frequent spelling in
    the training rows}}`` (``mapping`` None if it would be longer than ``MAX_REPLACEMENTS``),
    or None if there are no such values in the training rows.
    """
    keys = pd.Series(train_counts.index.astype(str), dtype=object).map(_variant_key)
    if not keys.duplicated().any():
        return None
    # The whole column: spellings in validation/test rows are replaced as well.
    counts = table.df[column].value_counts(dropna=True)
    train = {str(k): int(v) for k, v in train_counts.items()}
    by_key: dict[str, list[str]] = {}
    for value in counts.index.astype(str):
        by_key.setdefault(_variant_key(value), []).append(value)
    groups = []
    mapping: dict[str, str] = {}
    for spellings in by_key.values():
        if len(spellings) < 2:
            continue
        ranked = sorted(spellings, key=lambda v: (-train.get(v, 0), -int(counts.get(v, 0)), v))
        groups.append(ranked)
        mapping.update({v: ranked[0] for v in ranked[1:]})
    if not groups:
        return None
    groups.sort(key=lambda g: -sum(train.get(v, 0) for v in g))
    return {"groups": groups, "mapping": mapping if len(mapping) <= MAX_REPLACEMENTS else None}


def _variants_text(groups: list[list[str]]) -> str:
    text = " / ".join(_quoted(groups[0], 4).split(", "))
    if len(groups) > 1:
        text += f" and {plural(len(groups) - 1, 'more group')}"
    return text


def _named_id(name: str) -> bool:
    return bool(_ID_NAME.search(name) or _CAMEL_ID.search(name))


def string_feature_issues(
    table: Table, categorical: list[str], text: list[str], task: str | None
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Errors for string feature columns that need more than one-hot encoding or dropping:
    numbers or dates stored as text (fix: change the type), ID-like values (drop), spellings
    that differ only in case or spaces (replace), too many categories (frequency or target
    mean encoding). Returns the issues and the categorical and text columns left for the
    usual one-hot / drop errors."""
    positions = train_positions(table)
    issues: list[dict[str, Any]] = []
    numbers: dict[tuple[bool, bool], list[tuple[str, dict[str, Any]]]] = {}
    dates: list[str] = []
    ids: list[str] = []
    many: list[tuple[str, int]] = []
    onehot: list[str] = []
    free: list[str] = []
    for column in [*categorical, *text]:
        values = _text_sample(table, column, positions)
        found = numbers_in_text(values)
        if found is not None:
            numbers.setdefault((found["decimal_comma"], found["strip_units"]), []).append(
                (column, found)
            )
            continue
        if dates_in_text(values):
            dates.append(column)
            continue
        if table.kinds[column] == "text":
            free.append(column)
            continue
        train_counts = table.df[column].iloc[positions].value_counts(dropna=True)
        rows, distinct = int(train_counts.sum()), len(train_counts)
        if (rows >= ID_MIN_ROWS and distinct >= ID_UNIQUE_SHARE * rows) or (
            rows >= 2 and distinct == rows and _named_id(column)
        ):
            ids.append(column)
            continue
        variants = category_variants(table, column, train_counts)
        if variants is not None:
            mapping = variants["mapping"]
            issues.append(
                issue(
                    "inconsistent_categories",
                    "error",
                    f"{column} has values that differ only in upper/lower case or spaces "
                    f"({_variants_text(variants['groups'])}); they would become different "
                    "categories. Unify them, then encode the column.",
                    [column],
                    step_fix(
                        "replace_values",
                        {"columns": [column], "mapping": mapping},
                        f"Unify the spellings in {column}",
                    )
                    if mapping
                    else None,
                )
            )
            continue
        if distinct > HIGH_CARDINALITY:
            many.append((column, distinct))
            continue
        onehot.append(column)

    for (decimal_comma, strip_units), found in numbers.items():
        columns = [c for c, _ in found]
        if len(found) == 1:
            column, info = found[0]
            message = (
                f"{column} holds numbers as text ({_quoted(info['examples'])}). Convert it to "
                "numbers"
            )
            if info["share"] < 1:
                message += (
                    f"; {fmt_share(1 - info['share'], 1)} of its values are not numbers "
                    f"({_quoted(info['others'])}) and become missing"
                )
            message += "."
        else:
            listed = "; ".join(f"{c}: {_quoted(i['examples'], 2)}" for c, i in found[:4])
            message = (
                f"{len(found)} feature columns hold numbers as text ({listed}"
                f"{' …' if len(found) > 4 else ''}). Convert them to numbers."
            )
        params: dict[str, Any] = {"columns": columns, "to": "numeric"}
        if decimal_comma:
            params["decimal_comma"] = True
        if strip_units:
            params["strip_units"] = True
        issues.append(
            issue(
                "numbers_as_text",
                "error",
                message,
                columns,
                step_fix("cast", params, f"Convert {_columns_text(columns)} to numbers"),
            )
        )
    if dates:
        issues.append(
            issue(
                "dates_as_text",
                "error",
                (
                    f"{dates[0]} holds dates as text. Convert it to dates, then split it into "
                    "parts."
                    if len(dates) == 1
                    else f"{len(dates)} feature columns hold dates as text ({names_text(dates)}). "
                    "Convert them to dates, then split them into parts."
                ),
                dates,
                step_fix(
                    "cast",
                    {"columns": dates, "to": "datetime"},
                    f"Convert {_columns_text(dates)} to dates",
                ),
            )
        )
    if ids:
        issues.append(
            issue(
                "id_like",
                "error",
                (
                    f"{ids[0]} has a different value in (almost) every row, like an ID or a "
                    "name. It cannot help to predict new rows."
                    if len(ids) == 1
                    else f"{len(ids)} feature columns have a different value in (almost) every "
                    f"row ({names_text(ids)}), like IDs or names. They cannot help to predict "
                    "new rows."
                ),
                ids,
                step_fix("drop_columns", {"columns": ids}, f"Drop {_columns_text(ids)}"),
            )
        )
    if many:
        columns = [c for c, _ in many]
        method = "target" if task == "regression" else "frequency"
        how = "the mean target value" if method == "target" else "how often it occurs"
        what = (
            f"{many[0][0]} has {many[0][1]:,} categories"
            if len(many) == 1
            else f"{len(many)} feature columns have many categories ("
            + ", ".join(f"{c}: {n:,}" for c, n in many[:4])
            + (" …" if len(many) > 4 else "")
            + ")"
        )
        issues.append(
            issue(
                "high_cardinality",
                "error",
                f"{what}. One-hot encoding would keep only the {ONEHOT_CATEGORIES} most frequent "
                f"and add a column for each; encoding every category by {how} needs one column.",
                columns,
                step_fix(
                    "encode",
                    {"columns": columns, "method": method},
                    f"Encode {_columns_text(columns)} ("
                    + ("target mean" if method == "target" else "frequency")
                    + ")",
                ),
            )
        )
    return issues, onehot, free


# --------------------------------------------------------------------------- #
# Target
# --------------------------------------------------------------------------- #


def class_issues(
    table: Table,
    target: str,
    names: list[str],
    codes: np.ndarray,
    stratify: bool,
) -> list[dict[str, Any]]:
    """Rare classes (fewer than ``MIN_CLASS_ROWS`` training rows; fix: remove their rows)
    and classes missing from the validation or test rows."""
    issues: list[dict[str, Any]] = []
    split = table.split if table.split is not None else np.zeros(table.n_rows, dtype=np.int8)
    present = codes >= 0
    counts = [
        np.bincount(codes[present & (split == part)], minlength=len(names))
        for part in (TRAIN, VAL, TEST)
    ]
    train = counts[TRAIN]
    rare = [i for i in range(len(names)) if train[i] < MIN_CLASS_ROWS]
    if rare:
        listed = ", ".join(f"{names[i]}: {int(train[i])}" for i in rare[:5])
        more = f" and {len(rare) - 5:,} more" if len(rare) > 5 else ""
        fix = None
        if len(names) - len(rare) >= 2:
            kind = table.kinds[target]
            values: list[Any] = [float(names[i]) if kind == "numeric" else names[i] for i in rare]
            if kind != "boolean":
                fix = step_fix(
                    "filter_rows",
                    {"column": target, "operator": "not_in", "values": values},
                    f"Remove the rows of {names_text([names[i] for i in rare], 3)}"
                    if len(rare) <= 3
                    else f"Remove the rows of {len(rare)} rare classes",
                )
        issues.append(
            issue(
                "rare_classes",
                "warning",
                f"{plural(len(rare), 'class has', 'classes have')} fewer than {MIN_CLASS_ROWS} "
                f"training rows ({listed}{more}); the network can hardly learn "
                f"{'it' if len(rare) == 1 else 'them'}.",
                [target],
                fix,
            )
        )
    sizes = table.split_counts() or [table.n_rows, 0, 0]
    gaps = []
    for part, label in ((VAL, "validation"), (TEST, "test")):
        if sizes[part] == 0:
            continue
        lacking = [
            names[i]
            for i in range(len(names))
            if train[i] >= MIN_CLASS_ROWS and counts[part][i] == 0
        ]
        if lacking:
            gaps.append((label, lacking))
    if gaps:
        label, lacking = gaps[0]
        parts = " and ".join(g[0] for g in gaps)
        all_lacking = sorted({c for _, cs in gaps for c in cs}, key=names.index)
        hint = (
            "Turn on “Stratified” under Target & split."
            if not stratify
            else "Use larger validation and test shares."
        )
        issues.append(
            issue(
                "classes_not_in_split",
                "warning",
                f"{plural(len(all_lacking), 'class has', 'classes have')} no {parts} rows "
                f"({names_text(all_lacking)}), so the {parts} scores say nothing about "
                f"{'it' if len(all_lacking) == 1 else 'them'}. {hint}",
                [target],
            )
        )
    return issues


def target_variant_issues(table: Table, target: str) -> list[dict[str, Any]]:
    """Classes of a string target that differ only in case or spaces."""
    positions = train_positions(table)
    counts = table.df[target].iloc[positions].value_counts(dropna=True)
    variants = category_variants(table, target, counts)
    if variants is None:
        return []
    mapping = variants["mapping"]
    return [
        issue(
            "inconsistent_categories",
            "warning",
            f"The target {target} has classes that differ only in upper/lower case or spaces "
            f"({_variants_text(variants['groups'])}); they count as different classes.",
            [target],
            step_fix(
                "replace_values",
                {"columns": [target], "mapping": mapping},
                f"Unify the spellings in {target}",
            )
            if mapping
            else None,
        )
    ]


def _transform_fix(columns: list[str], function: str) -> dict[str, Any]:
    if function == "log1p":
        label = f"Log-transform {_columns_text(columns)} (log1p)"
    else:
        label = f"Yeo-Johnson transform {_columns_text(columns)}"
    return step_fix("transform", {"columns": columns, "function": function}, label)


def _skew_function(x: np.ndarray, skew: float) -> str | None:
    """``log1p`` if the values are ≥ 0 and log(1 + x) halves the skewness, ``yeo_johnson``
    for values with negative numbers, None if log1p does not help."""
    if x.min() < 0:
        return "yeo_johnson"
    after = _skewness(np.log1p(x))
    if after is not None and abs(after) <= abs(skew) / 2:
        return "log1p"
    return None


def regression_target_issues(table: Table, target: str) -> list[dict[str, Any]]:
    """A strongly skewed numeric target (fix: transform it)."""
    values = _floats(table.df[target].iloc[train_positions(table)])
    values = values[~np.isnan(values)]
    if len(values) < LEAK_MIN_ROWS or len(np.unique(values)) < SKEW_MIN_VALUES:
        return []
    skew = _skewness(values)
    if skew is None or abs(skew) <= SKEWNESS:
        return []
    function = _skew_function(values, skew)
    if function is None:
        return []
    how = "log(1 + x)" if function == "log1p" else "a Yeo-Johnson transform"
    return [
        issue(
            "skewed_target",
            "info",
            f"The target {target} is strongly skewed (skewness {skew:.1f}): the few largest "
            f"values dominate the loss. Training on {how} of it often works better; the "
            "network then predicts transformed values.",
            [target],
            _transform_fix([target], function),
        )
    ]


# --------------------------------------------------------------------------- #
# Feature checks: leakage, ID-like, near-constant, redundant, skewed, scales
# --------------------------------------------------------------------------- #


def _leak_name(feature: str, target: str) -> bool:
    """``<target>_id``, ``<target>_label``, ``is_<target>``, ``<Target>`` … (names only)."""
    f, t = feature.casefold().strip(), target.casefold().strip()
    if len(t) < 2:
        return False
    if f == t:
        return True
    sep = r"[\s_\-.]*"
    t = re.escape(t)
    patterns = (
        rf"{t}{sep}(?:{LEAK_SUFFIXES})",
        rf"(?:is|has|was){sep}{t}",
        rf"(?:{LEAK_SUFFIXES}){sep}(?:of{sep})?{t}",
    )
    return any(re.fullmatch(p, f) for p in patterns)


def _purity(groups: np.ndarray, classes: np.ndarray) -> tuple[float, float, int] | None:
    """Errors of predicting the class from the group (most frequent class of each group)
    and of always predicting the largest class, over the rows of groups with at least
    ``LEAK_VALUE_ROWS`` rows, and the number of those rows; None if those rows are fewer
    than ``LEAK_COVERAGE`` of all."""
    ok = (groups >= 0) & (classes >= 0)
    n = int(ok.sum())
    if n < LEAK_MIN_ROWS:
        return None
    g, c = groups[ok], classes[ok]
    sizes = np.bincount(g)
    frequent = sizes[g] >= LEAK_VALUE_ROWS
    rows = int(frequent.sum())
    if rows < LEAK_COVERAGE * n or rows < LEAK_MIN_ROWS:
        return None
    g, c = g[frequent], c[frequent]
    k = int(c.max()) + 1
    table = np.bincount(g * k + c, minlength=(int(g.max()) + 1) * k).reshape(-1, k)
    error = 1 - table.max(axis=1).sum() / rows
    base = 1 - np.bincount(c, minlength=k).max() / rows
    return float(error), float(base), rows


def _leaks_by_error(error: float, base: float, rows: int) -> bool:
    return base * rows >= LEAK_VALUE_ROWS and error <= min(LEAK_MAX_ERROR, LEAK_ERROR_SHARE * base)


def _stump_error(x: np.ndarray, classes: np.ndarray) -> float | None:
    """Errors of the best single threshold on ``x`` for a binary target (0/1)."""
    ok = ~np.isnan(x) & (classes >= 0)
    if ok.sum() < LEAK_MIN_ROWS:
        return None
    order = np.argsort(x[ok], kind="stable")
    values, y = x[ok][order], classes[ok][order]
    n = len(y)
    ones_left = np.cumsum(y)
    total_ones = int(ones_left[-1])
    # thresholds only between different values
    cut = np.flatnonzero(values[1:] != values[:-1])
    if len(cut) == 0:
        return None
    left = cut + 1
    ones = ones_left[cut]
    zeros = left - ones
    # predict 0 left / 1 right, or 1 left / 0 right
    correct_a = zeros + (total_ones - ones)
    correct_b = ones + ((n - left) - (total_ones - ones))
    best = max(int(correct_a.max()), int(correct_b.max()))
    return 1 - best / n


@dataclass
class _Leak:
    column: str
    detail: str


def _leak(
    name: str,
    x: np.ndarray | None,
    groups: np.ndarray | None,
    target: str,
    task: str,
    y: np.ndarray,
    discrete: bool,
) -> _Leak | None:
    """Whether feature ``name`` (numbers ``x`` and/or group codes ``groups`` on the training
    sample) (almost) determines the target ``y`` (values or class codes)."""
    if task == "regression":
        if x is not None:
            both = ~np.isnan(x) & ~np.isnan(y)
            if both.sum() < LEAK_MIN_ROWS:
                return None
            if np.mean(x[both] == y[both]) >= 0.99:
                return _Leak(name, f"{name} equals the target {target} in almost every row")
            r = _pearson(x, y)
            if r is not None and r >= LEAK_CORRELATION:
                return _Leak(
                    name, f"{name} correlates almost perfectly with the target (|r| = {r:.3f})"
                )
            return None
        assert groups is not None
        ok = (groups >= 0) & ~np.isnan(y)
        sizes = np.bincount(groups[ok]) if ok.any() else np.zeros(0, dtype=np.int64)
        frequent = ok.copy()
        frequent[ok] = sizes[groups[ok]] >= LEAK_VALUE_ROWS
        if frequent.sum() < max(LEAK_MIN_ROWS, LEAK_COVERAGE * ok.sum()):
            return None
        eta = _eta(y[frequent], groups[frequent])
        if eta is not None and eta >= LEAK_ETA:
            return _Leak(
                name, f"the target {target} is almost the same within each value of {name}"
            )
        return None
    # classification
    if discrete:
        codes = groups if groups is not None else _group_codes(x)
        found = _purity(codes, y)
        if found is not None and _leaks_by_error(*found):
            error = found[0]
            return _Leak(
                name,
                f"each value of {name} belongs to one class in "
                f"{fmt_share(1 - error, 1)} of the training rows",
            )
        return None
    assert x is not None
    classes = int(y.max()) + 1 if (y >= 0).any() else 0
    if classes == 2:
        error = _stump_error(x, y)
        ok = ~np.isnan(x) & (y >= 0)
        base = 1 - np.bincount(y[ok], minlength=2).max() / max(int(ok.sum()), 1)
        if error is not None and _leaks_by_error(error, base, int(ok.sum())):
            return _Leak(
                name,
                f"a single threshold on {name} separates the classes in "
                f"{fmt_share(1 - error, 1)} of the training rows",
            )
    ok = ~np.isnan(x) & (y >= 0)
    if ok.sum() >= LEAK_MIN_ROWS:
        eta = _eta(x[ok], y[ok])
        if eta is not None and eta >= LEAK_ETA:
            return _Leak(name, f"{name} is almost constant within each class")
    return None


def feature_issues(
    table: Table,
    target: str | None,
    task: str | None,
    class_codes: np.ndarray | None,
    *,
    constant: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Warnings and notes about the feature columns (all but the target): target leakage,
    ID-like numbers, near-constant, redundant and skewed features, very different scales.

    ``class_codes`` are the class codes of all rows (classification). Columns in
    ``constant`` are reported elsewhere and skipped.
    """
    features = [c for c in table.columns if c != target]
    numeric = [c for c in features if table.kinds[c] in NUMERIC_KINDS and c not in constant]
    issues: list[dict[str, Any]] = []
    if not features:
        return issues
    rows = max(MIN_CHECK_ROWS, CHECK_CELLS // max(len(numeric), 1))
    positions = train_positions(table, min(CHECK_ROWS, rows))
    stats = _numeric_stats(table, numeric, positions)
    flagged: set[str] = set(constant)

    # ---- target leakage
    has_target = target is not None and target in table.kinds and task is not None
    y: np.ndarray | None = None
    if has_target and task == "classification" and class_codes is not None:
        y = class_codes[positions]
    elif has_target and task == "regression" and table.kinds[target] in NUMERIC_KINDS:
        y = _floats(table.df[target].iloc[positions])
    leaks: list[_Leak] = []
    if y is not None and target is not None and task is not None:
        known = (y >= 0) if task == "classification" else ~np.isnan(y)
        if known.sum() < LEAK_MIN_ROWS:
            y = None
    if y is not None and target is not None and task is not None:
        for j, name in enumerate(stats.names):
            if stats.count[j] < LEAK_MIN_ROWS or stats.unique[j] < 2:
                continue
            discrete = table.kinds[name] == "boolean" or stats.unique[j] <= LEAK_MAX_VALUES
            found = _leak(name, stats.values[:, j], None, target, task, y, discrete)
            if found:
                leaks.append(found)
        for name in [c for c in features if table.kinds[c] == "categorical"]:
            groups = _group_codes(table.df[name].iloc[positions])
            if len(np.unique(groups[groups >= 0])) < 2:
                continue
            found = _leak(name, None, groups, target, task, y, True)
            if found:
                leaks.append(found)
    if target is not None and target in table.kinds:
        by_name = {leak.column for leak in leaks}
        for name in features:
            if name not in by_name and name not in constant and _leak_name(name, target):
                leaks.append(_Leak(name, ""))
    # One issue per column; the one-hot columns of one category column ("city=Berlin") together.
    grouped: dict[str, list[_Leak]] = {}
    for leak in leaks:
        source = leak.column.split("=", 1)[0] if leak.detail and "=" in leak.column else ""
        grouped.setdefault(source or leak.column, []).append(leak)
    for source, found in grouped.items():
        columns = [leak.column for leak in found]
        flagged.update(columns)
        first = found[0]
        if not first.detail:
            message = (
                f"Possible target leakage: by its name, {first.column} looks like another form "
                f"of the target {target}. If it is derived from it, drop it – new data would "
                "not have it."
            )
        elif len(found) == 1:
            message = (
                f"Possible target leakage: {first.detail}. If {first.column} is derived from "
                f"{target} or only known after it, drop it – new data would not have it."
            )
        else:
            message = (
                f"Possible target leakage: {first.detail} (and the same for "
                f"{names_text(columns[1:], 3)}). If {source} is derived from {target} or only "
                "known after it, drop these columns – new data would not have them."
            )
        issues.append(
            issue(
                "leakage",
                "warning",
                message,
                columns,
                step_fix("drop_columns", {"columns": columns}, f"Drop {_columns_text(columns)}"),
            )
        )

    # ---- ID-like numbers
    ids = []
    for j, name in enumerate(stats.names):
        if name in flagged or table.kinds[name] != "numeric":
            continue
        count = int(stats.count[j])
        if not stats.integer[j] or count < ID_MIN_ROWS:
            continue
        if stats.unique[j] == count and _row_numbers(table.df[name]):
            ids.append(name)
        elif stats.unique[j] >= ID_UNIQUE_SHARE * count and _named_id(name):
            # (repeated rows repeat their ID too)
            ids.append(name)
    if ids:
        flagged.update(ids)
        issues.append(
            issue(
                "id_like",
                "warning",
                (
                    f"{ids[0]} looks like an ID (a different whole number in (almost) every "
                    "row). As a feature, it lets the network memorise rows instead of learning "
                    "patterns."
                    if len(ids) == 1
                    else f"{len(ids)} feature columns look like IDs ({names_text(ids)}). As "
                    "features, they let the network memorise rows instead of learning patterns."
                ),
                ids,
                step_fix("drop_columns", {"columns": ids}, f"Drop {_columns_text(ids)}"),
            )
        )

    # ---- near-constant
    near = [
        (name, float(stats.dominant[j]))
        for j, name in enumerate(stats.names)
        if name not in flagged
        and stats.count[j] >= NEAR_CONSTANT_MIN_ROWS
        and stats.dominant[j] >= NEAR_CONSTANT
    ]
    if near:
        columns = [c for c, _ in near]
        flagged.update(columns)
        listed = ", ".join(f"{c}: {fmt_share(s, 1)}" for c, s in near[:5])
        more = f" and {len(near) - 5:,} more" if len(near) > 5 else ""
        issues.append(
            issue(
                "near_constant",
                "info",
                (
                    f"{columns[0]} has the same value in {fmt_share(near[0][1], 1)} of the "
                    "training rows; it carries almost no information."
                    if len(near) == 1
                    else f"{len(near)} features have the same value in at least "
                    f"{NEAR_CONSTANT:.0%} of the training rows ({listed}{more}); they carry "
                    "almost no information."
                ),
                columns,
                step_fix("drop_columns", {"columns": columns}, f"Drop {_columns_text(columns)}"),
            )
        )

    # ---- redundant pairs
    redundant = _redundant(stats, flagged, y, task)
    if redundant:
        drop = [d for d, _, _ in redundant]
        flagged.update(drop)
        pairs = [f"{d} ≈ {k} ({_r_text(r)})" for d, k, r in redundant[:4]]
        more = " …" if len(redundant) > 4 else ""
        issues.append(
            issue(
                "redundant",
                "info",
                (
                    f"{drop[0]} is almost a copy of {redundant[0][1]} "
                    f"({_r_text(redundant[0][2])}); one of them is enough."
                    if len(redundant) == 1
                    else f"{len(drop)} features are almost copies of others ({'; '.join(pairs)}"
                    f"{more}); one of each pair is enough."
                ),
                drop,
                step_fix("drop_columns", {"columns": drop}, f"Drop {_columns_text(drop)}"),
            )
        )

    # ---- skewed numbers
    skewed: dict[str, list[tuple[str, float]]] = {}
    for j, name in enumerate(stats.names):
        if name in flagged or table.kinds[name] != "numeric":
            continue
        if stats.unique[j] < SKEW_MIN_VALUES:
            continue
        x = stats.values[:, j]
        x = x[~np.isnan(x)]
        skew = _skewness(x)
        if skew is None or abs(skew) <= SKEWNESS:
            continue
        function = _skew_function(x, skew)
        if function is not None:
            skewed.setdefault(function, []).append((name, skew))
    for function, found in skewed.items():
        columns = [c for c, _ in found]
        listed = ", ".join(f"{c}: {s:.1f}" for c, s in found[:5])
        more = f" and {len(found) - 5:,} more" if len(found) > 5 else ""
        issues.append(
            issue(
                "skewed",
                "info",
                (
                    f"{columns[0]} is strongly skewed (skewness {found[0][1]:.1f}): a few "
                    "large values dominate it."
                    if len(found) == 1
                    else f"{len(found)} features are strongly skewed (skewness {listed}{more}): "
                    "a few large values dominate them."
                )
                + (
                    " log(1 + x) makes it more even."
                    if function == "log1p" and len(found) == 1
                    else " log(1 + x) makes them more even."
                    if function == "log1p"
                    else " A Yeo-Johnson transform makes "
                    + ("it" if len(found) == 1 else "them")
                    + " more even."
                ),
                columns,
                _transform_fix(columns, function),
            )
        )

    # ---- scales
    spread = [
        (name, float(stats.std[j]))
        for j, name in enumerate(stats.names)
        if name not in flagged
        and table.kinds[name] == "numeric"
        and stats.unique[j] > 2
        and stats.std[j] > 0
        and math.isfinite(stats.std[j])
    ]
    if len(spread) >= 2:
        largest = max(spread, key=lambda s: s[1])
        smallest = min(spread, key=lambda s: s[1])
        if largest[1] >= SCALE_RATIO * smallest[1]:
            columns = [c for c, _ in spread]
            issues.append(
                issue(
                    "scales",
                    "info",
                    f"The features have very different scales (standard deviation of "
                    f"{largest[0]}: {fmt_number(largest[1])}, of {smallest[0]}: "
                    f"{fmt_number(smallest[1])}). Training in Brick DNN standardises them "
                    "anyway; for the exported data and other tools, scale them.",
                    [largest[0], smallest[0]],
                    step_fix(
                        "scale",
                        {"columns": columns, "method": "standard"},
                        f"Standardise {_columns_text(columns)}",
                    ),
                )
            )
    return issues


def _r_text(r: float) -> str:
    """``r = 0.987``, ``r ≈ 1`` or ``r ≈ -1``."""
    if abs(r) >= 0.9995:
        return "r ≈ 1" if r > 0 else "r ≈ -1"
    return f"r = {r:.3f}"


def _row_numbers(series: pd.Series) -> bool:
    """Whole numbers 0 … n-1 or 1 … n, each once (row numbers saved with the data)."""
    values = _floats(series)
    values = values[~np.isnan(values)]
    if len(values) < ID_MIN_ROWS:
        return False
    low, high = float(values.min()), float(values.max())
    return (
        low in (0.0, 1.0)
        and high - low + 1 == len(values)
        and len(np.unique(values)) == len(values)
    )


def _redundant(
    stats: _Stats, flagged: set[str], y: np.ndarray | None, task: str | None
) -> list[tuple[str, str, float]]:
    """Pairs of numeric features with |r| ≥ ``REDUNDANT`` on the training sample:
    ``(dropped, kept, r)``; of each pair the one less associated with the target is dropped
    (the later one without a target)."""
    usable = [
        j
        for j, name in enumerate(stats.names)
        if name not in flagged and stats.unique[j] >= 2 and stats.std[j] > 0
    ]
    if len(usable) < 2 or len(usable) > MAX_PAIR_COLUMNS:
        return []
    x = stats.values[:, usable]
    count = (~np.isnan(x)).sum(axis=0)
    with np.errstate(all="ignore"):
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        z = np.nan_to_num((x - mean) / std)
        r = (z.T @ z) / np.maximum(np.minimum.outer(count, count), 1)
    np.fill_diagonal(r, 0)
    pairs = np.argwhere(np.triu(np.abs(r) >= REDUNDANT))
    if len(pairs) == 0:
        return []
    names = [stats.names[j] for j in usable]
    association = _associations(x, y, task) if y is not None else None
    ranked = sorted(pairs.tolist(), key=lambda p: (-abs(r[p[0], p[1]]), p[0], p[1]))
    dropped: set[int] = set()
    found = []
    for a, b in ranked:
        if a in dropped or b in dropped:
            continue
        drop, keep = b, a
        if association is not None and association[b] > association[a]:
            drop, keep = a, b
        dropped.add(drop)
        found.append((names[drop], names[keep], float(np.clip(r[a, b], -1, 1))))
    return found


def _associations(x: np.ndarray, y: np.ndarray, task: str | None) -> np.ndarray:
    result = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        if task == "classification":
            ok = ~np.isnan(x[:, j]) & (y >= 0)
            value = _eta(x[ok, j], y[ok]) if ok.sum() >= 3 else None
        else:
            value = _pearson(x[:, j], y)
        result[j] = value or 0.0
    return result


# --------------------------------------------------------------------------- #
# Rows: duplicates, rows shared by training and validation/test, conflicting labels, size
# --------------------------------------------------------------------------- #


def _row_hashes(df: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(df, index=False).to_numpy()


def row_issues(
    table: Table, target: str | None, task: str | None, num_features: int
) -> list[dict[str, Any]]:
    """Exact duplicate rows, validation/test rows that are copies of training rows (fix
    for both: remove duplicates), rows with the same features but different classes, and
    too few training rows for the number of features."""
    issues: list[dict[str, Any]] = []
    n = table.n_rows
    if n == 0 or not table.columns:
        return issues
    has_target = target is not None and target in table.kinds
    positions = None
    if n > DUPLICATE_ROWS:
        rng = np.random.default_rng(CHECK_SEED)
        positions = np.sort(rng.choice(n, DUPLICATE_ROWS, replace=False))
    df = table.df if positions is None else table.df.iloc[positions]
    split = table.split
    if split is not None and positions is not None:
        split = split[positions]
    features = [c for c in table.columns if c != target]
    rows = len(df)
    sampled = f"In a sample of {rows:,} rows, " if positions is not None else ""

    # One hash of the features and one of the target per row, combined for the whole row.
    zeros = np.zeros(rows, dtype=np.uint64)
    feature_hashes = _row_hashes(df[features]) if features else zeros
    target_hashes = _row_hashes(df[[target]]) if has_target else zeros
    with np.errstate(over="ignore"):
        hashes = feature_hashes ^ (target_hashes * np.uint64(0x9E3779B97F4A7C15))
    candidates = pd.Series(hashes).duplicated(keep=False).to_numpy()
    duplicates = 0
    if candidates.any():
        duplicates = int(df[candidates].duplicated(keep="first").sum())
    fix = step_fix("drop_duplicates", {"columns": [], "keep": "first"}, "Remove duplicates")
    overlap = held = 0
    if duplicates and split is not None:
        held_out = split != TRAIN
        held = int(held_out.sum())
        overlap = int(np.isin(hashes[held_out], hashes[~held_out]).sum())
    if overlap:
        share = overlap / max(held, 1)
        text = f"{plural(overlap, 'validation/test row')} ({fmt_share(overlap, held)})"
        issues.append(
            issue(
                "split_overlap",
                "warning" if share >= OVERLAP_WARNING else "info",
                f"{sampled}{text} {'is an exact copy' if overlap == 1 else 'are exact copies'} "
                "of training rows, so the validation and test scores are too optimistic "
                f"({plural(duplicates, 'duplicate row')} in all).",
                [],
                fix,
            )
        )
    elif duplicates:
        share = duplicates / rows
        message = (
            f"{plural(duplicates, 'row')} ({fmt_share(duplicates, rows)}) "
            f"{'is an exact copy' if duplicates == 1 else 'are exact copies'} of other rows."
        )
        if sampled:
            message = sampled + message[0].lower() + message[1:]
        issues.append(
            issue(
                "duplicate_rows",
                "warning" if share >= DUPLICATE_WARNING else "info",
                message,
                [],
                fix,
            )
        )

    if has_target and task == "classification" and features:
        frame = pd.DataFrame({"f": feature_hashes, "y": target_hashes})
        repeated = frame["f"].duplicated(keep=False).to_numpy()
        if repeated.any():
            classes = frame[repeated].groupby("f")["y"].nunique()
            mixed = classes.index[classes > 1]
            conflicting = int(frame["f"].isin(mixed).sum())
            if conflicting and conflicting >= CONFLICT_SHARE * rows:
                issues.append(
                    issue(
                        "conflicting_labels",
                        "info",
                        f"{sampled}{plural(conflicting, 'row')} ({fmt_share(conflicting, rows)}) "
                        "have the same features as another row but a different class; the "
                        "network cannot tell them apart. Check the labels, or add features "
                        "that separate them.",
                        [target] if target else [],
                    )
                )

    train_rows = (table.split_counts() or [n, 0, 0])[TRAIN]
    if num_features and train_rows >= 4:
        if train_rows < MIN_TRAIN_ROWS:
            message = (
                f"Only {plural(train_rows, 'training row')}: too few to train a network "
                "reliably. Collect more data if you can."
            )
        elif train_rows < ROWS_PER_FEATURE * num_features:
            message = (
                f"{plural(train_rows, 'training row')} for {plural(num_features, 'feature')} "
                f"({train_rows / num_features:.1f} per feature): the network will probably "
                "memorise the training rows. Use fewer features (e.g. a feature selection "
                "step), more rows, or a small network with dropout or weight decay."
            )
        else:
            message = ""
        if message:
            issues.append(issue("few_rows", "warning", message))
    return issues


# --------------------------------------------------------------------------- #
# Row structure: unusually typical rows, near twins (see brickdnn.data.rows)
# --------------------------------------------------------------------------- #


def group_column_name(table: Table) -> str:
    """``group``, or ``group_2`` … if that column exists already."""
    name, number = "group", 1
    while name in table.kinds:
        number += 1
        name = f"group_{number}"
    return name


def row_structure_issues(
    table: Table, target: str | None, settings: dict[str, Any]
) -> list[dict[str, Any]]:
    """Rows whose every value sits close to its column median (fix: remove atypical rows),
    and rows with a near twin, with the share of validation/test rows that have a twin in
    the training rows (fix: find near-duplicate groups and split by them).

    Both need a clear split in the distributions (:func:`rows.suggest_threshold`), at
    least ``rows.MIN_FEATURES`` feature columns and ``STRUCTURE_MIN_ROWS`` rows, so ordinary
    tables stay quiet. Large tables are checked on a seeded sample of
    ``STRUCTURE_ROWS`` rows (or fewer for wide tables)."""
    if table.n_rows < STRUCTURE_MIN_ROWS:
        return []
    group_by = (settings.get("split") or {}).get("group_by")
    try:
        stats = rowstats.cached_compute(
            table,
            target=target,
            exclude=[group_by] if group_by else [],
            max_rows=STRUCTURE_ROWS,
            with_projection=False,
        )
    except DataError:
        return []
    suggested = stats.suggestions()
    sampled = f"In a sample of {stats.rows:,} rows, " if stats.sampled else ""
    issues: list[dict[str, Any]] = []

    def sentence(text: str) -> str:
        return sampled + text[0].lower() + text[1:] if sampled else text

    if (threshold := suggested["atypical"]) is not None:
        count = int(np.sum(stats.spread < threshold))
        typical = float(np.nanmedian(stats.spread))
        issues.append(
            issue(
                "atypical_rows",
                "warning",
                sentence(
                    f"{plural(count, 'row')} ({fmt_share(count, stats.rows)}) "
                    f"{'is' if count == 1 else 'are'} unusually typical: every value is close "
                    f"to its column median (spread below {fmt_number(threshold)}; ordinary rows "
                    f"{fmt_number(typical)}). Such rows are often filler or synthetic rows "
                    "with made-up labels; look at them in Analysis → Rows."
                ),
                [],
                step_fix(
                    "remove_atypical",
                    {"features": "auto", "columns": [], "side": "low", "threshold": None},
                    "Remove the atypical rows",
                ),
            )
        )
    if (threshold := suggested["twins"]) is not None:
        ratio = stats.nn_ratio()
        twins = ratio < threshold
        count = int(twins.sum())
        if count >= STRUCTURE_TWIN_SHARE * stats.rows:
            closer = 1 / float(np.median(ratio[twins]))
            held_text, held_share = "", 0.0
            if table.split is not None:
                parts = table.split[stats.positions]
                held_out = parts != TRAIN
                ratios = stats.ratios()
                linked = (stats.neighbors >= 0) & (ratios < threshold)
                in_train = np.zeros_like(linked)
                valid = stats.neighbors >= 0
                in_train[valid] = parts[stats.neighbors[valid]] == TRAIN
                held = int(held_out.sum())
                with_twin = int((linked & in_train).any(axis=1)[held_out].sum())
                held_share = with_twin / held if held else 0.0
                held_text = (
                    f" {fmt_share(with_twin, held)} of the validation/test rows have a near "
                    "twin in the training rows"
                )
            base = sentence(
                f"{plural(count, 'row')} ({fmt_share(count, stats.rows)}) "
                f"{'has' if count == 1 else 'have'} a near twin: the nearest row is about "
                f"{closer:.0f}× closer than a typical pair of rows (repeated measurements of the "
                "same person or object?)."
            )
            if group_by:
                if held_share >= STRUCTURE_HELD_SHARE:
                    issues.append(
                        issue(
                            "near_twins",
                            "warning",
                            f"{base} Although the split keeps the groups of '{group_by}' "
                            f"together,{held_text}: the groups do not hold the twins. Check "
                            "the groups in Analysis → Rows.",
                        )
                    )
            elif held_share >= STRUCTURE_HELD_SHARE or table.split is None:
                name = group_column_name(table)
                fix = step_fix(
                    "find_groups",
                    {"features": "auto", "columns": [], "threshold": None, "name": name},
                    "Find near-duplicate groups and split by them",
                )
                fix["settings"] = {"split": {"group_by": name}}
                issues.append(
                    issue(
                        "near_twins",
                        "warning",
                        f"{base}{held_text}, so the validation and test scores are too "
                        "optimistic. Keep near twins in one part: find the groups and split by "
                        "them.",
                        [],
                        fix,
                    )
                )
    return issues


def group_column_issues(table: Table, target: str | None, group_by: str | None) -> list:
    """The column the split keeps groups of is still a feature (fix: drop it)."""
    if not group_by or group_by == target or group_by not in table.kinds:
        return []
    return [
        issue(
            "group_column_feature",
            "warning",
            f"'{group_by}' decides the split (keep groups together) and is still a feature: "
            "group numbers or ids mean nothing for new rows. Drop it after the split.",
            [group_by],
            step_fix("drop_columns", {"columns": [group_by]}, f"Drop {group_by}"),
        )
    ]
