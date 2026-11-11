"""Running a preparation: steps in order, the train/validation/test split and the check
whether the result can be used for training.

Order of work in :func:`run_pipeline`:

1. **Combine** steps (join, append) run first, on whole tables without a split.
2. **Row steps before the split** (``Step.pre_split``: remove atypical rows, find
   near-duplicate groups) run next, on all rows; they use no labels.
3. The **split** is assigned to every row (seeded; stratified by the target for
   classification if asked for and possible; with ``group_by`` whole groups of rows go to one
   part, see :func:`assign_split`). From here on every row keeps its split label, whatever
   later steps do, so steps that learn values learn them from training rows only.
4. All other steps run in order. Disabled steps are passed through. If a step fails, its
   report has ``status="error"`` and the message, all later steps are ``"skipped"``, and the
   result is the table before the failing step, so the preview stays usable.
5. With ``upto`` (a step id), steps after that one are ``"not_run"``: the UI uses it to show
   the data at any point of the pipeline.

Settings (``PrepSettings`` in ``frontend/src/api/types.ts``)::

    {"target": "price" | None,
     "task": "auto" | "classification" | "regression",
     "split": {"val": 0.15, "test": 0.15, "seed": 42, "stratify": True,
               "group_by": None | "patient_id"}}

A step is ``{"id": "s1", "type": "impute", "enabled": True, "params": {...}}``.

What a step learned (``StepResult.fitted``) is made JSON-safe (plain Python numbers, NaN and
infinity become None). Keys starting with an underscore (e.g. the reference rows of a KNN
imputation) are kept for the export (recipe, pandas script) but left out of the step
reports sent to the UI.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .checks import (
    class_issues,
    feature_issues,
    group_column_issues,
    issue,
    numbers_in_text,
    regression_target_issues,
    relation_issues,
    row_issues,
    row_structure_issues,
    sort_issues,
    step_fix,
    string_feature_issues,
    target_variant_issues,
)
from .relations import Relations
from .steps import REGISTRY, StepContext, StepResult, get_step
from .table import MAX_NEW_COLUMNS, NUMERIC_KINDS, TEST, TRAIN, VAL, DataError, Table

logger = logging.getLogger(__name__)

MAX_STEPS = 100
DEFAULT_SPLIT = {"val": 0.15, "test": 0.15, "seed": 42, "stratify": True, "group_by": None}
DEFAULT_SETTINGS: dict[str, Any] = {"target": None, "task": "auto", "split": dict(DEFAULT_SPLIT)}
TASKS = ("auto", "classification", "regression")
# "auto": a numeric target with at most this many distinct whole numbers is a class label.
AUTO_MAX_CLASSES = 20
# Same limit as training (brickdnn.training.datasets.MAX_CLASSES).
MAX_CLASSES = 1000
# Same limit as training (brickdnn.training.datasets.MIN_SAMPLES).
MIN_ROWS = 4
MAX_SHARE = 0.9
MAX_SEED = 2**31 - 1
MAX_STEP_ID = 64
# Group split: warn when a part is this far (share of all rows) from its asked share.
GROUP_SHARE_TOLERANCE = 0.03
MAX_COLUMN_NAME = 200
# Readiness warnings.
IMBALANCE_RATIO = 3.0
MANY_FEATURES = 1000

SourceLoader = Callable[[int], Table]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _count(n: int, singular: str, plural: str | None = None) -> str:
    return f"{n:,} {singular if n == 1 else (plural or singular + 's')}"


def _names(names: list[str], limit: int = 5) -> str:
    """``age, income`` or ``a, b, c, d, e and 3 more``."""
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + f" and {len(names) - limit:,} more"


def json_safe(value: Any) -> Any:
    """``value`` with numpy types turned into Python types, tuples into lists and
    non-finite floats into None, so it can be sent as JSON."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, np.bool_):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if value is pd.NaT or value is pd.NA:
        return None
    return str(value)


def _public(fitted: dict[str, Any] | None) -> dict[str, Any] | None:
    """``fitted`` without the keys meant for the export only (``_reference`` ...)."""
    if fitted is None:
        return None
    public = {k: v for k, v in fitted.items() if not str(k).startswith("_")}
    return public if public or not fitted else None


def class_label(value: Any) -> str:
    """A class name for a numeric label: ``3.0`` -> ``"3"``, ``2.5`` -> ``"2.5"``."""
    number = float(value)
    if number.is_integer() and abs(number) < 1e15:
        return str(int(number))
    return repr(number)


def class_labels(series: pd.Series, kind: str) -> tuple[list[str], np.ndarray]:
    """Class names in class index order and the class index of every row (-1 = missing).

    Numeric labels are sorted numerically (``"0"``, ``"1"``, ``"2.5"``), booleans are
    ``"false"`` < ``"true"`` (as the UI shows them), strings alphabetically (by code point,
    like the CSV reader of the training), dates in time order. This is the order of the
    class indices in exported training datasets.
    """
    if kind == "boolean":
        values = series.to_numpy(dtype=float, na_value=np.nan)
        present = np.isfinite(values)
        uniques, inverse = np.unique(values[present], return_inverse=True)
        codes = np.full(len(values), -1, dtype=np.int64)
        codes[present] = inverse
        return ["true" if v else "false" for v in uniques.tolist()], codes
    if kind in NUMERIC_KINDS:
        values = series.to_numpy(dtype=float, na_value=np.nan)
        present = np.isfinite(values)
        uniques, inverse = np.unique(values[present], return_inverse=True)
        codes = np.full(len(values), -1, dtype=np.int64)
        codes[present] = inverse
        return [class_label(v) for v in uniques.tolist()], codes
    codes, uniques = pd.factorize(series, sort=True)
    label = _date_label if kind == "datetime" else str
    return [label(v) for v in uniques], np.asarray(codes, dtype=np.int64)


def _date_label(value: Any) -> str:
    text = pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")
    return text[:-9] if text.endswith(" 00:00:00") else text


# --------------------------------------------------------------------------- #
# Settings and steps
# --------------------------------------------------------------------------- #


def _share(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataError(f"{label}: enter a share between 0 and {MAX_SHARE:g}, e.g. 0.15.")
    number = float(value)
    if not math.isfinite(number) or number < 0 or number > MAX_SHARE:
        raise DataError(f"{label}: enter a share between 0 and {MAX_SHARE:g}, e.g. 0.15.")
    return number


def clean_settings(data: Any) -> dict[str, Any]:
    """Validate settings and fill in defaults. Raises :class:`DataError`.

    ``split.val`` and ``split.test`` are shares in [0, 0.9] with ``val > 0`` and
    ``val + test <= 0.9``; ``seed`` is an int in [0, 2**31 - 1].
    """
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise DataError("The settings must be an object with target, task and split.")
    target = data.get("target")
    if target is not None:
        if not isinstance(target, str):
            raise DataError("Target: choose a column.")
        target = target.strip() or None
    task = data.get("task")
    if task is None:
        task = "auto"
    if task not in TASKS:
        raise DataError("Task: choose auto, classification or regression.")
    split = data.get("split")
    if split is None:
        split = {}
    if not isinstance(split, dict):
        raise DataError("Split: expected the validation and test shares, a seed and stratify.")
    val = _share(split.get("val", DEFAULT_SPLIT["val"]), "Validation share")
    test = _share(split.get("test", DEFAULT_SPLIT["test"]), "Test share")
    if val <= 0:
        raise DataError(
            "Validation share: must be more than 0; training needs validation data to "
            "measure progress."
        )
    if val + test > MAX_SHARE + 1e-9:
        raise DataError(
            f"Validation and test shares together can be at most {MAX_SHARE:g} "
            f"({MAX_SHARE:.0%}), so that rows remain for training."
        )
    seed = split.get("seed", DEFAULT_SPLIT["seed"])
    if isinstance(seed, float) and seed.is_integer():
        seed = int(seed)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        raise DataError(f"Seed: enter a whole number between 0 and {MAX_SEED:,}.")
    stratify = split.get("stratify", DEFAULT_SPLIT["stratify"])
    if not isinstance(stratify, bool):
        raise DataError("Stratify: expected true or false.")
    group_by = split.get("group_by")
    if group_by is not None:
        if not isinstance(group_by, str) or len(group_by) > MAX_COLUMN_NAME:
            raise DataError("Keep groups together: choose a column.")
        group_by = group_by.strip() or None
    return {
        "target": target,
        "task": task,
        "split": {
            "val": val,
            "test": test,
            "seed": seed,
            "stratify": stratify,
            "group_by": group_by,
        },
    }


def clean_steps(data: Any) -> list[dict[str, Any]]:
    """Validate the structure of a step list (not the step parameters, those depend on the
    data). Raises :class:`DataError`.

    Checks: a list of at most ``MAX_STEPS`` objects with a unique non-empty string ``id``
    (at most 64 characters), a known ``type``, a boolean ``enabled`` (default True) and a
    ``params`` object (default {}); combine steps only at the start of the list, the steps
    that run before the split (``pre_split``) right after them.
    """
    if data is None:
        return []
    if not isinstance(data, list):
        raise DataError("The steps must be a list.")
    if len(data) > MAX_STEPS:
        raise DataError(f"A preparation can have at most {MAX_STEPS} steps ({len(data)} given).")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    after_combine = False
    after_pre_split = False
    for number, step in enumerate(data, start=1):
        where = f"Step {number}"
        if not isinstance(step, dict):
            raise DataError(f"{where}: expected an object with id, type and params.")
        step_id = step.get("id")
        if not isinstance(step_id, str) or not step_id.strip():
            raise DataError(f"{where}: the step needs an id.")
        if len(step_id) > MAX_STEP_ID:
            raise DataError(f"{where}: the id is longer than {MAX_STEP_ID} characters.")
        if step_id in seen:
            raise DataError(f"{where}: the id '{step_id}' is used by two steps.")
        seen.add(step_id)
        step_type = step.get("type")
        cls = REGISTRY.get(step_type) if isinstance(step_type, str) else None
        if cls is None:
            raise DataError(f"{where}: unknown step type '{step_type}'.")
        enabled = step.get("enabled", True)
        if enabled is None:
            enabled = True
        if not isinstance(enabled, bool):
            raise DataError(f"{where} ({cls.label}): 'enabled' must be true or false.")
        params = step.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise DataError(f"{where} ({cls.label}): the settings must be an object.")
        if cls.combine and after_combine:
            raise DataError(
                f"{where} ({cls.label}) combines tables, so it must come before all other "
                "steps. Move it to the top."
            )
        if cls.pre_split and after_pre_split:
            raise DataError(
                f"{where} ({cls.label}) runs before the split, so it must come right after the "
                "steps that combine tables and before all other steps. Move it up."
            )
        after_combine = after_combine or not cls.combine
        after_pre_split = after_pre_split or not (cls.combine or cls.pre_split)
        result.append({"id": step_id, "type": cls.type, "enabled": enabled, "params": params})
    return result


# --------------------------------------------------------------------------- #
# Task and split
# --------------------------------------------------------------------------- #


def resolve_task(table: Table, settings: dict[str, Any]) -> str | None:
    """ "classification", "regression" or None (no target, or the target is missing).

    ``auto``: categorical / boolean / text targets and numeric targets with at most
    ``AUTO_MAX_CLASSES`` distinct whole numbers are classification, other numeric targets
    regression, datetime targets None.
    """
    target = settings.get("target")
    if not target or target not in table.kinds:
        return None
    task = settings.get("task") or "auto"
    if task in ("classification", "regression"):
        return task
    kind = table.kinds[target]
    if kind == "datetime":
        return None
    if kind != "numeric":
        return "classification"
    values = table.df[target].to_numpy(dtype=float, na_value=np.nan)
    values = values[np.isfinite(values)]
    whole = len(values) > 0 and bool(np.all(np.floor(values) == values))
    if whole and len(np.unique(values)) <= AUTO_MAX_CLASSES:
        return "classification"
    return "regression"


def split_sizes(n: int, val: float, test: float) -> tuple[int, int, int]:
    """Training, validation and test rows for ``n`` rows: ``round(n * share)`` with at least
    one row for every non-empty part, like ``brickdnn.training.data.split_sizes`` (but
    without an error for tiny tables: parts that do not fit stay empty)."""
    n_test = max(min(max(1, round(n * test)), n - 2), 0) if test > 0 else 0
    n_val = max(min(max(1, round(n * val)), n - n_test - 1), 0) if val > 0 else 0
    return n - n_val - n_test, n_val, n_test


def _allocate(counts: np.ndarray, total: int, capacity: np.ndarray) -> np.ndarray:
    """Distribute ``total`` rows over groups in proportion to ``counts`` (largest remainder
    method), giving no group more than ``capacity``."""
    alloc = np.zeros(len(counts), dtype=np.int64)
    n = int(counts.sum())
    if total <= 0 or n == 0:
        return alloc
    ideal = counts * (total / n)
    alloc = np.minimum(np.floor(ideal).astype(np.int64), capacity)
    while (rest := total - int(alloc.sum())) > 0:
        free = np.flatnonzero(capacity > alloc)
        if len(free) == 0:
            break
        order = free[np.argsort(-(ideal[free] - alloc[free]), kind="stable")]
        alloc[order[:rest]] += 1
    return alloc


def _target_codes(table: Table, settings: dict[str, Any], task: str | None) -> np.ndarray | None:
    target = settings.get("target")
    if task != "classification" or not target or target not in table.kinds:
        return None
    return class_labels(table.df[target], table.kinds[target])[1]


def assign_split(table: Table, settings: dict[str, Any], task: str | None) -> np.ndarray:
    """int8 split label (0 train, 1 val, 2 test) for every row, seeded with
    ``settings["split"]["seed"]``.

    Sizes follow ``round(n * share)`` with at least one row for every non-empty part (like
    ``brickdnn.training.data.split_sizes``). Stratified per class when ``stratify`` is set
    and the task is classification; rows with a missing target are split at random.

    With ``group_by`` (a column of ``table``) all rows with the same value go to the same
    part, see :func:`group_split`. Raises :class:`DataError` if that column is missing.
    """
    split_settings = {**DEFAULT_SPLIT, **(settings.get("split") or {})}
    n = table.n_rows
    _, n_val, n_test = split_sizes(n, split_settings["val"], split_settings["test"])
    rng = np.random.default_rng(int(split_settings["seed"]))
    codes = _target_codes(table, settings, task) if split_settings["stratify"] else None
    group_by = split_settings.get("group_by")
    if group_by:
        if group_by not in table.kinds:
            raise DataError(
                f"Keep groups together: there is no column '{group_by}' when the split is "
                "assigned (after the combine steps and the steps that run before the split)."
            )
        return group_split(group_codes(table.df[group_by]), n_val, n_test, codes, rng)
    split = np.full(n, TRAIN, dtype=np.int8)
    if codes is None or n == 0:
        order = rng.permutation(n)
        split[order[:n_test]] = TEST
        split[order[n_test : n_test + n_val]] = VAL
        return split

    # Stratified: every class (and the rows without a target, as one more group) gets its
    # share of the validation and test rows; each class keeps at least one training row
    # where possible.
    groups, counts = np.unique(codes, return_counts=True)
    keep_one = np.where(counts > 1, counts - 1, 0)
    test_c = _allocate(counts, n_test, keep_one)
    val_c = _allocate(counts, n_val, keep_one - test_c)
    if test_c.sum() < n_test:
        test_c += _allocate(counts, n_test - int(test_c.sum()), counts - test_c - val_c)
    if val_c.sum() < n_val:
        val_c += _allocate(counts, n_val - int(val_c.sum()), counts - test_c - val_c)
    by_group = np.argsort(codes, kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)])
    for i in range(len(groups)):
        rows = rng.permutation(by_group[starts[i] : starts[i + 1]])
        split[rows[: test_c[i]]] = TEST
        split[rows[test_c[i] : test_c[i] + val_c[i]]] = VAL
    return split


def group_codes(series: pd.Series) -> np.ndarray:
    """A group number per row (0, 1, …): equal values share a number; every row with a
    missing value is a group of its own."""
    codes, uniques = pd.factorize(series, use_na_sentinel=True)
    codes = np.asarray(codes, dtype=np.int64)
    missing = codes < 0
    if missing.any():
        codes[missing] = len(uniques) + np.arange(int(missing.sum()))
    return codes


def _majority(groups: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """The most common class of every group (ties: the smaller class code; -1 = no target)."""
    frame = pd.DataFrame({"g": groups, "c": classes})
    counts = frame.value_counts().reset_index(name="n")
    counts = counts.sort_values(["g", "n", "c"], ascending=[True, False, True], kind="stable")
    first = counts.drop_duplicates("g")
    result = np.full(int(groups.max()) + 1 if len(groups) else 0, -1, dtype=np.int64)
    result[first["g"].to_numpy()] = first["c"].to_numpy()
    return result


def group_split(
    groups: np.ndarray,
    n_val: int,
    n_test: int,
    classes: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """A split that keeps every group (``groups``: a number per row) in one part, with row
    counts as close to ``n_val`` / ``n_test`` as the group sizes allow.

    Groups with several rows are placed first, largest first (equal sizes in random order),
    each into the part where it reduces the distance to the wanted row counts the most;
    single rows then fill the remaining room at random. With ``classes`` (stratified
    classification) this happens per stratum, the most common class of each group, so every
    part gets about the same class shares. Every part that should have rows gets at least
    one group if possible.
    """
    n = len(groups)
    split = np.full(n, TRAIN, dtype=np.int8)
    if n == 0:
        return split
    sizes = np.bincount(groups)
    count = len(sizes)
    strata = _majority(groups, classes) if classes is not None else np.zeros(count, np.int64)
    _, strata = np.unique(strata, return_inverse=True)
    n_strata = int(strata.max()) + 1
    wanted = np.array([n - n_val - n_test, n_val, n_test], dtype=np.float64)
    stratum_rows = np.bincount(strata, weights=sizes, minlength=n_strata)
    target = stratum_rows[:, None] * (wanted / n)[None, :]
    assigned = np.zeros((n_strata, 3))
    part = np.zeros(count, dtype=np.int8)

    order = rng.permutation(count)
    order = order[np.argsort(-sizes[order], kind="stable")]
    multi = order[sizes[order] > 1]
    single = order[sizes[order] == 1]
    totals = np.zeros(3)
    for g in multi.tolist():
        st, size = int(strata[g]), float(sizes[g])
        need = target[st] - assigned[st]
        change = np.abs(need - size) - np.abs(need)
        best = np.flatnonzero(change <= change.min() + 1e-9)
        if len(best) > 1:  # ties: the part that misses the most rows overall
            best = best[np.argsort(-(wanted - totals)[best], kind="stable")]
        p = int(best[0])
        part[g] = p
        assigned[st, p] += size
        totals[p] += size
    for st in range(n_strata):
        members = single[strata[single] == st]
        if len(members) == 0:
            continue
        need = np.maximum(target[st] - assigned[st], 0)
        if need.sum() <= 0:
            need = target[st].copy()
        alloc = _allocate(need, len(members), np.full(3, len(members)))
        bounds = np.cumsum(alloc)
        for p in range(3):
            part[members[bounds[p] - alloc[p] : bounds[p]]] = p
    # Every part that should have rows gets at least one group (the smallest of the part
    # with the most groups), as long as that part keeps one.
    for p in (VAL, TEST, TRAIN):
        if wanted[p] > 0 and not (part == p).any():
            counts = np.bincount(part, minlength=3)
            donor = int(np.argmax(counts))
            if counts[donor] > 1:
                candidates = np.flatnonzero(part == donor)
                part[candidates[np.argmin(sizes[candidates])]] = p
    return part[groups].astype(np.int8)


def split_info(table: Table, settings: dict[str, Any], error: str | None = None) -> dict[str, Any]:
    """``SplitInfo`` (frontend/src/api/types.ts): the columns where the split is assigned and,
    with ``group_by``, how the groups were split (``mixed``: groups in more than one part,
    0 unless rows were added to a group later)."""
    group_by = (settings.get("split") or {}).get("group_by")
    info: dict[str, Any] = {
        "group_by": group_by,
        "columns": table.column_info(),
        "error": error,
        "groups": None,
    }
    if not group_by or error or group_by not in table.kinds or table.split is None:
        return info
    codes = group_codes(table.df[group_by])
    sizes = np.bincount(codes)
    parts = np.zeros((len(sizes), 3), dtype=np.int64)
    np.add.at(parts, (codes, table.split.astype(np.int64)), 1)
    info["groups"] = {
        "count": len(sizes),
        "multi": int((sizes > 1).sum()),
        "largest": int(sizes.max()) if len(sizes) else 0,
        "missing": int(table.df[group_by].isna().sum()),
        "per_part": [int((parts[:, p] > 0).sum()) for p in range(3)],
        "mixed": int(((parts > 0).sum(axis=1) > 1).sum()),
    }
    return info


def _group_split_warnings(table: Table, settings: dict[str, Any], task: str | None) -> list[str]:
    split_settings = settings["split"]
    assert table.split is not None
    counts = np.bincount(table.split.astype(np.int64), minlength=3)[:3] / table.n_rows
    val, test = split_settings["val"], split_settings["test"]
    asked = np.array([1 - val - test, val, test])
    warnings = []
    if np.abs(counts - asked).max() > GROUP_SHARE_TOLERANCE:
        shown = " / ".join(f"{v:.0%}" for v in counts)
        wanted = " / ".join(f"{v:.0%}" for v in asked)
        warnings.append(
            f"Split by '{split_settings['group_by']}': the groups are large, so training / "
            f"validation / test have {shown} of the rows instead of {wanted}."
        )
    if split_settings["stratify"] and task == "classification":
        warnings.append(
            "Stratified by the most common class of every group: the class shares of the "
            "parts are close to each other, not exactly equal."
        )
    return warnings


def _split_warnings(table: Table, settings: dict[str, Any], task: str | None) -> list[str]:
    split_settings = settings["split"]
    if split_settings.get("group_by") and table.split is not None and table.n_rows:
        return _group_split_warnings(table, settings, task)
    if not split_settings["stratify"] or task != "classification":
        return []
    codes = _target_codes(table, settings, task)
    if codes is None:
        return []
    names, _ = class_labels(table.df[settings["target"]], table.kinds[settings["target"]])
    counts = np.bincount(codes[codes >= 0], minlength=len(names))
    need = 1 + (split_settings["val"] > 0) + (split_settings["test"] > 0)
    small = [names[i] for i in np.flatnonzero(counts < need)]
    if not small:
        return []
    return [
        f"Stratified split: {_count(len(small), 'class', 'classes')} "
        f"({_names(small)}) {'has' if len(small) == 1 else 'have'} fewer than {need} rows, "
        "so not every part contains "
        f"{'it' if len(small) == 1 else 'them'}."
    ]


# --------------------------------------------------------------------------- #
# Running the steps
# --------------------------------------------------------------------------- #


@dataclass
class PipelineResult:
    # The data after the last step that ran (or after `upto`).
    table: Table
    # The input of the non-combine steps: after combining, with the split assigned.
    before: Table
    # One report per step, in order, see ``StepReport`` in frontend/src/api/types.ts.
    reports: list[dict[str, Any]]
    task: str | None
    # Step id -> what the step learned (None for steps that learn nothing / did not run).
    fitted: dict[str, dict[str, Any] | None]
    failed: bool
    warnings: list[str] = field(default_factory=list)
    # Step id -> the cleaned parameters of every step that ran (for the pandas script).
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    # The settings after clean_settings.
    settings: dict[str, Any] = field(default_factory=dict)
    # Where and how the split was assigned (SplitInfo, see split_info).
    split: dict[str, Any] = field(default_factory=dict)


def make_context(
    table: Table,
    settings: dict[str, Any],
    task: str | None,
    *,
    load_source: SourceLoader | None = None,
    source_name: Callable[[int], str] | None = None,
    source_info: Callable[[int], dict[str, Any]] | None = None,
    source_id: int | None = None,
) -> StepContext:
    """The :class:`StepContext` for the steps of a pipeline run.

    ``source_id`` is the id of the main data source (used by ``append`` to label rows).
    """
    split = settings.get("split") or {}
    return StepContext(
        target=settings.get("target") or None,
        task=task,
        seed=int(split.get("seed", DEFAULT_SPLIT["seed"])),
        load_source=load_source,
        source_name=source_name,
        source_info=source_info,
        source_id=source_id,
    )


def _report(
    step: dict[str, Any],
    status: str,
    before: Table,
    after: Table | None = None,
    *,
    result: StepResult | None = None,
    fitted: dict[str, Any] | None = None,
    error: str | None = None,
    duration: float = 0.0,
) -> dict[str, Any]:
    after = after if after is not None else before
    before_names, after_names = set(before.columns), set(after.columns)
    details = []
    if result is not None:
        for item in result.details or []:
            if isinstance(item, dict) and "text" in item:
                detail = {"text": str(item["text"])}
                if item.get("column") is not None:
                    detail["column"] = str(item["column"])
                details.append(detail)
    return {
        "id": step["id"],
        "type": step["type"],
        "status": status,
        "summary": str(result.summary) if result is not None and status == "ok" else "",
        "error": error,
        "warnings": [str(w) for w in (result.warnings or [])] if result is not None else [],
        "details": details,
        "rows_before": before.n_rows,
        "rows_after": after.n_rows,
        "columns": after.column_info(),
        "added": [c for c in after.columns if c not in before_names],
        "removed": [c for c in before.columns if c not in after_names],
        "fitted": _public(fitted),
        "duration_ms": round(duration * 1000, 1),
    }


def _short(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    text = f"{type(exc).__name__}: {text}" if text else type(exc).__name__
    return text if len(text) <= 160 else text[:157] + "…"


def _run_step(
    step: dict[str, Any], table: Table, ctx: StepContext
) -> tuple[Table, dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """Run one enabled step. Returns ``(table after, report, fitted, cleaned params)``; on an
    error the table is the input table and fitted/params are None."""
    start = time.perf_counter()
    try:
        cls = get_step(step["type"])
        if cls.combine and table.split is not None:
            raise DataError(
                f"'{cls.label}' combines tables, so it must come before all other steps."
            )
        if cls.pre_split and table.split is not None:
            raise DataError(
                f"'{cls.label}' runs before the split, so it must come before all steps "
                "except the ones that combine tables."
            )
        instance = cls(step.get("params") or {})
        params = instance.clean(table, ctx)
        result = instance.run(table, params, ctx)
        if not isinstance(result, StepResult) or not isinstance(result.table, Table):
            raise TypeError("the step returned no table")
        out = result.table
        if table.split is not None and (out.split is None or len(out.split) != out.n_rows):
            raise RuntimeError("the step lost the train/validation/test split of the rows")
        if len(out.columns) - len(table.columns) > MAX_NEW_COLUMNS:
            raise DataError(
                f"This step adds {len(out.columns) - len(table.columns):,} columns; at most "
                f"{MAX_NEW_COLUMNS:,} new columns per step are supported. Use fewer columns or "
                "categories (e.g. group rare categories first)."
            )
        fitted = json_safe(result.fitted) if result.fitted is not None else None
    except DataError as exc:
        report = _report(step, "error", table, error=str(exc), duration=time.perf_counter() - start)
        return table, report, None, None
    except Exception as exc:  # a bug in a step must not break the whole preview
        logger.exception("Data preparation step %s (%s) failed", step.get("id"), step.get("type"))
        report = _report(
            step,
            "error",
            table,
            error=f"Unexpected error in this step: {_short(exc)}",
            duration=time.perf_counter() - start,
        )
        return table, report, None, None
    report = _report(
        step, "ok", table, out, result=result, fitted=fitted, duration=time.perf_counter() - start
    )
    return out, report, fitted, json_safe(params)


def run_pipeline(
    source: Table,
    steps: list[dict[str, Any]],
    settings: dict[str, Any],
    *,
    load_source: SourceLoader | None = None,
    source_name: Callable[[int], str] | None = None,
    source_info: Callable[[int], dict[str, Any]] | None = None,
    upto: str | None = None,
    source_id: int | None = None,
) -> PipelineResult:
    """Run ``steps`` (already checked with :func:`clean_steps`) on ``source`` (an imported
    table without split), see the module docstring. Never raises for problems of a single
    step; raises :class:`DataError` only for invalid settings.

    ``source_id`` (optional) is the id of ``source``; ``append`` uses its name to label rows.
    """
    settings = clean_settings(settings)
    sources = {"load_source": load_source, "source_name": source_name, "source_info": source_info}
    table = source.with_split(None) if source.split is not None else source
    ctx = make_context(table, settings, None, source_id=source_id, **sources)
    task: str | None = None
    before: Table | None = None
    reports: list[dict[str, Any]] = []
    fitted: dict[str, dict[str, Any] | None] = {}
    params: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    state = "run"  # "run", "error" (later steps are skipped) or "upto" (later steps not run)
    if upto is not None and not any(s.get("id") == upto for s in steps):
        upto = None

    split_error: str | None = None

    def start_split() -> None:
        nonlocal table, before, task, ctx, split_error
        task = resolve_task(table, settings)
        try:
            split = assign_split(table, settings, task)
            used = settings
        except DataError as exc:
            # The group column is missing: a random split keeps the preview usable.
            used = {**settings, "split": {**settings["split"], "group_by": None}}
            split = assign_split(table, used, task)
            if state == "run":
                split_error = str(exc)
            else:
                warnings.append(
                    "This preview stops before the group column exists, so it shows a random "
                    "split; the split by groups is assigned once the steps before it have run."
                )
        table = table.with_split(split)
        before = table
        ctx = make_context(table, settings, task, source_id=source_id, **sources)
        warnings.extend(_split_warnings(table, used, task))

    for step in steps:
        cls = REGISTRY.get(step.get("type"))
        early = cls is not None and (cls.combine or cls.pre_split)
        if before is None and not early:
            start_split()
        fitted[step["id"]] = None
        if state == "error":
            reports.append(_report(step, "skipped", table))
        elif state == "upto":
            reports.append(_report(step, "not_run", table))
        elif not step.get("enabled", True):
            reports.append(_report(step, "disabled", table))
        else:
            table, report, step_fitted, step_params = _run_step(step, table, ctx)
            reports.append(report)
            if report["status"] == "error":
                state = "error"
            else:
                fitted[step["id"]] = step_fitted
                params[step["id"]] = step_params or {}
        if state == "run" and step["id"] == upto:
            state = "upto"
    if before is None:
        start_split()
    assert before is not None
    if task is None:
        # The target may be created by a step (formula, join); decide on the result.
        task = resolve_task(table, settings)
    return PipelineResult(
        table=table,
        before=before,
        reports=reports,
        task=task,
        fitted=fitted,
        failed=state == "error",
        warnings=warnings,
        params=params,
        settings=settings,
        split=split_info(before, settings, split_error),
    )


# --------------------------------------------------------------------------- #
# Training readiness
# --------------------------------------------------------------------------- #


def _columns_text(columns: list[str]) -> str:
    return columns[0] if len(columns) == 1 else _count(len(columns), "column")


def _mostly_numbers(series: pd.Series) -> dict[str, Any] | None:
    """How a string target can be read as numbers (see ``checks.numbers_in_text``)."""
    present = series.dropna()
    if len(present) == 0:
        return None
    values = present.astype(str).str.strip()
    parsed = pd.to_numeric(values, errors="coerce")
    if float(parsed.notna().mean()) >= 0.9:
        return {"decimal_comma": False, "strip_units": False}
    return numbers_in_text(values)


def readiness(
    table: Table,
    settings: dict[str, Any],
    task: str | None,
    *,
    checks: bool = True,
    split: dict[str, Any] | None = None,
    relations: Relations | None = None,
) -> dict[str, Any]:
    """Can ``table`` be exported as a training dataset? See ``Readiness`` in
    ``frontend/src/api/types.ts``.

    Errors: no target chosen / target missing / missing target values (fix: drop_missing on
    the target), fewer than 2 or more than ``MAX_CLASSES`` classes, a non-numeric regression
    target, no feature columns, categorical features (fix: encode one-hot; numbers or dates
    stored as text: cast; ID-like values: drop; spellings that differ only in case: replace
    values; many categories: frequency or target mean encoding), text features (fix:
    drop_columns), datetime features (fix: datetime_features), missing feature values (fix:
    impute), infinite values, an empty training or validation part.
    Warnings and notes: constant features (fix: drop_columns), more than 1,000 feature
    columns, strong class imbalance, and with ``checks`` the sanity checks of
    :mod:`brickdnn.data.checks` (target leakage, duplicates, rare classes …). Every issue is
    ``{"code", "severity", "column", "columns", "message", "fix"}`` with ``fix`` ``{"type":
    step type, "params": {...}, "label": "Impute 3 columns (median)"}`` or None. ``checks``
    False leaves out the sanity checks (only the errors decide whether data can be exported).
    ``split``: the ``PipelineResult.split`` of the run (``SplitInfo``); its ``error`` (the
    group column is missing where the split is assigned) is an error here.
    ``relations``: the column relations (:mod:`brickdnn.data.relations`) if already computed
    for the preview, for the check of columns without a strong partner.
    """
    target = settings.get("target") or None
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    df = table.df
    split_counts = table.split_counts()
    num_classes: int | None = None
    class_names: list[str] | None = None
    class_codes: np.ndarray | None = None
    has_target = bool(target) and target in table.kinds
    stratify = bool((settings.get("split") or {}).get("stratify", DEFAULT_SPLIT["stratify"]))

    # ---- the target
    if not target:
        errors.append(
            issue(
                "no_target",
                "error",
                "Choose the target column: the column the network should predict.",
            )
        )
    elif not has_target:
        errors.append(
            issue(
                "target_not_found",
                "error",
                f"The target column '{target}' does not exist (any more). A step may have "
                "renamed or dropped it; choose the target again.",
                [target],
            )
        )
    else:
        kind = table.kinds[target]
        missing = int(df[target].isna().sum())
        if missing:
            errors.append(
                issue(
                    "target_missing",
                    "error",
                    f"{_count(missing, 'row has', 'rows have')} no target value ({target}).",
                    [target],
                    step_fix(
                        "drop_missing",
                        {"columns": [target], "how": "any"},
                        f"Drop {_count(missing, 'row')} without a target",
                    ),
                )
            )
        if kind == "datetime" and task is None:
            errors.append(
                issue(
                    "target_date",
                    "error",
                    f"The target '{target}' is a date. Choose another target, or turn it into "
                    "numbers or classes first.",
                    [target],
                )
            )
        elif task == "classification":
            names, codes = class_labels(df[target], kind)
            num_classes = len(names)
            if num_classes > MAX_CLASSES:
                errors.append(
                    issue(
                        "too_many_classes",
                        "error",
                        f"The target '{target}' has {num_classes:,} different values; "
                        f"classification supports at most {MAX_CLASSES:,} classes. "
                        "Choose regression for numbers.",
                        [target],
                    )
                )
                num_classes = None
            else:
                class_names = names
                if num_classes < 2:
                    only = f" ('{names[0]}')" if names else ""
                    errors.append(
                        issue(
                            "one_class",
                            "error",
                            f"The target '{target}' has only {num_classes} class{only}. "
                            "Classification needs at least 2.",
                            [target],
                        )
                    )
                else:
                    class_codes = codes
                    train_codes = codes[table.train_mask() & (codes >= 0)]
                    counts = np.bincount(train_codes, minlength=num_classes)
                    if checks:
                        warnings.extend(class_issues(table, target, names, codes, stratify))
                        if kind in ("categorical", "text"):
                            warnings.extend(target_variant_issues(table, target))
                    else:
                        rare = [names[i] for i in np.flatnonzero(counts < 2)]
                        if rare:
                            warnings.append(
                                issue(
                                    "rare_classes",
                                    "warning",
                                    f"{_count(len(rare), 'class has', 'classes have')} fewer "
                                    f"than 2 training rows ({_names(rare)}).",
                                    [target],
                                )
                            )
                    present = counts[counts > 0]
                    if len(present) >= 2:
                        ratio = float(present.max() / present.min())
                        if ratio > IMBALANCE_RATIO:
                            warnings.append(
                                issue(
                                    "imbalanced",
                                    "warning",
                                    f"The classes are imbalanced ({ratio:.1f} : 1 in the "
                                    "training rows).",
                                    [target],
                                    step_fix(
                                        "balance_classes",
                                        {"method": "oversample"},
                                        "Balance the classes (oversample)",
                                    ),
                                )
                            )
        elif task == "regression" and kind not in NUMERIC_KINDS:
            fix = None
            found = _mostly_numbers(df[target]) if kind in ("categorical", "text") else None
            if found is not None:
                params: dict[str, Any] = {"columns": [target], "to": "numeric"}
                params.update({k: True for k in ("decimal_comma", "strip_units") if found[k]})
                fix = step_fix("cast", params, f"Change {target} to numbers")
            errors.append(
                issue(
                    "target_not_numeric",
                    "error",
                    f"The target '{target}' is {kind}, not numeric, so it cannot be used for "
                    "regression. Choose classification or another target.",
                    [target],
                    fix,
                )
            )
        elif task == "regression" and checks:
            warnings.extend(regression_target_issues(table, target))
        if kind in NUMERIC_KINDS:
            infinite = int(np.isinf(df[target].to_numpy(dtype=float, na_value=np.nan)).sum())
            if infinite:
                errors.append(
                    issue(
                        "infinite_values",
                        "error",
                        f"The target '{target}' has {_count(infinite, 'infinite value')}.",
                        [target],
                    )
                )

    # ---- the features
    features = [c for c in table.columns if c != target]
    by_kind: dict[str, list[str]] = {}
    for name in features:
        by_kind.setdefault(table.kinds[name], []).append(name)
    numeric = [c for c in features if table.kinds[c] in NUMERIC_KINDS]
    if not features:
        errors.append(
            issue("no_features", "error", "There are no feature columns besides the target.")
        )
    categorical = by_kind.get("categorical", [])
    text = by_kind.get("text", [])
    if checks and (categorical or text):
        found_issues, categorical, text = string_feature_issues(
            table, categorical, text, task if has_target else None
        )
        errors.extend(found_issues)
    if categorical:
        errors.append(
            issue(
                "categorical_features",
                "error",
                f"{_count(len(categorical), 'feature column is', 'feature columns are')} "
                f"categorical ({_names(categorical)}). Networks need numbers.",
                categorical,
                step_fix(
                    "encode",
                    {"columns": categorical, "method": "onehot"},
                    f"One-hot encode {_columns_text(categorical)}",
                ),
            )
        )
    if text:
        message = (
            f"{text[0]} is free text and cannot be used as a feature."
            if len(text) == 1
            else f"{len(text):,} feature columns are free text ({_names(text)}) and cannot be "
            "used as features."
        )
        errors.append(
            issue(
                "text_features",
                "error",
                message,
                text,
                step_fix("drop_columns", {"columns": text}, f"Drop {_columns_text(text)}"),
            )
        )
    if dates := by_kind.get("datetime"):
        message = (
            f"{dates[0]} is a date. Split it into numbers first."
            if len(dates) == 1
            else f"{len(dates):,} feature columns are dates ({_names(dates)}). Split them into "
            "numbers first."
        )
        what = dates[0] if len(dates) == 1 else f"{len(dates):,} date columns"
        errors.append(
            issue(
                "date_features",
                "error",
                message,
                dates,
                step_fix(
                    "datetime_features",
                    {
                        "columns": dates,
                        "parts": ["year", "month", "weekday"],
                        "drop_original": True,
                    },
                    f"Split {what} into year, month, weekday",
                ),
            )
        )
    constant: list[str] = []
    if numeric:
        values = df[numeric].to_numpy(dtype=float, na_value=np.nan)
        nan_counts = np.isnan(values).sum(axis=0)
        with_missing = [c for c, k in zip(numeric, nan_counts, strict=True) if k]
        if with_missing:
            errors.append(
                issue(
                    "missing_values",
                    "error",
                    f"{_count(len(with_missing), 'feature column has', 'feature columns have')} "
                    f"missing values ({_names(with_missing)}).",
                    with_missing,
                    step_fix(
                        "impute",
                        {"columns": with_missing, "strategy": "median"},
                        f"Impute {_columns_text(with_missing)} (median)",
                    ),
                )
            )
        inf_counts = np.isinf(values).sum(axis=0)
        with_inf = [c for c, k in zip(numeric, inf_counts, strict=True) if k]
        if with_inf:
            errors.append(
                issue(
                    "infinite_values",
                    "error",
                    f"{_count(len(with_inf), 'feature column has', 'feature columns have')} "
                    f"infinite values ({_names(with_inf)}).",
                    with_inf,
                )
            )
        finite = np.where(np.isfinite(values), values, np.nan)
        with np.errstate(all="ignore"):
            constant = [
                c
                for i, c in enumerate(numeric)
                if not np.any(np.isfinite(finite[:, i]))
                or np.nanmin(finite[:, i]) == np.nanmax(finite[:, i])
            ]
        if constant:
            warnings.append(
                issue(
                    "constant",
                    "warning",
                    f"{_count(len(constant), 'feature column is', 'feature columns are')} "
                    f"constant ({_names(constant)}) and carry no information.",
                    constant,
                    step_fix(
                        "drop_columns", {"columns": constant}, f"Drop {_columns_text(constant)}"
                    ),
                )
            )
    if features and all(_is_number_text(c) for c in features):
        mapping = {c: f"x{c}" for c in features}
        errors.append(
            issue(
                "numeric_names",
                "error",
                f"All feature names are numbers ({_names(features)}), so the column header "
                "of the training file cannot be recognised. Rename at least one column.",
                features,
                step_fix("rename_columns", {"mapping": mapping}, "Rename the columns (x1, x2 …)"),
            )
        )
    if len(features) > MANY_FEATURES:
        warnings.append(
            issue(
                "many_features",
                "warning",
                f"The data has {len(features):,} feature columns; training will be slow and "
                "needs much memory. Consider feature selection.",
            )
        )
    if checks and features and table.n_rows >= MIN_ROWS:
        with np.errstate(all="ignore"):
            warnings.extend(
                feature_issues(
                    table, target if has_target else None, task, class_codes, constant=constant
                )
            )
            warnings.extend(row_issues(table, target if has_target else None, task, len(features)))
            warnings.extend(row_structure_issues(table, target if has_target else None, settings))
            warnings.extend(relation_issues(table, target if has_target else None, task, relations))

    # ---- rows and split
    group_by = (settings.get("split") or {}).get("group_by")
    if split and split.get("error"):
        errors.append(
            issue(
                "split_group_missing",
                "error",
                f"{split['error']} Choose another column in Target & split, or add the step "
                "that creates it (e.g. Find near-duplicate groups).",
                [group_by] if group_by else [],
            )
        )
    elif checks:
        warnings.extend(group_column_issues(table, target, group_by))
    if table.n_rows < MIN_ROWS:
        errors.append(
            issue(
                "too_few_rows",
                "error",
                f"The data has {_count(table.n_rows, 'row')}; at least {MIN_ROWS} are needed.",
            )
        )
    if split_counts is None:
        errors.append(issue("no_split", "error", "The rows have no train/validation/test split."))
    elif table.n_rows:
        if split_counts[TRAIN] == 0:
            errors.append(issue("no_training_rows", "error", "There are no training rows left."))
        if split_counts[VAL] == 0:
            errors.append(
                issue(
                    "no_validation_rows",
                    "error",
                    "There are no validation rows left. Increase the validation share or keep "
                    "more rows.",
                )
            )

    return {
        "ready": not errors,
        "task": task if has_target else None,
        "target": target,
        "rows": table.n_rows,
        "num_features": len(features),
        "num_classes": num_classes,
        "class_names": class_names,
        "split_counts": split_counts,
        "issues": sort_issues(errors + warnings),
    }


def _is_number_text(name: str) -> bool:
    """Whether the training CSV reader would take ``name`` for a number (see
    ``brickdnn.training.datasets.parse_number``)."""
    try:
        return math.isfinite(float(name.strip()))
    except ValueError:
        return False
