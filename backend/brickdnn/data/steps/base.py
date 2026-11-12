"""What every preparation step has in common: parameters, context, result and the registry.

A step is a subclass of :class:`Step` registered with :func:`register`. Its class attributes
describe it for the UI (``GET /api/data/steps/`` is :func:`catalog`), :meth:`Step.run`
applies it to a table and :meth:`Step.to_pandas` writes the same operation as pandas code
for the exported script, with everything the step learned (``fitted``) written in as
constants.

Rules for every step:

* Never change the input table; return a new one (``Table.with_column``, ``Table.take`` ...).
* Learn values (means, bounds, categories ...) from the **training rows only**
  (``table.train_mask()``), then apply them to all rows. Mark such steps ``fitted = True``.
* Return a short, concrete ``summary`` ("Filled 132 missing values in 2 columns with the
  median.") and raise :class:`~brickdnn.data.table.DataError` with a helpful message for
  settings that cannot work.
* Keep ``fitted`` JSON-serialisable (plain floats, strings, lists, dicts; no numpy types).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..table import KINDS, DataError, Table

GROUPS = (
    ("combine", "Combine data"),
    ("columns", "Columns"),
    ("rows", "Rows"),
    ("missing", "Missing values"),
    ("outliers", "Outliers & clipping"),
    ("transform", "Transform numbers"),
    ("features", "Feature engineering"),
    ("encode", "Encode categories"),
    ("datetime", "Dates & times"),
    ("target", "Target & balance"),
    ("selection", "Feature selection"),
)
GROUP_IDS = tuple(g for g, _ in GROUPS)

PARAM_KINDS = (
    "column",  # one column name
    "columns",  # list of column names; [] usually means "all suitable columns"
    "source",  # id (int) of another imported data source
    "source_columns",  # list of column names of the source chosen in `source_param`
    "select",  # one value of `options`
    "multiselect",  # list of values of `options`
    "number",  # float
    "int",  # whole number
    "bool",
    "string",
    "value",  # a single cell value: string, number, bool or null
    "list",  # list of cell values (strings or numbers)
    "mapping",  # {"old value": new value, ...}; keys are strings
)


@dataclass(frozen=True)
class Param:
    """One setting of a step, as shown in the step's form."""

    name: str
    kind: str
    label: str
    default: Any = None
    help: str | None = None
    # select / multiselect: (value, label) pairs
    options: tuple[tuple[str, str], ...] | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    # column / columns: only offer columns of these kinds (None: all kinds)
    column_kinds: tuple[str, ...] | None = None
    # May be left empty / null.
    optional: bool = False
    # columns: what an empty list means, e.g. "All numeric columns"
    empty_means: str | None = None
    # Only shown when another param has one of these values: {"param": "method", "values": [...]}
    visible_if: dict[str, Any] | None = None
    # source_columns: name of the "source" param whose columns are offered
    source_param: str | None = None
    # mapping: whether the keys are column names ("column") or cell values ("value")
    key_kind: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in PARAM_KINDS:
            raise ValueError(f"Unknown param kind '{self.kind}'.")
        if self.column_kinds and not set(self.column_kinds) <= set(KINDS):
            raise ValueError(f"Unknown column kinds in {self.column_kinds}.")
        if self.key_kind not in (None, "column", "value"):
            raise ValueError(f"Unknown key kind '{self.key_kind}'.")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "label": self.label,
            "default": self.default,
            "help": self.help,
            "optional": self.optional,
        }
        if self.options is not None:
            data["options"] = [{"value": v, "label": label} for v, label in self.options]
        for key in ("min", "max", "step", "empty_means", "visible_if", "source_param", "key_kind"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.column_kinds is not None:
            data["column_kinds"] = list(self.column_kinds)
        return data

    def visible(self, params: dict[str, Any]) -> bool:
        if not self.visible_if:
            return True
        return params.get(self.visible_if["param"]) in self.visible_if["values"]


@dataclass
class StepContext:
    """What a step may need besides its table and parameters."""

    # Target column and resolved task ("classification" | "regression"), if chosen.
    target: str | None = None
    task: str | None = None
    # Seed for anything random (sampling, KNN ties ...), from the split settings.
    seed: int = 42
    # Loads another imported data source by id (combine steps). Raises DataError.
    load_source: Callable[[int], Table] | None = None
    # Display name of a data source by id (summaries, pandas export).
    source_name: Callable[[int], str] | None = None
    # How to read a data source again (pandas export of combine steps): see
    # ``export.SourceInfo``.
    source_info: Callable[[int], dict[str, Any]] | None = None
    # Id of the main data source of the preparation, if known (append labels its rows).
    source_id: int | None = None

    def source(self, source_id: int) -> Table:
        if self.load_source is None:
            raise DataError("Other data sources are not available here.")
        return self.load_source(source_id)

    def name_of(self, source_id: int) -> str:
        if self.source_name is None:
            return f"source {source_id}"
        return self.source_name(source_id)


@dataclass
class StepResult:
    table: Table
    # One sentence for the step card, e.g. "Clipped 57 values (1.2 %) in 2 columns."
    summary: str
    # Optional per-column facts for the step card: [{"column": "age", "text": "median 34"}].
    details: list[dict[str, Any]] = field(default_factory=list)
    # What the step learned from the training rows (JSON-serialisable), None if nothing.
    fitted: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)


class Step:
    """Base class of all steps. Subclasses set the class attributes and implement ``run``."""

    type: ClassVar[str]
    label: ClassVar[str]
    group: ClassVar[str]
    description: ClassVar[str]
    params: ClassVar[tuple[Param, ...]] = ()
    # Learns values from the training rows.
    fitted: ClassVar[bool] = False
    # Combines data sources (join, append): must come before all other steps, runs before
    # the split is assigned.
    combine: ClassVar[bool] = False
    # Only removes or adds training rows (e.g. class balancing): skipped in the pandas
    # script that prepares new data.
    training_only: ClassVar[bool] = False
    # Works on the rows as a whole before the split is assigned (remove atypical rows, find
    # near-duplicate groups): comes after the combine steps and before all other steps, so a
    # column it adds can decide the split. Learns nothing from labels; sees all rows.
    pre_split: ClassVar[bool] = False

    def __init__(self, params: dict[str, Any]):
        self.raw_params = params

    # ------------------------------------------------------------------ catalog

    @classmethod
    def spec(cls) -> dict[str, Any]:
        return {
            "type": cls.type,
            "label": cls.label,
            "group": cls.group,
            "description": cls.description,
            "fitted": cls.fitted,
            "combine": cls.combine,
            "training_only": cls.training_only,
            "pre_split": cls.pre_split,
            "params": [p.to_dict() for p in cls.params],
        }

    @classmethod
    def defaults(cls) -> dict[str, Any]:
        return {p.name: _copy_default(p.default) for p in cls.params}

    # ------------------------------------------------------------------ parameters

    def clean(self, table: Table, ctx: StepContext) -> dict[str, Any]:
        """Fill in defaults and check every parameter against the table.

        Returns the cleaned parameters; raises :class:`DataError` naming the bad field.
        Subclasses may extend this (call ``super().clean``) for checks across parameters.
        """
        raw = self.raw_params if isinstance(self.raw_params, dict) else {}
        unknown = set(raw) - {p.name for p in self.params}
        if unknown:
            raise DataError(f"Unknown setting '{sorted(unknown)[0]}' for '{self.label}'.")
        result: dict[str, Any] = {}
        for param in self.params:
            value = raw.get(param.name, _copy_default(param.default))
            result[param.name] = value
        for param in self.params:
            if not param.visible(result):
                result[param.name] = _copy_default(param.default)
                continue
            result[param.name] = clean_value(param, result[param.name], table)
        return result

    # ------------------------------------------------------------------ behaviour

    def run(self, table: Table, params: dict[str, Any], ctx: StepContext) -> StepResult:
        raise NotImplementedError

    def to_pandas(
        self, params: dict[str, Any], fitted: dict[str, Any] | None, ctx: StepContext
    ) -> list[str]:
        """Lines of pandas code that do the same to a DataFrame called ``df``.

        Use the fitted values as constants, so the script prepares new data exactly like
        the training data. Return comment lines (``# ...``) for steps that do not apply to
        new data.
        """
        raise NotImplementedError


REGISTRY: dict[str, type[Step]] = {}


def register(cls: type[Step]) -> type[Step]:
    """Class decorator: add a step type to the catalog."""
    if cls.group not in GROUP_IDS:
        raise ValueError(f"Unknown step group '{cls.group}' of {cls.__name__}.")
    if cls.type in REGISTRY:
        raise ValueError(f"Step type '{cls.type}' is registered twice.")
    REGISTRY[cls.type] = cls
    return cls


def get_step(step_type: Any) -> type[Step]:
    try:
        return REGISTRY[step_type]
    except (KeyError, TypeError):
        raise DataError(f"Unknown step type '{step_type}'.") from None


def catalog() -> dict[str, Any]:
    """Everything the UI needs to build the step menu and forms."""
    order = {g: i for i, g in enumerate(GROUP_IDS)}
    steps = sorted(REGISTRY.values(), key=lambda s: order[s.group])
    return {
        "groups": [{"id": g, "label": label} for g, label in GROUPS],
        "steps": [s.spec() for s in steps],
    }


# --------------------------------------------------------------------------- #
# Generic parameter checks
# --------------------------------------------------------------------------- #


def _copy_default(value: Any) -> Any:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        try:
            number = float(value.strip().replace(",", "."))
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _check_column(param: Param, name: Any, table: Table) -> str:
    if not isinstance(name, str) or not name:
        raise DataError(f"{param.label}: choose a column.")
    if name not in table.kinds:
        raise DataError(f"{param.label}: there is no column '{name}' (any more).")
    kind = table.kinds[name]
    if param.column_kinds and kind not in param.column_kinds:
        allowed = " or ".join(param.column_kinds)
        raise DataError(f"{param.label}: '{name}' is a {kind} column; {allowed} columns needed.")
    return name


def valid_text(value: Any) -> bool:
    """False if ``value`` (or a string inside a list/dict) is not valid Unicode, e.g. a lone
    surrogate like ``"\\ud800"`` that JSON allows but nothing can store or send back."""
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            return False
        return True
    if isinstance(value, (list, tuple)):
        return all(valid_text(v) for v in value)
    if isinstance(value, dict):
        return all(valid_text(k) and valid_text(v) for k, v in value.items())
    return True


def clean_value(param: Param, value: Any, table: Table) -> Any:
    """Check and convert one parameter value. Raises :class:`DataError`."""
    label = param.label
    if not valid_text(value):
        raise DataError(f"{label}: contains characters that are not valid text.")
    empty = value is None or value == "" or value == []
    if empty and param.optional:
        return [] if param.kind in ("columns", "source_columns", "multiselect", "list") else None

    kind = param.kind
    if kind == "column":
        return _check_column(param, value, table)
    if kind == "columns":
        if value is None:
            return []
        if not isinstance(value, list):
            raise DataError(f"{label}: expected a list of columns.")
        names = [_check_column(param, name, table) for name in value]
        return list(dict.fromkeys(names))
    if kind == "source":
        if isinstance(value, str) and value.isdigit():
            value = int(value)
        if not isinstance(value, int) or isinstance(value, bool):
            raise DataError(f"{label}: choose a data source.")
        return value
    if kind in ("source_columns", "list"):
        if value is None:
            return []
        if not isinstance(value, list) or not all(
            isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in value
        ):
            raise DataError(f"{label}: expected a list of values.")
        return list(value)
    if kind == "select":
        allowed = [v for v, _ in param.options or ()]
        if value not in allowed:
            raise DataError(f"{label}: choose one of {', '.join(map(str, allowed))}.")
        return value
    if kind == "multiselect":
        allowed = [v for v, _ in param.options or ()]
        if not isinstance(value, list) or any(v not in allowed for v in value):
            raise DataError(f"{label}: choose from {', '.join(map(str, allowed))}.")
        return [v for v in allowed if v in value]
    if kind in ("number", "int"):
        number = _number(value)
        if number is None:
            raise DataError(f"{label}: enter a number.")
        if kind == "int":
            if not number.is_integer():
                raise DataError(f"{label}: enter a whole number.")
            number = int(number)
        if param.min is not None and number < param.min:
            raise DataError(f"{label}: must be at least {param.min:g}.")
        if param.max is not None and number > param.max:
            raise DataError(f"{label}: must be at most {param.max:g}.")
        return number
    if kind == "bool":
        if not isinstance(value, bool):
            raise DataError(f"{label}: expected true or false.")
        return value
    if kind == "string":
        if not isinstance(value, str) or (not value.strip() and not param.optional):
            raise DataError(f"{label}: enter a text.")
        return value.strip()
    if kind == "value":
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise DataError(f"{label}: enter a single value.")
        if value is None and not param.optional:
            raise DataError(f"{label}: enter a value.")
        return value
    if kind == "mapping":
        if value is None:
            return {}
        if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
            raise DataError(f"{label}: expected pairs of old and new values.")
        for new in value.values():
            if new is not None and not isinstance(new, (str, int, float, bool)):
                raise DataError(f"{label}: new values must be single values.")
        return dict(value)
    raise DataError(f"{label}: unsupported setting.")  # pragma: no cover
