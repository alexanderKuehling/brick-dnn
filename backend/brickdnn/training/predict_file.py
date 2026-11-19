"""Predict every row of a CSV file: ``python -m brickdnn.training.predict_file JOB_FILE``.

A single prediction (:mod:`.predict`) takes a few rows as JSON. A file with hundreds of
thousands of rows is predicted here instead, in a process of its own like every prediction:
it loads the model once, reads the file in chunks (so the memory it needs does not grow with
the file), prepares the rows like a single prediction does and writes a result file with the
same rows in the same order plus the prediction.

``JOB_FILE`` is JSON, written by the server (``brick_dnn_core/predict_files.py``)::

    {"input": "/…/input.csv", "output": "/…/result.csv",
     "task": "classification" | "regression", "class_names": ["no", "yes"] | null,
     "threshold": 0.3 | null,            # the decision threshold of a binary classification
     "models": [{"run_dir": "/…", "name": "MLP", "preprocessing": {…}}],
     "ensemble": {"method": "average", "weights": [0.5, 0.5], "meta": null} | null,
     "raw": "/…/prepare" | null,         # the frozen data preparation (brickdnn.data.frozen)
     "input_format": "auto" | "raw" | "prepared",
     "options": {"columns": "all" | ["id"], "prediction_column": "prediction",
                 "output": "label" | "probability" | "probabilities", "class": "yes" | null},
     "max_rows": 1000000}

One model predicts like :func:`.predict.predict`; several are the members of an ensemble of
runs, combined like the server combines them (:mod:`.ensemble`).

The columns are matched by name: the feature columns of the training dataset, or (``raw``)
the columns of the source table of its data preparation, which is then applied to the rows
with the values it learned. Other columns (an id …) are ignored by the model and can be kept
in the result. A row that cannot be predicted (a value that is not a number, too few values,
no value in a feature after the preparation) keeps its place with an empty prediction and is
counted; the first ones are listed with the reason.

It prints one line of JSON: the summary of the result (see :func:`predict_file`), or an
error for the user; ``cpu_seconds`` is the CPU time the process used::

    {"rows": 1200, "predicted": 1198, "failed": 2, "problems": ["Row 7, 'age': …"], …}
    {"error": "The file has no column 'age'. …", "kind": "file", "cpu_seconds": 0.4}

(``kind`` "file": the file or the options do not fit, which the user can change.)
"""

from __future__ import annotations

import codecs
import csv
import json
import math
import re
import sys
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .datasets import parse_number
from .inference import BINARY_POSITIVE
from .predict import MAX_OUTPUT_VALUES, PredictionError, load
from .reporting import short_error
from .runner import lower_priority
from .state import cpu_seconds

# Rows read, prepared and predicted at a time.
CHUNK_ROWS = 2000
# Rows of the result shown as a preview, and problems and warnings listed with their text.
PREVIEW_ROWS = 20
MAX_MESSAGES = 20
OUTPUTS = ("label", "probability", "probabilities")
DEFAULT_COLUMN = "prediction"
MAX_COLUMN_NAME = 120
INPUT_FORMATS = ("auto", "raw", "prepared")
SNIFF_CHARS = 64 * 1024
DECODE_BLOCK = 1024 * 1024
# A step of a data preparation that fails on a chunk is tried on its halves, to find the
# rows it fails on; after this many failures the file is given up.
MAX_FAILED_PREPARATIONS = 60
# Probabilities below this count as this in the log loss (as in evaluation.py).
EPSILON = 1e-15
_ROW_LABEL = re.compile(r"^(?:Row (\d+)|The row)")
_SEP_LINE = re.compile(r"\s*sep=(.)\s*", re.IGNORECASE)


class FileError(Exception):
    """The file or the options do not fit the model. The message is shown to the user."""


# --------------------------------------------------------------------------- #
# Reading the file
# --------------------------------------------------------------------------- #


def _decodes(path: Path, encoding: str) -> bool:
    decoder = codecs.getincrementaldecoder(encoding)()
    try:
        with path.open("rb") as handle:
            while block := handle.read(DECODE_BLOCK):
                decoder.decode(block)
            decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        return False
    return True


def detect_encoding(path: Path) -> str:
    """The text encoding of the file, found like the CSV import of the Data tab does
    (brickdnn.data.io): UTF-16 with a byte order mark, UTF-8, else Windows-1252, else
    Latin-1. The file is read in blocks. Raises :class:`FileError` for other files."""
    with path.open("rb") as handle:
        head = handle.read(8192)
    if not head.strip():
        raise FileError("The file is empty.")
    if head.startswith((b"PK\x03\x04", b"\xd0\xcf\x11\xe0")):
        raise FileError(
            "This looks like an Excel workbook (or another binary file), not a CSV file. "
            "In Excel, use “Save as” → “CSV UTF-8 (comma delimited)” and choose that file."
        )
    if head.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    if b"\x00" in head:
        raise FileError(
            "This is not a text file. Choose a CSV file (values separated by commas, "
            "semicolons or tabs)."
        )
    for name in ("utf-8-sig", "cp1252"):
        if _decodes(path, name):
            return name
    return "latin-1"


def detect_delimiter(sample: str) -> str:
    """``,``, ``;``, tab or ``|``: the one that splits the first lines into the same number
    of values (the detection of the CSV import; without pandas the one of the training
    upload)."""
    try:
        from brickdnn.data.io import _detect_delimiter
    except ImportError:  # pandas is not installed
        from .datasets import _sniff_delimiter

        return _sniff_delimiter(sample)
    return _detect_delimiter(sample)


def _blank(record: list[str]) -> bool:
    """An empty line (not a row of empty values, which keeps its place in the result)."""
    return not record or (len(record) == 1 and not record[0].strip())


@dataclass
class CsvFile:
    """The open file: its first record, the records after it, and how it is written."""

    first: list[str]
    records: Iterator[list[str]]
    delimiter: str
    encoding: str


def open_table(path: Path, handle: Any, encoding: str) -> CsvFile:
    """The records of the CSV file behind ``handle`` (text, opened with ``encoding``);
    lines without any value are skipped. Raises :class:`FileError`."""
    sample = handle.read(SNIFF_CHARS)
    handle.seek(0)
    delimiter = None
    first_line = sample.split("\n", 1)[0].rstrip("\r")
    if match := _SEP_LINE.fullmatch(first_line):  # Excel's "sep=;" line
        delimiter = match.group(1)
        handle.readline()
        sample = sample.split("\n", 1)[1] if "\n" in sample else ""
    if delimiter is None:
        cut = sample.rfind("\n")
        delimiter = detect_delimiter(sample[:cut] if cut > 0 else sample)
    records = (record for record in csv.reader(handle, delimiter=delimiter) if not _blank(record))
    try:
        first = next(records, None)
    except csv.Error as exc:
        raise FileError(f"The CSV file cannot be read: {exc}.") from None
    if first is None:
        raise FileError("The file is empty.")
    return CsvFile(first, records, delimiter, "utf-8" if encoding == "utf-8-sig" else encoding)


# --------------------------------------------------------------------------- #
# The columns of the file and of the result
# --------------------------------------------------------------------------- #


def _quoted(names: list[str], limit: int = 5) -> str:
    shown = ", ".join(f"'{n}'" for n in names[:limit])
    return shown + (f" and {len(names) - limit} more" if len(names) > limit else "")


def _listed(names: list[str], limit: int = 40) -> str:
    return ", ".join(names[:limit]) + (f" … ({len(names)} in all)" if len(names) > limit else "")


@dataclass
class Layout:
    """Where the model finds its columns in a record of the file."""

    header: list[str]  # the columns of the file, as the result names them
    has_header: bool
    raw: bool  # the columns of the source table of a data preparation
    needed: list[tuple[str, int]]  # the columns the model reads and their positions
    target: str | None
    target_at: int | None
    decimal_comma: bool

    @property
    def width(self) -> int:
        return len(self.header)

    @property
    def ignored(self) -> list[str]:
        used = {at for _, at in self.needed}
        return [
            name for at, name in enumerate(self.header) if at not in used and at != self.target_at
        ]


def _positions(header: list[str], names: list[str]) -> dict[str, int]:
    found = {}
    for name in names:
        if name in header:
            if header.count(name) > 1:
                raise FileError(f"The file has more than one column named '{name}'.")
            found[name] = header.index(name)
    return found


def layout_of(
    table: CsvFile,
    prepared: list[str],
    target: str | None,
    recipe: dict[str, Any] | None,
    input_format: str,
    decimal_comma: bool,
) -> Layout:
    """Match the columns of the file with the model's by name. ``prepared``: the feature
    columns of the training dataset; ``recipe``: the frozen data preparation, whose source
    columns the file may have instead. Raises :class:`FileError`."""
    header = [cell.strip() for cell in table.first]
    comma = decimal_comma or table.delimiter != ","
    found = _positions(header, prepared)
    raw_names = [str(c["name"]) for c in (recipe or {}).get("columns") or [] if c.get("used", True)]
    wants_raw = input_format == "raw" or len(found) < len(prepared)
    if recipe is not None and input_format != "prepared" and wants_raw:
        raw_found = _positions(header, raw_names)
        missing = [name for name in raw_names if name not in raw_found]
        if missing:
            source = (recipe.get("source") or {}).get("name") or "the source table"
            raise FileError(
                f"The file has no column {_quoted(missing)}. The model needs the columns "
                f"of '{source}' as imported: {_listed(raw_names)}. The file has: "
                f"{_listed(header)}."
            )
        raw_target = recipe.get("target")
        at = header.index(raw_target) if raw_target in header else None
        decimal = (recipe.get("source") or {}).get("decimal") == ","
        return Layout(
            header, True, True, [(n, raw_found[n]) for n in raw_names], raw_target, at, decimal
        )
    # No header (like a training file without one): the columns in their order, with the
    # target last if there is one more.
    if (
        not found
        and len(header) in (len(prepared), len(prepared) + 1)
        and all(parse_number(cell, comma) is not None for cell in header[: len(prepared)])
    ):
        names = list(prepared)
        if len(header) > len(prepared):
            names.append(target or "target")
        return Layout(
            names,
            False,
            False,
            list(zip(prepared, range(len(prepared)), strict=True)),
            target,
            len(prepared) if len(header) > len(prepared) else None,
            comma,
        )
    missing = [name for name in prepared if name not in found]
    if missing:
        raise FileError(
            f"The file has no column {_quoted(missing)}. The model needs: "
            f"{_listed(prepared)}. The file has: {_listed(header)}."
        )
    at = header.index(target) if target in header else None
    return Layout(header, True, False, [(n, found[n]) for n in prepared], target, at, comma)


@dataclass
class Result:
    """What is written per row: the kept columns of the file, then the prediction."""

    keep: list[int]
    columns: list[str]  # the header of the result
    output: str  # "label", "probability", "probabilities" or "value"
    class_at: int | None  # "probability": of which class
    cells: int  # prediction cells per row


def result_of(layout: Layout, options: Any, task: str, names: list[str] | None) -> Result:
    """Check the options of the result file. Raises :class:`FileError`."""
    options = options if isinstance(options, dict) else {}
    columns = options.get("columns", "all")
    if columns in (None, "all"):
        keep = list(range(layout.width))
    elif isinstance(columns, list) and all(isinstance(c, str) for c in columns):
        absent = [c for c in columns if c not in layout.header]
        if absent:
            raise FileError(
                f"The file has no column {_quoted(absent)} to keep in the result. It has: "
                f"{_listed(layout.header)}."
            )
        keep = [layout.header.index(c) for c in dict.fromkeys(columns)]
    else:
        raise FileError("'columns' must be \"all\" or a list of column names of the file.")
    name = options.get("prediction_column")
    name = DEFAULT_COLUMN if name in (None, "") else name
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > MAX_COLUMN_NAME:
        raise FileError(
            f"The name of the prediction column must be text of 1 to {MAX_COLUMN_NAME} characters."
        )
    name = " ".join(name.split())
    kept = [layout.header[at] for at in keep]
    output = options.get("output") or "label"
    class_at = None
    if task != "classification" or not names:
        output, added = "value", [name]
    elif output not in OUTPUTS:
        raise FileError(f"'output' must be one of: {', '.join(OUTPUTS)}.")
    elif output == "probabilities":
        added = list(names) if not set(names) & set(kept) else [f"{name}_{c}" for c in names]
    else:
        added = [name]
        if output == "probability":
            wanted = options.get("class")
            if wanted in (None, "") and len(names) == 2:
                wanted = names[BINARY_POSITIVE]
            if wanted not in names:
                raise FileError(
                    f"'class' says whose probability is written; it must be one of: "
                    f"{_listed(list(names))}."
                )
            class_at = names.index(wanted)
    twice = [c for c in added if c in kept]
    if twice:
        raise FileError(
            f"The result would have two columns named '{twice[0]}': choose another name for "
            "the prediction column, or leave that column out of the result."
        )
    return Result(keep, kept + added, output, class_at, len(added))


# --------------------------------------------------------------------------- #
# The models
# --------------------------------------------------------------------------- #


def _softmax(scores: Any) -> Any:
    import numpy as np

    shifted = scores - scores.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)


def _flat(value: Any) -> list[Any]:
    if isinstance(value, list):
        return [x for item in value for x in _flat(item)]
    return [value]


class Model:
    """A trained run, loaded once: standardises the feature columns like its training and
    predicts them (see :mod:`.predict`)."""

    def __init__(self, spec: dict[str, Any]):
        import numpy as np

        self.name = str(spec.get("name") or "")
        self.preprocessing = spec["preprocessing"]
        self.columns = [str(c) for c in self.preprocessing["columns"]]
        self.mean = np.asarray(self.preprocessing["mean"], dtype=np.float64)
        self.std = np.asarray(self.preprocessing["std"], dtype=np.float64)
        self.at: list[int] = []
        self.answer: Callable[[Any], dict[str, Any]] = load(spec["run_dir"])

    def bind(self, features: list[str]) -> None:
        """Where the columns of this model are in the prepared rows."""
        absent = [c for c in self.columns if c not in features]
        if absent:
            raise FileError(
                f"The rows have no column {_quoted(absent)} that the model was trained on."
            )
        self.at = [features.index(c) for c in self.columns]

    def outputs(self, matrix: Any) -> tuple[Any, bool]:
        """The raw answer for the prepared rows (rows × outputs), and whether it is class
        probabilities already (classical models)."""
        import numpy as np

        x = (matrix[:, self.at] - self.mean) / self.std
        tensor = {"dtype": "float", "shape": list(x.shape), "values": x.ravel().tolist()}
        result = self.answer({"inputs": [tensor], "kind": "tabular"})
        outputs = result.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != 1 or len(outputs[0]) != len(x):
            raise PredictionError("The model returned an unexpected answer.")
        values = np.array(
            [[math.nan if v is None else v for v in _flat(row)] for row in outputs[0]],
            dtype=np.float64,
        ).reshape(len(x), -1)
        return values, bool(result.get("probabilities"))

    def in_target_units(self, values: Any) -> Any:
        scale = self.preprocessing.get("target_std")
        shift = self.preprocessing.get("target_mean")
        if scale is not None and shift is not None:
            return values * float(scale) + float(shift)
        return values


class Predictor:
    """One model, or the members of an ensemble of runs with how they are combined: class
    probabilities (rows × classes) or values in the units of the target (rows)."""

    def __init__(self, job: dict[str, Any]):
        self.task = job.get("task")
        self.names = job.get("class_names") if self.task == "classification" else None
        self.models = [Model(spec) for spec in job["models"]]
        self.ensemble = job.get("ensemble") if isinstance(job.get("ensemble"), dict) else None
        threshold = job.get("threshold")
        binary = self.names is not None and len(self.names) == 2
        self.threshold = float(threshold) if threshold is not None and binary else None

    @property
    def columns(self) -> list[str]:
        return list(dict.fromkeys(c for model in self.models for c in model.columns))

    @property
    def outputs_per_row(self) -> int:
        return max(len(self.names or [1]), *(len(model.columns) for model in self.models))

    def bind(self, features: list[str]) -> None:
        for model in self.models:
            model.bind(features)

    def predict(self, matrix: Any) -> Any:
        import numpy as np

        classification = self.task == "classification"
        if self.ensemble is None:
            model = self.models[0]
            values, probabilities = model.outputs(matrix)
            if not classification:
                if values.shape[1] != 1:
                    raise PredictionError("The network returned an unexpected number of values.")
                return model.in_target_units(values[:, 0])
            if values.shape[1] != len(self.names or []):
                raise PredictionError("The network returned an unexpected number of classes.")
            return self._shares(values) if probabilities else _softmax(values)
        from . import ensemble as combining

        parts = []
        for model in self.models:
            values, probabilities = model.outputs(matrix)
            if not classification:
                parts.append(model.in_target_units(values.reshape(len(matrix))))
            elif probabilities:
                sums = values.sum(axis=1, keepdims=True)
                parts.append(values / np.where(sums > 0, sums, 1.0))
            else:
                parts.append(_softmax(values))
        try:
            combined = combining.combine(
                self.task,
                self.ensemble.get("method"),
                parts,
                self.ensemble.get("weights"),
                self.ensemble.get("meta"),
            )
        except (combining.EnsembleError, KeyError, ValueError) as exc:
            raise PredictionError(f"The members' predictions cannot be combined: {exc}") from None
        combined = np.asarray(combined, dtype=np.float64)
        return self._shares(combined) if classification else combined.reshape(len(matrix))

    @staticmethod
    def _shares(values: Any) -> Any:
        """Probabilities that add up to 1 (as inference._classes reads them)."""
        import numpy as np

        total = values.sum(axis=1, keepdims=True)
        return np.maximum(values, 0.0) / np.where(total == 0, 1.0, total)

    def labels(self, probabilities: Any) -> Any:
        """The predicted class of every row: with a decision threshold the second class
        when its probability is at least the threshold, else the most probable one."""
        import numpy as np

        if self.threshold is not None:
            positive = probabilities[:, BINARY_POSITIVE] >= self.threshold
            return np.where(positive, BINARY_POSITIVE, 1 - BINARY_POSITIVE)
        return np.argmax(probabilities, axis=1)


# --------------------------------------------------------------------------- #
# Preparing the rows of a chunk
# --------------------------------------------------------------------------- #


def _shown(cell: str) -> str:
    return cell.strip()[:30]


def _numbered(message: str, numbers: list[int]) -> str:
    """A message of the data preparation about "Row 3" of the rows it got, with the number
    that row has in the file."""

    def replace(match: re.Match) -> str:
        index = int(match.group(1)) - 1 if match.group(1) else 0
        return f"Row {numbers[index]}" if 0 <= index < len(numbers) else match.group(0)

    return _ROW_LABEL.sub(replace, message, count=1)


class RawPreparation:
    """The frozen data preparation of the run's dataset, applied to the rows of a chunk."""

    def __init__(self, directory: str, recipe: dict[str, Any]):
        self.directory = directory
        self.recipe = recipe
        self.failures = 0

    def prepare(
        self, rows: list[dict[str, str]], numbers: list[int]
    ) -> tuple[list[str], Any, dict[int, str], list[str]]:
        """``(features, matrix, problems by row index, warnings)``. A step that fails on
        the rows is tried on their halves, so that only the rows it fails on are lost."""
        import numpy as np

        from brickdnn.data.frozen import RawInputError, prepare_table

        try:
            table = prepare_table(self.recipe, rows, directory=self.directory)
        except RawInputError as exc:
            self.failures += 1
            message = exc.messages[0] if exc.messages else "The rows cannot be prepared."
            if self.failures > MAX_FAILED_PREPARATIONS:
                raise FileError(_numbered(message, numbers)) from None
            if len(rows) == 1:
                features = [str(c) for c in self.recipe.get("features") or []]
                text = _numbered(message, numbers)
                if not _ROW_LABEL.match(text):
                    text = f"Row {numbers[0]}: {text}"
                return features, np.full((1, len(features)), np.nan), {0: text}, []
            half = len(rows) // 2
            first = self.prepare(rows[:half], numbers[:half])
            second = self.prepare(rows[half:], numbers[half:])
            problems = dict(first[2])
            problems.update({half + i: text for i, text in second[2].items()})
            return first[0], np.vstack([first[1], second[1]]), problems, first[3] + second[3]
        problems = {i: _numbered(text, numbers) for i, text in table.problems.items()}
        warnings = [_numbered(text, numbers) for text in table.warnings]
        return table.features, table.matrix, problems, warnings


def target_changed(recipe: dict[str, Any]) -> bool:
    """Whether a step of the preparation mentions the target column (and so may change its
    values): the target values of a file with the source columns then cannot be compared
    with what the model predicts."""
    target = recipe.get("target")
    if not target:
        return False
    quoted = json.dumps(target, ensure_ascii=False)
    for step in recipe.get("steps") or []:
        if step.get("status") != "ok" or step.get("type") in ("drop_missing", "filter_rows"):
            continue
        params = step.get("params") or {}
        text = json.dumps(params, ensure_ascii=False)
        if quoted in text or target in str(params.get("expression") or ""):
            return True
    return False


# --------------------------------------------------------------------------- #
# Summary and metrics
# --------------------------------------------------------------------------- #


def _number_text(value: float, comma: bool) -> str:
    text = format(float(value), ".8g")
    return text.replace(".", ",") if comma else text


@dataclass
class Messages:
    """The first messages with their text, and how many there were."""

    shown: list[str] = field(default_factory=list)
    count: int = 0

    def add(self, text: str) -> None:
        self.count += 1
        if len(self.shown) < MAX_MESSAGES:
            self.shown.append(text)


class Metrics:
    """The predictions against the target column of the file, where a row has one."""

    def __init__(self, task: str, names: list[str] | None, comma: bool):
        self.task = task
        self.names = names or []
        self.comma = comma
        self.index = {name: i for i, name in enumerate(self.names)}
        # A prepared dataset has the class index as its target; a target of whole numbers
        # is read as those numbers only.
        self.by_index = not any(name.strip().lstrip("+-").isdigit() for name in self.names)
        self.count = 0
        self.unknown = 0  # rows whose target value is no class of the model, or no number
        self.correct = 0
        self.log_loss_sum = 0.0
        self.scores: list[Any] = []
        self.actual: list[Any] = []
        self.predicted: list[Any] = []

    def _class(self, cell: str) -> int | None:
        text = cell.strip()
        if text in self.index:
            return self.index[text]
        number = parse_number(text, self.comma)
        if number is not None and number.is_integer():
            whole = str(int(number))
            if whole in self.index:
                return self.index[whole]
            if self.by_index and text.isdigit() and int(text) < len(self.names):
                return int(text)
        return None

    def add(self, cells: list[str], predictions: Any, labels: Any) -> None:
        """``cells``: the target cells of the predicted rows; ``predictions``: their
        probabilities or values; ``labels``: their predicted classes (None: regression)."""
        import numpy as np

        rows, truth = [], []
        for i, cell in enumerate(cells):
            if not cell.strip():
                continue  # no target in this row: nothing to compare
            value = self._class(cell) if labels is not None else parse_number(cell, self.comma)
            if value is None:
                self.unknown += 1
                continue
            rows.append(i)
            truth.append(value)
        if not rows:
            return
        self.count += len(rows)
        if labels is None:
            self.predicted.append(np.asarray(predictions)[rows])
            self.actual.append(np.asarray(truth, dtype=np.float64))
            return
        true = np.asarray(truth, dtype=np.int64)
        proba = np.asarray(predictions)[rows]
        self.correct += int((np.asarray(labels)[rows] == true).sum())
        p_true = proba[np.arange(len(true)), true]
        self.log_loss_sum += float(-np.log(np.clip(p_true, EPSILON, 1.0)).sum())
        if len(self.names) == 2:
            self.scores.append(proba[:, BINARY_POSITIVE].copy())
            self.actual.append(true)

    def result(self) -> dict[str, Any] | None:
        import numpy as np

        from .evaluation import binary_aucs, regression_metrics

        if self.count == 0:
            return None
        found: dict[str, Any] = {"rows": self.count, "skipped": self.unknown}
        if self.task != "classification":
            measured = regression_metrics(
                np.concatenate(self.predicted), np.concatenate(self.actual)
            )
            found.update({key: measured.get(key) for key in ("rmse", "mae", "r2")})
            return found
        found["accuracy"] = self.correct / self.count
        found["log_loss"] = self.log_loss_sum / self.count
        if self.scores:
            scores, labels = np.concatenate(self.scores), np.concatenate(self.actual)
            found["roc_auc"] = binary_aucs(scores, labels)[0]
        return found


# --------------------------------------------------------------------------- #
# The whole file
# --------------------------------------------------------------------------- #


def predict_file(job: dict[str, Any]) -> dict[str, Any]:
    """Predict the rows of ``job["input"]`` and write ``job["output"]``. Returns::

        {"rows": 1200, "predicted": 1198, "failed": 2,
         "problems": ["Row 7, 'age': 'n/a' is not a number.", …],   # the first ones
         "warnings": [...], "warning_count": 3,
         "columns": ["id", "prediction"], "preview": [["1", "yes"], …],
         "task": "classification", "output": "label", "input_format": "prepared",
         "class_counts": {"no": 900, "yes": 298},        # or
         "values": {"min": 0.1, "max": 9.4, "mean": 3.2},
         "metrics": {"rows": 1198, "accuracy": 0.91, "log_loss": 0.24, "roc_auc": 0.95} | null,
         "target": "label", "ignored_columns": ["id"], "delimiter": ",", "encoding": "utf-8",
         "decision_threshold": 0.3 | null, "size_bytes": 14811}

    Raises :class:`FileError` (the file or the options do not fit) or
    :class:`.predict.PredictionError`.
    """
    import numpy as np

    path, output = Path(job["input"]), Path(job["output"])
    input_format = job.get("input_format") or "auto"
    if input_format not in INPUT_FORMATS:
        raise FileError(f"input_format must be one of: {', '.join(INPUT_FORMATS)}.")
    max_rows = int(job.get("max_rows") or 0)
    predictor = Predictor(job)
    task, names = predictor.task, predictor.names
    recipe = None
    if job.get("raw"):
        from brickdnn.data.frozen import read_recipe

        recipe = read_recipe(job["raw"])
    if input_format == "raw" and recipe is None:
        raise FileError(
            "input_format 'raw' is for runs trained on a dataset exported from a data "
            "preparation; this run takes the feature columns of its dataset."
        )
    first_model = predictor.models[0].preprocessing
    encoding = detect_encoding(path)
    try:
        with path.open(encoding=encoding, newline="") as handle:
            table = open_table(path, handle, encoding)
            layout = layout_of(
                table,
                predictor.columns,
                first_model.get("target"),
                recipe,
                input_format,
                bool(first_model.get("decimal_comma")),
            )
            result = result_of(layout, job.get("options"), task, names)
            comma = layout.decimal_comma and table.delimiter != ","
            preparation = RawPreparation(job["raw"], recipe) if layout.raw and recipe else None
            if preparation is None:
                predictor.bind([name for name, _ in layout.needed])
            bound = preparation is None
            chunk_rows = max(1, min(CHUNK_ROWS, MAX_OUTPUT_VALUES // predictor.outputs_per_row))
            metrics = None
            metrics_note = None
            if layout.target_at is not None:
                if layout.raw and task != "classification" and target_changed(recipe or {}):
                    metrics_note = (
                        f"The data preparation changes the target column '{layout.target}', so "
                        "its values in the file cannot be compared with the predictions."
                    )
                else:
                    metrics = Metrics(task, names, layout.decimal_comma)
            problems, warnings = Messages(), Messages()
            preview: list[list[str]] = []
            rows = predicted = 0
            counts = np.zeros(len(names or []), dtype=np.int64)
            low, high, total = math.inf, -math.inf, 0.0

            def chunks() -> Iterator[list[list[str]]]:
                chunk = [] if layout.has_header else [table.first]
                for record in table.records:
                    chunk.append(record)
                    if len(chunk) >= chunk_rows:
                        yield chunk
                        chunk = []
                if chunk:
                    yield chunk

            with output.open("w", encoding="utf-8", newline="") as out:
                writer = csv.writer(out, delimiter=table.delimiter, lineterminator="\n")
                writer.writerow(result.columns)
                for chunk in chunks():
                    if max_rows and rows + len(chunk) > max_rows:
                        raise FileError(
                            f"The file has more than {max_rows:,} rows, the most that are "
                            "predicted from one file. Split it into several files."
                        )
                    numbers = list(range(rows + 1, rows + len(chunk) + 1))
                    reasons: dict[int, str] = {}
                    good: list[int] = []
                    for i, record in enumerate(chunk):
                        if len(record) != layout.width:
                            reasons[i] = (
                                f"Row {numbers[i]} has {len(record)} values, but the file "
                                f"has {layout.width} columns."
                            )
                        else:
                            good.append(i)
                    if preparation is not None:
                        raw_rows = [{n: chunk[i][at] for n, at in layout.needed} for i in good]
                        features, matrix, found, notes = (
                            preparation.prepare(raw_rows, [numbers[i] for i in good])
                            if good
                            else ([], np.empty((0, 0)), {}, [])
                        )
                        for text in notes:
                            warnings.add(text)
                        for k, text in found.items():
                            reasons[good[k]] = text
                        usable = [k for k in range(len(good)) if k not in found]
                        if good and not bound:
                            predictor.bind(features)
                            bound = True
                        matrix = matrix[usable] if usable else np.empty((0, 0))
                        good = [good[k] for k in usable]
                    else:
                        values: list[list[float]] = []
                        kept: list[int] = []
                        for i in good:
                            record, row = chunk[i], []
                            for name, at in layout.needed:
                                x = parse_number(record[at], layout.decimal_comma)
                                if x is None:
                                    cell = _shown(record[at])
                                    reasons[i] = f"Row {numbers[i]}, '{name}': " + (
                                        f"'{cell}' is not a number." if cell else "no value."
                                    )
                                    break
                                row.append(x)
                            else:
                                values.append(row)
                                kept.append(i)
                        good = kept
                        matrix = np.asarray(values, dtype=np.float64).reshape(
                            len(values), len(layout.needed)
                        )
                    cells: dict[int, list[str]] = {}
                    if good:
                        answers = predictor.predict(matrix)
                        fine = np.isfinite(answers.reshape(len(good), -1)).all(axis=1)
                        for k in np.flatnonzero(~fine):
                            reasons[good[k]] = (
                                f"Row {numbers[good[k]]}: the model returned no number."
                            )
                        rows_ok = [good[k] for k in np.flatnonzero(fine)]
                        answers = answers[fine]
                        labels = predictor.labels(answers) if names else None
                        if labels is not None:
                            counts += np.bincount(labels, minlength=len(counts))
                        written = answers
                        if result.output == "label":
                            written = None
                        elif result.output == "probability":
                            written = answers[:, result.class_at]
                        if (
                            written is not None
                            and len(rows_ok)
                            and result.output != "probabilities"
                        ):
                            low = min(low, float(written.min()))
                            high = max(high, float(written.max()))
                            total += float(written.sum())
                        for k, i in enumerate(rows_ok):
                            if result.output == "label":
                                cells[i] = [str(names[int(labels[k])])]
                            elif result.output == "probabilities":
                                cells[i] = [_number_text(p, comma) for p in answers[k]]
                            else:
                                cells[i] = [_number_text(written[k], comma)]
                        if metrics is not None and rows_ok:
                            targets = [chunk[i][layout.target_at] for i in rows_ok]
                            metrics.add(targets, answers, labels)
                        predicted += len(rows_ok)
                    empty = [""] * result.cells
                    for i, record in enumerate(chunk):
                        line = [record[at] if at < len(record) else "" for at in result.keep]
                        line += cells.get(i, empty)
                        writer.writerow(line)
                        if len(preview) < PREVIEW_ROWS:
                            preview.append(line)
                        if i in reasons:
                            problems.add(reasons[i])
                    rows += len(chunk)
    except UnicodeDecodeError:
        raise FileError("The text encoding of the file cannot be read.") from None
    except csv.Error as exc:
        raise FileError(f"The CSV file cannot be read: {exc}.") from None
    if rows == 0:
        raise FileError("The file has a header line but no rows to predict.")
    if predicted == 0:
        raise FileError(f"No row of the file could be predicted. {problems.shown[0]}")
    summary: dict[str, Any] = {
        "rows": rows,
        "predicted": predicted,
        "failed": rows - predicted,
        "problems": problems.shown,
        "warnings": warnings.shown,
        "warning_count": warnings.count,
        "columns": result.columns,
        "preview": preview,
        "task": task,
        "output": result.output,
        "input_format": "raw" if layout.raw else "prepared",
        "has_header": layout.has_header,
        "target": layout.target if layout.target_at is not None else None,
        "ignored_columns": layout.ignored,
        "delimiter": table.delimiter,
        "encoding": table.encoding,
        "decision_threshold": predictor.threshold,
        "metrics": metrics.result() if metrics is not None else None,
        "metrics_note": metrics_note,
        "size_bytes": output.stat().st_size,
    }
    if names:
        summary["class_counts"] = {str(n): int(c) for n, c in zip(names, counts, strict=True)}
        if result.output == "probability":
            summary["class"] = names[result.class_at or 0]
    if math.isfinite(low):
        summary["values"] = {"min": low, "max": high, "mean": total / predicted}
    return summary


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m brickdnn.training.predict_file JOB_FILE", file=sys.stderr)
        return 2
    lower_priority()
    try:
        job = json.loads(Path(args[0]).read_text(encoding="utf-8"))
        result = predict_file(job)
    except FileError as exc:
        result = {"error": str(exc), "kind": "file"}
    except PredictionError as exc:
        result = {"error": str(exc)}
    except Exception as exc:
        traceback.print_exc()
        result = {"error": f"The prediction failed: {short_error(exc)}"}
    result["cpu_seconds"] = round(cpu_seconds(), 3)
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
