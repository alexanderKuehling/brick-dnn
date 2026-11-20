"""Messages the ``brickdnn`` package builds from data, in the request's language.

The package writes its messages in English with the values put in (``f"Kernel size {k} is
larger than the padded input size {n}."``) and knows nothing of Django: the CLI and the MCP
server show them as they are. For the web app the API translates them on the way out:

- :func:`templates` reads the package's source (:mod:`ast`) and turns every message-like string
  into a *template*, the literal parts with numbered placeholders: ``"Kernel size {0} is larger
  than the padded input size {1}."``. These are the msgids (context ``brickdnn``) in
  ``locale/de/LC_MESSAGES/django.po``; ``manage.py update_translations`` collects them.
- :func:`translate_message` matches a finished English message against the templates (one
  regular expression each, longest literal text first), takes the values out of it and puts them
  into the translated template. A message made of several sentences is translated sentence by
  sentence. What matches no template stays English. The values are translated too where they
  are messages of their own (a hint put into a sentence, the label of a step or a layer),
  counts (``"3 rows"``, with the nouns of :func:`counts`), whole numbers with thousands
  separators (``"3,072"`` is ``"3.072"`` in German) and the metrics of a run
  (``"validation accuracy 97.1%; test accuracy 96.8%"``, :func:`_translate_metrics`).
- EXTRA_TEMPLATES and LABEL_TEMPLATES list what the first step cannot find: short messages
  (``"Finished: {0}"``), parts the package puts together and messages the Django side stores
  with a run while no request is around.
- :func:`translate_response` applies it to the text fields of an API answer (``message``,
  ``summary``, ``errors`` …).

So the package stays free of Django and its messages keep one source of truth, and a changed
English message only loses its translation until ``update_translations`` picks it up again.
"""

from __future__ import annotations

import ast
import functools
import re
from pathlib import Path
from typing import Any

from django.utils import translation
from django.utils.translation import gettext, npgettext, pgettext
from rest_framework import renderers

CONTEXT = "brickdnn"
PACKAGE = Path(__file__).resolve().parent.parent / "brickdnn"
# The modules whose messages users see: network analysis, training checks, data preparation
# and its analysis.
MODULES = (
    "graph.py",
    "layers.py",
    "training/compat.py",
    "training/engine.py",
    "training/classical_engine.py",
    "training/hybrid_engine.py",
    "training/state.py",
    "data/checks.py",
    "data/pipeline.py",
    "data/profile.py",
    "data/rows.py",
    "data/steps",
)
# The package's helpers that write a count with its noun: ``plural(3, "row")`` is "3 rows",
# ``_count(2, "class has", "classes have")`` is "2 classes have" (see :func:`counts`).
COUNT_FUNCTIONS = frozenset({"plural", "_count", "_plural"})
# Keys of API answers whose string values (or lists of strings) are messages.
MESSAGE_KEYS = frozenset(
    {
        "message",
        "messages",
        "summary",
        "detail",
        "errors",
        "warnings",
        "notes",
        "note",
        "hint",
        "reason",
        "explanation",
        "fix_summary",
        "log",
    }
)
# (parent key, key): messages only inside an object of the parent key, e.g. the label of the
# fix a readiness check offers ({"fix": {"type", "params", "label"}}).
NESTED_MESSAGE_KEYS = frozenset({("fix", "label")})
_PLACEHOLDER = re.compile(r"\{(\d+)\}")
# Messages :func:`templates` cannot find in the source: short ones, parts that the package puts
# together into one message, and messages the Django side stores with a run before a request
# (and its language) is around. tests/test_languages.py checks that their text is still in
# the source.
EXTRA_TEMPLATES = (
    # The final message of a run (training/engine.py, classical_engine.py, hybrid_engine.py,
    # ensemble.py) and its parts.
    "Finished: {0}",
    "Finished: {0}.",
    "Fitted on {0} rows in {1}.",
    "Stopped early after epoch {0}: no improvement for {1} epochs.",
    "Stopped early after epoch {0}: no improvement for {1} epochs; kept the weights of epoch {2}.",
    "Kept the weights of epoch {0}.",
    "Kept the weights of epoch {0} (lowest validation loss).",
    "less than 0.1 s",
    "Finished 1 trial.",
    "Finished {0} trials.",
    "Finished {0} trials ({1} failed).",
    "Best: trial {0} ({1}), {2}",
    "Best: trial {0} ({1}), {2}.",
    "Best so far: trial {0} ({1}).",
    "{0} features learned in 1 epoch",
    "{0} features learned in {1} epochs",
    "{0} features learned in {1} epochs (kept epoch {2})",
    "{0} frozen random features",
    "Combined {0} runs ({1}).",
    "Combined {0} runs ({1}): {2}.",
    "weighted average",
    "stacking",
    "Gaussian process",
    "support vector machine",
    "gradient boosting model",
    "Ensemble of {0} runs ({1}): {2}.",
    # Progress and failures (the status line and the log of a run).
    "Epoch {0} of {1}",
    "Trial {0} of {1}: epoch {2} of {3}",
    "Random seed: {0}",
    "Training failed: {0}",
    "Fitting failed: {0}",
    "Trial {0} failed: {1}",
    # What a training check says about its data (training/compat.py).
    "Demo data: random inputs and a hidden rule to learn (classification).",
    "Demo data: random inputs and a hidden rule to learn (regression).",
    "Demo data: random inputs and a hidden rule to learn (classification and regression).",
    "{0}: neural layers learn {1} features, {2} is fitted on them.",
    "{0} training, {1} validation and {2} test rows",
    "{0} training and {1} validation rows, no test set",
    "{0} training and {1} test rows, no validation set",
    "{0} training rows, no validation and no test set",
    "{0} training, {1} validation and {2} test images",
    "{0} training and {1} validation images, no test set",
    # Readiness checks and warnings of the analysis (data/checks.py, data/profile.py): parts
    # of their sentences.
    "In a sample of {0} rows, {1}",
    "{0} and {1} are almost perfectly correlated (r = {2}); one of them is enough.",
    "the nearest row has the same values in every feature",
    "the nearest row is about {0}× closer than a typical pair of rows",
    "{0} of the validation/test rows have a near twin in the training rows, so the validation "
    "and test scores are too optimistic.",
    "Keep near twins in one part: find the groups and split by them.",
    # A failed step of a preparation (brick_dnn_core/data_service.py).
    "Step {0} ({1}) failed: {2}",
    "log(1 + x) makes it more even.",
    "log(1 + x) makes them more even.",
    "A Yeo-Johnson transform makes it more even.",
    "A Yeo-Johnson transform makes them more even.",
    # Stored with a run by the Django side (training_queue.py, training_service.py).
    "Starting …",
    "Not started: {0}",
    "The training process could not be started. See the server log for details.",
)
# Found in the source, but not used. The first one: an entry of EXTRA_TEMPLATES translates
# the message better (the metric of "validation {2} {3}" is a value there, which would stay
# English). The others have values that are English words of the package ("is", "too
# typical", "year, month"): half a sentence in each language is worse than an English one.
SKIPPED_TEMPLATES = (
    "Best: trial {0} ({1}), validation {2} {3}",
    "Already {0}: {1}.",
    "Added {0} ({1}) from {2} ({3}) and {4}.",
    "Removed {0} ({1}) that {2} {3} on {4} ({5}).",
    "Removed {0} ({1}) {2}.",
    "Removed {0} ({1}) with {2}.",
)
# The label of the step a readiness check offers ({"fix": {"label": ...}}, data/checks.py):
# matched only there, since "Drop {0}" would fit many other sentences.
LABEL_TEMPLATES = (
    "Drop {0}",
    "Standardise {0}",
    "Remove duplicates",
    "Log-transform {0} (log1p)",
    "Encode {0} (target mean)",
    "Encode {0} (frequency)",
    "Impute {0} (median)",
    "Impute {0} (most frequent value)",
)
# The metrics in the final message of a run: "validation accuracy 97.1%", "test loss 0.123,
# RMSE 4.5, R² 0.912" (:func:`_translate_metrics`). The first metric of a group names the data
# it was measured on.
METRIC_SPLITS = ("validation", "test", "training")
METRIC_NAMES = (
    "accuracy",
    "loss",
    "RMSE",
    "R²",
    "mean IoU",
    "pixel accuracy",
    "balanced accuracy",
    "macro F1",
    "ROC AUC",
    "PR AUC",
    "MAE",
)
# The settings of the best trial of a search, written the same way: "learning rate 0.00312,
# batch size 64, Adam" (training/tuning.py).
SETTING_NAMES = (
    "learning rate",
    "batch size",
    "weight decay",
    "focal gamma",
    "label smoothing",
    "Huber delta",
)
METRIC_NOTES = (
    "(on the training data, optimistic)",
    "(on the training rows, optimistic)",
    "(optimistic: measured on the rows the stacking model was fitted on)",
)
METRIC_PREFIXES = ("stopped early, weights of epoch {0}: {1}", "weights of epoch {0}: {1}")
# The end of a list that names only its first items: "a, b and 3 more".
MORE_TEMPLATE = "{0} and {1} more"
# The rows a message is about, where the value stands directly before "rows", "scores" or
# "images" ("no {0} rows"): translated to the first part of a German compound ("Validierungs",
# for "{0}zeilen"). Elsewhere a value "test" may be the name of a column.
SPLIT_WORDS = ("validation …", "test …", "training …", "validation and test …")
_SPLIT_VALUE = re.compile(r"\{(\d+)\} (?:rows|scores|images)\b")


def _is_message(text: str, sentence: bool = False) -> bool:
    """A sentence for people, not an identifier, a format or a code snippet. ``text`` has an
    ``x`` for every value. ``sentence``: it is the rest of a sentence that starts with a
    value."""
    words = re.findall(r"[A-Za-z]{2,}", text)
    # A sentence that starts with a value, e.g. the name of a layer, column or dataset
    # ("x has no incoming connection."). Fragments that start with one end without a stop.
    # A hint in a value may follow ("x predicts one class per sample. x").
    if text.startswith("x ") and re.search(r"[.!?]( x)?$", text.rstrip()):
        rest = text[2:]
        return _is_message(rest[:1].upper() + rest[1:], sentence=True)
    # What a fix changes: "Input shape x → x".
    if "→" in text and text[:1].isupper() and len(words) >= 2 and "\n" not in text:
        return True
    # A verb and what it did, with a full stop: "Dropped x: x.", "Renamed x."
    if re.match(r"[A-Z][a-z]{3,} x\b", text) and text.rstrip().endswith(".") and "\n" not in text:
        return True
    # A short sentence with a value and a full stop: "Applied x to x.", "x missing values (x)."
    short = sentence or (text.rstrip().endswith(".") and " x" in text and text[:1].isupper())
    if len(words) < (2 if short else 3):
        return False
    if text.lstrip().startswith(("%", "{", "<", "#", "/", "http", "def ", "import ", "class ")):
        return False
    if "\n" in text.strip() or "    " in text or text.count("(") > 4:
        return False
    # Fragments that other messages are built from, and code (docstrings, assignments).
    if text[:1].islower() or text[:1] in " ,;" or '"""' in text or " = " in text:
        return False
    return bool(re.search(r"[A-Za-z]{2,}[,:;]? (?:[A-Za-z] )?[A-Za-z]{2,}", text))


_FORMAT_FIELD = re.compile(r"\{[A-Za-z_]\w*\}")


def _template_of(node: ast.AST, fields: bool = False) -> str | None:
    """The template of a string expression. ``fields``: a constant with the named fields of
    ``str.format`` ("Scaled {cols} to {low} … {high}.") is a template of its own, with a
    number for every field."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if fields and _FORMAT_FIELD.search(node.value):
            numbers = iter(range(len(node.value)))
            return _FORMAT_FIELD.sub(lambda _: f"{{{next(numbers)}}}", node.value)
        return node.value.replace("{", "{{").replace("}", "}}")
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        index = 0
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value.replace("{", "{{").replace("}", "}}"))
            elif isinstance(value, ast.FormattedValue):
                parts.append(f"{{{index}}}")
                index += 1
            else:
                return None
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _template_of(node.left), _template_of(node.right)
        if left is None or right is None:
            return None
        # Renumber the placeholders of the right part after those of the left one.
        offset = len(_PLACEHOLDER.findall(left))
        right = _PLACEHOLDER.sub(lambda m: f"{{{int(m.group(1)) + offset}}}", right)
        return left + right
    return None


def _docstrings(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def _module_files() -> list[Path]:
    files: list[Path] = []
    for name in MODULES:
        path = PACKAGE / name
        files.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])
    return [path for path in files if path.exists()]


@functools.lru_cache(maxsize=1)
def templates() -> tuple[str, ...]:
    """Every message template of the package, in a stable order without duplicates."""
    found: dict[str, None] = {}
    for path in _module_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        skip = _docstrings(tree)
        inner: set[int] = set()
        for node in ast.walk(tree):
            if id(node) in skip or id(node) in inner:
                continue
            if isinstance(node, (ast.JoinedStr, ast.BinOp, ast.Constant)):
                template = _template_of(node, fields=True)
                if template is None:
                    continue
                # The parts of a concatenation or f-string are not messages of their own.
                for child in ast.walk(node):
                    if child is not node:
                        inner.add(id(child))
                text = template.replace("{{", "{").replace("}}", "}")
                if _is_message(_PLACEHOLDER.sub("x", text)):
                    found.setdefault(template, None)
    for template in (*EXTRA_TEMPLATES, *_stored_messages()):
        found.setdefault(template, None)
    for template in SKIPPED_TEMPLATES:
        found.pop(template, None)
    return tuple(found)


def _stored_messages() -> tuple[str, ...]:
    """What the training queue writes into a run or its queue information: the worker that
    stores them knows no request, so they are translated on the way out like the package's
    messages."""
    from . import training_queue as queue

    return (
        queue.QUEUED_MESSAGE,
        queue.CANCELLED_MESSAGE,
        queue.RESTARTED_MESSAGE,
        queue.LOST_MESSAGE,
        queue.LOST_TWICE_MESSAGE,
        queue.NO_WORKER_MESSAGE,
        queue.MODE_OFF_MESSAGE,
    )


def phrases() -> tuple[str, ...]:
    """What is translated besides the templates (``pgettext``, context ``brickdnn``): the
    labels of the steps a readiness check offers, the names of metrics in the final message
    of a run, with and without the data they were measured on ("validation accuracy",
    "accuracy"), the notes after them and what may stand before them, and the end of a list
    that names only its first items."""
    return (
        *LABEL_TEMPLATES,
        *(f"{split} {name}" for split in METRIC_SPLITS for name in METRIC_NAMES),
        *METRIC_NAMES,
        *SETTING_NAMES,
        *METRIC_NOTES,
        *METRIC_PREFIXES,
        MORE_TEMPLATE,
        *SPLIT_WORDS,
    )


@functools.lru_cache(maxsize=1)
def counts() -> tuple[tuple[str, str], ...]:
    """The nouns the package counts with its helpers (COUNT_FUNCTIONS), as (singular, plural):
    ``("row", "rows")``, ``("class has", "classes have")``. They are translated with
    ``npgettext`` (context ``brickdnn``): "3 rows" in a message is "3 Zeilen" in German."""
    found: dict[tuple[str, str], None] = {}
    for path in _module_files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call) or len(node.args) < 2:
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            words = [
                arg.value
                for arg in node.args[1:3]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            ]
            if name not in COUNT_FUNCTIONS or len(words) != len(node.args[1:3]):
                continue
            if words[0].strip():
                found.setdefault((words[0], words[1] if len(words) > 1 else words[0] + "s"), None)
    return tuple(found)


@functools.lru_cache(maxsize=1)
def _count_nouns() -> dict[str, tuple[str, str]]:
    """The singular and the plural of every counted noun -> (singular, plural)."""
    nouns: dict[str, tuple[str, str]] = {}
    for singular, plural in counts():
        nouns.setdefault(singular, (singular, plural))
        nouns.setdefault(plural, (singular, plural))
    return nouns


_COUNT = re.compile(r"^(\d{1,3}(?:,\d{3})+|\d+) (\S.*)$", re.S)
_THOUSANDS = re.compile(r"^-?\d{1,3}(?:,\d{3})+$")


def _number(text: str) -> str:
    """A whole number with thousands separators in German: "3,072" -> "3.072"."""
    return text.replace(",", ".") if _THOUSANDS.match(text) else text


@functools.lru_cache(maxsize=1)
def _layer_labels() -> frozenset[str]:
    """The labels of the layers and of the steps of a preparation ("Gradient boosting
    classification", "Filter rows"): messages name them by their label, which the catalogue
    translates (catalog.py)."""
    from brickdnn import REGISTRY
    from brickdnn.data.steps import REGISTRY as STEPS

    return frozenset(spec.label for spec in (*REGISTRY.values(), *STEPS.values()))


_SECONDS = re.compile(r"^(\d+)\.(\d+) s$")
_MORE = re.compile(r"^(.*) and (\d{1,3}(?:,\d{3})*) more$", re.S)
_METRIC_VALUE = r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?%?"
_METRIC = re.compile(
    rf"^(?:(?P<split>{'|'.join(METRIC_SPLITS)}) )?"
    rf"(?P<name>{'|'.join(re.escape(n) for n in (*METRIC_NAMES, *SETTING_NAMES))}) "
    rf"(?P<value>{_METRIC_VALUE})$",
    re.I,
)
_METRIC_PREFIX = re.compile(r"^(?P<early>stopped early, )?weights of epoch (?P<epoch>\d+): ")


def _metric_value(text: str) -> str:
    """A number of a metric in German: "0.912" -> "0,912", "97.1%" -> "97,1 %"."""
    percent = text.endswith("%")
    number = text.rstrip("%").replace(".", ",")
    return f"{number}\u00a0%" if percent else number


def _translate_metrics(text: str) -> str | None:
    """The metrics in the final message of a run, in the active language: "validation accuracy
    97.1%; test accuracy 96.8%" or "Validation loss 0.123, RMSE 4.5, R² 0.912." (groups after
    ";", the metrics of a group after ","), and the settings of a trial, which are written the
    same way. None if ``text`` is something else."""
    body = text.rstrip()
    stop = "." if body.endswith(".") else ""
    body = body[: len(body) - len(stop)]
    groups = []
    for index, group in enumerate(body.split("; ")):
        prefix = _METRIC_PREFIX.match(group)
        if prefix:
            group = group[prefix.end() :]
        note = next((n for n in METRIC_NOTES if group.endswith(f" {n}")), None)
        if note:
            group = group[: -len(note) - 1]
        parts = []
        for position, part in enumerate(group.split(", ")):
            found = _METRIC.match(part)
            if not found and position > 0 and re.fullmatch(r"[A-Za-z][\w-]*", part):
                parts.append(part)  # the optimizer of a trial: "Adam"
                continue
            if not found or (found.group("split") and position > 0):
                return None
            split = (found.group("split") or "").lower()
            names = (*METRIC_NAMES, *SETTING_NAMES)
            name = next(n for n in names if n.lower() == found.group("name").lower())
            phrase = f"{split} {name}" if split else name
            translated = pgettext(CONTEXT, phrase)
            parts.append(f"{translated} {_metric_value(found.group('value'))}")
        result = ", ".join(parts)
        if note:
            result += f" {pgettext(CONTEXT, note)}"
        if prefix:
            template = METRIC_PREFIXES[0 if prefix.group("early") else 1]
            translated = pgettext(CONTEXT, template)
            result = translated.replace("{0}", prefix.group("epoch")).replace("{1}", result)
        if index == 0 and text[:1].isupper():
            result = result[:1].upper() + result[1:]
        groups.append(result)
    return "; ".join(groups) + stop


def _translate_value(value: str, depth: int) -> str:
    """A value taken out of a message, in the active language: a count ("3 rows"), a whole
    number ("3,072"), the label of a layer, the metrics of a run or a message of its own (a
    hint, the label of a step)."""
    if value != value.strip():
        # A sentence added to another one with its space (" A network with …").
        lead = value[: len(value) - len(value.lstrip())]
        tail = value[len(value.rstrip()) :]
        return f"{lead}{_translate_value(value.strip(), depth)}{tail}" if value.strip() else value
    if _THOUSANDS.match(value):
        return _number(value)
    count = _COUNT.match(value)
    if count and count.group(2) in _count_nouns():
        singular, plural = _count_nouns()[count.group(2)]
        n = int(count.group(1).replace(",", ""))
        noun = npgettext(CONTEXT, singular, plural, n)
        if noun in (singular, plural):
            return value
        return f"{_number(count.group(1))} {noun}"
    seconds = _SECONDS.match(value)
    if seconds:
        return f"{seconds.group(1)},{seconds.group(2)} s"
    if value in _layer_labels():
        return gettext(value)
    if depth >= 2 or " " not in value.strip():
        return value
    metrics = _translate_metrics(value)
    if metrics is not None:
        return metrics
    translated = translate_message(value, depth + 1)
    if translated == value and not value.rstrip().endswith((".", "!", "?", "…", ":")):
        # A sentence that lost its full stop inside another one.
        translated = translate_message(f"{value}.", depth + 1)
        translated = value if translated == f"{value}." else translated.removesuffix(".")
    more = _MORE.match(translated)
    if more:
        # "a, b and 3 more" at the end of a list.
        rest = pgettext(CONTEXT, MORE_TEMPLATE)
        if rest != MORE_TEMPLATE:
            translated = rest.replace("{0}", more.group(1)).replace("{1}", _number(more.group(2)))
    return translated


# The end of a sentence inside a value (not the stop of "e.g.").
_STOP = re.compile(r"(?<!e\.g)[.!?]\s+(?=[A-Z])")
_Matcher = tuple[re.Pattern[str], str, frozenset[str]]


def _compile(found: tuple[str, ...]) -> list[_Matcher]:
    """(regular expression, template, the values inside the text), the most specific (longest
    literal text) first. A value inside the text is followed by more text of the template; the
    others end the message."""
    result = []
    for template in found:
        pieces = re.split(r"(\{\d+\})", template)
        pattern = ""
        inside = set()
        for index, piece in enumerate(pieces):
            match = _PLACEHOLDER.fullmatch(piece)
            if not match:
                pattern += re.escape(piece.replace("{{", "{").replace("}}", "}"))
                continue
            if any(_PLACEHOLDER.sub("", later) for later in pieces[index + 1 :]):
                inside.add(f"p{match.group(1)}")
            if index >= 2 and pieces[index - 1] == "":
                # Directly after another value ("{2}{3}", e.g. a list and its "and 3 more"):
                # the first one takes all of it, so that it is translated as a whole.
                pattern += f"(?P<p{match.group(1)}>.*?)"
            elif index + 2 < len(pieces) and pieces[index + 1] == "":
                pattern += f"(?P<p{match.group(1)}>.+)"
            else:
                pattern += f"(?P<p{match.group(1)}>.+?)"
        literal = len(_PLACEHOLDER.sub("", template))
        result.append((literal, re.compile(f"^{pattern}$", re.S), template, frozenset(inside)))
    result.sort(key=lambda item: -item[0])
    return [(pattern, template, inside) for _, pattern, template, inside in result]


@functools.lru_cache(maxsize=1)
def _matchers() -> list[_Matcher]:
    return _compile(templates())


@functools.lru_cache(maxsize=1)
def _label_matchers() -> list[_Matcher]:
    return _compile(LABEL_TEMPLATES)


def _translate_one(text: str, depth: int = 0, matchers: list[_Matcher] | None = None) -> str | None:
    for pattern, template, inside in _matchers() if matchers is None else matchers:
        match = pattern.match(text)
        if not match:
            continue
        # A value inside the text that runs over the end of a sentence: the template fits a
        # longer message only by accident ("Finished: {0}." and a second sentence after it).
        if any(_STOP.search(match.group(name)) for name in inside):
            continue
        translated = pgettext(CONTEXT, template)
        if translated == template:
            # Texts of the catalogues are translated without the context (catalog.py).
            translated = gettext(template)
        if translated == template:
            return None
        values = {
            int(name[1:]): _translate_value(value, depth)
            for name, value in match.groupdict().items()
        }
        for index in _SPLIT_VALUE.findall(template):
            if f"{match.group(f'p{index}')} …" in SPLIT_WORDS:
                values[int(index)] = pgettext(CONTEXT, f"{match.group(f'p{index}')} …")
        try:
            result = _PLACEHOLDER.sub(lambda m, v=values: v[int(m.group(1))], translated)
        except KeyError:
            return None
        return result.replace("{{", "{").replace("}}", "}")
    return None


# Where a new sentence starts ("log(1 + x) …" is the one that starts with a small letter).
# After a closing bracket also the one that starts with a number ("… object?). 10.9 % of …").
_SENTENCE = re.compile(r"(?<!e\.g\.)(?<=[.!?:;])\s+(?=[A-Z“\"'(]|log\()|(?<=\)\.)\s+(?=\d)")


def translate_message(text: str, depth: int = 0) -> str:
    """``text`` in the active language, or unchanged if no template matches (or English).
    ``depth``: how deep ``text`` is inside the values of other messages."""
    if not text or (translation.get_language() or "en").split("-")[0] == "en":
        return text
    whole = _translate_one(text, depth)
    if whole is not None:
        return whole
    sentences = _SENTENCE.split(text)
    if len(sentences) == 1:
        return _translate_metrics(text) or text
    return " ".join(
        _translate_one(sentence, depth) or _translate_metrics(sentence) or sentence
        for sentence in sentences
    )


def translate_label(text: str) -> str:
    """The label of a step that a readiness check offers ("Drop id"), in the active language."""
    if not text or (translation.get_language() or "en").split("-")[0] == "en":
        return text
    return _translate_one(text) or _translate_one(text, 0, _label_matchers()) or text


class JSONRenderer(renderers.JSONRenderer):
    """DRF's JSON renderer, with the package's messages in the request's language."""

    def render(self, data, accepted_media_type=None, renderer_context=None):
        return super().render(translate_response(data), accepted_media_type, renderer_context)


def translate_response(data: Any, key: str | None = None, parent: str | None = None) -> Any:
    """``data`` (an API answer) with its messages translated."""
    if (translation.get_language() or "en").split("-")[0] == "en":
        return data
    if isinstance(data, dict):
        return {k: translate_response(v, k, key) for k, v in data.items()}
    if isinstance(data, list):
        return [translate_response(item, key, parent) for item in data]
    if isinstance(data, str) and (parent, key) in NESTED_MESSAGE_KEYS:
        return translate_label(data)
    if isinstance(data, str) and key in MESSAGE_KEYS:
        return translate_message(data)
    return data
