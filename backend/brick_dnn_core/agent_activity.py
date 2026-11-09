"""The activity log of AI agents: what assistants and scripts did with a user's API tokens and
connected apps, shown on the page "Agent activity".

What is logged, as an :class:`AgentAction` each:

- every MCP tool call (``record_tool_call``, called by mcp.py), and
- every other request made with an API token or an app's access token (the middleware below),
  also when it is refused (read-only token, app token outside the MCP server).

An entry says who (the token or app by name, the client's user agent), what (the tool or REST
route, the outcome with the error message, the duration), what it touched (networks, training
runs, datasets, data files, with ids and names) and a sentence for people. Requests of the
website (JWT logins) are not logged.

Arguments are summarised and redacted before they are stored (``summarize``): sizes and
counts instead of CSV text, file data, prediction inputs, graphs and long texts, links without
their query; tokens, passwords and other secrets never.

Logging never gets in the way: entries are described and written after the response has been
sent (``_defer``), and every error on the way is logged and swallowed. Each user sees their own
entries (``AgentActivityView``, website login only). The daily ``purge_old_data`` deletes
entries older than ``BRICKDNN_AGENT_LOG_DAYS`` days (default 90; 0 switches the log off) and
keeps at most ``MAX_ENTRIES_PER_USER`` per account. The account export includes them and
deleting the account deletes them; revoking a token or disconnecting an app keeps its entries
until they expire. See docs/mcp.md.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from django.db.models import Count, Max, Q, QuerySet
from django.utils import timezone
from django.utils.dateparse import parse_date, parse_datetime
from rest_framework import permissions
from rest_framework.exceptions import ValidationError
from rest_framework.request import Empty, Request
from rest_framework.response import Response
from rest_framework.views import APIView

from .api_tokens import MCP_URL_NAMES, READ_ONLY_POSTS, IsLoggedInUser
from .models import AgentAction, ApiToken, OAuthConnection

logger = logging.getLogger(__name__)

# The daily purge keeps at most this many entries per account (the newest), so that a busy
# script cannot fill the database.
MAX_ENTRIES_PER_USER = 20_000
PAGE_SIZE = 50
MAX_PAGE_SIZE = 200

MAX_NAME = 120  # names of networks, datasets, runs
MAX_SHORT = 40  # other short texts in the arguments
MAX_ERROR = 500
MAX_SUMMARY = 300
MAX_USER_AGENT = 200
MAX_KEYS = 20  # arguments (and settings inside them) per entry
# POST requests that change nothing.
READ_ONLY_ROUTES = {*READ_ONLY_POSTS, "data-prep-preview"}


def enabled() -> bool:
    return settings.AGENT_LOG_DAYS > 0


# --------------------------------------------------------------------------- #
# Redaction: what of the arguments is stored
# --------------------------------------------------------------------------- #

REDACTED = "[redacted]"
# Keys whose values are never stored: any key containing one of these words, or one of the
# short keys exactly.
SECRET_WORDS = (
    "password",
    "secret",
    "token",
    "authorization",
    "credential",
    "api_key",
    "apikey",
    "verifier",
    "cookie",
    "session",
)
SECRET_KEYS = {"key", "sig", "signature", "code", "refresh", "access"}
# Values that look like credentials wherever they appear: API tokens (bdnn_…), refresh tokens
# of apps (bdnr_…) and JWTs.
SECRET_VALUE = re.compile(
    r"bdn[nr]_[A-Za-z0-9_-]{6,}|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?:\.[A-Za-z0-9_-]*)?"
)
# Texts kept in full (up to MAX_NAME characters); other texts longer than MAX_SHORT characters
# are stored as their length only.
NAME_KEYS = {"name", "class_name", "label", "filename"}
# Texts only stored as their length: free text of the user (and e-mail addresses of others).
LENGTH_ONLY_KEYS = {"description", "text", "user", "email", "query"}
UUID = re.compile(r"^[0-9a-fA-F]{8}-?([0-9a-fA-F]{4}-?){3}[0-9a-fA-F]{12}$")
CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def is_secret_key(key: str) -> bool:
    key = key.lower()
    return key in SECRET_KEYS or any(word in key for word in SECRET_WORDS)


def clean_text(value: Any, limit: int) -> str:
    """One line of text, without secrets or control characters, at most ``limit`` long."""
    text = str(value).encode("utf-8", "replace").decode("utf-8")  # no lone surrogates
    text = SECRET_VALUE.sub(REDACTED, text)
    text = " ".join(CONTROL_CHARACTERS.sub(" ", text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def count(number: int, word: str, words: str | None = None) -> str:
    return f"{number:,} {word if number == 1 else (words or word + 's')}"


def size_text(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 10 * 1024:
        return f"{size / 1024:.1f} KB"
    if size < 1024**2:
        return f"{size / 1024:.0f} KB"
    return f"{size / 1024**2:.1f} MB"


def csv_summary(text: str) -> str:
    """'48 KB, 801 lines, 9 columns': the size of a CSV file, never its content."""
    size = len(text.encode("utf-8", "replace"))
    lines = [line for line in text.splitlines() if line.strip()]
    first = lines[0] if lines else ""
    separators = max((first.count(sep) for sep in (",", ";", "\t")), default=0)
    columns = separators + 1 if first else 0
    return f"{size_text(size)}, {count(len(lines), 'line')}, {count(columns, 'column')}"


def graph_summary(graph: Any) -> str:
    """'5 layers (Input, Linear ×2, ReLU, Softmax), 4 connections'."""
    if not isinstance(graph, dict):
        return shape_of(graph)
    nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
    edges = graph.get("edges") if isinstance(graph.get("edges"), list) else []
    types: dict[str, int] = {}
    for node in nodes:
        kind = node.get("type") if isinstance(node, dict) else None
        if isinstance(kind, str):
            kind = clean_text(kind, MAX_SHORT)
            types[kind] = types.get(kind, 0) + 1
    shown = [f"{kind} ×{n}" if n > 1 else kind for kind, n in list(types.items())[:6]]
    if len(types) > 6:
        shown.append("…")
    text = count(len(nodes), "layer")
    if shown:
        text += f" ({', '.join(shown)})"
    return f"{text}, {count(len(edges), 'connection')}"


def layer_count(graph: Any) -> int | None:
    if isinstance(graph, dict) and isinstance(graph.get("nodes"), list):
        return len(graph["nodes"])
    return None


def _dimensions(value: Any) -> list[int]:
    dims = []
    while isinstance(value, list) and len(dims) < 4:
        dims.append(len(value))
        value = value[0] if value else None
    return dims


def inputs_summary(inputs: list) -> str:
    """'3 samples with 8 values' or '2 samples of 1×28×28': how many, never the values."""
    text = count(len(inputs), "sample")
    first = inputs[0] if inputs else None
    if isinstance(first, dict):
        return f"{text} with {count(len(first), 'column')}"
    dims = _dimensions(first)
    if len(dims) == 1:
        return f"{text} with {count(dims[0], 'value')}"
    if dims:
        return f"{text} of {'×'.join(str(d) for d in dims)}"
    return text


def shape_of(value: Any) -> str:
    """What a value is, without its content."""
    if isinstance(value, str):
        return f"text, {count(len(value), 'character')}"
    if isinstance(value, (list, tuple)):
        return f"list of {count(len(value), 'item')}"
    if isinstance(value, Mapping):
        return count(len(value), "field")
    return type(value).__name__


def without_query(url: str) -> str:
    """Links are stored without query and fragment, which can hold keys (as in the Data tab)."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return shape_of(url)
    return urlunsplit((parts.scheme, parts.netloc.rpartition("@")[2], parts.path, "", ""))


def _number(value: int | float) -> int | float | str:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # JSON (and PostgreSQL) have no NaN or infinity
    if isinstance(value, int) and not -(2**53) <= value <= 2**53:
        return str(value)
    return value


def summarize_value(key: str, value: Any, depth: int = 0) -> Any:
    """A stored stand-in for one argument (see the module docstring)."""
    name = key.lower()
    if is_secret_key(name):
        return REDACTED
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return _number(value)
    if isinstance(value, UploadedFile):
        return f"{clean_text(value.name or 'file', MAX_SHORT)}, {size_text(value.size or 0)}"
    if name == "graph":
        return graph_summary(value)
    if name == "csv" and isinstance(value, str):
        return csv_summary(value)
    if name == "inputs" and isinstance(value, list):
        return inputs_summary(value)
    if isinstance(value, str):
        if name == "url":
            return clean_text(without_query(value), MAX_NAME)
        if name in LENGTH_ONLY_KEYS:
            return count(len(value), "character")
        if name in NAME_KEYS:
            return clean_text(value, MAX_NAME)
        if len(value) > MAX_SHORT:
            return count(len(value), "character")
        return clean_text(value, MAX_SHORT)
    if isinstance(value, Mapping):
        if depth >= 1:
            return count(len(value), "field")
        items = list(value.items())
        summary = {
            clean_text(sub, MAX_SHORT): summarize_value(str(sub), item, depth + 1)
            for sub, item in items[:MAX_KEYS]
        }
        if len(items) > MAX_KEYS:
            summary["…"] = f"{len(items) - MAX_KEYS} more"
        return summary
    if isinstance(value, (list, tuple)):
        if len(value) <= 8 and all(
            item is None or isinstance(item, (bool, int, float)) for item in value
        ):
            return [
                item if item is None or isinstance(item, bool) else _number(item) for item in value
            ]
        return f"list of {count(len(value), 'item')}"
    return type(value).__name__


def summarize(arguments: Any) -> dict[str, Any]:
    """The arguments of a call as they are stored: ``{"name": "Iris", "csv": "4 KB, …"}``."""
    if arguments is None:
        return {}
    if not isinstance(arguments, Mapping):
        return {"body": shape_of(arguments)}
    keys = list(arguments.keys())
    summary = {
        clean_text(key, MAX_SHORT): summarize_value(str(key), arguments.get(key))
        for key in keys[:MAX_KEYS]
    }
    if len(keys) > MAX_KEYS:
        summary["…"] = f"{len(keys) - MAX_KEYS} more"
    return summary


# --------------------------------------------------------------------------- #
# Describing an action: a sentence and what it touched
# --------------------------------------------------------------------------- #


@dataclass
class Described:
    """What an action did, as an infinitive ("save network “X” (#3)"): the summary is
    "Saved network …" when it worked and "Could not save network …" when it failed."""

    phrase: str
    # Appended either way, e.g. the size of an upload.
    detail: str = ""
    # Appended when it worked, e.g. ": valid" or ", now version 4".
    result: str = ""
    targets: list[dict[str, Any]] = field(default_factory=list)
    # The network it was about, if it is not among the targets (e.g. the project of a run).
    project_id: int | None = None
    # The whole summary when it worked, if it is not the phrase in the past ("Listed 3 …").
    done: str | None = None

    def summary(self, ok: bool) -> str:
        if not ok:
            return f"Could not {self.phrase}{self.detail}"
        if self.done is not None:
            return self.done
        return f"{past_tense(self.phrase)}{self.detail}{self.result}"


PAST_TENSE = {
    "add": "Added",
    "analyze": "Analyzed",
    "call": "Called",
    "change": "Changed",
    "check": "Checked",
    "comment": "Commented",
    "create": "Created",
    "delete": "Deleted",
    "download": "Downloaded",
    "duplicate": "Duplicated",
    "export": "Exported",
    "generate": "Generated",
    "get": "Got",
    "import": "Imported",
    "invite": "Invited",
    "list": "Listed",
    "look": "Looked",
    "name": "Named",
    "open": "Opened",
    "predict": "Predicted",
    "preview": "Previewed",
    "read": "Read",
    "remove": "Removed",
    "rename": "Renamed",
    "restore": "Restored",
    "save": "Saved",
    "send": "Sent",
    "share": "Shared",
    "start": "Started",
    "stop": "Stopped",
    "upload": "Uploaded",
    "use": "Used",
}


def past_tense(phrase: str) -> str:
    verb, _, rest = phrase.partition(" ")
    done = PAST_TENSE.get(verb, verb.capitalize())
    return f"{done} {rest}" if rest else done


def valid_id(value: Any) -> int | None:
    """A database id from an argument (agents may send "12" or nonsense)."""
    if isinstance(value, str) and value.isdigit() and len(value) < 18:
        value = int(value)
    if isinstance(value, int) and not isinstance(value, bool) and 0 < value < 2**63:
        return value
    return None


def target(kind: str, object_id: Any, name: Any = None) -> dict[str, Any]:
    number = valid_id(object_id) if kind != "run" else None
    ref: dict[str, Any] = {
        "kind": kind,
        "id": number if number is not None else clean_text(object_id, 64),
    }
    if isinstance(name, str) and name.strip():
        ref["name"] = clean_text(name, MAX_NAME)
    return ref


def _quoted(ref: dict[str, Any] | None) -> str:
    return f"“{ref['name']}”" if ref and ref.get("name") else ""


NOUNS = {
    "project": "network",
    "run": "training",
    "dataset": "dataset",
    "data_source": "data file",
    "data_prep": "data preparation",
}


def phrase_of(ref: dict[str, Any] | None, kind: str) -> str:
    """'network “House prices” (#12)', 'training “GP 1”', or 'a network' when unknown."""
    noun = NOUNS[kind]
    if ref is None:
        return f"a {noun} run" if kind == "run" else f"a {noun}"
    name = _quoted(ref)
    if kind == "run":
        return f"{noun} {name}" if name else "a training run"
    number = ref.get("id") if isinstance(ref.get("id"), int) else None
    if name:
        return f"{noun} {name} (#{number})" if number is not None else f"{noun} {name}"
    return f"{noun} #{number}" if number is not None else f"a {noun}"


def _safe(lookup: Callable[[], Any]) -> Any:
    try:
        return lookup()
    except Exception:  # e.g. a malformed id
        return None


def project_ref(user, project_id: Any, name: Any = None) -> dict[str, Any] | None:
    """The network with this id, named if the user can see it."""
    number = valid_id(project_id)
    if number is None:
        return None
    if not name:
        from .models import Project

        name = (
            Project.objects.filter(Q(owner=user) | Q(members__user=user), pk=number)
            .values_list("name", flat=True)
            .first()
        )
    return target("project", number, name)


def run_ref(user, run_id: Any, name: Any = None) -> dict[str, Any] | None:
    if not isinstance(run_id, str) or not run_id:
        return None
    if not name:
        from .shared_data import visible_runs

        name = _safe(
            lambda: visible_runs(user).filter(pk=run_id).values_list("name", flat=True).first()
        )
        if not name and not UUID.match(run_id):
            return None  # not a run id at all (agents make mistakes)
    return target("run", run_id, name)


def dataset_ref(user, dataset_id: Any, name: Any = None) -> dict[str, Any] | None:
    number = valid_id(dataset_id)
    if number is None:
        return None
    if not name:
        from .shared_data import visible_datasets

        name = visible_datasets(user).filter(pk=number).values_list("name", flat=True).first()
    return target("dataset", number, name)


def training_data(user, ref: Any) -> tuple[dict[str, Any] | None, str]:
    """The dataset a training uses (``{"kind", "id", "name"?}``): its target and words."""
    if not isinstance(ref, Mapping):
        return None, "a dataset"
    kind = ref.get("kind")
    if kind == "upload":
        dataset = dataset_ref(user, ref.get("id"), ref.get("name"))
        return dataset, phrase_of(dataset, "dataset") if dataset else "an uploaded dataset"
    if kind == "builtin":
        from brickdnn.training.datasets import BUILTIN_DATASETS

        builtin = BUILTIN_DATASETS.get(ref.get("id")) if isinstance(ref.get("id"), str) else None
        name = ref.get("name") or (builtin.name if builtin else ref.get("id"))
        return (
            None,
            f"the built-in dataset “{clean_text(name, MAX_NAME)}”"
            if name
            else "a built-in dataset",
        )
    if kind == "random":
        return None, "random demo data"
    return None, "a dataset"


# The most telling result of a run (as in the training panel): the test result if there is
# one, otherwise the validation result.
HEADLINE_METRICS = (
    ("test_accuracy", "test accuracy"),
    ("test_r2", "test R²"),
    ("test_rmse", "test RMSE"),
    ("test_loss", "test loss"),
    ("val_accuracy", "validation accuracy"),
    ("val_r2", "validation R²"),
    ("val_rmse", "validation RMSE"),
    ("val_loss", "validation loss"),
)


def _headline(metrics: Any) -> str:
    if not isinstance(metrics, Mapping):
        return ""
    for key, label in HEADLINE_METRICS:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            number = f"{value * 100:.1f} %" if key.endswith("accuracy") else f"{value:.3g}"
            return f", {label} {number}"
    return ""


def _run_progress(run: Mapping) -> str:
    """': running, epoch 3 of 20' or ': completed, test accuracy 96.7 %'."""
    status = run.get("status")
    if not isinstance(status, str):
        return ""
    text = f": {status}"
    tuning = run.get("tuning")
    if status == "running" and isinstance(tuning, Mapping) and tuning.get("trials"):
        text += f", trial {run.get('trials_done', 0)} of {tuning['trials']}"
    elif status == "running" and run.get("family") != "classical" and run.get("epochs"):
        text += f", epoch {run.get('epochs_done', 0)} of {run['epochs']}"
    elif status in ("completed", "stopped"):
        text += _headline(run.get("metrics"))
    return text


def _run_targets(user, run_id: Any, result: Any) -> tuple[dict[str, Any] | None, int | None]:
    run = result if isinstance(result, Mapping) else {}
    ref = run_ref(user, run.get("id", run_id), run.get("name"))
    return ref, valid_id(run.get("project"))


def _layers(args: Mapping) -> str:
    layers = layer_count(args.get("graph"))
    return f"a network of {count(layers, 'layer')}" if layers is not None else "a network"


def _listed(result: Any, word: str) -> str | None:
    """'Listed 3 saved networks' for a list answer."""
    return f"Listed {count(len(result), word)}" if isinstance(result, list) else None


# One per MCP tool: (user, arguments, result or None if it failed) -> Described.


def _list_layer_types(user, args, result) -> Described:
    return Described("look up the layer types")


def _list_examples(user, args, result) -> Described:
    return Described("look up the example networks")


def _analyze_network(user, args, result) -> Described:
    text = ""
    if isinstance(result, Mapping):
        if result.get("valid"):
            params = result.get("total_params")
            text = ": valid" + (f", {params:,} parameters" if isinstance(params, int) else "")
        else:
            text = f": {count(len(result.get('errors') or []), 'error')}"
    return Described(f"analyze {_layers(args)}", result=text)


def _generate_code(user, args, result) -> Described:
    language = (
        "scikit-learn" if isinstance(result, str) and "sklearn" in result[:5000] else "PyTorch"
    )
    return Described(f"generate code for {_layers(args)}", result=f" ({language})")


def _list_projects(user, args, result) -> Described:
    return Described("list the saved networks", done=_listed(result, "saved network"))


def _get_project(user, args, result) -> Described:
    data = result if isinstance(result, Mapping) else {}
    ref = project_ref(user, data.get("id", args.get("id")), data.get("name"))
    return Described(f"open {phrase_of(ref, 'project')}", targets=[ref] if ref else [])


def _save_project(user, args, result) -> Described:
    data = result if isinstance(result, Mapping) else {}
    layers = layer_count(args.get("graph"))
    detail = f" with {count(layers, 'layer')}" if layers is not None else ""
    if args.get("id") is None:  # a new network
        ref = project_ref(user, data["id"], data.get("name")) if data.get("id") else None
        name = data.get("name") or args.get("name")
        what = (
            phrase_of(ref, "project")
            if ref
            else (
                f"network “{clean_text(name, MAX_NAME)}”"
                if isinstance(name, str) and name
                else "a network"
            )
        )
        return Described(f"create {what}", detail, targets=[ref] if ref else [])
    ref = project_ref(user, data.get("id", args.get("id")), data.get("name"))
    version = data.get("version")
    result_text = f", now version {version}" if isinstance(version, int) else ""
    return Described(f"save {phrase_of(ref, 'project')}", detail, result_text, [ref] if ref else [])


def _delete_project(user, args, result) -> Described:
    data = result if isinstance(result, Mapping) else {}
    ref = project_ref(user, data.get("id", args.get("id")), data.get("name"))
    return Described(f"delete {phrase_of(ref, 'project')}", targets=[ref] if ref else [])


def _training_options(user, args, result) -> Described:
    return Described("look up the training options and datasets")


def _upload_dataset(user, args, result) -> Described:
    data = result.get("dataset") if isinstance(result, Mapping) else None
    ref = dataset_ref(user, data.get("id"), data.get("name")) if isinstance(data, Mapping) else None
    name = args.get("name")
    if ref:
        what = phrase_of(ref, "dataset")
    elif isinstance(name, str) and name.strip():
        what = f"dataset “{clean_text(name, MAX_NAME)}”"
    else:
        what = "a dataset"
    detail = f": {csv_summary(args['csv'])}" if isinstance(args.get("csv"), str) else ""
    targets = [ref] if ref else []
    project = project_ref(user, args.get("project_id"))
    if project:
        targets.append(project)
        what += f" in {phrase_of(project, 'project')}"
    return Described(f"upload {what}", detail, targets=targets)


def _check_training(user, args, result) -> Described:
    dataset, words = training_data(user, args.get("dataset"))
    text = ""
    if isinstance(result, Mapping):
        problems = len(result.get("errors") or [])
        text = ": it fits" if result.get("ok") else f": {count(problems, 'problem')}"
    return Described(
        f"check {_layers(args)} against {words}", result=text, targets=[dataset] if dataset else []
    )


def _start_training(user, args, result) -> Described:
    run = result if isinstance(result, Mapping) else {}
    ref = run_ref(user, run.get("id"), run.get("name")) if run.get("id") else None
    if ref is None:
        name = (
            args.get("name") if isinstance(args.get("name"), str) and args["name"].strip() else None
        )
        what = f"training “{clean_text(name, MAX_NAME)}”" if name else "a training"
    else:
        what = phrase_of(ref, "run")
    dataset, words = training_data(user, run.get("dataset") or args.get("dataset"))
    targets = [item for item in (ref, dataset) if item]
    project = project_ref(user, run.get("project", args.get("project_id")))
    phrase = f"start {what} on {words}"
    if project:
        targets.append(project)
        phrase += f" in {phrase_of(project, 'project')}"
    epochs = run.get("epochs")
    text = (
        f", {count(epochs, 'epoch')}"
        if isinstance(epochs, int) and run.get("family") != "classical"
        else ""
    )
    if isinstance(run.get("tuning"), Mapping) and run["tuning"].get("trials"):
        text = f", hyperparameter search with {count(run['tuning']['trials'], 'trial')}"
    return Described(phrase, result=text, targets=targets)


def _list_runs(user, args, result) -> Described:
    return Described("list the training runs", done=_listed(result, "training run"))


def _get_run(user, args, result) -> Described:
    ref, project = _run_targets(user, args.get("id"), result)
    progress = _run_progress(result) if isinstance(result, Mapping) else ""
    return Described(
        f"check {phrase_of(ref, 'run')}",
        result=progress,
        targets=[ref] if ref else [],
        project_id=project,
    )


def _stop_training(user, args, result) -> Described:
    ref, project = _run_targets(user, args.get("id"), result)
    return Described(
        f"stop {phrase_of(ref, 'run')}", targets=[ref] if ref else [], project_id=project
    )


def _weights_download(user, args, result) -> Described:
    ref, project = _run_targets(user, args.get("id"), None)
    return Described(
        f"get a download link for the weights of {phrase_of(ref, 'run')}",
        targets=[ref] if ref else [],
        project_id=project,
    )


def _predict(user, args, result) -> Described:
    ref, project = _run_targets(user, args.get("run_id"), None)
    inputs = args.get("inputs")
    samples = count(len(inputs), "sample") if isinstance(inputs, list) else "samples"
    return Described(
        f"predict {samples} with {phrase_of(ref, 'run')}",
        targets=[ref] if ref else [],
        project_id=project,
    )


def _delete_run(user, args, result) -> Described:
    data = result if isinstance(result, Mapping) else {}
    ref = run_ref(user, data.get("id", args.get("id")), data.get("name"))
    return Described(f"delete {phrase_of(ref, 'run')}", targets=[ref] if ref else [])


def _delete_dataset(user, args, result) -> Described:
    data = result if isinstance(result, Mapping) else {}
    ref = dataset_ref(user, data.get("id", args.get("id")), data.get("name"))
    return Described(f"delete {phrase_of(ref, 'dataset')}", targets=[ref] if ref else [])


TOOL_DESCRIPTIONS: dict[str, Callable[[Any, Mapping, Any], Described]] = {
    "list_layer_types": _list_layer_types,
    "list_examples": _list_examples,
    "analyze_network": _analyze_network,
    "generate_pytorch_code": _generate_code,
    "list_projects": _list_projects,
    "get_project": _get_project,
    "save_project": _save_project,
    "delete_project": _delete_project,
    "get_training_options": _training_options,
    "upload_dataset": _upload_dataset,
    "check_training": _check_training,
    "start_training": _start_training,
    "list_training_runs": _list_runs,
    "get_training_run": _get_run,
    "stop_training": _stop_training,
    "get_weights_download": _weights_download,
    "predict": _predict,
    "delete_training_run": _delete_run,
    "delete_dataset": _delete_dataset,
}


def describe_tool(user, tool: str, args: Any, result: Any) -> Described:
    describe = TOOL_DESCRIPTIONS.get(tool)
    if describe is None:
        return Described(f"call an unknown tool “{clean_text(tool, MAX_SHORT)}”")
    return describe(user, args if isinstance(args, Mapping) else {}, result)


# REST requests ------------------------------------------------------------- #

# Routes that do what an MCP tool does: (route, method) -> (tool, arguments from the URL
# kwargs and the body). Their results are alike, too.
REST_AS_TOOL: dict[tuple[str, str], tuple[str, Callable[[Mapping, Mapping], dict]]] = {
    ("layers", "GET"): ("list_layer_types", lambda kw, body: {}),
    ("examples", "GET"): ("list_examples", lambda kw, body: {}),
    ("analyze", "POST"): ("analyze_network", lambda kw, body: {"graph": body.get("graph", body)}),
    ("generate", "POST"): ("generate_pytorch_code", lambda kw, body: {"graph": body.get("graph")}),
    ("training", "GET"): ("get_training_options", lambda kw, body: {}),
    ("training-check", "POST"): ("check_training", lambda kw, body: dict(body)),
    ("project-list", "GET"): ("list_projects", lambda kw, body: {}),
    ("project-list", "POST"): (
        "save_project",
        lambda kw, body: {k: body[k] for k in ("name", "description", "graph") if k in body},
    ),
    ("project-detail", "GET"): ("get_project", lambda kw, body: {"id": kw.get("pk")}),
    ("project-detail", "PATCH"): ("save_project", lambda kw, body: {**body, "id": kw.get("pk")}),
    ("project-detail", "PUT"): ("save_project", lambda kw, body: {**body, "id": kw.get("pk")}),
    ("project-detail", "DELETE"): ("delete_project", lambda kw, body: {"id": kw.get("pk")}),
    ("training-run-list", "GET"): ("list_training_runs", lambda kw, body: {}),
    ("training-run-list", "POST"): (
        "start_training",
        lambda kw, body: {**body, "project_id": body.get("project")},
    ),
    ("training-run-detail", "GET"): ("get_training_run", lambda kw, body: {"id": kw.get("pk")}),
    ("training-run-detail", "DELETE"): (
        "delete_training_run",
        lambda kw, body: {"id": kw.get("pk")},
    ),
    ("training-run-stop", "POST"): ("stop_training", lambda kw, body: {"id": kw.get("pk")}),
    ("training-run-predict", "POST"): (
        "predict",
        lambda kw, body: {"run_id": kw.get("pk"), "inputs": body.get("inputs")},
    ),
    ("training-dataset-detail", "DELETE"): (
        "delete_dataset",
        lambda kw, body: {"id": kw.get("pk")},
    ),
}

# The other routes: (route, method) -> phrase; {object} is the network, run, dataset, data
# file or data preparation of the URL.
ROUTE_KINDS = {
    "training-dataset": "dataset",
    "training-run": "run",
    "data-source": "data_source",
    "data-prep": "data_prep",
    "project": "project",
}
REST_PHRASES: dict[tuple[str, str], str] = {
    ("training-dataset-list", "GET"): "list the datasets",
    ("training-dataset-list", "POST"): "upload {object}",
    ("training-dataset-detail", "GET"): "open {object}",
    ("training-run-weights", "GET"): "download the weights of {object}",
    ("data-source-list", "GET"): "list the data files",
    ("data-source-list", "POST"): "import {object}",
    ("data-source-detail", "GET"): "open {object}",
    ("data-source-detail", "PATCH"): "rename {object}",
    ("data-source-detail", "DELETE"): "delete {object}",
    ("data-prep-list", "GET"): "list the data preparations",
    ("data-prep-list", "POST"): "create {object}",
    ("data-prep-detail", "GET"): "open {object}",
    ("data-prep-detail", "PATCH"): "change {object}",
    ("data-prep-detail", "PUT"): "change {object}",
    ("data-prep-detail", "DELETE"): "delete {object}",
    ("data-prep-preview", "POST"): "preview {object}",
    ("data-prep-export", "POST"): "export {object} as a training dataset",
    ("data-prep-download", "GET"): "download {object}",
    ("project-duplicate", "POST"): "duplicate {object}",
    ("project-members", "GET"): "list the members of {object}",
    ("project-members", "POST"): "invite someone to {object}",
    ("project-member", "PATCH"): "change a member's role in {object}",
    ("project-member", "DELETE"): "remove a member from {object}",
    ("project-link", "POST"): "change the share link of {object}",
    ("project-data", "GET"): "list the data shared in {object}",
    ("project-data", "POST"): "share data in {object}",
    ("project-data-item", "DELETE"): "stop sharing data in {object}",
    ("project-versions", "GET"): "open the history of {object}",
    ("project-versions", "POST"): "name a version of {object}",
    ("project-version-detail", "GET"): "open a version of {object}",
    ("project-version-detail", "PATCH"): "rename a version of {object}",
    ("project-restore", "POST"): "restore a version of {object}",
    ("project-comments", "GET"): "read the comments on {object}",
    ("project-comments", "POST"): "comment on {object}",
    ("project-comment", "PATCH"): "change a comment on {object}",
    ("project-comment", "DELETE"): "delete a comment on {object}",
    ("data", "GET"): "look up the data preparation options",
    ("data-steps", "GET"): "look up the data preparation steps",
    ("me", "GET"): "read the account details",
    ("health", "GET"): "check the server status",
    ("api-root", "GET"): "read the API index",
    ("shared-project", "GET"): "open a network shared by link",
    ("shared-project-duplicate", "POST"): "duplicate a network shared by link",
    ("legal", "GET"): "read the list of legal pages",
    ("legal-page", "GET"): "read a legal page",
}


def source_ref(user, source_id: Any, name: Any = None) -> dict[str, Any] | None:
    number = valid_id(source_id)
    if number is None:
        return None
    if not name:
        from .models import DataSource

        name = (
            DataSource.objects.filter(owner=user, pk=number).values_list("name", flat=True).first()
        )
    return target("data_source", number, name)


def prep_ref(user, prep_id: Any, name: Any = None) -> dict[str, Any] | None:
    number = valid_id(prep_id)
    if number is None:
        return None
    if not name:
        from .shared_data import visible_preps

        name = visible_preps(user).filter(pk=number).values_list("name", flat=True).first()
    return target("data_prep", number, name)


REFS: dict[str, Callable[..., dict[str, Any] | None]] = {
    "project": project_ref,
    "run": lambda user, run_id, name=None: run_ref(user, str(run_id), name),
    "dataset": dataset_ref,
    "data_source": source_ref,
    "data_prep": prep_ref,
}


def _route_kind(route: str) -> str | None:
    return next(
        (kind for prefix, kind in ROUTE_KINDS.items() if route.startswith(f"{prefix}-")), None
    )


def _route_object(user, route: str, kwargs: Mapping, data: Any) -> dict[str, Any] | None:
    """The network, run, dataset ... a REST route is about: from the URL, or for a new one
    from the answer."""
    kind = _route_kind(route)
    if kind is None:
        return None
    data = data if isinstance(data, Mapping) else {}
    object_id = kwargs.get("pk")
    if object_id is None and route.endswith("-list"):
        object_id = data.get("id")  # just created
    if object_id is None:
        return None
    # The answer is the object itself for detail routes, creating and e.g. stopping a run.
    name = data.get("name") if str(data.get("id")) == str(object_id) else None
    return REFS[kind](user, object_id, name)


def describe_rest(
    user, route: str, method: str, kwargs: Mapping, data: Any, ok: bool, body: Any
) -> Described:
    body = body if isinstance(body, Mapping) else {}
    as_tool = REST_AS_TOOL.get((route, method))
    if as_tool is not None:
        tool, arguments = as_tool
        result = data if ok else None
        if tool == "generate_pytorch_code" and isinstance(result, Mapping):
            result = result.get("code")
        return describe_tool(user, tool, arguments(kwargs, body), result)
    phrase = REST_PHRASES.get((route, method))
    if phrase is None:
        return Described(f"call {method} {clean_text(route or 'an unknown address', MAX_SHORT)}")
    kind = _route_kind(route)
    ref = _route_object(user, route, kwargs, data if ok else None)
    name = body.get("name")
    if ref is None and kind and isinstance(name, str) and name.strip():
        what = f"{NOUNS[kind]} “{clean_text(name, MAX_NAME)}”"  # e.g. an upload that failed
    else:
        what = phrase_of(ref, kind) if kind else ""
    return Described(phrase.format(object=what), targets=[ref] if ref else [])


def _error_text(data: Any, status_code: int) -> str:
    """The message of an error answer of the API."""
    if isinstance(data, Mapping):
        for key in ("detail", "error_description", "error"):
            if isinstance(data.get(key), str) and data[key]:
                return data[key]
        for key, value in data.items():
            first = value[0] if isinstance(value, list) and value else value
            return f"{key}: {first}"
    if isinstance(data, list) and data:
        return str(data[0])
    return f"HTTP {status_code}"


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #


@dataclass
class _Call:
    """An action as it ended; described and written after the response."""

    token: ApiToken
    user_agent: str
    source: str
    tool: str
    method: str
    writes: bool
    error: str | None
    status_code: int | None
    duration_ms: int
    finished_at: datetime
    # (user) -> (what it did, the summarised arguments)
    describe: Callable[[Any], tuple[Described, dict[str, Any]]]


def _pending(request) -> list[_Call]:
    """The entries waiting to be written for a Django request."""
    calls = getattr(request, "_agent_calls", None)
    if calls is None:
        calls = []
        request._agent_calls = calls
    return calls


def _user_agent(request) -> str:
    return clean_text(request.headers.get("User-Agent", ""), MAX_USER_AGENT)


def record_tool_call(
    request: Request,
    tool: Any,
    args: Any,
    *,
    started: float,
    result: Any = None,
    error: str | None = None,
    writes: bool = False,
) -> None:
    """Queue an entry for an MCP tool call. ``error``: the message the assistant got."""
    try:
        token = request.auth
        if not enabled() or not isinstance(token, ApiToken):
            return
        name = tool if isinstance(tool, str) else shape_of(tool)
        _pending(request._request).append(
            _Call(
                token=token,
                user_agent=_user_agent(request),
                source="mcp",
                tool=clean_text(name, 64),
                method="",
                writes=writes,
                error=error,
                status_code=None,
                duration_ms=round((time.monotonic() - started) * 1000),
                finished_at=timezone.now(),
                describe=lambda user: (describe_tool(user, name, args, result), summarize(args)),
            )
        )
    except Exception:
        logger.exception("Could not record an MCP tool call in the agent activity log")


def _record_request(request, response, started: float) -> None:
    # Set by ApiTokenAuthentication (api_tokens.py) for every request it authenticates, also
    # for those it refuses afterwards; the DRF request is a weak reference.
    token = getattr(request, "agent_token", None)
    if not isinstance(token, ApiToken) or not enabled():
        return
    match = request.resolver_match
    route = (match.url_name if match is not None else None) or ""
    status_code = response.status_code
    data = getattr(response, "data", None)  # answers of DRF views
    if route in MCP_URL_NAMES:
        # Tool calls are logged one by one; only a request that never got to them is logged
        # here (too large, not JSON, wrong origin ...).
        if status_code < 400 or getattr(request, "_agent_calls", None):
            return
        source, method, writes = "mcp", "", False
        error = _mcp_error(response)
        route = "mcp"
    else:
        source, method = "api", request.method or ""
        writes = method not in permissions.SAFE_METHODS and route not in READ_ONLY_ROUTES
        error = (
            clean_text(_error_text(data, status_code), MAX_ERROR) if status_code >= 400 else None
        )
    kwargs = dict(match.kwargs) if match is not None else {}
    query = {key: request.GET.get(key) for key in list(request.GET.keys())[:MAX_KEYS]}
    drf_ref = getattr(request, "agent_request", None)
    drf_request = drf_ref() if drf_ref is not None else None
    # The body as the view parsed it (not parsed again: that would read the request now).
    body = getattr(drf_request, "_full_data", Empty) if drf_request is not None else Empty
    body = None if body is Empty or source == "mcp" else body
    ok = error is None

    def describe(user) -> tuple[Described, dict[str, Any]]:
        if source == "mcp":
            return Described("send a request to the MCP server"), {}
        arguments = {}
        if kwargs:
            ids = {"id" if k == "pk" else k: valid_id(v) or v for k, v in kwargs.items()}
            arguments.update(summarize(ids))
        if query:
            arguments["query"] = summarize(query)
        if body:
            arguments.update(summarize(body))
        return describe_rest(user, route, method, kwargs, data, ok, body), arguments

    _pending(request).append(
        _Call(
            token=token,
            user_agent=_user_agent(request),
            source=source,
            tool=clean_text(route or "unknown", 64),
            method=method,
            writes=writes,
            error=error,
            status_code=status_code if source == "api" else None,
            duration_ms=round((time.monotonic() - started) * 1000),
            finished_at=timezone.now(),
            describe=describe,
        )
    )


def _mcp_error(response) -> str:
    try:
        message = json.loads(response.content)["error"]["message"]
    except Exception:
        message = f"HTTP {response.status_code}"
    return clean_text(message, MAX_ERROR)


def _build(call: _Call) -> AgentAction:
    token = call.token
    described, arguments = call.describe(token.owner)
    ok = call.error is None
    targets = [ref for ref in described.targets if ref]
    project_id = next(
        (ref["id"] for ref in targets if ref["kind"] == "project" and isinstance(ref["id"], int)),
        described.project_id,
    )
    is_app = token.connection_id is not None
    return AgentAction(
        owner_id=token.owner_id,
        created_at=call.finished_at,
        source=call.source,
        agent_kind="app" if is_app else "token",
        agent_id=token.connection_id if is_app else token.pk,
        agent_name=clean_text(token.name, 200),
        token_prefix=token.prefix[:20],
        user_agent=call.user_agent,
        tool=call.tool,
        method=call.method,
        writes=call.writes,
        outcome="ok" if ok else "error",
        error="" if ok else clean_text(call.error, MAX_ERROR),
        status_code=call.status_code,
        duration_ms=max(0, min(call.duration_ms, 2**31 - 1)),
        summary=clean_text(described.summary(ok), MAX_SUMMARY),
        arguments=arguments,
        targets=targets,
        project_id=valid_id(project_id),
    )


def _write(calls: list[_Call]) -> None:
    entries = []
    for call in calls:
        try:
            entries.append(_build(call))
        except Exception:
            logger.exception("Could not describe an agent action for the activity log")
    if entries:
        try:
            AgentAction.objects.bulk_create(entries)
        except Exception:
            logger.exception("Could not write %d entries of the agent activity log", len(entries))


def _defer(response, calls: list[_Call]) -> None:
    """Write the entries once the response has been sent: Django runs a response's closers
    after the server has sent it (and before the database connection is given back)."""
    closers = getattr(response, "_resource_closers", None)
    if isinstance(closers, list):
        closers.append(lambda: _write(calls))
    else:
        _write(calls)


class AgentActivityMiddleware:
    """Logs requests made with API tokens and app tokens, and writes the queued entries of a
    request (also the MCP tool calls) after its response."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        started = time.monotonic()
        response = self.get_response(request)
        try:
            _record_request(request, response, started)
            calls = getattr(request, "_agent_calls", None)
            if calls:
                _defer(response, calls)
        except Exception:
            logger.exception("Could not record a request in the agent activity log")
        return response


# --------------------------------------------------------------------------- #
# Deleting old entries (manage.py purge_old_data)
# --------------------------------------------------------------------------- #


def purge(now: datetime, days: int, dry_run: bool = False) -> int:
    """Delete the entries older than ``days`` days (all of them if 0: the log is off) and,
    per account, those beyond the newest MAX_ENTRIES_PER_USER. Returns how many."""
    entries = AgentAction.objects.all()
    old = entries if days <= 0 else entries.filter(created_at__lt=now - timedelta(days=days))
    deleted = old.count()
    if not dry_run:
        old.delete()
    if days <= 0:
        return deleted
    kept = entries.filter(created_at__gte=now - timedelta(days=days))
    crowded = kept.values("owner_id").annotate(n=Count("id")).filter(n__gt=MAX_ENTRIES_PER_USER)
    for row in crowded:
        mine = kept.filter(owner_id=row["owner_id"])
        oldest_kept = mine.order_by("-id").values_list("id", flat=True)[MAX_ENTRIES_PER_USER - 1]
        extra = mine.filter(id__lt=oldest_kept)
        deleted += extra.count()
        if not dry_run:
            extra.delete()
    return deleted


# --------------------------------------------------------------------------- #
# Export (accounts.py) and API
# --------------------------------------------------------------------------- #


def entry_data(entry: AgentAction) -> dict[str, Any]:
    """``AgentAction`` (frontend/src/api/types.ts)."""
    return {
        "id": entry.pk,
        "created_at": entry.created_at,
        "source": entry.source,
        "agent": {
            "kind": entry.agent_kind,
            "id": entry.agent_id,
            "key": f"{entry.agent_kind}:{entry.agent_id}",
            "name": entry.agent_name,
            "prefix": entry.token_prefix or None,
        },
        "user_agent": entry.user_agent,
        "tool": entry.tool,
        "method": entry.method,
        "writes": entry.writes,
        "outcome": entry.outcome,
        "error": entry.error,
        "status_code": entry.status_code,
        "duration_ms": entry.duration_ms,
        "summary": entry.summary,
        "arguments": entry.arguments,
        "targets": entry.targets,
        "project": entry.project_id,
    }


def export_entries(user) -> list[dict[str, Any]]:
    return [entry_data(entry) for entry in AgentAction.objects.filter(owner=user).order_by("id")]


class WebsiteLogin(IsLoggedInUser):
    message = "See the agent activity after logging in on the website; API tokens cannot."


AGENT_FILTER = re.compile(r"^(token|app):(\d{1,18})$")


def _int_param(params, name: str, low: int = 0, high: int = 2**63 - 1) -> int | None:
    value = params.get(name)
    if value in (None, ""):
        return None
    if not value.isdigit() or not low <= int(value) <= high:
        raise ValidationError({name: f"Must be a whole number from {low} to {high}."})
    return int(value)


def filter_entries(entries: QuerySet[AgentAction], params) -> QuerySet[AgentAction]:
    """``?agent=token:<id>|app:<id>&tool=&outcome=ok|error&source=mcp|api&writes=true
    &project=<id>&since=<date or time>``."""
    agent = params.get("agent")
    if agent:
        match = AGENT_FILTER.match(agent)
        if match is None:
            raise ValidationError({"agent": "Use token:<id> or app:<id>."})
        entries = entries.filter(agent_kind=match[1], agent_id=int(match[2]))
    if params.get("tool"):
        entries = entries.filter(tool=params["tool"][:64])
    outcome = params.get("outcome")
    if outcome:
        if outcome not in dict(AgentAction.OUTCOMES):
            raise ValidationError({"outcome": "Use ok or error."})
        entries = entries.filter(outcome=outcome)
    source = params.get("source")
    if source:
        if source not in dict(AgentAction.SOURCES):
            raise ValidationError({"source": "Use mcp or api."})
        entries = entries.filter(source=source)
    if params.get("writes") in ("true", "1"):
        entries = entries.filter(writes=True)
    project = _int_param(params, "project", 1)
    if project is not None:
        entries = entries.filter(project_id=project)
    since = params.get("since")
    if since:
        moment = _parse_moment(since)
        if moment is None:
            raise ValidationError({"since": "Use an ISO date or time, e.g. 2026-11-09T14:00Z."})
        entries = entries.filter(created_at__gte=moment)
    return entries


def _parse_moment(text: str) -> datetime | None:
    try:
        moment = parse_datetime(text.replace(" ", "+"))  # "+" in a query string is a space
        if moment is None:
            day = parse_date(text)
            moment = datetime(day.year, day.month, day.day) if day else None
    except ValueError:
        return None
    if moment is not None and timezone.is_naive(moment):
        moment = timezone.make_aware(moment, timezone.get_default_timezone())
    return moment


class AgentActivityView(APIView):
    """``GET /api/agent-activity/``: the user's agent actions, newest first.

    Pages: ``?limit=`` (default 50, at most 200) and ``?before=<id>`` for older entries;
    ``?after=<id>`` for entries newer than the ones shown (polling). Filters: see
    ``filter_entries``. ``latest_id`` is the newest entry of the user (whatever the filters),
    the next ``after``.
    """

    permission_classes = [WebsiteLogin]

    def get(self, request: Request) -> Response:
        params = request.query_params
        mine = AgentAction.objects.filter(owner=request.user)
        latest = mine.aggregate(latest=Max("id"))["latest"] or 0
        entries = filter_entries(mine, params).filter(id__lte=latest)
        limit = _int_param(params, "limit", 1, MAX_PAGE_SIZE) or PAGE_SIZE
        after = _int_param(params, "after")
        before = _int_param(params, "before")
        if after is not None:
            entries = entries.filter(id__gt=after)
        if before is not None:
            entries = entries.filter(id__lt=before)
        page = list(entries.order_by("-id")[: limit + 1])
        return Response(
            {
                "results": [entry_data(entry) for entry in page[:limit]],
                "has_more": len(page) > limit,
                "latest_id": latest,
                "retention_days": settings.AGENT_LOG_DAYS,
            }
        )


class AgentActivitySummaryView(APIView):
    """``GET /api/agent-activity/summary/``: which tokens and apps acted (for the filters and
    the API tokens page), the tools they used and the latest entry."""

    permission_classes = [WebsiteLogin]

    def get(self, request: Request) -> Response:
        user = request.user
        mine = AgentAction.objects.filter(owner=user)
        rows = list(
            mine.values("agent_kind", "agent_id")
            .annotate(
                name=Max("agent_name"),
                prefix=Max("token_prefix"),
                actions=Count("id"),
                errors=Count("id", filter=Q(outcome="error")),
                last_at=Max("created_at"),
                last_id=Max("id"),
            )
            .order_by("-last_id")
        )
        token_ids = [row["agent_id"] for row in rows if row["agent_kind"] == "token"]
        app_ids = [row["agent_id"] for row in rows if row["agent_kind"] == "app"]
        tokens = set(
            ApiToken.objects.filter(
                owner=user, connection__isnull=True, pk__in=token_ids
            ).values_list("pk", flat=True)
        )
        apps = set(
            OAuthConnection.objects.filter(owner=user, pk__in=app_ids).values_list("pk", flat=True)
        )
        agents = [
            {
                "kind": row["agent_kind"],
                "id": row["agent_id"],
                "key": f"{row['agent_kind']}:{row['agent_id']}",
                "name": row["name"],
                # Apps get a new access token every hour: their prefix says nothing.
                "prefix": row["prefix"] if row["agent_kind"] == "token" else None,
                "active": row["agent_id"] in (tokens if row["agent_kind"] == "token" else apps),
                "actions": row["actions"],
                "errors": row["errors"],
                "last_at": row["last_at"],
                "last_id": row["last_id"],
            }
            for row in rows
        ]
        tools = list(mine.values("tool").annotate(actions=Count("id")).order_by("tool"))
        latest = mine.order_by("-id").values("id", "created_at").first()
        return Response(
            {
                "agents": agents,
                "tools": tools,
                "total": sum(agent["actions"] for agent in agents),
                "latest_id": latest["id"] if latest else 0,
                "latest_at": latest["created_at"] if latest else None,
                "retention_days": settings.AGENT_LOG_DAYS,
            }
        )
