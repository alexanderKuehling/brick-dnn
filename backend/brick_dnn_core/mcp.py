"""MCP server: lets AI assistants build, check, save and train networks for a user.

Implements the Model Context Protocol (https://modelcontextprotocol.io) over Streamable
HTTP, without sessions: every JSON-RPC message is a POST to ``/api/mcp/`` and gets a JSON
answer (no event streams). Clients log in with a personal API token
(``Authorization: Bearer bdnn_…``) or sign in with OAuth (see oauth.py); read-only access
only gets the tools that change nothing.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.files.uploadedfile import SimpleUploadedFile
from django.db.models import Q, RestrictedError
from django.http import HttpResponse
from rest_framework import permissions, serializers
from rest_framework.exceptions import ParseError
from rest_framework.renderers import JSONRenderer
from rest_framework.request import Request
from rest_framework.throttling import ScopedRateThrottle, UserRateThrottle
from rest_framework.views import APIView

import brickdnn
from brickdnn import service
from brickdnn.training import ACTIVE_STATUSES, RUNNING_STATUSES, DatasetError, inference
from brickdnn.training.config import TrainingConfig

from . import budgets, downloads, history, raw_predictions
from . import data_service as data
from . import mcp_compact as compact
from . import mcp_data as data_answers
from . import training_service as training
from . import upload_links as links
from .accounts import usage_data
from .agent_activity import record_tool_call
from .api_tokens import uses_api_token
from .models import ApiToken, DataPrep, DataSource, Project, TrainingRun
from .oauth import McpAuthentication
from .parsers import SafeJSONParser
from .projects import (
    VersionConflict,
    can_edit,
    project_data,
    projects_for,
    role_of,
    update_project,
)
from .projects import delete_project as remove_project
from .retention import touch
from .serializers import (
    DataPrepSerializer,
    ProjectSerializer,
    RunSummarySerializer,
)
from .shared_data import (
    can_delete_run,
    can_edit_prep,
    can_stop_run,
    editable_projects,
    visible_datasets,
    visible_preps,
    visible_runs,
)

logger = logging.getLogger(__name__)

# Newest first; an unknown version requested by a client is answered with the newest.
PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")

INSTRUCTIONS = """\
Brick DNN builds neural networks from layers ("bricks") and turns them into PyTorch code.
A network is a JSON graph: {"version": 1, "nodes": [{"id": "in", "type": "Input",
"params": {"shape": [784]}}, {"id": "fc", "type": "Linear", "params": {"out_features": 10}}],
"edges": [{"source": "in", "target": "fc"}]}. Every network starts with an Input layer whose
shape has no batch dimension; nodes without outgoing edges are the outputs.

It also builds classical models (scikit-learn) for tabular data: Gaussian processes
(GaussianProcessRegressor, GaussianProcessClassifier) and support vector machines (SVR,
SVC). Connect the Input (shape [number of features]) straight to the model; a Gaussian
process can also take a kernel built from kernel bricks (RBF, Matern, RationalQuadratic,
ExpSineSquared, DotProduct, WhiteKernel, ConstantKernel, combined with KernelSum and
KernelProduct), whose edges carry a kernel instead of a tensor. Classical models and neural
layers cannot be mixed in one network yet. Their code is a scikit-learn script, and they
are trained on CSV data or demo data (fitted once, no epochs).

Typical workflow: look up layer types with list_layer_types (or start from list_examples),
build the graph, call analyze_network until it is valid (it infers shapes and parameter
counts and explains errors), then generate_pytorch_code, save_project, or train it on a
dataset (get_training_options, upload_dataset for a small CSV file, check_training,
start_training, get_training_run). A finished run can predict new samples (predict) and
its weights can be downloaded with a short-lived link (get_weights_download).
Saved networks also appear in the user's Brick DNN web app.

Real tables usually need preparing first (missing values, categories, dates, outliers):
import_data (the text of a CSV or JSON file, base64 for Excel or Parquet, or a public link),
create_data_prep with the target column (the column to predict), then repeat
preview_data_prep (column profiles, sample rows and the readiness check: errors, warnings
and fixes) and apply_readiness_fix or update_data_prep (step types: list_prep_steps) until
readiness says ready. export_training_dataset then gives the dataset reference for
check_training and start_training. predict then takes raw rows of the source table
(input_format "raw"): the server prepares them like the training rows.
get_data_prep_script returns the pandas script that prepares new data the same way.
Files too large to pass as text: create_upload_link gives a one-time link for curl.

Answers are compact: lists are capped with their totals, and arguments such as include,
columns_offset or types ask for more.

Trainings and predictions use compute (CPU time), which can be limited per month: for the
account, and for your API token or app. get_usage shows what is left; check_training
estimates what a training will use. A training stops by itself (keeping its weights) when
the budget runs out. Only the user can raise budgets, in the web app.

A server may queue trainings: start_training then returns the run as queued, with its
position and a rough estimate of the wait (queue), and it trains when a worker is free.
get_training_run shows where it stands; stop_training cancels it while it waits."""

GRAPH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        'Network graph: {"version": 1, "nodes": [{"id", "type", "params", "name"?}], '
        '"edges": [{"source", "target"}]}. Layer types and their parameters: '
        "list_layer_types."
    ),
    "properties": {
        "version": {"type": "integer"},
        "nodes": {"type": "array", "items": {"type": "object"}},
        "edges": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["nodes", "edges"],
}

DATASET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        'Which data to train on: {"kind": "builtin", "id": "mnist"}, {"kind": "upload", '
        '"id": 3} (one of the user\'s datasets) or {"kind": "random"}. See '
        "get_training_options."
    ),
    "properties": {
        "kind": {"type": "string", "enum": ["builtin", "upload", "random"]},
        "id": {"type": ["string", "integer"]},
    },
    "required": ["kind"],
}

STEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": 'A preparation step: {"type": "impute", "params": {"strategy": "median"}}; '
    'saved steps also have "id" and "enabled". Types and params: list_prep_steps.',
    "properties": {
        "id": {"type": "string"},
        "type": {"type": "string"},
        "enabled": {"type": "boolean"},
        "params": {"type": "object"},
    },
    "required": ["type"],
}

SPLIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "Train/validation/test split (defaults: val 0.15, test 0.15, seed 42, "
    "stratify true); only the given keys change",
    "properties": {
        "val": {"type": "number", "minimum": 0, "maximum": 0.9},
        "test": {"type": "number", "minimum": 0, "maximum": 0.9},
        "seed": {"type": "integer", "minimum": 0},
        "stratify": {"type": "boolean"},
    },
}

PREP_SETTINGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": 'All settings at once: {"target": "price", "task": "auto", "split": {...}}',
    "properties": {
        "target": {"type": ["string", "null"]},
        "task": {"type": "string", "enum": ["auto", "classification", "regression"]},
        "split": SPLIT_SCHEMA,
    },
}


class McpError(Exception):
    """A JSON-RPC error (wrong method or parameters)."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class ToolError(Exception):
    """A tool could not do its job; the assistant sees the message and can react."""


@dataclass(frozen=True)
class Tool:
    name: str
    title: str
    description: str
    run: Callable[[Request, dict[str, Any]], Any]
    properties: dict[str, Any]
    required: tuple[str, ...] = ()
    # Changes something (saves a network, starts a training): not for read-only tokens.
    writes: bool = False
    destructive: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": self.properties,
                "required": list(self.required),
                "additionalProperties": False,
            },
            "annotations": {
                "title": self.title,
                "readOnlyHint": not self.writes,
                "destructiveHint": self.destructive,
                "idempotentHint": not self.writes,
                "openWorldHint": False,
            },
        }


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


def _int_id(args: dict[str, Any], what: str) -> int:
    value = args.get("id")
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ToolError(f"'id' must be the number of {what}.")
    return value


def _project(request: Request, args: dict[str, Any]) -> Project:
    project = projects_for(request.user).filter(pk=_int_id(args, "a project")).first()
    if project is None:
        raise ToolError("No such project. list_projects shows the user's projects.")
    return project


def _run(request: Request, args: dict[str, Any], key: str = "id") -> TrainingRun:
    try:
        run = visible_runs(request.user).filter(pk=args.get(key)).first()
    except (ValidationError, ValueError):
        run = None
    if run is None:
        raise ToolError("No such training run. list_training_runs shows the user's runs.")
    touch(run)
    return training.sync_run(run)


def _require_training() -> None:
    if not settings.TRAINING_ENABLED:
        raise ToolError(training.DISABLED_REASON)


@contextlib.contextmanager
def _compact_errors():
    try:
        yield
    except compact.CompactError as exc:
        raise ToolError(str(exc)) from None


def list_layer_types(request: Request, args: dict[str, Any]) -> Any:
    with _compact_errors():
        return compact.layer_catalog(service.layers(), args.get("types"))


def list_examples(request: Request, args: dict[str, Any]) -> Any:
    with _compact_errors():
        return compact.examples(service.examples(), args.get("id"))


def analyze_network(request: Request, args: dict[str, Any]) -> Any:
    return service.analyze_graph(args["graph"])


def generate_pytorch_code(request: Request, args: dict[str, Any]) -> Any:
    options = {
        key: args[key]
        for key in ("class_name", "include_main", "include_training")
        if args.get(key) is not None
    }
    status_code, payload = service.generate({"graph": args["graph"], "options": options})
    if status_code != 200:
        errors = payload.get("analysis", {}).get("errors", [])
        details = "".join(f"\n- {e.get('node') or 'graph'}: {e.get('message')}" for e in errors)
        raise ToolError(f"{payload.get('detail')}{details}")
    return payload["code"]


def _summary(project: Project, user) -> dict[str, Any]:
    data = project_data(project, user)
    data.pop("graph")
    data.pop("share_token")
    return data


def list_projects(request: Request, args: dict[str, Any]) -> Any:
    projects = projects_for(request.user).order_by("-updated_at")
    return [_summary(project, request.user) for project in projects]


def get_project(request: Request, args: dict[str, Any]) -> Any:
    data = project_data(_project(request, args), request.user)
    data.pop("share_token")
    return data


def save_project(request: Request, args: dict[str, Any]) -> Any:
    data = {key: args[key] for key in ("name", "description", "graph") if key in args}
    if args.get("id") is None:
        if "name" not in data or "graph" not in data:
            raise ToolError("A new project needs a name and a graph.")
        serializer = ProjectSerializer(data=data)
        if not serializer.is_valid():
            raise ToolError(json.dumps(serializer.errors))
        project = serializer.save(owner=request.user, updated_by=request.user)
        history.record(project, request.user)
        return _summary(project, request.user)

    project = _project(request, args)
    if not can_edit(request.user, project):
        raise ToolError("You can only view this network: its owner shared it read-only.")
    try:
        project = update_project(project, request.user, data, args.get("version"))
    except VersionConflict as exc:
        raise ToolError(
            f"Someone else changed this project (now version {exc.project.version}). "
            "Load it again with get_project and apply your changes to that state."
        ) from None
    except serializers.ValidationError as exc:
        raise ToolError(json.dumps(exc.detail)) from None
    return _summary(project, request.user)


def delete_project(request: Request, args: dict[str, Any]) -> Any:
    project = _project(request, args)
    role = role_of(request.user, project)
    if role != "owner":
        raise ToolError(
            f"Only its owner ({project.owner.get_username()}) can delete “{project.name}”; "
            f"the user is {'an editor' if role == 'editor' else 'a viewer'} of it. To remove it "
            "from their list, they can leave it in the Brick DNN web app (My networks → ⋯ → "
            "Leave)."
        )
    deleted = {"deleted": True, "id": project.pk, "name": project.name}
    remove_project(project)  # the same as in the web app: also closes it for live editors
    return {**deleted, "note": "Training runs started in it are kept for who started them."}


def get_training_options(request: Request, args: dict[str, Any]) -> Any:
    capabilities = training.capabilities(request.user)
    queue = capabilities["queue"]
    return {
        "available": capabilities["available"],
        "reason": capabilities["reason"],
        # With a training worker: trainings wait in a queue (whether a worker is online, its
        # slots, how many trainings run and wait); null: the server trains one at a time and
        # refuses a start while it is busy.
        "queue": {k: v for k, v in queue.items() if v is not None} if queue else None,
        "devices": capabilities["devices"],
        "builtin_datasets": capabilities["builtin_datasets"],
        # Your own datasets and the ones shared in your projects (see "owner" and "mine"),
        # without their columns (get_dataset).
        "your_datasets": [
            compact.dataset_summary(dataset, request.user)
            for dataset in visible_datasets(request.user).select_related("owner")
        ],
        "dataset_reference": DATASET_SCHEMA["description"],
        "config_defaults": TrainingConfig().to_dict(),
        "tuning": capabilities["tuning"],
        "classical": capabilities["classical"],
        "max_concurrent_runs": capabilities["max_concurrent_runs"],
        "upload_dataset_max_mb": _megabytes(upload_limit()),
    }


def get_dataset(request: Request, args: dict[str, Any]) -> Any:
    dataset_id = _int_id(args, "a dataset")
    dataset = (
        visible_datasets(request.user)
        .select_related("owner", "prepared_from")
        .filter(pk=dataset_id)
        .first()
    )
    if dataset is None:
        raise ToolError("No such dataset. get_training_options lists the user's datasets.")
    touch(dataset)
    with _compact_errors():
        return compact.dataset_detail(
            dataset, request.user, args.get("columns_limit"), args.get("columns_offset")
        )


def upload_limit() -> int:
    """Largest CSV text for upload_dataset, in bytes."""
    limit = int(float(settings.MCP_UPLOAD_MAX_MB) * 1024 * 1024)
    return max(0, min(limit, training.max_upload_bytes()))


def _megabytes(size: int) -> float:
    return round(size / 1024**2, 2)


class _UploadScope:
    """Uploads over MCP count against the same rate limit as in the web app."""

    throttle_scope = "uploads"


def upload_dataset(request: Request, args: dict[str, Any]) -> Any:
    _require_training()
    name, text = args["name"], args["csv"]
    if not isinstance(name, str) or not name.strip():
        raise ToolError("'name' must be a short text, e.g. 'Iris flowers'.")
    if len(name.strip()) > 120:
        raise ToolError("The name can be at most 120 characters long.")
    if not isinstance(text, str) or not text.strip():
        raise ToolError("'csv' must be the content of the CSV file as text.")
    data = text.encode("utf-8")
    limit = upload_limit()
    if len(data) > limit:
        raise ToolError(
            f"The CSV text has {_megabytes(len(data))} MB, but upload_dataset takes at most "
            f"{_megabytes(limit)} MB. Upload larger files in the Brick DNN web app instead."
        )
    project = None
    if args.get("project_id") is not None:
        project_id = args["project_id"]
        if isinstance(project_id, bool) or not isinstance(project_id, int):
            raise ToolError("'project_id' must be the number of a project.")
        project = editable_projects(request.user).filter(pk=project_id).first()
        if project is None:
            raise ToolError(
                "Datasets can only be shared in projects the user owns or edits; "
                "list_projects shows the user's role in each project."
            )
    if not ScopedRateThrottle().allow_request(request, _UploadScope()):
        raise ToolError("Too many datasets were uploaded recently. Try again later.")
    # The same path as an upload in the web app: stored, inspected and listed the same way.
    upload = SimpleUploadedFile("dataset.csv", data, content_type="text/csv")
    try:
        dataset = training.save_upload(request.user, upload, name)
    except DatasetError as exc:
        raise ToolError(f"The CSV file cannot be used: {exc}") from None
    if project is not None:
        dataset.projects.add(project)
    return {
        "dataset": compact.dataset_detail(dataset, request.user),
        "reference": {"kind": "upload", "id": dataset.pk},
        "next": "Pass the reference as dataset to check_training and start_training.",
    }


def create_upload_link(request: Request, args: dict[str, Any]) -> Any:
    try:
        return links.create_link(request, request.user, args["purpose"], args)
    except links.UploadLinkError as exc:
        raise ToolError(str(exc)) from None


# --------------------------------------------------------------------------- #
# Data preparation (the Data tab): the same service functions, roles, limits and rate
# limits as the REST API in data_views.py; mcp_data.py makes the answers compact.
# --------------------------------------------------------------------------- #


class _DataScope:
    """Previews and downloads of preparations count against the same rate limit as in the
    web app."""

    throttle_scope = "data"


def _require_data() -> None:
    reason = data.unavailable_reason()
    if reason is not None:
        raise ToolError(reason)


@contextlib.contextmanager
def _data_errors():
    """The messages of the Data tab (and of mcp_data) as tool errors."""
    try:
        yield
    except data.DataInputError as exc:
        message = str(exc)
        if exc.sheets:
            message += f" The sheets of the workbook: {', '.join(exc.sheets)}. Pass 'sheet'."
        raise ToolError(message) from None
    except data_answers.DataToolError as exc:
        raise ToolError(str(exc)) from None


def _check_text(args: dict[str, Any]) -> None:
    """Like the Data tab's API: no text that cannot be stored (e.g. lone surrogates)."""
    from brickdnn.data.steps.base import valid_text

    if not valid_text(args):
        raise ToolError("The arguments contain characters that are not valid text.")


def _data_id(args: dict[str, Any], key: str, what: str) -> int:
    value = args.get(key)
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ToolError(f"'{key}' must be the number of {what}.")
    return value


def _data_source(request: Request, args: dict[str, Any]) -> DataSource:
    """One of the user's sources or the source of a preparation shared with them."""
    user = request.user
    used = visible_preps(user).values("source")
    source = (
        DataSource.objects.filter(Q(owner=user) | Q(pk__in=used))
        .filter(pk=_data_id(args, "id", "a data source"))
        .first()
    )
    if source is None:
        raise ToolError("No such data source. list_data_sources shows the user's data sources.")
    return source


def _data_prep(request: Request, args: dict[str, Any], *, edit: bool = False) -> DataPrep:
    """A preparation the user can see (``edit``: and change)."""
    prep = (
        visible_preps(request.user)
        .select_related("source", "owner")
        .filter(pk=_data_id(args, "id", "a data preparation"))
        .first()
    )
    if prep is None:
        raise ToolError("No such data preparation. list_data_preps shows the user's preparations.")
    if edit and not can_edit_prep(request.user, prep):
        raise ToolError(
            f"This data preparation was shared with the user read-only: they view “{prep.name}” "
            f"of {prep.owner.get_username()} in a project. Only its owner and the project's "
            "editors can change it."
        )
    touch(prep.source)  # an open preparation keeps its data
    return prep


def _sharing_project(request: Request, args: dict[str, Any]) -> Project | None:
    if args.get("project_id") is None:
        return None
    project_id = args["project_id"]
    if isinstance(project_id, bool) or not isinstance(project_id, int):
        raise ToolError("'project_id' must be the number of a project.")
    project = editable_projects(request.user).filter(pk=project_id).first()
    if project is None:
        raise ToolError(
            "Data can only be shared in projects the user owns or edits; list_projects shows "
            "the user's role in each project."
        )
    return project


def _quick_check(request: Request, prep: DataPrep) -> dict[str, Any]:
    """The step reports and the readiness after a change (a preview, rate-limited like
    previews in the web app)."""
    if not ScopedRateThrottle().allow_request(request, _DataScope()):
        return {"skipped": "Too many previews recently; call preview_data_prep later."}
    try:
        result = data.preview(prep, {"sample_rows": data.MIN_SAMPLE_ROWS})
    except data.DataInputError as exc:
        return {"error": str(exc)}
    return data_answers.preview_view(result, detail=False)


def _prep_answer(request: Request, prep: DataPrep) -> dict[str, Any]:
    return {
        "preparation": data_answers.prep_detail(prep, request),
        "check": _quick_check(request, prep),
    }


def _save_prep(request: Request, prep: DataPrep | None, body: dict[str, Any]) -> DataPrep:
    """Create or change a preparation with the serializer of the REST API (same checks)."""
    serializer = DataPrepSerializer(
        prep, data=body, partial=prep is not None, context={"request": request}
    )
    if not serializer.is_valid():
        raise ToolError(data_answers.serializer_message(serializer.errors))
    return serializer.save(owner=request.user) if prep is None else serializer.save()


def list_data_sources(request: Request, args: dict[str, Any]) -> Any:
    _require_data()
    sources = DataSource.objects.filter(owner=request.user)
    return {
        "sources": [data_answers.source_summary(source) for source in sources],
        "import_data": data_answers.import_info(),
    }


def get_data_source(request: Request, args: dict[str, Any]) -> Any:
    _require_data()
    source = _data_source(request, args)
    touch(source)
    with _data_errors():
        rows = data_answers.sample_rows_arg(args.get("sample_rows"))
        offset = data_answers.offset_arg(args.get("columns_offset"))
        return data_answers.source_detail(source, request.user, rows, offset)


def import_data(request: Request, args: dict[str, Any]) -> Any:
    _require_data()
    _check_text(args)
    given = [key for key in ("content", "content_base64", "url") if args.get(key) is not None]
    if len(given) != 1:
        raise ToolError(
            "Pass exactly one of content (the text of a CSV, TSV, JSON or JSON Lines file), "
            "content_base64 (a binary file such as .xlsx or .parquet, with filename) or url "
            "(a public link to the file)."
        )
    name = args.get("name")
    if name is not None and (not isinstance(name, str) or len(name.strip()) > 120):
        raise ToolError("'name' must be a text of at most 120 characters.")
    fields = {key: args.get(key) for key in ("name", *data.OPTION_FIELDS)}
    if given == ["url"]:
        if not settings.DATA_URL_IMPORT_ENABLED:
            raise ToolError(f"{data.URL_DISABLED_MESSAGE} Send the file as content instead.")
        if not ScopedRateThrottle().allow_request(request, _UploadScope()):
            raise ToolError("Too many files were imported recently. Try again later.")
        # The same download as in the web app, with its checks against requests to
        # private addresses (url_import.py).
        with _data_errors():
            source = data.import_url(request.user, args["url"], fields)
    else:
        with _data_errors():
            content, filename = data_answers.decode_content(args)
        limit = data_answers.import_limit()
        if len(content) > limit:
            raise ToolError(
                f"The file has {_megabytes(len(content))} MB, but import_data takes at most "
                f"{_megabytes(limit)} MB as content. Import larger files from a link (url) or "
                "in the Brick DNN web app."
            )
        if not ScopedRateThrottle().allow_request(request, _UploadScope()):
            raise ToolError("Too many files were imported recently. Try again later.")
        upload = SimpleUploadedFile(filename, content)
        with _data_errors():
            source = data.import_source(request.user, upload, fields)
    with _data_errors():
        detail = data_answers.source_detail(source, request.user)
    return {
        "source": detail,
        "next": f"create_data_prep with source_id {source.pk} and the target column (the "
        "column to predict) checks what the data needs for training.",
    }


def list_data_preps(request: Request, args: dict[str, Any]) -> Any:
    _require_data()
    preps = visible_preps(request.user).select_related("source", "owner")
    return {"preparations": [data_answers.prep_summary(prep, request.user) for prep in preps]}


def get_data_prep(request: Request, args: dict[str, Any]) -> Any:
    _require_data()
    return data_answers.prep_detail(_data_prep(request, args), request)


def list_prep_steps(request: Request, args: dict[str, Any]) -> Any:
    _require_data()
    with _data_errors():
        return data_answers.step_catalog(args.get("type"))


def preview_data_prep(request: Request, args: dict[str, Any]) -> Any:
    _require_data()
    _check_text(args)
    prep = _data_prep(request, args)
    with _data_errors():
        rows = data_answers.sample_rows_arg(args.get("sample_rows"))
        columns = data_answers.check_columns(args.get("columns"))
        offset = data_answers.offset_arg(args.get("columns_offset"))
        body: dict[str, Any] = {"sample_rows": max(rows, data.MIN_SAMPLE_ROWS)}
        if args.get("steps") is not None:
            body["steps"] = data_answers.complete_steps(args["steps"])
        if (unsaved := data_answers.settings_from(args, prep.settings)) is not None:
            body["settings"] = unsaved
        if args.get("upto") is not None:
            body["upto"] = args["upto"]
    if not ScopedRateThrottle().allow_request(request, _DataScope()):
        raise ToolError("Too many previews were made recently. Try again later.")
    with _data_errors():
        result = data.preview(prep, body)
    if (message := data_answers.unknown_columns(columns, result)) is not None:
        raise ToolError(message)
    view = data_answers.preview_view(
        result, detail=True, sample_rows=rows, columns=columns, offset=offset
    )
    if "steps" in body or "settings" in body:
        view["unsaved"] = (
            "This previews the given steps/settings without saving them; update_data_prep "
            "saves them. fix_ids refer to the saved preparation only."
        )
    return view


def get_data_prep_script(request: Request, args: dict[str, Any]) -> Any:
    _require_data()
    prep = _data_prep(request, args)
    fmt = args.get("format") or "python"
    if fmt not in data.DOWNLOAD_FORMATS:
        raise ToolError("'format' must be python (pandas script), recipe (JSON) or csv.")
    max_rows = args.get("max_rows", data_answers.DEFAULT_CSV_ROWS)
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
        raise ToolError(f"'max_rows' must be a whole number from 1 to {data_answers.MAX_CSV_ROWS}.")
    if not ScopedRateThrottle().allow_request(request, _DataScope()):
        raise ToolError("Too many downloads were made recently. Try again later.")
    with _data_errors():
        file = data.download(prep, fmt)
    if fmt == "csv":
        with file.handle:
            text = data_answers.csv_text(file.handle, min(max_rows, data_answers.MAX_CSV_ROWS))
        return {"filename": file.filename, **text}
    content = file.content or b""
    if data_answers.too_large(content):
        note = (
            f"The {fmt} file has {round(len(content) / 1024)} KB, too much for MCP; download it "
            "in the Brick DNN web app (Data tab → Export)."
        )
        if fmt == "recipe":
            return {"filename": file.filename, "truncated": True, "note": note}
        cut = content[: data_answers.MAX_DOWNLOAD_BYTES].decode("utf-8", errors="ignore")
        return f"{cut}\n\n# … cut off here. {note}\n"
    if fmt == "recipe":
        return {"filename": file.filename, "recipe": json.loads(content)}
    return content.decode("utf-8")


def create_data_prep(request: Request, args: dict[str, Any]) -> Any:
    _require_data()
    _check_text(args)
    project = _sharing_project(request, args)
    body: dict[str, Any] = {"source": args["source_id"]}
    if args.get("name") is not None:
        body["name"] = args["name"]
    with _data_errors():
        if args.get("steps") is not None:
            body["steps"] = data_answers.complete_steps(args["steps"])
        if (prep_settings := data_answers.settings_from(args)) is not None:
            body["settings"] = prep_settings
    prep = _save_prep(request, None, body)
    if project is not None:
        prep.projects.add(project)
    return _prep_answer(request, prep)


STEP_EDITS = ("add_steps", "edit_steps", "move_steps", "remove_steps")


def update_data_prep(request: Request, args: dict[str, Any]) -> Any:
    _require_data()
    _check_text(args)
    prep = _data_prep(request, args, edit=True)
    body: dict[str, Any] = {}
    edits = [key for key in STEP_EDITS if args.get(key) is not None]
    with _data_errors():
        if args.get("steps") is not None:
            if edits:
                raise ToolError(
                    f"Pass either steps (the whole new list) or {', '.join(STEP_EDITS)}, not both."
                )
            body["steps"] = data_answers.complete_steps(args["steps"])
        elif edits:
            current = prep.steps if isinstance(prep.steps, list) else []
            body["steps"] = data_answers.edit_steps(current, args)
        if (prep_settings := data_answers.settings_from(args, prep.settings)) is not None:
            body["settings"] = prep_settings
    if args.get("name") is not None:
        body["name"] = args["name"]
    if args.get("source_id") is not None:
        body["source"] = args["source_id"]
    if not body:
        raise ToolError(
            "Nothing to change: pass steps, add_steps, edit_steps, move_steps, remove_steps, "
            "target, task, split, settings, name or source_id."
        )
    return _prep_answer(request, _save_prep(request, prep, body))


def apply_readiness_fix(request: Request, args: dict[str, Any]) -> Any:
    _require_data()
    prep = _data_prep(request, args, edit=True)
    wanted = args["fix_id"]
    if not isinstance(wanted, str) or not wanted.strip():
        raise ToolError("'fix_id' must be the fix_id of a readiness issue, or 'all'.")
    if not ScopedRateThrottle().allow_request(request, _DataScope()):
        raise ToolError("Too many previews were made recently. Try again later.")
    # The readiness of the saved preparation, as the web app shows it.
    with _data_errors():
        result = data.preview(prep, {"sample_rows": data.MIN_SAMPLE_ROWS})
    readiness = result["readiness"]
    fixes = data_answers.fixes_of(readiness)
    if wanted.strip() == "all":  # "Fix all": the fixes of all errors
        chosen = data_answers.error_fixes(readiness)
        if not chosen:
            errors = [i["message"] for i in readiness.get("issues", []) if i["severity"] == "error"]
            raise ToolError(
                "No error has a fix. "
                + (
                    "The remaining errors: " + " ".join(errors)
                    if errors
                    else "The data is ready for training: export_training_dataset."
                )
            )
    elif wanted.strip() in fixes:
        chosen = [(wanted.strip(), fixes[wanted.strip()])]
    else:
        offered = "; ".join(f"{key}: {fix['label']}" for key, (fix, _) in fixes.items())
        raise ToolError(
            f"No fix '{wanted}' in the readiness check of the saved preparation (fix ids change "
            "when the steps or the data change). "
            + (f"Its fixes now: {offered}." if offered else "It suggests no fixes now.")
        )
    steps = list(prep.steps) if isinstance(prep.steps, list) else []
    applied = []
    with _data_errors():
        for key, (fix, _) in chosen:
            new = {"type": fix["type"], "params": copy.deepcopy(fix.get("params") or {})}
            steps, created = data_answers.insert_step(steps, new, reports=result["steps"])
            applied.append(
                {
                    "fix_id": key,
                    "label": fix.get("label"),
                    "step": created["id"],
                    "type": fix["type"],
                }
            )
    prep = _save_prep(request, prep, {"steps": steps})
    return {
        "applied": applied,
        "steps": [{"id": s["id"], "type": s["type"]} for s in prep.steps],
        "check": _quick_check(request, prep),
    }


def export_training_dataset(request: Request, args: dict[str, Any]) -> Any:
    _require_data()
    _require_training()
    _check_text(args)
    prep = _data_prep(request, args, edit=True)
    if not ScopedRateThrottle().allow_request(request, _UploadScope()):
        raise ToolError("Too many datasets were created recently. Try again later.")
    try:
        dataset = data.export_dataset(prep, args.get("name"))
    except data.DataInputError as exc:
        message = str(exc)
        if exc.issues:
            errors = [i["message"] for i in exc.issues if i.get("severity") == "error"]
            message = (
                f"The data is not ready for training: {' '.join(errors) or message} "
                "preview_data_prep lists the problems with their fixes (apply_readiness_fix)."
            )
        raise ToolError(message) from None
    return data_answers.export_answer(dataset, request.user)


def delete_data_prep(request: Request, args: dict[str, Any]) -> Any:
    _require_data()
    prep_id = _data_id(args, "id", "a data preparation")
    prep = visible_preps(request.user).select_related("owner").filter(pk=prep_id).first()
    if prep is None:
        raise ToolError("No such data preparation. list_data_preps shows the user's preparations.")
    if prep.owner_id != request.user.pk:
        raise ToolError(
            f"Only its owner ({prep.owner.get_username()}) can delete “{prep.name}”; it is "
            "shared with the user in a project."
        )
    kept = prep.datasets.count()
    deleted = {"deleted": True, "id": prep.pk, "name": prep.name}
    prep.delete()  # the same as DELETE /api/data/preps/<id>/
    if kept:
        deleted["note"] = (
            f"The {kept} training dataset{'s' if kept > 1 else ''} made from it "
            f"{'are' if kept > 1 else 'is'} kept."
        )
    return deleted


def delete_data_source(request: Request, args: dict[str, Any]) -> Any:
    _require_data()
    source = _data_source(request, args)
    if source.owner_id != request.user.pk:
        raise ToolError(
            f"Only its owner ({source.owner.get_username()}) can delete this data source; the "
            "user sees it through a preparation shared in a project."
        )
    # As DELETE /api/data/sources/<id>/: not while a preparation uses it.
    hint = " delete_data_prep deletes a preparation (ask the user first)."
    preps = data.preps_using(source)
    if preps:
        raise ToolError(data.in_use_message(preps) + hint)
    deleted = {"deleted": True, "id": source.pk, "name": source.name}
    try:
        source.delete()  # the files are removed by a post_delete signal
    except RestrictedError:  # a preparation was created in the meantime
        raise ToolError(data.in_use_message(data.preps_using(source)) + hint) from None
    return deleted


def check_training(request: Request, args: dict[str, Any]) -> Any:
    _require_training()
    try:
        training.validate_graph(args["graph"])
        dataset = training.resolve_dataset(request.user, args["dataset"])
        config = training.parse_config(args.get("config"))
        search = training.parse_tuning(args.get("tuning"))
    except training.TrainingInputError as exc:
        raise ToolError(str(exc)) from None
    result = training.check(args["graph"], dataset, config)
    cost = training.budget_check(
        budgets.Actor.of(request), args["graph"], dataset, config, result, search
    )
    answer = result.to_dict()
    if isinstance(answer.get("plan"), dict):  # up to 1,000 class names
        plan = answer["plan"]
        plan["class_names"] = compact.capped_names(plan.get("class_names"))
    return {**answer, **cost}


class _TrainingScope:
    """Starting a training over MCP counts against the same rate limit as in the web app."""

    throttle_scope = "training"


def start_training(request: Request, args: dict[str, Any]) -> Any:
    _require_training()
    if not ScopedRateThrottle().allow_request(request, _TrainingScope()):
        raise ToolError("Too many trainings were started recently. Try again later.")
    try:
        run = training.start_run(
            request.user,
            args.get("name"),
            args["graph"],
            args["dataset"],
            args.get("config"),
            args.get("tuning"),
            args.get("project_id"),
            actor=budgets.Actor.of(request),
        )
    except training.IncompatibleNetwork as exc:
        raise ToolError(json.dumps({"detail": str(exc), **exc.compatibility.to_dict()})) from None
    except (
        training.TrainingUnavailable,
        training.TrainingInputError,
        training.TrainingBusy,
    ) as exc:
        raise ToolError(str(exc)) from None
    data = dict(RunSummarySerializer(run).data)
    if run.budget_warning:
        data["budget_warning"] = run.budget_warning
    return data


def get_usage(request: Request, args: dict[str, Any]) -> Any:
    return usage_data(request)


def list_training_runs(request: Request, args: dict[str, Any]) -> Any:
    runs = visible_runs(request.user).select_related(
        "owner", "api_token", "connection__client", "worker"
    )
    training.sync_runs(runs.filter(status__in=ACTIVE_STATUSES))
    limit = args.get("limit", 20)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ToolError("'limit' must be a whole number from 1 to 100.")
    if args.get("status") is not None:
        runs = runs.filter(status=args["status"])
    if args.get("project_id") is not None:
        runs = runs.filter(project_id=_data_id(args, "project_id", "a project"))
    runs = runs.order_by("-created_at")
    total = runs.count()
    context: dict[str, Any] = {}  # one look at the queue for all runs
    shown = [compact.run_summary(run, context) for run in runs[:limit]]
    answer: dict[str, Any] = {"total": total, "runs": shown}
    if total > len(shown):
        answer["note"] = f"The newest {len(shown)} of {total} runs; raise limit to see more."
    return answer


def get_training_run(request: Request, args: dict[str, Any]) -> Any:
    run = _run(request, args)
    with _compact_errors():
        includes = compact.run_includes(args.get("include"))
        return compact.run_detail(run, training.state_of(run), includes, args.get("log_lines"))


def stop_training(request: Request, args: dict[str, Any]) -> Any:
    run = _run(request, args)
    if not can_stop_run(request.user, run):
        raise ToolError("Only who started it and the project's editors can stop it.")
    return RunSummarySerializer(training.stop_run(run)).data


def get_weights_download(request: Request, args: dict[str, Any]) -> Any:
    run = _run(request, args)
    if training.weights_path(run) is None:
        raise ToolError(training.no_weights_reason(run))
    return downloads.weights_info(request, run, request.user)


class _PredictScope:
    """Predictions over MCP count against the same rate limit as in the web app."""

    throttle_scope = "predict"


def predict(request: Request, args: dict[str, Any]) -> Any:
    run = _run(request, args, "run_id")
    try:
        prediction = training.prepare_prediction(
            run,
            args["inputs"],
            args.get("input_scale"),
            args.get("top_k"),
            input_format=args.get("input_format"),
        )
    except training.TrainingInputError as exc:
        message = str(exc)
        if isinstance(exc, raw_predictions.RawRowsError) and len(exc.errors) > 1:
            message = "\n".join(exc.errors)
        expected = training.describe_inputs(run)
        raise ToolError(
            f"{message}\nExpected inputs: {expected}" if expected else message
        ) from None
    except (training.TrainingUnavailable, training.PredictionUnavailable) as exc:
        raise ToolError(str(exc)) from None
    # Only calls that start a process count against the rate limit.
    if not ScopedRateThrottle().allow_request(request, _PredictScope()):
        raise ToolError("Too many predictions were made recently. Try again later.")
    try:
        answer = training.run_prediction(prediction, budgets.Actor.of(request))
    except (training.TrainingBusy, training.PredictionFailed) as exc:
        raise ToolError(str(exc)) from None
    return compact.prediction_answer(answer)


def delete_training_run(request: Request, args: dict[str, Any]) -> Any:
    run = _run(request, args)
    if not can_delete_run(request.user, run):
        raise ToolError(
            f"Only who started this run ({run.owner.get_username()}) and the owner of its "
            "project can delete it."
        )
    deleted = {"deleted": True, "id": str(run.pk), "name": run.name}
    was_training = run.status in RUNNING_STATUSES
    training.delete_run(run)
    return {**deleted, "killed_running_training": was_training}


def delete_dataset(request: Request, args: dict[str, Any]) -> Any:
    dataset_id = _int_id(args, "a dataset")
    dataset = visible_datasets(request.user).select_related("owner").filter(pk=dataset_id).first()
    if dataset is None:
        raise ToolError("No such dataset. get_training_options lists the user's datasets.")
    if dataset.owner_id != request.user.pk:
        raise ToolError(
            f"Only its owner ({dataset.owner.get_username()}) can delete this dataset; it is "
            "shared with the user in a project."
        )
    deleted = {"deleted": True, "id": dataset.pk, "name": dataset.name}
    try:
        training.delete_dataset(dataset)
    except training.DatasetInUse as exc:
        raise ToolError(
            f"{exc} list_training_runs shows the running trainings; stop_training stops one."
        ) from None
    return deleted


TOOLS: dict[str, Tool] = {
    tool.name: tool
    for tool in [
        Tool(
            "list_layer_types",
            "List layer types",
            "All layer types (Input, Linear, Conv2d, LSTM, …, kernel bricks such as RBF and "
            "classical models such as GaussianProcessRegressor or SVC) with their parameters: "
            "kind, default, allowed range, and whether the value can be inferred ('auto'). "
            "family says whether a brick is a neural layer or classical (scikit-learn), "
            "output_kind whether it puts out a tensor or a kernel, input_slots what it takes. "
            "Compact by default (one line per parameter); types gives the full specs with "
            "help texts of some types.",
            list_layer_types,
            {
                "types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Full specs of these types, e.g. ['Conv2d', 'LSTM']",
                }
            },
        ),
        Tool(
            "list_examples",
            "List example networks",
            "Example networks (MLP, CNN, ResNet block, LSTM, Transformer, Gaussian process "
            "regression, SVM classifier, …): good starting points. Lists names and "
            "descriptions; id gives one example with its graph.",
            list_examples,
            {"id": {"type": "string", "description": "An example id: returns its graph"}},
        ),
        Tool(
            "analyze_network",
            "Analyze a network",
            "Validate a graph and infer the output shape, dtype and parameter count of every "
            "layer. Returns valid, errors (with the node they belong to), warnings, nodes and "
            "total_params. Kernel bricks and Gaussian processes report their kernel as "
            "scikit-learn code (kernel); family and classical describe a classical model. "
            "Use it after every change.",
            analyze_network,
            {"graph": GRAPH_SCHEMA},
            ("graph",),
        ),
        Tool(
            "generate_pytorch_code",
            "Generate PyTorch code",
            "Turn a valid graph into a runnable PyTorch script (an nn.Module class, optionally "
            "with a demo main and a CSV training loop). For a classical model it is a "
            "scikit-learn script instead: make_model() returns Pipeline(StandardScaler, model), "
            "the main fits it to a CSV file, saves it with joblib and predicts new rows "
            "(class_name does not apply).",
            generate_pytorch_code,
            {
                "graph": GRAPH_SCHEMA,
                "class_name": {"type": "string", "description": "Default: NeuralNetwork"},
                "include_main": {"type": "boolean", "description": "Default: true"},
                "include_training": {"type": "boolean", "description": "Default: true"},
            },
            ("graph",),
        ),
        Tool(
            "list_projects",
            "List saved networks",
            "The user's saved networks (id, name, description, number of layers), newest first.",
            list_projects,
            {},
        ),
        Tool(
            "get_project",
            "Open a saved network",
            "One saved network with its graph.",
            get_project,
            {"id": {"type": "integer"}},
            ("id",),
        ),
        Tool(
            "save_project",
            "Save a network",
            "Save a network in the user's account (it appears in the web app). Without id a "
            "new project is created (name and graph required); with id the given fields of "
            "that project are replaced (owner or editor only). Pass the version you loaded: "
            "if someone changed the project since, nothing is saved and you get an error.",
            save_project,
            {
                "id": {"type": "integer", "description": "Project to update; omit to create"},
                "version": {
                    "type": "integer",
                    "description": "Version from get_project/list_projects (recommended)",
                },
                "name": {"type": "string", "maxLength": 120},
                "description": {"type": "string", "maxLength": 2000},
                "graph": GRAPH_SCHEMA,
            },
            writes=True,
            destructive=True,
        ),
        Tool(
            "delete_project",
            "Delete a saved network",
            "Delete a saved network for good, with its history and comments (only its owner "
            "can; members are told how to leave it instead). Anyone editing it live in the web "
            "app is disconnected; training runs started in it are kept. Ask the user first.",
            delete_project,
            {"id": {"type": "integer", "description": "Project id from list_projects"}},
            ("id",),
            writes=True,
            destructive=True,
        ),
        Tool(
            "get_training_options",
            "Show training options",
            "Whether training is available on this server (available: neural networks with "
            "PyTorch; classical: Gaussian processes and SVMs with scikit-learn, with the "
            "most training rows they are fitted on), the built-in datasets and the user's own "
            "datasets, how to refer to a dataset, the default training settings (config) and "
            "the hyperparameter search options (tuning, neural networks only). Datasets are "
            "summaries (rows, features, target, task, classes, reference); get_dataset shows "
            "the columns of one.",
            get_training_options,
            {},
        ),
        Tool(
            "get_dataset",
            "Show a dataset",
            "One training dataset: rows, task, target, the class names (capped) and its "
            "columns, a page at a time (columns_limit, default 50; columns_offset), with the "
            "total in n_columns. The features come first, the target is the last column.",
            get_dataset,
            {
                "id": {"type": "integer", "description": "Dataset id from get_training_options"},
                "columns_limit": {"type": "integer", "minimum": 0, "maximum": 500},
                "columns_offset": {"type": "integer", "minimum": 0},
            },
            ("id",),
        ),
        Tool(
            "upload_dataset",
            "Upload a CSV dataset",
            "Store a small CSV file as one of the user's training datasets (the text of the "
            "file; get_training_options shows the size limit). Comma-, semicolon- or "
            "tab-separated, optionally with a header row; every column except the last is a "
            "numeric feature, the last column is the target: numbers for regression, class "
            "labels or class indices for classification. At least 4 rows. It is checked like "
            'an upload in the web app; train on it with {"kind": "upload", "id": <id>}. Data '
            "with missing values, text categories or dates needs preparing first: import_data.",
            upload_dataset,
            {
                "name": {"type": "string", "maxLength": 120, "description": "Dataset name"},
                "csv": {"type": "string", "description": "Content of the CSV file"},
                "project_id": {
                    "type": "integer",
                    "description": "Also share it in this project (owner or editor), so that "
                    "its members can train on it",
                },
            },
            ("name", "csv"),
            writes=True,
        ),
        Tool(
            "create_upload_link",
            "Create an upload link",
            "A one-time link (valid 15 minutes, no login needed) to upload a local file "
            "with a shell command such as curl, for files too large to pass as text. "
            "purpose data_source imports it into the Data tab like import_data (any format "
            "import_data reads, with its options); training_dataset stores a CSV or ZIP file "
            "like upload_dataset. Returns the url and ready-to-run curl commands; the curl "
            "answer is the JSON that import_data or upload_dataset would return (with the "
            "new id or the dataset reference). The size limit is that of the web app. Which "
            "way to send a file: small file (up to the import_data limit) as text or base64; "
            "a local file and a shell: this link; a public file: import_data with url; "
            "otherwise ask the user to upload it in the Brick DNN web app.",
            create_upload_link,
            {
                "purpose": {"type": "string", "enum": list(links.PURPOSES)},
                "filename": {
                    "type": "string",
                    "description": "Name of the file with its ending, e.g. sales.xlsx (the "
                    "ending decides the format)",
                },
                "name": {"type": "string", "maxLength": 120, "description": "Default: file name"},
                "project_id": {
                    "type": "integer",
                    "description": "training_dataset: also share it in this project",
                },
                "delimiter": {"type": "string", "description": "data_source: CSV delimiter"},
                "decimal": {"type": "string", "description": "data_source: '.' or ','"},
                "header": {"type": "string", "enum": ["auto", "yes", "no"]},
                "encoding": {"type": "string", "description": "data_source: auto, utf-8 …"},
                "missing_values": {"type": "array", "items": {"type": "string"}},
                "sheet": {"type": "string", "description": "data_source: sheet of a workbook"},
            },
            ("purpose", "filename"),
            writes=True,
        ),
        Tool(
            "list_data_sources",
            "List imported data files",
            "The user's imported data files for data preparation (data sources): id, name, "
            "file, format, rows and columns, newest first. Also what import_data accepts on "
            "this server: the size limit, the file types and whether links can be imported.",
            list_data_sources,
            {},
        ),
        Tool(
            "get_data_source",
            "Show an imported data file",
            "One data source: its columns with their types (numeric, boolean, datetime, "
            "categorical, text) and missing values, 50 at a time (columns_offset for more), "
            "what the import detected (format, delimiter, decimal mark, header, sheet, "
            "warnings), its first rows and the preparations that use it.",
            get_data_source,
            {
                "id": {"type": "integer", "description": "Source id from list_data_sources"},
                "sample_rows": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": data_answers.MAX_SAMPLE_ROWS,
                    "description": f"First rows to show (default {data_answers.DEFAULT_SAMPLE_ROWS})",
                },
                "columns_offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": f"Show the columns from this one on "
                    f"({data_answers.COLUMN_PAGE} at a time)",
                },
            },
            ("id",),
        ),
        Tool(
            "import_data",
            "Import a data file",
            "Import a data file as a data source for data preparation, like Import in the web "
            "app's Data tab. Pass exactly one of: content, the text of a CSV/TSV, JSON or JSON "
            "Lines file (filename defaults to data.csv; name a JSON file e.g. orders.json); "
            "content_base64, a binary file (Excel .xlsx/.xls, OpenDocument .ods, Parquet, "
            "Feather, or packed as .gz/.zip) with its filename; or url, a public http(s) link "
            "that the server downloads (share links of Google Sheets, Google Drive, GitHub and "
            "Dropbox work). Delimiter, decimal mark, header, encoding and missing-value markers "
            "are detected: pass the options only to override them. Every column gets a type. "
            "Returns the source with its columns and first rows; list_data_sources shows the "
            "size limit. Next: create_data_prep.",
            import_data,
            {
                "content": {"type": "string", "description": "Text of a CSV or JSON file"},
                "content_base64": {"type": "string", "description": "A binary file as base64"},
                "url": {"type": "string", "description": "Public link to the file"},
                "filename": {
                    "type": "string",
                    "description": "File name with its ending, e.g. sales.xlsx (decides the "
                    "format; required with content_base64)",
                },
                "name": {"type": "string", "maxLength": 120, "description": "Default: file name"},
                "delimiter": {
                    "type": "string",
                    "description": "CSV: auto (default), ',', ';', 'tab' or '|'",
                },
                "decimal": {"type": "string", "description": "CSV: auto (default), '.' or ','"},
                "header": {
                    "type": "string",
                    "enum": ["auto", "yes", "no"],
                    "description": "Whether the first row holds the column names (CSV, sheets)",
                },
                "encoding": {
                    "type": "string",
                    "description": "Only content_base64 and url: auto, utf-8, cp1252 or latin-1",
                },
                "missing_values": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Cell values that mean missing, e.g. ['-999', 'n/a'] "
                    "(default: empty cells, NA, N/A, null, -, ? …)",
                },
                "sheet": {
                    "type": "string",
                    "description": "Sheet of a workbook (default: the first one with values)",
                },
            },
            writes=True,
        ),
        Tool(
            "list_data_preps",
            "List data preparations",
            "The user's data preparations and the ones shared with them in projects: id, name, "
            "source, target, task, number of steps, owner and the user's role (owner, editor, "
            "viewer), last changed first.",
            list_data_preps,
            {},
        ),
        Tool(
            "get_data_prep",
            "Show a data preparation",
            "One data preparation: its source, settings (target, task, split), steps with "
            "their ids and params, the training datasets exported from it (with their "
            "references for training), owner and role. preview_data_prep runs it.",
            get_data_prep,
            {"id": {"type": "integer", "description": "Preparation id from list_data_preps"}},
            ("id",),
        ),
        Tool(
            "list_prep_steps",
            "List preparation steps",
            "The step types of data preparation (join and append tables; drop, keep, rename, "
            "cast columns, formulas, replace values; filter, sort and deduplicate rows; fill "
            "missing values; outliers and clipping; scale and transform numbers, bins; encode "
            "categories; date parts; balance classes; feature selection) with one line per "
            "parameter: what it accepts, its default and when it applies. With type: the help "
            "texts of that step type and an example.",
            list_prep_steps,
            {"type": {"type": "string", "description": "A step type, e.g. impute"}},
        ),
        Tool(
            "preview_data_prep",
            "Preview a data preparation",
            "Run a preparation and look at the result: a report per step (what it did, rows "
            "and columns added or removed, or why it failed, with the columns it saw), the "
            "columns after all steps with profiles (missing, unique, min/max/mean/median/std, "
            "top values, association with the target), the target (classes or distribution), "
            "sample rows with their split, and the readiness for training: errors that block "
            "the export, warnings (target leakage, duplicates, rare classes, imbalance …) and "
            "suggestions, each with a fix_id where a step fixes it (apply_readiness_fix). "
            "steps and/or target, task, split, settings try changes without saving them; upto "
            "shows the data after that step. Wide tables are shown in pages: 50 columns, "
            "profiles and samples of the first 20 of them; columns_offset or columns (names) "
            "choose others.",
            preview_data_prep,
            {
                "id": {"type": "integer", "description": "Preparation id"},
                "steps": {
                    "type": "array",
                    "items": STEP_SCHEMA,
                    "description": "Unsaved steps to try instead of the saved ones",
                },
                "target": {"type": ["string", "null"], "description": "Unsaved target column"},
                "task": {"type": "string", "enum": ["auto", "classification", "regression"]},
                "split": SPLIT_SCHEMA,
                "settings": PREP_SETTINGS_SCHEMA,
                "upto": {"type": "string", "description": "Show the data after this step id"},
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only these columns in profiles and samples",
                },
                "columns_offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": f"Columns from this one on ({data_answers.COLUMN_PAGE} at a "
                    f"time; profiles and samples of the first {data_answers.MAX_PROFILES})",
                },
                "sample_rows": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": data_answers.MAX_SAMPLE_ROWS,
                    "description": f"Sample rows (default {data_answers.DEFAULT_SAMPLE_ROWS})",
                },
            },
            ("id",),
        ),
        Tool(
            "get_data_prep_script",
            "Download a data preparation",
            "The saved preparation as a file: python (default) is a standalone pandas script "
            "whose prepare(df) applies the same steps with the values learned from the "
            "training rows, so new data (e.g. rows to predict) is prepared exactly like the "
            "training data; recipe the steps with their learned values as JSON; csv the "
            "prepared rows with a split column (the first max_rows rows).",
            get_data_prep_script,
            {
                "id": {"type": "integer", "description": "Preparation id"},
                "format": {"type": "string", "enum": list(data.DOWNLOAD_FORMATS)},
                "max_rows": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": data_answers.MAX_CSV_ROWS,
                    "description": f"csv: rows to return (default {data_answers.DEFAULT_CSV_ROWS})",
                },
            },
            ("id",),
        ),
        Tool(
            "create_data_prep",
            "Create a data preparation",
            "Start preparing one of the user's data sources for training: the target column "
            "(what a model should predict), the task (auto, classification or regression), "
            "the train/validation/test split and optional first steps. Returns the "
            "preparation and a check: the report of every step and the readiness for "
            "training (errors that block the export, warnings and suggestions, with fix_ids). "
            "Next: apply_readiness_fix or update_data_prep until it is ready; "
            "preview_data_prep shows profiles and sample rows.",
            create_data_prep,
            {
                "source_id": {"type": "integer", "description": "Source id (import_data)"},
                "name": {"type": "string", "maxLength": 120, "description": "Default: source"},
                "target": {"type": "string", "description": "The column to predict"},
                "task": {
                    "type": "string",
                    "enum": ["auto", "classification", "regression"],
                    "description": "Default auto: classes for text and few whole numbers",
                },
                "split": SPLIT_SCHEMA,
                "steps": {
                    "type": "array",
                    "items": STEP_SCHEMA,
                    "description": "First steps (list_prep_steps)",
                },
                "project_id": {
                    "type": "integer",
                    "description": "Also share it in this project (owner or editor): its "
                    "members see it, its editors can change it",
                },
            },
            ("source_id",),
            writes=True,
        ),
        Tool(
            "update_data_prep",
            "Change a data preparation",
            "Change a preparation (its owner, or the editors of a project it is shared in) and "
            "save it right away; the web app shows the change. Steps: either steps (the whole "
            "new list; steps without id are new and get the defaults of their type) or the "
            "edits remove_steps, edit_steps (params are merged, null resets one), move_steps "
            "and add_steps (at the end or after a step), applied in this order. Settings: "
            "target, task and split (changed on top of the current ones) or settings (all). "
            "Also name and source_id. Returns the preparation and the check: the report of "
            "every step (a failing step says why, e.g. a wrong column name with the columns "
            "it sees; later steps are skipped) and the readiness with fix_ids.",
            update_data_prep,
            {
                "id": {"type": "integer", "description": "Preparation id"},
                "steps": {
                    "type": "array",
                    "items": STEP_SCHEMA,
                    "description": "Replaces all steps",
                },
                "add_steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "params": {"type": "object"},
                            "enabled": {"type": "boolean"},
                            "after": {
                                "type": "string",
                                "description": "Insert after this step id (default: at the end)",
                            },
                        },
                        "required": ["type"],
                    },
                    "description": 'e.g. [{"type": "impute", "params": {"strategy": "mean"}}]',
                },
                "edit_steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "params": {"type": "object"},
                            "enabled": {"type": "boolean"},
                        },
                        "required": ["id"],
                    },
                    "description": 'e.g. [{"id": "s2", "params": {"k": 3}}]',
                },
                "move_steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "after": {
                                "type": ["string", "null"],
                                "description": "Step id to follow, null for the top",
                            },
                        },
                        "required": ["id"],
                    },
                },
                "remove_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Step ids",
                },
                "target": {"type": ["string", "null"], "description": "The column to predict"},
                "task": {"type": "string", "enum": ["auto", "classification", "regression"]},
                "split": SPLIT_SCHEMA,
                "settings": PREP_SETTINGS_SCHEMA,
                "name": {"type": "string", "maxLength": 120},
                "source_id": {"type": "integer", "description": "Another source of the owner"},
            },
            ("id",),
            writes=True,
            destructive=True,
        ),
        Tool(
            "apply_readiness_fix",
            "Apply a readiness fix",
            "Add the step that fixes a readiness issue, like its button in the web app: a "
            "fix_id from the readiness of the saved preparation (create_data_prep, "
            "update_data_prep, preview_data_prep without unsaved changes), or 'all' for the "
            "fixes of all errors at once (Fix all). The step goes to the end, in front of a "
            "failing step. Saves the preparation; returns the added steps and the new check.",
            apply_readiness_fix,
            {
                "id": {"type": "integer", "description": "Preparation id"},
                "fix_id": {"type": "string", "description": "A fix_id, or 'all'"},
            },
            ("id", "fix_id"),
            writes=True,
        ),
        Tool(
            "export_training_dataset",
            "Create a training dataset",
            "Turn the saved preparation into a training dataset (owner or editor) once its "
            "readiness says ready: the numeric feature columns, the target last and the "
            "split of the preparation. Returns the dataset reference for check_training and "
            "start_training, the task, the target, the feature columns in order (what "
            "predict takes) and the class names. Data that is not ready is refused with the "
            "errors that remain.",
            export_training_dataset,
            {
                "id": {"type": "integer", "description": "Preparation id"},
                "name": {
                    "type": "string",
                    "maxLength": 120,
                    "description": "Dataset name (default: the preparation's)",
                },
            },
            ("id",),
            writes=True,
        ),
        Tool(
            "delete_data_prep",
            "Delete a data preparation",
            "Delete a data preparation for good (only its owner can). Its data source and "
            "the training datasets made from it are kept. Ask the user first.",
            delete_data_prep,
            {"id": {"type": "integer", "description": "Preparation id from list_data_preps"}},
            ("id",),
            writes=True,
            destructive=True,
        ),
        Tool(
            "delete_data_source",
            "Delete an imported data file",
            "Delete one of the user's data sources with its files for good. Not possible "
            "while a preparation uses it (delete those first). Ask the user first.",
            delete_data_source,
            {"id": {"type": "integer", "description": "Source id from list_data_sources"}},
            ("id",),
            writes=True,
            destructive=True,
        ),
        Tool(
            "check_training",
            "Check a network against a dataset",
            "Check whether a network fits a dataset (input shape and dtype, output size, loss; "
            "classical models: tabular data and the right task) before training. Returns ok, "
            "errors, warnings, the plan (task, classes) and fixes: the params to set on the "
            "Input layer and the last layer to make it fit. Pass config to also check the "
            "settings (e.g. max_samples against the row limit of classical models). Also "
            "returns estimate (the rough CPU time the training would use, in CPU minutes, "
            "with a low-high range) and budget: whether the compute budgets allow it "
            "(allowed, reason, warning, run_limit_cpu_minutes: the most it may use). Pass "
            "config and tuning as for start_training for a precise estimate. On a server "
            "with a training queue, queue says where the run would wait (position) and "
            "roughly how long (estimated_wait_seconds, an estimate; null if unknown).",
            check_training,
            {
                "graph": GRAPH_SCHEMA,
                "dataset": DATASET_SCHEMA,
                "config": {"type": "object"},
                "tuning": {"type": "object"},
            },
            ("graph", "dataset"),
        ),
        Tool(
            "start_training",
            "Start a training",
            "Train a network on the server. Runs in the background: follow it with "
            "get_training_run. On a server with a training queue the run starts as queued "
            "(queue: its position and the estimated wait) and trains when a worker is "
            "free; stop_training cancels it. config overrides the defaults from "
            "get_training_options "
            "(epochs, batch_size, learning_rate, …); tuning turns the run into a "
            "hyperparameter search. A classical model is fitted once (no epochs): only "
            "val_split, test_split, max_samples and seed apply, and tuning is not available. "
            "Refused when a compute budget is used up (get_usage); the run stops by itself, "
            "keeping its weights, when it has used what is left (budget_warning says if that "
            "is likely).",
            start_training,
            {
                "graph": GRAPH_SCHEMA,
                "dataset": DATASET_SCHEMA,
                "name": {"type": "string", "maxLength": 120},
                "config": {"type": "object"},
                "tuning": {"type": "object"},
                "project_id": {
                    "type": "integer",
                    "description": "Shared project to train in: its members see the run",
                },
            },
            ("graph", "dataset"),
            writes=True,
        ),
        Tool(
            "get_usage",
            "Show usage and budgets",
            "This month's compute usage (CPU minutes of trainings and predictions) and storage "
            "of the user's account, its limits and what is left, when the budgets reset (the "
            "first day of the next month, UTC), the usage per API token and app, and the "
            "budget of the token or app you are using (caller). Check it before training. "
            "Reading it costs nothing.",
            get_usage,
            {},
        ),
        Tool(
            "list_training_runs",
            "List training runs",
            "The user's training runs, newest first: status, epochs done, final metrics, one "
            "line each, with the total. At most limit runs (default 20); status and "
            "project_id filter them.",
            list_training_runs,
            {
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                "status": {
                    "type": "string",
                    "enum": [
                        "queued",
                        "starting",
                        "preparing",
                        "running",
                        "completed",
                        "stopped",
                        "failed",
                    ],
                },
                "project_id": {"type": "integer", "description": "Runs of this project only"},
            },
        ),
        Tool(
            "get_training_run",
            "Show a training run",
            "One training run, as a summary by default: status and message (queued runs: "
            "queue with the position and the estimated wait), settings, epochs "
            "done, the best epoch, the final and test metrics (loss, accuracy; for regressions "
            "the RMSE in the units of the target and R²), "
            "the rows per split, tuning (trials and the best one), classical (the fitted "
            "kernel, log marginal likelihood or support vectors) and the end of the log if "
            "it failed. include adds history (metrics per epoch, at most 50 evenly spaced "
            "epochs plus the best and the last), graph, plan, log (the last log_lines "
            "lines), trials and preprocessing.",
            get_training_run,
            {
                "id": {"type": "string"},
                "include": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(compact.RUN_INCLUDES)},
                },
                "log_lines": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": compact.MAX_LOG_LINES,
                    "description": f"With include log (default {compact.DEFAULT_LOG_LINES})",
                },
            },
            ("id",),
        ),
        Tool(
            "stop_training",
            "Stop a training",
            "Stop a running training after the current batch; the weights trained so far are "
            "kept. A training that still waits in the queue is cancelled.",
            stop_training,
            {"id": {"type": "string"}},
            ("id",),
            writes=True,
        ),
        Tool(
            "get_weights_download",
            "Get the trained weights",
            "A download link for the trained weights of a finished or stopped run (a PyTorch "
            "state_dict, .pt; for a classical model the fitted scikit-learn pipeline, .joblib), "
            "valid for 15 minutes and usable without logging in: give it to the user or fetch "
            "it with a tool that can download files. Also returns the matching model code "
            "(class NeuralNetwork, or make_model()), how inputs were prepared during training "
            "(feature order, normalisation, image size, class names) and a Python snippet that "
            "loads the weights and predicts.",
            get_weights_download,
            {"id": {"type": "string", "description": "Run id from list_training_runs"}},
            ("id",),
        ),
        Tool(
            "predict",
            "Predict with a trained network",
            "Run the trained network of a finished or stopped run on up to "
            f"{inference.MAX_SAMPLES} samples on the server. Classification: the most likely "
            "class with its probability and the top_k classes; regression: the predicted "
            "value in the units of the target column (a Gaussian process adds std, its "
            "uncertainty). Runs trained on a dataset exported from a data preparation: send "
            "raw rows with input_format 'raw', one object {column: value} per sample in the "
            "columns of the imported source table (as in the file, before the preparation; "
            "dates as 2024-05-31, the target left out); the server applies the preparation "
            "with the values it learned from the training rows, so do not compute prepared "
            "columns (one-hot, date parts, scaling …) yourself. Other runs trained on a CSV "
            "file (and prepared rows, input_format 'prepared', the default): one row per "
            "sample, as an object {column: value} or a list of the feature "
            "values in column order, raw as in the CSV file (they are standardised like in "
            "training). Inputs of an image network: one image per sample as a nested array of "
            "channels×height×width (grayscale also height×width) or in the shape of its Input "
            "layer, with input_scale 'raw' (pixels 0–255), 'unit' (0–1) or 'normalized' "
            "(already normalised like in training). Runs trained on demo data take arrays of "
            "the Input shape as they are. get_weights_download describes a run's inputs.",
            predict,
            {
                "run_id": {"type": "string", "description": "Run id from list_training_runs"},
                "inputs": {
                    "type": "array",
                    "description": f"The samples, at most {inference.MAX_SAMPLES}",
                },
                "input_scale": {
                    "type": "string",
                    "enum": list(inference.INPUT_SCALES),
                    "description": "How the values are given (required for image networks): "
                    "'raw' as in the CSV file or image file (pixels 0–255), 'unit' for "
                    "pixels in 0–1, 'normalized' if already prepared like in training",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": inference.MAX_TOP_K,
                    "description": f"Classes listed per sample (default {inference.DEFAULT_TOP_K})",
                },
                "input_format": {
                    "type": "string",
                    "enum": list(raw_predictions.INPUT_FORMATS),
                    "description": "'raw': rows in the columns of the source table of a data "
                    "preparation, prepared by the server like the training rows; 'prepared' "
                    "(default): the feature columns of the training dataset",
                },
            },
            ("run_id", "inputs"),
        ),
        Tool(
            "delete_training_run",
            "Delete a training run",
            "Delete a training run with its trained weights and log for good; a training that "
            "is still running is killed first. Only who started it and the owner of its "
            "project can. Ask the user first.",
            delete_training_run,
            {"id": {"type": "string", "description": "Run id from list_training_runs"}},
            ("id",),
            writes=True,
            destructive=True,
        ),
        Tool(
            "delete_dataset",
            "Delete a dataset",
            "Delete one of the user's own training datasets with its files for good. Not "
            "possible while a training uses it, nor for datasets that others shared with the "
            "user. Runs trained on it are kept. Ask the user first.",
            delete_dataset,
            {"id": {"type": "integer", "description": "Dataset id from get_training_options"}},
            ("id",),
            writes=True,
            destructive=True,
        ),
    ]
}


# --------------------------------------------------------------------------- #
# JSON-RPC
# --------------------------------------------------------------------------- #


def _can_write(request: Request) -> bool:
    token = request.auth
    return not (isinstance(token, ApiToken) and token.read_only)


def _tools_for(request: Request) -> list[Tool]:
    return [tool for tool in TOOLS.values() if _can_write(request) or not tool.writes]


def _initialize(params: dict[str, Any]) -> dict[str, Any]:
    requested = params.get("protocolVersion")
    return {
        "protocolVersion": requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {
            "name": "brick-dnn",
            "title": "Brick DNN",
            "version": brickdnn.__version__,
        },
        "instructions": INSTRUCTIONS,
    }


def _run_tool(tool: Tool, request: Request, args: dict[str, Any]) -> tuple[Any, str | None]:
    """The tool's result, or the error message for the assistant."""
    try:
        if tool.writes and not _can_write(request):
            raise ToolError("This API token is read-only. Use a token with full access.")
        missing = [name for name in tool.required if args.get(name) is None]
        if missing:
            raise ToolError(f"Missing argument: {', '.join(missing)}.")
        return tool.run(request, args), None
    except ToolError as exc:
        return None, str(exc)
    except budgets.BudgetError as exc:
        # A budget or the storage quota does not allow it: say which, and how to get more.
        return None, f"{exc.message}\nget_usage shows the usage and budgets of this month."
    except Exception:
        logger.exception("MCP tool %s failed", tool.name)
        return None, "Internal error in the Brick DNN server."


def _call_tool(request: Request, params: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    name = params.get("name")
    tool = TOOLS.get(name) if isinstance(name, str) else None
    args = params.get("arguments") or {}
    if tool is None or not isinstance(args, dict):
        message = f"Unknown tool: {name}" if tool is None else "'arguments' must be an object."
        record_tool_call(request, name, args, started=started, error=message)
        raise McpError(-32602, message)

    result, error = _run_tool(tool, request, args)
    # The activity log (agent_activity.py); written after the response.
    record_tool_call(
        request, tool.name, args, started=started, result=result, error=error, writes=tool.writes
    )
    if error is not None:
        return {"content": [{"type": "text", "text": error}], "isError": True}
    if isinstance(result, str):
        return {"content": [{"type": "text", "text": result}]}
    text = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    response: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
    if isinstance(result, dict):
        response["structuredContent"] = json.loads(text)
    return response


def _error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}


def handle_message(request: Request, message: Any) -> dict[str, Any] | None:
    """Answer one JSON-RPC message; ``None`` for notifications and responses."""
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return _error(None, -32600, "Invalid JSON-RPC message.")
    method = message.get("method")
    if method is None and ("result" in message or "error" in message):
        return None  # a response to a request of ours; we never send any
    if not isinstance(method, str):
        return _error(message.get("id"), -32600, "Invalid JSON-RPC message.")
    if "id" not in message:
        return None  # notifications (initialized, cancelled, …) need no answer

    message_id = message["id"]
    params = message.get("params") or {}
    if not isinstance(params, dict):
        return _error(message_id, -32602, "'params' must be an object.")
    try:
        if method == "initialize":
            result = _initialize(params)
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": [tool.to_dict() for tool in _tools_for(request)]}
        elif method == "tools/call":
            result = _call_tool(request, params)
        else:
            raise McpError(-32601, f"Method not found: {method}")
    except McpError as exc:
        return _error(message_id, exc.code, exc.message)
    return {"jsonrpc": "2.0", "id": message_id, "result": result}


def _json(payload: Any, status: int = 200) -> HttpResponse:
    return HttpResponse(
        json.dumps(payload, ensure_ascii=False), status=status, content_type="application/json"
    )


class IsApiTokenUser(permissions.IsAuthenticated):
    message = (
        "Sign in with OAuth, or log in with a personal API token: "
        "Authorization: Bearer bdnn_… (see the API tokens page of Brick DNN)."
    )

    def has_permission(self, request, view) -> bool:
        return super().has_permission(request, view) and uses_api_token(request)


class McpView(APIView):
    authentication_classes = [McpAuthentication]
    permission_classes = [IsApiTokenUser]
    throttle_classes = [UserRateThrottle, ScopedRateThrottle]
    throttle_scope = "mcp"
    parser_classes = [SafeJSONParser]
    renderer_classes = [JSONRenderer]

    def perform_content_negotiation(self, request, force=False):
        # Clients accept "application/json, text/event-stream"; we always answer with JSON.
        return super().perform_content_negotiation(request, force=True)

    def post(self, request: Request) -> HttpResponse:
        if not _origin_allowed(request):
            return _json(_error(None, -32600, "Origin not allowed."), status=403)
        version = request.headers.get("MCP-Protocol-Version")
        if version and version not in PROTOCOL_VERSIONS:
            return _json(_error(None, -32600, f"Unsupported protocol version: {version}"), 400)
        limit = _body_limit()
        if not _read_body(request, limit):
            message = f"Request too large: MCP requests can have at most {_megabytes(limit)} MB."
            return _json(_error(None, -32600, message), status=413)
        try:
            payload = request.data
        except ParseError as exc:
            return _json(_error(None, -32700, f"Parse error: {exc.detail}"), status=400)

        if isinstance(payload, list):  # batches (protocol 2025-03-26)
            if not payload:
                return _json(_error(None, -32600, "Empty batch."), status=400)
            answers = [a for m in payload if (a := handle_message(request, m)) is not None]
            return _json(answers) if answers else HttpResponse(status=202)
        answer = handle_message(request, payload)
        return _json(answer) if answer is not None else HttpResponse(status=202)


def _body_limit() -> int | None:
    """Largest MCP request in bytes: upload_dataset and import_data carry a file as JSON text
    (which escapes quotes and line breaks) or base64, so the usual limit for JSON requests is
    not enough."""
    default = settings.DATA_UPLOAD_MAX_MEMORY_SIZE
    if default is None:
        return None
    return max(default, 2 * max(upload_limit(), data_answers.import_limit()) + 64 * 1024)


def _read_body(request: Request, limit: int | None) -> bool:
    """Read the body with the MCP limit instead of DATA_UPLOAD_MAX_MEMORY_SIZE. Called after
    authentication, so only logged-in users can send large requests. False: too large.

    Django checks DATA_UPLOAD_MAX_MEMORY_SIZE when ``request.body`` is first read; a body
    that is already loaded is used as it is (by DRF's parsers too).
    """
    django_request = request._request
    if limit is None or hasattr(django_request, "_body") or django_request._read_started:
        return True
    try:
        length = int(django_request.META.get("CONTENT_LENGTH") or 0)
    except ValueError:
        length = 0
    if length > limit:
        return False
    body = django_request.read(limit + 1)
    if len(body) > limit:
        return False
    django_request._body = body
    django_request._stream = io.BytesIO(body)
    return True


def _origin_allowed(request: Request) -> bool:
    """Browsers send Origin: only this site and the configured CORS origins may call us."""
    origin = request.headers.get("Origin")
    if not origin:
        return True
    return urlsplit(origin).netloc == request.get_host() or origin in getattr(
        settings, "CORS_ALLOWED_ORIGINS", []
    )
