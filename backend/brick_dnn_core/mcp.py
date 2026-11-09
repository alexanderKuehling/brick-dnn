"""MCP server: lets AI assistants build, check, save and train networks for a user.

Implements the Model Context Protocol (https://modelcontextprotocol.io) over Streamable
HTTP, without sessions: every JSON-RPC message is a POST to ``/api/mcp/`` and gets a JSON
answer (no event streams). Clients log in with a personal API token
(``Authorization: Bearer bdnn_…``) or sign in with OAuth (see oauth.py); read-only access
only gets the tools that change nothing.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from django.conf import settings
from django.core.exceptions import ValidationError
from django.http import HttpResponse
from rest_framework import permissions, serializers
from rest_framework.exceptions import ParseError
from rest_framework.renderers import JSONRenderer
from rest_framework.request import Request
from rest_framework.throttling import ScopedRateThrottle, UserRateThrottle
from rest_framework.views import APIView

import brickdnn
from brickdnn import service
from brickdnn.training import ACTIVE_STATUSES
from brickdnn.training.config import TrainingConfig

from . import history
from . import training_service as training
from .api_tokens import uses_api_token
from .models import ApiToken, Dataset, Project, TrainingRun
from .oauth import McpAuthentication
from .parsers import SafeJSONParser
from .projects import (
    VersionConflict,
    can_edit,
    project_data,
    projects_for,
    update_project,
)
from .serializers import (
    DatasetSerializer,
    ProjectSerializer,
    RunDetailSerializer,
    RunSummarySerializer,
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

Typical workflow: look up layer types with list_layer_types (or start from list_examples),
build the graph, call analyze_network until it is valid (it infers shapes and parameter
counts and explains errors), then generate_pytorch_code, save_project, or train it on a
dataset (get_training_options, check_training, start_training, get_training_run).
Saved networks also appear in the user's Brick DNN web app."""

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


def _run(request: Request, args: dict[str, Any]) -> TrainingRun:
    try:
        run = TrainingRun.objects.filter(owner=request.user, pk=args.get("id")).first()
    except (ValidationError, ValueError):
        run = None
    if run is None:
        raise ToolError("No such training run. list_training_runs shows the user's runs.")
    return training.sync_run(run)


def _require_training() -> None:
    if not settings.TRAINING_ENABLED:
        raise ToolError(training.DISABLED_REASON)


def list_layer_types(request: Request, args: dict[str, Any]) -> Any:
    return service.layers()


def list_examples(request: Request, args: dict[str, Any]) -> Any:
    return service.examples()


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


def get_training_options(request: Request, args: dict[str, Any]) -> Any:
    capabilities = training.capabilities()
    return {
        "available": capabilities["available"],
        "reason": capabilities["reason"],
        "devices": capabilities["devices"],
        "builtin_datasets": capabilities["builtin_datasets"],
        "your_datasets": DatasetSerializer(
            Dataset.objects.filter(owner=request.user), many=True
        ).data,
        "dataset_reference": DATASET_SCHEMA["description"],
        "config_defaults": TrainingConfig().to_dict(),
        "tuning": capabilities["tuning"],
        "max_concurrent_runs": capabilities["max_concurrent_runs"],
    }


def check_training(request: Request, args: dict[str, Any]) -> Any:
    _require_training()
    try:
        training.validate_graph(args["graph"])
        dataset = training.resolve_dataset(request.user, args["dataset"])
    except training.TrainingInputError as exc:
        raise ToolError(str(exc)) from None
    return training.check(args["graph"], dataset).to_dict()


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
        )
    except training.IncompatibleNetwork as exc:
        raise ToolError(json.dumps({"detail": str(exc), **exc.compatibility.to_dict()})) from None
    except (
        training.TrainingUnavailable,
        training.TrainingInputError,
        training.TrainingBusy,
    ) as exc:
        raise ToolError(str(exc)) from None
    return RunSummarySerializer(run).data


def list_training_runs(request: Request, args: dict[str, Any]) -> Any:
    runs = TrainingRun.objects.filter(owner=request.user)
    training.sync_runs(runs.filter(status__in=ACTIVE_STATUSES))
    return RunSummarySerializer(runs, many=True).data


def get_training_run(request: Request, args: dict[str, Any]) -> Any:
    return RunDetailSerializer(_run(request, args)).data


def stop_training(request: Request, args: dict[str, Any]) -> Any:
    return RunSummarySerializer(training.stop_run(_run(request, args))).data


TOOLS: dict[str, Tool] = {
    tool.name: tool
    for tool in [
        Tool(
            "list_layer_types",
            "List layer types",
            "All layer types (Input, Linear, Conv2d, LSTM, …) with their parameters: kind, "
            "default, allowed range, and whether the value can be inferred ('auto').",
            list_layer_types,
            {},
        ),
        Tool(
            "list_examples",
            "List example networks",
            "Example networks with their graphs (MLP, CNN, ResNet block, LSTM, Transformer, "
            "…): good starting points.",
            list_examples,
            {},
        ),
        Tool(
            "analyze_network",
            "Analyze a network",
            "Validate a graph and infer the output shape, dtype and parameter count of every "
            "layer. Returns valid, errors (with the node they belong to), warnings, nodes and "
            "total_params. Use it after every change.",
            analyze_network,
            {"graph": GRAPH_SCHEMA},
            ("graph",),
        ),
        Tool(
            "generate_pytorch_code",
            "Generate PyTorch code",
            "Turn a valid graph into a runnable PyTorch script (an nn.Module class, optionally "
            "with a demo main and a CSV training loop).",
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
            "get_training_options",
            "Show training options",
            "Whether training is available on this server, the built-in datasets and the "
            "user's own datasets, how to refer to a dataset, the default training settings "
            "(config) and the hyperparameter search options (tuning).",
            get_training_options,
            {},
        ),
        Tool(
            "check_training",
            "Check a network against a dataset",
            "Check whether a network fits a dataset (input shape and dtype, output size, loss) "
            "before training. Returns ok, errors, warnings and the task.",
            check_training,
            {"graph": GRAPH_SCHEMA, "dataset": DATASET_SCHEMA},
            ("graph", "dataset"),
        ),
        Tool(
            "start_training",
            "Start a training",
            "Train a network on the server. Runs in the background: follow it with "
            "get_training_run. config overrides the defaults from get_training_options "
            "(epochs, batch_size, learning_rate, …); tuning turns the run into a "
            "hyperparameter search.",
            start_training,
            {
                "graph": GRAPH_SCHEMA,
                "dataset": DATASET_SCHEMA,
                "name": {"type": "string", "maxLength": 120},
                "config": {"type": "object"},
                "tuning": {"type": "object"},
            },
            ("graph", "dataset"),
            writes=True,
        ),
        Tool(
            "list_training_runs",
            "List training runs",
            "The user's training runs with status, progress and final metrics, newest first.",
            list_training_runs,
            {},
        ),
        Tool(
            "get_training_run",
            "Show a training run",
            "One training run with its settings, its history of metrics per epoch and its status.",
            get_training_run,
            {"id": {"type": "string"}},
            ("id",),
        ),
        Tool(
            "stop_training",
            "Stop a training",
            "Stop a running training after the current batch; the weights trained so far are kept.",
            stop_training,
            {"id": {"type": "string"}},
            ("id",),
            writes=True,
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


def _call_tool(request: Request, params: dict[str, Any]) -> dict[str, Any]:
    tool = TOOLS.get(params.get("name"))  # type: ignore[arg-type]
    if tool is None:
        raise McpError(-32602, f"Unknown tool: {params.get('name')}")
    args = params.get("arguments") or {}
    if not isinstance(args, dict):
        raise McpError(-32602, "'arguments' must be an object.")

    try:
        if tool.writes and not _can_write(request):
            raise ToolError("This API token is read-only. Use a token with full access.")
        missing = [name for name in tool.required if args.get(name) is None]
        if missing:
            raise ToolError(f"Missing argument: {', '.join(missing)}.")
        result = tool.run(request, args)
    except ToolError as exc:
        return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
    except Exception:
        logger.exception("MCP tool %s failed", tool.name)
        return {
            "content": [{"type": "text", "text": "Internal error in the Brick DNN server."}],
            "isError": True,
        }

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


def _origin_allowed(request: Request) -> bool:
    """Browsers send Origin: only this site and the configured CORS origins may call us."""
    origin = request.headers.get("Origin")
    if not origin:
        return True
    return urlsplit(origin).netloc == request.get_host() or origin in getattr(
        settings, "CORS_ALLOWED_ORIGINS", []
    )
