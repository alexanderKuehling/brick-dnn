"""MCP tools that explain trained models (explain_service.py): ``explain_prediction`` and
``get_layer_activations``.

The answers of the web app hold pictures (PNG data URLs) and histograms; the tools leave
them out and keep the numbers an assistant can read (coarse grids of at most 14×14 or
16×16 values, the top attributions, statistics per brick). With ``images`` the pictures
are added as long as all of them together stay below MAX_IMAGE_CHARS.

The tools are added to mcp.TOOLS (and their descriptions to the activity log) when this
module is imported, at the end of mcp.py.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rest_framework.request import Request
from rest_framework.throttling import ScopedRateThrottle

from brickdnn.training import inference

from . import agent_activity
from . import explain_service as explaining
from . import training_service as training
from .budgets import Actor, BudgetError
from .mcp import TOOLS, Tool, ToolError, _run

# Pictures in an answer with ``images``: at most this many characters together.
MAX_IMAGE_CHARS = 60_000
# Values of a vector and channels of feature maps shown per brick.
VECTOR_VALUES = 32
MAX_POINTS = 40
EXPLAIN_METHODS = ("samples", "image", "tabular", "permutation", "boundary")


class _PredictScope:
    """Explanations count against the rate limit of predictions."""

    throttle_scope = "predict"


def _explain(request: Request, run_id: Any, body: dict[str, Any]) -> dict[str, Any]:
    run = _run(request, {"run_id": run_id}, "run_id")
    try:
        explanation = explaining.prepare(run, body)
    except training.TrainingInputError as exc:
        expected = training.describe_inputs(run)
        raise ToolError(f"{exc}\nExpected inputs: {expected}" if expected else str(exc)) from None
    except (training.TrainingUnavailable, training.PredictionUnavailable) as exc:
        raise ToolError(str(exc)) from None
    if explaining.needs_process(explanation) and not ScopedRateThrottle().allow_request(
        request, _PredictScope()
    ):
        raise ToolError("Too many predictions and explanations were made recently. Try later.")
    try:
        return explaining.run_explanation(explanation, Actor.of(request))
    except BudgetError as exc:
        raise ToolError(str(exc.body().get("detail") or exc)) from None
    except (training.TrainingBusy, training.PredictionFailed) as exc:
        raise ToolError(str(exc)) from None


def _request_body(args: dict[str, Any], method: str) -> dict[str, Any]:
    body: dict[str, Any] = {"method": method}
    for key in ("sample", "input", "input_scale", "target", "layer", "node"):
        if args.get(key) is not None:
            body[key] = args[key]
    return body


# --------------------------------------------------------------------------- #
# Compact answers
# --------------------------------------------------------------------------- #


def _images(value: Any, found: list[tuple[Any, str]], path: tuple = ()) -> None:
    """Collect every PNG data URL (with where it is) in ``value``."""
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and item.startswith("data:image/png"):
                found.append((value, key))
            else:
                _images(item, found, (*path, key))
    elif isinstance(value, list):
        for item in value:
            _images(item, found, path)


def _strip_images(answer: dict[str, Any], keep: bool) -> dict[str, Any]:
    found: list[tuple[Any, str]] = []
    _images(answer, found)
    total = sum(len(holder[key]) for holder, key in found)
    if keep and total <= MAX_IMAGE_CHARS:
        return answer
    for holder, key in found:
        del holder[key]
    if keep and found:
        answer.setdefault("notes", []).append(
            f"The pictures were left out: together they are larger than {MAX_IMAGE_CHARS:,} "
            "characters. The grids hold the same maps, coarser."
        )
    return answer


def compact_explanation(answer: dict[str, Any], images: bool = False) -> dict[str, Any]:
    answer = {k: v for k, v in answer.items() if k not in ("cached",)}
    method = answer.get("method")
    if method == "samples":
        answer["samples"] = answer.get("samples", [])[:24]
    if method == "boundary":
        points = answer.get("points") or []
        answer["points"] = points[:MAX_POINTS]
        if len(points) > MAX_POINTS:
            answer["points_note"] = f"The first {MAX_POINTS} of {len(points)} training points."
        if "grid" in answer:
            answer["grid_note"] = (
                "grid: the prediction (class index or value) on a coarse grid; row 0 is the "
                "largest value of the second feature, column 0 the smallest of the first."
            )
    if method == "image":
        answer["grid_note"] = (
            "Each grid is a coarse copy of the map (0 = no influence, 1 = the strongest; "
            "occlusion: how much the probability drops), row 0 at the top of the image."
        )
        gradcam = answer.get("gradcam")
        if isinstance(gradcam, dict) and isinstance(gradcam.get("layers"), list):
            gradcam["layers"] = [layer["id"] for layer in gradcam["layers"]]
    return _strip_images(answer, images)


def compact_activations(answer: dict[str, Any], images: bool = False) -> dict[str, Any]:
    nodes = []
    for node in answer.get("nodes") or []:
        view = dict(node.get("view") or {})
        entry: dict[str, Any] = {
            "id": node.get("id"),
            "type": node.get("type"),
            "name": node.get("name"),
            "shape": node.get("shape"),
            "stats": node.get("stats"),
        }
        if view.get("kind") == "vector":
            values = view.get("values") or []
            entry["values"] = values[:VECTOR_VALUES]
            if view.get("total", 0) > VECTOR_VALUES:
                entry["values_note"] = f"The first {VECTOR_VALUES} of {view['total']} values."
        elif view.get("kind") == "scalar":
            entry["value"] = view.get("value")
        elif view.get("kind") == "maps":
            entry["channel_means"] = view.get("channel_means")
            if images:
                entry["png"] = view.get("png")
        elif images and view.get("png"):
            entry["png"] = view.get("png")
        nodes.append(entry)
    compact = {
        k: v for k, v in answer.items() if k in ("run", "kind", "sample", "prediction", "notes")
    }
    compact["nodes"] = nodes
    return _strip_images(compact, images)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


def explain_prediction(request: Request, args: dict[str, Any]) -> Any:
    method = args.get("method")
    if method not in EXPLAIN_METHODS:
        raise ToolError(f"'method' must be one of: {', '.join(EXPLAIN_METHODS)}.")
    answer = _explain(request, args.get("run_id"), _request_body(args, method))
    return compact_explanation(answer, bool(args.get("images")))


def get_layer_activations(request: Request, args: dict[str, Any]) -> Any:
    answer = _explain(request, args.get("run_id"), _request_body(args, "activations"))
    return compact_activations(answer, bool(args.get("images")))


SAMPLE_PROPERTIES: dict[str, Any] = {
    "run_id": {"type": "string", "description": "Run id from list_training_runs"},
    "sample": {
        "type": "integer",
        "minimum": 0,
        "description": "A validation sample of the run: its number in the answer of method "
        "'samples'",
    },
    "input": {
        "description": "Or one sample of your own, like one entry of predict's inputs "
        "(a row of a CSV model, or an image as a nested array)",
    },
    "input_scale": {
        "type": "string",
        "enum": list(inference.INPUT_SCALES),
        "description": "How the values of input are given (images: required), as for predict",
    },
    "images": {
        "type": "boolean",
        "description": "Also return the pictures as PNG data URLs (if they are small enough)",
    },
}

TOOLS["explain_prediction"] = Tool(
    "explain_prediction",
    "Explain a trained model",
    "Explain what a trained model of a finished run does (computed on the server like a "
    "prediction, it uses compute; repeated questions are answered from a cache). Methods: "
    "'samples' lists validation samples with the model's predictions (pick one by its "
    "number); 'image' (image networks, one sample): the predicted classes, a saliency map "
    "(SmoothGrad: which pixels change the score most), Grad-CAM (where the last convolution, or "
    "'layer', sees evidence for the class) and occlusion (how much the probability drops when "
    "a patch is covered), each as a coarse grid of 0–1; 'tabular' (CSV models, one sample): "
    "Shapley values per feature against the mean of the training rows (they add up to the "
    "prediction minus the prediction for the mean row), for networks also integrated "
    "gradients, for Gaussian processes the predictive std; 'permutation' (CSV models): how "
    "much accuracy or R² drops on validation rows when a feature is shuffled, mean and std; "
    "'boundary' (models with exactly 2 input features): the predicted class or value on a "
    "grid with training points. target picks the class to explain (default: the predicted "
    "one). docs: what each method shows and its limits are in the answer's notes.",
    explain_prediction,
    {
        **SAMPLE_PROPERTIES,
        "method": {"type": "string", "enum": list(EXPLAIN_METHODS)},
        "target": {"type": "integer", "minimum": 0, "description": "Class index to explain"},
        "layer": {"type": "string", "description": "Grad-CAM: the id of a brick"},
    },
    ("run_id", "method"),
)

TOOLS["get_layer_activations"] = Tool(
    "get_layer_activations",
    "Look inside a trained network",
    "The output of every brick (or of one, 'node') of a trained network for one sample: "
    "shape, mean, std, min, max and the share of zeros; for vectors the first values, for "
    "feature maps the mean of each of the first 16 channels. Like a prediction it runs on "
    "the server and uses compute. A run on demo data takes a random input without "
    "sample or input.",
    get_layer_activations,
    {**SAMPLE_PROPERTIES, "node": {"type": "string", "description": "Only this brick (its id)"}},
    ("run_id",),
)


# --------------------------------------------------------------------------- #
# The activity log of AI agents
# --------------------------------------------------------------------------- #


def _describe_explain(user, args: Mapping, result: Any) -> agent_activity.Described:
    ref, project = agent_activity._run_targets(user, args.get("run_id"), None)
    method = args.get("method")
    what = {
        "samples": "the validation samples",
        "image": "a saliency map and Grad-CAM",
        "tabular": "feature attributions",
        "permutation": "the permutation importance",
        "boundary": "the decision boundary",
    }.get(method if isinstance(method, str) else "", "an explanation")
    return agent_activity.Described(
        f"get {what} of {agent_activity.phrase_of(ref, 'run')}",
        targets=[ref] if ref else [],
        project_id=project,
    )


def _describe_activations(user, args: Mapping, result: Any) -> agent_activity.Described:
    ref, project = agent_activity._run_targets(user, args.get("run_id"), None)
    return agent_activity.Described(
        f"get the layer activations of {agent_activity.phrase_of(ref, 'run')}",
        targets=[ref] if ref else [],
        project_id=project,
    )


agent_activity.TOOL_DESCRIPTIONS["explain_prediction"] = _describe_explain
agent_activity.TOOL_DESCRIPTIONS["get_layer_activations"] = _describe_activations
agent_activity.REST_AS_TOOL[("training-run-explain", "POST")] = (
    "explain_prediction",
    lambda kw, body: {"run_id": kw.get("pk"), "method": body.get("method")},
)
agent_activity.REST_PHRASES[("training-run-explain", "GET")] = (
    "look at how {object} can be explained"
)
