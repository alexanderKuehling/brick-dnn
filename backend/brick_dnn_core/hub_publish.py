"""Publishing a trained run to the Hugging Face Hub as a model repository.

Only when the user asks for it (the web app's "Publish to Hugging Face", the MCP tool
publish_to_huggingface): the user pastes a Hugging Face access token with write access,
which is used for that one request (hub_client.HubToken: never stored, logged or shown in
the activity log), and chooses the repository name, public or private, and the licence.

The repository gets (see :func:`repo_files`):

* the trained model: ``model.safetensors`` for a neural network (converted from the run's
  ``weights.pt`` in a process of its own, see brickdnn.training.safetensors_file; ``model.pt``
  if that fails), ``model.joblib`` for a classical model (the fitted scikit-learn pipeline),
  both ``model.joblib`` and ``model.features.pt`` for a hybrid network (the neural layers
  stay a ``torch.save`` checkpoint because the generated script and the loading code read
  it like that; the joblib file is a pickle anyway);
* the generated script (``<name>.py``: the PyTorch or scikit-learn code with training and
  prediction), the network graph (``network.json``) and the preprocessing
  (``preprocessing.json``: what new inputs need, as the run measured it);
* ``README.md``, the model card (:func:`model_card`): YAML metadata (library, licence, tags
  ``brickdnn`` and the task, the dataset, the metrics as a model-index) and an explanation
  of the architecture, the training settings, the results per split and how to load the
  files and predict.

Limits: the files together at most ``HUGGINGFACE_MAX_UPLOAD_MB``, the rate limit
``huggingface`` per user, and an overall time limit below the web server's. Errors of the
Hub (bad token, name taken, too large ...) are :class:`~.hub_client.HubError`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.exceptions import ValidationError
from django.utils.text import slugify
from rest_framework import permissions, status
from rest_framework.exceptions import NotFound
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle, UserRateThrottle
from rest_framework.views import APIView

from brickdnn.training import inference
from brickdnn.training.state import (
    CLASSICAL_WEIGHTS_FILE,
    WEIGHTS_FILE,
    preprocessing_path,
)
from brickdnn.training.textsets import TEXT_DATASETS

from . import training_service as training
from .hub_client import HubClient, HubError, HubToken, UploadFile
from .models import TrainingRun
from .retention import touch
from .shared_data import can_stop_run, visible_runs

logger = logging.getLogger(__name__)

SITE = "https://brickdnn.com"
DISABLED = "Publishing to Hugging Face is switched off on this server."
# Licences offered (Hugging Face licence identifiers); "other" for anything else.
LICENSES = (
    "apache-2.0",
    "mit",
    "bsd-3-clause",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc-by-nc-4.0",
    "cc0-1.0",
    "gpl-3.0",
    "openrail",
    "other",
)
DEFAULT_LICENSE = "apache-2.0"
# Repository names on the Hub: letters, digits, "-", "_" and ".", at most 96 characters,
# not starting or ending with "-" or ".", no "--" or "..".
NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?")
CONVERT_TIMEOUT = 120  # seconds for the conversion to safetensors
TIME_LIMIT = 240  # seconds for the whole publish (gunicorn stops a request after 300)
SUMMARY = "Upload model trained with Brick DNN"
# Built-in datasets of Brick DNN that are on the Hub too.
HUB_DATASETS = {
    "mnist": "ylecun/mnist",
    "fashion_mnist": "zalando-datasets/fashion_mnist",
    "cifar10": "uoft-cs/cifar10",
}
PIPELINE_TASKS = {
    ("tabular", "classification"): "tabular-classification",
    ("tabular", "regression"): "tabular-regression",
    ("image", "classification"): "image-classification",
    ("image", "segmentation"): "image-segmentation",
    ("tokens", "classification"): "text-classification",
}


class HubInputError(ValueError):
    """An invalid choice in the publish form (400); the message is for the user."""


def enabled() -> bool:
    return bool(settings.HUGGINGFACE_ENABLED)


def max_upload_bytes() -> int:
    return int(float(settings.HUGGINGFACE_MAX_UPLOAD_MB) * 1024 * 1024)


def capabilities() -> dict[str, Any]:
    """``HubCapabilities`` (frontend/src/api/types.ts), part of GET /api/training/."""
    return {
        "publish": enabled(),
        "max_upload_mb": settings.HUGGINGFACE_MAX_UPLOAD_MB,
        "licenses": list(LICENSES),
        "default_license": DEFAULT_LICENSE,
        "url": settings.HUGGINGFACE_URL.rstrip("/"),
    }


def client(token: HubToken | None, deadline: float | None = None) -> HubClient:
    return HubClient(
        settings.HUGGINGFACE_URL,
        token,
        proxy=settings.HUGGINGFACE_PROXY or None,
        deadline=deadline,
    )


# --------------------------------------------------------------------------- #
# What may be published, and the choices
# --------------------------------------------------------------------------- #


def can_publish(user, run: TrainingRun) -> bool:
    """Who started the run and the owner and editors of its network (like stopping it)."""
    return can_stop_run(user, run)


def problem(run: TrainingRun) -> str | None:
    """Why ``run`` cannot be published, or None."""
    if run.kind == "ensemble":
        return "An ensemble has no weights of its own: publish the runs it combines one by one."
    if training.weights_path(run) is None:
        return training.no_weights_reason(run)
    if run.status not in ("completed", "stopped"):
        return "Only finished runs can be published."
    return None


@dataclass
class Options:
    repo: str  # "name" or "owner/name"
    private: bool = True
    license: str = DEFAULT_LICENSE
    update_existing: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "private": self.private,
            "license": self.license,
            "update_existing": self.update_existing,
        }


def default_repo_name(run: TrainingRun) -> str:
    """A repository name from the network's (or run's) name: "iris-classifier"."""
    name = run.project.name if run.project is not None and run.project.name else run.name
    slug = re.sub(r"[-.]{2,}", "-", slugify(name or ""))[:96].strip("-.")
    return slug if slug and NAME.fullmatch(slug) else "brickdnn-model"


def parse_options(data: Any, run: TrainingRun) -> Options:
    """The choices of a publish request, checked. Raises HubInputError."""
    if not isinstance(data, dict):
        data = {}
    repo = data.get("repo")
    if repo is None or (isinstance(repo, str) and not repo.strip()):
        repo = default_repo_name(run)
    if not isinstance(repo, str):
        raise HubInputError("'repo' must be the name of the repository, e.g. “iris-classifier”.")
    repo = repo.strip().removeprefix("https://huggingface.co/").strip("/")
    parts = repo.split("/")
    if len(parts) > 2 or not all(_valid_name(part) for part in parts):
        raise HubInputError(
            "The repository name may only contain letters, digits, “-”, “_” and “.” (at most "
            "96 characters, not starting or ending with “-” or “.”), optionally after your "
            "user or organization name and “/”."
        )
    private = data.get("private", True)
    update = data.get("update_existing", False)
    if not isinstance(private, bool) or not isinstance(update, bool):
        raise HubInputError("'private' and 'update_existing' must be true or false.")
    license_ = data.get("license") or DEFAULT_LICENSE
    if license_ not in LICENSES:
        raise HubInputError(f"Choose one of these licences: {', '.join(LICENSES)}.")
    return Options(repo, private, license_, update)


def choices(args: dict[str, Any]) -> dict[str, Any]:
    """The publish choices among the arguments of the MCP tool (or a request)."""
    return {
        key: args[key]
        for key in ("repo", "private", "license", "update_existing")
        if args.get(key) is not None
    }


def grant_arguments(run: TrainingRun, options: Options) -> dict[str, Any]:
    """What an approval of publish_to_huggingface allows (approvals.py; never the token)."""
    return {"run_id": str(run.pk), **options.to_dict()}


def _valid_name(name: str) -> bool:
    return bool(NAME.fullmatch(name)) and ".." not in name and "--" not in name


# --------------------------------------------------------------------------- #
# The files of the repository
# --------------------------------------------------------------------------- #


@dataclass
class Bundle:
    """What gets uploaded: the files and what the model card needs to know about them."""

    files: list[UploadFile]
    family: str
    weights: str  # the file name of the model in the repository
    module: str  # the script's module name
    preprocessing: dict[str, Any] | None
    state: dict[str, Any]

    @property
    def size(self) -> int:
        return sum(file.size for file in self.files)


def _preprocessing(run: TrainingRun, state: dict[str, Any]) -> dict[str, Any] | None:
    found = state.get("preprocessing")
    if isinstance(found, dict) and found.get("kind") not in (None, "demo"):
        return found
    path = training.weights_path(run)
    if path is not None:
        saved = preprocessing_path(path)
        if saved.is_file():
            with contextlib.suppress(ValueError, OSError):
                data = json.loads(saved.read_text())
                if isinstance(data, dict):
                    return data
    return found if isinstance(found, dict) else None


def _script(run: TrainingRun) -> str | None:
    """The generated script, with the loss of the run (and its built-in text dataset) where
    the generator takes them."""
    from brickdnn.codegen import LOSS_OPTIONS, generate_code

    config = run.config if isinstance(run.config, dict) else {}
    dataset = run.dataset if isinstance(run.dataset, dict) else {}
    options: dict[str, Any] = {key: config[key] for key in LOSS_OPTIONS if key in config}
    if dataset.get("kind") == "builtin" and dataset.get("id") in TEXT_DATASETS:
        options["dataset"] = dataset["id"]
    for chosen in (options, {}):
        try:
            return generate_code(run.graph, chosen)
        except Exception:  # these options do not fit, or the network is no longer valid
            continue
    logger.info("No script for publishing run %s", run.pk)
    return None


def to_safetensors(source: Path, target: Path) -> bool:
    """Convert ``weights.pt`` in a process of its own (the web server does not import
    PyTorch); False if PyTorch is missing or the conversion fails."""
    if not training.torch_installed():
        return False
    command = [sys.executable, "-m", "brickdnn.training.safetensors_file", str(source), str(target)]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=CONVERT_TIMEOUT,
            env=training._subprocess_env(),
        )
    except (subprocess.TimeoutExpired, OSError):
        logger.warning("Converting %s to safetensors failed", source, exc_info=True)
        return False
    if result.returncode != 0 or not target.is_file():
        logger.warning("Converting %s to safetensors failed: %s", source, result.stdout[-500:])
        return False
    return True


def repo_files(run: TrainingRun, folder: Path | None, convert: bool = True) -> Bundle:
    """The files to upload (without the model card). ``folder``: where converted weights
    are written; ``convert=False`` names them without converting (for the preview)."""
    family = training.run_family(run)
    directory = training.run_dir(run.pk)
    state = training.state_of(run) or {}
    preprocessing = _preprocessing(run, state)
    module = inference.module_name(run.name)
    files: list[UploadFile] = []
    if family in ("classical", "hybrid"):
        files.append(UploadFile("model.joblib", directory / CLASSICAL_WEIGHTS_FILE))
        weights = "model.joblib"
        if family == "hybrid":
            files.append(UploadFile("model.features.pt", directory / WEIGHTS_FILE))
    else:
        source = directory / WEIGHTS_FILE
        weights = "model.safetensors"
        if not convert:
            planned = training.torch_installed()
            weights = "model.safetensors" if planned else "model.pt"
            files.append(UploadFile(weights, source))  # its size is about the same
        elif folder is not None and to_safetensors(source, folder / "model.safetensors"):
            files.append(UploadFile(weights, folder / "model.safetensors"))
        else:
            weights = "model.pt"
            files.append(UploadFile(weights, source))
    script = _script(run)
    if script is not None:
        files.append(UploadFile(f"{module}.py", content=script.encode()))
    files.append(
        UploadFile("network.json", content=(json.dumps(run.graph, indent=2) + "\n").encode("utf-8"))
    )
    if preprocessing is not None:
        files.append(
            UploadFile(
                "preprocessing.json",
                content=(json.dumps(preprocessing, indent=2) + "\n").encode("utf-8"),
            )
        )
    return Bundle(files, family, weights, module, preprocessing, state)


# --------------------------------------------------------------------------- #
# The model card
# --------------------------------------------------------------------------- #


def _yaml(value: Any) -> str:
    """A YAML scalar: JSON is valid YAML, so strings are written as JSON strings."""
    if isinstance(value, bool) or value is None:
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return json.dumps(round(value, 6) if isinstance(value, float) else value)
    return json.dumps(str(value), ensure_ascii=False)


def _task(run: TrainingRun, bundle: Bundle) -> str | None:
    """classification | regression | segmentation, or None (demo data)."""
    plan = run.plan if isinstance(run.plan, dict) else {}
    task = bundle.state.get("task") or plan.get("task")
    return task if task in ("classification", "regression", "segmentation") else None


def _pipeline_tag(run: TrainingRun, bundle: Bundle) -> str | None:
    kind = (bundle.preprocessing or {}).get("kind")
    if kind in (None, "demo"):
        return None
    return PIPELINE_TASKS.get((kind, _task(run, bundle) or ""))


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value == value and abs(value) != float("inf") else None


METRICS = {
    "classification": [
        ("accuracy", "accuracy", "Accuracy"),
        ("balanced_accuracy", "balanced_accuracy", "Balanced accuracy"),
        ("macro_f1", "f1", "Macro F1"),
        ("log_loss", "log_loss", "Log loss"),
    ],
    "regression": [
        ("mae", "mae", "MAE"),
        ("rmse", "rmse", "RMSE"),
        ("r2", "r_squared", "R²"),
    ],
    "segmentation": [
        ("iou", "mean_iou", "Mean IoU"),
        ("dice", "dice", "Dice"),
        ("pixel_accuracy", "accuracy", "Pixel accuracy"),
    ],
}
SPLIT_NAMES = {"val": "Validation", "test": "Test", "training": "Training (seen rows)"}
PREPROCESSING_TEXT = {
    "tabular": " (columns, means, standard deviations, classes)",
    "image": " (image size, channel means and standard deviations, classes)",
    "tokens": " (vocabulary, text length, classes)",
}
HUB_SPLITS = {"val": "validation", "test": "test", "training": "train"}


def results(run: TrainingRun, bundle: Bundle) -> tuple[str | None, dict[str, dict[str, float]]]:
    """The task and the metrics of every split: ``{"test": {"accuracy": 0.95, ...}}``."""
    task = _task(run, bundle)
    state = bundle.state
    found: dict[str, dict[str, float]] = {}
    if task == "segmentation":
        segmentation = (
            state.get("segmentation") if isinstance(state.get("segmentation"), dict) else {}
        )
        for key, split in (("validation", "val"), ("test", "test")):
            values = segmentation.get(key)
            if isinstance(values, dict):
                found[split] = {
                    name: number
                    for name, _, _ in METRICS["segmentation"]
                    if (number := _number(values.get(name))) is not None
                }
        return task, {k: v for k, v in found.items() if v}
    evaluation = state.get("evaluation") if isinstance(state.get("evaluation"), dict) else None
    if task in METRICS and evaluation:
        for split in ("val", "test", "training"):
            values = evaluation.get(split)
            if not isinstance(values, dict):
                continue
            found[split] = {
                name: number
                for name, _, _ in METRICS[task]
                if (number := _number(values.get(name))) is not None
            }
            curve = values.get("threshold_curve")
            if isinstance(curve, dict) and (auc := _number(curve.get("roc_auc"))) is not None:
                found[split]["roc_auc"] = auc
        return task, {k: v for k, v in found.items() if v}
    metrics = run.metrics if isinstance(run.metrics, dict) else {}
    for split, prefix in (("val", "val_"), ("test", "test_")):
        values = {}
        for name in ("accuracy", "rmse", "r2"):
            number = _number(metrics.get(prefix + name))
            if number is not None:
                values[name] = number
        if values:
            found[split] = values
    return task, found


LABELS = {name: label for task in METRICS.values() for name, _, label in task}
LABELS["roc_auc"] = "ROC AUC"
HUB_METRIC = {name: hub for task in METRICS.values() for name, hub, _ in task}
HUB_METRIC["roc_auc"] = "roc_auc"


def _format(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".") if abs(value) < 1000 else f"{value:,.1f}"


def _dataset(run: TrainingRun) -> tuple[str, str | None]:
    """The dataset's name and its id on the Hub (built-in datasets that are there)."""
    data = run.dataset if isinstance(run.dataset, dict) else {}
    name = data.get("name") or (run.upload.name if run.upload_id and run.upload else None)
    kind = data.get("kind")
    if kind == "random":
        return "random demo data", None
    hub_id = HUB_DATASETS.get(str(data.get("id"))) if kind == "builtin" else None
    return str(name or data.get("id") or "a dataset"), hub_id


def _layers(graph: Any) -> list[tuple[str, str, str]]:
    """(id, type, parameters) of every node of the graph, in its order."""
    rows = []
    for node in (graph or {}).get("nodes") or []:
        if not isinstance(node, dict):
            continue
        params = node.get("params") if isinstance(node.get("params"), dict) else {}
        shown = ", ".join(
            f"{key}={json.dumps(value, ensure_ascii=False)}"
            for key, value in list(params.items())[:6]
            if not isinstance(value, (dict, list)) or len(json.dumps(value)) < 40
        )
        rows.append((str(node.get("id", "")), str(node.get("type", "")), shown))
    return rows


def _md(text: Any) -> str:
    """Text for a Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ")


MODEL_FILES = {"model.safetensors", "model.pt", "model.joblib", "model.features.pt"}
FAMILY_TEXT = {
    "neural": "a neural network (PyTorch)",
    "classical": "a classical model (scikit-learn)",
    "hybrid": "a hybrid network: neural layers (PyTorch) learn the features of a classical "
    "model (scikit-learn)",
}
SETTINGS = [
    ("epochs", "Epochs"),
    ("batch_size", "Batch size"),
    ("optimizer", "Optimizer"),
    ("learning_rate", "Learning rate"),
    ("weight_decay", "Weight decay"),
    ("val_split", "Validation share"),
    ("test_split", "Test share"),
    ("patience", "Early stopping patience"),
    ("max_samples", "Max. samples"),
    ("seed", "Seed"),
]


def usage(run: TrainingRun, bundle: Bundle, repo_id: str) -> str | None:
    """Python code that downloads the repository, loads the model and predicts, or None if
    there is no script or the run's inputs are not known."""
    paths = {file.path_in_repo for file in bundle.files}
    if f"{bundle.module}.py" not in paths:
        return None
    archive = "model.zip" if bundle.family == "hybrid" else bundle.weights
    try:
        snippet = inference.usage_snippet(run.plan, bundle.preprocessing, bundle.module, archive)
    except (inference.UnknownPreprocessing, KeyError, TypeError, ValueError, IndexError):
        return None
    snippet = snippet.replace(f"# model_code, saved as {bundle.module}.py", f"# {bundle.module}.py")
    snippet = snippet.replace(f"# {bundle.module}.py from the ZIP file", f"# {bundle.module}.py")
    if bundle.weights == "model.safetensors":
        loader = (
            'model.load_state_dict(torch.load("model.safetensors", map_location="cpu", '
            "weights_only=True))"
        )
        if loader not in snippet:
            return None
        snippet = snippet.replace(loader, 'model.load_state_dict(load_file("model.safetensors"))')
        snippet = snippet.replace(
            "import torch\n", "import torch\nfrom safetensors.torch import load_file\n", 1
        )
    # The model files are read from the downloaded folder; the user's own files (an image to
    # predict) stay where the code runs.
    for name in MODEL_FILES & paths:
        if f'"{name}"' not in snippet:
            return None
        snippet = snippet.replace(f'"{name}"', f'folder / "{name}"')
    header = (
        "import sys\n"
        "from pathlib import Path\n\n"
        "from huggingface_hub import snapshot_download\n\n"
        f'folder = Path(snapshot_download("{repo_id}"))  # the files of this repository\n'
        "sys.path.insert(0, str(folder))  # to import the script\n\n"
    )
    return header + snippet


def _requirements(bundle: Bundle, code: str) -> str:
    """The packages the code of the model card needs."""
    packages = ["huggingface_hub"]
    if bundle.family != "classical":
        packages.append("torch")
    if bundle.weights == "model.safetensors":
        packages.append("safetensors")
    if bundle.family != "neural":
        packages += ["scikit-learn", "joblib"]
    if "from PIL import" in code:
        packages.append("pillow")
    return " ".join(packages)


def model_card(run: TrainingRun, bundle: Bundle, options: Options, repo_id: str) -> str:
    """``README.md`` of the repository: YAML metadata and the explanation."""
    task, splits = results(run, bundle)
    pipeline = _pipeline_tag(run, bundle)
    dataset_name, hub_dataset = _dataset(run)
    library = "sklearn" if bundle.family == "classical" else "pytorch"
    tags = ["brickdnn", "pytorch" if bundle.family != "classical" else "scikit-learn"]
    if bundle.family == "hybrid":
        tags.append("scikit-learn")
    if pipeline:
        tags.append(pipeline)
    if bundle.family != "neural":
        tags.append("tabular" if (bundle.preprocessing or {}).get("kind") == "tabular" else "")
    tags = [tag for tag in dict.fromkeys(tags) if tag]
    name = repo_id.split("/", 1)[1]

    yaml = ["---", f"library_name: {library}", f"license: {options.license}", "tags:"]
    yaml += [f"- {_yaml(tag)}" for tag in tags]
    if pipeline:
        yaml.append(f"pipeline_tag: {pipeline}")
    if hub_dataset:
        yaml += ["datasets:", f"- {_yaml(hub_dataset)}"]
    used = sorted({HUB_METRIC[key] for values in splits.values() for key in values})
    if used:
        yaml += ["metrics:"] + [f"- {_yaml(metric)}" for metric in used]
    if pipeline and splits:
        yaml += [
            "model-index:",
            f"- name: {_yaml(name)}",
            "  results:",
        ]
        for split, values in splits.items():
            yaml += [
                "  - task:",
                f"      type: {pipeline}",
                "    dataset:",
                f"      name: {_yaml(dataset_name)}",
                f"      type: {_yaml(hub_dataset or slugify(dataset_name) or 'dataset')}",
                f"      split: {HUB_SPLITS[split]}",
                "    metrics:",
            ]
            for key, value in values.items():
                yaml += [
                    f"    - type: {HUB_METRIC[key]}",
                    f"      value: {_yaml(value)}",
                    f"      name: {_yaml(f'{LABELS[key]} ({HUB_SPLITS[split]})')}",
                ]
    yaml.append("---")

    plan = run.plan if isinstance(run.plan, dict) else {}
    state = bundle.state
    lines = [
        "",
        f"# {run.name}",
        "",
        f"{_sentence(FAMILY_TEXT.get(bundle.family, 'a model'))}"
        + (f" for {task}" if task else "")
        + f", built from bricks and trained with [Brick DNN]({SITE}) on {dataset_name}.",
        "",
        "## Model",
        "",
    ]
    if run.total_params:
        lines.append(f"- Parameters: {run.total_params:,}")
    classical = plan.get("classical") if isinstance(plan.get("classical"), dict) else None
    if classical:
        lines.append(f"- Classical model: {classical.get('estimator') or classical.get('type')}")
    with contextlib.suppress(Exception):
        inputs = inference.describe_inputs(run.plan, bundle.preprocessing)
        if inputs:
            lines.append(f"- Inputs: {inputs}")
    names = None
    with contextlib.suppress(Exception):
        names = inference.class_names(run.plan, bundle.preprocessing)
    if names:
        shown = ", ".join(str(n) for n in names[:20]) + (" …" if len(names) > 20 else "")
        lines.append(f"- Classes: {shown}")
    layers = _layers(run.graph)
    if layers:
        lines += ["", "| Layer | Type | Parameters |", "| --- | --- | --- |"]
        lines += [f"| {_md(i)} | {_md(t)} | {_md(p)} |" for i, t, p in layers[:60]]
        if len(layers) > 60:
            lines.append(f"| … | {len(layers) - 60} more | |")
        lines += ["", "The whole graph is in `network.json` (open it in Brick DNN to edit it)."]

    lines += ["", "## Training", "", f"- Dataset: {dataset_name}"]
    sizes = [
        (label, state.get(key))
        for key, label in (
            ("train_size", "training"),
            ("val_size", "validation"),
            ("test_size", "test"),
        )
    ]
    sizes = [(label, value) for label, value in sizes if isinstance(value, int) and value > 0]
    if sizes:
        lines.append("- Rows / samples: " + ", ".join(f"{v:,} {label}" for label, v in sizes))
    loss = state.get("loss") if isinstance(state.get("loss"), dict) else None
    if loss and loss.get("label"):
        lines.append(f"- Loss: {loss['label']}")
    config = run.config if isinstance(run.config, dict) else {}
    if bundle.family != "classical":
        builtin = isinstance(run.dataset, dict) and run.dataset.get("kind") == "builtin"
        for key, label in SETTINGS:
            value = config.get(key)
            if key == "test_split" and builtin:
                continue  # built-in datasets are tested on their own test set
            if value not in (None, ""):
                lines.append(f"- {label}: {value}")
        if state.get("weights_epoch"):
            note = ", stopped early" if state.get("stopped_early") else ""
            lines.append(f"- Kept weights: epoch {state['weights_epoch']}{note}")
    best_trial = run.best_trial if isinstance(run.best_trial, dict) else None
    if best_trial and best_trial.get("params"):
        params = ", ".join(f"{k}={v}" for k, v in best_trial["params"].items())
        lines.append(f"- Hyperparameter search: best of {run.trials_done} trials ({params})")

    lines += ["", "## Results", ""]
    if splits:
        keys = list(dict.fromkeys(key for values in splits.values() for key in values))
        lines += [
            "| Split | " + " | ".join(LABELS[key] for key in keys) + " |",
            "| --- |" + " --- |" * len(keys),
        ]
        for split, values in splits.items():
            cells = [_format(values[key]) if key in values else "–" for key in keys]
            lines.append(f"| {SPLIT_NAMES[split]} | " + " | ".join(cells) + " |")
        if "training" in splits:
            lines += [
                "",
                "Without validation and test rows, the model is measured on its training rows: "
                "these numbers are optimistic.",
            ]
    elif task is None:
        lines.append("Trained on random demo data: the outputs mean nothing.")
    else:
        lines.append("No results were measured for this run.")

    script = next((f for f in bundle.files if f.path_in_repo == f"{bundle.module}.py"), None)
    # Networks for tables come with a training loop for a CSV file and a command line.
    trains = script is not None and b"argparse" in script.read()
    kind = (bundle.preprocessing or {}).get("kind")
    lines += ["", "## Files", "", "| File | What |", "| --- | --- |"]
    what = {
        "model.safetensors": "The trained weights (state_dict of `NeuralNetwork`), safetensors",
        "model.pt": "The trained weights (state_dict of `NeuralNetwork`), `torch.save`",
        "model.joblib": "The fitted scikit-learn pipeline (joblib, a pickle: only load files "
        "you trust)",
        "model.features.pt": "The neural layers that compute the features "
        "(`FeatureExtractor`, `torch.save`)",
        f"{bundle.module}.py": "The generated script: the model's code"
        + (", training on a CSV file and prediction" if trains else ""),
        "network.json": "The network as Brick DNN builds it (bricks and connections)",
        "preprocessing.json": "What new inputs need before the model sees them, as measured "
        "on the training data" + PREPROCESSING_TEXT.get(kind, ""),
    }
    for file in bundle.files:
        lines.append(f"| `{file.path_in_repo}` | {what.get(file.path_in_repo, '')} |")

    code = usage(run, bundle, repo_id)
    lines += ["", "## How to use", ""]
    if code:
        lines += [
            f"Download the files and predict (`pip install {_requirements(bundle, code)}`):",
            "",
            "```python",
            code.rstrip(),
            "```",
        ]
        if '"example.png"' in code:
            lines += ["", "`example.png` stands for an image of your own."]
        if options.private:
            lines += [
                "",
                "While the repository is private, log in first (`huggingface-cli login`) so "
                "that the files can be downloaded.",
            ]
    elif script is not None:
        lines.append(
            f"Load `{bundle.weights}` with the model's class in `{bundle.module}.py`; "
            "`preprocessing.json` says how inputs are prepared."
        )
    else:
        lines.append(
            "Open `network.json` in Brick DNN to get the model's code, then load "
            f"`{bundle.weights}` with it."
        )
    if trains:
        lines += [
            "",
            f"`{bundle.module}.py` also trains the model again on your own CSV file and "
            f"predicts from the command line: `python {bundle.module}.py --help` shows how.",
        ]
    lines += [
        "",
        "## About",
        "",
        f"Built and trained with [Brick DNN]({SITE}), which turns networks of bricks into "
        "clean PyTorch and scikit-learn code. This model card was generated from the run "
        f"“{run.name}”.",
        "",
    ]
    return "\n".join(yaml + lines)


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:]


# --------------------------------------------------------------------------- #
# Publishing
# --------------------------------------------------------------------------- #


def _too_large(size: int) -> HubError:
    return HubError(
        f"The files of this run have {size / 1024 / 1024:.1f} MB; at most "
        f"{settings.HUGGINGFACE_MAX_UPLOAD_MB:g} MB can be published from this server. "
        "Download them and upload them with the Hugging Face tools instead.",
        status=413,
        code="too_large",
    )


def preview(run: TrainingRun, data: Any = None) -> dict[str, Any]:
    """What publishing would upload (no token, nothing sent): the files with their
    (approximate) sizes and the model card."""
    options = parse_options(data or {}, run)
    bundle = repo_files(run, None, convert=False)
    repo_id = options.repo if "/" in options.repo else f"your-username/{options.repo}"
    card = model_card(run, bundle, options, repo_id)
    files = [{"path": "README.md", "size": len(card.encode())}] + [
        {"path": file.path_in_repo, "size": file.size} for file in bundle.files
    ]
    total = sum(file["size"] for file in files)
    return {
        "repo": options.repo,
        "default_repo": default_repo_name(run),
        "files": files,
        "size": total,
        "max_upload_mb": settings.HUGGINGFACE_MAX_UPLOAD_MB,
        "too_large": total > max_upload_bytes(),
        "model_card": card,
    }


def publish(user, run: TrainingRun, token: HubToken, options: Options) -> dict[str, Any]:
    """Create (or, if allowed, update) the model repository and upload the run's files in
    one commit. Raises HubError (for the user) or HubInputError."""
    deadline = time.monotonic() + TIME_LIMIT
    reason = problem(run)
    if reason is not None:
        raise HubInputError(reason)
    # Measure before anything is sent: the weights are about as large as converted.
    estimate = repo_files(run, None, convert=False)
    if estimate.size > max_upload_bytes():
        raise _too_large(estimate.size)
    hub = client(token, deadline)
    account = hub.whoami()
    if account.get("role") == "read":
        raise HubError(
            "This access token can only read. Create a token with the “Write” role (or a "
            "fine-grained one that may write to your repositories) and try again.",
            status=403,
            code="token_read_only",
        )
    repo_id = options.repo if "/" in options.repo else f"{account['name']}/{options.repo}"
    created = True
    url = f"{hub.endpoint}/{repo_id}"
    work = training.training_dir()
    work.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=".hub-publish-", dir=work))
    try:
        # Everything is ready (and small enough) before the repository is created.
        bundle = repo_files(run, folder)
        card = model_card(run, bundle, options, repo_id)
        files = [UploadFile("README.md", content=card.encode("utf-8")), *bundle.files]
        total = sum(file.size for file in files)
        if total > max_upload_bytes():
            raise _too_large(total)
        try:
            url, created = hub.create_repo(repo_id, options.private)
        except HubError as exc:
            if exc.code != "name_taken":
                raise
            if not options.update_existing:
                raise HubError(
                    f"A model repository “{repo_id}” exists already. Choose another name, or "
                    "allow updating the existing repository.",
                    status=409,
                    code="name_taken",
                ) from None
            created = False
        try:
            hub.preupload(repo_id, files)
            hub.upload_lfs(repo_id, [file for file in files if file.upload_mode == "lfs"])
            commit = hub.commit(
                repo_id,
                files,
                SUMMARY,
                f"Run “{run.name}” published from Brick DNN ({SITE}).",
            )
        except HubError as exc:
            if created:  # say so: publishing again under this name needs "update"
                exc.message += (
                    f" The repository “{repo_id}” was created and is still empty: allow "
                    "updating the existing repository to publish into it, or delete it on "
                    "Hugging Face."
                )
                exc.args = (exc.message,)
                exc.created_repo = repo_id
            raise
        uploaded = [{"path": file.path_in_repo, "size": file.size} for file in files]
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    touch(run)
    logger.info("User %s published run %s to Hugging Face (%s files)", user.pk, run.pk, len(files))
    return {
        "repo_id": repo_id,
        "url": url,
        "created": created,
        "private": options.private if created else None,
        "license": options.license,
        "commit_url": commit.get("commitUrl"),
        "commit": commit.get("commitOid"),
        "files": uploaded,
        "size": total,
    }


def find_run(user, run_id: Any) -> TrainingRun:
    try:
        run = visible_runs(user).filter(pk=run_id).first()
    except (ValueError, TypeError, ValidationError):  # not a run id
        run = None
    if run is None:
        raise NotFound("No such training run.")
    return training.sync_run(run)


# --------------------------------------------------------------------------- #
# API: GET (preview) and POST (publish) /api/training/runs/<id>/huggingface/
# --------------------------------------------------------------------------- #


def error_response(exc: HubError) -> Response:
    data = {"detail": exc.message, "code": exc.code}
    if exc.created_repo:
        data["created_repo"] = exc.created_repo
    return Response(data, status=exc.status)


class PublishView(APIView):
    """``GET``: what would be uploaded (``?repo=&license=``). ``POST {"hf_token", "repo"?,
    "private"?, "license"?, "update_existing"?}``: publish. The token is used for this one
    request and then forgotten."""

    permission_classes = [permissions.IsAuthenticated]
    throttle_scope = "huggingface"

    def get_throttles(self):
        throttles = [UserRateThrottle()]
        if self.request.method == "POST":
            throttles.append(ScopedRateThrottle())
        return throttles

    def get(self, request: Request, pk) -> Response:
        run = find_run(request.user, pk)
        if not enabled():
            return Response({"detail": DISABLED}, status=status.HTTP_403_FORBIDDEN)
        reason = problem(run)
        if reason is not None:
            return Response({"detail": reason}, status=status.HTTP_400_BAD_REQUEST)
        try:
            data = preview(run, {k: request.query_params.get(k) for k in ("repo", "license")})
        except HubInputError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        data["can_publish"] = can_publish(request.user, run)
        return Response(data)

    def post(self, request: Request, pk) -> Response:
        run = find_run(request.user, pk)
        if not enabled():
            return Response({"detail": DISABLED}, status=status.HTTP_403_FORBIDDEN)
        if not can_publish(request.user, run):
            return Response(
                {
                    "detail": "Only who started the run and the editors of its network can "
                    "publish it."
                },
                status=status.HTTP_403_FORBIDDEN,
            )
        body = request.data if isinstance(request.data, dict) else {}
        try:
            # Taken out of the request data at once: only the wrapper, which never shows
            # its value, goes further.
            token = HubToken(body.pop("hf_token", None))
            options = parse_options(body, run)
            result = publish(request.user, run, token, options)
        except ValueError as exc:  # HubInputError and an invalid token
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except HubError as exc:
            return error_response(exc)
        return Response(result, status=status.HTTP_201_CREATED)
