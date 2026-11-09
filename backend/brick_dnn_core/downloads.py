"""Short-lived links that download the trained weights of a run without logging in.

AI assistants cannot pass binary files through MCP, so the tool get_weights_download hands
out a link instead: ``/api/training/runs/<id>/weights/?sig=…``. The signature (Django's
``TimestampSigner`` with the SECRET_KEY and a salt of its own) binds the link to the run and
to the user who asked for it, and it is valid for :data:`LINK_SECONDS`. When the link is
used, that user must still exist, be active and be allowed to see the run, so removing
someone from a project also ends their links.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from urllib.parse import urlencode

from django.contrib.auth import get_user_model
from django.core import signing
from django.urls import reverse
from django.utils import timezone
from rest_framework.exceptions import PermissionDenied
from rest_framework.request import Request

from brickdnn import GraphValidationError
from brickdnn.codegen import generate_code
from brickdnn.training import inference

from . import training_service as training
from .models import TrainingRun
from .oauth import site_url
from .shared_data import visible_runs

SALT = "brick_dnn_core.downloads.weights"
LINK_SECONDS = 15 * 60
NEW_LINK = "Ask for a new one (MCP tool get_weights_download)."


def _signer() -> signing.TimestampSigner:
    return signing.TimestampSigner(salt=SALT)


def weights_link(request: Request, run: TrainingRun, user) -> dict[str, Any]:
    """A signed link to the weights of ``run`` for ``user``, with its expiry."""
    signature = _signer().sign_object({"run": str(run.pk), "user": user.pk})
    path = reverse("training-run-weights", kwargs={"pk": str(run.pk)})
    expires_at = timezone.now() + timedelta(seconds=LINK_SECONDS)
    return {
        "url": f"{site_url(request)}{path}?{urlencode({'sig': signature})}",
        "expires_at": expires_at.isoformat(timespec="seconds"),
        "expires_in_minutes": LINK_SECONDS // 60,
    }


def run_for_link(signature: str, run_id: str) -> TrainingRun:
    """The run a signed link gives access to. Raises ``PermissionDenied`` (403)."""
    try:
        data = _signer().unsign_object(signature, max_age=LINK_SECONDS)
    except signing.SignatureExpired:
        raise PermissionDenied(f"This download link has expired. {NEW_LINK}") from None
    except signing.BadSignature:
        raise PermissionDenied(f"This download link is invalid. {NEW_LINK}") from None
    if not isinstance(data, dict) or data.get("run") != str(run_id):
        raise PermissionDenied(f"This download link is for another run. {NEW_LINK}")
    user = get_user_model().objects.filter(pk=data.get("user"), is_active=True).first()
    run = visible_runs(user).filter(pk=data["run"]).first() if user is not None else None
    if run is None:
        raise PermissionDenied("This download link is no longer valid: no access to this run.")
    return training.sync_run(run)


def weights_info(request: Request, run: TrainingRun, user) -> dict[str, Any]:
    """Everything an assistant needs to hand over the weights: the link, the model code,
    how inputs were prepared and a snippet that loads the weights and predicts."""
    path = training.weights_path(run)
    assert path is not None
    filename = training.weights_filename(run)
    module = inference.module_name(run.name)
    info: dict[str, Any] = {
        "run": {"id": str(run.pk), "name": run.name, "status": run.status},
        **weights_link(request, run, user),
        "filename": filename,
        "size_bytes": path.stat().st_size,
        "format": (
            "PyTorch state_dict of the class NeuralNetwork in model_code (torch.save); load it "
            "with model.load_state_dict(torch.load(path, weights_only=True))."
        ),
        "notes": [
            "The link works without logging in until it expires; treat it like a password.",
        ],
    }
    try:
        code = generate_code(run.graph, {"include_main": False, "include_training": False})
    except (GraphValidationError, ValueError):
        code = None
        info["notes"].append(
            "The network of this run cannot be turned into code any more; call "
            "generate_pytorch_code with its graph (get_training_run) for details."
        )
    info["model_code_filename"] = f"{module}.py"
    info["model_code"] = code
    state = training.state_of(run) or {}
    preprocessing = state.get("preprocessing")
    try:
        info["task"] = run.plan.get("task") if isinstance(run.plan, dict) else None
        info["inputs"] = inference.describe_inputs(run.plan, preprocessing)
        info["class_names"] = inference.class_names(run.plan, preprocessing)
        info["preprocessing"] = preprocessing
        info["how_to_use"] = inference.usage_snippet(run.plan, preprocessing, module, filename)
    except inference.UnknownPreprocessing as exc:
        info["notes"].append(str(exc))
    if code is not None:
        info["notes"].append(
            "Save model_code as "
            f"{module}.py next to the weights; generate_pytorch_code gives the full script "
            "with a demo main and a CSV training loop."
        )
    return info
