"""``/api/training/runs/<id>/explain/``: explanations of a trained model (explain_service.py).

GET lists the methods that work for the run; POST computes one. Everyone who can see the
run may (like predictions), and the compute counts against the budgets of who asks.
"""

from __future__ import annotations

from django.core.exceptions import ValidationError
from django.http import Http404
from rest_framework import permissions, status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView

from . import explain_service as explaining
from . import training_service as training
from .budgets import Actor, BudgetError
from .models import TrainingRun
from .retention import touch
from .shared_data import visible_runs


def visible_run(user, pk) -> TrainingRun:
    try:
        run = visible_runs(user).filter(pk=pk).first()
    except (ValidationError, ValueError):
        run = None
    if run is None:
        raise Http404("No such training run.")
    touch(run)
    return training.sync_run(run)


class ExplainView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    # Only requests that start a process count (see post).
    throttle_scope = "predict"

    def get(self, request: Request, pk=None) -> Response:
        """``{"family", "kind", "unavailable", "methods": [{"id", "label", "available",
        "reason"}]}``."""
        return Response(explaining.available(visible_run(request.user, pk)))

    def post(self, request: Request, pk=None) -> Response:
        """``{"method", "sample"?: n | "input"?: ..., "input_scale"?, "target"?, "layer"?,
        "node"?}``, see docs/explain.md."""
        run = visible_run(request.user, pk)
        try:
            explanation = explaining.prepare(run, request.data)
        except training.TrainingUnavailable as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except training.TrainingInputError as exc:
            return Response(
                {"detail": str(exc), "inputs": training.describe_inputs(run)},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except training.PredictionUnavailable as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_409_CONFLICT)
        if explaining.needs_process(explanation):
            throttle = ScopedRateThrottle()
            if not throttle.allow_request(request, self):
                self.throttled(request, throttle.wait())
        try:
            return Response(explaining.run_explanation(explanation, Actor.of(request)))
        except BudgetError as exc:
            return Response(exc.body(), status=exc.status_code)
        except training.TrainingBusy as exc:
            return Response(
                {"detail": str(exc)},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
                headers={"Retry-After": "5"},
            )
        except training.PredictionFailed as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
