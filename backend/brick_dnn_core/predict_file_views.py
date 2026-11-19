"""``/api/training/runs/<id>/predict-file/``: predict every row of an uploaded CSV file and
download the result (predict_files.py).

POST uploads the file (multipart form) with the options of the result and answers with the
summary and a preview; GET ``…/predict-file/<result id>/`` downloads the result file, which
deletes it; DELETE discards it. Everyone who can see the run may predict (like single
predictions); a result is only for who made it.
"""

from __future__ import annotations

from django.http import FileResponse
from rest_framework import permissions, status
from rest_framework.parsers import MultiPartParser
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView

from . import predict_files
from . import training_service as training
from .budgets import Actor, BudgetError
from .explain_views import visible_run


def _refusal(exc: Exception) -> Response:
    """Why the run cannot predict the file, as the ``predict`` action answers."""
    if isinstance(exc, training.TrainingUnavailable):
        return Response({"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    if isinstance(exc, training.PredictionUnavailable):
        return Response({"detail": str(exc)}, status=status.HTTP_409_CONFLICT)
    return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)


REFUSALS = (
    training.TrainingUnavailable,
    training.PredictionUnavailable,
    training.TrainingInputError,
)


class PredictFileView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser]
    # Only requests that start a process count (see post).
    throttle_scope = "predict"

    def post(self, request: Request, pk=None) -> Response:
        """Form fields: ``file`` (CSV), ``columns`` ("all" or a JSON list of the file's
        columns to keep), ``prediction_column``, ``output`` ("label", "probability",
        "probabilities"), ``class`` and ``input_format`` ("auto", "raw", "prepared"); see
        docs/api.md."""
        too_large = Response(
            {"detail": predict_files.too_large_message()},
            status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )
        # Reject before the body is read (allowing a little room for the form fields).
        try:
            length = int(request.META.get("CONTENT_LENGTH") or 0)
        except ValueError:
            length = 0
        if length > predict_files.max_bytes() + 64 * 1024:
            return too_large
        run = visible_run(request.user, pk)
        upload = request.FILES.get("file")
        if upload is None:
            return Response(
                {"detail": "Send the CSV file as the form field 'file'."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            predict_files.describe(run)  # can this run predict a file at all?
        except REFUSALS as exc:
            return _refusal(exc)
        throttle = ScopedRateThrottle()
        if not throttle.allow_request(request, self):
            self.throttled(request, throttle.wait())
        try:
            return Response(predict_files.predict(run, upload, request.data, Actor.of(request)))
        except predict_files.FileTooLarge:
            return too_large
        except BudgetError as exc:
            return Response(exc.body(), status=exc.status_code)
        except REFUSALS as exc:
            return _refusal(exc)
        except training.TrainingBusy as exc:
            return Response(
                {"detail": str(exc)},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
                headers={"Retry-After": "5"},
            )
        except training.PredictionFailed as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class PredictFileResultView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request: Request, pk=None, result: str = "") -> Response | FileResponse:
        """The result file (CSV, UTF-8). It is deleted with this download; 404 afterwards
        and once it has expired."""
        run = visible_run(request.user, pk)
        try:
            handle, filename, size = predict_files.open_result(run, result, request.user)
        except predict_files.ResultGone as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_404_NOT_FOUND)
        response = FileResponse(
            handle, as_attachment=True, filename=filename, content_type="text/csv; charset=utf-8"
        )
        response["Content-Length"] = str(size)
        response["Cache-Control"] = "private, no-store"
        return response

    def delete(self, request: Request, pk=None, result: str = "") -> Response:
        predict_files.discard(visible_run(request.user, pk), result, request.user)
        return Response(status=status.HTTP_204_NO_CONTENT)
