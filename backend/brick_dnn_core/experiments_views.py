"""API of the experiment tracker (experiments.py): shared comparisons of runs.

The runs table, the curves, the remembered columns and the tags of a run are actions of
``TrainingRunViewSet`` (views.py)."""

from __future__ import annotations

from rest_framework import permissions, status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView

from . import experiments
from .models import RunComparison


class ComparisonListView(APIView):
    """``GET``: the user's shared comparisons, newest first. ``POST {name?, runs, columns?,
    options?, include_notes?}``: share these runs read-only with a link."""

    permission_classes = [permissions.IsAuthenticated]
    throttle_scope = "sharing"

    def get_throttles(self):
        throttles = super().get_throttles()
        if self.request.method == "POST":
            throttles.append(ScopedRateThrottle())
        return throttles

    def get(self, request: Request) -> Response:
        comparisons = RunComparison.objects.filter(owner=request.user)
        return Response([experiments.comparison_data(c) for c in comparisons])

    def post(self, request: Request) -> Response:
        try:
            comparison = experiments.create_comparison(request.user, request.data)
        except experiments.ExperimentInputError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(experiments.comparison_data(comparison), status=status.HTTP_201_CREATED)


class ComparisonDetailView(APIView):
    """``GET`` one of the user's comparisons; ``DELETE`` it: the link stops working."""

    permission_classes = [permissions.IsAuthenticated]

    def _get(self, request: Request, pk: int) -> RunComparison | None:
        return RunComparison.objects.filter(owner=request.user, pk=pk).first()

    def get(self, request: Request, pk: int) -> Response:
        comparison = self._get(request, pk)
        if comparison is None:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        return Response(experiments.comparison_data(comparison))

    def delete(self, request: Request, pk: int) -> Response:
        comparison = self._get(request, pk)
        if comparison is None:
            return Response({"detail": "Not found."}, status=status.HTTP_404_NOT_FOUND)
        comparison.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class SharedComparisonView(APIView):
    """``GET /api/shared/runs/<token>/``: a comparison shared by link, for everyone."""

    permission_classes = [permissions.AllowAny]

    def get(self, request: Request, token: str) -> Response:
        data = experiments.shared_comparison(token)
        if data is None:
            return Response(
                {"detail": "This link does not exist or was turned off."},
                status=status.HTTP_404_NOT_FOUND,
            )
        return Response(data)
