import json

from django.conf import settings
from django.http import FileResponse, HttpResponse
from rest_framework import mixins, permissions, renderers, status, viewsets
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.exceptions import APIException, NotAuthenticated, PermissionDenied
from rest_framework.parsers import MultiPartParser
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView
from rest_framework_simplejwt.views import TokenObtainPairView

import brickdnn
from brickdnn import service
from brickdnn.training import ACTIVE_STATUSES, DatasetError

from . import downloads, shared_data
from . import training_service as training
from .accounts import LoginSerializer, tokens_for
from .api_tokens import MAX_TOKENS_PER_USER, IsLoggedInUser, create_token
from .models import ApiToken, Dataset, TrainingRun
from .retention import touch
from .serializers import (
    ApiTokenCreateSerializer,
    ApiTokenSerializer,
    DatasetSerializer,
    RegisterSerializer,
    RunDetailSerializer,
    RunSummarySerializer,
    UserSerializer,
)

# --------------------------------------------------------------------------- #
# Code generation (public, no login needed)
# The logic lives in brickdnn.service, which the browser demo uses as well.
# --------------------------------------------------------------------------- #


@api_view(["GET"])
def health(request: Request) -> Response:
    return Response({"status": "ok", "version": brickdnn.__version__})


@api_view(["GET"])
def layers(request: Request) -> Response:
    """All layer types with their parameters: the UI builds its palette and forms from this."""
    return Response(service.layers())


@api_view(["GET"])
def examples(request: Request) -> Response:
    return Response(service.examples())


@api_view(["POST"])
def analyze_graph(request: Request) -> Response:
    """Validate a graph and return the inferred shape and parameter count of every layer."""
    return Response(service.analyze_graph(request.data))


@api_view(["POST"])
def generate(request: Request) -> Response | HttpResponse:
    """Generate a PyTorch script (see ``brickdnn.service.generate`` for the body).

    Add ``?download=1`` to get the file as an attachment instead of JSON.
    """
    status_code, payload = service.generate(request.data)
    if status_code == status.HTTP_200_OK and request.query_params.get("download"):
        response = HttpResponse(payload["code"], content_type="text/x-python; charset=utf-8")
        response["Content-Disposition"] = f'attachment; filename="{payload["filename"]}"'
        return response
    return Response(payload, status=status_code)


# --------------------------------------------------------------------------- #
# Accounts
# --------------------------------------------------------------------------- #


class LoginView(TokenObtainPairView):
    serializer_class = LoginSerializer
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "auth"


class RegisterView(APIView):
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "auth"

    def post(self, request: Request) -> Response:
        if not settings.ALLOW_REGISTRATION:
            return Response(
                {"detail": "Registration is disabled on this server."},
                status=status.HTTP_403_FORBIDDEN,
            )
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        return Response(
            {"user": UserSerializer(user).data, **tokens_for(user)},
            status=status.HTTP_201_CREATED,
        )


@api_view(["GET"])
@permission_classes([permissions.IsAuthenticated])
def me(request: Request) -> Response:
    return Response(UserSerializer(request.user).data)


class ApiTokenViewSet(
    mixins.ListModelMixin,
    mixins.DestroyModelMixin,
    viewsets.GenericViewSet,
):
    """Personal API tokens of the logged-in user (for scripts and the MCP server).

    Only a website login can manage them, not an API token. The token itself is returned
    once, by ``create``; deleting a token revokes it at once.
    """

    serializer_class = ApiTokenSerializer
    permission_classes = [IsLoggedInUser]

    def get_queryset(self):
        # The access tokens of apps connected with OAuth are managed as connections.
        return ApiToken.objects.filter(owner=self.request.user, connection__isnull=True)

    def create(self, request: Request) -> Response:
        serializer = ApiTokenCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        if self.get_queryset().count() >= MAX_TOKENS_PER_USER:
            return Response(
                {"detail": f"You can have at most {MAX_TOKENS_PER_USER} API tokens."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        token, key = create_token(request.user, **serializer.validated_data)
        return Response(
            {**ApiTokenSerializer(token).data, "token": key}, status=status.HTTP_201_CREATED
        )


# --------------------------------------------------------------------------- #
# Saved projects
# --------------------------------------------------------------------------- #


# ProjectViewSet and sharing: see projects.py.


# --------------------------------------------------------------------------- #
# Local training
# --------------------------------------------------------------------------- #


class Conflict(APIException):
    status_code = status.HTTP_409_CONFLICT
    default_detail = "The request conflicts with the current state."
    default_code = "conflict"


class BinaryRenderer(renderers.BaseRenderer):
    """Lets clients ask for ``application/octet-stream`` (errors are still JSON)."""

    media_type = "application/octet-stream"
    format = "bin"
    charset = None
    render_style = "binary"

    def render(self, data, accepted_media_type=None, renderer_context=None):
        if isinstance(data, bytes):
            return data
        return json.dumps(data).encode()


def _disabled_response() -> Response | None:
    if not settings.TRAINING_ENABLED:
        return Response(
            {"detail": training.DISABLED_REASON}, status=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    return None


@api_view(["GET"])
def training_info(request: Request) -> Response:
    """Whether training is possible on this server: PyTorch version, devices, datasets."""
    return Response(training.capabilities())


@api_view(["POST"])
@permission_classes([permissions.IsAuthenticated])
def training_check(request: Request) -> Response:
    """Check a network against a dataset before training: ``{"graph", "dataset",
    "config"?}``. With ``config`` (the training settings), the checks that depend on them
    are made too (e.g. Max. samples against the row limit of classical models)."""
    if (response := _disabled_response()) is not None:
        return response
    data = request.data if isinstance(request.data, dict) else {}
    graph = data.get("graph")
    try:
        training.validate_graph(graph)
        dataset = training.resolve_dataset(request.user, data.get("dataset"))
        config = training.parse_config(data.get("config")) if "config" in data else None
    except training.TrainingInputError as exc:
        return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    return Response(training.check(graph, dataset, config).to_dict())


class DatasetViewSet(
    mixins.ListModelMixin,
    mixins.CreateModelMixin,
    mixins.RetrieveModelMixin,
    mixins.DestroyModelMixin,
    viewsets.GenericViewSet,
):
    """Uploaded training data: a CSV file or a ZIP file with one folder per class."""

    serializer_class = DatasetSerializer
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser]
    throttle_scope = "uploads"

    def get_queryset(self):
        # Your own datasets and the ones shared in your projects; only your own can be
        # deleted.
        user = self.request.user
        datasets = (
            Dataset.objects.filter(owner=user)
            if self.action == "destroy"
            else shared_data.visible_datasets(user)
        )
        return datasets.select_related("prepared_from", "owner")

    def get_throttles(self):
        throttles = super().get_throttles()
        if self.action == "create":
            throttles.append(ScopedRateThrottle())
        return throttles

    def get_object(self) -> Dataset:
        dataset = super().get_object()
        if self.action == "retrieve":
            touch(dataset)
        return dataset

    def create(self, request: Request, *args, **kwargs) -> Response:
        if (response := _disabled_response()) is not None:
            return response
        limit = training.max_upload_bytes()
        too_large = Response(
            {
                "detail": "The file is too large. Uploads are limited to "
                f"{settings.TRAINING_MAX_UPLOAD_MB} MB."
            },
            status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )
        # Reject before the body is read (allowing a little room for the form fields).
        try:
            length = int(request.META.get("CONTENT_LENGTH") or 0)
        except ValueError:
            length = 0
        if length > limit + 64 * 1024:
            return too_large
        upload = request.FILES.get("file")
        if upload is None:
            return Response(
                {"detail": "Choose a file to upload."}, status=status.HTTP_400_BAD_REQUEST
            )
        if upload.size is not None and upload.size > limit:
            return too_large
        name = request.data.get("name")
        try:
            dataset = training.save_upload(
                request.user, upload, name if isinstance(name, str) else None
            )
        except DatasetError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(
            DatasetSerializer(dataset, context={"request": request}).data,
            status=status.HTTP_201_CREATED,
        )

    def perform_destroy(self, instance: Dataset) -> None:
        try:
            training.delete_dataset(instance)
        except training.DatasetInUse as exc:
            raise Conflict(str(exc)) from None


class TrainingRunViewSet(
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    mixins.DestroyModelMixin,
    viewsets.GenericViewSet,
):
    """Training runs of the logged-in user and of their shared projects (``?project=<id>``).
    Training happens in a separate process."""

    permission_classes = [permissions.IsAuthenticated]
    throttle_scope = "training"

    def get_queryset(self):
        runs = shared_data.visible_runs(self.request.user).select_related("owner")
        project = self.request.query_params.get("project")
        if project is not None and project.isdigit():
            runs = runs.filter(project_id=int(project))
        return runs

    def get_serializer_class(self):
        return RunSummarySerializer if self.action == "list" else RunDetailSerializer

    def get_throttles(self):
        throttles = super().get_throttles()
        if self.action == "create":
            throttles.append(ScopedRateThrottle())
        return throttles

    def get_object(self) -> TrainingRun:
        run = super().get_object()
        touch(run)
        return training.sync_run(run)

    def list(self, request: Request, *args, **kwargs) -> Response:
        training.sync_runs(self.get_queryset().filter(status__in=ACTIVE_STATUSES))
        return super().list(request, *args, **kwargs)

    def create(self, request: Request) -> Response:
        """Start a training: ``{"name", "graph", "dataset", "config", "tuning"?, "project"?}``.

        With ``project`` (a project you own or edit), the run belongs to that project too."""
        data = request.data if isinstance(request.data, dict) else {}
        try:
            run = training.start_run(
                request.user,
                data.get("name"),
                data.get("graph"),
                data.get("dataset"),
                data.get("config"),
                data.get("tuning"),
                data.get("project"),
            )
        except training.TrainingUnavailable as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except training.TrainingInputError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except training.IncompatibleNetwork as exc:
            result = exc.compatibility
            return Response(
                {"detail": str(exc), "errors": result.errors, "warnings": result.warnings},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except training.TrainingBusy as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_409_CONFLICT)
        return Response(RunDetailSerializer(run).data, status=status.HTTP_201_CREATED)

    def perform_destroy(self, instance: TrainingRun) -> None:
        if not shared_data.can_delete_run(self.request.user, instance):
            raise PermissionDenied("Only who started it and the project's owner can delete it.")
        training.delete_run(instance)

    @action(detail=True, methods=["post"])
    def stop(self, request: Request, pk=None) -> Response:
        """Stop after the current batch; the weights trained so far are kept."""
        run = self.get_object()
        if not shared_data.can_stop_run(request.user, run):
            raise PermissionDenied("Only who started it and the project's editors can stop it.")
        run = training.stop_run(run)
        return Response(RunDetailSerializer(run).data)

    @action(detail=True, methods=["post"], throttle_scope="predict")
    def predict(self, request: Request, pk=None) -> Response:
        """Run the trained network on a few inputs: ``{"inputs", "input_scale"?, "top_k"?}``
        (see brickdnn.training.inference). Everyone who can see the run may."""
        run = self.get_object()
        data = request.data if isinstance(request.data, dict) else {}
        try:
            prediction = training.prepare_prediction(
                run, data.get("inputs"), data.get("input_scale"), data.get("top_k")
            )
        except training.TrainingUnavailable as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except training.TrainingInputError as exc:
            return Response(
                {"detail": str(exc), "inputs": training.describe_inputs(run)},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except training.PredictionUnavailable as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_409_CONFLICT)
        # Only requests that start a process count against the rate limit.
        throttle = ScopedRateThrottle()
        if not throttle.allow_request(request, self):
            self.throttled(request, throttle.wait())
        try:
            return Response(training.run_prediction(prediction))
        except training.TrainingBusy as exc:
            return Response(
                {"detail": str(exc)},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
                headers={"Retry-After": "5"},
            )
        except training.PredictionFailed as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(
        detail=True,
        methods=["get"],
        renderer_classes=[renderers.JSONRenderer, BinaryRenderer],
        permission_classes=[permissions.AllowAny],  # checked below: login or signed link
    )
    def weights(self, request: Request, pk=None) -> Response | FileResponse:
        """The trained weights (``model.state_dict()``) as a ``.pt`` file; for a classical
        model the fitted scikit-learn pipeline as a ``.joblib`` file.

        Logged in, or with ``?sig=`` from a download link (see downloads.py), which works
        without logging in until it expires.
        """
        signature = request.query_params.get("sig")
        if signature is not None:
            run = downloads.run_for_link(signature, pk)
            touch(run)
        elif not request.user.is_authenticated:
            raise NotAuthenticated()
        else:
            run = self.get_object()
        path = training.weights_path(run)
        if path is None:
            return Response(
                {"detail": "No weights available for this run."},
                status=status.HTTP_404_NOT_FOUND,
            )
        response = FileResponse(
            path.open("rb"),
            as_attachment=True,
            filename=training.weights_filename(run),
            content_type="application/octet-stream",
        )
        if signature is not None:
            response["Cache-Control"] = "private, no-store"
            response["Referrer-Policy"] = "no-referrer"
        return response


class ApiRoot(APIView):
    def get(self, request: Request) -> Response:
        base = request.build_absolute_uri("/api/")
        return Response(
            {
                "name": "Brick DNN API",
                "version": brickdnn.__version__,
                "endpoints": {
                    "layers": f"{base}layers/",
                    "examples": f"{base}examples/",
                    "analyze": f"{base}analyze/",
                    "generate": f"{base}generate/",
                    "register": f"{base}auth/register/",
                    "login": f"{base}auth/token/",
                    "api_tokens": f"{base}auth/api-tokens/",
                    "account": f"{base}auth/account/",
                    "legal": f"{base}legal/",
                    "mcp": f"{base}mcp/",
                    "projects": f"{base}projects/",
                    "training": f"{base}training/",
                    "training_check": f"{base}training/check/",
                    "training_datasets": f"{base}training/datasets/",
                    "training_runs": f"{base}training/runs/",
                    "data": f"{base}data/",
                    "data_steps": f"{base}data/steps/",
                    "data_sources": f"{base}data/sources/",
                    "data_preps": f"{base}data/preps/",
                },
            }
        )
