from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from django.utils import timezone
from rest_framework import serializers

from brickdnn.graph import MAX_EDGES, MAX_NODES

from . import legal
from .api_tokens import EXPIRY_DAYS
from .models import ApiToken, DataPrep, Dataset, DataSource, Project, TrainingRun, UserProfile

User = get_user_model()


def validate_graph_document(value):
    """Light structural check. Semantic errors are allowed: users may save work in progress."""
    if not isinstance(value, dict):
        raise serializers.ValidationError("The graph must be a JSON object.")
    nodes, edges = value.get("nodes", []), value.get("edges", [])
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise serializers.ValidationError("'nodes' and 'edges' must be lists.")
    if len(nodes) > MAX_NODES or len(edges) > MAX_EDGES:
        raise serializers.ValidationError(
            f"Graphs are limited to {MAX_NODES} layers and {MAX_EDGES} connections."
        )
    return value


class ProjectSerializer(serializers.ModelSerializer):
    num_layers = serializers.SerializerMethodField()

    class Meta:
        model = Project
        fields = ["id", "name", "description", "graph", "num_layers", "created_at", "updated_at"]
        read_only_fields = ["id", "created_at", "updated_at"]

    def get_num_layers(self, obj: Project) -> int:
        nodes = obj.graph.get("nodes", []) if isinstance(obj.graph, dict) else []
        return len(nodes)

    def validate_graph(self, value):
        return validate_graph_document(value)


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["id", "username", "email", "date_joined"]


def email_field() -> serializers.EmailField:
    # Required: people are invited to shared networks by e-mail, too.
    return serializers.EmailField(
        max_length=254,
        error_messages={
            "required": "Enter your e-mail address.",
            "blank": "Enter your e-mail address.",
        },
    )


def clean_email(value: str, user=None) -> str:
    """A normalised e-mail address that no other account uses (``user``: the own account)."""
    value = User.objects.normalize_email(value.strip())
    others = User.objects.filter(email__iexact=value)
    if user is not None:
        others = others.exclude(pk=user.pk)
    if others.exists():
        raise serializers.ValidationError("An account with this e-mail address exists already.")
    return value


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, style={"input_type": "password"})
    email = email_field()
    # Required when the server has terms of use (see legal.py).
    accept_terms = serializers.BooleanField(required=False, default=False, write_only=True)

    class Meta:
        model = User
        fields = ["username", "email", "password", "accept_terms"]

    def validate_email(self, value: str) -> str:
        return clean_email(value)

    def validate(self, attrs):
        self.legal_versions = legal.versions()
        if self.legal_versions["terms"] is not None and attrs.get("accept_terms") is not True:
            raise serializers.ValidationError(
                {"accept_terms": "Please agree to the terms of use to create an account."}
            )
        candidate = User(username=attrs.get("username"), email=attrs.get("email", ""))
        validate_password(attrs["password"], user=candidate)
        return attrs

    def create(self, validated_data):
        accepted = validated_data.pop("accept_terms", False)
        user = User.objects.create_user(**validated_data)
        versions = getattr(self, "legal_versions", {})
        profile = UserProfile(user=user)
        if accepted and versions.get("terms"):
            profile.terms_accepted_at = timezone.now()
            profile.terms_version = versions["terms"]
            profile.privacy_version = versions.get("privacy") or ""
        profile.save()
        return user


# --------------------------------------------------------------------------- #
# Local training
# --------------------------------------------------------------------------- #


class ApiTokenSerializer(serializers.ModelSerializer):
    class Meta:
        model = ApiToken
        fields = ["id", "name", "prefix", "read_only", "created_at", "last_used_at", "expires_at"]
        read_only_fields = fields


class ApiTokenCreateSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=100)
    read_only = serializers.BooleanField(default=False)
    expires_in_days = serializers.ChoiceField(
        choices=EXPIRY_DAYS, allow_null=True, required=False, default=None
    )


class DatasetSerializer(serializers.ModelSerializer):
    prepared_from = serializers.SerializerMethodField()
    owner = serializers.SerializerMethodField()
    mine = serializers.SerializerMethodField()
    shared_in = serializers.SerializerMethodField()

    class Meta:
        model = Dataset
        fields = [
            "id",
            "name",
            "kind",
            "size_bytes",
            "created_at",
            "info",
            "prepared_from",
            "owner",
            "mine",
            "shared_in",
        ]
        read_only_fields = fields

    def _user(self):
        request = self.context.get("request")
        return getattr(request, "user", None) or self.context.get("user")

    def get_owner(self, obj: Dataset) -> str:
        return obj.owner.get_username()

    def get_mine(self, obj: Dataset) -> bool:
        user = self._user()
        return user is not None and obj.owner_id == user.pk

    def get_shared_in(self, obj: Dataset) -> list:
        """Projects it is shared in (those the requesting user can see)."""
        from .shared_data import shared_in

        user = self._user()
        return shared_in(user, obj) if user is not None and user.is_authenticated else []

    def get_prepared_from(self, obj: Dataset) -> dict | None:
        """The data preparation (Data tab) the dataset was made from: ``{id, name}``."""
        prep = obj.prepared_from
        return {"id": prep.pk, "name": prep.name} if prep is not None else None


class RunSummarySerializer(serializers.ModelSerializer):
    task = serializers.SerializerMethodField()
    epochs = serializers.SerializerMethodField()
    stop_requested = serializers.SerializerMethodField()

    class Meta:
        model = TrainingRun
        fields = [
            "id",
            "name",
            "status",
            "message",
            "dataset",
            "config",
            "tuning",
            "task",
            "epochs",
            "epochs_done",
            "metrics",
            "best",
            "trials_done",
            "best_trial",
            "has_weights",
            "stop_requested",
            "total_params",
            "created_at",
            "finished_at",
            "project",
            "started_by",
        ]
        read_only_fields = fields

    started_by = serializers.SerializerMethodField()

    def get_started_by(self, obj: TrainingRun) -> str:
        return obj.owner.get_username()

    def get_task(self, obj: TrainingRun) -> str | None:
        return obj.plan.get("task") if isinstance(obj.plan, dict) else None

    def get_epochs(self, obj: TrainingRun) -> int | None:
        return obj.config.get("epochs") if isinstance(obj.config, dict) else None

    def get_stop_requested(self, obj: TrainingRun) -> bool:
        return obj.stop_requested_at is not None


class RunDetailSerializer(RunSummarySerializer):
    state = serializers.SerializerMethodField()

    class Meta(RunSummarySerializer.Meta):
        fields = [*RunSummarySerializer.Meta.fields, "graph", "plan", "state"]
        read_only_fields = fields

    def get_state(self, obj: TrainingRun) -> dict | None:
        from .training_service import state_of

        return state_of(obj)


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #


class DataSourceSerializer(serializers.ModelSerializer):
    """An imported data file. Only the name can be changed."""

    name = serializers.CharField(max_length=120)

    class Meta:
        model = DataSource
        fields = [
            "id",
            "name",
            "filename",
            "size_bytes",
            "n_rows",
            "n_columns",
            "columns",
            "import_options",
            "report",
            "created_at",
        ]
        read_only_fields = [f for f in fields if f != "name"]


class PrepDatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = ["id", "name", "created_at"]
        read_only_fields = fields


class DataPrepSerializer(serializers.ModelSerializer):
    """A preparation. ``steps`` and ``settings`` are checked with brickdnn.data.pipeline;
    the source must belong to the requesting user."""

    name = serializers.CharField(max_length=120, required=False, allow_blank=True)
    source = serializers.PrimaryKeyRelatedField(
        queryset=DataSource.objects.none(),
        error_messages={
            "does_not_exist": "This data source does not exist (any more).",
            "incorrect_type": "Choose one of your data sources.",
        },
    )
    source_name = serializers.CharField(source="source.name", read_only=True)
    steps = serializers.JSONField(required=False)
    settings = serializers.JSONField(required=False)
    # Training datasets made from this preparation, newest first.
    datasets = PrepDatasetSerializer(many=True, read_only=True)

    class Meta:
        model = DataPrep
        fields = [
            "id",
            "name",
            "source",
            "source_name",
            "steps",
            "settings",
            "datasets",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "source_name", "datasets", "created_at", "updated_at"]

    def get_fields(self):
        fields = super().get_fields()
        request = self.context.get("request")
        user = getattr(request, "user", None)
        if user is not None and user.is_authenticated:
            # Editors of a shared preparation choose among the owner's sources.
            owner = self.instance.owner if isinstance(self.instance, DataPrep) else user
            fields["source"].queryset = DataSource.objects.filter(owner=owner)
        return fields

    def to_representation(self, instance):
        data = super().to_representation(instance)
        request = self.context.get("request")
        user = getattr(request, "user", None)
        if user is not None and user.is_authenticated and isinstance(instance, DataPrep):
            from .shared_data import prep_role, shared_in

            data["owner"] = instance.owner.get_username()
            data["role"] = prep_role(user, instance)
            data["shared_in"] = shared_in(user, instance)
        return data

    def validate_steps(self, value):
        from . import data_service

        try:
            return data_service.clean_steps(value)
        except data_service.DataInputError as exc:
            raise serializers.ValidationError(str(exc)) from None

    def validate_settings(self, value):
        from . import data_service

        try:
            return data_service.clean_settings(value)
        except data_service.DataInputError as exc:
            raise serializers.ValidationError(str(exc)) from None

    def validate(self, attrs):
        if self.instance is None:
            attrs.setdefault("steps", [])
            if "settings" not in attrs:
                attrs["settings"] = self.validate_settings({})
        if "name" in attrs or self.instance is None:
            name = (attrs.get("name") or "").strip()
            if not name:
                source = attrs.get("source") or getattr(self.instance, "source", None)
                name = source.name if source is not None else "Data preparation"
            attrs["name"] = name[:120]
        return attrs
