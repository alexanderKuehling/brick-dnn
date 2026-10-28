from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from rest_framework import serializers

from brickdnn.graph import MAX_EDGES, MAX_NODES

from .models import Dataset, Project, TrainingRun

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


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, style={"input_type": "password"})
    email = serializers.EmailField(required=False, allow_blank=True)

    class Meta:
        model = User
        fields = ["username", "email", "password"]

    def validate(self, attrs):
        candidate = User(username=attrs.get("username"), email=attrs.get("email", ""))
        validate_password(attrs["password"], user=candidate)
        return attrs

    def create(self, validated_data):
        return User.objects.create_user(**validated_data)


# --------------------------------------------------------------------------- #
# Local training
# --------------------------------------------------------------------------- #


class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = ["id", "name", "kind", "size_bytes", "created_at", "info"]
        read_only_fields = fields


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
        ]
        read_only_fields = fields

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
