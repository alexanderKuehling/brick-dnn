"""Training runs and data that members of a shared project use together.

- A training run started from a project (``project`` when starting it) belongs to that
  project: all its members see it; its starter, the project's editors and owner can stop
  it; its starter and the project's owner can delete it.
- Owners of a dataset or data preparation can add it to a project where they are owner or
  editor. Members of the project see it and can train on the dataset; editors of the
  project can also change the preparation (but not its source) and export it. Deleting,
  renaming a dataset and sharing it further stay with its owner.
"""

from __future__ import annotations

from typing import Any

from django.db.models import Q, QuerySet

from .models import DataPrep, Dataset, Project, TrainingRun


def _projects(user) -> QuerySet[Project]:
    from .projects import projects_for

    return projects_for(user)


def editable_projects(user) -> QuerySet[Project]:
    from .models import ProjectMember

    editor_of = ProjectMember.objects.filter(user=user, role="editor").values("project")
    return Project.objects.filter(Q(owner=user) | Q(pk__in=editor_of))


# --------------------------------------------------------------------------- #
# What a user can see
# --------------------------------------------------------------------------- #


def visible_runs(user) -> QuerySet[TrainingRun]:
    project_ids = _projects(user).values("pk")
    return TrainingRun.objects.filter(Q(owner=user) | Q(project__in=project_ids)).distinct()


def visible_datasets(user) -> QuerySet[Dataset]:
    project_ids = _projects(user).values("pk")
    return Dataset.objects.filter(Q(owner=user) | Q(projects__in=project_ids)).distinct()


def visible_preps(user) -> QuerySet[DataPrep]:
    project_ids = _projects(user).values("pk")
    return DataPrep.objects.filter(Q(owner=user) | Q(projects__in=project_ids)).distinct()


# --------------------------------------------------------------------------- #
# What a user may do
# --------------------------------------------------------------------------- #


def can_stop_run(user, run: TrainingRun) -> bool:
    if run.owner_id == user.pk:
        return True
    return run.project_id is not None and editable_projects(user).filter(pk=run.project_id).exists()


def can_delete_run(user, run: TrainingRun) -> bool:
    if run.owner_id == user.pk:
        return True
    return run.project is not None and run.project.owner_id == user.pk


def can_edit_prep(user, prep: DataPrep) -> bool:
    if prep.owner_id == user.pk:
        return True
    return prep.projects.filter(pk__in=editable_projects(user).values("pk")).exists()


def prep_role(user, prep: DataPrep) -> str:
    if prep.owner_id == user.pk:
        return "owner"
    return "editor" if can_edit_prep(user, prep) else "viewer"


def shared_in(user, item: Dataset | DataPrep) -> list[dict[str, Any]]:
    """The projects an item is shared in, as far as ``user`` can see them."""
    visible = _projects(user).values("pk")
    return [
        {"id": p.pk, "name": p.name} for p in item.projects.filter(pk__in=visible).order_by("name")
    ]


# --------------------------------------------------------------------------- #
# The data of a project
# --------------------------------------------------------------------------- #


def item_ref(item: Dataset | DataPrep) -> dict[str, Any]:
    data = {
        "id": item.pk,
        "name": item.name,
        "owner": item.owner.get_username(),
        "created_at": item.created_at,
    }
    if isinstance(item, Dataset):
        data["kind"] = item.kind
        data["size_bytes"] = item.size_bytes
    return data


def project_data_items(project: Project) -> dict[str, Any]:
    return {
        "datasets": [
            item_ref(d) for d in project.datasets.select_related("owner").order_by("name")
        ],
        "preps": [item_ref(p) for p in project.data_preps.select_related("owner").order_by("name")],
    }


KINDS = {"dataset": Dataset, "prep": DataPrep}


def find_item(kind: Any, item_id: Any) -> Dataset | DataPrep | None:
    model = KINDS.get(kind)
    if model is None or not isinstance(item_id, int) or isinstance(item_id, bool):
        return None
    return model.objects.select_related("owner").filter(pk=item_id).first()
