"""Ensembles: one model from several finished runs (see brickdnn.training.ensemble).

An ensemble is a :class:`TrainingRun` of kind "ensemble" with one :class:`EnsembleMember`
per run it combines. It shows up wherever runs do (lists, the run view, usage, the export of
an account) and is deleted like a run. Its "training" runs like any other (the queue, the
runner, budgets): it lets every member predict the validation and test rows of their dataset,
combines the predictions and measures them (ensemble_engine.py). That is cheap, but its CPU
time counts like a training's.

The members are finished runs the user can see (their own, or shared in a project), of one
task (the same classes), trained on the same dataset made in the Data tab: its stored split
makes their validation and test rows line up.

A member cannot be deleted while an ensemble the deleting user can see uses it
(:func:`deletion_refusal`). If it goes another way (with its owner's account, or deleted by
its owner when the ensemble belongs to someone else), the ensemble keeps its results but
cannot predict any more (:func:`broken_reason`).

Predictions prepare raw rows once (the members share their data preparation), let every
member predict them in its own prediction process, combine the probabilities or values and
apply the ensemble's own decision threshold.
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from django.core.exceptions import ValidationError
from django.db import transaction

from brickdnn.training import TERMINAL_STATUSES, DatasetInfo, inference
from brickdnn.training import ensemble as combining
from brickdnn.training.state import ENSEMBLE_FILE, finite, write_state

from . import budgets
from . import training_service as training
from .budgets import Actor
from .models import EnsembleMember, TrainingRun, empty_graph

FINISHED = ("completed", "stopped")
NOT_TABULAR = (
    "'{name}' was not trained on a table. Ensembles combine runs trained on the same dataset "
    "made in the Data tab."
)
NO_SPLIT = (
    "The dataset '{dataset}' has no stored split. Ensembles need a dataset made in the Data "
    "tab: its split makes the members' validation and test rows the same."
)


class EnsembleInputError(training.TrainingInputError):
    """The members or settings of an ensemble are not valid. The message is for the user."""


# --------------------------------------------------------------------------- #
# Members
# --------------------------------------------------------------------------- #


def _ids(value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise EnsembleInputError("'members' must be a list of run ids.")
    if len(value) < combining.MIN_MEMBERS:
        raise EnsembleInputError(
            f"An ensemble combines at least {combining.MIN_MEMBERS} finished runs."
        )
    if len(value) > combining.MAX_MEMBERS:
        raise EnsembleInputError(f"An ensemble combines at most {combining.MAX_MEMBERS} runs.")
    if len(set(value)) != len(value):
        raise EnsembleInputError("Each run can be a member only once.")
    return value


def _quoted(names: list[str]) -> str:
    return ", ".join(f"'{name}'" for name in names)


def _preprocessing(run: TrainingRun) -> dict[str, Any] | None:
    found = (training.read_run_state(run) or {}).get("preprocessing")
    return found if isinstance(found, dict) else None


def class_targets(run: TrainingRun) -> str:
    """How the target column of the member's dataset names the classes (see
    ensemble_engine._read_table): "labels" (text), "indices" or "values"."""
    plan = run.plan if isinstance(run.plan, dict) else {}
    classical = plan.get("classical")
    if isinstance(classical, dict) and classical.get("class_targets"):
        return str(classical["class_targets"])
    info = run.upload.info if run.upload is not None else {}
    return "labels" if info.get("target_kind") == "text" else "indices"


def check_members(user, member_ids: Any) -> list[TrainingRun]:
    """The runs ``member_ids`` in that order, after checking that they can be combined.
    Raises :class:`EnsembleInputError` or :class:`training.TrainingUnavailable`."""
    from .shared_data import visible_runs

    ids = _ids(member_ids)
    runs: list[TrainingRun] = []
    for run_id in ids:
        try:
            run = visible_runs(user).select_related("upload").filter(pk=run_id).first()
        except (ValidationError, ValueError):  # not a run id
            run = None
        if run is None:
            raise EnsembleInputError(
                f"No such training run: '{run_id[:40]}'. Use runs of your own or of your "
                "shared networks."
            )
        runs.append(training.sync_run(run))
    for run in runs:
        if run.kind == "ensemble":
            raise EnsembleInputError(
                f"'{run.name}' is an ensemble itself: combine the runs it is made of instead."
            )
        task = run.plan.get("task") if isinstance(run.plan, dict) else None
        if task not in ("classification", "regression"):
            raise EnsembleInputError(
                f"'{run.name}' is a {task or 'demo'} run. Ensembles combine classifications "
                "or regressions."
            )
        if run.status not in FINISHED or training.weights_path(run) is None:
            raise EnsembleInputError(
                f"'{run.name}' has not finished: {training.no_weights_reason(run)}"
            )
        targets = run.plan.get("targets") or []
        inputs = run.plan.get("inputs") or []
        if len(targets) != 1 or len(inputs) != 1:
            raise EnsembleInputError(
                f"'{run.name}' has more than one input or output; ensembles combine runs "
                "with one of each."
            )
        upload = run.upload
        if (run.dataset or {}).get("kind") != "upload":
            raise EnsembleInputError(NOT_TABULAR.format(name=run.name))
        if upload is None:
            raise EnsembleInputError(
                f"The dataset of '{run.name}' was deleted. An ensemble measures its members on "
                "the validation and test rows of their dataset."
            )
        if upload.kind != "csv":
            raise EnsembleInputError(NOT_TABULAR.format(name=run.name))
        preprocessing = _preprocessing(run)
        if preprocessing is None or preprocessing.get("kind") != "tabular":
            raise EnsembleInputError(
                f"How the inputs of '{run.name}' were prepared is not known (the training did "
                "not get far enough to record it), so it cannot be a member."
            )
        reason = training.unavailable_reason(training.run_family(run))
        if reason:
            raise training.TrainingUnavailable(reason)
    first = runs[0]
    tasks = {run.plan["task"] for run in runs}
    if len(tasks) > 1:
        other = next(run for run in runs if run.plan["task"] != first.plan["task"])
        raise EnsembleInputError(
            f"The runs solve different tasks: '{first.name}' is a {first.plan['task']}, "
            f"'{other.name}' a {other.plan['task']}."
        )
    datasets = {run.upload_id for run in runs}
    if len(datasets) > 1:
        other = next(run for run in runs if run.upload_id != first.upload_id)
        raise EnsembleInputError(
            f"The runs were trained on different datasets: '{first.name}' on "
            f"'{first.upload.name}', '{other.name}' on '{other.upload.name}'. An ensemble needs "
            "runs trained on the same dataset, so that their validation and test rows line up."
        )
    info = DatasetInfo.from_dict(first.upload.info)
    if info.predefined_split is None:
        raise EnsembleInputError(NO_SPLIT.format(dataset=first.upload.name))
    if not training.dataset_path(first.upload).exists():
        raise EnsembleInputError(
            f"The files of the dataset '{first.upload.name}' are missing. Export it again."
        )
    if first.plan["task"] == "classification":
        names = first.plan.get("class_names")
        for run in runs[1:]:
            if run.plan.get("class_names") != names:
                raise EnsembleInputError(
                    "The runs predict different classes: "
                    f"'{first.name}' has {_classes(names)}, '{run.name}' has "
                    f"{_classes(run.plan.get('class_names'))}. Give every network one output "
                    "per class of the dataset."
                )
    columns = _preprocessing(first).get("columns")
    for run in runs[1:]:
        if _preprocessing(run).get("columns") != columns:
            raise EnsembleInputError(
                f"'{run.name}' takes other columns than '{first.name}'. Train both on the "
                "same dataset."
            )
    return runs


def _classes(names: Any) -> str:
    names = [str(n) for n in names or []]
    shown = ", ".join(names[:8]) + (" …" if len(names) > 8 else "")
    return f"{len(names)} classes ({shown})"


def _plan(runs: list[TrainingRun], method: str) -> dict[str, Any]:
    """The plan of the ensemble: task, inputs, targets and classes of its members (like a
    trained run's, see compat.TrainingPlan, without a model)."""
    first = runs[0].plan
    label = combining.METHOD_LABELS[method].lower()
    return {
        "task": first["task"],
        "summary": f"Ensemble of {len(runs)} runs ({label}): {_quoted([r.name for r in runs])}.",
        "inputs": first["inputs"],
        "targets": first["targets"],
        "image": None,
        "num_classes": first.get("num_classes"),
        "class_names": first.get("class_names"),
        "notes": [
            "The members predict the validation and test rows of their dataset; the "
            "ensemble combines what they predict."
        ],
        "classical": None,
        "segmentation": None,
    }


# --------------------------------------------------------------------------- #
# Creating
# --------------------------------------------------------------------------- #


def create_ensemble(
    user,
    name: Any,
    member_ids: Any,
    method: Any = None,
    weights: Any = None,
    project_id: Any = None,
    actor: Actor | None = None,
) -> TrainingRun:
    """Check the members and start the ensemble like a training (see
    training_service.start_run for the queue, the budgets and what it raises).
    ``weights``: one number per member for the weighted average (default: equal)."""
    actor = actor or Actor(user)
    reason = training.unavailable_reason("ensemble")
    if reason:
        raise training.TrainingUnavailable(reason)
    name = training.clean_name(name, "Ensemble")
    try:
        method = combining.check_method(method)
    except combining.EnsembleError as exc:
        raise EnsembleInputError(str(exc)) from None
    project = training.run_project(user, project_id)
    runs = check_members(user, member_ids)
    try:
        if method == "stacking" and weights is not None:
            raise combining.EnsembleError(
                "A stacking ensemble learns how to combine its members: leave out the weights."
            )
        shares = combining.normalized_weights(weights, len(runs))
    except combining.EnsembleError as exc:
        raise EnsembleInputError(str(exc)) from None
    upload = runs[0].upload
    info = DatasetInfo.from_dict(upload.info)
    if method == "stacking":
        counts = [*(info.predefined_split or []), 0, 0]
        if not int(counts[1]):
            raise EnsembleInputError(combining.STACKING_NEEDS_VALIDATION)
        reason = training.unavailable_reason("classical")  # the model is fitted with sklearn
        if reason:
            raise training.TrainingUnavailable(reason)
    budgets.check_storage(user)
    first = runs[0]

    def create(limit: budgets.RunLimit, **fields: Any) -> TrainingRun:
        with transaction.atomic():
            run = TrainingRun.objects.create(
                owner=user,
                kind="ensemble",
                project=project,
                name=name,
                graph=empty_graph(),
                dataset=dict(first.dataset),
                upload=upload,
                config={"method": method},
                plan=_plan(runs, method),
                api_token=actor.token,
                connection=actor.connection,
                cpu_limit_seconds=limit.seconds,
                **fields,
            )
            EnsembleMember.objects.bulk_create(
                EnsembleMember(
                    ensemble=run, run=member, position=index, weight=share, name=member.name
                )
                for index, (member, share) in enumerate(zip(runs, shares, strict=True))
            )
        return run

    return training.start_new(user, actor, None, create, info, training.dataset_path(upload))


def ensemble_job(
    run: TrainingRun, info: DatasetInfo, path: Path | None, limit: budgets.RunLimit
) -> dict[str, Any]:
    """job.json of an ensemble (see brickdnn.training.ensemble_engine)."""
    rows = members_of(run)
    plan = run.plan if isinstance(run.plan, dict) else {}
    first = next((row.run for row in rows if row.run is not None), None)
    method = (run.config or {}).get("method", "average")
    return {
        "name": run.name,
        "graph": run.graph,
        "dataset": {
            "kind": info.kind,
            "path": str(path) if path is not None else None,
            "info": info.to_dict(),
        },
        "config": run.config,
        "tuning": None,
        "limits": limit.job(),
        "ensemble": {
            "method": method,
            "weights": [row.weight for row in rows] if method == "average" else None,
            "task": plan.get("task"),
            "num_classes": plan.get("num_classes"),
            "class_names": plan.get("class_names"),
            "class_targets": class_targets(first) if first is not None else "indices",
            "members": [
                {
                    "id": str(row.run_id) if row.run_id else None,
                    "name": row.name,
                    # A member deleted meanwhile has no files: the ensemble fails clearly.
                    "dir": str(training.run_dir(row.run_id)) if row.run_id else "",
                    "family": training.run_family(row.run) if row.run else "neural",
                }
                for row in rows
            ],
        },
    }


# --------------------------------------------------------------------------- #
# What an ensemble is made of
# --------------------------------------------------------------------------- #


def members_of(run: TrainingRun) -> list[EnsembleMember]:
    return list(run.ensemble_members.select_related("run").order_by("position"))


def method_of(run: TrainingRun) -> str:
    method = (run.config or {}).get("method") if isinstance(run.config, dict) else None
    return method if method in combining.METHODS else "average"


def broken_reason(run: TrainingRun, rows: list[EnsembleMember] | None = None) -> str | None:
    """Why the ensemble cannot predict any more (a member is gone), or None."""
    rows = members_of(run) if rows is None else rows
    gone = [row.name for row in rows if row.run is None]
    if not gone:
        return None
    word = "member" if len(gone) == 1 else "members"
    return (
        f"The {word} {_quoted(gone)} of this ensemble {'was' if len(gone) == 1 else 'were'} "
        "deleted, so it cannot predict any more. Its results are kept."
    )


def describe(run: TrainingRun, user=None) -> dict[str, Any] | None:
    """What the API shows of an ensemble (None for other runs)::

        {"method": "average", "members": [{"id", "name", "weight", "family", "status",
         "task", "visible", "deleted"}], "broken": null}

    ``visible``: the user can open the member's run (a member deleted meanwhile has no id)."""
    if run.kind != "ensemble":
        return None
    from .shared_data import visible_runs

    rows = members_of(run)
    ids = [row.run_id for row in rows if row.run_id is not None]
    visible: set[Any] = set()
    if user is not None and getattr(user, "is_authenticated", False) and ids:
        visible = set(visible_runs(user).filter(pk__in=ids).values_list("pk", flat=True))
    method = method_of(run)
    members = []
    for row in rows:
        member = row.run
        members.append(
            {
                "id": str(row.run_id) if row.run_id else None,
                "name": member.name if member is not None else row.name,
                "weight": row.weight if method == "average" else None,
                "family": training.run_family(member) if member is not None else None,
                "model": _model_label(member) if member is not None else None,
                "status": member.status if member is not None else None,
                "visible": row.run_id in visible,
                "deleted": member is None,
            }
        )
    return {
        "method": method,
        "method_label": combining.METHOD_LABELS[method],
        "members": members,
        "broken": broken_reason(run, rows),
    }


def _model_label(run: TrainingRun) -> str | None:
    """'HistGradientBoostingClassifier' or 'Neural network' (for the member list)."""
    plan = run.plan if isinstance(run.plan, dict) else {}
    classical = plan.get("classical")
    if isinstance(classical, dict):
        return str(classical.get("label") or classical.get("type") or "Classical model")
    return "Neural network"


def deletion_refusal(user, run: TrainingRun) -> str | None:
    """Why ``user`` may not delete ``run``: an ensemble they can see predicts with it. (An
    ensemble of somebody else's that they cannot see does not keep them from deleting their
    run; that ensemble then cannot predict any more.)"""
    if run.kind == "ensemble":
        return None
    from .shared_data import visible_runs

    names = list(
        visible_runs(user)
        .filter(kind="ensemble", ensemble_members__run=run)
        .values_list("name", flat=True)
        .distinct()
    )
    if not names:
        return None
    word = "ensemble" if len(names) == 1 else "ensembles"
    return (
        f"'{run.name}' is a member of the {word} {_quoted(names)}, which "
        f"{'predicts' if len(names) == 1 else 'predict'} with it. Delete the {word} first."
    )


# --------------------------------------------------------------------------- #
# Changing the weights
# --------------------------------------------------------------------------- #


def set_weights(run: TrainingRun, weights: Any) -> TrainingRun:
    """New weights for a weighted average: measured again on the members' saved
    predictions (no member predicts again). The CPU time counts as the ensemble's training.
    Raises :class:`EnsembleInputError`."""
    if run.kind != "ensemble":
        raise EnsembleInputError("Only ensembles have weights.")
    if method_of(run) != "average":
        raise EnsembleInputError(
            "A stacking ensemble learns how to combine its members; it has no weights to set."
        )
    path = training.run_dir(run.pk) / ENSEMBLE_FILE
    state = training.read_run_state(run)
    if run.status != "completed" or not path.is_file() or not state:
        raise EnsembleInputError(
            "The ensemble has not finished yet: its weights can be changed once it has."
        )
    rows = members_of(run)
    try:
        shares = combining.normalized_weights(weights, len(rows))
    except combining.EnsembleError as exc:
        raise EnsembleInputError(str(exc)) from None
    started = time.thread_time()
    plan = run.plan if isinstance(run.plan, dict) else {}
    outputs, targets = combining.load_outputs(path, len(rows))
    task = plan.get("task")
    result = combining.ensemble_result(
        task,
        "average",
        outputs,
        targets,
        weights=shares,
        num_classes=plan.get("num_classes"),
        class_names=plan.get("class_names"),
        member_names=[row.name for row in rows],
    )
    # The seconds of the combining the training did stay what they were.
    result["history"][0]["seconds"] = ((state.get("history") or [{}])[0] or {}).get("seconds")
    state.update(
        evaluation=result["evaluation"],
        ensemble=result["ensemble"],
        history=result["history"],
        best=result["best"],
        test=result["test"],
        message=combining.summary(task, result),
    )
    write_state(training.run_dir(run.pk), finite(state))
    with transaction.atomic():
        for row, share in zip(rows, shares, strict=True):
            EnsembleMember.objects.filter(pk=row.pk).update(weight=share)
        training.apply_state(run, state)
        run.save(update_fields=["metrics", "best", "message"])
    spent = time.thread_time() - started
    budgets.record_run(run, (run.cpu_seconds or 0.0) + spent, final=True)
    return run


# --------------------------------------------------------------------------- #
# Predictions
# --------------------------------------------------------------------------- #


@dataclass
class EnsembleInputs:
    """The prepared inputs of every member (training.Prediction each)."""

    parts: list[Any]
    members: bool  # also return what every member predicts
    top_k: int


def prediction_reason(run: TrainingRun) -> str | None:
    """Why the ensemble cannot predict now (cheap), or None."""
    if not run.has_weights or run.status not in TERMINAL_STATUSES:
        return _not_done(run)
    rows = members_of(run)
    reason = broken_reason(run, rows)
    if reason:
        return reason
    for row in rows:
        member = row.run
        reason = training.unavailable_reason(training.run_family(member))
        if reason:
            return reason
        if training.weights_path(member) is None:
            return (
                f"The member '{member.name}' cannot predict: {training.no_weights_reason(member)}"
            )
    return None


def _not_done(run: TrainingRun) -> str:
    if run.status == "queued":
        return "This ensemble still waits in the queue."
    if run.status in ("starting", "preparing", "running"):
        return "This ensemble is still measuring its members; it predicts once it has finished."
    if run.status == "failed":
        return f"This ensemble failed: {run.message}"
    return "This ensemble was stopped before it had measured its members."


def _raw_source(run: TrainingRun, members: list[TrainingRun]) -> TrainingRun:
    """The run whose data preparation prepares raw rows: the ensemble's (that of its
    dataset), else the first member's own copy of it."""
    from .raw_predictions import recipe_of

    for candidate in (run, *members):
        with contextlib.suppress(Exception):
            if recipe_of(candidate) is not None:
                return candidate
    return run


def prepare_prediction(
    run: TrainingRun,
    inputs: Any,
    input_scale: Any = None,
    top_k: Any = None,
    image: Any = None,
    input_format: Any = None,
    members: Any = False,
) -> training.Prediction:
    """Prepare ``inputs`` for every member (raw rows are prepared once). Raises like
    training.prepare_prediction."""
    from . import raw_predictions

    reason = training.unavailable_reason("ensemble")
    if reason:
        raise training.TrainingUnavailable(reason)
    if image is not None:
        raise training.TrainingInputError("Ensembles predict rows of a table; they take no images.")
    if members is not None and not isinstance(members, bool):
        raise training.TrainingInputError("'members' must be true or false.")
    for row in members_of(run):
        if row.run is not None:
            reason = training.unavailable_reason(training.run_family(row.run))
            if reason:
                raise training.TrainingUnavailable(reason)
    reason = prediction_reason(run)
    if reason:
        raise training.PredictionUnavailable(reason)
    if input_scale == "normalized":
        raise training.TrainingInputError(
            "Ensembles take the values as in the CSV file (input_scale 'raw'): their members "
            "standardise them each in their own way."
        )
    runs = [row.run for row in members_of(run)]
    warnings: list[str] = []
    started = time.thread_time()
    raw = raw_predictions.check_format(input_format) == "raw"
    if raw:
        inputs, warnings = raw_predictions.prepare_raw(_raw_source(run, runs), inputs)
        input_scale = None  # the prepared rows are in the units of the training CSV file
    parts = []
    for member in runs:
        preprocessing = (training.state_of(member) or {}).get("preprocessing")
        try:
            prepared = inference.prepare_inputs(
                member.plan, preprocessing, inputs, input_scale, top_k
            )
        except inference.UnknownPreprocessing as exc:
            raise training.PredictionUnavailable(f"'{member.name}': {exc}") from None
        except inference.InvalidInputs as exc:
            message = str(exc)
            if "unknown column" in message and not raw and raw_predictions.recipe_of(run):
                message += (
                    " Rows in the columns of the source table (before the data preparation) "
                    "need input_format 'raw'."
                )
            raise training.TrainingInputError(message) from None
        parts.append(training.Prediction(member, preprocessing, prepared))
    spent = time.thread_time() - started if raw else 0.0
    own = (training.state_of(run) or {}).get("preprocessing") or parts[0].preprocessing
    return training.Prediction(
        run,
        own,
        parts[0].inputs,
        warnings=warnings,
        prepare_seconds=spent,
        ensemble=EnsembleInputs(parts, bool(members), parts[0].inputs.top_k),
    )


def _member_array(task: str, part: Any, result: dict[str, Any]):
    """A member's answer as probabilities (rows × classes) or values in target units."""
    import numpy as np

    outputs = result.get("outputs")
    count = part.inputs.count
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise training.PredictionFailed(f"'{part.run.name}' returned an unexpected answer.")
    values = np.array(
        [[math.nan if v is None else v for v in _flat(row)] for row in outputs[0]],
        dtype=np.float64,
    )
    if len(values) != count:
        raise training.PredictionFailed(f"'{part.run.name}' returned an unexpected answer.")
    if task == "classification":
        if result.get("probabilities"):
            sums = values.sum(axis=1, keepdims=True)
            return values / np.where(sums > 0, sums, 1.0)
        return combining.softmax(values)
    values = values.reshape(count)
    scale = part.preprocessing.get("target_std")
    shift = part.preprocessing.get("target_mean")
    if scale is not None and shift is not None:
        values = values * float(scale) + float(shift)
    return values


def _flat(value: Any) -> list[Any]:
    if isinstance(value, list):
        return [x for item in value for x in _flat(item)]
    return [value]


def run_prediction(prediction: training.Prediction, actor: Actor) -> dict[str, Any]:
    """Let every member predict (one process each, one after the other), combine and
    answer like training.run_prediction, with ``ensemble`` (method and members) and with
    ``members`` (what each member predicted) when asked for."""
    import numpy as np

    run = prediction.run
    spec: EnsembleInputs = prediction.ensemble
    budgets.check_compute(actor)
    if prediction.prepare_seconds > 0:  # preparing raw rows (in this process)
        budgets.record_prediction(actor, run, prediction.prepare_seconds)
    answers = []
    with training.prediction_slot():
        for part in spec.parts:
            request = {"inputs": part.inputs.tensors, "kind": part.inputs.kind}
            answers.append(training.predict_process(part.run, request, actor, usage_run=run))
    plan = run.plan if isinstance(run.plan, dict) else {}
    task = plan.get("task")
    arrays = [
        _member_array(task, part, answer) for part, answer in zip(spec.parts, answers, strict=True)
    ]
    rows = members_of(run)
    method = method_of(run)
    meta = ((training.state_of(run) or {}).get("ensemble") or {}).get("meta")
    weights = [row.weight for row in rows]
    try:
        combined = combining.combine(task, method, arrays, weights, meta)
    except (combining.EnsembleError, KeyError, ValueError) as exc:
        raise training.PredictionFailed(
            f"The members' predictions cannot be combined: {exc}"
        ) from None
    threshold = training.decision_threshold(run)
    names = inference.class_names(plan, prediction.preprocessing)
    predictions = []
    for values in combined:
        if task == "classification":
            row = [float(v) if math.isfinite(v) else None for v in np.ravel(values)]
            predictions.append(
                inference.classes_from_probabilities(row, names or [], spec.top_k, threshold)
            )
        else:
            value = float(values)
            predictions.append({"value": value if math.isfinite(value) else None})
    answer: dict[str, Any] = {
        "run": {"id": str(run.pk), "name": run.name},
        "task": task,
        "predictions": predictions,
        "ensemble": {
            "method": method,
            "members": [
                {
                    "id": str(row.run_id),
                    "name": row.run.name,
                    "weight": row.weight if method == "average" else None,
                }
                for row in rows
            ],
        },
    }
    if threshold is not None and names:
        positive = names[inference.BINARY_POSITIVE]
        answer["decision_threshold"] = threshold
        answer["positive_class"] = positive
        answer["threshold_note"] = (
            f"The prediction is '{positive}' when its combined probability "
            f"(positive_probability) is at least the ensemble's decision threshold "
            f"{threshold:g}, else '{names[1 - inference.BINARY_POSITIVE]}'."
        )
    if task != "classification":
        answer["target"] = prediction.preprocessing.get("target")
    if spec.members:
        answer["members"] = []
        for part, result in zip(spec.parts, answers, strict=True):
            member = part.run
            try:
                found = inference.read_outputs(
                    member.plan,
                    part.preprocessing,
                    result.get("outputs"),
                    part.inputs.count,
                    spec.top_k,
                    probabilities=bool(result.get("probabilities")),
                    std=result.get("std"),
                    threshold=training.decision_threshold(member),
                )
            except ValueError as exc:
                raise training.PredictionFailed(str(exc)) from None
            answer["members"].append(
                {"id": str(member.pk), "name": member.name, "predictions": found}
            )
    if prediction.warnings:
        answer["warnings"] = prediction.warnings
    return answer
