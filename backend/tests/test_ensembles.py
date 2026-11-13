"""Ensembles of finished runs (brickdnn.training.ensemble, ensemble_engine.py and
brick_dnn_core/ensembles.py).

The math is checked against NumPy and scikit-learn without a database. The members' checks,
deletion, permissions, the API and the MCP tools use members written by the tests: small
scikit-learn pipelines fitted here and saved like a classical run, which the real runner and
prediction processes use without PyTorch. The end-to-end test trains a tiny neural network
and a gradient boosting model on a dataset exported from the Data tab and predicts raw rows.
"""

# ruff: noqa: F811 (fixtures are imported from other test modules)

from __future__ import annotations

import json
import re
import time

import numpy as np
import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

pytest.importorskip("sklearn")

import joblib  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from brick_dnn_core import accounts, retention, training_service  # noqa: E402
from brick_dnn_core.models import (  # noqa: E402
    AgentAction,
    Dataset,
    EnsembleMember,
    Project,
    ProjectMember,
    TrainingRun,
    UsageRecord,
)
from brickdnn.training import ensemble, torch_installed  # noqa: E402
from brickdnn.training.datasets import SPLIT_FILE, inspect_csv  # noqa: E402
from brickdnn.training.evaluation import ClassificationCounter, evaluation  # noqa: E402
from brickdnn.training.state import (  # noqa: E402
    CLASSICAL_WEIGHTS_FILE,
    ENSEMBLE_FILE,
    initial_state,
    write_job,
    write_state,
)

from .helpers import chain  # noqa: E402
from .test_mcp import call, mcp_client  # noqa: E402
from .test_training_api import training_dir, wait_for  # noqa: E402, F401

# --------------------------------------------------------------------------- #
# The math (no database)
# --------------------------------------------------------------------------- #


def binary_outputs(n: int = 300, seed: int = 0):
    """Two members' probabilities of two classes, both better than chance."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    first = np.clip(0.5 * y + 0.25 + 0.25 * rng.normal(size=n), 0.01, 0.99)
    second = np.clip(0.4 * y + 0.3 + 0.3 * rng.normal(size=n), 0.01, 0.99)
    return [np.column_stack([1 - p, p]) for p in (first, second)], y


def multiclass_outputs(n: int = 300, k: int = 3, seed: int = 1):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    outputs = []
    for strength in (2.0, 1.2):
        scores = rng.normal(size=(n, k))
        scores[np.arange(n), y] += strength
        outputs.append(ensemble.softmax(scores))
    return outputs, y


def test_weighted_average_matches_numpy():
    outputs, _ = binary_outputs()
    got = ensemble.combine("classification", "average", outputs, [0.25, 0.75])
    expected = np.average(np.stack(outputs), axis=0, weights=[0.25, 0.75])
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(got.sum(axis=1), 1.0)
    values = [np.array([1.0, 2.0, 3.0]), np.array([3.0, 6.0, 9.0])]
    got = ensemble.combine("regression", "average", values, [0.5, 0.5])
    np.testing.assert_allclose(got, np.mean(values, axis=0))


def test_weights_are_checked_and_scaled():
    assert ensemble.normalized_weights(None, 4) == [0.25] * 4
    assert ensemble.normalized_weights([2, 1, "1"], 3) == [0.5, 0.25, 0.25]
    assert ensemble.normalized_weights([0, 3], 2) == [0.0, 1.0]
    for weights, message in (
        ([1], "one weight per member (2 numbers)"),
        ([1, -1], "at least 0"),
        ([0, 0], "At least one weight must be above 0"),
        ([True, 1], "at least 0"),
        ([float("nan"), 1], "at least 0"),
        ("1,1", "one weight per member"),
    ):
        with pytest.raises(ensemble.EnsembleError, match=re.escape(message)):
            ensemble.normalized_weights(weights, 2)
    with pytest.raises(ensemble.EnsembleError, match="average, stacking"):
        ensemble.check_method("vote")


def test_binary_stacking_matches_scikit_learn():
    outputs, y = binary_outputs()
    meta = ensemble.fit_stacking("classification", outputs, y, 2)
    # One feature per member: the probability of the second class.
    x = np.column_stack([o[:, 1] for o in outputs])
    model = LogisticRegression(max_iter=1000).fit(x, y)
    new, _ = binary_outputs(50, seed=9)
    got = ensemble.apply_stacking(meta, "classification", new)
    expected = model.predict_proba(np.column_stack([o[:, 1] for o in new]))
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-9)
    assert meta["model"] == "logistic_regression" and meta["classes"] == [0, 1]
    assert len(meta["coef"][0]) == 2 and meta["rows"] == len(y)
    json.dumps(meta)  # plain numbers, kept in state.json


def test_multiclass_stacking_matches_scikit_learn():
    outputs, y = multiclass_outputs()
    meta = ensemble.fit_stacking("classification", outputs, y, 3)
    x = np.hstack(outputs)  # all probabilities of every member
    model = LogisticRegression(max_iter=1000).fit(x, y)
    got = ensemble.apply_stacking(meta, "classification", outputs)
    np.testing.assert_allclose(got, model.predict_proba(x), rtol=0, atol=1e-9)

    # A class the validation rows do not hold gets the probability 0.
    rows = y != 1
    meta = ensemble.fit_stacking("classification", [o[rows] for o in outputs], y[rows], 3)
    assert meta["classes"] == [0, 2]
    got = ensemble.apply_stacking(meta, "classification", outputs)
    model = LogisticRegression(max_iter=1000).fit(x[rows], y[rows])
    np.testing.assert_allclose(got[:, [0, 2]], model.predict_proba(x), rtol=0, atol=1e-9)
    assert (got[:, 1] == 0).all()


def test_ridge_stacking_matches_scikit_learn():
    rng = np.random.default_rng(4)
    y = rng.normal(size=120) * 10 + 50
    outputs = [y + rng.normal(size=120), 0.8 * y + 12 + 3 * rng.normal(size=120)]
    meta = ensemble.fit_stacking("regression", outputs, y)
    model = Ridge().fit(np.column_stack(outputs), y)
    got = ensemble.combine("regression", "stacking", outputs, meta=meta)
    np.testing.assert_allclose(got, model.predict(np.column_stack(outputs)), rtol=1e-12)


def test_stacking_refuses_one_class_and_missing_predictions():
    outputs, y = binary_outputs(40)
    with pytest.raises(ensemble.EnsembleError, match="only one class"):
        ensemble.fit_stacking("classification", outputs, np.zeros(40, dtype=int), 2)
    broken = [outputs[0].copy(), outputs[1]]
    broken[0][3, 1] = np.nan
    with pytest.raises(ensemble.EnsembleError, match="NaN"):
        ensemble.fit_stacking("classification", broken, y, 2)


def splits_of(outputs, y, val, test=None):
    """``outputs`` and ``y`` cut into a validation and a test part."""
    found = {"val": ([o[val] for o in outputs], y[val])}
    if test is not None:
        found["test"] = ([o[test] for o in outputs], y[test])
    return {k: v[0] for k, v in found.items()}, {k: v[1] for k, v in found.items()}


def test_the_evaluation_of_an_average_is_the_one_of_a_trained_run():
    outputs, y = binary_outputs(400)
    val, test = np.arange(0, 200), np.arange(200, 400)
    per_split, targets = splits_of(outputs, y, val, test)
    result = ensemble.ensemble_result(
        "classification",
        "average",
        per_split,
        targets,
        weights=[3, 1],
        num_classes=2,
        class_names=["ok", "fraud"],
        member_names=["MLP", "Trees"],
    )
    combined = {key: 0.75 * per_split[key][0] + 0.25 * per_split[key][1] for key in ("val", "test")}
    counters = {}
    for key in ("val", "test"):
        counter = ClassificationCounter(2)
        counter.add(combined[key], targets[key])
        counters[key] = counter.result()
    expected = evaluation(
        "classification", counters["val"], counters["test"], None, class_names=["ok", "fraud"]
    )
    assert json.dumps(result["evaluation"], sort_keys=True) == json.dumps(expected, sort_keys=True)
    found = result["evaluation"]
    # The threshold analysis of two classes: suggestions from the validation rows, a curve
    # for both splits.
    assert found["binary"]["source"] == "val" and found["binary"]["best_f1"]["threshold"] > 0
    assert found["test"]["threshold_curve"]["roc_auc"] == pytest.approx(
        roc_auc_score(targets["test"], combined["test"][:, 1])
    )
    info = result["ensemble"]
    assert info["method"] == "average" and info["weights"] == [0.75, 0.25]
    assert info["optimistic"] is False and info["meta"] is None
    assert info["rows"] == {"val": 200, "test": 200}
    # The comparison: every member and the ensemble on the same rows.
    rows = {row["name"]: row for row in info["comparison"]}
    assert list(rows) == ["MLP", "Trees", "Ensemble"]
    for index, name in enumerate(["MLP", "Trees"]):
        accuracy = float(np.mean(per_split["test"][index].argmax(axis=1) == targets["test"]))
        assert rows[name]["test"]["accuracy"] == pytest.approx(accuracy)
        assert rows[name]["test"]["roc_auc"] == pytest.approx(
            roc_auc_score(targets["test"], per_split["test"][index][:, 1])
        )
    assert rows["Ensemble"]["val"]["accuracy"] == found["val"]["accuracy"]
    # What the run list shows: the validation metrics as an epoch, the test metrics.
    assert result["history"][0]["val_accuracy"] == found["val"]["accuracy"]
    assert result["history"][0]["val_loss"] == found["val"]["log_loss"]
    assert result["test"]["accuracy"] == found["test"]["accuracy"]
    assert ensemble.summary("classification", result).startswith(
        "Combined 2 runs (weighted average): validation accuracy"
    )


def test_stacking_is_fitted_on_validation_and_labelled_optimistic_without_test_rows():
    outputs, y = binary_outputs(400)
    val, test = np.arange(0, 200), np.arange(200, 400)
    per_split, targets = splits_of(outputs, y, val, test)
    result = ensemble.ensemble_result(
        "classification", "stacking", per_split, targets, num_classes=2
    )
    info = result["ensemble"]
    assert info["fitted_on"] == "val" and info["optimistic"] is False
    assert info["weights"] is None
    assert info["meta"] == ensemble.fit_stacking("classification", per_split["val"], y[val], 2)
    stacked = ensemble.apply_stacking(info["meta"], "classification", per_split["test"])
    accuracy = float(np.mean(stacked.argmax(axis=1) == y[test]))
    assert result["evaluation"]["test"]["accuracy"] == pytest.approx(accuracy)
    assert "optimistic" not in ensemble.summary("classification", result)

    # Without test rows: measured on the rows it was fitted on.
    per_split, targets = splits_of(outputs, y, val)
    result = ensemble.ensemble_result(
        "classification", "stacking", per_split, targets, num_classes=2
    )
    assert result["ensemble"]["optimistic"] is True
    assert result["test"] is None and result["evaluation"]["test"] is None
    assert "(optimistic: measured on the rows the stacking model was fitted on)" in (
        ensemble.summary("classification", result)
    )

    # Without validation rows there is nothing to fit it on.
    with pytest.raises(ensemble.EnsembleError, match="validation rows"):
        ensemble.ensemble_result(
            "classification",
            "stacking",
            {"test": per_split["val"]},
            {"test": targets["val"]},
            num_classes=2,
        )

    # An average without validation and test rows is measured on the training rows.
    result = ensemble.ensemble_result(
        "classification",
        "average",
        {"training": per_split["val"]},
        {"training": targets["val"]},
        num_classes=2,
    )
    assert result["ensemble"]["optimistic"] is True
    assert result["evaluation"]["training"]["accuracy"] is not None
    assert "(on the training rows, optimistic)" in ensemble.summary("classification", result)


def test_regression_results_and_saved_outputs(tmp_path):
    rng = np.random.default_rng(2)
    y = rng.normal(size=100) * 5 + 20
    outputs = [y + rng.normal(size=100), y + 2 * rng.normal(size=100)]
    val, test = np.arange(50), np.arange(50, 100)
    per_split, targets = splits_of(outputs, y, val, test)
    result = ensemble.ensemble_result("regression", "average", per_split, targets)
    combined = (per_split["test"][0] + per_split["test"][1]) / 2
    rmse = float(np.sqrt(np.mean((combined - targets["test"]) ** 2)))
    assert result["evaluation"]["test"]["rmse"] == pytest.approx(rmse)
    assert result["test"]["rmse"] == pytest.approx(rmse)
    assert result["history"][0]["val_rmse"] == result["evaluation"]["val"]["rmse"]
    names = [row["name"] for row in result["ensemble"]["comparison"]]
    assert names == ["Member 1", "Member 2", "Ensemble"]

    path = tmp_path / ENSEMBLE_FILE
    ensemble.save_outputs(path, per_split, targets, {"val": val, "test": test})
    loaded, loaded_targets = ensemble.load_outputs(path, 2)
    assert set(loaded) == {"val", "test"}
    np.testing.assert_array_equal(loaded["test"][1], per_split["test"][1])
    np.testing.assert_array_equal(loaded_targets["val"], targets["val"])


# --------------------------------------------------------------------------- #
# Members written by the tests: datasets with a stored split and fitted pipelines
# --------------------------------------------------------------------------- #

CLASSES = ["ok", "fraud"]
CLASSICAL_GRAPH = {
    "version": 1,
    "nodes": [
        {"id": "in", "type": "Input", "params": {"shape": [3]}},
        {"id": "m", "type": "SVC", "params": {}},
    ],
    "edges": [{"source": "in", "target": "m"}],
}
SPLIT = [1 if i % 5 == 0 else 2 if i % 5 == 1 else 0 for i in range(200)]


def payments(n: int = 200, seed: int = 3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = ((x[:, 0] + 0.8 * x[:, 1] + 0.4 * rng.normal(size=n)) > 0.7).astype(int)
    return x, y


def make_dataset(owner, split=SPLIT, classes=CLASSES, name="Payments", seed=3, stored=True):
    """A dataset as the Data tab exports it: data.csv (the class index last) and split.bin."""
    x, y = payments(len(split), seed)
    dataset = Dataset.objects.create(owner=owner, name=name, kind="csv", info={"kind": "csv"})
    folder = training_service.dataset_dir(dataset.pk)
    folder.mkdir(parents=True)
    lines = ["a,b,c,label"] + [
        f"{r[0]:.5f},{r[1]:.5f},{r[2]:.5f},{t}" for r, t in zip(x, y, strict=True)
    ]
    path = folder / "data.csv"
    path.write_text("\n".join(lines) + "\n")
    info = inspect_csv(path, name=name)
    if stored:
        (folder / SPLIT_FILE).write_bytes(bytes(split))
        info.predefined_split = [split.count(0), split.count(1), split.count(2)]
    info.class_names = list(classes)
    dataset.info = info.to_dict()
    dataset.save()
    return dataset, x, y


def member(
    owner,
    dataset,
    name,
    *,
    c=1.0,
    columns=(0, 1, 2),
    status="completed",
    class_names=CLASSES,
    task="classification",
    project=None,
):
    """A finished classical run on ``dataset``: a logistic regression (``c``: its
    regularisation, ``columns``: the features it looks at, by zeroing the others) fitted on
    the training rows of the stored split and saved like the classical engine does."""
    x, y = payments(len(SPLIT), 3)
    train = np.flatnonzero(np.array(SPLIT) == 0)
    mask = np.zeros(3)
    mask[list(columns)] = 1.0
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("model", LogisticRegression(C=c, max_iter=1000))]
    )
    fitted_x = x * mask
    pipeline.fit(fitted_x[train], y[train])
    # Zeroed columns stay zero: a scaler with mean 0 and scale 1 there.
    scaler = pipeline.named_steps["scaler"]
    scaler.mean_ = scaler.mean_ * mask
    scaler.scale_ = np.where(mask > 0, scaler.scale_, 1.0)
    model = pipeline.named_steps["model"]
    model.coef_ = model.coef_ * mask
    plan = {
        "task": task,
        "summary": "",
        "inputs": [{"shape": [3], "dtype": "float", "vocab": None}],
        "targets": [{"kind": "class", "num_classes": len(class_names)}],
        "num_classes": len(class_names),
        "class_names": list(class_names),
        "notes": [],
        "classical": {
            "type": "SVC",
            "label": "Logistic regression",
            "estimator": "LogisticRegression",
            "class_targets": "indices",
        },
    }
    run = TrainingRun.objects.create(
        owner=owner,
        name=name,
        dataset={"kind": "upload", "id": dataset.pk, "name": dataset.name},
        upload=dataset,
        plan=plan,
        status=status,
        has_weights=status == "completed",
        project=project,
    )
    directory = training_service.run_dir(run.pk)
    directory.mkdir(parents=True)
    # The prediction process tells classical models by their graph.
    write_job(directory, {"name": name, "graph": CLASSICAL_GRAPH})
    joblib.dump(
        {
            "model": pipeline,
            "task": "classification",
            "class_indices": [0, 1],
            "num_classes": 2,
            "class_names": CLASSES,
        },
        directory / CLASSICAL_WEIGHTS_FILE,
    )
    state = initial_state(1, "classical")
    state.update(
        status=status,
        weights=status == "completed",
        task=task,
        preprocessing={
            "kind": "tabular",
            "columns": ["a", "b", "c"],
            "target": "label",
            "delimiter": ",",
            "decimal_comma": False,
            "mean": scaler.mean_.tolist(),
            "std": scaler.scale_.tolist(),
            "target_mean": None,
            "target_std": None,
            "class_names": list(class_names),
        },
    )
    write_state(directory, state)
    return run, pipeline, mask


@pytest.fixture
def real_processes(training_dir, settings):
    """The real runner and prediction processes (the members need scikit-learn only)."""
    settings.TRAINING_RUNNER_COMMAND = None
    settings.TRAINING_PREDICT_COMMAND = None
    settings.TRAINING_MAX_RUNS = 5
    return training_dir


@pytest.fixture
def team(db):
    users = {
        name: get_user_model().objects.create_user(name, password="correct-horse-1")
        for name in ("ada", "bob", "cy")
    }
    project = Project.objects.create(owner=users["ada"], name="Fraud", graph={})
    ProjectMember.objects.create(project=project, user=users["bob"], role="editor")
    ProjectMember.objects.create(project=project, user=users["cy"], role="viewer")
    return project, users


def client_for(user) -> APIClient:
    client = APIClient()
    client.force_authenticate(user)
    return client


def create(client, members, **body):
    return client.post(
        "/api/training/ensembles/", {"members": [str(m) for m in members], **body}, format="json"
    )


def finished(client, run_id):
    data = wait_for(client, run_id, timeout=60, interval=0.1)
    assert data["status"] == "completed", data["message"]
    return data


# --------------------------------------------------------------------------- #
# Checking the members
# --------------------------------------------------------------------------- #


@pytest.mark.django_db
def test_members_are_checked(team, real_processes):
    project, users = team
    ada = users["ada"]
    client = client_for(ada)
    dataset, _, _ = make_dataset(ada)
    first, _, _ = member(ada, dataset, "LR")
    second, _, _ = member(ada, dataset, "LR strong", c=0.05)

    def refused(members, **body):
        response = create(client, members, **body)
        assert response.status_code == 400, response.data
        return response.data["detail"]

    assert "at least 2 finished runs" in refused([first.pk])
    assert "only once" in refused([first.pk, first.pk])
    assert "at most 10 runs" in refused([first.pk] * 11)
    assert refused(["not-a-run", first.pk]).startswith("No such training run: 'not-a-run'")

    running, _, _ = member(ada, dataset, "Still training", status="running")
    TrainingRun.objects.filter(pk=running.pk).update(status="failed", message="Out of memory")
    assert refused([first.pk, running.pk]) == (
        "'Still training' has not finished: This run failed before it saved weights: Out of memory"
    )
    other_task, _, _ = member(ada, dataset, "Regression")
    other_task.plan = {**other_task.plan, "task": "regression"}
    other_task.save()
    assert refused([first.pk, other_task.pk]) == (
        "The runs solve different tasks: 'LR' is a classification, 'Regression' a regression."
    )
    three, _, _ = member(ada, dataset, "Three outputs", class_names=[*CLASSES, "class 2"])
    assert refused([first.pk, three.pk]).startswith(
        "The runs predict different classes: 'LR' has 2 classes (ok, fraud), 'Three outputs' "
        "has 3 classes (ok, fraud, class 2)."
    )
    elsewhere, _, _ = make_dataset(ada, name="Other payments")
    moved, _, _ = member(ada, elsewhere, "Other data")
    assert refused([first.pk, moved.pk]) == (
        "The runs were trained on different datasets: 'LR' on 'Payments', 'Other data' on "
        "'Other payments'. An ensemble needs runs trained on the same dataset, so that their "
        "validation and test rows line up."
    )
    loose, _, _ = make_dataset(ada, name="Plain upload", stored=False)
    plain_a, _, _ = member(ada, loose, "Plain A")
    plain_b, _, _ = member(ada, loose, "Plain B")
    assert refused([plain_a.pk, plain_b.pk]).startswith(
        "The dataset 'Plain upload' has no stored split."
    )
    demo = TrainingRun.objects.create(
        owner=ada,
        name="Demo",
        dataset={"kind": "random"},
        plan={"task": "demo"},
        status="completed",
        has_weights=True,
    )
    assert "is a demo run" in refused([first.pk, demo.pk])

    # Runs of other people: only those shared in a project the user belongs to.
    stranger = get_user_model().objects.create_user("eve", password="correct-horse-1")
    theirs, _, _ = member(stranger, dataset, "Eve's run")
    assert refused([first.pk, theirs.pk]).startswith(f"No such training run: '{theirs.pk}'")
    shared, _, _ = member(users["bob"], dataset, "Bob's run", project=project)
    response = create(client, [first.pk, shared.pk], name="With Bob's run")
    assert response.status_code == 201, response.data
    assert response.data["kind"] == "ensemble" and response.data["family"] == "ensemble"

    # An ensemble is no member, and stacking needs validation rows and no weights.
    ensemble_id = response.data["id"]
    assert "is an ensemble itself" in refused([first.pk, ensemble_id])
    assert "leave out the weights" in refused(
        [first.pk, second.pk], method="stacking", weights=[1, 2]
    )
    no_val, _, _ = make_dataset(ada, split=[2 if i % 5 == 1 else 0 for i in range(200)])
    a, _, _ = member(ada, no_val, "A")
    b, _, _ = member(ada, no_val, "B")
    assert "this dataset has none" in refused([a.pk, b.pk], method="stacking")
    assert "one weight per member" in refused([first.pk, second.pk], weights=[1])
    finished(client, ensemble_id)


# --------------------------------------------------------------------------- #
# Through the API: evaluation, weights, threshold, predictions, deletion
# --------------------------------------------------------------------------- #


@pytest.mark.django_db
def test_average_ensemble_through_the_api(team, real_processes, django_capture_on_commit_callbacks):
    project, users = team
    ada = users["ada"]
    client = client_for(ada)
    dataset, x, y = make_dataset(ada)
    first, first_model, first_mask = member(ada, dataset, "Only a")
    second, second_model, second_mask = member(ada, dataset, "Only b", columns=(1,))

    response = create(
        client, [first.pk, second.pk], name="Fraud ensemble", weights=[3, 1], project=project.pk
    )
    assert response.status_code == 201, response.data
    run_id = response.data["id"]
    data = finished(client, run_id)
    assert data["message"].startswith("Combined 2 runs (weighted average): validation accuracy")
    assert data["ensemble"]["method"] == "average" and data["ensemble"]["broken"] is None
    assert [(m["name"], m["weight"], m["visible"]) for m in data["ensemble"]["members"]] == [
        ("Only a", 0.75, True),
        ("Only b", 0.25, True),
    ]
    assert data["project"] == project.pk and data["has_weights"] is True
    assert data["epochs"] == 1 and data["decision_threshold"] == 0.5

    # The evaluation is that of the weighted average of the members' probabilities.
    val = np.flatnonzero(np.array(SPLIT) == 1)
    test = np.flatnonzero(np.array(SPLIT) == 2)
    proba = [first_model.predict_proba(x * first_mask), second_model.predict_proba(x * second_mask)]
    combined = 0.75 * proba[0] + 0.25 * proba[1]
    state = data["state"]
    found = state["evaluation"]
    assert found["test"]["accuracy"] == pytest.approx(
        np.mean(combined[test].argmax(axis=1) == y[test])
    )
    assert found["val"]["threshold_curve"]["roc_auc"] == pytest.approx(
        roc_auc_score(y[val], combined[val, 1])
    )
    assert found["binary"]["source"] == "val"
    assert data["metrics"]["val_accuracy"] == found["val"]["accuracy"]
    assert data["metrics"]["test_accuracy"] == found["test"]["accuracy"]
    rows = {row["name"]: row for row in state["ensemble"]["comparison"]}
    assert rows["Only b"]["test"]["accuracy"] == pytest.approx(
        np.mean(proba[1][test].argmax(axis=1) == y[test])
    )
    assert state["val_size"] == len(val) and state["test_size"] == len(test)
    assert state["preprocessing"]["columns"] == ["a", "b", "c"]
    # Its CPU time counts like a training's.
    assert UsageRecord.objects.filter(run_id=run_id, kind="training").exists()

    # New weights: measured again on the saved predictions.
    url = f"/api/training/runs/{run_id}/ensemble/"
    bob = client_for(users["bob"])
    assert bob.patch(url, {"weights": [1, 1]}, format="json").status_code == 403
    assert client.patch(url, {"weights": [1]}, format="json").status_code == 400
    before = TrainingRun.objects.get(pk=run_id).cpu_seconds
    response = client.patch(url, {"weights": [1, 1]}, format="json")
    assert response.status_code == 200, response.data
    assert [m["weight"] for m in response.data["ensemble"]["members"]] == [0.5, 0.5]
    combined = (proba[0] + proba[1]) / 2
    found = response.data["state"]["evaluation"]
    assert found["test"]["accuracy"] == pytest.approx(
        np.mean(combined[test].argmax(axis=1) == y[test])
    )
    assert response.data["metrics"]["test_accuracy"] == found["test"]["accuracy"]
    assert response.data["state"]["ensemble"]["weights"] == [0.5, 0.5]
    assert TrainingRun.objects.get(pk=run_id).cpu_seconds > before  # counted as its training
    assert list(
        EnsembleMember.objects.filter(ensemble_id=run_id).values_list("weight", flat=True)
    ) == [
        0.5,
        0.5,
    ]

    # Its own decision threshold.
    response = client.post(
        f"/api/training/runs/{run_id}/decision-threshold/", {"threshold": 0.3}, format="json"
    )
    assert response.status_code == 200, response.data

    # Predictions: every member predicts, the ensemble averages and applies its threshold.
    rows_x = x[:6]
    inputs = [dict(zip(["a", "b", "c"], map(float, r), strict=True)) for r in rows_x]
    response = client.post(
        f"/api/training/runs/{run_id}/predict/", {"inputs": inputs, "members": True}, format="json"
    )
    assert response.status_code == 200, response.data
    answer = response.data
    expected = (
        first_model.predict_proba(rows_x * first_mask)
        + second_model.predict_proba(rows_x * second_mask)
    ) / 2
    got = [p["positive_probability"] for p in answer["predictions"]]
    np.testing.assert_allclose(got, expected[:, 1], rtol=0, atol=1e-5)
    assert [p["class"] for p in answer["predictions"]] == [
        "fraud" if p >= 0.3 else "ok" for p in expected[:, 1]
    ]
    assert answer["decision_threshold"] == 0.3 and answer["positive_class"] == "fraud"
    assert answer["ensemble"]["method"] == "average"
    assert [m["name"] for m in answer["members"]] == ["Only a", "Only b"]
    member_probability = [p["positive_probability"] for p in answer["members"][1]["predictions"]]
    np.testing.assert_allclose(
        member_probability, second_model.predict_proba(rows_x * second_mask)[:, 1], atol=1e-4
    )
    # Without "members" only the ensemble's predictions; the viewer of the project may too.
    response = client_for(users["cy"]).post(
        f"/api/training/runs/{run_id}/predict/", {"inputs": inputs[:1]}, format="json"
    )
    assert response.status_code == 200, response.data
    assert "members" not in response.data
    assert UsageRecord.objects.filter(run_id=run_id, kind="prediction").exists()
    form = client.get(f"/api/training/runs/{run_id}/predict-form/").data
    assert form["available"] is True and form["kind"] == "tabular"
    assert form["class_names"] == CLASSES
    assert len(form["prepared"]["samples"]) == 5
    response = client.post(
        f"/api/training/runs/{run_id}/predict/",
        {"inputs": inputs[:1], "input_scale": "normalized"},
        format="json",
    )
    assert response.status_code == 400 and "input_scale 'raw'" in response.data["detail"]

    # An ensemble has no weights of its own to download or to explain.
    response = client.get(f"/api/training/runs/{run_id}/weights/")
    assert response.status_code == 404

    # Deleting: a member is kept while the ensemble uses it.
    response = client.delete(f"/api/training/runs/{first.pk}/")
    assert response.status_code == 409
    assert response.data["detail"] == (
        "'Only a' is a member of the ensemble 'Fraud ensemble', which predicts with it. "
        "Delete the ensemble first."
    )
    assert TrainingRun.objects.filter(pk=first.pk).exists()
    with django_capture_on_commit_callbacks(execute=True):
        assert client.delete(f"/api/training/runs/{run_id}/").status_code == 204
    assert not EnsembleMember.objects.exists()
    assert not (real_processes / "runs" / run_id).exists()
    assert client.delete(f"/api/training/runs/{first.pk}/").status_code == 204


@pytest.mark.django_db
def test_stacking_through_mcp(team, real_processes):
    _, users = team
    ada = users["ada"]
    dataset, x, y = make_dataset(ada)
    first, first_model, first_mask = member(ada, dataset, "Only a")
    second, second_model, second_mask = member(ada, dataset, "Only b", columns=(1,))
    mcp = mcp_client(ada)

    result, text = call(
        mcp, "create_ensemble", members=[str(first.pk), str(second.pk)], method="stacking"
    )
    assert not result.get("isError"), text
    run_id = result["structuredContent"]["id"]
    assert result["structuredContent"]["family"] == "ensemble"
    finished(client_for(ada), run_id)

    result, text = call(mcp, "list_training_runs")
    line = next(r for r in result["structuredContent"]["runs"] if r["id"] == run_id)
    assert line["ensemble"] == "stacking of 2 runs: Only a, Only b"
    assert line["family"] == "ensemble"

    result, text = call(mcp, "get_training_run", id=run_id)
    detail = result["structuredContent"]
    assert detail["ensemble"]["method"] == "stacking"
    assert [m["name"] for m in detail["ensemble"]["members"]] == ["Only a", "Only b"]
    assert detail["ensemble"]["note"].startswith(
        "The stacking model was fitted on the validation rows"
    )
    names = [row["name"] for row in detail["ensemble"]["comparison"]]
    assert names == ["Only a", "Only b", "Ensemble"]
    assert detail["ensemble"]["stacking_model"]["model"] == "logistic_regression"
    assert detail["evaluation"]["test"]["roc_auc"] is not None

    # The stacking model fitted on the members' validation probabilities.
    val = np.flatnonzero(np.array(SPLIT) == 1)
    features = np.column_stack(
        [
            first_model.predict_proba(x * first_mask)[:, 1],
            second_model.predict_proba(x * second_mask)[:, 1],
        ]
    )
    meta = LogisticRegression(max_iter=1000).fit(features[val], y[val])
    inputs = [dict(zip(["a", "b", "c"], map(float, r), strict=True)) for r in x[:4]]
    result, text = call(mcp, "predict", run_id=run_id, inputs=inputs, members=True)
    assert not result.get("isError"), text
    answer = result["structuredContent"]
    got = [p["positive_probability"] for p in answer["predictions"]]
    np.testing.assert_allclose(got, meta.predict_proba(features[:4])[:, 1], atol=1e-5)
    assert len(answer["members"]) == 2

    result, text = call(mcp, "set_decision_threshold", id=run_id, threshold=0.2)
    assert not result.get("isError"), text
    # Stacking has no weights to change.
    response = client_for(ada).patch(
        f"/api/training/runs/{run_id}/ensemble/", {"weights": [1, 2]}, format="json"
    )
    assert response.status_code == 400 and "no weights to set" in response.data["detail"]

    result, text = call(mcp, "delete_training_run", id=str(first.pk))
    assert result.get("isError") and "Delete the ensemble first" in text
    logged = AgentAction.objects.get(tool="create_ensemble")
    assert logged.summary == "Combined 2 runs into ensemble “Ensemble” (stacking)"

    # The account's export lists the ensemble with its members.
    exported = {run["id"]: run for run in accounts._runs(ada)}
    assert exported[run_id]["kind"] == "ensemble"
    assert exported[run_id]["ensemble"]["members"][0]["name"] == "Only a"
    assert exported[str(first.pk)]["ensemble"] is None


@pytest.mark.django_db
def test_a_member_that_goes_breaks_only_its_predictions(team, real_processes):
    """Bob deletes his run, a member of Ada's ensemble that he cannot see: allowed, and
    the ensemble keeps its results but cannot predict."""
    project, users = team
    ada, bob = users["ada"], users["bob"]
    dataset, x, _ = make_dataset(ada)
    first, _, _ = member(ada, dataset, "Ada's run")
    bobs, _, _ = member(bob, dataset, "Bob's run", columns=(1,), project=project)
    client = client_for(ada)
    response = create(client, [first.pk, bobs.pk], name="Private ensemble")
    assert response.status_code == 201, response.data
    run_id = response.data["id"]
    finished(client, run_id)

    # Members of an ensemble are kept by the clean-up of old runs.
    from django.utils import timezone

    TrainingRun.objects.update(last_used_at=timezone.now() - timezone.timedelta(days=400))
    unused = set(retention.unused_runs(timezone.now()).values_list("pk", flat=True))
    assert unused == {uuid_of(run_id)}

    assert client_for(bob).delete(f"/api/training/runs/{bobs.pk}/").status_code == 204
    data = client.get(f"/api/training/runs/{run_id}/").data
    assert data["status"] == "completed" and data["state"]["evaluation"]["val"]
    assert data["ensemble"]["members"][1] == {
        "id": None,
        "name": "Bob's run",
        "weight": 0.5,
        "family": None,
        "model": None,
        "status": None,
        "visible": False,
        "deleted": True,
    }
    assert data["ensemble"]["broken"] == (
        "The member 'Bob's run' of this ensemble was deleted, so it cannot predict any more. "
        "Its results are kept."
    )
    inputs = [{"a": 0.1, "b": 0.2, "c": 0.3}]
    response = client.post(
        f"/api/training/runs/{run_id}/predict/", {"inputs": inputs}, format="json"
    )
    assert response.status_code == 409 and "cannot predict any more" in response.data["detail"]
    form = client.get(f"/api/training/runs/{run_id}/predict-form/").data
    assert form["available"] is False and "was deleted" in form["reason"]

    # Deleting the account takes the ensembles with it.
    ada.delete()
    assert not TrainingRun.objects.filter(owner_id=ada.pk).exists()
    assert not EnsembleMember.objects.exists()


def uuid_of(value):
    import uuid

    return uuid.UUID(str(value))


@pytest.mark.django_db
def test_login_and_permissions(team, real_processes):
    project, users = team
    ada = users["ada"]
    dataset, _, _ = make_dataset(ada)
    first, _, _ = member(ada, dataset, "A")
    second, _, _ = member(ada, dataset, "B", c=0.1)
    response = APIClient().post(
        "/api/training/ensembles/", {"members": [str(first.pk), str(second.pk)]}, format="json"
    )
    assert response.status_code == 401
    # Bob cannot see Ada's own runs (they are not in the project).
    response = create(client_for(users["bob"]), [first.pk, second.pk])
    assert response.status_code == 400 and response.data["detail"].startswith("No such training")
    # Ensembles go into projects the user edits, not into those they only view.
    shared_a, _, _ = member(ada, dataset, "Shared A", project=project)
    shared_b, _, _ = member(ada, dataset, "Shared B", c=0.1, project=project)
    response = create(client_for(users["cy"]), [shared_a.pk, shared_b.pk], project=project.pk)
    assert response.status_code == 400 and "projects you own or can edit" in response.data["detail"]
    response = create(client_for(users["bob"]), [shared_a.pk, shared_b.pk], project=project.pk)
    assert response.status_code == 201, response.data
    run_id = response.data["id"]
    finished(client_for(users["bob"]), run_id)
    # Members of the project see it; the viewer cannot delete it, Ada (the owner) can.
    assert client_for(users["cy"]).get(f"/api/training/runs/{run_id}/").status_code == 200
    assert client_for(users["cy"]).delete(f"/api/training/runs/{run_id}/").status_code == 403
    # Ada sees the ensemble that uses her runs: she has to delete it first.
    response = client_for(ada).delete(f"/api/training/runs/{shared_a.pk}/")
    assert response.status_code == 409
    assert client_for(ada).delete(f"/api/training/runs/{run_id}/").status_code == 204
    assert client_for(ada).delete(f"/api/training/runs/{shared_a.pk}/").status_code == 204


# --------------------------------------------------------------------------- #
# End to end: a neural network and gradient boosting on an exported dataset
# --------------------------------------------------------------------------- #


def payments_csv(n: int = 240) -> bytes:
    """Raw payments with a category and a rare-ish class."""
    rng = np.random.default_rng(11)
    lines = ["amount;merchant;hour;label"]
    merchants = ["shop", "travel", "games"]
    for i in range(n):
        amount = float(rng.gamma(2.0, 40.0))
        merchant = merchants[i % 3]
        hour = int(rng.integers(0, 24))
        risky = amount > 110 or (merchant == "games" and hour < 6)
        label = "fraud" if risky else "ok"
        lines.append(f"{amount:.2f};{merchant};{hour};{label}")
    return ("\n".join(lines) + "\n").encode()


@pytest.mark.skipif(not torch_installed(), reason="PyTorch is not installed")
@pytest.mark.django_db
def test_neural_and_classical_members_predict_raw_rows(real_processes, settings, tmp_path):
    from .test_data_api import import_csv

    settings.DATA_PREP_DIR = tmp_path / "prepare"
    settings.DATA_PREP_ENABLED = True
    user = get_user_model().objects.create_user("ada", password="correct-horse-1")
    client = client_for(user)
    source = import_csv(client, "payments.csv", payments_csv()).data
    body = {
        "name": "Payments",
        "source": source["id"],
        "steps": [
            {
                "id": "onehot",
                "type": "encode",
                "params": {"columns": ["merchant"], "method": "onehot"},
            }
        ],
        "settings": {
            "target": "label",
            "task": "classification",
            "split": {"val": 0.25, "test": 0.25, "seed": 1},
        },
    }
    prep = client.post("/api/data/preps/", body, format="json")
    assert prep.status_code == 201, prep.data
    exported = client.post(f"/api/data/preps/{prep.data['id']}/export/", {}, format="json")
    assert exported.status_code == 201, exported.data
    dataset = Dataset.objects.get(pk=exported.data["id"])
    features = len(dataset.info["columns"]) - 1
    ref = {"kind": "upload", "id": dataset.pk}

    mlp = chain(
        ("Input", {"shape": [features]}),
        ("Linear", {"out_features": 8}),
        ("Activation", {}),
        ("Linear", {"out_features": 2}),
    )
    boosting = {
        "version": 1,
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [features]}},
            {
                "id": "m",
                "type": "HistGradientBoostingClassifier",
                "params": {"max_iter": 30, "min_samples_leaf": 5, "early_stopping": "off"},
            },
        ],
        "edges": [{"source": "in", "target": "m"}],
    }
    ids = []
    for name, graph, config in (
        ("MLP", mlp, {"epochs": 15, "batch_size": 16, "device": "cpu", "seed": 1}),
        ("Boosting", boosting, {"seed": 1}),
    ):
        response = client.post(
            "/api/training/runs/",
            {"name": name, "graph": graph, "dataset": ref, "config": config},
            format="json",
        )
        assert response.status_code == 201, response.data
        data = wait_for(client, response.data["id"], timeout=120, interval=0.25)
        assert data["status"] == "completed", data["message"]
        ids.append(data["id"])
    own = [client.get(f"/api/training/runs/{i}/").data["state"]["evaluation"] for i in ids]

    response = create(client, ids, name="MLP + trees")
    assert response.status_code == 201, response.data
    run_id = response.data["id"]
    data = wait_for(client, run_id, timeout=120, interval=0.25)
    assert data["status"] == "completed", data["message"]
    comparison = {row["name"]: row for row in data["state"]["ensemble"]["comparison"]}
    # The members measured on the same rows as in their own evaluation.
    for name, found in zip(("MLP", "Boosting"), own, strict=True):
        assert comparison[name]["val"]["accuracy"] == pytest.approx(found["val"]["accuracy"])
        assert comparison[name]["test"]["size"] == found["test"]["size"]
    assert data["ensemble"]["members"][0]["model"] == "Neural network"

    raw = [
        {"amount": "250.00", "merchant": "travel", "hour": 13},
        {"amount": "12.50", "merchant": "shop", "hour": 10},
        {"amount": "40.00", "merchant": "games", "hour": 3},
    ]
    response = client.post(
        f"/api/training/runs/{run_id}/predict/",
        {"inputs": raw, "input_format": "raw", "members": True},
        format="json",
    )
    assert response.status_code == 200, response.data
    answer = response.data
    # The average of what each member predicts on its own.
    alone = []
    for member_id in ids:
        found = client.post(
            f"/api/training/runs/{member_id}/predict/",
            {"inputs": raw, "input_format": "raw"},
            format="json",
        )
        assert found.status_code == 200, found.data
        alone.append([p["positive_probability"] for p in found.data["predictions"]])
    expected = np.mean(alone, axis=0)
    got = [p["positive_probability"] for p in answer["predictions"]]
    np.testing.assert_allclose(got, expected, atol=1e-3)
    for index, found in enumerate(answer["members"]):
        member_got = [p["positive_probability"] for p in found["predictions"]]
        np.testing.assert_allclose(member_got, alone[index], atol=1e-6)
    assert answer["predictions"][0]["class"] in ("fraud", "ok")


@pytest.mark.django_db
def test_a_training_worker_combines_a_queued_ensemble(team, real_processes, settings):
    from brick_dnn_core.training_worker import Worker
    from brickdnn.training.state import read_job

    settings.TRAINING_WORKER = True
    settings.WORKER_NAME = ""
    settings.WORKER_SLOTS = 0
    settings.WORKER_TIMEOUT = 60
    settings.WORKER_MAX_RESTARTS = 3
    settings.WORKER_DRAIN_SECONDS = 0
    settings.QUEUE_MAX_WAITING = None
    _, users = team
    ada = users["ada"]
    dataset, _, _ = make_dataset(ada)
    first, _, _ = member(ada, dataset, "A")
    second, _, _ = member(ada, dataset, "B", columns=(1,))
    client = client_for(ada)
    response = create(client, [first.pk, second.pk])
    assert response.status_code == 201, response.data
    assert response.data["status"] == "queued"
    run_id = response.data["id"]

    worker = Worker(name="worker-1", slots=1)
    worker.start()
    worker.step()
    job = read_job(real_processes / "runs" / run_id)
    assert [m["name"] for m in job["ensemble"]["members"]] == ["A", "B"]
    assert job["ensemble"]["weights"] == [0.5, 0.5]
    assert job["ensemble"]["class_targets"] == "indices"
    deadline = time.monotonic() + 60
    while TrainingRun.objects.get(pk=run_id).status != "completed":
        assert time.monotonic() < deadline, TrainingRun.objects.get(pk=run_id).message
        worker.step()
        time.sleep(0.05)
    data = client.get(f"/api/training/runs/{run_id}/").data
    assert data["has_weights"] is True and data["state"]["evaluation"]["test"]["accuracy"]
    worker.stop()
