"""Binary classification: the threshold analysis of the final evaluation (threshold curves,
ROC AUC and PR AUC against scikit-learn, the suggestions, which split the thresholds come
from, bounded size) and the decision threshold of a run (the REST endpoint, who may set it,
predictions that use it, the MCP tools)."""

# ruff: noqa: F811 (the fixtures training_dir and predictor are imported)

from __future__ import annotations

import json
import math

import pytest

np = pytest.importorskip("numpy")
sklearn_metrics = pytest.importorskip("sklearn.metrics")

from brickdnn.training import evaluation as ev  # noqa: E402
from brickdnn.training import inference  # noqa: E402

# --------------------------------------------------------------------------- #
# The threshold analysis
# --------------------------------------------------------------------------- #


def imbalanced(rng, n, share=0.02, ties=False):
    """Probabilities of the positive class and true classes, about ``share`` positives."""
    y = (rng.random(n) < share).astype(np.int64)
    if y.sum() == 0:
        y[0] = 1
    p = np.clip(rng.normal(0.25 + 0.4 * y, 0.15), 0.0, 1.0)
    if ties:
        p = np.round(p, 2)
    return p, y


def split_of(p, y, loss=0.1):
    counter = ev.ClassificationCounter(2)
    counter.add(np.c_[1 - p, p], y)
    return counter.result(loss)


def binary_evaluation(val=None, test=None, training=None):
    return ev.evaluation(
        "classification",
        val,
        test,
        {"loss": 0.2, "accuracy": 0.9},
        class_names=["ok", "fraud"],
        epoch=3,
        training=training,
    )


def brute_force(p, y, threshold):
    predicted = p >= threshold
    tp = int((predicted & (y == 1)).sum())
    fp = int((predicted & (y == 0)).sum())
    return tp, fp, int((y == 0).sum()) - fp, int((y == 1).sum()) - tp


@pytest.mark.parametrize("ties", [False, True])
def test_aucs_match_scikit_learn(ties):
    rng = np.random.default_rng(3)
    p, y = imbalanced(rng, 3000, ties=ties)
    found = binary_evaluation(split_of(p, y))
    curve = found["val"]["threshold_curve"]
    assert curve["positive_class"] == 1
    assert curve["positives"] == int(y.sum()) and curve["negatives"] == int((y == 0).sum())
    assert curve["roc_auc"] == pytest.approx(sklearn_metrics.roc_auc_score(y, p))
    assert curve["pr_auc"] == pytest.approx(sklearn_metrics.average_precision_score(y, p))


def test_curve_points_count_the_rows_at_or_above_each_threshold():
    rng = np.random.default_rng(4)
    p, y = imbalanced(rng, 2000, share=0.1, ties=True)
    found = binary_evaluation(split_of(p, y))
    points = found["val"]["threshold_curve"]["points"]
    thresholds = points["threshold"]
    assert thresholds == sorted(set(thresholds)) and thresholds[0] == 0.0
    assert thresholds[-1] == 1.0 and 0.5 in thresholds
    assert len(thresholds) <= ev.MAX_CURVE_POINTS
    keys = ("threshold", "precision", "recall", "fpr", "tp", "fp", "tn", "fn")
    assert set(points) == set(keys) and all(len(points[k]) == len(thresholds) for k in keys)
    for i, threshold in enumerate(thresholds):
        tp, fp, tn, fn = brute_force(p, y, threshold)
        assert (points["tp"][i], points["fp"][i], points["tn"][i], points["fn"][i]) == (
            tp,
            fp,
            tn,
            fn,
        )
        if tp + fp:
            assert points["precision"][i] == pytest.approx(tp / (tp + fp), abs=1e-4)
        else:
            assert points["precision"][i] is None
        assert points["recall"][i] == pytest.approx(tp / (tp + fn), abs=1e-4)
        assert points["fpr"][i] == pytest.approx(fp / (fp + tn), abs=1e-4)
    # Recall falls (or stays) as the threshold rises.
    assert all(a >= b for a, b in zip(points["recall"], points["recall"][1:], strict=False))


def test_suggestions_best_f1_and_the_recall_target():
    rng = np.random.default_rng(5)
    p, y = imbalanced(rng, 4000)
    found = binary_evaluation(split_of(p, y))
    binary = found["binary"]
    assert binary["positive_class"] == 1 and binary["recall_target"] == ev.RECALL_TARGET
    assert binary["source"] == "val"
    # Brute force over every distinct probability.
    best = max(
        2 * tp / (2 * tp + fp + fn)
        for tp, fp, _, fn in map(lambda t: brute_force(p, y, t), np.unique(p))
    )
    suggested = binary["best_f1"]
    tp, fp, tn, fn = brute_force(p, y, suggested["threshold"])
    assert (tp, fp, tn, fn) == tuple(suggested[k] for k in ("tp", "fp", "tn", "fn"))
    assert suggested["f1"] == pytest.approx(best, abs=1e-4)
    recall = binary["recall"]
    assert recall["recall"] >= ev.RECALL_TARGET
    assert recall["tp"] == brute_force(p, y, recall["threshold"])[0]
    # Any higher threshold finds fewer than the target.
    higher = np.unique(p[p > recall["threshold"] + 1e-6])
    if len(higher):
        tp_higher = brute_force(p, y, higher[0])[0]
        assert tp_higher / y.sum() < ev.RECALL_TARGET
    # Both suggestions are points of the curve, so the slider can show them exactly.
    thresholds = found["val"]["threshold_curve"]["points"]["threshold"]
    assert suggested["threshold"] in thresholds and recall["threshold"] in thresholds


def test_the_same_thresholds_for_validation_and_test():
    rng = np.random.default_rng(6)
    val, test = split_of(*imbalanced(rng, 1500)), split_of(*imbalanced(rng, 900))
    found = binary_evaluation(val, test)
    assert found["binary"]["source"] == "val"
    val_points = found["val"]["threshold_curve"]["points"]
    test_points = found["test"]["threshold_curve"]["points"]
    assert val_points["threshold"] == test_points["threshold"]
    assert (
        found["test"]["threshold_curve"]["positives"]
        + found["test"]["threshold_curve"]["negatives"]
        == (found["test"]["size"])
    )


def test_thresholds_from_test_or_training_rows_without_validation_rows():
    rng = np.random.default_rng(7)
    test = split_of(*imbalanced(rng, 800))
    found = binary_evaluation(None, test)
    assert found["binary"]["source"] == "test" and found["val"] is None
    assert found["test"]["threshold_curve"]["roc_auc"] is not None
    training = split_of(*imbalanced(rng, 800))
    found = binary_evaluation(None, None, training)
    assert found["binary"]["source"] == "training"
    assert found["training"]["threshold_curve"]["pr_auc"] is not None
    assert found["binary"]["best_f1"] is not None


def test_without_positive_rows_there_are_no_suggestions():
    p = np.linspace(0.0, 0.6, 50)
    found = binary_evaluation(split_of(p, np.zeros(50, dtype=np.int64)))
    curve = found["val"]["threshold_curve"]
    assert curve["roc_auc"] is None and curve["pr_auc"] is None and curve["positives"] == 0
    assert found["binary"]["best_f1"] is None and found["binary"]["recall"] is None
    assert all(r is None for r in curve["points"]["recall"])


def test_rows_without_a_probability_are_left_out():
    p = np.array([0.9, np.nan, 0.2, 0.6, np.nan])
    y = np.array([1, 1, 0, 0, 0])
    found = binary_evaluation(split_of(p, y))
    curve = found["val"]["threshold_curve"]
    assert (curve["positives"], curve["negatives"]) == (1, 2)
    assert curve["roc_auc"] == 1.0
    json.dumps(found, allow_nan=False)
    found = binary_evaluation(split_of(np.full(3, np.nan), np.array([0, 1, 0])))
    assert found["val"]["threshold_curve"]["positives"] == 0
    assert found["binary"]["best_f1"] is None


def test_state_json_stays_small_for_many_rows():
    rng = np.random.default_rng(8)
    val, test = split_of(*imbalanced(rng, 200_000)), split_of(*imbalanced(rng, 100_000))
    found = binary_evaluation(val, test)
    for key in ("val", "test"):
        assert len(found[key]["threshold_curve"]["points"]["threshold"]) <= ev.MAX_CURVE_POINTS
    text = json.dumps(found, allow_nan=False)
    assert len(text) < 40_000
    assert "binary_scores" not in text


def test_no_threshold_analysis_for_more_classes_and_regressions():
    rng = np.random.default_rng(9)
    counter = ev.ClassificationCounter(3)
    counter.add(rng.dirichlet(np.ones(3), size=60), rng.integers(0, 3, size=60))
    found = ev.evaluation("classification", counter.result(0.5), None, {"loss": 0.4})
    assert "binary" not in found and "threshold_curve" not in found["val"]
    collector = ev.RegressionCollector()
    collector.add(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.5, 2.0]))
    found = ev.evaluation("regression", collector.result(0.1), None, {"loss": 0.1})
    assert "binary" not in found and "threshold_curve" not in found["val"]


def test_a_failing_analysis_keeps_the_rest_of_the_evaluation(monkeypatch, capsys):
    rng = np.random.default_rng(10)
    val = split_of(*imbalanced(rng, 100))

    def broken(splits):
        raise RuntimeError("no curves today")

    monkeypatch.setattr(ev, "threshold_analysis", broken)
    found = binary_evaluation(val)
    assert "binary" not in found and found["val"]["accuracy"] is not None
    assert "no curves today" in capsys.readouterr().err


def test_mcp_summary_has_the_aucs_and_the_metrics_at_the_threshold():
    from brick_dnn_core import mcp_compact

    rng = np.random.default_rng(11)
    p, y = imbalanced(rng, 1000)
    found = binary_evaluation(split_of(p, y))
    summary = mcp_compact.evaluation_summary(found, 0.3)
    val = summary["validation"]
    assert val["roc_auc"] == pytest.approx(found["val"]["threshold_curve"]["roc_auc"], rel=1e-3)
    binary = summary["binary"]
    assert binary["positive_class"] == "fraud" and binary["suggestions_from"] == "validation"
    assert binary["best_f1"]["threshold"] == pytest.approx(found["binary"]["best_f1"]["threshold"])
    assert binary["recall_80"]["recall"] >= 0.8
    point = binary["at_decision_threshold"]["validation"]
    assert point["threshold"] == 0.3
    assert (point["tp"], point["fp"], point["tn"], point["fn"]) == brute_force(p, y, 0.3)
    assert "points" not in json.dumps(summary) and len(json.dumps(summary)) < 2500


def test_cli_prints_the_aucs():
    import io

    from brickdnn.training import cli
    from brickdnn.training.state import Path

    rng = np.random.default_rng(12)
    out = io.StringIO()
    reporter = cli.ConsoleReporter(1, Path("w.pt"), out)
    reporter.evaluation_end(binary_evaluation(split_of(*imbalanced(rng, 500))))
    text = out.getvalue()
    assert "ROC AUC" in text and "PR AUC" in text


# --------------------------------------------------------------------------- #
# Predictions with a threshold (no processes)
# --------------------------------------------------------------------------- #

BINARY_PLAN = {
    "task": "classification",
    "inputs": [{"shape": [2], "dtype": "float", "vocab": None}],
    "targets": [{"kind": "class", "num_classes": 2}],
    "class_names": ["ok", "fraud"],
}
BINARY_PREPROCESSING = {
    "kind": "tabular",
    "columns": ["amount", "hour"],
    "target": "label",
    "mean": [0.0, 0.0],
    "std": [1.0, 1.0],
    "class_names": ["ok", "fraud"],
}


def test_is_binary():
    assert inference.is_binary(BINARY_PLAN)
    assert not inference.is_binary(
        {**BINARY_PLAN, "targets": [{"kind": "class", "num_classes": 3}]}
    )
    assert not inference.is_binary({"task": "regression", "targets": [{"kind": "value"}]})
    assert not inference.is_binary(
        {"task": "segmentation", "targets": [{"kind": "mask", "num_classes": 2}]}
    )
    assert not inference.is_binary(None)


def test_read_outputs_uses_the_threshold_on_the_second_class():
    # p(fraud) of the two samples: 0.2 and 0.6.
    outputs = [[[0.8, 0.2], [0.4, 0.6]]]
    plain = inference.read_outputs(
        BINARY_PLAN, BINARY_PREPROCESSING, outputs, 2, probabilities=True
    )
    assert [p["class"] for p in plain] == ["ok", "fraud"] and "positive_probability" not in plain[0]
    low = inference.read_outputs(
        BINARY_PLAN, BINARY_PREPROCESSING, outputs, 2, probabilities=True, threshold=0.15
    )
    assert [p["class"] for p in low] == ["fraud", "fraud"]
    assert low[0]["probability"] == 0.2 and low[0]["positive_probability"] == 0.2
    assert low[0]["index"] == 1 and low[0]["top"][0]["class"] == "ok"  # order of probability
    high = inference.read_outputs(
        BINARY_PLAN, BINARY_PREPROCESSING, outputs, 2, probabilities=True, threshold=0.7
    )
    assert [p["class"] for p in high] == ["ok", "ok"] and high[1]["probability"] == 0.4
    # At the threshold itself: the positive class.
    edge = inference.read_outputs(
        BINARY_PLAN, BINARY_PREPROCESSING, outputs, 2, probabilities=True, threshold=0.6
    )
    assert edge[1]["class"] == "fraud"
    # Raw scores (a neural network): softmax first.
    logits = [[[0.0, math.log(3)]]]  # p(fraud) = 0.75
    (found,) = inference.read_outputs(BINARY_PLAN, BINARY_PREPROCESSING, logits, 1, threshold=0.8)
    assert found["class"] == "ok" and found["positive_probability"] == pytest.approx(0.75)
    # Other runs ignore it.
    three = {**BINARY_PLAN, "targets": [{"kind": "class", "num_classes": 3}]}
    (found,) = inference.read_outputs(
        three,
        {**BINARY_PREPROCESSING, "class_names": ["a", "b", "c"]},
        [[[0.1, 0.2, 0.7]]],
        1,
        probabilities=True,
        threshold=0.05,
    )
    assert found["class"] == "c" and "positive_probability" not in found


# --------------------------------------------------------------------------- #
# The REST endpoint, predictions and MCP
# --------------------------------------------------------------------------- #

from .test_mcp import call, mcp_client  # noqa: E402
from .test_predict import (  # noqa: E402, F401 (fixtures)
    PETS_GRAPH,
    PETS_PREPROCESSING,
    client_for,
    image_zip,
    mcp_predict,
    predict,
    predictor,
    team,
    trained_run,
    training_dir,
    use_predictor,
)


def pets_run(client, training_dir, project=None):
    pets = image_zip(["cats/1.png", "cats/2.png", "dogs/1.png", "dogs/2.png"])
    return trained_run(
        client, training_dir, PETS_GRAPH, "pets.zip", pets, PETS_PREPROCESSING, project
    )


def set_threshold(client, run_id, value):
    return client.post(
        f"/api/training/runs/{run_id}/decision-threshold/", {"threshold": value}, format="json"
    )


def with_evaluation(training_dir, run_id, p, y):
    from brickdnn.training.state import read_state, write_state

    directory = training_dir / "runs" / run_id
    state = read_state(directory)
    found = binary_evaluation(split_of(p, y))
    found["class_names"] = ["cats", "dogs"]
    state["evaluation"] = json.loads(json.dumps(found))
    write_state(directory, state)


# A raw 4×4 image whose first two standardised pixels are 2 and -2: the fake network
# echoes them as the scores of cats and dogs, so p(dogs) = 1 / (1 + e^4) ≈ 0.018.
IMAGE = [[255, 0, 0, 0]] + [[0] * 4] * 3


@pytest.mark.django_db
def test_set_the_threshold_and_predict_with_it(team, predictor, training_dir):
    _, users = team
    client = client_for(users["ada"])
    use_predictor(predictor, "echo:2")
    run_id = pets_run(client, training_dir)
    detail = client.get(f"/api/training/runs/{run_id}/").json()
    assert detail["decision_threshold"] == 0.5
    listed = client.get("/api/training/runs/").json()
    rows = listed["results"] if isinstance(listed, dict) else listed
    assert next(r for r in rows if r["id"] == run_id)["decision_threshold"] == 0.5

    response = predict(client, run_id, inputs=[IMAGE], input_scale="raw")
    assert response.status_code == 200, response.data
    (found,) = response.data["predictions"]
    assert found["class"] == "cats" and found["positive_probability"] == pytest.approx(
        0.018, abs=1e-3
    )
    assert response.data["decision_threshold"] == 0.5
    assert response.data["positive_class"] == "dogs"
    assert "at least the run's decision threshold 0.5" in response.data["threshold_note"]

    response = set_threshold(client, run_id, 0.01)
    assert response.status_code == 200, response.data
    assert response.data == {"id": run_id, "decision_threshold": 0.01}
    assert client.get(f"/api/training/runs/{run_id}/").json()["decision_threshold"] == 0.01
    response = predict(client, run_id, inputs=[IMAGE], input_scale="raw")
    (found,) = response.data["predictions"]
    assert found["class"] == "dogs" and found["index"] == 1
    assert found["probability"] == pytest.approx(0.018, abs=1e-3)
    assert response.data["decision_threshold"] == 0.01

    # The same over MCP.
    result, text = mcp_predict(mcp_client(users["ada"]), run_id, inputs=[IMAGE], input_scale="raw")
    assert "isError" not in result, text
    answer = result["structuredContent"]
    assert answer["predictions"][0]["class"] == "dogs" and answer["decision_threshold"] == 0.01
    assert "positive_probability" in answer["predictions"][0]


@pytest.mark.django_db
@pytest.mark.parametrize("value", [-0.1, 1.5, "high", None, True, [0.3], float("nan")])
def test_the_threshold_is_a_number_from_0_to_1(team, predictor, training_dir, value):
    _, users = team
    client = client_for(users["ada"])
    run_id = pets_run(client, training_dir)
    if isinstance(value, float) and math.isnan(value):
        response = client.post(
            f"/api/training/runs/{run_id}/decision-threshold/",
            '{"threshold": NaN}',
            content_type="application/json",
        )
        assert response.status_code == 400
    else:
        response = set_threshold(client, run_id, value)
        assert response.status_code == 400, response.data
        assert "from 0 to 1" in response.data["detail"]
    assert client.get(f"/api/training/runs/{run_id}/").json()["decision_threshold"] == 0.5


@pytest.mark.django_db
def test_the_ends_and_text_numbers_are_valid(team, predictor, training_dir):
    _, users = team
    client = client_for(users["ada"])
    run_id = pets_run(client, training_dir)
    for value, saved in ((0, 0.0), (1, 1.0), ("0.25", 0.25)):
        response = set_threshold(client, run_id, value)
        assert response.status_code == 200 and response.data["decision_threshold"] == saved


@pytest.mark.django_db
def test_only_binary_classifications_have_a_threshold(team, predictor, training_dir):
    from .test_predict import iris_run

    _, users = team
    client = client_for(users["ada"])
    run_id = iris_run(client, training_dir)  # three classes
    assert client.get(f"/api/training/runs/{run_id}/").json()["decision_threshold"] is None
    response = set_threshold(client, run_id, 0.3)
    assert response.status_code == 400 and "two classes" in response.data["detail"]


@pytest.mark.django_db
def test_only_who_started_the_run_may_set_the_threshold(team, predictor, training_dir):
    project, users = team
    run_id = pets_run(client_for(users["ada"]), training_dir, project)
    for name in ("bob", "cy"):  # editor and viewer of the project see the run
        client = client_for(users[name])
        assert client.get(f"/api/training/runs/{run_id}/").status_code == 200
        response = set_threshold(client, run_id, 0.2)
        assert response.status_code == 403, name
    assert set_threshold(client_for(users["eve"]), run_id, 0.2).status_code == 404
    assert set_threshold(client_for(users["ada"]), run_id, 0.2).status_code == 200
    # A read-only API token cannot either.
    from rest_framework.test import APIClient

    from brick_dnn_core.api_tokens import create_token

    _, key = create_token(users["ada"], "reader", read_only=True)
    reader = APIClient()
    reader.credentials(HTTP_AUTHORIZATION=f"Bearer {key}")
    assert set_threshold(reader, run_id, 0.3).status_code == 403
    _, key = create_token(users["ada"], "writer")
    writer = APIClient()
    writer.credentials(HTTP_AUTHORIZATION=f"Bearer {key}")
    assert set_threshold(writer, run_id, 0.3).status_code == 200


@pytest.mark.django_db
def test_mcp_shows_and_sets_the_threshold(team, predictor, training_dir):
    project, users = team
    run_id = pets_run(client_for(users["ada"]), training_dir, project)
    rng = np.random.default_rng(13)
    p, y = imbalanced(rng, 600, share=0.1)
    with_evaluation(training_dir, run_id, p, y)
    mcp = mcp_client(users["ada"])

    result, text = call(mcp, "get_training_run", id=run_id)
    assert "isError" not in result, text
    answer = result["structuredContent"]
    assert answer["decision_threshold"] == 0.5
    binary = answer["evaluation"]["binary"]
    assert binary["decision_threshold"] == 0.5 and binary["positive_class"] == "dogs"
    point = binary["at_decision_threshold"]["validation"]
    assert (point["tp"], point["fp"], point["tn"], point["fn"]) == brute_force(p, y, 0.5)
    assert answer["evaluation"]["validation"]["roc_auc"] > 0.5

    result, text = call(mcp, "set_decision_threshold", id=run_id, threshold=0.3)
    assert "isError" not in result, text
    answer = result["structuredContent"]
    assert answer["decision_threshold"] == 0.3
    point = answer["at_decision_threshold"]["validation"]
    assert point["threshold"] == 0.3
    assert (point["tp"], point["fp"], point["tn"], point["fn"]) == brute_force(p, y, 0.3)
    result, _ = call(mcp, "get_training_run", id=run_id)
    assert result["structuredContent"]["decision_threshold"] == 0.3

    result, text = call(mcp, "set_decision_threshold", id=run_id, threshold=2)
    assert result["isError"] is True and "from 0 to 1" in text
    result, text = call(
        mcp_client(users["bob"]), "set_decision_threshold", id=run_id, threshold=0.1
    )
    assert result["isError"] is True and "Only who started this run (ada)" in text
    # Not offered to read-only tokens.
    from .test_mcp import rpc

    tools = rpc(mcp_client(users["ada"], read_only=True), "tools/list").json()["result"]["tools"]
    assert "set_decision_threshold" not in {tool["name"] for tool in tools}
    tools = rpc(mcp, "tools/list").json()["result"]["tools"]
    assert "set_decision_threshold" in {tool["name"] for tool in tools}


@pytest.mark.django_db
def test_the_weights_download_names_the_threshold(team, predictor, training_dir):
    _, users = team
    client = client_for(users["ada"])
    run_id = pets_run(client, training_dir)
    set_threshold(client, run_id, 0.2)
    result, text = call(mcp_client(users["ada"]), "get_weights_download", id=run_id)
    assert "isError" not in result, text
    answer = result["structuredContent"]
    assert answer["decision_threshold"] == 0.2
    assert any("probabilities[:, 1] >= 0.2" in note for note in answer["notes"])


@pytest.mark.django_db
def test_rest_activity_of_a_token_names_the_threshold(team, predictor, training_dir):
    from rest_framework.test import APIClient

    from brick_dnn_core.agent_activity import describe_tool
    from brick_dnn_core.api_tokens import create_token

    _, users = team
    run_id = pets_run(client_for(users["ada"]), training_dir)
    _, key = create_token(users["ada"], "writer")
    writer = APIClient()
    writer.credentials(HTTP_AUTHORIZATION=f"Bearer {key}")
    assert set_threshold(writer, run_id, 0.25).status_code == 200
    described = describe_tool(
        users["ada"], "set_decision_threshold", {"id": run_id, "threshold": 0.25}, None
    )
    assert "set the decision threshold of" in described.phrase and "to 0.25" in described.phrase


# --------------------------------------------------------------------------- #
# Real trainings
# --------------------------------------------------------------------------- #


def test_neural_binary_run_gets_curves(tmp_path):
    pytest.importorskip("torch")
    from brickdnn.training import inspect_csv

    from .test_run_evaluation import run_neural
    from .test_training_engine import iris_like, mlp

    path = tmp_path / "spam.csv"
    path.write_text(iris_like(80, labels=("ham", "spam")))
    result, rec = run_neural(
        mlp(4, 2), inspect_csv(path), tmp_path, path, epochs=2, batch_size=8, test_split=0.2
    )
    assert result.status == "completed", result.message
    found = rec.evaluation
    assert found["binary"]["source"] == "val"
    for key in ("val", "test"):
        curve = found[key]["threshold_curve"]
        assert curve["positives"] + curve["negatives"] == found[key]["size"]
        assert 0 <= curve["roc_auc"] <= 1
    json.dumps(found, allow_nan=False)


def test_neural_binary_run_without_validation_rows(tmp_path):
    pytest.importorskip("torch")
    from brickdnn.training import inspect_csv

    from .test_run_evaluation import run_neural
    from .test_training_engine import iris_like, mlp

    path = tmp_path / "spam.csv"
    path.write_text(iris_like(40, labels=("ham", "spam")))
    result, rec = run_neural(
        mlp(4, 2), inspect_csv(path), tmp_path, path, epochs=1, batch_size=8, val_split=0.0
    )
    assert result.status == "completed", result.message
    found = rec.evaluation
    assert found["val"] is None and found["binary"]["source"] == "training"
    assert found["training"]["threshold_curve"]["positives"] == 20


def test_classical_binary_run_gets_curves(tmp_path):
    pytest.importorskip("sklearn")
    from brickdnn.training import inspect_csv

    from .test_classical_training import classification_csv, model_graph
    from .test_classical_training import run_job as run_classical_job
    from .test_run_evaluation import ClassicalRec

    path = classification_csv(tmp_path / "mail.csv", n=120, labels=("ham", "spam", "spam"))
    rec = ClassicalRec()
    result, _ = run_classical_job(
        model_graph("SVC"), inspect_csv(path), tmp_path, path, reporter=rec, test_split=0.2
    )
    assert result.status == "completed", result.message
    found = rec.evaluation
    assert found["class_names"] == ["ham", "spam"]
    assert found["binary"]["source"] == "val"
    assert found["val"]["threshold_curve"]["roc_auc"] is not None
    assert (
        found["val"]["threshold_curve"]["points"]["threshold"]
        == found["test"]["threshold_curve"]["points"]["threshold"]
    )
