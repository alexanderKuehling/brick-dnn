"""Explanations over REST and MCP: which methods a run offers, checks of the request, the
cache, permissions, limits and CPU accounting (with a fake explanation process), compact
MCP answers, and at the end the real process on trained runs.
"""

# ruff: noqa: F811 (the fixture training_dir is imported from test_training_api)

from __future__ import annotations

import fcntl
import json
import sys
import time

import pytest
from rest_framework.test import APIClient
from rest_framework.throttling import ScopedRateThrottle

from brick_dnn_core import explain_service, mcp_explain, training_service
from brick_dnn_core.models import UsageRecord
from brickdnn.training.state import read_state, write_state

from .helpers import chain
from .test_mcp import call, mcp_client, rpc
from .test_predict import (
    IRIS_PREPROCESSING,
    PETS_GRAPH,
    PETS_PREPROCESSING,
    client_for,
    iris_run,
    needs_torch,
    team,  # noqa: F401 (a fixture)
    trained_run,
)
from .test_training_api import (  # noqa: F401 (training_dir is a fixture)
    QUICK,
    image_zip,
    training_dir,
    upload,
    use_runner,
    wait_for,
)

# Behaves like the explanation process as far as the server can tell: it records the
# request and answers with the method (or fails, or takes too long).
FAKE_EXPLAINER = """
import json
import sys
import time
from pathlib import Path

mode, run_dir = sys.argv[1], Path(sys.argv[-1])
request = json.load(sys.stdin)
(run_dir / "explain-request.json").write_text(json.dumps(request))
if mode == "slow":
    time.sleep(30)
elif mode == "error":
    print(json.dumps({"error": "There are 60 validation rows: 0 to 59.", "cpu_seconds": 0.5}))
    sys.exit(0)
elif mode == "big":
    picture = "data:image/png;base64," + "A" * 30000
    grid = [[0.5] * 14 for _ in range(14)]
    print(json.dumps({
        "method": "image", "kind": "image", "image": picture, "size": [28, 28],
        "prediction": {"class": "7", "index": 7, "probability": 0.9, "top": []},
        "target": {"index": 7, "class": "7"},
        "saliency": {"png": picture, "grid": grid, "samples": 16, "noise": 0.15},
        "gradcam": {"png": picture, "grid": grid, "layer": {"id": "c2"},
                    "layers": [{"id": "c1", "name": "c1", "type": "Conv2d", "shape": [8, 28, 28]}]},
        "occlusion": {"png": picture, "grid": grid, "patch": 4, "stride": 2},
        "cpu_seconds": 1.5}))
    sys.exit(0)
print("some log line of PyTorch")
explain = request["explain"]
print(json.dumps({"method": explain["method"], "kind": "tabular", "echo": explain,
                  "cpu_seconds": 2.5}))
"""


@pytest.fixture
def explainer(settings, training_dir, tmp_path):
    script = tmp_path / "fake_explainer.py"
    script.write_text(FAKE_EXPLAINER)
    settings.TRAINING_PREDICT_COMMAND = [sys.executable, str(script), "echo"]
    settings.TRAINING_MAX_RUNS = 5
    return settings


def use_explainer(settings, mode: str) -> None:
    settings.TRAINING_PREDICT_COMMAND = [*settings.TRAINING_PREDICT_COMMAND[:2], mode]


def explain(client, run_id, **body):
    return client.post(f"/api/training/runs/{run_id}/explain/", body, format="json")


def recorded(training_dir, run_id) -> dict:
    return json.loads((training_dir / "runs" / run_id / "explain-request.json").read_text())


def methods_of(client, run_id) -> dict[str, dict]:
    response = client.get(f"/api/training/runs/{run_id}/explain/")
    assert response.status_code == 200, response.data
    return {m["id"]: m for m in response.data["methods"]}


@pytest.mark.django_db
def test_methods_depend_on_the_run(team, explainer, training_dir):
    _, users = team
    client = client_for(users["ada"])
    iris = iris_run(client, training_dir)
    methods = methods_of(client, iris)
    assert {k for k, m in methods.items() if m["available"]} == {
        "samples",
        "tabular",
        "permutation",
        "activations",
        "weights",
    }
    assert methods["boundary"]["reason"] == "For models with exactly 2 input features."
    assert methods["image"]["reason"] == "For image models only."

    pets = image_zip(["cats/1.png", "cats/2.png", "dogs/1.png", "dogs/2.png"])
    images = trained_run(client, training_dir, PETS_GRAPH, "pets.zip", pets, PETS_PREPROCESSING)
    methods = methods_of(client, images)
    assert methods["image"]["available"] and not methods["tabular"]["available"]

    houses = trained_run(
        client,
        training_dir,
        chain(("Input", {"shape": [2]}), ("Linear", {"out_features": 1})),
        "houses.csv",
        b"rooms,area,price\n" + b"".join(f"{i},{50 + i},{100 + i}\n".encode() for i in range(12)),
        {
            **IRIS_PREPROCESSING,
            "columns": ["rooms", "area"],
            "target": "price",
            "mean": [0, 0],
            "std": [1, 1],
            "target_mean": 0.0,
            "target_std": 1.0,
            "class_names": None,
        },
    )
    assert methods_of(client, houses)["boundary"]["available"]

    # Demo runs: only what needs no data.
    response = client.post(
        "/api/training/runs/",
        {
            "name": "Demo",
            "graph": chain(("Input", {"shape": [4]}), ("Linear", {"out_features": 3})),
            "dataset": {"kind": "random"},
            "config": QUICK,
        },
        format="json",
    )
    demo = response.data["id"]
    wait_for(client, demo)
    state = read_state(training_dir / "runs" / demo)
    write_state(training_dir / "runs" / demo, {**state, "preprocessing": {"kind": "demo"}})
    response = client.get(f"/api/training/runs/{demo}/explain/")
    assert response.data["demo"] is True
    available = {m["id"] for m in response.data["methods"] if m["available"]}
    assert available == {"activations", "weights"}


@pytest.mark.django_db
def test_explanations_are_computed_once_and_counted_as_predictions(team, explainer, training_dir):
    _, users = team
    client = client_for(users["ada"])
    run_id = iris_run(client, training_dir)

    response = explain(client, run_id, method="tabular", sample=3, target=1)
    assert response.status_code == 200, response.data
    assert response.data["cached"] is False and response.data["run"]["id"] == run_id
    request = recorded(training_dir, run_id)["explain"]
    assert request["method"] == "tabular" and request["sample"] == {"index": 3}
    assert request["target"] == 1 and request["preprocessing"] == IRIS_PREPROCESSING
    assert request["plan"]["targets"][0]["kind"] == "class"
    record = UsageRecord.objects.get(kind="prediction")
    assert record.quantity == pytest.approx(2.5) and record.run_id is not None

    # The same question again: from the cache, no process, no compute.
    (training_dir / "runs" / run_id / "explain-request.json").unlink()
    again = explain(client, run_id, method="tabular", sample=3, target=1)
    assert again.status_code == 200 and again.data["cached"] is True
    assert again.data["echo"] == response.data["echo"]
    assert not (training_dir / "runs" / run_id / "explain-request.json").exists()
    assert UsageRecord.objects.count() == 1

    # An input of one's own is prepared like for a prediction (standardised).
    response = explain(client, run_id, method="tabular", input=[6.0, 3.5, 1.5, 0.3])
    assert response.status_code == 200
    tensor = recorded(training_dir, run_id)["explain"]["sample"]["tensor"]
    assert tensor["shape"] == [1, 4]
    assert tensor["values"] == pytest.approx([1.0, 0.0, 1.0, 1.0])

    # Methods without a sample, and the cache stays bounded.
    assert explain(client, run_id, method="permutation").status_code == 200
    assert "sample" not in recorded(training_dir, run_id)["explain"]
    cache = training_dir / "runs" / run_id / "explain" / "cache"
    assert len(list(cache.glob("*.json"))) == 3


@pytest.mark.django_db
def test_bad_requests(team, explainer, training_dir, monkeypatch):
    _, users = team
    client = client_for(users["ada"])
    run_id = iris_run(client, training_dir)
    response = explain(client, run_id, method="magic")
    assert response.status_code == 400 and "'method' must be one of" in response.data["detail"]
    response = explain(client, run_id, method="tabular")
    assert response.status_code == 400 and "Pick a sample" in response.data["detail"]
    response = explain(client, run_id, method="tabular", input=[1, 2])
    assert response.status_code == 400 and "has 2 values" in response.data["detail"]
    assert "sepal_length" in response.data["inputs"]
    response = explain(client, run_id, method="image", sample=0)
    assert response.status_code == 409 and response.data["detail"] == "For image models only."
    response = explain(client, run_id, method="tabular", sample=-1)
    assert response.status_code == 400
    response = explain(client, run_id, method="image", sample=0, layer=["x"])
    assert response.status_code == 409  # the method is checked first

    use_explainer(explainer, "error")
    response = explain(client, run_id, method="tabular", sample=99)
    assert response.status_code == 500 and "60 validation rows" in response.data["detail"]
    assert UsageRecord.objects.get(kind="prediction").quantity == pytest.approx(0.5)

    use_explainer(explainer, "slow")
    monkeypatch.setattr(explain_service, "TIMEOUT_SECONDS", 1)
    started = time.monotonic()
    response = explain(client, run_id, method="tabular", sample=1)
    assert response.status_code == 500 and "longer than 1 seconds" in response.data["detail"]
    assert time.monotonic() - started < 10


@pytest.mark.django_db
def test_who_may_explain(team, explainer, training_dir):
    project, users = team
    run_id = iris_run(client_for(users["bob"]), training_dir, project)
    for index, name in enumerate(("ada", "bob", "cy")):  # the owner, editor and viewer
        response = explain(client_for(users[name]), run_id, method="tabular", sample=index)
        assert response.status_code == 200, (name, response.data)
    assert explain(client_for(users["eve"]), run_id, method="permutation").status_code == 404
    assert client_for(users["eve"]).get(f"/api/training/runs/{run_id}/explain/").status_code == 404
    assert explain(APIClient(), run_id, method="permutation").status_code == 401
    assert explain(client_for(users["ada"]), "not-a-uuid", method="weights").status_code == 404
    # Each one counts for who asked (the cache is shared: answers are the same for all).
    assert UsageRecord.objects.filter(kind="prediction", user=users["cy"]).count() == 1

    result, text = call(
        mcp_client(users["eve"]), "explain_prediction", run_id=run_id, method="samples"
    )
    assert result["isError"] is True and "No such training run" in text
    read_only = mcp_client(users["cy"], read_only=True)
    tools = {t["name"]: t for t in rpc(read_only, "tools/list").json()["result"]["tools"]}
    assert tools["explain_prediction"]["annotations"]["readOnlyHint"] is True
    assert "get_layer_activations" in tools


@pytest.mark.django_db
def test_explanations_are_limited_like_predictions(team, explainer, training_dir, monkeypatch):
    _, users = team
    client = client_for(users["ada"])
    run_id = iris_run(client, training_dir)
    lock = open(training_dir / ".predict-0.lock", "a")  # noqa: SIM115
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        response = explain(client, run_id, method="permutation")
        assert response.status_code == 503 and response["Retry-After"] == "5"
    finally:
        lock.close()

    # Rate limit (the busy request above counted): only requests that start a process count.
    monkeypatch.setitem(ScopedRateThrottle.THROTTLE_RATES, "predict", "3/hour")
    assert explain(client, run_id, method="permutation").status_code == 200
    assert explain(client, run_id, method="samples").status_code == 200
    # Cached answers do not count against the rate limit.
    assert explain(client, run_id, method="permutation").status_code == 200
    assert explain(client, run_id, method="tabular", sample=0).status_code == 429
    result, text = call(
        mcp_client(users["ada"]), "explain_prediction", run_id=run_id, method="tabular", sample=1
    )
    assert result["isError"] is True and "Too many" in text


@pytest.mark.django_db
def test_mcp_answers_are_compact(team, explainer, training_dir):
    _, users = team
    client = client_for(users["ada"])
    run_id = iris_run(client, training_dir)
    mcp = mcp_client(users["ada"])
    result, text = call(mcp, "explain_prediction", run_id=run_id, method="tabular", sample=2)
    assert "isError" not in result, text
    assert result["structuredContent"]["echo"]["sample"] == {"index": 2}

    use_explainer(explainer, "big")
    result, text = call(mcp, "explain_prediction", run_id=run_id, method="permutation")
    answer = result["structuredContent"]
    assert "data:image" not in text and len(text) < 12_000  # three 14×14 grids
    assert len(answer["saliency"]["grid"]) == 14 and answer["gradcam"]["layers"] == ["c1"]
    # With images, only while they are small enough: these are not.
    result, text = call(mcp, "explain_prediction", run_id=run_id, method="weights")
    assert result["isError"] is True  # not an explanation method of the tool
    result, text = call(mcp, "explain_prediction", run_id=run_id, method="samples", images=True)
    assert "data:image" not in text and "left out" in text

    small = mcp_explain.compact_explanation(
        {"method": "image", "image": "data:image/png;base64,AAAA", "saliency": {"grid": [[1]]}},
        images=True,
    )
    assert small["image"].startswith("data:image/png")

    activations = {
        "method": "activations",
        "kind": "image",
        "sample": {"source": "validation", "index": 0},
        "nodes": [
            {
                "id": f"n{i}",
                "type": "Conv2d",
                "name": f"conv {i}",
                "shape": [64, 28, 28],
                "stats": {"mean": 0.1, "std": 1.0, "min": -3.0, "max": 3.0, "zeros": 0.4},
                "histogram": {"edges": [0.0] * 25, "counts": [1] * 24, "count": 24},
                "view": {
                    "kind": "maps",
                    "png": "data:image/png;base64," + "A" * 8000,
                    "channels": 64,
                    "shown": 16,
                    "channel_means": [0.123] * 16,
                },
            }
            for i in range(30)
        ]
        + [
            {
                "id": "fc",
                "type": "Linear",
                "name": "fc",
                "shape": [512],
                "stats": {},
                "histogram": {},
                "view": {"kind": "vector", "values": [0.5] * 256, "total": 512},
            }
        ],
    }
    compact = mcp_explain.compact_activations(activations)
    text = json.dumps(compact)
    assert len(text) < 12_000 and "data:image" not in text
    assert len(compact["nodes"][-1]["values"]) == 32
    assert len(json.dumps(mcp_explain.compact_activations(activations, images=True))) < 20_000


# --------------------------------------------------------------------------- #
# For real: trained runs and the real process
# --------------------------------------------------------------------------- #


def moons_csv(rows: int = 160) -> bytes:
    import math
    import random

    generator = random.Random(3)
    lines = ["x1,x2,label"]
    for i in range(rows):
        t = generator.uniform(0, math.pi)
        x, y = (math.cos(t), math.sin(t)) if i % 2 == 0 else (1 - math.cos(t), 0.5 - math.sin(t))
        lines.append(
            f"{x + generator.gauss(0, 0.1):.3f},{y + generator.gauss(0, 0.1):.3f},{'ab'[i % 2]}"
        )
    return ("\n".join(lines) + "\n").encode()


@needs_torch
@pytest.mark.django_db
def test_real_explanations_of_trained_runs(team, training_dir, settings):
    settings.TRAINING_RUNNER_COMMAND = None
    settings.TRAINING_PREDICT_COMMAND = None
    _, users = team
    client = client_for(users["ada"])
    dataset = upload(client, "moons.csv", moons_csv()).data["id"]

    def train(graph, config):
        response = client.post(
            "/api/training/runs/",
            {
                "name": "Moons",
                "graph": graph,
                "dataset": {"kind": "upload", "id": dataset},
                "config": config,
            },
            format="json",
        )
        assert response.status_code == 201, response.data
        data = wait_for(client, response.data["id"], timeout=180, interval=0.5)
        assert data["status"] == "completed", data["message"]
        return data["id"]

    svm = train(chain(("Input", {"shape": [2]}), ("SVC", {})), {"seed": 2})
    started = time.monotonic()
    response = explain(client, svm, method="boundary")
    assert response.status_code == 200, response.data
    assert response.data["classes"] == ["a", "b"] and len(response.data["grid"]) == 16
    assert time.monotonic() - started < training_service.PREDICT_TIMEOUT_SECONDS
    response = explain(client, svm, method="samples")
    assert response.status_code == 200 and response.data["total"] > 10
    assert explain(client, svm, method="weights").status_code == 409

    net = train(
        chain(
            ("Input", {"shape": [2]}),
            ("Linear", {"out_features": 16}),
            ("Activation", {"function": "relu"}),
            ("Linear", {"out_features": 2}),
        ),
        {**QUICK, "max_samples": None, "epochs": 3},  # the seed is chosen by the run
    )
    for body in (
        {"method": "tabular", "sample": 0},
        {"method": "permutation"},
        {"method": "boundary"},
        {"method": "activations", "sample": 1},
        {"method": "weights"},
    ):
        response = explain(client, net, **body)
        assert response.status_code == 200, (body, response.data)
    mcp = mcp_client(users["ada"])
    result, text = call(mcp, "explain_prediction", run_id=net, method="tabular", input=[0.5, 0.2])
    assert "isError" not in result, text
    features = result["structuredContent"]["features"]
    assert {f["name"] for f in features} == {"x1", "x2"}
    result, text = call(mcp, "get_layer_activations", run_id=net, sample=0)
    assert "isError" not in result, text
    assert [n["shape"] for n in result["structuredContent"]["nodes"]] == [[2], [16], [16], [2]]
    assert len(text) < 4000
