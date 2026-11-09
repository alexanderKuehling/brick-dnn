"""Predictions with trained weights: inputs, outputs, the REST endpoint and the MCP tool.

Most tests use a fake prediction process (like the fake runner of the training tests): it
records what it was given and returns the first values of every sample as the network's
outputs, so the tests need no PyTorch. The tests at the end train and predict for real.
"""

# ruff: noqa: F811 (the fixture training_dir is imported from test_training_api)

from __future__ import annotations

import fcntl
import json
import subprocess
import sys
import time
from urllib.parse import urlsplit

import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient
from rest_framework.throttling import ScopedRateThrottle

from brick_dnn_core import training_service
from brick_dnn_core.api_tokens import create_token
from brick_dnn_core.models import Project, ProjectMember
from brickdnn.training import inference, torch_installed
from brickdnn.training.state import read_state, write_state

from .helpers import chain
from .test_mcp import call, mcp_client, rpc
from .test_training_api import (  # noqa: F401 (training_dir is a fixture)
    QUICK,
    image_zip,
    iris_csv,
    training_dir,
    upload,
    use_runner,
    wait_for,
)

IRIS_GRAPH = chain(
    ("Input", {"shape": [4]}),
    ("Linear", {"out_features": 8}),
    ("Activation", {"function": "relu"}),
    ("Linear", {"out_features": 3}),
)
HOUSES_GRAPH = chain(("Input", {"shape": [2]}), ("Linear", {"out_features": 1}))
PETS_GRAPH = chain(
    ("Input", {"shape": [1, 4, 4]}), ("Flatten", {}), ("Linear", {"out_features": 2})
)
COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
SPECIES = ["setosa", "versicolor", "virginica"]
IRIS_PREPROCESSING = {
    "kind": "tabular",
    "columns": COLUMNS,
    "target": "species",
    "delimiter": ",",
    "decimal_comma": False,
    "mean": [5.5, 3.5, 1.3, 0.2],
    "std": [0.5, 0.25, 0.2, 0.1],
    "target_mean": None,
    "target_std": None,
    "class_names": SPECIES,
}
HOUSES_PREPROCESSING = {
    "kind": "tabular",
    "columns": ["rooms", "area"],
    "target": "price",
    "mean": [3.0, 80.0],
    "std": [1.0, 20.0],
    "target_mean": 250.0,
    "target_std": 50.0,
    "class_names": None,
}
PETS_PREPROCESSING = {
    "kind": "image",
    "channels": 1,
    "height": 4,
    "width": 4,
    "layout": "chw",
    "mean": [0.5],
    "std": [0.25],
    "class_names": ["cats", "dogs"],
}

# Behaves like brickdnn.training.predict as far as the API can tell. The mode is the first
# argument, the run directory the last one.
FAKE_PREDICTOR = """
import json
import sys
import time
from pathlib import Path

mode, run_dir = sys.argv[1], Path(sys.argv[-1])
request = json.load(sys.stdin)
(run_dir / "predict-request.json").write_text(json.dumps(request))
if mode == "slow":
    time.sleep(30)
elif mode == "crash":
    print("RuntimeError: out of cheese", file=sys.stderr)
    sys.exit(1)
elif mode == "error":
    print(json.dumps({"error": "The trained weights do not fit the network of this run."}))
    sys.exit(0)
elif mode == "nan":
    print(json.dumps({"outputs": [[[None, 1.0, 2.0]]]}))
    sys.exit(0)
tensor = request["inputs"][0]
count = tensor["shape"][0]
size = len(tensor["values"]) // count
rows = [tensor["values"][i * size:(i + 1) * size] for i in range(count)]
width = int(mode.split(":")[1])  # echo:<outputs per sample>
print("log output of PyTorch")
print(json.dumps({"outputs": [[row[:width] for row in rows]]}))
"""


# --------------------------------------------------------------------------- #
# Inputs and outputs (no database, no processes)
# --------------------------------------------------------------------------- #


def plan_for(task: str, **extra) -> dict:
    plans = {
        "iris": {
            "task": "classification",
            "inputs": [{"shape": [4], "dtype": "float", "vocab": None}],
            "targets": [{"kind": "class", "num_classes": 3}],
            "class_names": SPECIES,
        },
        "houses": {
            "task": "regression",
            "inputs": [{"shape": [2], "dtype": "float", "vocab": None}],
            "targets": [{"kind": "value", "shape": [1]}],
        },
        "pets": {
            "task": "classification",
            "inputs": [{"shape": [1, 4, 4], "dtype": "float", "vocab": None}],
            "targets": [{"kind": "class", "num_classes": 2}],
            "image": {"channels": 1, "height": 4, "width": 4, "layout": "chw"},
            "class_names": ["cats", "dogs"],
        },
    }
    return {**plans[task], **extra}


def values_of(prepared: inference.PreparedInputs) -> list[float]:
    return [round(v, 6) for v in prepared.tensors[0]["values"]]


def test_tabular_inputs_are_standardised_like_in_training():
    plan = plan_for("iris")
    row = {"sepal_length": 6.0, "sepal_width": "3.0", "petal_length": 1.3, "petal_width": 0.3}
    prepared = inference.prepare_inputs(plan, IRIS_PREPROCESSING, [row, [5.0, 3.5, 1.5, 0.2]])
    assert prepared.kind == "tabular" and prepared.count == 2 and prepared.top_k == 3
    assert prepared.tensors[0]["shape"] == [2, 4] and prepared.tensors[0]["dtype"] == "float"
    assert values_of(prepared) == [1.0, -2.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0]

    # A single row may be given without the list; the target column is ignored.
    single = inference.prepare_inputs(plan, IRIS_PREPROCESSING, {**row, "species": "?"})
    assert single.count == 1 and values_of(single) == [1.0, -2.0, 0.0, 1.0]
    assert inference.prepare_inputs(plan, IRIS_PREPROCESSING, [5.5, 3.5, 1.3, 0.2]).count == 1
    # Already standardised values are used as they are.
    normalized = inference.prepare_inputs(
        plan, IRIS_PREPROCESSING, [[1, 2, 3, 4]], "normalized", top_k=1
    )
    assert values_of(normalized) == [1, 2, 3, 4] and normalized.top_k == 1
    # Decimal commas like in the CSV file.
    comma = {**IRIS_PREPROCESSING, "decimal_comma": True}
    assert values_of(inference.prepare_inputs(plan, comma, [["6,0", 3.5, 1.3, 0.2]]))[0] == 1.0


@pytest.mark.parametrize(
    ("inputs", "options", "message"),
    [
        ([], {}, "'inputs' must be a list of samples"),
        ("5,3,1,0", {}, "'inputs' must be a list of samples"),
        ([[1, 2, 3]], {}, "has 3 values, but the network takes 4: sepal_length"),
        ([{"sepal_length": 1}], {}, "the column 'sepal_width' is missing (and 2 more)"),
        ([{c: 1 for c in [*COLUMNS, "colour"]}], {}, "unknown column 'colour'"),
        ([[1, 2, "three", 4]], {}, "'petal_length': 'three' is not a number"),
        ([[1, 2, float("nan"), 4]], {}, "is not a number"),
        ([[1, 2, True, 4]], {}, "is not a number"),
        ([[1, 2, 3, 4], "x"], {}, "Sample 2 must be an object"),
        ([[1, 2, 3, 4]] * 65, {}, "65 samples are too many"),
        ([[1, 2, 3, 4]], {"input_scale": "unit"}, "'unit' is for image models"),
        ([[1, 2, 3, 4]], {"input_scale": "pixels"}, "input_scale must be one of"),
        ([[1, 2, 3, 4]], {"top_k": 11}, "top_k must be a whole number from 1 to 10"),
        ([[1, 2, 3, 4]], {"top_k": "3"}, "top_k must be a whole number"),
    ],
)
def test_bad_tabular_inputs(inputs, options, message):
    with pytest.raises(inference.InvalidInputs) as error:
        inference.prepare_inputs(plan_for("iris"), IRIS_PREPROCESSING, inputs, **options)
    assert message in str(error.value)


def test_image_inputs_are_normalised_like_in_training():
    plan = plan_for("pets")
    image = [[0, 255, 127.5, 255]] * 4  # 4×4 pixels
    raw = inference.prepare_inputs(plan, PETS_PREPROCESSING, [[image]], "raw")
    assert raw.count == 1 and raw.tensors[0]["shape"] == [1, 1, 4, 4]
    assert values_of(raw)[:4] == [-2.0, 2.0, 0.0, 2.0]  # (pixel / 255 - 0.5) / 0.25
    unit = inference.prepare_inputs(plan, PETS_PREPROCESSING, [[[[0, 1, 0.5, 1]] * 4]], "unit")
    assert values_of(unit) == values_of(raw)
    normalized = inference.prepare_inputs(plan, PETS_PREPROCESSING, [[image]], "normalized")
    assert values_of(normalized)[:4] == [0, 255, 127.5, 255]

    # A grayscale image may leave out its channel axis, also as a single sample.
    for inputs in ([image, image], image, [[image]], [image]):
        prepared = inference.prepare_inputs(plan, PETS_PREPROCESSING, inputs, "raw")
        assert values_of(prepared)[:4] == [-2.0, 2.0, 0.0, 2.0]
    assert inference.prepare_inputs(plan, PETS_PREPROCESSING, [image, image], "raw").count == 2

    # RGB images are normalised per channel; flat layouts keep the channel order.
    rgb = {**PETS_PREPROCESSING, "channels": 3, "layout": "flat"}
    rgb.update(mean=[0.0, 0.5, 1.0], std=[1.0, 0.5, 0.25], height=1, width=2)
    rgb_plan = plan_for("pets", inputs=[{"shape": [6], "dtype": "float"}])
    prepared = inference.prepare_inputs(rgb_plan, rgb, [[0, 1, 0, 1, 0, 1]], "unit")
    assert values_of(prepared) == [0.0, 1.0, -1.0, 1.0, -4.0, 0.0]
    as_image = inference.prepare_inputs(rgb_plan, rgb, [[[[0, 1]], [[0, 1]], [[0, 1]]]], "unit")
    assert as_image.tensors[0]["shape"] == [1, 6] and values_of(as_image) == values_of(prepared)

    # Networks that take a flattened image (e.g. an MLP on MNIST) or rows of pixels also take
    # the image itself.
    flat = [v for row in image for v in row]
    for layout, shape, samples in (("flat", [16], [flat]), ("hw", [4, 4], [image])):
        other = plan_for("pets", inputs=[{"shape": shape, "dtype": "float"}])
        preprocessing = {**PETS_PREPROCESSING, "layout": layout}
        for inputs in ([image, image], [[image]], samples, [*samples, image]):
            prepared = inference.prepare_inputs(other, preprocessing, inputs, "raw")
            assert prepared.tensors[0]["shape"][1:] == shape
            assert values_of(prepared)[:4] == [-2.0, 2.0, 0.0, 2.0]


@pytest.mark.parametrize(
    ("inputs", "scale", "message"),
    [
        ([[[[0] * 4] * 4]], None, "Say how the pixel values are given with input_scale"),
        (
            [[[[0] * 4] * 3]],
            "raw",
            "has the shape 1×3×4, but the network takes a grayscale image as a nested list of 1×4×4 or 4×4",
        ),
        ([[[0] * 4] * 5], "raw", "has the shape 5×4, but the network takes a grayscale image"),
        ([[[[300] * 4] * 4]], "raw", "300 is not a pixel value between 0 and 255"),
        ([[[[2] * 4] * 4]], "unit", "Use input_scale 'raw' for values up to 255"),
        ([[[["x"] * 4] * 4]], "raw", "'x' is not a number"),
    ],
)
def test_bad_image_inputs(inputs, scale, message):
    with pytest.raises(inference.InvalidInputs) as error:
        inference.prepare_inputs(plan_for("pets"), PETS_PREPROCESSING, inputs, scale)
    assert message in str(error.value)


def test_input_size_is_limited():
    big = {**PETS_PREPROCESSING, "height": 100, "width": 100}
    plan = plan_for("pets", inputs=[{"shape": [1, 100, 100], "dtype": "float"}])
    image = [[[0] * 100] * 100]
    assert inference.prepare_inputs(plan, big, [image] * 25, "raw").count == 25
    with pytest.raises(inference.InvalidInputs, match="at most 250,000 numbers"):
        inference.prepare_inputs(plan, big, [image] * 26, "raw")


def test_demo_inputs():
    plan = {
        "task": "demo",
        "inputs": [{"shape": [3], "dtype": "long", "vocab": 10}],
        "targets": [{"kind": "class", "num_classes": 2}],
    }
    prepared = inference.prepare_inputs(plan, {"kind": "demo"}, [[1, 2, 9]])
    assert prepared.tensors[0] == {"dtype": "long", "shape": [1, 3], "values": [1, 2, 9]}
    with pytest.raises(inference.InvalidInputs, match="whole numbers from 0 to 9"):
        inference.prepare_inputs(plan, {"kind": "demo"}, [[1, 2, 10]])
    two = {**plan, "inputs": [*plan["inputs"], *plan["inputs"]]}
    with pytest.raises(inference.InvalidInputs, match="2 Input layers"):
        inference.prepare_inputs(two, {"kind": "demo"}, [[1, 2, 3]])


@pytest.mark.parametrize(
    ("plan", "preprocessing"),
    [
        (plan_for("iris"), None),
        (plan_for("iris"), {"kind": "unknown"}),
        (plan_for("iris"), {**IRIS_PREPROCESSING, "mean": [0, 0]}),
        (plan_for("iris", inputs=[{"shape": [5]}]), IRIS_PREPROCESSING),
        (plan_for("houses"), {**HOUSES_PREPROCESSING, "target_std": None}),
        (plan_for("pets"), {**PETS_PREPROCESSING, "layout": "hwc"}),
        (plan_for("pets"), {**PETS_PREPROCESSING, "height": 5}),  # not the Input shape
        (plan_for("iris"), {**IRIS_PREPROCESSING, "std": [1, 0, 1, 1]}),
        ({}, {"kind": "demo"}),
    ],
)
def test_runs_without_known_preprocessing_cannot_predict(plan, preprocessing):
    with pytest.raises(inference.UnknownPreprocessing, match="not known"):
        inference.prepare_inputs(plan, preprocessing, [[1, 2, 3, 4]])


def test_reading_outputs():
    plan = plan_for("iris")
    outputs = [[[2.0, 1.0, 0.0], [0.0, 0.0, 5.0]]]
    first, second = inference.read_outputs(plan, IRIS_PREPROCESSING, outputs, 2, top_k=2)
    assert first["class"] == "setosa" and first["index"] == 0
    assert first["probability"] == pytest.approx(0.6652, abs=1e-4)
    assert [t["class"] for t in first["top"]] == ["setosa", "versicolor"]
    assert second["class"] == "virginica" and len(second["top"]) == 2
    nan = inference.read_outputs(plan, IRIS_PREPROCESSING, [[[None, 1.0, 2.0]]], 1)
    assert nan == [{"class": None, "index": None, "probability": None, "top": []}]
    with pytest.raises(ValueError, match="number of classes"):
        inference.read_outputs(plan, IRIS_PREPROCESSING, [[[1.0, 2.0]]], 1)
    with pytest.raises(ValueError, match="number of samples"):
        inference.read_outputs(plan, IRIS_PREPROCESSING, outputs, 3)

    # Regression: back in the units of the target column.
    values = inference.read_outputs(plan_for("houses"), HOUSES_PREPROCESSING, [[[1.0], [-0.5]]], 2)
    assert values == [{"value": 300.0}, {"value": 225.0}]

    # Demo runs with several outputs.
    demo = {
        "task": "demo",
        "inputs": [{"shape": [2], "dtype": "float"}],
        "targets": [{"kind": "class", "num_classes": 2}, {"kind": "value", "shape": [1]}],
    }
    (result,) = inference.read_outputs(demo, {"kind": "demo"}, [[[0.0, 0.0]], [[1.5]]], 1)
    assert result["outputs"][0]["class"] == "class 0"
    assert result["outputs"][0]["probability"] == 0.5
    assert result["outputs"][1] == {"value": 1.5}


def test_usage_snippets_are_python():
    cases = [
        (plan_for("iris"), IRIS_PREPROCESSING, "CLASS_NAMES = ['setosa'"),
        (plan_for("houses"), HOUSES_PREPROCESSING, "* 50 + 250).tolist())  # predicted price"),
        (plan_for("pets"), PETS_PREPROCESSING, 'convert("L")'),
        (
            plan_for("pets", inputs=[{"shape": [16]}]),
            {**PETS_PREPROCESSING, "layout": "flat"},
            "flatten(1)",
        ),
    ]
    for plan, preprocessing, expected in cases:
        snippet = inference.usage_snippet(plan, preprocessing, "my_net", "my-net.pt")
        compile(snippet, "<snippet>", "exec")
        assert expected in snippet, snippet
    assert inference.module_name("Iris net (v2)") == "iris_net_v2"
    assert inference.module_name("2nd try") == "model_2nd_try"
    assert inference.module_name("class") == "model" and inference.module_name("Ä") == "model"
    assert "feature columns of the CSV file in this order: sepal_length" in (
        inference.describe_inputs(plan_for("iris"), IRIS_PREPROCESSING)
    )


# --------------------------------------------------------------------------- #
# The REST endpoint and the MCP tool, with a fake prediction process
# --------------------------------------------------------------------------- #


@pytest.fixture
def predictor(settings, training_dir, tmp_path):
    script = tmp_path / "fake_predictor.py"
    script.write_text(FAKE_PREDICTOR)
    settings.TRAINING_PREDICT_COMMAND = [sys.executable, str(script), "echo:3"]
    settings.TRAINING_MAX_RUNS = 5
    return settings


def use_predictor(settings, mode: str) -> None:
    settings.TRAINING_PREDICT_COMMAND = [*settings.TRAINING_PREDICT_COMMAND[:2], mode]


@pytest.fixture
def team(db):
    users = {
        name: get_user_model().objects.create_user(name, password="correct-horse-1")
        for name in ("ada", "bob", "cy", "eve")
    }
    project = Project.objects.create(owner=users["ada"], name="Shared", graph=IRIS_GRAPH)
    ProjectMember.objects.create(project=project, user=users["bob"], role="editor")
    ProjectMember.objects.create(project=project, user=users["cy"], role="viewer")
    return project, users


def client_for(user) -> APIClient:
    client = APIClient()
    client.force_authenticate(user)
    return client


def trained_run(client, training_dir, graph, filename, content, preprocessing, project=None):
    """A completed run (fake runner) on an uploaded dataset, with its preprocessing."""
    dataset = upload(client, filename, content)
    assert dataset.status_code == 201, dataset.data
    body = {
        "name": "Net",
        "graph": graph,
        "dataset": {"kind": "upload", "id": dataset.data["id"]},
        "config": QUICK,
    }
    if project is not None:
        body["project"] = project.pk
    response = client.post("/api/training/runs/", body, format="json")
    assert response.status_code == 201, response.data
    run_id = response.data["id"]
    assert wait_for(client, run_id)["status"] == "completed"
    if preprocessing is not None:
        directory = training_dir / "runs" / run_id
        state = read_state(directory)
        state["preprocessing"] = preprocessing
        write_state(directory, state)
    return run_id


def iris_run(client, training_dir, project=None, preprocessing=IRIS_PREPROCESSING):
    return trained_run(
        client, training_dir, IRIS_GRAPH, "iris.csv", iris_csv(), preprocessing, project
    )


def houses_csv() -> bytes:
    rows = ["rooms,area,price"] + [
        f"{1 + i % 5},{50 + 7 * i},{120.5 + 13.25 * i}" for i in range(12)
    ]
    return ("\n".join(rows) + "\n").encode()


def recorded_request(training_dir, run_id) -> dict:
    return json.loads((training_dir / "runs" / run_id / "predict-request.json").read_text())


def predict(client, run_id, **body):
    return client.post(f"/api/training/runs/{run_id}/predict/", body, format="json")


def mcp_predict(client, run_id, **arguments):
    result, text = call(client, "predict", run_id=run_id, **arguments)
    return result, text


@pytest.mark.django_db
def test_predict_tabular_classification_over_mcp(team, predictor, training_dir):
    _, users = team
    run_id = iris_run(client_for(users["ada"]), training_dir)
    mcp = mcp_client(users["ada"])
    rows = [
        {"sepal_length": 5.5, "sepal_width": 3.5, "petal_length": 1.7, "petal_width": 0.2},
        [5.5, 3.5, 1.3, 0.4],
    ]
    result, text = mcp_predict(mcp, run_id, inputs=rows, top_k=2)
    assert "isError" not in result, text
    answer = result["structuredContent"]
    assert answer["run"] == {"id": run_id, "name": "Net"} and answer["task"] == "classification"
    first, second = answer["predictions"]
    # The fake network echoes the first three standardised features as class scores.
    assert first["class"] == "virginica" and second["class"] == "setosa"
    assert (
        len(first["top"]) == 2 and first["top"][0]["probability"] > first["top"][1]["probability"]
    )
    assert first["probability"] == first["top"][0]["probability"]
    request = recorded_request(training_dir, run_id)
    (tensor,) = request["inputs"]
    assert tensor["dtype"] == "float" and tensor["shape"] == [2, 4]
    assert tensor["values"] == pytest.approx([0, 0, 2, 0, 0, 0, 0, 2])


@pytest.mark.django_db
def test_predict_regression_over_rest(team, predictor, training_dir):
    _, users = team
    client = client_for(users["ada"])
    use_predictor(predictor, "echo:1")
    run_id = trained_run(
        client, training_dir, HOUSES_GRAPH, "houses.csv", houses_csv(), HOUSES_PREPROCESSING
    )
    response = predict(client, run_id, inputs=[{"rooms": 4, "area": 100}, [2, 60]])
    assert response.status_code == 200, response.data
    assert response.data["task"] == "regression" and response.data["target"] == "price"
    # Standardised rooms (1.0 and -1.0) come back as values, in the units of the price.
    assert response.data["predictions"] == [{"value": 300.0}, {"value": 200.0}]


@pytest.mark.django_db
def test_predict_images(team, predictor, training_dir):
    _, users = team
    client = client_for(users["ada"])
    use_predictor(predictor, "echo:2")
    pets = image_zip(["cats/1.png", "cats/2.png", "dogs/1.png", "dogs/2.png"])
    run_id = trained_run(client, training_dir, PETS_GRAPH, "pets.zip", pets, PETS_PREPROCESSING)
    mcp = mcp_client(users["ada"])
    image = [[255, 0, 0, 0]] + [[0] * 4] * 3

    # Without input_scale the assistant learns what to send.
    result, text = mcp_predict(mcp, run_id, inputs=[image])
    assert result["isError"] is True
    assert "input_scale" in text and "Expected inputs: One grayscale image of 4×4 pixels" in text

    result, text = mcp_predict(mcp, run_id, inputs=[image], input_scale="raw")
    assert "isError" not in result, text
    (prediction,) = result["structuredContent"]["predictions"]
    assert prediction["class"] == "cats" and [t["class"] for t in prediction["top"]] == [
        "cats",
        "dogs",
    ]
    values = recorded_request(training_dir, run_id)["inputs"][0]["values"]
    assert values[:2] == [2.0, -2.0] and len(values) == 16

    response = predict(client, run_id, inputs=[[[1, 2]]], input_scale="raw")
    assert response.status_code == 400
    assert (
        "network takes a grayscale image as a nested list of 1×4×4 or 4×4"
        in response.data["detail"]
    )
    assert response.data["inputs"].startswith("One grayscale image")


@pytest.mark.django_db
def test_who_may_predict(team, predictor, training_dir):
    project, users = team
    run_id = iris_run(client_for(users["bob"]), training_dir, project)
    row = [[5.5, 3.5, 1.3, 0.2]]
    # Everyone who sees the run, viewers too; strangers don't see it.
    for name in ("ada", "bob", "cy"):
        result, text = mcp_predict(mcp_client(users[name]), run_id, inputs=row)
        assert "isError" not in result, (name, text)
    result, text = mcp_predict(mcp_client(users["eve"]), run_id, inputs=row)
    assert result["isError"] is True and "No such training run" in text
    assert predict(client_for(users["eve"]), run_id, inputs=row).status_code == 404
    assert predict(APIClient(), run_id, inputs=row).status_code == 401

    # Predicting changes nothing: read-only tokens may, over MCP and REST.
    read_only = mcp_client(users["cy"], read_only=True)
    tools = {t["name"]: t for t in rpc(read_only, "tools/list").json()["result"]["tools"]}
    assert tools["predict"]["annotations"]["readOnlyHint"] is True
    result, _ = mcp_predict(read_only, run_id, inputs=row)
    assert "isError" not in result
    _, key = create_token(users["cy"], "script", read_only=True)
    rest = APIClient()
    rest.credentials(HTTP_AUTHORIZATION=f"Bearer {key}")
    assert predict(rest, run_id, inputs=row).status_code == 200
    assert rest.post(f"/api/training/runs/{run_id}/stop/").status_code == 403


@pytest.mark.django_db
def test_runs_that_cannot_predict(team, predictor, training_dir, settings, monkeypatch):
    _, users = team
    client = client_for(users["ada"])
    mcp = mcp_client(users["ada"])
    row = [[5.5, 3.5, 1.3, 0.2]]

    # Recorded no preprocessing (the fake runner does not).
    run_id = iris_run(client, training_dir, preprocessing=None)
    response = predict(client, run_id, inputs=row)
    assert response.status_code == 409 and "not known" in response.data["detail"]

    # Still running: no weights yet.
    use_runner(settings, "until_stop")
    response = client.post(
        "/api/training/runs/",
        {"name": "Busy", "graph": IRIS_GRAPH, "dataset": {"kind": "random"}, "config": QUICK},
        format="json",
    )
    running = response.data["id"]
    result, text = mcp_predict(mcp, running, inputs=row)
    assert result["isError"] is True and "weights are saved when it finishes" in text
    assert predict(client, running, inputs=row).status_code == 409
    client.post(f"/api/training/runs/{running}/stop/")
    wait_for(client, running)

    # Training switched off, or no PyTorch.
    use_runner(settings, "complete")
    run_id = iris_run(client, training_dir)
    settings.TRAINING_ENABLED = False
    assert predict(client, run_id, inputs=row).status_code == 503
    result, text = mcp_predict(mcp, run_id, inputs=row)
    assert result["isError"] is True and "disabled" in text
    settings.TRAINING_ENABLED = True
    monkeypatch.setattr(training_service, "torch_installed", lambda: False)
    response = predict(client, run_id, inputs=row)
    assert response.status_code == 503 and "PyTorch is not installed" in response.data["detail"]


@pytest.mark.django_db
def test_failing_predictions(team, predictor, training_dir, monkeypatch):
    _, users = team
    client = client_for(users["ada"])
    mcp = mcp_client(users["ada"])
    run_id = iris_run(client, training_dir)
    row = [[5.5, 3.5, 1.3, 0.2]]

    use_predictor(predictor, "crash")
    response = predict(client, run_id, inputs=row)
    assert response.status_code == 500
    assert response.data["detail"] == "The prediction failed: RuntimeError: out of cheese"

    use_predictor(predictor, "error")
    result, text = mcp_predict(mcp, run_id, inputs=row)
    assert result["isError"] is True and text.startswith("The trained weights do not fit")

    use_predictor(predictor, "nan")
    response = predict(client, run_id, inputs=row)
    assert response.status_code == 200 and response.data["predictions"][0]["class"] is None

    use_predictor(predictor, "echo:2")  # two classes instead of three
    response = predict(client, run_id, inputs=row)
    assert response.status_code == 500 and "number of classes" in response.data["detail"]

    use_predictor(predictor, "slow")
    monkeypatch.setattr(training_service, "PREDICT_TIMEOUT_SECONDS", 1)
    started = time.monotonic()
    response = predict(client, run_id, inputs=row)
    assert response.status_code == 500 and "longer than 1 seconds" in response.data["detail"]
    assert time.monotonic() - started < 10


@pytest.mark.django_db
def test_predictions_are_limited(team, predictor, training_dir, settings, monkeypatch):
    _, users = team
    client = client_for(users["ada"])
    mcp = mcp_client(users["ada"])
    run_id = iris_run(client, training_dir)
    row = [[5.5, 3.5, 1.3, 0.2]]

    # One at a time (across server processes): a second one is turned away.
    lock = open(training_dir / ".predict-0.lock", "a")  # noqa: SIM115
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        response = predict(client, run_id, inputs=row)
        assert response.status_code == 503 and response["Retry-After"] == "5"
        assert "busy" in response.data["detail"]
        result, text = mcp_predict(mcp, run_id, inputs=row)
        assert result["isError"] is True and "busy" in text
        settings.TRAINING_MAX_PREDICTIONS = 2
        assert predict(client, run_id, inputs=row).status_code == 200
    finally:
        lock.close()

    # Rate limit: only requests that start a process count.
    monkeypatch.setitem(ScopedRateThrottle.THROTTLE_RATES, "predict", "2/hour")
    other = client_for(users["bob"])
    bobs_run = iris_run(other, training_dir)
    assert predict(other, bobs_run, inputs=[[1, 2]]).status_code == 400
    assert predict(other, bobs_run, inputs=row).status_code == 200
    result, _ = mcp_predict(mcp_client(users["bob"]), bobs_run, inputs=row)
    assert "isError" not in result
    assert predict(other, bobs_run, inputs=row).status_code == 429
    result, text = mcp_predict(mcp_client(users["bob"]), bobs_run, inputs=row)
    assert result["isError"] is True and "Too many predictions" in text


# --------------------------------------------------------------------------- #
# For real, with PyTorch
# --------------------------------------------------------------------------- #


needs_torch = pytest.mark.skipif(not torch_installed(), reason="PyTorch is not installed")


def direct_probabilities(training_dir, run_id, graph, x):
    """The network of the run, computed in this process: the reference for the server."""
    import torch

    from brickdnn.training.engine import build_model

    model = build_model(graph)
    weights = torch.load(training_dir / "runs" / run_id / "weights.pt", weights_only=True)
    model.load_state_dict(weights)
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).softmax(dim=1).tolist()


@needs_torch
@pytest.mark.django_db
def test_real_prediction_on_a_trained_csv_model(team, training_dir, settings, tmp_path):
    settings.TRAINING_RUNNER_COMMAND = None
    settings.TRAINING_PREDICT_COMMAND = None
    _, users = team
    client = client_for(users["ada"])
    dataset_id = upload(client, "iris.csv", iris_csv(rows=60)).data["id"]
    response = client.post(
        "/api/training/runs/",
        {
            "name": "Iris net",
            "graph": IRIS_GRAPH,
            "dataset": {"kind": "upload", "id": dataset_id},
            "config": {"epochs": 2, "batch_size": 8, "device": "cpu", "seed": 1},
        },
        format="json",
    )
    assert response.status_code == 201, response.data
    run_id = response.data["id"]
    data = wait_for(client, run_id, timeout=120, interval=0.5)
    assert data["status"] == "completed", data["message"]
    preprocessing = data["state"]["preprocessing"]
    assert preprocessing["kind"] == "tabular" and preprocessing["columns"] == COLUMNS

    rows = [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3], [5.9, 3.0, 4.2, 1.5]]
    mcp = mcp_client(users["ada"])
    result, text = mcp_predict(mcp, run_id, inputs=rows, top_k=3)
    assert "isError" not in result, text
    predictions = result["structuredContent"]["predictions"]
    assert len(predictions) == 3

    standardised = [
        [
            (v - m) / s
            for v, m, s in zip(row, preprocessing["mean"], preprocessing["std"], strict=True)
        ]
        for row in rows
    ]
    expected = direct_probabilities(training_dir, run_id, IRIS_GRAPH, standardised)
    for prediction, probabilities in zip(predictions, expected, strict=True):
        assert sum(t["probability"] for t in prediction["top"]) == pytest.approx(1, abs=1e-3)
        best = max(range(3), key=lambda i: probabilities[i])
        assert prediction["class"] == SPECIES[best]
        assert prediction["probability"] == pytest.approx(probabilities[best], abs=1e-4)

    # The download link and the snippet work together: load the weights and predict.
    info, _ = call(mcp, "get_weights_download", id=run_id)
    info = info["structuredContent"]
    (tmp_path / info["model_code_filename"]).write_text(info["model_code"])
    weights = APIClient().get(urlsplit(info["url"]).path + "?" + urlsplit(info["url"]).query)
    assert weights.status_code == 200
    (tmp_path / info["filename"]).write_bytes(b"".join(weights.streaming_content))
    script = subprocess.run(
        [sys.executable, "-c", info["how_to_use"]],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert script.returncode == 0, script.stderr
    assert json.loads(script.stdout.strip().replace("'", '"'))[0] in SPECIES


@needs_torch
@pytest.mark.django_db
def test_real_prediction_with_an_image_model(team, predictor, training_dir, tmp_path):
    """Weights made directly with PyTorch, predicted by the real prediction process."""
    import torch

    from brickdnn.training.engine import build_model

    _, users = team
    client = client_for(users["ada"])
    pets = image_zip(["cats/1.png", "cats/2.png", "dogs/1.png", "dogs/2.png"])
    run_id = trained_run(client, training_dir, PETS_GRAPH, "pets.zip", pets, PETS_PREPROCESSING)
    torch.manual_seed(0)
    torch.save(build_model(PETS_GRAPH).state_dict(), training_dir / "runs" / run_id / "weights.pt")
    predictor.TRAINING_PREDICT_COMMAND = None

    images = [[[(16 * r + 4 * c) % 256 for c in range(4)] for r in range(4)], [[255] * 4] * 4]
    response = predict(client, run_id, inputs=images, input_scale="raw")
    assert response.status_code == 200, response.data
    normalised = [[[[(p / 255 - 0.5) / 0.25 for p in row] for row in image]] for image in images]
    expected = direct_probabilities(training_dir, run_id, PETS_GRAPH, normalised)
    for prediction, probabilities in zip(response.data["predictions"], expected, strict=True):
        assert prediction["top"][0]["probability"] == pytest.approx(max(probabilities), abs=1e-4)
        assert prediction["class"] == ["cats", "dogs"][probabilities.index(max(probabilities))]

    # The snippet of get_weights_download reads an image file the same way.
    from PIL import Image

    info, _ = call(mcp_client(users["ada"]), "get_weights_download", id=run_id)
    info = info["structuredContent"]
    (tmp_path / info["model_code_filename"]).write_text(info["model_code"])
    (tmp_path / info["filename"]).write_bytes(
        (training_dir / "runs" / run_id / "weights.pt").read_bytes()
    )
    picture = Image.new("L", (4, 4))
    picture.putdata([p for row in images[0] for p in row])
    picture.save(tmp_path / "example.png")
    script = subprocess.run(
        [sys.executable, "-c", info["how_to_use"]],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert script.returncode == 0, script.stderr
    assert script.stdout.strip() == f"['{response.data['predictions'][0]['class']}']"

    # Weights that do not fit the network are reported, not crashed on.
    torch.save({"nonsense": torch.zeros(1)}, training_dir / "runs" / run_id / "weights.pt")
    response = predict(client, run_id, inputs=images, input_scale="raw")
    assert response.status_code == 500
    assert response.data["detail"].startswith("The trained weights do not fit the network")
