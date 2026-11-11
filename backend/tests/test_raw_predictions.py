"""Predictions from raw rows: the API, the MCP tool and the prediction form of the web app
for runs trained on a dataset exported from a data preparation (see raw_predictions.py).

The tests use the fake runner and the fake prediction process of the training tests: the fake
network echoes its inputs (standardised with mean 0 and std 1), so the tests see the prepared
rows.
"""

# ruff: noqa: F811 (fixtures are imported from other test modules)

from __future__ import annotations

import csv
import io
import json

import numpy as np
import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from brick_dnn_core.models import Dataset, Project, ProjectMember, TrainingRun
from brickdnn.data import frozen
from brickdnn.training.state import read_state, write_state

from .helpers import chain
from .test_data_api import import_csv
from .test_mcp import call, mcp_client
from .test_predict import FAKE_PREDICTOR, recorded_request
from .test_training_api import QUICK, training_dir, wait_for  # noqa: F401

# --------------------------------------------------------------------------- #
# The API, the MCP tool and the prediction form
# --------------------------------------------------------------------------- #

STORES = "store,size\nBerlin,120\nHamburg,80\nKöln,95\n"


def sales_csv(rows: int = 40) -> bytes:
    """Daily sales with a date, a store (category) and missing temperatures; ';' and ','."""
    lines = ["date;store;promo;temp;sales"]
    stores = ["Berlin", "Hamburg", "Köln"]
    for i in range(rows):
        temp = "" if i % 7 == 3 else f"{5 + (i * 3) % 20},5"
        day = f"2024-{1 + i // 28:02d}-{1 + i % 28:02d}"
        lines.append(f"{day};{stores[i % 3]};{i % 2};{temp};{100 + 3 * i + 10 * (i % 3)}")
    return ("\n".join(lines) + "\n").encode()


SALES_STEPS = [
    {
        "id": "join",
        "type": "join",
        "params": {"source": None, "how": "left", "left_on": ["store"]},
    },
    {"id": "dates", "type": "datetime_features", "params": {"columns": ["date"]}},
    {"id": "fill", "type": "impute", "params": {"columns": ["temp"], "strategy": "median"}},
    {"id": "onehot", "type": "encode", "params": {"columns": ["store"], "method": "onehot"}},
    {"id": "target", "type": "drop_missing", "params": {"columns": ["sales"]}},
]


@pytest.fixture
def predictor(settings, training_dir, tmp_path):
    script = tmp_path / "fake_predictor.py"
    script.write_text(FAKE_PREDICTOR)
    settings.TRAINING_PREDICT_COMMAND = [__import__("sys").executable, str(script), "echo:1"]
    settings.TRAINING_MAX_RUNS = 5
    settings.DATA_PREP_DIR = tmp_path / "prepare"
    settings.DATA_PREP_ENABLED = True
    return settings


@pytest.fixture
def team(db):
    users = {
        name: get_user_model().objects.create_user(name, password="correct-horse-1")
        for name in ("ada", "bob", "cy", "eve")
    }
    project = Project.objects.create(owner=users["ada"], name="Shop", graph={})
    ProjectMember.objects.create(project=project, user=users["cy"], role="viewer")
    return project, users


def client_for(user) -> APIClient:
    client = APIClient()
    client.force_authenticate(user)
    return client


def sales_run(client, training_dir, project=None):
    """Import, prepare, export and train (fake runner); the run's preprocessing standardises
    nothing (mean 0, std 1), so the fake network echoes the prepared rows."""
    source = import_csv(client, "sales.csv", sales_csv()).data
    stores = import_csv(client, "stores.csv", STORES.encode()).data
    prep_steps = json.loads(json.dumps(SALES_STEPS))
    prep_steps[0]["params"]["source"] = stores["id"]
    body = {"name": "Sales", "source": source["id"], "steps": prep_steps}
    body["settings"] = {"target": "sales", "task": "regression"}
    prep = client.post("/api/data/preps/", body, format="json")
    assert prep.status_code == 201, prep.data
    exported = client.post(f"/api/data/preps/{prep.data['id']}/export/", {}, format="json")
    assert exported.status_code == 201, exported.data
    dataset = Dataset.objects.get(pk=exported.data["id"])
    features = dataset.info["columns"][:-1]
    graph = chain(("Input", {"shape": [len(features)]}), ("Linear", {"out_features": 1}))
    start = {
        "name": "Sales net",
        "graph": graph,
        "dataset": {"kind": "upload", "id": dataset.pk},
        "config": QUICK,
    }
    if project is not None:
        start["project"] = project.pk
    response = client.post("/api/training/runs/", start, format="json")
    assert response.status_code == 201, response.data
    run_id = response.data["id"]
    assert wait_for(client, run_id)["status"] == "completed"
    directory = training_dir / "runs" / run_id
    state = read_state(directory)
    state["preprocessing"] = {
        "kind": "tabular",
        "columns": features,
        "target": "sales",
        "delimiter": ",",
        "decimal_comma": False,
        "mean": [0.0] * len(features),
        "std": [1.0] * len(features),
        "target_mean": 0.0,
        "target_std": 1.0,
        "class_names": None,
    }
    write_state(directory, state)
    return run_id, dataset, features


def raw_sales_rows() -> list[dict]:
    """The first rows of sales.csv as a client sends them (strings as in the file)."""
    reader = csv.DictReader(io.StringIO(sales_csv().decode()), delimiter=";")
    return [{k: v for k, v in row.items() if k != "sales"} for row in reader]


def dataset_rows(dataset, features) -> list[list[float]]:
    from brick_dnn_core import training_service

    path = training_service.dataset_path(dataset)
    with path.open(encoding="utf-8") as handle:
        return [[float(row[f]) for f in features] for row in csv.DictReader(handle)]


def predict(client, run_id, **body):
    return client.post(f"/api/training/runs/{run_id}/predict/", body, format="json")


@pytest.mark.django_db
def test_predict_from_raw_rows_over_rest(team, predictor, training_dir):
    _, users = team
    client = client_for(users["ada"])
    run_id, dataset, features = sales_run(client, training_dir)
    assert "store=Berlin" in features and "date_year" in features and "size" in features

    rows = raw_sales_rows()[:5]
    response = predict(client, run_id, inputs=rows, input_format="raw")
    assert response.status_code == 200, response.data
    assert response.data["target"] == "sales" and len(response.data["predictions"]) == 5
    # The network got the rows of the training dataset (the prepared source rows).
    (tensor,) = recorded_request(training_dir, run_id)["inputs"]
    assert tensor["shape"] == [5, len(features)]
    values = np.array(tensor["values"]).reshape(5, -1)
    np.testing.assert_allclose(values, dataset_rows(dataset, features)[:5], rtol=0, atol=1e-12)
    assert "warnings" not in response.data

    # A new day: the store's size comes from the stored table, the missing temperature is
    # filled with the median of the training rows.
    new = {"date": "2024-12-24", "store": "Hamburg", "promo": 1, "temp": None}
    response = predict(client, run_id, inputs=[new, {**new, "store": "Bonn"}], input_format="raw")
    assert response.status_code == 400
    assert response.data["errors"] == [
        "Row 2: after the preparation, 'size' has no value (no match in 'stores'). The model "
        "needs a number in every feature column."
    ]
    response = predict(client, run_id, inputs=[new], input_format="raw")
    assert response.status_code == 200, response.data
    (tensor,) = recorded_request(training_dir, run_id)["inputs"]
    got = dict(zip(features, tensor["values"], strict=True))
    assert got["size"] == 80 and got["store=Hamburg"] == 1 and got["date_month"] == 12
    recipe = frozen.read_recipe(training_dir / "datasets" / str(dataset.pk) / "prepare")
    fill = next(s for s in recipe["steps"] if s["id"] == "fill")
    assert got["temp"] == fill["fitted"]["values"]["temp"]

    # Errors name the row and the column; prepared rows still work as before.
    response = predict(
        client, run_id, inputs=[{**new, "store": "Berlin", "temp": "warm"}], input_format="raw"
    )
    assert response.status_code == 400
    assert response.data["errors"] == ["The row, 'temp': 'warm' is not a number."]
    assert "input_format 'raw'" in response.data["inputs"]
    response = predict(client, run_id, inputs=[rows[0]])
    assert response.status_code == 400
    assert "need input_format 'raw'" in response.data["detail"]
    prepared = dict(zip(features, dataset_rows(dataset, features)[0], strict=True))
    assert predict(client, run_id, inputs=[prepared]).status_code == 200
    response = predict(client, run_id, inputs=[prepared], input_format="cooked")
    assert response.status_code == 400 and "input_format must be one of" in response.data["detail"]
    # The limit of a request is the same as for prepared rows.
    response = predict(client, run_id, inputs=[new] * 65, input_format="raw")
    assert response.status_code == 400
    assert response.data["detail"] == "65 rows are too many: predict at most 64 at a time."


@pytest.mark.django_db
def test_raw_rows_over_mcp_and_the_form(team, predictor, training_dir):
    project, users = team
    run_id, dataset, features = sales_run(client_for(users["ada"]), training_dir, project)
    mcp = mcp_client(users["ada"])
    rows = raw_sales_rows()[:3]
    result, text = call(mcp, "predict", run_id=run_id, inputs=rows, input_format="raw")
    assert "isError" not in result, text
    answer = result["structuredContent"]
    assert [set(p) for p in answer["predictions"]] == [{"value"}] * 3
    result, text = call(
        mcp,
        "predict",
        run_id=run_id,
        inputs=[{**rows[0], "temp": "hot"}, {**rows[1], "promo": "x"}],
        input_format="raw",
    )
    assert result.get("isError") is True
    assert "Row 1, 'temp': 'hot' is not a number.\nRow 2, 'promo': 'x' is not a number." in text
    assert "Expected inputs:" in text and "input_format 'raw'" in text
    result, text = call(mcp, "get_weights_download", id=run_id)
    assert "isError" not in result, text
    inputs = result["structuredContent"]["inputs"]
    assert "Or raw rows with input_format 'raw': the columns of 'sales'" in inputs
    assert "(date, store, promo, temp)" in inputs

    # The form: raw columns with their kinds and categories, validation rows as examples.
    # A viewer of the project may predict too.
    viewer = client_for(users["cy"])
    form = viewer.get(f"/api/training/runs/{run_id}/predict-form/").data
    assert form["available"] is True and form["kind"] == "tabular"
    assert form["task"] == "regression" and form["max_rows"] == 64
    assert [c["name"] for c in form["prepared"]["columns"]] == features
    raw = form["raw"]
    assert raw["source"] == "sales" and raw["target"] == "sales"
    columns = {c["name"]: c for c in raw["columns"]}
    assert list(columns) == ["date", "store", "promo", "temp"]
    assert columns["date"]["kind"] == "datetime"
    assert sorted(columns["store"]["categories"]) == ["Berlin", "Hamburg", "Köln"]
    assert columns["temp"]["missing"] is True
    assert raw["joined"] == ["stores"]
    sample = raw["samples"][0]
    assert set(sample["values"]) == set(columns) and isinstance(sample["target"], int)
    assert len(form["prepared"]["samples"]) == 5
    response = predict(viewer, run_id, inputs=[sample["values"]], input_format="raw")
    assert response.status_code == 200, response.data
    # Others cannot see the run.
    outsider = client_for(users["eve"])
    assert outsider.get(f"/api/training/runs/{run_id}/predict-form/").status_code == 404
    response = predict(outsider, run_id, inputs=[sample["values"]], input_format="raw")
    assert response.status_code == 404


@pytest.mark.django_db
def test_runs_keep_the_preparation_when_the_dataset_is_deleted(
    team, predictor, training_dir, django_capture_on_commit_callbacks
):
    _, users = team
    client = client_for(users["ada"])
    run_id, dataset, features = sales_run(client, training_dir)
    with django_capture_on_commit_callbacks(execute=True):
        dataset.delete()
    assert not (training_dir / "datasets" / str(dataset.pk)).exists()
    assert (training_dir / "runs" / run_id / "prepare" / "recipe.json").is_file()
    response = predict(client, run_id, inputs=raw_sales_rows()[:2], input_format="raw")
    assert response.status_code == 200, response.data


@pytest.mark.django_db
def test_raw_rows_need_a_run_from_a_preparation(team, predictor, training_dir):
    from .test_predict import IRIS_PREPROCESSING, iris_run

    _, users = team
    client = client_for(users["ada"])
    run_id = iris_run(client, training_dir)
    row = dict(zip(IRIS_PREPROCESSING["columns"], [5.0, 3.0, 1.0, 0.2], strict=True))
    response = predict(client, run_id, inputs=[row], input_format="raw")
    assert response.status_code == 400
    assert "this run takes the feature columns of its dataset" in response.data["detail"]
    form = client.get(f"/api/training/runs/{run_id}/predict-form/").data
    assert form["available"] is True and "raw" not in form
    assert form["class_names"] == ["setosa", "versicolor", "virginica"]
    assert len(form["prepared"]["samples"]) == 5  # rows of the CSV file
    assert set(form["prepared"]["samples"][0]["values"]) == set(IRIS_PREPROCESSING["columns"])

    # A run without weights says why.
    run = TrainingRun.objects.get(pk=run_id)
    run.has_weights = False
    run.save()
    form = client.get(f"/api/training/runs/{run_id}/predict-form/").data
    assert form["available"] is False and "no trained weights" in form["reason"]


def test_compact_prediction_answers_for_mcp():
    from brick_dnn_core.mcp_compact import prediction_answer

    top = [
        {"class": "a", "index": 0, "probability": 0.7},
        {"class": "b", "index": 1, "probability": 0.3},
    ]
    answer = {
        "run": {"id": "x", "name": "Net"},
        "task": "classification",
        "predictions": [
            {"class": "a", "index": 0, "probability": 0.7, "top": top},
            {"class": "a", "index": 0, "probability": 0.7, "top": top[:1]},
            {"outputs": [{"class": "b", "index": 1, "probability": 1.0, "top": top}]},
        ],
        "warnings": [f"Row {i}: note." for i in range(1, 15)],
    }
    compact = prediction_answer(answer)
    assert compact["predictions"][0] == {
        "class": "a",
        "probability": 0.7,
        "top": [{"class": "a", "probability": 0.7}, {"class": "b", "probability": 0.3}],
    }
    assert compact["predictions"][1] == {"class": "a", "probability": 0.7}  # top is itself
    assert compact["predictions"][2]["outputs"][0]["top"][1] == {"class": "b", "probability": 0.3}
    assert compact["warnings"][-1] == "… and 4 more" and len(compact["warnings"]) == 11
    regression = {"predictions": [{"value": 3.0, "std": 0.1}]}
    assert prediction_answer(regression)["predictions"] == [{"value": 3.0, "std": 0.1}]
