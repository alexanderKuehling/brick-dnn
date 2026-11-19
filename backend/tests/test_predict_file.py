"""Predicting every row of an uploaded CSV file (predict_files.py and
brickdnn/training/predict_file.py), with the real runner and prediction processes."""

# ruff: noqa: F811 (the fixtures are imported from test_training_api)

from __future__ import annotations

import csv
import io
import json
import os
import time

import pytest

pytest.importorskip("sklearn")

import numpy as np  # noqa: E402
from django.contrib.auth import get_user_model  # noqa: E402
from django.core.files.uploadedfile import SimpleUploadedFile  # noqa: E402
from rest_framework.test import APIClient  # noqa: E402

from brick_dnn_core import predict_files  # noqa: E402
from brick_dnn_core.api_tokens import create_token  # noqa: E402
from brick_dnn_core.models import Dataset, Project, ProjectMember, UsageRecord  # noqa: E402
from brickdnn.training import predict_file as engine  # noqa: E402
from brickdnn.training import torch_installed  # noqa: E402

from .helpers import chain  # noqa: E402
from .test_classical_training import (  # noqa: E402
    classification_csv,
    model_graph,
    regression_csv,
)
from .test_training_api import (  # noqa: E402, F401 (fixtures)
    auth_client,
    training_dir,
    upload,
    user,
    wait_for,
)

needs_torch = pytest.mark.skipif(not torch_installed(), reason="PyTorch is not installed")


@pytest.fixture
def real_processes(training_dir, settings):
    settings.TRAINING_RUNNER_COMMAND = None
    settings.TRAINING_PREDICT_COMMAND = None
    settings.TRAINING_MAX_RUNS = 5
    return training_dir


def train(client, name, graph, dataset, config):
    response = client.post(
        "/api/training/runs/",
        {"name": name, "graph": graph, "dataset": dataset, "config": config},
        format="json",
    )
    assert response.status_code == 201, response.data
    data = wait_for(client, response.data["id"], timeout=180, interval=0.25)
    assert data["status"] == "completed", data["message"]
    return data["id"]


def uploaded(client, path):
    response = upload(client, path.name, path.read_bytes())
    assert response.status_code == 201, response.data
    return {"kind": "upload", "id": response.data["id"]}


def mail_run(client, tmp_path, labels=("ham", "spam", "eggs")):
    """A gradient boosting classifier on a,b,c,d -> label."""
    dataset = uploaded(client, classification_csv(tmp_path / "mail.csv", labels=labels))
    graph = model_graph("HistGradientBoostingClassifier")
    return train(client, "Mail trees", graph, dataset, {"seed": 1})


def predict_file(client, run_id, content, name="test.csv", **fields):
    if isinstance(content, str):
        content = content.encode()
    if isinstance(fields.get("columns"), list):
        fields["columns"] = json.dumps(fields["columns"])
    return client.post(
        f"/api/training/runs/{run_id}/predict-file/",
        {"file": SimpleUploadedFile(name, content), **fields},
        format="multipart",
    )


def download(client, run_id, result_id, delimiter=","):
    response = client.get(f"/api/training/runs/{run_id}/predict-file/{result_id}/")
    if response.status_code != 200:
        return response, None
    text = b"".join(response.streaming_content).decode()
    return response, list(csv.reader(io.StringIO(text), delimiter=delimiter))


def single(client, run_id, rows, **body):
    """What the predict action answers for the same rows (at most 64 at a time)."""
    found = []
    for start in range(0, len(rows), 64):
        response = client.post(
            f"/api/training/runs/{run_id}/predict/",
            {"inputs": rows[start : start + 64], "top_k": 3, **body},
            format="json",
        )
        assert response.status_code == 200, response.data
        found += response.data["predictions"]
    return found


def mail_rows(n=150, seed=5):
    rng = np.random.default_rng(seed)
    return [[f"{v:.4f}" for v in row] for row in rng.normal(size=(n, 4))]


def temp_entries(training_dir):
    folder = training_dir / predict_files.FOLDER
    return sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []


# --------------------------------------------------------------------------- #
# A classical model: the options of the result, problems, limits, clean-up
# --------------------------------------------------------------------------- #


@pytest.mark.django_db
def test_a_classifier_predicts_a_file_with_every_output(
    auth_client, user, real_processes, tmp_path
):
    run_id = mail_run(auth_client, tmp_path)
    rows = mail_rows()
    # An id column first, the model's columns in another order, and a column it ignores.
    lines = ["id,d,c,b,a,note"]
    lines += [f"row-{i},{r[3]},{r[2]},{r[1]},{r[0]},x{i}" for i, r in enumerate(rows)]
    content = "\n".join(lines) + "\n"
    expected = single(auth_client, run_id, [[float(v) for v in r] for r in rows])

    # The default: every column of the file plus the predicted class.
    response = predict_file(auth_client, run_id, content)
    assert response.status_code == 200, response.data
    data = response.data
    assert data["rows"] == data["predicted"] == 150 and data["failed"] == 0
    assert data["columns"] == ["id", "d", "c", "b", "a", "note", "prediction"]
    assert data["output"] == "label" and data["input_format"] == "prepared"
    assert data["ignored_columns"] == ["id", "note"] and data["metrics"] is None
    assert data["filename"] == "test-predictions.csv" and data["keep_minutes"] == 60
    assert sum(data["class_counts"].values()) == 150
    assert len(data["preview"]) == 20 and data["preview"][0][0] == "row-0"
    # The uploaded file is gone already; the result waits for its download.
    assert temp_entries(real_processes) == [data["id"]]
    folder = real_processes / predict_files.FOLDER / data["id"]
    assert sorted(p.name for p in folder.iterdir()) == ["meta.json", "result.csv"]
    answer, table = download(auth_client, run_id, data["id"])
    assert answer["Content-Disposition"] == 'attachment; filename="test-predictions.csv"'
    assert table[0] == data["columns"] and len(table) == 151
    # Same rows, same order, the classes of single predictions.
    assert [line[0] for line in table[1:]] == [f"row-{i}" for i in range(150)]
    assert [line[5] for line in table[1:]] == [f"x{i}" for i in range(150)]
    assert [line[-1] for line in table[1:]] == [p["class"] for p in expected]
    assert table[1:21] == data["preview"]
    # Downloaded once: deleted.
    assert temp_entries(real_processes) == []
    again, _ = download(auth_client, run_id, data["id"])
    assert again.status_code == 404 and "Predict the file again" in again.data["detail"]

    # A submission: only the id and the prediction, under another name.
    response = predict_file(auth_client, run_id, content, columns=["id"], prediction_column="label")
    assert response.status_code == 200, response.data
    assert response.data["columns"] == ["id", "label"]
    _, table = download(auth_client, run_id, response.data["id"])
    assert table[0] == ["id", "label"]
    assert table[1:] == [[f"row-{i}", p["class"]] for i, p in enumerate(expected)]

    # The probability of one class, and of all classes (columns named like the classes).
    response = predict_file(
        auth_client,
        run_id,
        content,
        columns=["id"],
        output="probability",
        prediction_column="p_spam",
        **{"class": "spam"},
    )
    assert response.status_code == 200, response.data
    assert response.data["class"] == "spam" and response.data["columns"] == ["id", "p_spam"]
    assert 0 <= response.data["values"]["min"] <= response.data["values"]["max"] <= 1
    _, table = download(auth_client, run_id, response.data["id"])
    spam = {i: {t["class"]: t["probability"] for t in p["top"]} for i, p in enumerate(expected)}
    got = [float(line[1]) for line in table[1:]]
    np.testing.assert_allclose(got, [spam[i]["spam"] for i in range(150)], atol=6e-5)

    response = predict_file(auth_client, run_id, content, columns=[], output="probabilities")
    assert response.status_code == 200, response.data
    assert response.data["columns"] == ["eggs", "ham", "spam"]
    _, table = download(auth_client, run_id, response.data["id"])
    matrix = np.array(table[1:], dtype=float)
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(matrix[:, 2], [spam[i]["spam"] for i in range(150)], atol=6e-5)

    # The CPU time counts as predictions of the run, like single ones.
    records = UsageRecord.objects.filter(user=user, kind="prediction", run_id=run_id)
    assert records.count() >= 4 + 3 and all(r.quantity > 0 for r in records)


@pytest.mark.django_db
def test_the_target_column_gives_metrics_and_bad_rows_stay(auth_client, real_processes, tmp_path):
    run_id = mail_run(auth_client, tmp_path)
    # The training file itself (with its target), a ';' file with decimal commas.
    reader = csv.reader(io.StringIO((tmp_path / "mail.csv").read_text()))
    header, *rows = list(reader)
    lines = [";".join(header)]
    lines += [";".join([*(v.replace(".", ",") for v in r[:4]), r[4]]) for r in rows]
    lines[3] = lines[3].replace(lines[3].split(";")[1], "n/a")  # row 3: b is no number
    lines[6] = "1,5;2,5"  # row 6: too few values
    lines.insert(9, "")  # an empty line is no row
    lines[11] = ";".join([*lines[11].split(";")[:4], ""])  # row 10: no target
    lines[12] = ";".join(["", *lines[12].split(";")[1:]])  # row 11: a is empty
    content = "﻿" + "\r\n".join(lines) + "\r\n"

    response = predict_file(auth_client, run_id, content.encode("utf-8"), name="Mail März.csv")
    assert response.status_code == 200, response.data
    data = response.data
    assert data["delimiter"] == ";" and data["encoding"] == "utf-8"
    assert data["rows"] == 90 and data["predicted"] == 87 and data["failed"] == 3
    assert data["problems"] == [
        "Row 3, 'b': 'n/a' is not a number.",
        "Row 6 has 2 values, but the file has 5 columns.",
        "Row 11, 'a': no value.",
    ]
    assert data["target"] == "label" and data["filename"] == "mail-marz-predictions.csv"
    metrics = data["metrics"]
    assert metrics["rows"] == 86 and metrics["skipped"] == 0  # 87 predicted, one without target
    assert metrics["accuracy"] > 0.9 and 0 < metrics["log_loss"] < 0.5
    assert "roc_auc" not in metrics  # three classes
    _, table = download(auth_client, run_id, data["id"], delimiter=";")
    assert len(table) == 91  # the header and all 90 rows
    assert table[3][-1] == "" and table[6] == ["1,5", "2,5", "", "", "", ""]
    assert table[11][-1] == "" and table[1][-1] in ("ham", "spam", "eggs")
    assert [line[4] for line in table[1:6]] == [r[4] for r in rows[:5]]  # the target is kept

    # Latin-1 text and a tab-separated file are read too.
    tabbed = "a\tb\tc\td\tstraße\n" + "\n".join("\t".join([*r[:4], "Köln"]) for r in rows[:5])
    response = predict_file(auth_client, run_id, tabbed.encode("cp1252"), name="t.tsv")
    assert response.status_code == 200, response.data
    assert response.data["encoding"] == "cp1252" and response.data["delimiter"] == "\t"
    assert response.data["preview"][0][4] == "Köln"
    assert response.data["columns"][4] == "straße"


@pytest.mark.django_db
def test_files_that_do_not_fit_are_refused_and_leave_nothing(
    auth_client, user, real_processes, settings, tmp_path
):
    run_id = mail_run(auth_client, tmp_path)
    rows = mail_rows(12)
    good = "a,b,c,d\n" + "\n".join(",".join(r) for r in rows) + "\n"
    url = f"/api/training/runs/{run_id}/predict-file/"

    def refused(response, status, *words):
        assert response.status_code == status, response.data
        for word in words:
            assert word in response.data["detail"], response.data
        assert temp_entries(real_processes) == []

    refused(auth_client.post(url, {}, format="multipart"), 400, "form field 'file'")
    refused(predict_file(auth_client, run_id, good, name="test.xlsx"), 400, "CSV file")
    refused(predict_file(auth_client, run_id, b"PK\x03\x04binary"), 400, "Excel workbook")
    refused(predict_file(auth_client, run_id, b"  \n"), 400, "empty")
    refused(predict_file(auth_client, run_id, "a,b,c,d\n"), 400, "no rows to predict")
    # Missing columns: which ones, what the model needs and what the file has.
    response = predict_file(auth_client, run_id, "id,a,b,x\n1,1,2,3\n")
    refused(response, 400, "no column 'c', 'd'", "The model needs: a, b, c, d", "id, a, b, x")
    refused(predict_file(auth_client, run_id, "a,a,b,c,d\n1,1,2,3,4\n"), 400, "more than one")
    # No row can be predicted: the first reason.
    response = predict_file(auth_client, run_id, "a,b,c,d\nx,1,2,3\ny,1,2,3\n")
    refused(response, 400, "No row of the file could be predicted", "Row 1, 'a': 'x'")
    # Options that do not fit the file or the model.
    refused(predict_file(auth_client, run_id, good, columns=["id"]), 400, "no column 'id'")
    refused(predict_file(auth_client, run_id, good, columns="{"), 400, "JSON list")
    refused(predict_file(auth_client, run_id, good, output="logits"), 400, "'output'")
    response = predict_file(auth_client, run_id, good, output="probability")
    refused(response, 400, "'class'", "eggs, ham, spam")
    refused(predict_file(auth_client, run_id, good, prediction_column="a"), 400, "two columns")
    refused(predict_file(auth_client, run_id, good, input_format="raw"), 400, "data preparation")
    # The limits: rows and size.
    settings.TRAINING_PREDICT_FILE_MAX_ROWS = 10
    refused(predict_file(auth_client, run_id, good), 400, "more than 10 rows", "Split it")
    settings.TRAINING_PREDICT_FILE_MAX_ROWS = 1000
    settings.TRAINING_PREDICT_FILE_MAX_MB = 0.0001
    refused(predict_file(auth_client, run_id, good), 413, "too large", "0.0001 MB")
    settings.TRAINING_PREDICT_FILE_MAX_MB = 200
    # ... and never more than a dataset upload.
    settings.TRAINING_MAX_UPLOAD_MB = 0.0001
    refused(predict_file(auth_client, run_id, good), 413, "too large")
    settings.TRAINING_MAX_UPLOAD_MB = 5
    form = auth_client.get(f"/api/training/runs/{run_id}/predict-form/").data
    assert form["file"] == {"max_mb": 5, "max_rows": 1000, "keep_minutes": 60}
    assert form["max_form_rows"] == 300

    # A used-up compute budget refuses the file like a single prediction.
    usage = "/api/account/usage/"
    assert auth_client.patch(usage, {"budget_cpu_minutes": 0.001}, format="json").status_code == 200
    response = predict_file(auth_client, run_id, good)
    assert response.status_code == 403 and response.data["code"] == "budget_exceeded"
    assert temp_entries(real_processes) == []
    assert auth_client.patch(usage, {"budget_cpu_minutes": None}, format="json").status_code == 200

    # A prediction that takes too long is stopped; its time counts.
    settings.TRAINING_PREDICT_FILE_COMMAND = [
        __import__("sys").executable,
        "-c",
        "import time; time.sleep(30)",
    ]
    settings.TRAINING_PREDICT_FILE_TIMEOUT = 1
    before = UsageRecord.objects.filter(user=user, kind="prediction").count()
    refused(predict_file(auth_client, run_id, good), 500, "longer than 1 seconds")
    assert UsageRecord.objects.filter(user=user, kind="prediction").count() == before + 1
    settings.TRAINING_PREDICT_FILE_COMMAND = None
    settings.TRAINING_PREDICT_FILE_TIMEOUT = 270

    # Results that are not downloaded expire.
    response = predict_file(auth_client, run_id, good)
    assert response.status_code == 200, response.data
    result_id = response.data["id"]
    folder = real_processes / predict_files.FOLDER / result_id
    old = time.time() - 61 * 60
    os.utime(folder, (old, old))
    gone, _ = download(auth_client, run_id, result_id)
    assert gone.status_code == 404 and not folder.exists()
    # ... also without anyone asking for them (purge_old_data, the next file), like a work
    # folder that a restart left behind.
    response = predict_file(auth_client, run_id, good)
    kept = real_processes / predict_files.FOLDER / response.data["id"]
    stale = real_processes / predict_files.FOLDER / ".work-0123"
    stale.mkdir()
    os.utime(kept, (old, old))
    os.utime(stale, (old, old))
    assert predict_files.remove_expired() == 2
    assert temp_entries(real_processes) == []
    # A result can also be discarded.
    response = predict_file(auth_client, run_id, good)
    deleted = auth_client.delete(f"{url}{response.data['id']}/")
    assert deleted.status_code == 204 and temp_entries(real_processes) == []


@pytest.mark.django_db
def test_who_may_predict_a_file_and_get_its_result(real_processes, tmp_path):
    users = {
        name: get_user_model().objects.create_user(name, password="correct-horse-1")
        for name in ("ada", "cy", "eve")
    }
    clients = {}
    for name, account in users.items():
        clients[name] = APIClient()
        clients[name].force_authenticate(account)
    project = Project.objects.create(owner=users["ada"], name="Mail", graph={})
    ProjectMember.objects.create(project=project, user=users["cy"], role="viewer")
    dataset = uploaded(clients["ada"], classification_csv(tmp_path / "mail.csv"))
    response = clients["ada"].post(
        "/api/training/runs/",
        {
            "name": "Mail trees",
            "graph": model_graph("HistGradientBoostingClassifier"),
            "dataset": dataset,
            "config": {"seed": 1},
            "project": project.pk,
        },
        format="json",
    )
    assert response.status_code == 201, response.data
    run_id = response.data["id"]
    assert wait_for(clients["ada"], run_id, timeout=120, interval=0.25)["status"] == "completed"
    content = "a,b,c,d\n" + "\n".join(",".join(r) for r in mail_rows(5)) + "\n"

    assert predict_file(APIClient(), run_id, content).status_code == 401
    # Not in the project: the run does not exist for them.
    assert predict_file(clients["eve"], run_id, content).status_code == 404
    # A viewer of the project may predict (like single predictions), on their own account.
    response = predict_file(clients["cy"], run_id, content)
    assert response.status_code == 200, response.data
    result_id = response.data["id"]
    assert UsageRecord.objects.filter(user=users["cy"], kind="prediction").exists()
    assert not UsageRecord.objects.filter(user=users["ada"], kind="prediction").exists()
    # The result holds the rows they uploaded: only they get it.
    assert download(APIClient(), run_id, result_id)[0].status_code == 401
    assert download(clients["eve"], run_id, result_id)[0].status_code == 404
    assert download(clients["ada"], run_id, result_id)[0].status_code == 404
    assert (
        clients["ada"].delete(f"/api/training/runs/{run_id}/predict-file/{result_id}/").status_code
        == 204
    )  # nothing happens
    assert download(clients["cy"], run_id, "../" + result_id)[0].status_code == 404
    answer, table = download(clients["cy"], run_id, result_id)
    assert answer.status_code == 200 and len(table) == 6

    # A read-only API token may predict a file (it changes nothing), like single predictions.
    _, key = create_token(users["ada"], "reader", read_only=True)
    reader = APIClient()
    reader.credentials(HTTP_AUTHORIZATION=f"Bearer {key}")
    response = predict_file(reader, run_id, content, columns=[])
    assert response.status_code == 200, response.data
    record = UsageRecord.objects.get(user=users["ada"], kind="prediction")
    assert record.source == "token" and record.via == "reader"
    assert download(reader, run_id, response.data["id"])[0].status_code == 200


# --------------------------------------------------------------------------- #
# The other model families
# --------------------------------------------------------------------------- #


@needs_torch
@pytest.mark.django_db
def test_a_neural_regression_writes_values_in_the_units_of_the_target(
    auth_client, real_processes, tmp_path
):
    dataset = uploaded(auth_client, regression_csv(tmp_path / "houses.csv", n=120))
    graph = chain(
        ("Input", {"shape": [4]}),
        ("Linear", {"out_features": 8}),
        ("Activation", {}),
        ("Linear", {"out_features": 1}),
    )
    config = {"epochs": 30, "batch_size": 16, "device": "cpu", "seed": 1, "learning_rate": 0.01}
    run_id = train(auth_client, "Houses", graph, dataset, config)
    reader = csv.reader(io.StringIO((tmp_path / "houses.csv").read_text()))
    header, *rows = list(reader)
    expected = single(auth_client, run_id, [[float(v) for v in r[:4]] for r in rows])

    response = predict_file(auth_client, run_id, (tmp_path / "houses.csv").read_bytes())
    assert response.status_code == 200, response.data
    data = response.data
    assert data["task"] == "regression" and data["output"] == "value"
    assert data["columns"] == [*header, "prediction"] and data["target"] == "price"
    assert "class_counts" not in data
    # Around 100, the mean of the target: not the standardised values of the network.
    assert 90 < data["values"]["min"] < data["values"]["mean"] < data["values"]["max"] < 110
    metrics = data["metrics"]
    assert metrics["rows"] == 120 and metrics["rmse"] > 0 and metrics["mae"] <= metrics["rmse"]
    assert metrics["r2"] is not None and metrics["r2"] <= 1
    _, table = download(auth_client, run_id, data["id"])
    got = [float(line[-1]) for line in table[1:]]
    np.testing.assert_allclose(got, [p["value"] for p in expected], rtol=1e-5)
    actual = np.array([float(r[4]) for r in rows])
    assert metrics["rmse"] == pytest.approx(
        float(np.sqrt(np.mean((np.array(got) - actual) ** 2))), rel=1e-4
    )

    # A file without a header line (like a training file without one): by position.
    bare = "\n".join(",".join(r[:4]) for r in rows[:7]) + "\n"
    response = predict_file(auth_client, run_id, bare, columns=[])
    assert response.status_code == 200, response.data
    assert response.data["has_header"] is False and response.data["rows"] == 7
    _, table = download(auth_client, run_id, response.data["id"])
    assert table[0] == ["prediction"]
    np.testing.assert_allclose([float(line[0]) for line in table[1:]], got[:7], rtol=1e-6)


@needs_torch
@pytest.mark.django_db
def test_a_hybrid_network_predicts_a_file(auth_client, real_processes, tmp_path):
    from .test_hybrid import hybrid_graph

    dataset = uploaded(auth_client, classification_csv(tmp_path / "mail.csv", n=150))
    config = {"seed": 1, "epochs": 4, "device": "cpu"}
    run_id = train(auth_client, "Mail features", hybrid_graph("SVC"), dataset, config)
    rows = mail_rows(70)
    content = "a,b,c,d\n" + "\n".join(",".join(r) for r in rows) + "\n"
    expected = single(auth_client, run_id, [[float(v) for v in r] for r in rows])
    response = predict_file(auth_client, run_id, content, output="probabilities", columns=[])
    assert response.status_code == 200, response.data
    assert response.data["predicted"] == 70
    names = response.data["columns"]
    assert sorted(names) == ["eggs", "ham", "spam"]
    _, table = download(auth_client, run_id, response.data["id"])
    matrix = np.array(table[1:], dtype=float)
    assert [names[i] for i in matrix.argmax(axis=1)] == [p["class"] for p in expected]
    np.testing.assert_allclose(matrix.max(axis=1), [p["probability"] for p in expected], atol=6e-5)


def payments_csv(n: int = 240, seed: int = 11, target: bool = True) -> str:
    rng = np.random.default_rng(seed)
    lines = ["amount;merchant;hour" + (";label" if target else "")]
    merchants = ["shop", "travel", "games"]
    for i in range(n):
        amount = float(rng.gamma(2.0, 40.0))
        merchant = merchants[i % 3]
        hour = int(rng.integers(0, 24))
        risky = amount > 110 or (merchant == "games" and hour < 6)
        line = f"{amount:.2f};{merchant};{hour}"
        lines.append(line + (f";{'fraud' if risky else 'ok'}" if target else ""))
    return "\n".join(lines) + "\n"


@needs_torch
@pytest.mark.django_db
def test_an_ensemble_predicts_the_raw_rows_of_a_prepared_dataset(
    auth_client, real_processes, settings, tmp_path
):
    """Runs trained on a dataset from the Data tab take a file in the columns of the source
    table: the frozen preparation is applied (imputation as in the training), the members
    predict, the ensemble combines them and uses its decision threshold."""
    from .test_data_api import import_csv

    settings.DATA_PREP_DIR = tmp_path / "prepare"
    settings.DATA_PREP_ENABLED = True
    client = auth_client
    source = import_csv(client, "payments.csv", payments_csv().encode()).data
    body = {
        "name": "Payments",
        "source": source["id"],
        "steps": [
            {"id": "fill", "type": "impute", "params": {"columns": ["hour"], "strategy": "median"}},
            {
                "id": "onehot",
                "type": "encode",
                "params": {"columns": ["merchant"], "method": "onehot"},
            },
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
    boosting = model_graph(
        "HistGradientBoostingClassifier",
        features,
        {"max_iter": 30, "min_samples_leaf": 5, "early_stopping": "off"},
    )
    ids = [
        train(
            client, "MLP", mlp, ref, {"epochs": 15, "batch_size": 16, "device": "cpu", "seed": 1}
        ),
        train(client, "Boosting", boosting, ref, {"seed": 1}),
    ]
    response = client.post(
        "/api/training/ensembles/", {"members": ids, "name": "MLP + trees"}, format="json"
    )
    assert response.status_code == 201, response.data
    run_id = response.data["id"]
    assert wait_for(client, run_id, timeout=120, interval=0.25)["status"] == "completed"

    # New payments with their labels, an id, a merchant the training never saw (row 2), an
    # hour that is missing (row 3: filled like in the training) and no amount (row 5).
    reader = csv.reader(io.StringIO(payments_csv(60, seed=4)), delimiter=";")
    _, *rows = list(reader)
    rows[1][1] = "casino"
    rows[2][2] = ""
    rows[4][0] = ""
    lines = ["id;hour;merchant;amount;label"]
    lines += [f"{1000 + i};{r[2]};{r[1]};{r[0]};{r[3]}" for i, r in enumerate(rows)]
    content = "\n".join(lines) + "\n"
    good = [i for i in range(60) if i != 4]
    raw = [{"amount": rows[i][0], "merchant": rows[i][1], "hour": rows[i][2]} for i in good]
    expected = single(client, run_id, raw, input_format="raw")

    response = predict_file(
        client, run_id, content, columns=["id"], output="probability", prediction_column="p_ok"
    )
    assert response.status_code == 200, response.data
    data = response.data
    assert data["input_format"] == "raw" and data["class"] == "ok"  # the second class
    assert data["rows"] == 60 and data["predicted"] == 59 and data["failed"] == 1
    assert data["problems"][0].startswith("Row 5: after the preparation, ")
    assert "no value given for amount" in data["problems"][0]
    assert data["warning_count"] == 1 and data["warnings"][0].startswith("Row 2, 'merchant'")
    assert data["ignored_columns"] == ["id"] and data["target"] == "label"
    assert data["decision_threshold"] == 0.5
    metrics = data["metrics"]
    assert metrics["rows"] == 59 and metrics["accuracy"] > 0.6 and 0.5 < metrics["roc_auc"] <= 1
    _, table = download(client, run_id, data["id"], delimiter=";")
    assert table[0] == ["id", "p_ok"] and len(table) == 61
    assert [line[0] for line in table[1:]] == [str(1000 + i) for i in range(60)]
    assert table[5] == ["1004", ""]
    got = [float(table[1 + i][1]) for i in good]
    np.testing.assert_allclose(got, [p["positive_probability"] for p in expected], atol=1e-5)

    # The class follows the ensemble's decision threshold.
    def labels():
        answer = predict_file(client, run_id, content, columns=[])
        assert answer.status_code == 200, answer.data
        _, lines = download(client, run_id, answer.data["id"])
        return answer.data, [line[0] if line else "" for line in lines[1:]]

    _, before = labels()
    assert [before[i] for i in good] == [p["class"] for p in expected]
    changed = client.post(
        f"/api/training/runs/{run_id}/decision-threshold/", {"threshold": 0.999}, format="json"
    )
    assert changed.status_code == 200, changed.data
    summary, after = labels()
    assert summary["decision_threshold"] == 0.999
    assert summary["class_counts"]["ok"] < sum(1 for label in before if label == "ok")
    assert [after[i] for i in good] == ["ok" if p >= 0.999 else "fraud" for p in got]

    # The feature columns of the prepared dataset work as well (a member alone).
    names = dataset.info["columns"][:-1]
    prepared = client.post(
        f"/api/training/runs/{ids[1]}/predict/",
        {"inputs": raw[:3], "input_format": "raw"},
        format="json",
    )
    assert prepared.status_code == 200, prepared.data
    # ... which a file without one of the source columns cannot replace.
    response = predict_file(client, ids[1], "id;hour;merchant\n1;3;shop\n")
    assert response.status_code == 400
    assert "no column 'amount'" in response.data["detail"]
    assert "payments" in response.data["detail"]
    values = {"amount": 250.0, "hour": 13.0}
    line = ",".join(str(values.get(n, 1.0 if n.endswith("travel") else 0.0)) for n in names)
    response = predict_file(client, ids[1], ",".join(names) + "\n" + line + "\n")
    assert response.status_code == 200, response.data
    assert response.data["input_format"] == "prepared"
    assert response.data["preview"][0][-1] == "fraud"


# --------------------------------------------------------------------------- #
# The parts of the prediction process
# --------------------------------------------------------------------------- #


def test_backups_leave_the_temporary_files_out(tmp_path):
    from brick_dnn_core import backups

    assert predict_files.FOLDER in backups.TEMPORARY_FOLDERS
    kept = tmp_path / "training" / "runs" / "r1"
    temporary = tmp_path / "training" / predict_files.FOLDER / "0123"
    for folder in (kept, temporary):
        folder.mkdir(parents=True)
        (folder / "file.csv").write_text("a\n")
    found = list(backups._walk(backups._data_entries(tmp_path, set())))
    assert found == [kept / "file.csv"]


def test_encodings_and_delimiters_are_detected(tmp_path):
    path = tmp_path / "file.csv"
    for text, codec, found in (
        ("a,b\n1,ä\n", "utf-8", "utf-8-sig"),
        ("a,b\n1,ä\n", "utf-8-sig", "utf-8-sig"),
        ("a,b\n1,ä€\n", "cp1252", "cp1252"),
        ("a,b\n1,ä\n", "utf-16", "utf-16"),
    ):
        path.write_bytes(text.encode(codec))
        assert engine.detect_encoding(path) == found
    path.write_bytes(b"a,b\n1,\x81\n")  # not Windows-1252 either
    assert engine.detect_encoding(path) == "latin-1"
    path.write_bytes(b"a\x00b")
    with pytest.raises(engine.FileError, match="not a text file"):
        engine.detect_encoding(path)
    path.write_text("sep=|\nx|y\n1|2\n\n3|4\n")
    with path.open(newline="") as handle:
        table = engine.open_table(path, handle, "utf-8-sig")
        assert table.delimiter == "|" and table.first == ["x", "y"]
        assert list(table.records) == [["1", "2"], ["3", "4"]]
    assert engine.detect_delimiter("a;b;c\n1,5;2;3\n") == ";"


def test_the_columns_of_the_result_are_checked():
    file = engine.CsvFile(["id", "a", "b", "label"], iter(()), ",", "utf-8")
    layout = engine.layout_of(file, ["a", "b"], "label", None, "auto", False)
    assert layout.needed == [("a", 1), ("b", 2)] and layout.target_at == 3
    assert layout.ignored == ["id"]
    names = ["no", "yes"]
    result = engine.result_of(layout, None, "classification", names)
    assert result.columns == ["id", "a", "b", "label", "prediction"] and result.output == "label"
    result = engine.result_of(
        layout, {"columns": ["id", "id"], "output": "probability"}, "classification", names
    )
    assert result.columns == ["id", "prediction"] and result.class_at == 1  # the positive class
    options = {"columns": ["id"], "output": "probabilities", "prediction_column": " p "}
    assert engine.result_of(layout, options, "classification", names).columns == ["id", "no", "yes"]
    # Class names that the kept columns have already get the name of the prediction column.
    clash = engine.result_of(layout, options, "classification", ["id", "b"])
    assert clash.columns == ["id", "p_id", "p_b"]
    assert engine.result_of(layout, {"output": "label"}, "regression", None).output == "value"
    for options, message in (
        ({"columns": "id"}, "list of column names"),
        ({"prediction_column": "x" * 121}, "1 to 120 characters"),
        ({"prediction_column": "label"}, "two columns named 'label'"),
        ({"output": "probability", "class": "maybe"}, "must be one of: no, yes"),
    ):
        with pytest.raises(engine.FileError, match=message):
            engine.result_of(layout, options, "classification", names)


def test_metrics_read_class_names_numbers_and_indices():
    metrics = engine.Metrics("classification", ["1", "2", "7"], False)
    assert [metrics._class(c) for c in ("1", " 2.0 ", "7", "0", "x", "3")] == [
        0,
        1,
        2,
        None,
        None,
        None,
    ]
    named = engine.Metrics("classification", ["no", "yes"], False)
    assert [named._class(c) for c in ("yes", "0", "1", "2", "")] == [1, 0, 1, None, None]
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    named.add(["no", "yes", "yes", "", "maybe"][:4], proba, np.array([0, 1, 0, 1]))
    named.add(["maybe"], proba[:1], np.array([0]))
    found = named.result()
    assert found["rows"] == 3 and found["skipped"] == 1
    assert found["accuracy"] == pytest.approx(2 / 3) and found["roc_auc"] == 1.0
    assert found["log_loss"] == pytest.approx(-np.log([0.9, 0.8, 0.4]).mean())
    assert engine.Metrics("regression", None, False).result() is None
    # A preparation that names the target may change it: no metrics from raw files then.
    recipe = {
        "target": "price",
        "steps": [{"status": "ok", "type": "scale", "params": {"columns": ["price"]}}],
    }
    assert engine.target_changed(recipe) is True
    recipe["steps"][0]["params"]["columns"] = ["area"]
    assert engine.target_changed(recipe) is False
