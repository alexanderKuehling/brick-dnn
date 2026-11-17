"""The client that logs a training elsewhere into Brick DNN (brickdnn/tracking.py) and the
code generator's option that embeds it ("log to Brick DNN"): inert without BRICKDNN_TOKEN,
never stops a training, and logs into a real server."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from brickdnn import generate_code
from brickdnn.tracking import BrickDNNRun

from .helpers import chain

CLASSIFIER = chain(
    ("Input", {"shape": [4]}),
    ("Linear", {"out_features": 8}),
    ("Activation", {"function": "relu"}),
    ("Linear", {"out_features": 3}),
)
REGRESSION = chain(("Input", {"shape": [3]}), ("Linear", {"out_features": 1}))
PROXIES = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy")


@pytest.fixture(autouse=True)
def no_proxy(monkeypatch):
    """Requests to the fake server go straight to it; no token from the environment."""
    for name in (
        *PROXIES,
        "BRICKDNN_TOKEN",
        "BRICKDNN_URL",
        "BRICKDNN_PROJECT",
        "BRICKDNN_RUN_NAME",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setattr(BrickDNNRun, "TIMEOUT", 2.0)


class FakeServer:
    """A Brick DNN that answers the tracking endpoints, or fails as told."""

    def __init__(self):
        self.requests: list[tuple[str, str, dict, dict]] = []
        self.failures: list[int] = []  # status codes to answer with first
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _answer(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                server.requests.append((self.command, self.path, body, dict(self.headers)))
                if server.failures:
                    code = server.failures.pop(0)
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"detail": f"refused with {code}"}).encode())
                    return
                answer = {"id": "run-1", "status": "running"} if self.path.endswith("runs/") else {}
                data = json.dumps(answer).encode()
                self.send_response(201 if self.path.endswith("runs/") else 200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_POST = do_PATCH = _answer

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def paths(self) -> list[str]:
        return [f"{method} {path}" for method, path, _, _ in self.requests]

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def server():
    fake = FakeServer()
    yield fake
    fake.close()


# --------------------------------------------------------------------------- #
# The client
# --------------------------------------------------------------------------- #


def test_without_a_token_nothing_is_sent(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("no request without BRICKDNN_TOKEN")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    with BrickDNNRun.start(name="x", config={"lr": 1}) as run:
        assert run.active is False and run.id is None
        run.log(epoch=1, train_loss=0.5)
        run.flush()
    run.finish(summary={"test_loss": 1})


def test_a_run_is_created_logged_in_batches_and_finished(server):
    with BrickDNNRun.start(
        name="MLP",
        config={"lr": 0.01},
        dataset="d.csv",
        task="regression",
        epochs=3,
        url=server.url,
        token="bdnn_test",
    ) as run:
        run.INTERVAL = 3600  # send only when the batch is full or at the end
        run.BATCH = 2
        for epoch in range(1, 4):
            run.log(epoch=epoch, train_loss=1 / epoch, val_loss=float("nan"), note="text")
        run.finish(summary={"test_rmse": 1.5, "bad": float("inf")})
    assert server.paths() == [
        "POST /api/tracking/runs/",
        "POST /api/tracking/runs/run-1/log/",
        "POST /api/tracking/runs/run-1/log/",
        "POST /api/tracking/runs/run-1/finish/",
    ]
    _, _, created, headers = server.requests[0]
    assert headers["Authorization"] == "Bearer bdnn_test"
    assert created == {
        "name": "MLP",
        "config": {"lr": 0.01},
        "dataset": "d.csv",
        "task": "regression",
        "epochs": 3,
        "client": "brickdnn-tracking/1",
    }
    first = server.requests[1][2]["points"]
    assert [p["epoch"] for p in first] == [1, 2]
    assert "val_loss" not in first[0] and "note" not in first[0]  # NaN and text left out
    assert first[0]["train_loss"] == 1.0 and first[0]["time"] >= 0
    assert server.requests[3][2] == {"status": "completed", "summary": {"test_rmse": 1.5}}


def test_the_environment_names_the_server_network_and_run(monkeypatch, server):
    monkeypatch.setenv("BRICKDNN_TOKEN", "bdnn_env")
    monkeypatch.setenv("BRICKDNN_URL", server.url)
    monkeypatch.setenv("BRICKDNN_PROJECT", "12")
    monkeypatch.setenv("BRICKDNN_RUN_NAME", "From the shell")
    run = BrickDNNRun.start(name="ignored", url="https://elsewhere.example")
    assert run.url == server.url
    body = server.requests[0][2]
    assert body["project"] == 12 and body["name"] == "From the shell"


def test_server_errors_warn_once_keep_the_points_and_give_up(server, capsys):
    run = BrickDNNRun.start(url=server.url, token="bdnn_test")
    run.RETRY_SECONDS = 0
    run.INTERVAL = 0
    server.failures = [500, 503]
    run.log(epoch=1, loss=1.0)  # fails: kept
    run.log(epoch=2, loss=0.9)  # fails again: kept
    assert run.active and len(run.points) == 2
    run.log(epoch=3, loss=0.8)  # works: all three arrive
    assert run.points == []
    assert [len(r[2]["points"]) for r in server.requests if r[1].endswith("/log/")] == [1, 2, 3]
    err = capsys.readouterr().err
    assert err.count("could not log") == 1 and "refused with 500" in err
    server.failures = [500, 500, 500]
    for epoch in range(4, 8):
        run.log(epoch=epoch, loss=0.5)
    assert run.active is False and run.points == []  # gave up after three failures
    sent = len(server.requests)
    run.log(epoch=9, loss=0.1)
    run.finish()
    assert len(server.requests) == sent


def test_a_refused_token_switches_logging_off_at_once(server, capsys):
    server.failures = [401]
    run = BrickDNNRun.start(url=server.url, token="bdnn_wrong")
    assert run.active is False
    run.log(epoch=1, loss=1)
    run.finish()
    assert len(server.requests) == 1
    assert "refused with 401" in capsys.readouterr().err


def test_a_run_stopped_in_the_app_stops_logging(server):
    run = BrickDNNRun.start(url=server.url, token="bdnn_test")
    run.INTERVAL = 0
    server.failures = [409]
    run.log(epoch=1, loss=1)
    assert run.active is False


def test_no_server_at_all_never_stops_the_training(capsys):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]  # closed again: nothing listens there
    run = BrickDNNRun.start(url=f"http://127.0.0.1:{port}", token="bdnn_test")
    run.RETRY_SECONDS = 0
    for epoch in range(1, 6):
        run.log(epoch=epoch, loss=1.0)
    run.finish()
    assert run.active is False
    assert capsys.readouterr().err.count("could not log") == 1


def test_a_server_that_was_down_at_the_start_gets_the_kept_points(server, monkeypatch):
    monkeypatch.setattr(BrickDNNRun, "RETRY_SECONDS", 0)
    monkeypatch.setattr(BrickDNNRun, "INTERVAL", 0)
    server.failures = [502]
    run = BrickDNNRun.start(url=server.url, token="bdnn_test")
    assert run.active and run.id is None
    run.log(epoch=1, loss=1.0)
    assert run.id == "run-1"
    assert server.paths()[-1] == "POST /api/tracking/runs/run-1/log/"


def test_errors_and_ctrl_c_end_the_run_as_failed_and_stopped(server):
    with pytest.raises(ZeroDivisionError), BrickDNNRun.start(url=server.url, token="bdnn_test"):
        1 / 0  # noqa: B018
    assert server.requests[-1][2]["status"] == "failed"
    assert "ZeroDivisionError" in server.requests[-1][2]["message"]
    with pytest.raises(KeyboardInterrupt), BrickDNNRun.start(url=server.url, token="bdnn_test"):
        raise KeyboardInterrupt
    assert server.requests[-1][2]["status"] == "stopped"


def test_tensors_and_numpy_numbers_are_logged():
    np = pytest.importorskip("numpy")
    assert BrickDNNRun._number(np.float32(0.5)) == 0.5
    assert BrickDNNRun._number(np.int64(3)) == 3
    assert BrickDNNRun._number(np.array([1.0, 2.0])) is None  # not one number
    assert BrickDNNRun._number(True) is None


# --------------------------------------------------------------------------- #
# The generated script
# --------------------------------------------------------------------------- #


def test_the_script_embeds_the_client_but_no_token():
    code = generate_code(CLASSIFIER, {"tracking_project": 12, "tracking_url": "https://my.server"})
    assert "class BrickDNNRun:" in code and "import urllib.request" in code
    assert 'BRICKDNN_URL = "https://my.server"' in code and "BRICKDNN_PROJECT = 12" in code
    assert "export BRICKDNN_TOKEN=<your token>" in code
    assert "BRICKDNN_TOKEN=bdnn_" not in code and "bdnn_" not in code.replace("bdnn_...", "")
    compile(code, "<generated>", "exec")
    off = generate_code(CLASSIFIER, {"log_to_brickdnn": False})
    assert "BrickDNNRun" not in off and "urllib" not in off and "BRICKDNN_TOKEN" not in off
    assert "--test-split" in off
    # Networks without the CSV loop have nothing to log.
    images = chain(
        ("Input", {"shape": [1, 8, 8]}), ("Flatten", {}), ("Linear", {"out_features": 2})
    )
    assert "BrickDNNRun" not in generate_code(images)


@pytest.mark.parametrize(
    "options",
    [
        {"tracking_url": "javascript:alert(1)"},
        {"tracking_url": 'https://x"; import os'},
        {"tracking_project": "12"},
        {"tracking_project": 0},
    ],
)
def test_the_tracking_options_are_checked(options):
    with pytest.raises(ValueError):
        generate_code(CLASSIFIER, options)


@pytest.mark.django_db
def test_the_api_writes_the_servers_address_into_the_script(settings, client):
    settings.PUBLIC_URL = "https://brickdnn.example"
    answer = client.post(
        "/api/generate/", {"graph": CLASSIFIER, "options": {}}, content_type="application/json"
    )
    assert 'BRICKDNN_URL = "https://brickdnn.example"' in answer.json()["code"]
    off = client.post(
        "/api/generate/",
        {"graph": CLASSIFIER, "options": {"log_to_brickdnn": False}},
        content_type="application/json",
    )
    assert "BRICKDNN_URL" not in off.json()["code"]


def _csv(path, rows=60, regression=False):
    lines = ["a,b,c,d,y"] if not regression else ["a,b,c,y"]
    for i in range(rows):
        label = i % 3
        if regression:
            lines.append(f"{i % 7},{i % 5},{i % 3},{(i % 7) * 2 + 1}")
        else:
            lines.append(f"{label + i % 5 / 10},{label * 2},{i % 4},{label - 1},{label}")
    path.write_text("\n".join(lines) + "\n")
    return path


def _run(script, *args, env=None, cwd):
    environment = {k: v for k, v in os.environ.items() if k not in PROXIES}
    environment.pop("BRICKDNN_TOKEN", None)
    environment.update({"NO_PROXY": "127.0.0.1,localhost", **(env or {})})
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        env=environment,
        timeout=300,
    )


@pytest.mark.parametrize("graph,regression", [(CLASSIFIER, False), (REGRESSION, True)])
def test_the_script_runs_offline_with_the_same_results(tmp_path, graph, regression):
    torch = pytest.importorskip("torch")
    data = _csv(tmp_path / "data.csv", regression=regression)
    outputs = []
    for logging in (True, False):
        script = tmp_path / f"model_{logging}.py"
        script.write_text(generate_code(graph, {"log_to_brickdnn": logging}))
        weights = tmp_path / f"weights_{logging}.pt"
        done = _run(
            script, "--csv", str(data), "--epochs", "3", "--save", str(weights), cwd=tmp_path
        )
        assert done.returncode == 0, done.stderr
        assert "Brick DNN" not in done.stderr
        outputs.append(done.stdout.replace(str(weights), "W"))
        if logging:
            on = torch.load(weights, weights_only=True)["state_dict"]
        else:
            off = torch.load(weights, weights_only=True)["state_dict"]
    assert outputs[0] == outputs[1]
    assert all(torch.equal(on[k], off[k]) for k in on)


def test_the_script_logs_to_a_fake_server(tmp_path, server):
    pytest.importorskip("torch")
    data = _csv(tmp_path / "data.csv")
    script = tmp_path / "model.py"
    script.write_text(generate_code(CLASSIFIER, {"tracking_project": 4}))
    done = _run(
        script,
        "--csv",
        str(data),
        "--epochs",
        "3",
        "--test-split",
        "0.2",
        "--save",
        str(tmp_path / "w.pt"),
        env={"BRICKDNN_TOKEN": "bdnn_fake", "BRICKDNN_URL": server.url},
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    assert "Test: loss" in done.stdout
    created = server.requests[0][2]
    assert created["project"] == 4 and created["task"] == "classification"
    assert created["dataset"] == "data.csv" and created["config"]["test_split"] == 0.2
    assert created["config"]["loss"] == "cross_entropy"
    logged = [p for r in server.requests if r[1].endswith("/log/") for p in r[2]["points"]]
    assert [p["epoch"] for p in logged] == [1, 2, 3]
    assert {"train_loss", "val_loss", "val_accuracy"} <= set(logged[0])
    final = server.requests[-1][2]
    assert final["status"] == "completed"
    assert {"test_loss", "test_accuracy", "val_loss"} <= set(final["summary"])


def test_the_script_keeps_training_when_the_server_fails(tmp_path, server):
    pytest.importorskip("torch")
    server.failures = [500] * 20
    data = _csv(tmp_path / "data.csv")
    script = tmp_path / "model.py"
    script.write_text(generate_code(CLASSIFIER))
    done = _run(
        script,
        "--csv",
        str(data),
        "--epochs",
        "2",
        "--save",
        str(tmp_path / "w.pt"),
        env={"BRICKDNN_TOKEN": "bdnn_fake", "BRICKDNN_URL": server.url},
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    assert done.stderr.count("could not log") == 1
    assert "Saved the weights" in done.stdout


@pytest.mark.django_db(transaction=True)
def test_the_script_logs_into_brick_dnn(tmp_path, live_server, settings):
    """End to end: the generated script, a logging-only token and the real API."""
    pytest.importorskip("torch")
    from django.contrib.auth import get_user_model

    from brick_dnn_core.api_tokens import create_token
    from brick_dnn_core.models import Project, TrainingRun

    settings.TRAINING_DIR = tmp_path / "training"
    user = get_user_model().objects.create_user("ada", email="ada@example.com", password="x")
    project = Project.objects.create(owner=user, name="Iris", graph=CLASSIFIER, updated_by=user)
    _, key = create_token(user, "laptop", scope="runs:log")
    data = _csv(tmp_path / "iris.csv")
    script = tmp_path / "model.py"
    script.write_text(generate_code(CLASSIFIER, {"tracking_project": project.pk}))
    done = _run(
        script,
        "--csv",
        str(data),
        "--epochs",
        "4",
        "--test-split",
        "0.2",
        "--save",
        str(tmp_path / "w.pt"),
        env={"BRICKDNN_TOKEN": key, "BRICKDNN_URL": live_server.url},
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    assert "Brick DNN: logging this run" in done.stderr
    run = TrainingRun.objects.get(kind="external")
    assert run.status == "completed" and run.project_id == project.pk
    assert run.epochs_done == 4 and run.dataset["name"] == "iris.csv"
    assert run.config["epochs"] == 4 and run.config["optimizer"] == "adam"
    assert run.metrics["test_accuracy"] is not None
    assert run.api_token.scope == "runs:log"
