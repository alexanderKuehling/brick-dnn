"""Publishing runs to the Hugging Face Hub (hub_client.py, hub_publish.py, the MCP tool
publish_to_huggingface and its approval), against a fake Hub (hub_fake.py): the requests and
the commit, the model card, the errors, and that the access token is never stored, logged or
sent anywhere but to the Hub."""

# ruff: noqa: F811 (fixtures are imported from other test modules)

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import logging
import os
import pickle
import subprocess
import sys

import pytest
from django.core.management import call_command
from rest_framework.throttling import ScopedRateThrottle

from brick_dnn_core import approvals, hub_publish
from brick_dnn_core.hub_client import HubClient, HubError, HubToken, clean
from brick_dnn_core.models import AgentAction, ApprovalRequest, Project, ProjectMember, TrainingRun
from brickdnn.training import torch_installed
from brickdnn.training.state import read_state, write_state

from .hub_fake import GOOD_TOKEN, OTHER_TOKEN, READ_TOKEN, FakeHub
from .test_budgets import limits, token_client  # noqa: F401 (fixtures)
from .test_mcp import call
from .test_predict import IRIS_PREPROCESSING, client_for, iris_run, team  # noqa: F401
from .test_training_api import training_dir  # noqa: F401 (fixture)

pytestmark = pytest.mark.django_db

EVALUATION = {
    "task": "classification",
    "epoch": 1,
    "val": {
        "size": 30,
        "accuracy": 0.9,
        "balanced_accuracy": 0.88,
        "macro_f1": 0.87,
        "log_loss": 0.31,
    },
    "test": {
        "size": 30,
        "accuracy": 0.93,
        "balanced_accuracy": 0.92,
        "macro_f1": 0.91,
        "log_loss": 0.25,
    },
    "train": {"loss": 0.2, "accuracy": 0.95},
    "class_names": ["setosa", "versicolor", "virginica"],
}


@pytest.fixture
def hub(settings):
    fake = FakeHub()
    settings.HUGGINGFACE_URL = fake.url
    settings.HUGGINGFACE_ENABLED = True
    settings.HUGGINGFACE_PROXY = ""
    settings.HUGGINGFACE_MAX_UPLOAD_MB = 50
    yield fake
    fake.close()


@pytest.fixture
def ada(team):
    return team[1]["ada"]


@pytest.fixture
def web(ada):
    return client_for(ada)


def finished(client, training_dir, project=None, evaluation=EVALUATION, real_weights=True):
    """A completed run of the Iris network with preprocessing, results and (if PyTorch is
    installed) real weights."""
    run_id = iris_run(client, training_dir, project)
    directory = training_dir / "runs" / run_id
    state = read_state(directory)
    state.update(
        task="classification",
        train_size=90,
        val_size=30,
        test_size=30,
        weights_epoch=1,
        loss={"name": "cross_entropy", "label": "cross-entropy"},
        evaluation=evaluation,
    )
    write_state(directory, state)
    if real_weights and torch_installed():
        import torch

        weights = {
            "layers.0.weight": torch.arange(32, dtype=torch.float32).reshape(8, 4),
            "layers.0.bias": torch.zeros(8),
            "layers.2.weight": torch.ones(3, 8),
            "layers.2.bias": torch.tensor([0.5, -0.5, 1.0]),
        }
        torch.save(weights, directory / "weights.pt")
    return run_id


def publish(client, run_id, **body):
    return client.post(
        f"/api/training/runs/{run_id}/huggingface/",
        {"hf_token": GOOD_TOKEN, **body},
        format="json",
    )


def committed(hub) -> dict[str, dict]:
    """The files of the last commit: path -> {"kind", "content" | "oid"}."""
    files = {}
    for line in hub.commits[-1]["lines"]:
        value = line["value"]
        if line["key"] == "file":
            files[value["path"]] = {"kind": "file", "content": base64.b64decode(value["content"])}
        elif line["key"] == "lfsFile":
            files[value["path"]] = {"kind": "lfs", "oid": value["oid"], "size": value["size"]}
    return files


def front_matter(card: str) -> str:
    assert card.startswith("---\n")
    return card.split("---\n")[1]


# --------------------------------------------------------------------------- #
# The pieces: safetensors, the token, the client
# --------------------------------------------------------------------------- #


def test_safetensors_file_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    from brickdnn.training import safetensors_file

    tensors = {
        "b.weight": torch.tensor([[1.5, -2.0], [0.25, 4.0]]),
        "a.count": torch.tensor([1, 2, 3], dtype=torch.int64),
        "c.half": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
        "d.mask": torch.tensor([True, False]),
    }
    torch.save({"state_dict": tensors, "preprocessing": {"kind": "demo"}}, tmp_path / "w.pt")
    result = safetensors_file.convert(tmp_path / "w.pt", tmp_path / "m.safetensors")
    assert result["tensors"] == 4
    header = safetensors_file.read_header(tmp_path / "m.safetensors")
    assert header["__metadata__"] == {"format": "pt"}
    assert header["b.weight"]["dtype"] == "F32" and header["b.weight"]["shape"] == [2, 2]
    assert header["a.count"]["dtype"] == "I64"
    assert header["c.half"]["dtype"] == "BF16" and header["d.mask"]["dtype"] == "BOOL"
    raw = (tmp_path / "m.safetensors").read_bytes()
    length = int.from_bytes(raw[:8], "little")
    assert length % 8 == 0
    data = raw[8 + length :]
    begin, end = header["b.weight"]["data_offsets"]
    assert torch.frombuffer(bytearray(data[begin:end]), dtype=torch.float32).tolist() == [
        1.5,
        -2.0,
        0.25,
        4.0,
    ]
    begin, end = header["c.half"]["data_offsets"]
    restored = torch.frombuffer(bytearray(data[begin:end]), dtype=torch.bfloat16)
    assert restored.tolist() == [1.0, 2.0]
    # Offsets follow each other without gaps, sorted by name.
    names = sorted(k for k in header if k != "__metadata__")
    assert [header[n]["data_offsets"][0] for n in names][0] == 0
    assert header[names[-1]]["data_offsets"][1] == len(data)
    # Something else than a state_dict is refused.
    torch.save({"x": "not a tensor"}, tmp_path / "bad.pt")
    assert safetensors_file.main([str(tmp_path / "bad.pt"), str(tmp_path / "o")]) == 1


def test_the_token_never_shows_its_value():
    token = HubToken(f"  {GOOD_TOKEN}\n")
    assert token.authorization() == f"Bearer {GOOD_TOKEN}"
    assert GOOD_TOKEN not in repr(token) and GOOD_TOKEN not in str(token)
    assert GOOD_TOKEN not in f"{[token]} { ({'t': token}) }"
    with pytest.raises(TypeError):
        pickle.dumps(token)
    for bad in (None, "", "   ", 12, "has space", "x" * 401):
        with pytest.raises(ValueError):
            HubToken(bad)
    assert HubToken.optional("") is None
    assert clean(f"bad token {GOOD_TOKEN}!") == "bad token [token]!"


def test_the_token_goes_to_the_hub_only(hub):
    client = HubClient(hub.url, HubToken(GOOD_TOKEN))
    assert client.whoami() == {"name": "ada-hf", "orgs": ["team"], "role": "write"}
    assert hub.requests[-1]["headers"]["Authorization"] == f"Bearer {GOOD_TOKEN}"
    # Another host (here: the storage server) never gets it.
    answer = client.request(
        "PUT", f"{hub.storage.url}/upload/{hashlib.sha256(b'').hexdigest()}", body=b""
    )
    assert answer.status == 200
    assert "Authorization" not in hub.storage_requests[-1]["headers"]
    with pytest.raises(HubError) as error:
        HubClient(hub.url, HubToken(OTHER_TOKEN)).whoami()
    assert error.value.status == 400 and error.value.code == "token_invalid"


def test_unreachable_hub():
    client = HubClient("http://127.0.0.1:9", HubToken(GOOD_TOKEN), timeout=2)
    with pytest.raises(HubError) as error:
        client.whoami()
    assert error.value.code == "unreachable" and error.value.status == 502


# --------------------------------------------------------------------------- #
# Preview and publishing in the web app
# --------------------------------------------------------------------------- #


def test_capabilities(web, hub, training_dir):
    data = web.get("/api/training/").data["huggingface"]
    assert data["publish"] is True and data["max_upload_mb"] == 50
    assert "apache-2.0" in data["licenses"] and data["default_license"] == "apache-2.0"


def test_preview_shows_files_and_model_card_without_sending_anything(web, hub, training_dir):
    run_id = finished(web, training_dir)
    response = web.get(f"/api/training/runs/{run_id}/huggingface/?repo=iris-demo")
    assert response.status_code == 200, response.data
    data = response.data
    paths = [f["path"] for f in data["files"]]
    assert paths[0] == "README.md"
    assert {"network.json", "preprocessing.json", "net.py"} <= set(paths)
    assert "model.safetensors" in paths
    assert data["repo"] == "iris-demo" and data["default_repo"] == "net"
    assert data["can_publish"] is True and data["too_large"] is False
    assert "your-username/iris-demo" in data["model_card"]
    assert hub.requests == []


def test_publish_a_neural_network(web, hub, training_dir, caplog):
    caplog.set_level(logging.DEBUG)
    run_id = finished(web, training_dir)
    response = publish(web, run_id, repo="iris-net", private=True, license="mit")
    assert response.status_code == 201, response.data
    data = response.data
    assert data["repo_id"] == "ada-hf/iris-net" and data["created"] is True
    assert data["url"] == f"{hub.url}/ada-hf/iris-net" and data["commit"] == "abc123"
    assert hub.repos["ada-hf/iris-net"] == {"private": True}

    create = next(r for r in hub.requests if r["path"] == "/api/repos/create")
    assert json.loads(create["body"]) == {
        "type": "model",
        "name": "iris-net",
        "private": True,
        "organization": "ada-hf",
    }
    files = committed(hub)
    assert set(files) == {
        "README.md",
        "model.safetensors" if torch_installed() else "model.pt",
        "net.py",
        "network.json",
        "preprocessing.json",
    }
    weights = files.get("model.safetensors") or files["model.pt"]
    assert weights["kind"] == "lfs"
    assert hashlib.sha256(hub.lfs[weights["oid"]]).hexdigest() == weights["oid"]
    if torch_installed():
        from brickdnn.training.safetensors_file import read_header

        stored = training_dir / "lfs.safetensors"
        stored.write_bytes(hub.lfs[weights["oid"]])
        assert read_header(stored)["layers.0.weight"]["shape"] == [8, 4]
    assert json.loads(files["network.json"]["content"])["nodes"][0]["type"] == "Input"
    assert json.loads(files["preprocessing.json"]["content"]) == IRIS_PREPROCESSING
    assert b"class NeuralNetwork" in files["net.py"]["content"]

    card = files["README.md"]["content"].decode()
    meta = front_matter(card)
    assert "library_name: pytorch" in meta and "license: mit" in meta
    assert '- "brickdnn"' in meta and '- "tabular-classification"' in meta
    assert "pipeline_tag: tabular-classification" in meta
    assert "model-index:" in meta and "split: test" in meta and "value: 0.93" in meta
    assert '- "accuracy"' in meta and '- "f1"' in meta
    assert "# Net" in card and "https://brickdnn.com" in card
    assert "| Test | 0.93 | 0.92 | 0.91 | 0.25 |" in card
    assert "Features in this order: sepal_length" in card
    assert 'snapshot_download("ada-hf/iris-net")' in card
    assert '      name: "Macro F1 (test)"' in meta
    assert "While the repository is private, log in first" in card
    assert "python net.py --help" in card
    if torch_installed():
        assert 'model.load_state_dict(load_file(folder / "model.safetensors"))' in card
        assert "from safetensors.torch import load_file" in card
        assert "`pip install huggingface_hub torch safetensors`" in card

    # The token went to the Hub's API only, never to the storage of the uploads.
    assert all(r["headers"].get("Authorization") == f"Bearer {GOOD_TOKEN}" for r in hub.requests)
    assert hub.storage_requests and all(
        "Authorization" not in r["headers"] for r in hub.storage_requests
    )
    assert all(
        {k.lower(): v for k, v in r["headers"].items()}.get("x-amz-test") == "1"
        for r in hub.storage_requests
    )
    # ... and was neither stored nor logged.
    dump = io.StringIO()
    call_command("dumpdata", stdout=dump)
    assert GOOD_TOKEN not in dump.getvalue()
    assert GOOD_TOKEN not in caplog.text
    assert not list((training_dir / "runs" / run_id).glob("*.safetensors"))


def test_weights_that_cannot_be_converted_are_uploaded_as_they_are(web, hub, training_dir):
    run_id = finished(web, training_dir, real_weights=False)  # b"fake weights"
    response = publish(web, run_id)
    assert response.status_code == 201, response.data
    assert response.data["repo_id"] == "ada-hf/net"  # the default name
    files = committed(hub)
    assert "model.pt" in files and "model.safetensors" not in files
    assert hub.lfs[files["model.pt"]["oid"]] == b"fake weights"
    card = files["README.md"]["content"].decode()
    assert 'torch.load(folder / "model.pt"' in card
    # Published again: the Hub has the LFS file already, nothing is uploaded twice.
    uploads = len(hub.storage_requests)
    response = publish(web, run_id, update_existing=True)
    assert response.status_code == 201 and response.data["created"] is False
    assert len(hub.storage_requests) == uploads


def _model_run(ada, training_dir, family: str) -> TrainingRun:
    """A finished classical or hybrid run (files written by hand): the plan of an Iris run
    with the classical model added."""
    from brickdnn import EXAMPLES

    neural = TrainingRun.objects.get(pk=finished(client_for(ada), training_dir))
    plan = {
        **neural.plan,
        "classical": {"type": "SVC", "estimator": "SVC", "label": "Support vector classifier"},
    }
    graph = {
        "version": 1,
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [4]}},
            {"id": "m", "type": "SVC", "params": {}},
        ],
        "edges": [{"source": "in", "target": "m"}],
    }
    if family == "hybrid":
        plan["hybrid"] = {"features": 8}
        graph = EXAMPLES["hybrid_mlp_gradient_boosting"]["graph"]
    run = TrainingRun.objects.create(
        owner=ada,
        name=f"Iris {family}",
        graph=graph,
        dataset={"kind": "upload", "id": 1, "name": "iris.csv"},
        plan=plan,
        status="completed",
        has_weights=True,
    )
    directory = training_dir / "runs" / str(run.pk)
    directory.mkdir(parents=True)
    (directory / "model.joblib").write_bytes(b"joblib bundle")
    if family == "hybrid":
        (directory / "weights.pt").write_bytes(b"feature layers")
    write_state(
        directory,
        {
            "status": "completed",
            "family": family,
            "task": "classification",
            "weights": True,
            "preprocessing": IRIS_PREPROCESSING,
            "evaluation": EVALUATION,
        },
    )
    return run


@pytest.mark.parametrize("family", ["classical", "hybrid"])
def test_publish_classical_and_hybrid_models(web, ada, hub, training_dir, family):
    run = _model_run(ada, training_dir, family)
    response = publish(web, run.pk, repo=f"iris-{family}", private=False)
    assert response.status_code == 201, response.data
    files = committed(hub)
    assert hub.lfs[files["model.joblib"]["oid"]] == b"joblib bundle"
    card = files["README.md"]["content"].decode()
    if family == "classical":
        assert "library_name: sklearn" in front_matter(card)
        assert "model.features.pt" not in files
        assert 'joblib.load(folder / "model.joblib")' in card
        assert "`pip install huggingface_hub scikit-learn joblib`" in card
        assert "log in first" not in card  # a public repository
    else:
        assert "library_name: pytorch" in front_matter(card)
        assert '- "scikit-learn"' in front_matter(card)
        assert hub.lfs[files["model.features.pt"]["oid"]] == b"feature layers"
        assert 'torch.load(folder / "model.features.pt"' in card
    assert hub.repos[f"ada-hf/iris-{family}"] == {"private": False}


# What stands in for the packages of the model card's code that are not installed here.
HUB_STUB = "import os\n\n\ndef snapshot_download(repo_id):\n    return os.environ['HUB_FOLDER']\n"
SAFETENSORS_STUB = '''
import json
import struct

import torch

DTYPES = {"F32": torch.float32, "F64": torch.float64, "I64": torch.int64, "BOOL": torch.bool}


def load_file(path):
    """Reads the format as its specification describes it (not with the code under test)."""
    raw = open(path, "rb").read()
    (length,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + length])
    data = raw[8 + length :]
    tensors = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        begin, end = entry["data_offsets"]
        values = torch.frombuffer(bytearray(data[begin:end]), dtype=DTYPES[entry["dtype"]])
        tensors[name] = values.reshape(entry["shape"])
    return tensors
'''


def test_the_code_of_the_model_card_loads_the_published_files_and_predicts(
    web, hub, training_dir, tmp_path
):
    torch = pytest.importorskip("torch")
    from brickdnn.codegen import generate_code

    run_id = finished(web, training_dir)
    run = TrainingRun.objects.get(pk=run_id)
    # The weights of the network as the generated code builds it.
    namespace: dict = {"__name__": "net"}
    exec(generate_code(run.graph), namespace)
    torch.manual_seed(0)
    model = namespace["NeuralNetwork"]()
    torch.save(model.state_dict(), training_dir / "runs" / run_id / "weights.pt")

    assert publish(web, run_id, repo="runs", private=False).status_code == 201
    files = committed(hub)
    folder = tmp_path / "snapshot"
    folder.mkdir()
    for path, file in files.items():
        (folder / path).write_bytes(
            file["content"] if file["kind"] == "file" else hub.lfs[file["oid"]]
        )
    card = (folder / "README.md").read_text()
    code = card.split("```python\n")[1].split("```")[0]
    stubs = tmp_path / "stubs"
    (stubs / "safetensors").mkdir(parents=True)
    (stubs / "huggingface_hub.py").write_text(HUB_STUB)
    paths = [str(stubs)]
    if importlib.util.find_spec("safetensors") is None:
        (stubs / "safetensors" / "__init__.py").write_text("")
        (stubs / "safetensors" / "torch.py").write_text(SAFETENSORS_STUB)
    else:
        (stubs / "safetensors").rmdir()
    work = tmp_path / "elsewhere"  # the code does not depend on where it runs
    work.mkdir()
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=work,
        timeout=120,
        env={**os.environ, "HUB_FOLDER": str(folder), "PYTHONPATH": os.pathsep.join(paths)},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    # The same answer as the model in this process for the example row of the card.
    mean = torch.tensor(IRIS_PREPROCESSING["mean"])
    with torch.no_grad():
        expected = model.eval()((mean - mean).unsqueeze(0)).argmax(dim=1).item()
    assert result.stdout.strip() == str([IRIS_PREPROCESSING["class_names"][expected]])


def _run_like(ada, training_dir, client, example: str, dataset: str, name: str, **state):
    """A finished neural run of an example network on a built-in dataset: the plan as the
    training check makes it, the files written by hand."""
    from brickdnn import EXAMPLES

    graph = json.loads(json.dumps(EXAMPLES[example]["graph"]))
    body = {"graph": graph, "dataset": {"kind": "builtin", "id": dataset}}
    check = client.post("/api/training/check/", body, format="json").data
    for fix in check["fixes"]:  # e.g. the number of classes of the last layer
        node = next(n for n in graph["nodes"] if n["id"] == fix["node"])
        node["params"] = {**node["params"], **fix["params"]}
    check = client.post("/api/training/check/", body, format="json").data
    assert check["ok"], check["errors"]
    run = TrainingRun.objects.create(
        owner=ada,
        name=name,
        graph=graph,
        dataset={"kind": "builtin", "id": dataset, "name": dataset.title()},
        config={"epochs": 2, "val_split": 0.2, "test_split": 0.0, "loss": "cross_entropy"},
        plan=check["plan"],
        status="completed",
        has_weights=True,
    )
    directory = training_dir / "runs" / str(run.pk)
    directory.mkdir(parents=True)
    (directory / "weights.pt").write_bytes(b"not a state_dict")
    write_state(directory, {"status": "completed", "family": "neural", "weights": True, **state})
    return run


def test_model_card_of_a_segmentation(web, ada, hub, training_dir):
    preprocessing = {
        "kind": "image",
        "channels": 3,
        "height": 64,
        "width": 64,
        "layout": "chw",
        "resize": "bilinear",
        "scale": "pixel / 255",
        "mean": [0.5556, 0.5575, 0.5419],
        "std": [0.1903, 0.192, 0.2007],
        "class_names": ["background", "square", "circle", "triangle"],
        "task": "segmentation",
        "mask": {"resize": "nearest", "binary": False, "ignore_index": None},
        "dataset": "shapes",
    }
    run = _run_like(
        ada,
        training_dir,
        web,
        "unet",
        "shapes",
        "Shapes U-Net",
        task="segmentation",
        preprocessing=preprocessing,
        segmentation={
            "validation": {"epoch": 2, "iou": 0.61, "dice": 0.72, "pixel_accuracy": 0.9},
            "test": {"iou": 0.58, "dice": 0.7, "pixel_accuracy": 0.88},
        },
    )
    response = web.get(f"/api/training/runs/{run.pk}/huggingface/?repo=shapes-unet")
    assert response.status_code == 200, response.data
    card = response.data["model_card"]
    meta = front_matter(card)
    assert "pipeline_tag: image-segmentation" in meta and "- type: mean_iou" in meta
    assert '      name: "Mean IoU (validation)"' in meta and "value: 0.58" in meta
    assert "| Test | 0.58 | 0.7 | 0.88 |" in card
    assert "Classes: background, square, circle, triangle" in card
    # Images: Pillow is needed, the user's own image stays where the code runs, and the
    # script of an image network has no command line.
    assert "torch safetensors pillow`" in card or "torch pillow`" in card
    assert 'Image.open("example.png")' in card and "stands for an image of your own" in card
    assert "--help" not in card and "Test share" not in card
    assert "shapes_u_net.py" in [f["path"] for f in response.data["files"]]


def test_model_card_of_a_text_classifier(web, ada, hub, training_dir):
    from brickdnn.training.textsets import TEXT_DATASETS

    assert "sentiment" in TEXT_DATASETS
    preprocessing = {
        "kind": "tokens",
        "dataset": "sentiment",
        "length": 24,
        "vocabulary": ["<pad>", "<unk>", "the", "food", "was", "good", "bad"],
        "pad_id": 0,
        "unknown_id": 1,
        "class_names": ["negative", "positive"],
    }
    run = _run_like(
        ada,
        training_dir,
        web,
        "lstm_sentiment",
        "sentiment",
        "Sentiment LSTM",
        task="classification",
        preprocessing=preprocessing,
        evaluation={
            "task": "classification",
            "val": {"accuracy": 0.9, "threshold_curve": {"roc_auc": 0.95}},
            "test": {"accuracy": 0.88},
        },
    )
    response = web.get(f"/api/training/runs/{run.pk}/huggingface/")
    assert response.status_code == 200, response.data
    assert response.data["default_repo"] == "sentiment-lstm"
    card = response.data["model_card"]
    meta = front_matter(card)
    assert "pipeline_tag: text-classification" in meta and '- "roc_auc"' in meta
    assert "| Validation | 0.9 | 0.95 |" in card and "| Test | 0.88 | – |" in card
    assert "def encode(text):" in card and "import re" in card
    assert "Classes: negative, positive" in card


def test_errors(web, hub, training_dir, settings):
    run_id = finished(web, training_dir, real_weights=False)
    response = publish(web, run_id, hf_token=OTHER_TOKEN)
    # Not 401: that would log the user out of Brick DNN.
    assert response.status_code == 400 and response.data["code"] == "token_invalid"
    assert "did not accept the access token" in response.data["detail"]
    response = publish(web, run_id, hf_token=READ_TOKEN)
    assert response.status_code == 403 and response.data["code"] == "token_read_only"
    response = publish(web, run_id, repo="taken")
    assert response.status_code == 409 and response.data["code"] == "name_taken"
    assert "exists already" in response.data["detail"]
    response = publish(web, run_id, repo="someone-else/model")
    assert response.status_code == 403 and response.data["code"] == "forbidden"
    for body in ({"repo": "bad name!"}, {"repo": "a/b/c"}, {"license": "wtfpl"}, {"private": "no"}):
        response = publish(web, run_id, **body)
        assert response.status_code == 400, body
    assert publish(web, run_id, hf_token="").status_code == 400
    assert hub.commits == []

    hub.commit_limit = 10
    response = publish(web, run_id, repo="big")
    assert response.status_code == 413 and response.data["code"] == "too_large"
    # The Hub refused the commit after the repository was created: the answer says so.
    assert response.data["created_repo"] == "ada-hf/big" and "ada-hf/big" in hub.repos
    assert "was created and is still empty" in response.data["detail"]
    hub.commit_limit = None
    assert publish(web, run_id, repo="big").status_code == 409
    assert publish(web, run_id, repo="big", update_existing=True).status_code == 201

    settings.HUGGINGFACE_MAX_UPLOAD_MB = 0.000001
    before = len(hub.requests)
    response = publish(web, run_id, repo="bigger")
    assert response.status_code == 413 and "at most" in response.data["detail"]
    assert len(hub.requests) == before  # refused before anything was sent

    settings.HUGGINGFACE_ENABLED = False
    assert publish(web, run_id).status_code == 403
    assert web.get("/api/training/").data["huggingface"]["publish"] is False


def test_who_may_publish(team, hub, training_dir):
    project, users = team
    owner = client_for(users["ada"])
    run_id = finished(owner, training_dir, project=project, real_weights=False)
    assert publish(client_for(users["cy"]), run_id).status_code == 403  # a viewer
    assert publish(client_for(users["eve"]), run_id).status_code == 404  # a stranger
    response = publish(client_for(users["bob"]), run_id, repo="bobs")  # an editor
    assert response.status_code == 201, response.data
    assert Project.objects.filter(pk=project.pk).exists()
    assert ProjectMember.objects.filter(project=project).count() == 2


def test_rate_limit(web, hub, training_dir, monkeypatch):
    monkeypatch.setitem(ScopedRateThrottle.THROTTLE_RATES, "huggingface", "1/hour")
    run_id = finished(web, training_dir, real_weights=False)
    assert publish(web, run_id, repo="one").status_code == 201
    assert publish(web, run_id, repo="two").status_code == 429


def test_unfinished_runs_cannot_be_published(web, ada, hub, training_dir):
    run = TrainingRun.objects.create(owner=ada, name="Empty", status="failed", message="Boom")
    response = web.get(f"/api/training/runs/{run.pk}/huggingface/")
    assert response.status_code == 400 and "failed" in response.data["detail"]
    assert publish(web, run.pk).status_code == 400


# --------------------------------------------------------------------------- #
# API tokens and MCP
# --------------------------------------------------------------------------- #


def test_rest_api_token_whose_rule_covers_publishing_is_refused(ada, hub, training_dir, limits):
    run_id = finished(client_for(ada), training_dir, real_weights=False)
    client, token = token_client(ada)
    approvals.set_rule(ada, approvals.Rule(publishing=True), token=token)
    response = publish(client, run_id)
    assert response.status_code == 403 and "approval" in response.data["detail"]
    approvals.set_rule(ada, approvals.Rule(publishing=False), token=token)
    response = publish(client, run_id, repo="by-script")
    assert response.status_code == 201, response.data
    entry = AgentAction.objects.filter(tool="training-run-huggingface").latest("id")
    # Taken out of the request before anything else sees it.
    assert entry.arguments.get("hf_token", "[redacted]") == "[redacted]"
    assert GOOD_TOKEN not in json.dumps(entry.arguments) and entry.arguments["repo"] == "by-script"
    assert entry.summary == "Published training “Net” to Hugging Face"


def mcp_publish(client, run_id, **arguments):
    return call(client, "publish_to_huggingface", run_id=run_id, hf_token=GOOD_TOKEN, **arguments)


def test_mcp_publishes_without_approval_rule(ada, hub, training_dir, limits, caplog):
    caplog.set_level(logging.DEBUG)
    run_id = finished(client_for(ada), training_dir, real_weights=False)
    client, token = token_client(ada)
    approvals.set_rule(ada, approvals.OFF, token=token)
    result, text = mcp_publish(client, run_id, repo="from-mcp")
    assert "isError" not in result, text
    assert result["structuredContent"]["repo_id"] == "ada-hf/from-mcp"
    entry = AgentAction.objects.filter(tool="publish_to_huggingface").latest("id")
    assert entry.arguments["hf_token"] == "[redacted]"
    assert (
        entry.summary.startswith("Published training “Net” to Hugging Face as")
        and "ada-hf/from-mcp" in entry.summary
    )
    dump = io.StringIO()
    call_command("dumpdata", stdout=dump)
    assert GOOD_TOKEN not in dump.getvalue() and GOOD_TOKEN not in caplog.text
    tools = {t["name"]: t for t in call_tools(client)}
    annotations = tools["publish_to_huggingface"]["annotations"]
    assert annotations["openWorldHint"] is True and annotations["readOnlyHint"] is False


def call_tools(client):
    from .test_mcp import rpc

    return rpc(client, "tools/list").json()["result"]["tools"]


def test_mcp_publish_needs_approval_and_the_grant_works_once(
    ada, web, hub, training_dir, limits, caplog
):
    caplog.set_level(logging.DEBUG)
    run_id = finished(web, training_dir, real_weights=False)
    client, token = token_client(ada)
    approvals.set_rule(ada, approvals.Rule(publishing=True), token=token)

    result, text = mcp_publish(client, run_id, repo="approved", private=False)
    answer = json.loads(text)
    assert answer["approval_required"] is True, text
    assert hub.requests == []  # nothing was sent to Hugging Face
    approval = ApprovalRequest.objects.get(pk=answer["approval"]["id"])
    assert approval.action == "publish_to_huggingface" and approval.category == "publishing"
    assert approval.arguments == {
        "run_id": run_id,
        "repo": "approved",
        "private": False,
        "license": "apache-2.0",
        "update_existing": False,
    }
    assert "public model repository “approved”" in approval.summary
    assert "anyone can see" in approval.details["exposes"]
    # Before the decision the grant cannot be used.
    result, text = mcp_publish(
        client, run_id, repo="approved", private=False, approval_id=approval.pk
    )
    assert result["isError"] and "still waits" in text

    response = web.post(f"/api/approvals/{approval.pk}/approve/")
    assert response.status_code == 200 and response.data["status"] == "executed", response.data
    assert "Allowed agent to publish" in response.data["done"]
    assert hub.requests == []  # approving alone sends nothing: the agent has the token
    granted = call(client, "get_approval", id=approval.pk)[0]["structuredContent"]
    assert granted["result"]["granted"] is True

    # Other settings than approved are refused.
    result, text = mcp_publish(client, run_id, repo="other", private=False, approval_id=approval.pk)
    assert result["isError"] and "other settings" in text
    # A failed attempt keeps the grant.
    result, text = call(
        client,
        "publish_to_huggingface",
        run_id=run_id,
        hf_token=OTHER_TOKEN,
        repo="approved",
        private=False,
        approval_id=approval.pk,
    )
    assert result["isError"] and "did not accept the access token" in text
    result, text = mcp_publish(
        client, run_id, repo="approved", private=False, approval_id=approval.pk
    )
    assert "isError" not in result, text
    assert hub.repos["ada-hf/approved"] == {"private": False}
    approval.refresh_from_db()
    assert approval.result["granted"] is False and approval.result["repo_id"] == "ada-hf/approved"
    assert "published run “Net” to Hugging Face" in approval.details["done"]
    # Once only.
    result, text = mcp_publish(
        client, run_id, repo="approved", private=False, approval_id=approval.pk
    )
    assert result["isError"] and "used already" in text

    dump = io.StringIO()
    call_command("dumpdata", stdout=dump)
    assert GOOD_TOKEN not in dump.getvalue() and OTHER_TOKEN not in dump.getvalue()
    assert GOOD_TOKEN not in caplog.text and OTHER_TOKEN not in caplog.text
    for entry in AgentAction.objects.filter(tool="publish_to_huggingface"):
        assert entry.arguments.get("hf_token", "[redacted]") == "[redacted]"
        assert GOOD_TOKEN not in json.dumps(entry.arguments) + entry.summary + entry.error


def test_mcp_publish_errors(ada, hub, training_dir, limits):
    run_id = finished(client_for(ada), training_dir, real_weights=False)
    client, token = token_client(ada)
    approvals.set_rule(ada, approvals.OFF, token=token)
    result, text = call(client, "publish_to_huggingface", run_id=run_id)
    assert result["isError"] and "Missing argument: hf_token" in text
    result, text = call(client, "publish_to_huggingface", run_id=run_id, hf_token="with space")
    assert result["isError"] and "does not look like" in text
    result, text = mcp_publish(client, run_id, repo="taken")
    assert result["isError"] and "exists already" in text
    result, text = mcp_publish(client, "00000000-0000-0000-0000-000000000000")
    assert result["isError"] and "No such training run" in text


def test_mcp_publish_counts_against_the_rate_limit(ada, hub, training_dir, limits, monkeypatch):
    monkeypatch.setitem(ScopedRateThrottle.THROTTLE_RATES, "huggingface", "1/hour")
    run_id = finished(client_for(ada), training_dir, real_weights=False)
    client, token = token_client(ada)
    approvals.set_rule(ada, approvals.OFF, token=token)
    result, text = mcp_publish(client, run_id, repo="one")
    assert "isError" not in result, text
    result, text = mcp_publish(client, run_id, repo="two")
    assert result["isError"] and "Too many requests to Hugging Face" in text
    assert "ada-hf/two" not in hub.repos
    assert hub_publish.LICENSES[0] == "apache-2.0"
