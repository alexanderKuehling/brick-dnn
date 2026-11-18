"""The built-in text datasets in the REST API and over MCP: the training options (also in
German), the check with its fixes, and a real training on the CPU whose network predicts
typed texts (and token ids) and whose download explains how to read new texts."""

# ruff: noqa: F811 (the fixture training_dir is imported from test_training_api)

from __future__ import annotations

import copy
import subprocess
import sys

import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from brickdnn import EXAMPLES
from brickdnn.training import torch_installed
from brickdnn.training.textsets import text_examples

from .helpers import chain
from .test_mcp import call, mcp_client
from .test_training_api import training_dir, wait_for  # noqa: F401 (a fixture)

pytestmark = pytest.mark.django_db

needs_torch = pytest.mark.skipif(not torch_installed(), reason="PyTorch is not installed")
GERMAN = {"HTTP_ACCEPT_LANGUAGE": "de-DE,de;q=0.9,en;q=0.8"}
TOPICS = {"kind": "builtin", "id": "topics"}
SENTIMENT = {"kind": "builtin", "id": "sentiment"}


@pytest.fixture
def user():
    return get_user_model().objects.create_user("ada", password="correct-horse-battery")


@pytest.fixture
def web(user):
    client = APIClient()
    client.force_authenticate(user)
    return client


@pytest.fixture
def mcp(user):
    return mcp_client(user)


def answer(client, tool, **arguments) -> dict:
    result, text = call(client, tool, **arguments)
    assert "isError" not in result, text
    return result.get("structuredContent", text)


def test_text_datasets_in_the_training_options(web, mcp, training_dir):
    options = web.get("/api/training/").data
    texts = {d["id"]: d for d in options["builtin_datasets"] if d.get("input_kind") == "tokens"}
    assert list(texts) == ["sentiment", "lookup", "topics"]
    lookup = texts["lookup"]
    assert (lookup["name"], lookup["available"], lookup["task"]) == (
        "Key lookup",
        True,
        "classification",
    )
    assert (lookup["vocab_size"], lookup["sequence_length"], lookup["sample_shape"]) == (
        289,
        10,
        [10],
    )
    assert lookup["examples"] == text_examples("lookup")
    # In German: the descriptions, not the class names nor the texts.
    german = APIClient().get("/api/training/", **GERMAN).data
    sentiment = next(d for d in german["builtin_datasets"] if d["id"] == "sentiment")
    assert sentiment["description"].startswith("Text: kurze Bewertungen, positiv oder negativ.")
    assert sentiment["class_names"] == ["negative", "positive"]
    assert sentiment["examples"] == text_examples("sentiment")
    # MCP lists them as well.
    listed = answer(mcp, "get_training_options")["builtin_datasets"]
    assert [d["id"] for d in listed if d.get("input_kind") == "tokens"] == list(texts)


def test_the_check_explains_and_fixes_a_text_network(web, mcp, training_dir):
    graph = copy.deepcopy(EXAMPLES["transformer_classifier"]["graph"])
    check = web.post(
        "/api/training/check/", {"graph": graph, "dataset": SENTIMENT}, format="json"
    ).data
    assert check["ok"] is False
    assert check["errors"][0].startswith("Every text of Sentiment has 24 tokens")
    assert [fix["summary"] for fix in check["fixes"]] == [
        "Input shape 64 → 24",
        "Vocabulary size of the Embedding 10000 → 151",
        "Outputs of the last layer 4 → 2",
    ]
    german = APIClient()
    german.force_authenticate(web.handler._force_user)
    # The messages are translated when the answer is rendered.
    check = german.post(
        "/api/training/check/", {"graph": graph, "dataset": SENTIMENT}, format="json", **GERMAN
    ).json()
    assert check["errors"][0].startswith("Jeder Text von Sentiment hat 24 Tokens")
    assert check["warnings"][0].startswith("Das Embedding hat 10,000 Einträge")
    assert check["fixes"][1]["summary"] == "Vokabulargröße des Embeddings 10000 → 151"
    # Over MCP, with the fixes applied, it fits.
    for fix in check["fixes"]:
        node = next(n for n in graph["nodes"] if n["id"] == fix["node"])
        node["params"].update(fix["params"])
    result = answer(mcp, "check_training", graph=graph, dataset=SENTIMENT)
    assert result["ok"] is True
    assert result["plan"]["tokens"] == {"dataset": "sentiment", "length": 24, "vocab_size": 151}


def test_generated_code_over_mcp_embeds_the_dataset(mcp):
    graph = EXAMPLES["transformer_classifier"]["graph"]
    code = answer(mcp, "generate_pytorch_code", graph=graph)
    code = code if isinstance(code, str) else code["result"]
    assert 'DATASET = "topics"' in code
    result, text = call(mcp, "generate_pytorch_code", graph=graph, dataset="lookup")
    assert result.get("isError") is True and "does not fit the built-in dataset" in text


@needs_torch
def test_train_and_predict_texts(web, mcp, training_dir, settings, monkeypatch, tmp_path):
    settings.TRAINING_RUNNER_COMMAND = None
    settings.TRAINING_PREDICT_COMMAND = None
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    graph = chain(
        ("Input", {"shape": [64], "dtype": "long"}),
        ("Embedding", {"num_embeddings": 161, "embedding_dim": 16}),
        ("SequencePool", {"mode": "mean"}),
        ("Linear", {"out_features": 4}),
    )
    config = {"epochs": 3, "learning_rate": 0.02, "max_samples": 1200, "seed": 1, "device": "cpu"}
    response = web.post(
        "/api/training/runs/",
        {"name": "Topics net", "graph": graph, "dataset": TOPICS, "config": config},
        format="json",
    )
    assert response.status_code == 201, response.data
    run = wait_for(web, response.data["id"], timeout=600, interval=0.5)
    assert run["status"] == "completed", run["message"]
    assert run["task"] == "classification" and run["metrics"]["val_accuracy"] > 0.6

    form = web.get(f"/api/training/runs/{run['id']}/predict-form/").data
    assert form["available"] is True and form["kind"] == "tokens"
    assert form["text"]["length"] == 64 and form["text"]["vocab_size"] == 161
    assert form["text"]["examples"] == text_examples("topics", 4)

    url = f"/api/training/runs/{run['id']}/predict/"
    texts = [
        "The chef put garlic, butter and onion in the pan for the sauce",
        "the team scored a late goal in the match and the fans cheered",
    ]
    response = web.post(url, {"inputs": texts}, format="json")
    assert response.status_code == 200, response.data
    cooking, sports = response.data["predictions"]
    assert cooking["class"] == "cooking" and sports["class"] == "sports"
    # What the network read: words it does not know are <unk>.
    assert cooking["tokens"][:5] == ["the", "chef", "<unk>", "garlic", "<unk>"]
    assert cooking["unknown_words"] == ["put", ","]
    assert sports["unknown_words"] == ["scored", "late", "cheered"]
    # Token ids work too, and errors say what the network takes.
    response = web.post(url, {"inputs": [[2, 3, 4]]}, format="json")
    assert response.status_code == 200 and len(response.data["predictions"][0]["tokens"]) == 3
    response = web.post(url, {"inputs": [[999]]}, format="json")
    assert response.status_code == 400
    assert "token ids must be whole numbers from 0 to 160" in response.data["detail"]
    assert response.data["inputs"].startswith("One text per sample")

    text = "a recipe for dinner: bake the dough in the oven with cheese, tomato and herbs"
    result = answer(mcp, "predict", run_id=run["id"], inputs=[text])
    (prediction,) = result["predictions"]
    assert prediction["class"] == "cooking"
    assert prediction["tokens"][:4] == ["a", "recipe", "for", "dinner"]
    assert prediction["unknown_words"] == [":", ","]

    # Of the explanations, the weights work for texts.
    methods = web.get(f"/api/training/runs/{run['id']}/explain/").data["methods"]
    assert {m["id"] for m in methods if m["available"]} == {"weights"}

    # The download explains the texts, and its snippet reads them like the server.
    info = answer(mcp, "get_weights_download", id=run["id"])
    assert info["inputs"].startswith("One text per sample, as a string")
    (tmp_path / info["model_code_filename"]).write_text(info["model_code"])
    (tmp_path / info["filename"]).write_bytes(
        (training_dir / "runs" / run["id"] / "weights.pt").read_bytes()
    )
    snippet = info["how_to_use"].replace(text_examples("topics", 1)[0]["text"], texts[0])
    script = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert script.returncode == 0, script.stderr
    assert script.stdout.strip() == "['cooking']"
