"""Importing files of datasets on the Hugging Face Hub (hub_datasets.py) against a fake Hub
(hub_fake.py): references, configs and splits, the listing, the import (a link import with
its checks) with and without an access token, and the MCP tool import_data."""

from __future__ import annotations

import io
import json
import logging

import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from rest_framework.test import APIClient

from brick_dnn_core import hub_datasets
from brick_dnn_core.hub_datasets import DatasetRef, classify, parse
from brick_dnn_core.models import AgentAction, DataSource

from .hub_fake import GOOD_TOKEN, OTHER_TOKEN, FakeHub
from .test_mcp import call, mcp_client

pytestmark = pytest.mark.django_db

CSV = b"sepal_length,sepal_width,species\n5.1,3.5,setosa\n6.2,2.9,versicolor\n5.9,3.0,virginica\n"


@pytest.fixture(autouse=True)
def data_dirs(settings, tmp_path):
    settings.DATA_PREP_DIR = tmp_path / "prepare"
    settings.TRAINING_DIR = tmp_path / "training"
    settings.DATA_PREP_ENABLED = True
    settings.DATA_URL_IMPORT_ENABLED = True
    settings.DATA_URL_ALLOW_PRIVATE = True  # the fake Hub runs on 127.0.0.1
    settings.DATA_MAX_UPLOAD_MB = 5


@pytest.fixture
def hub(settings):
    fake = FakeHub()
    settings.HUGGINGFACE_URL = fake.url
    settings.HUGGINGFACE_ENABLED = True
    settings.HUGGINGFACE_PROXY = ""
    fake.add_dataset(
        "ada/iris",
        {
            "README.md": b"# Iris",
            "data/train.csv": CSV,
            "data/test.csv": CSV,
            "extra/notes.txt": b"a;b\n1;2\n3;4\n",
            "extra/loader.py": b"# not data",
        },
    )
    yield fake
    fake.close()


@pytest.fixture
def user():
    return get_user_model().objects.create_user("ada", password="correct-horse-battery")


@pytest.fixture
def web(user):
    client = APIClient()
    client.force_authenticate(user)
    return client


def files(client, dataset, **body):
    return client.post("/api/data/huggingface/files/", {"dataset": dataset, **body}, format="json")


def import_ref(client, url, **body):
    return client.post("/api/data/sources/", {"url": url, **body}, format="json")


# --------------------------------------------------------------------------- #
# References, configs and splits
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hf://datasets/stanfordnlp/imdb", DatasetRef("stanfordnlp/imdb")),
        (
            "hf://datasets/stanfordnlp/imdb@refs%2Fconvert%2Fparquet/plain_text/train/0000.parquet",
            DatasetRef("stanfordnlp/imdb", "refs/convert/parquet", "plain_text/train/0000.parquet"),
        ),
        ("hf://datasets/imdb", DatasetRef("imdb")),
        ("https://huggingface.co/datasets/scikit-learn/iris", DatasetRef("scikit-learn/iris")),
        (
            "https://huggingface.co/datasets/scikit-learn/iris/blob/main/Iris.csv",
            DatasetRef("scikit-learn/iris", "main", "Iris.csv"),
        ),
        (
            "https://huggingface.co/datasets/scikit-learn/iris/raw/main/Iris.csv",
            DatasetRef("scikit-learn/iris", "main", "Iris.csv"),
        ),
        (  # a folder: the dataset at that revision, not a file
            "https://huggingface.co/datasets/o/n/tree/refs%2Fconvert%2Fparquet/default",
            DatasetRef("o/n", "refs/convert/parquet"),
        ),
        (
            "https://huggingface.co/datasets/imdb/resolve/main/a.csv",
            DatasetRef("imdb", path="a.csv"),
        ),
        ("https://huggingface.co/datasets/o/n/discussions/3", None),  # another page
        (
            "https://huggingface.co/datasets/o/n/viewer/default/test",
            DatasetRef("o/n", config="default", split="test"),
        ),
        ("https://huggingface.co/bert-base-uncased", None),  # a model
        ("https://example.org/datasets/a/b", None),
        ("o/n", None),  # only with bare=True
    ],
)
def test_references(text, expected):
    assert parse(text) == expected


def test_bare_and_invalid_references():
    assert parse("scikit-learn/iris", bare=True) == DatasetRef("scikit-learn/iris")
    for bad in ("hf://models/x/y", "hf://datasets/", "hf://datasets/a/b!c", "hf://datasets/a--b/c"):
        with pytest.raises(hub_datasets.HubInputError):
            parse(bad)
    assert DatasetRef("o/n", "refs/convert/parquet").file_ref("a/0.parquet") == (
        "hf://datasets/o/n@refs%2Fconvert%2Fparquet/a/0.parquet"
    )


def test_configs_and_splits():
    paths = [
        "data/train-00000-of-00002.parquet",
        "data/validation-00000-of-00001.parquet",
        "plain_text/test.jsonl",
        "dev.csv",
        "misc.csv",
    ]
    assert classify(paths, {}, converted=False) == {
        "data/train-00000-of-00002.parquet": ("default", "train"),
        "data/validation-00000-of-00001.parquet": ("default", "validation"),
        "plain_text/test.jsonl": ("plain_text", "test"),
        "dev.csv": ("default", "validation"),
        "misc.csv": ("default", None),
    }
    card = {
        "cardData": {
            "configs": [
                {
                    "config_name": "en",
                    "data_files": [
                        {"split": "train", "path": "en/part-*.csv"},
                        {"split": "test", "path": ["en/holdout.csv"]},
                    ],
                },
                {"config_name": "de", "data_files": "de/*.csv"},
            ]
        }
    }
    found = classify(["en/part-1.csv", "en/holdout.csv", "de/x.csv", "other.csv"], card, False)
    assert found == {
        "en/part-1.csv": ("en", "train"),
        "en/holdout.csv": ("en", "test"),
        "de/x.csv": ("de", "train"),
        "other.csv": ("default", None),
    }
    converted = classify(["plain_text/partial-train/0000.parquet"], {}, converted=True)
    assert converted == {"plain_text/partial-train/0000.parquet": ("plain_text", "train")}


# --------------------------------------------------------------------------- #
# Listing
# --------------------------------------------------------------------------- #


def test_listing(web, hub):
    response = files(web, "hf://datasets/ada/iris")
    assert response.status_code == 200, response.data
    data = response.data
    assert data["repo_id"] == "ada/iris" and data["revision"] == "main"
    assert data["converted"] is False and data["private"] is False and data["gated"] is False
    assert [f["path"] for f in data["files"]] == ["data/train.csv", "data/test.csv"]
    assert data["files"][0] == {
        "path": "data/train.csv",
        "size": len(CSV),
        "format": "csv",
        "config": "default",
        "split": "train",
        "ref": "hf://datasets/ada/iris/data/train.csv",
    }
    assert data["configs"] == ["default"] and data["splits"] == ["train", "test"]
    assert data["page"] == f"{hub.url}/datasets/ada/iris"
    # Public: no token was sent.
    assert all("Authorization" not in r["headers"] for r in hub.requests)
    # The same with the page's link and with owner/name.
    assert files(web, "https://huggingface.co/datasets/ada/iris").data["repo_id"] == "ada/iris"
    assert files(web, "ada/iris").status_code == 200


def test_listing_pages_and_errors(web, hub):
    hub.page_size = 2
    hub.add_dataset("ada/many", {f"part{i}/train.csv": CSV for i in range(5)})
    data = files(web, "ada/many").data
    assert len(data["files"]) == 5 and data["configs"] == [f"part{i}" for i in range(5)]
    response = files(web, "ada/missing")
    assert response.status_code == 404 and "or it is private" in response.data["detail"]
    assert files(web, "not a dataset").status_code == 400
    assert files(web, "hf://models/a/b").status_code == 400
    response = files(web, "ada/iris", hf_token=OTHER_TOKEN)
    assert response.status_code == 200  # a token the Hub accepts for public data
    response = files(web, "ada/iris", hf_token="has space")
    assert response.status_code == 400


def test_private_dataset_needs_a_token(web, hub):
    hub.add_dataset("ada/secret", {"train.csv": CSV}, private=True)
    response = files(web, "ada/secret")
    assert response.status_code == 404 and "access token" in response.data["detail"]
    response = files(web, "ada/secret", hf_token=GOOD_TOKEN)
    assert response.status_code == 200, response.data
    assert response.data["private"] is True and response.data["files"][0]["path"] == "train.csv"


def test_datasets_without_data_files_are_listed_from_the_converted_parquet(web, hub):
    hub.add_dataset("ada/scripted", {"scripted.py": b"# a loading script"})
    hub.add_dataset(
        "ada/scripted",
        {"default/train/0000.parquet": b"PAR1", "default/test/0000.parquet": b"PAR1"},
        revision="refs/convert/parquet",
    )
    data = files(web, "ada/scripted").data
    assert data["converted"] is True and data["revision"] == "refs/convert/parquet"
    assert [f["split"] for f in data["files"]] == ["train", "test"]
    assert data["files"][0]["ref"] == (
        "hf://datasets/ada/scripted@refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    )


def test_switched_off(web, user, hub, settings):
    settings.HUGGINGFACE_ENABLED = False
    assert files(web, "ada/iris").status_code == 403
    assert web.get("/api/data/").data["huggingface"] is False
    response = import_ref(web, "hf://datasets/ada/iris/data/train.csv")
    assert response.status_code == 403
    url = f"{hub.url}/datasets/ada/iris/resolve/main/data/train.csv"
    assert import_ref(web, url, hf_token=GOOD_TOKEN).status_code == 403
    assert all("Authorization" not in r["headers"] for r in hub.requests)
    # The link of a file on the Hub stays what it was before: an ordinary link import.
    response = import_ref(web, url)
    assert response.status_code == 201, response.data
    assert response.data["n_rows"] == 3
    result, text = call(mcp_client(user), "import_data", url="hf://datasets/a/b")
    assert result["isError"] and "switched off" in text


# --------------------------------------------------------------------------- #
# Importing
# --------------------------------------------------------------------------- #


def test_import_a_file(web, hub):
    response = import_ref(web, "hf://datasets/ada/iris/data/train.csv")
    assert response.status_code == 201, response.data
    data = response.data
    assert data["name"] == "iris · train" and data["n_rows"] == 3 and data["n_columns"] == 3
    assert data["report"]["url"] == f"{hub.url}/datasets/ada/iris/resolve/main/data/train.csv"
    # The download was redirected to the storage server, like on the real Hub.
    assert hub.storage_requests[-1]["method"] == "GET"
    # A name of the user wins; a whole dataset is not a file.
    assert (
        import_ref(web, "hf://datasets/ada/iris/data/test.csv", name="Mine").data["name"] == "Mine"
    )
    response = import_ref(web, "hf://datasets/ada/iris")
    assert response.status_code == 400 and "Choose a file" in response.data["detail"]
    requests = len(hub.requests)
    response = import_ref(web, "hf://datasets/ada/iris/extra/loader.py")
    assert response.status_code == 400 and "Parquet" in response.data["detail"]
    assert len(hub.requests) == requests  # refused before anything was asked
    # A file named directly need not be one of the listed formats, and the link of its page
    # on the Hub works like its hf:// reference.
    response = import_ref(web, f"{hub.url}/datasets/ada/iris/blob/main/extra/notes.txt")
    assert response.status_code == 201, response.data
    assert response.data["n_rows"] == 2 and response.data["name"] == "iris · extra · notes"
    response = import_ref(web, "hf://datasets/ada/iris/data/nothing.csv")
    assert response.status_code == 400 and "404" in response.data["detail"]
    # A token only goes to Hugging Face.
    response = import_ref(web, f"{hub.storage.url}/x.csv", hf_token=GOOD_TOKEN)
    assert (
        response.status_code == 400 and "only sent along to Hugging Face" in response.data["detail"]
    )


def test_gated_dataset_with_a_token(web, hub, caplog):
    caplog.set_level(logging.DEBUG)
    hub.add_dataset("ada/gated", {"train.csv": CSV}, gated="manual")
    assert files(web, "ada/gated").data["gated"] == "manual"
    response = import_ref(web, "hf://datasets/ada/gated/train.csv")
    assert response.status_code == 400
    assert (
        "HTTP 403" in response.data["detail"] and "accept its conditions" in response.data["detail"]
    )
    response = import_ref(web, "hf://datasets/ada/gated/train.csv", hf_token=GOOD_TOKEN)
    assert response.status_code == 201, response.data
    # Sent to the Hub, not to the storage server it redirected to.
    resolve = [r for r in hub.requests if "/resolve/" in r["path"]][-1]
    assert resolve["headers"]["Authorization"] == f"Bearer {GOOD_TOKEN}"
    assert all("Authorization" not in r["headers"] for r in hub.storage_requests)
    # Stored nowhere: not in the database, not in the logs, not in the files of the import.
    dump = io.StringIO()
    call_command("dumpdata", stdout=dump)
    assert GOOD_TOKEN not in dump.getvalue() and GOOD_TOKEN not in caplog.text
    source = DataSource.objects.get(pk=response.data["id"])
    assert GOOD_TOKEN not in json.dumps(source.report)


def test_mcp_import_data(user, hub, caplog):
    caplog.set_level(logging.DEBUG)
    client = mcp_client(user)
    result, text = call(client, "import_data", url="hf://datasets/ada/iris")
    assert result["isError"] and "2 files that can be imported" in text
    assert "- hf://datasets/ada/iris/data/train.csv (config default, split train" in text
    result, text = call(client, "import_data", url="hf://datasets/ada/iris/data/test.csv")
    assert "isError" not in result, text
    assert result["structuredContent"]["source"]["name"] == "iris · test"
    hub.add_dataset("ada/gated", {"train.csv": CSV}, gated="auto")
    result, text = call(
        client, "import_data", url="hf://datasets/ada/gated/train.csv", hf_token=GOOD_TOKEN
    )
    assert "isError" not in result, text
    result, text = call(client, "import_data", url="https://example.org/a.csv", hf_token=GOOD_TOKEN)
    assert result["isError"] and "only sent along" in text
    result, text = call(
        client, "import_data", url="hf://datasets/ada/gated/train.csv", hf_token=" "
    )
    assert result["isError"] and "HTTP 403" in text
    for entry in AgentAction.objects.filter(tool="import_data"):
        assert GOOD_TOKEN not in json.dumps(entry.arguments)
        if "hf_token" in entry.arguments:
            assert entry.arguments["hf_token"] == "[redacted]"
    dump = io.StringIO()
    call_command("dumpdata", stdout=dump)
    assert GOOD_TOKEN not in dump.getvalue() and GOOD_TOKEN not in caplog.text
