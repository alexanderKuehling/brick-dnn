"""MCP answers are compact by default: summaries of datasets and runs, capped and paged
lists, more on request (``include``, ``columns_offset`` …). The REST API keeps its full
answers."""

# ruff: noqa: F811 (the fixture training_dir is imported from test_training_api)

from __future__ import annotations

import json
import time

import pytest
from django.contrib.auth import get_user_model

from brick_dnn_core import training_service
from brick_dnn_core.models import Dataset, TrainingRun
from brick_dnn_core.serializers import DatasetSerializer, RunDetailSerializer
from brickdnn.training.state import initial_state, write_state

from .test_mcp import call, mcp_client
from .test_training_api import DEMO_GRAPH, training_dir  # noqa: F401 (a fixture)

pytestmark = pytest.mark.django_db

WIDE = 800  # columns of the wide tables


@pytest.fixture(autouse=True)
def data_dirs(settings, tmp_path):
    settings.DATA_PREP_DIR = tmp_path / "prepare"
    settings.DATA_PREP_ENABLED = True
    settings.DATA_MAX_UPLOAD_MB = 5


@pytest.fixture
def user():
    return get_user_model().objects.create_user("ada", password="correct-horse-1")


@pytest.fixture
def mcp(user):
    return mcp_client(user)


def answer(client, tool, **arguments) -> tuple[dict, int]:
    """The structured result and the size of the text the assistant reads."""
    result, text = call(client, tool, **arguments)
    assert "isError" not in result, text
    return result.get("structuredContent", text), len(text.encode())


def error_of(client, tool, **arguments) -> str:
    result, text = call(client, tool, **arguments)
    assert result.get("isError") is True, text
    return text


def wide_csv(rows: int = 30, gaps: bool = False) -> str:
    """WIDE - 1 numeric features (with ``gaps`` every third with missing values) and a
    class label."""
    header = [f"pixel_{i}" for i in range(WIDE - 1)] + ["label"]
    lines = [",".join(header)]
    for r in range(rows):
        cells = [
            "" if (gaps and i % 3 == 0 and r % 7 == 0) else str((r * i) % 13)
            for i in range(WIDE - 1)
        ]
        lines.append(",".join([*cells, "abc"[r % 3]]))
    return "\n".join(lines) + "\n"


def size(data) -> int:
    return len(json.dumps(data, ensure_ascii=False, indent=2, default=str).encode())


# --------------------------------------------------------------------------- #
# Sizes
# --------------------------------------------------------------------------- #


def long_run(user, training_dir, epochs: int = 200) -> TrainingRun:
    run = TrainingRun.objects.create(
        owner=user,
        name="Long run",
        graph=DEMO_GRAPH,
        dataset={"kind": "random", "id": None, "name": "Demo data"},
        config={"epochs": epochs, "batch_size": 64, "learning_rate": 0.001, "seed": None},
        plan={"task": "classification", "summary": "Classification into 3 classes.", "notes": []},
        status="completed",
        message=f"Finished after {epochs} epochs.",
        epochs_done=epochs,
    )
    state = initial_state(epochs)
    history = [
        {
            "epoch": e,
            "train_loss": 1 / e,
            "val_loss": 1 / e + (0.01 if e > 120 else 0),
            "train_accuracy": 1 - 1 / (e + 1),
            "val_accuracy": 1 - 1 / (e + 2),
            "seconds": 0.5,
        }
        for e in range(1, epochs + 1)
    ]
    state.update(
        status="completed",
        message=run.message,
        history=history,
        best={"epoch": 120, "val_loss": 1 / 120, "val_accuracy": 0.99},
        weights_epoch=120,
        test={"loss": 0.01, "accuracy": 0.98, "rmse": None},
        log=[f"Epoch {e}: loss {1 / e:.4f} " + "x" * 80 for e in range(1, epochs + 1)],
        train_size=800,
        val_size=100,
        test_size=100,
        finished_at=state["started_at"] + 100,
    )
    directory = training_service.run_dir(run.pk)
    directory.mkdir(parents=True)
    write_state(directory, state)
    return run


def test_answers_stay_small_for_wide_data_and_long_runs(user, mcp, training_dir, capsys):
    sizes = {}
    # A dataset with 800 columns.
    uploaded, _ = answer(mcp, "upload_dataset", name="Pixels", csv=wide_csv())
    dataset = Dataset.objects.get(pk=uploaded["reference"]["id"])
    options, sizes["get_training_options"] = answer(mcp, "get_training_options")
    (summary,) = options["your_datasets"]
    assert summary == {
        "id": dataset.pk,
        "name": "Pixels",
        "kind": "csv",
        "owner": "ada",
        "mine": True,
        "rows": 30,
        "features": WIDE - 1,
        "target": "label",
        "task": "classification",
        "classes": 3,
        "prepared": False,
        "size_bytes": dataset.size_bytes,
        "reference": {"kind": "upload", "id": dataset.pk},
    }
    full = size(DatasetSerializer([dataset], many=True, context={"user": user}).data)
    # Four built-in datasets with their class names, the limits and the defaults, and what
    # a hyperparameter search can rank by and vary (about 1,800 bytes of it).
    assert sizes["get_training_options"] < 6500 < full
    tuning = options["classical"]["tuning"]
    assert tuning["models"]["SVC"] == "C*, gamma*, kernel, class_weight"
    assert tuning["params"]["C"] == "number > 0, log; e.g. 0.1:100"
    assert tuning["params"]["kernel"] == "linear|poly|rbf|sigmoid"
    assert "roc_auc (max; two classes)" in options["tuning"]["metrics"]

    # A run with 200 epochs and a long log.
    run = long_run(user, training_dir)
    detail, sizes["get_training_run"] = answer(mcp, "get_training_run", id=str(run.pk))
    full_run = size(RunDetailSerializer(run).data)
    assert sizes["get_training_run"] < 3000 < 20 * 3000 < full_run
    assert detail["best"]["epoch"] == 120 and detail["test"] == {"loss": 0.01, "accuracy": 0.98}
    assert detail["epochs"] == {"done": 200, "total": 200}
    assert detail["rows"] == {"train": 800, "validation": 100, "test": 100}
    assert "history" not in detail and "log" not in detail and "graph" not in detail
    assert "history" in detail["more"]

    # A data source and a preparation with 800 columns.
    result, _ = answer(mcp, "import_data", content=wide_csv(gaps=True), filename="pixels.csv")
    source = result["source"]
    _, sizes["get_data_source"] = answer(mcp, "get_data_source", id=source["id"])
    created, sizes["create_data_prep"] = answer(
        mcp, "create_data_prep", source_id=source["id"], target="label"
    )
    prep_id = created["preparation"]["id"]
    preview, sizes["preview_data_prep"] = answer(mcp, "preview_data_prep", id=prep_id)
    assert sizes["get_data_source"] < 6000
    assert sizes["create_data_prep"] < 8000 and sizes["preview_data_prep"] < 16000
    # Issues about many columns name a few of them.
    (missing,) = [i for i in preview["readiness"]["errors"] if i["code"] == "missing_values"]
    assert len(missing["columns"]) == 11 and missing["columns"][-1] == "… and 257 more"
    assert len(preview["columns"]) == 50 and "columns_offset=50" in preview["columns_note"]
    assert len(preview["profiles"]) == 20 and len(preview["sample"]["rows"][0]) == 20
    # How the 799 number columns relate: counts, a coarse histogram, the weakest ones.
    relations = preview["column_relations"]
    assert 700 < relations["columns"] <= 799 and size(relations) < 2500
    assert len(relations["weakest"]) == 15
    source_answer, _ = answer(mcp, "get_data_source", id=source["id"])
    assert 700 < source_answer["column_relations"]["columns"] <= 799
    with capsys.disabled():
        print("\nMCP answer sizes (bytes):", sizes, {"before": {"options": full, "run": full_run}})


# --------------------------------------------------------------------------- #
# Asking for more
# --------------------------------------------------------------------------- #


def test_get_dataset_pages_its_columns(user, mcp, training_dir):
    uploaded, _ = answer(mcp, "upload_dataset", name="Pixels", csv=wide_csv())
    dataset_id = uploaded["reference"]["id"]
    detail, _ = answer(mcp, "get_dataset", id=dataset_id)
    assert detail["n_columns"] == WIDE and detail["columns"][:2] == ["pixel_0", "pixel_1"]
    assert len(detail["columns"]) == 50 and "columns_offset=50" in detail["more_columns"]
    assert detail["class_names"] == ["a", "b", "c"]
    last, _ = answer(mcp, "get_dataset", id=dataset_id, columns_offset=790, columns_limit=50)
    assert last["columns"][-1] == "label" and len(last["columns"]) == 10
    assert "more_columns" not in last
    assert "columns_limit" in error_of(mcp, "get_dataset", id=dataset_id, columns_limit=501)
    assert "No such dataset" in error_of(mcp, "get_dataset", id=dataset_id + 1)


def test_get_training_run_includes(user, mcp, training_dir):
    run = long_run(user, training_dir)
    detail, _ = answer(
        mcp, "get_training_run", id=str(run.pk), include=["history", "log", "graph"], log_lines=5
    )
    epochs = [e["epoch"] for e in detail["history"]]
    assert len(epochs) <= 52 and epochs[0] == 1 and epochs[-1] == 200 and 120 in epochs
    assert epochs == sorted(epochs) and "evenly spaced" in detail["history_note"]
    assert len(detail["log"]) == 5 and detail["log"][-1].startswith("Epoch 200")
    assert detail["graph"] == DEMO_GRAPH
    assert "include" in error_of(mcp, "get_training_run", id=str(run.pk), include=["state"])

    # A failed run shows the end of its log by default.
    TrainingRun.objects.filter(pk=run.pk).update(status="failed", message="It broke.")
    detail, _ = answer(mcp, "get_training_run", id=str(run.pk))
    assert len(detail["log_tail"]) == 10 and detail["status"] == "failed"


def test_list_training_runs_is_paged_and_filtered(user, mcp, training_dir):
    runs = [long_run(user, training_dir, epochs=3) for _ in range(3)]
    TrainingRun.objects.filter(pk=runs[0].pk).update(status="failed")
    listing, _ = answer(mcp, "list_training_runs", limit=2)
    assert listing["total"] == 3 and len(listing["runs"]) == 2 and "limit" in listing["note"]
    first = listing["runs"][0]
    assert first["epochs"] == "3/3" and "config" not in first and "graph" not in first
    failed, _ = answer(mcp, "list_training_runs", status="failed")
    assert [r["id"] for r in failed["runs"]] == [str(runs[0].pk)]
    assert "'limit'" in error_of(mcp, "list_training_runs", limit=0)
    assert answer(mcp, "list_training_runs", project_id=12345)[0] == {"total": 0, "runs": []}


def test_layer_types_are_compact_unless_asked(mcp):
    catalog, compact_size = answer(mcp, "list_layer_types")
    layers = {layer["type"]: layer for layer in catalog["layers"]}
    assert {"Input", "Linear", "Conv2d", "LSTM", "RBF", "SVC"} <= set(layers)
    assert {"HistGradientBoostingRegressor", "HistGradientBoostingClassifier"} <= set(layers)
    boosting = layers["HistGradientBoostingClassifier"]["params"]
    assert {"learning_rate", "max_iter", "class_weight", "early_stopping"} <= set(boosting)
    assert layers["Linear"]["params"]["out_features"].startswith("int")
    assert "null = inferred" in layers["Linear"]["params"]["in_features"]
    assert layers["RBF"]["output_kind"] == "kernel"
    detail, _ = answer(mcp, "list_layer_types", types=["Linear"])
    (linear,) = detail["layers"]
    assert linear["params"][0]["help"] and linear["input_slots"]
    full = size(detail) * 0  # (the full catalogue is GET /api/layers/)
    assert compact_size < 25_000 + full
    assert "Unknown layer type 'Dense'" in error_of(mcp, "list_layer_types", types=["Dense"])


def test_check_training_caps_class_names(user, mcp, training_dir):
    lines = ["x,y,label"] + [f"{i},{i % 7},c{i}" for i in range(120)]
    uploaded, _ = answer(mcp, "upload_dataset", name="Many", csv="\n".join(lines) + "\n")
    graph = {
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [2]}},
            {"id": "fc", "type": "Linear", "params": {"out_features": 120}},
        ],
        "edges": [{"source": "in", "target": "fc"}],
    }
    started = time.monotonic()
    check, _ = answer(mcp, "check_training", graph=graph, dataset=uploaded["reference"])
    names = check["plan"]["class_names"]
    assert len(names) == 51 and names[-1] == "… and 70 more" and time.monotonic() - started < 30


def test_column_relations_summary_points_at_noise_columns():
    import numpy as np
    import pandas as pd

    from brick_dnn_core.mcp_data import relations_view
    from brickdnn.data.relations import column_relations, to_json
    from brickdnn.data.table import Table

    rng = np.random.default_rng(0)
    latent = rng.normal(size=(500, 3))
    data = {f"r{i}": latent[:, i % 3] + 0.5 * rng.normal(size=500) for i in range(30)}
    data.update({f"n{i}": rng.normal(size=500) for i in range(60)})
    relations = to_json(column_relations(Table.from_frame(pd.DataFrame(data))))
    view = relations_view(relations)
    assert view["columns"] == 90
    assert view["without_partner"]["threshold"] == relations["suggested_level"]
    assert 55 <= view["without_partner"]["columns"] <= 60
    assert view["with_partner"]["columns"] == 30
    assert all(name.startswith("n") for name in view["weakest"])
    assert '"method": "no_partner"' in view["drop_them"] and size(view) < 2000
    assert relations_view(None) is None
