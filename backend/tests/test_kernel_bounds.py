"""Kernel hyperparameters that a Gaussian process fit left at a bound
(brickdnn.training.kernel_bounds): found after the fit, stored in state.classical["at_bounds"],
logged, and summarised by the MCP tool get_training_run."""

from __future__ import annotations

import pytest

pytest.importorskip("sklearn")

from brickdnn.graph import analyze  # noqa: E402
from brickdnn.training import inspect_csv, random_info  # noqa: E402
from brickdnn.training.classical_engine import build_pipeline  # noqa: E402
from brickdnn.training.kernel_bounds import at_bounds  # noqa: E402

from .test_classical_training import classification_csv, regression_csv, run_job  # noqa: E402


def gp_graph(white=None, rbf=None, model="GaussianProcessRegressor", features=4):
    return {
        "version": 1,
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [features]}},
            {"id": "rbf_1", "type": "RBF", "params": rbf or {}},
            {"id": "white_1", "type": "WhiteKernel", "params": white or {}},
            {"id": "sum", "type": "KernelSum", "params": {}},
            {"id": "gp", "type": model, "params": {}},
        ],
        "edges": [
            {"source": "in", "target": "gp"},
            {"source": "rbf_1", "target": "sum"},
            {"source": "white_1", "target": "sum"},
            {"source": "sum", "target": "gp"},
        ],
    }


# Almost noise-free data (noise 0.05 on targets with a spread of about 4): the fit wants a
# noise level far below 0.5 and a length scale above 0.5.
TIGHT_NOISE = {
    "noise_level": 1.0,
    "hyperparameters": "custom bounds",
    "noise_level_min": 0.5,
    "noise_level_max": 10.0,
}
TIGHT_LENGTH = {
    "length_scale": 0.3,
    "hyperparameters": "custom bounds",
    "length_scale_min": 0.01,
    "length_scale_max": 0.5,
}


def test_a_fit_at_its_bounds_is_reported_with_a_suggestion(tmp_path):
    path = regression_csv(tmp_path / "prices.csv")
    graph = gp_graph(white=TIGHT_NOISE, rbf=TIGHT_LENGTH)
    result, rec = run_job(graph, inspect_csv(path), tmp_path / "run", path=path)
    assert result.status == "completed", result.message
    found = {(e["node"], e["side"]): e for e in rec.info["classical"]["at_bounds"]}
    assert set(found) == {("white_1", "lower"), ("rbf_1", "upper")}

    noise = found[("white_1", "lower")]
    assert noise["parameter"] == "noise_level"
    assert noise["label"] == "Noise level"
    assert noise["type"] == "WhiteKernel"
    assert noise["bound"] == pytest.approx(0.5)
    assert noise["value"] == pytest.approx(0.5, rel=1e-3)
    assert noise["message"] == (
        "Noise level of white_1 ended at its lower bound 0.5: the model may be overfitting; "
        "set custom bounds, e.g. ≥ 50."
    )
    # The suggestion is a valid set of parameters for the brick.
    assert noise["suggestion"] == {
        "hyperparameters": "custom bounds",
        "noise_level_min": 50.0,
        "noise_level_max": 500.0,
        "noise_level": 50.0,
    }
    changed = [
        {**n, "params": {**n["params"], **noise["suggestion"]}} if n["id"] == "white_1" else n
        for n in graph["nodes"]
    ]
    assert analyze({**graph, "nodes": changed}).valid

    length = found[("rbf_1", "upper")]
    assert length["bound"] == pytest.approx(0.5)
    assert "upper bound 0.5" in length["message"]
    assert length["suggestion"]["length_scale_max"] == pytest.approx(50.0)
    assert length["suggestion"]["length_scale_min"] == pytest.approx(0.01)
    # The run log says it too.
    assert noise["message"] in rec.logs


def test_nothing_is_reported_for_a_fit_inside_its_bounds(tmp_path):
    path = regression_csv(tmp_path / "prices.csv")
    graph = gp_graph(white={"noise_level": 0.1})
    result, rec = run_job(graph, inspect_csv(path), tmp_path / "run", path=path)
    assert result.status == "completed", result.message
    entries = rec.info["classical"]["at_bounds"]
    # The default bounds (1e-05 … 1e+05) are wide: the noise of this data is inside them.
    assert all(e["node"] != "rbf_1" for e in entries)


def test_fixed_hyperparameters_are_never_at_a_bound(tmp_path):
    path = regression_csv(tmp_path / "prices.csv")
    graph = gp_graph(white={"noise_level": 1e-5, "hyperparameters": "fixed"})
    result, rec = run_job(graph, inspect_csv(path), tmp_path / "run", path=path)
    assert result.status == "completed", result.message
    assert all(e["node"] != "white_1" for e in rec.info["classical"]["at_bounds"])


def test_svms_have_no_kernel_bounds(tmp_path):
    graph = {
        "version": 1,
        "nodes": [
            {"id": "in", "type": "Input", "params": {"shape": [3]}},
            {"id": "m", "type": "SVR", "params": {}},
        ],
        "edges": [{"source": "in", "target": "m"}],
    }
    result, rec = run_job(graph, random_info(), tmp_path / "run")
    assert result.status == "completed", result.message
    assert "at_bounds" not in rec.info["classical"]


def test_classifiers_with_one_kernel_per_class(tmp_path):
    path = classification_csv(tmp_path / "food.csv")
    graph = gp_graph(white=TIGHT_NOISE, model="GaussianProcessClassifier")
    analysis = analyze(graph)
    pipeline = build_pipeline(graph)
    data = [line.split(",") for line in path.read_text().splitlines()[1:]]
    x = [[float(v) for v in row[:4]] for row in data]
    y = [row[4] for row in data]
    pipeline.fit(x, y)
    model = pipeline.named_steps["model"]
    found = at_bounds(model, analysis.model.layer.kernel, analysis, ["eggs", "ham", "spam"])
    noise = [e for e in found if e["node"] == "white_1"]
    # One entry, naming the classes whose kernel ended at the bound.
    assert len(noise) == 1
    assert noise[0]["classes"] and set(noise[0]["classes"]) <= {"eggs", "ham", "spam"}


@pytest.mark.django_db
def test_get_training_run_names_the_bounds_in_its_summary():
    from django.contrib.auth import get_user_model

    from brick_dnn_core.mcp_compact import run_detail
    from brick_dnn_core.models import TrainingRun

    user = get_user_model().objects.create_user("ada")
    run = TrainingRun.objects.create(
        owner=user,
        name="GP",
        graph=gp_graph(),
        dataset={"kind": "csv", "id": None, "name": "prices"},
        config={},
        plan={"task": "regression", "classical": {"estimator": "GaussianProcessRegressor"}},
        status="completed",
    )
    message = "Noise level of white_1 ended at its lower bound 1e-05: the model may be overfitting."
    entry = {"node": "white_1", "parameter": "noise_level", "suggestion": {}, "message": message}
    state = {"classical": {"kernel": "RBF(length_scale=1)", "at_bounds": [entry] * 7}}
    classical = run_detail(run, state, set())["classical"]
    assert classical["kernel"] == "RBF(length_scale=1)"
    assert classical["at_bounds"][:5] == [message] * 5
    assert classical["at_bounds"][5] == "… and 2 more"
    # Without any, the key is left out.
    state["classical"]["at_bounds"] = []
    assert "at_bounds" not in run_detail(run, state, set())["classical"]
