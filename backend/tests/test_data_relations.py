"""Column relations (strongest partner of every number column), the readiness check for
columns without a strong partner and the "no strong partner" feature selection."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from brickdnn.data import relations as rel
from brickdnn.data.checks import relation_issues
from brickdnn.data.pipeline import readiness, run_pipeline
from brickdnn.data.profile import profile_table
from brickdnn.data.relations import (
    chance_level,
    column_relations,
    correlation_matrix,
    suggested_level,
    to_json,
    without_partner,
)
from brickdnn.data.steps import StepContext, get_step
from brickdnn.data.table import TEST, TRAIN, VAL, Table

SETTINGS = {"target": "y", "task": "regression", "split": {"val": 0.15, "test": 0.15, "seed": 1}}


def frame_table(df: pd.DataFrame, split: np.ndarray | None = None) -> Table:
    return Table.from_frame(df, split=split)


def related_table(n: int = 300, seed: int = 0, missing: float = 0.0) -> pd.DataFrame:
    """Two groups of related columns (a*, b*), independent ones (n*), a monotonic but
    non-linear partner (c = exp(a0)), a constant and a target."""
    rng = np.random.default_rng(seed)
    f, g = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame(
        {
            "a0": f + 0.3 * rng.normal(size=n),
            "a1": f + 0.5 * rng.normal(size=n),
            "a2": -f + 0.7 * rng.normal(size=n),
            "b0": g + 0.4 * rng.normal(size=n),
            "b1": g + 0.4 * rng.normal(size=n),
            "n0": rng.normal(size=n),
            "n1": rng.uniform(size=n),
            "const": np.ones(n),
        }
    )
    df["c"] = np.exp(3 * df["a0"])
    df["y"] = f + g + 0.1 * rng.normal(size=n)
    if missing:
        for column in ("a0", "a1", "b0", "n0"):
            df.loc[rng.random(n) < missing, column] = np.nan
    return df


def noise_table(
    rows: int = 600,
    real: int = 40,
    noise: int = 80,
    factors: int = 4,
    seed: int = 0,
) -> pd.DataFrame:
    """Like the brain-scan table: real columns built from a few latent factors and columns
    of injected noise (half uniform, half Gaussian), shuffled; a target from the factors."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(rows, factors))
    data = {}
    for i in range(real):
        weights = rng.normal(size=factors) * (rng.random(factors) < 0.6)
        weights[i % factors] += 1.5
        data[f"r{i}"] = latent @ weights + rng.normal(size=rows)
    for i in range(noise):
        data[f"n{i}"] = rng.uniform(size=rows) if i % 2 else rng.normal(size=rows)
    df = pd.DataFrame(data)
    df = df[list(rng.permutation(df.columns))]
    df["y"] = 60 + 10 * latent[:, 0] + rng.normal(size=rows)
    return df


def split_of(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice([TRAIN, VAL, TEST], n, p=[0.7, 0.15, 0.15]).astype(np.int8)


# --------------------------------------------------------------------------- #
# Statistics against a straightforward reference
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["spearman", "pearson"])
def test_matrix_matches_pandas_without_missing_values(method):
    df = related_table().drop(columns=["const", "y"])
    matrix, pair_rows = correlation_matrix(df.to_numpy(dtype=float), method)
    reference = df.corr(method=method).to_numpy().copy()
    np.fill_diagonal(reference, np.nan)
    np.testing.assert_allclose(matrix, reference, atol=1e-5)
    assert pair_rows == len(df)


def test_matrix_with_missing_values_is_close_to_pairwise_complete_pandas():
    df = related_table(missing=0.1).drop(columns=["const", "y"])
    matrix, pair_rows = correlation_matrix(df.to_numpy(dtype=float), "spearman")
    reference = df.corr(method="spearman").to_numpy().copy()
    np.fill_diagonal(reference, np.nan)
    # Ranks and means per column instead of per pair: close, not exact.
    np.testing.assert_allclose(matrix, reference, atol=0.03)
    assert 0.75 * len(df) < pair_rows < len(df)


def test_strongest_partner_count_and_top_partners():
    df = related_table()
    table = frame_table(df)
    relations = column_relations(table, "y")
    names = relations.columns
    assert "y" not in names and "const" not in names
    assert relations.skipped == ["const"]
    reference = df[names].corr(method="spearman")
    for i, name in enumerate(names):
        others = reference[name].drop(name).abs()
        assert names[relations.best[i]] == others.idxmax()
        assert relations.best_value[i] == pytest.approx(reference[name][others.idxmax()], abs=1e-5)
        assert relations.partners[i] == int((others >= rel.PARTNER_LEVEL).sum())
        # (c and a0 have the same ranks, so compare strengths, not names)
        expected_top = others.sort_values(ascending=False).to_numpy()[: rel.TOP_PARTNERS]
        np.testing.assert_allclose(np.abs(relations.top_value[i]), expected_top, atol=1e-5)
    # c = exp(a0): Spearman sees a perfect partner, Pearson much less.
    c = names.index("c")
    assert names[relations.best[c]] == "a0" and relations.best_value[c] == pytest.approx(1.0)
    pearson = relations.extra["best_pearson"][c]
    assert pearson == pytest.approx(df["c"].corr(df["a0"]), abs=1e-6) and pearson < 0.9
    # the target: same rows, same method
    for i, name in enumerate(names):
        expected = df[name].corr(df["y"], method="spearman")
        assert relations.target_value[i] == pytest.approx(expected, abs=1e-5)


def test_only_training_rows_are_used():
    df = related_table(n=400)
    split = split_of(len(df))
    table = frame_table(df, split)
    relations = column_relations(table, "y")
    train = df[split == TRAIN]
    assert relations.rows == relations.pool_rows == len(train) and relations.training_rows
    expected = train[relations.columns].corr(method="spearman").to_numpy().copy()
    np.fill_diagonal(expected, np.nan)
    np.testing.assert_allclose(relations.matrix, expected, atol=1e-5)
    # Changing validation/test rows changes nothing.
    other = df.copy()
    other.loc[split != TRAIN, "n0"] = other.loc[split != TRAIN, "a0"] * 7
    again = column_relations(frame_table(other, split), "y")
    np.testing.assert_array_equal(again.matrix, relations.matrix)


def test_row_sample_limits(monkeypatch):
    df = related_table(n=3000)
    monkeypatch.setattr(rel, "MAX_RELATION_ROWS", 500)
    monkeypatch.setattr(rel, "MIN_RELATION_ROWS", 100)
    relations = column_relations(frame_table(df), "y")
    assert relations.rows == 500 and relations.pool_rows == 3000
    assert not relations.training_rows
    # the work budget: rows × columns² ≤ RELATION_WORK (at least MIN_RELATION_ROWS rows)
    monkeypatch.setattr(rel, "RELATION_WORK", 81 * 200)
    assert column_relations(frame_table(df), "y").rows == 200
    monkeypatch.setattr(rel, "RELATION_WORK", 1)
    assert column_relations(frame_table(df), "y").rows == 100
    # seeded: the same rows every time
    first = column_relations(frame_table(df), "y")
    second = column_relations(frame_table(df), "y")
    np.testing.assert_array_equal(first.matrix, second.matrix)


def test_column_sample_and_blocks(monkeypatch):
    df = related_table()
    full = column_relations(frame_table(df), "y")
    monkeypatch.setattr(rel, "BLOCK_CELLS", 1)  # blocks of 64 columns at least: one block
    monkeypatch.setattr(rel, "_block_size", lambda n: 3)
    blocked = column_relations(frame_table(df), "y")
    np.testing.assert_allclose(blocked.matrix, full.matrix, atol=1e-6)
    np.testing.assert_array_equal(blocked.best, full.best)
    np.testing.assert_array_equal(blocked.top, full.top)
    monkeypatch.setattr(rel, "MAX_RELATION_COLUMNS", 5)
    sampled = column_relations(frame_table(df), "y")
    assert sampled.sampled_columns and sampled.total == 9
    assert len(sampled.columns) + len(sampled.skipped) == 5
    everything = column_relations(frame_table(df), "y", sample_columns=False, cluster=False)
    assert not everything.sampled_columns and everything.matrix is None
    np.testing.assert_array_equal(everything.best, full.best)


def test_clusters_put_related_columns_together():
    relations = column_relations(frame_table(related_table()), "y")
    names = relations.columns
    groups = {names[i]: g for i, g in enumerate(relations.groups)}
    assert groups["a0"] == groups["a1"] == groups["a2"] == groups["c"] >= 0
    assert groups["b0"] == groups["b1"] >= 0 and groups["b0"] != groups["a0"]
    assert groups["n0"] == groups["n1"] == -1
    order = [names[i] for i in relations.order]
    assert sorted(order) == sorted(names)
    a_positions = sorted(order.index(c) for c in ("a0", "a1", "a2", "c"))
    assert a_positions[-1] - a_positions[0] == 3  # contiguous
    overview = rel.overview(relations)
    assert len(overview["starts"]) == len(names) and len(overview["values"]) == len(names)


def test_overview_downsamples_into_blocks():
    df = noise_table(rows=200, real=30, noise=60)
    relations = column_relations(frame_table(df), "y")
    overview = rel.overview(relations, blocks=10)
    assert overview["starts"][0] == 0 and len(overview["starts"]) == 10
    assert all(len(row) == 10 for row in overview["values"])
    values = np.array(overview["values"], dtype=float)
    assert np.nanmax(values) <= 1 and np.nanmin(values) >= 0


def test_chance_level_matches_independent_columns():
    rng = np.random.default_rng(5)
    values = rng.normal(size=(400, 120))
    matrix, rows = correlation_matrix(values, "spearman")
    strongest = np.nanmax(np.abs(matrix), axis=1)
    median = chance_level(rows, 119, 0.5)
    assert np.median(strongest) == pytest.approx(median, rel=0.1)
    assert (strongest < chance_level(rows, 119, 0.999)).all()
    assert chance_level(3, 10) is None and chance_level(100, 0) is None


def test_json_is_compact_and_safe():
    df = noise_table(rows=300, real=30, noise=60)
    table = frame_table(df, split_of(len(df)))
    relations = column_relations(table, "y")
    data = to_json(relations)
    text = json.dumps(data, allow_nan=False)
    assert len(text) < 30_000
    assert data["columns"] == relations.columns and len(data["best"]) == 90
    assert sum(data["histogram"]["counts"]) == 90 and len(data["histogram"]["edges"]) == 51
    assert data["chance"] < data["chance_high"] <= data["suggested_level"]
    profile = profile_table(table, target="y", task="regression")
    assert profile["relations"]["columns"] == data["columns"]
    json.dumps(profile, allow_nan=False)
    assert profile_table(table, relations=False)["relations"] is None


def test_tables_without_two_number_columns():
    df = pd.DataFrame({"x": [1.0, 2, 3, 4], "c": ["a", "b", "a", "b"]})
    relations = column_relations(frame_table(df))
    assert relations.columns == ["x"] and relations.matrix is None
    assert to_json(relations)["overview"] is None
    empty = column_relations(frame_table(pd.DataFrame({"c": ["a", "b"]})))
    assert empty.columns == [] and to_json(empty)["histogram"]["counts"] == [0] * 50
    allmissing = pd.DataFrame({"x": [np.nan] * 5, "z": [1.0, 2, 3, 4, 5], "w": [2.0, 1, 4, 3, 5]})
    relations = column_relations(frame_table(allmissing))
    assert relations.skipped == ["x"] and len(relations.columns) == 2


# --------------------------------------------------------------------------- #
# Readiness check
# --------------------------------------------------------------------------- #


def prepared(df: pd.DataFrame, steps: list | None = None):
    result = run_pipeline(frame_table(df), steps or [], SETTINGS)
    return result.table, result.task


def test_check_finds_injected_noise_and_its_fix_drops_exactly_those_columns():
    df = noise_table()
    table, task = prepared(df)
    issues = [i for i in readiness(table, SETTINGS, task)["issues"] if i["code"] == "no_partner"]
    assert len(issues) == 1
    found = issues[0]
    noise = {c for c in df.columns if c.startswith("n")}
    assert set(found["columns"]) == noise
    assert found["severity"] == "warning"
    assert found["message"].startswith("80 of 120 number features have no strong partner")
    assert "at chance level" in found["message"]
    fix = found["fix"]
    assert fix["type"] == "feature_selection" and fix["params"]["method"] == "no_partner"
    step = {"id": "s1", "type": fix["type"], "params": fix["params"]}
    result = run_pipeline(frame_table(df), [step], SETTINGS)
    assert set(result.reports[0]["removed"]) == noise
    after = readiness(result.table, SETTINGS, result.task)["issues"]
    assert not any(i["code"] == "no_partner" for i in after)


def test_check_notes_a_column_without_partner_that_predicts_the_target():
    df = noise_table()
    df["y"] = df["y"] + 15 * df["n0"]  # n0 relates to no other feature, but to the target
    table, task = prepared(df)
    (found,) = relation_issues(table, "y", task)
    assert "n0" in found["columns"]
    assert "Check before dropping: n0 correlates with the target y" in found["message"]


@pytest.mark.parametrize(
    "make",
    [
        # iris-like: 4 related measurements and a class
        lambda rng: pd.DataFrame(
            {
                **{
                    name: base + rng.normal(scale=0.3, size=150)
                    for name, base in zip(
                        ["sl", "sw", "pl", "pw"], np.repeat(rng.normal(size=(1, 150)), 4, 0)
                    )
                },
                "y": rng.normal(size=150),
            }
        ),
        # california-housing-like: 8 features, partly related, 5,000 rows
        lambda rng: (
            lambda f: pd.DataFrame(
                {
                    "income": f[:, 0],
                    "age": rng.normal(size=5000),
                    "rooms": f[:, 1] + 0.5 * rng.normal(size=5000),
                    "bedrooms": f[:, 1] + 0.6 * rng.normal(size=5000),
                    "population": f[:, 2],
                    "households": f[:, 2] + 0.3 * rng.normal(size=5000),
                    "lat": f[:, 3],
                    "lon": -f[:, 3] + 0.2 * rng.normal(size=5000),
                    "y": f[:, 0] + rng.normal(size=5000),
                }
            )
        )(rng.normal(size=(5000, 4))),
        # 60 independent features (e.g. a questionnaire): no related group at all
        lambda rng: pd.DataFrame(
            {**{f"q{i}": rng.normal(size=500) for i in range(60)}, "y": rng.normal(size=500)}
        ),
        # 60 features that are all related
        lambda rng: noise_table(rows=500, real=60, noise=0),
        # a few noise columns among many related ones: not a clear group
        lambda rng: noise_table(rows=500, real=60, noise=4),
        # many columns, few rows: chance correlations are large, nothing is clear
        lambda rng: noise_table(rows=40, real=40, noise=80),
    ],
    ids=["iris", "housing", "independent", "all-related", "few-noise", "few-rows"],
)
def test_check_stays_quiet_on_ordinary_tables(make):
    df = make(np.random.default_rng(0))
    table, task = prepared(df)
    issues = readiness(table, SETTINGS, task)["issues"]
    assert not any(i["code"] == "no_partner" for i in issues)


def test_suggested_level_grows_with_fewer_rows():
    many = column_relations(frame_table(noise_table(rows=2000)), "y")
    few = column_relations(frame_table(noise_table(rows=150)), "y")
    assert suggested_level(many) == rel.WEAK_LEVEL
    assert suggested_level(few) > 0.3


# --------------------------------------------------------------------------- #
# Feature selection: no strong partner
# --------------------------------------------------------------------------- #


def run_step(table: Table, params: dict, target: str = "y"):
    step = get_step("feature_selection")(params)
    ctx = StepContext(target=target, task="regression", seed=1)
    cleaned = step.clean(table, ctx)
    return step, cleaned, step.run(table, cleaned, ctx), ctx


def test_selection_drops_columns_without_partner_on_training_rows():
    df = related_table(n=400)
    split = split_of(len(df))
    table = frame_table(df, split)
    _, _, result, _ = run_step(table, {"method": "no_partner", "min_partner": 0.3})
    assert set(result.fitted["drop"]) == {"n0", "n1", "const"}
    assert "y" in result.table.columns and "a0" in result.table.columns
    assert result.summary.startswith("Dropped 3 of 9 features without a partner at |ρ| ≥ 0.3")
    details = {d["column"]: d["text"] for d in result.details}
    assert details["const"] == "constant" and details["n0"].startswith("|ρ| 0.")
    # Validation/test rows do not change what is dropped.
    other = df.copy()
    other.loc[split != TRAIN, "n0"] = other.loc[split != TRAIN, "a0"]
    _, _, again, _ = run_step(
        frame_table(other, split), {"method": "no_partner", "min_partner": 0.3}
    )
    assert again.fitted == result.fitted


def test_selection_pearson_and_threshold():
    df = related_table(n=400)
    df["d"] = np.sin(8 * df["a0"])  # related, but not monotonic: weak for both measures
    table = frame_table(df)
    _, _, spearman, _ = run_step(table, {"method": "no_partner", "min_partner": 0.5})
    _, _, pearson, _ = run_step(
        table, {"method": "no_partner", "min_partner": 0.5, "partner_method": "pearson"}
    )
    # c = exp(a0) has a perfect Spearman partner but a weaker Pearson one
    assert "c" not in spearman.fitted["drop"]
    assert "|r|" in pearson.summary
    _, _, nothing, _ = run_step(table, {"method": "no_partner", "min_partner": 0})
    assert nothing.fitted == {"drop": ["const"]}
    single = frame_table(pd.DataFrame({"x": [1.0, 2, 3], "y": [1.0, 2, 4]}))
    _, _, result, _ = run_step(single, {"method": "no_partner"})
    assert result.fitted == {"drop": []} and result.warnings


def test_selection_script_reproduces_the_table(tmp_path):
    from brickdnn.data.export import pandas_script, write_processed_csv

    df = noise_table(rows=600, real=20, noise=30)
    df.insert(0, "id", np.arange(len(df)))
    (tmp_path / "data.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    from brickdnn.data.readers import read_file

    table, _ = read_file(tmp_path / "data.csv")
    steps = [
        {
            "id": "s1",
            "type": "feature_selection",
            "enabled": True,
            "params": {"method": "no_partner", "min_partner": 0.3},
        }
    ]
    result = run_pipeline(table, steps, SETTINGS)
    noise = {c for c in df.columns if c.startswith("n")}
    assert not result.failed and set(result.reports[0]["removed"]) == noise | {"id"}
    info = {
        1: {
            "id": 1,
            "name": "data",
            "filename": "data.csv",
            "delimiter": ",",
            "decimal": ".",
            "header": True,
            "encoding": "utf-8",
            "missing_values": None,
            "columns": table.column_info(),
        }
    }
    script = pandas_script(info, 1, steps, SETTINGS, result, name="noise")
    path = tmp_path / "prepare.py"
    path.write_text(script, encoding="utf-8")
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, str(path), "--output", "out.csv"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    write_processed_csv(result.table, tmp_path / "expected.csv", include_split=False)
    assert (tmp_path / "out.csv").read_text() == (tmp_path / "expected.csv").read_text()


def test_without_partner_lists_constant_columns():
    relations = column_relations(frame_table(related_table()), "y")
    assert without_partner(relations, 0.0) == ["const"]
    assert set(without_partner(relations, 1.01)) == set(relations.columns) | {"const"}
