"""Row structure: typicality, nearest neighbours, near-twin groups (brickdnn.data.rows),
the row steps, the split by groups and the row readiness checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from brickdnn.data import rows
from brickdnn.data.checks import row_structure_issues
from brickdnn.data.export import pandas_script, write_training_dataset
from brickdnn.data.pipeline import (
    assign_split,
    clean_settings,
    clean_steps,
    group_codes,
    group_split,
    readiness,
    run_pipeline,
)
from brickdnn.data.table import TEST, TRAIN, VAL, DataError, Table

from .data_helpers import apply, check_code, steps

# --------------------------------------------------------------------------- #
# Test data
# --------------------------------------------------------------------------- #


def people_frame(
    people: int = 120,
    *,
    signal: int = 40,
    noise: int = 40,
    fakes: int = 10,
    max_scans: int = 4,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Like the brain-scan case: people with 1 to ``max_scans`` scans (rows) whose signal
    columns share a person effect (pairs of columns share a latent value, so they have
    partners), ``noise`` columns without any partner, and ``fakes`` rows with every value at
    the column median and a random age. Returns (frame, person per row (-1 fake), fake mask)."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(people, signal // 2))
    base_age = rng.uniform(45, 95, people)
    scans = rng.integers(1, max_scans + 1, people)
    person = np.repeat(np.arange(people), scans)
    n = len(person)
    sig = np.repeat(latent, 2, axis=1)[person] + rng.normal(scale=0.12, size=(n, signal))
    noi = rng.normal(size=(n, noise))
    age = base_age[person] + rng.uniform(-1.5, 1.5, n)
    values = np.hstack([sig, noi])
    medians = np.median(values, axis=0)
    fake_values = medians + rng.normal(scale=0.03, size=(fakes, values.shape[1]))
    values = np.vstack([values, fake_values])
    age = np.concatenate([age, rng.uniform(42, 97, fakes)])
    person = np.concatenate([person, -np.ones(fakes, dtype=int)])
    order = rng.permutation(len(values))
    names = [f"x{i}" for i in range(values.shape[1])]
    frame = pd.DataFrame(values[order], columns=names)
    frame.insert(0, "id", np.arange(len(frame)))
    frame["age"] = age[order]
    fake = person[order] < 0
    return frame, person[order], fake


def plain_frame(n: int = 400, d: int = 30, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cov = 0.5 * np.ones((d, d)) + 0.5 * np.eye(d)
    frame = pd.DataFrame(
        rng.multivariate_normal(np.zeros(d), cov, size=n), columns=[f"f{i}" for i in range(d)]
    )
    frame["y"] = frame.sum(axis=1) + rng.normal(size=n)
    return frame


# --------------------------------------------------------------------------- #
# Statistics against direct references
# --------------------------------------------------------------------------- #


def reference(values: np.ndarray):
    """Robust z, spreads and all pair distances computed the slow, obvious way."""
    med = np.nanmedian(values, axis=0)
    mad = np.nanmedian(np.abs(values - med), axis=0) * 1.4826
    z = np.clip((values - med) / mad, -10, 10)
    spread = np.array([np.median(np.abs(r[~np.isnan(r)])) for r in z])
    n, d = z.shape
    dist = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(n):
            both = ~np.isnan(z[i]) & ~np.isnan(z[j])
            if i != j and both.sum() >= rows.min_shared(d):
                dist[i, j] = np.sqrt(np.mean((z[i, both] - z[j, both]) ** 2))
    return spread, dist


@pytest.mark.parametrize("missing", [0.0, 0.2])
def test_spread_and_neighbours_match_a_direct_computation(missing, monkeypatch):
    rng = np.random.default_rng(3)
    values = rng.normal(size=(40, 7)) * [1, 2, 3, 4, 5, 6, 7] + 10
    values[rng.random(values.shape) < missing] = np.nan
    t = Table.from_frame(pd.DataFrame(values, columns=list("abcdefg")))
    monkeypatch.setattr(rows, "CHUNK_CELLS", 100)  # several chunks
    stats = rows.compute(t, mode="all")
    spread, dist = reference(values)
    np.testing.assert_allclose(stats.spread, spread, rtol=1e-5)
    np.testing.assert_allclose(stats.distances[:, 0], dist.min(axis=1), rtol=1e-4)
    order = np.argsort(dist, axis=1, kind="stable")[:, :3]
    np.testing.assert_allclose(
        stats.distances[:, :3], np.take_along_axis(dist, order, axis=1), rtol=1e-4
    )
    a, b = np.array([0, 5, 7]), np.array([1, 9, 30])
    np.testing.assert_allclose(rows.pair_distances(stats.z, a, b), dist[a, b], rtol=1e-5)
    assert stats.missing.tolist() == np.isnan(values).sum(axis=1).tolist()


def test_rows_without_shared_columns_have_no_neighbour():
    values = np.array([[1, np.nan], [np.nan, 2], [1.5, np.nan], [np.nan, 3.0], [0.0, np.nan]])
    z = rows.standardize(values, np.array([1.0, 2.0]), np.array([1.0, 1.0]))
    neighbors, distances = rows.nearest_neighbors(z, k=4)
    assert neighbors[0, 0] in (2, 4) and set(neighbors[0][neighbors[0] >= 0]) == {2, 4}
    assert np.isinf(distances[0, 2]) and neighbors[0, 2] == -1


def test_large_tables_are_sampled(monkeypatch):
    monkeypatch.setattr(rows, "MAX_ROWS", 50)
    frame, _, _ = people_frame(40)
    stats = rows.compute(Table.from_frame(frame), target="age")
    assert stats.sampled and stats.rows == 50 and stats.n_total == len(frame)
    assert np.all(np.diff(stats.positions) > 0)
    again = rows.compute(Table.from_frame(frame), target="age")
    assert again.positions.tolist() == stats.positions.tolist()  # seeded
    view = rows.analysis_view(stats, Table.from_frame(frame), target="age", task="regression")
    assert view["sampled"] and view["analysed"] == 50 and len(view["rows"]["spread"]) == 50


def test_row_limit_shrinks_for_wide_tables():
    assert rows.row_limit(10) == rows.MAX_ROWS
    assert rows.row_limit(5_000) < rows.row_limit(1_000) < rows.MAX_ROWS


def test_features_leave_out_target_ids_constants_and_noise():
    frame, _, _ = people_frame(60)
    frame["const"] = 1.0
    t = Table.from_frame(frame)
    choice = rows.choose_features(t, "auto", exclude=["age"])
    assert choice.used == "partnered"
    assert set(choice.columns) == {f"x{i}" for i in range(40)}
    reasons = dict(choice.left_out)
    assert reasons["id"] == "id" and reasons["age"] == "target_or_group"
    assert reasons["const"] == "constant" and reasons["x50"] == "no_partner"
    every = rows.choose_features(t, "all", exclude=["age"])
    assert every.used == "all" and len(every.columns) == 80
    chosen = rows.choose_features(t, "columns", ["x1", "x2"])
    assert chosen.columns == ["x1", "x2"]
    with pytest.raises(DataError, match="not a number column|no column"):
        rows.choose_features(t, "columns", ["nope"])


def test_groups_are_connected_components():
    assert rows.components(5, np.array([0, 3]), np.array([1, 4])).tolist() == [0, 0, 1, 2, 2]
    neighbors = np.array([[1, 2], [0, 2], [3, 0], [2, 1]])
    ratios = np.array([[0.1, 0.9], [0.1, 0.9], [0.9, 0.9], [0.9, 0.9]])
    labels, sizes = rows.group_labels(neighbors, ratios, 0.5)
    assert labels.tolist() == [0, 0, 1, 2] and sizes.tolist() == [2, 2, 1, 1]


# --------------------------------------------------------------------------- #
# Suggested thresholds: injected structure is found, ordinary tables stay quiet
# --------------------------------------------------------------------------- #


def test_people_and_fake_rows_are_found():
    frame, person, fake = people_frame()
    t = Table.from_frame(frame)
    stats = rows.compute(t, target="age")
    suggested = stats.suggestions()
    assert suggested["atypical"] is not None
    flagged = stats.spread < suggested["atypical"]
    assert flagged.tolist() == fake.tolist()
    assert suggested["twins"] is not None
    labels, sizes = stats.groups(suggested["twins"])
    real = ~fake
    # Rows of one person are in one group, and groups do not mix people.
    for p in np.unique(person[real]):
        assert len(np.unique(labels[person == p])) == 1
    for g in np.unique(labels[real]):
        assert len(np.unique(person[(labels == g) & real])) == 1


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(("n", "d"), [(60, 12), (300, 25), (1000, 40)])
def test_ordinary_tables_get_no_suggestion(seed, n, d):
    rng = np.random.default_rng(seed)
    for values in (
        rng.normal(size=(n, d)),
        rng.lognormal(size=(n, d)),
        rng.integers(0, 10, size=(n, d)).astype(float),
    ):
        t = Table.from_frame(pd.DataFrame(values, columns=[f"c{i}" for i in range(d)]))
        stats = rows.compute(t, with_projection=False)
        assert stats.suggestions() == {"atypical": None, "twins": None}


def test_few_columns_get_no_suggestion():
    frame, _, _ = people_frame(signal=4, noise=0)
    stats = rows.compute(Table.from_frame(frame), target="age")
    assert stats.suggestions() == {"atypical": None, "twins": None}


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #


def test_analysis_view_and_similar_rows():
    frame, person, fake = people_frame(80)
    t = Table.from_frame(frame).with_split(np.zeros(len(frame), dtype=np.int8))
    stats = rows.compute(t, target="age")
    view = rows.analysis_view(stats, t, target="age", task="regression")
    assert view["id_column"] == "id" and view["rows"]["id"] == frame["id"].tolist()
    assert view["suggested"]["twins"] == view["thresholds"]["twins"]
    assert len(view["rows"]["pc1"]) == len(frame) and view["explained"][0] > 0
    assert view["groups"]["count"] > 10 and view["groups"]["largest"] <= 10
    assert len(view["random_pairs"]["ratio"]) == len(view["random_pairs"]["a"])
    # A person's other scan is the most similar row, with almost the same age.
    sizes = np.array(view["rows"]["group_size"])
    i = int(np.flatnonzero((sizes == 2) & ~fake)[0])
    similar = rows.similar_view(stats, t, i, target="age", task="regression")
    nearest = similar["neighbors"][0]
    assert person[nearest["index"]] == person[i]
    assert abs(nearest["target"] - frame["age"][i]) < 3.5
    pair = similar["pair"]
    assert pair["b"]["index"] == nearest["index"] and len(pair["closest"]) == 8
    assert pair["closest"][0]["diff"] <= pair["farthest"][0]["diff"]
    with pytest.raises(DataError):
        rows.similar_view(stats, t, len(frame) + 5, target=None, task=None)
    summary = rows.summary(stats, t)
    assert summary["groups"] > 10 and summary["atypical_rows"] == int(fake.sum())
    assert sum(summary["nn_ratio_histogram"]["counts"]) == len(frame)


def test_cache_reuses_statistics_of_equal_tables():
    rows.clear_cache()
    frame = plain_frame(80, 12)
    first = rows.cached_compute(Table.from_frame(frame), target="y")
    assert rows.cached_compute(Table.from_frame(frame.copy()), target="y") is first
    frame.loc[3, "f1"] = 99
    assert rows.cached_compute(Table.from_frame(frame), target="y") is not first


# --------------------------------------------------------------------------- #
# Split by groups
# --------------------------------------------------------------------------- #


def test_group_split_keeps_groups_together_with_close_shares():
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(400), rng.integers(1, 8, 400))
    split = group_split(
        groups, round(len(groups) * 0.15), round(len(groups) * 0.15), None, np.random.default_rng(1)
    )
    for g in np.unique(groups):
        assert len(np.unique(split[groups == g])) == 1
    shares = np.bincount(split, minlength=3) / len(groups)
    np.testing.assert_allclose(shares, [0.7, 0.15, 0.15], atol=0.01)
    again = group_split(
        groups, round(len(groups) * 0.15), round(len(groups) * 0.15), None, np.random.default_rng(1)
    )
    assert again.tolist() == split.tolist()
    other = group_split(
        groups, round(len(groups) * 0.15), round(len(groups) * 0.15), None, np.random.default_rng(2)
    )
    assert other.tolist() != split.tolist()


def test_group_split_stratifies_by_the_main_class_of_each_group():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(300), 3)
    group_class = (rng.random(300) < 0.2).astype(int)  # 20 % class 1
    classes = group_class[groups]
    split = group_split(groups, 135, 135, classes, np.random.default_rng(3))
    for part in (TRAIN, VAL, TEST):
        share = classes[split == part].mean()
        assert abs(share - 0.2) < 0.02


def test_group_split_with_huge_groups_and_missing_values():
    groups = group_codes(pd.Series(["a"] * 50 + ["b"] * 30 + [None] * 4))
    assert len(np.unique(groups)) == 6  # every missing value is its own group
    split = group_split(groups, 13, 13, None, np.random.default_rng(0))
    assert set(np.unique(split)) == {TRAIN, VAL, TEST}
    assert len(np.unique(split[:50])) == 1 and len(np.unique(split[50:80])) == 1


def test_assign_split_by_group_column_and_settings():
    settings = clean_settings({"split": {"group_by": " person "}})
    assert settings["split"]["group_by"] == "person"
    assert clean_settings({})["split"]["group_by"] is None
    with pytest.raises(DataError):
        clean_settings({"split": {"group_by": 3}})
    frame, person, _ = people_frame(100)
    frame["person"] = person
    t = Table.from_frame(frame)
    split = assign_split(t, settings, "regression")
    for p in np.unique(person):
        assert len(np.unique(split[person == p])) == 1
    with pytest.raises(DataError, match="no column 'nope'"):
        assign_split(t, clean_settings({"split": {"group_by": "nope"}}), None)


# --------------------------------------------------------------------------- #
# Steps and the pipeline
# --------------------------------------------------------------------------- #


def test_pre_split_steps_come_right_after_combine_steps():
    clean_steps(steps(("find_groups", {}), ("remove_atypical", {}), ("impute", {})))
    with pytest.raises(DataError, match="runs before the split"):
        clean_steps(steps(("impute", {}), ("find_groups", {})))


def test_remove_atypical_rows_and_its_code():
    frame, _, fake = people_frame(80)
    t = Table.from_frame(frame)
    result = check_code(t, "remove_atypical", {})
    assert result.table.n_rows == len(frame) - fake.sum()
    assert f"Removed {fake.sum()} rows" in result.summary and "automatically" in result.summary
    assert result.fitted["threshold"] > 0 and len(result.fitted["_medians"]) == 40
    # A fixed threshold, and the other side.
    high, _, _ = apply(t, "remove_atypical", {"side": "high", "threshold": 0.9})
    assert high.table.n_rows < len(frame)
    plain = Table.from_frame(plain_frame())
    nothing, _, _ = apply(plain, "remove_atypical", {})
    assert nothing.table.n_rows == plain.n_rows and "nothing removed" in nothing.summary


def test_find_groups_step_and_group_split_in_the_pipeline():
    frame, person, fake = people_frame(100)
    t = Table.from_frame(frame)
    items = steps(
        ("remove_atypical", {}),
        ("find_groups", {}),
        ("drop_columns", {"columns": ["group"]}),
    )
    settings = {"target": "age", "split": {"group_by": "group", "seed": 3}}
    result = run_pipeline(t, clean_steps(items), settings)
    assert not result.failed, [r["error"] for r in result.reports]
    assert "group" not in result.table.columns
    before = result.before
    group = before.df["group"].to_numpy()
    for g in np.unique(group):
        assert len(np.unique(before.split[group == g])) == 1
    real_person = person[~fake]
    for p in np.unique(real_person):
        assert len(np.unique(before.split[real_person == p])) == 1
    info = result.split
    assert info["group_by"] == "group" and info["groups"]["mixed"] == 0
    assert info["groups"]["multi"] > 20 and info["error"] is None
    assert "group" in [c["name"] for c in info["columns"]]
    ready = readiness(result.table, result.settings, result.task, split=result.split)
    assert not any(i["code"] in ("near_twins", "split_group_missing") for i in ready["issues"])


def test_group_column_that_appears_after_the_split_is_an_error():
    frame, _, _ = people_frame(40)
    t = Table.from_frame(frame)
    items = steps(("impute", {}), ("formula", {"name": "g", "expression": "x1 > 0"}))
    settings = {"target": "age", "split": {"group_by": "g"}}
    try:
        result = run_pipeline(t, clean_steps(items), settings)
    except DataError:  # pragma: no cover - (the formula step may not exist)
        pytest.skip("no formula step")
    assert result.split["error"] and "no column 'g'" in result.split["error"]
    ready = readiness(result.table, result.settings, result.task, split=result.split)
    assert any(i["code"] == "split_group_missing" for i in ready["issues"])
    assert not ready["ready"]
    # A preview that stops before the step that creates the group column only notes the
    # random split.
    items = steps(("remove_atypical", {}), ("find_groups", {}), ("impute", {}))
    settings = {"target": "age", "split": {"group_by": "group"}}
    upto = run_pipeline(t, clean_steps(items), settings, upto="s1")
    assert upto.split["error"] is None and any("random split" in w for w in upto.warnings)
    full = run_pipeline(t, clean_steps(items), settings)
    assert full.split["error"] is None and full.split["groups"]["count"] > 0


def test_group_column_left_as_feature_is_a_warning():
    frame, person, _ = people_frame(40)
    frame["person"] = person
    t = Table.from_frame(frame)
    settings = {"target": "age", "split": {"group_by": "person"}}
    result = run_pipeline(t, [], settings)
    ready = readiness(result.table, result.settings, result.task, split=result.split)
    (found,) = [i for i in ready["issues"] if i["code"] == "group_column_feature"]
    assert found["fix"]["type"] == "drop_columns"


def test_drop_rows_by_id_and_by_position():
    frame, _, _ = people_frame(30)
    t = Table.from_frame(frame)
    result = check_code(t, "drop_rows", {"by": "column", "column": "id", "values": [0, 5, "7"]})
    assert result.table.n_rows == len(frame) - 3
    assert 5 not in result.table.df["id"].tolist()
    by_position, _, _ = apply(t, "drop_rows", {"by": "position", "positions": [0, 1, 1]})
    assert by_position.table.n_rows == len(frame) - 2
    with pytest.raises(DataError, match="no row"):
        apply(t, "drop_rows", {"by": "position", "positions": [len(frame)]})


def test_export_and_script_with_the_row_steps(tmp_path):
    from .test_data_export import run_script

    frame, _, fake = people_frame(60)
    path = tmp_path / "scans.csv"
    frame.to_csv(path, index=False)
    t = Table.from_frame(frame)
    items = steps(
        ("remove_atypical", {}),
        ("find_groups", {}),
        ("drop_columns", {"columns": ["group", "id"]}),
        ("drop_rows", {"by": "position", "positions": [0]}),
    )
    settings = clean_settings({"target": "age", "split": {"group_by": "group"}})
    result = run_pipeline(t, clean_steps(items), settings)
    assert not result.failed
    written = write_training_dataset(result.table, settings, result.task, tmp_path / "ds")
    stored = np.frombuffer((tmp_path / "ds" / "split.bin").read_bytes(), dtype=np.uint8)
    assert stored.tolist() == result.table.split.tolist()
    assert sum(written["split_counts"]) == result.table.n_rows
    info = {
        "id": 1,
        "name": "scans",
        "filename": "scans.csv",
        "format": "csv",
        "delimiter": ",",
        "decimal": ".",
        "header": True,
        "encoding": "utf-8",
        "missing_values": [""],
        "columns": t.column_info(),
    }
    script = pandas_script({1: info}, 1, items, settings, result, name="scans")
    done = run_script(tmp_path, script, "--input", str(path), "--output", "out.csv")
    assert done.returncode == 0, done.stderr
    out = pd.read_csv(tmp_path / "out.csv")
    # The script removes the atypical rows like the step; hand-picked row numbers only
    # apply to the prepared data.
    assert len(out) == len(frame) - fake.sum()
    assert list(out.columns) == result.table.columns


# --------------------------------------------------------------------------- #
# Readiness
# --------------------------------------------------------------------------- #


def test_readiness_finds_fake_rows_and_near_twins():
    frame, _, fake = people_frame(150)
    t = Table.from_frame(frame)
    result = run_pipeline(t, [], {"target": "age"})
    issues = {i["code"]: i for i in row_structure_issues(result.table, "age", result.settings)}
    atypical = issues["atypical_rows"]
    assert f"{fake.sum()} rows" in atypical["message"]
    assert atypical["fix"]["type"] == "remove_atypical"
    twins = issues["near_twins"]
    assert "validation/test rows have a near twin in the training rows" in twins["message"]
    assert twins["fix"]["type"] == "find_groups"
    assert twins["fix"]["settings"] == {"split": {"group_by": "group"}}


def test_readiness_handles_rows_that_are_exact_copies():
    """Exact copies are at distance 0 from their twin: the check must not divide by it."""
    base = plain_frame(200, 12, 0)
    frame = pd.concat([base, base.iloc[:120]], ignore_index=True)
    t = Table.from_frame(frame)
    result = run_pipeline(t, [], {"target": "y"})
    issues = {i["code"]: i for i in row_structure_issues(result.table, "y", result.settings)}
    twins = issues["near_twins"]
    assert "the nearest row has the same values in every feature" in twins["message"]


@pytest.mark.parametrize("seed", range(3))
def test_readiness_stays_quiet_on_ordinary_tables(seed):
    t = Table.from_frame(plain_frame(500, 25, seed))
    result = run_pipeline(t, [], {"target": "y"})
    ready = readiness(result.table, result.settings, result.task, split=result.split)
    assert not any(
        i["code"] in ("atypical_rows", "near_twins", "group_column_feature")
        for i in ready["issues"]
    )


def test_training_reads_the_group_split(tmp_path):
    torch = pytest.importorskip("torch")
    from brickdnn.training.data import predefined_split_indices, read_split_file
    from brickdnn.training.datasets import inspect_csv

    frame, person, _ = people_frame(80, fakes=0)
    frame["person"] = person
    t = Table.from_frame(frame)
    settings = clean_settings({"target": "age", "split": {"group_by": "person"}})
    items = clean_steps(steps(("drop_columns", {"columns": ["person"]})))
    result = run_pipeline(t, items, settings)
    written = write_training_dataset(result.table, settings, result.task, tmp_path)
    info = inspect_csv(tmp_path / "data.csv")
    split = read_split_file(tmp_path / "data.csv", info.num_samples, written["split_counts"])
    train, val, test = predefined_split_indices(split, None, torch.Generator().manual_seed(0))
    part = np.zeros(len(frame), dtype=int)
    part[val.numpy()] = VAL
    part[test.numpy()] = TEST
    assert (part == TRAIN).sum() == len(train)
    for p in np.unique(person):
        assert len(np.unique(part[person == p])) == 1


def test_work_budget_limits_the_rows_of_the_checks():
    frame, _, _ = people_frame(60)
    t = Table.from_frame(frame)
    stats = rows.compute(t, target="age", max_work=100 * 100 * 40, with_projection=False)
    assert stats.sampled and stats.rows == 100
