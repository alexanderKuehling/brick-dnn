"""The frozen preparation (brickdnn.data.frozen): new rows in the columns of the source
table are prepared exactly like the training rows, with the values the steps learned from
them, for predictions from raw rows."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from brickdnn.data import frozen
from brickdnn.data.export import DATA_FILE, write_training_dataset
from brickdnn.data.steps import REGISTRY
from brickdnn.data.table import TRAIN

from .data_helpers import run, steps, table
from .test_data_export import customers_table, regions_table

# --------------------------------------------------------------------------- #
# The frozen preparation (no database)
# --------------------------------------------------------------------------- #

CUSTOMER_INFO = {"id": 1, "name": "Kunden", "decimal": ",", "missing_values": ["", "NA", "?"]}
# Join, placeholders, renaming, a formula, four kinds of imputation (KNN with indicator
# columns), replacing values, date parts, encoding, clipping and scaling.
STEPS = (
    ("join", {"source": 2, "how": "left", "left_on": ["Stadt"], "right_on": ["city"]}),
    ("mark_missing", {"columns": ["Alter"], "values": [-999]}),
    ("rename_columns", {"mapping": {"Alter": "age", "Einkommen": "income"}}),
    ("formula", {"name": "income_k", "expression": "round(income / 1000, 1) + (age > 40)"}),
    ("impute", {"columns": ["age", "income"], "strategy": "knn", "k": 2, "add_indicator": True}),
    ("impute", {"columns": ["Stadt", "region"], "strategy": "most_frequent"}),
    ("impute", {"columns": ["Mitglied", "population", "income_k"], "strategy": "median"}),
    ("impute", {"columns": ["Seit"], "strategy": "constant", "fill_value": "2020-01-01"}),
    ("replace_values", {"columns": ["Stadt"], "mapping": {"Köln": "Cologne", "Rom": "Rome"}}),
    ("datetime_features", {"columns": ["Seit"], "parts": ["year", "month", "weekday"]}),
    ("encode", {"columns": ["Stadt", "region"], "method": "onehot"}),
    ("clip", {"columns": ["income"], "mode": "quantiles", "lower_q": 0.05, "upper_q": 0.95}),
    ("scale", {"columns": ["age", "income_k"], "method": "standard"}),
    ("drop_columns", {"columns": ["Name"]}),
)


def freeze_run(tmp_path, main, items, settings, other=None):
    """Run the pipeline, write the training dataset and freeze the preparation."""
    other = other if other is not None else regions_table()
    loaders = {"load_source": {2: other}.__getitem__, "source_name": {2: "regions"}.get}
    result = run(main, *items, settings=settings, source_id=1, **loaders)
    assert not result.failed, [r["error"] for r in result.reports if r["error"]]
    written = write_training_dataset(result.table, result.settings, result.task, tmp_path)
    directory = tmp_path / frozen.FROZEN_DIR
    frozen.freeze(
        directory,
        result=result,
        steps=steps(*items),
        source=main,
        source_info=CUSTOMER_INFO,
        prep={"id": 7, "name": "Churn"},
        **loaders,
    )
    return result, written, frozen.read_recipe(directory), directory


def raw_rows(source, recipe) -> list[dict]:
    """The rows of the source table as a client sends them (JSON values)."""
    columns = recipe["columns"]
    return [
        {c["name"]: frozen._jsonable_cell(source.df[c["name"]].iloc[i], c["kind"]) for c in columns}
        for i in range(source.n_rows)
    ]


def training_rows(tmp_path, features) -> np.ndarray:
    with (tmp_path / DATA_FILE).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return np.array([[float(row[f]) for f in features] for row in rows])


def test_raw_rows_reproduce_the_exported_training_table(tmp_path):
    main = customers_table()
    main = main.take(np.arange(main.n_rows - 1))  # without the duplicate last row
    result, written, recipe, directory = freeze_run(tmp_path, main, STEPS, {"target": "Kündigung"})
    features = written["features"]
    assert recipe["features"] == features and recipe["target"] == "Kündigung"
    assert "Stadt=Cologne" in features and "Seit_weekday" in features
    assert "age_missing" in features  # the KNN imputation's indicator

    prepared = frozen.prepare_rows(recipe, raw_rows(main, recipe), directory=directory)
    got = np.array([[row[f] for f in features] for row in prepared.rows])
    # Exactly the values of the training dataset, row by row (all parts of the split).
    np.testing.assert_array_equal(got, training_rows(tmp_path, features))

    # Every row alone gives the same values: nothing is learned from the new rows.
    for i in (1, 2, 5):
        alone = frozen.prepare_rows(recipe, raw_rows(main, recipe)[i], directory=directory)
        np.testing.assert_array_equal([alone.rows[0][f] for f in features], got[i])


def test_more_step_types_reproduce_the_training_table(tmp_path):
    rng = np.random.default_rng(5)
    n = 80
    code = [f"{rng.integers(0, 99):02d}" for _ in range(n)]  # numbers stored as text
    main = table(
        {
            "amount": np.round(rng.lognormal(3, 1, n), 2),
            "size": rng.normal(10, 3, n),
            "code": code,
            "shop": rng.choice(["a", "b", "c", "d", "rare"], n, p=[0.3, 0.3, 0.2, 0.17, 0.03]),
            "color": rng.choice(["red", "blue", None], n),
            "y": rng.choice(["no", "yes"], n),
        },
        kinds={"code": "categorical"},
    )
    items = (
        ("cast", {"columns": ["code"], "to": "numeric"}),
        ("outliers", {"columns": ["size"], "method": "iqr", "action": "clip"}),
        ("transform", {"columns": ["amount"], "function": "yeo_johnson", "replace": False}),
        ("quantile_transform", {"columns": ["size"]}),
        ("bin", {"column": "code", "bins": 4}),
        ("rare_categories", {"columns": ["shop"]}),
        ("encode", {"columns": ["shop"], "method": "frequency"}),
        ("impute", {"columns": ["color"], "strategy": "most_frequent"}),
        ("encode", {"columns": ["color"], "method": "ordinal"}),
    )
    probe = run(main, *items, settings={"target": "y"})
    failed = [(r["type"], r["error"]) for r in probe.reports if r["status"] != "ok"]
    assert not failed, failed
    result, written, recipe, directory = freeze_run(tmp_path, main, items, {"target": "y"})
    prepared = frozen.prepare_rows(recipe, raw_rows(main, recipe), directory=directory)
    features = written["features"]
    got = np.array([[row[f] for f in features] for row in prepared.rows])
    np.testing.assert_array_equal(got, training_rows(tmp_path, features))


def test_target_encoding_of_new_rows_uses_the_training_means(tmp_path):
    rng = np.random.default_rng(3)
    n = 60
    main = table(
        {
            "shop": rng.choice(["a", "b", "c"], n),
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
        }
    )
    items = (("encode", {"columns": ["shop"], "method": "target", "smoothing": 2}),)
    result, written, recipe, directory = freeze_run(tmp_path, main, items, {"target": "y"})
    prepared = frozen.prepare_rows(recipe, raw_rows(main, recipe), directory=directory)
    features = written["features"]
    got = np.array([[row[f] for f in features] for row in prepared.rows])
    expected = training_rows(tmp_path, features)
    # Training rows got out-of-fold values; validation and test rows (like new rows) the
    # means of all training rows.
    unseen = result.table.split != TRAIN
    np.testing.assert_array_equal(got[unseen], expected[unseen])


def fraud_transactions(n: int = 120, seed: int = 11):
    """Card transactions with amounts, balances, risk flags and categories, some missing."""
    rng = np.random.default_rng(seed)
    amount = np.round(rng.lognormal(4, 1.2, n), 2)
    amount[[3, 40]] = np.nan
    balance = np.round(rng.lognormal(7, 1, n), 2)
    balance[[5, 17, 60]] = 0.0  # a division by zero for amount / balance
    balance[8] = np.nan
    category = rng.choice(["food", "travel", "electronics", "games", "fuel"], n).astype(object)
    category[[11, 90]] = None
    category[-1] = "jewelry"  # only in one row, likely outside the training rows
    flags = {
        name: rng.choice([0.0, 1.0], n, p=[0.8, 0.2])
        for name in ("used_vpn", "is_new_merchant", "ip_country_mismatch")
    }
    flags["is_new_merchant"][12] = np.nan
    return table(
        {
            "amount_usd": amount,
            "account_balance_usd": balance,
            "txn_count_last_24h": rng.integers(0, 12, n).astype(float),
            "distance_from_home_km": np.round(rng.exponential(30, n), 1),
            **flags,
            "merchant_category": category,
            "channel": rng.choice(["online", "pos", "atm"], n),
            "is_fraud": rng.choice(["no", "yes"], n, p=[0.85, 0.15]),
        },
        kinds={name: "boolean" for name in flags},
    )


FEATURE_STEPS = (
    (
        "combine_columns",
        {
            "columns": ["used_vpn", "is_new_merchant", "ip_country_mismatch"],
            "operations": ["nonzero", "sum", "mean"],
            "name": "risk_flags",
        },
    ),
    (
        "combine_columns",
        {
            "columns": ["amount_usd", "distance_from_home_km", "txn_count_last_24h"],
            "operations": ["std", "product", "max"],
            "missing": "any",
        },
    ),
    (
        "interactions",
        {
            "columns": ["amount_usd", "account_balance_usd", "distance_from_home_km"],
            "operations": ["multiply", "divide", "subtract", "add"],
            "both_directions": True,
        },
    ),
    (
        "group_features",
        {
            "group_by": ["merchant_category"],
            "value": "amount_usd",
            "stats": ["mean", "median", "std", "min", "max", "count", "rows", "diff_mean"],
            "unseen": "overall",
            "min_rows": 3,
        },
    ),
    (
        "group_features",
        {
            "group_by": ["merchant_category", "channel", "used_vpn"],
            "value": "distance_from_home_km",
            "stats": ["ratio_mean", "rows"],
            "unseen": "missing",
        },
    ),
)


def test_feature_steps_reproduce_the_training_table(tmp_path):
    main = fraud_transactions()
    items = (
        *FEATURE_STEPS,
        # Missing values (missing inputs, divisions by zero, unseen groups) get the median.
        ("impute", {"strategy": "median"}),
        ("impute", {"columns": ["merchant_category"], "strategy": "most_frequent"}),
        ("encode", {"columns": ["merchant_category", "channel"], "method": "onehot"}),
    )
    probe = run(main, *items, settings={"target": "is_fraud"})
    failed = [(r["type"], r["error"]) for r in probe.reports if r["status"] != "ok"]
    assert not failed, failed
    result, written, recipe, directory = freeze_run(tmp_path, main, items, {"target": "is_fraud"})
    features = written["features"]
    assert {
        "risk_flags_nonzero",
        "amount_usd/account_balance_usd",
        "rows_by_merchant_category",
    } <= (set(features))
    assert "distance_from_home_km_ratio_mean_by_merchant_category_channel_used_vpn" in features
    raw = raw_rows(main, recipe)
    prepared = frozen.prepare_rows(recipe, raw, directory=directory)
    got = np.array([[row[f] for f in features] for row in prepared.rows])
    np.testing.assert_array_equal(got, training_rows(tmp_path, features))
    # A single new row gets the same values: the group statistics come from the recipe.
    for i in (3, 5, 11, len(raw) - 1):
        alone = frozen.prepare_rows(recipe, raw[i], directory=directory)
        np.testing.assert_array_equal([alone.rows[0][f] for f in features], got[i])


def test_steps_that_only_change_rows(tmp_path):
    main = customers_table()
    items = (
        ("drop_duplicates", {}),
        ("drop_missing", {"columns": ["Kündigung", "Einkommen"]}),
        ("filter_rows", {"column": "Alter", "operator": ">", "value": 0}),
        ("sort_rows", {"column": "Einkommen"}),
        ("outliers", {"columns": ["Einkommen"], "action": "remove", "remove_from": "all"}),
        ("drop_columns", {"columns": ["Name", "Stadt", "Seit"]}),
        ("impute", {"columns": ["Einkommen", "Alter", "Mitglied"], "strategy": "median"}),
    )
    if "balance_classes" in REGISTRY:
        items += (("balance_classes", {"method": "oversample"}),)
    result, written, recipe, directory = freeze_run(tmp_path, main, items, {"target": "Kündigung"})
    rows = [
        {"id": 1, "Alter": 30, "Einkommen": "41.000,5", "Mitglied": "ja"},
        {"id": 1, "Alter": 30, "Einkommen": "41.000,5", "Mitglied": "ja"},  # a duplicate
        {"id": 2, "Alter": -5, "Einkommen": None, "Mitglied": False},
        {"id": 3, "Alter": 40, "Einkommen": 900000, "Mitglied": 1, "Kündigung": "yes"},
    ]
    prepared = frozen.prepare_rows(recipe, rows, directory=directory)
    # Every row is prepared and stays in its place (no duplicates removed, no sorting).
    assert len(prepared.rows) == 4 and prepared.rows[0] == prepared.rows[1]
    assert prepared.rows[0]["Einkommen"] == 41000.5 and prepared.rows[1]["Mitglied"] == 1.0
    assert prepared.rows[3]["Alter"] == 40.0
    text = "\n".join(prepared.warnings)
    # Filters warn instead of removing rows; the filter on the target is left out.
    assert "Row 3: the training data only has rows where Alter > 0" in text
    assert "Row 3: the training data only has rows with a value in Einkommen" in text
    assert "Row 4: the training data only has values of Einkommen within" in text
    assert "Kündigung" not in text and "Row 1" not in text
    # The median of the training rows fills the missing income.
    medians = recipe["steps"][-2 if "balance_classes" in REGISTRY else -1]["fitted"]["values"]
    assert prepared.rows[2]["Einkommen"] == medians["Einkommen"]
    # Name, Stadt and Seit are dropped before anything reads them: not needed.
    unused = {c["name"] for c in recipe["columns"] if c.get("used") is False}
    assert unused == {"Name", "Stadt", "Seit"}
    assert "not given" not in text
    assert set(frozen.describe(recipe)["skipped_steps"]) >= {"Remove duplicates", "Sort rows"}


def test_messages_for_rows_that_cannot_be_prepared(tmp_path):
    main = customers_table()
    items = (
        ("drop_missing", {"columns": ["Seit"]}),
        ("impute", {"columns": ["Alter", "Einkommen", "Mitglied"], "strategy": "median"}),
        ("encode", {"columns": ["Stadt"], "method": "onehot"}),
        ("datetime_features", {"columns": ["Seit"], "parts": ["year"]}),
        ("drop_columns", {"columns": ["Name", "id"]}),
    )
    result, written, recipe, directory = freeze_run(tmp_path, main, items, {"target": "Kündigung"})

    def errors(rows):
        with pytest.raises(frozen.RawInputError) as caught:
            frozen.prepare_rows(recipe, rows, directory=directory)
        return caught.value.messages

    good = {"Alter": 30, "Einkommen": 50000, "Stadt": "Berlin", "Mitglied": True}
    found = errors(
        [
            {**good, "Alter": "thirty", "Seit": "2024-02-30"},
            {**good, "Mitglied": "maybe", "Seit": "31.01.2024"},
            {**good, "Wohnort": "Bonn"},
        ]
    )
    assert found == [
        "Row 1, 'Alter': 'thirty' is not a number.",
        "Row 1, 'Seit': '2024-02-30' is not a date such as 2024-05-31 (or 2024-05-31T14:30).",
        "Row 2, 'Mitglied': 'maybe' is not true or false.",
        "Row 3: unknown column 'Wohnort'. The columns of 'Kunden' are: id, Name, Alter, "
        "Einkommen, Stadt, Mitglied, Seit.",
    ]
    # A date without a value: the year column has no value after the preparation.
    found = errors([{**good, "Seit": None}])
    assert found == [
        "The row: after the preparation, 'Seit_year' has no value (no value given for Seit). "
        "The model needs a number in every feature column."
    ]
    assert errors([[1, 2]])[0].startswith("The row has 2 values, but 7 are needed")
    assert errors("text")[0].startswith("'inputs' must be a list of rows")
    assert errors([good] * 70)[0] != ""  # no limit without max_rows …
    with pytest.raises(frozen.RawInputError, match="70 rows are too many"):
        frozen.prepare_rows(recipe, [good] * 70, directory=directory, max_rows=64)

    # Unknown categories and columns that are not given are warnings; dates in other formats
    # and numbers with the decimal comma of the file are read like the import did.
    prepared = frozen.prepare_rows(
        recipe,
        [{**good, "Stadt": "Paris", "Seit": "31.01.2024", "Einkommen": "1.234,5"}],
        directory=directory,
    )
    row = prepared.rows[0]
    assert row["Seit_year"] == 2024.0 and row["Einkommen"] == 1234.5
    assert row["Stadt=Berlin"] == 0.0 and row["Stadt=Köln"] == 0.0
    assert prepared.warnings == [
        "The row, 'Stadt': 'Paris' does not occur in the training data; the model treats it "
        "as an unknown category.",
    ]
    prepared = frozen.prepare_rows(recipe, [{"Seit": "2024-01-31"}], directory=directory)
    assert prepared.warnings == [
        "Columns not given, treated as missing values: Alter, Einkommen, Stadt, Mitglied."
    ]
    medians = recipe["steps"][1]["fitted"]["values"]
    assert prepared.rows[0]["Alter"] == medians["Alter"]  # filled with the training median


def test_joined_table_is_stored_with_the_dataset(tmp_path):
    main = customers_table()
    items = (
        ("join", {"source": 2, "how": "inner", "left_on": ["Stadt"], "right_on": ["city"]}),
        ("drop_columns", {"columns": ["Name", "Seit", "id"]}),
        ("encode", {"columns": ["Stadt", "region"], "method": "onehot"}),
        ("impute", {"columns": ["Alter", "Einkommen", "Mitglied", "population"]}),
    )
    result, written, recipe, directory = freeze_run(tmp_path, main, items, {"target": "Kündigung"})
    assert recipe["sources"] == {"2": {"name": "regions", "table": "sources/2"}}
    assert (directory / "sources" / "2" / "table.json").is_file()
    rows = [
        {"Alter": 30, "Einkommen": 1000, "Stadt": "Hamburg", "Mitglied": 1},
        {"Alter": 30, "Einkommen": 1000, "Stadt": "Paris", "Mitglied": 1},
        {"Alter": 30, "Einkommen": 1000, "Stadt": "Bonn", "Mitglied": 1},
    ]
    prepared = frozen.prepare_rows(recipe, rows, directory=directory)
    assert prepared.rows[0]["population"] == 1.9 and prepared.rows[0]["region=North"] == 1.0
    # Paris is in the stored table but not in the training rows (no customer there).
    assert prepared.rows[1]["population"] == 2.1
    # Bonn is not in it: the population of the training rows' median, no region.
    medians = recipe["steps"][-1]["fitted"]["values"]
    assert prepared.rows[2]["population"] == medians["population"]
    assert not any(v for k, v in prepared.rows[2].items() if k.startswith("region="))
    text = "\n".join(prepared.warnings)
    assert "Row 2, 'Stadt': 'Paris' does not occur in the training data" in text
    assert (
        "Row 3: no match in 'regions' (step 1 (Join another table)); the training data only "
        "has rows with a match." in text
    )


def test_old_recipes_and_other_formats_are_not_read(tmp_path):
    assert frozen.read_recipe(tmp_path) is None
    (tmp_path / frozen.RECIPE_FILE).write_text(json.dumps({"format": "x"}))
    assert frozen.read_recipe(tmp_path) is None
    (tmp_path / frozen.RECIPE_FILE).write_text("{")
    assert frozen.read_recipe(tmp_path) is None
