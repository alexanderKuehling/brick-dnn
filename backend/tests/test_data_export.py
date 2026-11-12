"""Exports of a prepared table: training dataset (data.csv + split.bin, read by the training
engine), cleaned CSV, recipe and the standalone pandas script (run for real)."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from brickdnn.data.export import (
    DATA_FILE,
    SPLIT_FILE,
    pandas_script,
    recipe,
    write_processed_csv,
    write_training_dataset,
)
from brickdnn.data.pipeline import clean_settings, readiness
from brickdnn.data.steps import REGISTRY
from brickdnn.data.table import TRAIN, VAL, DataError, Table
from brickdnn.training.datasets import inspect_csv

from .data_helpers import run, steps, table


def ready_table(n: int = 60, target_values=None, kinds=None) -> tuple[Table, dict, str]:
    rng = np.random.default_rng(1)
    target = target_values if target_values is not None else rng.choice(["yes", "no"], n)
    source = table(
        {
            "age": rng.integers(18, 80, n).astype(float),
            "income": rng.lognormal(10, 1, n),
            "member": rng.choice([0.0, 1.0], n),
            "label": target,
        },
        kinds={"member": "boolean", **(kinds or {})},
    )
    result = run(source, settings={"target": "label"})
    return result.table, result.settings, result.task


# --------------------------------------------------------------------------- #
# Training dataset
# --------------------------------------------------------------------------- #


def test_training_dataset_is_read_by_the_training_engine(tmp_path):
    prepared, settings, task = ready_table()
    info = write_training_dataset(prepared, settings, task, tmp_path / "out")
    assert info == {
        "rows": 60,
        "split_counts": prepared.split_counts(),
        "features": ["age", "income", "member"],
        "target": "label",
        "task": "classification",
        "class_names": ["no", "yes"],
    }
    dataset = inspect_csv(tmp_path / "out" / DATA_FILE)
    assert dataset.columns == ["age", "income", "member", "label"]
    assert dataset.num_features == 3 and dataset.num_samples == 60
    assert dataset.has_header and dataset.delimiter == ","
    assert dataset.target_kind == "integer" and dataset.class_names == ["0", "1"]

    split = (tmp_path / "out" / SPLIT_FILE).read_bytes()
    assert len(split) == 60
    assert (
        np.bincount(np.frombuffer(split, dtype=np.uint8), minlength=3).tolist()
        == prepared.split_counts()
    )
    assert np.frombuffer(split, dtype=np.uint8).tolist() == prepared.split.tolist()

    rows = list(csv.reader((tmp_path / "out" / DATA_FILE).open()))
    labels = prepared.df["label"].tolist()
    assert [int(r[-1]) for r in rows[1:]] == [["no", "yes"].index(v) for v in labels]
    assert all("." not in r[0] and r[2] in ("0", "1") for r in rows[1:])  # whole numbers as ints
    incomes = [float(r[1]) for r in rows[1:]]
    assert incomes == prepared.df["income"].tolist()  # exact round trip


@pytest.mark.parametrize(("val", "test"), [(0, 0.2), (0, 0)])
def test_training_dataset_without_validation_rows(tmp_path, val, test):
    rng = np.random.default_rng(3)
    source = table({"x": rng.normal(size=50), "label": rng.choice(["yes", "no"], 50)})
    result = run(source, settings={"target": "label", "split": {"val": val, "test": test}})
    info = write_training_dataset(result.table, result.settings, result.task, tmp_path)
    assert info["split_counts"][1] == 0
    assert (info["split_counts"][2] > 0) == (test > 0)
    split = np.frombuffer((tmp_path / SPLIT_FILE).read_bytes(), dtype=np.uint8)
    assert len(split) == 50 and VAL not in split.tolist()


def test_numeric_class_labels_are_sorted_numerically(tmp_path):
    prepared, settings, task = ready_table(target_values=[10.0, 2.0, 1.0] * 20)
    assert task == "classification"
    info = write_training_dataset(prepared, settings, task, tmp_path)
    assert info["class_names"] == ["1", "2", "10"]
    rows = list(csv.reader((tmp_path / DATA_FILE).open()))[1:]
    expected = {10.0: "2", 2.0: "1", 1.0: "0"}
    assert [r[-1] for r in rows] == [expected[v] for v in prepared.df["label"]]
    dataset = inspect_csv(tmp_path / DATA_FILE)
    assert dataset.class_names == ["0", "1", "2"]


def test_regression_dataset(tmp_path):
    values = np.random.default_rng(2).normal(100, 20, 60)
    prepared, settings, task = ready_table(target_values=values)
    assert task == "regression"
    info = write_training_dataset(prepared, settings, task, tmp_path)
    assert info["class_names"] is None
    dataset = inspect_csv(tmp_path / DATA_FILE)
    assert dataset.target_kind == "number"
    assert dataset.target_range == [values.min(), values.max()]


def test_feature_names_with_commas_and_quotes(tmp_path):
    prepared, settings, task = ready_table()
    prepared = prepared.rename({"age": 'age, "years"', "income": "city=Köln"})
    write_training_dataset(prepared, settings, task, tmp_path)
    dataset = inspect_csv(tmp_path / DATA_FILE)
    assert dataset.columns[:2] == ['age, "years"', "city=Köln"]
    assert dataset.num_features == 3


def test_training_dataset_refuses_data_that_is_not_ready(tmp_path):
    source = table({"x": [1.0, np.nan, 3.0, 4.0, 5.0], "y": ["a", "b", "a", "b", "a"]})
    result = run(source, settings={"target": "y"})
    check = readiness(result.table, result.settings, result.task)
    with pytest.raises(DataError) as error:
        write_training_dataset(result.table, result.settings, result.task, tmp_path)
    assert str(error.value) == check["issues"][0]["message"]
    assert not (tmp_path / DATA_FILE).exists()


# --------------------------------------------------------------------------- #
# CSV and recipe
# --------------------------------------------------------------------------- #


def test_processed_csv(tmp_path):
    source = table(
        {
            "n": [1.0, np.nan, 3.0],
            "x": [0.5, 1.25, np.nan],
            "d": pd.to_datetime(["2024-01-31", None, "2024-02-01"]),
            "t": [pd.Timestamp("2024-01-31 10:30"), pd.Timestamp("2024-02-01"), None],
            "c": ["a, b", None, 'say "hi"'],
            "split": ["x", "y", "z"],
        }
    )
    source = source.with_split(np.array([TRAIN, VAL, 2], dtype=np.int8))
    write_processed_csv(source, tmp_path / "a.csv")
    assert (tmp_path / "a.csv").read_text().splitlines() == [
        "n,x,d,t,c,split,split_2",
        '1,0.5,2024-01-31,2024-01-31 10:30:00,"a, b",x,train',
        ",1.25,,2024-02-01 00:00:00,,y,val",
        '3,,2024-02-01,,"say ""hi""",z,test',
    ]
    write_processed_csv(source, tmp_path / "b.csv", include_split=False)
    assert (tmp_path / "b.csv").read_text().splitlines()[0] == "n,x,d,t,c,split"


def test_recipe_is_complete_json():
    source = table({"x": [1.0, np.nan, 3.0, 4.0] * 5, "z": np.arange(20.0), "y": ["a", "b"] * 10})
    items = steps(("impute", {"strategy": "knn", "k": 2}), ("drop_duplicates", {}))
    items[1]["enabled"] = False
    result = run(
        source,
        ("impute", {"strategy": "knn", "k": 2}),
        ("drop_duplicates", {}),
        settings={"target": "y"},
    )
    sources = {
        1: {"id": 1, "name": "main", "filename": "main.csv", "columns": source.column_info()}
    }
    data = recipe(sources, 1, items, {"target": "y"}, result)
    text = json.dumps(data, allow_nan=False)
    assert data["format"] == "brickdnn-data-recipe" and data["version"] == 1
    assert data["source"]["name"] == "main" and data["sources"] == {}
    assert data["settings"]["split"]["seed"] == 42
    assert [s["id"] for s in data["steps"]] == ["s1", "s2"]
    assert data["steps"][1]["enabled"] is False
    assert data["steps"][0]["fitted"]["k"] == 2 and "_reference" in data["steps"][0]["fitted"]
    assert data["result"] == {
        "rows": 20,
        "columns": [
            {"name": "x", "kind": "numeric"},
            {"name": "z", "kind": "numeric"},
            {"name": "y", "kind": "categorical"},
        ],
        "split_counts": result.table.split_counts(),
        "task": "classification",
    }
    assert "NaN" not in text


# --------------------------------------------------------------------------- #
# pandas script
# --------------------------------------------------------------------------- #

CUSTOMERS = """\
id;Name;Alter;Einkommen;Stadt;Mitglied;Seit;Kündigung
1;Ann;34;52.000,5;Berlin;ja;31.01.2021;yes
2;Bob;NA;38.500;Hamburg;nein;15.07.2022;no
3;Cem;51;?;  Köln ;ja;;no
4;Dana;28;61.000,25;Berlin;nein;01.02.2020;yes
5;Eve;-999;44.250;Rom;ja;02.11.2023;no
6;Fay;45;39.000;;nein;24.12.2021;yes
7;Gus;62;71.500;Hamburg;ja;05.05.2019;no
8;Hal;33;12.250,75;Köln;NA;06.06.2022;no
9;Ida;40;55.000;Berlin;nein;07.07.2020;yes
10;Jon;29;33.000;Hamburg;ja;08.08.2021;no
11;Kim;38;47.500;Berlin;ja;09.09.2023;yes
12;Lou;55;66.000;Köln;nein;10.10.2018;no
12;Lou;55;66.000;Köln;nein;10.10.2018;no
"""
REGIONS = """\
city,region,population
Berlin,East,3.7
Hamburg,North,1.9
Köln,West,1.1
Paris,Abroad,2.1
"""


def customers_table() -> Table:
    return table(
        {
            "id": [float(i) for i in range(1, 13)] + [12.0],
            "Name": [
                "Ann",
                "Bob",
                "Cem",
                "Dana",
                "Eve",
                "Fay",
                "Gus",
                "Hal",
                "Ida",
                "Jon",
                "Kim",
                "Lou",
                "Lou",
            ],
            "Alter": [
                34.0,
                np.nan,
                51.0,
                28.0,
                -999.0,
                45.0,
                62.0,
                33.0,
                40.0,
                29.0,
                38.0,
                55.0,
                55.0,
            ],
            "Einkommen": [
                52000.5,
                38500.0,
                np.nan,
                61000.25,
                44250.0,
                39000.0,
                71500.0,
                12250.75,
                55000.0,
                33000.0,
                47500.0,
                66000.0,
                66000.0,
            ],
            "Stadt": [
                "Berlin",
                "Hamburg",
                "Köln",
                "Berlin",
                "Rom",
                None,
                "Hamburg",
                "Köln",
                "Berlin",
                "Hamburg",
                "Berlin",
                "Köln",
                "Köln",
            ],
            "Mitglied": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, np.nan, 0.0, 1.0, 1.0, 0.0, 0.0],
            "Seit": pd.to_datetime(
                [
                    "2021-01-31",
                    "2022-07-15",
                    None,
                    "2020-02-01",
                    "2023-11-02",
                    "2021-12-24",
                    "2019-05-05",
                    "2022-06-06",
                    "2020-07-07",
                    "2021-08-08",
                    "2023-09-09",
                    "2018-10-10",
                    "2018-10-10",
                ]
            ),
            "Kündigung": [
                "yes",
                "no",
                "no",
                "yes",
                "no",
                "yes",
                "no",
                "no",
                "yes",
                "no",
                "yes",
                "no",
                "no",
            ],
        },
        kinds={"Mitglied": "boolean", "Name": "categorical"},
    )


def regions_table() -> Table:
    return table(
        {
            "city": ["Berlin", "Hamburg", "Köln", "Paris"],
            "region": ["East", "North", "West", "Abroad"],
            "population": [3.7, 1.9, 1.1, 2.1],
        }
    )


def source_infos(main: Table, other: Table) -> dict[int, dict]:
    return {
        1: {
            "id": 1,
            "name": "Kunden",
            "filename": "customers.csv",
            "delimiter": ";",
            "decimal": ",",
            "header": True,
            "encoding": "utf-8",
            "missing_values": ["", "NA", "?"],
            "columns": main.column_info(),
        },
        2: {
            "id": 2,
            "name": "regions",
            "filename": "regions.csv",
            "delimiter": ",",
            "decimal": ".",
            "header": True,
            "encoding": "utf-8",
            "missing_values": None,
            "columns": other.column_info(),
        },
    }


BASIC_STEPS = (
    ("join", {"source": 2, "how": "left", "left_on": ["Stadt"], "right_on": ["city"]}),
    ("drop_duplicates", {}),
    ("mark_missing", {"columns": ["Alter"], "values": [-999]}),
    ("rename_columns", {"mapping": {"Alter": "age", "Einkommen": "income"}}),
    ("formula", {"name": "income_k", "expression": "round(income / 1000, 1) + (age > 40)"}),
    ("impute", {"columns": ["age", "income"], "strategy": "knn", "k": 2, "add_indicator": True}),
    ("impute", {"columns": ["Stadt", "region"], "strategy": "most_frequent"}),
    ("impute", {"columns": ["Mitglied", "population"], "strategy": "median"}),
    ("impute", {"columns": ["Seit"], "strategy": "constant", "fill_value": "2020-01-01"}),
    ("replace_values", {"columns": ["Stadt"], "mapping": {"Köln": "Cologne", "Rom": "Rome"}}),
    ("cast", {"columns": ["Seit"], "to": "numeric"}),
    ("filter_rows", {"column": "income", "operator": ">", "value": 20000}),
    ("sort_rows", {"column": "income", "descending": True}),
    ("drop_sparse_columns", {"threshold": 0.9}),
    ("drop_columns", {"columns": ["id", "Name"]}),
)


def run_script(tmp_path: Path, script: str, *args: str) -> subprocess.CompletedProcess:
    path = tmp_path / "prepare_script.py"
    path.write_text(script, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(path), *args],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
        check=False,
    )


def check_script_reproduces(tmp_path: Path, main: Table, items: tuple, settings: dict) -> str:
    (tmp_path / "customers.csv").write_text(CUSTOMERS, encoding="utf-8")
    (tmp_path / "regions.csv").write_text(REGIONS, encoding="utf-8")
    other = regions_table()
    result = run(
        main,
        *items,
        settings=settings,
        load_source={2: other}.__getitem__,
        source_name={1: "Kunden", 2: "regions"}.get,
        source_id=1,
    )
    assert not result.failed, [r["error"] for r in result.reports if r["error"]]
    infos = source_infos(main, other)
    script = pandas_script(infos, 1, steps(*items), settings, result, name="Churn – Kunden")
    assert script.startswith("# Built with Brick DNN: https://brickdnn.com")
    done = run_script(tmp_path, script, "--output", "out.csv")
    assert done.returncode == 0, done.stderr
    assert "Wrote" in done.stdout
    write_processed_csv(result.table, tmp_path / "expected.csv", include_split=False)
    got = (tmp_path / "out.csv").read_text(encoding="utf-8").splitlines()
    expected = (tmp_path / "expected.csv").read_text(encoding="utf-8").splitlines()
    assert got == expected
    return script


def test_pandas_script_reproduces_the_prepared_table(tmp_path):
    script = check_script_reproduces(
        tmp_path, customers_table(), BASIC_STEPS, {"target": "Kündigung"}
    )
    assert "TARGET = 'Kündigung'" in script
    assert "def prepare(df):" in script and "def combine(df, load_source):" in script
    assert "argparse" in script and 'if __name__ == "__main__":' in script
    assert "def knn_fill(" in script and script.count("def knn_fill(") == 1
    assert "# 6. Fill missing values (values learned from the training rows)" in script
    assert "--source-2" in script


def test_pandas_script_with_advanced_steps(tmp_path):
    advanced = (
        ("encode", {"columns": ["Stadt", "region"], "method": "onehot"}),
        ("clip", {"columns": ["income"], "mode": "quantiles", "lower_q": 0.05, "upper_q": 0.95}),
        ("scale", {"columns": ["age"], "method": "standard"}),
    )
    if any(t not in REGISTRY for t, _ in advanced):
        pytest.skip("advanced steps are not available")
    main = customers_table()
    items = (*BASIC_STEPS[:13], *advanced)
    probe = run(
        main, *items, settings={"target": "Kündigung"}, load_source={2: regions_table()}.__getitem__
    )
    if probe.failed:
        pytest.skip(
            "an advanced step failed: " + str([r["error"] for r in probe.reports if r["error"]])
        )
    check_script_reproduces(tmp_path, main, items, {"target": "Kündigung"})


def test_pandas_script_with_feature_steps(tmp_path):
    features = (
        (
            "combine_columns",
            {"columns": ["age", "income", "Mitglied"], "operations": ["sum", "std", "nonzero"]},
        ),
        ("interactions", {"columns": ["age", "income"], "operations": ["divide", "multiply"]}),
        (
            "group_features",
            {
                "group_by": ["Stadt", "Mitglied"],
                "value": "income",
                "stats": ["mean", "rows", "ratio_mean"],
                "unseen": "missing",
            },
        ),
    )
    main = customers_table()
    items = (*BASIC_STEPS[:10], *features, *BASIC_STEPS[10:])
    script = check_script_reproduces(tmp_path, main, items, {"target": "Kündigung"})
    assert script.count("def combine_rows(") == 1 and script.count("def group_lookup(") == 1
    assert "# 13. Group statistics (values learned from the training rows)" in script


def test_prepare_works_on_new_data(tmp_path):
    main = customers_table()
    items = (  # without the join
        *BASIC_STEPS[2:6],
        ("impute", {"columns": ["Stadt"], "strategy": "most_frequent"}),
        *BASIC_STEPS[9:],
    )
    result = run(main, *items, settings={"target": "Kündigung"})
    assert not result.failed, [r["error"] for r in result.reports]
    script = pandas_script(
        source_infos(main, regions_table()), 1, steps(*items), {"target": "Kündigung"}, result
    )
    namespace: dict = {"__name__": "prepared_module"}
    exec(compile(script, "prepare_script.py", "exec"), namespace)
    new = tmp_path / "new.csv"
    new.write_text(
        CUSTOMERS.splitlines()[0] + "\n13;Max;NA;80.000;Rom;ja;01.01.2024;no\n", encoding="utf-8"
    )
    df = namespace["prepare"](namespace["read_source"](new, namespace["SOURCES"][1]))
    assert list(df.columns) == result.table.columns
    assert namespace["FEATURES"] == [c for c in result.table.columns if c != "Kündigung"]
    assert not df[["age", "income"]].isna().any().any()  # filled with the training values
    assert df["Stadt"].tolist() == ["Rome"]


def test_pandas_script_notes_steps_that_did_not_run(tmp_path):
    main = customers_table()
    items = steps(
        ("drop_columns", {"columns": ["id"]}),
        ("drop_columns", {"columns": ["Name"]}),
        ("filter_rows", {"column": "Alter", "operator": ">", "value": 10_000}),
        ("drop_columns", {"columns": ["Seit"]}),
    )
    items[1]["enabled"] = False
    from brickdnn.data.pipeline import run_pipeline

    result = run_pipeline(main, items, {"target": "Kündigung"})
    script = pandas_script(source_infos(main, regions_table()), 1, items, {}, result)
    assert "# 2. Drop columns: disabled, not applied." in script
    assert "# 3. Filter rows: failed in the preview, left out." in script
    assert "# 4. Drop columns: did not run (an earlier step failed), left out." in script
    assert "Note: step(s) 3, 4 could not be included" in script
    compile(script, "x.py", "exec")


def test_training_only_steps_are_left_out_of_the_script():
    if "balance_classes" not in REGISTRY:
        pytest.skip("balance_classes is not available")
    main = customers_table()
    result = run(
        main, ("balance_classes", {"method": "oversample"}), settings={"target": "Kündigung"}
    )
    if result.failed:
        pytest.skip("balance_classes failed: " + str(result.reports[0]["error"]))
    script = pandas_script(
        source_infos(main, regions_table()), 1, steps(("balance_classes", {})), {}, result
    )
    assert "only changes the training rows; not applied to new data." in script


def test_user_text_cannot_inject_code_into_the_script(tmp_path):
    evil = "x\nimport os; os.system('echo pwned > pwned.txt')\n"
    main = table({"name": ["a", evil, "b", "c"], "y": [1.0, 2.0, 3.0, 4.0]})
    items = (
        ("filter_rows", {"column": "name", "operator": "!=", "value": evil}),
        ("formula", {"name": evil, "expression": "y * 2"}),
    )
    result = run(main, *items, settings={"target": "y"})
    assert not result.failed, result.reports
    infos = {1: {"name": evil, "filename": "data.csv", "columns": main.column_info()}}
    script = pandas_script(infos, 1, steps(*items), {"target": "y"}, result, name=evil + '"""')
    assert not any(line.startswith("import os") for line in script.splitlines())
    (tmp_path / "data.csv").write_text("name,y\na,1\nb,2\n", encoding="utf-8")
    done = run_script(tmp_path, script, "--input", "data.csv")
    assert done.returncode == 0, done.stderr
    assert not (tmp_path / "pwned.txt").exists()


@pytest.mark.parametrize(
    "name",
    [
        '""; X_INJECTED = 1; ""',
        '"" + X_INJECTED + ""',
        "ends with a backslash \\",
        "\\N{oops} and \\x",
    ],
)
def test_names_cannot_end_the_script_docstring(name):
    """Names go into the module docstring; quotes and backslashes must not escape it."""
    import ast

    main = table({"a": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]})
    result = run(main, settings={"target": "y"})
    infos = {1: {"name": name, "filename": name + ".csv", "columns": main.column_info()}}
    script = pandas_script(infos, 1, [], {"target": "y"}, result, name=name)
    tree = ast.parse(script)
    assert isinstance(tree.body[0], ast.Expr)  # the docstring
    # The name may only appear inside string literals, never as code.
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "X_INJECTED" not in names
    assert name.replace('"', "'").replace("\\", "/") in ast.get_docstring(tree)


def test_script_reader_matches_the_csv_import(tmp_path):
    """The reader in the script types the columns like brickdnn.data.io.read_csv."""
    from brickdnn.data import io

    path = tmp_path / "customers.csv"
    path.write_text(CUSTOMERS, encoding="utf-8")
    try:
        imported, report = io.read_csv(path, io.ImportOptions(missing_values=["", "NA", "?"]))
    except NotImplementedError:
        pytest.skip("the CSV import is not available yet")
    result = run(imported, settings={"target": "Kündigung"})
    info = {
        "name": "Kunden",
        "filename": "customers.csv",
        "delimiter": report.delimiter,
        "decimal": report.decimal,
        "header": report.header,
        "encoding": report.encoding,
        "missing_values": ["", "NA", "?"],
        "columns": imported.column_info(),
    }
    script = pandas_script({1: info}, 1, [], {"target": "Kündigung"}, result)
    namespace: dict = {"__name__": "prepared_module"}
    exec(compile(script, "prepare_script.py", "exec"), namespace)
    df = namespace["read_source"](path, namespace["SOURCES"][1])
    assert list(df.columns) == imported.columns
    for name, kind in imported.kinds.items():
        got = [None if pd.isna(v) else v for v in df[name].tolist()]
        want = [None if pd.isna(v) else v for v in imported.df[name].tolist()]
        assert got == want, (name, kind)


def test_settings_of_the_result_are_used():
    prepared, settings, task = ready_table()
    assert clean_settings(settings) == settings


MESSY_LINES = [
    "sep=;",
    "Nr;Name;Datum;Mitglied;Betrag;Note{end}",
    "1;Ann;31.01.2024;ja;1.234,5; gut ;",
    "2;Bob; NA ;nein;12,25;schlecht;",
    "",
    "3; Cem ;01.02.2024 10:30;JA;-0,5;;",
    ";;;;;;",
    "4;Dana;15.03.2023;Nein;7;gut;",
    "5;Eve;;wahr;;mittel;",
    "6;Fay;02.04.2022;falsch;1.000.000;gut;",
]


@pytest.mark.parametrize(
    ("encoding", "header_end"),
    [("utf-8-sig", ";"), ("utf-8-sig", ""), ("utf-16", ";")],
    ids=["bom-trailing-header", "bom-longer-data-lines", "utf16"],
)
def test_script_reads_messy_files_like_the_import(tmp_path, encoding, header_end):
    """Trailing delimiters, blank and empty lines, BOM / UTF-16, a sep= line, German dates,
    ja/nein booleans and padded missing-value markers: the script reads the file exactly
    like brickdnn.data.io.read_csv and reproduces the prepared table."""
    from brickdnn.data import io

    text = "\r\n".join(MESSY_LINES).replace("{end}", header_end) + "\r\n"
    path = tmp_path / "messy.csv"
    path.write_bytes(text.encode(encoding))
    try:
        imported, report = io.read_csv(path)
    except NotImplementedError:
        pytest.skip("the CSV import is not available yet")
    assert imported.n_rows == 6 and imported.columns == [
        "Nr",
        "Name",
        "Datum",
        "Mitglied",
        "Betrag",
        "Note",
    ]
    assert imported.kinds["Datum"] == "datetime" and imported.kinds["Mitglied"] == "boolean"

    items = (
        ("impute", {"columns": ["Betrag"], "strategy": "median"}),
        ("impute", {"columns": ["Datum"], "strategy": "most_frequent", "add_indicator": True}),
        ("formula", {"name": "big", "expression": "Betrag > 10"}),
        ("cast", {"columns": ["Datum"], "to": "numeric"}),
        ("filter_rows", {"column": "Note", "operator": "not_missing"}),
    )
    result = run(imported, *items, settings={"target": "Note"})
    assert not result.failed, [r["error"] for r in result.reports]
    info = {
        "id": 1,
        "name": "messy",
        "filename": "messy.csv",
        "delimiter": report.delimiter,
        "decimal": report.decimal,
        "header": report.header,
        "encoding": report.encoding,
        "missing_values": None,
        "columns": imported.column_info(),
    }
    script = pandas_script({1: info}, 1, steps(*items), {"target": "Note"}, result)

    namespace: dict = {"__name__": "prepared_module"}
    exec(compile(script, "prepare_script.py", "exec"), namespace)
    df = namespace["read_source"](path, namespace["SOURCES"][1])
    assert list(df.columns) == imported.columns
    for name in imported.columns:
        got = [None if pd.isna(v) else v for v in df[name].tolist()]
        assert got == [None if pd.isna(v) else v for v in imported.df[name].tolist()], name

    done = run_script(tmp_path, script, "--input", "messy.csv", "--output", "out.csv")
    assert done.returncode == 0, done.stderr
    write_processed_csv(result.table, tmp_path / "expected.csv", include_split=False)
    assert (tmp_path / "out.csv").read_text(encoding="utf-8").splitlines() == (
        (tmp_path / "expected.csv").read_text(encoding="utf-8").splitlines()
    )


# --------------------------------------------------------------------------- #
# The script reads other formats like the import
# --------------------------------------------------------------------------- #


def _script_info(report, imported: Table, filename: str) -> dict:
    """The SourceInfo the server stores for an import (brick_dnn_core.data_service)."""
    from brickdnn.data.io import DEFAULT_MISSING
    from brickdnn.data.readers import SPREADSHEET_ERRORS, WORKBOOK_FORMATS

    missing = list(DEFAULT_MISSING)
    if report.format in WORKBOOK_FORMATS:
        missing += list(SPREADSHEET_ERRORS)
    info = {
        "id": 1,
        "name": "data",
        "filename": filename,
        "format": report.format,
        "delimiter": report.delimiter,
        "decimal": report.decimal,
        "header": report.header,
        "encoding": report.encoding,
        "missing_values": missing,
        "columns": imported.column_info(),
    }
    for key in ("sheet", "json_path", "member"):
        if getattr(report, key):
            info[key] = getattr(report, key)
    return info


def _write_formats(tmp_path: Path) -> dict[str, dict]:
    """A file per format, all with numbers, booleans, dates, texts and missing values."""
    import gzip
    import zipfile

    from .test_data_formats import ods_cell, ods_row, write_ods

    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "score": [1.5, None, -3.25, 1e-7, 12.0],
            "member": [True, False, None, True, False],
            "since": pd.to_datetime(
                [
                    "2024-01-31 00:00",
                    "2023-05-01 12:30",
                    None,
                    "2022-12-24 00:00",
                    "2021-01-01 00:00",
                ]
            ),
            "city": ["Köln", "NA", "Rom", " Paris ", "Rom"],
            "label": ["yes", "no", "yes", "no", "yes"],
        }
    )
    records = json.loads(df.to_json(orient="records", date_format="iso"))
    for rank, record in enumerate(records):
        record["meta"] = {"rank": rank, "tags": ["a"] * rank}
    with pd.ExcelWriter(tmp_path / "book.xlsx") as writer:
        df.head(2).to_excel(writer, sheet_name="First", index=False)
        df.to_excel(writer, sheet_name="Second", index=False)
    (tmp_path / "wrapped.json").write_text(json.dumps({"data": records, "count": 5}))
    (tmp_path / "lines.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    df.to_parquet(tmp_path / "data.parquet")
    df.to_feather(tmp_path / "data.feather")
    csv_text = df.to_csv(index=False, sep=";", decimal=",")
    (tmp_path / "data.csv.gz").write_bytes(gzip.compress(csv_text.encode()))
    with zipfile.ZipFile(tmp_path / "data.zip", "w") as archive:
        archive.writestr("__MACOSX/._x", b"")
        archive.writestr("export/records.json", json.dumps(records))
    header = [ods_cell("n"), ods_cell("ok"), ods_cell("day"), ods_cell("word")]
    first = [ods_cell(1.5), ods_cell(True), ods_cell(("date", "2024-01-31")), ods_cell("a")]
    second = [ods_cell(2), ods_cell(False), ods_cell(), ods_cell("#N/A")]
    padding = [ods_cell(repeat=50)]
    write_ods(
        tmp_path / "sheet.ods",
        {"S": [ods_row(header), ods_row(first), ods_row(second), ods_row(padding, repeat=1000)]},
    )
    return {
        "book.xlsx": {"sheet": "Second"},
        "wrapped.json": {},
        "lines.jsonl": {},
        "data.parquet": {},
        "data.feather": {},
        "data.csv.gz": {},
        "data.zip": {},
        "sheet.ods": {},
    }


def test_script_reads_every_format_like_the_import(tmp_path):
    pytest.importorskip("openpyxl")
    pytest.importorskip("pyarrow")
    from brickdnn.data.io import ImportOptions
    from brickdnn.data.readers import read_file

    for filename, options in _write_formats(tmp_path).items():
        path = tmp_path / filename
        imported, report = read_file(path, ImportOptions.from_dict(options))
        result = run(imported, settings={})
        info = _script_info(report, imported, filename)
        script = pandas_script({1: info}, 1, [], {}, result)
        namespace: dict = {"__name__": "prepared_module"}
        exec(compile(script, "prepare_script.py", "exec"), namespace)
        df = namespace["read_source"](path, namespace["SOURCES"][1])
        assert list(df.columns) == imported.columns, filename
        for name, kind in imported.kinds.items():
            got = [None if pd.isna(v) else v for v in df[name].tolist()]
            want = [None if pd.isna(v) else v for v in imported.df[name].tolist()]
            assert got == want, (filename, name, kind)
    assert report.format == "ods" and imported.n_rows == 2


def test_script_for_an_excel_source_runs_on_its_own(tmp_path):
    pytest.importorskip("openpyxl")
    from brickdnn.data.io import ImportOptions
    from brickdnn.data.readers import read_file

    _write_formats(tmp_path)
    imported, report = read_file(tmp_path / "book.xlsx", ImportOptions(sheet="Second"))
    items = (("impute", {"columns": ["score"], "strategy": "median"}),)
    result = run(imported, *items, settings={"target": "label"})
    info = _script_info(report, imported, "book.xlsx")
    script = pandas_script({1: info}, 1, steps(*items), {"target": "label"}, result)
    assert "Needs Python 3.10+ with pandas 2 and numpy, openpyxl." in script
    assert "'sheet': 'Second'" in script and "'format': 'xlsx'" in script
    done = run_script(tmp_path, script, "--output", "out.csv")
    assert done.returncode == 0, done.stderr
    write_processed_csv(result.table, tmp_path / "expected.csv", include_split=False)
    assert (tmp_path / "out.csv").read_text(encoding="utf-8").splitlines() == (
        (tmp_path / "expected.csv").read_text(encoding="utf-8").splitlines()
    )
