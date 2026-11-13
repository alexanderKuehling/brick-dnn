"""The "Extract from text" step (extract_text in brickdnn.data.steps.columns): a part of a
text column into a new column, optionally mapped to numbers. The step, the pandas script and
predictions from raw rows give the same values."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from brickdnn.data import frozen, io
from brickdnn.data.export import pandas_script, write_processed_csv, write_training_dataset
from brickdnn.data.steps import StepContext, catalog
from brickdnn.data.table import DataError

from .data_helpers import apply, check_code, run, steps, table, values
from .test_data_export import run_script
from .test_data_frozen import raw_rows, training_rows

DECKS = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}
CABINS = ["C85", None, "D32", "E46", "B57 B59 B63 B66", "T", "a6", "G6", "F G73", "X1"]


def cabins_table():
    return table(
        {
            "Cabin": CABINS,
            "Name": [
                "Braund, Mr. Owen Harris",
                "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
                "Heikkinen, Miss. Laina",
                None,
                "Allen, Mr. William Henry",
                "Moran, Mr. James",
                "McCarthy, Mr. Timothy J",
                "Palsson, Master. Gosta Leonard",
                "Johnson, Mrs. Oscar W",
                "Nasser, Mrs. Nicholas",
            ],
            "Fare": [71.3, 7.25, 8.05, 53.1, 8.05, 8.46, 51.86, 21.08, 11.13, 30.07],
        },
        kinds={"Cabin": "categorical", "Name": "text"},
    )


def extract(source, params, ctx=None):
    """Run the step and check that its pandas code gives the same table."""
    return check_code(source, "extract_text", params, ctx, re=re)


def error_of(source, params, ctx=None) -> str:
    with pytest.raises(DataError) as info:
        apply(source, "extract_text", params, ctx)
    return str(info.value)


def test_catalog_entry():
    spec = next(s for s in catalog()["steps"] if s["type"] == "extract_text")
    assert spec["label"] == "Extract from text" and spec["group"] == "columns"
    assert spec["fitted"] is False and spec["example"]["mapping"]["D"] == 4
    params = {p["name"]: p for p in spec["params"]}
    assert list(params) == [
        "column",
        "name",
        "part",
        "length",
        "pattern",
        "index",
        "strip",
        "case",
        "mapping",
    ]
    assert params["column"]["column_kinds"] == ["categorical", "text"]
    assert [o["value"] for o in params["part"]["options"]] == ["first", "last", "pattern", "word"]
    assert params["part"]["default"] == "first" and params["case"]["default"] == "upper"
    assert params["length"]["visible_if"] == {"param": "part", "values": ["first", "last"]}
    assert params["pattern"]["visible_if"] == {"param": "part", "values": ["pattern"]}
    assert params["index"]["visible_if"] == {"param": "part", "values": ["word"]}
    assert params["mapping"]["key_kind"] == "value" and params["mapping"]["optional"] is True


# --------------------------------------------------------------------------- #
# Parts
# --------------------------------------------------------------------------- #


def test_first_and_last_characters():
    source = cabins_table()
    result = extract(source, {"column": "Cabin", "name": "Deck"})
    assert values(result.table, "Deck") == ["C", None, "D", "E", "B", "T", "A", "G", "F", "X"]
    assert result.table.kinds["Deck"] == "categorical"
    assert result.table.columns[-1] == "Deck"
    assert result.summary == (
        "Took the first character of 'Cabin' into 'Deck': 9 values in 9 categories, 1 missing."
    )
    assert result.details == [
        {
            "column": "Deck",
            "text": "parts taken: 'C' (1), 'D' (1), 'E' (1), 'B' (1), 'T' (1) and 4 more",
        }
    ]

    result = extract(source, {"column": "Cabin", "name": "end", "part": "last", "length": 2})
    assert values(result.table, "end") == [
        "85",
        None,
        "32",
        "46",
        "66",
        "T",
        "A6",
        "G6",
        "73",
        "X1",
    ]
    assert result.summary.startswith("Took the last 2 characters of 'Cabin' into 'end'")

    # The part taken can replace the column itself.
    result = extract(source, {"column": "Cabin", "name": "Cabin", "length": 3, "case": "keep"})
    assert values(result.table, "Cabin")[:3] == ["C85", None, "D32"]
    assert result.table.columns == source.columns


def test_pattern_takes_the_first_group_or_the_whole_match():
    source = cabins_table()
    title = extract(
        source,
        {
            "column": "Name",
            "name": "Title",
            "part": "pattern",
            "pattern": r",\s*([^.]+)\.",
            "case": "keep",
        },
    )
    assert values(title.table, "Title") == [
        "Mr",
        "Mrs",
        "Miss",
        None,
        "Mr",
        "Mr",
        "Mr",
        "Master",
        "Mrs",
        "Mrs",
    ]
    assert title.summary.startswith(r"Took the part matching ',\s*([^.]+)\.' of 'Name' into")
    whole = extract(source, {"column": "Cabin", "name": "x", "part": "pattern", "pattern": r"\d+"})
    assert values(whole.table, "x") == ["85", None, "32", "46", "57", None, "6", "6", "73", "1"]
    # The pattern is used as typed: case matters and spaces count.
    upper = extract(
        source, {"column": "Cabin", "name": "x", "part": "pattern", "pattern": "[A-Z]\\d"}
    )
    assert values(upper.table, "x")[5:7] == [None, None]
    spaced = extract(
        source, {"column": "Cabin", "name": "x", "part": "pattern", "pattern": " (B\\d+)"}
    )
    assert values(spaced.table, "x") == [None] * 4 + ["B59"] + [None] * 5
    # An optional group that does not take part gives a missing value.
    optional = extract(
        source, {"column": "Cabin", "name": "x", "part": "pattern", "pattern": "C(X)?"}
    )
    assert values(optional.table, "x") == [None] * 10
    assert optional.warnings == ["Every value of 'x' is missing."]


def test_words_from_the_start_and_the_end():
    source = cabins_table()
    second = extract(source, {"column": "Cabin", "name": "x", "part": "word", "index": 2})
    assert values(second.table, "x") == [
        None,
        None,
        None,
        None,
        "B59",
        None,
        None,
        None,
        "G73",
        None,
    ]
    assert second.summary.startswith("Took the 2nd word of 'Cabin'")
    last = extract(
        source, {"column": "Name", "name": "x", "part": "word", "index": -1, "case": "lower"}
    )
    assert values(last.table, "x")[:4] == ["harris", "thayer)", "laina", None]
    assert last.summary.startswith("Took the last word of 'Name'")
    third_last = extract(source, {"column": "Cabin", "name": "x", "part": "word", "index": -3})
    assert values(third_last.table, "x")[4] == "B59"
    assert third_last.summary.startswith("Took the 3rd last word of 'Cabin'")


def test_strip_and_case():
    source = table({"code": ["  ab-1 ", "Cd-2", None, "   ", "e"]}, kinds={"code": "categorical"})
    stripped = extract(source, {"column": "code", "name": "x", "length": 2})
    assert values(stripped.table, "x") == ["AB", "CD", None, None, "E"]
    kept = extract(
        source, {"column": "code", "name": "x", "length": 2, "strip": False, "case": "keep"}
    )
    assert values(kept.table, "x") == ["  ", "Cd", None, "  ", "e"]
    lower = extract(
        source, {"column": "code", "name": "x", "part": "last", "length": 3, "case": "lower"}
    )
    assert values(lower.table, "x") == ["b-1", "d-2", None, None, "e"]


# --------------------------------------------------------------------------- #
# Mapping
# --------------------------------------------------------------------------- #


def test_mapping_to_numbers_makes_unmapped_values_missing():
    source = cabins_table()
    result = extract(source, {"column": "Cabin", "name": "Deck", "mapping": DECKS})
    assert values(result.table, "Deck") == [3, None, 4, 5, 2, 8, 1, 7, 6, None]
    assert result.table.kinds["Deck"] == "numeric"
    assert result.table.df["Deck"].dtype == np.float64
    assert result.summary == (
        "Took the first character of 'Cabin' into 'Deck': 8 values, 2 missing; mapped to "
        "numbers 1–8."
    )
    assert result.warnings == ["1 value not in the mapping became missing: 'X'."]

    # Numbers typed as text count as numbers; a value mapped to nothing is missing too.
    texts = extract(
        source,
        {"column": "Cabin", "name": "Deck", "mapping": {"C": "3", "D": "4,5", "E": None, "X": 0}},
    )
    assert values(texts.table, "Deck") == [3, None, 4.5, None, None, None, None, None, None, 0]
    assert texts.table.kinds["Deck"] == "numeric"
    assert texts.warnings == [
        "5 values not in the mapping became missing: 'B', 'T', 'A', 'G' and 'F'."
    ]


def test_mapping_to_categories_and_case_hints():
    source = cabins_table()
    result = extract(
        source,
        {
            "column": "Cabin",
            "name": "level",
            "mapping": {"A": "top", "B": "top", "C": 2, "d": "low"},
        },
    )
    assert values(result.table, "level") == [
        "2",
        None,
        None,
        None,
        "top",
        None,
        "top",
        None,
        None,
        None,
    ]
    assert result.table.kinds["level"] == "categorical"
    assert result.summary.endswith("3 values, 7 missing; mapped to 2 categories.")
    assert result.warnings == [
        "6 values not in the mapping became missing: 'D', 'E', 'T', 'G', 'F' and 1 more.",
        "The parts are in upper case, so 'd' in the mapping never matches. Change the mapping "
        "or the case.",
    ]
    single = extract(source, {"column": "Cabin", "name": "x", "mapping": {"C": 1}})
    assert single.summary.endswith("1 value, 9 missing; mapped to the number 1.")


def test_titanic_cabin_deck():
    """The deck letter of the Titanic cabins as a number: A → 1 … G → 7, T → 8."""
    cabins = ["C85", None, "C123", None, None, "E46", None, "G6", "C103", "D56"]
    cabins += [None, "A6", "C23 C25 C27", None, "B78", "D33", None, "B30", "F E69", "T"]
    cabins += ["D32", "F G73", "B57 B59 B63 B66", None, None]
    source = table({"Cabin": cabins, "Survived": [1.0, 0.0] * 12 + [1.0]})
    ctx = StepContext(target="Survived", task="classification")
    params = {"column": "Cabin", "name": "Deck", "part": "first", "mapping": DECKS}
    result = extract(source, params, ctx)
    assert values(result.table, "Deck") == [
        3,
        None,
        3,
        None,
        None,
        5,
        None,
        7,
        3,
        4,
        None,
        1,
        3,
        None,
        2,
        4,
        None,
        2,
        6,
        8,
        4,
        6,
        2,
        None,
        None,
    ]
    assert result.summary == (
        "Took the first character of 'Cabin' into 'Deck': 16 values, 9 missing; mapped to "
        "numbers 1–8."
    )
    assert not result.warnings


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def test_errors():
    source = cabins_table()
    base = {"column": "Cabin", "name": "Deck"}
    text = error_of(source, {**base, "column": "Fare"})
    assert text.startswith("Text column: 'Fare' holds numbers, not text.")
    assert "Change type" in text
    assert error_of(source, {**base, "column": "Deck"}) == (
        "Text column: there is no column 'Deck' (any more)."
    )
    assert error_of(source, {**base, "column": None}) == "Text column: choose a column."
    assert error_of(source, {**base, "name": " "}) == "New column: enter a text."
    assert error_of(source, {**base, "length": 0}) == "Characters: must be at least 1."
    assert error_of(source, {**base, "part": "last", "length": 1.5}) == (
        "Characters: enter a whole number."
    )
    assert error_of(source, {**base, "part": "middle"}).startswith("Take: choose one of first")
    assert error_of(source, {**base, "part": "pattern"}) == "Pattern: enter a text."
    assert error_of(source, {**base, "part": "pattern", "pattern": "([A-Z]"}).startswith(
        "Pattern: this is not a valid regular expression (missing ), unterminated subpattern"
    )
    assert error_of(source, {**base, "part": "pattern", "pattern": "a" * 201}) == (
        "Pattern: at most 200 characters."
    )
    assert error_of(source, {**base, "part": "word", "index": 0}) == (
        "Word number: use 1 for the first word or -1 for the last one."
    )
    assert error_of(source, {**base, "case": "title"}).startswith("Case: choose one of keep")
    assert error_of(source, {**base, "mapping": ["A"]}) == (
        "Map to new values: expected pairs of old and new values."
    )
    ctx = StepContext(target="Cabin", task="classification")
    assert "is the target" in error_of(source, base, ctx)
    # Hidden settings are not checked: a bad pattern does not matter for the first characters.
    result, cleaned, _ = apply(source, "extract_text", {**base, "pattern": "(", "index": 0})
    assert cleaned["pattern"] == "" and cleaned["index"] == 1


def test_replacing_the_target_warns():
    source = cabins_table()
    ctx = StepContext(target="Cabin", task="classification")
    result = extract(source, {"column": "Cabin", "name": "Cabin"}, ctx)
    assert result.warnings == ["This step replaces the target column 'Cabin'."]


# --------------------------------------------------------------------------- #
# Script and raw-row predictions
# --------------------------------------------------------------------------- #


def titanic_csv(n: int = 90) -> str:
    rng = np.random.default_rng(7)
    decks = list("ABCDEFG")
    titles = ["Mr", "Mrs", "Miss", "Master", "Dr"]
    lines = ["PassengerId,Survived,Pclass,Name,Sex,Age,Ticket,Fare,Cabin"]
    for i in range(1, n + 1):
        cabin = ""
        if rng.random() < 0.35:
            deck = decks[int(rng.integers(0, len(decks)))]
            cabin = " ".join(f"{deck}{rng.integers(1, 130)}" for _ in range(rng.integers(1, 4)))
        if i == 7:
            cabin = "T"
        if i == 11:
            cabin = "F G73"
        title = titles[int(rng.integers(0, len(titles)))]
        name = f'"Family{i}, {title}. Given Name"'
        sex = "female" if title in ("Mrs", "Miss") else "male"
        age = "" if rng.random() < 0.15 else str(int(rng.integers(1, 70)))
        ticket = rng.choice(["A/5 ", "PC ", "STON/O2. ", ""]) + str(rng.integers(1000, 400000))
        fare = f"{rng.lognormal(3, 1):.4f}"
        survived = int(rng.random() < 0.4)
        pclass = int(rng.integers(1, 4))
        lines.append(f"{i},{survived},{pclass},{name},{sex},{age},{ticket},{fare},{cabin}")
    return "\n".join(lines) + "\n"


TITANIC_STEPS = (
    ("extract_text", {"column": "Cabin", "name": "Deck", "mapping": DECKS}),
    (
        "extract_text",
        {
            "column": "Name",
            "name": "Title",
            "part": "pattern",
            "pattern": r",\s*([^.]+)\.",
            "case": "keep",
        },
    ),
    ("extract_text", {"column": "Ticket", "name": "TicketNumber", "part": "word", "index": -1}),
    (
        "extract_text",
        {"column": "Ticket", "name": "TicketPrefix", "part": "pattern", "pattern": r"^([A-Za-z]+)"},
    ),
    (
        "extract_text",
        {
            "column": "Cabin",
            "name": "Rooms",
            "part": "pattern",
            "pattern": r"\d+$",
            "mapping": {"6": 1, "73": 2},
        },
    ),
    ("cast", {"columns": ["TicketNumber"], "to": "numeric"}),
    ("impute", {"columns": ["Deck", "Age", "Rooms"], "strategy": "median"}),
    ("impute", {"columns": ["TicketPrefix"], "strategy": "constant", "fill_value": "none"}),
    ("encode", {"columns": ["Title", "Sex", "TicketPrefix"], "method": "onehot"}),
    ("drop_columns", {"columns": ["PassengerId", "Name", "Ticket", "Cabin"]}),
)


def test_script_and_raw_rows_reproduce_the_titanic_preparation(tmp_path):
    path = tmp_path / "titanic.csv"
    path.write_text(titanic_csv(), encoding="utf-8")
    imported, report = io.read_csv(path)
    assert imported.kinds["Cabin"] in ("categorical", "text")
    settings = {"target": "Survived"}
    result = run(imported, *TITANIC_STEPS, settings=settings)
    assert not result.failed, [r["error"] for r in result.reports if r["error"]]
    summary = result.reports[0]["summary"]
    assert re.fullmatch(
        r"Took the first character of 'Cabin' into 'Deck': \d+ values, \d+ missing; mapped to "
        r"numbers 1–8\.",
        summary,
    ), summary
    assert "Deck" in result.table.columns and "Title=Mr" in result.table.columns

    # The pandas script gives the same table.
    info = {
        "id": 1,
        "name": "titanic",
        "filename": "titanic.csv",
        "delimiter": report.delimiter,
        "decimal": report.decimal,
        "header": report.header,
        "encoding": report.encoding,
        "missing_values": None,
        "columns": imported.column_info(),
    }
    script = pandas_script({1: info}, 1, steps(*TITANIC_STEPS), settings, result)
    assert script.count("def extract_text(") == 1
    done = run_script(tmp_path, script, "--input", "titanic.csv", "--output", "out.csv")
    assert done.returncode == 0, done.stderr
    write_processed_csv(result.table, tmp_path / "expected.csv", include_split=False)
    assert (tmp_path / "out.csv").read_text(encoding="utf-8").splitlines() == (
        (tmp_path / "expected.csv").read_text(encoding="utf-8").splitlines()
    )

    # Raw rows give the values of the training dataset.
    written = write_training_dataset(result.table, result.settings, result.task, tmp_path)
    directory = tmp_path / frozen.FROZEN_DIR
    frozen.freeze(
        directory,
        result=result,
        steps=steps(*TITANIC_STEPS),
        source=imported,
        source_info=info,
        prep={"id": 3, "name": "Titanic"},
    )
    recipe = frozen.read_recipe(directory)
    features = written["features"]
    raw = raw_rows(imported, recipe)
    prepared = frozen.prepare_rows(recipe, raw, directory=directory)
    got = np.array([[row[f] for f in features] for row in prepared.rows])
    np.testing.assert_array_equal(got, training_rows(tmp_path, features))
    script_out = pd.read_csv(tmp_path / "out.csv")
    np.testing.assert_array_equal(got, script_out[features].to_numpy(dtype=float))

    # A new passenger in cabin D32 is on deck 4; an unknown deck gets the training median.
    deck_median = recipe["steps"][6]["fitted"]["values"]["Deck"]
    row = {**raw[0], "Cabin": "D32"}
    assert frozen.prepare_rows(recipe, row, directory=directory).rows[0]["Deck"] == 4.0
    for cabin in ("X9", None, "  "):
        alone = frozen.prepare_rows(recipe, {**raw[0], "Cabin": cabin}, directory=directory)
        assert alone.rows[0]["Deck"] == deck_median
