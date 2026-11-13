"""Text functions of the Formula step (FormulaCompiler in brickdnn.data.steps.columns): what
they compute, that unsafe formulas are refused while compiling, and that the step, the pandas
script and predictions from raw rows give the same values."""

from __future__ import annotations

import ast
import re

import numpy as np
import pandas as pd
import pytest

from brickdnn.data import frozen, io
from brickdnn.data.export import pandas_script, write_processed_csv, write_training_dataset
from brickdnn.data.steps import catalog
from brickdnn.data.steps.columns import (
    _ALLOWED_TEXT,
    FORMULA_FUNCTIONS,
    FORMULA_HELPERS,
    MAX_FORMULA_LENGTH,
    MAX_MAP_ENTRIES,
    FormulaCompiler,
    evaluate_formula,
    formula_code,
)
from brickdnn.data.table import DataError

from .data_helpers import apply, check_code, run, steps, table, values
from .test_data_export import run_script
from .test_data_frozen import raw_rows, training_rows

DECKS = '{"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}'


@pytest.fixture
def passengers():
    return table(
        {
            "Cabin": ["C85", None, "d32", "T", "X1", "B57 B59"],
            "Name": [
                "Braund, Mr. Owen Harris",
                "Cumings, Mrs. John Bradley",
                None,
                "Heikkinen, Miss. Laina",
                "Nobody",
                "Allen, Mr. William Henry",
            ],
            "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282", "113803", None, "373450"],
            "SibSp": [1.0, 1.0, 0.0, np.nan, 0.0, 2.5],
            "Parch": [0.0, 2.0, 0.0, 1.0, 0.0, 0.0],
            "Embarked": ["S", "C", None, "S", "Q", "S"],
        },
        kinds={"Cabin": "categorical", "Name": "text", "Ticket": "categorical"},
    )


def formula(source, expression, name="out"):
    """Run the step and check that its pandas code gives the same table."""
    return check_code(source, "formula", {"name": name, "expression": expression}, re=re)


def compile_error(expression: str, kinds: dict[str, str] | None = None) -> str:
    kinds = kinds or {"Name": "text", "Cabin": "categorical", "x": "numeric"}
    with pytest.raises(DataError) as info:
        FormulaCompiler(kinds).compile(expression)
    return str(info.value)


# --------------------------------------------------------------------------- #
# The functions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("expression", "kind", "expected"),
    [
        ("left(`Cabin`, 1)", "categorical", ["C", None, "d", "T", "X", "B"]),
        ("right(Ticket, 3)", "categorical", ["171", "599", "282", "803", None, "450"]),
        ("left(`Cabin`, 0)", "categorical", [None] * 6),  # an empty text is missing
        ("mid(Name, 1, 3)", "categorical", ["Bra", "Cum", None, "Hei", "Nob", "All"]),
        ("mid(Ticket, 4, 2)", "categorical", [" 2", "17", "N/", "80", None, "45"]),
        ("upper(`Cabin`)", "categorical", ["C85", None, "D32", "T", "X1", "B57 B59"]),
        ("lower(Embarked)", "categorical", ["s", "c", None, "s", "q", "s"]),
        ("strip(concat(' ', Embarked, ' '))", "categorical", ["S", "C", None, "S", "Q", "S"]),
        ("len(`Cabin`)", "numeric", [3.0, None, 3.0, 1.0, 2.0, 7.0]),
        ('contains(Name, "Mr.")', "boolean", [1.0, 0.0, None, 0.0, 0.0, 1.0]),
        ('contains(Name, "mr", ignore_case=1)', "boolean", [1.0, 1.0, None, 0.0, 0.0, 1.0]),
        ('contains(Name, "mr")', "boolean", [0.0, 0.0, None, 0.0, 0.0, 0.0]),
        ('contains(Ticket, ".")', "boolean", [0.0, 0.0, 1.0, 0.0, None, 0.0]),  # not a pattern
        ('startswith(Ticket, "PC")', "boolean", [0.0, 1.0, 0.0, 0.0, None, 0.0]),
        ('endswith(`Cabin`, "D32", 1)', "boolean", [0.0, None, 1.0, 0.0, 0.0, 0.0]),
        (
            'replace(Ticket, ".", "")',
            "categorical",
            ["A/5 21171", "PC 17599", "STON/O2 3101282", "113803", None, "373450"],
        ),
        ('replace(Embarked, "S", "")', "categorical", [None, "C", None, None, "Q", None]),
        (
            'extract(Name, ",\\s*([^.]+)\\.")',
            "categorical",
            ["Mr", "Mrs", None, "Miss", None, "Mr"],
        ),
        ('extract(Ticket, "\\d+")', "categorical", ["5", "17599", "2", "113803", None, "373450"]),
        (
            'split_part(Ticket, " ", 1)',
            "categorical",
            ["A/5", "PC", "STON/O2.", "113803", None, "373450"],
        ),
        (
            'split_part(Ticket, " ", -1)',
            "categorical",
            ["21171", "17599", "3101282", "113803", None, "373450"],
        ),
        (
            'split_part(Ticket, " ", 2)',
            "categorical",
            ["21171", "17599", "3101282", None, None, None],
        ),
        ('split_part(`Cabin`, " ", -3)', "categorical", [None] * 6),
        ('concat(Embarked, "-", SibSp)', "categorical", ["S-1", "C-1", None, None, "Q-0", "S-2.5"]),
        ('concat("a", "b")', "categorical", ["ab"] * 6),
        (
            'to_number(split_part(Ticket, " ", -1))',
            "numeric",
            [21171.0, 17599.0, 3101282.0, 113803.0, None, 373450.0],
        ),
        ("to_number(`Cabin`)", "numeric", [None] * 6),
        ('to_number("2.5") * 2', "numeric", [5.0] * 6),
        ("to_text(SibSp)", "categorical", ["1", "1", "0", None, "0", "2.5"]),
        (
            "coalesce(`Cabin`, Embarked, 'none')",
            "categorical",
            ["C85", "C", "d32", "T", "X1", "B57 B59"],
        ),
        ("coalesce(SibSp, Parch, 0)", "numeric", [1.0, 1.0, 0.0, 1.0, 0.0, 2.5]),
        ('Embarked == "S"', "boolean", [1.0, 0.0, None, 1.0, 0.0, 1.0]),
        ("upper(`Cabin`) != 'T'", "boolean", [1.0, None, 1.0, 0.0, 1.0, 1.0]),
        (
            'where(contains(Name, "Mrs"), "married", Embarked)',
            "categorical",
            ["S", "married", None, "S", "Q", "S"],
        ),
        ("SibSp + Parch + 1", "numeric", [2.0, 4.0, 1.0, None, 1.0, 3.5]),
    ],
)
def test_text_functions(passengers, expression, kind, expected):
    result = formula(passengers, expression)
    assert result.table.kinds["out"] == kind
    assert values(result.table, "out") == expected


@pytest.mark.parametrize(
    ("expression", "kind", "expected"),
    [
        # All values numbers: a number column; unmapped values and a missing cabin: missing.
        (f"map(upper(left(`Cabin`, 1)), {DECKS})", "numeric", [3.0, None, 4.0, 8.0, None, 2.0]),
        (f"map(upper(left(`Cabin`, 1)), {DECKS}, 0)", "numeric", [3.0, None, 4.0, 8.0, 0.0, 2.0]),
        (
            'map(Embarked, {"S": "Southampton", "C": "Cherbourg"}, "other")',
            "categorical",
            ["Southampton", "Cherbourg", None, "Southampton", "other", "Southampton"],
        ),
        ('map(Embarked, {"S": 1, "C": "two"})', "categorical", ["1", "two", None, "1", None, "1"]),
        # Keys of a number column may be numbers or numbers in quotes.
        (
            'map(SibSp, {0: "none", "1": "one"})',
            "categorical",
            ["one", "one", "none", None, "none", None],
        ),
        ("map(to_text(SibSp), {'2.5': -1, '1': 1})", "numeric", [1.0, 1.0, None, None, None, -1.0]),
    ],
)
def test_map(passengers, expression, kind, expected):
    result = formula(passengers, expression)
    assert result.table.kinds["out"] == kind
    assert values(result.table, "out") == expected


def test_summary_counts_values_without_a_result(passengers):
    result, _, _ = apply(
        passengers, "formula", {"name": "Deck", "expression": f"map(left(`Cabin`, 1), {DECKS})"}
    )
    assert result.summary == (
        "Added 'Deck' (numbers): 1 value missing because an input is missing; 2 values could "
        "not be computed (e.g. no match, no such part, a value not in the mapping or an empty "
        "text)."
    )


def test_constants_are_computed_while_compiling():
    compiler = FormulaCompiler({"x": "numeric"})
    assert compiler.compile("concat('a', to_text(2.5), 'b')").value == "a2.5b"
    assert compiler.compile("to_number(concat('1', '5')) + 1").value == 16.0
    assert compiler.compile("coalesce(to_number('x'), 7)").value == 7.0
    assert compiler.compile("'a' == 'a'").value == 1.0


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("left(Name, x)", "left(x, n) needs n as a whole number from 0 to 10,000 written"),
        ("left(Name, -1)", "left(x, n) needs n as a whole number"),
        ("right(Name, 1.5)", "right(x, n) needs n"),
        ("mid(Name, 0, 2)", "mid(x, start, length) needs start as a whole number from 1"),
        ('left(Name, "2")', "left(x, n) needs n"),
        ("contains(Name, Cabin)", 'contains(x, "text") needs the text to look for as a text'),
        ('contains(Name, "")', "the text to look for is empty"),
        ('contains(Name, "a", ignore_case=2)', "ignore_case is 1"),
        ('contains(Name, "a", ignore_case=x)', "ignore_case is 1"),
        ('contains(Name, "a", flags=1)', "contains(…) takes plain arguments only"),
        ('replace(Name, "a", Cabin)', 'replace(x, "old", "new") needs the new text'),
        ('replace(Name, "", "b")', "the text to replace is empty"),
        (f'replace(Name, "{"a" * 201}", "b")', "at most 200 characters"),
        ("extract(Name, Cabin)", 'extract(x, "pattern") needs the pattern'),
        ('extract(Name, "(")', "not a valid regular expression"),
        ('extract(Name, "(a+)+$")', "nested repetition"),
        ('extract(Name, "(\\w+\\s?)*$")', "nested repetition"),
        (f'extract(Name, "{"a" * 201}")', "at most 200 characters"),
        ('split_part(Name, " ", 0)', "k is 1 for the first part"),
        ('split_part(Name, "", 1)', "the separator is empty"),
        ('split_part(Name, " ", x)', "needs k as a whole number"),
        ("map(Name, Cabin)", "needs a dictionary of values written in the formula"),
        ('map(Name, {"a": x})', "keys and values are texts in quotes or numbers"),
        ("map(Name, {Cabin: 1})", "keys and values are texts in quotes or numbers"),
        ('map(Name, {"a": 1 + 1})', "keys and values are texts in quotes or numbers"),
        ('map(Name, {"a": 1}, Cabin)', "the default is a text in quotes or a number"),
        ('map(Name, {"a": 1, "a": 2})', "the key 'a' is listed twice"),
        ('map(x, {"a": 1})', "the key 'a' is a text, but x holds numbers"),
        ("map(Name, {})", "the dictionary is empty"),
        ('map(Name, {**{"a": 1}})', "keys and values are texts in quotes or numbers"),
        ("concat(Name)", "concat(…) takes 2 to 50 argument(s)"),
        ("coalesce(Name, 1)", "coalesce(a, b, …) needs values of the same type"),
        ("where(x > 1, Name, 1)", "needs a and b of the same type"),
        ("upper(Name) + 1", "text (formula_upper(df['Name'])) cannot be used with '+'"),
        ("Name < 'B'", "only be compared with == and !="),
        ("Name == 1", "cannot compare text with numbers"),
        ("upper", "upper is a function"),
    ],
)
def test_errors_name_the_function_and_what_is_allowed(expression, message):
    assert message in compile_error(expression)


def test_extract_from_text_refuses_nested_repetition(passengers):
    with pytest.raises(DataError, match="nested repetition"):
        apply(
            passengers,
            "extract_text",
            {"column": "Name", "name": "t", "part": "pattern", "pattern": "^(\\w+\\s?)*$"},
        )


def test_map_takes_at_most_500_entries():
    entries = ", ".join(f'"k{i}": {i}' for i in range(MAX_MAP_ENTRIES))
    expression = f"map(Name, {{{entries}}})"
    assert len(expression) > MAX_FORMULA_LENGTH  # dictionaries do not count
    expr = FormulaCompiler({"Name": "text"}).compile(expression)
    df = pd.DataFrame({"Name": ["k7", "k499", "nope", None]})
    got = evaluate_formula(formula_code(expr), df)
    assert got.tolist()[:2] == [7.0, 499.0] and np.isnan(got.tolist()[2:]).all()
    too_many = f'map(Name, {{{entries}, "last": 1}})'
    assert "at most 500 entries" in compile_error(too_many)


# --------------------------------------------------------------------------- #
# Safety: refused while compiling
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("__import__('os')", "__import__(…) is not allowed"),
        ("__import__('os').system('echo hacked')", "Attribute access"),
        ("`Name`.__class__", "Attribute access"),
        ("`Name`.__class__.__bases__[0].__subclasses__()", "not allowed"),
        ("upper(Name).__class__", "Attribute access"),
        ("open('/etc/passwd')", "open(…) is not allowed"),
        ("open('/etc/passwd').read()", "Attribute access"),
        ("eval('1')", "eval(…) is not allowed"),
        ("exec('x = 1')", "exec(…) is not allowed"),
        ("getattr(Name, 'upper')", "getattr(…) is not allowed"),
        ("globals()", "globals(…) is not allowed"),
        ("df", "no column 'df'"),
        ("np", "no column 'np'"),
        ("formula_upper(Name)", "formula_upper(…) is not allowed"),
        ("__builtins__", "not allowed"),
        ("(lambda: 1)()", "Functions (lambda) cannot be defined"),
        ("map(Name, {'a': 1}, (lambda: 1)())", "default is a text in quotes or a number"),
        ("upper(lambda: 1)", "Functions (lambda) cannot be defined"),
        ("[c for c in Name]", "Comprehensions"),
        ("concat(Name, (c for c in Name))", "Comprehensions"),
        ("map(Name, {c: 1 for c in 'ab'})", "needs a dictionary of values"),
        ("{1: 2}", "can only be the second argument of map"),
        ("{1, 2}", "'Set' is not allowed"),
        ("[1, 2]", "'List' is not allowed"),
        ("(1, 2)", "'Tuple' is not allowed"),
        ("Name[0]", "Indexing"),
        ("f'{Name}'", "f-strings are not allowed"),
        ("'{}'.format(Name)", "Attribute access"),
        ("'%s' % Name", "text ('%s') cannot be used with '%'"),
        ("concat(*Name)", "takes plain arguments only"),
        ("concat(**Name)", "takes plain arguments only"),
        ("upper(x=Name)", "takes plain arguments only"),
        ("(y := 1)", "Assignments (:=)"),
        ("b'abc'", "is not allowed"),
        ("Name if x else Cabin", "where(condition, a, b)"),
        ("~x", "'~' is not allowed"),
        ("x @ x", "'BinOp' is not allowed"),
        ("x is x", "'in' and 'is' are not allowed"),
        ("'a' in Name", "'in' and 'is' are not allowed"),
        ("...", "is not allowed"),
        ("1j", "is not allowed"),
        ("map(Name, Cabin)", "needs a dictionary of values written in the formula"),
        ("extract(Name, concat('(', 'a)'))", 'extract(x, "pattern") needs the pattern'),
        ("extract(Name, upper('a'))", 'extract(x, "pattern") needs the pattern'),
        ('split_part(Name, upper(" "), 1)', "needs the separator as a text in quotes"),
        ("left(Name, len(Name))", "needs n as a whole number"),
    ],
)
def test_injections_are_refused_while_compiling(expression, message):
    assert message in compile_error(expression)


def test_huge_dictionaries_and_formulas_are_refused():
    entries = ", ".join(f'"k{i}": {i}' for i in range(MAX_MAP_ENTRIES + 1))
    assert "at most 500 entries" in compile_error(f"map(Name, {{{entries}}})")
    huge = "{" + ", ".join(f'"k{i}": "{"v" * 40}"' for i in range(500)) + "}"
    assert "up to 20,000 with them" in compile_error(f"map(Name, {huge})")
    text = "concat(Name, " + ", ".join(['"' + "a" * 150 + '"'] * 14) + ")"
    assert len(text) > MAX_FORMULA_LENGTH
    assert "at most 2,000 characters (map(…) dictionaries not counted" in compile_error(text)
    # Column names in backticks count as written.
    name = "n" * 1995
    assert "at most 2,000" in compile_error(f"upper(`{name}`)", {name: "text"})
    assert FormulaCompiler({name: "text"}).compile(f"`{name}`").kind == "str"


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        pytest.param("upper(" * 120 + "Name" + ")" * 120, "too long", id="nested-calls"),
        pytest.param("(" * 300 + "x" + ")" * 300, "nested too deeply", id="parentheses"),
        pytest.param("-" * 120 + "x", "nested too deeply", id="signs"),
        pytest.param("not " * 120 + "x", "nested too deeply", id="not"),
        pytest.param("not " * 40 + "x", "too complex", id="generated-code"),
        pytest.param("concat(" + ", ".join(["Name"] * 51) + ")", "2 to 50", id="arguments"),
        pytest.param(" + ".join(["len(Name)"] * 80), "too long", id="nodes"),
    ],
)
def test_deep_or_large_formulas_are_refused(expression, message):
    assert message in compile_error(expression)


def test_generated_code_only_calls_whitelisted_names():
    kinds = {"Name": "text", "Cabin": "categorical", "x": "numeric"}
    expression = (
        f'where(contains(Name, "__import__"), map(upper(left(Cabin, 1)), {DECKS}), '
        'to_number(extract(Name, "(\\d+)\'")) + len(concat(Name, "\'), exec(\'1"))) '
        "+ len(replace(split_part(Name, ' ', -1), '\"', '`x`'))"
    )
    code = formula_code(FormulaCompiler(kinds).compile(expression))
    names = {n.id for n in ast.walk(ast.parse(code, mode="eval")) if isinstance(n, ast.Name)}
    assert names <= {"df", "np", "pd", *FORMULA_HELPERS}, names
    attributes = {
        n.attr for n in ast.walk(ast.parse(code, mode="eval")) if isinstance(n, ast.Attribute)
    }
    assert not any(a.startswith("_") for a in attributes), attributes
    # Literal texts stay literals: the payloads are data, not code.
    df = pd.DataFrame({"Name": ["a __import__ b", "x 7') y"], "Cabin": ["C1", "B2"], "x": [1, 2]})
    assert evaluate_formula(code, df).tolist() == [3.0 + 1, 7.0 + 18 + 1]


def test_every_function_is_listed_in_the_help():
    spec = next(s for s in catalog()["steps"] if s["type"] == "formula")
    help_text = next(p["help"] for p in spec["params"] if p["name"] == "expression")
    for name in FORMULA_FUNCTIONS:
        assert name in _ALLOWED_TEXT and name in help_text, name
    assert (
        spec["example"]["name"] == "Deck"
        and "map(upper(left(`Cabin`, 1))" in (spec["example"]["expression"])
    )
    cabins = table({"Cabin": ["C85", None, "t"]}, kinds={"Cabin": "categorical"})
    result = formula(cabins, spec["example"]["expression"], name="Deck")
    assert values(result.table, "Deck") == [3.0, None, 8.0]


# --------------------------------------------------------------------------- #
# Script and raw-row predictions
# --------------------------------------------------------------------------- #


def titanic_csv(n: int = 90) -> str:
    rng = np.random.default_rng(11)
    decks = list("ABCDEFG")
    titles = ["Mr", "Mrs", "Miss", "Master", "Dr"]
    lines = ["PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked"]
    for i in range(1, n + 1):
        cabin = ""
        if rng.random() < 0.4:
            deck = decks[int(rng.integers(0, len(decks)))]
            cabin = " ".join(f"{deck}{rng.integers(1, 130)}" for _ in range(rng.integers(1, 4)))
            cabin = cabin.lower() if i % 9 == 0 else cabin
        if i == 7:
            cabin = "T"
        if i == 13:
            cabin = "X12"  # no such deck
        title = titles[int(rng.integers(0, len(titles)))]
        name = f'"Family{i}, {title}. Given Name"' if i != 5 else '"Nameless"'
        sex = "female" if title in ("Mrs", "Miss") else "male"
        age = "" if rng.random() < 0.15 else str(int(rng.integers(1, 70)))
        sibsp, parch = int(rng.integers(0, 4)), int(rng.integers(0, 3))
        ticket = rng.choice(["A/5 ", "PC ", "STON/O2. ", ""]) + str(rng.integers(1000, 400000))
        fare = f"{rng.lognormal(3, 1):.4f}"
        embarked = rng.choice(["S", "C", "Q", ""])
        survived = int(rng.random() < 0.4)
        pclass = int(rng.integers(1, 4))
        lines.append(
            f"{i},{survived},{pclass},{name},{sex},{age},{sibsp},{parch},{ticket},{fare},"
            f"{cabin},{embarked}"
        )
    return "\n".join(lines) + "\n"


TITLES = '{"Mr": 1, "Mrs": 2, "Miss": 3, "Master": 4}'
TITANIC_STEPS = (
    ("formula", {"name": "Deck", "expression": f"map(upper(left(`Cabin`, 1)), {DECKS})"}),
    ("formula", {"name": "Title", "expression": 'extract(Name, ",\\s*([^.]+)\\.")'}),
    ("formula", {"name": "TitleCode", "expression": f"map(Title, {TITLES}, 0)"}),
    ("formula", {"name": "FamilySize", "expression": "SibSp + Parch + 1"}),
    ("formula", {"name": "IsAlone", "expression": "FamilySize == 1"}),
    ("formula", {"name": "Married", "expression": 'contains(Name, "mrs.", ignore_case=1)'}),
    (
        "formula",
        {"name": "TicketNumber", "expression": 'to_number(split_part(Ticket, " ", -1))'},
    ),
    (
        "formula",
        {
            "name": "Port",
            "expression": 'where(Embarked == "S", "Southampton", coalesce(Embarked, "unknown"))',
        },
    ),
    ("formula", {"name": "CabinCount", "expression": 'len(replace(`Cabin`, " ", "")) / 3'}),
    ("impute", {"columns": ["Deck", "TitleCode", "Age", "CabinCount"], "strategy": "median"}),
    ("encode", {"columns": ["Title", "Sex", "Port"], "method": "onehot"}),
    ("drop_columns", {"columns": ["PassengerId", "Name", "Ticket", "Cabin", "Embarked"]}),
)


def test_script_and_raw_rows_reproduce_the_titanic_formulas(tmp_path):
    path = tmp_path / "titanic.csv"
    path.write_text(titanic_csv(), encoding="utf-8")
    imported, report = io.read_csv(path)
    settings = {"target": "Survived"}
    result = run(imported, *TITANIC_STEPS, settings=settings)
    assert not result.failed, [r["error"] for r in result.reports if r["error"]]
    assert result.reports[0]["summary"].startswith("Added 'Deck' (numbers): ")
    columns = result.table.columns
    assert {"Deck", "TitleCode", "FamilySize", "IsAlone", "Married", "TicketNumber"} <= set(columns)
    assert "Title=Mr" in columns and "Port=Southampton" in columns

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
    assert script.count("def formula_map(") == 1 and "def formula_extract(" in script
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
    script_out = pd.read_csv(tmp_path / "out.csv", float_precision="round_trip")
    np.testing.assert_array_equal(got, script_out[features].to_numpy(dtype=float))

    # A new passenger: cabin d32 is on deck 4, "Mrs." is married, a family of 3.
    row = {**raw[0], "Cabin": "d32", "Name": "Doe, Mrs. Jane", "SibSp": 1, "Parch": 1}
    alone = frozen.prepare_rows(recipe, row, directory=directory).rows[0]
    assert alone["Deck"] == 4.0 and alone["Married"] == 1.0 and alone["TitleCode"] == 2.0
    assert alone["FamilySize"] == 3.0 and alone["IsAlone"] == 0.0
    deck_median = recipe["steps"][9]["fitted"]["values"]["Deck"]
    for cabin in ("X9", None):
        other = frozen.prepare_rows(recipe, {**raw[0], "Cabin": cabin}, directory=directory)
        assert other.rows[0]["Deck"] == deck_median
