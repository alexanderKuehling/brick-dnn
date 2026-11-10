"""The torch-free part of brickdnn.training: settings, datasets, compatibility, run files."""

from __future__ import annotations

import io
import json
import os
import stat
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

from brickdnn import EXAMPLES
from brickdnn.training import (
    BUILTIN_DATASETS,
    ConfigError,
    DatasetError,
    DatasetInfo,
    TrainingConfig,
    TrainingPlan,
    builtin_info,
    check_compatibility,
    extract_image_zip,
    inspect_csv,
    inspect_image_folder,
    random_info,
    runner,
    torch_installed,
    torchvision_installed,
)
from brickdnn.training import state as st
from brickdnn.training.datasets import find_image_root, list_image_files, parse_number, read_csv
from brickdnn.training.tuning import TuningConfig, describe_params, trial_config

from .helpers import chain

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


def test_config_defaults():
    config = TrainingConfig.from_dict(None)
    assert config == TrainingConfig()
    assert config.to_dict() == {
        "epochs": 10,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "weight_decay": 0.0,
        "val_split": 0.2,
        "test_split": 0.0,
        "device": "auto",
        "max_samples": None,
        "seed": None,
        "patience": None,
        "restore_best": True,
        "loss": "cross_entropy",
    }
    assert TrainingConfig.from_dict(config.to_dict()) == config
    assert not config.has_test
    assert TrainingConfig.from_dict({"test_split": "0.15"}).has_test


def test_config_parses_strings_and_normalises_choices():
    config = TrainingConfig.from_dict(
        {
            "epochs": "5",
            "batch_size": 32.0,
            "learning_rate": "1e-2",
            "optimizer": " SGD ",
            "device": "CPU",
            "max_samples": "1e3",
            "seed": 0,
            "val_split": "",
        }
    )
    assert config.epochs == 5
    assert config.batch_size == 32
    assert config.learning_rate == 0.01
    assert config.optimizer == "sgd"
    assert config.device == "cpu"
    assert config.max_samples == 1000
    assert config.seed == 0
    assert config.val_split == 0.2


def test_config_early_stopping_and_best_weights():
    config = TrainingConfig.from_dict({"patience": "5", "restore_best": False})
    assert config.patience == 5 and config.restore_best is False
    assert TrainingConfig.from_dict({"patience": "", "restore_best": "true"}).patience is None
    assert TrainingConfig.from_dict({"restore_best": "False"}).restore_best is False
    assert TrainingConfig.from_dict({"restore_best": None}).restore_best is True


@pytest.mark.parametrize(
    "data,message",
    [
        ({"epochs": 0}, "Epochs must be between 1 and 1,000."),
        ({"epochs": 1.5}, "Epochs must be a whole number."),
        ({"epochs": True}, "Epochs must be a whole number."),
        ({"epochs": "many"}, "Epochs must be a whole number."),
        ({"epochs": [3]}, "Epochs must be a whole number."),
        ({"batch_size": 10_000}, "Batch size must be between 1 and 8,192."),
        ({"learning_rate": 0}, "Learning rate must be greater than 0 and at most 10."),
        ({"learning_rate": "nan"}, "Learning rate must be greater than 0"),
        ({"weight_decay": -0.1}, "Weight decay must be at least 0 and at most 1."),
        ({"val_split": 0.9}, "Validation split must be at least 0.05 and at most 0.5."),
        ({"test_split": 0.6}, "Test split must be at least 0 and at most 0.5."),
        ({"test_split": -0.1}, "Test split must be at least 0 and at most 0.5."),
        (
            {"val_split": 0.5, "test_split": 0.4},
            "Validation and test split together can be at most 80%.",
        ),
        ({"optimizer": "rmsprop"}, "Optimizer must be one of: adam, adamw, sgd."),
        ({"device": "tpu"}, "Device must be one of: auto, cpu, cuda, mps."),
        ({"max_samples": 5}, "Max. samples must be between 10 and"),
        ({"seed": -1}, "Seed must be between 0 and"),
        ({"patience": 0}, "Patience must be between 1 and 1,000."),
        ({"patience": 2.5}, "Patience must be a whole number."),
        ({"restore_best": 1}, "Restore best weights must be true or false."),
        ({"restore_best": "yes"}, "Restore best weights must be true or false."),
    ],
)
def test_config_errors(data, message):
    with pytest.raises(ConfigError) as excinfo:
        TrainingConfig.from_dict(data)
    assert str(excinfo.value).startswith(message)


def test_config_must_be_an_object():
    with pytest.raises(ConfigError, match="must be an object"):
        TrainingConfig.from_dict([1, 2])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Hyperparameter search settings
# --------------------------------------------------------------------------- #


def test_tuning_defaults():
    tuning = TuningConfig.from_dict(None)
    assert tuning == TuningConfig()
    assert tuning.to_dict() == {
        "method": "random",
        "trials": 10,
        "space": {"learning_rate": {"min": 1e-4, "max": 1e-2}, "batch_size": [32, 64, 128]},
    }
    assert TuningConfig.from_dict(tuning.to_dict()) == tuning
    assert tuning.describe() == "random search, 10 trials"


def test_random_search_is_reproducible_and_within_the_space():
    tuning = TuningConfig.from_dict(
        {
            "trials": 25,
            "space": {
                "learning_rate": {"min": "1e-5", "max": 0.1},
                "optimizer": ["Adam", "sgd", "adam"],
                "weight_decay": [0, 1e-4],
            },
        }
    )
    assert tuning.space["optimizer"] == ("adam", "sgd")  # normalised, without duplicates
    trials = tuning.sample(seed=7)
    assert trials == tuning.sample(seed=7)
    assert trials != tuning.sample(seed=8)
    assert len(trials) == 25
    assert len({tuple(t.values()) for t in trials}) == 25  # no trial twice
    for params in trials:
        assert list(params) == ["learning_rate", "optimizer", "weight_decay"]
        assert 1e-5 <= params["learning_rate"] <= 0.1
        assert params["learning_rate"] == float(f"{params['learning_rate']:.3g}")  # rounded
        assert params["optimizer"] in ("adam", "sgd")
        assert params["weight_decay"] in (0, 1e-4)
    rates = sorted(t["learning_rate"] for t in trials)
    assert rates[0] < 1e-3 < rates[-1]  # log-uniform: small rates are drawn as well


def test_random_search_over_lists_tries_each_combination_at_most_once():
    tuning = TuningConfig.from_dict(
        {"trials": 50, "space": {"batch_size": [16, 32], "optimizer": ["adam", "sgd"]}}
    )
    assert tuning.trials == 4
    trials = tuning.sample(seed=0)
    assert sorted((t["batch_size"], t["optimizer"]) for t in trials) == [
        (16, "adam"),
        (16, "sgd"),
        (32, "adam"),
        (32, "sgd"),
    ]


def test_grid_search_tries_every_combination_in_order():
    tuning = TuningConfig.from_dict(
        {
            "method": "grid",
            "trials": 2,  # ignored: a grid always tries every combination
            "space": {"learning_rate": [0.01, 0.001], "batch_size": [32, 64, 128]},
        }
    )
    assert tuning.trials == 6
    assert tuning.describe() == "grid search, 6 trials"
    assert tuning.sample(seed=1) == [
        {"learning_rate": 0.01, "batch_size": 32},
        {"learning_rate": 0.01, "batch_size": 64},
        {"learning_rate": 0.01, "batch_size": 128},
        {"learning_rate": 0.001, "batch_size": 32},
        {"learning_rate": 0.001, "batch_size": 64},
        {"learning_rate": 0.001, "batch_size": 128},
    ]


def test_trial_config_and_description():
    base = TrainingConfig.from_dict({"epochs": 3, "learning_rate": 0.1})
    params = {"learning_rate": 0.00312, "batch_size": 16, "optimizer": "adamw"}
    config = trial_config(base, params)
    assert (config.epochs, config.learning_rate, config.batch_size) == (3, 0.00312, 16)
    assert config.optimizer == "adamw" and base.learning_rate == 0.1
    assert describe_params(params) == "learning rate 0.00312, batch size 16, AdamW"
    assert describe_params({"weight_decay": 0.0001}) == "weight decay 0.0001"


@pytest.mark.parametrize(
    "data,message",
    [
        ("fast", "The hyperparameter search settings must be an object."),
        ({"method": "bayes"}, "the method must be one of: random, grid."),
        ({"trials": 0}, "the number of trials must be at least 1 and at most 100."),
        ({"trials": 2.5}, "the number of trials must be whole numbers."),
        ({"space": []}, "the search space must be an object."),
        ({"space": {}}, "choose at least one setting to tune."),
        ({"space": {"learning_rate": None}}, "choose at least one setting to tune."),
        ({"space": {"momentum": [0.9]}}, "'momentum' cannot be tuned."),
        ({"space": {"learning_rate": [0.1, 0]}}, "learning rates must be greater than 0"),
        ({"space": {"learning_rate": {"min": 0.1}}}, 'needs "min" and "max".'),
        ({"space": {"learning_rate": {"min": 0.1, "max": 0.01}}}, "must not be larger"),
        ({"space": {"weight_decay": {"min": 0, "max": 0.1}}}, "must be greater than 0"),
        ({"space": {"batch_size": {"min": 8, "max": 64}}}, "give a list of values"),
        ({"space": {"batch_size": [32, 0]}}, "batch sizes must be at least 1"),
        ({"space": {"batch_size": [32.5]}}, "batch sizes must be whole numbers."),
        ({"space": {"batch_size": []}}, "give at least one batch size."),
        ({"space": {"batch_size": list(range(1, 30))}}, "at most 20 values per setting."),
        ({"space": {"optimizer": ["rmsprop"]}}, "the optimizer must be one of"),
        (
            {"method": "grid", "space": {"learning_rate": {"min": 1e-4, "max": 1e-2}}},
            "Grid search needs a list of values for the learning rate, not a range.",
        ),
        (
            {
                "method": "grid",
                "space": {
                    "learning_rate": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
                    "batch_size": [8, 16, 32, 64, 128],
                    "weight_decay": [0, 1e-5, 1e-4, 1e-3, 1e-2],
                },
            },
            "The grid has 125 combinations, at most 100 are possible.",
        ),
    ],
)
def test_tuning_errors(data, message):
    with pytest.raises(ConfigError) as excinfo:
        TuningConfig.from_dict(data)
    assert message in str(excinfo.value)


def test_optional_dependency_checks_return_booleans():
    assert isinstance(torch_installed(), bool)
    assert isinstance(torchvision_installed(), bool)


# --------------------------------------------------------------------------- #
# CSV files
# --------------------------------------------------------------------------- #


def write(tmp_path, name, text, encoding="utf-8"):
    path = tmp_path / name
    path.write_text(text, encoding=encoding)
    return path


@pytest.mark.parametrize(
    "cell,decimal_comma,expected",
    [
        ("1.5", False, 1.5),
        (" -2 ", False, -2.0),
        ("1e3", False, 1000.0),
        ("1,5", False, None),
        ("1,5", True, 1.5),
        ("1.234,5", True, 1234.5),
        ("", False, None),
        ("abc", True, None),
        ("nan", False, None),
        ("inf", False, None),
    ],
)
def test_parse_number(cell, decimal_comma, expected):
    assert parse_number(cell, decimal_comma) == expected


def test_csv_with_header_and_text_labels(tmp_path):
    rows = ["5.1,3.5,setosa", "7.0,3.2,versicolor", "6.3,3.3,virginica", "4.9,3.0,setosa"]
    path = write(tmp_path, "iris.csv", "sepal_length,sepal_width,species\n" + "\n".join(rows))
    header, data, delimiter = read_csv(path)
    assert header == ["sepal_length", "sepal_width", "species"]
    assert len(data) == 4 and delimiter == ","
    info = inspect_csv(path)
    assert info.kind == "csv" and info.name == "iris"
    assert info.has_header is True
    assert info.num_samples == 4 and info.num_features == 2
    assert info.columns == ["sepal_length", "sepal_width", "species"]
    assert info.target_kind == "text"
    assert info.class_names == ["setosa", "versicolor", "virginica"]
    assert info.num_classes == 3


def test_csv_without_header_and_integer_target(tmp_path):
    path = write(tmp_path, "digits.csv", "0.1,0.2,0\n0.3,0.4,2\n0.5,0.6,1\n0.7,0.8,2\n\n")
    info = inspect_csv(path, name="Digits")
    assert info.name == "Digits"
    assert info.has_header is False
    assert info.columns == ["column_1", "column_2", "target"]
    assert info.target_kind == "integer"
    assert info.class_names == ["0", "1", "2"]
    assert info.target_range == [0.0, 2.0]


def test_csv_semicolon_and_decimal_comma(tmp_path):
    text = "\ufeffa;b;price\n1,5;2;10,25\n2,5;3;11,5\n3,5;4;12,75\n4,5;5;14\n"
    path = write(tmp_path, "prices.csv", text)
    header, rows, delimiter = read_csv(path)
    assert delimiter == ";"
    assert header == ["a", "b", "price"]  # the byte order mark is removed
    info = inspect_csv(path)
    assert info.delimiter == ";"
    assert info.target_kind == "number"
    assert info.target_range == [10.25, 14.0]
    assert info.class_names is None


def test_csv_tab_separated(tmp_path):
    path = write(tmp_path, "data.tsv", "x\ty\n1\t2.5\n2\t3.5\n3\t4.5\n4\t5.5\n")
    info = inspect_csv(path)
    assert info.delimiter == "\t"
    assert info.num_features == 1


def test_csv_ragged_rows(tmp_path):
    path = write(tmp_path, "bad.csv", "a,b,c\n1,2,3\n4,5\n6,7,8\n9,10,11\n")
    with pytest.raises(DatasetError, match="Line 3 has 2 values, expected 3."):
        inspect_csv(path)


def test_csv_non_numeric_feature_names_line_and_column(tmp_path):
    path = write(tmp_path, "bad.csv", "a,b,c\n1,2,x\n4,five,y\n6,7,z\n9,10,x\n")
    with pytest.raises(DatasetError) as excinfo:
        inspect_csv(path)
    assert str(excinfo.value).startswith("Line 3, column 'b': 'five' is not a number.")


@pytest.mark.parametrize(
    "text,message",
    [
        ("", "The CSV file is empty."),
        ("1\n2\n3\n4\n5\n", "needs at least two columns"),
        ("a,b\n1,2\n3,4\n", "needs at least 4 data rows"),
        ("a,b\n1,2\n3,\n5,6\n7,8\n", "Line 3: the target (last column) is empty."),
    ],
)
def test_csv_errors(tmp_path, text, message):
    path = write(tmp_path, "bad.csv", text)
    with pytest.raises(DatasetError) as excinfo:
        inspect_csv(path)
    assert message in str(excinfo.value)


def test_csv_must_be_utf8(tmp_path):
    path = tmp_path / "latin1.csv"
    path.write_bytes("größe,ziel\n1,ä\n2,ö\n3,ü\n4,ß\n".encode("latin-1"))
    with pytest.raises(DatasetError, match="not UTF-8"):
        inspect_csv(path)


# --------------------------------------------------------------------------- #
# Image folders and ZIP files
# --------------------------------------------------------------------------- #


def make_images(root, names):
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not decoded during inspection")


def test_image_folder(tmp_path):
    make_images(
        tmp_path,
        ["cats/1.png", "cats/2.JPG", "cats/sub/3.png", "dogs/1.png", "dogs/.hidden.png"],
    )
    make_images(tmp_path, ["dogs/notes.txt", ".DS_Store/x.png", "__MACOSX/cats/._1.png"])
    info = inspect_image_folder(tmp_path, name="pets")
    assert info.kind == "images" and info.name == "pets"
    assert info.class_names == ["cats", "dogs"]
    assert info.class_counts == [3, 1]
    assert info.num_samples == 4
    assert info.subdir == ""
    files = list_image_files(tmp_path)
    assert [p.name for p in files["cats"]] == ["1.png", "2.JPG", "3.png"]


def test_image_folder_skips_wrapper_folders(tmp_path):
    make_images(tmp_path, ["export/data/cats/1.png", "export/data/dogs/1.png"])
    make_images(tmp_path, ["export/data/cats/2.png", "export/data/dogs/2.png"])
    (tmp_path / "__MACOSX").mkdir()
    assert find_image_root(tmp_path) == tmp_path / "export" / "data"
    info = inspect_image_folder(tmp_path)
    assert info.subdir == "export/data"
    assert info.class_names == ["cats", "dogs"]


@pytest.mark.parametrize(
    "names,message",
    [
        ([], "No images found."),
        (["a.png", "b.png", "c.png", "d.png"], "one folder per class"),
        (["cats/1.png", "cats/2.png"], "Found only one class folder ('cats')"),
        (["data/cats/1.png", "data/cats/2.png"], "Found only one class folder ('cats')"),
        (["cats/1.png", "dogs/1.png", "dogs/2.png"], "at least 4 images"),
    ],
)
def test_image_folder_errors(tmp_path, names, message):
    make_images(tmp_path, names)
    with pytest.raises(DatasetError) as excinfo:
        inspect_image_folder(tmp_path)
    assert message in str(excinfo.value)


def test_image_folder_must_exist(tmp_path):
    with pytest.raises(DatasetError, match="is not a folder"):
        inspect_image_folder(tmp_path / "missing")


def make_zip(path, entries):
    """``entries``: name -> bytes, or ZipInfo -> bytes."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return path


def test_extract_image_zip(tmp_path):
    archive = make_zip(
        tmp_path / "pets.zip",
        {
            "pets/cats/1.png": b"a" * 10,
            "pets/cats/2.png": b"b" * 10,
            "pets/dogs/1.jpg": b"c" * 10,
            "pets/dogs/readme.txt": b"skipped",
            "pets/.DS_Store": b"skipped",
            "__MACOSX/pets/cats/._1.png": b"skipped",
        },
    )
    dest = tmp_path / "out"
    assert extract_image_zip(archive, dest) == 3
    files = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
    assert files == ["pets/cats/1.png", "pets/cats/2.png", "pets/dogs/1.jpg"]
    assert (dest / "pets/cats/1.png").read_bytes() == b"a" * 10


@pytest.mark.parametrize(
    "name", ["../evil.png", "cats/../../evil.png", "/etc/evil.png", "C:/evil.png", "..\\evil.png"]
)
def test_extract_image_zip_rejects_unsafe_paths(tmp_path, name):
    info = zipfile.ZipInfo("cats/1.png")
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(info, b"ok")
        evil = zipfile.ZipInfo("placeholder.png")
        evil.filename = name  # bypass any normalisation of ZipInfo()
        zf.writestr(evil, b"evil")
    dest = tmp_path / "sandbox" / "out"
    with pytest.raises(DatasetError, match="unsafe path"):
        extract_image_zip(archive, dest)
    assert not dest.exists()
    assert not list(tmp_path.rglob("evil.png"))


def test_extract_image_zip_rejects_symlinks(tmp_path):
    link = zipfile.ZipInfo("cats/link.png")
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive = tmp_path / "links.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("cats/1.png", b"ok")
        zf.writestr(link, "/etc/passwd")
    dest = tmp_path / "out"
    with pytest.raises(DatasetError, match="symbolic links"):
        extract_image_zip(archive, dest)
    assert not dest.exists()


def test_extract_image_zip_limits_file_count(tmp_path):
    archive = make_zip(tmp_path / "many.zip", {f"c/{i}.png": b"x" for i in range(4)})
    with pytest.raises(DatasetError, match="more than 3 images"):
        extract_image_zip(archive, tmp_path / "out", max_files=3)
    assert not (tmp_path / "out").exists()


def test_extract_image_zip_limits_unpacked_size(tmp_path):
    archive = tmp_path / "bomb.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("c/1.png", b"\0" * 5000)  # compresses to a few bytes
    with pytest.raises(DatasetError, match="larger than"):
        extract_image_zip(archive, tmp_path / "out", max_bytes=1000)
    assert not (tmp_path / "out").exists()


def test_extract_image_zip_without_images(tmp_path):
    archive = make_zip(tmp_path / "docs.zip", {"readme.txt": b"hi", "data/a.csv": b"1,2"})
    with pytest.raises(DatasetError, match="contains no images"):
        extract_image_zip(archive, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_extract_invalid_or_damaged_zip(tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"PK this is not a zip")
    with pytest.raises(DatasetError, match="This is not a valid ZIP file."):
        extract_image_zip(bad, tmp_path / "out")
    assert not (tmp_path / "out").exists()

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("cats/1.png", bytes(range(256)) * 40)
    data = bytearray(buffer.getvalue())
    start = data.find(b"cats/1.png") + len("cats/1.png")
    for i in range(start + 4, start + 40):
        data[i] ^= 0xA5  # corrupt the compressed stream
    damaged = tmp_path / "damaged.zip"
    damaged.write_bytes(bytes(data))
    with pytest.raises(DatasetError):
        extract_image_zip(damaged, tmp_path / "out2")
    assert not (tmp_path / "out2").exists()


def test_extracted_zip_with_wrapper_folder_can_be_inspected(tmp_path):
    names = [f"flowers/{c}/{i}.png" for c in ("rose", "tulip") for i in range(3)]
    archive = make_zip(tmp_path / "flowers.zip", {name: b"x" for name in names})
    dest = tmp_path / "images"
    extract_image_zip(archive, dest)
    info = inspect_image_folder(dest, name="flowers")
    assert info.subdir == "flowers"
    assert info.class_names == ["rose", "tulip"]
    assert info.class_counts == [3, 3]


# --------------------------------------------------------------------------- #
# Dataset metadata
# --------------------------------------------------------------------------- #


def test_dataset_info_round_trip():
    info = builtin_info("cifar10")
    data = info.to_dict()
    assert data["num_classes"] == 10
    assert data["sample_shape"] == [3, 32, 32]
    assert DatasetInfo.from_dict(json.loads(json.dumps(data))) == info
    assert random_info().to_dict()["kind"] == "random"


def test_builtin_datasets():
    assert set(BUILTIN_DATASETS) == {"mnist", "fashion_mnist", "cifar10", "shapes"}
    data = BUILTIN_DATASETS["mnist"].to_dict()
    assert data["download_mb"] == 12 and data["train_size"] == 60_000
    assert data["test_size"] == 10_000 and "val_size" not in data
    with pytest.raises(DatasetError, match="Unknown dataset 'imagenet'"):
        builtin_info("imagenet")


# --------------------------------------------------------------------------- #
# Compatibility
# --------------------------------------------------------------------------- #


def check(graph, info):
    """``graph`` is a graph document or the id of a bundled example."""
    if isinstance(graph, str):
        graph = EXAMPLES[graph]["graph"]
    return check_compatibility(graph, info)


@pytest.mark.parametrize("example_id", sorted(EXAMPLES))
def test_every_example_trains_on_demo_data(example_id):
    result = check(example_id, random_info())
    assert result.ok and not result.errors
    plan = result.plan
    assert plan.task == "demo"
    assert plan.summary.startswith("Demo data")
    assert plan.notes
    assert len(plan.inputs) >= 1 and len(plan.targets) >= 1
    assert TrainingPlan.from_dict(json.loads(json.dumps(plan.to_dict()))) == plan


def test_demo_plan_details():
    lstm = check("lstm_sentiment", random_info()).plan
    assert lstm.inputs == [{"shape": [128], "dtype": "long", "vocab": 20000}]
    assert lstm.targets == [{"kind": "class", "num_classes": 2}]
    multi = check("multi_input", random_info()).plan
    assert [i["shape"] for i in multi.inputs] == [[1, 28, 28], [10]]
    assert multi.targets == [{"kind": "value", "shape": [1]}]
    assert "regression" in multi.summary


@pytest.mark.parametrize(
    "example_id,layout",
    [("mlp_mnist", "flat"), ("cnn_mnist", "chw"), ("resnet_block", "chw")],
)
def test_image_examples_train_on_mnist(example_id, layout):
    result = check(example_id, builtin_info("mnist"))
    assert result.ok, result.errors
    assert result.plan.task == "classification"
    assert result.plan.image["layout"] == layout
    assert result.plan.class_names == [str(i) for i in range(10)]
    assert any("downloaded on first use (about 12 MB)" in n for n in result.plan.notes)
    assert result.plan.summary == (
        "Classification into 10 classes (MNIST, 60,000 training and 10,000 test images)."
    )
    assert (
        "Validation data is taken from the 60,000 training images (validation split). The "
        "10,000 official test images are the test set: the kept weights are measured on them "
        "once, after training." in result.plan.notes
    )


def test_mnist_into_rgb_network_is_resized_and_copied():
    result = check("resnet_block", builtin_info("mnist"))
    assert result.warnings == ["MNIST images (28×28) are resized to 32×32."]
    assert "The grayscale images are copied into 3 channels." in result.plan.notes
    assert result.plan.image == {"channels": 3, "height": 32, "width": 32, "layout": "chw"}


def test_cifar_into_grayscale_network():
    result = check("cnn_mnist", builtin_info("cifar10"))
    assert result.ok
    assert result.warnings == ["CIFAR-10 images (32×32) are resized to 28×28."]
    assert "CIFAR-10 images are converted to grayscale." in result.plan.notes


@pytest.mark.parametrize(
    "example_id,dataset,message",
    [
        ("mlp_mnist", "cifar10", "Set the Input shape to 3072 or to 3, 32, 32."),
        ("tabular_regression", "mnist", "Set the Input shape to 784 or to 1, 28, 28."),
        ("lstm_sentiment", "mnist", "Text models can only be trained on demo data"),
        ("transformer_classifier", "cifar10", "integer token ids"),
        ("multi_input", "mnist", "needs a network with exactly one Input layer"),
    ],
)
def test_incompatible_examples(example_id, dataset, message):
    result = check(example_id, builtin_info(dataset))
    assert not result.ok
    assert message in result.errors[0]
    assert result.to_dict()["plan"] is None


def test_hw_layout_for_rnn_over_image_rows():
    graph = chain(
        ("Input", {"shape": [28, 28]}),
        ("LSTM", {"hidden_size": 16}),
        ("Linear", {"out_features": 10}),
    )
    result = check(graph, builtin_info("fashion_mnist"))
    assert result.ok, result.errors
    assert result.plan.image["layout"] == "hw"
    assert result.plan.class_names[0] == "T-shirt/top"


def test_too_few_outputs_for_the_classes():
    graph = chain(
        ("Input", {"shape": [1, 28, 28]}), ("Flatten", {}), ("Linear", {"out_features": 5})
    )
    result = check(graph, builtin_info("mnist"))
    assert not result.ok
    assert "has 10 classes" in result.errors[0]
    assert "Linear with 10 outputs" in result.errors[0]


def test_extra_outputs_are_a_warning():
    graph = chain(("Input", {"shape": [784]}), ("Linear", {"out_features": 12}))
    result = check(graph, builtin_info("mnist"))
    assert result.ok
    assert "only 10 classes" in result.warnings[0]
    assert result.plan.num_classes == 12
    assert result.plan.class_names[-1] == "class 11"


def test_invalid_network():
    result = check({"nodes": [], "edges": []}, random_info())
    assert not result.ok
    assert result.errors == ["The network has problems. Fix the layers marked in red first."]


def csv_info(tmp_path, text):
    return inspect_csv(write(tmp_path, "data.csv", text))


IRIS = "a,b,c,d,species\n" + "\n".join(
    f"{i},{i + 1},{i + 2},{i + 3},{name}" for i, name in enumerate(["x", "y", "z", "x", "y", "z"])
)


def test_csv_classification(tmp_path):
    info = csv_info(tmp_path, IRIS)
    graph = chain(("Input", {"shape": [4]}), ("Linear", {"out_features": 3}))
    result = check(graph, info)
    assert result.ok, result.errors
    plan = result.plan
    assert plan.task == "classification"
    assert plan.targets == [{"kind": "class", "num_classes": 3}]
    assert plan.class_names == ["x", "y", "z"]
    assert any("alphabetical order" in n for n in plan.notes)


def test_csv_regression(tmp_path):
    rows = "\n".join(f"{i},{i * 2},{i * 0.5 + 0.25}" for i in range(6))
    info = csv_info(tmp_path, "a,b,price\n" + rows)
    graph = chain(("Input", {"shape": [2]}), ("Linear", {"out_features": 1}))
    result = check(graph, info)
    assert result.ok
    assert result.plan.task == "regression"
    assert result.plan.targets == [{"kind": "value", "shape": [1]}]


def test_csv_integer_classes(tmp_path):
    info = csv_info(tmp_path, "a,label\n1,0\n2,1\n3,2\n4,1\n")
    result = check(chain(("Input", {"shape": [1]}), ("Linear", {"out_features": 3})), info)
    assert result.ok and result.plan.class_names == ["0", "1", "2"]
    small = check(chain(("Input", {"shape": [1]}), ("Linear", {"out_features": 2})), info)
    assert "class indices must be between 0 and 1" in small.errors[0]
    regression = check(chain(("Input", {"shape": [1]}), ("Linear", {"out_features": 1})), info)
    assert regression.ok and regression.plan.task == "regression"
    assert "only contains the whole numbers 0, 1, 2" in regression.warnings[0]


@pytest.mark.parametrize(
    "shape,outputs,message",
    [
        ([3], 3, "Set the Input shape to 4."),
        ([4], 1, "the last layer needs 3 outputs"),
        ([4], 2, "has 3 classes"),
    ],
)
def test_csv_mismatches(tmp_path, shape, outputs, message):
    info = csv_info(tmp_path, IRIS)
    graph = chain(("Input", {"shape": shape}), ("Linear", {"out_features": outputs}))
    result = check(graph, info)
    assert not result.ok
    assert message in result.errors[0]


def test_csv_decimal_target_needs_one_output(tmp_path):
    info = csv_info(tmp_path, "a,y\n1,0.5\n2,1.5\n3,2.5\n4,3.25\n")
    result = check(chain(("Input", {"shape": [1]}), ("Linear", {"out_features": 4})), info)
    assert "decimal numbers, which is a regression task" in result.errors[0]


def test_csv_needs_flat_output(tmp_path):
    info = csv_info(tmp_path, IRIS)
    graph = chain(
        ("Input", {"shape": [4]}), ("Linear", {"out_features": 6}), ("Reshape", {"shape": [2, 3]})
    )
    result = check(graph, info)
    assert not result.ok
    assert "For CSV data it must output 1 value" in result.errors[0]


def image_info(tmp_path, classes=("cats", "dogs")):
    make_images(tmp_path / "pets", [f"{c}/{i}.png" for c in classes for i in range(3)])
    return inspect_image_folder(tmp_path / "pets")


def test_image_folder_plan(tmp_path):
    info = image_info(tmp_path)
    graph = chain(
        ("Input", {"shape": [3, 64, 64]}),
        ("Flatten", {}),
        ("Linear", {"out_features": 2}),
    )
    result = check(graph, info)
    assert result.ok, result.errors
    assert result.plan.image == {"channels": 3, "height": 64, "width": 64, "layout": "chw"}
    assert result.plan.class_names == ["cats", "dogs"]
    assert "Images are converted to RGB and resized to 64×64." in result.plan.notes


@pytest.mark.parametrize(
    "shape,message",
    [
        ([1000], "set the Input shape to channels, height, width"),
        ([4], "set the Input shape to channels, height, width"),  # not 2×2 images
        ([2, 8, 8], "The Input has 2 channels"),
        ([1, 2048, 2048], "at most 1024×1024"),
    ],
)
def test_image_folder_input_errors(tmp_path, shape, message):
    info = image_info(tmp_path)
    graph = chain(("Input", {"shape": shape}), ("Flatten", {}), ("Linear", {"out_features": 2}))
    result = check(graph, info)
    assert not result.ok
    assert message in result.errors[0]


@pytest.mark.parametrize(
    "shape,image",
    [
        ([12288], {"channels": 3, "height": 64, "width": 64, "layout": "flat"}),
        ([784], {"channels": 1, "height": 28, "width": 28, "layout": "flat"}),
    ],
)
def test_image_folder_into_a_flat_input(tmp_path, shape, image):
    info = image_info(tmp_path)
    graph = chain(("Input", {"shape": shape}), ("Linear", {"out_features": 2}))
    result = check(graph, info)
    assert result.ok, result.errors
    assert result.plan.image == image
    side = image["height"]
    assert (
        f"Each image is flattened to {shape[0]:,} values ({image['channels']}×{side}×{side})."
        in (result.plan.notes)
    )


# --------------------------------------------------------------------------- #
# Suggested fixes
# --------------------------------------------------------------------------- #


def fixes_of(graph, info):
    """``{node id: params}`` of the suggested fixes; they must make the network fit."""
    if isinstance(graph, str):
        graph = EXAMPLES[graph]["graph"]
    result = check_compatibility(graph, info)
    assert result.to_dict()["fixes"] == result.fixes
    if result.fixes:
        changes = {fix["node"]: fix["params"] for fix in result.fixes}
        fixed = {
            **graph,
            "nodes": [
                {**n, "params": {**n.get("params", {}), **changes.get(n["id"], {})}}
                for n in graph["nodes"]
            ],
        }
        assert check_compatibility(fixed, info).ok
    return {fix["node"]: fix["params"] for fix in result.fixes}


def test_fixes_for_a_csv_file(tmp_path):
    info = csv_info(tmp_path, IRIS)
    result = check("mlp_mnist", info)
    assert not result.ok
    assert result.fixes == [
        {
            "node": "input",
            "type": "Input",
            "params": {"shape": [4]},
            "previous": {"shape": [784]},
            "summary": "Input shape 784 → 4",
        },
        {
            "node": "fc3",
            "type": "Linear",
            "params": {"out_features": 3},
            "previous": {"out_features": 10},
            "summary": "Outputs of the last layer 10 → 3",
        },
    ]
    # The errors and warnings stay as they are.
    assert "Set the Input shape to 4." in result.errors[0]


def test_fixes_for_csv_targets(tmp_path):
    regression = csv_info(tmp_path, "a,b,price\n" + "\n".join(f"{i},{i},{i}.5" for i in range(6)))
    assert fixes_of("mlp_mnist", regression) == {
        "input": {"shape": [2]},
        "fc3": {"out_features": 1},
    }
    assert fixes_of("tabular_regression", regression) == {"features": {"shape": [2]}}
    integers = csv_info(tmp_path, "a,label\n1,0\n2,1\n3,2\n4,1\n")
    # Classification by class index: one output per index; with 1 output a regression.
    assert fixes_of("mlp_mnist", integers) == {
        "input": {"shape": [1]},
        "fc3": {"out_features": 3},
    }
    assert fixes_of("tabular_regression", integers) == {"features": {"shape": [1]}}
    negative = csv_info(tmp_path, "a,label\n1,-1\n2,1\n3,2\n4,1\n")
    assert fixes_of(
        chain(("Input", {"shape": [1]}), ("Linear", {"out_features": 4})), negative
    ) == {"n1": {"out_features": 1}}


def test_fixes_for_prepared_data(tmp_path):
    """Data tab datasets: class names only for classification, class indices as targets."""
    info = csv_info(tmp_path, "a,b,y\n" + "\n".join(f"{i},{i},{i % 3}" for i in range(9)))
    info.predefined_split = [6, 2, 1]
    info.class_names = ["low", "mid", "high"]
    assert fixes_of("tabular_regression", info) == {
        "features": {"shape": [2]},
        "out": {"out_features": 3},
    }
    # Fits, but 7 outputs are never the right answer: offered without an error.
    seven = chain(("Input", {"shape": [2]}), ("Linear", {"out_features": 7}))
    assert check(seven, info).ok
    assert fixes_of(seven, info) == {"n1": {"out_features": 3}}
    info.class_names = None  # regression
    assert fixes_of("mlp_mnist", info) == {"input": {"shape": [2]}, "fc3": {"out_features": 1}}


@pytest.mark.parametrize(
    "example_id,dataset,expected",
    [
        ("mlp_mnist", "mnist", {}),
        ("mlp_mnist", "cifar10", {"input": {"shape": [3072]}}),  # stays flat
        ("cnn_mnist", "cifar10", {}),  # converted to grayscale and resized: works
        ("resnet_block", "mnist", {}),
        (
            "tabular_regression",
            "mnist",
            {"features": {"shape": [784]}, "out": {"out_features": 10}},
        ),
        ("lstm_sentiment", "mnist", {}),  # text models: nothing to suggest
        ("multi_input", "mnist", {}),
    ],
)
def test_fixes_for_builtin_datasets(example_id, dataset, expected):
    assert fixes_of(example_id, builtin_info(dataset)) == expected


def test_fixes_keep_the_kind_of_input():
    cnn = chain(
        ("Input", {"shape": [4]}),
        ("Reshape", {"shape": [1, 2, 2]}),
        ("Conv2d", {"out_channels": 4, "kernel_size": [1, 1]}),
        ("Flatten", {}),
        ("Linear", {"out_features": 10}),
    )
    # A Reshape needs the flat size; a channels-first Input for Conv2d and Flatten.
    assert fixes_of(cnn, builtin_info("mnist")) == {}  # 784 cannot be reshaped into 1×2×2
    channels = chain(
        ("Input", {"shape": [2, 16, 16]}),
        ("Conv2d", {"out_channels": 4}),
        ("Flatten", {}),
        ("Linear", {"out_features": 10}),
    )
    assert fixes_of(channels, builtin_info("cifar10")) == {"n0": {"shape": [3, 16, 16]}}
    rows = chain(("Input", {"shape": [2000, 2000]}), ("LSTM", {"hidden_size": 8}), ("Linear", {}))
    assert fixes_of(rows, builtin_info("mnist")) == {
        "n0": {"shape": [28, 28]},
        "n2": {"out_features": 10},
    }


def test_fixes_for_image_folders(tmp_path):
    info = image_info(tmp_path)
    # The MLP takes the images flattened (28×28 grayscale); only the outputs change.
    assert fixes_of("mlp_mnist", info) == {"fc3": {"out_features": 2}}
    assert fixes_of("tabular_regression", info) == {
        "features": {"shape": [12288]},
        "out": {"out_features": 2},
    }
    flatten = chain(("Input", {"shape": [5]}), ("Flatten", {}), ("Linear", {"out_features": 2}))
    assert fixes_of(flatten, info) == {"n0": {"shape": [12288]}}
    conv = chain(
        ("Input", {"shape": [4, 2048, 1024]}),
        ("Conv2d", {"out_channels": 2}),
        ("AdaptiveAvgPool2d", {}),
        ("Flatten", {}),
        ("Linear", {"out_features": 2}),
    )
    assert fixes_of(conv, info) == {"n0": {"shape": [3, 1024, 512]}}


def test_fixes_for_the_last_layer_behind_activations(tmp_path):
    info = csv_info(tmp_path, IRIS)
    graph = chain(
        ("Input", {"shape": [4]}),
        ("Linear", {"out_features": 8}),
        ("Dropout", {"p": 0.1}),
        ("Activation", {"function": "relu"}),
    )
    assert fixes_of(graph, info) == {"n1": {"out_features": 3}}
    # A layer that changes the shape (or a second branch) is not changed.
    graph["nodes"][2] = {"id": "n2", "type": "Reshape", "params": {"shape": [2, 4]}}
    assert fixes_of(graph, info) == {}


def test_no_fixes_when_other_layers_would_have_to_change(tmp_path):
    info = csv_info(tmp_path, IRIS)
    fixed = chain(("Input", {"shape": [5]}), ("Linear", {"in_features": 5, "out_features": 3}))
    assert fixes_of(fixed, info) == {}
    assert fixes_of("cnn_mnist", info) == {}
    assert fixes_of("mlp_mnist", random_info()) == {}
    # An analysis (instead of the graph) cannot be changed: no fixes.
    from brickdnn.graph import analyze

    assert check_compatibility(analyze(EXAMPLES["mlp_mnist"]["graph"]), info).fixes == []


def test_fixes_with_long_inputs_and_saved_projects(tmp_path):
    info = csv_info(tmp_path, IRIS)
    graph = chain(("Input", {"shape": [4], "dtype": "long"}), ("Embedding", {}), ("Flatten", {}))
    assert fixes_of(graph, info) == {}  # the Embedding needs token ids
    # Saved project files ({"name", "graph"}) work as well.
    project = {"name": "MLP", "graph": EXAMPLES["mlp_mnist"]["graph"]}
    assert [f["node"] for f in check_compatibility(project, info).fixes] == ["input", "fc3"]


# --------------------------------------------------------------------------- #
# Run directory files
# --------------------------------------------------------------------------- #


def test_state_files(tmp_path):
    assert st.read_state(tmp_path) is None
    state = st.initial_state(epochs=7)
    assert state["status"] == "starting" and state["epochs"] == 7
    assert state["history"] == [] and state["weights"] is False
    st.write_state(tmp_path, state)
    assert st.read_state(tmp_path) == state
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]  # no temp files left

    (tmp_path / "state.json").write_text("{broken")
    assert st.read_state(tmp_path) is None
    (tmp_path / "state.json").write_text("[1, 2]")
    assert st.read_state(tmp_path) is None


def test_state_json_is_always_valid_json(tmp_path):
    state = st.initial_state()
    state["history"] = [{"epoch": 1, "train_loss": float("inf"), "val_loss": float("nan")}]
    state["progress"] = {"loss": float("-inf"), "steps": [1.5, float("nan")]}
    st.write_state(tmp_path, state)
    text = (tmp_path / "state.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    saved = st.read_state(tmp_path)
    assert saved["history"][0] == {"epoch": 1, "train_loss": None, "val_loss": None}
    assert saved["progress"] == {"loss": None, "steps": [1.5, None]}
    with pytest.raises(ValueError):
        st.write_json_atomic(tmp_path / "other.json", {"value": float("nan")})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state.json"]


def test_job_files(tmp_path):
    run_dir = tmp_path / "runs" / "abc"
    job = {"name": "x", "graph": {"nodes": [], "edges": []}}
    st.write_job(run_dir, job)
    assert st.read_job(run_dir) == job
    assert st.read_job(tmp_path) is None


def test_stop_file(tmp_path):
    assert not st.stop_requested(tmp_path)
    st.request_stop(tmp_path)
    assert st.stop_requested(tmp_path)
    assert (tmp_path / st.STOP_FILE).exists()


def test_is_stale():
    now = time.time()
    assert not st.is_stale(None)
    assert not st.is_stale({"status": "completed", "updated_at": 0})
    assert st.is_stale({"status": "running"})
    assert st.is_stale({"status": "running", "updated_at": now - st.STALE_SECONDS - 1})
    assert not st.is_stale({"status": "preparing", "updated_at": now - 1})
    assert not st.is_stale({"status": "running", "updated_at": 100.0}, now=110.0)


def test_status_constants():
    # "queued" is the server's: a run waits there for a training worker. Runners never report it.
    assert st.STATUSES == (
        "queued",
        "starting",
        "preparing",
        "running",
        "completed",
        "stopped",
        "failed",
    )
    assert st.RUNNING_STATUSES == ("starting", "preparing", "running")
    assert (st.QUEUED, *st.RUNNING_STATUSES) == st.ACTIVE_STATUSES
    assert set(st.ACTIVE_STATUSES).isdisjoint(st.TERMINAL_STATUSES)
    assert st.initial_state()["status"] == "starting"


def test_runner_lowers_its_priority_before_training(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "lower_priority", lambda: calls.append("lower_priority"))
    monkeypatch.setattr(runner, "run", lambda run_dir: calls.append("run"))
    assert runner.main([str(tmp_path)]) == 0
    assert calls == ["lower_priority", "run"]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="oom_score_adj is Linux only")
def test_lower_priority():
    # In its own process: a lowered priority cannot be raised again.
    code = (
        "import os; from brickdnn.training import runner; runner.lower_priority(); "
        "print(os.getpriority(os.PRIO_PROCESS, 0), open('/proc/self/oom_score_adj').read())"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    nice, oom_score_adj = map(int, result.stdout.split())
    assert nice == min(os.getpriority(os.PRIO_PROCESS, 0) + runner.NICE, 19)
    assert oom_score_adj == runner.OOM_SCORE_ADJ
