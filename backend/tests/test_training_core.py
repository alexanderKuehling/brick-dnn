"""The torch-free part of brickdnn.training: settings, datasets, compatibility, run files."""

from __future__ import annotations

import io
import json
import stat
import time
import zipfile

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
    torch_installed,
    torchvision_installed,
)
from brickdnn.training import state as st
from brickdnn.training.datasets import find_image_root, list_image_files, parse_number, read_csv

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
        "device": "auto",
        "max_samples": None,
        "seed": None,
    }
    assert TrainingConfig.from_dict(config.to_dict()) == config


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
        ({"optimizer": "rmsprop"}, "Optimizer must be one of: adam, adamw, sgd."),
        ({"device": "tpu"}, "Device must be one of: auto, cpu, cuda, mps."),
        ({"max_samples": 5}, "Max. samples must be between 10 and"),
        ({"seed": -1}, "Seed must be between 0 and"),
    ],
)
def test_config_errors(data, message):
    with pytest.raises(ConfigError) as excinfo:
        TrainingConfig.from_dict(data)
    assert str(excinfo.value).startswith(message)


def test_config_must_be_an_object():
    with pytest.raises(ConfigError, match="must be an object"):
        TrainingConfig.from_dict([1, 2])  # type: ignore[arg-type]


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
    assert set(BUILTIN_DATASETS) == {"mnist", "fashion_mnist", "cifar10"}
    data = BUILTIN_DATASETS["mnist"].to_dict()
    assert data["download_mb"] == 12 and data["train_size"] == 60_000
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
        ([12288], "set the Input shape to channels, height, width"),
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
    assert st.STATUSES == ("starting", "preparing", "running", "completed", "stopped", "failed")
    assert set(st.ACTIVE_STATUSES).isdisjoint(st.TERMINAL_STATUSES)
