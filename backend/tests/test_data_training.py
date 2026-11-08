"""Training on datasets made in the Data tab: they bring their own split (``split.bin``).

The compatibility tests need no PyTorch; the engine tests train tiny networks on the CPU.
"""

from __future__ import annotations

import pytest

from brickdnn.training import TrainingConfig, check_compatibility, inspect_csv, torch_installed
from brickdnn.training.datasets import SPLIT_FILE
from brickdnn.training.reporting import Reporter

from .helpers import chain

needs_torch = pytest.mark.skipif(not torch_installed(), reason="PyTorch is not installed")

# 40 rows: every 5th row is validation, the row after it test, the rest training.
PATTERN = [1 if i % 5 == 0 else 2 if i % 5 == 1 else 0 for i in range(40)]


def mlp(features: int, outputs: int):
    return chain(
        ("Input", {"shape": [features]}),
        ("Linear", {"out_features": 8}),
        ("Activation", {}),
        ("Linear", {"out_features": outputs}),
    )


def write_prepared(tmp_path, split, *, classes=True, split_bytes=None):
    """A prepared dataset as the export writes it: data.csv (target = class index or value)
    and split.bin. Feature ``a`` is the row number, so statistics show which rows were used."""
    folder = tmp_path / "dataset"
    folder.mkdir()
    lines = ["a,b,target"]
    for i in range(len(split)):
        target = i % 2 if classes else 10.0 + 3 * i
        lines.append(f"{i},{(i * 7) % 11},{target}")
    path = folder / "data.csv"
    path.write_text("\n".join(lines) + "\n")
    (folder / SPLIT_FILE).write_bytes(bytes(split) if split_bytes is None else split_bytes)
    info = inspect_csv(path, name="prepared")
    info.predefined_split = [split.count(0), split.count(1), split.count(2)]
    if classes:
        info.class_names = ["no", "yes"]
    else:
        info.class_names = None
    return path, info


# --------------------------------------------------------------------------- #
# Compatibility (no PyTorch needed)
# --------------------------------------------------------------------------- #


def test_check_uses_the_class_names_and_mentions_the_split(tmp_path):
    _, info = write_prepared(tmp_path, PATTERN)
    result = check_compatibility(mlp(2, 2), info)
    assert result.ok, result.errors
    assert result.plan.task == "classification"
    assert result.plan.class_names == ["no", "yes"]
    assert result.warnings == [
        "Uses the split of the data preparation (24 training, 8 validation and 8 test rows); "
        "the validation and test split settings are ignored."
    ]
    assert "Classes in index order: no, yes." in result.plan.notes

    # More outputs than classes: the extra outputs get placeholder names.
    assert check_compatibility(mlp(2, 3), info).plan.class_names == ["no", "yes", "class 2"]

    # One output: trained as a regression, with a hint that names the classes.
    result = check_compatibility(mlp(2, 1), info)
    assert result.plan.task == "regression"
    assert "holds 2 classes (no, yes)" in result.warnings[1]


def test_check_of_a_split_without_test_rows(tmp_path):
    split = [1 if i % 4 == 0 else 0 for i in range(40)]
    _, info = write_prepared(tmp_path, split, classes=False)
    result = check_compatibility(mlp(2, 1), info)
    assert result.ok and result.plan.task == "regression"
    assert result.warnings == [
        "Uses the split of the data preparation (30 training and 10 validation rows, no test "
        "set); the validation and test split settings are ignored."
    ]


def test_check_of_ordinary_csv_files_is_unchanged(tmp_path):
    path, info = write_prepared(tmp_path, PATTERN)
    info = inspect_csv(path)  # an upload: no predefined split
    result = check_compatibility(mlp(2, 2), info)
    assert result.plan.class_names == ["0", "1"]
    assert result.warnings == []


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


class Recorder(Reporter):
    """Keeps the facts the engine reports at the start and the test result."""

    def __init__(self):
        self.info: dict = {}
        self.test = None

    def setup(self, info):
        self.info.update(info)

    def test_end(self, result):
        self.test = result


def run_job(graph, info, path, tmp_path, **settings):
    from brickdnn.training.engine import TrainingJob, train

    values = {"epochs": 1, "batch_size": 8, "device": "cpu", "seed": 0, **settings}
    job = TrainingJob(
        graph=graph, dataset=info, config=TrainingConfig.from_dict(values), path=str(path)
    )
    reporter = Recorder()
    result = train(job, reporter, tmp_path / "weights.pt")
    return result, reporter


@needs_torch
def test_training_uses_exactly_the_stored_split(tmp_path):
    path, info = write_prepared(tmp_path, PATTERN)
    # val_split / test_split are ignored for prepared datasets.
    result, rec = run_job(mlp(2, 2), info, path, tmp_path, val_split=0.5, test_split=0.0)
    assert result.status == "completed", result.message
    assert (rec.info["train_size"], rec.info["val_size"], rec.info["test_size"]) == (24, 8, 8)
    assert rec.info["class_names"] == ["no", "yes"]
    pre = rec.info["preprocessing"]
    assert pre["split"] == {"source": "dataset", "counts": [24, 8, 8]}
    # The features are standardised with the training rows of split.bin: feature a is the
    # row number, so its mean shows which rows they were.
    train_rows = [i for i, part in enumerate(PATTERN) if part == 0]
    assert pre["mean"][0] == pytest.approx(sum(train_rows) / len(train_rows))
    assert rec.test is not None  # the test rows were evaluated once at the end


@needs_torch
def test_regression_target_statistics_come_from_the_training_rows(tmp_path):
    split = [1 if i % 4 == 0 else 0 for i in range(40)]  # no test rows
    path, info = write_prepared(tmp_path, split, classes=False)
    result, rec = run_job(mlp(2, 1), info, path, tmp_path, test_split=0.3)
    assert result.status == "completed", result.message
    assert (rec.info["train_size"], rec.info["val_size"], rec.info["test_size"]) == (30, 10, None)
    assert rec.test is None
    train_targets = [10.0 + 3 * i for i, part in enumerate(split) if part == 0]
    pre = rec.info["preprocessing"]
    assert pre["target_mean"] == pytest.approx(sum(train_targets) / len(train_targets))
    assert pre["split"]["counts"] == [30, 10, 0]


@needs_torch
def test_max_samples_shrinks_every_part_in_proportion(tmp_path):
    path, info = write_prepared(tmp_path, PATTERN)
    result, rec = run_job(mlp(2, 2), info, path, tmp_path, max_samples=12)
    assert result.status == "completed", result.message
    # 24 / 8 / 8 rows -> 12 training rows and half of the validation and test rows.
    assert (rec.info["train_size"], rec.info["val_size"], rec.info["test_size"]) == (12, 4, 4)
    assert rec.info["preprocessing"]["split"]["counts"] == [12, 4, 4]


@needs_torch
def test_predefined_split_indices_stay_in_their_part():
    import torch

    from brickdnn.training import data

    split = torch.tensor(PATTERN, dtype=torch.uint8)
    train, val, test = data.predefined_split_indices(split, None, torch.Generator())
    assert train.tolist() == [i for i, p in enumerate(PATTERN) if p == 0]
    assert val.tolist() == [i for i, p in enumerate(PATTERN) if p == 1]
    assert test.tolist() == [i for i, p in enumerate(PATTERN) if p == 2]

    train, val, test = data.predefined_split_indices(split, 6, torch.Generator().manual_seed(1))
    assert (len(train), len(val), len(test)) == (6, 2, 2)
    assert all(PATTERN[i] == 0 for i in train.tolist())
    assert all(PATTERN[i] == 1 for i in val.tolist())
    assert all(PATTERN[i] == 2 for i in test.tolist())
    assert train.tolist() == sorted(train.tolist())

    no_test = torch.tensor([0, 0, 1, 0], dtype=torch.uint8)
    assert data.predefined_split_indices(no_test, None, torch.Generator())[2] is None


@needs_torch
@pytest.mark.parametrize(
    ("split_bytes", "counts", "message"),
    [
        (None, None, "split.bin) is missing"),
        (bytes(PATTERN[:-1]), None, "does not match its rows"),
        (bytes([3, *PATTERN[1:]]), None, "does not match its rows"),
        (bytes(PATTERN), [30, 5, 5], "does not match its rows"),
        (bytes([0] * 40), [40, 0, 0], "needs training and validation rows"),
    ],
    ids=["missing", "too-short", "bad-value", "other-counts", "no-validation"],
)
def test_broken_split_files(tmp_path, split_bytes, counts, message):
    path, info = write_prepared(tmp_path, PATTERN, split_bytes=split_bytes or b"")
    if split_bytes is None:
        (path.parent / SPLIT_FILE).unlink()
    if counts is not None:
        info.predefined_split = counts
    result, _ = run_job(mlp(2, 2), info, path, tmp_path)
    assert result.status == "failed"
    assert message in result.message


# --------------------------------------------------------------------------- #
# End to end: export in the Data tab, then train with the engine
# --------------------------------------------------------------------------- #


@needs_torch
@pytest.mark.django_db
def test_training_on_an_exported_dataset(tmp_path, settings):
    pytest.importorskip("pandas")
    from django.contrib.auth import get_user_model
    from rest_framework.test import APIClient

    from brick_dnn_core import training_service
    from brick_dnn_core.models import Dataset
    from brickdnn.training import DatasetInfo

    from .test_data_api import churn_csv, create_prep, import_csv

    settings.DATA_PREP_DIR = tmp_path / "prepare"
    settings.TRAINING_DIR = tmp_path / "training"
    settings.DATA_PREP_ENABLED = settings.TRAINING_ENABLED = True
    client = APIClient()
    client.force_authenticate(get_user_model().objects.create_user("ada", password="x-12345678"))
    source = import_csv(client, "churn.csv", churn_csv(60)).data
    prep = create_prep(client, source["id"], settings={"target": "churn"}).data
    response = client.post(f"/api/data/preps/{prep['id']}/export/", {}, format="json")
    assert response.status_code == 201, response.data
    dataset = Dataset.objects.get(pk=response.data["id"])
    info = DatasetInfo.from_dict(dataset.info)
    counts = info.predefined_split
    assert counts is not None and sum(counts) == 60

    result, rec = run_job(
        mlp(info.num_features, len(info.class_names)),
        info,
        training_service.dataset_path(dataset),
        tmp_path,
    )
    assert result.status == "completed", result.message
    sizes = [rec.info["train_size"], rec.info["val_size"], rec.info["test_size"] or 0]
    assert sizes == counts
    assert rec.info["class_names"] == ["leave", "stay"]
