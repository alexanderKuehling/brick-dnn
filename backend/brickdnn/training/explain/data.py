"""Samples of a run's own data for explanations: a few validation images, or validation and
training rows of a CSV file, read again exactly as the training split them. A run without
validation data takes its test rows instead, or, without those too, its training rows (the
pool's ``meta["source"]`` says which: "validation", "test" or "training").

The training does not store its split, but it is reproducible: the same seed (the setting,
or the one the run chose and logged), the same split functions and settings give the same
rows. The samples are read once and kept in ``RUN_DIR/explain/pool.npz`` (small: at most
POOL_IMAGES images or POOL_ROWS + POOL_TRAIN_ROWS rows), so later explanations do not
read the dataset again.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..config import TrainingConfig
from ..datasets import BUILTIN_DATASETS, DEFAULT_DATA_ROOT, DatasetError, DatasetInfo
from ..job import TrainingJob
from ..reporting import TrainingError
from ..state import read_state
from ..tabular import csv_columns, csv_rows, read_split

EXPLAIN_DIR = "explain"
POOL_FILE = "pool.npz"
POOL_VERSION = 1
# Validation images kept for explanations (the sample picker shows some of them).
POOL_IMAGES = 64
# Validation rows (permutation importance, sample picker) and training rows (the points of
# a decision boundary).
POOL_ROWS = 1000
POOL_TRAIN_ROWS = 400
# A CSV file larger than this is not read again (it would take too long in a prediction).
MAX_CSV_BYTES = 64 * 1024**2
SEED_PATTERN = re.compile(r"Random seed: (\d+)")


class ExplainError(Exception):
    """The message is shown to the user."""


NO_DATA = (
    "The samples of this run's dataset are not available: {reason} You can still explain "
    "a sample you enter, draw or upload."
)


def explain_dir(run_dir: Path) -> Path:
    return run_dir / EXPLAIN_DIR


def find_seed(run_dir: Path, config: TrainingConfig) -> int | None:
    """The seed of the run: the setting, else the one the run chose and logged."""
    if config.seed is not None:
        return config.seed
    state = read_state(run_dir) or {}
    for line in state.get("log") or []:
        match = SEED_PATTERN.search(str(line))
        if match:
            return int(match.group(1))
    log = run_dir / "output.log"
    try:
        with log.open(encoding="utf-8", errors="replace") as handle:
            for _, line in zip(range(2000), handle, strict=False):
                match = SEED_PATTERN.search(line)
                if match:
                    return int(match.group(1))
    except OSError:
        pass
    return None


# --------------------------------------------------------------------------- #
# The pool file
# --------------------------------------------------------------------------- #


def load_pool(run_dir: Path, job: TrainingJob, preprocessing: dict[str, Any]) -> dict[str, Any]:
    """The samples of the run (read once, then from ``pool.npz``). Raises ExplainError."""
    path = explain_dir(run_dir) / POOL_FILE
    if path.is_file():
        try:
            with np.load(path, allow_pickle=False) as data:
                pool = {key: data[key] for key in data.files}
            meta = json.loads(str(pool.pop("meta")))
            if meta.get("version") == POOL_VERSION:
                pool["meta"] = meta
                return pool
        except (OSError, ValueError, KeyError):
            pass
    pool = build_pool(run_dir, job, preprocessing)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.npz")
    arrays = {k: v for k, v in pool.items() if k != "meta"}
    np.savez_compressed(tmp, meta=np.array(json.dumps(pool["meta"])), **arrays)
    tmp.replace(path)
    return pool


def build_pool(run_dir: Path, job: TrainingJob, preprocessing: dict[str, Any]) -> dict[str, Any]:
    info = job.dataset
    if info.kind == "random" and job.family != "classical":
        raise ExplainError("This run was trained on random demo data, which is not kept.")
    seed = find_seed(run_dir, job.config)
    if seed is None:
        raise ExplainError(NO_DATA.format(reason="the random seed of the run is not known."))
    try:
        if job.family == "classical":
            return _classical_rows(job, seed, preprocessing)
        if info.kind == "csv":
            return _neural_rows(job, seed, preprocessing)
        if info.kind == "builtin":
            return _builtin_images(job, seed, preprocessing)
        if info.kind == "images":
            return _folder_images(job, seed, preprocessing)
    except (TrainingError, DatasetError, OSError) as exc:
        raise ExplainError(NO_DATA.format(reason=_sentence(str(exc)))) from None
    raise ExplainError(NO_DATA.format(reason=f"unknown dataset kind '{info.kind}'."))


def _sentence(text: str) -> str:
    text = text.strip() or "it cannot be read."
    return text if text.endswith(".") else text + "."


def _pick(indices: np.ndarray, count: int, seed: int) -> np.ndarray:
    """At most ``count`` of ``indices`` (random, but always the same for the same run)."""
    indices = np.asarray(indices)
    if len(indices) <= count:
        return indices
    rng = np.random.default_rng([seed, 7])
    return indices[np.sort(rng.choice(len(indices), size=count, replace=False))]


def _meta(source: str, **extra: Any) -> dict[str, Any]:
    return {"version": POOL_VERSION, "source": source, **extra}


def _held_out(train: Any, val: Any, test: Any) -> tuple[np.ndarray, str]:
    """The rows the samples come from and their name: the validation rows, else the test
    rows, else (a run that holds nothing out) the training rows."""
    if len(val):
        return np.asarray(val), "validation"
    if test is not None and len(test):
        return np.asarray(test), "test"
    return np.asarray(train), "training"


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #


def _torch_split(n: int, config: TrainingConfig, seed: int, test_split: float) -> Any:
    import torch

    from ..data import split_indices

    generator = torch.Generator().manual_seed(seed)
    return split_indices(n, config.val_split, config.max_samples, generator, test_split)


def _to_network_size(images: Any, preprocessing: dict[str, Any]) -> np.ndarray:
    """uint8 N×C×H×W of any size -> uint8 at the planned channels and size (as training
    converts them, but rounded back to whole pixel values)."""
    from ..data import ImageTransform

    image = {k: preprocessing[k] for k in ("channels", "height", "width")}
    image["layout"] = "chw"
    unit = ImageTransform(image, [0.0], [1.0]).to_unit(images)
    return (unit.clamp(0, 1) * 255).round().to(dtype=__import__("torch").uint8).numpy()


def _builtin_images(job: TrainingJob, seed: int, preprocessing: dict[str, Any]) -> dict:
    import torch

    from ..data import _as_nchw, open_torchvision

    spec = BUILTIN_DATASETS.get(job.dataset.builtin or "")
    if spec is None:
        raise ExplainError(NO_DATA.format(reason="the built-in dataset is unknown."))
    root = Path(job.data_root or DEFAULT_DATA_ROOT) / spec.id
    try:
        train_set = open_torchvision(spec.id, root, True, download=False)
    except TrainingError:
        raise
    except Exception:
        raise ExplainError(
            NO_DATA.format(reason=f"{spec.name} is not downloaded on this server.")
        ) from None
    # The validation rows come from the training split (see data.builtin_data); without
    # them the official test images (the test set) are used.
    _, val, _ = _torch_split(len(train_set.data), job.config, seed, 0.0)
    source = "validation"
    if len(val) == 0:
        try:
            train_set = open_torchvision(spec.id, root, False, download=False)
        except TrainingError:
            raise
        except Exception:
            raise ExplainError(
                NO_DATA.format(reason=f"{spec.name} is not downloaded on this server.")
            ) from None
        val, source = torch.arange(len(train_set.data)), "test"
    rows = _pick(val.numpy(), POOL_IMAGES, seed)
    images = _as_nchw(train_set.data)[torch.as_tensor(rows)]
    labels = torch.as_tensor(train_set.targets, dtype=torch.long)[torch.as_tensor(rows)]
    return {
        "images": _to_network_size(images, preprocessing),
        "labels": labels.numpy(),
        "rows": rows,
        "meta": _meta(source, dataset=spec.name),
    }


def _folder_images(job: TrainingJob, seed: int, preprocessing: dict[str, Any]) -> dict:
    import torch

    from ..data import load_image
    from ..datasets import list_image_files

    if job.path is None:
        raise ExplainError(NO_DATA.format(reason="the dataset has no files."))
    root = Path(job.path) / (job.dataset.subdir or "")
    if not root.is_dir():
        raise ExplainError(NO_DATA.format(reason="the image folder was deleted."))
    classes = list_image_files(root)
    if list(classes) != list(job.dataset.class_names or []):
        raise ExplainError(NO_DATA.format(reason="the image folder has changed."))
    files = [f for names in classes.values() for f in names]
    labels = [label for label, names in enumerate(classes.values()) for _ in names]
    train, val, test = _torch_split(len(files), job.config, seed, job.config.test_split)
    held, source = _held_out(train.numpy(), val.numpy(), test.numpy() if test is not None else None)
    rows = _pick(held, POOL_IMAGES, seed)
    image = {k: preprocessing[k] for k in ("channels", "height", "width")}
    loaded = torch.stack([load_image(files[i], image, root) for i in rows.tolist()])
    return {
        "images": loaded.numpy(),
        "labels": np.asarray([labels[i] for i in rows], dtype=np.int64),
        "rows": rows,
        "meta": _meta(source),
    }


# --------------------------------------------------------------------------- #
# Rows of a CSV file
# --------------------------------------------------------------------------- #


def _csv_path(job: TrainingJob) -> Path:
    if job.path is None:
        raise ExplainError(NO_DATA.format(reason="the dataset has no file."))
    path = Path(job.path)
    if not path.is_file():
        raise ExplainError(NO_DATA.format(reason="the CSV file was deleted."))
    if path.stat().st_size > MAX_CSV_BYTES:
        raise ExplainError(
            NO_DATA.format(
                reason=f"the CSV file is larger than {MAX_CSV_BYTES // 1024**2} MB, too large "
                "to read again for an explanation."
            )
        )
    return path


def _read_rows(
    path: Path, info: DatasetInfo, wanted: np.ndarray, target: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Features and targets of the rows ``wanted`` (sorted row numbers) of the CSV file.
    ``target(cell, number)`` turns the target cell into a class index or value."""
    position = {int(r): i for i, r in enumerate(wanted)}
    last = int(wanted.max()) if len(wanted) else -1
    x = np.zeros((len(wanted), info.num_features or 0), dtype=np.float64)
    y = np.zeros(len(wanted), dtype=np.float64)
    with csv_rows(path, info) as (_, _, rows):
        for index, (_, values, cell, number) in enumerate(rows):
            if index in position:
                x[position[index]] = values
                y[position[index]] = target(cell, number)
            if index >= last:
                break
    return x, y


def _class_target(names: list[str], mode: str) -> Any:
    """Class index of a target cell: text labels, whole-number indices or values."""
    index = {name: i for i, name in enumerate(names)}

    def convert(cell: str, number: float | None) -> float:
        if mode == "labels":
            if cell not in index:
                raise TrainingError(f"unknown class '{cell[:30]}'")
            return index[cell]
        if number is None or not float(number).is_integer():
            raise TrainingError(f"the target '{cell[:30]}' is not a whole number")
        if mode == "values":
            key = str(int(number))
            if key not in index:
                raise TrainingError(f"unknown class '{cell[:30]}'")
            return index[key]
        return int(number)

    return convert


def _value_target(cell: str, number: float | None) -> float:
    if number is None:
        raise TrainingError(f"the target '{cell[:30]}' is not a number")
    return float(number)


def _tabular_pool(
    path: Path,
    info: DatasetInfo,
    train: np.ndarray,
    val: np.ndarray,
    target: Any,
    seed: int,
    classification: bool,
    source: str = "validation",
) -> dict:
    """``val_x``/``val_y``: rows of ``val`` (the held-out rows the samples come from, see
    :func:`_held_out`), ``train_x``/``train_y``: training rows."""
    val_rows = np.sort(_pick(val, POOL_ROWS, seed))
    train_rows = np.sort(_pick(train, POOL_TRAIN_ROWS, seed + 1))
    both = np.union1d(val_rows, train_rows)
    x, y = _read_rows(path, info, both, target)
    where = {int(r): i for i, r in enumerate(both)}
    vi = np.array([where[int(r)] for r in val_rows], dtype=np.int64)
    ti = np.array([where[int(r)] for r in train_rows], dtype=np.int64)
    dtype = np.int64 if classification else np.float64
    return {
        "val_x": x[vi],
        "val_y": y[vi].astype(dtype),
        "train_x": x[ti],
        "train_y": y[ti].astype(dtype),
        "rows": val_rows,
        "meta": _meta(source, columns=csv_columns(info)),
    }


def _neural_rows(job: TrainingJob, seed: int, preprocessing: dict[str, Any]) -> dict:
    import torch

    from ..compat import check_compatibility
    from ..data import predefined_split_indices, read_split_file

    info = job.dataset
    path = _csv_path(job)
    compat = check_compatibility(job.graph, info)
    if compat.plan is None:
        raise ExplainError(NO_DATA.format(reason="the network no longer fits the dataset."))
    plan = compat.plan
    n = info.num_samples or 0
    generator = torch.Generator().manual_seed(seed)
    if info.predefined_split is not None:
        split = read_split_file(path, n, info.predefined_split)
        train, val, test = predefined_split_indices(split, job.config.max_samples, generator)
    else:
        train, val, test = _torch_split(n, job.config, seed, job.config.test_split)
    classification = plan.targets[0]["kind"] == "class"
    if classification:
        mode = "labels" if info.target_kind == "text" else "indices"
        target = _class_target(list(info.class_names or []), mode)
    else:
        target = _value_target
    held, source = _held_out(train.numpy(), val.numpy(), test.numpy() if test is not None else None)
    return _tabular_pool(path, info, train.numpy(), held, target, seed, classification, source)


def _classical_rows(job: TrainingJob, seed: int, preprocessing: dict[str, Any]) -> dict:
    from ...graph import analyze
    from ..classical_engine import _ClassicalRun
    from ..compat import check_compatibility

    info = job.dataset
    compat = check_compatibility(job.graph, info, job.config)
    if compat.plan is None or compat.plan.classical is None:
        raise ExplainError(NO_DATA.format(reason="the model no longer fits the dataset."))
    plan = compat.plan
    classification = plan.targets[0]["kind"] == "class"
    # The split functions (and the demo data) of the classical engine only need the
    # settings, the seed and the model.
    shim = _SplitShim(job.config, seed)
    if info.kind == "random":
        model = analyze(job.graph).model
        assert model is not None
        shim.layer = model.layer
        shim.classification = classification
        shim.num_classes = plan.num_classes or 0
        x, y = _ClassicalRun._demo(shim, plan.inputs[0]["shape"][0])
        train, val, test = _ClassicalRun._random_split(shim, len(x), job.config.max_samples)
        limit = int(plan.classical["max_rows"])
        if len(train) > limit:
            train, val, test = _ClassicalRun._random_split(shim, len(x), limit)
        held, source = _held_out(train, val, test)
        val_rows = np.sort(_pick(held, POOL_ROWS, seed))
        train_rows = np.sort(_pick(train, POOL_TRAIN_ROWS, seed + 1))
        n = x.shape[1]
        return {
            "val_x": x[val_rows],
            "val_y": y[val_rows],
            "train_x": x[train_rows],
            "train_y": y[train_rows],
            "rows": val_rows,
            "meta": _meta(source, columns=[*(f"x{i + 1}" for i in range(n)), "y"]),
        }
    path = _csv_path(job)
    n = info.num_samples or 0
    if info.predefined_split is not None:
        stored = np.frombuffer(read_split(path, n, info.predefined_split), dtype=np.uint8)

        def split(max_samples: int | None) -> Any:
            return _ClassicalRun._predefined_split(shim, stored, max_samples)
    else:

        def split(max_samples: int | None) -> Any:
            return _ClassicalRun._random_split(shim, n, max_samples)

    train, val, test = split(job.config.max_samples)
    limit = int(plan.classical["max_rows"])
    if len(train) > limit:
        train, val, test = split(limit)
    held, source = _held_out(train, val, test)
    if classification:
        mode = str(plan.classical.get("class_targets") or "indices")
        target = _class_target(list(plan.class_names or []), mode)
    else:
        target = _value_target
    return _tabular_pool(path, info, np.asarray(train), held, target, seed, classification, source)


class _SplitShim:
    """What ``_ClassicalRun._random_split`` and ``_predefined_split`` use of the run."""

    def __init__(self, config: TrainingConfig, seed: int):
        self.config = config
        self.seed = seed
        self.layer: Any = None
        self.classification = False
        self.num_classes = 0

    def _rng(self, purpose: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, purpose])
