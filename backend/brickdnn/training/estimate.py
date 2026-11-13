"""A rough estimate of the CPU time a training will use (no PyTorch needed).

Budgets (see brick_dnn_core.budgets) compare it with what is left before a run starts. It is
an estimate, not a measurement: the real CPU time depends on the machine, the number of CPU
threads, the data and on early stopping, and is often off by a factor of two or three
(``low`` and ``high`` give the range). What is counted afterwards is always the CPU time the
training really used.

Neural networks: the arithmetic of one forward pass per sample is estimated from the layers
(multiply-adds of a layer ≈ its weights × the positions they are applied to, e.g. the pixels
of a convolution's output; layers without weights cost one operation per output value), a
training step costs about three forward passes (forward and backward), validation and test
one. Added are a fixed cost per optimizer step, the preparation of the samples per epoch
and the start of the process (Python, PyTorch, the data).

Classical models are fitted once: a Gaussian process needs about n³ operations per
optimizer step for n training rows (a classifier once per class), a support vector machine
between n² and n³. Gradient boosting builds one tree per iteration (per class with more than
two classes); a tree costs about n × features histogram updates plus a fixed overhead.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from ..graph import Analysis
from ..layers import ClassicalModel
from .config import TrainingConfig, classical_max_rows
from .datasets import DatasetInfo
from .tabular import held_out_size, split_sizes

# Calibrated with the examples on a 4-core Xeon server with PyTorch 2 and scikit-learn 1:
# the estimates were 0.4 to 2.2 times the measured CPU time (Gaussian processes vary most).
# Arithmetic of a training per CPU second (all threads count, so more threads do not help).
FLOPS_PER_CPU_SECOND = 25e9
# CPU time of one optimizer step besides its arithmetic (Python, autograd, the optimizer).
STEP_CPU_SECONDS = 0.002
# CPU time per layer and optimizer step (each layer is a few kernel launches).
LAYER_STEP_CPU_SECONDS = 0.0003
# Preparing one sample per epoch (decoding, normalising, copying into a batch).
SAMPLE_CPU_SECONDS = {
    "builtin": 4e-6,
    "images": 2e-4,
    "segmentation": 4e-4,
    "csv": 1e-6,
    "random": 1e-6,
}
# Starting the process, importing PyTorch, loading the data and building the model.
STARTUP_CPU_SECONDS = 3.0
# Built-in datasets are read once (and downloaded the first time).
BUILTIN_LOAD_CPU_SECONDS = 2.0
# Classical models: arithmetic per CPU second (NumPy, LAPACK) and optimizer steps of a
# Gaussian process (L-BFGS, per start).
CLASSICAL_FLOPS_PER_CPU_SECOND = 3e9
GP_OPTIMIZER_STEPS = 80
CLASSICAL_STARTUP_CPU_SECONDS = 2.0
# The estimate is typically within this factor of the real CPU time.
UNCERTAINTY = 3.0
# Demo data: samples in total (see data.py, which also limits them to DEMO_BYTES).
DEMO_SAMPLES = 2560
DEMO_BYTES = 500 * 1024**2
CLASSICAL_DEMO_ROWS = {"gaussian_process": 500, "svm": 1000, "gradient_boosting": 2000}
# Gradient boosting (measured, all threads together): CPU seconds per tree, per row and
# feature of a tree (histograms and splits), and per row and tree of a prediction.
TREE_CPU_SECONDS = 0.006
TREE_ROW_FEATURE_CPU_SECONDS = 8e-9
TREE_PREDICT_CPU_SECONDS = 5e-8
# Built-in datasets: their official test split is shrunk with max_samples as if it were
# this share of all samples (see data.py).
BUILTIN_TEST_SHARE = 0.2


@dataclass
class Estimate:
    """Estimated CPU seconds of a training (``low`` to ``high``) and what it is based on."""

    cpu_seconds: float
    low: float
    high: float
    basis: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("cpu_seconds", "low", "high"):
            data[key] = round(data[key], 1)
        return data


def _prod(values: Any) -> int:
    return math.prod(int(v) for v in values) if values else 1


def _layer_macs(analysis: Analysis, node_id: str) -> float:
    """Multiply-adds of one layer for one sample (rough)."""
    info = analysis.nodes[node_id]
    shape = tuple(info.shape or ())
    params = info.num_params or 0
    numel = _prod(shape) if shape else 0
    source = analysis.nodes.get(info.inputs[0]) if info.inputs else None
    in_shape = tuple(source.shape or ()) if source is not None else ()
    kind = info.type
    if kind == "Input":
        return 0.0
    if kind in ("Embedding", "PositionalEmbedding"):
        return float(numel)  # lookups, not multiplications
    if kind == "Linear":
        return float(params * _prod(shape[:-1]))
    if kind.startswith("Conv"):
        return float(params * _prod(shape[1:]))
    if kind in ("RNN", "LSTM", "GRU"):
        steps = in_shape[0] if len(in_shape) >= 2 else 1
        return float(params * steps)
    if kind == "TransformerEncoder":
        steps = in_shape[0] if len(in_shape) >= 2 else 1
        width = in_shape[-1] if in_shape else 1
        layers = int(info.layer.p.get("num_layers", 1)) if info.layer is not None else 1
        return float(params * steps + 2 * layers * steps * steps * width)
    return float(max(params, numel))


def forward_flops(analysis: Analysis) -> float:
    """Floating point operations of one forward pass for one sample (rough)."""
    return 2.0 * sum(_layer_macs(analysis, node_id) for node_id in analysis.nodes)


def _demo_sample_bytes(plan: dict[str, Any]) -> int:
    size = 0
    for spec in plan.get("inputs") or []:
        size += _prod(spec.get("shape") or []) * (8 if spec.get("dtype") == "long" else 4)
    for target in plan.get("targets") or []:
        if target.get("kind") == "class":
            size += 8
        else:
            size += 4 * _prod(target.get("shape") or [])
    return max(size, 1)


def sample_counts(
    info: DatasetInfo, config: TrainingConfig, plan: dict[str, Any] | None = None
) -> tuple[int, int, int]:
    """About how many training, validation and test samples a run uses."""
    val_split, test_split, cap = config.val_split, config.test_split, config.max_samples
    train_share = max(1e-6, 1 - val_split - test_split)
    if info.kind == "builtin":
        total_train = int(info.train_size or info.num_samples or 0)
        n_val = round(total_train * val_split)
        n_train, n_test = total_train - n_val, int(info.test_size or 0)
        if cap is not None and cap < n_train:
            n_train = cap
            n_val = min(n_val, held_out_size(cap, val_split, 1 - val_split))
            n_test = min(n_test, held_out_size(cap, BUILTIN_TEST_SHARE, 1 - BUILTIN_TEST_SHARE))
        return n_train, n_val, n_test
    if info.kind == "random":
        total = min(DEMO_SAMPLES, max(3, DEMO_BYTES // _demo_sample_bytes(plan or {})))
    elif info.predefined_split:
        n_train, n_val, n_test = (int(n) for n in info.predefined_split)
        if cap is not None and cap < n_train:
            n_val = min(n_val, held_out_size(cap, n_val, n_train))
            n_test = min(n_test, held_out_size(cap, n_test, n_train))
            n_train = cap
        return n_train, n_val, n_test
    else:
        total = int(info.num_samples or 0)
    try:
        n_train, n_val, n_test = split_sizes(total, val_split, test_split)
    except Exception:
        return 0, 0, 0
    if cap is not None and cap < n_train:
        n_train = cap
        n_val = min(n_val, held_out_size(cap, val_split, train_share))
        n_test = min(n_test, held_out_size(cap, test_split, train_share)) if n_test else 0
    return n_train, n_val, n_test


def estimate_training(
    analysis: Analysis,
    info: DatasetInfo,
    config: TrainingConfig | None,
    plan: dict[str, Any] | None,
    trials: int = 1,
) -> Estimate | None:
    """The estimated CPU time of training ``analysis`` on ``info``; None if the network is
    not valid. ``trials``: the trials of a hyperparameter search (1 for a normal run)."""
    if not analysis.valid:
        return None
    config = config or TrainingConfig()
    n_train, n_val, n_test = sample_counts(info, config, plan)
    if analysis.family == "classical":
        return _classical(analysis, info, config, plan, n_train, n_val + n_test)

    flops = forward_flops(analysis)
    layers = sum(1 for info_ in analysis.nodes.values() if info_.type != "Input")
    batch = max(1, config.batch_size)
    steps = math.ceil(n_train / batch) if n_train else 0
    epochs = config.epochs * max(1, trials)
    per_sample = SAMPLE_CPU_SECONDS.get(info.kind, 1e-6)
    arithmetic = (3 * n_train + n_val) * flops * epochs + n_test * flops
    step_overhead = steps * epochs * (STEP_CPU_SECONDS + layers * LAYER_STEP_CPU_SECONDS)
    data = (n_train + n_val) * epochs * per_sample
    startup = STARTUP_CPU_SECONDS + (BUILTIN_LOAD_CPU_SECONDS if info.kind == "builtin" else 0)
    seconds = startup + arithmetic / FLOPS_PER_CPU_SECOND + step_overhead + data
    runs = f"{config.epochs} epochs" + (f" × {trials} trials" if trials > 1 else "")
    basis = (
        f"{n_train:,} training and {n_val + n_test:,} validation/test samples, {runs}, "
        f"about {_amount(flops)} operations per sample and pass"
    )
    details = {
        "family": "neural",
        "train_samples": n_train,
        "val_samples": n_val,
        "test_samples": n_test,
        "epochs": config.epochs,
        "trials": max(1, trials),
        "steps_per_epoch": steps,
        "forward_flops_per_sample": round(flops),
        "total_params": analysis.total_params,
    }
    return _estimate(seconds, basis, details)


def _classical(
    analysis: Analysis,
    info: DatasetInfo,
    config: TrainingConfig,
    plan: dict[str, Any] | None,
    n_train: int,
    n_held_out: int,
) -> Estimate | None:
    node = next(
        (n for n in analysis.nodes.values() if isinstance(n.layer, ClassicalModel)),
        None,
    )
    if node is None or node.layer is None:
        return None
    layer: ClassicalModel = node.layer  # type: ignore[assignment]
    max_rows = classical_max_rows(layer.limit)
    if info.kind == "random":
        n_train = CLASSICAL_DEMO_ROWS.get(layer.scale, 1000)
    n = max(1, min(n_train, max_rows))
    features = max(1, _prod(analysis.nodes[analysis.inputs[0]].shape or (1,)))
    classes = int((plan or {}).get("num_classes") or 1)
    if layer.scale == "gradient_boosting":
        # At most max_iter iterations: early stopping often ends sooner.
        trees = int(layer.p["max_iter"]) * (classes if classes > 2 else 1)
        fit = trees * (TREE_CPU_SECONDS + TREE_ROW_FEATURE_CPU_SECONDS * n * features)
        predict = trees * (n_held_out + n) * TREE_PREDICT_CPU_SECONDS
        seconds = CLASSICAL_STARTUP_CPU_SECONDS + fit + predict
        basis = (
            f"gradient boosting with up to {trees:,} trees fitted on {n:,} training rows with "
            f"{features:,} features"
        )
        details = {
            "family": "classical",
            "train_rows": n,
            "features": features,
            "max_rows": max_rows,
            "trees": trees,
        }
        return _estimate(seconds, basis, details)
    if layer.scale == "gaussian_process":
        restarts = int(layer.p.get("n_restarts_optimizer") or 0)
        per_class = classes if classes > 2 else 1
        steps = GP_OPTIMIZER_STEPS * (1 + restarts) * per_class
        # Cholesky (n³/3) plus the kernel and its gradient (n² × features) per step.
        flops = steps * (n**3 / 3 + 10 * n * n * features)
        if layer.type == "GaussianProcessClassifier":
            flops *= 3  # the Laplace approximation iterates inside every step
        model = "Gaussian process"
    else:
        # Kernel evaluations: about n² (up to n³ for hard problems); SVC also calibrates its
        # probabilities with 5 folds.
        flops = 2.0 * n * n * features
        if layer.type == "SVC":
            flops *= 5
        model = "support vector machine"
    flops += (n_held_out + n) * n * features * 2  # predictions on all parts
    seconds = CLASSICAL_STARTUP_CPU_SECONDS + flops / CLASSICAL_FLOPS_PER_CPU_SECOND
    basis = f"{model} fitted on {n:,} training rows with {features:,} features"
    details = {"family": "classical", "train_rows": n, "features": features, "max_rows": max_rows}
    return _estimate(seconds, basis, details)


def _estimate(seconds: float, basis: str, details: dict[str, Any]) -> Estimate:
    seconds = max(1.0, seconds)
    return Estimate(
        cpu_seconds=seconds,
        low=seconds / UNCERTAINTY,
        high=seconds * UNCERTAINTY,
        basis=basis,
        details=details,
    )


def _amount(flops: float) -> str:
    for unit, size in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if flops >= size:
            return f"{flops / size:.3g}{unit}"
    return f"{flops:.0f}"
