"""Kernel hyperparameters that a Gaussian process fit left at one of their bounds.

scikit-learn only warns (``ConvergenceWarning: The optimal value found for dimension 0 of
parameter k2__noise_level is close to the specified lower bound 1e-05``) when the optimiser
stops at a bound. That usually means something: a noise level at its lower bound lets the
model run through every training point (overfitting), a length scale at its upper bound means
the kernel hardly varies. :func:`at_bounds` finds these cases after the fit and names the brick,
the parameter and a change of its bounds that the web app can apply.

The fitted kernel and the kernel built from the bricks (:class:`~brickdnn.layers.KernelExpr`)
have their leaves in the same order: the kernel is built from ``KernelExpr.code()``, and an
in-order walk of scikit-learn's sums and products (``k1`` before ``k2``) visits the calls in the
order in which the code writes them.
"""

from __future__ import annotations

import contextlib
import math
from typing import Any

import numpy as np

from ..graph import Analysis
from ..layers import DEFAULT_BOUNDS, KernelExpr

# Closer than this in log space counts as "at the bound" (scikit-learn's own warning uses
# np.isclose on the log values, which is stricter).
LOG_TOLERANCE = 1e-3
# The suggested new bound: two orders of magnitude further in (lower) or out (upper).
SUGGEST_FACTOR = 100.0


def _leaves(kernel: Any) -> list[Any]:
    """The kernel calls of a fitted scikit-learn kernel, left to right."""
    if hasattr(kernel, "k1") and hasattr(kernel, "k2"):
        return _leaves(kernel.k1) + _leaves(kernel.k2)
    return [kernel]


def fitted_kernels(
    model: Any, class_names: list[str] | None = None
) -> list[tuple[str | None, Any]]:
    """``(class name or None, fitted kernel)``: one, or one per class when a multi-class
    Gaussian process classifier fitted one against the rest."""
    per_class = getattr(getattr(model, "base_estimator_", None), "estimators_", None)
    if not per_class:
        kernel = getattr(model, "kernel_", None)
        return [(None, kernel)] if kernel is not None else []
    names = class_names or []
    found = []
    for value, estimator in zip(model.classes_, per_class, strict=False):
        # The engine trains on class indices; other callers may use the labels themselves.
        index = int(value) if isinstance(value, (int, np.integer)) else None
        name = names[index] if index is not None and index < len(names) else value
        found.append((str(name), estimator.kernel_))
    return found


def _brick_names(analysis: Analysis) -> dict[int, tuple[str, str, Any]]:
    """For each leaf kernel of a kernel brick (by identity): its node id, display name and
    layer. The ClassicalModel's kernel is built from these very objects."""
    found = {}
    for nid, info in analysis.nodes.items():
        if info.kernel is not None and not info.kernel.op and info.layer is not None:
            found[id(info.kernel)] = (nid, info.name or nid, info.layer)
    return found


def _fmt(value: float) -> str:
    return f"{value:.3g}"


def _suggestion(layer: Any, parameter: str, side: str, bound: float) -> dict[str, Any]:
    """New parameters for the brick: custom bounds that move the bound that was hit."""
    params = getattr(layer, "p", {})
    custom = params.get("hyperparameters") == "custom bounds"
    low = float(params.get(f"{parameter}_min", DEFAULT_BOUNDS[0])) if custom else DEFAULT_BOUNDS[0]
    high = float(params.get(f"{parameter}_max", DEFAULT_BOUNDS[1])) if custom else DEFAULT_BOUNDS[1]
    if side == "lower":
        low = float(_fmt(bound * SUGGEST_FACTOR))
        high = max(high, low * 10)
    else:
        high = float(_fmt(bound * SUGGEST_FACTOR))
    value = float(params.get(parameter, 1.0))
    value = min(max(value, low), high)
    return {
        "hyperparameters": "custom bounds",
        f"{parameter}_min": low,
        f"{parameter}_max": high,
        parameter: value,
    }


def _message(label: str, brick: str, side: str, bound: float, suggestion: dict | None) -> str:
    text = f"{label} of {brick} ended at its {side} bound {_fmt(bound)}"
    if side == "lower":
        text += ": the model may be overfitting"
        if suggestion:
            name = next(k for k in suggestion if k.endswith("_min"))
            text += f"; set custom bounds, e.g. ≥ {_fmt(suggestion[name])}"
    else:
        text += ": the fit wanted a larger value"
        if suggestion:
            name = next(k for k in suggestion if k.endswith("_max"))
            text += (
                f"; if that makes sense, raise the upper bound (custom bounds, e.g. "
                f"≤ {_fmt(suggestion[name])}), otherwise this kernel may not be needed"
            )
    return text + "."


def at_bounds(
    model: Any, expr: KernelExpr | None, analysis: Analysis, class_names: list[str] | None = None
) -> list[dict[str, Any]]:
    """The optimised hyperparameters that ended at (or very near) a bound, one entry per
    brick, parameter and side::

        {"node": "white_1", "name": "white_1", "type": "WhiteKernel",
         "parameter": "noise_level", "label": "Noise level", "value": 1e-05,
         "bound": 1e-05, "side": "lower", "classes": None,
         "suggestion": {"hyperparameters": "custom bounds", "noise_level_min": 0.001, ...},
         "message": "Noise level of white_1 ended at its lower bound 1e-05: ..."}

    ``node`` and ``suggestion`` are None for the default kernel (no kernel bricks)."""
    if expr is None or getattr(model, "optimizer", "fmin_l_bfgs_b") is None:
        return []
    leaves = expr.leaves()
    bricks = _brick_names(analysis)
    found: dict[tuple[str | None, str, str], dict[str, Any]] = {}
    for class_name, kernel in fitted_kernels(model, class_names):
        fitted = _leaves(kernel)
        if len(fitted) != len(leaves):
            continue  # not the kernel the bricks describe (should not happen)
        for leaf, sk_kernel in zip(leaves, fitted, strict=True):
            node, name, layer = bricks.get(id(leaf), (None, "the default kernel", None))
            for hyper in sk_kernel.hyperparameters:
                if hyper.fixed:
                    continue
                value = float(getattr(sk_kernel, hyper.name))
                low, high = (float(b) for b in hyper.bounds[0])
                side = None
                for candidate, bound in (("lower", low), ("upper", high)):
                    if value > 0 and bound > 0:
                        distance = abs(math.log(value) - math.log(bound))
                        if distance <= LOG_TOLERANCE:
                            side = candidate
                if side is None:
                    continue
                key = (node, hyper.name, side)
                if key in found:
                    if class_name is not None:
                        found[key]["classes"].append(class_name)
                    continue
                bound = low if side == "lower" else high
                label = hyper.name.replace("_", " ").capitalize()
                if layer is not None:
                    with contextlib.suppress(StopIteration, AttributeError):
                        label = layer.param(hyper.name).label
                suggestion = _suggestion(layer, hyper.name, side, bound) if layer else None
                found[key] = {
                    "node": node,
                    "name": name,
                    "type": getattr(layer, "type", sk_kernel.__class__.__name__),
                    "parameter": hyper.name,
                    "label": label,
                    "value": value,
                    "bound": bound,
                    "side": side,
                    "classes": [class_name] if class_name is not None else None,
                    "suggestion": suggestion,
                    "message": _message(label, name, side, bound, suggestion),
                }
    return list(found.values())
