"""The loss functions of neural networks (needs PyTorch); which loss a run uses and its
settings are in losses.py.

Class weights are kept on the CPU and moved to the device of the outputs when the loss is
computed, so a loss works wherever the model is (also while the Apple GPU is checked
against the CPU, see engine._Run._try_mps).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import class_weight_vector, class_weights_of, loss_for, uses
from .segmetrics import SegmentationLoss


class CrossEntropy(nn.Module):
    """The cross-entropy with optional class weights and label smoothing
    (``torch.nn.CrossEntropyLoss(weight=…, label_smoothing=…)``). With weights the mean is
    weighted: the sum of the weighted losses divided by the sum of the weights of the
    targets of the batch."""

    def __init__(self, weight: torch.Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        self.weight = weight
        self.label_smoothing = float(label_smoothing)

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(output.device, output.dtype) if self.weight is not None else None
        return F.cross_entropy(output, target, weight=weight, label_smoothing=self.label_smoothing)


class FocalLoss(nn.Module):
    """The focal loss (Lin et al., 2017) of class scores: the mean over the samples of
    ``-alpha_t · (1 - p_t)^gamma · log(p_t)``, where ``p_t`` is the predicted probability of
    the true class and ``alpha_t`` its class weight (1 without weights). Well-classified
    samples (``p_t`` near 1) count less, so training concentrates on the hard and rare ones;
    ``gamma`` 0 is the cross-entropy."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(output, dim=1).gather(1, target.long().unsqueeze(1)).squeeze(1)
        p = log_p.exp()
        loss = -((1.0 - p).clamp(min=0.0) ** self.gamma) * log_p
        if self.weight is not None:
            loss = loss * self.weight.to(output.device, output.dtype)[target.long()]
        return loss.mean()


def build_loss(config: Any, target: dict[str, Any], weights: list[float] | None) -> nn.Module:
    """The loss function of one output (``target`` of the plan) with the settings
    ``config``; ``weights``: its class weights (see :func:`class_weights_for`)."""
    kind = target["kind"]
    name = getattr(config, "loss", "auto") or "auto"
    weight = torch.tensor(weights, dtype=torch.float32) if weights is not None else None
    if kind == "mask":
        name = loss_for(name, "segmentation")
        return SegmentationLoss(
            int(target["num_classes"]),
            bool(target.get("binary")),
            dice=name == "cross_entropy_dice",
            weight=weight if name == "weighted_cross_entropy" else None,
        )
    if kind == "class":
        name = loss_for(name, "classification")
        smoothing = float(config.label_smoothing) if uses("label_smoothing", name) else 0.0
        if name == "focal":
            return FocalLoss(config.focal_gamma, weight)
        if name == "cross_entropy" and smoothing == 0.0:
            return nn.CrossEntropyLoss()
        return CrossEntropy(weight if name == "weighted_cross_entropy" else None, smoothing)
    name = loss_for(name, "regression")
    if name == "mae":
        return nn.L1Loss()
    if name == "huber":
        return nn.HuberLoss(delta=float(config.huber_delta))
    return nn.MSELoss()


def needs_counts(config: Any, target: dict[str, Any]) -> bool:
    """True if the loss of ``target`` needs the class counts of the training data (balanced
    class weights)."""
    task = {"class": "classification", "mask": "segmentation"}.get(target["kind"])
    if task is None:
        return False
    return class_weights_of(config, loss_for(config.loss or "auto", task)) == "balanced"


def class_weights_for(
    config: Any,
    target: dict[str, Any],
    class_names: list[str] | None,
    counts: list[float] | None,
) -> list[float] | None:
    """The class weights the loss of ``target`` uses with ``config`` (None: none);
    ``counts``: how often every class occurs in the training data (needed for
    "balanced")."""
    task = {"class": "classification", "mask": "segmentation"}.get(target["kind"])
    if task is None:
        return None
    given = class_weights_of(config, loss_for(config.loss or "auto", task))
    return class_weight_vector(given, int(target["num_classes"]), class_names, counts)
