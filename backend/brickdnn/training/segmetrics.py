"""Loss and metrics of a segmentation training (needs PyTorch).

The network outputs one score per class for every pixel, (N, K, H, W), or for two classes
one channel, (N, 1, H, W): the score (logit) of class 1. The masks are class maps (N, H, W);
pixels with the value IGNORE_INDEX count neither for the loss nor for the metrics.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .segmentation import IGNORE_INDEX

# Added to both sides of the Dice ratio: a class without pixels in a batch counts as matched
# when nothing of it is predicted.
DICE_SMOOTH = 1.0


class SegmentationLoss(nn.Module):
    """The cross-entropy of every pixel (two classes with one output channel: the binary
    cross-entropy on logits), averaged over the pixels that are not ignored. With ``dice``,
    the soft Dice loss is added: one minus the Dice score of the predicted probabilities,
    averaged over the classes (of the whole batch)."""

    def __init__(self, num_classes: int, binary: bool, dice: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.binary = binary
        self.dice = dice

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != IGNORE_INDEX
        if self.binary:
            logits = output[:, 0]
            labels = (target == 1).to(logits.dtype)
            weights = valid.to(logits.dtype)
            pixels = weights.sum().clamp(min=1.0)
            loss = (
                F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * weights
            ).sum() / pixels
            if self.dice:
                probability = torch.sigmoid(logits) * weights
                labels = labels * weights
                overlap = (probability * labels).sum()
                loss = (
                    loss
                    + 1
                    - (2 * overlap + DICE_SMOOTH) / (probability.sum() + labels.sum() + DICE_SMOOTH)
                )
            return loss
        if not bool(valid.any()):
            return output.sum() * 0.0  # a batch of ignored pixels only: nothing to learn
        loss = F.cross_entropy(output, target, ignore_index=IGNORE_INDEX)
        if self.dice:
            weights = valid.unsqueeze(1).to(output.dtype)
            probability = output.softmax(dim=1) * weights
            classes = output.shape[1]
            labels = F.one_hot(target.clamp(max=classes - 1), classes).permute(0, 3, 1, 2)
            labels = labels.to(output.dtype) * weights
            overlap = (probability * labels).sum(dim=(0, 2, 3))
            sizes = probability.sum(dim=(0, 2, 3)) + labels.sum(dim=(0, 2, 3))
            loss = loss + 1 - ((2 * overlap + DICE_SMOOTH) / (sizes + DICE_SMOOTH)).mean()
        return loss


def predicted_classes(output: torch.Tensor, binary: bool) -> torch.Tensor:
    """The class map (N, H, W) the network predicts: the class with the highest score, or
    for one output channel class 1 where its score is positive (probability above 0.5)."""
    return (output[:, 0] > 0).long() if binary else output.argmax(dim=1)


def confusion(truth: torch.Tensor, predicted: torch.Tensor, num_classes: int) -> torch.Tensor:
    """The confusion matrix (num_classes × num_classes, rows: true class, columns: predicted
    class) of class maps of any shape. Pixels whose true value is no class (IGNORE_INDEX) are
    left out."""
    truth = truth.reshape(-1).long()
    predicted = predicted.reshape(-1).long()
    valid = truth < num_classes
    pairs = truth[valid] * num_classes + predicted[valid].clamp(0, num_classes - 1)
    return torch.bincount(pairs, minlength=num_classes * num_classes).view(num_classes, num_classes)


def confusion_metrics(matrix: torch.Tensor) -> dict[str, Any]:
    """IoU, Dice and pixel accuracy from a confusion matrix (rows: true class).

    Per class, IoU is hits / (true pixels + predicted pixels - hits) and Dice is 2 · hits /
    (true pixels + predicted pixels). ``class_iou`` and ``class_dice`` have one entry per
    class, None for a class that occurs neither in the masks nor in the predictions; ``iou``
    and ``dice`` are the means over the other classes; ``pixel_accuracy`` is the share of
    pixels with the right class. Everything is None without pixels.
    """
    matrix = matrix.double()
    hits = matrix.diagonal()
    true_pixels, predicted_pixels = matrix.sum(dim=1), matrix.sum(dim=0)
    union = true_pixels + predicted_pixels - hits
    present = union > 0
    total = float(matrix.sum())
    if total == 0:
        empty: list[float | None] = [None] * len(hits)
        return {
            "iou": None,
            "dice": None,
            "pixel_accuracy": None,
            "class_iou": empty,
            "class_dice": list(empty),
        }
    iou = hits / union.clamp(min=1)
    dice = 2 * hits / (true_pixels + predicted_pixels).clamp(min=1)
    return {
        "iou": float(iou[present].mean()),
        "dice": float(dice[present].mean()),
        "pixel_accuracy": float(hits.sum()) / total,
        "class_iou": [float(v) if p else None for v, p in zip(iou, present, strict=True)],
        "class_dice": [float(v) if p else None for v, p in zip(dice, present, strict=True)],
    }


class SegmentationMetrics:
    """Adds up the confusion matrix of a split batch by batch, so IoU and Dice are those of
    all its pixels together (not the mean over batches or images)."""

    def __init__(self, num_classes: int, binary: bool):
        self.num_classes = num_classes
        self.binary = binary
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)

    def add(self, output: torch.Tensor, target: torch.Tensor) -> None:
        predicted = predicted_classes(output.detach(), self.binary).cpu()
        self.matrix += confusion(target.cpu(), predicted, self.num_classes)

    def result(self) -> dict[str, Any]:
        return confusion_metrics(self.matrix)
