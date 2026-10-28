"""How the training engine talks to whoever runs it (no PyTorch needed).

The engine never prints or writes files itself. It calls a :class:`Reporter`, so the same
training code drives the web runner (which writes ``state.json``) and the command line
(which prints to the console).
"""

from __future__ import annotations

from typing import Any


class TrainingError(Exception):
    """Ends a run as ``failed``. The message is shown to the user."""


def short_error(exc: BaseException) -> str:
    """``"ValueError: first line of the message"`` (at most about 300 characters)."""
    lines = str(exc).strip().splitlines()
    detail = lines[0][:300] if lines else ""
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


class StopTraining(Exception):
    """Raised inside the engine when the user asked to stop."""


class Reporter:
    """Receives progress from the engine. Every method is optional to override.

    Call order: ``status``/``log`` while preparing, ``setup`` once data and model are
    ready, then per epoch ``status`` + ``progress`` (about twice per second) +
    ``epoch_end``, and exactly one ``finish`` at the end.
    """

    def status(self, status: str, message: str) -> None:
        """Phase (``preparing`` or ``running``) and current activity, e.g. "Epoch 3 of 10"."""

    def log(self, line: str) -> None:
        """A line for the run log, e.g. "Downloading MNIST (about 12 MB) …"."""

    def setup(self, info: dict[str, Any]) -> None:
        """Facts about the run (state.json keys): device, device_label, task, num_classes,
        class_names, train_size, val_size, steps_per_epoch, total_params, preprocessing.
        May be called again with a subset (e.g. after falling back to the CPU)."""

    def progress(self, progress: dict[str, Any]) -> None:
        """``{"epoch", "step", "steps", "loss", "samples_per_second"}`` during an epoch."""

    def epoch_end(self, entry: dict[str, Any], best: dict[str, Any]) -> None:
        """One history entry (see state.py) and the best epoch so far."""

    def finish(self, status: str, message: str, weights: bool) -> None:
        """Terminal status (completed | stopped | failed) and the result for the user."""

    def should_stop(self) -> bool:
        """Checked between batches; return True to stop and keep the weights."""
        return False
