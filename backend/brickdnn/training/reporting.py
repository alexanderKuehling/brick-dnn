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


class DivergedError(TrainingError):
    """The loss became NaN or infinite (the settings, e.g. the learning rate, are to blame)."""


class StopTraining(Exception):
    """Raised inside the engine when the user asked to stop."""


class Reporter:
    """Receives progress from the engine. Every method is optional to override.

    Call order: ``status``/``log`` while preparing, ``setup`` once data and model are
    ready, then per epoch ``status`` + ``progress`` (about twice per second) +
    ``epoch_end``, ``test_end`` if there is a test set, and exactly one ``finish`` at the
    end. A hyperparameter search wraps the epochs of every trial in ``trial_start`` and
    ``trial_end`` and calls ``search_end`` before testing the best trial.
    """

    def status(self, status: str, message: str) -> None:
        """Phase (``preparing`` or ``running``) and current activity, e.g. "Epoch 3 of 10"."""

    def log(self, line: str) -> None:
        """A line for the run log, e.g. "Downloading MNIST (about 12 MB) …"."""

    def setup(self, info: dict[str, Any]) -> None:
        """Facts about the run (state.json keys): device, device_label, task, num_classes,
        class_names, train_size, val_size, test_size, steps_per_epoch, total_params,
        preprocessing; for a hyperparameter search also tuning and trials_total.
        May be called again with a subset (e.g. after falling back to the CPU)."""

    def progress(self, progress: dict[str, Any]) -> None:
        """``{"epoch", "step", "steps", "loss", "samples_per_second"}`` during an epoch."""

    def epoch_end(self, entry: dict[str, Any], best: dict[str, Any]) -> None:
        """One history entry (see state.py) and the best epoch so far (of this trial)."""

    def trial_start(self, trial: dict[str, Any]) -> None:
        """Hyperparameter search: ``{"trial", "trials", "params", "steps_per_epoch"}``
        before the epochs of a trial (``trial`` counts from 1)."""

    def trial_end(self, trial: dict[str, Any], best_trial: int | None) -> None:
        """Hyperparameter search: the result of a trial (see state.py, ``trials``) and the
        number of the best trial so far."""

    def search_end(self, result: dict[str, Any]) -> None:
        """Hyperparameter search: the winner ``{"best_trial", "params", "history", "best"}``.
        Its weights are the ones that are tested and saved."""

    def test_end(self, result: dict[str, Any]) -> None:
        """``{"loss", "accuracy", "rmse"}`` of the final model on the test set."""

    def finish(self, status: str, message: str, weights: bool) -> None:
        """Terminal status (completed | stopped | failed) and the result for the user."""

    def should_stop(self) -> bool:
        """Checked between batches; return True to stop and keep the weights."""
        return False
