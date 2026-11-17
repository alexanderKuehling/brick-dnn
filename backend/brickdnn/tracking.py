"""Log a training that runs elsewhere (your laptop, a GPU server, Colab) into Brick DNN.

Its curves then appear on the runs page of Brick DNN next to the runs trained there (see
docs/experiments.md). Only the standard library is needed::

    from brickdnn.tracking import BrickDNNRun

    with BrickDNNRun.start(name="MLP on houses", config={"lr": 1e-3, "epochs": 20}) as run:
        for epoch in range(1, 21):
            ...  # train
            run.log(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
        run.finish(summary={"test_rmse": rmse})

Logging is **off** unless the environment variable ``BRICKDNN_TOKEN`` holds a personal API
token (create one on the page *API tokens* of Brick DNN; a token that may only log runs is
enough). ``BRICKDNN_URL`` is the server (default: the one the script came from, else
https://brickdnn.com), ``BRICKDNN_PROJECT`` the id of the saved network the run belongs to
and ``BRICKDNN_RUN_NAME`` its name. Without a token every method does nothing, so a script
behaves exactly as before; the token is only ever read from the environment.

It never stops a training: on the first error it warns once on stderr, keeps up to
``BUFFER`` points and tries again a little later; after ``ATTEMPTS`` failures in a row, or
when the server refuses for good (a wrong token, a limit, a run stopped in the app), it
gives up quietly and the training goes on. The code generator copies the class
:class:`BrickDNNRun` into the scripts it writes (``inspect.getsource``), so the class may
only use the standard library modules imported here.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

__all__ = ["DEFAULT_URL", "BrickDNNRun"]

DEFAULT_URL = "https://brickdnn.com"


class BrickDNNRun:
    """A run logged to Brick DNN (see brickdnn/tracking.py), or an inert one without the
    environment variable BRICKDNN_TOKEN."""

    # Points are sent in batches: when this many are waiting, or after INTERVAL seconds.
    BATCH = 200
    INTERVAL = 2.0
    # Points kept while the server cannot be reached; failures in a row before giving up.
    BUFFER = 1000
    ATTEMPTS = 3
    RETRY_SECONDS = 10.0
    TIMEOUT = 5.0

    def __init__(self, url: str | None = None, token: str | None = None) -> None:
        url = os.environ.get("BRICKDNN_URL") or url or "https://brickdnn.com"
        self.url = url.strip().rstrip("/")
        self.token = (token if token is not None else os.environ.get("BRICKDNN_TOKEN", "")).strip()
        self.id: str | None = None
        self.active = bool(self.token)
        self.points: list[dict] = []
        self._create: dict = {}
        self._started = time.monotonic()
        self._sent = self._started  # the first batch waits INTERVAL too
        self._failures = 0
        self._retry_at = 0.0
        self._warned = False

    @classmethod
    def start(
        cls,
        name: str | None = None,
        config: dict | None = None,
        project: int | None = None,
        dataset: str | None = None,
        task: str | None = None,
        epochs: int | None = None,
        url: str | None = None,
        token: str | None = None,
    ) -> BrickDNNRun:
        """Create the run on the server (nothing happens without a token)."""
        run = cls(url, token)
        if not run.active:
            return run
        if os.environ.get("BRICKDNN_PROJECT", "").strip().isdigit():
            project = int(os.environ["BRICKDNN_PROJECT"].strip())
        body = {
            "name": os.environ.get("BRICKDNN_RUN_NAME") or name,
            "config": config or {},
            "project": project,
            "dataset": dataset,
            "task": task,
            "epochs": epochs,
            "client": "brickdnn-tracking/1",
        }
        run._create = {key: value for key, value in body.items() if value is not None}
        run._open()
        return run

    def _open(self) -> bool:
        """Create the run on the server if that has not worked yet."""
        if self.id is None and self.active and time.monotonic() >= self._retry_at:
            answer = self._send("POST", "/api/tracking/runs/", self._create)
            if isinstance(answer, dict) and isinstance(answer.get("id"), str):
                self.id = answer["id"]
                print(f"Brick DNN: logging this run to {self.url}/runs", file=sys.stderr)
        return self.id is not None

    def log(self, epoch: int | None = None, step: int | None = None, **metrics) -> None:
        """One point of the curves: ``epoch`` (or ``step``) and numbers such as
        ``train_loss``, ``val_loss`` or ``val_accuracy``; other values are left out."""
        if not self.active:
            return
        try:
            point: dict = {"time": round(time.monotonic() - self._started, 3)}
            if epoch is not None:
                point["epoch"] = int(epoch)
            if step is not None:
                point["step"] = int(step)
            for key, value in metrics.items():
                number = self._number(value)
                if number is not None:
                    point[str(key)] = number
            self.points.append(point)
            del self.points[: max(0, len(self.points) - self.BUFFER)]
            if len(self.points) >= self.BATCH or time.monotonic() - self._sent >= self.INTERVAL:
                self.flush()
        except Exception as error:  # logging never stops the training
            self._fail(error)

    def flush(self) -> None:
        """Send the points that wait."""
        if not self.active or not self.points or time.monotonic() < self._retry_at:
            return
        if not self._open():
            return
        while self.active and self.points:
            batch = self.points[: self.BATCH]
            if self._send("POST", f"/api/tracking/runs/{self.id}/log/", {"points": batch}) is None:
                return
            del self.points[: len(batch)]
            self._sent = time.monotonic()

    def finish(
        self, status: str = "completed", summary: dict | None = None, message: str | None = None
    ) -> None:
        """End the run: ``completed``, ``failed`` or ``stopped``, with its final numbers
        (e.g. ``{"test_accuracy": 0.93}``)."""
        if not self.active:
            return
        try:
            self._retry_at = 0.0
            self.flush()
            if self.active and self._open():
                numbers = {str(k): self._number(v) for k, v in (summary or {}).items()}
                body: dict = {
                    "status": status,
                    "summary": {k: v for k, v in numbers.items() if v is not None},
                }
                if message:
                    body["message"] = str(message)[:500]
                self._send("POST", f"/api/tracking/runs/{self.id}/finish/", body)
        except Exception as error:
            self._fail(error)
        self.active = False

    def __enter__(self) -> BrickDNNRun:
        return self

    def __exit__(self, kind, error, traceback) -> bool:
        if kind is None:
            self.finish("completed")
        elif issubclass(kind, KeyboardInterrupt):
            self.finish("stopped", message="Stopped with Ctrl+C.")
        else:
            self.finish("failed", message=f"{kind.__name__}: {error}")
        return False  # an error goes on as before

    @staticmethod
    def _number(value) -> float | int | None:
        try:
            if hasattr(value, "item"):
                value = value.item()  # a tensor or a NumPy number
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            return value if math.isfinite(value) else None
        except Exception:
            return None

    def _send(self, method: str, path: str, body: dict) -> dict | None:
        """The server's answer, or None if the request failed (warned once)."""
        try:
            request = urllib.request.Request(
                self.url + path,
                data=json.dumps(body).encode(),
                method=method,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                    "User-Agent": "brickdnn-tracking/1",
                },
            )
            with urllib.request.urlopen(request, timeout=self.TIMEOUT) as response:
                answer = json.loads(response.read().decode() or "{}")
            self._failures = 0
            return answer if isinstance(answer, dict) else {}
        except urllib.error.HTTPError as error:
            # Busy or broken for a moment: again later. Refused (token, limit, the run was
            # stopped in the app): for good.
            self._fail(error, final=error.code not in (408, 429) and error.code < 500)
        except Exception as error:
            self._fail(error)
        return None

    def _fail(self, error: Exception, final: bool = False) -> None:
        self._failures += 1
        if not self._warned:
            self._warned = True
            detail = str(error)
            if isinstance(error, urllib.error.HTTPError):
                try:
                    detail = json.loads(error.read().decode()).get("detail") or detail
                except Exception:
                    detail = f"HTTP {error.code}"
            print(
                f"Brick DNN: could not log to {self.url} ({detail}). The training goes on"
                + ("." if final else "; logging is tried again a few times."),
                file=sys.stderr,
            )
        if final or self._failures >= self.ATTEMPTS:
            self.active = False
            self.points = []
        else:
            self._retry_at = time.monotonic() + self.RETRY_SECONDS
