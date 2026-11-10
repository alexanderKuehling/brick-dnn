"""``manage.py training_worker``: work off the training queue.

    manage.py training_worker                    train until stopped (SIGTERM, Ctrl+C)
    manage.py training_worker --slots 2          two trainings at the same time
    manage.py training_worker --once             train one run, then end
    manage.py training_worker --once --idle-exit 60   … or end if none came for a minute

Needs BRICKDNN_TRAINING_WORKER=true (also for the web server, which then puts new trainings
into the queue instead of starting them itself), the same database and the same data
directory as the web server. See brick_dnn_core/training_worker.py and docs/training.md.
"""

from __future__ import annotations

import signal
import time

from django.core.management.base import BaseCommand

from brick_dnn_core import training_queue as queue
from brick_dnn_core.training_worker import Worker, touch_alive_file


class Command(BaseCommand):
    help = "Take trainings from the queue and train them (the training worker)."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--once", action="store_true", help="Train one run, then end (e.g. a Kubernetes Job)"
        )
        parser.add_argument(
            "--slots",
            type=int,
            help="Trainings at the same time (default: BRICKDNN_WORKER_SLOTS, else from the "
            "CPUs and OMP_NUM_THREADS)",
        )
        parser.add_argument(
            "--name", help="Name of this worker (default: BRICKDNN_WORKER_NAME, else the host name)"
        )
        parser.add_argument(
            "--poll", type=float, default=2.0, help="Seconds between two looks at the queue"
        )
        parser.add_argument(
            "--idle-exit",
            type=float,
            help="End when there was nothing to do for this many seconds (default: never)",
        )

    def handle(self, *args, **options) -> None:
        if not queue.enabled():
            self.stdout.write(
                "BRICKDNN_TRAINING_WORKER is off: the web server starts trainings itself, so "
                "there is nothing for a worker to do. Set it to true for the web server and "
                "this worker to use the queue."
            )
            if not options["once"]:
                self._idle()
            return
        worker = Worker(
            name=options["name"],
            slots=options["slots"],
            once=options["once"],
            poll=options["poll"],
            idle_exit=options["idle_exit"],
            log=self._log,
        )
        worker.run()

    def _log(self, line: str) -> None:
        self.stdout.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}")
        self.stdout.flush()

    def _idle(self) -> None:
        """Stay alive without doing anything, so that a container that is restarted when it
        ends does not start over and over."""
        stop = []
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: stop.append(True))
        while not stop:
            touch_alive_file()  # for the health check of the container
            time.sleep(0.5)
