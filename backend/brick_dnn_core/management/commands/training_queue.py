"""``manage.py training_queue``: the training queue at a glance.

    manage.py training_queue           workers, running and waiting trainings, the last 7 days
    manage.py training_queue --days 30 the statistics of the last 30 days
    manage.py training_queue --recover also put the runs of dead workers back into the queue

The same as *Training workers* in the Django admin. The numbers at the end say whether the
server is large enough: how long trainings waited for a free slot (median and 95th
percentile) and how busy the worker's slots were.
"""

from __future__ import annotations

from django.core.management.base import BaseCommand

from brick_dnn_core import training_queue as queue
from brick_dnn_core.training_queue import duration


class Command(BaseCommand):
    help = "Show the training queue: workers, running and waiting trainings, statistics."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--days", type=int, default=7, help="Days of the statistics")
        parser.add_argument(
            "--recover",
            action="store_true",
            help="First put the runs of workers without a heartbeat back into the queue",
        )

    def handle(self, *args, **options) -> None:
        write = self.stdout.write
        if not queue.enabled():
            write(
                "The training queue is off (BRICKDNN_TRAINING_WORKER): the web server starts "
                "trainings itself."
            )
        if options["recover"]:
            for run, outcome in queue.recover():
                write(f"Recovered “{run.name}” ({run.pk}): {outcome}")
        data = queue.overview()
        data["stats"] = queue.stats(days=max(1, options["days"]), now=data["now"])

        write(f"Workers ({data['slots']} slot{'' if data['slots'] == 1 else 's'} online)")
        if data["message"]:
            write(f"  {data['message']}")
        for worker in data["workers"]:
            memory = f", {worker['memory_mb']} MB" if worker["memory_mb"] else ""
            jobs = ", ".join(worker["jobs"]) or "idle"
            write(
                f"  {worker['name']}: {worker['state']}, heartbeat "
                f"{duration(worker['heartbeat_seconds'])} ago, {worker['slots']} slot(s) × "
                f"{worker['threads']} thread(s){memory}: {jobs}"
            )
        if not data["workers"]:
            write("  none")

        write(f"Running ({len(data['running'])})")
        for run in data["running"]:
            lost = "" if run["worker_alive"] else "  [its worker is gone]"
            write(
                f"  {run['name']}  {run['account']} ({run['plan']}, priority {run['priority']})"
                f"  {run['status']} for {duration(run['running_seconds'])}, "
                f"{run['cpu_minutes']:g} CPU min, on {run['worker'] or '-'}{lost}"
            )
        write(f"Waiting ({len(data['waiting'])})")
        for run in data["waiting"]:
            notes = []
            if run["blocked"]:
                notes.append("waits for a training of the same account")
            if run["restarts"]:
                notes.append(f"restart {run['restarts']}")
            estimate = run["estimated_wait_seconds"]
            start = (
                f"starts in about {duration(estimate)}" if estimate is not None else "start unknown"
            )
            write(
                f"  {run['position']}. {run['name']}  {run['account']} ({run['plan']}, priority "
                f"{run['priority']})  waiting {duration(run['waiting_seconds'])}, {start}"
                + (f"  [{'; '.join(notes)}]" if notes else "")
            )

        stats = data["stats"]
        write(f"Last {stats['days']} days")
        write(
            f"  trainings started: {stats['started']}; wait for a slot: median "
            f"{duration(stats['median_wait_seconds'])}, 95th percentile "
            f"{duration(stats['p95_wait_seconds'])}, longest {duration(stats['max_wait_seconds'])}"
        )
        for plan in stats["by_plan"]:
            write(
                f"    {plan['plan']}: {plan['started']} started, median "
                f"{duration(plan['median_wait_seconds'])}, 95th percentile "
                f"{duration(plan['p95_wait_seconds'])}"
            )
        share = stats["utilisation"]
        busy = duration(stats["busy_slot_seconds"])
        there = duration(stats["available_slot_seconds"])
        write(
            "  worker utilisation: "
            + (
                f"{share * 100:.0f} % ({busy} busy of {there} slot time)"
                if share is not None
                else "-"
            )
        )
