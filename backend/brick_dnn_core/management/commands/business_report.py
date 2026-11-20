"""``manage.py business_report``: does the service work as a business?

    manage.py business_report                    the last 6 months and where things stand now
    manage.py business_report --period week      by week (8), or --period day (14)
    manage.py business_report --count 12         more periods
    manage.py business_report --json             the same as JSON, for a spreadsheet or script

Sign-ups, activated accounts (first training within 7 days), accounts that trained, paying
accounts by plan, estimated MRR, new and ended subscriptions, free to paid conversion, CPU
hours used against the server's capacity, storage against the disk, runs logged from
elsewhere and links shared. Aggregates from the database only: nothing is tracked for this
(business_report.py says how each number is counted; docs/operations.md how to read them).
The same is *Business report* in the Django admin.
"""

from __future__ import annotations

import json

from django.core.management.base import BaseCommand

from brick_dnn_core import business_report


class Command(BaseCommand):
    help = (
        "Business numbers from the database: sign-ups, activation, paying accounts, MRR, "
        "conversion, CPU hours against capacity, storage, runs logged and links shared."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--period",
            choices=list(business_report.PERIODS),
            default="month",
            help="Count by day, week (from Monday) or month, in UTC (default: month)",
        )
        parser.add_argument(
            "--count",
            type=int,
            help="How many periods, the running one included (default: 14 days, 8 weeks, "
            f"6 months; at most {business_report.MAX_PERIODS})",
        )
        parser.add_argument("--json", action="store_true", help="Print JSON instead of text")

    def handle(self, *args, **options) -> None:
        data = business_report.report(options["period"], options["count"])
        if options["json"]:
            self.stdout.write(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            self.stdout.write(business_report.text(data))
