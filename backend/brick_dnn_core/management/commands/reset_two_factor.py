"""``python manage.py reset_two_factor <username>``: turn two-factor authentication off.

For an account that lost both its authenticator app and its recovery codes, e.g. the
operator's own. Make sure the request comes from the account's owner first. The account can
then log in with its password alone and set it up again. The Django admin can do the same
(delete the entry under "Two-factor authentication").
"""

from __future__ import annotations

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError

from brick_dnn_core import notices, two_factor


class Command(BaseCommand):
    help = "Turn two-factor authentication off for the account USERNAME."

    def add_arguments(self, parser) -> None:
        parser.add_argument("username")

    def handle(self, *args, **options) -> None:
        user = get_user_model().objects.filter(username=options["username"]).first()
        if user is None:
            raise CommandError(f"There is no account {options['username']!r}.")
        if two_factor.disable(user):
            notices.two_factor_changed(user, enabled=False)
            self.stdout.write(f"Two-factor authentication is off for {user.get_username()} now.")
        else:
            self.stdout.write(f"{user.get_username()} had no two-factor authentication.")
