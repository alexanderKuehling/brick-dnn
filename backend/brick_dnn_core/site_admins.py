"""Accounts the operator names as administrators in the settings.

``BRICKDNN_ADMIN_USERS`` lists usernames (see deploy/sites/README.md for a public server).
After every ``migrate``, which the backend runs when it starts, the accounts with these names
become staff and superusers: they can open the Django admin and, by default, have no limits
(budgets.py). It only ever gives the rights; removing a name from the list takes nothing
away (do that in the Django admin). The names cannot be registered by anyone, so that nobody
can take one over after its account was deleted: create the account first, then list it.
"""

from __future__ import annotations

import logging

from django.conf import settings
from django.contrib.auth import get_user_model

logger = logging.getLogger(__name__)


def admin_usernames() -> list[str]:
    return [name for name in settings.ADMIN_USERS if name]


def is_reserved(username: str) -> bool:
    """Whether ``username`` is one of the administrators' names, whatever its case."""
    wanted = username.strip().casefold()
    return any(wanted == name.casefold() for name in admin_usernames())


def promote() -> list[str]:
    """Make the listed accounts staff and superusers. Returns the names that changed."""
    names = admin_usernames()
    if not names:
        return []
    changed = []
    users = get_user_model().objects.filter(username__in=names, is_active=True)
    for user in users.exclude(is_staff=True, is_superuser=True):
        user.is_staff = user.is_superuser = True
        user.save(update_fields=["is_staff", "is_superuser"])
        changed.append(user.get_username())
        logger.warning("%s is an administrator now (BRICKDNN_ADMIN_USERS).", user.get_username())
    return changed


def promote_after_migrate(sender, **kwargs) -> None:
    promote()
