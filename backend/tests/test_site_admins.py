"""Administrators named in the settings (BRICKDNN_ADMIN_USERS)."""

from __future__ import annotations

import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from rest_framework.test import APIClient

from brick_dnn_core import budgets, site_admins

pytestmark = pytest.mark.django_db
PASSWORD = "correct-horse-1"


def user(name: str, **fields):
    return get_user_model().objects.create_user(name, password=PASSWORD, **fields)


def register(username: str):
    body = {"username": username, "password": PASSWORD, "email": f"{username.lower()}@example.com"}
    return APIClient().post("/api/auth/register/", body, format="json")


def test_listed_accounts_become_administrators_after_migrate(settings):
    ada, bob = user("ada"), user("bob")
    gone = user("eve", is_active=False)
    settings.ADMIN_USERS = ["ada", "eve", "nobody"]
    call_command("migrate", verbosity=0)  # what the backend does when it starts
    for account in (ada, bob, gone):
        account.refresh_from_db()
    assert ada.is_staff and ada.is_superuser
    assert not bob.is_staff and not bob.is_superuser
    assert not gone.is_staff  # a deactivated account gets nothing
    assert budgets.is_unlimited(ada) and not budgets.is_unlimited(bob)
    assert site_admins.promote() == []  # nothing left to do


def test_nothing_happens_without_the_setting(settings):
    ada = user("ada")
    settings.ADMIN_USERS = []
    assert site_admins.promote() == []
    ada.refresh_from_db()
    assert not ada.is_staff
    assert register("carol").status_code == 201


def test_removing_a_name_takes_nothing_away(settings):
    ada = user("ada")
    settings.ADMIN_USERS = ["ada"]
    site_admins.promote()
    settings.ADMIN_USERS = []
    site_admins.promote()
    ada.refresh_from_db()
    assert ada.is_staff and ada.is_superuser


def test_nobody_can_register_an_administrators_name(settings):
    settings.ADMIN_USERS = ["alexklg"]
    for name in ("alexklg", "AlexKLG"):
        response = register(name)
        assert response.status_code == 400
        assert response.data["username"] == ["A user with that username already exists."]
    assert register("alexklg2").status_code == 201
    # Somebody who got the name before it was listed is not promoted by a different spelling.
    other = user("ALEXKLG")
    site_admins.promote()
    other.refresh_from_db()
    assert not other.is_staff
