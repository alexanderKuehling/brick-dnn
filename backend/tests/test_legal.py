"""Imprint, privacy policy and terms of use (Markdown files on the server), and accepting the
terms at sign-up."""

from __future__ import annotations

import hashlib

import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from brick_dnn_core import legal
from brick_dnn_core.models import UserProfile

pytestmark = pytest.mark.django_db

IMPRINT = "# Impressum\n\nMax Mustermann\nMusterstraße 1\n"
PRIVACY = "# Datenschutzerklärung\n\nWir speichern so wenig wie möglich.\n"
TERMS = "# Nutzungsbedingungen\n\n1. Seien Sie nett.\n"


@pytest.fixture
def legal_dir(settings, tmp_path):
    settings.LEGAL_DIR = tmp_path / "legal"
    settings.DELETE_UNUSED_UPLOADS_DAYS = 0
    settings.DELETE_INACTIVE_ACCOUNTS_DAYS = 0
    return settings.LEGAL_DIR


def write(directory, page: str, text: str) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{page}.md").write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def register(client, username="ada", **extra):
    body = {
        "username": username,
        "email": f"{username}@example.com",
        "password": "correct-horse-battery",
        **extra,
    }
    return client.post("/api/auth/register/", body, format="json")


# --------------------------------------------------------------------------- #
# The pages
# --------------------------------------------------------------------------- #


def test_no_legal_pages_by_default(legal_dir):
    client = APIClient()
    response = client.get("/api/legal/")
    assert response.status_code == 200
    assert response.data == {
        "pages": {"imprint": False, "privacy": False, "terms": False},
        "terms_version": None,
        "privacy_version": None,
        "accept_terms_required": False,
        "retention": {"unused_uploads_days": 0, "inactive_accounts_days": 0},
    }
    for page in ("imprint", "privacy", "terms"):
        response = client.get(f"/api/legal/{page}/")
        assert response.status_code == 404
        assert response.data["detail"] == "This page has not been set up on this server."


def test_pages_come_from_the_legal_dir(legal_dir, settings):
    write(legal_dir, "imprint", IMPRINT)
    terms_version = write(legal_dir, "terms", TERMS)
    settings.DELETE_UNUSED_UPLOADS_DAYS = 90
    client = APIClient()

    info = client.get("/api/legal/").data
    assert info["pages"] == {"imprint": True, "privacy": False, "terms": True}
    assert info["terms_version"] == terms_version
    assert info["accept_terms_required"] is True
    assert info["retention"] == {"unused_uploads_days": 90, "inactive_accounts_days": 0}

    page = client.get("/api/legal/imprint/").data
    assert page["page"] == "imprint"
    assert page["title"] == "Imprint"
    assert page["content"] == IMPRINT.strip()
    assert page["version"] == hashlib.sha256(IMPRINT.encode()).hexdigest()[:16]
    assert page["updated_at"]
    assert client.get("/api/legal/privacy/").status_code == 404


def test_changes_are_online_at_once(legal_dir):
    first = write(legal_dir, "terms", TERMS)
    client = APIClient()
    assert client.get("/api/legal/terms/").data["version"] == first
    second = write(legal_dir, "terms", TERMS + "\n2. Neu.\n")
    assert second != first
    response = client.get("/api/legal/terms/")
    assert response.data["version"] == second
    assert response.data["content"].endswith("2. Neu.")


def test_empty_and_oversized_files_are_ignored(legal_dir, monkeypatch):
    write(legal_dir, "imprint", "  \n\n")
    write(legal_dir, "privacy", PRIVACY)
    monkeypatch.setattr(legal, "MAX_BYTES", 10)
    pages = APIClient().get("/api/legal/").data["pages"]
    assert pages == {"imprint": False, "privacy": False, "terms": False}


def test_several_folders_first_one_wins(settings, tmp_path):
    # The server: files on the server first, then the texts of its domain in the repository.
    server, site, missing = tmp_path / "server", tmp_path / "site", tmp_path / "missing"
    settings.LEGAL_DIR = f"{server}:{missing}:{site}"
    write(site, "imprint", IMPRINT)
    write(site, "privacy", PRIVACY)
    write(server, "privacy", "# Datenschutz\n\nVom Server.\n")
    write(server, "imprint", "  \n")  # empty: the next folder's file counts
    client = APIClient()
    assert client.get("/api/legal/").data["pages"] == {
        "imprint": True,
        "privacy": True,
        "terms": False,
    }
    assert client.get("/api/legal/imprint/").data["content"] == IMPRINT.strip()
    assert "Vom Server." in client.get("/api/legal/privacy/").data["content"]
    assert client.get("/api/legal/terms/").status_code == 404


def test_markdown_is_returned_as_text(legal_dir):
    """The API hands out the Markdown as it is; the app renders it without raw HTML."""
    text = "# Hi\n\n<script>alert(1)</script>\n"
    write(legal_dir, "privacy", text)
    response = APIClient().get("/api/legal/privacy/")
    assert response["Content-Type"].startswith("application/json")
    assert response.data["content"] == text.strip()


def test_byte_order_mark_and_bad_bytes(legal_dir):
    legal_dir.mkdir(parents=True)
    (legal_dir / "imprint.md").write_bytes("﻿# Impressum\n".encode() + b"\xff\n")
    content = APIClient().get("/api/legal/imprint/").data["content"]
    assert content.startswith("# Impressum")
    assert "�" in content


def test_only_known_pages(legal_dir):
    write(legal_dir, "secret", "nope")
    client = APIClient()
    assert client.get("/api/legal/secret/").status_code == 404
    assert client.get("/api/legal/..%2Fsecret_key/").status_code == 404
    assert legal.read_page("../secret") is None


def test_api_root_lists_account_and_legal():
    endpoints = APIClient().get("/api/").data["endpoints"]
    assert endpoints["legal"].endswith("/api/legal/")
    assert endpoints["account"].endswith("/api/auth/account/")


# --------------------------------------------------------------------------- #
# Accepting the terms at sign-up
# --------------------------------------------------------------------------- #


def test_sign_up_without_terms_needs_no_acceptance(legal_dir):
    write(legal_dir, "privacy", PRIVACY)  # a privacy policy alone asks for nothing
    response = register(APIClient())
    assert response.status_code == 201, response.data
    profile = UserProfile.objects.get(user__username="ada")
    assert profile.terms_accepted_at is None
    assert profile.terms_version == ""
    assert profile.last_active_at is not None


def test_sign_up_requires_accepting_the_terms(legal_dir):
    terms_version = write(legal_dir, "terms", TERMS)
    privacy_version = write(legal_dir, "privacy", PRIVACY)
    client = APIClient()

    for extra in ({}, {"accept_terms": False}, {"accept_terms": "yes please"}):
        response = register(client, **extra)
        assert response.status_code == 400
        assert "accept_terms" in response.data
    assert not get_user_model().objects.exists()

    response = register(client, accept_terms=True)
    assert response.status_code == 201, response.data
    profile = UserProfile.objects.get(user__username="ada")
    assert profile.terms_accepted_at is not None
    assert profile.terms_version == terms_version
    assert profile.privacy_version == privacy_version


def test_terms_without_privacy_policy(legal_dir):
    terms_version = write(legal_dir, "terms", TERMS)
    assert register(APIClient(), accept_terms=True).status_code == 201
    profile = UserProfile.objects.get(user__username="ada")
    assert (profile.terms_version, profile.privacy_version) == (terms_version, "")
