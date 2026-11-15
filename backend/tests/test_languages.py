"""English and German: the account's language, Accept-Language, e-mails, the catalogues."""

from __future__ import annotations

import gettext
import json
import re
from datetime import date

import pytest
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.utils import timezone, translation
from rest_framework.test import APIClient

from brick_dnn_core import languages, notices, po, translations
from brick_dnn_core.models import UserProfile

from .mail_helpers import html_of
from .test_account import PASSWORD, ada, login  # noqa: F401 (fixture)
from .test_mcp import call, mcp_client

pytestmark = pytest.mark.django_db

User = get_user_model()

GERMAN = {"HTTP_ACCEPT_LANGUAGE": "de-DE,de;q=0.9,en;q=0.8"}


def register(client: APIClient, name: str, **extra):
    body = {"username": name, "email": f"{name}@example.com", "password": PASSWORD, **extra}
    return client.post("/api/auth/register/", body, format="json")


# --------------------------------------------------------------------------- #
# The language of the account
# --------------------------------------------------------------------------- #


def test_accepted_language():
    assert languages.accepted_language("de-DE,de;q=0.9,en;q=0.8") == "de"
    assert languages.accepted_language("en-US,en;q=0.9,de;q=0.8") == "en"
    assert languages.accepted_language("fr-FR,fr;q=0.9,de;q=0.5") == "de"
    assert languages.accepted_language("fr, it") == "en"
    assert languages.accepted_language("de;q=0, en") == "en"
    assert languages.accepted_language("") == "en"
    assert languages.accepted_language("garbage;q=x, ,,") == "en"


def test_sign_up_saves_the_language_of_the_app():
    response = register(APIClient(), "grace", language="de")
    assert response.status_code == 201, response.data
    assert response.data["user"]["language"] == "de"
    assert UserProfile.objects.get(user__username="grace").language == "de"


def test_sign_up_without_a_language_takes_the_requests():
    response = register(APIClient(), "grace", **{})
    assert response.data["user"]["language"] == "en"
    response = APIClient().post(
        "/api/auth/register/",
        {"username": "hans", "email": "hans@example.com", "password": PASSWORD},
        format="json",
        **GERMAN,
    )
    assert response.status_code == 201, response.data
    assert response.data["user"]["language"] == "de"


def test_sign_up_rejects_other_languages():
    response = register(APIClient(), "grace", language="fr")
    assert response.status_code == 400
    assert "language" in response.data


def test_language_field_via_api(ada):  # noqa: F811
    client, _ = login("ada")
    # No language chosen yet (accounts from before languages).
    assert client.get("/api/auth/me/").data["language"] == ""
    assert client.get("/api/auth/account/language/").data == {"language": ""}

    response = client.patch("/api/auth/account/language/", {"language": "de"}, format="json")
    assert response.status_code == 200
    assert response.data == {"language": "de"}
    assert client.get("/api/auth/me/").data["language"] == "de"
    assert UserProfile.objects.get(user=ada).language == "de"

    response = client.patch("/api/auth/account/language/", {"language": "xx"}, format="json")
    assert response.status_code == 400
    assert client.get("/api/auth/me/").data["language"] == "de"

    client.patch("/api/auth/account/language/", {"language": "en"}, format="json")
    assert client.get("/api/auth/me/").data["language"] == "en"
    # The data export has it too.
    assert client.get("/api/auth/account/").status_code == 200


def test_language_needs_a_login():
    response = APIClient().patch("/api/auth/account/language/", {"language": "de"}, format="json")
    assert response.status_code == 401


# --------------------------------------------------------------------------- #
# Accept-Language
# --------------------------------------------------------------------------- #


def _linear(data) -> dict:
    return next(layer for layer in data["layers"] if layer["type"] == "Linear")


def test_layer_descriptions_follow_accept_language():
    english = APIClient().get("/api/layers/")
    german = APIClient().get("/api/layers/", **GERMAN)
    assert english["Content-Language"] == "en"
    assert german["Content-Language"] == "de"
    assert "Accept-Language" in german["Vary"]

    linear_en, linear_de = _linear(english.data), _linear(german.data)
    assert linear_en["description"].startswith("Fully connected layer")
    assert linear_de["description"].startswith("Vollständig verbundene Schicht")
    # Identifiers stay: type, label (the type name), category, parameter names.
    for key in ("type", "label", "category"):
        assert linear_de[key] == linear_en[key]
    assert [p["name"] for p in linear_de["params"]] == [p["name"] for p in linear_en["params"]]
    labels = {p["name"]: p["label"] for p in linear_de["params"]}
    assert labels["out_features"] == "Ausgabemerkmale"
    assert german.data["categories"] == english.data["categories"]


def test_without_accept_language_the_api_is_english():
    response = APIClient().get("/api/layers/")
    assert _linear(response.data)["params"][1]["label"] == "Output features"


def test_step_catalogue_and_training_options_in_german(settings):
    settings.DATA_ENABLED = True
    response = APIClient().get("/api/data/steps/", **GERMAN)
    if response.status_code == 200:
        join = next(step for step in response.data["steps"] if step["type"] == "join")
        assert join["label"] == "Andere Tabelle verknüpfen (Join)"
        how = next(param for param in join["params"] if param["name"] == "how")
        assert [option["value"] for option in how["options"]][0] == "left"
        assert how["options"][0]["label"] == "Alle Zeilen dieser Tabelle (Left Join)"
    options = APIClient().get("/api/training/", **GERMAN)
    assert options.status_code == 200
    text = json.dumps(options.data, ensure_ascii=False)
    assert "Handgeschriebene Ziffern" in text


def test_validation_errors_in_german(ada):  # noqa: F811
    response = APIClient().post(
        "/api/auth/register/",
        {"username": "ada", "email": "ada@example.com", "password": PASSWORD},
        format="json",
        **GERMAN,
    )
    assert response.status_code == 400
    assert response.data["email"] == ["Es gibt bereits ein Konto mit dieser E-Mail-Adresse."]
    # Django's and DRF's own messages are translated as well.
    response = APIClient().post("/api/auth/register/", {}, format="json", **GERMAN)
    assert "Dieses Feld ist zwingend erforderlich." in response.data["username"]


def test_the_mcp_server_stays_english(ada):  # noqa: F811
    client = mcp_client(ada)
    client.credentials(HTTP_AUTHORIZATION=client._credentials["HTTP_AUTHORIZATION"], **GERMAN)
    _, text = call(client, "list_layer_types")
    linear = _linear(json.loads(text))
    assert linear["description"].startswith("Fully connected layer")


# --------------------------------------------------------------------------- #
# E-mails
# --------------------------------------------------------------------------- #


def test_emails_in_the_language_of_the_account(ada, email_on):  # noqa: F811
    UserProfile.objects.update_or_create(user=ada, defaults={"language": "de"})
    assert notices.password_reset(ada, "https://brickdnn.example/reset-password#x")
    message = email_on[-1]
    assert message.subject.startswith("Passwort zurücksetzen")
    assert message.body.startswith("Hallo ada,")
    assert "Er gilt eine Stunde" in message.body
    html = html_of(message)
    assert '<html lang="de">' in html
    assert "Falls der Button nicht funktioniert" in html
    assert "You get this e-mail" not in message.body


def test_deletion_warning_in_german_with_a_german_date(ada, email_on):  # noqa: F811
    UserProfile.objects.update_or_create(user=ada, defaults={"language": "de"})
    assert notices.deletion_warning(ada, date(2026, 12, 3), 365)
    body = email_on[-1].body
    assert email_on[-1].subject.startswith("Ihr Konto wird gelöscht")
    assert "am 3. Dezember 2026 oder danach" in body


def test_emails_without_a_language_follow_the_request_then_english(ada, email_on):  # noqa: F811
    notices.password_changed(ada)
    assert email_on == []  # only to confirmed addresses
    assert notices.password_reset(ada, "https://brickdnn.example/x")
    assert email_on[-1].subject.startswith("Reset your password")
    assert '<html lang="en">' in html_of(email_on[-1])


def test_sign_up_in_german_sends_a_german_link(email_on):
    response = APIClient().post(
        "/api/auth/register/",
        {"username": "hans", "email": "hans@example.com", "password": PASSWORD, "language": "de"},
        format="json",
    )
    assert response.status_code == 201
    message = email_on[-1]
    assert message.subject.startswith("Bestätigen Sie Ihre E-Mail-Adresse")
    assert "Der Link gilt" in message.body


def test_english_accounts_get_english_mail_from_a_german_request(ada, email_on):  # noqa: F811
    UserProfile.objects.update_or_create(
        user=ada,
        defaults={
            "language": "en",
            "email_verified_at": timezone.now(),
            "email_verified_for": ada.email,
        },
    )
    response = APIClient().post(
        "/api/auth/password-reset/", {"login": "ada"}, format="json", **GERMAN
    )
    assert response.status_code == 200
    # The answer follows the request, …
    assert response.json()["detail"].startswith("Falls es ein Konto")
    # … the e-mail the account.
    assert email_on[-1].subject.startswith("Reset your password")


# --------------------------------------------------------------------------- #
# The catalogues
# --------------------------------------------------------------------------- #


def test_every_string_has_a_german_translation():
    """Fails when a new string has no translation or the .po/.mo files are out of date
    (run ``manage.py update_translations``, docs/translations.md)."""
    call_command("update_translations", check=True)


def test_translations_keep_their_placeholders():
    catalog = po.read(translations.po_path("de"))
    placeholder = re.compile(r"%\(\w+\)[sd]|\{\w+\}")
    for entry in catalog.entries:
        assert sorted(placeholder.findall(entry.msgid)) == sorted(
            placeholder.findall(entry.msgstr)
        ), entry.msgid


def test_po_round_trip_and_mo(tmp_path):
    catalog = po.Catalog(
        header=translations.HEADER.format(language="de"),
        entries=[
            po.Entry(msgid="Hello", msgstr="Hallo"),
            po.Entry(msgid='A "quoted"\nline', msgstr="Eine „zitierte“\nZeile"),
            po.Entry(
                msgid="One file",
                msgid_plural="%(n)d files",
                msgstr_plural={0: "Eine Datei", 1: "%(n)d Dateien"},
            ),
            po.Entry(msgid="May", msgctxt="month", msgstr="Mai"),
            po.Entry(msgid="Untranslated"),
        ],
    )
    text = po.dump(catalog)
    again = po.parse(text)
    assert again.header == catalog.header
    assert [
        (e.msgctxt, e.msgid, e.msgstr, e.msgid_plural, e.msgstr_plural) for e in again.entries
    ] == [(e.msgctxt, e.msgid, e.msgstr, e.msgid_plural, e.msgstr_plural) for e in catalog.entries]
    path = tmp_path / "test.mo"
    path.write_bytes(po.compile_mo(catalog))
    with path.open("rb") as handle:
        compiled = gettext.GNUTranslations(handle)
    assert compiled.gettext("Hello") == "Hallo"
    assert compiled.gettext('A "quoted"\nline') == "Eine „zitierte“\nZeile"
    assert compiled.ngettext("One file", "%(n)d files", 1) == "Eine Datei"
    assert compiled.ngettext("One file", "%(n)d files", 5) == "%(n)d Dateien"
    assert compiled.pgettext("month", "May") == "Mai"
    assert compiled.gettext("Untranslated") == "Untranslated"


# --------------------------------------------------------------------------- #
# Legal pages
# --------------------------------------------------------------------------- #


def test_legal_pages_in_the_language_of_the_request(settings, tmp_path):
    settings.LEGAL_DIR = str(tmp_path)
    (tmp_path / "privacy.md").write_text("# Privacy policy\n\nEnglish text.", encoding="utf-8")
    (tmp_path / "privacy.de.md").write_text("# Datenschutz\n\nDeutscher Text.", encoding="utf-8")
    (tmp_path / "terms.md").write_text("# Terms\n\nOnly English.", encoding="utf-8")

    english = APIClient().get("/api/legal/privacy/")
    german = APIClient().get("/api/legal/privacy/", **GERMAN)
    assert "English text." in english.data["content"]
    assert "Deutscher Text." in german.data["content"]
    assert german.data["title"] == "Datenschutzerklärung"
    # Without a translation the page itself, in any language.
    assert "Only English." in APIClient().get("/api/legal/terms/", **GERMAN).data["content"]
    # Sign-up stores the version of the page itself.
    info = APIClient().get("/api/legal/", **GERMAN).data
    assert info["privacy_version"] == english.data["version"]


# --------------------------------------------------------------------------- #
# Messages built from data (messages.py)
# --------------------------------------------------------------------------- #

BAD_GRAPH = {
    "version": 1,
    "nodes": [
        {"id": "in", "type": "Input", "params": {"shape": [784]}},
        {"id": "reshape", "type": "Reshape", "params": {"shape": [1, 28, 29]}},
    ],
    "edges": [{"source": "in", "target": "reshape"}],
}


def test_analysis_errors_in_german():
    english = APIClient().post("/api/analyze/", {"graph": BAD_GRAPH}, format="json")
    german = APIClient().post("/api/analyze/", {"graph": BAD_GRAPH}, format="json", **GERMAN)
    message_en = english.json()["errors"][0]["message"]
    message_de = german.json()["errors"][0]["message"]
    assert message_en.startswith("Cannot reshape")
    assert "lässt sich nicht in" in message_de and "umformen" in message_de
    # Identifiers stay.
    assert german.json()["errors"][0]["node"] == english.json()["errors"][0]["node"]


def test_message_templates_cover_the_package():
    from brick_dnn_core import messages

    with translation.override("de"):
        assert (
            messages.translate_message("Graphs are limited to 200 layers and 400 connections.")
            == "Graphen sind auf 200 Schichten und 400 Verbindungen begrenzt."
        )
        # Several sentences, one of them unknown: the known ones are translated.
        mixed = messages.translate_message(
            "Only one entry of the target shape may be -1. Something new happened."
        )
        assert mixed == "Nur ein Eintrag der Zielform darf -1 sein. Something new happened."
    # English stays as it is.
    assert messages.translate_message("Only one entry of the target shape may be -1.") == (
        "Only one entry of the target shape may be -1."
    )
