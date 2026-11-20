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
    # The generated datasets: the description in German, the class names as they are (they
    # name the classes in masks, metrics and predictions).
    cells = next(d for d in options.data["builtin_datasets"] if d["id"] == "cells")
    assert cells["name"] == "Cells"
    assert cells["description"].startswith("Segmentierung: Zellen mit ihren Zellkernen")
    assert cells["class_names"] == ["background", "cell", "nucleus"]


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


def test_messages_that_start_with_a_value_and_their_values():
    from brick_dnn_core import messages

    with translation.override("de"):
        # A sentence that starts with the name of a layer.
        assert messages.translate_message("Linear has no incoming connection.") == (
            "Linear hat keine eingehende Verbindung."
        )
        # Counts, numbers and a hint inside the message are translated too.
        assert messages.translate_message(
            "The network outputs 4 values, but Key lookup has 10 classes. The last layer "
            "should output 10 values (e.g. Linear with 10 outputs)."
        ) == (
            "Das Netz gibt 4 Werte aus, aber Key lookup hat 10 Klassen. Die letzte Schicht "
            "sollte 10 Werte ausgeben (z. B. Linear mit 10 Ausgaben)."
        )
        assert "10.000 Einträge" in messages.translate_message(
            "The Embedding has 10,000 entries, but Key lookup has only 289 words: the other "
            "9,711 are never used. Set its vocabulary size to 289."
        )
        # What a fix of the training check changes.
        assert messages.translate_message("Input shape 64 → 10") == "Form der Eingabe 64 → 10"
        # The label of a readiness fix, only inside "fix".
        answer = {"fix": {"label": "Drop rows without churn"}, "label": "Drop rows without x"}
        assert messages.translate_response(answer) == {
            "fix": {"label": "Zeilen ohne churn entfernen"},
            "label": "Drop rows without x",
        }


def test_training_check_in_german(ada):  # noqa: F811
    from brickdnn.examples import get_example

    client = APIClient()
    client.force_authenticate(ada)
    graph = get_example("transformer_classifier")["graph"]
    body = {"graph": graph, "dataset": {"kind": "builtin", "id": "lookup"}}
    answer = client.post("/api/training/check/", body, format="json", **GERMAN).json()
    assert any(e.startswith("Jeder Text von Key lookup hat 10 Tokens") for e in answer["errors"])
    assert "Form der Eingabe 64 → 10" in [fix["summary"] for fix in answer["fixes"]]


def test_run_messages_in_german():
    """The final message of a run, its metrics and what the queue stores with a run."""
    from brick_dnn_core import messages

    cases = {
        "Finished: validation accuracy 16.8%; test accuracy 12.0%": (
            "Abgeschlossen: Validierungsgenauigkeit 16,8 %; Testgenauigkeit 12,0 %"
        ),
        "Finished: validation RMSE 9.682, R² 0.965. Fitted on 240,000 rows in 0.8 s.": (
            "Abgeschlossen: Validierungs-RMSE 9,682, R² 0,965. "
            "Auf 240.000 Zeilen in 0,8 s angepasst."
        ),
        "Stopped early after epoch 7: no improvement for 3 epochs; kept the weights of epoch "
        "4. Validation loss 0.1234, RMSE 4.5.": (
            "Nach Epoche 7 vorzeitig gestoppt: keine Verbesserung seit 3 Epochen; die Gewichte "
            "von Epoche 4 wurden behalten. Validierungsverlust 0,1234, RMSE 4,5."
        ),
        "Finished: 3 features learned in 3 epochs; HistGradientBoostingClassifier on them: "
        "validation accuracy 91.0%. Fitted on 800 rows in 0.1 s.": (
            "Abgeschlossen: 3 Merkmale in 3 Epochen gelernt; HistGradientBoostingClassifier "
            "darauf: Validierungsgenauigkeit 91,0 %. Auf 800 Zeilen in 0,1 s angepasst."
        ),
        "Finished 5 trials. Best: trial 3 (learning rate 0.01, batch size 32, Adam), "
        "validation accuracy 91.0%": (
            "5 Versuche abgeschlossen. Bester: Versuch 3 (Lernrate 0,01, Batchgröße 32, Adam), "
            "Validierungsgenauigkeit 91,0 %"
        ),
        "Combined 2 runs (weighted average): validation accuracy 92.0%; test accuracy 94.0%.": (
            "2 Läufe kombiniert (gewichteter Mittelwert): Validierungsgenauigkeit 92,0 %; "
            "Testgenauigkeit 94,0 %."
        ),
        "Waiting for a free training slot.": "Wartet auf einen freien Trainingsplatz.",
        "Demo data: random inputs and a hidden rule to learn (classification).": (
            "Demodaten: zufällige Eingaben und eine versteckte Regel zum Lernen (Klassifikation)."
        ),
        # A value that only looks like a metric stays as it is.
        "Finished: accuracy unknown": "Abgeschlossen: accuracy unknown",
    }
    with translation.override("de"):
        for english, german in cases.items():
            assert messages.translate_message(english) == german
    assert messages.translate_message("Finished: validation accuracy 16.8%") == (
        "Finished: validation accuracy 16.8%"
    )


def test_readiness_messages_and_fix_labels_in_german():
    from brick_dnn_core import messages

    with translation.override("de"):
        # A label of a layer and a sentence put into a message as values.
        assert messages.translate_message(
            "Gradient boosting classification predicts one class per sample, but Shapes has a "
            "class for every pixel. A segmentation needs a neural network that outputs a class "
            "per pixel, e.g. the example U-Net."
        ).startswith("Gradient-Boosting-Klassifikation sagt eine Klasse pro Beispiel vorher")
        # Two values that follow each other: the first one takes the whole hint.
        assert "Die letzte Schicht sollte 10 Werte ausgeben (z. B. Linear mit 10 Ausgaben)." in (
            messages.translate_message(
                "The network outputs shape (4, 64, 64), but MNIST needs one score per class. "
                "The last layer should output 10 values (e.g. Linear with 10 outputs). A network "
                "with one output per pixel is trained on a segmentation dataset (images with "
                "masks), e.g. the built-in Shapes or Cells."
            )
        )
        assert messages.translate_message("2 feature columns have missing values (age, x).") == (
            "2 Merkmalsspalten haben fehlende Werte (age, x)."
        )
        assert messages.translate_message(
            "income is strongly skewed (skewness 3.6): a few large values dominate it. "
            "log(1 + x) makes it more even."
        ).endswith("log(1 + x) macht sie gleichmäßiger.")
        assert messages.translate_message(
            "Standardised 2 columns (mean 0, standard deviation 1 on the training rows)."
        ).startswith("2 Spalten standardisiert")
        answer = {"fix": {"label": "Drop id"}, "label": "Drop id", "message": "Drop id"}
        assert messages.translate_response(answer) == {
            "fix": {"label": "id entfernen"},
            "label": "Drop id",
            "message": "Drop id",
        }
        assert messages.translate_label("Log-transform 2 columns (log1p)") == (
            "2 Spalten logarithmieren (log1p)"
        )


def test_extra_templates_are_still_in_the_source():
    """EXTRA_TEMPLATES and LABEL_TEMPLATES repeat texts of the package: when a text changes
    there, its copy in messages.py (and its translation) has to follow."""
    import ast
    from pathlib import Path

    from brick_dnn_core import messages

    backend = Path(messages.__file__).resolve().parent.parent
    pieces: set[str] = set()
    for folder in ("brickdnn", "brick_dnn_core"):
        for path in (backend / folder).rglob("*.py"):
            if path.name == "messages.py":
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    pieces.add(node.value)
    source = "\n".join(pieces)
    for template in (*messages.EXTRA_TEMPLATES, *messages.LABEL_TEMPLATES):
        # The longest piece between two values, numbers, brackets or commas, or all but its last word
        # or its first words (a plural "s" or a conditional part may follow in the source).
        literal = max(re.split(r"\{\d+\}|\d+|\(|, ", template), key=len).strip(" .,:;()")
        start = " ".join(literal.split(" ")[:4])
        assert literal in source or literal.rsplit(" ", 1)[0] in source or start in source, template
