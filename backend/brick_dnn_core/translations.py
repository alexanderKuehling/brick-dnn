"""Collecting the strings to translate and keeping the catalogues up to date
(``manage.py update_translations``, docs/translations.md).

Sources:

- calls of ``gettext``/``_``, ``gettext_lazy``, ``ngettext`` and ``pgettext`` with literal
  strings in the Python files of ``brick_dnn_core`` (found with :mod:`ast`);
- ``{% translate "…" %}`` in the e-mail templates;
- the texts of the layer and step catalogues and the training options (catalog.py);
- the message templates of the ``brickdnn`` package (messages.py), with the context
  ``brickdnn``.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from django.conf import settings

from . import po

APP_DIR = Path(__file__).resolve().parent
LANGUAGES = ("de",)
GETTEXT_NAMES = {"_", "gettext", "gettext_lazy", "gettext_noop"}
NGETTEXT_NAMES = {"ngettext", "ngettext_lazy"}
PGETTEXT_NAMES = {"pgettext", "pgettext_lazy"}
# Files that are not looked at: the MCP server and the admin stay English.
SKIP = ("migrations", "tests")

HEADER = (
    "Project-Id-Version: Brick DNN\n"
    "Language: {language}\n"
    "MIME-Version: 1.0\n"
    "Content-Type: text/plain; charset=UTF-8\n"
    "Content-Transfer-Encoding: 8bit\n"
    "Plural-Forms: nplurals=2; plural=(n != 1);\n"
)


@dataclass(frozen=True)
class Message:
    msgid: str
    plural: str = ""
    context: str | None = None
    where: str = ""


def _literal(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def python_messages(path: Path) -> list[Message]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        args = node.args
        where = f"{path.relative_to(APP_DIR.parent)}:{node.lineno}"
        if name in GETTEXT_NAMES and args and (text := _literal(args[0])) is not None:
            found.append(Message(text, where=where))
        elif name in NGETTEXT_NAMES and len(args) >= 2:
            one, many = _literal(args[0]), _literal(args[1])
            if one is not None and many is not None:
                found.append(Message(one, many, where=where))
        elif name in PGETTEXT_NAMES and len(args) >= 2:
            context, text = _literal(args[0]), _literal(args[1])
            if context is not None and text is not None:
                found.append(Message(text, context=context, where=where))
    return found


TEMPLATE_TAG = re.compile(r"""\{%\s*(?:translate|trans)\s+(["'])(.*?)\1""")


def template_messages(path: Path) -> list[Message]:
    text = path.read_text(encoding="utf-8")
    return [
        Message(match.group(2), where=f"{path.relative_to(APP_DIR.parent)}")
        for match in TEMPLATE_TAG.finditer(text)
    ]


def collect() -> list[Message]:
    """Every string to translate, in a stable order, without duplicates."""
    from . import catalog

    messages: list[Message] = []
    for path in sorted(APP_DIR.rglob("*.py")):
        if any(part in SKIP for part in path.relative_to(APP_DIR).parts):
            continue
        messages.extend(python_messages(path))
    # The e-mails; the admin pages use Django's own translations.
    for path in sorted((APP_DIR / "templates" / "brick_dnn_core" / "email").rglob("*.*")):
        messages.extend(template_messages(path))
    catalog_texts = catalog.catalog_texts()
    messages.extend(Message(text, where="catalog") for text in catalog_texts)
    # Messages the brickdnn package builds from data (messages.py), unless the catalogue has them.
    from . import messages as package_messages

    known = set(catalog_texts)
    messages.extend(
        Message(template, context=package_messages.CONTEXT, where="brickdnn")
        for template in package_messages.templates()
        if template not in known
    )
    unique: dict[tuple[str | None, str], Message] = {}
    for message in messages:
        unique.setdefault((message.context, message.msgid), message)
    return list(unique.values())


def po_path(language: str) -> Path:
    return Path(settings.LOCALE_PATHS[0]) / language / "LC_MESSAGES" / "django.po"


def update(language: str, messages: list[Message] | None = None) -> tuple[po.Catalog, list[str]]:
    """The catalogue of ``language`` with exactly the collected strings (translations kept,
    new ones empty); also returns the strings without a translation."""
    messages = collect() if messages is None else messages
    old = po.read(po_path(language))
    catalog = po.Catalog(header=old.header or HEADER.format(language=language))
    for message in messages:
        previous = old.get(message.msgid, message.context)
        entry = po.Entry(
            msgid=message.msgid,
            msgid_plural=message.plural,
            msgctxt=message.context,
            comments=[f"#: {message.where}"] if message.where else [],
        )
        if previous is not None:
            entry.msgstr = previous.msgstr
            entry.msgstr_plural = dict(previous.msgstr_plural)
            entry.fuzzy = previous.fuzzy
        if message.plural and not entry.msgstr_plural:
            entry.msgstr_plural = {0: "", 1: ""}
        catalog.entries.append(entry)
    missing = [entry.msgid for entry in catalog.entries if not entry.translated]
    return catalog, missing


def write(language: str, catalog: po.Catalog) -> None:
    path = po_path(language)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(po.dump(catalog), encoding="utf-8")
    path.with_suffix(".mo").write_bytes(po.compile_mo(catalog))
