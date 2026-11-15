"""Messages the ``brickdnn`` package builds from data, in the request's language.

The package writes its messages in English with the values put in (``f"Kernel size {k} is
larger than the padded input size {n}."``) and knows nothing of Django: the CLI and the MCP
server show them as they are. For the web app the API translates them on the way out:

- :func:`templates` reads the package's source (:mod:`ast`) and turns every message-like string
  into a *template*, the literal parts with numbered placeholders: ``"Kernel size {0} is larger
  than the padded input size {1}."``. These are the msgids (context ``brickdnn``) in
  ``locale/de/LC_MESSAGES/django.po``; ``manage.py update_translations`` collects them.
- :func:`translate_message` matches a finished English message against the templates (one
  regular expression each, longest literal text first), takes the values out of it and puts them
  into the translated template. A message made of several sentences is translated sentence by
  sentence. What matches no template stays English.
- :func:`translate_response` applies it to the text fields of an API answer (``message``,
  ``summary``, ``errors`` …).

So the package stays free of Django and its messages keep one source of truth, and a changed
English message only loses its translation until ``update_translations`` picks it up again.
"""

from __future__ import annotations

import ast
import functools
import re
from pathlib import Path
from typing import Any

from django.utils import translation
from django.utils.translation import gettext, pgettext
from rest_framework import renderers

CONTEXT = "brickdnn"
PACKAGE = Path(__file__).resolve().parent.parent / "brickdnn"
# The modules whose messages users see: network analysis, training checks, data preparation.
MODULES = (
    "graph.py",
    "layers.py",
    "training/compat.py",
    "training/engine.py",
    "training/classical_engine.py",
    "training/hybrid_engine.py",
    "training/state.py",
    "data/checks.py",
    "data/pipeline.py",
    "data/steps",
)
# Keys of API answers whose string values (or lists of strings) are messages.
MESSAGE_KEYS = frozenset(
    {
        "message",
        "messages",
        "summary",
        "detail",
        "errors",
        "warnings",
        "notes",
        "note",
        "hint",
        "reason",
        "explanation",
        "fix_summary",
        "log",
    }
)
_PLACEHOLDER = re.compile(r"\{(\d+)\}")


def _is_message(text: str) -> bool:
    """A sentence for people, not an identifier, a format or a code snippet."""
    words = re.findall(r"[A-Za-z]{2,}", text)
    if len(words) < 3:
        return False
    if text.lstrip().startswith(("%", "{", "<", "#", "/", "http", "def ", "import ", "class ")):
        return False
    if "\n" in text.strip() or "    " in text or text.count("(") > 4:
        return False
    # Fragments that other messages are built from, and code (docstrings, assignments).
    if text[:1].islower() or text[:1] in " ,;" or '"""' in text or " = " in text:
        return False
    return bool(re.search(r"[A-Za-z]{2,} [A-Za-z]{2,}", text))


def _template_of(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.replace("{", "{{").replace("}", "}}")
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        index = 0
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value.replace("{", "{{").replace("}", "}}"))
            elif isinstance(value, ast.FormattedValue):
                parts.append(f"{{{index}}}")
                index += 1
            else:
                return None
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _template_of(node.left), _template_of(node.right)
        if left is None or right is None:
            return None
        # Renumber the placeholders of the right part after those of the left one.
        offset = len(_PLACEHOLDER.findall(left))
        right = _PLACEHOLDER.sub(lambda m: f"{{{int(m.group(1)) + offset}}}", right)
        return left + right
    return None


def _docstrings(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def _module_files() -> list[Path]:
    files: list[Path] = []
    for name in MODULES:
        path = PACKAGE / name
        files.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])
    return [path for path in files if path.exists()]


@functools.lru_cache(maxsize=1)
def templates() -> tuple[str, ...]:
    """Every message template of the package, in a stable order without duplicates."""
    found: dict[str, None] = {}
    for path in _module_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        skip = _docstrings(tree)
        inner: set[int] = set()
        for node in ast.walk(tree):
            if id(node) in skip or id(node) in inner:
                continue
            if isinstance(node, (ast.JoinedStr, ast.BinOp, ast.Constant)):
                template = _template_of(node)
                if template is None:
                    continue
                # The parts of a concatenation or f-string are not messages of their own.
                for child in ast.walk(node):
                    if child is not node:
                        inner.add(id(child))
                text = template.replace("{{", "{").replace("}}", "}")
                if _is_message(_PLACEHOLDER.sub("x", text)):
                    found.setdefault(template, None)
    return tuple(found)


@functools.lru_cache(maxsize=1)
def _matchers() -> list[tuple[re.Pattern[str], str]]:
    """(regular expression, template), the most specific (longest literal text) first."""
    result = []
    for template in templates():
        pieces = re.split(r"(\{\d+\})", template)
        pattern = ""
        for piece in pieces:
            match = _PLACEHOLDER.fullmatch(piece)
            if match:
                pattern += f"(?P<p{match.group(1)}>.+?)"
            else:
                pattern += re.escape(piece.replace("{{", "{").replace("}}", "}"))
        literal = len(_PLACEHOLDER.sub("", template))
        result.append((literal, re.compile(f"^{pattern}$", re.S), template))
    result.sort(key=lambda item: -item[0])
    return [(pattern, template) for _, pattern, template in result]


def _translate_one(text: str) -> str | None:
    for pattern, template in _matchers():
        match = pattern.match(text)
        if not match:
            continue
        translated = pgettext(CONTEXT, template)
        if translated == template:
            # Texts of the catalogues are translated without the context (catalog.py).
            translated = gettext(template)
        if translated == template:
            return None
        values = {int(name[1:]): value for name, value in match.groupdict().items()}
        try:
            result = _PLACEHOLDER.sub(lambda m, v=values: v[int(m.group(1))], translated)
        except KeyError:
            return None
        return result.replace("{{", "{").replace("}}", "}")
    return None


_SENTENCE = re.compile(r"(?<=[.!?:;])\s+(?=[A-Z“\"'(])")


def translate_message(text: str) -> str:
    """``text`` in the active language, or unchanged if no template matches (or English)."""
    if not text or (translation.get_language() or "en").split("-")[0] == "en":
        return text
    whole = _translate_one(text)
    if whole is not None:
        return whole
    sentences = _SENTENCE.split(text)
    if len(sentences) == 1:
        return text
    return " ".join(_translate_one(sentence) or sentence for sentence in sentences)


class JSONRenderer(renderers.JSONRenderer):
    """DRF's JSON renderer, with the package's messages in the request's language."""

    def render(self, data, accepted_media_type=None, renderer_context=None):
        return super().render(translate_response(data), accepted_media_type, renderer_context)


def translate_response(data: Any, key: str | None = None) -> Any:
    """``data`` (an API answer) with its messages translated."""
    if (translation.get_language() or "en").split("-")[0] == "en":
        return data
    if isinstance(data, dict):
        return {k: translate_response(v, k) for k, v in data.items()}
    if isinstance(data, list):
        return [translate_response(item, key) for item in data]
    if isinstance(data, str) and key in MESSAGE_KEYS:
        return translate_message(data)
    return data
