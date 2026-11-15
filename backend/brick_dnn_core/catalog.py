"""The descriptions of layers, preparation steps and training options in the request's language.

The ``brickdnn`` package describes its layers, steps and options in English and knows nothing
of Django or gettext (the CLI and the MCP server use it as it is). The API translates those
texts when it returns them: :func:`translate` replaces every text field (``label``, ``help``,
``description`` …) by its gettext translation in the active language (languages.py), leaving
identifiers (types, names, values, categories) as they are. The English texts are the msgids
in ``locale/de/LC_MESSAGES/django.po``; :func:`catalog_texts` lists them for
``manage.py update_translations``.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator
from typing import Any

from django.utils import translation
from django.utils.translation import gettext

# Keys whose string values are texts for people.
TEXT_FIELDS = frozenset({"label", "help", "description", "note", "empty_means", "title"})


def _is_text(key: str, value: Any, parent: dict[str, Any]) -> bool:
    if key not in TEXT_FIELDS or not isinstance(value, str) or not value.strip():
        return False
    # A layer's label is its type name (Linear, Conv2d): not translated.
    return value != parent.get("type")


def _walk(data: Any, change) -> Any:
    if isinstance(data, dict):
        return {
            key: (change(value) if _is_text(key, value, data) else _walk(value, change))
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [_walk(item, change) for item in data]
    return data


def translate(data: Any) -> Any:
    """``data`` with its texts in the active language (unchanged in English)."""
    if (translation.get_language() or "en").split("-")[0] == "en":
        return data
    return _walk(data, gettext)


def texts(data: Any) -> Iterator[str]:
    """The texts of ``data`` that :func:`translate` would translate."""
    found: list[str] = []

    def collect(value: str) -> str:
        found.append(value)
        return value

    _walk(data, collect)
    return iter(found)


@functools.lru_cache(maxsize=4)
def layers(language: str) -> dict[str, Any]:
    from brickdnn import service

    with translation.override(language):
        return translate(service.layers())


@functools.lru_cache(maxsize=4)
def examples(language: str) -> list[dict[str, Any]]:
    from brickdnn import service

    with translation.override(language):
        return translate(service.examples())


@functools.lru_cache(maxsize=4)
def steps(language: str) -> dict[str, Any]:
    from . import data_service

    with translation.override(language):
        return translate(data_service.step_catalog())


def catalog_texts() -> list[str]:
    """Every text of the layer and step catalogues, the examples and the training options, in
    order."""
    from django.contrib.auth.models import AnonymousUser

    from brickdnn import service

    from . import data_service, training_service

    sources: list[Any] = [
        service.layers(),
        service.examples(),
        data_service.step_catalog(),
        training_service.capabilities(AnonymousUser()),
    ]
    seen: dict[str, None] = {}
    for source in sources:
        for text in texts(source):
            seen.setdefault(text, None)
    return list(seen)
