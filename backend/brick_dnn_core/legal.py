"""Imprint, privacy policy and terms of use: Markdown files the operator puts on the server.

The files live in ``settings.LEGAL_DIR`` (``BRICKDNN_LEGAL_DIR``, by default ``legal/`` in the
data directory) and are read on every request, so a changed file is online at once. It can name
several folders, separated by ``:``: each page comes from the first folder that has it (the
server takes files on the server first, then the texts of its domain in the repository). A
page whose file is missing or empty is not offered. If ``terms.md`` exists, sign-up asks to accept
the terms (see ``RegisterSerializer``); the acceptance is stored with the version, a hash of
the file. Templates: ``deploy/legal/``; more in docs/privacy.md.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone as dt_timezone
from pathlib import Path
from typing import Any

from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.request import Request
from rest_framework.response import Response

logger = logging.getLogger(__name__)

PAGES = {"imprint": "Imprint", "privacy": "Privacy policy", "terms": "Terms of use"}
# Larger files are ignored (and logged): legal texts are a few dozen kilobytes.
MAX_BYTES = 512 * 1024


@dataclass
class LegalPage:
    page: str
    content: str
    # The first 16 hex digits of the SHA-256 of the file: changes with every edit.
    version: str
    updated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "title": PAGES[self.page],
            "content": self.content,
            "version": self.version,
            "updated_at": self.updated_at,
        }


def legal_dirs() -> list[Path]:
    """The folders of ``LEGAL_DIR`` (separated by ``:``), in order."""
    return [Path(part) for part in str(settings.LEGAL_DIR).split(os.pathsep) if part.strip()]


def read_page(page: str) -> LegalPage | None:
    """The page from ``<folder>/<page>.md`` of the first folder of ``LEGAL_DIR`` that has a
    non-empty one, or None if it is not set up."""
    if page not in PAGES:
        return None
    for folder in legal_dirs():
        found = _read_file(page, folder / f"{page}.md")
        if found is not None:
            return found
    return None


def _read_file(page: str, path: Path) -> LegalPage | None:
    try:
        if not path.is_file():
            return None
        if path.stat().st_size > MAX_BYTES:
            logger.warning("%s is larger than %d KB and is not shown.", path, MAX_BYTES // 1024)
            return None
        raw = path.read_bytes()
        modified = path.stat().st_mtime
    except OSError:
        logger.warning("Could not read %s.", path, exc_info=True)
        return None
    content = raw.decode("utf-8", errors="replace").lstrip("﻿").strip()
    if not content:
        return None
    return LegalPage(
        page=page,
        content=content,
        version=hashlib.sha256(raw).hexdigest()[:16],
        updated_at=datetime.fromtimestamp(modified, tz=dt_timezone.utc),
    )


def versions() -> dict[str, str | None]:
    """The version of every page, None for pages that are not set up."""
    return {page: (found.version if (found := read_page(page)) else None) for page in PAGES}


def terms_required() -> bool:
    """Sign-up asks to accept the terms of use when there are some."""
    return read_page("terms") is not None


def info() -> dict[str, Any]:
    """``LegalInfo`` (frontend/src/api/types.ts): which pages exist and what sign-up needs."""
    current = versions()
    return {
        "pages": {page: version is not None for page, version in current.items()},
        "terms_version": current["terms"],
        "privacy_version": current["privacy"],
        "accept_terms_required": current["terms"] is not None,
        # Automatic deletion (0: off), so the app can tell people how long uploads are kept.
        "retention": {
            "unused_uploads_days": max(0, settings.DELETE_UNUSED_UPLOADS_DAYS),
            "inactive_accounts_days": max(0, settings.DELETE_INACTIVE_ACCOUNTS_DAYS),
        },
    }


@api_view(["GET"])
def legal_index(request: Request) -> Response:
    """Which legal pages this server has (public)."""
    return Response(info())


@api_view(["GET"])
def legal_page(request: Request, page: str) -> Response:
    """One page as Markdown (public): ``{"page", "title", "content", "version", "updated_at"}``."""
    found = read_page(page)
    if found is None:
        return Response(
            {"detail": "This page has not been set up on this server."},
            status=status.HTTP_404_NOT_FOUND,
        )
    return Response(found.to_dict())
