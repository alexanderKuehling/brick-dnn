"""Importing a file of a dataset on the Hugging Face Hub (the Data tab, MCP import_data).

The user names a dataset: ``hf://datasets/<owner>/<name>`` (optionally ``@<revision>`` and a
file path, as in ``huggingface_hub``), a huggingface.co dataset page
(``https://huggingface.co/datasets/<owner>/<name>``, also its ``tree``, ``blob``, ``resolve``
and ``viewer`` pages) or just ``<owner>/<name>``. :func:`list_files` asks the Hub for its
files (``GET /api/datasets/<repo>`` and ``/tree/<revision>?recursive=true``) and keeps the
CSV, TSV, JSON, JSON Lines and Parquet files (also gzip/bzip2/xz-packed), each with the
config and split it belongs to: from the ``configs`` of the dataset card if it has them,
else from the usual names (``data/train-00000-of-00001.parquet``, ``test.csv``,
``<config>/validation.jsonl``). A dataset without such files (e.g. one with a loading
script) is listed from the Parquet files the Hub converted it to
(revision ``refs/convert/parquet``), if there are any.

:func:`import_file` then imports one file like a link import (url_import.py: the same size
limit, time limit and address checks), from ``/datasets/<repo>/resolve/<revision>/<path>``.

Public datasets need no token. For a gated or private one the user can enter a Hugging Face
access token: it goes to the Hub's own host for this one request only (not to the storage
server the download is redirected to) and is never stored (hub_client.HubToken).
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from django.conf import settings
from rest_framework import permissions, status
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle, UserRateThrottle
from rest_framework.views import APIView

from brickdnn.data import file_types

from . import data_service as data
from .hub_client import HubClient, HubError, HubToken

# Formats the Data tab reads that datasets on the Hub come in.
FORMATS = {
    ".csv": "csv",
    ".tsv": "csv",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".parquet": "parquet",
}
PACKED = (".gz", ".bz2", ".xz")
PARQUET_REVISION = "refs/convert/parquet"
MAX_FILES = 500  # files in a listing
NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?")
SPLIT_WORDS = {
    "train": "train",
    "training": "train",
    "validation": "validation",
    "valid": "validation",
    "val": "validation",
    "dev": "validation",
    "test": "test",
    "testing": "test",
    "eval": "test",
}
SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}
DISABLED = "Importing from Hugging Face is switched off on this server."


class HubInputError(ValueError):
    """Not a valid dataset reference (400); the message is for the user."""


class HubDisabled(Exception):
    """The import from the Hub is switched off on this server (403)."""


@dataclass
class DatasetRef:
    repo_id: str
    revision: str = "main"
    path: str | None = None
    config: str | None = None
    split: str | None = None

    def file_ref(self, path: str | None = None) -> str:
        """``hf://datasets/<repo>[@<revision>]/<path>``."""
        at = "" if self.revision == "main" else "@" + quote(self.revision, safe="")
        tail = path if path is not None else self.path
        return f"hf://datasets/{self.repo_id}{at}" + (f"/{tail}" if tail else "")


def enabled() -> bool:
    return bool(settings.HUGGINGFACE_ENABLED) and bool(settings.DATA_URL_IMPORT_ENABLED)


def _hub_host() -> str:
    return (urlsplit(settings.HUGGINGFACE_URL).hostname or "huggingface.co").lower()


def client(token: HubToken | None) -> HubClient:
    return HubClient(settings.HUGGINGFACE_URL, token, proxy=settings.HUGGINGFACE_PROXY or None)


def _repo(parts: list[str]) -> str:
    if not 1 <= len(parts) <= 2 or not all(
        NAME.fullmatch(p) and ".." not in p and "--" not in p for p in parts
    ):
        raise HubInputError(
            "This is not the name of a dataset on Hugging Face: use "
            "“hf://datasets/<owner>/<name>” or the link of its page."
        )
    return "/".join(parts)


def parse(text: Any, bare: bool = False) -> DatasetRef | None:
    """The dataset ``text`` names, or None if it is no Hugging Face dataset reference (any
    other link). ``bare``: also take ``<owner>/<name>``. Raises HubInputError for a
    reference that is not valid."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.startswith("hf://"):
        rest = text[len("hf://") :]
        if not rest.startswith("datasets/"):
            raise HubInputError(
                "Only datasets can be imported from Hugging Face: hf://datasets/<owner>/<name>."
            )
        segments = [s for s in rest[len("datasets/") :].split("/") if s]
        if not segments:
            raise HubInputError("Add the dataset: hf://datasets/<owner>/<name>.")
        # The revision follows the name: owner/name@rev/path (or name@rev for old ones).
        count = 2 if len(segments) >= 2 and "@" not in segments[0] else 1
        head, tail = segments[:count], segments[count:]
        revision = "main"
        if "@" in head[-1]:
            head[-1], revision = head[-1].split("@", 1)
            revision = unquote(revision) or "main"
        return DatasetRef(_repo(head), revision, "/".join(tail) or None)
    parts = urlsplit(text) if "://" in text else None
    if parts is not None:
        host = (parts.hostname or "").lower()
        if host not in ("huggingface.co", "www.huggingface.co", _hub_host()):
            return None
        segments = [unquote(s) for s in parts.path.split("/") if s]
        if not segments or segments[0] != "datasets":
            return None
        segments = segments[1:]
        # <owner>/<name> (or just <name> for old datasets), then the kind of page.
        markers = ("tree", "blob", "resolve", "raw", "viewer")
        cut = next((i for i, s in enumerate(segments) if i and s in markers), len(segments))
        if cut > 2:
            return None  # another page of the Hub (discussions ...): an ordinary link
        ref = DatasetRef(_repo(segments[:cut]))
        rest = segments[cut:]
        if rest and rest[0] == "viewer":
            ref.config = rest[1] if len(rest) > 1 else None
            ref.split = rest[2] if len(rest) > 2 else None
        elif rest:
            ref.revision = rest[1] if len(rest) > 1 else "main"
            if rest[0] != "tree":  # a folder is not a file to import
                ref.path = "/".join(rest[2:]) or None
        return ref
    if bare and re.fullmatch(r"[^/\s]+/[^/\s]+", text):
        return DatasetRef(_repo(text.split("/")))
    return None


def import_reference(text: Any, token: HubToken | None) -> DatasetRef | None:
    """What an import of ``text`` (the ``url`` of a link import) means: a dataset (file) of the
    Hub, or None for an ordinary link. While the Hub is switched off, links to huggingface.co
    stay ordinary links; ``hf://`` references and access tokens are refused
    (:class:`HubDisabled`). Raises HubInputError."""
    ref = parse(text)
    if ref is None:
        if token is not None:
            raise HubInputError("An access token is only sent along to Hugging Face (hf://…).")
        return None
    if not enabled():
        if token is not None or (isinstance(text, str) and text.strip().startswith("hf://")):
            raise HubDisabled(DISABLED)
        return None
    return ref


def file_format(path: str) -> str | None:
    """csv, json, jsonl or parquet for a file the import reads, else None."""
    suffixes = [s.lower() for s in PurePosixPath(path).suffixes]
    if suffixes and suffixes[-1] in PACKED:
        suffixes = suffixes[:-1]
    return FORMATS.get(suffixes[-1]) if suffixes else None


def importable(path: str) -> bool:
    """True if the import reads files with this ending. A file that is named directly need
    not be one of the listed formats (an Excel file or a ZIP file works as with any link)."""
    return PurePosixPath(path).suffix.lower() in set(file_types())


# --------------------------------------------------------------------------- #
# Configs and splits
# --------------------------------------------------------------------------- #


def _patterns(data_files: Any) -> list[tuple[str, list[str]]]:
    """(split, patterns) of a config's ``data_files`` in the dataset card."""
    if isinstance(data_files, str):
        return [("train", [data_files])]
    if isinstance(data_files, list) and all(isinstance(p, str) for p in data_files):
        return [("train", list(data_files))]
    found = []
    if isinstance(data_files, list):
        for item in data_files:
            if not isinstance(item, dict):
                continue
            paths = item.get("path")
            paths = [paths] if isinstance(paths, str) else paths
            if isinstance(paths, list):
                found.append((str(item.get("split") or "train"), [str(p) for p in paths]))
    elif isinstance(data_files, dict):
        for split, paths in data_files.items():
            paths = [paths] if isinstance(paths, str) else paths
            if isinstance(paths, list):
                found.append((str(split), [str(p) for p in paths]))
    return found


def _match(path: str, pattern: str) -> bool:
    pattern = pattern.lstrip("./")
    return fnmatch.fnmatchcase(path, pattern) or fnmatch.fnmatchcase(
        path, pattern.replace("**/", "")
    )


def _card_configs(info: dict[str, Any]) -> list[tuple[str, list[tuple[str, list[str]]]]]:
    card = info.get("cardData") if isinstance(info.get("cardData"), dict) else {}
    configs = card.get("configs")
    found = []
    if isinstance(configs, list):
        for config in configs:
            if isinstance(config, dict) and config.get("data_files") is not None:
                name = str(config.get("config_name") or "default")
                found.append((name, _patterns(config["data_files"])))
    return found


def _split_word(text: str) -> str | None:
    for word in re.split(r"[^a-z]+", text.lower()):
        if word in SPLIT_WORDS:
            return SPLIT_WORDS[word]
    return None


def _guess(path: str, converted: bool) -> tuple[str, str | None]:
    """(config, split) from the usual file names."""
    parts = path.split("/")
    if converted and len(parts) >= 3:  # <config>/<split>/0000.parquet
        split = parts[1].removeprefix("partial-")
        return parts[0], split
    stem = PurePosixPath(parts[-1]).name
    split = _split_word(stem)
    if split is None:
        for folder in reversed(parts[:-1]):
            split = _split_word(folder)
            if split is not None:
                break
    config = "default"
    if len(parts) >= 2 and parts[0] not in ("data", "") and _split_word(parts[0]) is None:
        config = parts[0]
    return config, split


def classify(
    paths: list[str], info: dict[str, Any], converted: bool
) -> dict[str, tuple[str, str | None]]:
    """config and split of every path."""
    configs = [] if converted else _card_configs(info)
    found: dict[str, tuple[str, str | None]] = {}
    for path in paths:
        for config, splits in configs:
            split = next(
                (name for name, patterns in splits if any(_match(path, p) for p in patterns)),
                None,
            )
            if split is not None:
                found[path] = (config, split)
                break
        else:
            found[path] = _guess(path, converted)
    return found


# --------------------------------------------------------------------------- #
# Listing and importing
# --------------------------------------------------------------------------- #


def _listing_error(exc: HubError, ref: DatasetRef, token: HubToken | None) -> HubError:
    if exc.code in ("not_found", "token_invalid") and token is None:
        return HubError(
            f"The dataset “{ref.repo_id}” was not found on Hugging Face, or it is private. "
            "Check its name; for a private or gated dataset enter an access token that may "
            "read it.",
            status=404,
            code="not_found",
        )
    if exc.code == "not_found":
        return HubError(
            f"The dataset “{ref.repo_id}” (revision “{ref.revision}”) was not found on "
            "Hugging Face, or the access token may not read it.",
            status=404,
            code="not_found",
        )
    return exc


def list_files(ref: DatasetRef, token: HubToken | None) -> dict[str, Any]:
    """``HubDatasetFiles`` (frontend/src/api/types.ts): the importable files of a dataset."""
    hub = client(token)
    try:
        info = hub.dataset_info(ref.repo_id)
        revision = ref.revision
        entries, truncated = hub.dataset_tree(ref.repo_id, revision)
        converted = revision == PARQUET_REVISION
        files = [e for e in entries if e.get("type") == "file" and file_format(str(e.get("path")))]
        if not files and revision == "main":
            try:
                entries, truncated = hub.dataset_tree(ref.repo_id, PARQUET_REVISION)
            except HubError as exc:
                if exc.code != "not_found":
                    raise
            else:
                files = [
                    e
                    for e in entries
                    if e.get("type") == "file" and file_format(str(e.get("path")))
                ]
                if files:
                    revision, converted = PARQUET_REVISION, True
    except HubError as exc:
        raise _listing_error(exc, ref, token) from None
    repo_id = str(info.get("id") or ref.repo_id)
    if not _valid_repo(repo_id):
        repo_id = ref.repo_id
    shown = DatasetRef(repo_id, revision)
    paths = [str(e["path"]) for e in files]
    groups = classify(paths, info, converted)
    items = []
    for entry in files:
        path = str(entry["path"])
        config, split = groups[path]
        size = entry.get("size")
        lfs = entry.get("lfs")
        if isinstance(lfs, dict) and isinstance(lfs.get("size"), int):
            size = lfs["size"]
        items.append(
            {
                "path": path,
                "size": size if isinstance(size, int) else None,
                "format": file_format(path),
                "config": config,
                "split": split,
                "ref": shown.file_ref(path),
            }
        )
    items.sort(
        key=lambda f: (
            f["config"] != "default",
            f["config"],
            SPLIT_ORDER.get(f["split"] or "", 3),
            f["path"],
        )
    )
    configs = list(dict.fromkeys(f["config"] for f in items))
    splits = list(dict.fromkeys(f["split"] for f in items if f["split"]))
    gated = info.get("gated")
    return {
        "repo_id": repo_id,
        "revision": revision,
        "converted": converted,
        "page": f"{hub.endpoint}/datasets/{repo_id}",
        "private": bool(info.get("private")),
        "gated": gated if gated in ("auto", "manual") else False,
        "files": items[:MAX_FILES],
        "total": len(items),
        "truncated": truncated or len(items) > MAX_FILES,
        "configs": configs,
        "splits": splits,
        "config": ref.config if ref.config in configs else None,
        "split": ref.split if ref.split in splits else None,
        "max_upload_mb": settings.DATA_MAX_UPLOAD_MB,
    }


def _valid_repo(repo_id: str) -> bool:
    try:
        _repo(repo_id.split("/"))
    except HubInputError:
        return False
    return True


def _default_name(ref: DatasetRef, path: str) -> str:
    """ "imdb · train" for data/train-00000-of-00001.parquet of stanfordnlp/imdb."""
    config, split = _guess(path, ref.revision == PARQUET_REVISION)
    words = [ref.repo_id.split("/")[-1]]
    if config != "default":
        words.append(config)
    words.append(split or PurePosixPath(path).name.split(".")[0])
    return " · ".join(words)[:120]


def import_file(user, ref: DatasetRef, token: HubToken | None, fields: dict[str, Any]):
    """Import the file ``ref.path`` like a link import (data_service.import_url)."""
    if not ref.path:
        raise data.DataInputError(
            "Choose a file of the dataset: hf://datasets/<owner>/<name>/<path of the file>."
        )
    if not importable(ref.path):
        raise data.DataInputError(
            f"“{PurePosixPath(ref.path).name}” is not a data file that can be imported: choose "
            "a CSV, TSV, JSON, JSON Lines or Parquet file of the dataset."
        )
    url = client(None).dataset_file_url(ref.repo_id, ref.revision, ref.path)
    fields = dict(fields)
    if not (isinstance(fields.get("name"), str) and fields["name"].strip()):
        fields["name"] = _default_name(ref, ref.path)
    auth = (settings.HUGGINGFACE_URL, token)  # also says: a download from the Hub
    return data.import_url(user, url, fields, auth=auth)


# --------------------------------------------------------------------------- #
# API: POST /api/data/huggingface/files/
# --------------------------------------------------------------------------- #


class DatasetFilesView(APIView):
    """``POST {"dataset", "hf_token"?}``: the importable files of a dataset (POST, so that a
    token never ends up in a URL). The import itself goes through ``POST
    /api/data/sources/`` with the file's ``ref`` as ``url``."""

    permission_classes = [permissions.IsAuthenticated]
    throttle_scope = "huggingface"
    throttle_classes = [UserRateThrottle, ScopedRateThrottle]

    def post(self, request: Request) -> Response:
        reason = data.unavailable_reason()
        if reason is not None:
            return Response({"detail": reason}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        if not enabled():
            return Response({"detail": DISABLED}, status=status.HTTP_403_FORBIDDEN)
        body = request.data if isinstance(request.data, dict) else {}
        try:
            token = HubToken.optional(body.pop("hf_token", None))
            ref = parse(body.get("dataset"), bare=True)
            if ref is None:
                raise HubInputError(
                    "Enter a dataset of Hugging Face: hf://datasets/<owner>/<name>, "
                    "<owner>/<name> or the link of its page on huggingface.co."
                )
            listing = list_files(ref, token)
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except HubError as exc:
            return Response({"detail": exc.message, "code": exc.code}, status=exc.status)
        return Response(listing)
