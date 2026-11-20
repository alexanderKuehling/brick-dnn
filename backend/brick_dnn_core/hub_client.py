"""A small client for the public HTTP API of the Hugging Face Hub (huggingface.co).

Only what Brick DNN needs, with the standard library (``urllib``), so the server gets no new
dependency (``huggingface_hub`` would bring requests or httpx, tqdm, fsspec, filelock and a
native Xet extension, and it reads and caches tokens on disk by itself):

* :meth:`HubClient.whoami`: who a token belongs to and whether it may write;
* :meth:`HubClient.create_repo`: ``POST /api/repos/create``;
* :meth:`HubClient.preupload`, :meth:`HubClient.upload_lfs`: which files go to Git LFS
  (``POST /api/models/{repo}/preupload/{revision}``) and their upload with the LFS batch API
  (``POST /{repo}.git/info/lfs/objects/batch``, then the ``basic`` transfer);
* :meth:`HubClient.commit`: ``POST /api/models/{repo}/commit/{revision}`` with NDJSON lines;
* :meth:`HubClient.dataset_info`, :meth:`HubClient.dataset_tree`: a dataset and its files.

**The access token** is wrapped in :class:`HubToken`: it is sent in the ``Authorization``
header to the Hub's own host (setting ``HUGGINGFACE_URL``) and nowhere else (not to the
storage hosts the Hub hands out for uploads, not after a redirect to another host), it never
appears in ``repr``/``str`` (tracebacks, error mails, Sentry), cannot be pickled (cache,
sessions), and nothing here logs or stores it. It lives as long as the request that uses it.

Errors are :class:`HubError` with a message for the user and the HTTP status the Brick DNN
API answers with (never 401: that would log the user out of Brick DNN).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urljoin, urlsplit

logger = logging.getLogger(__name__)

DEFAULT_URL = "https://huggingface.co"
TIMEOUT = 30  # seconds per request without an answer
USER_AGENT = "BrickDNN (+https://brickdnn.com)"
MAX_ANSWER = 16 * 1024 * 1024  # bytes of a JSON answer
MAX_REDIRECTS = 3
# Text that looks like a Hugging Face token, removed from every message shown or logged.
TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{8,}")


class HubError(Exception):
    """The Hub refused or could not be reached; ``message`` is for the user, ``status`` the
    answer of the Brick DNN API and ``code`` a short name for the frontend."""

    def __init__(self, message: str, status: int = 502, code: str = "hub_error", hub_status=None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        self.hub_status = hub_status
        # A repository this request created before it failed (hub_publish.publish).
        self.created_repo: str | None = None


class HubToken:
    """A Hugging Face access token, held only for the request that uses it."""

    __slots__ = ("_value",)

    def __init__(self, value: Any):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Enter a Hugging Face access token.")
        value = value.strip()
        if len(value) > 400 or not re.fullmatch(r"[\x21-\x7e]+", value):
            raise ValueError("This does not look like a Hugging Face access token.")
        self._value = value

    @classmethod
    def optional(cls, value: Any) -> HubToken | None:
        """None for a missing or empty value, else the token (ValueError if invalid)."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return cls(value)

    def authorization(self) -> str:
        return f"Bearer {self._value}"

    def __repr__(self) -> str:
        return "HubToken(hidden)"

    __str__ = __repr__

    def __reduce__(self):
        raise TypeError("A Hugging Face access token is never stored.")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HubToken) and other._value == self._value

    def __hash__(self) -> int:
        return hash(("HubToken", self._value))


class InvalidToken:
    """Stands in for a value that is no valid token (the message says why); like HubToken
    it never shows the value."""

    __slots__ = ("message",)

    def __init__(self, message: str):
        self.message = message

    def __repr__(self) -> str:
        return "InvalidToken(hidden)"

    __str__ = __repr__


def wrap(value: Any) -> HubToken | InvalidToken | None:
    """A token argument as it is passed on: a HubToken, None if empty, or InvalidToken."""
    try:
        return HubToken.optional(value)
    except ValueError as exc:
        return InvalidToken(str(exc))


def clean(text: Any, limit: int = 300) -> str:
    """A message of the Hub, without anything that looks like a token, on one line."""
    value = TOKEN_PATTERN.sub("[token]", str(text or ""))
    value = " ".join(value.split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


@dataclass
class Answer:
    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        try:
            return json.loads(self.body or b"null")
        except ValueError:
            raise HubError("Hugging Face sent an answer that is not JSON.") from None


@dataclass
class UploadFile:
    """A file of a commit: its path in the repository and its content (a file or bytes)."""

    path_in_repo: str
    local: Path | None = None
    content: bytes | None = None
    upload_mode: str = "regular"  # or "lfs", as the Hub tells (preupload)

    def __post_init__(self):
        self._sha256: str | None = None

    @property
    def size(self) -> int:
        if self.content is not None:
            return len(self.content)
        assert self.local is not None
        return self.local.stat().st_size

    def read(self, size: int = -1) -> bytes:
        if self.content is not None:
            return self.content if size < 0 else self.content[:size]
        assert self.local is not None
        with self.local.open("rb") as source:
            return source.read() if size < 0 else source.read(size)

    def sha256(self) -> str:
        if self._sha256 is None:
            digest = hashlib.sha256()
            if self.content is not None:
                digest.update(self.content)
            else:
                assert self.local is not None
                with self.local.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
            self._sha256 = digest.hexdigest()
        return self._sha256


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """Redirects are followed by hand: the token must not go to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HubClient:
    """Requests to the Hub at ``endpoint`` (``https://huggingface.co``; a fake Hub in
    tests), with ``token`` if given. ``proxy``: an outbound proxy for these requests (by
    default none, not even from the environment)."""

    def __init__(
        self,
        endpoint: str = DEFAULT_URL,
        token: HubToken | None = None,
        *,
        proxy: str | None = None,
        timeout: float = TIMEOUT,
        deadline: float | None = None,
    ):
        self.endpoint = endpoint.rstrip("/")
        parts = urlsplit(self.endpoint)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError(f"Not a valid address of the Hugging Face Hub: {endpoint!r}")
        self._origin = (parts.scheme, parts.netloc.lower())
        self.token = token
        self.timeout = timeout
        self.deadline = deadline  # time.monotonic() after which no request starts
        proxies = {"http": proxy, "https": proxy} if proxy else {}
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler(proxies),
            urllib.request.HTTPSHandler(context=ssl.create_default_context()),
            _NoRedirects(),
        )

    # ------------------------------------------------------------------ requests

    def url(self, path: str, query: dict[str, Any] | None = None) -> str:
        return f"{self.endpoint}{path}" + (f"?{urlencode(query)}" if query else "")

    def _same_origin(self, url: str) -> bool:
        parts = urlsplit(url)
        return (parts.scheme, parts.netloc.lower()) == self._origin

    def request(
        self,
        method: str,
        url: str,
        *,
        body: bytes | Any = None,
        headers: dict[str, str] | None = None,
        length: int | None = None,
        expected: Iterable[int] = (200,),
        what: str = "",
    ) -> Answer:
        """One request; the token goes along only to the Hub's own host. Redirects of GET
        requests within the Hub are followed. Raises HubError for other answers."""
        for _ in range(MAX_REDIRECTS + 1):
            if self.deadline is not None and time.monotonic() > self.deadline:
                raise HubError(
                    "Publishing takes too long; it was stopped. Nothing more was uploaded.",
                    status=504,
                    code="timeout",
                )
            sent = {"User-Agent": USER_AGENT, **(headers or {})}
            if self.token is not None and self._same_origin(url):
                sent["Authorization"] = self.token.authorization()
            if length is not None:
                sent["Content-Length"] = str(length)
            req = urllib.request.Request(url, data=body, headers=sent, method=method)
            try:
                with self._opener.open(req, timeout=self.timeout) as response:
                    answer = Answer(
                        response.status, dict(response.headers), response.read(MAX_ANSWER + 1)
                    )
                if len(answer.body) > MAX_ANSWER:
                    raise HubError("Hugging Face sent an answer that is too long.")
            except urllib.error.HTTPError as exc:
                with exc:
                    answer = Answer(exc.code, dict(exc.headers or {}), exc.read(MAX_ANSWER))
            except (TimeoutError, OSError) as exc:
                reason = getattr(exc, "reason", None) or exc
                logger.warning("Hugging Face Hub not reachable (%s %s): %s", method, what, reason)
                raise HubError(
                    "Hugging Face cannot be reached right now. Try again later.",
                    status=502,
                    code="unreachable",
                ) from None
            if answer.status in (301, 302, 303, 307, 308) and method in ("GET", "HEAD"):
                location = answer.headers.get("Location") or answer.headers.get("location")
                if location:
                    target = urljoin(url, location)
                    if self._same_origin(target):
                        url = target
                        continue
            if answer.status in tuple(expected):
                return answer
            raise self._error(answer, what)
        raise HubError("Hugging Face redirects too often.", status=502)

    def _error(self, answer: Answer, what: str) -> HubError:
        try:
            data = json.loads(answer.body)
        except ValueError:
            data = None
        detail = ""
        if isinstance(data, dict):
            detail = clean(data.get("error") or data.get("message") or "")
        status = answer.status
        logger.info("Hugging Face Hub answered HTTP %s (%s)", status, what)
        said = f" Hugging Face says: “{detail}”" if detail else ""
        if status == 401:
            return HubError(
                "Hugging Face did not accept the access token (HTTP 401). Check that you "
                "copied all of it and that it has not been revoked." + said,
                status=400,
                code="token_invalid",
                hub_status=status,
            )
        if status == 403:
            return HubError(
                "Hugging Face refused this with the access token (HTTP 403): it may not "
                "write there, or the repository belongs to someone else." + said,
                status=403,
                code="forbidden",
                hub_status=status,
            )
        if status == 404:
            return HubError(
                "Hugging Face found nothing there (HTTP 404)." + said,
                status=404,
                code="not_found",
                hub_status=status,
            )
        if status == 409:
            return HubError(
                "This name is taken on Hugging Face (HTTP 409)." + said,
                status=409,
                code="name_taken",
                hub_status=status,
            )
        if status == 413:
            return HubError(
                "Hugging Face refused the upload as too large (HTTP 413)." + said,
                status=413,
                code="too_large",
                hub_status=status,
            )
        if status == 429:
            return HubError(
                "Hugging Face is busy (HTTP 429, too many requests). Try again later.",
                status=429,
                code="busy",
                hub_status=status,
            )
        if status == 400 or status == 422:
            return HubError(
                f"Hugging Face refused the request (HTTP {status})." + said,
                status=400,
                code="refused",
                hub_status=status,
            )
        return HubError(
            f"Hugging Face answered HTTP {status}; try again later." + said,
            status=502,
            code="hub_error",
            hub_status=status,
        )

    def get_json(self, path: str, query=None, what: str = "") -> Any:
        return self.request("GET", self.url(path, query), what=what or path).json()

    def post_json(self, path: str, data: Any, what: str = "", expected=(200,)) -> Answer:
        return self.request(
            "POST",
            self.url(path),
            body=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
            expected=expected,
            what=what or path,
        )

    # ------------------------------------------------------------------ account

    def whoami(self) -> dict[str, Any]:
        """``{"name", "orgs": [names], "role": "write" | "read" | "fineGrained" | None}``."""
        data = self.get_json("/api/whoami-v2", what="whoami")
        if not isinstance(data, dict) or not isinstance(data.get("name"), str):
            raise HubError("Hugging Face sent an unexpected answer about the token.")
        access = (
            ((data.get("auth") or {}).get("accessToken") or {}) if isinstance(data, dict) else {}
        )
        orgs = [org.get("name") for org in data.get("orgs") or [] if isinstance(org, dict)]
        return {
            "name": data["name"],
            "orgs": [name for name in orgs if isinstance(name, str)],
            "role": access.get("role") if isinstance(access, dict) else None,
        }

    # ------------------------------------------------------------------ models

    def create_repo(self, repo_id: str, private: bool) -> tuple[str, bool]:
        """Create the model repository ``owner/name``: (its URL, True). If it exists
        already: HubError with code ``name_taken`` (status 409)."""
        owner, name = repo_id.split("/", 1)
        body: dict[str, Any] = {"type": "model", "name": name, "private": bool(private)}
        body["organization"] = owner
        answer = self.post_json("/api/repos/create", body, what="create repo")
        data = answer.json()
        url = data.get("url") if isinstance(data, dict) else None
        return (url if isinstance(url, str) else f"{self.endpoint}/{repo_id}"), True

    def model_info(self, repo_id: str) -> dict[str, Any]:
        data = self.get_json(f"/api/models/{_repo_path(repo_id)}", what="model info")
        return data if isinstance(data, dict) else {}

    def preupload(self, repo_id: str, files: list[UploadFile], revision: str = "main") -> None:
        """Ask the Hub which files go to Git LFS; sets ``upload_mode`` of each file."""
        payload = {
            "files": [
                {
                    "path": file.path_in_repo,
                    "sample": base64.b64encode(file.read(512)).decode("ascii"),
                    "size": file.size,
                }
                for file in files
            ]
        }
        answer = self.post_json(
            f"/api/models/{_repo_path(repo_id)}/preupload/{quote(revision, safe='')}",
            payload,
            what="preupload",
        )
        modes = {
            item.get("path"): item.get("uploadMode")
            for item in (answer.json() or {}).get("files") or []
            if isinstance(item, dict)
        }
        for file in files:
            file.upload_mode = "lfs" if modes.get(file.path_in_repo) == "lfs" else "regular"

    def upload_lfs(self, repo_id: str, files: list[UploadFile], revision: str = "main") -> None:
        """Upload the LFS files that the Hub does not have yet (batch API, basic transfer)."""
        if not files:
            return
        payload = {
            "operation": "upload",
            "transfers": ["basic"],
            "objects": [{"oid": file.sha256(), "size": file.size} for file in files],
            "hash_algo": "sha256",
            "ref": {"name": revision},
        }
        answer = self.request(
            "POST",
            self.url(f"/{_repo_path(repo_id)}.git/info/lfs/objects/batch"),
            body=json.dumps(payload).encode(),
            headers={
                "Accept": "application/vnd.git-lfs+json",
                "Content-Type": "application/vnd.git-lfs+json",
            },
            what="lfs batch",
        )
        objects = (answer.json() or {}).get("objects") or []
        by_oid = {file.sha256(): file for file in files}
        for item in objects:
            if not isinstance(item, dict):
                continue
            file = by_oid.get(item.get("oid"))
            if file is None:
                continue
            error = item.get("error")
            if isinstance(error, dict):
                code = error.get("code")
                raise HubError(
                    f"Hugging Face refused “{file.path_in_repo}”: {clean(error.get('message'))}",
                    status=413 if code == 413 else 502,
                    code="too_large" if code == 413 else "hub_error",
                    hub_status=code,
                )
            actions = item.get("actions") or {}
            upload = actions.get("upload")
            if not upload:
                continue  # the Hub has it already
            header = dict(upload.get("header") or {})
            if "chunk_size" in header:
                raise HubError(
                    f"“{file.path_in_repo}” is too large for a single upload.",
                    status=413,
                    code="too_large",
                )
            self._put(upload["href"], file, header)
            verify = actions.get("verify")
            if verify:
                self.request(
                    "POST",
                    verify["href"],
                    body=json.dumps({"oid": file.sha256(), "size": file.size}).encode(),
                    headers={
                        "Content-Type": "application/vnd.git-lfs+json",
                        "Accept": "application/vnd.git-lfs+json",
                        **(verify.get("header") or {}),
                    },
                    what="lfs verify",
                )

    def _put(self, href: str, file: UploadFile, header: dict[str, str]) -> None:
        if urlsplit(href).scheme not in ("https", self._origin[0]):
            raise HubError("Hugging Face sent an upload address that is not secure.")
        if file.content is not None:
            self.request("PUT", href, body=file.content, headers=header, what="lfs upload")
            return
        assert file.local is not None
        with file.local.open("rb") as source:
            self.request(
                "PUT", href, body=source, headers=header, length=file.size, what="lfs upload"
            )

    def commit(
        self,
        repo_id: str,
        files: list[UploadFile],
        summary: str,
        description: str = "",
        revision: str = "main",
    ) -> dict[str, Any]:
        """One commit with ``files`` (regular ones inline, LFS files by their hash)."""
        lines = [{"key": "header", "value": {"summary": summary, "description": description}}]
        for file in files:
            if file.upload_mode == "lfs":
                lines.append(
                    {
                        "key": "lfsFile",
                        "value": {
                            "path": file.path_in_repo,
                            "algo": "sha256",
                            "oid": file.sha256(),
                            "size": file.size,
                        },
                    }
                )
            else:
                lines.append(
                    {
                        "key": "file",
                        "value": {
                            "path": file.path_in_repo,
                            "encoding": "base64",
                            "content": base64.b64encode(file.read()).decode("ascii"),
                        },
                    }
                )
        body = "\n".join(json.dumps(line) for line in lines).encode()
        answer = self.request(
            "POST",
            self.url(f"/api/models/{_repo_path(repo_id)}/commit/{quote(revision, safe='')}"),
            body=body,
            headers={"Content-Type": "application/x-ndjson"},
            what="commit",
        )
        data = answer.json()
        return data if isinstance(data, dict) else {}

    # ------------------------------------------------------------------ datasets

    def dataset_info(self, repo_id: str) -> dict[str, Any]:
        data = self.get_json(f"/api/datasets/{_repo_path(repo_id)}", what="dataset info")
        return data if isinstance(data, dict) else {}

    def dataset_tree(
        self, repo_id: str, revision: str = "main", max_entries: int = 5000
    ) -> tuple[list[dict[str, Any]], bool]:
        """The files of a dataset (recursively, pages followed); True if cut off."""
        url: str | None = self.url(
            f"/api/datasets/{_repo_path(repo_id)}/tree/{quote(revision, safe='')}",
            {"recursive": "true", "expand": "false"},
        )
        entries: list[dict[str, Any]] = []
        while url:
            answer = self.request("GET", url, what="dataset tree")
            page = answer.json()
            if not isinstance(page, list):
                raise HubError("Hugging Face sent an unexpected list of files.")
            entries.extend(item for item in page if isinstance(item, dict))
            if len(entries) >= max_entries:
                return entries[:max_entries], True
            url = _next_link(answer.headers)
            if url and not self._same_origin(url):
                url = None
        return entries, False

    def dataset_file_url(self, repo_id: str, revision: str, path: str) -> str:
        return (
            f"{self.endpoint}/datasets/{_repo_path(repo_id)}/resolve/"
            f"{quote(revision, safe='')}/{quote(path, safe='/')}"
        )


def _repo_path(repo_id: str) -> str:
    return "/".join(quote(part, safe="") for part in repo_id.split("/"))


def _next_link(headers: dict[str, str]) -> str | None:
    """The ``rel="next"`` address of a ``Link`` header."""
    value = headers.get("Link") or headers.get("link") or ""
    for part in value.split(","):
        match = re.match(r'\s*<([^>]+)>\s*;\s*rel="?next"?', part)
        if match:
            return match.group(1)
    return None
