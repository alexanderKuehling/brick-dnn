"""Downloading a data file from a link (http/https) for an import, safely.

The server fetches the link, so it must not become a way into internal services
(server-side request forgery, SSRF):

* only ``http`` and ``https``; links with a user name or password are refused, and no
  cookies, credentials or other headers of the user are sent;
* the host name is resolved once and **every** address must be public: private, loopback,
  link-local (e.g. cloud metadata at 169.254.169.254), multicast, reserved, shared and other
  special IPv4/IPv6 addresses are refused, also when they hide in IPv4-mapped, 6to4,
  Teredo or NAT64 IPv6 addresses. The connection then goes to exactly the checked address
  (a second DNS lookup could answer differently), with TLS certificates verified for the
  host name;
* redirects are followed by hand, at most ``MAX_REDIRECTS``, and every new location is
  checked the same way;
* connect and read timeouts and an overall time limit; the size limit of uploads applies
  while the file streams in (a larger ``Content-Length`` is refused right away);
* proxies from the environment are not used (a proxy would resolve names itself).

``allow_private`` (setting ``DATA_URL_ALLOW_PRIVATE``) lifts the address check for local
installations, e.g. to import from a file server in the same network.

Links that people share are rewritten to the file itself: Google Sheets to its CSV export
(``gid`` selects the tab), Google Drive files to their download, GitHub ``blob`` pages to
raw.githubusercontent.com, Hugging Face ``blob`` pages to ``resolve``, Dropbox ``dl=0`` to
``dl=1``.
"""

from __future__ import annotations

import email.message
import http.client
import ipaddress
import re
import socket
import ssl
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, quote, unquote, urlencode, urljoin, urlsplit, urlunsplit

from brickdnn.data import COMPRESSED_SUFFIXES, FORMAT_SUFFIXES

MAX_URL_LENGTH = 2048
MAX_REDIRECTS = 5
CONNECT_TIMEOUT = 10  # seconds
READ_TIMEOUT = 30  # seconds without any data
TIME_LIMIT = 180  # seconds for the whole download
CHUNK = 64 * 1024
USER_AGENT = "BrickDNN-data-import (+https://github.com/alexanderKuehling/brick-dnn)"

# File name endings for common content types, for links whose name has none.
CONTENT_TYPE_SUFFIXES = {
    "text/csv": ".csv",
    "application/csv": ".csv",
    "text/tab-separated-values": ".tsv",
    "text/plain": ".txt",
    "application/json": ".json",
    "application/geo+json": ".json",
    "application/x-ndjson": ".jsonl",
    "application/jsonl": ".jsonl",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.oasis.opendocument.spreadsheet": ".ods",
    "application/vnd.apache.parquet": ".parquet",
    "application/x-parquet": ".parquet",
    "application/vnd.apache.arrow.file": ".arrow",
    "application/vnd.apache.arrow.stream": ".arrows",
    "application/gzip": ".gz",
    "application/x-gzip": ".gz",
    "application/zip": ".zip",
}
# IPv6 prefixes that carry an IPv4 address in their last 32 bits (NAT64).
_NAT64 = (ipaddress.ip_network("64:ff9b::/96"), ipaddress.ip_network("64:ff9b:1::/48"))


class LinkError(ValueError):
    """The link cannot be imported; the message is shown to the user."""


@dataclass
class Download:
    path: Path
    filename: str  # from Content-Disposition or the link; with a known ending if possible
    url: str  # the link as given
    final_url: str  # after rewriting and redirects
    size: int
    content_type: str


# --------------------------------------------------------------------------- #
# Links
# --------------------------------------------------------------------------- #


def check_url(url: object) -> str:
    """The link, stripped, if it is a http(s) link without credentials; else LinkError."""
    if not isinstance(url, str) or not url.strip():
        raise LinkError("Enter the link (URL) of a data file.")
    url = url.strip()
    if len(url) > MAX_URL_LENGTH:
        raise LinkError(f"The link is too long (at most {MAX_URL_LENGTH:,} characters).")
    if re.search(r"[\x00-\x20\x7f]", url):
        raise LinkError("The link must not contain spaces or control characters.")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        raise LinkError("This is not a valid link.") from None
    if parts.scheme.lower() not in ("http", "https"):
        raise LinkError("Only http:// and https:// links can be imported.")
    if not parts.hostname:
        raise LinkError("The link has no host name.")
    if parts.username is not None or parts.password is not None:
        raise LinkError("Links with a user name or password cannot be imported. Use a public link.")
    if port == 0:
        raise LinkError("This is not a valid link.")
    return url


def rewrite(url: str) -> str:
    """The direct download link for share links of Google Sheets / Drive, GitHub, Hugging
    Face and Dropbox; other links unchanged."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    path = parts.path
    query = parse_qs(parts.query)
    if host == "docs.google.com":
        match = re.match(r"/spreadsheets/d/(e/)?([\w-]+)", path)
        if match:
            published, key = match.groups()
            gid = (query.get("gid") or re.findall(r"gid=(\d+)", parts.fragment) or [None])[0]
            if published:
                params = {"output": "csv", **({"gid": gid} if gid else {})}
                return f"https://docs.google.com/spreadsheets/d/e/{key}/pub?{urlencode(params)}"
            if path.rstrip("/").endswith("/export"):
                return url  # already an export link (maybe xlsx)
            params = {"format": "csv", **({"gid": gid} if gid else {})}
            return f"https://docs.google.com/spreadsheets/d/{key}/export?{urlencode(params)}"
    if host == "drive.google.com":
        match = re.match(r"/file/d/([\w-]+)", path)
        key = match.group(1) if match else (query.get("id") or [None])[0]
        if key:
            return f"https://drive.google.com/uc?{urlencode({'export': 'download', 'id': key})}"
    if host == "github.com":
        match = re.match(r"/([^/]+)/([^/]+)/(?:blob|raw)/(.+)", path)
        if match:
            owner, repo, rest = match.groups()
            return f"https://raw.githubusercontent.com/{owner}/{repo}/{rest}"
    if host in ("huggingface.co", "www.huggingface.co"):
        match = re.match(r"(/datasets/[^/]+/[^/]+)/blob/(.+)", path)
        if match:
            return urlunsplit(("https", "huggingface.co", f"{match[1]}/resolve/{match[2]}", "", ""))
    if host in ("www.dropbox.com", "dropbox.com") and query.get("dl") == ["0"]:
        query["dl"] = ["1"]
        return urlunsplit(parts._replace(query=urlencode(query, doseq=True)))
    return url


def address_allowed(address: str) -> bool:
    """True if ``address`` (IPv4 or IPv6) is a public internet address."""
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = ip.ipv4_mapped or ip.sixtofour or (ip.teredo[1] if ip.teredo else None)
        if embedded is None and any(ip in network for network in _NAT64):
            embedded = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        if embedded is not None and not address_allowed(str(embedded)):
            return False
    return ip.is_global and not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def resolve(host: str, port: int) -> list[str]:
    """The addresses of ``host`` (all of them, as the system resolver returns them)."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError):
        raise LinkError(f"The server “{host}” was not found. Check the link.") from None
    addresses: list[str] = []
    for info in infos:
        address = str(info[4][0])
        if address not in addresses:
            addresses.append(address)
    if not addresses:
        raise LinkError(f"The server “{host}” was not found. Check the link.")
    return addresses


def _addresses(host: str, port: int, allow_private: bool) -> list[str]:
    addresses = resolve(host, port)
    if not allow_private and not all(address_allowed(a) for a in addresses):
        raise LinkError(
            f"“{host}” is not a public internet address. Only files on the public internet "
            "can be imported."
        )
    return addresses


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #


def fetch(
    url: object,
    folder: Path,
    *,
    max_bytes: int,
    allow_private: bool = False,
    time_limit: float = TIME_LIMIT,
) -> Download:
    """Download the file at ``url`` into ``folder`` (see the module docstring for the
    checks). Raises :class:`LinkError` with a message for the user."""
    given = check_url(url)
    current = rewrite(given)
    deadline = time.monotonic() + time_limit
    for _ in range(MAX_REDIRECTS + 1):
        current = check_url(current)
        parts = urlsplit(current)
        https = parts.scheme.lower() == "https"
        host = _ascii_host(parts.hostname or "")
        port = parts.port or (443 if https else 80)
        addresses = _addresses(host, port, allow_private)
        connection, response = _get(host, port, https, addresses, _target(parts))
        try:
            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader("Location")
                if not location:
                    raise LinkError("The server sent a redirect without a target.")
                current = urljoin(current, location.strip())
                continue
            _check_status(response, given)
            return _save(response, given, current, folder, max_bytes, deadline)
        finally:
            response.close()
            connection.close()
    raise LinkError(f"The link redirects more than {MAX_REDIRECTS} times.")


def _ascii_host(host: str) -> str:
    try:
        return host.encode("idna").decode("ascii") if not host.isascii() else host
    except UnicodeError:
        raise LinkError("The host name of the link is not valid.") from None


def _target(parts) -> str:
    """Path and query of the request, with characters that need it percent-encoded."""
    path = quote(parts.path or "/", safe="/%:@!$&'()*+,;=-._~")
    query = quote(parts.query, safe="/%:@!$&'()*+,;=-._~?")
    return path + (f"?{query}" if query else "")


def _get(host: str, port: int, https: bool, addresses: list[str], target: str):
    """Send the request to the first checked address that answers."""
    last_error: Exception | None = None
    for address in addresses:
        if https:
            connection: http.client.HTTPConnection = http.client.HTTPSConnection(
                host, port, timeout=READ_TIMEOUT, context=ssl.create_default_context()
            )
        else:
            connection = http.client.HTTPConnection(host, port, timeout=READ_TIMEOUT)
        # Connect to the checked address, not to whatever the name resolves to now. TLS
        # (server name and certificate) still uses the host name.
        connection._create_connection = _connector(address)  # type: ignore[attr-defined]
        try:
            connection.request("GET", target, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
            return connection, connection.getresponse()
        except ssl.SSLCertVerificationError:
            connection.close()
            raise LinkError(
                f"The secure connection to “{host}” failed: its certificate is not valid."
            ) from None
        except TimeoutError:
            connection.close()
            last_error = LinkError(f"The server “{host}” did not answer in time.")
        except (OSError, http.client.HTTPException) as exc:
            connection.close()
            last_error = exc
    if isinstance(last_error, LinkError):
        raise last_error
    reason = getattr(last_error, "strerror", None) or str(last_error) or "no connection"
    raise LinkError(f"The server “{host}” cannot be reached ({reason}).")


def _connector(address: str):
    def connect(_address, timeout=None, source_address=None):
        sock = socket.create_connection((address, _address[1]), CONNECT_TIMEOUT)
        sock.settimeout(READ_TIMEOUT)
        return sock

    return connect


def _check_status(response: http.client.HTTPResponse, given: str) -> None:
    status = response.status
    google = (urlsplit(given).hostname or "").endswith("google.com")
    if status in (401, 403):
        hint = (
            " For Google Sheets: Share → “Anyone with the link” (Viewer)."
            if google
            else " Only links that work without logging in can be imported."
        )
        raise LinkError(f"The server refused access (HTTP {status}).{hint}")
    if status == 404:
        raise LinkError("Nothing was found at this link (HTTP 404). Check the link.")
    if status == 429:
        raise LinkError("The server is busy (HTTP 429, too many requests). Try again later.")
    if status != 200:
        reason = f" {response.reason}" if response.reason else ""
        raise LinkError(f"The server answered HTTP {status}{reason}; nothing was imported.")
    content_type = (response.getheader("Content-Type") or "").split(";")[0].strip().lower()
    if content_type in ("text/html", "application/xhtml+xml"):
        if google:
            raise LinkError(
                "Google answered with a web page instead of the data: the sheet is probably "
                "not public. Share it with “Anyone with the link” (Viewer) and try again."
            )
        raise LinkError(
            "The link leads to a web page, not to a data file. Use the link of the file "
            "itself (e.g. “Raw” on GitHub, the download link on Hugging Face)."
        )


def _save(response, given: str, final: str, folder: Path, max_bytes: int, deadline: float):
    length = response.getheader("Content-Length")
    too_large = LinkError(
        f"The file at the link is larger than {max_bytes / 1024 / 1024:g} MB, the limit for "
        "imports on this server."
    )
    if length and length.strip().isdigit() and int(length) > max_bytes:
        raise too_large
    path = folder / "download"
    size = 0
    try:
        with path.open("wb") as out:
            while chunk := response.read(CHUNK):
                size += len(chunk)
                if size > max_bytes:
                    raise too_large
                if time.monotonic() > deadline:
                    raise LinkError("The download takes too long; nothing was imported.")
                out.write(chunk)
    except TimeoutError:
        raise LinkError("The server stopped sending the file.") from None
    except (OSError, http.client.HTTPException) as exc:
        raise LinkError(f"The download was interrupted ({exc}).") from None
    if size == 0:
        raise LinkError("The file at the link is empty.")
    content_type = (response.getheader("Content-Type") or "").split(";")[0].strip().lower()
    filename = _filename(response.getheader("Content-Disposition"), final, content_type)
    return Download(path, filename, given, final, size, content_type)


def _filename(disposition: str | None, url: str, content_type: str) -> str:
    """A safe file name from Content-Disposition or the link, with a known ending if the
    content type tells one."""
    name = None
    if disposition:
        message = email.message.Message()
        message["Content-Disposition"] = disposition
        name = message.get_filename()
    if not name:
        name = unquote(PurePosixPath(urlsplit(url).path).name)
    name = re.sub(r"[\x00-\x1f\x7f/\\]+", "_", name or "").strip(" .")[:200] or "download"
    suffixes = [s.lower() for s in PurePosixPath(name).suffixes]
    known = any(s in FORMAT_SUFFIXES or s in COMPRESSED_SUFFIXES for s in suffixes[-1:])
    if not known and content_type in CONTENT_TYPE_SUFFIXES:
        name += CONTENT_TYPE_SUFFIXES[content_type]
    return name
