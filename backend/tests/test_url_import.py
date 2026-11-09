"""Downloads for imports from links: link rewriting (Google Sheets ...), the address checks
against server-side request forgery, redirects, limits and error messages. The downloads
run against a local HTTP server."""

from __future__ import annotations

import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from brick_dnn_core import url_import
from brick_dnn_core.url_import import LinkError, address_allowed, check_url, fetch, rewrite

CSV = b"a,b\n1,2\n3,4\n"


class Server:
    """A local HTTP server: ``routes[path] = (status, headers, body)``."""

    def __init__(self):
        self.routes: dict[str, tuple[int, dict[str, str], bytes]] = {}
        self.requests: list[dict[str, str]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                server.requests.append({"path": self.path, **dict(self.headers)})
                status, headers, body = server.routes.get(self.path, (404, {}, b"missing"))
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                if "Content-Length" not in headers:
                    self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    server = Server()
    yield server
    server.close()


def download(url, tmp_path, max_bytes=1024 * 1024, allow_private=True, **kwargs):
    return fetch(url, tmp_path, max_bytes=max_bytes, allow_private=allow_private, **kwargs)


# --------------------------------------------------------------------------- #
# Links
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://docs.google.com/spreadsheets/d/1AbC-dEf_9/edit?usp=sharing",
            "https://docs.google.com/spreadsheets/d/1AbC-dEf_9/export?format=csv",
        ),
        (
            "https://docs.google.com/spreadsheets/d/1AbC/edit#gid=12345",
            "https://docs.google.com/spreadsheets/d/1AbC/export?format=csv&gid=12345",
        ),
        (
            "https://docs.google.com/spreadsheets/d/1AbC/edit?gid=7#gid=7",
            "https://docs.google.com/spreadsheets/d/1AbC/export?format=csv&gid=7",
        ),
        (
            "https://docs.google.com/spreadsheets/d/1AbC/export?format=xlsx",
            "https://docs.google.com/spreadsheets/d/1AbC/export?format=xlsx",
        ),
        (
            "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ/pubhtml?gid=3&single=true",
            "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ/pub?output=csv&gid=3",
        ),
        (
            "https://drive.google.com/file/d/1XyZ/view?usp=sharing",
            "https://drive.google.com/uc?export=download&id=1XyZ",
        ),
        (
            "https://github.com/owner/repo/blob/main/data/titanic.csv",
            "https://raw.githubusercontent.com/owner/repo/main/data/titanic.csv",
        ),
        (
            "https://huggingface.co/datasets/org/name/blob/main/train.parquet",
            "https://huggingface.co/datasets/org/name/resolve/main/train.parquet",
        ),
        (
            "https://www.dropbox.com/s/abc/data.csv?dl=0",
            "https://www.dropbox.com/s/abc/data.csv?dl=1",
        ),
        ("https://example.org/data.csv?x=1", "https://example.org/data.csv?x=1"),
    ],
)
def test_share_links_are_rewritten(url, expected):
    assert rewrite(url) == expected


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("", "Enter the link"),
        (None, "Enter the link"),
        ("ftp://example.org/data.csv", "Only http:// and https://"),
        ("file:///etc/passwd", "Only http:// and https://"),
        ("javascript:alert(1)", "Only http:// and https://"),
        ("https://user:secret@example.org/data.csv", "user name or password"),
        ("https://token@example.org/data.csv", "user name or password"),
        ("https:///data.csv", "no host name"),
        ("https://example.org/a b.csv", "spaces"),
        ("https://example.org:99999/x", "not a valid link"),
        ("https://example.org/" + "x" * 3000, "too long"),
    ],
)
def test_invalid_links(url, message):
    with pytest.raises(LinkError, match=message):
        check_url(url)


@pytest.mark.parametrize(
    ("address", "allowed"),
    [
        ("93.184.215.14", True),
        ("2606:4700:4700::1111", True),
        ("127.0.0.1", False),
        ("10.1.2.3", False),
        ("172.16.0.1", False),
        ("192.168.178.1", False),
        ("169.254.169.254", False),  # cloud metadata
        ("100.64.0.1", False),  # shared address space (carrier-grade NAT)
        ("0.0.0.0", False),
        ("224.0.0.1", False),
        ("255.255.255.255", False),
        ("192.0.2.1", False),  # documentation
        ("::1", False),
        ("::", False),
        ("fe80::1", False),
        ("fe80::1%eth0", False),
        ("fc00::1", False),
        ("ff02::1", False),
        ("::ffff:127.0.0.1", False),  # IPv4-mapped
        ("::ffff:10.0.0.1", False),
        ("2002:7f00:1::", False),  # 6to4 of 127.0.0.1
        ("64:ff9b::a9fe:a9fe", False),  # NAT64 of 169.254.169.254
        ("not an address", False),
    ],
)
def test_only_public_addresses_are_allowed(address, allowed):
    assert address_allowed(address) is allowed


# --------------------------------------------------------------------------- #
# Downloads from the local server (private addresses allowed, as for local installations)
# --------------------------------------------------------------------------- #


def test_download_with_file_name_and_redirects(server, tmp_path):
    server.routes["/files/data.csv"] = (200, {"Content-Type": "text/csv"}, CSV)
    server.routes["/latest"] = (302, {"Location": "/files/data.csv"}, b"")
    server.routes["/export"] = (
        200,
        {
            "Content-Type": "text/csv; charset=utf-8",
            "Content-Disposition": "attachment; filename*=UTF-8''Umsatz%20K%C3%B6ln.csv",
        },
        CSV,
    )
    server.routes["/api/table"] = (200, {"Content-Type": "application/json"}, b"[]")

    result = download(f"{server.url}/latest", tmp_path)
    assert result.path.read_bytes() == CSV and result.size == len(CSV)
    assert result.filename == "data.csv" and result.url == f"{server.url}/latest"
    assert result.final_url == f"{server.url}/files/data.csv"

    assert download(f"{server.url}/export", tmp_path).filename == "Umsatz Köln.csv"
    # No name ending in the link: the content type gives one.
    assert download(f"{server.url}/api/table", tmp_path).filename == "table.json"

    headers = server.requests[0]
    assert headers["User-Agent"].startswith("BrickDNN")
    assert headers["Accept-Encoding"] == "identity"
    assert "Cookie" not in headers and "Authorization" not in headers


def test_too_many_redirects(server, tmp_path):
    for i in range(10):
        server.routes[f"/r{i}"] = (301, {"Location": f"/r{i + 1}"}, b"")
    with pytest.raises(LinkError, match="redirects more than 5 times"):
        download(f"{server.url}/r0", tmp_path)
    assert len(server.requests) == 6


@pytest.mark.parametrize(
    ("status", "headers", "body", "message"),
    [
        (404, {}, b"", "Nothing was found at this link"),
        (403, {}, b"", "refused access"),
        (500, {}, b"", "HTTP 500"),
        (200, {"Content-Type": "text/html"}, b"<html></html>", "leads to a web page"),
        (200, {"Content-Type": "text/csv"}, b"", "is empty"),
        (302, {}, b"", "redirect without a target"),
    ],
)
def test_server_answers_that_are_not_a_file(server, tmp_path, status, headers, body, message):
    server.routes["/x.csv"] = (status, headers, body)
    with pytest.raises(LinkError, match=message):
        download(f"{server.url}/x.csv", tmp_path)


def test_size_limit_while_downloading(server, tmp_path):
    big = b"1,2\n" * 1000
    server.routes["/declared.csv"] = (200, {}, big)
    with pytest.raises(LinkError, match="larger than"):
        download(f"{server.url}/declared.csv", tmp_path, max_bytes=1000)
    # Without Content-Length (streamed until the connection closes): counted while reading.
    server.routes["/streamed.csv"] = (200, {"Content-Length": ""}, big)
    with pytest.raises(LinkError, match="larger than"):
        download(f"{server.url}/streamed.csv", tmp_path, max_bytes=1000)
    assert sorted(p.name for p in tmp_path.iterdir()) in ([], ["download"])


def test_time_limit(server, tmp_path):
    server.routes["/slow.csv"] = (200, {}, CSV)
    with pytest.raises(LinkError, match="takes too long"):
        download(f"{server.url}/slow.csv", tmp_path, time_limit=-1)


def test_unreachable_server(tmp_path):
    with socket.socket() as sock:  # a port where nobody listens
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with pytest.raises(LinkError, match="cannot be reached"):
        download(f"http://127.0.0.1:{port}/data.csv", tmp_path)


# --------------------------------------------------------------------------- #
# Server-side request forgery
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://localhost:8000/api/",
        "http://[::1]/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/data.csv",
        "http://2130706433/",  # 127.0.0.1 as a number
        "http://0x7f.1/",
        "http://[::ffff:127.0.0.1]/",
    ],
)
def test_private_addresses_are_refused(url, tmp_path, monkeypatch):
    def no_connection(*args, **kwargs):
        raise AssertionError("must not connect")

    monkeypatch.setattr(socket, "create_connection", no_connection)
    with pytest.raises(LinkError, match="not a public internet address"):
        download(url, tmp_path, allow_private=False)


def fake_dns(monkeypatch, names: dict[str, list[str]]):
    real = socket.getaddrinfo

    def getaddrinfo(host, port, *args, **kwargs):
        if host in names:
            return [
                (
                    socket.AF_INET6 if ":" in a else socket.AF_INET,
                    socket.SOCK_STREAM,
                    6,
                    "",
                    (a, port),
                )
                for a in names[host]
            ]
        return real(host, port, *args, **kwargs)

    monkeypatch.setattr(url_import.socket, "getaddrinfo", getaddrinfo)


def test_host_names_resolving_to_private_addresses_are_refused(tmp_path, monkeypatch):
    fake_dns(
        monkeypatch,
        {
            "internal.example": ["10.0.0.7"],
            # One public and one private address: refused (the connection could use either).
            "mixed.example": ["93.184.215.14", "192.168.0.10"],
        },
    )
    for host in ("internal.example", "mixed.example"):
        with pytest.raises(LinkError, match="not a public internet address"):
            download(f"http://{host}/data.csv", tmp_path, allow_private=False)


@pytest.mark.parametrize(
    "target",
    [
        "http://127.0.0.1:{port}/data.csv",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]:{port}/data.csv",
        "http://internal.example:{port}/data.csv",
    ],
)
def test_redirects_to_private_addresses_are_refused(server, tmp_path, monkeypatch, target):
    """The first host is public (simulated), its redirect leads inside: refused."""
    fake_dns(monkeypatch, {"internal.example": ["10.0.0.1"]})
    checked = url_import._addresses

    def addresses(host, port, allow_private):
        if host == "public.example":
            return ["127.0.0.1"]  # pretend the test server is on the internet
        return checked(host, port, allow_private)

    monkeypatch.setattr(url_import, "_addresses", addresses)
    server.routes["/start"] = (302, {"Location": target.format(port=server.port)}, b"")
    server.routes["/data.csv"] = (200, {}, CSV)
    with pytest.raises(LinkError, match="not a public internet address"):
        download(f"http://public.example:{server.port}/start", tmp_path, allow_private=False)
    assert [r["path"] for r in server.requests] == ["/start"]


def test_the_checked_address_is_the_one_connected_to(server, tmp_path, monkeypatch):
    """No second DNS lookup: the request goes to the address that was checked, with the
    host name in the Host header (allowed here because private addresses are allowed)."""
    fake_dns(monkeypatch, {"files.example": ["127.0.0.1"]})
    server.routes["/data.csv"] = (200, {}, CSV)
    lookups = []
    real = url_import.resolve
    monkeypatch.setattr(url_import, "resolve", lambda h, p: lookups.append(h) or real(h, p))
    result = download(f"http://files.example:{server.port}/data.csv", tmp_path)
    assert result.path.read_bytes() == CSV
    assert server.requests[0]["Host"] == f"files.example:{server.port}"
    assert lookups == ["files.example"]
