"""A fake Hugging Face Hub for the tests (hub_client.py, hub_publish.py, hub_datasets.py).

Two local servers: the Hub itself (the API, Git LFS batch, dataset downloads) and a storage
server on another port, where LFS uploads go and downloads are redirected to, like the
buckets and CDNs of the real Hub. Both record every request with its headers, so tests can
check what was sent where (and that the token never reached the storage server).
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

GOOD_TOKEN = "hf_" + "A" * 34
READ_TOKEN = "hf_" + "R" * 34
OTHER_TOKEN = "hf_" + "B" * 34  # valid, but not for the gated dataset


class _Server:
    def __init__(self, handler):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class FakeHub:
    """The Hub at ``url`` and its storage at ``storage_url``."""

    def __init__(self):
        self.requests: list[dict] = []  # every request to the Hub
        self.storage_requests: list[dict] = []
        self.repos: dict[str, dict] = {"ada-hf/taken": {"private": False}}
        self.commits: list[dict] = []
        self.lfs: dict[str, bytes] = {}  # uploaded LFS objects by sha256
        self.commit_limit: int | None = None  # bytes; above: HTTP 413
        self.datasets: dict[str, dict] = {}
        self.files: dict[tuple[str, str, str], bytes] = {}  # (repo, revision, path)
        self.page_size = 1000
        hub = self

        class Hub(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, data=None, headers=None, raw=None):
                body = raw if raw is not None else json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(body)

            def _record(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                entry = {
                    "method": self.command,
                    "path": self.path,
                    "headers": dict(self.headers),
                    "body": body,
                }
                hub.requests.append(entry)
                return entry

            def _token(self):
                auth = self.headers.get("Authorization") or ""
                return auth.removeprefix("Bearer ") if auth.startswith("Bearer ") else None

            def do_GET(self):
                entry = self._record()
                parts = urlsplit(self.path)
                path = parts.path
                token = self._token()
                if path == "/api/whoami-v2":
                    if token == GOOD_TOKEN:
                        role = "write"
                    elif token == READ_TOKEN:
                        role = "read"
                    else:
                        return self._send(
                            401, {"error": "Invalid credentials in Authorization header"}
                        )
                    return self._send(
                        200,
                        {
                            "type": "user",
                            "name": "ada-hf",
                            "orgs": [{"name": "team"}],
                            "auth": {"type": "access_token", "accessToken": {"role": role}},
                        },
                    )
                if path.startswith("/api/datasets/"):
                    return self._dataset_api(unquote(path[len("/api/datasets/") :]), parts, token)
                if path.startswith("/datasets/") and "/resolve/" in path:
                    head, rest = path[len("/datasets/") :].split("/resolve/", 1)
                    revision, file_path = rest.split("/", 1)
                    repo, revision, file_path = unquote(head), unquote(revision), unquote(file_path)
                    info = hub.datasets.get(repo)
                    if info is None:
                        return self._send(404, {"error": "Repository not found"})
                    if info.get("gated") and token != GOOD_TOKEN:
                        return self._send(403, {"error": "Access to this dataset is restricted."})
                    if (repo, revision, file_path) not in hub.files:
                        return self._send(404, {"error": "Entry not found"})
                    key = base64.urlsafe_b64encode(
                        json.dumps([repo, revision, file_path]).encode()
                    ).decode()
                    self.send_response(302)
                    self.send_header("Location", f"{hub.storage.url}/files/{key}?signed=1")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return None
                del entry
                return self._send(404, {"error": "Not found"})

            def _dataset_api(self, rest, parts, token):
                if "/tree/" in rest:
                    repo, revision = rest.split("/tree/", 1)
                    info = hub.datasets.get(repo)
                    if info is None or (info.get("private") and token != GOOD_TOKEN):
                        return self._send(
                            401 if token is None else 404, {"error": "Repository not found"}
                        )
                    revision = unquote(revision)
                    files = [
                        {"type": "file", "path": p, "size": len(content), "oid": "x"}
                        for (r, rev, p), content in sorted(hub.files.items())
                        if r == repo and rev == revision
                    ]
                    if not files and revision != "main":
                        return self._send(404, {"error": "Revision not found"})
                    folders = sorted(
                        {p.rsplit("/", 1)[0] for p in (f["path"] for f in files) if "/" in p}
                    )
                    entries = [
                        {"type": "directory", "path": f, "oid": "d"} for f in folders
                    ] + files
                    query = parse_qs(parts.query)
                    start = int((query.get("cursor") or ["0"])[0])
                    page = entries[start : start + hub.page_size]
                    headers = {}
                    if start + hub.page_size < len(entries):
                        nxt = f"{hub.url}{parts.path}?recursive=true&expand=false&cursor={start + hub.page_size}"
                        headers["Link"] = f'<{nxt}>; rel="next"'
                    return self._send(200, page, headers)
                info = hub.datasets.get(rest)
                if info is None or (info.get("private") and token != GOOD_TOKEN):
                    return self._send(
                        401 if token is None else 404, {"error": "Repository not found"}
                    )
                return self._send(200, {"id": rest, **info})

            def do_POST(self):
                entry = self._record()
                path = urlsplit(self.path).path
                token = self._token()
                if token != GOOD_TOKEN:
                    return self._send(401, {"error": "Invalid credentials in Authorization header"})
                if path == "/api/repos/create":
                    data = json.loads(entry["body"])
                    repo = f"{data['organization']}/{data['name']}"
                    if data["organization"] not in ("ada-hf", "team"):
                        return self._send(
                            403,
                            {
                                "error": "You don't have the rights to create a model under this namespace"
                            },
                        )
                    if repo in hub.repos:
                        return self._send(
                            409,
                            {
                                "error": "You already created this model repo",
                                "url": f"{hub.url}/{repo}",
                            },
                        )
                    hub.repos[repo] = {"private": data["private"]}
                    return self._send(200, {"url": f"{hub.url}/{repo}", "name": repo})
                if path.startswith("/api/models/") and "/preupload/" in path:
                    data = json.loads(entry["body"])
                    answer = [
                        {
                            "path": f["path"],
                            "uploadMode": "lfs"
                            if f["path"].endswith((".safetensors", ".pt", ".joblib"))
                            or f["size"] > 1_000_000
                            else "regular",
                            "shouldIgnore": False,
                        }
                        for f in data["files"]
                    ]
                    return self._send(200, {"files": answer})
                if path.endswith(".git/info/lfs/objects/batch"):
                    data = json.loads(entry["body"])
                    objects = []
                    for item in data["objects"]:
                        found = {"oid": item["oid"], "size": item["size"], "authenticated": True}
                        if item["oid"] not in hub.lfs:
                            found["actions"] = {
                                "upload": {
                                    "href": f"{hub.storage.url}/upload/{item['oid']}",
                                    "header": {"x-amz-test": "1"},
                                },
                                "verify": {"href": f"{hub.url}/lfs-verify", "header": {}},
                            }
                        objects.append(found)
                    return self._send(200, {"transfer": "basic", "objects": objects})
                if path == "/lfs-verify":
                    data = json.loads(entry["body"])
                    ok = data["oid"] in hub.lfs and len(hub.lfs[data["oid"]]) == data["size"]
                    return self._send(200 if ok else 404, {})
                if path.startswith("/api/models/") and "/commit/" in path:
                    if hub.commit_limit is not None and len(entry["body"]) > hub.commit_limit:
                        return self._send(413, {"error": "Payload too large"})
                    lines = [json.loads(line) for line in entry["body"].decode().splitlines()]
                    repo = unquote(path[len("/api/models/") :].split("/commit/")[0])
                    for line in lines:
                        if line["key"] == "lfsFile" and line["value"]["oid"] not in hub.lfs:
                            return self._send(400, {"error": "LFS file not uploaded"})
                    hub.commits.append({"repo": repo, "lines": lines, "headers": entry["headers"]})
                    return self._send(
                        200,
                        {"commitUrl": f"{hub.url}/{repo}/commit/abc123", "commitOid": "abc123"},
                    )
                return self._send(404, {"error": "Not found"})

        class Storage(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _record(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                hub.storage_requests.append(
                    {"method": self.command, "path": self.path, "headers": dict(self.headers)}
                )
                return body

            def do_PUT(self):
                body = self._record()
                oid = urlsplit(self.path).path.rsplit("/", 1)[-1]
                status = 200 if hashlib.sha256(body).hexdigest() == oid else 400
                if status == 200:
                    hub.lfs[oid] = body
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self):
                self._record()
                key = urlsplit(self.path).path.rsplit("/", 1)[-1]
                repo, revision, path = json.loads(base64.urlsafe_b64decode(key))
                content = hub.files[(repo, revision, path)]
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

        self.storage = _Server(Storage)
        self.hub = _Server(Hub)

    @property
    def url(self) -> str:
        return self.hub.url

    def close(self):
        self.hub.close()
        self.storage.close()

    def add_dataset(self, repo: str, files: dict[str, bytes], revision: str = "main", **info):
        self.datasets.setdefault(repo, {"private": False, "gated": False, **info})
        for path, content in files.items():
            self.files[(repo, revision, path)] = content

    def all_headers(self) -> list[dict]:
        return [r["headers"] for r in self.requests]
