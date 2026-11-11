"""deploy/server/uptime.py, the script of .github/workflows/uptime.yml: checks a site served
locally and keeps one outage issue in a fake GitHub up to date."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "deploy" / "server" / "uptime.py"
spec = importlib.util.spec_from_file_location("uptime", SCRIPT)
uptime = importlib.util.module_from_spec(spec)
sys.modules["uptime"] = uptime  # dataclasses look the module up
_write_bytecode, sys.dont_write_bytecode = (
    sys.dont_write_bytecode,
    True,
)  # no __pycache__ in deploy/
spec.loader.exec_module(uptime)
sys.dont_write_bytecode = _write_bytecode

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)


class Site:
    """What the fake site answers."""

    health = (200, {"status": "ok", "checks": {"database": "ok"}})
    page = (200, "<title>Brick DNN</title>")


@pytest.fixture
def site():
    state = Site()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/health/":
                status, body = state.health
                text = body if isinstance(body, str) else json.dumps(body)
            else:
                status, text = state.page
            data = text.encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield state
    server.shutdown()


class FakeGitHub(uptime.GitHub):
    def __init__(self):
        super().__init__("https://api.example", "owner/repo", "token")
        self.issues: list[dict] = []
        self.comments_of: dict[int, list[dict]] = {}
        self.labels: set[str] = set()
        self.now = NOW

    def call(self, method, path, body=None):
        stamp = self.now.isoformat().replace("+00:00", "Z")
        if method == "GET" and path.startswith("/issues?"):
            return [i for i in self.issues if i["state"] == "open" and "outage" in i["labels"]]
        if method == "POST" and path == "/labels":
            self.labels.add(body["name"])
            return body
        if method == "POST" and path == "/issues":
            assert body["labels"][0] in self.labels
            issue = {**body, "number": len(self.issues) + 1, "state": "open", "created_at": stamp}
            self.issues.append(issue)
            return issue
        number = int(path.split("/")[2].split("?")[0])
        if path.endswith("/comments") and method == "POST":
            self.comments_of.setdefault(number, []).append({**body, "created_at": stamp})
            return body
        if "/comments" in path:
            return self.comments_of.get(number, [])
        if method == "PATCH":
            self.issues[number - 1].update(body)
            return self.issues[number - 1]
        raise AssertionError(f"unexpected call {method} {path}")


def test_a_healthy_site_is_up(site):
    result = uptime.check_site(site.url, timeout=5)
    assert result.up, result.problems


def test_problems_are_named_without_details(site):
    site.health = (503, {"status": "error", "checks": {"database": "failed", "disk": "ok"}})
    site.page = (502, "Bad gateway")
    result = uptime.check_site(site.url, timeout=5)
    assert not result.up
    assert result.problems == [
        "API: HTTP 503, status error (failed: database)",
        "front page: HTTP 502",
    ]
    site.health = (200, {"status": "degraded", "checks": {"worker": "failed"}})
    site.page = (200, "<html>Welcome to nginx</html>")
    assert uptime.check_site(site.url, timeout=5).problems == [
        "API: HTTP 200, status degraded (failed: worker)",
        "front page: not the Brick DNN app",
    ]
    site.health = (200, "<html>not json</html>")
    assert uptime.check_site(site.url, timeout=5).problems[0] == "API: HTTP 200, no health report"


def test_no_answer():
    result = uptime.check_site("http://127.0.0.1:9", timeout=2)
    assert not result.up
    assert result.problems[0].startswith("API: no answer")


def test_a_short_hiccup_is_retried(site, monkeypatch):
    answers = iter([uptime.Result(False, ["API: no answer (timed out)"]), uptime.Result(True, [])])
    monkeypatch.setattr(uptime, "check_site", lambda url: next(answers))
    pauses = []
    assert uptime.check_with_retries(site.url, 3, 30, sleep=pauses.append).up
    assert pauses == [30]


def test_one_issue_per_outage_comments_on_changes_and_closes_when_back():
    github = FakeGitHub()
    down = uptime.Result(False, ["front page: HTTP 502"])
    run = "https://github.com/owner/repo/actions/runs/1"

    assert uptime.report(github, "https://brickdnn.com", down, run, 6, NOW).endswith("#1")
    assert github.issues[0]["title"] == "brickdnn.com is down"
    assert "front page: HTTP 502" in github.issues[0]["body"]
    assert github.labels == {"outage"}

    # The same problem 15 minutes later: nothing new (no notification).
    github.now = NOW + timedelta(minutes=15)
    message = uptime.report(github, "https://brickdnn.com", down, run, 6, github.now)
    assert "up to date" in message
    assert len(github.issues) == 1 and not github.comments_of

    # Another problem: a comment on the same issue, not a second issue.
    worse = uptime.Result(False, ["API: no answer (timed out)", "front page: HTTP 502"])
    github.now = NOW + timedelta(minutes=30)
    uptime.report(github, "https://brickdnn.com", worse, run, 6, github.now)
    assert len(github.issues) == 1
    assert "the problem changed" in github.comments_of[1][0]["body"]

    # Still down six hours after the last comment: a reminder.
    github.now = NOW + timedelta(hours=7)
    uptime.report(github, "https://brickdnn.com", worse, run, 6, github.now)
    assert "Still down after about 7 h" in github.comments_of[1][1]["body"]

    # Back: a last comment, and the issue is closed.
    github.now = NOW + timedelta(hours=7, minutes=15)
    up = uptime.Result(True, [])
    assert "closed issue #1" in uptime.report(
        github, "https://brickdnn.com", up, run, 6, github.now
    )
    assert "back up" in github.comments_of[1][2]["body"]
    assert "7 h 15 min" in github.comments_of[1][2]["body"]
    assert github.issues[0]["state"] == "closed"

    # Up and no open issue: nothing to do. The next outage opens a new issue.
    assert uptime.report(github, "https://brickdnn.com", up, run, 6, github.now) == (
        "up, no open issue"
    )
    uptime.report(github, "https://brickdnn.com", down, run, 6, github.now)
    assert len(github.issues) == 2


def test_without_site_url_nothing_runs(capsys):
    assert uptime.main([], env={}) == 0
    assert "nothing to check" in capsys.readouterr().out


def test_dry_run_against_a_local_site(site, capsys):
    env = {"SITE_URL": site.url + "/", "UPTIME_ATTEMPTS": "1"}
    assert uptime.main(["--dry-run"], env=env) == 0
    assert "dry run: site up" in capsys.readouterr().out
