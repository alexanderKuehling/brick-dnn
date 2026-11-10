#!/usr/bin/env python3
"""Is the site up? Checks it and keeps one GitHub issue (label ``outage``) in line with it.

Run by .github/workflows/uptime.yml every 15 minutes; Python's standard library only.

- ``SITE_URL/api/health/`` must answer 200 with ``"status": "ok"`` and the front page must
  answer 200 and contain "Brick DNN". A failed check is repeated (``UPTIME_ATTEMPTS`` times,
  ``UPTIME_RETRY_SECONDS`` apart) before the site counts as down, so a deploy or a slow
  moment does not open an issue.
- Down: opens an issue labelled ``outage``, or, if one is open, comments on it when the
  problem changed or every ``UPTIME_REMIND_HOURS`` hours while it lasts. Never a second issue.
- Up again: comments with the length of the outage and closes the issue.

GitHub notifies everyone who watches the repository about new issues and comments (e-mail,
app). The issue only says which checks failed; details are in the Django admin or in
``/api/health/`` with the X-Health-Token header (deploy/server/README.md, "Monitoring").

Environment: SITE_URL, GITHUB_TOKEN, GITHUB_REPOSITORY (owner/name); optional GITHUB_API_URL,
GITHUB_SERVER_URL and GITHUB_RUN_ID (a link to the run in the issue), and the UPTIME_* above.
``--dry-run`` only checks the site and prints what it would do on GitHub.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlsplit

LABEL = "outage"
MARKER = re.compile(r"<!-- uptime-state: (.*?) -->")
USER_AGENT = "brickdnn-uptime-check"


@dataclass
class Result:
    up: bool
    problems: list[str]

    @property
    def summary(self) -> str:
        return "; ".join(self.problems) if self.problems else "all checks passed"


def fetch(url: str, timeout: float) -> tuple[int | None, str]:
    """The HTTP status and the body (``None`` and the reason if there was no answer)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(1_000_000).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(100_000).decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        return None, "timed out" if "timed out" in str(reason) else str(reason)[:120]


def check_site(site: str, timeout: float = 20.0) -> Result:
    problems: list[str] = []
    status, body = fetch(f"{site}/api/health/", timeout)
    if status is None:
        problems.append(f"API: no answer ({body})")
    else:
        try:
            data = json.loads(body)
        except ValueError:
            data = None
        if not isinstance(data, dict):
            problems.append(f"API: HTTP {status}, no health report")
        elif status != 200 or data.get("status") != "ok":
            failed = sorted(
                name for name, value in (data.get("checks") or {}).items() if value != "ok"
            )
            text = f"API: HTTP {status}, status {data.get('status')!s}"
            problems.append(text + (f" (failed: {', '.join(failed)})" if failed else ""))
    status, body = fetch(f"{site}/", timeout)
    if status is None:
        problems.append(f"front page: no answer ({body})")
    elif status != 200:
        problems.append(f"front page: HTTP {status}")
    elif "Brick DNN" not in body:
        problems.append("front page: not the Brick DNN app")
    return Result(up=not problems, problems=problems)


def check_with_retries(site: str, attempts: int, pause: float, sleep=time.sleep) -> Result:
    result = check_site(site)
    for _ in range(attempts - 1):
        if result.up:
            break
        print(f"Not up ({result.summary}); trying again in {pause:g} s.")
        sleep(pause)
        result = check_site(site)
    return result


class GitHub:
    def __init__(self, api: str, repository: str, token: str):
        self.base = f"{api.rstrip('/')}/repos/{repository}"
        self.token = token

    def call(self, method: str, path: str, body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self.base + path,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": USER_AGENT,
                **({"Content-Type": "application/json"} if data else {}),
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            text = response.read().decode()
        return json.loads(text) if text else None

    def open_issue(self) -> dict | None:
        issues = self.call("GET", f"/issues?labels={LABEL}&state=open&per_page=20")
        issues = [issue for issue in issues if "pull_request" not in issue]
        return min(issues, key=lambda issue: issue["number"]) if issues else None

    def ensure_label(self) -> None:
        try:
            self.call(
                "POST",
                "/labels",
                {"name": LABEL, "color": "b60205", "description": "The site is down"},
            )
        except urllib.error.HTTPError as exc:
            if exc.code != 422:  # 422: the label exists
                raise

    def comments(self, number: int) -> list[dict]:
        found: list[dict] = []
        page = 1
        while True:
            batch = self.call("GET", f"/issues/{number}/comments?per_page=100&page={page}")
            found += batch
            if len(batch) < 100:
                return found
            page += 1


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def duration(seconds: float) -> str:
    days, minutes = divmod(max(1, round(seconds / 60)), 24 * 60)
    hours, minutes = divmod(minutes, 60)
    parts = [(days, "d"), (hours, "h"), (minutes, "min")]
    return " ".join(f"{value} {unit}" for value, unit in parts if value)


def last_state(issue: dict, comments: list[dict]) -> tuple[str, datetime]:
    """The problem this script last reported on the issue, and when."""
    state, when = "", parse_time(issue["created_at"])
    for item in [issue, *comments]:
        match = MARKER.search(item.get("body") or "")
        if match:
            state, when = match.group(1), parse_time(item["created_at"])
    return state, when


def report(
    github: GitHub | None,
    site: str,
    result: Result,
    run_url: str,
    remind_hours: float,
    now: datetime,
) -> str:
    """Opens, comments on or closes the issue. Returns what it did (for the log)."""
    host = urlsplit(site).hostname or site
    marker = f"<!-- uptime-state: {result.summary} -->"
    run = f"\n\n[Check run]({run_url})" if run_url else ""
    stamp = now.strftime("%Y-%m-%d %H:%M UTC")
    if github is None:
        return f"dry run: site {'up' if result.up else 'down'} ({result.summary})"
    issue = github.open_issue()
    if result.up:
        if not issue:
            return "up, no open issue"
        down_for = duration((now - parse_time(issue["created_at"])).total_seconds())
        github.call(
            "POST",
            f"/issues/{issue['number']}/comments",
            {
                "body": f"{site} is back up ({stamp}), about {down_for} after this issue was "
                f"opened. All checks passed.{run}\n\n{marker}"
            },
        )
        github.call(
            "PATCH",
            f"/issues/{issue['number']}",
            {"state": "closed", "state_reason": "completed"},
        )
        return f"up again, closed issue #{issue['number']}"
    problems = "\n".join(f"- {problem}" for problem in result.problems)
    if not issue:
        github.ensure_label()
        created = github.call(
            "POST",
            "/issues",
            {
                "title": f"{host} is down",
                "labels": [LABEL],
                "body": (
                    f"The uptime check found {site} not working at {stamp}, several tries "
                    f"in a row:\n\n{problems}\n\nThis issue gets a comment when the "
                    f"problem changes and is closed when the site is back. What to look at: "
                    f'deploy/server/README.md, section "Monitoring".{run}\n\n{marker}'
                ),
            },
        )
        return f"down, opened issue #{created['number']}"
    state, when = last_state(issue, github.comments(issue["number"]))
    hours = (now - when).total_seconds() / 3600
    if state != result.summary:
        text = f"Still down ({stamp}), the problem changed:\n\n{problems}"
    elif hours >= remind_hours:
        down_for = duration((now - parse_time(issue["created_at"])).total_seconds())
        text = f"Still down after about {down_for} ({stamp}):\n\n{problems}"
    else:
        return f"down, issue #{issue['number']} is open and up to date"
    github.call("POST", f"/issues/{issue['number']}/comments", {"body": f"{text}{run}\n\n{marker}"})
    return f"down, commented on issue #{issue['number']}"


def main(argv: list[str], env=os.environ) -> int:
    site = env.get("SITE_URL", "").strip().rstrip("/")
    if not site:
        print("SITE_URL is not set: nothing to check.")
        return 0
    dry_run = "--dry-run" in argv
    attempts = max(1, int(env.get("UPTIME_ATTEMPTS") or 3))
    pause = float(env.get("UPTIME_RETRY_SECONDS") or 30)
    remind_hours = float(env.get("UPTIME_REMIND_HOURS") or 6)
    result = check_with_retries(site, attempts, pause)
    print(f"{site}: {'up' if result.up else 'DOWN'} ({result.summary})")
    github = None
    if not dry_run:
        github = GitHub(
            env.get("GITHUB_API_URL") or "https://api.github.com",
            env["GITHUB_REPOSITORY"],
            env["GITHUB_TOKEN"],
        )
    run_url = ""
    if env.get("GITHUB_RUN_ID") and env.get("GITHUB_REPOSITORY"):
        server = env.get("GITHUB_SERVER_URL") or "https://github.com"
        run_url = f"{server}/{env['GITHUB_REPOSITORY']}/actions/runs/{env['GITHUB_RUN_ID']}"
    print(report(github, site, result, run_url, remind_hours, datetime.now(UTC)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
