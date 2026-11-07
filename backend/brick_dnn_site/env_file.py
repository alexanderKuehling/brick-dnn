"""Settings from the optional file ``.env`` in the repository root.

Docker Compose reads this file itself; ``manage.py`` (``make backend``) and the WSGI/ASGI
servers read it with :func:`load_env_file`, so local runs get the same settings. The test
suite does not read it, so tests never run against the database configured there.

Format: one ``NAME=value`` per line, ``#`` starts a comment. Variables that are already set in
the environment win.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"
# 'value' or "value", optionally followed by a comment: taken as it is.
QUOTED = re.compile(r"""^\s*(['"])(.*)\1\s*(?:#.*)?$""")


def parse_env(text: str) -> dict[str, str]:
    """The variables in the text of a ``.env`` file (a subset of Docker Compose's format)."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").lstrip()
        name, sep, value = line.partition("=")
        name = name.strip()
        if not sep or not name.isidentifier():
            continue
        quoted = QUOTED.match(value)
        if quoted:
            values[name] = quoted.group(2)
        else:  # a # after a space starts a comment
            values[name] = re.split(r"\s#", value, maxsplit=1)[0].strip()
    return values


def load_env_file(path: Path = ENV_FILE) -> None:
    """Set the variables of the ``.env`` file that are not set yet. No file: nothing happens."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    for name, value in parse_env(text).items():
        os.environ.setdefault(name, value)
