"""Example graphs shipped with Brick DNN (also shown in the UI's "Examples" menu)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

_DIR = Path(__file__).parent


def _load() -> dict[str, dict[str, Any]]:
    docs = [json.loads(path.read_text()) for path in _DIR.glob("*.json")]
    return {doc["id"]: doc for doc in sorted(docs, key=lambda d: d["order"])}


EXAMPLES: dict[str, dict[str, Any]] = _load()


def get_example(example_id: str) -> dict[str, Any]:
    """Return a deep copy of the example with the given id (raises KeyError)."""
    return copy.deepcopy(EXAMPLES[example_id])
