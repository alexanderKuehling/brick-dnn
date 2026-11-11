"""Read an imported file in a process of its own: ``python -m brickdnn.data.import_job``.

Reading a large CSV or Excel file can need several times the file's size in memory for a
moment (text cells before their types are known, copies while converting). In the web
server that can get the whole worker killed when the container's memory runs out. This
process reads the file, stores the table and exits, so the memory is given back. It asks
the kernel to kill it first when memory runs out (``oom_score_adj``), not the web server.

Input (JSON on stdin): ``path``, ``filename``, ``options`` (``ImportOptions.to_dict()``),
``out`` (the folder for the table), ``max_rows``, ``max_columns``, ``max_bytes``.
Output (one JSON line on stdout): ``{"ok": true, "n_rows", "columns", "report"}`` or
``{"ok": false, "error", "sheets"?}`` for data the user has to fix.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any

# The kernel kills the process with the highest score first when memory runs out.
OOM_SCORE_ADJ = 1000


def _make_way() -> None:
    with contextlib.suppress(OSError):  # Linux only
        Path("/proc/self/oom_score_adj").write_text(f"{OOM_SCORE_ADJ}\n")


def run(job: dict[str, Any]) -> dict[str, Any]:
    from brickdnn.data.io import ImportOptions
    from brickdnn.data.readers import read_file
    from brickdnn.data.store import save_table
    from brickdnn.data.table import DataError
    from brickdnn.training.datasets import DatasetError

    try:
        options = ImportOptions.from_dict(job.get("options") or {})
        table, report = read_file(
            Path(job["path"]),
            options,
            filename=job["filename"],
            max_rows=job.get("max_rows"),
            max_columns=job.get("max_columns"),
            max_bytes=job.get("max_bytes"),
        )
        save_table(table, Path(job["out"]))
    except (DataError, DatasetError) as exc:
        answer: dict[str, Any] = {"ok": False, "error": str(exc)}
        if getattr(exc, "sheets", None) is not None:
            answer["sheets"] = exc.sheets
        return answer
    missing = table.df.isna().sum()
    return {
        "ok": True,
        "n_rows": table.n_rows,
        "columns": [
            {"name": column, "kind": table.kinds[column], "missing": int(missing[column])}
            for column in table.columns
        ],
        "report": report.to_dict(),
    }


def main() -> int:
    _make_way()
    job = json.loads(sys.stdin.read())
    print(json.dumps(run(job)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
