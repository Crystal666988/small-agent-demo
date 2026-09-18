"""Execution trace / logging.

Every significant event in the loop (user input, model decision, tool call,
tool result, error, final answer, compression) is written as one JSON line to
logs/trace-<session>.jsonl and optionally echoed to stderr. This gives an
auditable per-session execution log.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


class Tracer:
    def __init__(self, session_id: str, *, echo: bool = False, log_dir: Path | None = None):
        self.session_id = session_id
        self.echo = echo
        d = log_dir or _LOG_DIR
        d.mkdir(parents=True, exist_ok=True)
        # sanitize session id for filenames
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
        self.path = d / f"trace-{safe}.jsonl"

    def emit(self, event: str, **fields: Any) -> None:
        rec = {"ts": round(time.time(), 3), "session": self.session_id, "event": event, **fields}
        line = json.dumps(rec, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if self.echo:
            print(f"  [trace] {event}: {fields}", file=sys.stderr)
