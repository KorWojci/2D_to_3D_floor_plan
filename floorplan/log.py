"""Minimal structured logger used by the pipeline (streamed to the web UI)."""
from __future__ import annotations

import time
from typing import Callable, Optional


class Log:
    def __init__(self, sink: Optional[Callable[[dict], None]] = None):
        self.entries: list[dict] = []
        self._sink = sink
        self._t0 = time.time()

    def _emit(self, level: str, msg: str) -> None:
        e = {"t": round(time.time() - self._t0, 3), "level": level, "msg": msg}
        self.entries.append(e)
        if self._sink:
            self._sink(e)

    def info(self, msg: str) -> None:
        self._emit("info", msg)

    def step(self, msg: str) -> None:
        self._emit("step", msg)

    def warn(self, msg: str) -> None:
        self._emit("warn", msg)

    def error(self, msg: str) -> None:
        self._emit("error", msg)

    @property
    def warnings(self) -> list[str]:
        return [e["msg"] for e in self.entries if e["level"] == "warn"]
