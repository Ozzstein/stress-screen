"""Lightweight progress reporting to stderr with elapsed-time prefix."""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import Iterator


class ProgressReporter:
    """Stage-level progress reporter. Prints to stderr with [N.Ns] prefix.

    If quiet=True, all output is suppressed (used for tests / scripted invocations).
    """

    def __init__(self, quiet: bool = False) -> None:
        self._t0 = time.monotonic()
        self.quiet = quiet

    def _elapsed(self) -> float:
        return time.monotonic() - self._t0

    def stage(self, message: str) -> None:
        if self.quiet:
            return
        print(f"[{self._elapsed():.1f}s] {message}", file=sys.stderr, flush=True)

    @contextmanager
    def step(self, start_msg: str, end_msg: str | None = None) -> Iterator[None]:
        """Context manager that prints start and (optional) end message."""
        self.stage(start_msg)
        try:
            yield
        finally:
            if end_msg is not None:
                self.stage(end_msg)


# Module-level singleton — default verbose. CLI can replace with a quiet instance.
_default = ProgressReporter()


def get() -> ProgressReporter:
    return _default


def set_quiet(quiet: bool) -> None:
    global _default
    _default = ProgressReporter(quiet=quiet)
