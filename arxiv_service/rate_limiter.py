"""
A single, thread-safe rate limiter, shared by `arxiv.py` and `semantic_scholar.py`.

The limiter is deliberately kept process-global (not per-caller): ArXiv and
Semantic Scholar see all of this server's traffic as a single identity
regardless of how many end users sit behind it, so a single shared pace is
the correct behavior for a single-instance deployment.
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """Blocks the calling thread until at least `min_interval_seconds` has
    elapsed since the last call across *all* threads, serialized by a lock
    so concurrent callers queue instead of racing."""

    def __init__(self, min_interval_seconds: float) -> None:
        self._min_interval = min_interval_seconds
        self._lock = threading.Lock()
        self._last_call_time: float = 0.0

    def wait(self) -> float:
        """Block until it's this caller's turn. Returns the seconds actually slept."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call_time
            sleep_for = self._min_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last_call_time = time.monotonic()
            return max(sleep_for, 0.0)

    def set_min_interval(self, min_interval_seconds: float) -> None:
        """Adjust the pace at runtime (e.g. once a Semantic Scholar API key is known)."""
        with self._lock:
            self._min_interval = min_interval_seconds