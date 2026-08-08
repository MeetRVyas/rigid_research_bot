"""
Shared HTTP-call plumbing for both API clients.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, TypeVar

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .errors import UpstreamUnavailableError

T = TypeVar("T")

# A small dedicated pool for enforcing call deadlines.
_deadline_executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="arxiv-mcp-http")


def build_retry_session(
    *,
    user_agent: str,
    total_retries: int = 2,
    backoff_factor: float = 1.0,
    status_forcelist: tuple[int, ...] = (429, 500, 502, 503, 504),
) -> requests.Session:
    """Build a `requests.Session` with one deliberate, shared retry policy.

    Defaults: 2 retries, backoff 1s/2s — a worst case of 3 attempts. Kept
    intentionally modest so it fits under `call_with_deadline`'s default
    budget even when every attempt also pays its own per-request timeout.
    """
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=list(status_forcelist),
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": user_agent})
    return session


def call_with_deadline(
    fn: Callable[..., T],
    *args: Any,
    deadline_seconds: float,
    upstream_name: str,
    **kwargs: Any,
) -> T:
    """Run `fn(*args, **kwargs)` with a hard wall-clock deadline.

    Raises `UpstreamUnavailableError` (rather than a raw `TimeoutError`) if
    the call hasn't completed within `deadline_seconds`, regardless of what
    the underlying retry adapter was doing internally.
    """
    future = _deadline_executor.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=deadline_seconds)
    except FutureTimeoutError as exc:
        raise UpstreamUnavailableError(
            f"{upstream_name} did not respond within {deadline_seconds:.0f}s "
            "(including retries). It may be slow or temporarily unavailable — try again shortly."
        ) from exc