"""
llm_router.py
─────────────────────────────────────────────────────────────────────────────
Async-first LangChain multi-provider LLM router with smart failover.

Routing policy
──────────────
• Providers are tried in config insertion order.
• A provider is fully exhausted before moving to the next one.
• Within a provider, (api_key × model) permutations are round-robined.

Error taxonomy
──────────────
  Class          HTTP / signal       Action
  ─────────────────────────────────────────────────────────────────────────
  RATE_LIMIT     429                 endpoint → COOLING for cooldown_seconds
  OVERLOADED     5xx / demand msg    endpoint → COOLING (no retry)
  UNAVAILABLE    404 / model msg     all endpoints for that model → DEAD
  AUTH_FAILURE   401                 all endpoints for that key  → DEAD
  SERVER_ERROR   500                 retry w/ exponential backoff → COOLING
  TIMEOUT        timeout / conn      retry w/ exponential backoff → COOLING
  UNKNOWN        anything else       retry w/ exponential backoff → COOLING

Circuit breaker (per provider)
──────────────────────────────
After `cb_threshold` consecutive full exhaustions the circuit opens for
`cb_timeout` seconds. While open the provider is skipped entirely.
Resets automatically when the timeout expires.

Logging
───────
  llm_router.log  — DEBUG-level: every attempt, state change, timing
  console         — INFO-level: lifecycle events only
                    (provider_switched, circuit, dead endpoints, recovered)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set

from pydantic import Field, PrivateAttr
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError


# ══════════════════════════════════════════════════════════════════════════════
# § 1  LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def _setup_loggers(log_file: str = "logs/llm_router.log"):
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s")
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    fl = logging.getLogger("llm_router.file")
    fl.setLevel(logging.DEBUG)
    fl.addHandler(fh)
    fl.propagate = False

    cl = logging.getLogger("llm_router.console")
    cl.setLevel(logging.INFO)
    cl.addHandler(ch)
    cl.propagate = False

    return fl, cl


_flog, _clog = _setup_loggers()


# ══════════════════════════════════════════════════════════════════════════════
# § 2  ERROR CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

class ErrorType(Enum):
    RATE_LIMIT   = auto()  # 429                         → cooldown, no retry
    OVERLOADED   = auto()  # high-demand 5xx / message   → cooldown, no retry
    UNAVAILABLE  = auto()  # 404 / model-not-found msg   → DEAD (model)
    AUTH_FAILURE = auto()  # 401                         → DEAD (key)
    SERVER_ERROR = auto()  # generic 5xx                 → retry → cooldown
    TIMEOUT      = auto()  # timeout / connection error  → retry → cooldown
    UNKNOWN      = auto()  # anything else               → retry → cooldown


_OVERLOAD_HINTS: tuple[str, ...] = (
    "overloaded", "high demand", "at capacity", "too many requests",
    "server is busy", "model is busy", "currently overloaded",
)
_UNAVAIL_HINTS: tuple[str, ...] = (
    "model not found", "no such model", "does not exist",
    "model_not_found", "invalid model", "deprecated",
    "not available in your region",
)


def classify_error(exc: Exception) -> ErrorType:
    msg = str(exc).lower()

    if isinstance(exc, (APITimeoutError, asyncio.TimeoutError)):
        return ErrorType.TIMEOUT

    if isinstance(exc, APIConnectionError):
        return ErrorType.TIMEOUT

    if isinstance(exc, RateLimitError):
        return (
            ErrorType.OVERLOADED
            if any(h in msg for h in _OVERLOAD_HINTS)
            else ErrorType.RATE_LIMIT
        )

    if isinstance(exc, APIStatusError):
        s = exc.status_code
        if s == 401:
            return ErrorType.AUTH_FAILURE
        if s == 404 or any(h in msg for h in _UNAVAIL_HINTS):
            return ErrorType.UNAVAILABLE
        if s == 429:
            return (
                ErrorType.OVERLOADED
                if any(h in msg for h in _OVERLOAD_HINTS)
                else ErrorType.RATE_LIMIT
            )
        if s in (503, 529) or any(h in msg for h in _OVERLOAD_HINTS):
            return ErrorType.OVERLOADED
        if s >= 500:
            return ErrorType.SERVER_ERROR

    # Fallback: plain message scan
    if any(h in msg for h in _OVERLOAD_HINTS):
        return ErrorType.OVERLOADED
    if any(h in msg for h in _UNAVAIL_HINTS):
        return ErrorType.UNAVAILABLE
    return ErrorType.UNKNOWN


# ══════════════════════════════════════════════════════════════════════════════
# § 3  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RetryConfig:
    max_retries:      int   = 3
    base_delay:       float = 1.0    # seconds before first retry
    max_delay:        float = 30.0   # cap on retry backoff delay
    backoff_factor:   float = 2.0
    cooldown_seconds: float = 60.0   # how long a rate-limited endpoint waits
    cb_threshold:     int   = 2      # full-exhaustion events before circuit opens
    cb_timeout:       float = 120.0  # seconds the circuit stays open


class EndpointState(Enum):
    ACTIVE  = "active"
    COOLING = "cooling"  # temporarily unavailable; recovers after cooldown
    DEAD    = "dead"     # permanently unavailable (bad key / gone model)


@dataclass
class EndpointStatus:
    state:          EndpointState = EndpointState.ACTIVE
    cooldown_until: float         = 0.0
    last_error:     str           = ""


@dataclass
class Endpoint:
    provider: str
    model:    str
    api_key:  str
    base_url: Optional[str] = None

    @property
    def id(self) -> str:
        return f"{self.provider}:{self.model}:{self.api_key[:8]}"


# ══════════════════════════════════════════════════════════════════════════════
# § 4  PROVIDER ROUTER
# ══════════════════════════════════════════════════════════════════════════════

class ProviderRouter:
    """
    Manages all (key × model) endpoints for one provider.

    Uses threading.Lock — fast, cross-thread safe, event-loop-agnostic.
    The lock is held only for brief in-memory operations; all I/O happens
    outside it.
    """

    def __init__(self, name: str, endpoints: List[Endpoint], cfg: RetryConfig):
        self.name      = name
        self.endpoints = endpoints
        self.cfg       = cfg

        self._statuses: Dict[str, EndpointStatus] = {
            ep.id: EndpointStatus() for ep in endpoints
        }
        self._idx              = 0
        self._lock             = threading.Lock()
        self._cb_open_until    = 0.0   # monotonic timestamp; 0 = circuit closed
        self._exhaustion_count = 0     # consecutive full-exhaustion events

    # ─── internal (always call within self._lock) ────────────────────────────

    def _is_active(self, ep: Endpoint) -> bool:
        """Returns True if the endpoint is usable right now; auto-recovers cooling."""
        st = self._statuses[ep.id]
        if st.state == EndpointState.DEAD:
            return False
        if st.state == EndpointState.COOLING:
            if time.monotonic() >= st.cooldown_until:
                st.state = EndpointState.ACTIVE
                _flog.info(f"cooldown_recovered  endpoint={ep.id}")
                _clog.info(f"  ↺  recovered: {self.name} / {ep.model}")
                return True
            return False
        return True

    def _active_list(self) -> List[Endpoint]:
        return [ep for ep in self.endpoints if self._is_active(ep)]

    # ─── public API ──────────────────────────────────────────────────────────

    def circuit_is_open(self) -> bool:
        with self._lock:
            if time.monotonic() < self._cb_open_until:
                return True
            if self._cb_open_until > 0:          # timeout just expired → reset
                self._cb_open_until    = 0.0
                self._exhaustion_count = 0
                _flog.info(f"circuit_closed  provider={self.name}")
                _clog.info(f"  ↺  circuit closed: {self.name}")
            return False

    def is_exhausted(self) -> bool:
        """True when no endpoints are active right now (all cooling or dead)."""
        with self._lock:
            return not self._active_list()

    def is_permanently_dead(self) -> bool:
        """True when every endpoint is DEAD — no cooldown recovery is possible."""
        with self._lock:
            return all(
                self._statuses[ep.id].state == EndpointState.DEAD
                for ep in self.endpoints
            )

    def next_endpoint(self) -> Optional[Endpoint]:
        """Return the next active endpoint in round-robin order, or None."""
        with self._lock:
            if time.monotonic() < self._cb_open_until:
                return None
            active = self._active_list()
            if not active:
                return None
            ep         = active[self._idx % len(active)]
            self._idx  = (self._idx + 1) % len(active)
            return ep

    def mark_endpoint(self, ep: Endpoint, error_type: ErrorType) -> None:
        """Update endpoint state after a failure; may trip the circuit breaker."""
        with self._lock:
            if error_type == ErrorType.AUTH_FAILURE:
                # Entire key is invalid → kill every endpoint sharing it
                for e in self.endpoints:
                    if e.api_key == ep.api_key:
                        s            = self._statuses[e.id]
                        s.state      = EndpointState.DEAD
                        s.last_error = "auth_failure"
                _flog.warning(
                    f"auth_failure  provider={self.name}"
                    f"  key={ep.api_key[:8]}…"
                    f"  →  all endpoints for this key DEAD"
                )
                _clog.info(
                    f"  ✗  auth failure: {self.name}"
                    f"  key={ep.api_key[:8]}… → dead"
                )

            elif error_type == ErrorType.UNAVAILABLE:
                # Model is gone → kill every endpoint using it
                for e in self.endpoints:
                    if e.model == ep.model:
                        s            = self._statuses[e.id]
                        s.state      = EndpointState.DEAD
                        s.last_error = "unavailable"
                _flog.warning(
                    f"model_unavailable  provider={self.name}"
                    f"  model={ep.model}  → DEAD"
                )
                _clog.info(
                    f"  ✗  unavailable: {self.name} / {ep.model} → dead"
                )

            else:
                # RATE_LIMIT, OVERLOADED, or post-retry exhaustion → cooldown
                s                = self._statuses[ep.id]
                s.state          = EndpointState.COOLING
                s.cooldown_until = time.monotonic() + self.cfg.cooldown_seconds
                s.last_error     = error_type.name.lower()
                _flog.info(
                    f"cooling  endpoint={ep.id}"
                    f"  reason={error_type.name}"
                    f"  duration={self.cfg.cooldown_seconds}s"
                )

            # After every failure check if the whole provider is now exhausted
            if not self._active_list():
                self._exhaustion_count += 1
                _flog.warning(
                    f"provider_fully_exhausted  provider={self.name}"
                    f"  count={self._exhaustion_count}/{self.cfg.cb_threshold}"
                )
                if self._exhaustion_count >= self.cfg.cb_threshold:
                    self._cb_open_until = time.monotonic() + self.cfg.cb_timeout
                    _flog.warning(
                        f"circuit_opened  provider={self.name}"
                        f"  timeout={self.cfg.cb_timeout}s"
                    )
                    _clog.info(
                        f"  ⚡  circuit open: {self.name}"
                        f"  (resets in {self.cfg.cb_timeout:.0f}s)"
                    )

    def on_success(self) -> None:
        """Reset failure counters after a successful call."""
        with self._lock:
            self._exhaustion_count = 0

    def status_snapshot(self) -> Dict[str, Any]:
        """Return a snapshot of the current state (useful for debugging)."""
        now = time.monotonic()
        with self._lock:
            return {
                "circuit_open":      now < self._cb_open_until,
                "circuit_reopens_in": max(0.0, self._cb_open_until - now),
                "exhaustion_count":  self._exhaustion_count,
                "endpoints": {
                    ep.id: {
                        "state":      self._statuses[ep.id].state.value,
                        "last_error": self._statuses[ep.id].last_error,
                        "cooldown_remaining_s": (
                            max(0.0, self._statuses[ep.id].cooldown_until - now)
                            if self._statuses[ep.id].state == EndpointState.COOLING
                            else 0.0
                        ),
                    }
                    for ep in self.endpoints
                },
            }


# ══════════════════════════════════════════════════════════════════════════════
# § 5  MULTI-PROVIDER CHAT LLM
# ══════════════════════════════════════════════════════════════════════════════

class MultiProviderChatLLM(BaseChatModel):
    """
    Async-first LangChain chat model with cascading provider failover.

    Prefer `ainvoke` / `astream` for async callers.
    `invoke` / `generate` work synchronously via a thread-pool shim.
    """

    providers_config: Dict[str, Dict]
    retry_config:     RetryConfig = Field(default_factory=RetryConfig)

    _routers: List[ProviderRouter] = PrivateAttr(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        providers_config: Dict[str, Dict],
        retry_config: Optional[RetryConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            providers_config=providers_config,
            retry_config=retry_config or RetryConfig(),
            **kwargs,
        )
        self._routers = self._build_routers()

    # ─── setup ──────────────────────────────────────────────────────────────

    def _build_routers(self) -> List[ProviderRouter]:
        routers: List[ProviderRouter] = []
        for name, params in self.providers_config.items():
            # Cartesian product: every key paired with every model
            endpoints = [
                Endpoint(
                    provider=name,
                    model=model,
                    api_key=key,
                    base_url=params.get("base_url"),
                )
                for key in params["keys"]
                for model in params["models"]
            ]
            routers.append(ProviderRouter(name, endpoints, self.retry_config))
            _flog.info(
                f"router_built  provider={name}"
                f"  endpoints={len(endpoints)}"
                f"  ({len(params['keys'])} keys × {len(params['models'])} models)"
            )
        return routers

    @property
    def _llm_type(self) -> str:
        return "multi_provider_chat"

    # ─── internal helpers ────────────────────────────────────────────────────

    def _build_client(self, ep: Endpoint) -> ChatOpenAI:
        kw: Dict[str, Any] = {
            "model":       ep.model,
            "api_key":     ep.api_key,
            "temperature": 0,
        }
        if ep.base_url:
            kw["base_url"] = ep.base_url
        _clog.info(ep.model)
        return ChatOpenAI(**kw)

    async def _call_endpoint(
        self, ep: Endpoint, messages: List[BaseMessage]
    ) -> str:
        client = self._build_client(ep)
        response = await client.ainvoke(messages)
        return response.content

    async def _try_with_retry(
        self,
        router:   ProviderRouter,
        ep:       Endpoint,
        messages: List[BaseMessage],
    ) -> Optional[ChatResult]:
        """
        Retry a transient failure (SERVER_ERROR / TIMEOUT / UNKNOWN) with
        exponential backoff.  Returns a ChatResult on success, or None after
        all retries are exhausted (the endpoint is then put into cooldown).
        """
        delay = self.retry_config.base_delay

        for attempt in range(1, self.retry_config.max_retries + 1):
            _flog.debug(
                f"retry  endpoint={ep.id}"
                f"  attempt={attempt}/{self.retry_config.max_retries}"
                f"  delay={delay:.1f}s"
            )
            await asyncio.sleep(delay)
            delay = min(delay * self.retry_config.backoff_factor, self.retry_config.max_delay)

            try:
                content = await self._call_endpoint(ep, messages)
                _flog.info(f"retry_success  endpoint={ep.id}  attempt={attempt}")
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=content))]
                )
            except Exception as exc:
                et = classify_error(exc)
                _flog.warning(
                    f"retry_failed  endpoint={ep.id}"
                    f"  attempt={attempt}  error={et.name}  exc={exc!r}"
                )
                # Non-transient error mid-retry → stop immediately, mark it
                if et not in (
                    ErrorType.SERVER_ERROR, ErrorType.TIMEOUT, ErrorType.UNKNOWN
                ):
                    router.mark_endpoint(ep, et)
                    return None

        # All retries exhausted → put this endpoint in cooldown
        router.mark_endpoint(ep, ErrorType.RATE_LIMIT)   # reuses the COOLING path
        return None

    async def _try_provider(
        self,
        router:   ProviderRouter,
        messages: List[BaseMessage],
    ) -> Optional[ChatResult]:
        """
        Attempt every active endpoint for this provider in round-robin order.
        Returns a ChatResult on the first success, or None when the provider
        is fully exhausted.
        """
        attempted: Set[str] = set()

        while True:
            if router.circuit_is_open():
                _flog.info(f"circuit_open_skip  provider={router.name}")
                return None

            ep = router.next_endpoint()

            # None   → all endpoints are cooling/dead right now
            # in set → safety net: we went full-circle (shouldn't happen in practice)
            if ep is None or ep.id in attempted:
                return None

            attempted.add(ep.id)
            _flog.debug(f"attempt  endpoint={ep.id}")

            try:
                content = await self._call_endpoint(ep, messages)
                _flog.info(f"success  endpoint={ep.id}")
                router.on_success()
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=content))]
                )

            except Exception as exc:
                et = classify_error(exc)
                _flog.warning(
                    f"endpoint_error  endpoint={ep.id}"
                    f"  error={et.name}  exc={exc!r}"
                )

                if et in (ErrorType.SERVER_ERROR, ErrorType.TIMEOUT, ErrorType.UNKNOWN):
                    # Worth retrying with backoff
                    result = await self._try_with_retry(router, ep, messages)
                    if result is not None:
                        router.on_success()
                        return result
                    # Retries failed; endpoint is now in COOLING → try next
                else:
                    # Immediate non-retryable failure → mark and move on
                    router.mark_endpoint(ep, et)

                # Continue loop → next active endpoint in this provider

    # ─── LangChain interface ─────────────────────────────────────────────────

    async def _agenerate(
        self,
        messages:    List[BaseMessage],
        stop:        Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs:    Any,
    ) -> ChatResult:
        _flog.info(
            f"request_start"
            f"  messages={len(messages)}"
            f"  providers={[r.name for r in self._routers]}"
        )

        for i, router in enumerate(self._routers):
            if router.is_permanently_dead():
                _flog.info(f"skip_dead  provider={router.name}")
                continue
            if router.circuit_is_open():
                _flog.info(f"skip_circuit  provider={router.name}")
                continue

            result = await self._try_provider(router, messages)

            if result is not None:
                _flog.info(f"request_done  provider={router.name}")
                return result

            # Provider exhausted — find next candidate for the log message
            remaining = [
                r.name for r in self._routers[i + 1:]
                if not r.is_permanently_dead()
            ]
            if remaining:
                _clog.info(
                    f"  →  provider_switched:"
                    f" {router.name} exhausted → {remaining[0]}"
                )
            _flog.warning(
                f"provider_exhausted  from={router.name}"
                f"  remaining={remaining}"
            )

        _clog.info("  ✗  all_exhausted")
        _flog.critical("all_exhausted  no response available")
        raise RuntimeError(
            "All providers exhausted — no response available."
        )

    def _generate(
        self,
        messages:    List[BaseMessage],
        stop:        Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs:    Any,
    ) -> ChatResult:
        """
        Sync shim: runs _agenerate in a dedicated thread with a fresh event
        loop.  Avoids "event loop already running" issues regardless of caller
        context.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run,
                self._agenerate(messages, stop, run_manager, **kwargs),
            ).result()

    # ─── debug helpers ───────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Return a full snapshot of every provider's routing state."""
        return {router.name: router.status_snapshot() for router in self._routers}