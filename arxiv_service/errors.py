"""
Shared error convention for the ArXiv MCP Server.

Every *expected* failure mode a tool can hit — bad input, nothing found,
an upstream API that's temporarily unavailable — should be raised as one of
the exceptions below rather than left to whatever the underlying HTTP
library or Python itself raises.
"""

from __future__ import annotations


class ArxivError(Exception):
    """Base class for all expected, purpose-raised errors in this server."""


class ValidationError(ArxivError):
    """The caller's input was malformed (bad arxiv_id shape, out-of-range value, etc.)."""


class NotFoundError(ArxivError):
    """The requested resource genuinely doesn't exist upstream."""


class UpstreamUnavailableError(ArxivError):
    """A dependency (ArXiv or Semantic Scholar) could not be reached or timed out."""


def error_envelope(message: str, **extra: object) -> dict[str, object]:
    """Build the one consistently-shaped error dict every tool returns on failure."""
    return {"error": message, **extra}