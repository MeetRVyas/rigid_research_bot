"""
Semantic Scholar API client.

Uses the public Graph API (https://api.semanticscholar.org/graph/v1).
No API key required, but an optional key raises rate limits significantly.
Set the SEMANTIC_SCHOLAR_API_KEY env var to use one.

Free-tier limits  : ~1 req/s   (~100 req/5 min)
With API key      : ~10 req/s  (apply at https://www.semanticscholar.org/product/api)
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from .config import get_settings
from .errors import NotFoundError, UpstreamUnavailableError
from .http_utils import build_retry_session, call_with_deadline
from models import CitationPaper
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# Constants
SS_BASE = "https://api.semanticscholar.org/graph/v1"

# Fields to request for full paper metadata
_PAPER_FIELDS = (
    "paperId,title,authors,year,abstract,"
    "citationCount,influentialCitationCount,"
    "externalIds,venue,fieldsOfStudy,publicationDate"
)

# Lighter field set for citation / reference lists
_REF_FIELDS = "title,authors,year,externalIds,citationCount,influentialCitationCount,venue"

_NO_KEY_DELAY = 1.1  # seconds; conservative for no-key usage
_WITH_KEY_DELAY = 0.1  # seconds


def _get_delay() -> float:
    settings = get_settings()
    return _WITH_KEY_DELAY if settings.semantic_scholar_api_key else _NO_KEY_DELAY


# Single shared, thread-safe limiter (concurrency-and-reliability.md #3).
_rate_limiter = RateLimiter(_get_delay())


# HTTP Session
def _build_ss_session() -> requests.Session:
    settings = get_settings()
    session = build_retry_session(user_agent="ArXivMCPServer/1.0")

    if settings.has_semantic_scholar_key and settings.semantic_scholar_api_key:
        session.headers.update({"x-api-key": settings.semantic_scholar_api_key})
        _rate_limiter.set_min_interval(_WITH_KEY_DELAY)
        logger.info("Semantic Scholar: using API key (elevated rate limits)")
    else:
        logger.info("Semantic Scholar: no API key — rate-limited to ~1 req/s")

    return session


_ss_session = _build_ss_session()


def _get(url: str, params: dict[str, Any] | None = None, timeout: int = 10) -> dict[str, Any]:
    """Rate-limited, deadline-bounded GET that returns parsed JSON.

    Raises `NotFoundError` on a genuine 404 and `UpstreamUnavailableError` for
    anything else that goes wrong (network failure, timeout, non-2xx status
    after retries). The one manual 429 retry this used to add on top of the
    retrying adapter's own backoff has been removed — it was redundant with
    what the adapter already does and could stack two full retry cycles back
    to back (concurrency-and-reliability.md #6).
    """
    _rate_limiter.wait()
    settings = get_settings()

    logger.debug(f"SS GET {url}  params={params}")
    try:
        resp = call_with_deadline(
            _ss_session.get,
            url,
            params=params or {},
            timeout=timeout,
            deadline_seconds=settings.request_deadline_seconds,
            upstream_name="Semantic Scholar",
        )
    except UpstreamUnavailableError:
        raise
    except requests.RequestException as exc:
        logger.warning(f"Semantic Scholar request failed: {exc}")
        raise UpstreamUnavailableError(
            "Could not reach Semantic Scholar right now. This is usually temporary."
        ) from exc

    if resp.status_code == 404:
        logger.info(f"Semantic Scholar 404: {url}")
        raise NotFoundError("Not found in Semantic Scholar.")

    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        logger.warning(f"Semantic Scholar returned HTTP {resp.status_code}: {exc}")
        raise UpstreamUnavailableError(
            f"Semantic Scholar returned an error (HTTP {resp.status_code}) after retries."
        ) from exc

    result: dict[str, Any] = resp.json()
    return result


# Helpers
def _ss_paper_id(arxiv_id: str) -> str:
    """Semantic Scholar accepts 'arXiv:2301.00001' as a paper identifier."""
    return f"arXiv:{arxiv_id.strip()}"


def _citation_paper_from_raw(raw: dict[str, Any]) -> CitationPaper:
    ext = raw.get("externalIds") or {}
    arxiv_id = ext.get("ArXiv")
    return CitationPaper(
        arxiv_id=arxiv_id,
        semantic_scholar_id=raw.get("paperId"),
        title=raw.get("title") or "",
        authors=[a.get("name", "") for a in (raw.get("authors") or [])],
        year=raw.get("year"),
        citation_count=raw.get("citationCount") or 0,
        influential_citation_count=raw.get("influentialCitationCount") or 0,
        abstract_url=f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None,
        venue=raw.get("venue"),
    )


# Public API
def get_paper_metadata(arxiv_id: str) -> dict[str, Any] | None:
    """
    Fetch enriched metadata from Semantic Scholar for a given ArXiv paper.
    Returns citation count, influential citations, venue, fields of study, etc.

    Returns None if the paper simply isn't indexed on Semantic Scholar yet
    (a normal, expected outcome — ArXiv papers can take a few days to
    appear). Raises `UpstreamUnavailableError` if Semantic Scholar itself
    could not be reached, so callers can tell the two apart instead of both
    silently looking like "no enrichment data" (concurrency-and-
    reliability.md #7).
    """
    try:
        data = _get(
            f"{SS_BASE}/paper/{_ss_paper_id(arxiv_id)}",
            params={"fields": _PAPER_FIELDS},
        )
    except NotFoundError:
        return None

    ext = data.get("externalIds") or {}
    return {
        "semantic_scholar_id": data.get("paperId"),
        "title": data.get("title", ""),
        "authors": [a.get("name", "") for a in (data.get("authors") or [])],
        "year": data.get("year"),
        "publication_date": data.get("publicationDate"),
        "abstract": data.get("abstract", ""),
        "citation_count": data.get("citationCount", 0),
        "influential_citation_count": data.get("influentialCitationCount", 0),
        "venue": data.get("venue"),
        "fields_of_study": data.get("fieldsOfStudy") or [],
        "external_ids": ext,
    }


def get_references(arxiv_id: str, max_results: int = 15) -> list[dict[str, Any]]:
    """
    Papers referenced BY the given paper (outgoing citations / bibliography).
    These are typically the papers the authors built upon.

    Returns an empty list if the paper isn't indexed on Semantic Scholar.
    Raises `UpstreamUnavailableError` if Semantic Scholar could not be reached.
    """
    try:
        data = _get(
            f"{SS_BASE}/paper/{_ss_paper_id(arxiv_id)}/references",
            params={"fields": _REF_FIELDS, "limit": min(max_results, 500)},
        )
    except NotFoundError:
        return []

    results = []
    for item in (data.get("data") or [])[:max_results]:
        cited = item.get("citedPaper") or {}
        if cited:
            results.append(_citation_paper_from_raw(cited).to_dict())
    return results


def get_citations(arxiv_id: str, max_results: int = 20) -> list[dict[str, Any]]:
    """
    Papers that CITE the given paper (incoming citations / forward citations).
    Useful for finding follow-up work.

    Returns an empty list if the paper isn't indexed on Semantic Scholar.
    Raises `UpstreamUnavailableError` if Semantic Scholar could not be reached.
    """
    try:
        data = _get(
            f"{SS_BASE}/paper/{_ss_paper_id(arxiv_id)}/citations",
            params={"fields": _REF_FIELDS, "limit": min(max_results, 500)},
        )
    except NotFoundError:
        return []

    results = []
    for item in (data.get("data") or [])[:max_results]:
        citing = item.get("citingPaper") or {}
        if citing:
            results.append(_citation_paper_from_raw(citing).to_dict())
    return results


def search_semantic_scholar(
    query: str,
    max_results: int = 10,
    fields_of_study: list[str] | None = None,
    year_range: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search Semantic Scholar directly (broader than ArXiv — includes non-ArXiv papers).

    Args:
        query: keyword query
        max_results: results to return
        fields_of_study: filter by field, e.g. ["Computer Science", "Mathematics"]
        year_range: e.g. "2020-2024" or "2023-"
    """
    params: dict[str, Any] = {
        "query": query,
        "limit": min(max_results, 100),
        "fields": _PAPER_FIELDS,
    }
    if fields_of_study:
        params["fieldsOfStudy"] = ",".join(fields_of_study)
    if year_range:
        params["year"] = year_range

    try:
        data = _get(f"{SS_BASE}/paper/search", params=params)
    except NotFoundError:
        return []

    results = []
    for paper in (data.get("data") or [])[:max_results]:
        ext = paper.get("externalIds") or {}
        arxiv_id = ext.get("ArXiv")
        results.append(
            {
                "semantic_scholar_id": paper.get("paperId"),
                "arxiv_id": arxiv_id,
                "title": paper.get("title", ""),
                "authors": [a.get("name", "") for a in (paper.get("authors") or [])],
                "year": paper.get("year"),
                "citation_count": paper.get("citationCount", 0),
                "influential_citation_count": paper.get("influentialCitationCount", 0),
                "venue": paper.get("venue"),
                "fields_of_study": paper.get("fieldsOfStudy") or [],
                "abstract_url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None,
                "abstract": paper.get("abstract", ""),
            }
        )
    return results