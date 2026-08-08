"""
app/services/arxiv_service.py

Thin adapter over the arxiv-mcp-server's underlying clients
(.arxiv / .semantic_scholar) — exposes the same 12
capabilities as the MCP tools in /server.py, but as plain,
directly-importable functions returning bare dicts/lists (no MCP envelope,
no Context, no ToolAnnotations), so Veritas_Service.TOOL_REGISTRY can call
them straight, keyed by the same names it already imports:

    search_papers, get_paper_details, get_paper_pdf_url, get_recent_papers,
    get_related_papers, get_paper_citations, get_author_papers,
    search_by_category, search_title, search_abstract, batch_get_papers,
    search_semantic_scholar

Every function here mirrors the corresponding tool body in server.py
(validation, clamping, Semantic Scholar enrichment) — only the MCP-protocol
plumbing (ctx, envelope dicts, _run_tool's error-to-envelope wrapping) is
stripped out, since Veritas_Service.retrieve_and_refine already wraps each
tool call in its own try/except and treats a raised exception as "no
results" for that step.

Requires the arxiv-mcp-server package to be an importable dependency of this
app (pip install / path dependency exposing the `` package). If
instead you're calling a *deployed* arxiv-mcp-server over MCP protocol
(streamable-http), this direct-import approach doesn't apply — you'd want
an MCP client adapter here instead, happy to write that version if that's
actually your setup.
"""

from __future__ import annotations

from typing import Any

from arxiv_service.arxiv import (
    _strip_version,
    fetch_paper_by_id,
    fetch_papers_by_ids,
    search_arxiv,
)
from arxiv_service.arxiv import get_author_papers as _author_papers
from arxiv_service.arxiv import get_recent_papers as _recent_papers
from arxiv_service.arxiv import search_by_abstract as _search_by_abstract
from arxiv_service.arxiv import search_by_category as _search_by_cat
from arxiv_service.arxiv import search_by_title as _search_by_title
from arxiv_service.errors import UpstreamUnavailableError
from arxiv_service.semantic_scholar import get_citations as _citations
from arxiv_service.semantic_scholar import get_paper_metadata as _ss_metadata
from arxiv_service.semantic_scholar import get_references as _references
from arxiv_service.semantic_scholar import search_semantic_scholar as _ss_search
from arxiv_service.validation import (
    validate_arxiv_id,
    validate_batch_ids,
    validate_category,
    validate_query,
)

_VALID_SORT_BY = {"relevance", "submittedDate", "lastUpdatedDate"}


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(value, hi))


# ---------------------------------------------------------------------------
# search_papers — keyword/field search across all of ArXiv
# ---------------------------------------------------------------------------
def search_papers(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
    sort_order: str = "descending",
) -> list[dict[str, Any]]:
    q = validate_query(query)
    clamped = _clamp(max_results, 1, 50)
    papers = search_arxiv(q, max_results=clamped, sort_by=sort_by, sort_order=sort_order)
    return [p.to_dict() for p in papers]


# ---------------------------------------------------------------------------
# get_paper_details — full metadata, enriched with Semantic Scholar
# ---------------------------------------------------------------------------
def get_paper_details(paper_id: str) -> dict[str, Any]:
    clean_id = validate_arxiv_id(paper_id)
    paper = fetch_paper_by_id(clean_id)
    if not paper:
        return {}

    result = paper.to_dict()
    result["html_url"] = paper.html_url

    try:
        ss = _ss_metadata(paper.arxiv_id)
    except UpstreamUnavailableError:
        # Enrichment is best-effort — a missing/unreachable Semantic Scholar
        # shouldn't fail the whole lookup, ArXiv metadata alone is still useful.
        ss = None

    if ss:
        result["citation_count"] = ss.get("citation_count")
        result["influential_citation_count"] = ss.get("influential_citation_count")
        result["fields_of_study"] = ss.get("fields_of_study", [])
        result["venue"] = ss.get("venue")
        result["semantic_scholar_id"] = ss.get("semantic_scholar_id")

    return result


# ---------------------------------------------------------------------------
# get_paper_pdf_url — PDF / HTML (ar5iv) / LaTeX source links, no network call
# ---------------------------------------------------------------------------
def get_paper_pdf_url(paper_id: str) -> dict[str, Any]:
    clean_id = _strip_version(validate_arxiv_id(paper_id))
    return {
        "paper_id": clean_id,
        "arxiv_id": clean_id,
        "pdf_url": f"https://arxiv.org/pdf/{clean_id}",
        "abstract_url": f"https://arxiv.org/abs/{clean_id}",
        "html_url": f"https://ar5iv.org/abs/{clean_id}",
        "latex_source_url": f"https://arxiv.org/src/{clean_id}",
    }


# ---------------------------------------------------------------------------
# get_recent_papers — newest submissions to a category
# ---------------------------------------------------------------------------
def get_recent_papers(
    category: str,
    days: int = 7,
    max_results: int = 20,
) -> list[dict[str, Any]]:
    cat = validate_category(category)
    clamped_days = _clamp(days, 1, 30)
    clamped_results = _clamp(max_results, 1, 50)
    papers = _recent_papers(cat, days_back=clamped_days, max_results=clamped_results)
    return [p.to_dict() for p in papers]


# ---------------------------------------------------------------------------
# get_related_papers — outgoing references / bibliography (Semantic Scholar)
# ---------------------------------------------------------------------------
def get_related_papers(paper_id: str, max_results: int = 10) -> list[dict[str, Any]]:
    clean_id = validate_arxiv_id(paper_id)
    clamped = _clamp(max_results, 1, 30)
    return _references(clean_id, max_results=clamped)


# ---------------------------------------------------------------------------
# get_paper_citations — incoming / forward citations (Semantic Scholar)
# ---------------------------------------------------------------------------
def get_paper_citations(paper_id: str, max_results: int = 20) -> list[dict[str, Any]]:
    clean_id = validate_arxiv_id(paper_id)
    clamped = _clamp(max_results, 1, 50)
    return _citations(clean_id, max_results=clamped)


# ---------------------------------------------------------------------------
# get_author_papers — all ArXiv papers by a researcher, partial name OK
# ---------------------------------------------------------------------------
def get_author_papers(
    author: str,
    max_results: int = 15,
    sort_by: str = "submittedDate",
) -> list[dict[str, Any]]:
    name = validate_query(author, field_name="author_name")
    effective_sort = sort_by if sort_by in _VALID_SORT_BY else "submittedDate"
    clamped = _clamp(max_results, 1, 50)
    papers = _author_papers(name, max_results=clamped, sort_by=effective_sort)
    return [p.to_dict() for p in papers]


# ---------------------------------------------------------------------------
# search_by_category — keyword search scoped to one ArXiv category
# ---------------------------------------------------------------------------
def search_by_category(
    category: str,
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
) -> list[dict[str, Any]]:
    cat = validate_category(category)
    q = validate_query(query)
    clamped = _clamp(max_results, 1, 50)
    papers = _search_by_cat(cat, q, max_results=clamped, sort_by=sort_by)
    return [p.to_dict() for p in papers]


# ---------------------------------------------------------------------------
# search_title — search specifically in paper titles
# ---------------------------------------------------------------------------
def search_title(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    q = validate_query(query, field_name="title_query")
    clamped = _clamp(max_results, 1, 30)
    papers = _search_by_title(q, max_results=clamped)
    return [p.to_dict() for p in papers]


# ---------------------------------------------------------------------------
# search_abstract — search specifically in abstracts
# ---------------------------------------------------------------------------
def search_abstract(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    q = validate_query(query)
    clamped = _clamp(max_results, 1, 30)
    papers = _search_by_abstract(q, max_results=clamped)
    return [p.to_dict() for p in papers]


# ---------------------------------------------------------------------------
# batch_get_papers — up to 20 papers in a single round-trip
# ---------------------------------------------------------------------------
def batch_get_papers(paper_ids: list[str]) -> list[dict[str, Any]]:
    clean_ids = validate_batch_ids(paper_ids)
    papers = fetch_papers_by_ids(clean_ids)
    return [p.to_dict() for p in papers]


# ---------------------------------------------------------------------------
# search_semantic_scholar — broader search, includes non-ArXiv papers
# ---------------------------------------------------------------------------
def search_semantic_scholar(
    query: str,
    max_results: int = 10,
    fields_of_study: list[str] | None = None,
    year_range: str | None = None,
) -> list[dict[str, Any]]:
    q = validate_query(query)
    clamped = _clamp(max_results, 1, 50)
    return _ss_search(q, max_results=clamped, fields_of_study=fields_of_study, year_range=year_range)