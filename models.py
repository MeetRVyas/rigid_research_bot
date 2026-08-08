"""Data models for ArXiv MCP Server."""

from dataclasses import asdict, dataclass
from typing import Any, Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field

from config import (
    EMBEDDING_PROVIDER,
    PROVIDER,
    EMBEDDING_MODEL,
    LLM_MODEL,
)


@dataclass
class Paper:
    """Represents a single ArXiv paper with full metadata."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str  # ISO 8601, e.g. "2023-01-01T00:00:00Z"
    updated: str
    primary_category: str
    categories: list[str]
    pdf_url: str
    abstract_url: str
    doi: str | None = None
    journal_ref: str | None = None
    comment: str | None = None  # Author-submitted comment (e.g. "15 pages, 4 figures")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def html_url(self) -> str:
        """ar5iv HTML rendering of the paper (LaTeX → HTML)."""
        return f"https://ar5iv.org/abs/{self.arxiv_id}"


@dataclass
class CitationPaper:
    """A paper in a citation / reference list (lighter weight)."""

    arxiv_id: str | None
    semantic_scholar_id: str | None
    title: str
    authors: list[str]
    year: int | None
    citation_count: int
    influential_citation_count: int
    abstract_url: str | None
    pdf_url: str | None
    venue: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Score(BaseModel):
    score: float = Field(description="Score between 0.0 and 1.0")
    reason: str = Field(description="Brief justification for the score")


class KeepOrDrop(BaseModel):
    keep: bool = Field(description="True if sentence is directly relevant")


class CitationSupport(BaseModel):
    supported: bool = Field(
        description="True only if the source text factually supports the claim — topical overlap alone does not count"
    )
    reason: str = Field(description="Brief justification for the verdict")


class CRAGRequest(BaseModel):
    question: str = Field(description="Original user query", min_length=3, max_length=2000)
    embedding_provider: str = Field(
        description="Embedding provider, e.g. ollama, google, huggingface",
        default=EMBEDDING_PROVIDER,
    )
    llm_provider: str = Field(
        description="LLM provider, e.g. ollama, google, huggingface",
        default=PROVIDER,
    )
    embedding_model: Optional[str] = Field(
        description="Embedding model name, e.g. nomic-embed-text",
        default=EMBEDDING_MODEL,
    )
    llm_model: Optional[str] = Field(
        description="LLM model name, e.g. phi3:mini, gemini-1.5-flash",
        default=LLM_MODEL,
    )
    # When provided, the chat endpoint looks for a cached answer inside these
    # snapshots before running the live pipeline.  Most-recent snapshot wins
    # when the same question is cached under multiple snapshot IDs.
    # When omitted, only the current active snapshot is checked.
    snapshot_ids: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional list of snapshot IDs to search for a cached answer. "
            "If the question is cached under multiple IDs the most recently "
            "created snapshot is used. Omit to use the current snapshot only."
        ),
    )


class CRAGResponse(BaseModel):
    answer: str = Field(description="Answer given by LLM")
    verdict: str = Field(description="CORRECT, AMBIGUOUS, or INCORRECT")
    web_search_used: bool = Field(
        description="True when Tavily web search was triggered"
    )
    cached: bool = Field(
        description="True when this answer was served from the Redis cache",
        default=False,
    )
    snapshot_id: Optional[str] = Field(
        default=None,
        description=(
            "The snapshot ID under which this answer was cached or stored. "
            "None when the answer was not cached."
        ),
    )


class Intent(str, Enum):
    AUTHOR_LOOKUP = "AUTHOR_LOOKUP"      # papers by a named author
    RECENT_DIGEST = "RECENT_DIGEST"      # what's new in a category / time window
    CITATION_GRAPH = "CITATION_GRAPH"    # what a paper cites / what cites a paper
    PAPER_LOOKUP = "PAPER_LOOKUP"        # one specific, clearly identified paper — abstract/metadata is enough
    PAPER_QA = "PAPER_QA"                # needs the paper's FULL TEXT, not just its abstract
    OPEN_ENDED = "OPEN_ENDED"            # broader / comparative / under-specified
    GENERAL_CHAT = "GENERAL_CHAT"        # does not require paper search


class RetrievalAction(BaseModel):
    """One additional, independent retrieval need beyond the primary intent/slots
    on ClassifyResult.

    Only populated when a single question has more than one distinct part —
    e.g. "compare paper 2301.00001's method with recent cs.CL work" needs a
    PAPER_QA action for the paper *and* a RECENT_DIGEST action for cs.CL.
    Ordinary single-part questions leave this list empty; the primary
    intent/slots on ClassifyResult already cover them.
    """

    intent: Intent
    author_name: Optional[str] = None
    category: Optional[str] = None
    days: Optional[int] = None
    paper_id: Optional[str] = None
    paper_title: Optional[str] = Field(
        default=None,
        description=(
            "PAPER_LOOKUP / PAPER_QA. Natural-language paper name/title, when the "
            "user names a paper without giving a strict alphanumeric arXiv ID "
            "(e.g. '1706.03762'). Populate this instead of paper_id in that case."
        ),
    )
    citation_direction: Optional[Literal["incoming", "outgoing"]] = None
    paper_ids: Optional[list[str]] = Field(
        default=None,
        description=(
            'PAPER_QA only. ArXiv id(s) to search the full text of. Use ["all"] '
            "to search every paper already indexed this session instead of a "
            "specific one."
        ),
    )
    query: Optional[str] = Field(
        default=None,
        description=(
            "Focused search string for this action — full-text search terms for "
            "PAPER_QA, or rewritten keyword-search terms for OPEN_ENDED. Leave "
            "unset for other intents."
        ),
    )


class ClassifyResult(BaseModel):
    intent: Intent
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the intent label itself")
    ambiguous: bool = Field(
        default=False,
        description="True only for OPEN_ENDED questions that are genuinely under-specified",
    )

    # Slots — populate whichever apply to the classified (primary) intent so
    # build_query can call the right tool without a second LLM round-trip.
    author_name: Optional[str] = None
    category: Optional[str] = None          # ArXiv category code, e.g. cs.CL
    days: Optional[int] = None              # lookback window for RECENT_DIGEST
    paper_id: Optional[str] = None          # strict alphanumeric ArXiv id only, e.g. "1706.03762"
    paper_title: Optional[str] = Field(
        default=None,
        description=(
            "PAPER_LOOKUP / PAPER_QA. Natural-language paper name/title, when the "
            "user names a paper without giving a strict alphanumeric arXiv ID. "
            "Populate this instead of paper_id in that case."
        ),
    )
    citation_direction: Optional[Literal["incoming", "outgoing"]] = Field(
        default=None,
        description="incoming = papers that cite this one, outgoing = this paper's references",
    )
    paper_ids: Optional[list[str]] = Field(
        default=None,
        description=(
            'PAPER_QA only. ArXiv id(s) to search the full text of. Use ["all"] '
            "to search every paper already indexed this session instead of a "
            "specific one."
        ),
    )
    query: Optional[str] = Field(
        default=None,
        description=(
            "Focused search string — full-text search terms for PAPER_QA, or "
            "rewritten keyword-search terms for OPEN_ENDED."
        ),
    )

    # Extra, independent retrieval needs beyond the primary intent/slots above.
    # Leave empty for ordinary single-part questions — this is what lets the
    # retrieve stage fan out to more than one tool call for one question.
    additional_actions: list[RetrievalAction] = Field(default_factory=list)


class DraftAnswer(BaseModel):
    answer: str = Field(..., description="Best-effort answer from general knowledge, no tools used")
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Self-rated confidence that the draft is complete and correct on its own",
    )


# ---------------------------------------------------------------------------
# Snapshot models (used by /documents/snapshots and /crag/cache)
# ---------------------------------------------------------------------------

class SnapshotFile(BaseModel):
    filename: str
    uploaded_at: str  # ISO-8601


class Snapshot(BaseModel):
    id: str
    created_at: str  # ISO-8601
    files: list[SnapshotFile]


class SnapshotListResponse(BaseModel):
    snapshots: list[Snapshot]


class CacheEntry(BaseModel):
    question: str
    answer: str
    verdict: str
    web_search_used: bool
    snapshot_id: str
    created_at: str  # ISO-8601


class SnapshotCacheResponse(BaseModel):
    """Response for GET /crag/cache — all cached answers for the requested snapshots."""
    entries: list[CacheEntry]