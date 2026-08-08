"""
Shared, validated input shapes.

Used everywhere an `arxiv_id`-shaped string, a free-text query, or an ArXiv
category code is accepted, instead of relying solely on FastMCP's
auto-generated primitive-type schema (str/int/list[str]). See
tool-design-and-protocol.md #2 and security-and-access-control.md #18.
"""

from __future__ import annotations

import re

from .errors import ValidationError

# New-style: 4 digits . 4-5 digits, optional version suffix (e.g. 2301.00001v2)
_NEW_STYLE_ID = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
# Old-style: archive(.subject-class)?/7 digits, optional version suffix (e.g. cs/0001001, math.GT/0309136v1)
_OLD_STYLE_ID = re.compile(r"^[a-zA-Z-]+(\.[A-Za-z]{2})?/\d{7}(v\d+)?$")

# ArXiv category codes look like "cs.LG", "math.OC", "q-bio.NC", or a bare top-level archive like "physics".
_CATEGORY = re.compile(r"^[a-zA-Z][a-zA-Z-]+(\.[A-Za-z]{1,10}[A-Za-z-]{1,10})?$")

MAX_QUERY_LENGTH = 400
MAX_NAME_LENGTH = 200
MAX_CATEGORY_LENGTH = 30
MAX_BATCH_SIZE = 20


def validate_arxiv_id(raw_id: str) -> str:
    """Validate and normalize (strip whitespace from) a single arxiv_id.

    Raises ValidationError with a clear, caller-facing message instead of
    letting an empty/malformed string reach `_strip_version` and crash with
    a raw IndexError.
    """
    if raw_id is None:
        raise ValidationError("arxiv_id is required and cannot be empty.")
    candidate = raw_id.strip()
    if not candidate:
        raise ValidationError(
            "arxiv_id cannot be blank or whitespace-only. "
            "Expected a real ArXiv ID, e.g. '1706.03762' or 'cs/0001001'."
        )
    if not (_NEW_STYLE_ID.match(candidate) or _OLD_STYLE_ID.match(candidate)):
        raise ValidationError(
            f"'{raw_id}' doesn't look like a valid ArXiv ID. Expected either the "
            "new style ('2301.00001' or '2301.00001v2') or the old style "
            "('cs/0001001' or 'math.GT/0309136v1')."
        )
    return candidate


def validate_query(query: str, *, field_name: str = "query") -> str:
    """Validate a free-text query/author-name field: non-empty, bounded length."""
    if query is None or not query.strip():
        raise ValidationError(f"{field_name} cannot be blank.")
    if len(query) > MAX_QUERY_LENGTH:
        raise ValidationError(
            f"{field_name} is too long ({len(query)} chars). "
            f"Keep it under {MAX_QUERY_LENGTH} characters."
        )
    return query


def validate_category(category: str) -> str:
    """Validate an ArXiv category code's shape (not its membership in the live taxonomy)."""
    if category is None or not category.strip():
        raise ValidationError("category cannot be blank, e.g. 'cs.LG'.")
    candidate = category.strip()
    if len(candidate) > MAX_CATEGORY_LENGTH:
        raise ValidationError(f"category '{category}' is too long to be a valid ArXiv code.")
    if not _CATEGORY.match(candidate):
        raise ValidationError(
            f"'{category}' doesn't look like a valid ArXiv category code, e.g. 'cs.LG', "
            "'stat.ML', 'q-bio.NC'. See the arxiv://reference/categories resource for the "
            "full list."
        )
    return candidate


def validate_batch_ids(arxiv_ids: list[str]) -> list[str]:
    """Validate a list of arxiv_ids for batch_get_papers: bounded size, every entry valid."""
    if not arxiv_ids:
        raise ValidationError("Provide at least one arxiv_id.")
    if len(arxiv_ids) > MAX_BATCH_SIZE:
        raise ValidationError(
            f"Maximum {MAX_BATCH_SIZE} IDs per batch request "
            f"(got {len(arxiv_ids)}). Split into multiple calls."
        )
    return [validate_arxiv_id(aid) for aid in arxiv_ids]