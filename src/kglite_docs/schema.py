"""Node + edge type names and embedding-store keys for the kglite-docs graph.

Centralised here so renames are a one-file change and so the MCP layer
and the storage layer agree on names.
"""

from __future__ import annotations

import re as _re
from typing import Final

# Node types
DOCUMENT: Final = "Document"
PAGE: Final = "Page"
CHUNK: Final = "Chunk"
SUMMARY: Final = "Summary"
TAG: Final = "Tag"
CLUSTER: Final = "Cluster"
AGENT: Final = "Agent"
VIEW: Final = "View"
NOTE: Final = "Note"
REVIEW_TICKET: Final = "ReviewTicket"
REVIEW_EVENT: Final = "ReviewEvent"

# Edge types
HAS_PAGE: Final = "HAS_PAGE"
HAS_CHUNK: Final = "HAS_CHUNK"
NEXT_CHUNK: Final = "NEXT_CHUNK"
IN_CLUSTER: Final = "IN_CLUSTER"
SIMILAR_TO: Final = "SIMILAR_TO"
SUMMARIZES: Final = "SUMMARIZES"
VERIFIES: Final = "VERIFIES"
TAGGED_AS: Final = "TAGGED_AS"
AUTHORED: Final = "AUTHORED"
VERIFIED_BY: Final = "VERIFIED_BY"
VIEWED: Final = "VIEWED"
ANNOTATED: Final = "ANNOTATED"
TARGETS: Final = "TARGETS"
CLAIMED: Final = "CLAIMED"
REVIEWED: Final = "REVIEWED"
HAS_REVIEW_EVENT: Final = "HAS_REVIEW_EVENT"

# Review ticket states (event-sourced via ReviewEvent)
REVIEW_NEW: Final = "new"
REVIEW_IN_REVIEW: Final = "in_review"
REVIEW_REVIEWED: Final = "reviewed"
REVIEW_REJECTED: Final = "rejected"
REVIEW_NEEDS_REVISION: Final = "needs_revision"
REVIEW_UNCLAIMED: Final = "unclaimed"  # event type, not a status
REVIEW_STATES: Final = frozenset({
    REVIEW_NEW, REVIEW_IN_REVIEW, REVIEW_REVIEWED,
    REVIEW_REJECTED, REVIEW_NEEDS_REVISION,
})

# Embedding-store text columns. kglite keys the store as `{text_column}_emb`.
CHUNK_TEXT_COL: Final = "text"
SUMMARY_TEXT_COL: Final = "text"
DOCUMENT_TITLE_COL: Final = "title"

CHUNK_TEXT_EMB: Final = (CHUNK, CHUNK_TEXT_COL)
SUMMARY_TEXT_EMB: Final = (SUMMARY, SUMMARY_TEXT_COL)
DOCUMENT_TITLE_EMB: Final = (DOCUMENT, DOCUMENT_TITLE_COL)

# Summary verification statuses
VERIFICATION_UNVERIFIED: Final = "unverified"
VERIFICATION_VERIFIED: Final = "verified"
VERIFICATION_DISPUTED: Final = "disputed"
VERIFICATION_NEEDS_REVISION: Final = "needs_revision"
VERIFICATION_STALE: Final = "stale"

VALID_VERDICTS: Final = frozenset(
    {VERIFICATION_VERIFIED, VERIFICATION_DISPUTED, VERIFICATION_NEEDS_REVISION}
)

# Chunk status
CHUNK_STATUS_READY: Final = "ready"
CHUNK_STATUS_NEEDS_OCR: Final = "needs_ocr"
CHUNK_STATUS_EMPTY: Final = "empty"

# Summary depths
DEPTH_CHUNK: Final = "chunk"
DEPTH_SECTION: Final = "section"
DEPTH_DOCUMENT: Final = "document"
VALID_DEPTHS: Final = frozenset({DEPTH_CHUNK, DEPTH_SECTION, DEPTH_DOCUMENT})

# Agent kinds
AGENT_LLM: Final = "llm"
AGENT_HUMAN: Final = "human"
AGENT_SERVICE: Final = "service"

# Tag kinds
TAG_TOPIC: Final = "topic"
TAG_ENTITY: Final = "entity"
TAG_CUSTOM: Final = "custom"

# ─── Secondary labels (kglite 0.10.5+) ────────────────────────────────────
#
# Categorical / lifecycle state lives as secondary labels on the relevant
# node type, queryable via `MATCH (n:Label)`. The user-facing API still
# accepts snake_case values ("verified", "needs_ocr"); `_label_for(...)`
# below maps user values → label names at the boundary.
#
# Cross-type collisions are intentional: `MATCH (n:Reviewed)` matches both
# completed review tickets and reviewed translations — the kglite team
# endorsed this in their 0.10.5 letter.

# Agent.kind → labels
LABEL_LLM: Final = "LLM"
LABEL_HUMAN: Final = "Human"
LABEL_SERVICE: Final = "Service"

# Chunk.status → labels
LABEL_READY: Final = "Ready"
LABEL_NEEDS_OCR: Final = "NeedsOcr"
LABEL_EMPTY: Final = "Empty"

# Summary verification → labels (also used for ReviewTicket overlap)
LABEL_UNVERIFIED: Final = "Unverified"
LABEL_VERIFIED: Final = "Verified"
LABEL_DISPUTED: Final = "Disputed"
LABEL_STALE: Final = "Stale"

# Review ticket + Summary share "NeedsRevision"/"Reviewed" (intentional)
LABEL_NEW: Final = "New"
LABEL_IN_REVIEW: Final = "InReview"
LABEL_REVIEWED: Final = "Reviewed"
LABEL_NEEDS_REVISION: Final = "NeedsRevision"
LABEL_REJECTED: Final = "Rejected"

# Translation.status → labels (Reviewed collides with review status —
# intentional, `MATCH (n:Reviewed)` returns both classes)
LABEL_DRAFT: Final = "Draft"
# LABEL_REVIEWED reused

# Tag.kind → labels
LABEL_TOPIC: Final = "Topic"
LABEL_ENTITY: Final = "Entity"
LABEL_CUSTOM: Final = "Custom"
LABEL_REVIEW_TAG: Final = "ReviewTag"

# ─── Discriminator → label maps ───────────────────────────────────────────
#
# Per-discriminator dict from user-facing string to canonical label name.
# `_label_for(discriminator, value)` consults these.

_AGENT_KIND_LABELS: Final[dict[str, str]] = {
    AGENT_LLM: LABEL_LLM,
    AGENT_HUMAN: LABEL_HUMAN,
    AGENT_SERVICE: LABEL_SERVICE,
}

_CHUNK_STATUS_LABELS: Final[dict[str, str]] = {
    CHUNK_STATUS_READY: LABEL_READY,
    CHUNK_STATUS_NEEDS_OCR: LABEL_NEEDS_OCR,
    CHUNK_STATUS_EMPTY: LABEL_EMPTY,
}

_SUMMARY_STATUS_LABELS: Final[dict[str, str]] = {
    VERIFICATION_UNVERIFIED: LABEL_UNVERIFIED,
    VERIFICATION_VERIFIED: LABEL_VERIFIED,
    VERIFICATION_DISPUTED: LABEL_DISPUTED,
    VERIFICATION_NEEDS_REVISION: LABEL_NEEDS_REVISION,
    VERIFICATION_STALE: LABEL_STALE,
}

_REVIEW_STATUS_LABELS: Final[dict[str, str]] = {
    REVIEW_NEW: LABEL_NEW,
    REVIEW_IN_REVIEW: LABEL_IN_REVIEW,
    REVIEW_REVIEWED: LABEL_REVIEWED,
    REVIEW_NEEDS_REVISION: LABEL_NEEDS_REVISION,
    REVIEW_REJECTED: LABEL_REJECTED,
}

_TRANSLATION_STATUS_LABELS: Final[dict[str, str]] = {
    "draft": LABEL_DRAFT,
    "reviewed": LABEL_REVIEWED,
}

_TAG_KIND_LABELS: Final[dict[str, str]] = {
    TAG_TOPIC: LABEL_TOPIC,
    TAG_ENTITY: LABEL_ENTITY,
    TAG_CUSTOM: LABEL_CUSTOM,
    "review": LABEL_REVIEW_TAG,
}

# Free-text discriminators (no enum): we slug-case → PascalCase
# e.g. "fact-checker" → "FactChecker", "vision_ocr" → "VisionOcr"
_SLUG_SPLIT: Final = _re.compile(r"[^A-Za-z0-9]+")


def _pascal_case(value: str) -> str:
    """Turn an agent-role-style string into a PascalCase label.
    `'fact-checker'` → `'FactChecker'`. `'reviewer'` → `'Reviewer'`."""
    parts = [p for p in _SLUG_SPLIT.split(value.strip()) if p]
    return "".join(p[:1].upper() + p[1:].lower() for p in parts) if parts else ""


# Public lookup — what the rest of the code calls.
_DISCRIMINATOR_MAPS: Final[dict[str, dict[str, str]]] = {
    "agent.kind": _AGENT_KIND_LABELS,
    "chunk.status": _CHUNK_STATUS_LABELS,
    "summary.verification_status": _SUMMARY_STATUS_LABELS,
    "review.status": _REVIEW_STATUS_LABELS,
    "translation.status": _TRANSLATION_STATUS_LABELS,
    "tag.kind": _TAG_KIND_LABELS,
}


def label_for(discriminator: str, value: str) -> str:
    """Canonical label name for a user-supplied discriminator value.

    `discriminator` is one of the keys in `_DISCRIMINATOR_MAPS`. For
    free-text discriminators not in the map (notably `'agent.role'`),
    falls back to PascalCase of the value (`'reviewer'` → `'Reviewer'`,
    `'fact-checker'` → `'FactChecker'`).

    Empty / missing values return `""` so callers can guard with truthy
    checks.
    """
    if not value:
        return ""
    table = _DISCRIMINATOR_MAPS.get(discriminator)
    if table is not None:
        mapped = table.get(value)
        if mapped is not None:
            return mapped
    return _pascal_case(value)


# Convenience: all the labels that map to a single discriminator,
# useful for `remove_label` calls that need to drop "any of these".
def labels_for(discriminator: str) -> tuple[str, ...]:
    """Every label name produced by a discriminator. Used by state
    transitions that need to remove any prior label of the same kind
    before adding a new one."""
    table = _DISCRIMINATOR_MAPS.get(discriminator)
    return tuple(table.values()) if table else ()
