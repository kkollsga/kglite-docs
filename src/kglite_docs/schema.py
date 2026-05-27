"""Node + edge type names and embedding-store keys for the kglite-docs graph.

Centralised here so renames are a one-file change and so the MCP layer
and the storage layer agree on names.
"""

from __future__ import annotations

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
