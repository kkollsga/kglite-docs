"""Public type aliases.

`TypedDict`s for the dict-shaped values our methods return, and `Literal`s
for the enum-ish string arguments. Importing from here gives IDE/type-checker
autocomplete on result fields and rejects typos in arguments.

Example::

    from kglite_docs.types import Verdict, SearchHit

    def my_fn(verdict: Verdict) -> list[SearchHit]:
        ...
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


# ─── Literal enums ─────────────────────────────────────────────────────────

#: Verdict an agent assigns when verifying a `Summary`.
SummaryVerdict = Literal["verified", "disputed", "needs_revision"]

#: Statuses a `Summary` can be in (computed from the latest VerificationEvent).
SummaryStatus = Literal["unverified", "verified", "disputed", "needs_revision", "stale"]

#: `depth` argument to `add_summary` — what scope does this summary cover.
SummaryDepth = Literal["chunk", "section", "document"]

#: `target_kind` argument — what kind of node a Summary, View, ReviewTicket targets.
TargetKind = Literal["Chunk", "Page", "Document", "Summary"]

#: `kind` argument when creating a Tag.
TagKind = Literal["topic", "entity", "custom", "review"]

#: Lifecycle of a `Chunk` after ingest.
ChunkStatus = Literal["ready", "needs_ocr", "empty"]

#: Lifecycle of a `Translation`.
TranslationStatus = Literal["draft", "reviewed"]

#: Algorithms `cluster_chunks` accepts.
ClusterAlgorithm = Literal["louvain", "kmeans", "dbscan"]

#: Verdict an agent assigns when completing a review ticket.
ReviewVerdict = Literal["reviewed", "needs_revision", "rejected"]

#: All states a review ticket can be in.
ReviewStatus = Literal["new", "in_review", "reviewed", "needs_revision", "rejected"]

#: Kinds of agent identity recorded on the Agent node.
AgentKind = Literal["llm", "human", "service"]

#: Export targets for `export_document` / `export_cluster` / `export_bundle`.
ExportFormat = Literal["md", "docx", "pdf"]


# ─── Result shapes ─────────────────────────────────────────────────────────


class SearchHit(TypedDict, total=False):
    """One row from `Corpus.search()`. Fields beyond `id`, `score`,
    `type` are best-effort — some are added by post-join enrichment and
    may be missing when the underlying chunk has unusual state."""

    id: str
    score: float
    type: str
    title: str
    text: str
    doc_id: str
    page: int
    headings: str
    status: ChunkStatus
    summaries: list[dict[str, Any]]


class DocumentRow(TypedDict):
    """One row from `Corpus.list_documents()`."""

    id: str
    title: str
    pages: int
    ingested_at: str
    bytes: int
    chunk_count: int


class DocumentDetail(TypedDict, total=False):
    """`Corpus.get_document()` return shape."""

    id: str
    title: str
    pages: int
    ingested_at: str
    bytes: int
    path: str
    metadata_json: str
    toc: list[dict[str, Any]]


class ChunkDetail(TypedDict, total=False):
    """`Corpus.get_chunk()` return shape."""

    id: str
    title: str
    doc_id: str
    page: int
    chunk_index: int
    text: str
    token_count: int
    headings: str
    status: ChunkStatus
    view_count: int
    next_id: str
    prev_id: str
    summaries: list[dict[str, Any]]


class SummaryRow(TypedDict, total=False):
    """`Corpus.get_summaries()` row."""

    id: str
    text: str
    depth: SummaryDepth
    status: SummaryStatus
    created_at: str
    verified_at: str | None
    verifier_agent: str | None
    notes: str
    model: str
    source_text_hash: str
    author_agent: str


class IngestSummary(TypedDict, total=False):
    """Aggregate of `Corpus.ingest_dir()` results."""

    ingested: int
    skipped: int
    total_chunks: int
    ocr_pending: int


class PendingOcrRow(TypedDict, total=False):
    """`Corpus.list_pending_ocr()` row."""

    page_id: str
    doc_id: str
    page_number: int
    doc_path: str
    doc_title: str
    image_b64: str
    image_mime: str
    image_error: str


class OcrStatusRow(TypedDict):
    """Per-document OCR status (the `documents` array in `ocr_status()`)."""

    doc_id: str
    title: str
    format: str
    pages: int
    ready: int
    pending: int
    pending_fraction: float


class OcrStatus(TypedDict):
    """`Corpus.ocr_status()` return shape."""

    total_pages: int
    ready_pages: int
    pending_pages: int
    documents_total: int
    documents_with_pending: int
    documents: list[OcrStatusRow]


class ReviewTicketRow(TypedDict, total=False):
    """One row from `Corpus.list_review_queue()`. Add `target` and
    `events` by going through `Corpus.get_review_ticket()`."""

    id: str
    target_id: str
    target_kind: TargetKind
    priority: int
    created_at: str
    created_by: str
    note: str
    status: ReviewStatus
    claimed_by: str | None


class ReviewEvent(TypedDict, total=False):
    """One event in a ticket's audit trail."""

    type: ReviewStatus
    notes: str
    accuracy: float | None
    authenticity: str
    at: str
    agent: str


class ReviewTicketDetail(ReviewTicketRow, total=False):
    """`Corpus.get_review_ticket()` return shape."""

    target: dict[str, Any]
    events: list[ReviewEvent]


class ReviewStats(TypedDict):
    """`Corpus.review_stats()` return shape."""

    tickets_total: int
    by_status: dict[str, int]
    in_review_by_agent: dict[str, int]


class AgentRow(TypedDict, total=False):
    """`Corpus.list_agents()` row."""

    id: str
    kind: AgentKind
    model: str
    first_seen: str
    last_seen: str
    actions: int


class TagRow(TypedDict, total=False):
    """`Corpus.list_tags()` row."""

    tag_id: str
    name: str
    kind: TagKind
    by_agent: str
    at: str
    tagging_id: str
    chunk_id: str
    doc_id: str


class ContextItem(TypedDict, total=False):
    """One entry in `Corpus.compose_context()['items']`."""

    chunk_id: str
    doc_id: str
    page: int
    headings: str
    score: float
    text: str
    summaries: list[dict[str, Any]]
    tokens: int


class ComposedContext(TypedDict):
    """`Corpus.compose_context()` return shape."""

    query: str
    budget_tokens: int
    used_tokens: int
    items: list[ContextItem]


class GroundingSentence(TypedDict):
    """One sentence's grounding analysis inside `check_grounding`."""

    sentence: str
    best_chunk_id: str
    best_score: float
    supported: bool


class GroundingReport(TypedDict):
    """`Corpus.check_grounding()` return shape."""

    summary_id: str
    sentences: list[GroundingSentence]
    supported_fraction: float
    grounding_score: float
    weak_sentences: list[GroundingSentence]
    threshold: float


__all__ = [
    "AgentKind", "AgentRow",
    "ChunkDetail", "ChunkStatus",
    "ClusterAlgorithm",
    "ComposedContext", "ContextItem",
    "DocumentDetail", "DocumentRow",
    "ExportFormat",
    "GroundingReport", "GroundingSentence",
    "IngestSummary",
    "OcrStatus", "OcrStatusRow",
    "PendingOcrRow",
    "ReviewEvent", "ReviewStats", "ReviewStatus",
    "ReviewTicketDetail", "ReviewTicketRow", "ReviewVerdict",
    "SearchHit",
    "SummaryDepth", "SummaryRow", "SummaryStatus", "SummaryVerdict",
    "TagKind", "TagRow",
    "TargetKind",
    "TranslationStatus",
]
