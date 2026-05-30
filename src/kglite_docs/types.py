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
#: `Study` lets a study conclusion be stored as a Summary on the Study node.
TargetKind = Literal["Chunk", "Page", "Document", "Summary", "Study"]

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

#: Stance of an evidence `Assessment` toward a `Study`'s question.
Stance = Literal["supports", "against", "neutral", "deferred"]

#: Provenance of an `Assessment` — what was actually checked (the basis), as
#: opposed to `weight` (the strength).
Provenance = Literal["primary_text", "characterization", "scanned_unread"]

#: Verdict a second agent assigns when verifying an `Assessment`.
AssessmentVerdict = Literal["verified", "disputed", "duplicate"]

#: Verification status an `Assessment` can be in.
AssessmentStatus = Literal["unverified", "verified", "disputed", "duplicate"]

#: Lifecycle of a `Study`.
StudyStatus = Literal["open", "closed"]


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


class SectionRow(TypedDict, total=False):
    """One row from `Corpus.list_sections()` — the grain between document and
    chunk (derived from the PDF outline or top-level headings at ingest)."""

    id: str
    doc_id: str
    title: str
    page_start: int
    page_end: int
    level: int
    doc_type: str
    chunk_count: int


class ChunkDetail(TypedDict, total=False):
    """`Corpus.get_chunk()` return shape."""

    id: str
    title: str
    doc_id: str
    page: int
    chunk_index: int
    text: str
    token_count: int
    word_count: int
    char_count: int
    content_kind: str
    quality_score: float
    boilerplate: bool
    entities: dict[str, list[str]]
    headings: str
    section_id: str
    doc_type: str
    status: ChunkStatus
    view_count: int
    next_id: str
    prev_id: str
    context_before: list[dict[str, Any]]
    context_after: list[dict[str, Any]]
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


class CoverageDocRow(TypedDict):
    """Per-document row in `coverage_report()`."""

    doc_id: str
    title: str
    format: str
    pages: int
    pending_ocr: int
    image_pages: int
    low_text_pages: int
    extractable_text_ratio: float


class CoverageReport(TypedDict):
    """`Corpus.coverage_report()` return shape — honest extraction + embedding
    coverage with a human-readable `summary`."""

    documents: list[CoverageDocRow]
    total_pages: int
    image_pages: int
    low_text_pages: int
    pending_ocr: int
    embedded: int
    unembedded: int
    summary: str


class CorpusStatus(TypedDict):
    """`Corpus.status()` — one-call snapshot of the corpus."""

    docs: int
    pages: int
    chunks: int
    embedded: int
    unembedded: int
    image_pages: int
    pending_ocr: int
    classified: int
    unclassified: int
    contested: int
    studies: int


class TriageMap(TypedDict, total=False):
    """`Corpus.triage_map()` — aggregated content signals for orientation."""

    chunks: int
    ready: int
    embedded: int
    unembedded: int
    pending_ocr: int
    sections: int
    content_kinds: dict[str, int]
    boilerplate: int
    low_quality: int
    entities: dict[str, int]
    classified: int
    unclassified: int
    contested: int
    elements: dict[str, int]
    summary: str


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
    """`Corpus.list_agents()` row — identity + counters, no template."""

    id: str
    kind: AgentKind
    model: str
    role: str
    description: str
    first_seen: str
    last_seen: str
    actions: int


class AgentConfig(TypedDict, total=False):
    """`Corpus.get_agent()` return shape — full template + counters.

    `tools` and `context` are hydrated from the underlying JSON
    properties so callers receive a real list / dict, not strings."""

    id: str
    kind: AgentKind
    model: str
    role: str
    system_prompt: str
    tools: list[str]
    context: dict[str, Any]
    description: str
    first_seen: str
    last_seen: str
    action_count: int


class AgentActivity(TypedDict, total=False):
    """`Corpus.agent_activity()` return shape — what an agent has
    done, optionally scoped to one target node."""

    agent: AgentConfig | None
    views: list[dict[str, Any]]
    summaries: list[dict[str, Any]]
    tags: list[dict[str, Any]]
    translations: list[dict[str, Any]]
    review_events: list[dict[str, Any]]
    verification_events: list[dict[str, Any]]


class TagRow(TypedDict, total=False):
    """`Corpus.list_tags()` row."""

    tag_id: str
    name: str
    kind: TagKind
    by_agent: str
    at: str
    tagging_id: str
    confidence: float | None
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
    searched_fraction: float
    items: list[ContextItem]


class GroundingSentence(TypedDict):
    """One sentence's grounding analysis inside `check_grounding`."""

    sentence: str
    best_chunk_id: str
    best_score: float
    supported: bool


class ComparisonQueryResult(TypedDict):
    """Per-query hits from each side of a `compare_documents` call."""

    query: str
    doc_a_hits: list[SearchHit]
    doc_b_hits: list[SearchHit]
    merged_context: ComposedContext


class ComparisonResult(TypedDict):
    """`Corpus.compare_documents()` return shape.

    Side-by-side cross-document retrieval result. For each query in the
    input list, you get the top hits from each document independently
    plus a budgeted merged context bundle suitable for handing to a
    downstream LLM that's writing a comparison."""

    doc_a_id: str
    doc_b_id: str
    doc_a_title: str
    doc_b_title: str
    queries: list[ComparisonQueryResult]


class GroundingReport(TypedDict):
    """`Corpus.check_grounding()` return shape."""

    summary_id: str
    sentences: list[GroundingSentence]
    supported_fraction: float
    grounding_score: float
    weak_sentences: list[GroundingSentence]
    threshold: float


# ─── evidence study ────────────────────────────────────────────────────────


class StudyRow(TypedDict, total=False):
    """One study from `Corpus.list_studies()` / `get_study()`."""

    id: str
    title: str
    question: str
    status: StudyStatus
    created_by: str
    created_at: str
    assessment_count: int
    tallies: dict[str, Any]          # {supports, against, neutral, *_weight}
    conclusions: list[SummaryRow]    # only on get_study


class AssessmentRow(TypedDict, total=False):
    """One ranked row in a study `ledger`."""

    assessment_id: str
    chunk_id: str
    doc_id: str
    page: int
    stance: Stance
    weight: float
    provenance: Provenance
    rationale: str
    quote: str
    char_start: int
    char_end: int
    by_agent: str
    verification_status: AssessmentStatus
    superseded: bool
    context_chunk_ids: list[str]
    text: str


class Ledger(TypedDict, total=False):
    """`Corpus.study_ledger()` return shape — weight-ranked evidence."""

    study_id: str
    question: str
    status: StudyStatus
    rows: list[AssessmentRow]
    total: int
    returned: int
    tallies: dict[str, Any]
    scope_coverage: dict[str, int]  # present only when `element=` scoped


class FindingRow(TypedDict, total=False):
    """One cross-chunk Finding from `Corpus.list_findings()` — a pattern asserted
    over a set of chunks."""

    finding_id: str
    statement: str
    finding_type: str
    stance: Stance
    weight: float
    provenance: Provenance
    rationale: str
    by_agent: str
    verification_status: AssessmentStatus
    supporting: list[dict[str, Any]]  # [{id, doc_id, page}, …]


class ConflictRow(TypedDict, total=False):
    """One contested chunk — current `supports` vs `against` assessments."""

    chunk_id: str
    doc_id: str
    page: int
    text: str
    supports: list[AssessmentRow]
    against: list[AssessmentRow]


class ConflictReport(TypedDict, total=False):
    """`Corpus.study_conflicts()` return shape."""

    study_id: str
    question: str
    conflicts: list[ConflictRow]
    total: int


__all__ = [
    "AgentActivity", "AgentConfig", "AgentKind", "AgentRow",
    "ChunkDetail", "ChunkStatus",
    "ComparisonQueryResult", "ComparisonResult",
    "ClusterAlgorithm",
    "ComposedContext", "ContextItem",
    "DocumentDetail", "DocumentRow",
    "ExportFormat",
    "GroundingReport", "GroundingSentence",
    "IngestSummary",
    "OcrStatus", "OcrStatusRow",
    "CoverageReport", "CoverageDocRow", "CorpusStatus", "TriageMap",
    "PendingOcrRow",
    "ReviewEvent", "ReviewStats", "ReviewStatus",
    "ReviewTicketDetail", "ReviewTicketRow", "ReviewVerdict",
    "SearchHit",
    "SectionRow",
    "SummaryDepth", "SummaryRow", "SummaryStatus", "SummaryVerdict",
    "TagKind", "TagRow",
    "TargetKind",
    "TranslationStatus",
    "Stance", "Provenance", "AssessmentVerdict", "AssessmentStatus", "StudyStatus",
    "StudyRow", "AssessmentRow", "Ledger", "ConflictRow", "ConflictReport",
    "FindingRow",
]
