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
SECTION: Final = "Section"
CHUNK: Final = "Chunk"
SUMMARY: Final = "Summary"
TAG: Final = "Tag"
CLUSTER: Final = "Cluster"
AGENT: Final = "Agent"
VIEW: Final = "View"
NOTE: Final = "Note"
REVIEW_TICKET: Final = "ReviewTicket"
REVIEW_EVENT: Final = "ReviewEvent"
STUDY: Final = "Study"
ASSESSMENT: Final = "Assessment"
FINDING: Final = "Finding"  # a cross-chunk pattern asserted over a SET of chunks
VERIFICATION_EVENT: Final = "VerificationEvent"
SYNTHESIS_EVENT: Final = "SynthesisEvent"  # a ledger-wide pattern pass (or its acknowledged skip)
REVIEW_ROUND: Final = "ReviewRound"  # one escalation pass (a level/kind/lens over a unit set)
STUDY_RECOMMENDATION: Final = "StudyRecommendation"  # a proposed follow-on study (never auto-runs)
EVENT: Final = "Event"  # a dated occurrence (actor/action/outcome) on the timeline
REPORT: Final = "Report"  # a versioned markdown report attached to a study
CHECKOUT: Final = "Checkout"  # punchcard: a batch of chunks claimed by an agent

# Edge types
HAS_PAGE: Final = "HAS_PAGE"
HAS_SECTION: Final = "HAS_SECTION"            # Document → Section (Section → Chunk reuses HAS_CHUNK)
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
HAS_VERIFICATION: Final = "HAS_VERIFICATION"  # Summary/Assessment → VerificationEvent
ASSESSED_AS: Final = "ASSESSED_AS"            # Chunk → Assessment (mirrors TAGGED_AS)
OF_STUDY: Final = "OF_STUDY"                  # Assessment → Study (mirrors OF_TAG)
CHECKED_OUT: Final = "CHECKED_OUT"            # Checkout → Chunk (the punched cards)
HOLDS: Final = "HOLDS"                        # Agent → Checkout
USED_CONTEXT: Final = "USED_CONTEXT"          # Assessment → Chunk (neighbors read to interpret the focal chunk)
SUPERSEDES: Final = "SUPERSEDES"              # Assessment → Assessment (the one it replaces)
SUPPORTED_BY: Final = "SUPPORTED_BY"          # Finding → Chunk (the chunks a cross-chunk pattern rests on)
HAS_SYNTHESIS_EVENT: Final = "HAS_SYNTHESIS_EVENT"  # Study → SynthesisEvent
HAS_ROUND: Final = "HAS_ROUND"                # Study → ReviewRound
CONDUCTED_BY: Final = "CONDUCTED_BY"          # ReviewRound → Agent
EXAMINED: Final = "EXAMINED"                  # ReviewRound → Chunk|Finding (the coverage record)
RECOMMENDS: Final = "RECOMMENDS"              # Study → StudyRecommendation
SPAWNED_FROM: Final = "SPAWNED_FROM"          # Study(child) → Study(source) (approved follow-on)
HAS_EVENT: Final = "HAS_EVENT"                # Document → Event (and Chunk → Event when chunk-anchored)
HAS_REPORT: Final = "HAS_REPORT"              # Study → Report (versioned markdown report)

# Punchcard lease: a checkout older than this is treated as abandoned and
# its chunks become claimable again (and are GC'd on the next claim).
CLAIM_TTL_SECONDS: Final = 1800  # 30 min

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

# Chunk content kind — a deterministic triage signal computed at ingest.
CONTENT_PROSE: Final = "prose"
CONTENT_TABLE: Final = "table"
CONTENT_LIST: Final = "list"
CONTENT_CODE: Final = "code"
CONTENT_SPARSE: Final = "sparse"

# Chunk embedding lifecycle (independent of status). Embedding is an
# optional, explicit phase: `ingest` writes ready chunks as pending;
# `index` flips them to done. Lets non-semantic workflows skip the model.
CHUNK_EMBED_PENDING: Final = "pending"
CHUNK_EMBED_DONE: Final = "done"

# Evidence-study: stance of an Assessment toward a Study's question
STANCE_SUPPORTS: Final = "supports"
STANCE_AGAINST: Final = "against"
STANCE_NEUTRAL: Final = "neutral"
STANCE_DEFERRED: Final = "deferred"  # read but unjudgeable yet (blocked/needs evidence)
VALID_STANCES: Final = frozenset(
    {STANCE_SUPPORTS, STANCE_AGAINST, STANCE_NEUTRAL, STANCE_DEFERRED}
)

# Evidence-study: provenance of an Assessment — *what was actually checked* to
# reach it (the basis), orthogonal to `weight` (the strength).
PROVENANCE_PRIMARY: Final = "primary_text"        # read the actual source text
PROVENANCE_CHARACTERIZATION: Final = "characterization"  # a paraphrase/summary, not the source
PROVENANCE_SCANNED_UNREAD: Final = "scanned_unread"      # a scan no one actually read (provisional)
PROVENANCE_DEFAULT: Final = PROVENANCE_PRIMARY
VALID_PROVENANCE: Final = frozenset(
    {PROVENANCE_PRIMARY, PROVENANCE_CHARACTERIZATION, PROVENANCE_SCANNED_UNREAD}
)

# Study lifecycle
STUDY_OPEN: Final = "open"
STUDY_CLOSED: Final = "closed"

# Assessment verification verdicts (verified/disputed mirror summaries;
# duplicate is the "these are the same" outcome from second-agent review)
ASSESSMENT_UNVERIFIED: Final = "unverified"
ASSESSMENT_VERIFIED: Final = "verified"
ASSESSMENT_DISPUTED: Final = "disputed"
ASSESSMENT_DUPLICATE: Final = "duplicate"
VALID_ASSESSMENT_VERDICTS: Final = frozenset(
    {ASSESSMENT_VERIFIED, ASSESSMENT_DISPUTED, ASSESSMENT_DUPLICATE}
)

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

# Chunk embedding lifecycle → labels. `MATCH (c:Chunk:Unembedded)` is the
# work-list `index` drains; `:Embedded` marks searchable chunks.
LABEL_EMBEDDED: Final = "Embedded"
LABEL_UNEMBEDDED: Final = "Unembedded"

# Chunk content kind → labels (deterministic triage signal; `MATCH (c:Chunk:Table)`)
LABEL_PROSE: Final = "Prose"
LABEL_TABLE: Final = "Table"
LABEL_LIST_BLOCK: Final = "ListBlock"
LABEL_CODE: Final = "Code"
LABEL_SPARSE: Final = "Sparse"

# Independent chunk flags (additive, not part of a one-of-N swap set)
LABEL_LOW_QUALITY: Final = "LowQuality"   # text looks garbled (bad OCR/encoding)
LABEL_BOILERPLATE: Final = "Boilerplate"  # repeated header/footer across pages

# Domain element-classification markers (core; the element *values* are supplied
# by a domain schema via register_element_discriminator).
CLASSIFY_DONE: Final = "classified"
CLASSIFY_NONE: Final = "unclassified"
LABEL_CLASSIFIED: Final = "Classified"
LABEL_UNCLASSIFIED: Final = "Unclassified"
LABEL_CONTESTED: Final = "Contested"      # multi-agent classification divergence

# Structured-entity presence flags (multi-valued → independent labels, not a
# discriminator): `MATCH (c:Chunk:HasMoney)`.
LABEL_HAS_DATE: Final = "HasDate"
LABEL_HAS_MONEY: Final = "HasMoney"
LABEL_HAS_EMAIL: Final = "HasEmail"
LABEL_HAS_URL: Final = "HasUrl"
LABEL_HAS_IDENTIFIER: Final = "HasIdentifier"

#: Entity type (from `signals.extract_entities`) → presence label.
ENTITY_LABELS: Final[dict[str, str]] = {
    "date": LABEL_HAS_DATE,
    "money": LABEL_HAS_MONEY,
    "email": LABEL_HAS_EMAIL,
    "url": LABEL_HAS_URL,
    "identifier": LABEL_HAS_IDENTIFIER,
}

# Assessment stance → labels (`MATCH (a:Assessment:Supports)`)
LABEL_SUPPORTS: Final = "Supports"
LABEL_AGAINST: Final = "Against"
LABEL_NEUTRAL: Final = "Neutral"
LABEL_DEFERRED: Final = "Deferred"

# Assessment provenance → labels (`MATCH (a:Assessment:PrimaryText)`)
LABEL_PRIMARY_TEXT: Final = "PrimaryText"
LABEL_CHARACTERIZATION: Final = "Characterization"
LABEL_SCANNED_UNREAD: Final = "ScannedUnread"

# Study lifecycle → labels
LABEL_OPEN: Final = "Open"
LABEL_CLOSED: Final = "Closed"

# Study synthesis status — has a ledger-wide cross-chunk pattern pass run? The
# honest-completeness gate: `conclude_study` refuses while this is `pending`
# unless an override is explicitly recorded. (one-of-N via swap_label)
SYNTHESIS_PENDING: Final = "pending"
SYNTHESIS_DONE: Final = "done"
LABEL_SYNTHESIS_PENDING: Final = "SynthesisPending"
LABEL_SYNTHESIS_DONE: Final = "SynthesisDone"
VALID_SYNTHESIS_STATUS: Final = frozenset({SYNTHESIS_PENDING, SYNTHESIS_DONE})

# Review round — one escalation pass. `kind` is a fixed taxonomy; `lens` is an
# open registry (see lenses.py), so lens is a property, not a discriminator.
ROUND_SCORE: Final = "score"
ROUND_VERIFY: Final = "verify"
ROUND_SYNTHESIZE: Final = "synthesize"
ROUND_PANEL: Final = "panel"
ROUND_EXPERT: Final = "expert"
VALID_ROUND_KINDS: Final = frozenset(
    {ROUND_SCORE, ROUND_VERIFY, ROUND_SYNTHESIZE, ROUND_PANEL, ROUND_EXPERT}
)
ROUND_OPEN: Final = "open"
ROUND_DONE: Final = "done"
LABEL_ROUND_OPEN: Final = "RoundOpen"
LABEL_ROUND_DONE: Final = "RoundDone"
VALID_ROUND_SCOPES: Final = frozenset({"contested", "low_depth", "uncovered", "all"})

# Study recommendation — a proposed follow-on study (a finding implies a new
# question). A proposal a human approves, never auto-run.
REC_PROPOSED: Final = "proposed"
REC_APPROVED: Final = "approved"
REC_DISMISSED: Final = "dismissed"
LABEL_REC_PROPOSED: Final = "RecProposed"
LABEL_REC_APPROVED: Final = "RecApproved"
LABEL_REC_DISMISSED: Final = "RecDismissed"
VALID_REC_STATUS: Final = frozenset({REC_PROPOSED, REC_APPROVED, REC_DISMISSED})

# Assessment verification → labels (Unverified/Verified/Disputed reuse the
# summary label *names*; Duplicate is study-specific)
LABEL_DUPLICATE: Final = "Duplicate"

# Summary verification → labels (also used for ReviewTicket overlap)
LABEL_UNVERIFIED: Final = "Unverified"
LABEL_VERIFIED: Final = "Verified"
LABEL_DISPUTED: Final = "Disputed"
LABEL_STALE: Final = "Stale"

# Finding escalation state — the aggregate of independent verification votes on a
# cross-chunk Finding (one-of-N via swap_label): `MATCH (f:Finding:Contested)`.
# (LABEL_CONTESTED is reused; labels are per-node-type so the Chunk/Finding
# `Contested` namespaces don't collide.)
ESCALATION_NEEDS_MORE: Final = "needs_more"   # too few reviewers to be confident
ESCALATION_SETTLED: Final = "settled"         # reviewers concur
ESCALATION_CONTESTED: Final = "contested"     # reviewers split
LABEL_SETTLED: Final = "Settled"
LABEL_NEEDS_MORE: Final = "NeedsMore"
VALID_ESCALATION_STATES: Final = frozenset(
    {ESCALATION_NEEDS_MORE, ESCALATION_SETTLED, ESCALATION_CONTESTED}
)

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

_CHUNK_EMBED_LABELS: Final[dict[str, str]] = {
    CHUNK_EMBED_PENDING: LABEL_UNEMBEDDED,
    CHUNK_EMBED_DONE: LABEL_EMBEDDED,
}

_CHUNK_CONTENT_KIND_LABELS: Final[dict[str, str]] = {
    CONTENT_PROSE: LABEL_PROSE,
    CONTENT_TABLE: LABEL_TABLE,
    CONTENT_LIST: LABEL_LIST_BLOCK,
    CONTENT_CODE: LABEL_CODE,
    CONTENT_SPARSE: LABEL_SPARSE,
}

# Classification lifecycle marker (one-of-N via swap_label): a ready chunk is
# either Classified or Unclassified — exhaustive, so the work-list never leaks.
_CHUNK_CLASSIFY_LABELS: Final[dict[str, str]] = {
    CLASSIFY_DONE: LABEL_CLASSIFIED,
    CLASSIFY_NONE: LABEL_UNCLASSIFIED,
}

_STUDY_STANCE_LABELS: Final[dict[str, str]] = {
    STANCE_SUPPORTS: LABEL_SUPPORTS,
    STANCE_AGAINST: LABEL_AGAINST,
    STANCE_NEUTRAL: LABEL_NEUTRAL,
    STANCE_DEFERRED: LABEL_DEFERRED,
}

_ASSESSMENT_PROVENANCE_LABELS: Final[dict[str, str]] = {
    PROVENANCE_PRIMARY: LABEL_PRIMARY_TEXT,
    PROVENANCE_CHARACTERIZATION: LABEL_CHARACTERIZATION,
    PROVENANCE_SCANNED_UNREAD: LABEL_SCANNED_UNREAD,
}

_FINDING_ESCALATION_LABELS: Final[dict[str, str]] = {
    ESCALATION_NEEDS_MORE: LABEL_NEEDS_MORE,
    ESCALATION_SETTLED: LABEL_SETTLED,
    ESCALATION_CONTESTED: LABEL_CONTESTED,
}

_STUDY_STATUS_LABELS: Final[dict[str, str]] = {
    STUDY_OPEN: LABEL_OPEN,
    STUDY_CLOSED: LABEL_CLOSED,
}

_STUDY_SYNTHESIS_LABELS: Final[dict[str, str]] = {
    SYNTHESIS_PENDING: LABEL_SYNTHESIS_PENDING,
    SYNTHESIS_DONE: LABEL_SYNTHESIS_DONE,
}

_ROUND_STATUS_LABELS: Final[dict[str, str]] = {
    ROUND_OPEN: LABEL_ROUND_OPEN,
    ROUND_DONE: LABEL_ROUND_DONE,
}

_REC_STATUS_LABELS: Final[dict[str, str]] = {
    REC_PROPOSED: LABEL_REC_PROPOSED,
    REC_APPROVED: LABEL_REC_APPROVED,
    REC_DISMISSED: LABEL_REC_DISMISSED,
}

_ASSESSMENT_STATUS_LABELS: Final[dict[str, str]] = {
    ASSESSMENT_UNVERIFIED: LABEL_UNVERIFIED,
    ASSESSMENT_VERIFIED: LABEL_VERIFIED,
    ASSESSMENT_DISPUTED: LABEL_DISPUTED,
    ASSESSMENT_DUPLICATE: LABEL_DUPLICATE,
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
    "chunk.embedding": _CHUNK_EMBED_LABELS,
    "chunk.content_kind": _CHUNK_CONTENT_KIND_LABELS,
    "chunk.classify": _CHUNK_CLASSIFY_LABELS,
    "summary.verification_status": _SUMMARY_STATUS_LABELS,
    "review.status": _REVIEW_STATUS_LABELS,
    "translation.status": _TRANSLATION_STATUS_LABELS,
    "tag.kind": _TAG_KIND_LABELS,
    "study.stance": _STUDY_STANCE_LABELS,
    "study.status": _STUDY_STATUS_LABELS,
    "study.synthesis_status": _STUDY_SYNTHESIS_LABELS,
    "round.status": _ROUND_STATUS_LABELS,
    "recommendation.status": _REC_STATUS_LABELS,
    "assessment.verification_status": _ASSESSMENT_STATUS_LABELS,
    "assessment.provenance": _ASSESSMENT_PROVENANCE_LABELS,
    "finding.escalation_state": _FINDING_ESCALATION_LABELS,
}


# ─── Domain element-classification registry ───────────────────────────────
# Built-in discriminators above are frozen. Domain schemas (e.g. the legal data
# pack) register *element* discriminators at import — kept in a separate mutable
# registry so the literal stays `Final`. Core never names a domain term; it only
# routes opaque element tokens. Element values are unique across all registered
# element discriminators, so a flat `element="holding"` lookup is unambiguous.
_REGISTERED: dict[str, dict[str, str]] = {}
_ELEMENT_DISCRIMINATORS: set[str] = set()
_ELEMENT_VALUE_LABELS: dict[str, str] = {}  # element value → label, flat union


def register_element_discriminator(name: str, mapping: dict[str, str]) -> None:
    """Register a domain element discriminator (value → label). Idempotent for an
    identical re-registration; raises on a conflicting redefinition or on an
    element value already owned by another element discriminator (values are
    globally unique so `element="…"` routing is unambiguous)."""
    if name in _DISCRIMINATOR_MAPS:
        raise ValueError(f"{name!r} is a built-in discriminator, not registrable")
    existing = _REGISTERED.get(name)
    if existing is not None:
        if existing == mapping:
            return  # idempotent
        raise ValueError(f"element discriminator {name!r} already registered differently")
    for value, label in mapping.items():
        owner = _ELEMENT_VALUE_LABELS.get(value)
        if owner is not None and owner != label:
            raise ValueError(
                f"element value {value!r} already registered (as {owner!r}); "
                "element values must be unique across element discriminators"
            )
    _REGISTERED[name] = dict(mapping)
    _ELEMENT_DISCRIMINATORS.add(name)
    _ELEMENT_VALUE_LABELS.update(mapping)


def valid_element_values() -> frozenset[str]:
    """Every element value registered by a domain schema. The allow-list a
    study's `element=` scope is validated against (unknown ⇒ raise, never a
    silent zero-match)."""
    return frozenset(_ELEMENT_VALUE_LABELS)


def element_label(value: str) -> str | None:
    """Label for a registered element value, or `None` if it isn't registered.
    Callers MUST raise on `None` rather than fall back to PascalCase — an
    unknown/skewed element token must never silently match zero chunks."""
    return _ELEMENT_VALUE_LABELS.get(value)


def label_for(discriminator: str, value: str) -> str:
    """Canonical label name for a user-supplied discriminator value.

    `discriminator` is a key in `_DISCRIMINATOR_MAPS` or a registered element
    discriminator. For free-text discriminators not in either (notably
    `'agent.role'`), falls back to PascalCase of the value (`'reviewer'` →
    `'Reviewer'`, `'fact-checker'` → `'FactChecker'`).

    Empty / missing values return `""` so callers can guard with truthy checks.
    """
    if not value:
        return ""
    table = _DISCRIMINATOR_MAPS.get(discriminator) or _REGISTERED.get(discriminator)
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
    table = _DISCRIMINATOR_MAPS.get(discriminator) or _REGISTERED.get(discriminator)
    return tuple(table.values()) if table else ()
