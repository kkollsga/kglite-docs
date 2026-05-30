"""Evidence studies — judge document chunks for/against a stored question.

A `Study` is a question/claim. Agents record an `Assessment` on each chunk:
a stance (supports / against / neutral / deferred), a probative `weight` in
[0, 1], and a free-text `rationale`. `deferred` = "read but can't judge yet"
(blocked / needs evidence) — counted distinctly and kept in the work-list. Assessments are reified nodes (like `Tagging` /
`ReviewEvent`) so multiple agents can co-assess the same chunk and each
assessment is independently verifiable.

Design notes:

- **Append-only, latest-wins.** Re-assessing the same (study, chunk, agent)
  writes a new `Assessment`; reads dedup to the most recent. This sidesteps
  kglite's String-`SET` hazard, gives free revision history, and matches the
  review/verify event-sourcing house style.
- **Off the embedding path.** `assess` never calls the embedder — rationale is
  a plain property. A whole study can run on an un-`index`ed corpus.
- **Work-list = absence**, not tickets: "chunks lacking an Assessment for this
  study" (mirrors the `embedded` / `count_unembedded` pattern). Resumable.
- **Stance / status / verification are secondary labels** for index-speed
  filtering (`MATCH (a:Assessment:Supports:Verified)`).
- The study **conclusion** is a `Summary` targeting the `Study` (reuses
  `enrich.add_summary` / `verify_summary`).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, cast

from kglite_docs.activity import register_agent
from kglite_docs.checkout import claim_or_preview
from kglite_docs.errors import InvalidEnumError, SelfVerificationError
from kglite_docs.schema import (
    AGENT,
    ASSESSED_AS,
    ASSESSMENT,
    ASSESSMENT_UNVERIFIED,
    AUTHORED,
    CHUNK,
    CHUNK_TEXT_COL,
    CLAIM_TTL_SECONDS,
    HAS_VERIFICATION,
    OF_STUDY,
    PROVENANCE_DEFAULT,
    STANCE_AGAINST,
    STANCE_DEFERRED,
    STANCE_SUPPORTS,
    STUDY,
    STUDY_CLOSED,
    STUDY_OPEN,
    SUPERSEDES,
    USED_CONTEXT,
    VALID_ASSESSMENT_VERDICTS,
    VALID_PROVENANCE,
    VALID_STANCES,
    VERIFICATION_EVENT,
    VERIFIED_BY,
    label_for,
    labels_for,
)
from kglite_docs.store import Store
from kglite_docs.store import rows as _df_dicts

# Reverse map: stance label → user-facing stance string.
_STANCE_BY_LABEL = {label_for("study.stance", s): s for s in VALID_STANCES}
# Reverse map: verification label → status string.
_VSTATUS_BY_LABEL = {
    "Unverified": "unverified", "Verified": "verified",
    "Disputed": "disputed", "Duplicate": "duplicate",
}
_ASSESS_LABEL_SET = set(labels_for("assessment.verification_status"))
_STANCE_LABEL_SET = set(labels_for("study.stance"))
# Reverse map: provenance label → user-facing provenance string.
_PROVENANCE_BY_LABEL = {
    label_for("assessment.provenance", v): v for v in VALID_PROVENANCE
}
_PROVENANCE_LABEL_SET = set(labels_for("assessment.provenance"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── studies ────────────────────────────────────────────────────────────────


def define_study(
    store: Store,
    *,
    question: str,
    title: str | None = None,
    created_by: str,
    status: str = STUDY_OPEN,
) -> str:
    """Create a `Study` (a question/claim agents will assess evidence for).
    Returns the study id."""
    question = (question or "").strip()
    if not question:
        raise InvalidEnumError("study question must be non-empty")
    if status not in (STUDY_OPEN, STUDY_CLOSED):
        raise InvalidEnumError(f"invalid study status: {status!r}")
    register_agent(store, agent_id=created_by)
    sid = "study_" + uuid.uuid4().hex[:16]
    store.upsert_nodes(
        STUDY,
        [{
            "id": sid,
            "title": (title or question)[:120],
            "question": question,
            "created_by": created_by,
            "created_at": _now(),
            "status": status,
        }],
    )
    status_label = label_for("study.status", status)
    if status_label:
        store.add_label(STUDY, [sid], status_label)
    return sid


def list_studies(
    store: Store, *, status: str | None = None, created_by: str | None = None,
) -> list[dict[str, Any]]:
    """List studies (newest first), each with a cheap assessment count."""
    status_label = label_for("study.status", status) if status else ""
    label_clause = f":{status_label}" if status_label else ""
    where: list[str] = []
    params: dict[str, Any] = {}
    if created_by:
        where.append("s.created_by = $cb")
        params["cb"] = created_by
    where_clause = ("WHERE " + " AND ".join(where)) if where else ""
    df = store.cypher(
        f"MATCH (s:Study{label_clause}) {where_clause} "
        "OPTIONAL MATCH (s)<-[:OF_STUDY]-(a:Assessment) "
        "RETURN s.id AS id, s.title AS title, s.question AS question, "
        "s.status AS status, s.created_by AS created_by, "
        "s.created_at AS created_at, count(a) AS assessment_count "
        "ORDER BY s.created_at DESC",
        params=params,
    )
    return _df_dicts(df)


def get_study(store: Store, *, study_id: str) -> dict[str, Any] | None:
    """Study metadata + tallies + its conclusion summaries. None if missing."""
    df = store.cypher(
        "MATCH (s:Study {id: $id}) RETURN s.id AS id, s.title AS title, "
        "s.question AS question, s.status AS status, "
        "s.created_by AS created_by, s.created_at AS created_at",
        params={"id": study_id},
    )
    rows = _df_dicts(df)
    if not rows:
        return None
    study = rows[0]
    study["tallies"] = _tallies(store, study_id)
    # Conclusions are Summaries targeting the Study.
    from kglite_docs.enrich import get_summaries
    study["conclusions"] = get_summaries(
        store, target_id=study_id, target_kind=STUDY,
    )
    return study


def reopen_study(store: Store, *, study_id: str, agent_id: str) -> dict[str, Any]:
    """Flip a study back to `open` (via label swap) for deeper analysis."""
    if not _study_exists(store, study_id):
        raise InvalidEnumError(f"study not found: {study_id}")
    register_agent(store, agent_id=agent_id)
    store.cypher(
        "MATCH (s:Study {id: $id}) SET s.status = $st",
        params={"id": study_id, "st": STUDY_OPEN},
    )
    store.swap_label(
        STUDY, [study_id],
        add=label_for("study.status", STUDY_OPEN),
        remove_any_of=labels_for("study.status"),
    )
    return {"study_id": study_id, "status": STUDY_OPEN}


def conclude_study(
    store: Store,
    embedder: Any,
    *,
    study_id: str,
    text: str,
    agent_id: str,
    model: str = "",
    embed: bool = False,
) -> str:
    """Write a conclusion for the study — stored as a `Summary` on the `Study`
    node (so it is attributed, revisable, and verifiable via `summary.verify`).
    Defaults to no embedding. Returns the conclusion (Summary) id."""
    if not _study_exists(store, study_id):
        raise InvalidEnumError(f"study not found: {study_id}")
    from kglite_docs.enrich import add_summary
    return add_summary(
        store, embedder,
        target_id=study_id, target_kind=STUDY, depth="document",
        text=text, agent_id=agent_id, model=model, embed=embed,
    )


def delete_study(store: Store, *, study_id: str) -> dict[str, Any]:
    """Cascade-delete a study: its Assessments, their VerificationEvents, and
    its conclusion Summaries (+ their events). Destructive."""
    if not _study_exists(store, study_id):
        raise InvalidEnumError(f"study not found: {study_id}")
    counts = _df_dicts(store.cypher(
        "MATCH (s:Study {id: $id}) "
        "OPTIONAL MATCH (s)<-[:OF_STUDY]-(a:Assessment) "
        "OPTIONAL MATCH (a)-[:HAS_VERIFICATION]->(ae:VerificationEvent) "
        "OPTIONAL MATCH (s)<-[:SUMMARIZES]-(c:Summary) "
        "OPTIONAL MATCH (c)-[:HAS_VERIFICATION]->(ce:VerificationEvent) "
        "RETURN count(DISTINCT a) AS assessments, "
        "count(DISTINCT c) AS conclusions, "
        "count(DISTINCT ae) + count(DISTINCT ce) AS events",
        params={"id": study_id},
    ))
    # One-statement cascade: empty OPTIONAL MATCH branches resolve to NULL and
    # are skipped by DETACH DELETE (kglite >= 0.10.8).
    store.cypher(
        "MATCH (s:Study {id: $id}) "
        "OPTIONAL MATCH (s)<-[:OF_STUDY]-(a:Assessment) "
        "OPTIONAL MATCH (a)-[:HAS_VERIFICATION]->(ae:VerificationEvent) "
        "OPTIONAL MATCH (s)<-[:SUMMARIZES]-(c:Summary) "
        "OPTIONAL MATCH (c)-[:HAS_VERIFICATION]->(ce:VerificationEvent) "
        "DETACH DELETE s, a, ae, c, ce",
        params={"id": study_id},
    )
    # Checkouts reference the study by property, not edge — clean separately.
    store.cypher("MATCH (co:Checkout {study_id: $id}) DETACH DELETE co", params={"id": study_id})
    c = counts[0] if counts else {}
    return {
        "deleted_study": study_id,
        "assessments": int(c.get("assessments", 0)),
        "conclusions": int(c.get("conclusions", 0)),
        "events": int(c.get("events", 0)),
    }


# ─── assessments ──────────────────────────────────────────────────────────


def assess(
    store: Store,
    *,
    study_id: str,
    chunk_id: str,
    stance: str,
    weight: float,
    rationale: str = "",
    agent_id: str,
    model: str = "",
    provenance: str = PROVENANCE_DEFAULT,
    quote: str = "",
    char_start: int | None = None,
    char_end: int | None = None,
    context_chunk_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Record one agent's stance + probative weight + rationale on a chunk,
    toward a study's question. Append-only (latest wins). Never embeds.

    `quote`/`char_start`/`char_end` are an optional **pinpoint span** — the exact
    passage the assessment rests on, surfaced in the ledger for pinpoint cites.
    Pass `quote` alone (located in the chunk) or `char_start`+`char_end` (validated
    against the chunk length; if `quote` is also given it must match the cited
    text). An out-of-range span or a quote not found in the chunk is rejected.

    `stance` is one of supports / against / neutral / deferred. Use `deferred`
    when the chunk was read but can't be judged yet (e.g. an image-only /
    needs_ocr chunk, or a claim awaiting a source not yet ingested): it's tallied
    separately and the chunk stays in `next_unassessed` for a later pass — never
    silently dropped.

    `context_chunk_ids` are neighbor chunks the agent had to read to interpret
    the focal chunk (e.g. an incoherent chunk understood only via the ones
    around it). They're recorded as ``USED_CONTEXT`` edges so retrieval can
    pull the full span later, and so they are **excluded from the work-list**
    (no one re-judges a chunk that was already covered as context)."""
    if not _study_exists(store, study_id):
        raise InvalidEnumError(f"study not found: {study_id}")
    a = _assemble_assessment(
        store, study_id=study_id, chunk_id=chunk_id, stance=stance, weight=weight,
        rationale=rationale, agent_id=agent_id, model=model, provenance=provenance,
        quote=quote, char_start=char_start, char_end=char_end,
        context_chunk_ids=context_chunk_ids, now=_now(),
    )
    register_agent(store, agent_id=agent_id)
    _write_assessments(store, [a])
    return cast("dict[str, Any]", a["result"])


def _assemble_assessment(
    store: Store,
    *,
    study_id: str,
    chunk_id: str,
    stance: str,
    weight: float,
    agent_id: str,
    rationale: str = "",
    model: str = "",
    provenance: str = PROVENANCE_DEFAULT,
    quote: str = "",
    char_start: int | None = None,
    char_end: int | None = None,
    context_chunk_ids: list[str] | None = None,
    now: str,
) -> dict[str, Any]:
    """Validate one assessment and build its node row + edges/labels metadata —
    **no writes**. Shared by `assess` (single) and `assess_many` (batch) so both
    go through one validated code path."""
    if stance not in VALID_STANCES:
        raise InvalidEnumError(
            f"invalid stance: {stance!r} (expected one of {sorted(VALID_STANCES)})"
        )
    if provenance not in VALID_PROVENANCE:
        raise InvalidEnumError(
            f"invalid provenance: {provenance!r} (expected one of {sorted(VALID_PROVENANCE)})"
        )
    try:
        weight = float(weight)
    except (TypeError, ValueError):
        raise InvalidEnumError(f"weight must be a number in [0,1] (got {weight!r})") from None
    if not 0.0 <= weight <= 1.0:
        raise InvalidEnumError(f"weight must be in [0,1] (got {weight})")
    quote, span_start, span_end = _resolve_span(store, chunk_id, quote, char_start, char_end)
    aid = "assess_" + uuid.uuid4().hex[:16]
    ctx = [cid for cid in (context_chunk_ids or []) if cid and cid != chunk_id]
    return {
        "aid": aid,
        "agent_id": agent_id,
        "study_id": study_id,
        "chunk_id": chunk_id,
        "ctx": ctx,
        "stance_label": label_for("study.stance", stance),
        "prov_label": label_for("assessment.provenance", provenance),
        "node": {
            "id": aid,
            "title": f"{stance} ({weight:.2f}) {chunk_id}",
            "study_id": study_id,
            "chunk_id": chunk_id,
            "stance": stance,
            "weight": weight,
            "provenance": provenance,
            "rationale": rationale,
            "quote": quote,
            "char_start": span_start,
            "char_end": span_end,
            "by_agent": agent_id,
            "model": model,
            "created_at": now,
            "verification_status": ASSESSMENT_UNVERIFIED,
        },
        "result": {
            "assessment_id": aid, "study_id": study_id, "chunk_id": chunk_id,
            "stance": stance, "weight": weight, "provenance": provenance,
            "quote": quote, "char_start": span_start, "char_end": span_end,
            "context_chunk_ids": ctx,
        },
    }


def _write_assessments(store: Store, assembled: list[dict[str, Any]]) -> None:
    """Persist a batch of assembled assessments — nodes, edges, and labels — via
    the bulk API (one call per kind). Works for one row or many."""
    if not assembled:
        return
    store.upsert_nodes(ASSESSMENT, [a["node"] for a in assembled])
    store.upsert_edges(
        ASSESSED_AS, [{"src": a["chunk_id"], "dst": a["aid"]} for a in assembled],
        source_type=CHUNK, target_type=ASSESSMENT,
    )
    store.upsert_edges(
        OF_STUDY, [{"src": a["aid"], "dst": a["study_id"]} for a in assembled],
        source_type=ASSESSMENT, target_type=STUDY,
    )
    store.upsert_edges(
        AUTHORED, [{"src": a["agent_id"], "dst": a["aid"]} for a in assembled],
        source_type=AGENT, target_type=ASSESSMENT,
    )
    stance_groups: dict[str, list[str]] = {}
    prov_groups: dict[str, list[str]] = {}
    for a in assembled:
        if a["stance_label"]:
            stance_groups.setdefault(a["stance_label"], []).append(a["aid"])
        if a["prov_label"]:
            prov_groups.setdefault(a["prov_label"], []).append(a["aid"])
    for lbl, ids in stance_groups.items():
        store.add_label(ASSESSMENT, ids, lbl)
    for lbl, ids in prov_groups.items():
        store.add_label(ASSESSMENT, ids, lbl)
    init_label = label_for("assessment.verification_status", ASSESSMENT_UNVERIFIED)
    if init_label:
        store.add_label(ASSESSMENT, [a["aid"] for a in assembled], init_label)
    ctx_edges = [
        {"src": a["aid"], "dst": cid} for a in assembled for cid in a["ctx"]
    ]
    if ctx_edges:
        store.upsert_edges(
            USED_CONTEXT, ctx_edges, source_type=ASSESSMENT, target_type=CHUNK,
        )


def assess_many(
    store: Store, *, study_id: str, rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Batch-assess many chunks in one shot — one validated, batched write for a
    fan-out (and, through the MCP layer, a single persist). Each row is a dict
    with `chunk_id`/`stance`/`weight`/`agent_id` (+ optional `rationale`,
    `model`, `provenance`, `quote`/`char_start`/`char_end`, `context_chunk_ids`).
    All rows are validated *before* any write, so one bad row aborts the batch
    with nothing written."""
    if not _study_exists(store, study_id):
        raise InvalidEnumError(f"study not found: {study_id}")
    if not rows:
        return {"created": 0, "assessments": []}
    now = _now()
    assembled: list[dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            raise InvalidEnumError(
                f"assess_many: each row must be a dict (got {type(r).__name__})"
            )
        missing = [k for k in ("chunk_id", "stance", "weight", "agent_id") if k not in r]
        if missing:
            raise InvalidEnumError(f"assess_many: row missing required field(s) {missing}")
        assembled.append(_assemble_assessment(
            store, study_id=study_id, chunk_id=r["chunk_id"], stance=r["stance"],
            weight=r["weight"], agent_id=r["agent_id"], rationale=r.get("rationale", ""),
            model=r.get("model", ""), provenance=r.get("provenance", PROVENANCE_DEFAULT),
            quote=r.get("quote", ""), char_start=r.get("char_start"),
            char_end=r.get("char_end"), context_chunk_ids=r.get("context_chunk_ids"),
            now=now,
        ))
    for ag in {a["agent_id"] for a in assembled}:
        register_agent(store, agent_id=ag)
    _write_assessments(store, assembled)
    return {"created": len(assembled), "assessments": [a["result"] for a in assembled]}


def supersede_assessment(
    store: Store,
    *,
    old_id: str,
    stance: str,
    weight: float,
    agent_id: str,
    rationale: str = "",
    model: str = "",
    provenance: str = PROVENANCE_DEFAULT,
    context_chunk_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Audit-preserving correction: record a new assessment that explicitly
    *supersedes* an existing one (`(:Assessment)-[:SUPERSEDES]->(:Assessment)`).

    The old assessment is **never deleted** — the correction trail stays
    legible — but `ledger`/tallies hide it by default (current-by-default), so a
    cross-agent correction yields one current row per chunk instead of two
    competing ones (BUG-4). The replacement inherits the old assessment's study
    and chunk; pass the new stance/weight/rationale/provenance."""
    old = _df_dicts(store.cypher(
        "MATCH (o:Assessment {id: $id}) RETURN o.study_id AS study_id, o.chunk_id AS chunk_id",
        params={"id": old_id},
    ))
    if not old or not old[0].get("study_id"):
        raise InvalidEnumError(f"assessment not found: {old_id}")

    res = assess(
        store, study_id=old[0]["study_id"], chunk_id=old[0]["chunk_id"],
        stance=stance, weight=weight, rationale=rationale, agent_id=agent_id,
        model=model, provenance=provenance, context_chunk_ids=context_chunk_ids,
    )
    store.upsert_edges(
        SUPERSEDES, [{"src": res["assessment_id"], "dst": old_id}],
        source_type=ASSESSMENT, target_type=ASSESSMENT,
    )
    res["supersedes"] = old_id
    return res


def verify_assessment(
    store: Store,
    *,
    assessment_id: str,
    verdict: str,
    verifier_agent_id: str,
    notes: str = "",
    provenance: str | None = None,
) -> dict[str, Any]:
    """A second agent verifies an assessment: verified / disputed / duplicate.
    Self-verification is rejected. Mirrors `enrich.verify_summary`.

    `provenance` (optional) records what the *verifier* checked (primary_text /
    characterization / scanned_unread) — stored on the verification event, so a
    'verified' that was itself based on a characterization is still legible."""
    if verdict not in VALID_ASSESSMENT_VERDICTS:
        raise InvalidEnumError(
            f"invalid verdict: {verdict!r} (expected one of {sorted(VALID_ASSESSMENT_VERDICTS)})"
        )
    if provenance is not None and provenance not in VALID_PROVENANCE:
        raise InvalidEnumError(
            f"invalid provenance: {provenance!r} (expected one of {sorted(VALID_PROVENANCE)})"
        )
    author_df = _df_dicts(store.cypher(
        "MATCH (a:Agent)-[:AUTHORED]->(x:Assessment {id: $id}) RETURN a.id AS id",
        params={"id": assessment_id},
    ))
    if not author_df:
        raise InvalidEnumError(f"assessment not found: {assessment_id}")
    if author_df[0]["id"] == verifier_agent_id:
        raise SelfVerificationError(
            f"agent {verifier_agent_id!r} can't verify assessment {assessment_id} — they authored it"
        )

    register_agent(store, agent_id=verifier_agent_id)
    now = _now()
    event_id = str(uuid.uuid4())
    store.upsert_nodes(
        VERIFICATION_EVENT,
        [{
            "id": event_id,
            "title": f"{verdict} by {verifier_agent_id}",
            "assessment_id": assessment_id,
            "verdict": verdict,
            "verifier_agent_id": verifier_agent_id,
            "notes": notes,
            "provenance": provenance or "",
            "created_at": now,
        }],
    )
    store.upsert_edges(
        HAS_VERIFICATION, [{"src": assessment_id, "dst": event_id}],
        source_type=ASSESSMENT, target_type=VERIFICATION_EVENT,
    )
    store.upsert_edges(
        VERIFIED_BY, [{"src": assessment_id, "dst": verifier_agent_id}],
        source_type=ASSESSMENT, target_type=AGENT,
    )
    store.upsert_edges(
        AUTHORED, [{"src": verifier_agent_id, "dst": event_id}],
        source_type=AGENT, target_type=VERIFICATION_EVENT,
    )
    store.swap_label(
        ASSESSMENT, [assessment_id],
        add=label_for("assessment.verification_status", verdict),
        remove_any_of=labels_for("assessment.verification_status"),
    )
    return {
        "assessment_id": assessment_id, "status": verdict,
        "verified_at": now, "verified_by": verifier_agent_id, "event_id": event_id,
        "provenance": provenance or "",
    }


# ─── retrieval ────────────────────────────────────────────────────────────


def ledger(
    store: Store,
    *,
    study_id: str,
    stance: str | None = None,
    min_weight: float | None = None,
    verified_only: bool = False,
    doc_id: str | None = None,
    section_id: str | None = None,
    include_superseded: bool = False,
    limit: int = 200,
) -> dict[str, Any]:
    """Weight-ranked evidence ledger for a study — the orchestrator's one call.

    Dedups to the latest assessment per (chunk, agent), ranks by weight DESC.
    `stance` ("supports"/"against"/"neutral"/"deferred") and `verified_only`
    filter via label predicates; `doc_id`/`section_id` scope to one document or
    section. **Current-by-
    default:** assessments explicitly superseded (FEAT-5) are hidden unless
    `include_superseded=True` (each row carries a `superseded` flag). Returns
    `rows` plus support/against `tallies`, and — for honest coverage — `total`
    (matches before the `limit`) and `returned` (rows handed back); `total >
    returned` means the ledger was truncated.
    """
    meta = _df_dicts(store.cypher(
        "MATCH (s:Study {id: $id}) RETURN s.question AS question, s.status AS status",
        params={"id": study_id},
    ))
    if not meta:
        raise InvalidEnumError(f"study not found: {study_id}")

    label_parts = ""
    if stance:
        if stance not in VALID_STANCES:
            raise InvalidEnumError(f"invalid stance: {stance!r}")
        label_parts += ":" + label_for("study.stance", stance)
    if verified_only:
        label_parts += ":" + label_for("assessment.verification_status", "verified")

    params: dict[str, Any] = {"id": study_id, "lim": int(limit)}
    # doc_id/section_id filter the chunk *before* the latest-per-(chunk,agent)
    # grouping; min_weight filters the surviving `latest` *after* it.
    pre_conds = []
    if doc_id:
        pre_conds.append("c.doc_id = $doc_id")
        params["doc_id"] = doc_id
    if section_id:
        pre_conds.append("c.section_id = $section_id")
        params["section_id"] = section_id
    pre_where = ("WHERE " + " AND ".join(pre_conds)) if pre_conds else ""
    # Assessments an explicit SUPERSEDES edge points at — hidden by default.
    # (EXISTS{} isn't projectable in a WITH on this engine, so resolve the set
    # up front and filter by id; the ledger already reads latest.id/latest.weight
    # off the collected node, so this is safe.)
    superseded_ids = _superseded_ids(store, study_id)

    post_conds = []
    if min_weight is not None:
        post_conds.append("latest.weight >= $minw")
        params["minw"] = float(min_weight)
    if not include_superseded and superseded_ids:
        post_conds.append("NOT latest.id IN $superseded")
        params["superseded"] = list(superseded_ids)
    post_where = ("WHERE " + " AND ".join(post_conds)) if post_conds else ""

    # Shared MATCH + dedup prefix — reused for the row query and the total count
    # so `total` counts exactly the latest-per-(chunk,agent) groups the rows
    # would contain without the LIMIT.
    base = (
        f"MATCH (c:Chunk)-[:ASSESSED_AS]->(a:Assessment{label_parts})-[:OF_STUDY]->(:Study {{id: $id}}) "
        f"{pre_where} "
        "WITH c, a.by_agent AS ag, a ORDER BY a.created_at DESC "
        "WITH c, ag, collect(a)[0] AS latest "
        f"{post_where} "
    )

    total_rows = _df_dicts(store.cypher(base + "RETURN count(latest) AS n", params=params))
    total = int(total_rows[0]["n"]) if total_rows else 0

    df = store.cypher(
        base + "RETURN latest.id AS assessment_id, c.id AS chunk_id, c.doc_id AS doc_id, "
        "c.page_number AS page, latest.stance AS stance, latest.weight AS weight, "
        "latest.rationale AS rationale, latest.by_agent AS by_agent, "
        "latest.quote AS quote, latest.char_start AS char_start, "
        "latest.char_end AS char_end, "
        "labels(latest) AS labels, c.text AS text "
        "ORDER BY weight DESC LIMIT $lim",
        params=params,
    )
    rows = _df_dicts(df)
    for r in rows:
        node_labels = r.pop("labels", []) or []
        r["verification_status"] = _verification_from_labels(node_labels)
        r["provenance"] = _provenance_from_labels(node_labels)
        r["superseded"] = r["assessment_id"] in superseded_ids
        # Normalize pinpoint span for pre-FEAT-6 rows (null → unset).
        r["quote"] = r.get("quote") or ""
        r["char_start"] = cs if isinstance(cs := r.get("char_start"), int) else -1
        r["char_end"] = ce if isinstance(ce := r.get("char_end"), int) else -1
    # Attach each row's USED_CONTEXT span (the neighbor chunks the agent read
    # to judge it) so retrieval can pull the full relevant span.
    ids = [r["assessment_id"] for r in rows if r.get("assessment_id")]
    if ids:
        ctx_rows = _df_dicts(store.cypher(
            "MATCH (a:Assessment)-[:USED_CONTEXT]->(ctx:Chunk) WHERE a.id IN $ids "
            "RETURN a.id AS aid, collect(ctx.id) AS ctx_ids",
            params={"ids": ids},
        ))
        ctx_map = {r["aid"]: r["ctx_ids"] for r in ctx_rows}
        for r in rows:
            r["context_chunk_ids"] = ctx_map.get(r["assessment_id"], [])
    return {
        "study_id": study_id,
        "question": meta[0]["question"],
        "status": meta[0]["status"],
        "rows": rows,
        "total": total,
        "returned": len(rows),
        "tallies": _tallies(store, study_id),
    }


def conflicts(store: Store, *, study_id: str) -> dict[str, Any]:
    """Chunks with *both* a current `supports` and a current `against`
    assessment — the contested evidence an orchestrator should review first.

    Computed over the current set (latest-per-(chunk, agent), excluding
    superseded), so a correction that resolves a disagreement removes it from
    the list. Each conflict carries its opposing rows split by side.
    """
    meta = _df_dicts(store.cypher(
        "MATCH (s:Study {id: $id}) RETURN s.question AS question",
        params={"id": study_id},
    ))
    if not meta:
        raise InvalidEnumError(f"study not found: {study_id}")

    superseded_ids = _superseded_ids(store, study_id)
    params: dict[str, Any] = {"id": study_id}
    sup_filter = ""
    if superseded_ids:
        sup_filter = "WHERE NOT latest.id IN $superseded "
        params["superseded"] = list(superseded_ids)
    rows = _df_dicts(store.cypher(
        "MATCH (c:Chunk)-[:ASSESSED_AS]->(a:Assessment)-[:OF_STUDY]->(:Study {id: $id}) "
        "WITH c, a.by_agent AS ag, a ORDER BY a.created_at DESC "
        "WITH c, ag, collect(a)[0] AS latest "
        f"{sup_filter}"
        "RETURN latest.id AS assessment_id, c.id AS chunk_id, c.doc_id AS doc_id, "
        "c.page_number AS page, latest.stance AS stance, latest.weight AS weight, "
        "latest.provenance AS provenance, latest.rationale AS rationale, "
        "latest.by_agent AS by_agent, c.text AS text "
        "ORDER BY c.doc_id, c.page_number",
        params=params,
    ))

    by_chunk: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for r in rows:
        cid = r["chunk_id"]
        if cid not in by_chunk:
            by_chunk[cid] = []
            order.append(cid)
        by_chunk[cid].append(r)

    out: list[dict[str, Any]] = []
    for cid in order:
        rs = by_chunk[cid]
        supports = [r for r in rs if r["stance"] == STANCE_SUPPORTS]
        against = [r for r in rs if r["stance"] == STANCE_AGAINST]
        if supports and against:
            first = rs[0]
            out.append({
                "chunk_id": cid,
                "doc_id": first["doc_id"],
                "page": first["page"],
                "text": first["text"],
                "supports": supports,
                "against": against,
            })
    # Most-contested first (total opposing rows).
    out.sort(key=lambda x: len(x["supports"]) + len(x["against"]), reverse=True)
    return {
        "study_id": study_id,
        "question": meta[0]["question"],
        "conflicts": out,
        "total": len(out),
    }


def next_unassessed(
    store: Store,
    *,
    study_id: str,
    doc_id: str | None = None,
    section_id: str | None = None,
    agent_id: str | None = None,
    limit: int = 20,
    ttl_seconds: int = CLAIM_TTL_SECONDS,
) -> list[dict[str, Any]]:
    """The work-list of chunks not yet assessed for this study, in reading
    order.

    **Punchcard semantics.** When `agent_id` is given, this atomically
    *claims* the returned chunks for that agent (a "checkout"), excluding any
    chunks already claimed by someone else — so parallel analysts never
    overlap. Without `agent_id` it's a read-only preview (no claim).

    Claims auto-expire after `ttl_seconds` (default 30 min) so an analyst that
    pulls but never assesses doesn't lock chunks forever; assessing a chunk
    excludes it regardless of claim (implicit release). Stale checkouts are
    garbage-collected on the next claim.
    """
    if not _study_exists(store, study_id):
        raise InvalidEnumError(f"study not found: {study_id}")

    chunk_where = ["c.status = 'ready'"]
    base_params: dict[str, Any] = {"sid": study_id, "lim": int(limit)}
    if doc_id:
        chunk_where.append("c.doc_id = $doc")
        base_params["doc"] = doc_id
    if section_id:
        chunk_where.append("c.section_id = $sec")
        base_params["sec"] = section_id
    where_sql = " AND ".join(chunk_where)
    # "Done for this study" = given a *real* (non-deferred) stance as a focal
    # chunk, OR already read as context for another chunk's assessment (so we
    # never re-judge it). A `deferred` assessment means "read but unjudgeable
    # yet" — it parks the chunk, keeping it in the work-list to revisit once the
    # blocker clears, instead of silently dropping blocked evidence.
    not_done = (
        f"NOT EXISTS {{ MATCH (c)-[:ASSESSED_AS]->(a:Assessment)-[:OF_STUDY]->(:Study {{id: $sid}}) "
        f"WHERE a.stance <> '{STANCE_DEFERRED}' }} "
        "AND NOT EXISTS { MATCH (c)<-[:USED_CONTEXT]-(:Assessment)-[:OF_STUDY]->(:Study {id: $sid}) }"
    )

    # Claim (agent_id) or preview, via the shared punchcard — keyed on the study
    # id so study claims stay disjoint from the classification work-list.
    return claim_or_preview(
        store, where_sql=where_sql, not_done=not_done, base_params=base_params,
        checkout_key=study_id, agent_id=agent_id, ttl_seconds=ttl_seconds,
    )


# ─── internals ────────────────────────────────────────────────────────────


def _study_exists(store: Store, study_id: str) -> bool:
    df = _df_dicts(store.cypher(
        "MATCH (s:Study {id: $id}) RETURN s.id AS id", params={"id": study_id},
    ))
    return bool(df)


def _chunk_text(store: Store, chunk_id: str) -> str | None:
    rows = _df_dicts(store.cypher(
        f"MATCH (c:Chunk {{id: $id}}) RETURN c.{CHUNK_TEXT_COL} AS text",
        params={"id": chunk_id},
    ))
    return rows[0]["text"] if rows else None


def _resolve_span(
    store: Store, chunk_id: str, quote: str,
    char_start: int | None, char_end: int | None,
) -> tuple[str, int, int]:
    """Validate/locate a pinpoint span (FEAT-6) against the chunk text. Returns
    `(quote, char_start, char_end)` with `-1` for unset offsets. Honest cites:
    an out-of-range span or a quote not found in the chunk is rejected."""
    if not quote and char_start is None and char_end is None:
        return "", -1, -1
    text = _chunk_text(store, chunk_id)
    if text is None:
        raise InvalidEnumError(f"chunk not found: {chunk_id}")
    n = len(text)
    if char_start is not None or char_end is not None:
        if char_start is None or char_end is None:
            raise InvalidEnumError("char_start and char_end must be given together")
        if not (0 <= char_start <= char_end <= n):
            raise InvalidEnumError(
                f"span [{char_start}, {char_end}] out of range for chunk of length {n}"
            )
        if quote and text[char_start:char_end] != quote:
            raise InvalidEnumError("quote does not match the text at [char_start, char_end]")
        return (quote or text[char_start:char_end]), char_start, char_end
    idx = text.find(quote)
    if idx < 0:
        raise InvalidEnumError("quote not found in the chunk text")
    return quote, idx, idx + len(quote)


def _superseded_ids(store: Store, study_id: str) -> set[str]:
    """Ids of this study's assessments that an explicit SUPERSEDES edge points
    at (i.e. have been corrected) — hidden from the current ledger/tallies."""
    return {
        r["id"] for r in _df_dicts(store.cypher(
            "MATCH (:Assessment)-[:SUPERSEDES]->(o:Assessment)-[:OF_STUDY]->(:Study {id: $id}) "
            "RETURN DISTINCT o.id AS id",
            params={"id": study_id},
        ))
    }


def _verification_from_labels(node_labels: list[str]) -> str:
    for lbl in node_labels:
        if lbl != "Unverified" and lbl in _ASSESS_LABEL_SET and lbl in _VSTATUS_BY_LABEL:
            return _VSTATUS_BY_LABEL[lbl]
    return "unverified"


def _provenance_from_labels(node_labels: list[str]) -> str:
    """Provenance string from an Assessment's labels. Defaults to the historic
    assumption (`primary_text`) for assessments written before FEAT-4."""
    for lbl in node_labels:
        if lbl in _PROVENANCE_LABEL_SET and lbl in _PROVENANCE_BY_LABEL:
            return _PROVENANCE_BY_LABEL[lbl]
    return PROVENANCE_DEFAULT


def _tallies(store: Store, study_id: str) -> dict[str, Any]:
    """Counts + summed weight per stance, deduped to latest per (chunk, agent).
    Reflects *current* truth — superseded assessments (FEAT-5) are excluded so
    the tallies match the default ledger."""
    superseded_ids = _superseded_ids(store, study_id)
    params: dict[str, Any] = {"id": study_id}
    sup_filter = ""
    if superseded_ids:
        sup_filter = "WHERE NOT latest.id IN $superseded "
        params["superseded"] = list(superseded_ids)
    df = store.cypher(
        "MATCH (c:Chunk)-[:ASSESSED_AS]->(a:Assessment)-[:OF_STUDY]->(:Study {id: $id}) "
        "WITH c, a.by_agent AS ag, a ORDER BY a.created_at DESC "
        "WITH c, ag, collect(a)[0] AS latest "
        f"{sup_filter}"
        "WITH latest.stance AS stance, latest.weight AS wt "
        "RETURN stance, count(*) AS n, sum(wt) AS w",
        params=params,
    )
    out: dict[str, Any] = {
        "supports": 0, "against": 0, "neutral": 0, "deferred": 0,
        "supports_weight": 0.0, "against_weight": 0.0,
        "neutral_weight": 0.0, "deferred_weight": 0.0,
    }
    for r in _df_dicts(df):
        st = r.get("stance")
        if st in VALID_STANCES:
            out[st] = int(r.get("n", 0))
            out[f"{st}_weight"] = round(float(r.get("w", 0.0) or 0.0), 4)
    return out
