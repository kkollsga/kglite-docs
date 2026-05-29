"""Kanban-style review queue.

Tickets target an existing node (Chunk / Summary / Document / Page).
Their state is **event-sourced**: every state transition writes an
immutable `ReviewEvent` node, and the current status is the verdict of
the most-recent event. This matches the verification model and dodges
kglite 0.10.3's in-place String SET panic.

Lifecycle::

       enqueue          claim            complete
    ── new ──→ in_review ──→ reviewed
                  │
                  └ unclaim ──→ new          (release without verdict)
                  └ complete(needs_revision) ─→ needs_revision
                  └ complete(rejected)       ─→ rejected

A single ticket can be re-claimed and re-completed any number of times
— each transition is just another `ReviewEvent`.

Atomicity caveat: claims are guarded by a process-local `threading.Lock`,
so within one Python process two callers can't race to claim the same
ticket. Across processes (rare for our use case — `.kgl` is
single-writer) you'd want kglite-level row locking, which isn't
exposed today.
"""

from __future__ import annotations

import contextlib
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from kglite_docs.activity import register_agent
from kglite_docs.errors import InvalidEnumError, ReviewConflict
from kglite_docs.schema import (
    AGENT,
    AUTHORED,
    CLAIMED,
    HAS_REVIEW_EVENT,
    REVIEW_EVENT,
    REVIEW_IN_REVIEW,
    REVIEW_NEEDS_REVISION,
    REVIEW_NEW,
    REVIEW_REJECTED,
    REVIEW_REVIEWED,
    REVIEW_TICKET,
    REVIEWED,
    TARGETS,
    label_for,
    labels_for,
)
from kglite_docs.store import Store
from kglite_docs.store import rows as _rows
from kglite_docs.tagging import tag_chunk

_claim_lock = threading.Lock()

_VALID_COMPLETION_STATUSES = {
    REVIEW_REVIEWED, REVIEW_NEEDS_REVISION, REVIEW_REJECTED,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── enqueue ───────────────────────────────────────────────────────────────


def enqueue(
    store: Store,
    *,
    target_id: str,
    target_kind: str = "Chunk",
    priority: int = 0,
    note: str = "",
    enqueued_by: str = "system",
) -> str:
    """Create a `new` ticket for a target node. Returns the ticket id.

    Multiple tickets can exist on the same target (e.g. for re-review).
    Callers who want at-most-one-open-ticket semantics should check
    first with `current_ticket_for(target_id)`.
    """
    ticket_id = str(uuid.uuid4())
    now = _now()
    store.upsert_nodes(REVIEW_TICKET, [{
        "id": ticket_id,
        "title": f"review {target_kind} {target_id[:18]}…",
        "target_id": target_id,
        "target_kind": target_kind,
        "priority": int(priority),
        "created_at": now,
        "created_by": enqueued_by,
        "note": note,
    }])
    store.upsert_edges(
        TARGETS, [{"src": ticket_id, "dst": target_id}],
        source_type=REVIEW_TICKET, target_type=target_kind,
    )
    _record_event(store, ticket_id=ticket_id, event_type=REVIEW_NEW,
                  agent_id=enqueued_by, notes=note)
    # Initial status label
    store.add_label(REVIEW_TICKET, [ticket_id], label_for("review.status", REVIEW_NEW))
    return ticket_id


def enqueue_chunks(
    store: Store, *, doc_id: str | None = None, status_filter: str | None = None,
    priority: int = 0, enqueued_by: str = "system",
) -> dict[str, Any]:
    """Bulk-enqueue chunks for review. Skips chunks that already have an
    open (not-yet-reviewed) ticket. Returns counts.

    Pass `doc_id` to scope to one document, `status_filter` to scope to
    chunks with a specific `Chunk.status` (e.g. 'ready' to skip placeholders).
    """
    where = []
    params: dict[str, Any] = {}
    if doc_id:
        where.append("c.doc_id = $doc_id")
        params["doc_id"] = doc_id
    # Chunk status is now a label — fold into MATCH instead of WHERE
    chunk_label = label_for("chunk.status", status_filter) if status_filter else ""
    label_clause = f":{chunk_label}" if chunk_label else ""
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    chunks = _rows(store.cypher(
        f"MATCH (c:Chunk{label_clause}) {clause} RETURN c.id AS id",
        params=params,
    ))
    enqueued = 0
    skipped = 0
    for c in chunks:
        if current_ticket_for(store, target_id=c["id"]) is None:
            enqueue(store, target_id=c["id"], target_kind="Chunk",
                    priority=priority, enqueued_by=enqueued_by)
            enqueued += 1
        else:
            skipped += 1
    return {"enqueued": enqueued, "skipped": skipped, "total_candidates": len(chunks)}


# ─── claim ─────────────────────────────────────────────────────────────────


def claim(
    store: Store, *, ticket_id: str, agent_id: str
) -> dict[str, Any]:
    """Atomically transition a ticket from `new` → `in_review` by `agent_id`.
    Raises `ReviewConflict` if the ticket isn't currently `new`."""
    register_agent(store, agent_id=agent_id)
    with _claim_lock:
        status, claimer = _current_state(store, ticket_id)
        if status != REVIEW_NEW:
            raise ReviewConflict(
                f"ticket {ticket_id} is in state {status!r} (claimed by {claimer!r}), not new"
            )
        eid = _record_event(
            store, ticket_id=ticket_id, event_type=REVIEW_IN_REVIEW,
            agent_id=agent_id, notes="",
        )
        store.upsert_edges(
            CLAIMED, [{"src": agent_id, "dst": ticket_id}],
            source_type=AGENT, target_type=REVIEW_TICKET,
        )
        # New → InReview label swap
        store.swap_label(
            REVIEW_TICKET, [ticket_id],
            add=label_for("review.status", REVIEW_IN_REVIEW),
            remove_any_of=labels_for("review.status"),
        )
    return {"ticket_id": ticket_id, "status": REVIEW_IN_REVIEW,
            "claimed_by": agent_id, "event_id": eid}


def claim_next(
    store: Store, *, agent_id: str, target_kind: str | None = None,
    min_priority: int | None = None,
) -> dict[str, Any] | None:
    """Atomic 'find highest-priority new ticket and claim it'. Returns
    `None` when the queue is empty."""
    with _claim_lock:
        params: dict[str, Any] = {}
        where = []
        if target_kind:
            where.append("t.target_kind = $tk")
            params["tk"] = target_kind
        if min_priority is not None:
            where.append("t.priority >= $mp")
            params["mp"] = int(min_priority)
        clause = ("AND " + " AND ".join(where)) if where else ""
        # Pull candidate tickets ordered by priority desc, created_at asc
        candidates = _rows(store.cypher(
            f"""
            MATCH (t:ReviewTicket)
            WHERE EXISTS {{ MATCH (t)-[:HAS_REVIEW_EVENT]->(e:ReviewEvent) }}
            {clause}
            RETURN t.id AS id, t.target_id AS target_id, t.target_kind AS target_kind,
                   t.priority AS priority, t.created_at AS created_at
            ORDER BY t.priority DESC, t.created_at ASC
            LIMIT 100
            """,
            params=params,
        ))
        for cand in candidates:
            status, _ = _current_state(store, cand["id"])
            if status == REVIEW_NEW:
                # claim it inside the same lock — but reuse claim() logic via
                # the internal helper (we already hold the lock).
                register_agent(store, agent_id=agent_id)
                eid = _record_event(
                    store, ticket_id=cand["id"], event_type=REVIEW_IN_REVIEW,
                    agent_id=agent_id, notes="",
                )
                store.upsert_edges(
                    CLAIMED, [{"src": agent_id, "dst": cand["id"]}],
                    source_type=AGENT, target_type=REVIEW_TICKET,
                )
                # New → InReview label swap
                store.swap_label(
                    REVIEW_TICKET, [cand["id"]],
                    add=label_for("review.status", REVIEW_IN_REVIEW),
                    remove_any_of=labels_for("review.status"),
                )
                # Hydrate the target so the caller has everything they need
                target = _hydrate_target(store, cand["target_id"], cand["target_kind"])
                return {
                    "ticket_id": cand["id"], "status": REVIEW_IN_REVIEW,
                    "claimed_by": agent_id, "event_id": eid,
                    "target_id": cand["target_id"],
                    "target_kind": cand["target_kind"],
                    "priority": cand.get("priority"),
                    "target": target,
                }
    return None


# ─── unclaim ───────────────────────────────────────────────────────────────


def unclaim(
    store: Store, *, ticket_id: str, agent_id: str, reason: str = ""
) -> dict[str, Any]:
    """Release a claim without a verdict — ticket returns to `new` and is
    re-claimable. Only the current claimer can unclaim."""
    status, claimer = _current_state(store, ticket_id)
    if status != REVIEW_IN_REVIEW:
        raise ReviewConflict(f"ticket {ticket_id} not in_review (currently {status!r})")
    if claimer != agent_id:
        raise ReviewConflict(
            f"agent {agent_id!r} can't unclaim ticket held by {claimer!r}"
        )
    _record_event(store, ticket_id=ticket_id, event_type=REVIEW_NEW,
                  agent_id=agent_id, notes=reason)
    # InReview → New label swap (release without verdict)
    store.swap_label(
        REVIEW_TICKET, [ticket_id],
        add=label_for("review.status", REVIEW_NEW),
        remove_any_of=labels_for("review.status"),
    )
    return {"ticket_id": ticket_id, "status": REVIEW_NEW, "released_by": agent_id}


# ─── complete ──────────────────────────────────────────────────────────────


def complete(
    store: Store, *, ticket_id: str, agent_id: str,
    verdict: str = REVIEW_REVIEWED,
    accuracy: float | None = None,
    authenticity: str | None = None,
    notes: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Mark a ticket as completed with a verdict.

    - `verdict`: one of `reviewed` (default), `needs_revision`, `rejected`.
    - `accuracy`: optional [0.0, 1.0] score the agent assigns.
    - `authenticity`: optional free-text or enum (`verified`/`disputed`).
    - `notes`: free-text rationale.
    - `tags`: tag names to apply to the *target* chunk (only when target is Chunk).
    """
    if verdict not in _VALID_COMPLETION_STATUSES:
        raise InvalidEnumError(
            f"invalid verdict {verdict!r}; expected one of {sorted(_VALID_COMPLETION_STATUSES)}"
        )
    status, claimer = _current_state(store, ticket_id)
    if status != REVIEW_IN_REVIEW:
        raise ReviewConflict(
            f"ticket {ticket_id} not in_review (currently {status!r}); "
            "claim it first"
        )
    if claimer != agent_id:
        raise ReviewConflict(
            f"agent {agent_id!r} can't complete ticket held by {claimer!r}"
        )
    eid = _record_event(
        store, ticket_id=ticket_id, event_type=verdict,
        agent_id=agent_id, notes=notes,
        accuracy=accuracy, authenticity=authenticity or "",
    )
    store.upsert_edges(
        REVIEWED, [{"src": agent_id, "dst": ticket_id}],
        source_type=AGENT, target_type=REVIEW_TICKET,
    )
    # InReview → <verdict> label swap
    store.swap_label(
        REVIEW_TICKET, [ticket_id],
        add=label_for("review.status", verdict),
        remove_any_of=labels_for("review.status"),
    )
    # Apply any requested tags to the target (only chunks for now)
    target_id, target_kind = _ticket_target(store, ticket_id)
    if target_kind == "Chunk" and tags:
        for t in tags:
            with contextlib.suppress(Exception):
                tag_chunk(store, chunk_id=target_id, tag_name=t,
                          kind="review", agent_id=agent_id)
    return {
        "ticket_id": ticket_id, "status": verdict, "completed_by": agent_id,
        "event_id": eid, "accuracy": accuracy, "authenticity": authenticity,
    }


# ─── queries ───────────────────────────────────────────────────────────────


def list_queue(
    store: Store,
    *,
    status: str | None = None,
    target_kind: str | None = None,
    agent_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List tickets with their current event-sourced status.

    Filters:
    - `status`: only tickets in this state (new / in_review / reviewed / ...).
    - `target_kind`: Chunk / Summary / Document / Page.
    - `agent_id`: only tickets currently claimed by this agent (implies
      status=in_review unless `status` is also set).
    """
    params: dict[str, Any] = {}
    where = []
    if target_kind:
        where.append("t.target_kind = $tk")
        params["tk"] = target_kind
    # Status filter via label predicate inline.
    status_label = label_for("review.status", status) if status else ""
    label_clause = f":{status_label}" if status_label else ""
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    rows = _rows(store.cypher(
        f"""
        MATCH (t:ReviewTicket{label_clause})
        {clause}
        RETURN t.id AS id, t.target_id AS target_id, t.target_kind AS target_kind,
               t.priority AS priority, t.created_at AS created_at,
               t.created_by AS created_by, t.note AS note
        ORDER BY t.priority DESC, t.created_at ASC
        """,
        params=params,
    ))
    out = []
    for r in rows:
        # We still call _current_state to get the claimer (the label only
        # tells us "in_review", not who holds it). For agent_id filtering
        # we drop tickets not held by that agent.
        st, claimer = _current_state(store, r["id"])
        if agent_id and claimer != agent_id:
            continue
        r["status"] = st
        r["claimed_by"] = claimer
        out.append(r)
        if len(out) >= limit:
            break
    return out


def get_ticket(
    store: Store, *, ticket_id: str, with_target: bool = True,
    with_events: bool = True,
) -> dict[str, Any] | None:
    rows = _rows(store.cypher(
        "MATCH (t:ReviewTicket {id: $id}) "
        "RETURN t.id AS id, t.target_id AS target_id, t.target_kind AS target_kind, "
        "t.priority AS priority, t.created_at AS created_at, t.note AS note",
        params={"id": ticket_id},
    ))
    if not rows:
        return None
    ticket = rows[0]
    ticket["status"], ticket["claimed_by"] = _current_state(store, ticket_id)
    if with_target:
        ticket["target"] = _hydrate_target(store, ticket["target_id"], ticket["target_kind"])
    if with_events:
        ticket["events"] = _rows(store.cypher(
            "MATCH (t:ReviewTicket {id: $id})-[:HAS_REVIEW_EVENT]->(e:ReviewEvent) "
            "OPTIONAL MATCH (a:Agent)-[:AUTHORED]->(e) "
            "RETURN e.event_type AS type, e.notes AS notes, "
            "e.accuracy AS accuracy, e.authenticity AS authenticity, "
            "e.created_at AS at, a.id AS agent "
            "ORDER BY e.created_at ASC",
            params={"id": ticket_id},
        ))
    return ticket


def current_ticket_for(
    store: Store, *, target_id: str
) -> dict[str, Any] | None:
    """The most-recent ticket targeting `target_id`, or None. Useful before
    enqueueing to avoid duplicates."""
    rows = _rows(store.cypher(
        "MATCH (t:ReviewTicket {target_id: $tid}) "
        "RETURN t.id AS id, t.created_at AS at "
        "ORDER BY t.created_at DESC LIMIT 1",
        params={"tid": target_id},
    ))
    if not rows:
        return None
    return get_ticket(store, ticket_id=rows[0]["id"], with_target=False, with_events=False)


def stats(store: Store) -> dict[str, Any]:
    """Counts by current status + per-agent in-review counts.
    Drives the kanban board summary."""
    by_status: dict[str, int] = {}
    by_agent_in_review: dict[str, int] = {}
    tickets = _rows(store.cypher(
        "MATCH (t:ReviewTicket) RETURN t.id AS id"
    ))
    for t in tickets:
        st, claimer = _current_state(store, t["id"])
        by_status[st] = by_status.get(st, 0) + 1
        if st == REVIEW_IN_REVIEW and claimer:
            by_agent_in_review[claimer] = by_agent_in_review.get(claimer, 0) + 1
    return {
        "tickets_total": len(tickets),
        "by_status": by_status,
        "in_review_by_agent": by_agent_in_review,
    }


# ─── internals ─────────────────────────────────────────────────────────────


def _record_event(
    store: Store,
    *,
    ticket_id: str,
    event_type: str,
    agent_id: str,
    notes: str = "",
    accuracy: float | None = None,
    authenticity: str = "",
) -> str:
    event_id = str(uuid.uuid4())
    row: dict[str, Any] = {
        "id": event_id,
        "title": f"{event_type} by {agent_id}",
        "ticket_id": ticket_id,
        "event_type": event_type,
        "agent_id": agent_id,
        "notes": notes,
        "authenticity": authenticity,
        "created_at": _now(),
    }
    if accuracy is not None:
        row["accuracy"] = float(accuracy)
    store.upsert_nodes(REVIEW_EVENT, [row])
    store.upsert_edges(
        HAS_REVIEW_EVENT, [{"src": ticket_id, "dst": event_id}],
        source_type=REVIEW_TICKET, target_type=REVIEW_EVENT,
    )
    if agent_id and agent_id != "system":
        store.upsert_edges(
            AUTHORED, [{"src": agent_id, "dst": event_id}],
            source_type=AGENT, target_type=REVIEW_EVENT,
        )
    return event_id


def _current_state(store: Store, ticket_id: str) -> tuple[str, str | None]:
    """Return `(status, current_claimer)` from the latest event."""
    rows = _rows(store.cypher(
        "MATCH (t:ReviewTicket {id: $id})-[:HAS_REVIEW_EVENT]->(e:ReviewEvent) "
        "RETURN e.event_type AS type, e.agent_id AS agent, e.created_at AS at "
        "ORDER BY e.created_at DESC LIMIT 1",
        params={"id": ticket_id},
    ))
    if not rows:
        return (REVIEW_NEW, None)
    row = rows[0]
    et = row["type"]
    if et == REVIEW_IN_REVIEW:
        return (REVIEW_IN_REVIEW, row.get("agent"))
    return (et, None)


def _ticket_target(store: Store, ticket_id: str) -> tuple[str, str]:
    rows = _rows(store.cypher(
        "MATCH (t:ReviewTicket {id: $id}) "
        "RETURN t.target_id AS target_id, t.target_kind AS target_kind",
        params={"id": ticket_id},
    ))
    if not rows:
        raise ValueError(f"ticket not found: {ticket_id}")
    return rows[0]["target_id"], rows[0]["target_kind"]


def _hydrate_target(store: Store, target_id: str, target_kind: str) -> dict[str, Any]:
    if target_kind == "Chunk":
        rows = _rows(store.cypher(
            "MATCH (c:Chunk {id: $id}) "
            "RETURN c.id AS id, c.doc_id AS doc_id, c.page_number AS page, "
            "c.text AS text, c.headings_json AS headings, c.status AS status",
            params={"id": target_id},
        ))
    elif target_kind == "Summary":
        rows = _rows(store.cypher(
            "MATCH (s:Summary {id: $id}) "
            "RETURN s.id AS id, s.target_id AS target_id, s.text AS text, s.depth AS depth",
            params={"id": target_id},
        ))
    elif target_kind == "Document":
        rows = _rows(store.cypher(
            "MATCH (d:Document {id: $id}) "
            "RETURN d.id AS id, d.title AS title, d.page_count AS pages, d.format AS format",
            params={"id": target_id},
        ))
    elif target_kind == "Page":
        rows = _rows(store.cypher(
            "MATCH (p:Page {id: $id}) "
            "RETURN p.id AS id, p.doc_id AS doc_id, p.page_number AS page_number",
            params={"id": target_id},
        ))
    else:
        return {"id": target_id, "kind": target_kind}
    return rows[0] if rows else {"id": target_id, "kind": target_kind}
