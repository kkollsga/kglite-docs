"""Timeline / Event layer — analyze the ordered record, not just chunks.

A whole class of legal defect is *sequential*: a judge defaults one party for a
justified absence but extinguishes the case for another's unjustified absence,
then condemns on the merits — three routine-looking events whose impropriety
appears only when you align them on `party × trigger × outcome × date`. Per-chunk
scoring can't see it; a timeline can.

kglite-docs is agent-first, so the library provides the `Event` node, an
extraction prompt, and deterministic sequence analyzers — the agent supplies the
events (no heavy in-process extraction model). An `Event` is generic
(`date / actor / action / outcome`); the legal action/outcome vocabulary stays in
the vertical. `timeline_conflicts` completes the disparate-treatment detector
that `semantic_conflicts` could only approximate without subjects.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from kglite_docs.activity import register_agent
from kglite_docs.errors import InvalidEnumError
from kglite_docs.schema import CHUNK, DOCUMENT, EVENT, HAS_EVENT
from kglite_docs.store import Store
from kglite_docs.store import rows as _df_dicts

EVENT_EXTRACTION_PROMPT = """\
TIMELINE EXTRACTION — for each dated occurrence in the document, record an event
with study("add_event", …): `date` (ISO yyyy-mm-dd if possible, else as written),
`actor` (who acted — court, a party, …), `action` (what happened — the trigger:
non_appearance, ruling, filing, …), `outcome` (the consequence — default,
dismissal, condemnation, …), and the `chunk_id` it came from. Normalize `action`
and `outcome` to short stable tokens so like events line up (same `action`
across parties is how disparate treatment surfaces)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_event(
    store: Store,
    *,
    doc_id: str,
    date: str,
    actor: str,
    action: str,
    outcome: str,
    chunk_id: str = "",
    ruling_type: str = "",
    agent_id: str = "",
) -> dict[str, Any]:
    """Record one timeline event on a document (optionally anchored to the chunk
    it came from). Generic shape — the action/outcome vocabulary is the caller's."""
    doc = _df_dicts(store.cypher(
        "MATCH (d:Document {id: $id}) RETURN d.id AS id", params={"id": doc_id},
    ))
    if not doc:
        raise InvalidEnumError(f"document not found: {doc_id}")
    if agent_id:
        register_agent(store, agent_id=agent_id)
    eid = "event_" + uuid.uuid4().hex[:16]
    store.upsert_nodes(EVENT, [{
        "id": eid, "title": f"{date} {actor} {action}→{outcome}"[:120],
        "doc_id": doc_id, "chunk_id": chunk_id, "date": date, "actor": actor,
        "action": action, "outcome": outcome, "ruling_type": ruling_type,
        "by_agent": agent_id, "created_at": _now(),
    }])
    store.upsert_edges(HAS_EVENT, [{"src": doc_id, "dst": eid}], source_type=DOCUMENT, target_type=EVENT)
    if chunk_id:
        store.upsert_edges(HAS_EVENT, [{"src": chunk_id, "dst": eid}], source_type=CHUNK, target_type=EVENT)
    return {"event_id": eid, "doc_id": doc_id, "date": date, "actor": actor,
            "action": action, "outcome": outcome}


def timeline(store: Store, *, doc_id: str) -> list[dict[str, Any]]:
    """A document's events in chronological order (by `date`, then insertion)."""
    return _df_dicts(store.cypher(
        "MATCH (:Document {id: $id})-[:HAS_EVENT]->(e:Event) "
        "RETURN e.id AS event_id, e.date AS date, e.actor AS actor, e.action AS action, "
        "e.outcome AS outcome, e.ruling_type AS ruling_type, e.chunk_id AS chunk_id "
        "ORDER BY e.date, e.created_at",
        params={"id": doc_id},
    ))


def timeline_conflicts(store: Store, *, doc_id: str) -> dict[str, Any]:
    """Sequence analysis over the events (deterministic, no model):

    - **disparate_treatment** — the same `action` (trigger) resolved to *different
      outcomes for different actors* (the like-cases-treated-alike test);
    - **contradictory_outcomes** — the same `(actor, action)` carrying conflicting
      outcomes across events (two operative outcomes that can't both stand).

    Honest coverage: reports `events` scanned, so a `0` reads as "looked, none"
    only when there were events to look at."""
    events = timeline(store, doc_id=doc_id)
    by_action: dict[str, list[dict[str, Any]]] = {}
    by_actor_action: dict[tuple[str, str], set[str]] = {}
    for e in events:
        action = (e.get("action") or "").strip()
        actor = (e.get("actor") or "").strip()
        outcome = (e.get("outcome") or "").strip()
        if action:
            by_action.setdefault(action, []).append(e)
        if action and actor:
            by_actor_action.setdefault((actor, action), set()).add(outcome)

    disparate: list[dict[str, Any]] = []
    for action, evs in by_action.items():
        by_actor: dict[str, set[str]] = {}
        for e in evs:
            by_actor.setdefault((e.get("actor") or "").strip(), set()).add((e.get("outcome") or "").strip())
        # ≥2 distinct actors, and not all outcomes identical across them
        outcomes_per_actor = {a: o for a, o in by_actor.items() if a}
        all_outcomes = {frozenset(o) for o in outcomes_per_actor.values()}
        if len(outcomes_per_actor) >= 2 and len(all_outcomes) >= 2:
            disparate.append({
                "action": action,
                "actors": [{"actor": a, "outcomes": sorted(o)} for a, o in outcomes_per_actor.items()],
                "events": [e["event_id"] for e in evs],
            })

    contradictory: list[dict[str, Any]] = []
    for (actor, action), outcomes in by_actor_action.items():
        real = {o for o in outcomes if o}
        if len(real) >= 2:
            contradictory.append({"actor": actor, "action": action, "outcomes": sorted(real)})

    return {
        "doc_id": doc_id,
        "events": len(events),
        "disparate_treatment": disparate,
        "contradictory_outcomes": contradictory,
        "total": len(disparate) + len(contradictory),
    }
