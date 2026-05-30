"""Domain element classification — tag each chunk *once* into a registered
controlled vocabulary so many studies route to their relevant chunks instead of
re-scanning the corpus. Stateless functions over `store` (mirrors `tagging.py`).

Storage is labels-first, property-canonical:
- element types are multi-valued secondary chunk labels (`:Holding:Statute`) —
  the hot routing path; **add-only** (recall-safe: a chunk is never silently
  dropped from a route).
- `element_types_json` is the canonical per-agent record (`[{type, conf, by, at}]`).
- every classified chunk gets `:Classified` (xor `:Unclassified` when an agent
  finds no element applies) — exhaustive, so `next_unclassified` never leaks.
- two agents disagreeing on a chunk's element set add `:Contested` (never a
  silent clobber).

Core stays domain-opaque: `elements` are validated against the registered
allow-list (`schema.valid_element_values`); an unknown element raises rather than
minting a stray label.
"""

from __future__ import annotations

import json
from typing import Any

from kglite_docs.activity import register_agent
from kglite_docs.checkout import CLASSIFY_CHECKOUT_KEY, _now, claim_or_preview
from kglite_docs.errors import InvalidEnumError
from kglite_docs.schema import (
    CHUNK,
    CLAIM_TTL_SECONDS,
    CLASSIFY_DONE,
    CLASSIFY_NONE,
    LABEL_CONTESTED,
    element_label,
    label_for,
    labels_for,
    valid_element_values,
)
from kglite_docs.store import Store
from kglite_docs.store import rows as _df_dicts

# A chunk is "not yet classified" when it carries neither marker.
_NOT_DONE = "NOT c:Classified AND NOT c:Unclassified"


def next_unclassified(
    store: Store,
    *,
    doc_id: str | None = None,
    section_id: str | None = None,
    agent_id: str | None = None,
    limit: int = 20,
    ttl_seconds: int = CLAIM_TTL_SECONDS,
) -> list[dict[str, Any]]:
    """Ready chunks not yet classified, in reading order. With `agent_id`,
    atomically claims them (punchcard) on the classify checkout key — disjoint
    from study claims, so a classify fan-out and a study fan-out never collide.
    Without it, a read-only preview."""
    chunk_where = ["c.status = 'ready'"]
    base_params: dict[str, Any] = {"lim": int(limit)}
    if doc_id:
        chunk_where.append("c.doc_id = $doc")
        base_params["doc"] = doc_id
    if section_id:
        chunk_where.append("c.section_id = $sec")
        base_params["sec"] = section_id
    return claim_or_preview(
        store, where_sql=" AND ".join(chunk_where), not_done=_NOT_DONE,
        base_params=base_params, checkout_key=CLASSIFY_CHECKOUT_KEY,
        agent_id=agent_id, ttl_seconds=ttl_seconds,
    )


def classify_chunk(
    store: Store,
    *,
    chunk_id: str,
    elements: list[str],
    agent_id: str,
    model: str = "",
    confidence: float | None = None,
) -> dict[str, Any]:
    """Classify one chunk into zero or more registered element types. An empty
    `elements` is a deliberate "no schema element applies" → `:Unclassified`
    (still *covered*, not a blind spot). Re-classifying replaces this agent's
    prior record; labels are an add-only union across agents (recall-safe).
    A divergent second agent adds `:Contested`."""
    valid = valid_element_values()
    bad = [e for e in elements if e not in valid]
    if bad:
        raise InvalidEnumError(
            f"unknown element type(s) {bad} — not in the registered schema "
            f"({sorted(valid)})"
        )
    existing = _read_records(store, chunk_id)
    if existing is None:
        raise InvalidEnumError(f"chunk not found: {chunk_id}")

    register_agent(store, agent_id=agent_id)
    now = _now()
    conf = float(confidence) if confidence is not None else 1.0
    mine = [{"type": e, "conf": conf, "by": agent_id, "at": now, "model": model}
            for e in dict.fromkeys(elements)]
    records = [r for r in existing if r.get("by") != agent_id] + mine

    # Persist the canonical record, add this agent's element labels (add-only
    # union), and mark Classified/Unclassified exhaustively.
    store.cypher(
        "MATCH (c:Chunk {id: $id}) SET c.element_types_json = $j",
        params={"id": chunk_id, "j": json.dumps(records, ensure_ascii=False)},
    )
    for e in elements:
        lbl = element_label(e)
        if lbl:
            store.add_label(CHUNK, [chunk_id], lbl)
    any_elements = any(r.get("type") for r in records)
    store.swap_label(
        CHUNK, [chunk_id],
        add=label_for("chunk.classify", CLASSIFY_DONE if any_elements else CLASSIFY_NONE),
        remove_any_of=labels_for("chunk.classify"),
    )
    contested = _is_contested(records)
    if contested:
        store.add_label(CHUNK, [chunk_id], LABEL_CONTESTED)

    return {
        "chunk_id": chunk_id, "elements": list(dict.fromkeys(elements)),
        "classified": any_elements, "contested": contested, "by_agent": agent_id,
    }


def classify_many(
    store: Store, *, items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Classify many chunks in one call. Each item: `{chunk_id, elements,
    agent_id}` (+ optional `model`, `confidence`). Validated per item; a bad
    item raises (the items before it are already written — this is a convenience
    batch, not a transaction). Returns counts + per-chunk results."""
    results = []
    for it in items:
        if not isinstance(it, dict):
            raise InvalidEnumError(f"classify_many: each item must be a dict (got {type(it).__name__})")
        missing = [k for k in ("chunk_id", "elements", "agent_id") if k not in it]
        if missing:
            raise InvalidEnumError(f"classify_many: item missing required field(s) {missing}")
        results.append(classify_chunk(
            store, chunk_id=it["chunk_id"], elements=list(it["elements"]),
            agent_id=it["agent_id"], model=it.get("model", ""),
            confidence=it.get("confidence"),
        ))
    return {"classified": len(results), "results": results}


# ─── internals ────────────────────────────────────────────────────────────


def _read_records(store: Store, chunk_id: str) -> list[dict[str, Any]] | None:
    """Existing element records for a chunk, or None if the chunk doesn't exist."""
    rows = _df_dicts(store.cypher(
        "MATCH (c:Chunk {id: $id}) RETURN c.element_types_json AS j",
        params={"id": chunk_id},
    ))
    if not rows:
        return None
    raw = rows[0].get("j")
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _is_contested(records: list[dict[str, Any]]) -> bool:
    """True when ≥2 distinct agents classified the chunk with differing element
    sets (genuine disagreement, never a silent clobber)."""
    by_agent: dict[str, set[str]] = {}
    for r in records:
        if r.get("type"):
            by_agent.setdefault(str(r.get("by")), set()).add(str(r["type"]))
    sets = list(by_agent.values())
    return len(sets) >= 2 and any(s != sets[0] for s in sets[1:])
