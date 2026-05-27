"""Agent identity + view tracking.

Agents are lazily registered on their first mutation; views can be
recorded explicitly (with context) or implicitly (when `search` /
`get_chunk` are called with an `agent_id`).

Aggregate `view_count` + `last_viewed_at` on the Chunk is updated on
every recorded view — a cheap denormalisation so listings can sort by
attention without joining View nodes.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from kglite_docs.schema import (
    AGENT,
    AUTHORED,
    CHUNK,
    VIEW,
    VIEWED,
)
from kglite_docs.store import Store


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


from kglite_docs.store import rows as _df_dicts  # noqa: E402


def register_agent(
    store: Store, *, agent_id: str, kind: str = "llm", model: str = ""
) -> dict[str, Any]:
    """Idempotent. Touches `last_seen` if the agent exists; creates otherwise."""
    now = _now()
    existing = _df_dicts(
        store.cypher("MATCH (a:Agent {id: $id}) RETURN a.id AS id", params={"id": agent_id})
    )
    if existing:
        store.cypher(
            "MATCH (a:Agent {id: $id}) SET a.last_seen = $now, a.action_count = coalesce(a.action_count, 0) + 1",
            params={"id": agent_id, "now": now},
        )
        return {"id": agent_id, "created": False, "last_seen": now}
    store.upsert_nodes(
        AGENT,
        [{
            "id": agent_id,
            "title": agent_id,
            "kind": kind,
            "model": model,
            "first_seen": now,
            "last_seen": now,
            "action_count": 1,
        }],
    )
    return {"id": agent_id, "created": True, "last_seen": now}


def list_agents(store: Store) -> list[dict[str, Any]]:
    df = store.cypher(
        "MATCH (a:Agent) RETURN a.id AS id, a.kind AS kind, a.model AS model, "
        "a.first_seen AS first_seen, a.last_seen AS last_seen, a.action_count AS actions "
        "ORDER BY a.last_seen DESC"
    )
    return _df_dicts(df)


def record_view(
    store: Store,
    *,
    agent_id: str,
    target_id: str,
    target_kind: str = CHUNK,
    context: str = "",
) -> dict[str, Any]:
    """Record an agent viewing a target. Lazy-registers the agent.

    - Always bumps the target's `view_count` and `last_viewed_at`.
    - Creates a `View` node + edges when `context` is non-empty (so we
      can surface "the query that led here" later); pure visits skip
      the View node to keep the graph lean.
    """
    register_agent(store, agent_id=agent_id)
    now = _now()
    if target_kind == CHUNK:
        store.cypher(
            "MATCH (c:Chunk {id: $id}) "
            "SET c.view_count = coalesce(c.view_count, 0) + 1, c.last_viewed_at = $now",
            params={"id": target_id, "now": now},
        )
    if not context:
        return {"recorded": True, "view_node": None}
    vid = str(uuid.uuid4())
    store.upsert_nodes(
        VIEW,
        [{
            "id": vid,
            "title": context[:60],
            "agent_id": agent_id,
            "target_id": target_id,
            "target_kind": target_kind,
            "at": now,
            "context": context,
        }],
    )
    store.upsert_edges(
        AUTHORED, [{"src": agent_id, "dst": vid}],
        source_type=AGENT, target_type=VIEW,
    )
    # Aggregate VIEWED edge (Agent → Chunk) — multiple writes are tolerated;
    # we don't need uniqueness here.
    if target_kind == CHUNK:
        store.upsert_edges(
            VIEWED, [{"src": agent_id, "dst": target_id, "at": now, "context": context}],
            source_type=AGENT, target_type=CHUNK,
        )
    return {"recorded": True, "view_node": vid}


def agent_activity(store: Store, agent_id: str, *, limit: int = 50) -> dict[str, Any]:
    """Return summary + recent activity for an agent."""
    a_df = _df_dicts(store.cypher(
        "MATCH (a:Agent {id: $id}) RETURN a.id AS id, a.kind AS kind, "
        "a.first_seen AS first_seen, a.last_seen AS last_seen, a.action_count AS actions",
        params={"id": agent_id},
    ))
    if not a_df:
        return {"agent": None, "views": [], "summaries": [], "tags": []}
    views = _df_dicts(store.cypher(
        "MATCH (a:Agent {id: $id})-[:AUTHORED]->(v:View) "
        f"RETURN v.target_id AS target_id, v.target_kind AS target_kind, v.context AS context, v.at AS at "
        f"ORDER BY v.at DESC LIMIT {int(limit)}",
        params={"id": agent_id},
    ))
    sums = _df_dicts(store.cypher(
        "MATCH (a:Agent {id: $id})-[:AUTHORED]->(s:Summary) "
        f"RETURN s.id AS id, s.target_id AS target_id, s.text AS text, s.verification_status AS status "
        f"ORDER BY s.created_at DESC LIMIT {int(limit)}",
        params={"id": agent_id},
    ))
    return {"agent": a_df[0], "views": views, "summaries": sums}
