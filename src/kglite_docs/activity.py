"""Agent identity, configuration, view tracking, and activity rollups.

An `Agent` node is *both* an identity (the actor whose ``AUTHORED``
edges record activity) and an optional **template** carrying the
loading context for that agent — `role`, `system_prompt`, `model`,
`tools`, free-form `context` JSON, and a human description. Orchestrators
fetch an agent with `get_agent(agent_id)` and use the returned template
to launch the actual LLM call; every subsequent corpus mutation under
the same `agent_id` is attributed back to the same template.

Agents are lazy-registered on first mutation if you haven't called
`upsert_agent` explicitly first. The lazy path won't clobber existing
configuration (template fields stay intact across writes).
"""

from __future__ import annotations

import json
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
from kglite_docs.store import rows as _df_dicts


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(obj: object) -> str:
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        return "{}"


def _load_json(text: str | None, fallback: Any) -> Any:
    if not text:
        return fallback
    try:
        return json.loads(text)
    except Exception:
        return fallback


# ─── identity / templates ──────────────────────────────────────────────────


def upsert_agent(
    store: Store,
    *,
    agent_id: str,
    kind: str = "llm",
    model: str = "",
    role: str = "",
    system_prompt: str = "",
    tools: list[str] | None = None,
    context: dict[str, Any] | None = None,
    description: str = "",
) -> dict[str, Any]:
    """Create or update an agent template.

    All template fields are optional — pass only the ones you want to
    set. Existing fields you *don't* pass are preserved (the function
    is field-level merge, not whole-record replace).

    Returns the resulting agent config + a ``created`` flag.
    """
    now = _now()
    existing = _df_dicts(store.cypher(
        "MATCH (a:Agent {id: $id}) "
        "RETURN a.kind AS kind, a.model AS model, a.role AS role, "
        "a.system_prompt AS system_prompt, a.tools_json AS tools_json, "
        "a.context_json AS context_json, a.description AS description",
        params={"id": agent_id},
    ))
    if not existing:
        store.upsert_nodes(
            AGENT,
            [{
                "id": agent_id,
                "title": agent_id,
                "kind": kind,
                "model": model,
                "role": role,
                "system_prompt": system_prompt,
                "tools_json": _safe_json(tools or []),
                "context_json": _safe_json(context or {}),
                "description": description,
                "first_seen": now,
                "last_seen": now,
                "action_count": 0,
            }],
        )
        return {**get_agent(store, agent_id=agent_id), "created": True}

    # Merge: only SET the properties the caller actually supplied.
    cur = existing[0]
    updates: dict[str, Any] = {}
    if kind and kind != cur.get("kind"):
        updates["kind"] = kind
    if model and model != cur.get("model"):
        updates["model"] = model
    if role and role != cur.get("role"):
        updates["role"] = role
    if system_prompt and system_prompt != cur.get("system_prompt"):
        updates["system_prompt"] = system_prompt
    if tools is not None:
        updates["tools_json"] = _safe_json(tools)
    if context is not None:
        updates["context_json"] = _safe_json(context)
    if description and description != cur.get("description"):
        updates["description"] = description

    for key, value in updates.items():
        store.cypher(
            f"MATCH (a:Agent {{id: $id}}) SET a.{key} = $v",
            params={"id": agent_id, "v": value},
        )
    return {**get_agent(store, agent_id=agent_id), "created": False}


def register_agent(
    store: Store, *, agent_id: str, kind: str = "llm", model: str = "",
) -> dict[str, Any]:
    """Lazy registration — touches `last_seen` + bumps `action_count`
    if the agent exists, creates a minimal Agent node otherwise.

    Critically, this does **not** overwrite template fields
    (`role`, `system_prompt`, etc.) when the agent already exists.
    Use `upsert_agent` for explicit configuration writes.
    """
    now = _now()
    existing = _df_dicts(
        store.cypher("MATCH (a:Agent {id: $id}) RETURN a.id AS id", params={"id": agent_id})
    )
    if existing:
        store.cypher(
            "MATCH (a:Agent {id: $id}) "
            "SET a.last_seen = $now, "
            "    a.action_count = coalesce(a.action_count, 0) + 1",
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
            "role": "",
            "system_prompt": "",
            "tools_json": "[]",
            "context_json": "{}",
            "description": "",
            "first_seen": now,
            "last_seen": now,
            "action_count": 1,
        }],
    )
    return {"id": agent_id, "created": True, "last_seen": now}


def get_agent(store: Store, *, agent_id: str) -> dict[str, Any]:
    """Fetch the full agent config — template + counters. Returns
    `{}` if the agent isn't registered yet."""
    rows = _df_dicts(store.cypher(
        "MATCH (a:Agent {id: $id}) "
        "RETURN a.id AS id, a.kind AS kind, a.model AS model, a.role AS role, "
        "a.system_prompt AS system_prompt, a.tools_json AS tools_json, "
        "a.context_json AS context_json, a.description AS description, "
        "a.first_seen AS first_seen, a.last_seen AS last_seen, "
        "a.action_count AS action_count",
        params={"id": agent_id},
    ))
    if not rows:
        return {}
    row = rows[0]
    # Hydrate the JSON fields so callers get lists/dicts, not strings
    row["tools"] = _load_json(row.pop("tools_json", None), [])
    row["context"] = _load_json(row.pop("context_json", None), {})
    return row


def list_agents(
    store: Store, *, role: str | None = None, kind: str | None = None,
) -> list[dict[str, Any]]:
    """List configured agents. Filter by `role` (free-text — whatever
    you wrote at upsert time) or `kind` (llm/human/service)."""
    where: list[str] = []
    params: dict[str, Any] = {}
    if role:
        where.append("a.role = $role")
        params["role"] = role
    if kind:
        where.append("a.kind = $kind")
        params["kind"] = kind
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    df = store.cypher(
        f"MATCH (a:Agent) {clause} "
        "RETURN a.id AS id, a.kind AS kind, a.model AS model, a.role AS role, "
        "a.description AS description, a.first_seen AS first_seen, "
        "a.last_seen AS last_seen, a.action_count AS actions "
        "ORDER BY a.last_seen DESC",
        params=params,
    )
    return _df_dicts(df)


# ─── view tracking (unchanged) ────────────────────────────────────────────


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
    if target_kind == CHUNK:
        store.upsert_edges(
            VIEWED, [{"src": agent_id, "dst": target_id, "at": now, "context": context}],
            source_type=AGENT, target_type=CHUNK,
        )
    return {"recorded": True, "view_node": vid}


# ─── activity rollups ─────────────────────────────────────────────────────


def agent_activity(
    store: Store,
    agent_id: str,
    *,
    target_id: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """What has this agent done — optionally scoped to one target node.

    Returns a dict with the agent config + lists of activity bucketed
    by kind (`views`, `summaries`, `tags`, `translations`,
    `review_events`, `verification_events`). With `target_id` set,
    only activity touching that target is returned.

    Lets you answer: "what has the reviewer agent done on chunk X?"
    in one call.
    """
    cfg = get_agent(store, agent_id=agent_id)
    if not cfg:
        return {
            "agent": None, "views": [], "summaries": [], "tags": [],
            "translations": [], "review_events": [], "verification_events": [],
        }

    target_clause = ""
    params: dict[str, Any] = {"id": agent_id}
    if target_id:
        target_clause = " AND v.target_id = $tid"
        params["tid"] = target_id

    views = _df_dicts(store.cypher(
        "MATCH (a:Agent {id: $id})-[:AUTHORED]->(v:View) "
        f"WHERE 1=1 {target_clause} "
        "RETURN v.id AS id, v.target_id AS target_id, v.target_kind AS target_kind, "
        "v.context AS context, v.at AS at "
        f"ORDER BY v.at DESC LIMIT {int(limit)}",
        params=params,
    ))

    sum_params: dict[str, Any] = {"id": agent_id}
    sum_where = ""
    if target_id:
        sum_where = "WHERE s.target_id = $tid"
        sum_params["tid"] = target_id
    summaries = _df_dicts(store.cypher(
        "MATCH (a:Agent {id: $id})-[:AUTHORED]->(s:Summary) "
        f"{sum_where} "
        "RETURN s.id AS id, s.target_id AS target_id, s.text AS text, "
        "s.verification_status AS status, s.created_at AS created_at "
        f"ORDER BY s.created_at DESC LIMIT {int(limit)}",
        params=sum_params,
    ))

    tags_params: dict[str, Any] = {"id": agent_id}
    tags_where = ""
    if target_id:
        tags_where = "WHERE tg.chunk_id = $tid"
        tags_params["tid"] = target_id
    tags = _df_dicts(store.cypher(
        "MATCH (a:Agent {id: $id})-[:AUTHORED]->(tg:Tagging)-[:OF_TAG]->(t:Tag) "
        f"{tags_where} "
        "RETURN tg.id AS tagging_id, tg.chunk_id AS chunk_id, "
        "tg.created_at AS created_at, t.name AS tag, t.kind AS tag_kind "
        f"ORDER BY tg.created_at DESC LIMIT {int(limit)}",
        params=tags_params,
    ))

    tr_params: dict[str, Any] = {"id": agent_id}
    tr_where = ""
    if target_id:
        tr_where = "WHERE t.chunk_id = $tid"
        tr_params["tid"] = target_id
    translations = _df_dicts(store.cypher(
        "MATCH (a:Agent {id: $id})-[:AUTHORED]->(t:Translation) "
        f"{tr_where} "
        "RETURN t.id AS id, t.chunk_id AS chunk_id, t.target_lang AS lang, "
        "t.status AS status, t.created_at AS created_at "
        f"ORDER BY t.created_at DESC LIMIT {int(limit)}",
        params=tr_params,
    ))

    # Review + verification events are scoped through their target
    # ticket / summary, not directly by target_id. We filter them by
    # *target_id chain* when one is requested.
    rev_params: dict[str, Any] = {"id": agent_id}
    rev_q = (
        "MATCH (a:Agent {id: $id})-[:AUTHORED]->(e:ReviewEvent) "
        "OPTIONAL MATCH (tk:ReviewTicket)-[:HAS_REVIEW_EVENT]->(e) "
    )
    rev_where = ""
    if target_id:
        rev_where = "WHERE tk.target_id = $tid"
        rev_params["tid"] = target_id
    review_events = _df_dicts(store.cypher(
        rev_q + rev_where +
        " RETURN e.id AS id, tk.id AS ticket_id, tk.target_id AS target_id, "
        "e.event_type AS event_type, e.created_at AS at "
        f"ORDER BY e.created_at DESC LIMIT {int(limit)}",
        params=rev_params,
    ))

    ver_params: dict[str, Any] = {"id": agent_id}
    ver_q = (
        "MATCH (a:Agent {id: $id})-[:AUTHORED]->(e:VerificationEvent) "
        "OPTIONAL MATCH (s:Summary)-[:HAS_VERIFICATION]->(e) "
    )
    ver_where = ""
    if target_id:
        ver_where = "WHERE s.target_id = $tid"
        ver_params["tid"] = target_id
    verification_events = _df_dicts(store.cypher(
        ver_q + ver_where +
        " RETURN e.id AS id, e.summary_id AS summary_id, "
        "e.verdict AS verdict, e.created_at AS at "
        f"ORDER BY e.created_at DESC LIMIT {int(limit)}",
        params=ver_params,
    ))

    return {
        "agent": cfg,
        "views": views,
        "summaries": summaries,
        "tags": tags,
        "translations": translations,
        "review_events": review_events,
        "verification_events": verification_events,
    }
