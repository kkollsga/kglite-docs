"""Tag application on chunks.

kglite enforces at-most-one edge per `(src, dst, type)` triple, so we
*reify* tagging as a `Tagging` node — one per (chunk, tag, agent) — to
let multiple agents distinctly tag the same chunk:

    (Chunk) -[:TAGGED_AS]-> (Tagging) -[:OF_TAG]-> (Tag)
                                ↑
                       (Agent) -[:AUTHORED]-

`list_tags()` joins through `Tagging` to surface `by_agent` /
`created_at` / `confidence` cleanly. Idempotency for re-tagging by the
same agent is enforced by checking for an existing Tagging node first.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

from kglite_docs.activity import register_agent
from kglite_docs.schema import (
    AGENT,
    AUTHORED,
    CHUNK,
    TAG,
    TAGGED_AS,
    label_for,
)
from kglite_docs.store import Store
from kglite_docs.store import rows as _df_dicts

TAGGING: str = "Tagging"
OF_TAG: str = "OF_TAG"

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(name: str) -> str:
    return _SLUG_RE.sub("-", name.strip().lower()).strip("-")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_tag(store: Store, name: str, *, kind: str = "custom", description: str = "") -> str:
    tag_id = _slug(name)
    existing = _df_dicts(
        store.cypher("MATCH (t:Tag {id: $id}) RETURN t.id AS id", params={"id": tag_id})
    )
    if not existing:
        store.upsert_nodes(
            TAG,
            [{"id": tag_id, "title": name, "name": name, "kind": kind, "description": description}],
        )
        # Tag.kind → label so `MATCH (t:Tag:Topic)` filters efficiently.
        kind_label = label_for("tag.kind", kind)
        if kind_label:
            store.add_label(TAG, [tag_id], kind_label)
    return tag_id


def tag_chunk(
    store: Store,
    *,
    chunk_id: str,
    tag_name: str,
    kind: str = "custom",
    agent_id: str,
    confidence: float | None = None,
) -> dict[str, Any]:
    register_agent(store, agent_id=agent_id)
    tag_id = _ensure_tag(store, tag_name, kind=kind)
    now = _now()
    # idempotency: this (chunk, tag, agent) already exists?
    existing = _df_dicts(store.cypher(
        "MATCH (c:Chunk {id: $cid})-[:TAGGED_AS]->(tg:Tagging)-[:OF_TAG]->(t:Tag {id: $tid}) "
        "WHERE tg.by_agent = $aid RETURN tg.id AS id",
        params={"cid": chunk_id, "tid": tag_id, "aid": agent_id},
    ))
    if existing:
        return {"created": False, "tag_id": tag_id, "chunk_id": chunk_id,
                "tagging_id": existing[0]["id"]}

    tagging_id = str(uuid.uuid4())
    row: dict[str, Any] = {
        "id": tagging_id,
        "title": f"{tag_name} by {agent_id}",
        "chunk_id": chunk_id,
        "tag_id": tag_id,
        "by_agent": agent_id,
        "created_at": now,
    }
    if confidence is not None:
        row["confidence"] = float(confidence)
    store.upsert_nodes(TAGGING, [row])
    store.upsert_edges(TAGGED_AS, [{"src": chunk_id, "dst": tagging_id}],
                       source_type=CHUNK, target_type=TAGGING)
    store.upsert_edges(OF_TAG, [{"src": tagging_id, "dst": tag_id}],
                       source_type=TAGGING, target_type=TAG)
    store.upsert_edges(AUTHORED, [{"src": agent_id, "dst": tagging_id}],
                       source_type=AGENT, target_type=TAGGING)
    return {
        "created": True, "tag_id": tag_id, "chunk_id": chunk_id,
        "tagging_id": tagging_id, "by_agent": agent_id,
    }


def untag_chunk(
    store: Store, *, chunk_id: str, tag_name: str, agent_id: str
) -> dict[str, Any]:
    tag_id = _slug(tag_name)
    store.cypher(
        "MATCH (c:Chunk {id: $cid})-[:TAGGED_AS]->(tg:Tagging)-[:OF_TAG]->(t:Tag {id: $tid}) "
        "WHERE tg.by_agent = $aid DETACH DELETE tg",
        params={"cid": chunk_id, "tid": tag_id, "aid": agent_id},
    )
    return {"removed": True, "tag_id": tag_id, "chunk_id": chunk_id, "by_agent": agent_id}


def list_tags(
    store: Store,
    *,
    doc_id: str | None = None,
    chunk_id: str | None = None,
    agent_id: str | None = None,
    kind: str | None = None,
) -> list[dict[str, Any]]:
    # Tag kind becomes a label predicate inline; everything else is
    # a normal WHERE filter.
    tag_label = ":" + label_for("tag.kind", kind) if kind else ""
    parts = [f"MATCH (c:Chunk)-[:TAGGED_AS]->(tg:Tagging)-[:OF_TAG]->(t:Tag{tag_label})"]
    where: list[str] = []
    params: dict[str, Any] = {}
    if doc_id:
        where.append("c.doc_id = $doc_id")
        params["doc_id"] = doc_id
    if chunk_id:
        where.append("c.id = $chunk_id")
        params["chunk_id"] = chunk_id
    if agent_id:
        where.append("tg.by_agent = $agent_id")
        params["agent_id"] = agent_id
    if where:
        parts.append("WHERE " + " AND ".join(where))
    parts.append(
        "RETURN t.id AS tag_id, t.name AS name, t.kind AS kind, "
        "tg.by_agent AS by_agent, tg.created_at AS at, tg.id AS tagging_id, "
        "tg.confidence AS confidence, "
        "c.id AS chunk_id, c.doc_id AS doc_id"
    )
    return _df_dicts(store.cypher(" ".join(parts), params=params))


def chunks_by_tag(store: Store, *, tag_name: str, limit: int = 100) -> list[dict[str, Any]]:
    tag_id = _slug(tag_name)
    # Rank by the strongest confidence on the tag (so a "supports"/weight tag
    # surfaces the most-confident chunks first), then by tagger count. Both
    # `confidence` and `taggers` are returned so callers can re-rank.
    return _df_dicts(store.cypher(
        "MATCH (c:Chunk)-[:TAGGED_AS]->(tg:Tagging)-[:OF_TAG]->(t:Tag {id: $tid}) "
        "WITH c, count(tg) AS taggers, max(tg.confidence) AS confidence "
        "RETURN c.id AS id, c.doc_id AS doc_id, c.page_number AS page, "
        "c.title AS title, c.text AS text, taggers, confidence "
        f"ORDER BY confidence DESC, taggers DESC LIMIT {int(limit)}",
        params={"tid": tag_id},
    ))
