"""Summaries + verification.

Key invariants:

- Author and verifier must be different agents (server-enforced).
- `source_text_hash` is captured at write time; re-ingest of a
  document compares and flips affected summaries to ``stale``.
- Summaries carry their own embeddings (separate store) so an agent
  can search "consensus about X" across enrichments without dredging
  raw chunks.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from kglite_docs.activity import register_agent
from kglite_docs.errors import InvalidEnumError, SelfVerificationError
from kglite_docs.ingest.hashing import combined_hash, text_hash
from kglite_docs.schema import (
    AGENT,
    AUTHORED,
    CHUNK,
    SUMMARIZES,
    SUMMARY,
    SUMMARY_TEXT_COL,
    VALID_DEPTHS,
    VALID_VERDICTS,
    VERIFICATION_STALE,
    VERIFICATION_UNVERIFIED,
    VERIFIED_BY,
    VERIFIES,
)
from kglite_docs.store import Store


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


from kglite_docs.store import rows as _df_dicts  # noqa: E402


def _source_text_hash(store: Store, target_id: str, target_kind: str) -> str:
    """Compute the rolling hash of the underlying chunks for a target."""
    if target_kind == CHUNK:
        df = _df_dicts(store.cypher(
            "MATCH (c:Chunk {id: $id}) RETURN c.text_hash AS h",
            params={"id": target_id},
        ))
        return df[0]["h"] if df else ""
    if target_kind == "Document":
        df = _df_dicts(store.cypher(
            "MATCH (d:Document {id: $id})-[:HAS_CHUNK]->(c:Chunk) "
            "RETURN c.text_hash AS h ORDER BY c.page_number, c.chunk_index",
            params={"id": target_id},
        ))
        return combined_hash([r["h"] for r in df if r.get("h")])
    if target_kind == "Page":
        df = _df_dicts(store.cypher(
            "MATCH (p:Page {id: $id})-[:HAS_CHUNK]->(c:Chunk) "
            "RETURN c.text_hash AS h ORDER BY c.chunk_index",
            params={"id": target_id},
        ))
        return combined_hash([r["h"] for r in df if r.get("h")])
    return ""


def add_summary(
    store: Store,
    embedder: Any,
    *,
    target_id: str,
    target_kind: str = CHUNK,
    depth: str = "chunk",
    text: str,
    agent_id: str,
    model: str = "",
    tags: Iterable[str] = (),
) -> str:
    if depth not in VALID_DEPTHS:
        raise InvalidEnumError(
            f"invalid depth: {depth!r} (expected one of {sorted(VALID_DEPTHS)})"
        )
    text = text.strip()
    if not text:
        raise InvalidEnumError("summary text must be non-empty")

    register_agent(store, agent_id=agent_id)
    sid = str(uuid.uuid4())
    sth = _source_text_hash(store, target_id, target_kind)
    now = _now()

    store.upsert_nodes(
        SUMMARY,
        [{
            "id": sid,
            "title": text[:80],
            "target_id": target_id,
            "target_kind": target_kind,
            "depth": depth,
            SUMMARY_TEXT_COL: text,
            "model": model,
            "created_at": now,
            "verification_status": VERIFICATION_UNVERIFIED,
            "verified_at": "",
            "verifier_notes": "",
            "source_text_hash": sth,
        }],
    )
    store.upsert_edges(
        AUTHORED, [{"src": agent_id, "dst": sid}],
        source_type=AGENT, target_type=SUMMARY,
    )
    store.upsert_edges(
        SUMMARIZES, [{"src": sid, "dst": target_id}],
        source_type=SUMMARY, target_type=target_kind,
    )
    # Embed the summary text into its own store
    try:
        vec = embedder.embed([text])[0]
        store.add_embeddings(SUMMARY, SUMMARY_TEXT_COL, {sid: vec})
    except Exception:
        # Embedding is optional; failure shouldn't block the write
        pass

    # Optional tag application
    for tag in tags:
        from kglite_docs.tagging import tag_chunk  # local import to avoid cycles
        if target_kind == CHUNK:
            tag_chunk(store, chunk_id=target_id, tag_name=tag, agent_id=agent_id)
    return sid


def verify_summary(
    store: Store,
    *,
    summary_id: str,
    verdict: str,
    verifier_agent_id: str,
    notes: str = "",
) -> dict[str, Any]:
    if verdict not in VALID_VERDICTS:
        raise InvalidEnumError(
            f"invalid verdict: {verdict!r} (expected one of {sorted(VALID_VERDICTS)})"
        )
    # Self-verification guard
    author_df = _df_dicts(store.cypher(
        "MATCH (a:Agent)-[:AUTHORED]->(s:Summary {id: $sid}) RETURN a.id AS id",
        params={"sid": summary_id},
    ))
    if not author_df:
        raise InvalidEnumError(f"summary not found: {summary_id}")
    author_id = author_df[0]["id"]
    if author_id == verifier_agent_id:
        raise SelfVerificationError(
            f"agent {verifier_agent_id!r} can't verify summary {summary_id} — they authored it"
        )

    register_agent(store, agent_id=verifier_agent_id)
    now = _now()
    # We store each verification as an immutable event node and recompute
    # the "current" status from the most-recent event. This avoids
    # kglite 0.10.3's mmap_vec panic on String SET updates.
    event_id = str(uuid.uuid4())
    store.upsert_nodes(
        "VerificationEvent",
        [{
            "id": event_id,
            "title": f"{verdict} by {verifier_agent_id}",
            "summary_id": summary_id,
            "verdict": verdict,
            "verifier_agent_id": verifier_agent_id,
            "notes": notes,
            "created_at": now,
        }],
    )
    store.upsert_edges(
        "HAS_VERIFICATION", [{"src": summary_id, "dst": event_id}],
        source_type=SUMMARY, target_type="VerificationEvent",
    )
    store.upsert_edges(
        VERIFIED_BY, [{"src": summary_id, "dst": verifier_agent_id}],
        source_type=SUMMARY, target_type=AGENT,
    )
    store.upsert_edges(
        AUTHORED, [{"src": verifier_agent_id, "dst": event_id}],
        source_type=AGENT, target_type="VerificationEvent",
    )
    return {
        "summary_id": summary_id, "status": verdict,
        "verified_at": now, "verified_by": verifier_agent_id,
        "event_id": event_id,
    }


def link_verification(
    store: Store, *, verifier_summary_id: str, target_summary_id: str
) -> dict[str, Any]:
    store.upsert_edges(
        VERIFIES, [{"src": verifier_summary_id, "dst": target_summary_id}],
        source_type=SUMMARY, target_type=SUMMARY,
    )
    return {"linked": True, "from": verifier_summary_id, "to": target_summary_id}


def get_summaries(
    store: Store,
    *,
    target_id: str,
    target_kind: str | None = None,
    status: str | None = None,
    depth: str | None = None,
) -> list[dict[str, Any]]:
    where = ["s.target_id = $id"]
    params: dict[str, Any] = {"id": target_id}
    if target_kind:
        where.append("s.target_kind = $tk")
        params["tk"] = target_kind
    if depth:
        where.append("s.depth = $d")
        params["d"] = depth
    df = store.cypher(
        f"MATCH (s:Summary) WHERE {' AND '.join(where)} "
        "OPTIONAL MATCH (a:Agent)-[:AUTHORED]->(s) "
        "RETURN s.id AS id, s.text AS text, s.depth AS depth, "
        "s.verification_status AS initial_status, s.created_at AS created_at, "
        "s.model AS model, s.source_text_hash AS source_text_hash, "
        "a.id AS author_agent "
        "ORDER BY s.created_at DESC",
        params=params,
    )
    rows = _df_dicts(df)
    # Layer in the latest verification event per summary (event-sourced).
    for row in rows:
        ev = _df_dicts(store.cypher(
            "MATCH (s:Summary {id: $sid})-[:HAS_VERIFICATION]->(e:VerificationEvent) "
            "RETURN e.verdict AS verdict, e.verifier_agent_id AS verifier, "
            "e.created_at AS at, e.notes AS notes "
            "ORDER BY e.created_at DESC LIMIT 1",
            params={"sid": row["id"]},
        ))
        if ev:
            row["status"] = ev[0]["verdict"]
            row["verified_at"] = ev[0]["at"]
            row["verifier_agent"] = ev[0]["verifier"]
            row["notes"] = ev[0]["notes"]
        else:
            row["status"] = row.get("initial_status") or VERIFICATION_UNVERIFIED
            row["verified_at"] = None
            row["verifier_agent"] = None
            row["notes"] = ""
    if status:
        rows = [r for r in rows if r["status"] == status]
    return rows


def find_consensus(
    store: Store, embedder: Any, *, query: str, top_k: int = 20
) -> list[dict[str, Any]]:
    """Vector search over Summary nodes — surfaces agreement / conflict.

    Returns summaries grouped by target_id with a count and the
    verification statuses present.
    """
    vec = embedder.embed([query])[0]
    hits = store.vector_search(SUMMARY, SUMMARY_TEXT_COL, vec, top_k=top_k)
    if not hits:
        return []
    ids = [h["id"] for h in hits]
    df = _df_dicts(store.cypher(
        "MATCH (s:Summary) WHERE s.id IN $ids "
        "RETURN s.id AS id, s.target_id AS target_id, s.text AS text, "
        "s.verification_status AS status",
        params={"ids": ids},
    ))
    by_id = {r["id"]: r for r in df}
    grouped: dict[str, dict[str, Any]] = {}
    for h in hits:
        info = by_id.get(h["id"], {})
        target_id = info.get("target_id", "")
        g = grouped.setdefault(
            target_id,
            {"target_id": target_id, "summaries": [], "statuses": []},
        )
        g["summaries"].append({
            "id": h["id"],
            "score": h.get("score"),
            "text": info.get("text", ""),
            "status": info.get("status", ""),
        })
        g["statuses"].append(info.get("status", ""))
    out: list[dict[str, Any]] = []
    for g in grouped.values():
        statuses = g.pop("statuses")
        g["count"] = len(g["summaries"])
        g["status_counts"] = {s: statuses.count(s) for s in set(statuses)}
        out.append(g)
    out.sort(key=lambda x: x["count"], reverse=True)
    return out


def mark_stale_for_doc(store: Store, doc_id: str) -> int:
    """After a re-ingest, flip any summary whose `source_text_hash` no
    longer matches the regenerated chunks to ``stale``. Returns count flipped.
    """
    # Find summaries that target this doc's chunks
    df = _df_dicts(store.cypher(
        "MATCH (s:Summary)-[:SUMMARIZES]->(t) "
        "WHERE (t:Chunk AND t.doc_id = $doc_id) OR (t:Document AND t.id = $doc_id) OR "
        "      (t:Page AND t.doc_id = $doc_id) "
        "RETURN s.id AS id, s.target_id AS target_id, s.target_kind AS target_kind, "
        "s.source_text_hash AS sth, s.verification_status AS status",
        params={"doc_id": doc_id},
    ))
    count = 0
    for row in df:
        if row["status"] == VERIFICATION_STALE:
            continue
        current = _source_text_hash(store, row["target_id"], row["target_kind"])
        if current != row["sth"]:
            # Record a STALE verification event (event-sourced; avoids
            # the kglite 0.10.3 mmap_vec panic on String SET).
            event_id = str(uuid.uuid4())
            store.upsert_nodes(
                "VerificationEvent",
                [{
                    "id": event_id, "title": "stale (auto)",
                    "summary_id": row["id"], "verdict": VERIFICATION_STALE,
                    "verifier_agent_id": "system:stale-check",
                    "notes": "source text hash drifted on re-ingest",
                    "created_at": _now(),
                }],
            )
            store.upsert_edges(
                "HAS_VERIFICATION", [{"src": row["id"], "dst": event_id}],
                source_type=SUMMARY, target_type="VerificationEvent",
            )
            count += 1
    return count
