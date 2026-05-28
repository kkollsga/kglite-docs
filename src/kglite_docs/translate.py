"""Translation layer.

Per-chunk translations are stored as `Translation` nodes (one per
chunk × target_lang × agent), so multiple translators can co-exist
and a translation has its own provenance + lifecycle. The original
text on `Chunk.text` is never overwritten.

Schema additions::

    (Chunk)-[:HAS_TRANSLATION]->(Translation)
    (Agent)-[:AUTHORED]->(Translation)

`Translation` properties: ``id``, ``chunk_id``, ``target_lang``,
``text``, ``model``, ``status`` (``draft`` / ``reviewed``),
``created_at``, ``source_text_hash``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from kglite_docs.activity import register_agent
from kglite_docs.schema import AGENT, AUTHORED, CHUNK, label_for, labels_for
from kglite_docs.store import Store
from kglite_docs.store import rows as _rows

TRANSLATION: str = "Translation"
HAS_TRANSLATION: str = "HAS_TRANSLATION"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_translation(
    store: Store,
    *,
    chunk_id: str,
    target_lang: str,
    text: str,
    agent_id: str,
    model: str = "",
    status: str = "draft",
) -> str:
    """Store an agent-produced translation of a chunk."""
    text = text.strip()
    if not text:
        raise ValueError("translation text must be non-empty")
    register_agent(store, agent_id=agent_id)
    tid = str(uuid.uuid4())
    # Capture source text_hash so we can flag stale translations later
    src = _rows(store.cypher(
        "MATCH (c:Chunk {id: $cid}) RETURN c.text_hash AS h",
        params={"cid": chunk_id},
    ))
    sth = src[0]["h"] if src else ""
    store.upsert_nodes(
        TRANSLATION,
        [{
            "id": tid, "title": text[:80],
            "chunk_id": chunk_id, "target_lang": target_lang,
            "text": text, "model": model, "status": status,
            "created_at": _now(), "source_text_hash": sth,
        }],
    )
    store.upsert_edges(HAS_TRANSLATION, [{"src": chunk_id, "dst": tid}],
                       source_type=CHUNK, target_type=TRANSLATION)
    store.upsert_edges(AUTHORED, [{"src": agent_id, "dst": tid}],
                       source_type=AGENT, target_type=TRANSLATION)
    # Initial status label (Draft / Reviewed). Translations that ship
    # straight to `status="reviewed"` (unusual but allowed) skip Draft.
    status_label = label_for("translation.status", status)
    if status_label:
        store.add_label(TRANSLATION, [tid], status_label)
    return tid


def get_translations(
    store: Store, *, chunk_id: str, target_lang: str | None = None
) -> list[dict[str, Any]]:
    where = ["t.chunk_id = $cid"]
    params: dict[str, Any] = {"cid": chunk_id}
    if target_lang:
        where.append("t.target_lang = $lang")
        params["lang"] = target_lang
    return _rows(store.cypher(
        f"MATCH (c:Chunk {{id: $cid}})-[:HAS_TRANSLATION]->(t:Translation) "
        f"OPTIONAL MATCH (a:Agent)-[:AUTHORED]->(t) "
        f"WHERE {' AND '.join(where)} "
        "RETURN t.id AS id, t.target_lang AS lang, t.text AS text, "
        "t.status AS status, t.created_at AS created_at, t.model AS model, "
        "a.id AS by_agent",
        params=params,
    ))


def mark_translation_reviewed(
    store: Store, *, translation_id: str, reviewer_agent_id: str
) -> dict[str, Any]:
    """Flip a translation's status from draft → reviewed. Author/reviewer
    can be the same agent — translation review is less adversarial than
    summary verification."""
    register_agent(store, agent_id=reviewer_agent_id)
    store.cypher(
        "MATCH (t:Translation {id: $tid}) SET t.status = 'reviewed', "
        "t.reviewed_at = $now, t.reviewed_by = $rid",
        params={"tid": translation_id, "now": _now(), "rid": reviewer_agent_id},
    )
    # Swap Draft → Reviewed label
    store.swap_label(
        TRANSLATION, [translation_id],
        add=label_for("translation.status", "reviewed"),
        remove_any_of=labels_for("translation.status"),
    )
    return {"translation_id": translation_id, "status": "reviewed"}


def assemble_translated_document(
    store: Store, *, doc_id: str, target_lang: str, prefer_reviewed: bool = True
) -> dict[str, Any]:
    """Stitch a document's translated chunks back together in reading
    order. Pages without a translation in `target_lang` retain the
    original chunk text (flagged so callers can highlight gaps).
    """
    # First, get all chunks
    chunks = _rows(store.cypher(
        "MATCH (d:Document {id: $did})-[:HAS_CHUNK]->(c:Chunk) "
        "RETURN c.id AS id, c.page_number AS page, c.chunk_index AS chunk_index, "
        "c.text AS original "
        "ORDER BY c.page_number, c.chunk_index",
        params={"did": doc_id},
    ))
    # Then, fetch translations for this language separately and join in Python
    tr_rows = _rows(store.cypher(
        "MATCH (c:Chunk)-[:HAS_TRANSLATION]->(t:Translation {target_lang: $lang}) "
        "WHERE c.doc_id = $did "
        "RETURN c.id AS chunk_id, t.id AS id, t.text AS text, t.status AS status, "
        "t.target_lang AS lang",
        params={"did": doc_id, "lang": target_lang},
    ))
    by_chunk: dict[str, list[dict[str, Any]]] = {}
    for r in tr_rows:
        by_chunk.setdefault(r["chunk_id"], []).append(r)
    for row in chunks:
        row["translations"] = by_chunk.get(row["id"], [])
    out: list[dict[str, Any]] = []
    missing = 0
    for row in chunks:
        trs = [t for t in row.get("translations", []) if t.get("id")]
        if prefer_reviewed:
            trs.sort(key=lambda t: 0 if t.get("status") == "reviewed" else 1)
        chosen = trs[0] if trs else None
        if chosen is None:
            missing += 1
        out.append({
            "chunk_id": row["id"],
            "page": row.get("page"),
            "text": (chosen["text"] if chosen else row.get("original", "")),
            "translated": bool(chosen),
            "status": chosen.get("status") if chosen else None,
        })
    return {
        "doc_id": doc_id, "target_lang": target_lang,
        "chunks": out, "missing_translation_count": missing,
        "coverage": (len(out) - missing) / len(out) if out else 1.0,
    }
