"""Scanned-page detection + agent-driven OCR submission.

`list_pending_ocr` returns pages where text extraction came up empty
but the page carries images — typically scanned PDFs. Each entry
includes a base64-encoded PNG render so the agent can read the page
visually and return its markdown via `submit_ocr`.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from kglite_docs.activity import register_agent
from kglite_docs.errors import InvalidEnumError
from kglite_docs.ingest.chunker import chunk_page
from kglite_docs.ingest.parser import render_page_png
from kglite_docs.schema import (
    CHUNK,
    CHUNK_STATUS_READY,
    CHUNK_TEXT_COL,
    HAS_CHUNK,
    NEXT_CHUNK,
    PAGE,
    label_for,
)
from kglite_docs.store import Store
from kglite_docs.store import rows as _df_dicts  # noqa: E402


def ocr_status(
    store: Store,
    *,
    doc_id: str | None = None,
) -> dict[str, Any]:
    """Per-document OCR coverage summary plus corpus-wide totals.

    Returns::

        {
          "total_pages": int,
          "ready_pages": int,
          "pending_pages": int,
          "documents_total": int,
          "documents_with_pending": int,
          "documents": [
            {
              "doc_id": ..., "title": ..., "format": ...,
              "pages": int, "ready": int, "pending": int,
              "pending_fraction": float,  # 0.0 = fully OCR'd, 1.0 = entirely unread
            },
            ...
          ],
        }

    Ordered with documents that still need work first. Pass `doc_id` to
    restrict to a single document; the corpus-wide totals still reflect
    the whole graph so you can sanity-check.
    """
    where = ""
    params: dict[str, Any] = {}
    if doc_id:
        where = "WHERE d.id = $doc_id"
        params["doc_id"] = doc_id
    rows = _df_dicts(store.cypher(
        f"""
        MATCH (d:Document)
        {where}
        OPTIONAL MATCH (d)-[:HAS_PAGE]->(p:Page)
        WITH d, count(p) AS pages,
             sum(CASE WHEN p.needs_ocr = true THEN 1 ELSE 0 END) AS pending
        RETURN d.id AS doc_id, d.title AS title, d.format AS format,
               pages, pending
        ORDER BY pending DESC, d.title ASC
        """,
        params=params,
    ))
    documents: list[dict[str, Any]] = []
    total_pages = 0
    total_pending = 0
    for r in rows:
        pages = int(r.get("pages") or 0)
        pending = int(r.get("pending") or 0)
        documents.append({
            "doc_id": r["doc_id"],
            "title": r.get("title"),
            "format": r.get("format"),
            "pages": pages,
            "ready": pages - pending,
            "pending": pending,
            "pending_fraction": (pending / pages) if pages else 0.0,
        })
        total_pages += pages
        total_pending += pending
    return {
        "total_pages": total_pages,
        "ready_pages": total_pages - total_pending,
        "pending_pages": total_pending,
        "documents_total": len(documents),
        "documents_with_pending": sum(1 for d in documents if d["pending"] > 0),
        "documents": documents,
    }


def list_pending_ocr(
    store: Store,
    *,
    doc_id: str | None = None,
    limit: int = 20,
    include_images: bool = True,
    dpi: int = 200,
) -> list[dict[str, Any]]:
    where = ["p.needs_ocr = true"]
    params: dict[str, Any] = {}
    if doc_id:
        where.append("p.doc_id = $doc_id")
        params["doc_id"] = doc_id
    rows = _df_dicts(store.cypher(
        f"MATCH (d:Document)-[:HAS_PAGE]->(p:Page) WHERE {' AND '.join(where)} "
        "RETURN p.id AS page_id, p.doc_id AS doc_id, p.page_number AS page_number, "
        "d.path AS doc_path, d.title AS doc_title "
        f"ORDER BY p.doc_id, p.page_number LIMIT {int(limit)}",
        params=params,
    ))
    if include_images:
        for r in rows:
            doc_path = r.get("doc_path") or ""
            if not doc_path or not Path(doc_path).exists():
                r["image_b64"] = ""
                r["image_error"] = (
                    f"source file missing for doc {r['doc_id'][:18]}…: "
                    f"{doc_path or '<no path recorded>'}"
                )
                continue
            try:
                png = render_page_png(doc_path, int(r["page_number"]), dpi=dpi)
                r["image_b64"] = base64.b64encode(png).decode("ascii")
                r["image_mime"] = "image/png"
            except Exception as exc:  # pragma: no cover
                r["image_b64"] = ""
                r["image_error"] = str(exc)
    return rows


def submit_ocr(
    store: Store,
    embedder: Any,
    *,
    page_id: str,
    markdown: str,
    agent_id: str,
    model: str = "",
    confidence: float | None = None,
) -> dict[str, Any]:
    """Patch OCR-derived markdown back into the graph.

    - Marks the Page as having text, `needs_ocr=False`.
    - Deletes the placeholder Chunk(s) on that page (status=needs_ocr).
    - Re-chunks the supplied markdown into fresh Chunks (status=ready),
      wires HAS_CHUNK + NEXT_CHUNK locally, and writes embeddings.
    """
    register_agent(store, agent_id=agent_id)
    markdown = markdown.strip()
    page_rows = _df_dicts(store.cypher(
        "MATCH (p:Page {id: $id}) RETURN p.doc_id AS doc_id, p.page_number AS page_number",
        params={"id": page_id},
    ))
    if not page_rows:
        raise InvalidEnumError(f"page not found: {page_id}")
    doc_id = page_rows[0]["doc_id"]
    page_number = int(page_rows[0]["page_number"])

    # Delete existing needs_ocr chunks on this page (label predicate
    # under kglite 0.10.5 multi-label: `(:Chunk:NeedsOcr)`)
    store.cypher(
        "MATCH (p:Page {id: $pid})-[:HAS_CHUNK]->(c:Chunk:NeedsOcr) "
        "DETACH DELETE c",
        params={"pid": page_id},
    )
    # Update page state
    store.cypher(
        "MATCH (p:Page {id: $pid}) "
        "SET p.markdown = $md, p.has_text = true, p.needs_ocr = false, "
        "p.ocr_agent = $aid, p.ocr_model = $m, p.ocr_confidence = $conf",
        params={
            "pid": page_id, "md": markdown, "aid": agent_id,
            "m": model, "conf": confidence if confidence is not None else -1.0,
        },
    )

    if not markdown:
        return {"page_id": page_id, "chunks_added": 0}

    # Chunk + insert
    chunks = chunk_page(markdown)
    chunk_rows: list[dict[str, Any]] = []
    for ch in chunks:
        cid = f"{doc_id}#p{page_number}#c{ch.chunk_index}"
        chunk_rows.append({
            "id": cid,
            "title": (ch.text[:80] + "…") if ch.text else f"ocr p.{page_number}",
            "doc_id": doc_id,
            "page_number": page_number,
            "page_id": page_id,
            "chunk_index": ch.chunk_index,
            CHUNK_TEXT_COL: ch.text,
            "token_count": ch.token_count,
            "headings_json": "[]",
            "status": CHUNK_STATUS_READY,
            "text_hash": ch.text_hash_value,
            "view_count": 0,
            "last_viewed_at": "",
        })
    if not chunk_rows:
        return {"page_id": page_id, "chunks_added": 0}

    store.upsert_nodes(CHUNK, chunk_rows)
    # New chunks are all ready
    ready_label = label_for("chunk.status", CHUNK_STATUS_READY)
    if ready_label:
        store.add_label(CHUNK, [r["id"] for r in chunk_rows], ready_label)
    store.upsert_edges(
        HAS_CHUNK, [{"src": page_id, "dst": r["id"]} for r in chunk_rows],
        source_type=PAGE, target_type=CHUNK,
    )
    store.upsert_edges(
        HAS_CHUNK, [{"src": doc_id, "dst": r["id"]} for r in chunk_rows],
        source_type="Document", target_type=CHUNK,
    )
    # Wire NEXT_CHUNK within this page
    if len(chunk_rows) > 1:
        store.upsert_edges(
            NEXT_CHUNK,
            [{"src": chunk_rows[i]["id"], "dst": chunk_rows[i + 1]["id"]}
             for i in range(len(chunk_rows) - 1)],
            source_type=CHUNK, target_type=CHUNK,
        )

    # Embed
    vecs = embedder.embed([r[CHUNK_TEXT_COL] for r in chunk_rows])
    store.add_embeddings(
        CHUNK, CHUNK_TEXT_COL,
        {r["id"]: vecs[i] for i, r in enumerate(chunk_rows)},
    )
    return {
        "page_id": page_id, "chunks_added": len(chunk_rows),
        "agent_id": agent_id, "model": model,
    }
