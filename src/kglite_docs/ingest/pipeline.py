"""Orchestrate parse → chunk → embed → graph-write for a single PDF.

`ingest_pdf(store, embedder, path)` is the entry the Corpus uses. It is
idempotent: if a Document with the same file hash already exists, returns
early with `created=False`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from kglite_docs.ingest.chunker import Chunk, chunk_page
from kglite_docs.ingest.formats import detect_format, parse_document
from kglite_docs.ingest.hashing import file_hash
from kglite_docs.ingest.parser import PageContent
from kglite_docs.schema import (
    CHUNK,
    CHUNK_STATUS_EMPTY,
    CHUNK_STATUS_NEEDS_OCR,
    CHUNK_STATUS_READY,
    CHUNK_TEXT_COL,
    DOCUMENT,
    HAS_CHUNK,
    HAS_PAGE,
    LABEL_EMBEDDED,
    NEXT_CHUNK,
    PAGE,
    label_for,
)
from kglite_docs.store import Store
from kglite_docs.store import rows as _rows


class EmbedderLike(Protocol):
    """Subset of the kglite EmbeddingModel protocol we use."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class IngestResult:
    doc_id: str
    created: bool  # False when the document was already present
    page_count: int
    chunk_count: int
    ocr_pending_pages: int
    format: str = ""
    embedded: int = 0  # chunks embedded during this ingest (0 unless embed=True)


def ingest_pdf(
    store: Store,
    embedder: EmbedderLike,
    path: str | Path,
    *,
    source_uri: str | None = None,
    title: str | None = None,
    metadata: dict[str, object] | None = None,
    embed: bool = False,
) -> IngestResult:
    """Back-compat alias; new code should prefer `ingest_document`."""
    return ingest_document(
        store, embedder, path,
        source_uri=source_uri, title=title, metadata=metadata, embed=embed,
    )


_MIME_BY_FORMAT = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "html": "text/html",
    "htm": "text/html",
    "md": "text/markdown",
    "markdown": "text/markdown",
    "txt": "text/plain",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "bmp": "image/bmp",
}


def ingest_document(
    store: Store,
    embedder: EmbedderLike,
    path: str | Path,
    *,
    source_uri: str | None = None,
    title: str | None = None,
    metadata: dict[str, object] | None = None,
    format: str | None = None,
    embed: bool = False,
) -> IngestResult:
    """Ingest any supported document format (PDF, DOCX, PPTX, MD, HTML,
    TXT, images). Format is auto-detected from the extension; pass
    `format=` to override.

    Embedding is **opt-in** (`embed=False` by default): ingest parses,
    chunks, and writes the graph without touching the embedding model, so
    it never blocks on model load / embedding compute and non-semantic
    workflows pay nothing. Ready chunks are marked ``:Unembedded``; run
    :func:`Corpus.index` (or pass ``embed=True``) to compute vectors and
    enable ``search``."""
    path = Path(path)
    doc_id = file_hash(path)
    fmt = (format or detect_format(path)).lower()
    title = title or path.stem

    # Idempotency: skip if doc already exists
    existing = store.cypher(
        "MATCH (d:Document {id: $id}) RETURN d.id AS id, d.page_count AS pc",
        params={"id": doc_id},
    )
    existing_rows = _rows(existing)
    if existing_rows:
        row = existing_rows[0]
        return IngestResult(
            doc_id=doc_id, created=False, page_count=int(row.get("pc", 0)),
            chunk_count=_count_chunks(store, doc_id),
            ocr_pending_pages=_count_pending_ocr(store, doc_id),
            format=fmt,
            embedded=_count_embedded(store, doc_id),
        )

    pages = parse_document(path, format=fmt)
    byte_size = path.stat().st_size
    now = datetime.now(timezone.utc).isoformat()

    # Insert Document
    store.upsert_nodes(
        DOCUMENT,
        [{
            "id": doc_id,
            "title": title,
            "path": str(path.resolve()),
            "source_uri": source_uri or "",
            "ingested_at": now,
            "page_count": len(pages),
            "mime": _MIME_BY_FORMAT.get(fmt, "application/octet-stream"),
            "format": fmt,
            "byte_size": byte_size,
            "metadata_json": _safe_json(metadata or {}),
        }],
    )

    # Insert Pages
    page_rows = [
        {
            "id": _page_id(doc_id, p.page_number),
            "title": f"{title} p.{p.page_number}",
            "doc_id": doc_id,
            "page_number": p.page_number,
            "has_text": p.has_text,
            "needs_ocr": p.needs_ocr,
            "markdown": p.markdown,
            "width_pt": p.width_pt,
            "height_pt": p.height_pt,
        }
        for p in pages
    ]
    store.upsert_nodes(PAGE, page_rows)
    store.upsert_edges(
        HAS_PAGE,
        [{"src": doc_id, "dst": r["id"]} for r in page_rows],
        source_type=DOCUMENT, target_type=PAGE,
    )

    # Chunk each page; collect across pages
    all_chunks: list[tuple[str, Chunk, PageContent]] = []
    for p in pages:
        if p.needs_ocr:
            # placeholder chunk so the agent can later see what's pending
            placeholder = Chunk(
                chunk_index=0,
                text="",
                token_count=0,
                headings=[],
            )
            all_chunks.append((CHUNK_STATUS_NEEDS_OCR, placeholder, p))
            continue
        if not p.markdown.strip():
            placeholder = Chunk(chunk_index=0, text="", token_count=0, headings=[])
            all_chunks.append((CHUNK_STATUS_EMPTY, placeholder, p))
            continue
        for ch in chunk_page(p.markdown):
            all_chunks.append((CHUNK_STATUS_READY, ch, p))

    # Insert Chunks
    chunk_rows: list[dict[str, object]] = []
    chunk_ids_by_page: dict[int, list[str]] = {}
    # Group chunk ids by status so we can add labels in one batch per kind.
    chunk_ids_by_status: dict[str, list[str]] = {}
    for status, ch, p in all_chunks:
        cid = _chunk_id(doc_id, p.page_number, ch.chunk_index)
        chunk_rows.append({
            "id": cid,
            "title": (ch.text[:80] + "…") if ch.text else f"[needs ocr] p.{p.page_number}",
            "doc_id": doc_id,
            "page_number": p.page_number,
            "page_id": _page_id(doc_id, p.page_number),
            "chunk_index": ch.chunk_index,
            CHUNK_TEXT_COL: ch.text,
            "token_count": ch.token_count,
            "headings_json": _safe_json(ch.headings),
            "status": status,                   # property still written
            "embedded": False,                  # flipped by index() / embed=True
            "text_hash": ch.text_hash_value if ch.text else "",
            "view_count": 0,
            "last_viewed_at": "",
        })
        chunk_ids_by_page.setdefault(p.page_number, []).append(cid)
        chunk_ids_by_status.setdefault(status, []).append(cid)
    store.upsert_nodes(CHUNK, chunk_rows)

    # Multi-label (kglite 0.10.5): tag each chunk with its status label
    # in batches per status — `MATCH (c:Chunk:Ready)` is now O(label-index).
    for status, ids in chunk_ids_by_status.items():
        status_label = label_for("chunk.status", status)
        if status_label:
            store.add_label(CHUNK, ids, status_label)

    # Embedding lifecycle is tracked by the boolean `c.embedded` property
    # (set False above). The work-list is "ready chunks where embedded =
    # false"; index() adds the additive :Embedded label as it goes. We do
    # NOT use a removable :Unembedded label — kglite's remove_label leaves
    # the label-predicate index stale (labels(c) updates, MATCH (:Label)
    # does not), so a removed-label MATCH would over-report.

    # Edges: Page→Chunk, Document→Chunk, and NEXT_CHUNK in reading order
    page_chunk_edges = [
        {"src": _page_id(doc_id, p.page_number), "dst": cid}
        for status, ch, p in [(s, c, pg) for (s, c, pg) in all_chunks]
        for cid in [_chunk_id(doc_id, p.page_number, ch.chunk_index)]
    ]
    store.upsert_edges(HAS_CHUNK, page_chunk_edges, source_type=PAGE, target_type=CHUNK)
    doc_chunk_edges = [{"src": doc_id, "dst": r["id"]} for r in chunk_rows]
    store.upsert_edges(HAS_CHUNK, doc_chunk_edges, source_type=DOCUMENT, target_type=CHUNK)

    # NEXT_CHUNK in document reading order across pages
    ordered_ids = [r["id"] for r in chunk_rows]
    if len(ordered_ids) > 1:
        next_edges = [
            {"src": ordered_ids[i], "dst": ordered_ids[i + 1]}
            for i in range(len(ordered_ids) - 1)
        ]
        store.upsert_edges(NEXT_CHUNK, next_edges, source_type=CHUNK, target_type=CHUNK)

    # Optional inline embed. The default (embed=False) leaves chunks
    # :Unembedded for a later index() pass — keeping ingest off the model
    # entirely. embed=True is the one-shot convenience path.
    embedded_count = 0
    if embed:
        embeddable = [
            (r["id"], r[CHUNK_TEXT_COL]) for r in chunk_rows
            if r["status"] == CHUNK_STATUS_READY and r[CHUNK_TEXT_COL]
        ]
        if embeddable:
            emb_ids, texts = zip(*embeddable, strict=False)
            vecs = embedder.embed(list(texts))
            store.add_embeddings(CHUNK, CHUNK_TEXT_COL, dict(zip(emb_ids, vecs, strict=False)))
            store.cypher(
                "MATCH (c:Chunk) WHERE c.id IN $ids SET c.embedded = true",
                params={"ids": list(emb_ids)},
            )
            store.add_label(CHUNK, list(emb_ids), LABEL_EMBEDDED)
            embedded_count = len(emb_ids)

    return IngestResult(
        doc_id=doc_id,
        created=True,
        page_count=len(pages),
        chunk_count=len(chunk_rows),
        ocr_pending_pages=sum(1 for p in pages if p.needs_ocr),
        format=fmt,
        embedded=embedded_count,
    )


def _page_id(doc_id: str, page_number: int) -> str:
    return f"{doc_id}#p{page_number}"


def _chunk_id(doc_id: str, page_number: int, chunk_index: int) -> str:
    return f"{doc_id}#p{page_number}#c{chunk_index}"


def _count_chunks(store: Store, doc_id: str) -> int:
    r = store.cypher(
        "MATCH (d:Document {id: $id})-[:HAS_CHUNK]->(c:Chunk) RETURN count(c) AS n",
        params={"id": doc_id},
    )
    rs = _rows(r)
    return int(rs[0]["n"]) if rs else 0


def _count_embedded(store: Store, doc_id: str) -> int:
    r = store.cypher(
        "MATCH (d:Document {id: $id})-[:HAS_CHUNK]->(c:Chunk:Embedded) RETURN count(c) AS n",
        params={"id": doc_id},
    )
    rs = _rows(r)
    return int(rs[0]["n"]) if rs else 0


def _count_pending_ocr(store: Store, doc_id: str) -> int:
    r = store.cypher(
        "MATCH (d:Document {id: $id})-[:HAS_PAGE]->(p:Page) WHERE p.needs_ocr = true RETURN count(p) AS n",
        params={"id": doc_id},
    )
    rs = _rows(r)
    return int(rs[0]["n"]) if rs else 0


def _safe_json(obj: object) -> str:
    import json
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return "{}"
