"""Orchestrate parse → chunk → embed → graph-write for a single PDF.

`ingest_pdf(store, embedder, path)` is the entry the Corpus uses. It is
idempotent: if a Document with the same file hash already exists, returns
early with `created=False`.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Protocol

from kglite_docs.ingest.chunker import Chunk, chunk_page
from kglite_docs.ingest.formats import detect_format, parse_document
from kglite_docs.ingest.hashing import file_hash
from kglite_docs.ingest.parser import PageContent, _extractable_alnum
from kglite_docs.schema import (
    CHUNK,
    CHUNK_STATUS_EMPTY,
    CHUNK_STATUS_NEEDS_OCR,
    CHUNK_STATUS_READY,
    CHUNK_TEXT_COL,
    DOCUMENT,
    HAS_CHUNK,
    HAS_PAGE,
    HAS_SECTION,
    LABEL_EMBEDDED,
    NEXT_CHUNK,
    PAGE,
    SECTION,
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
    section_count: int = 0  # Section nodes derived from outline/headings


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
    structure_aware: bool = False,
    context_summary: str = "",
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
            # FEAT-11: a doc-level summary prepended to each chunk *before*
            # embedding (vector carries global context; stored text stays clean).
            "embed_context": context_summary,
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
            "image_block_count": p.image_block_count,
            "extractable_alnum": _extractable_alnum(p.markdown),
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
        for ch in chunk_page(p.markdown, structure_aware=structure_aware):
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

    # Derive Sections (the middle grain between document and chunk) and stamp
    # each chunk with its section_id/doc_type *before* the upsert.
    outline = pages[0].metadata.get("doc_outline") if pages else None
    section_rows, chunk_section = _derive_sections(doc_id, chunk_rows, outline)
    for r in chunk_rows:
        sid, dtype = chunk_section.get(str(r["id"]), ("", ""))
        r["section_id"] = sid
        r["doc_type"] = dtype

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

    # Sections: Document→Section, Section→Chunk (reusing HAS_CHUNK).
    if section_rows:
        store.upsert_nodes(SECTION, section_rows)
        store.upsert_edges(
            HAS_SECTION, [{"src": doc_id, "dst": s["id"]} for s in section_rows],
            source_type=DOCUMENT, target_type=SECTION,
        )
        sec_chunk_edges = [
            {"src": str(r["section_id"]), "dst": str(r["id"])}
            for r in chunk_rows if r.get("section_id")
        ]
        if sec_chunk_edges:
            store.upsert_edges(
                HAS_CHUNK, sec_chunk_edges, source_type=SECTION, target_type=CHUNK,
            )

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
            embed_inputs = [
                f"{context_summary}\n\n{t}" if context_summary else t for t in texts
            ]
            vecs = embedder.embed(embed_inputs)
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
        section_count=len(section_rows),
    )


def _page_id(doc_id: str, page_number: int) -> str:
    return f"{doc_id}#p{page_number}"


def _chunk_id(doc_id: str, page_number: int, chunk_index: int) -> str:
    return f"{doc_id}#p{page_number}#c{chunk_index}"


def _row_page(r: dict[str, object]) -> int:
    """A chunk row's 1-based page number (0 if missing/non-int)."""
    v = r.get("page_number")
    return v if isinstance(v, int) else 0


def _top_heading(headings_json: object) -> str | None:
    """The top-level (first) heading of a chunk, from its headings_json."""
    if not isinstance(headings_json, str):
        return None
    try:
        hs = json.loads(headings_json)
    except Exception:
        return None
    return str(hs[0]) if isinstance(hs, list) and hs else None


def _normalize_outline(outline: object) -> list[tuple[int, str, int]]:
    """Coerce a `doc.get_toc(simple=True)` result into sorted
    `(level, title, page)` tuples; `[]` if absent/malformed."""
    if not isinstance(outline, list):
        return []
    entries: list[tuple[int, str, int]] = []
    for e in outline:
        if isinstance(e, (list, tuple)) and len(e) >= 3:
            try:
                entries.append((int(e[0]), str(e[1]), int(e[2])))
            except (TypeError, ValueError):
                continue
    entries.sort(key=lambda t: t[2])  # by page
    return entries


def _derive_sections(
    doc_id: str,
    chunk_rows: list[dict[str, object]],
    outline: object,
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, str]]]:
    """Group chunks into Sections — the grain between document and chunk.

    Prefers the PDF outline (`doc.get_toc`) when present; else falls back to
    top-level heading boundaries in the chunk stream. Returns `(section_rows,
    {chunk_id: (section_id, doc_type)})`. Generic + best-effort; `doc_type`
    defaults to `""` in core (verticals classify it). Page ranges are computed
    from the chunks actually assigned, so empty outline sections are dropped.
    """
    if not chunk_rows:
        return [], {}

    entries = _normalize_outline(outline)
    # Per-chunk grouping key + display title + level, in reading order.
    keyed: list[tuple[dict[str, object], Any, str, int]] = []
    if entries:
        starts = [e[2] for e in entries]
        for r in chunk_rows:
            pg = _row_page(r)
            idx = max(0, bisect.bisect_right(starts, pg) - 1)
            level, title, _pg = entries[idx]
            keyed.append((r, ("o", idx), title or "(untitled)", level or 1))
    else:
        cur = -1
        prev_top: object = _SECTION_SENTINEL
        for r in chunk_rows:
            top = _top_heading(r.get("headings_json"))
            if top != prev_top:
                cur += 1
                prev_top = top
            keyed.append((r, ("h", cur), top or "(untitled)", 1))

    order: list[Any] = []
    meta: dict[Any, dict[str, Any]] = {}
    for r, key, title, level in keyed:
        pg = _row_page(r)
        if key not in meta:
            meta[key] = {"title": title, "level": level, "page_start": pg, "page_end": pg}
            order.append(key)
        else:
            m = meta[key]
            if pg:
                m["page_start"] = min(m["page_start"] or pg, pg)
                m["page_end"] = max(m["page_end"], pg)

    key_to_sid: dict[Any, str] = {}
    section_rows: list[dict[str, Any]] = []
    for ordinal, key in enumerate(order):
        sid = f"{doc_id}#s{ordinal}"
        key_to_sid[key] = sid
        m = meta[key]
        section_rows.append({
            "id": sid,
            "doc_id": doc_id,
            "title": m["title"],
            "page_start": m["page_start"],
            "page_end": m["page_end"],
            "level": m["level"],
            "doc_type": "",
            "ordinal": ordinal,
        })
    chunk_section = {str(r["id"]): (key_to_sid[key], "") for r, key, _t, _l in keyed}
    return section_rows, chunk_section


_SECTION_SENTINEL: Final = object()


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
