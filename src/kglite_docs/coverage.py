"""Honest-coverage reporting.

`coverage_report` and `corpus_status` make every coverage-reducing fact
observable: how many pages are image-only / low-text (unanalyzed unless OCR'd),
and how many chunks are unembedded (so `search` is blind until `index()`).
The north star is that a user never trusts a green light while half the record
is invisible — see `ROADMAP.md`.
"""

from __future__ import annotations

from typing import Any

from kglite_docs.ingest.parser import OCR_TEXT_THRESHOLD
from kglite_docs.store import Store
from kglite_docs.store import rows as _df_dicts


def coverage_report(store: Store, *, doc_id: str | None = None) -> dict[str, Any]:
    """Per-document + corpus-wide extraction/embedding coverage.

    Each document reports `pages`, `pending_ocr`, `image_pages`,
    `low_text_pages` (< OCR_TEXT_THRESHOLD extractable chars), and
    `extractable_text_ratio` (0.0 = entirely image/low-text, 1.0 = fully
    extractable). Corpus totals add `unembedded`/`embedded` chunk counts and a
    one-line human `summary`. Documents are ordered worst-coverage first. Pass
    `doc_id` to scope the per-doc rows; corpus totals still span the graph.
    """
    where = "WHERE d.id = $doc_id" if doc_id else ""
    params: dict[str, Any] = {"thr": OCR_TEXT_THRESHOLD}
    if doc_id:
        params["doc_id"] = doc_id
    rows = _df_dicts(store.cypher(
        f"""
        MATCH (d:Document)
        {where}
        OPTIONAL MATCH (d)-[:HAS_PAGE]->(p:Page)
        WITH d,
             count(p) AS pages,
             sum(CASE WHEN p.needs_ocr = true THEN 1 ELSE 0 END) AS pending_ocr,
             sum(CASE WHEN coalesce(p.image_block_count, 0) > 0 THEN 1 ELSE 0 END) AS image_pages,
             sum(CASE WHEN coalesce(p.extractable_alnum, 0) < $thr THEN 1 ELSE 0 END) AS low_text_pages
        RETURN d.id AS doc_id, d.title AS title, d.format AS format,
               pages, pending_ocr, image_pages, low_text_pages
        ORDER BY low_text_pages DESC, pending_ocr DESC, d.title ASC
        """,
        params=params,
    ))
    documents: list[dict[str, Any]] = []
    tot_pages = tot_pending = tot_image = tot_low = 0
    for r in rows:
        pages = int(r.get("pages") or 0)
        pending = int(r.get("pending_ocr") or 0)
        image_pages = int(r.get("image_pages") or 0)
        low = int(r.get("low_text_pages") or 0)
        documents.append({
            "doc_id": r["doc_id"],
            "title": r.get("title"),
            "format": r.get("format"),
            "pages": pages,
            "pending_ocr": pending,
            "image_pages": image_pages,
            "low_text_pages": low,
            "extractable_text_ratio": ((pages - low) / pages) if pages else 1.0,
        })
        tot_pages += pages
        tot_pending += pending
        tot_image += image_pages
        tot_low += low

    embedded = _count(store, "MATCH (c:Chunk:Embedded) RETURN count(c) AS n")
    unembedded = _count(
        store, "MATCH (c:Chunk:Ready) WHERE c.embedded = false RETURN count(c) AS n"
    )

    summary = (
        f"{len(documents)} docs / {tot_pages} pages — "
        f"{tot_image} image-only and {tot_low} low-text page(s) are unanalyzed "
        f"unless OCR'd ({tot_pending} flagged needs_ocr); "
        f"{unembedded} chunk(s) unembedded (search is blind until index())."
    )
    return {
        "documents": documents,
        "total_pages": tot_pages,
        "image_pages": tot_image,
        "low_text_pages": tot_low,
        "pending_ocr": tot_pending,
        "embedded": embedded,
        "unembedded": unembedded,
        "summary": summary,
    }


def corpus_status(store: Store) -> dict[str, Any]:
    """One-call snapshot of what's in the corpus and what's unread/unindexed —
    the first thing an agent should check."""
    return {
        "docs": _count(store, "MATCH (d:Document) RETURN count(d) AS n"),
        "pages": _count(store, "MATCH (p:Page) RETURN count(p) AS n"),
        "chunks": _count(store, "MATCH (c:Chunk) RETURN count(c) AS n"),
        "embedded": _count(store, "MATCH (c:Chunk:Embedded) RETURN count(c) AS n"),
        "unembedded": _count(
            store, "MATCH (c:Chunk:Ready) WHERE c.embedded = false RETURN count(c) AS n"
        ),
        "image_pages": _count(
            store, "MATCH (p:Page) WHERE coalesce(p.image_block_count, 0) > 0 RETURN count(p) AS n"
        ),
        "pending_ocr": _count(
            store, "MATCH (p:Page) WHERE p.needs_ocr = true RETURN count(p) AS n"
        ),
        "studies": _count(store, "MATCH (s:Study) RETURN count(s) AS n"),
    }


def _count(store: Store, query: str) -> int:
    r = _df_dicts(store.cypher(query))
    return int(r[0]["n"]) if r else 0
