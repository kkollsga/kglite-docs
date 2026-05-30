"""Scanned-page detection + agent-driven OCR submission.

`list_pending_ocr` returns pages where text extraction came up empty
but the page carries images — typically scanned PDFs. Each entry
includes a base64-encoded PNG render so the agent can read the page
visually and return its markdown via `submit_ocr`.
"""

from __future__ import annotations

import base64
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kglite_docs.activity import register_agent
from kglite_docs.errors import InvalidEnumError
from kglite_docs.ingest.chunker import chunk_page
from kglite_docs.ingest.parser import (
    OCR_TEXT_THRESHOLD,
    _extractable_alnum,
    render_page_images,
    render_page_png,
)
from kglite_docs.schema import (
    CHUNK,
    CHUNK_STATUS_READY,
    CHUNK_TEXT_COL,
    HAS_CHUNK,
    LABEL_EMBEDDED,
    NEXT_CHUNK,
    PAGE,
    label_for,
)
from kglite_docs.store import Store
from kglite_docs.store import rows as _df_dicts  # noqa: E402

#: The verbatim-transcription instruction handed to the agent that performs OCR.
#: kglite-docs is agent-first and ships no OCR engine — the vision-capable agent
#: is the engine. Discipline matters: this text will be *quoted* in a brief, so
#: smoothing/inventing is a defect, not a feature.
OCR_PROMPT = (
    "Transcribe this page image VERBATIM. Reproduce the text exactly as written "
    "— same wording, numbers, names, punctuation, and line order. Do NOT "
    "summarize, correct, translate, modernize, or infer. Mark anything you "
    "cannot read confidently as [illegible]; never guess. Preserve structure as "
    "markdown (headings, lists, tables) where it is visually unambiguous. Return "
    "only the transcription. The result will be quoted as primary evidence."
)


#: Model tier guidance for OCR (A/B-tested on legal scans). We bundle no model —
#: the agent is the engine — but the right tier matters: small models don't just
#: do *worse* OCR, they **contaminate** (corrupt names; fabricate whole documents).
RECOMMENDED_OCR_MODEL = "claude-sonnet-4-6"
MODEL_GUIDANCE = (
    "Use Sonnet (default) for OCR — faithful verbatim + honest [illegible]. "
    "Escalate the hardest / decisive pages to Opus (via force re-OCR). Do NOT use "
    "small models (e.g. Haiku) for legal/forensic OCR: they corrupt names and can "
    "fabricate entire documents — contamination, not merely low quality."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# OCR outcome — *did the transcription actually yield readable text?* A page can
# be OCR'd (attempted) and still be noise: an honest agent returns "[ilegível]" /
# "[illegible]" for an unreadable scan. That honesty must surface at the page +
# coverage level, never sink inside chunk text — otherwise "OCR'd" reads as
# "readable" and a quarter of a record can be noise behind a green light.
OCR_OK: str = "ocr_ok"
OCR_PARTIAL: str = "ocr_partial"
OCR_ILLEGIBLE: str = "ocr_illegible"

_BRACKET_RE = re.compile(r"\[[^\]]*\]")  # [ilegível] / [illegible] / [página ilegível] / …


def _legible_chars(markdown: str) -> int:
    """Count genuinely-readable alphanumeric chars in an OCR transcription —
    after removing bracketed illegibility markers (any `[…]`) and the image /
    placeholder markup `_extractable_alnum` already strips. A page that is all
    `[ilegível]` scores 0."""
    return _extractable_alnum(_BRACKET_RE.sub(" ", markdown or ""))


def _ocr_outcome(legible: int) -> str:
    """`ocr_illegible` (no readable letters) | `ocr_partial` (< the text floor) |
    `ocr_ok`. Reuses the same `OCR_TEXT_THRESHOLD` floor detection uses."""
    if legible <= 0:
        return OCR_ILLEGIBLE
    if legible < OCR_TEXT_THRESHOLD:
        return OCR_PARTIAL
    return OCR_OK


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
             sum(CASE WHEN p.needs_ocr = true THEN 1 ELSE 0 END) AS pending,
             sum(CASE WHEN p.ocr_outcome = 'ocr_illegible' THEN 1 ELSE 0 END) AS illegible,
             sum(CASE WHEN p.ocr_outcome = 'ocr_partial' THEN 1 ELSE 0 END) AS partial
        RETURN d.id AS doc_id, d.title AS title, d.format AS format,
               pages, pending, illegible, partial
        ORDER BY pending DESC, illegible DESC, d.title ASC
        """,
        params=params,
    ))
    documents: list[dict[str, Any]] = []
    total_pages = total_pending = total_illegible = total_partial = 0
    for r in rows:
        pages = int(r.get("pages") or 0)
        pending = int(r.get("pending") or 0)
        illegible = int(r.get("illegible") or 0)
        partial = int(r.get("partial") or 0)
        documents.append({
            "doc_id": r["doc_id"],
            "title": r.get("title"),
            "format": r.get("format"),
            "pages": pages,
            "ready": pages - pending,
            "pending": pending,
            # OCR'd but not actually readable — surfaced, never silently "covered".
            "illegible": illegible,
            "partial": partial,
            "pending_fraction": (pending / pages) if pages else 0.0,
        })
        total_pages += pages
        total_pending += pending
        total_illegible += illegible
        total_partial += partial
    return {
        "total_pages": total_pages,
        "ready_pages": total_pages - total_pending,
        "pending_pages": total_pending,
        # Pages OCR'd but illegible/partial are *attempted* yet effectively
        # unreadable — readable_pages discounts them so "done" can't hide noise.
        "illegible_pages": total_illegible,
        "partial_pages": total_partial,
        "readable_pages": total_pages - total_pending - total_illegible - total_partial,
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


def export_ocr(store: Store, *, doc_id: str, out_path: str | None = None) -> dict[str, Any]:
    """Write a document's OCR to a **sidecar JSON** (`<source>.ocr.json` next to
    the PDF, unless `out_path` is given). Makes OCR portable, auditable,
    diffable, hand-correctable, and re-importable — done once, travels with the
    file. Carries each page's `ocr_status`/`legible_chars` (the illegibility flag)
    so downstream tools never mistake noise for text."""
    doc = _df_dicts(store.cypher(
        "MATCH (d:Document {id: $id}) RETURN d.path AS path, d.title AS title",
        params={"id": doc_id},
    ))
    if not doc:
        raise InvalidEnumError(f"document not found: {doc_id}")
    doc_path = doc[0].get("path") or ""
    pages = _df_dicts(store.cypher(
        "MATCH (d:Document {id: $id})-[:HAS_PAGE]->(p:Page) "
        "WHERE p.ocr_outcome IS NOT NULL "
        "RETURN p.page_number AS page_number, p.ocr_model AS ocr_model, "
        "p.ocr_agent AS ocr_agent, p.ocr_confidence AS ocr_confidence, "
        "p.ocr_outcome AS ocr_status, p.legible_chars AS legible_chars, "
        "p.markdown AS text ORDER BY p.page_number",
        params={"id": doc_id},
    ))
    if out_path is None:
        if not doc_path:
            raise InvalidEnumError("no source path recorded — pass out_path explicitly")
        p = Path(doc_path)
        out_path = str(p.with_name(p.stem + ".ocr.json"))
    source_file = Path(doc_path).name if doc_path else (doc[0].get("title") or doc_id)
    by_model = dict(Counter(str(pg.get("ocr_model") or "") for pg in pages if pg.get("ocr_model")))
    payload = {
        "source_file": source_file, "doc_id": doc_id, "generated_at": _now(),
        "by_model": by_model, "page_count": len(pages), "pages": pages,
    }
    Path(out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"out_path": out_path, "pages": len(pages), "by_model": by_model}


def import_ocr(store: Store, embedder: Any, *, path: str) -> dict[str, Any]:
    """Round-trip a sidecar JSON (`export_ocr`) back into the graph: apply each
    page's transcription via `submit_ocr` (re-chunk + embed + recompute the
    legibility outcome). The document must already be ingested (matched by
    `doc_id`); a human can fix the failed pages in the JSON and re-import without
    re-OCR. Pages not present in the corpus are skipped (reported)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    doc_id = data.get("doc_id") or ""
    if not _df_dicts(store.cypher(
        "MATCH (d:Document {id: $id}) RETURN d.id AS id", params={"id": doc_id},
    )):
        raise InvalidEnumError(
            f"document {doc_id!r} not in this corpus — ingest the source PDF first, "
            "then import its OCR sidecar."
        )
    imported = skipped = 0
    for rec in data.get("pages", []):
        prow = _df_dicts(store.cypher(
            "MATCH (p:Page {doc_id: $d, page_number: $pn}) RETURN p.id AS id",
            params={"d": doc_id, "pn": int(rec.get("page_number", -1))},
        ))
        if not prow:
            skipped += 1
            continue
        submit_ocr(
            store, embedder, page_id=prow[0]["id"], markdown=str(rec.get("text") or ""),
            agent_id=str(rec.get("ocr_agent") or "import"), model=str(rec.get("ocr_model") or ""),
            confidence=rec.get("ocr_confidence"),
        )
        imported += 1
    return {"doc_id": doc_id, "pages_imported": imported, "pages_skipped": skipped}


def list_illegible_pages(
    store: Store,
    *,
    doc_id: str | None = None,
    limit: int = 50,
    include_images: bool = False,
    dpi: int = 200,
) -> list[dict[str, Any]]:
    """Pages that were OCR'd but came back **illegible or partial** (effectively
    unreadable) — the worklist for human review or a stronger-model retry
    (`request_ocr(..., force=True)`). Without this they'd silently count as
    covered. Optional `include_images` renders each page for re-OCR."""
    where = ["p.ocr_outcome IN ['ocr_illegible', 'ocr_partial']"]
    params: dict[str, Any] = {}
    if doc_id:
        where.append("p.doc_id = $doc_id")
        params["doc_id"] = doc_id
    rows = _df_dicts(store.cypher(
        f"MATCH (d:Document)-[:HAS_PAGE]->(p:Page) WHERE {' AND '.join(where)} "
        "RETURN p.id AS page_id, p.doc_id AS doc_id, p.page_number AS page_number, "
        "p.ocr_outcome AS ocr_outcome, p.legible_chars AS legible_chars, "
        "p.ocr_model AS ocr_model, d.path AS doc_path, d.title AS doc_title "
        f"ORDER BY p.legible_chars ASC, p.doc_id, p.page_number LIMIT {int(limit)}",
        params=params,
    ))
    if include_images:
        for r in rows:
            doc_path = r.get("doc_path") or ""
            if not doc_path or not Path(doc_path).exists():
                r["image_b64"] = ""
                r["image_error"] = f"source file missing: {doc_path or '<no path recorded>'}"
                continue
            try:
                png = render_page_png(doc_path, int(r["page_number"]), dpi=dpi)
                r["image_b64"] = base64.b64encode(png).decode("ascii")
                r["image_mime"] = "image/png"
            except Exception as exc:  # pragma: no cover
                r["image_b64"] = ""
                r["image_error"] = str(exc)
    return rows


def request_ocr(
    store: Store,
    *,
    page_id: str | None = None,
    doc_id: str | None = None,
    page_number: int | None = None,
    agent_id: str,
    agent_type: str = "",
    dpi: int = 200,
    force: bool = False,
) -> dict[str, Any]:
    """Lazy, agent-driven OCR. The first time an agent asks for a `needs_ocr`
    page, hand it the OCR **task** to perform itself — the rendered page (base64
    PNG) plus the verbatim `OCR_PROMPT` — instead of serving empty/junk text.
    The agent transcribes and calls `submit_ocr`.

    Identify the page by `page_id`, or by `doc_id` + `page_number`. Raises if the
    page isn't found, isn't flagged `needs_ocr` (nothing to OCR), or its source
    file is missing. Pass `force=True` to **re-OCR an already-transcribed page**
    — e.g. to escalate an illegible/partial result to a stronger model
    (`ocr("illegible")` → re-`request_ocr(force=True)` → re-`submit_ocr`); the new
    submit replaces the page's chunks. `agent_type` is echoed (and recorded) so an
    orchestrator can route the task to a specific OCR subagent — the library never
    dispatches it (agent-first). The request is recorded on the page
    (who/when/agent_type; first request preserved) for an audit trail."""
    params: dict[str, Any]
    if page_id:
        where, params = "p.id = $pid", {"pid": page_id}
    elif doc_id and page_number is not None:
        where, params = "p.doc_id = $doc AND p.page_number = $pn", {"doc": doc_id, "pn": int(page_number)}
    else:
        raise InvalidEnumError("request_ocr needs page_id, or doc_id + page_number")
    rows = _df_dicts(store.cypher(
        f"MATCH (d:Document)-[:HAS_PAGE]->(p:Page) WHERE {where} "
        "RETURN p.id AS page_id, p.doc_id AS doc_id, p.page_number AS page_number, "
        "p.needs_ocr AS needs_ocr, p.ocr_requested_at AS prior, d.path AS doc_path",
        params=params,
    ))
    if not rows:
        raise InvalidEnumError(f"page not found: {page_id or f'{doc_id} p.{page_number}'}")
    r = rows[0]
    if not r.get("needs_ocr") and not force:
        raise InvalidEnumError(
            f"page {r['page_id']} is not flagged needs_ocr — nothing to OCR "
            "(it already has extractable text). Pass force=True to re-OCR it "
            "anyway (e.g. to escalate an illegible result to a stronger model)."
        )
    doc_path = r.get("doc_path") or ""
    if not doc_path or not Path(doc_path).exists():
        raise InvalidEnumError(
            f"source file missing for page {r['page_id']}: {doc_path or '<no path recorded>'} "
            "— cannot render the page to OCR."
        )
    register_agent(store, agent_id=agent_id)
    pno = int(r["page_number"])
    # Right-size the image to the model's input: one full-page tile if it fits,
    # else detail-preserving overlapping tiles (don't ship a blur the model
    # would downscale anyway).
    tiles = render_page_images(doc_path, pno, dpi=dpi)
    prior = r.get("prior") or ""
    requested_at = prior or _now()
    store.cypher(
        "MATCH (p:Page {id: $pid}) SET p.ocr_requested_by = $aid, "
        "p.ocr_agent_type = $at, p.ocr_requested_at = $ra, "
        "p.ocr_render_dpi = $dpi, p.ocr_tiles = $nt",
        params={"pid": r["page_id"], "aid": agent_id, "at": agent_type,
                "ra": requested_at, "dpi": int(dpi), "nt": len(tiles)},
    )
    out: dict[str, Any] = {
        "page_id": r["page_id"], "doc_id": r["doc_id"], "page_number": pno,
        "prompt": OCR_PROMPT, "agent_type": agent_type,
        "tiles": tiles, "tile_count": len(tiles),
        "recommended_model": RECOMMENDED_OCR_MODEL, "model_guidance": MODEL_GUIDANCE,
        "already_requested": bool(prior),
    }
    if len(tiles) == 1:
        # Backward-compatible single-image shape; transcribe → submit markdown.
        out["image_b64"] = tiles[0]["image_b64"]
        out["image_mime"] = tiles[0]["image_mime"]
        out["submit_with"] = "ocr('submit', page_id=…, markdown=<transcription>, agent_id=…)"
    else:
        # Transcribe each tile (top→bottom, overlapping); submit them to stitch.
        out["submit_with"] = (
            "ocr('submit', page_id=…, agent_id=…, "
            "tiles=[{tile_index, markdown}, …])  # library stitches in order"
        )
    return out


def submit_ocr(
    store: Store,
    embedder: Any,
    *,
    page_id: str,
    markdown: str = "",
    agent_id: str,
    model: str = "",
    confidence: float | None = None,
    tiles: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Patch OCR-derived markdown back into the graph.

    - Pass whole-page `markdown`, **or** `tiles=[{tile_index, markdown}, …]` from a
      tiled `request_ocr` — the library stitches them in `tile_index` order.
    - Marks the Page `needs_ocr=False` + records its legibility `ocr_outcome`.
    - Replaces all Chunk(s) on the page with fresh `ready` Chunks (HAS_CHUNK +
      NEXT_CHUNK), and writes embeddings.
    """
    register_agent(store, agent_id=agent_id)
    if tiles:
        ordered = sorted(tiles, key=lambda t: int(t.get("tile_index", 0)))
        markdown = "\n\n".join(
            str(t.get("markdown", "")).strip() for t in ordered if str(t.get("markdown", "")).strip()
        )
    markdown = markdown.strip()
    page_rows = _df_dicts(store.cypher(
        "MATCH (p:Page {id: $id}) RETURN p.doc_id AS doc_id, p.page_number AS page_number",
        params={"id": page_id},
    ))
    if not page_rows:
        raise InvalidEnumError(f"page not found: {page_id}")
    doc_id = page_rows[0]["doc_id"]
    page_number = int(page_rows[0]["page_number"])

    # Legibility outcome — did this transcription actually yield readable text?
    legible = _legible_chars(markdown)
    outcome = _ocr_outcome(legible)

    # Replace ALL chunks on this page (not just :NeedsOcr) so a re-OCR overwrites
    # cleanly instead of leaving the prior :Ready chunks beside the new ones.
    store.cypher(
        "MATCH (p:Page {id: $pid})-[:HAS_CHUNK]->(c:Chunk) DETACH DELETE c",
        params={"pid": page_id},
    )
    # Update page state. needs_ocr=false (it WAS attempted); ocr_outcome carries
    # the honest "is it actually readable?" signal.
    store.cypher(
        "MATCH (p:Page {id: $pid}) "
        "SET p.markdown = $md, p.has_text = true, p.needs_ocr = false, "
        "p.ocr_agent = $aid, p.ocr_model = $m, p.ocr_confidence = $conf, "
        "p.ocr_outcome = $outcome, p.legible_chars = $legible",
        params={
            "pid": page_id, "md": markdown, "aid": agent_id,
            "m": model, "conf": confidence if confidence is not None else -1.0,
            "outcome": outcome, "legible": legible,
        },
    )

    if not markdown:
        return {"page_id": page_id, "chunks_added": 0,
                "ocr_outcome": outcome, "legible_chars": legible}

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
            # Embedded inline below — a first-class searchable chunk, same as a
            # native-text chunk (so search/coverage don't treat it as unembedded).
            "embedded": True,
            # Honesty marker: this text is OCR-derived, not native extraction —
            # a reviewer should eyeball the page image before quoting it.
            "ocr_derived": True,
            "ocr_model": model,
            "ocr_by": agent_id,
        })
    if not chunk_rows:
        return {"page_id": page_id, "chunks_added": 0,
                "ocr_outcome": outcome, "legible_chars": legible}

    store.upsert_nodes(CHUNK, chunk_rows)
    # New chunks are ready AND (after embedding below) embedded — label both so
    # they're first-class searchable chunks, not counted as unembedded.
    chunk_ids = [r["id"] for r in chunk_rows]
    ready_label = label_for("chunk.status", CHUNK_STATUS_READY)
    if ready_label:
        store.add_label(CHUNK, chunk_ids, ready_label)
    store.add_label(CHUNK, chunk_ids, LABEL_EMBEDDED)
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
        "ocr_outcome": outcome, "legible_chars": legible,
    }


def submit_ocr_many(
    store: Store, embedder: Any, *, rows: list[dict[str, Any]],
    agent_id: str = "ocr", model: str = "",
) -> dict[str, Any]:
    """Submit many pages' OCR in one call. Each row is
    `{page_id, markdown}` or `{page_id, tiles:[…]}` (+ optional per-row
    `agent_id`/`model`/`confidence`). `rows` is a **structured argument** — the
    MCP/SDK layer escapes it — so an agent never hand-serializes multi-line,
    quote-heavy verbatim transcriptions into a JSON file (which silently corrupts
    on the very content this tool handles). A failing row is reported, not fatal."""
    results: list[dict[str, Any]] = []
    for rec in rows:
        pid = rec.get("page_id")
        if not pid:
            results.append({"error": "missing page_id", "record": rec})
            continue
        try:
            results.append(submit_ocr(
                store, embedder, page_id=str(pid),
                markdown=str(rec.get("markdown") or ""),
                agent_id=str(rec.get("agent_id") or agent_id),
                model=str(rec.get("model") or model),
                confidence=rec.get("confidence"),
                tiles=rec.get("tiles"),
            ))
        except Exception as exc:  # one bad row mustn't sink the batch
            results.append({"page_id": pid, "error": str(exc)})
    return {
        "submitted": sum(1 for r in results if "error" not in r),
        "failed": sum(1 for r in results if "error" in r),
        "results": results,
    }
