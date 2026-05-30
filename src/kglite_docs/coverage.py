"""Honest-coverage reporting.

`coverage_report` and `corpus_status` make every coverage-reducing fact
observable: how many pages are image-only / low-text (unanalyzed unless OCR'd),
and how many chunks are unembedded (so `search` is blind until `index()`).
The north star is that a user never trusts a green light while half the record
is invisible — see `ROADMAP.md`.
"""

from __future__ import annotations

import json
from typing import Any

from kglite_docs.errors import InvalidEnumError
from kglite_docs.ingest.parser import OCR_TEXT_THRESHOLD
from kglite_docs.schema import (
    ENTITY_LABELS,
    LABEL_BOILERPLATE,
    LABEL_CLASSIFIED,
    LABEL_CONTESTED,
    LABEL_EMBEDDED,
    LABEL_LOW_QUALITY,
    LABEL_READY,
    LABEL_UNCLASSIFIED,
    element_label,
    valid_element_values,
)
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
        # Element classification state — a non-zero `unclassified` is a standing
        # signal that element-scoped studies have a blind spot.
        "classified": _count(store, "MATCH (c:Chunk:Ready:Classified) RETURN count(c) AS n"),
        "unclassified": _count(store, "MATCH (c:Chunk:Ready:Unclassified) RETURN count(c) AS n"),
        "contested": _count(store, "MATCH (c:Chunk:Ready:Contested) RETURN count(c) AS n"),
        "studies": _count(store, "MATCH (s:Study) RETURN count(s) AS n"),
    }


def _count(store: Store, query: str, params: dict[str, Any] | None = None) -> int:
    r = _df_dicts(store.cypher(query, params=params or {}))
    return int(r[0]["n"]) if r else 0


def triage_map(store: Store, *, doc_id: str | None = None) -> dict[str, Any]:
    """One cheap call that aggregates the deterministic content signals so an
    agent can orient *without reading the corpus*: chunk counts, the
    content_kind breakdown, boilerplate / low-quality flags, structured-entity
    coverage, embedding state, and OCR-pending pages. All from label-indexed
    counts (fast). Scope with ``doc_id``."""
    where = "WHERE c.doc_id = $d " if doc_id else ""
    params: dict[str, Any] = {"d": doc_id} if doc_id else {}

    def lcount(label: str) -> int:
        return _count(store, f"MATCH (c:Chunk:{label}) {where}RETURN count(c) AS n", params)

    kind_rows = _df_dicts(store.cypher(
        f"MATCH (c:Chunk:{LABEL_READY}) {where}RETURN c.content_kind AS k, count(c) AS n",
        params=params,
    ))
    content_kinds = {r["k"]: int(r["n"]) for r in kind_rows if r.get("k")}
    entities = {et: lcount(lbl) for et, lbl in ENTITY_LABELS.items()}
    entities = {k: v for k, v in entities.items() if v}

    total = _count(store, f"MATCH (c:Chunk) {where}RETURN count(c) AS n", params)
    ready = lcount(LABEL_READY)
    embedded = lcount(LABEL_EMBEDDED)
    boilerplate = lcount(LABEL_BOILERPLATE)
    low_quality = lcount(LABEL_LOW_QUALITY)
    pwhere = "WHERE p.doc_id = $d AND " if doc_id else "WHERE "
    pending_ocr = _count(
        store, f"MATCH (p:Page) {pwhere}p.needs_ocr = true RETURN count(p) AS n", params,
    )
    sections = _count(
        store, f"MATCH (s:Section) {('WHERE s.doc_id = $d ' if doc_id else '')}RETURN count(s) AS n",
        params,
    )
    # Element classification coverage (per registered element type, ready chunks).
    classified = lcount(LABEL_CLASSIFIED)
    unclassified = lcount(LABEL_UNCLASSIFIED)
    contested = lcount(LABEL_CONTESTED)
    elements = {v: lcount(lbl) for v in valid_element_values() if (lbl := element_label(v))}
    elements = {k: v for k, v in sorted(elements.items(), key=lambda kv: -kv[1]) if v}

    top = ", ".join(f"{n} {k}" for k, n in sorted(content_kinds.items(), key=lambda kv: -kv[1]))
    ent = ", ".join(f"{n} {k}" for k, n in entities.items())
    el = ", ".join(f"{n} {k}" for k, n in elements.items())
    summary = (
        f"{total} chunks ({ready} ready, {embedded} embedded); kinds: {top or 'n/a'}; "
        f"{boilerplate} boilerplate, {low_quality} low-quality; "
        f"{pending_ocr} page(s) need OCR"
        + (f"; entities: {ent}" if ent else "")
        + (f"; elements: {el}" if el else "")
        + (f"; {unclassified} unclassified, {contested} contested" if (classified or unclassified) else "")
    )
    return {
        "chunks": total,
        "ready": ready,
        "embedded": embedded,
        "unembedded": ready - embedded,
        "pending_ocr": pending_ocr,
        "sections": sections,
        "content_kinds": content_kinds,
        "boilerplate": boilerplate,
        "low_quality": low_quality,
        "entities": entities,
        "classified": classified,
        "unclassified": unclassified,
        "contested": contested,
        "elements": elements,
        "summary": summary,
    }


def element_scope_coverage(
    store: Store, *, element: str, doc_id: str | None = None, section_id: str | None = None,
) -> dict[str, Any]:
    """How an `element=` scope partitions the ready chunks — the non-lossy block
    attached to a scoped `study_ledger`. Reconciles: `in_scope + excluded_total
    == ready_total`. A non-zero `excluded_unclassified` is a loud signal that the
    scope's universe has un-routed chunks (its blind spot). Unknown element raises."""
    lbl = element_label(element)
    if lbl is None:
        raise InvalidEnumError(
            f"unknown element {element!r} — not in the registered schema "
            f"({sorted(valid_element_values())})"
        )
    preds = []
    params: dict[str, Any] = {}
    if doc_id:
        preds.append("c.doc_id = $d")
        params["d"] = doc_id
    if section_id:
        preds.append("c.section_id = $s")
        params["s"] = section_id
    where = ("WHERE " + " AND ".join(preds) + " ") if preds else ""

    def c(extra_label: str = "") -> int:
        return _count(store, f"MATCH (c:Chunk:{LABEL_READY}{extra_label}) {where}RETURN count(c) AS n", params)

    ready = c()
    in_scope = c(f":{lbl}")
    classified = c(f":{LABEL_CLASSIFIED}")
    unclassified = c(f":{LABEL_UNCLASSIFIED}")
    excluded_other_element = classified - in_scope          # classified, but not this element
    excluded_unclassified = (ready - classified - unclassified) + unclassified  # not-yet + unclassified
    excluded_total = excluded_other_element + excluded_unclassified
    return {
        "element": element,
        "in_scope": in_scope,
        "excluded_other_element": excluded_other_element,
        "excluded_unclassified": excluded_unclassified,
        "excluded_total": excluded_total,
        "ready_total": ready,
    }


def element_consistency(store: Store) -> dict[str, Any]:
    """Audit the two-sources-of-truth invariant: a chunk's element *labels* must
    match the element set derived from its canonical `element_types_json`. Returns
    `{checked, inconsistent, sample}` — drift (e.g. from a reclassification that
    dropped a type) is observable, never silent."""
    rows = _df_dicts(store.cypher(
        "MATCH (c:Chunk) WHERE c.element_types_json IS NOT NULL "
        "RETURN c.id AS id, c.element_types_json AS j, labels(c) AS labels"
    ))
    valid_labels = {lbl for v in valid_element_values() if (lbl := element_label(v))}
    inconsistent: list[str] = []
    for r in rows:
        try:
            recs = json.loads(r.get("j") or "[]")
        except (TypeError, ValueError):
            recs = []
        derived = {element_label(rec["type"]) for rec in recs if rec.get("type") and element_label(rec.get("type"))}
        on_node = {lbl for lbl in (r.get("labels") or []) if lbl in valid_labels}
        if derived != on_node:
            inconsistent.append(str(r["id"]))
    return {"checked": len(rows), "inconsistent": len(inconsistent), "sample": inconsistent[:20]}


def study_confidence(store: Store, *, study_id: str) -> dict[str, Any]:
    """Confidence + **blind spots** for a study (R4/R5). Turns un-run lenses into
    a named, listed gap rather than a silent one:

    - per-finding `confidence` / `escalation_state` / `reviewer_count`;
    - `contested` and `low_depth_units` (the next-round worklists);
    - `coverage_by_lens` — every *registered* lens with whether it has run and how
      many units it examined; un-run lenses are the `blind_spots`;
    - `recommended_next_escalation` — the most valuable next step (or None);
    - `settled` — nothing contested, nothing shallow, and all required lenses run.
    """
    from kglite_docs import study as study_mod
    from kglite_docs.lenses import available_lenses

    meta = _df_dicts(store.cypher(
        "MATCH (s:Study {id: $id}) RETURN s.question AS question",
        params={"id": study_id},
    ))
    if not meta:
        raise InvalidEnumError(f"study not found: {study_id}")

    findings = study_mod.list_findings(store, study_id=study_id)
    contested = [f["finding_id"] for f in findings if f.get("escalation_state") == "contested"]
    low_depth = [f["finding_id"] for f in findings if int(f.get("reviewer_count", 0)) < 2]

    # Per-lens coverage from the rounds' EXAMINED edges.
    rows = _df_dicts(store.cypher(
        "MATCH (:Study {id: $id})-[:HAS_ROUND]->(r:ReviewRound) "
        "OPTIONAL MATCH (r)-[:EXAMINED]->(u) "
        "WHERE r.lens <> '' "
        "RETURN r.lens AS lens, count(DISTINCT u) AS units, count(DISTINCT r) AS rounds",
        params={"id": study_id},
    ))
    examined_by_lens = {r["lens"]: {"units": int(r.get("units") or 0), "rounds": int(r.get("rounds") or 0)}
                        for r in rows if r.get("lens")}
    coverage_by_lens: dict[str, Any] = {}
    blind_spots: list[str] = []
    for lens in available_lenses():
        info = examined_by_lens.get(lens)
        if info is None:
            coverage_by_lens[lens] = {"run": False, "rounds": 0, "units_examined": 0}
            blind_spots.append(lens)
        else:
            coverage_by_lens[lens] = {
                "run": True, "rounds": info["rounds"], "units_examined": info["units"],
            }

    policy = study_mod.completion_policy(store, study_id)
    required = policy.get("required_lenses", []) if policy else []
    required_blind = [le for le in required if le not in examined_by_lens]

    # Recommend the highest-leverage next step.
    rec: dict[str, Any] | None = None
    if contested:
        rec = {"action": "escalate", "scope": "contested", "kind": "panel",
               "why": f"{len(contested)} contested finding(s) need more reviewers"}
    elif required_blind:
        rec = {"action": "escalate", "scope": "uncovered", "lens": required_blind[0],
               "why": f"required lens {required_blind[0]!r} has not run"}
    elif low_depth:
        rec = {"action": "escalate", "scope": "low_depth", "kind": "panel",
               "why": f"{len(low_depth)} finding(s) reviewed by <2 agents"}
    elif blind_spots:
        rec = {"action": "escalate", "scope": "uncovered", "lens": blind_spots[0],
               "why": f"lens {blind_spots[0]!r} has not run (a blind spot)"}

    settled = not contested and not low_depth and not required_blind

    return {
        "study_id": study_id,
        "question": meta[0]["question"],
        "findings": [{
            "finding_id": f["finding_id"], "statement": f.get("statement", ""),
            "confidence": f.get("confidence", 0.0), "escalation_state": f.get("escalation_state"),
            "reviewer_count": f.get("reviewer_count", 0),
        } for f in findings],
        "contested": contested,
        "low_depth_units": low_depth,
        "coverage_by_lens": coverage_by_lens,
        "blind_spots": blind_spots,
        "recommended_next_escalation": rec,
        "completion_policy": policy,
        "settled": settled,
    }
