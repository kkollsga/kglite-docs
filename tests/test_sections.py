"""FEAT-9: Section nodes (grain between document and chunk) + section-scoped
studies. Sections come from the PDF outline when present, else top-level
heading boundaries — generic and best-effort."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from kglite_docs import Corpus


def _heading_doc(corpus: Corpus, tmp_path: Path) -> str:
    p = tmp_path / "doc.md"
    p.write_text(
        "# Alpha\n\nAlpha body with plenty of words to chunk here.\n\n"
        "# Beta\n\nBeta body with plenty of words to chunk here too.\n\n"
        "# Gamma\n\nGamma body, also enough words to make a chunk.\n",
        encoding="utf-8",
    )
    return corpus.ingest(p).doc_id


def test_sections_from_headings(corpus: Corpus, tmp_path: Path) -> None:
    doc_id = _heading_doc(corpus, tmp_path)
    secs = corpus.list_sections(doc_id)
    assert [s["title"] for s in secs] == ["Alpha", "Beta", "Gamma"]
    assert all(s["chunk_count"] >= 1 for s in secs)
    # Every chunk is linked to a section.
    rows = corpus.cypher(
        "MATCH (c:Chunk:Ready) WHERE c.doc_id = $d RETURN c.section_id AS sid",
        params={"d": doc_id},
    ).to_list()
    assert rows and all(r["sid"] for r in rows)
    # Linkage Document->Section->Chunk holds.
    linked = corpus.cypher(
        "MATCH (:Document {id: $d})-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(c:Chunk) "
        "RETURN count(DISTINCT s) AS sections, count(c) AS chunks",
        params={"d": doc_id},
    ).to_list()[0]
    assert linked["sections"] == 3 and linked["chunks"] >= 3


def test_get_chunk_exposes_section_id(corpus: Corpus, tmp_path: Path) -> None:
    doc_id = _heading_doc(corpus, tmp_path)
    cid = corpus.cypher(
        "MATCH (c:Chunk:Ready) WHERE c.doc_id = $d RETURN c.id AS id ORDER BY c.chunk_index",
        params={"d": doc_id},
    ).to_list()[0]["id"]
    detail = corpus.get_chunk(cid)
    assert detail is not None and detail["section_id"]


def test_section_scoped_study(corpus: Corpus, tmp_path: Path) -> None:
    doc_id = _heading_doc(corpus, tmp_path)
    secs = corpus.list_sections(doc_id)
    alpha, beta = secs[0]["id"], secs[1]["id"]

    # Preview work-list scoped to one section returns only that section's chunks.
    nxt = corpus.next_unassessed(_define(corpus), section_id=alpha)
    assert nxt  # the study id is fresh, so all alpha chunks are unassessed
    alpha_chunks = {
        r["id"] for r in corpus.cypher(
            "MATCH (c:Chunk:Ready) WHERE c.section_id = $s RETURN c.id AS id",
            params={"s": alpha},
        ).to_list()
    }
    assert {r["id"] for r in nxt} <= alpha_chunks

    # Ledger scoped to a section returns only its rows.
    sid = corpus.define_study("Q", created_by="lead")
    a_chunk = next(iter(alpha_chunks))
    b_chunk = next(iter(
        r["id"] for r in corpus.cypher(
            "MATCH (c:Chunk:Ready) WHERE c.section_id = $s RETURN c.id AS id",
            params={"s": beta},
        ).to_list()
    ))
    corpus.assess(sid, a_chunk, stance="supports", weight=0.8, agent_id="a1")
    corpus.assess(sid, b_chunk, stance="against", weight=0.6, agent_id="a1")
    led = corpus.study_ledger(sid, section_id=alpha)
    assert led["total"] == led["returned"] == 1
    assert led["rows"][0]["chunk_id"] == a_chunk


def _define(corpus: Corpus) -> str:
    return corpus.define_study("Scoped", created_by="lead")


def _outline_pdf(out: Path) -> Path:
    doc = pymupdf.open()
    for body in ("Intro page text body here.", "Methods page text body here."):
        page = doc.new_page()
        page.insert_text((72, 72), f"{body} " * 8)
    doc.set_toc([[1, "Intro", 1], [1, "Methods", 2]])
    doc.save(str(out))
    doc.close()
    return out


def test_sections_from_pdf_outline(corpus: Corpus, tmp_path: Path) -> None:
    doc_id = corpus.ingest(_outline_pdf(tmp_path / "outlined.pdf")).doc_id
    secs = corpus.list_sections(doc_id)
    titles = [s["title"] for s in secs]
    assert "Intro" in titles and "Methods" in titles
    # Chunks on page 1 belong to Intro; page 2 to Methods.
    by_page = corpus.cypher(
        "MATCH (s:Section)-[:HAS_CHUNK]->(c:Chunk) WHERE c.doc_id = $d "
        "RETURN s.title AS title, c.page_number AS page",
        params={"d": doc_id},
    ).to_list()
    assert by_page  # outline produced linked sections
    for r in by_page:
        if r["page"] == 1:
            assert r["title"] == "Intro"
        elif r["page"] == 2:
            assert r["title"] == "Methods"
