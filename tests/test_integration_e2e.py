"""End-to-end integration across the 0.0.7–0.0.10 feature set, exercised
*together* on a real PDF (with an outline) and the real bge-m3 embedder — and
once through the live MCP server. Complements the per-feature unit tests by
catching cross-feature interactions the stub embedder / isolated tests can miss.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from kglite_docs import Corpus
from kglite_docs.errors import NotIndexedError


def _make_outlined_pdf(out: Path) -> Path:
    doc = pymupdf.open()
    bodies = [
        "Introduction. Dense passage retrieval encodes a passage into a single "
        "vector for semantic search over large corpora of documents.",
        "Methods. We compare late interaction token-level matching against the "
        "single-vector retrieval baseline across several benchmark datasets.",
        "Results. Late interaction substantially improves recall on long "
        "documents where a single vector loses fine-grained token information.",
    ]
    for body in bodies:
        page = doc.new_page()
        # Repeat + wrap into a textbox so pymupdf4llm reliably extracts the page
        # (a single short line can come back empty).
        page.insert_textbox(pymupdf.Rect(72, 72, 520, 760), (body + " ") * 6, fontsize=11)
    doc.set_toc([[1, "Introduction", 1], [1, "Methods", 2], [1, "Results", 3]])
    doc.save(str(out))
    doc.close()
    return out


@pytest.mark.embed
def test_e2e_structure_augmentation_and_full_study(tmp_path: Path) -> None:
    c = Corpus.create(tmp_path / "e2e.kgl")  # real bge-m3 embedder
    pdf = _make_outlined_pdf(tmp_path / "paper.pdf")

    # FEAT-9 (sections) + FEAT-10 (structure-aware) + FEAT-11 (summary-augmented).
    res = c.ingest(
        pdf, structure_aware=True,
        context_summary="A paper comparing dense single-vector vs late-interaction retrieval.",
    )
    assert res.section_count >= 2

    # FEAT-1/2: coverage + status are observable pre-index.
    assert c.status()["unembedded"] > 0
    assert c.coverage_report()["unembedded"] > 0

    # BUG-2: retrieval over an unindexed corpus is loud, not a silent [].
    with pytest.raises(NotIndexedError):
        c.search("retrieval")

    while c.index()["pending"]:
        pass

    # FEAT-9: outline → ordered sections; scoping is total (every chunk has one).
    secs = c.list_sections(res.doc_id)
    titles = [s["title"] for s in secs]
    assert {"Introduction", "Methods", "Results"} <= set(titles)
    rows = c.cypher(
        "MATCH (c:Chunk:Ready) RETURN c.id AS id, c.section_id AS sid ORDER BY c.page_number, c.chunk_index"
    ).to_list()
    assert rows and all(r["sid"] for r in rows)

    # Augmented vectors still retrieve; FEAT-3: searched_fraction == 1.0 when full.
    assert c.search("late interaction token matching", top_k=3)
    assert c.compose_context("retrieval", max_tokens=400)["searched_fraction"] == 1.0

    ids = [r["id"] for r in rows]
    sid = c.define_study("Late interaction is necessary for long-document recall", created_by="lead")

    # FEAT-12 (assess_many) + FEAT-4 (provenance) + FEAT-6 (pinpoint span) + FEAT-7 (deferred).
    text0 = c.get_chunk(ids[0]).text            # FEAT-14: attribute access
    c.assess_many(sid, [
        {"chunk_id": ids[0], "stance": "supports", "weight": 0.9, "agent_id": "a1",
         "provenance": "primary_text", "char_start": 0, "char_end": min(12, len(text0))},
        {"chunk_id": ids[1], "stance": "against", "weight": 0.6, "agent_id": "a2",
         "provenance": "characterization"},
        {"chunk_id": ids[2], "stance": "deferred", "weight": 0.3, "agent_id": "a1"},
    ])
    led = c.study_ledger(sid)
    assert led["total"] == 3
    assert led["tallies"]["deferred"] == 1 and led["tallies"]["supports"] == 1
    by_chunk = {r["chunk_id"]: r for r in led["rows"]}
    assert by_chunk[ids[0]]["char_end"] >= 1 and by_chunk[ids[0]]["quote"]
    assert by_chunk[ids[1]]["provenance"] == "characterization"

    # FEAT-8 (conflicts): an opposing current assessment surfaces the chunk.
    c.assess(sid, ids[1], stance="supports", weight=0.7, agent_id="a3")
    assert any(x["chunk_id"] == ids[1] for x in c.study_conflicts(sid)["conflicts"])

    # FEAT-5 (supersede) resolves the disagreement → conflict clears, current-only.
    against = next(
        r for r in c.study_ledger(sid, include_superseded=True)["rows"]
        if r["chunk_id"] == ids[1] and r["stance"] == "against"
    )
    c.supersede_assessment(against["assessment_id"], stance="supports", weight=0.8, agent_id="a2")
    assert not any(x["chunk_id"] == ids[1] for x in c.study_conflicts(sid)["conflicts"])
    assert against["assessment_id"] not in {r["assessment_id"] for r in c.study_ledger(sid)["rows"]}

    # FEAT-9: section-scoped ledger + BUG-3 honest counts.
    intro_sid = next(s["id"] for s in secs if s["title"] == "Introduction")
    scoped = c.study_ledger(sid, section_id=intro_sid)
    assert scoped["total"] == scoped["returned"]
    assert all(
        c.get_chunk(r["chunk_id"]).section_id == intro_sid for r in scoped["rows"]
    )

    # FEAT-7: the deferred chunk is parked — still in the work-list.
    assert ids[2] in {r["id"] for r in c.next_unassessed(sid, limit=1000)}


@pytest.mark.mcp
def test_e2e_mcp_new_surface_round_trip(tmp_path: Path) -> None:
    """The new study/document actions work through the live MCP app with
    persistence (validates tools.py wiring, not just the Corpus layer)."""
    from kglite_docs.mcp_server.server import build_app
    from tests.test_mcp_smoke import _call

    c = Corpus.create(tmp_path / "mcp_e2e.kgl")
    app = build_app(c, warm_embedder=False)

    md = tmp_path / "doc.md"
    md.write_text(
        "# Alpha\n\nAlpha section body with enough words to chunk.\n\n"
        "# Beta\n\nBeta section body with enough words to chunk.\n",
        encoding="utf-8",
    )
    ing = _call(app, "document", {"action": "ingest", "path": str(md),
                                  "structure_aware": True})
    doc_id = ing["doc_id"]

    secs = _call(app, "document", {"action": "sections", "doc_id": doc_id}, as_list=True)
    assert [s["title"] for s in secs] == ["Alpha", "Beta"]

    chunk_ids = [r["id"] for r in c.cypher(
        "MATCH (c:Chunk:Ready) RETURN c.id AS id ORDER BY c.chunk_index"
    ).to_list()]

    sid = _call(app, "study", {"action": "define", "question": "Q", "agent_id": "lead"})
    _call(app, "study", {"action": "assess_many", "study_id": sid, "rows": [
        {"chunk_id": chunk_ids[0], "stance": "supports", "weight": 0.8, "agent_id": "a1"},
        {"chunk_id": chunk_ids[1], "stance": "against", "weight": 0.6, "agent_id": "a2"},
    ]})
    led = _call(app, "study", {"action": "ledger", "study_id": sid})
    assert led["total"] == 2

    conflicts = _call(app, "study", {"action": "conflicts", "study_id": sid})
    assert conflicts["total"] == 0  # different chunks, no conflict

    # supersede through the tool, then current-only ledger shrinks the loser out.
    aid = led["rows"][0]["assessment_id"]
    _call(app, "study", {"action": "supersede", "assessment_id": aid,
                         "stance": "neutral", "weight": 0.1, "agent_id": "a9"})
    led2 = _call(app, "study", {"action": "ledger", "study_id": sid})
    assert aid not in {r["assessment_id"] for r in led2["rows"]}

    # persistence: reopen (same process — lock allows) and the study survives.
    c.save()
    c2 = Corpus.open(tmp_path / "mcp_e2e.kgl")
    assert c2.study_ledger(sid)["total"] == 2
    c.store.close()
    c2.store.close()
