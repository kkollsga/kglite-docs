"""0.0.15 Phase 1: an OCR'd page that's all [ilegível] must NOT count as covered."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from kglite_docs import Corpus
from kglite_docs.ocr import _legible_chars, _ocr_outcome


def _scan(path: Path, marker: str = "x") -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 600, 800))
    pix.clear_with(180)
    page.insert_image(pymupdf.Rect(0, 0, 600, 800), pixmap=pix)
    page.insert_textbox(pymupdf.Rect(40, 700, 560, 790), marker, fontsize=10)
    doc.save(str(path))
    doc.close()


def test_legibility_metric() -> None:
    assert _legible_chars("[ilegível]\n\n[página ilegível]") == 0
    assert _ocr_outcome(0) == "ocr_illegible"
    assert _ocr_outcome(50) == "ocr_partial"
    assert _ocr_outcome(500) == "ocr_ok"
    # bracketed markers don't count; real words do
    assert _legible_chars("The court ruled. [ilegível] Final.") > 10


def test_illegible_submit_is_surfaced_not_covered(corpus: Corpus, tmp_path: Path) -> None:
    _scan(tmp_path / "s.pdf")
    corpus.ingest(tmp_path / "s.pdf")
    pid = corpus.list_pending_ocr(include_images=False)[0]["page_id"]
    r = corpus.submit_ocr(pid, "[ilegível]\n\n[página ilegível]", agent_id="ocr")
    assert r["ocr_outcome"] == "ocr_illegible" and r["legible_chars"] == 0
    st = corpus.ocr_status()
    # OCR'd (pending 0) but NOT readable — surfaced, not silently covered.
    assert st["pending_pages"] == 0
    assert st["illegible_pages"] == 1 and st["readable_pages"] == 0
    work = corpus.list_illegible_pages()
    assert len(work) == 1 and work[0]["ocr_outcome"] == "ocr_illegible"
    assert "illegible" in corpus.coverage_report()["summary"].lower()


def test_force_reocr_replaces_chunks(corpus: Corpus, tmp_path: Path) -> None:
    import pytest

    from kglite_docs.errors import InvalidEnumError
    _scan(tmp_path / "s.pdf")
    corpus.ingest(tmp_path / "s.pdf")
    pid = corpus.list_pending_ocr(include_images=False)[0]["page_id"]
    # First OCR: illegible.
    corpus.submit_ocr(pid, "[ilegível]", agent_id="sonnet")
    assert corpus.list_illegible_pages()[0]["page_id"] == pid
    # Without force, the page is no longer needs_ocr → request refuses.
    with pytest.raises(InvalidEnumError, match="force=True"):
        corpus.request_ocr(page_id=pid, agent_id="opus")
    # Force re-OCR (escalate to a stronger model) → task again.
    task = corpus.request_ocr(page_id=pid, agent_id="opus", force=True)
    assert task["image_b64"]
    # Re-submit replaces the prior chunk (no duplicates) and clears illegibility.
    corpus.submit_ocr(pid, "# Recovered\n\n" + ("Clear legible ruling text here. " * 8),
                      agent_id="opus", model="opus")
    n = corpus.cypher("MATCH (p:Page {id:$p})-[:HAS_CHUNK]->(c:Chunk) RETURN count(c) AS n",
                      params={"p": pid}).to_list()[0]["n"]
    assert n == 1  # replaced, not duplicated
    assert corpus.ocr_status()["readable_pages"] == 1 and corpus.list_illegible_pages() == []


def test_legible_submit_counts_as_readable(corpus: Corpus, tmp_path: Path) -> None:
    _scan(tmp_path / "s.pdf")
    corpus.ingest(tmp_path / "s.pdf")
    pid = corpus.list_pending_ocr(include_images=False)[0]["page_id"]
    r = corpus.submit_ocr(
        pid, "# Ruling\n\n" + ("The court found the defendant liable for damages. " * 8),
        agent_id="ocr",
    )
    assert r["ocr_outcome"] == "ocr_ok"
    st = corpus.ocr_status()
    assert st["readable_pages"] == 1 and st["illegible_pages"] == 0
    assert corpus.list_illegible_pages() == []
