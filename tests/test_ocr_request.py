"""0.0.14 Phase 2: lazy agent-driven OCR — request → task → submit."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from kglite_docs import Corpus
from kglite_docs.errors import InvalidEnumError


def _scan_pdf(path: Path) -> None:
    """A one-page full-image scan with a stray char → flagged needs_ocr."""
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 600, 800))
    pix.clear_with(180)
    page.insert_image(pymupdf.Rect(0, 0, 600, 800), pixmap=pix)
    page.insert_textbox(pymupdf.Rect(40, 700, 560, 790), "x", fontsize=10)
    doc.save(str(path))
    doc.close()


def _pending_page_id(corpus: Corpus, tmp_path: Path) -> str:
    _scan_pdf(tmp_path / "scan.pdf")
    corpus.ingest(tmp_path / "scan.pdf")
    return corpus.list_pending_ocr(include_images=False)[0]["page_id"]


def test_request_returns_task_and_records(corpus: Corpus, tmp_path: Path) -> None:
    pid = _pending_page_id(corpus, tmp_path)
    task = corpus.request_ocr(page_id=pid, agent_id="lead", agent_type="vision-ocr")
    assert task["tiles"] and task["tiles"][0]["image_b64"] and task["tiles"][0]["image_mime"] == "image/png"
    assert "VERBATIM" in task["prompt"]
    assert task["agent_type"] == "vision-ocr"
    assert task["already_requested"] is False
    # First request is preserved across re-requests (audit of who asked first).
    assert corpus.request_ocr(page_id=pid, agent_id="other")["already_requested"] is True


def test_request_then_submit_flips_and_stamps(corpus: Corpus, tmp_path: Path) -> None:
    pid = _pending_page_id(corpus, tmp_path)
    corpus.request_ocr(page_id=pid, agent_id="lead")
    res = corpus.submit_ocr(pid, "# Deposition\n\nI admit the debt.", agent_id="lead", model="claude")
    assert res["chunks_added"] == 1
    assert corpus.ocr_status()["pending_pages"] == 0
    row = corpus.cypher(
        "MATCH (c:Chunk:Ready) WHERE c.ocr_derived = true RETURN c.text AS t, c.ocr_by AS by"
    ).to_list()
    assert row and "admit the debt" in row[0]["t"] and row[0]["by"] == "lead"


def test_request_on_ready_page_raises(corpus: Corpus, tmp_path: Path) -> None:
    pid = _pending_page_id(corpus, tmp_path)
    corpus.submit_ocr(pid, "real text now", agent_id="lead")
    with pytest.raises(InvalidEnumError, match="not flagged needs_ocr"):
        corpus.request_ocr(page_id=pid, agent_id="lead")


def test_request_needs_an_identifier(corpus: Corpus) -> None:
    with pytest.raises(InvalidEnumError):
        corpus.request_ocr(agent_id="lead")  # neither page_id nor doc_id+page_number
