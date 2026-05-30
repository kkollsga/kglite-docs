"""0.0.14 Phase 1: robust OCR-requirement detection.

A scanned exhibit must never silently pass as `ready` (the confession-page miss):
detection is image-coverage- and density-aware, recall-biased."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from kglite_docs import Corpus
from kglite_docs.ingest.parser import parse_pdf

_PARA = "The quick brown fox jumps over the lazy dog near the riverbank today. " * 8


def _build(path: Path, *, text: str = "", image_frac: float = 0.0) -> None:
    """A one-page PDF with an image covering `image_frac` of the page + `text`."""
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    if image_frac > 0:
        h = int(800 * image_frac)
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 600, h))
        pix.clear_with(180)
        page.insert_image(pymupdf.Rect(0, 0, 600, h), pixmap=pix)
    if text:
        y0 = min(int(800 * image_frac) + 20, 700)
        page.insert_textbox(pymupdf.Rect(40, y0, 560, 790), text, fontsize=10)
    doc.save(str(path))
    doc.close()


def test_full_page_scan_flagged(tmp_path: Path) -> None:
    # A full-page image with only a few stray chars → a scan, must OCR.
    _build(tmp_path / "a.pdf", image_frac=1.0, text="x y z")
    assert parse_pdf(tmp_path / "a.pdf")[0].needs_ocr is True


def test_image_dominated_with_junk_over_threshold_flagged(tmp_path: Path) -> None:
    # The regression: an image-dominated page whose junk fragments clear the
    # 120-char sparse floor used to pass as `ready`. It must still be flagged.
    _build(tmp_path / "b.pdf", image_frac=0.7, text="abcde " * 30)  # ~150 alnum
    pc = parse_pdf(tmp_path / "b.pdf")[0]
    assert pc.image_coverage >= 0.6 and pc.needs_ocr is True


def test_text_page_not_flagged(tmp_path: Path) -> None:
    _build(tmp_path / "c.pdf", image_frac=0.0, text=_PARA)
    assert parse_pdf(tmp_path / "c.pdf")[0].needs_ocr is False


def test_small_image_with_rich_text_not_over_flagged(tmp_path: Path) -> None:
    # A real text page with a small figure must NOT be sent to OCR.
    _build(tmp_path / "d.pdf", image_frac=0.12, text=_PARA)
    pc = parse_pdf(tmp_path / "d.pdf")[0]
    assert pc.needs_ocr is False


def test_scanned_page_is_visible_in_ocr_status(corpus: Corpus, tmp_path: Path) -> None:
    # End-to-end: a flagged scan lands status needs_ocr and is counted (not hidden).
    _build(tmp_path / "scan.pdf", image_frac=1.0, text="x")
    corpus.ingest(tmp_path / "scan.pdf")
    st = corpus.ocr_status()
    assert st["pending_pages"] >= 1
    assert corpus.cypher(
        "MATCH (c:Chunk:NeedsOcr) RETURN count(c) AS n"
    ).to_list()[0]["n"] >= 1


def test_needs_ocr_unit_rules() -> None:
    from kglite_docs.ingest.parser import _needs_ocr
    # sparse text + image-bearing → OCR
    assert _needs_ocr("a b c", has_images=True, image_coverage=0.0) is True
    # sparse text, no image at all → nothing to OCR
    assert _needs_ocr("a b c", has_images=False, image_coverage=0.0) is False
    # image-dominated + not text-rich → OCR (even above the sparse floor)
    assert _needs_ocr("x " * 90, has_images=False, image_coverage=0.8) is True
    # text-rich page → not flagged regardless of a banner image
    assert _needs_ocr("word " * 200, has_images=True, image_coverage=0.2) is False
