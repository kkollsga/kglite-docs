"""BUG-1: image-only pages must be flagged needs_ocr by text density, not by
'has any text at all' (pymupdf4llm emits a placeholder fragment for image pages
that otherwise sneaks them through as 'ready')."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from kglite_docs import Corpus
from kglite_docs.ingest.parser import (
    OCR_TEXT_THRESHOLD,
    _extractable_alnum,
    _needs_ocr,
)

_MARKER = "==> picture 1 intentionally omitted <=="
_REAL = "# Findings\n\n" + ("The defendant was dismissed via PAD 003. " * 8)


def test_extractable_alnum_ignores_image_placeholders() -> None:
    assert _extractable_alnum(_MARKER) < 5            # placeholder → ~no real text
    assert _extractable_alnum("![fig](x.png)") < 5    # markdown image syntax
    assert _extractable_alnum(_REAL) > OCR_TEXT_THRESHOLD


def test_needs_ocr_decision() -> None:
    # image page whose only "text" is a placeholder → OCR
    assert _needs_ocr(_MARKER, has_images=True) is True
    assert _needs_ocr("", has_images=True) is True
    # genuine text page (even with an image) → not OCR
    assert _needs_ocr(_REAL, has_images=True) is False
    # no image → never OCR (a sparse text-only page isn't a scan)
    assert _needs_ocr(_MARKER, has_images=False) is False


def _image_only_pdf(out: Path) -> Path:
    """A page carrying a real embedded image and no text layer — unlike a mere
    fill, this makes page.get_images() non-empty (the bug's trigger shape)."""
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=200)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 64, 64))
    pix.clear_with(200)
    page.insert_image(pymupdf.Rect(10, 10, 74, 74), pixmap=pix)
    doc.save(str(out))
    doc.close()
    return out


def test_image_only_pdf_page_flagged_needs_ocr(corpus: Corpus, tmp_path: Path) -> None:
    pdf = _image_only_pdf(tmp_path / "scan.pdf")
    r = corpus.ingest(pdf)
    assert r.created is True
    assert r.ocr_pending_pages >= 1
    assert corpus.ocr_status()["pending_pages"] >= 1


def test_real_text_pdf_not_over_flagged(corpus: Corpus, tmp_path: Path) -> None:
    sample = Path("sample_data/subset/dpr.pdf")
    if not sample.exists():
        import pytest
        pytest.skip("sample dpr.pdf not available")
    corpus.ingest(sample)
    # A text paper must not be mass-flagged for OCR by the density heuristic.
    assert corpus.ocr_status()["pending_pages"] == 0
