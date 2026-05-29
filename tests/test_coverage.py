"""FEAT-1/2: coverage_report() + corpus.status() make extraction/embedding
coverage observable instead of silently assumed."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from kglite_docs import Corpus


def _image_only_pdf(out: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=200)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 64, 64))
    pix.clear_with(200)
    page.insert_image(pymupdf.Rect(10, 10, 74, 74), pixmap=pix)
    doc.save(str(out))
    doc.close()
    return out


def _ingest_mixed(corpus: Corpus, tmp_path: Path) -> None:
    corpus.ingest(_image_only_pdf(tmp_path / "scan.pdf"))
    md = tmp_path / "notes.md"
    md.write_text("# Notes\n\n" + ("Plenty of real extractable words here. " * 12), encoding="utf-8")
    corpus.ingest(md)


def test_coverage_report_surfaces_image_and_embedding_gaps(corpus: Corpus, tmp_path: Path) -> None:
    _ingest_mixed(corpus, tmp_path)
    rep = corpus.coverage_report()
    assert rep["image_pages"] >= 1
    assert rep["pending_ocr"] >= 1
    assert rep["unembedded"] > 0          # ingest is embed=False
    assert rep["embedded"] == 0
    assert isinstance(rep["summary"], str) and "image-only" in rep["summary"]
    # the scanned doc is low-coverage; the text doc is fully extractable
    by_id = {d["doc_id"]: d for d in rep["documents"]}
    scan = min(rep["documents"], key=lambda d: d["extractable_text_ratio"])
    txt = max(rep["documents"], key=lambda d: d["extractable_text_ratio"])
    assert scan["extractable_text_ratio"] < 1.0
    assert txt["extractable_text_ratio"] == 1.0
    assert len(by_id) == 2


def test_coverage_report_doc_scope(corpus: Corpus, tmp_path: Path) -> None:
    _ingest_mixed(corpus, tmp_path)
    all_docs = corpus.coverage_report()["documents"]
    one = all_docs[0]["doc_id"]
    scoped = corpus.coverage_report(doc_id=one)
    assert [d["doc_id"] for d in scoped["documents"]] == [one]


def test_status_snapshot(corpus: Corpus, tmp_path: Path) -> None:
    _ingest_mixed(corpus, tmp_path)
    s = corpus.status()
    assert set(s) == {
        "docs", "pages", "chunks", "embedded", "unembedded",
        "image_pages", "pending_ocr", "studies",
    }
    assert s["docs"] == 2
    assert s["image_pages"] >= 1
    assert s["studies"] == 0
    assert s["embedded"] == 0
    assert s["unembedded"] > 0
