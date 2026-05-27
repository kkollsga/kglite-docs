"""Export to MD / DOCX / PDF."""

from __future__ import annotations

from pathlib import Path

from kglite_docs import Corpus


def _ingest(corpus: Corpus, tmp_path: Path) -> str:
    p = tmp_path / "doc.md"
    p.write_text(
        "# Title\n\nFirst paragraph.\n\n## Sub\n\nSecond paragraph.\n",
        encoding="utf-8",
    )
    return corpus.ingest(p).doc_id


def test_export_document_md(corpus: Corpus, tmp_path: Path) -> None:
    doc_id = _ingest(corpus, tmp_path)
    out = corpus.export_document(doc_id, tmp_path / "out.md")
    assert out.exists()
    body = out.read_text(encoding="utf-8")
    assert "Title" in body or "First paragraph" in body


def test_export_document_docx(corpus: Corpus, tmp_path: Path) -> None:
    doc_id = _ingest(corpus, tmp_path)
    out = corpus.export_document(doc_id, tmp_path / "out.docx")
    assert out.exists()
    # Docx is a ZIP — sanity-check it's non-empty + has the right header
    head = out.read_bytes()[:4]
    assert head == b"PK\x03\x04"


def test_export_document_pdf(corpus: Corpus, tmp_path: Path) -> None:
    doc_id = _ingest(corpus, tmp_path)
    out = corpus.export_document(doc_id, tmp_path / "out.pdf")
    assert out.exists()
    assert out.read_bytes()[:5] == b"%PDF-"


def test_export_bundle_combines_items(corpus: Corpus, tmp_path: Path) -> None:
    doc_id = _ingest(corpus, tmp_path)
    out = corpus.export_bundle(
        [
            {"kind": "markdown", "text": "## Custom preamble"},
            {"kind": "doc", "id": doc_id},
        ],
        tmp_path / "bundle.md",
        title="Bundle",
    )
    body = out.read_text(encoding="utf-8")
    assert "Custom preamble" in body
    assert "Title" in body or "First paragraph" in body
