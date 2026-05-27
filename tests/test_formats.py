"""Multi-format ingestion."""

from __future__ import annotations

from pathlib import Path

from kglite_docs.ingest.formats import SUPPORTED_FORMATS, detect_format, parse_document


def test_supported_formats_contains_core() -> None:
    expected = {"pdf", "docx", "pptx", "md", "txt", "html", "png", "jpg"}
    assert expected.issubset(set(SUPPORTED_FORMATS))


def test_detect_format_by_extension() -> None:
    assert detect_format("foo.pdf") == "pdf"
    assert detect_format("/a/b/c.MD") == "md"
    assert detect_format("img.JPEG") == "jpeg"


def test_parse_md_splits_on_h1(tmp_path: Path) -> None:
    f = tmp_path / "x.md"
    f.write_text(
        "Preamble before any heading.\n\n# Section A\n\nBody of A.\n\n"
        "# Section B\n\nBody of B.\n",
        encoding="utf-8",
    )
    pages = parse_document(f)
    assert len(pages) == 3  # preamble + 2 sections
    assert "Section A" in pages[1].markdown
    assert "Section B" in pages[2].markdown


def test_parse_md_single_page_when_no_h1(tmp_path: Path) -> None:
    f = tmp_path / "x.md"
    f.write_text("Just plain text, no headings.", encoding="utf-8")
    pages = parse_document(f)
    assert len(pages) == 1
    assert pages[0].markdown.strip() == "Just plain text, no headings."


def test_parse_txt(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("plain content", encoding="utf-8")
    pages = parse_document(f)
    assert len(pages) == 1
    assert pages[0].has_text
    assert not pages[0].needs_ocr


def test_parse_html_to_markdown_pages(tmp_path: Path) -> None:
    f = tmp_path / "x.html"
    f.write_text(
        "<html><body><h1>One</h1><p>first</p><h1>Two</h1><p>second</p></body></html>",
        encoding="utf-8",
    )
    pages = parse_document(f)
    assert len(pages) == 2
    assert "first" in pages[0].markdown
    assert "second" in pages[1].markdown


def test_unsupported_format_raises(tmp_path: Path) -> None:
    f = tmp_path / "x.rtf"
    f.write_text("never indexed", encoding="utf-8")
    try:
        parse_document(f)
    except ValueError as e:
        assert "unsupported" in str(e).lower()
    else:
        raise AssertionError("expected ValueError for unsupported format")
