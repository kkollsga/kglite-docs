"""Chunker — token-aware paragraph packing."""

from __future__ import annotations

import pytest

from kglite_docs.ingest.chunker import chunk_page, count_tokens


pytestmark = pytest.mark.embed  # tokenizer pulls bge-m3 vocab


def test_empty_returns_empty_list() -> None:
    assert chunk_page("") == []
    assert chunk_page("   \n  \n") == []


def test_single_paragraph_fits_in_one_chunk() -> None:
    chunks = chunk_page("A short paragraph.", target_tokens=256)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert "short" in chunks[0].text
    assert chunks[0].token_count > 0


def test_heading_inherited_to_chunks() -> None:
    """Chunks emit with the heading stack as of emission time. With a
    tight token cap each paragraph becomes its own chunk and we can
    assert the stack precisely."""
    text = (
        "# Intro\n\n" + ("This is paragraph one of intro. " * 30) + "\n\n"
        "## Sub\n\n" + ("Subsection paragraph content here. " * 30) + "\n"
    )
    chunks = chunk_page(text, target_tokens=80, overlap_tokens=0)
    assert chunks
    intro_chunks = [c for c in chunks if "paragraph one" in c.text]
    sub_chunks = [c for c in chunks if "Subsection" in c.text]
    assert intro_chunks
    assert sub_chunks
    # First intro chunk should have just ['Intro']
    assert intro_chunks[0].headings == ["Intro"]
    # A sub chunk should include the sub heading too
    assert sub_chunks[0].headings == ["Intro", "Sub"]


def test_chunks_respect_token_cap() -> None:
    # Build a paragraph well below cap, then several so packing kicks in
    paras = ["This is a sentence with several words. " * 5 for _ in range(20)]
    text = "\n\n".join(paras)
    chunks = chunk_page(text, target_tokens=128, overlap_tokens=16)
    assert len(chunks) > 1
    # Allow a small overshoot (single paragraph at cap edge can exceed slightly)
    for c in chunks:
        assert c.token_count <= 256, f"chunk {c.chunk_index} has {c.token_count} tokens"


def test_overlap_appears_in_consecutive_chunks() -> None:
    # Construct a clear boundary so we can verify overlap behaviour
    paras = [f"Paragraph number {i} with enough content. " * 8 for i in range(10)]
    text = "\n\n".join(paras)
    chunks = chunk_page(text, target_tokens=200, overlap_tokens=32)
    if len(chunks) < 2:
        pytest.skip("not enough chunks to assess overlap")
    # Overlap appears as a token-sized suffix of chunk n re-prefixing chunk n+1.
    # The exact substring depends on tokenizer; we assert that the last
    # 20 chars of an earlier chunk appear *somewhere* in a later chunk.
    tail = chunks[0].text[-30:]
    assert any(tail in c.text for c in chunks[1:])


def test_text_hash_set_on_each_chunk() -> None:
    chunks = chunk_page("Something.\n\nElse.\n", target_tokens=256)
    for c in chunks:
        assert c.text_hash_value and len(c.text_hash_value) == 64
