"""FEAT-10: opt-in structure-aware chunking — start a fresh chunk at every
top-level heading. Uses a whitespace stub tokenizer so it runs in the fast
suite (the real bge-m3 chunker tests live in test_chunker.py, marked `embed`)."""

from __future__ import annotations

from pathlib import Path

from kglite_docs import Corpus
from kglite_docs.ingest.chunker import chunk_page


class _Enc:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids


class _StubTok:
    """Word-count tokenizer: one token per whitespace-split word."""

    def encode(self, text: str, add_special_tokens: bool = False) -> _Enc:
        return _Enc(text.split())

    def decode(self, ids: list[str]) -> str:
        return " ".join(ids)


_TOK = _StubTok()

_DOC = (
    "# Alpha\n\nAlpha body words here.\n\n"
    "# Beta\n\nBeta body words here.\n\n"
    "# Gamma\n\nGamma body words here.\n"
)


def test_default_packs_across_headings() -> None:
    # Whole doc is far under target → default packs into a single chunk that
    # straddles all three sections (its headings reflect only the last one).
    chunks = chunk_page(_DOC, target_tokens=1000, overlap_tokens=0, tokenizer=_TOK)
    assert len(chunks) == 1
    assert chunks[0].headings == ["Gamma"]


def test_structure_aware_one_chunk_per_section() -> None:
    chunks = chunk_page(
        _DOC, target_tokens=1000, overlap_tokens=0, tokenizer=_TOK, structure_aware=True
    )
    assert [c.headings[0] for c in chunks] == ["Alpha", "Beta", "Gamma"]
    # No chunk bleeds another section's body.
    assert "Beta" not in chunks[0].text and "Gamma" not in chunks[0].text


def test_structure_aware_still_honors_token_target() -> None:
    # One oversized section must still split by size (structure-aware doesn't
    # disable size splitting).
    big = "# Big\n\n" + ("word " * 60)
    chunks = chunk_page(
        big, target_tokens=10, overlap_tokens=0, tokenizer=_TOK, structure_aware=True
    )
    assert len(chunks) > 1
    assert all(c.token_count <= 20 for c in chunks)  # ≤ target (small overshoot)
    assert all(c.headings == ["Big"] for c in chunks)


def test_structure_aware_end_to_end_ingest(corpus: Corpus, tmp_path: Path) -> None:
    p = tmp_path / "secs.md"
    p.write_text(_DOC, encoding="utf-8")
    corpus.ingest(p, structure_aware=True)
    # Each chunk belongs to exactly one section's heading — no cross-section bleed.
    rows = corpus.cypher(
        "MATCH (c:Chunk:Ready) RETURN c.headings_json AS h, c.text AS t",
    ).to_list()
    assert rows
    for r in rows:
        body = r["t"]
        # A chunk under one heading must not contain another section's body.
        present = [w for w in ("Alpha body", "Beta body", "Gamma body") if w in body]
        assert len(present) <= 1, f"chunk bled across sections: {present}"
