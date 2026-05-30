"""FEAT-11: opt-in summary-augmented chunking — a doc-level summary is prepended
to each chunk *before embedding* (the vector carries global context) while the
stored chunk text stays clean. The stub embedder is sha256(text)→vector, so the
augmented embedding is exactly assertable."""

from __future__ import annotations

from pathlib import Path

import pytest

from kglite_docs import Corpus
from tests.conftest import StubEmbedder

_SUMMARY = "CONTEXT: 2021 merger agreement between Acme and Beta."
_BODY = "# Topic\n\nThe parties agree to the terms set out below in full.\n"


def _one_chunk(corpus: Corpus) -> tuple[str, str, list[float]]:
    rows = corpus.cypher(
        "MATCH (c:Chunk:Ready) RETURN c.id AS id, c.text AS t ORDER BY c.chunk_index",
    ).to_list()
    cid, text = rows[0]["id"], rows[0]["t"]
    vec = corpus.store.g.embeddings("Chunk", "text")[cid]
    return cid, text, list(vec)


def test_context_summary_augments_vector_not_stored_text(
    corpus: Corpus, tmp_path: Path, stub_embedder: StubEmbedder,
) -> None:
    p = tmp_path / "d.md"
    p.write_text(_BODY, encoding="utf-8")
    corpus.ingest(p, context_summary=_SUMMARY)
    corpus.index()

    _cid, text, vec = _one_chunk(corpus)
    assert "CONTEXT" not in text            # stored text is clean
    augmented = stub_embedder.embed([f"{_SUMMARY}\n\n{text}"])[0]
    clean = stub_embedder.embed([text])[0]
    assert vec == pytest.approx(augmented)  # vector embeds the augmented form
    assert vec != pytest.approx(clean)


def test_default_no_augmentation(
    corpus: Corpus, tmp_path: Path, stub_embedder: StubEmbedder,
) -> None:
    p = tmp_path / "d.md"
    p.write_text(_BODY, encoding="utf-8")
    corpus.ingest(p)            # no context_summary
    corpus.index()
    _cid, text, vec = _one_chunk(corpus)
    assert vec == pytest.approx(stub_embedder.embed([text])[0])


def test_context_summary_inline_embed(
    corpus: Corpus, tmp_path: Path, stub_embedder: StubEmbedder,
) -> None:
    p = tmp_path / "d.md"
    p.write_text(_BODY, encoding="utf-8")
    corpus.ingest(p, context_summary=_SUMMARY, embed=True)  # one-shot, no index()
    _cid, text, vec = _one_chunk(corpus)
    assert vec == pytest.approx(stub_embedder.embed([f"{_SUMMARY}\n\n{text}"])[0])
