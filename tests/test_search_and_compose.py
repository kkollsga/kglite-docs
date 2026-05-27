"""Search, similar_chunks, compose_context — with the stub embedder."""

from __future__ import annotations

from pathlib import Path

from kglite_docs import Corpus


def _make_corpus_with_text(corpus: Corpus, tmp_path: Path) -> str:
    p = tmp_path / "x.md"
    p.write_text(
        "# Topic A\n\nDense passage retrieval uses BERT.\n\n"
        "# Topic B\n\nLSTM recurrent networks were superseded.\n\n"
        "# Topic C\n\nAttention is the core building block of transformers.\n",
        encoding="utf-8",
    )
    return corpus.ingest(p).doc_id


def test_search_returns_hits_in_descending_score(corpus: Corpus, tmp_path: Path) -> None:
    _make_corpus_with_text(corpus, tmp_path)
    hits = corpus.search("Dense passage retrieval uses BERT.", top_k=3)
    assert len(hits) >= 1
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_attaches_page_and_text(corpus: Corpus, tmp_path: Path) -> None:
    _make_corpus_with_text(corpus, tmp_path)
    hits = corpus.search("LSTM", top_k=5)
    assert hits
    assert all("text" in h and "page" in h for h in hits)


def test_search_filters(corpus: Corpus, tmp_path: Path) -> None:
    doc1 = _make_corpus_with_text(corpus, tmp_path)
    other = tmp_path / "other.md"
    other.write_text("# Foo\n\nUnrelated content here.\n", encoding="utf-8")
    corpus.ingest(other)
    hits = corpus.search("retrieval", top_k=10, filters={"doc_id": doc1})
    assert hits and all(h["doc_id"] == doc1 for h in hits)


def test_search_records_view_for_agent(corpus: Corpus, tmp_path: Path) -> None:
    _make_corpus_with_text(corpus, tmp_path)
    corpus.search("transformer", top_k=2, agent_id="claude-1")
    agents = corpus.list_agents()
    assert any(a["id"] == "claude-1" for a in agents)


def test_compose_context_respects_budget(corpus: Corpus, tmp_path: Path) -> None:
    p = tmp_path / "x.md"
    # Many chunks to ensure budget actually clips
    body = "# H\n\n" + "\n\n".join(f"Paragraph {i} with content." for i in range(30))
    p.write_text(body, encoding="utf-8")
    corpus.ingest(p)
    bundle = corpus.compose_context("Paragraph", max_tokens=50)
    assert bundle["used_tokens"] <= 50
    assert all("text" in item for item in bundle["items"])
