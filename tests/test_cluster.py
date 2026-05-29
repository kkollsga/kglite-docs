"""Clustering pipeline."""

from __future__ import annotations

from pathlib import Path

from kglite_docs import Corpus


def _ingest_two_topics(corpus: Corpus, tmp_path: Path) -> None:
    paragraphs = []
    for i in range(8):
        paragraphs.append(f"# Topic A {i}\n\nDense passage retrieval uses BERT and contrastive learning.\n")
    for i in range(8):
        paragraphs.append(f"# Topic B {i}\n\nGraph neural networks aggregate messages from neighbors.\n")
    p = tmp_path / "c.md"
    p.write_text("\n\n".join(paragraphs), encoding="utf-8")
    corpus.ingest(p)
    corpus.index()


def test_kmeans_clustering_runs(corpus: Corpus, tmp_path: Path) -> None:
    _ingest_two_topics(corpus, tmp_path)
    r = corpus.cluster_chunks(algorithm="kmeans", params={"k": 4})
    assert r["clusters"] >= 1
    assert r["members"] > 0
    overview = corpus.cluster_overview()
    assert len(overview) == r["clusters"]


def test_get_cluster_returns_members(corpus: Corpus, tmp_path: Path) -> None:
    _ingest_two_topics(corpus, tmp_path)
    corpus.cluster_chunks(algorithm="kmeans", params={"k": 3})
    overview = corpus.cluster_overview()
    assert overview
    cid = overview[0]["id"]
    detail = corpus.get_cluster(cid)
    assert detail
    assert "members" in detail and detail["size"] > 0
    assert "top_terms" in detail


def test_louvain_falls_back_to_embedding_cluster(corpus: Corpus, tmp_path: Path) -> None:
    """When `CALL louvain()` lacks input edges, we fall back to a numpy
    k-means; the test just asserts we still get clusters back."""
    _ingest_two_topics(corpus, tmp_path)
    r = corpus.cluster_chunks(algorithm="louvain")
    assert r["clusters"] >= 1
