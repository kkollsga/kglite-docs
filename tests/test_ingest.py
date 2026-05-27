"""Ingestion pipeline end-to-end (uses the stub embedder for speed)."""

from __future__ import annotations

from pathlib import Path

from kglite_docs import Corpus


def _md_doc(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / f"{name}.md"
    p.write_text(body, encoding="utf-8")
    return p


def test_ingest_md_creates_document_pages_chunks(corpus: Corpus, tmp_path: Path) -> None:
    doc = _md_doc(tmp_path, "alpha", "# Alpha\n\nBody of alpha.\n\n# Beta\n\nBody of beta.\n")
    r = corpus.ingest(doc)
    assert r.created is True
    assert r.page_count == 2
    assert r.chunk_count == 2
    assert r.ocr_pending_pages == 0
    docs = corpus.list_documents()
    assert any(d["id"] == r.doc_id for d in docs)


def test_ingest_is_idempotent_on_same_file(corpus: Corpus, tmp_path: Path) -> None:
    doc = _md_doc(tmp_path, "a", "# T\n\nbody\n")
    r1 = corpus.ingest(doc)
    r2 = corpus.ingest(doc)
    assert r1.created is True
    assert r2.created is False
    assert r1.doc_id == r2.doc_id


def test_ingest_dir_picks_up_multiple_formats(corpus: Corpus, tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\ntext\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("plain b\n", encoding="utf-8")
    (tmp_path / "c.html").write_text("<h1>C</h1><p>html c</p>", encoding="utf-8")
    results = corpus.ingest_dir(tmp_path)
    assert len(results) == 3
    assert {r.format for r in results} == {"md", "txt", "html"}


def test_ingest_text_creates_a_doc(corpus: Corpus) -> None:
    r = corpus.ingest_text(
        "# Synthesis\n\nThis is an agent-authored note.\n",
        title="agent-synth-1",
    )
    assert r.created is True
    assert r.format == "md"


def test_chunk_neighbors_are_linked(corpus: Corpus, tmp_path: Path) -> None:
    doc = _md_doc(tmp_path, "n", "# A\n\nbody a\n\n# B\n\nbody b\n")
    r = corpus.ingest(doc)
    rows = corpus.cypher(
        "MATCH (a:Chunk)-[:NEXT_CHUNK]->(b:Chunk) WHERE a.doc_id = $id "
        "RETURN a.id AS a, b.id AS b",
        params={"id": r.doc_id},
    ).to_list()
    assert len(rows) == 1  # 2 chunks → 1 NEXT_CHUNK edge
