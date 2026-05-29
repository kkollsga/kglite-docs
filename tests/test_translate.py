"""Translation layer."""

from __future__ import annotations

from pathlib import Path

from kglite_docs import Corpus


def _ingest(corpus: Corpus, tmp_path: Path) -> str:
    p = tmp_path / "doc.md"
    p.write_text("# Topic\n\nThe cat sat on the mat.\n", encoding="utf-8")
    corpus.ingest(p)
    corpus.index()
    return corpus.search("cat sat", top_k=1)[0]["id"]


def test_add_and_get_translation(corpus: Corpus, tmp_path: Path) -> None:
    cid = _ingest(corpus, tmp_path)
    tid = corpus.add_translation(cid, "no", "Katten satt på matten.", agent_id="translator-1")
    rows = corpus.get_translations(cid)
    assert any(t["id"] == tid for t in rows)
    nor = corpus.get_translations(cid, target_lang="no")
    assert nor and nor[0]["text"].startswith("Katten")


def test_mark_translation_reviewed(corpus: Corpus, tmp_path: Path) -> None:
    cid = _ingest(corpus, tmp_path)
    tid = corpus.add_translation(cid, "fr", "Le chat est assis.", agent_id="t-a")
    corpus.mark_translation_reviewed(tid, reviewer_agent_id="t-b")
    rows = corpus.get_translations(cid, target_lang="fr")
    assert rows[0]["status"] == "reviewed"


def test_assemble_translated_document(corpus: Corpus, tmp_path: Path) -> None:
    p = tmp_path / "d.md"
    p.write_text("# A\n\nFirst.\n\n# B\n\nSecond.\n", encoding="utf-8")
    r = corpus.ingest(p)
    # Translate only the first chunk
    chunks = corpus.cypher(
        "MATCH (c:Chunk) WHERE c.doc_id = $id RETURN c.id AS id, c.page_number AS p ORDER BY p",
        params={"id": r.doc_id},
    ).to_list()
    corpus.add_translation(chunks[0]["id"], "no", "Først.", agent_id="t")
    out = corpus.assemble_translated_document(r.doc_id, target_lang="no")
    assert out["missing_translation_count"] == 1
    assert any(c["translated"] for c in out["chunks"])
    assert any(not c["translated"] for c in out["chunks"])
