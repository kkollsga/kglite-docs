"""0.0.12 Phase 2: classify.py — one-pass element classification + punchcard.
Uses a throwaway registered schema (the legal pack lands in a later phase)."""

from __future__ import annotations

from pathlib import Path

import pytest

from kglite_docs import Corpus, classify, schema
from kglite_docs.errors import InvalidEnumError

schema.register_element_discriminator(
    "chunk.test_element", {"alpha": "Alpha", "beta": "Beta", "gamma": "Gamma"}
)


def _ingest(corpus: Corpus, tmp_path: Path, n: int = 6) -> list[str]:
    p = tmp_path / "d.md"
    p.write_text(
        "\n\n".join(
            f"# Section {i}\n\nParagraph {i} with several distinct words for chunk {i} here please."
            for i in range(n)
        ),
        encoding="utf-8",
    )
    corpus.ingest(p, structure_aware=True)  # one chunk per section
    return [r["id"] for r in corpus.cypher(
        "MATCH (c:Chunk:Ready) RETURN c.id AS id ORDER BY c.chunk_index"
    ).to_list()]


def _labels(corpus: Corpus, cid: str) -> set[str]:
    return set(corpus.cypher(
        "MATCH (c:Chunk {id: $id}) RETURN labels(c) AS L", params={"id": cid}
    ).to_list()[0]["L"])


def test_classify_chunk_labels_marker_and_routing(corpus: Corpus, tmp_path: Path) -> None:
    ch = _ingest(corpus, tmp_path)
    r = classify.classify_chunk(corpus.store, chunk_id=ch[0], elements=["alpha", "beta"], agent_id="a1")
    assert r["classified"] is True
    labels = _labels(corpus, ch[0])
    assert {"Alpha", "Beta", "Classified"} <= labels and "Unclassified" not in labels
    # The routing predicate is the whole point.
    assert corpus.cypher("MATCH (c:Chunk:Alpha) RETURN count(c) AS n").to_list()[0]["n"] == 1


def test_empty_elements_is_unclassified(corpus: Corpus, tmp_path: Path) -> None:
    ch = _ingest(corpus, tmp_path)
    r = classify.classify_chunk(corpus.store, chunk_id=ch[0], elements=[], agent_id="a1")
    assert r["classified"] is False
    labels = _labels(corpus, ch[0])
    assert "Unclassified" in labels and "Classified" not in labels


def test_unknown_element_and_missing_chunk_raise(corpus: Corpus, tmp_path: Path) -> None:
    ch = _ingest(corpus, tmp_path)
    with pytest.raises(InvalidEnumError, match="unknown element"):
        classify.classify_chunk(corpus.store, chunk_id=ch[0], elements=["nonsense"], agent_id="a1")
    with pytest.raises(InvalidEnumError, match="chunk not found"):
        classify.classify_chunk(corpus.store, chunk_id="nope", elements=["alpha"], agent_id="a1")


def test_exhaustive_invariant(corpus: Corpus, tmp_path: Path) -> None:
    ch = _ingest(corpus, tmp_path)
    for i, c in enumerate(ch):
        classify.classify_chunk(corpus.store, chunk_id=c, elements=(["alpha"] if i % 2 else []), agent_id="a1")
    n = lambda lbl: corpus.cypher(f"MATCH (c:Chunk:{lbl}) RETURN count(c) AS n").to_list()[0]["n"]  # noqa: E731
    assert n("Classified") + n("Unclassified") == n("Ready")


def test_contested_on_disagreement(corpus: Corpus, tmp_path: Path) -> None:
    ch = _ingest(corpus, tmp_path)
    classify.classify_chunk(corpus.store, chunk_id=ch[0], elements=["alpha"], agent_id="a1")
    classify.classify_chunk(corpus.store, chunk_id=ch[0], elements=["alpha"], agent_id="a2")  # agree
    assert "Contested" not in _labels(corpus, ch[0])
    r = classify.classify_chunk(corpus.store, chunk_id=ch[0], elements=["beta"], agent_id="a3")  # differ
    assert r["contested"] is True and "Contested" in _labels(corpus, ch[0])


def test_punchcard_no_overlap_and_disjoint_from_study(corpus: Corpus, tmp_path: Path) -> None:
    _ingest(corpus, tmp_path, n=6)
    b1 = classify.next_unclassified(corpus.store, agent_id="c1", limit=3)
    b2 = classify.next_unclassified(corpus.store, agent_id="c2", limit=3)
    ids1, ids2 = {r["id"] for r in b1}, {r["id"] for r in b2}
    assert len(ids1) == 3 and len(ids2) == 3 and ids1.isdisjoint(ids2)
    # classify has now claimed all 6 chunks. A *study* work-list must be
    # unaffected — the checkout keys are disjoint.
    sid = corpus.define_study("Q", created_by="lead")
    assert len(corpus.next_unassessed(sid, agent_id="s1", limit=3)) == 3


def test_classify_many(corpus: Corpus, tmp_path: Path) -> None:
    ch = _ingest(corpus, tmp_path)
    res = classify.classify_many(corpus.store, items=[
        {"chunk_id": ch[0], "elements": ["alpha"], "agent_id": "a1"},
        {"chunk_id": ch[1], "elements": ["beta", "gamma"], "agent_id": "a1"},
    ])
    assert res["classified"] == 2
    assert {"Beta", "Gamma"} <= _labels(corpus, ch[1])
