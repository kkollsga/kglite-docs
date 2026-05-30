"""FEAT-14: result/detail ergonomics — cypher ResultView is iterable/indexable
and get_chunk supports both item and attribute access."""

from __future__ import annotations

from pathlib import Path

import pytest

from kglite_docs import Corpus


def _ingest(corpus: Corpus, tmp_path: Path) -> str:
    p = tmp_path / "d.md"
    p.write_text("# Topic\n\nSome body words to chunk here.\n", encoding="utf-8")
    return corpus.ingest(p).doc_id


def test_cypher_resultview_is_iterable_and_indexable(corpus: Corpus, tmp_path: Path) -> None:
    _ingest(corpus, tmp_path)
    res = corpus.cypher("MATCH (c:Chunk) RETURN c.id AS id, c.doc_id AS doc_id")
    rows = [row for row in res]          # iterable — each row a dict
    assert rows and rows[0]["id"]
    assert res[0]["id"] == rows[0]["id"]  # index access
    assert len(res) == len(rows)          # len
    assert res.to_list() == rows          # to_list still works


def test_get_chunk_supports_item_and_attr_access(corpus: Corpus, tmp_path: Path) -> None:
    doc_id = _ingest(corpus, tmp_path)
    cid = corpus.cypher(
        "MATCH (c:Chunk:Ready) WHERE c.doc_id = $d RETURN c.id AS id",
        params={"d": doc_id},
    ).to_list()[0]["id"]
    detail = corpus.get_chunk(cid)
    assert detail is not None
    assert isinstance(detail, dict)               # still a dict
    assert detail["section_id"] == detail.section_id   # both access styles agree
    assert detail.id == cid
    with pytest.raises(AttributeError):
        _ = detail.does_not_exist
