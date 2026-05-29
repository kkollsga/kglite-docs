"""Tags + agent registry + view tracking."""

from __future__ import annotations

from pathlib import Path

from kglite_docs import Corpus


def _ingest_md(corpus: Corpus, tmp_path: Path) -> str:
    p = tmp_path / "d.md"
    p.write_text("# T\n\nBody.\n", encoding="utf-8")
    corpus.ingest(p)
    corpus.index()
    return corpus.search("Body", top_k=1)[0]["id"]


def test_tag_chunk_is_idempotent_per_agent(corpus: Corpus, tmp_path: Path) -> None:
    cid = _ingest_md(corpus, tmp_path)
    a = corpus.tag_chunk(cid, "Topic A", kind="topic", agent_id="alice")
    b = corpus.tag_chunk(cid, "Topic A", kind="topic", agent_id="alice")
    assert a["created"] is True
    assert b["created"] is False


def test_two_agents_can_each_apply_same_tag(corpus: Corpus, tmp_path: Path) -> None:
    cid = _ingest_md(corpus, tmp_path)
    corpus.tag_chunk(cid, "shared", agent_id="alice")
    corpus.tag_chunk(cid, "shared", agent_id="bob")
    rows = corpus.list_tags(chunk_id=cid)
    by_agents = {r["by_agent"] for r in rows}
    assert by_agents == {"alice", "bob"}


def test_chunks_by_tag_returns_members(corpus: Corpus, tmp_path: Path) -> None:
    cid = _ingest_md(corpus, tmp_path)
    corpus.tag_chunk(cid, "important", agent_id="alice")
    rows = corpus.chunks_by_tag("important")
    assert any(r["id"] == cid for r in rows)


def test_untag_chunk_removes_application(corpus: Corpus, tmp_path: Path) -> None:
    cid = _ingest_md(corpus, tmp_path)
    corpus.tag_chunk(cid, "tmp", agent_id="alice")
    corpus.untag_chunk(cid, "tmp", agent_id="alice")
    rows = corpus.list_tags(chunk_id=cid, agent_id="alice")
    assert not any(r["tag_id"] == "tmp" for r in rows)


def test_record_view_creates_view_node(corpus: Corpus, tmp_path: Path) -> None:
    cid = _ingest_md(corpus, tmp_path)
    r = corpus.record_view(cid, "alice", context="my query")
    assert r["view_node"]


def test_list_agents_after_activity(corpus: Corpus, tmp_path: Path) -> None:
    cid = _ingest_md(corpus, tmp_path)
    corpus.tag_chunk(cid, "x", agent_id="alice")
    corpus.add_summary(cid, "y", agent_id="bob")
    agents = {a["id"] for a in corpus.list_agents()}
    assert {"alice", "bob"}.issubset(agents)


def test_search_with_agent_id_records_views(corpus: Corpus, tmp_path: Path) -> None:
    _ingest_md(corpus, tmp_path)
    corpus.search("Body", top_k=3, agent_id="claude-x")
    agents = {a["id"] for a in corpus.list_agents()}
    assert "claude-x" in agents
