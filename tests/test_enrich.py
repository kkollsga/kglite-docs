"""Summary writes + verification rules + staleness."""

from __future__ import annotations

from pathlib import Path

import pytest

from kglite_docs import Corpus


def _ingest_md(corpus: Corpus, tmp_path: Path, name: str = "doc") -> str:
    p = tmp_path / f"{name}.md"
    p.write_text("# T\n\nFirst paragraph.\n\n# U\n\nSecond paragraph.\n", encoding="utf-8")
    corpus.ingest(p)
    corpus.index()
    hits = corpus.search("First paragraph", top_k=1)
    assert hits
    return hits[0]["id"]


def test_add_summary_returns_id_and_stores(corpus: Corpus, tmp_path: Path) -> None:
    cid = _ingest_md(corpus, tmp_path)
    sid = corpus.add_summary(cid, "summary text", agent_id="alice", model="m")
    assert isinstance(sid, str) and sid
    sums = corpus.get_summaries(cid)
    assert any(s["id"] == sid for s in sums)


def test_self_verification_rejected(corpus: Corpus, tmp_path: Path) -> None:
    from kglite_docs.errors import SelfVerificationError
    cid = _ingest_md(corpus, tmp_path)
    sid = corpus.add_summary(cid, "x", agent_id="alice")
    with pytest.raises(SelfVerificationError):
        corpus.verify_summary(sid, verdict="verified", verifier_agent_id="alice")


def test_verification_by_different_agent(corpus: Corpus, tmp_path: Path) -> None:
    cid = _ingest_md(corpus, tmp_path)
    sid = corpus.add_summary(cid, "x", agent_id="alice")
    r = corpus.verify_summary(sid, verdict="verified", verifier_agent_id="bob", notes="ok")
    assert r["status"] == "verified"
    sums = corpus.get_summaries(cid, status="verified")
    assert any(s["id"] == sid for s in sums)


def test_invalid_verdict_rejected(corpus: Corpus, tmp_path: Path) -> None:
    cid = _ingest_md(corpus, tmp_path)
    sid = corpus.add_summary(cid, "x", agent_id="alice")
    with pytest.raises(ValueError):
        corpus.verify_summary(sid, verdict="bogus", verifier_agent_id="bob")


def test_invalid_depth_rejected(corpus: Corpus, tmp_path: Path) -> None:
    cid = _ingest_md(corpus, tmp_path)
    with pytest.raises(ValueError):
        corpus.add_summary(cid, "x", agent_id="alice", depth="not-a-real-depth")


def test_empty_text_rejected(corpus: Corpus, tmp_path: Path) -> None:
    cid = _ingest_md(corpus, tmp_path)
    with pytest.raises(ValueError):
        corpus.add_summary(cid, "   ", agent_id="alice")
