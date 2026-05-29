"""Evidence-study workflow tests — run entirely on the stub embedder
(no model load), which also locks in the "assess needs no embeddings" claim."""

from __future__ import annotations

from pathlib import Path

import pytest

from kglite_docs import Corpus
from kglite_docs.errors import InvalidEnumError, SelfVerificationError


def _ingest_chunks(corpus: Corpus, tmp_path: Path, n_paras: int = 250) -> list[str]:
    """Ingest a doc large enough to yield several chunks; return chunk ids."""
    p = tmp_path / "doc.md"
    para = (
        "Paragraph {i}: dense passage retrieval compresses a passage into a "
        "single vector, while late interaction keeps per-token embeddings for "
        "fine-grained matching, point number {i} with extra filler words."
    )
    p.write_text("# Doc\n\n" + "\n\n".join(para.format(i=i) for i in range(n_paras)), encoding="utf-8")
    corpus.ingest(p)  # no embed — study workflow doesn't need it
    rows = corpus.cypher(
        "MATCH (c:Chunk:Ready) RETURN c.id AS id ORDER BY c.chunk_index"
    ).to_list()
    ids = [r["id"] for r in rows]
    assert len(ids) >= 4, f"need several chunks, got {len(ids)}"
    return ids


def test_define_assess_ledger_ranking_and_tallies(corpus: Corpus, tmp_path: Path) -> None:
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Late interaction is necessary", created_by="lead")
    corpus.assess(sid, ch[0], stance="supports", weight=0.9, agent_id="a1", rationale="strong")
    corpus.assess(sid, ch[1], stance="against", weight=0.7, agent_id="a1", rationale="counter")
    corpus.assess(sid, ch[2], stance="supports", weight=0.4, agent_id="a1", rationale="weak")

    led = corpus.study_ledger(sid)
    weights = [r["weight"] for r in led["rows"]]
    assert weights == sorted(weights, reverse=True), "ledger must rank by weight DESC"
    assert led["tallies"]["supports"] == 2
    assert led["tallies"]["against"] == 1
    assert round(led["tallies"]["supports_weight"], 2) == 1.3
    # rationale + author come through
    top = led["rows"][0]
    assert top["weight"] == 0.9 and top["by_agent"] == "a1" and top["rationale"] == "strong"


def test_stance_filter_returns_supporting_or_contradicting(corpus: Corpus, tmp_path: Path) -> None:
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    corpus.assess(sid, ch[0], stance="supports", weight=0.8, agent_id="a1")
    corpus.assess(sid, ch[1], stance="against", weight=0.6, agent_id="a1")
    corpus.assess(sid, ch[2], stance="supports", weight=0.5, agent_id="a1")

    assert len(corpus.study_ledger(sid, stance="supports")["rows"]) == 2
    assert len(corpus.study_ledger(sid, stance="against")["rows"]) == 1
    assert len(corpus.study_ledger(sid, min_weight=0.7)["rows"]) == 1


def test_multi_agent_coexistence(corpus: Corpus, tmp_path: Path) -> None:
    """Two agents assessing the same chunk → two distinct Assessments."""
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    corpus.assess(sid, ch[0], stance="supports", weight=0.9, agent_id="a1")
    corpus.assess(sid, ch[0], stance="against", weight=0.3, agent_id="a2")
    rows = [r for r in corpus.study_ledger(sid)["rows"] if r["chunk_id"] == ch[0]]
    assert len(rows) == 2
    assert {r["by_agent"] for r in rows} == {"a1", "a2"}


def test_latest_wins_dedup(corpus: Corpus, tmp_path: Path) -> None:
    """Same agent re-assessing the same chunk → ledger shows only the latest."""
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    corpus.assess(sid, ch[0], stance="supports", weight=0.2, agent_id="a1", rationale="first")
    corpus.assess(sid, ch[0], stance="against", weight=0.8, agent_id="a1", rationale="revised")
    rows = [r for r in corpus.study_ledger(sid)["rows"] if r["chunk_id"] == ch[0]]
    assert len(rows) == 1
    assert rows[0]["stance"] == "against" and rows[0]["weight"] == 0.8
    assert rows[0]["rationale"] == "revised"


def test_verify_and_self_verify_rejected(corpus: Corpus, tmp_path: Path) -> None:
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    corpus.assess(sid, ch[0], stance="supports", weight=0.9, agent_id="a1")
    aid = corpus.study_ledger(sid)["rows"][0]["assessment_id"]

    res = corpus.verify_assessment(aid, verdict="verified", verifier_agent_id="checker")
    assert res["status"] == "verified"
    assert corpus.study_ledger(sid)["rows"][0]["verification_status"] == "verified"
    assert len(corpus.study_ledger(sid, verified_only=True)["rows"]) == 1

    with pytest.raises(SelfVerificationError):
        corpus.verify_assessment(aid, verdict="verified", verifier_agent_id="a1")


def test_verify_duplicate_verdict(corpus: Corpus, tmp_path: Path) -> None:
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    corpus.assess(sid, ch[0], stance="supports", weight=0.5, agent_id="a1")
    aid = corpus.study_ledger(sid)["rows"][0]["assessment_id"]
    corpus.verify_assessment(aid, verdict="duplicate", verifier_agent_id="checker")
    assert corpus.study_ledger(sid)["rows"][0]["verification_status"] == "duplicate"


def test_next_unassessed_is_resumable(corpus: Corpus, tmp_path: Path) -> None:
    ch = _ingest_chunks(corpus, tmp_path)
    total = len(ch)
    sid = corpus.define_study("Q", created_by="lead")
    assert len(corpus.next_unassessed(sid, limit=1000)) == total
    corpus.assess(sid, ch[0], stance="supports", weight=0.5, agent_id="a1")
    corpus.assess(sid, ch[1], stance="against", weight=0.5, agent_id="a1")
    remaining = corpus.next_unassessed(sid, limit=1000)
    assert len(remaining) == total - 2
    assert ch[0] not in {r["id"] for r in remaining}
    # ordered by reading position
    pages = [(r["page"], r["chunk_index"]) for r in remaining]
    assert pages == sorted(pages)


def test_punchcard_no_overlap(corpus: Corpus, tmp_path: Path) -> None:
    """Two agents claiming via next(agent_id=...) get DISJOINT batches."""
    _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    b1 = corpus.next_unassessed(sid, agent_id="a1", limit=3)
    b2 = corpus.next_unassessed(sid, agent_id="a2", limit=3)
    ids1, ids2 = {r["id"] for r in b1}, {r["id"] for r in b2}
    assert len(ids1) == 3 and len(ids2) == 3
    assert ids1.isdisjoint(ids2), "punchcard claims must not overlap"


def test_next_preview_does_not_claim(corpus: Corpus, tmp_path: Path) -> None:
    """next() without agent_id is a read-only preview — no checkout written."""
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    prev = corpus.next_unassessed(sid, limit=1000)
    assert len(prev) == len(ch)
    assert corpus.cypher("MATCH (co:Checkout) RETURN count(co) AS n").to_list()[0]["n"] == 0
    # a claiming agent still gets chunks (preview locked nothing)
    assert len(corpus.next_unassessed(sid, agent_id="a1", limit=3)) == 3


def test_punchcard_ttl_expiry(corpus: Corpus, tmp_path: Path) -> None:
    """A stale claim (older than ttl) is reclaimable — abandoned work frees up."""
    _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    b1 = corpus.next_unassessed(sid, agent_id="a1", limit=3)
    ids1 = {r["id"] for r in b1}
    # ttl_seconds=0 → a1's checkout is already expired → a2 reclaims the same chunks
    b2 = corpus.next_unassessed(sid, agent_id="a2", limit=3, ttl_seconds=0)
    assert {r["id"] for r in b2} == ids1


def test_assess_releases_claim(corpus: Corpus, tmp_path: Path) -> None:
    """Assessing a claimed chunk excludes it; an unassessed claim stays locked."""
    _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    b1 = corpus.next_unassessed(sid, agent_id="a1", limit=2)
    assessed, still_claimed = b1[0]["id"], b1[1]["id"]
    corpus.assess(sid, assessed, stance="supports", weight=0.5, agent_id="a1")
    got = {r["id"] for r in corpus.next_unassessed(sid, agent_id="a2", limit=100)}
    assert assessed not in got       # assessed → excluded (implicit release)
    assert still_claimed not in got  # claimed by a1, not yet expired → still locked


def test_delete_removes_checkouts(corpus: Corpus, tmp_path: Path) -> None:
    _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    corpus.next_unassessed(sid, agent_id="a1", limit=3)
    assert corpus.cypher("MATCH (co:Checkout) RETURN count(co) AS n").to_list()[0]["n"] == 1
    corpus.delete_study(sid)
    assert corpus.cypher("MATCH (co:Checkout) RETURN count(co) AS n").to_list()[0]["n"] == 0


def test_assess_with_context_records_span_dedups_and_backlinks(corpus: Corpus, tmp_path: Path) -> None:
    """Recording context_chunk_ids: ledger surfaces the span, the context
    chunks are excluded from the work-list, and they backlink to the assessment."""
    ch = _ingest_chunks(corpus, tmp_path)
    focal, ctx1, ctx2 = ch[0], ch[1], ch[2]
    sid = corpus.define_study("Q", created_by="lead")
    corpus.assess(
        sid, focal, stance="supports", weight=0.8, agent_id="a1",
        rationale="only makes sense with the next two paragraphs",
        context_chunk_ids=[ctx1, ctx2],
    )
    # Ledger surfaces the span on the focal row
    row = corpus.study_ledger(sid)["rows"][0]
    assert row["chunk_id"] == focal
    assert set(row["context_chunk_ids"]) == {ctx1, ctx2}

    # Dedup: context chunks are NOT handed out as fresh work (covered)
    remaining = {r["id"] for r in corpus.next_unassessed(sid, agent_id="a2", limit=1000)}
    assert focal not in remaining
    assert ctx1 not in remaining and ctx2 not in remaining

    # Backlink: from a context chunk, recover the assessment that used it
    back = corpus.cypher(
        "MATCH (c:Chunk {id: $id})<-[:USED_CONTEXT]-(a:Assessment) RETURN a.chunk_id AS focal",
        params={"id": ctx1},
    ).to_list()
    assert back and back[0]["focal"] == focal


def test_get_chunk_window_returns_neighbor_text(corpus: Corpus, tmp_path: Path) -> None:
    """window=N returns the N chunks before/after in reading order, with text."""
    ch = _ingest_chunks(corpus, tmp_path)
    mid = ch[2]
    detail = corpus.get_chunk(mid, window=2)
    assert detail is not None
    before_ids = [c["id"] for c in detail["context_before"]]
    after_ids = [c["id"] for c in detail["context_after"]]
    assert before_ids == [ch[0], ch[1]]
    assert after_ids == [ch[3], ch[4]]
    # text is included so the agent can actually read the context
    assert all(c.get("text") for c in detail["context_before"] + detail["context_after"])


def test_conclude_writes_verifiable_summary(corpus: Corpus, tmp_path: Path) -> None:
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    corpus.assess(sid, ch[0], stance="supports", weight=0.9, agent_id="a1")
    cid = corpus.conclude_study(sid, "Evidence supports the claim.", agent_id="lead")
    assert isinstance(cid, str)

    study = corpus.get_study(sid)
    assert len(study["conclusions"]) == 1
    assert study["conclusions"][0]["text"] == "Evidence supports the claim."
    # the conclusion is a Summary → verifiable via the summary machinery
    v = corpus.verify_summary(cid, verdict="verified", verifier_agent_id="checker")
    assert v["status"] == "verified"


def test_reopen_and_list(corpus: Corpus, tmp_path: Path) -> None:
    sid = corpus.define_study("Q", created_by="lead", status="closed")
    assert corpus.get_study(sid)["status"] == "closed"
    assert len(corpus.list_studies(status="closed")) == 1
    assert len(corpus.list_studies(status="open")) == 0
    corpus.reopen_study(sid, agent_id="lead")
    assert corpus.get_study(sid)["status"] == "open"
    assert len(corpus.list_studies(status="open")) == 1


def test_delete_cascades_and_isolates(corpus: Corpus, tmp_path: Path) -> None:
    ch = _ingest_chunks(corpus, tmp_path)
    s1 = corpus.define_study("Q1", created_by="lead")
    s2 = corpus.define_study("Q2", created_by="lead")
    corpus.assess(s1, ch[0], stance="supports", weight=0.9, agent_id="a1")
    corpus.assess(s2, ch[0], stance="against", weight=0.5, agent_id="a1")  # same chunk, other study
    aid = corpus.study_ledger(s1)["rows"][0]["assessment_id"]
    corpus.verify_assessment(aid, verdict="verified", verifier_agent_id="checker")
    corpus.conclude_study(s1, "done", agent_id="lead")

    res = corpus.delete_study(s1)
    assert res["assessments"] == 1 and res["conclusions"] == 1 and res["events"] == 1
    assert len(corpus.list_studies()) == 1
    assert corpus.get_study(s1) is None
    # s2 untouched (cross-study isolation)
    assert len(corpus.study_ledger(s2)["rows"]) == 1
    # no orphaned assessments / verification events
    assert corpus.cypher("MATCH (a:Assessment) RETURN count(a) AS n").to_list()[0]["n"] == 1
    assert corpus.cypher("MATCH (v:VerificationEvent) RETURN count(v) AS n").to_list()[0]["n"] == 0


def test_cross_study_isolation_same_chunk(corpus: Corpus, tmp_path: Path) -> None:
    ch = _ingest_chunks(corpus, tmp_path)
    s1 = corpus.define_study("Q1", created_by="lead")
    s2 = corpus.define_study("Q2", created_by="lead")
    corpus.assess(s1, ch[0], stance="supports", weight=0.9, agent_id="a1")
    corpus.assess(s2, ch[0], stance="against", weight=0.2, agent_id="a1")
    assert corpus.study_ledger(s1)["rows"][0]["stance"] == "supports"
    assert corpus.study_ledger(s2)["rows"][0]["stance"] == "against"


def test_validation(corpus: Corpus, tmp_path: Path) -> None:
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    with pytest.raises(InvalidEnumError):
        corpus.assess(sid, ch[0], stance="maybe", weight=0.5, agent_id="a1")
    with pytest.raises(InvalidEnumError):
        corpus.assess(sid, ch[0], stance="supports", weight=1.5, agent_id="a1")
    with pytest.raises(InvalidEnumError):
        corpus.define_study("", created_by="lead")
