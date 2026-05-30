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
    corpus.synthesize_study(sid, agent_id="lead")  # clears the conclude gate
    cid = corpus.conclude_study(sid, "Evidence supports the claim.", agent_id="lead")
    assert isinstance(cid, str)

    study = corpus.get_study(sid)
    assert len(study["conclusions"]) == 1
    assert study["conclusions"][0]["text"] == "Evidence supports the claim."
    # the conclusion is a Summary → verifiable via the summary machinery
    v = corpus.verify_summary(cid, verdict="verified", verifier_agent_id="checker")
    assert v["status"] == "verified"


def test_synthesis_gate_blocks_conclude(corpus: Corpus, tmp_path: Path) -> None:
    from kglite_docs.errors import SynthesisRequiredError
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    corpus.assess(sid, ch[0], stance="supports", weight=0.9, agent_id="a1")
    assert corpus.get_study(sid)["synthesis_status"] == "pending"
    # The happy path can't silently skip the cross-chunk pass.
    with pytest.raises(SynthesisRequiredError):
        corpus.conclude_study(sid, "done", agent_id="lead")
    # synthesize clears the gate.
    corpus.synthesize_study(sid, agent_id="lead", note="ledger reviewed")
    assert corpus.get_study(sid)["synthesis_status"] == "done"
    assert isinstance(corpus.conclude_study(sid, "done", agent_id="lead"), str)


def test_synthesis_skip_is_recorded(corpus: Corpus, tmp_path: Path) -> None:
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    corpus.assess(sid, ch[0], stance="supports", weight=0.5, agent_id="a1")
    # An override is allowed but never silent — it leaves an audited event.
    corpus.conclude_study(sid, "done", agent_id="lead", acknowledge_no_synthesis=True)
    events = corpus.get_study(sid)["synthesis_events"]
    assert [e["kind"] for e in events] == ["acknowledged_skip"]
    assert events[0]["by_agent"] == "lead"


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
    corpus.conclude_study(s1, "done", agent_id="lead", acknowledge_no_synthesis=True)

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


def test_deferred_stance_tallied_distinctly(corpus: Corpus, tmp_path: Path) -> None:
    """FEAT-7: `deferred` is counted on its own, not folded into neutral."""
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    corpus.assess(sid, ch[0], stance="supports", weight=0.9, agent_id="a1")
    corpus.assess(sid, ch[1], stance="neutral", weight=0.1, agent_id="a1")
    corpus.assess(sid, ch[2], stance="deferred", weight=0.4, agent_id="a1",
                  rationale="image-only — needs OCR")

    tallies = corpus.study_ledger(sid)["tallies"]
    assert tallies["deferred"] == 1
    assert tallies["deferred_weight"] == 0.4
    assert tallies["neutral"] == 1          # deferred not folded into neutral

    # The new label routes through label_for, so stance filtering works.
    deferred_rows = corpus.study_ledger(sid, stance="deferred")["rows"]
    assert [r["chunk_id"] for r in deferred_rows] == [ch[2]]
    assert deferred_rows[0]["stance"] == "deferred"


def test_deferred_chunk_stays_in_work_list(corpus: Corpus, tmp_path: Path) -> None:
    """FEAT-7: a deferred chunk is parked, not done — it reappears in the
    work-list, whereas a real stance removes the chunk."""
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    corpus.assess(sid, ch[0], stance="deferred", weight=0.3, agent_id="a1")
    corpus.assess(sid, ch[1], stance="supports", weight=0.8, agent_id="a1")

    remaining = {r["id"] for r in corpus.next_unassessed(sid, limit=1000)}
    assert ch[0] in remaining, "deferred chunk must stay in the work-list"
    assert ch[1] not in remaining, "a judged chunk must drop out"


def test_provenance_round_trips_into_ledger(corpus: Corpus, tmp_path: Path) -> None:
    """FEAT-4: each provenance value is recorded and surfaced per ledger row."""
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    corpus.assess(sid, ch[0], stance="supports", weight=0.9, agent_id="a1",
                  provenance="primary_text")
    corpus.assess(sid, ch[1], stance="supports", weight=0.8, agent_id="a1",
                  provenance="characterization")
    corpus.assess(sid, ch[2], stance="against", weight=0.7, agent_id="a1",
                  provenance="scanned_unread")
    corpus.assess(sid, ch[3], stance="neutral", weight=0.1, agent_id="a1")  # default

    prov = {r["chunk_id"]: r["provenance"] for r in corpus.study_ledger(sid)["rows"]}
    assert prov[ch[0]] == "primary_text"
    assert prov[ch[1]] == "characterization"
    assert prov[ch[2]] == "scanned_unread"
    assert prov[ch[3]] == "primary_text"   # default when omitted


def test_provenance_validation(corpus: Corpus, tmp_path: Path) -> None:
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    with pytest.raises(InvalidEnumError):
        corpus.assess(sid, ch[0], stance="supports", weight=0.5, agent_id="a1",
                      provenance="hearsay")


def test_verify_records_provenance_without_changing_assessment(
    corpus: Corpus, tmp_path: Path,
) -> None:
    """FEAT-4: the verifier's provenance is stored on the event; the
    assessment's own provenance is untouched by verification."""
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    corpus.assess(sid, ch[0], stance="supports", weight=0.9, agent_id="a1",
                  provenance="characterization")
    aid = corpus.study_ledger(sid)["rows"][0]["assessment_id"]

    res = corpus.verify_assessment(aid, verdict="verified", verifier_agent_id="checker",
                                   provenance="primary_text")
    assert res["provenance"] == "primary_text"
    # The event carries it.
    ev = corpus.cypher(
        "MATCH (:Assessment {id: $id})-[:HAS_VERIFICATION]->(e:VerificationEvent) "
        "RETURN e.provenance AS p", params={"id": aid},
    ).to_list()
    assert ev and ev[0]["p"] == "primary_text"
    # The assessment's own provenance is unchanged.
    assert corpus.study_ledger(sid)["rows"][0]["provenance"] == "characterization"


def test_supersede_resolves_cross_agent_correction(corpus: Corpus, tmp_path: Path) -> None:
    """FEAT-5/BUG-4: agent a2 supersedes a1's row → one current winner per chunk;
    the old is hidden by default, shown with include_superseded, never deleted."""
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    first = corpus.assess(sid, ch[0], stance="supports", weight=0.9, agent_id="prefilter")
    old_id = first["assessment_id"]
    new = corpus.supersede_assessment(old_id, stance="against", weight=0.6,
                                      agent_id="analyst", rationale="prefilter was wrong")
    assert new["supersedes"] == old_id

    # Default ledger: one current row for this chunk — the correction (a2).
    cur = [r for r in corpus.study_ledger(sid)["rows"] if r["chunk_id"] == ch[0]]
    assert len(cur) == 1
    assert cur[0]["assessment_id"] == new["assessment_id"]
    assert cur[0]["stance"] == "against" and cur[0]["superseded"] is False

    # History: both rows; the old one flagged superseded.
    hist = {r["assessment_id"]: r for r in
            corpus.study_ledger(sid, include_superseded=True)["rows"]
            if r["chunk_id"] == ch[0]}
    assert set(hist) == {old_id, new["assessment_id"]}
    assert hist[old_id]["superseded"] is True
    assert hist[new["assessment_id"]]["superseded"] is False


def test_supersede_tallies_and_counts_are_current(corpus: Corpus, tmp_path: Path) -> None:
    """Tallies + total/returned reflect the post-supersede current set."""
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    a = corpus.assess(sid, ch[0], stance="supports", weight=0.9, agent_id="a1")
    corpus.supersede_assessment(a["assessment_id"], stance="against", weight=0.5, agent_id="a2")

    led = corpus.study_ledger(sid)
    assert led["tallies"]["supports"] == 0   # the superseded support is gone
    assert led["tallies"]["against"] == 1
    assert led["total"] == led["returned"] == 1
    # include_superseded widens the count.
    assert corpus.study_ledger(sid, include_superseded=True)["total"] == 2


def test_supersede_keeps_audit_edge_and_old_node(corpus: Corpus, tmp_path: Path) -> None:
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    a = corpus.assess(sid, ch[0], stance="neutral", weight=0.2, agent_id="a1")
    old_id = a["assessment_id"]
    new = corpus.supersede_assessment(old_id, stance="supports", weight=0.8, agent_id="a2")
    edge = corpus.cypher(
        "MATCH (n:Assessment {id: $new})-[:SUPERSEDES]->(o:Assessment {id: $old}) "
        "RETURN o.id AS old", params={"new": new["assessment_id"], "old": old_id},
    ).to_list()
    assert edge and edge[0]["old"] == old_id           # edge exists
    still = corpus.cypher("MATCH (o:Assessment {id: $id}) RETURN o.id AS id",
                          params={"id": old_id}).to_list()
    assert still and still[0]["id"] == old_id          # old node not deleted


def test_supersede_unknown_assessment_raises(corpus: Corpus, tmp_path: Path) -> None:
    _ingest_chunks(corpus, tmp_path)
    with pytest.raises(InvalidEnumError):
        corpus.supersede_assessment("nope", stance="supports", weight=0.5, agent_id="a1")


def test_conflicts_surfaces_opposing_assessments(corpus: Corpus, tmp_path: Path) -> None:
    """FEAT-8: a chunk with both supports and against is a conflict; agreement
    is not."""
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    # ch[0]: contested (a1 supports, a2 against)
    corpus.assess(sid, ch[0], stance="supports", weight=0.8, agent_id="a1", rationale="for")
    corpus.assess(sid, ch[0], stance="against", weight=0.6, agent_id="a2", rationale="vs")
    # ch[1]: agreement (both supports) — not a conflict
    corpus.assess(sid, ch[1], stance="supports", weight=0.7, agent_id="a1")
    corpus.assess(sid, ch[1], stance="supports", weight=0.5, agent_id="a2")
    # ch[2]: supports + neutral — not a conflict
    corpus.assess(sid, ch[2], stance="supports", weight=0.4, agent_id="a1")
    corpus.assess(sid, ch[2], stance="neutral", weight=0.1, agent_id="a2")

    rep = corpus.study_conflicts(sid)
    assert rep["total"] == 1
    c = rep["conflicts"][0]
    assert c["chunk_id"] == ch[0]
    assert {r["by_agent"] for r in c["supports"]} == {"a1"}
    assert {r["by_agent"] for r in c["against"]} == {"a2"}
    assert c["supports"][0]["provenance"] == "primary_text"


def test_conflicts_resolved_by_supersede(corpus: Corpus, tmp_path: Path) -> None:
    """A correction that removes the opposition clears the conflict."""
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    corpus.assess(sid, ch[0], stance="supports", weight=0.8, agent_id="a1")
    against = corpus.assess(sid, ch[0], stance="against", weight=0.6, agent_id="a2")
    assert corpus.study_conflicts(sid)["total"] == 1

    # a2 corrects their against → supports; no opposition remains.
    corpus.supersede_assessment(against["assessment_id"], stance="supports",
                                weight=0.7, agent_id="a2")
    assert corpus.study_conflicts(sid)["total"] == 0


def test_conflicts_unknown_study_raises(corpus: Corpus, tmp_path: Path) -> None:
    _ingest_chunks(corpus, tmp_path)
    with pytest.raises(InvalidEnumError):
        corpus.study_conflicts("nope")


def _chunk_text(corpus: Corpus, chunk_id: str) -> str:
    return corpus.cypher(
        "MATCH (c:Chunk {id: $id}) RETURN c.text AS t", params={"id": chunk_id}
    ).to_list()[0]["t"]


def test_pinpoint_span_offsets_round_trip(corpus: Corpus, tmp_path: Path) -> None:
    """FEAT-6: a char span is stored and surfaced per ledger row."""
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    text = _chunk_text(corpus, ch[0])
    quote = text[5:25]
    corpus.assess(sid, ch[0], stance="supports", weight=0.9, agent_id="a1",
                  char_start=5, char_end=25)
    row = corpus.study_ledger(sid)["rows"][0]
    assert row["char_start"] == 5 and row["char_end"] == 25
    assert row["quote"] == quote          # filled from the cited text


def test_pinpoint_quote_only_is_located(corpus: Corpus, tmp_path: Path) -> None:
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    text = _chunk_text(corpus, ch[0])
    quote = text[10:30]
    corpus.assess(sid, ch[0], stance="against", weight=0.5, agent_id="a1", quote=quote)
    row = corpus.study_ledger(sid)["rows"][0]
    assert row["quote"] == quote
    assert text[row["char_start"]:row["char_end"]] == quote


def test_pinpoint_no_span_defaults(corpus: Corpus, tmp_path: Path) -> None:
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    corpus.assess(sid, ch[0], stance="neutral", weight=0.1, agent_id="a1")
    row = corpus.study_ledger(sid)["rows"][0]
    assert row["quote"] == "" and row["char_start"] == -1 and row["char_end"] == -1


def test_pinpoint_span_validation(corpus: Corpus, tmp_path: Path) -> None:
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    n = len(_chunk_text(corpus, ch[0]))
    with pytest.raises(InvalidEnumError):       # out of range
        corpus.assess(sid, ch[0], stance="supports", weight=0.5, agent_id="a1",
                      char_start=0, char_end=n + 50)
    with pytest.raises(InvalidEnumError):       # quote not in chunk
        corpus.assess(sid, ch[0], stance="supports", weight=0.5, agent_id="a1",
                      quote="this phrase is definitely not present in the chunk zzz")


def test_assess_many_batches_one_write(corpus: Corpus, tmp_path: Path) -> None:
    """FEAT-12: a batch assess writes N assessments and round-trips through the
    ledger (tallies, provenance, span included)."""
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    res = corpus.assess_many(sid, [
        {"chunk_id": ch[0], "stance": "supports", "weight": 0.9, "agent_id": "a1"},
        {"chunk_id": ch[1], "stance": "against", "weight": 0.6, "agent_id": "a1",
         "provenance": "characterization"},
        {"chunk_id": ch[2], "stance": "neutral", "weight": 0.2, "agent_id": "a2",
         "char_start": 0, "char_end": 5},
    ])
    assert res["created"] == 3
    led = corpus.study_ledger(sid)
    assert led["total"] == 3
    assert led["tallies"]["supports"] == 1 and led["tallies"]["against"] == 1
    by_chunk = {r["chunk_id"]: r for r in led["rows"]}
    assert by_chunk[ch[1]]["provenance"] == "characterization"
    assert by_chunk[ch[2]]["char_start"] == 0 and by_chunk[ch[2]]["char_end"] == 5


def test_assess_many_bad_row_writes_nothing(corpus: Corpus, tmp_path: Path) -> None:
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    with pytest.raises(InvalidEnumError):
        corpus.assess_many(sid, [
            {"chunk_id": ch[0], "stance": "supports", "weight": 0.9, "agent_id": "a1"},
            {"chunk_id": ch[1], "stance": "maybe", "weight": 0.5, "agent_id": "a1"},  # bad
        ])
    # Nothing was written — the whole batch aborted before any upsert.
    assert corpus.study_ledger(sid)["total"] == 0


def test_assess_many_empty_and_missing_field(corpus: Corpus, tmp_path: Path) -> None:
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    assert corpus.assess_many(sid, [])["created"] == 0
    with pytest.raises(InvalidEnumError):
        corpus.assess_many(sid, [{"chunk_id": ch[0], "stance": "supports"}])  # no weight/agent


def test_ledger_reports_total_and_returned_on_truncation(corpus: Corpus, tmp_path: Path) -> None:
    """BUG-3: a clipped ledger must say so — total > returned, not a silent cut."""
    ch = _ingest_chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    for i in range(6):
        corpus.assess(sid, ch[i], stance="supports", weight=0.5 + i / 100, agent_id="a1")

    clipped = corpus.study_ledger(sid, limit=3)
    assert clipped["returned"] == 3
    assert len(clipped["rows"]) == 3
    assert clipped["total"] == 6          # truncation is observable
    assert clipped["total"] > clipped["returned"]

    full = corpus.study_ledger(sid, limit=200)
    assert full["total"] == full["returned"] == 6   # nothing hidden


def _doc_id_of(corpus: Corpus, chunk_id: str) -> str:
    return corpus.cypher(
        "MATCH (c:Chunk {id: $id}) RETURN c.doc_id AS d", params={"id": chunk_id}
    ).to_list()[0]["d"]


def test_ledger_doc_id_scope(corpus: Corpus, tmp_path: Path) -> None:
    """BUG-3: doc_id scopes the ledger (rows + total) to one document."""
    ch_a = _ingest_chunks(corpus, tmp_path)
    doc_a = _doc_id_of(corpus, ch_a[0])

    # A second, distinct document.
    p2 = tmp_path / "doc2.md"
    p2.write_text("# Doc2\n\n" + "\n\n".join(f"Other paragraph {i}." for i in range(250)),
                  encoding="utf-8")
    corpus.ingest(p2)
    ch_b = [r["id"] for r in corpus.cypher(
        "MATCH (c:Chunk:Ready) WHERE c.doc_id <> $da RETURN c.id AS id ORDER BY c.chunk_index",
        params={"da": doc_a},
    ).to_list()]

    sid = corpus.define_study("Q", created_by="lead")
    corpus.assess(sid, ch_a[0], stance="supports", weight=0.9, agent_id="a1")
    corpus.assess(sid, ch_a[1], stance="against", weight=0.7, agent_id="a1")
    corpus.assess(sid, ch_b[0], stance="supports", weight=0.8, agent_id="a1")

    scoped = corpus.study_ledger(sid, doc_id=doc_a)
    assert scoped["total"] == scoped["returned"] == 2
    assert all(r["doc_id"] == doc_a for r in scoped["rows"])
    assert corpus.study_ledger(sid)["total"] == 3   # unscoped sees both docs
