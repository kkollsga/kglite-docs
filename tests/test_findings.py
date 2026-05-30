"""0.0.13 Phase 1: cross-chunk Finding unit — a pattern asserted over a SET of
chunks (what per-chunk assess structurally can't see)."""

from __future__ import annotations

from pathlib import Path

import pytest

from kglite_docs import Corpus, classify, schema
from kglite_docs.errors import InvalidEnumError
from kglite_docs.testing import make_chunks

schema.register_element_discriminator("chunk.sc_element", {"rulex": "RuleX"})


def _chunks(corpus: Corpus, tmp_path: Path, n: int = 4) -> list[str]:
    return make_chunks(corpus, n)


def test_finding_round_trips_with_supporting_chunks(corpus: Corpus, tmp_path: Path) -> None:
    ch = _chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    r = corpus.create_finding(
        sid, statement="The court treated the parties' absences unequally",
        supporting_chunk_ids=[ch[0], ch[2]], stance="against", weight=0.9,
        agent_id="synth", finding_type="disparate_treatment", rationale="A vs C",
    )
    assert r["finding_id"].startswith("finding_")
    findings = corpus.list_findings(sid)
    assert len(findings) == 1
    f = findings[0]
    assert f["statement"].startswith("The court treated")
    assert f["finding_type"] == "disparate_treatment" and f["stance"] == "against"
    assert {s["id"] for s in f["supporting"]} == {ch[0], ch[2]}
    assert all("page" in s for s in f["supporting"])
    # finding_type is a routing label.
    assert corpus.cypher(
        "MATCH (f:Finding:DisparateTreatment) RETURN count(f) AS n"
    ).to_list()[0]["n"] == 1
    # Surfaced in get_study.
    assert len(corpus.get_study(sid)["findings"]) == 1


def test_finding_type_filter(corpus: Corpus, tmp_path: Path) -> None:
    ch = _chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    corpus.create_finding(sid, statement="disparate", supporting_chunk_ids=[ch[0], ch[1]],
                          stance="against", weight=0.8, agent_id="s", finding_type="disparate_treatment")
    corpus.create_finding(sid, statement="contradiction", supporting_chunk_ids=[ch[1], ch[2]],
                          stance="against", weight=0.7, agent_id="s", finding_type="contradiction")
    assert len(corpus.list_findings(sid)) == 2
    only = corpus.list_findings(sid, finding_type="contradiction")
    assert len(only) == 1 and only[0]["statement"] == "contradiction"


def test_findings_separate_from_assessment_ledger(corpus: Corpus, tmp_path: Path) -> None:
    ch = _chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    corpus.assess(sid, ch[0], stance="supports", weight=0.5, agent_id="a1")
    corpus.create_finding(sid, statement="pattern", supporting_chunk_ids=[ch[0], ch[1]],
                          stance="against", weight=0.9, agent_id="synth")
    led = corpus.study_ledger(sid)
    assert led["total"] == 1                          # only the assessment
    assert led["tallies"]["against"] == 0             # the finding isn't in the ledger
    assert len(corpus.list_findings(sid)) == 1


def test_finding_validation(corpus: Corpus, tmp_path: Path) -> None:
    ch = _chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    with pytest.raises(InvalidEnumError, match="must cite supporting"):
        corpus.create_finding(sid, statement="x", supporting_chunk_ids=[],
                              stance="against", weight=0.5, agent_id="s")
    with pytest.raises(InvalidEnumError, match="not found"):
        corpus.create_finding(sid, statement="x", supporting_chunk_ids=[ch[0], "nope"],
                              stance="against", weight=0.5, agent_id="s")
    with pytest.raises(InvalidEnumError):  # bad stance
        corpus.create_finding(sid, statement="x", supporting_chunk_ids=[ch[0]],
                              stance="maybe", weight=0.5, agent_id="s")
    with pytest.raises(InvalidEnumError):  # study missing
        corpus.create_finding("nope", statement="x", supporting_chunk_ids=[ch[0]],
                              stance="against", weight=0.5, agent_id="s")


def test_finding_verification_builds_confidence(corpus: Corpus, tmp_path: Path) -> None:
    ch = _chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    f = corpus.create_finding(sid, statement="unequal", supporting_chunk_ids=[ch[0], ch[2]],
                              stance="against", weight=0.9, agent_id="synth", provenance="primary_text")
    fid = f["finding_id"]
    # A fresh finding is unreviewed → low confidence, needs_more.
    fresh = corpus.list_findings(sid)[0]
    assert fresh["reviewer_count"] == 0 and fresh["confidence"] == 0.0
    assert fresh["escalation_state"] == "needs_more"
    # Two concurring independent reviewers → settled, confidence rises.
    corpus.verify_finding(fid, verdict="verified", verifier_agent_id="r1", provenance="primary_text")
    r2 = corpus.verify_finding(fid, verdict="verified", verifier_agent_id="r2", provenance="primary_text")
    assert r2["escalation_state"] == "settled" and r2["agreement"] == 1.0 and r2["confidence"] == 1.0
    row = corpus.list_findings(sid)[0]
    assert row["reviewer_count"] == 2 and row["vote_tally"]["verified"] == 2
    assert len(row["review_events"]) == 2
    # Escalation state is a routing label.
    assert corpus.cypher("MATCH (f:Finding:Settled) RETURN count(f) AS n").to_list()[0]["n"] == 1


def test_finding_dispute_is_contested(corpus: Corpus, tmp_path: Path) -> None:
    ch = _chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    fid = corpus.create_finding(sid, statement="x", supporting_chunk_ids=[ch[0]],
                                stance="against", weight=0.8, agent_id="synth")["finding_id"]
    corpus.verify_finding(fid, verdict="verified", verifier_agent_id="r1")
    r = corpus.verify_finding(fid, verdict="disputed", verifier_agent_id="r2")
    assert r["escalation_state"] == "contested"
    assert corpus.list_findings(sid)[0]["escalation_state"] == "contested"


def test_finding_self_verification_rejected(corpus: Corpus, tmp_path: Path) -> None:
    ch = _chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    fid = corpus.create_finding(sid, statement="x", supporting_chunk_ids=[ch[0]],
                                stance="against", weight=0.5, agent_id="synth")["finding_id"]
    from kglite_docs.errors import SelfVerificationError
    with pytest.raises(SelfVerificationError):
        corpus.verify_finding(fid, verdict="verified", verifier_agent_id="synth")
    with pytest.raises(InvalidEnumError):  # finding missing
        corpus.verify_finding("nope", verdict="verified", verifier_agent_id="r1")


def test_semantic_conflicts_cross_chunk(corpus: Corpus, tmp_path: Path) -> None:
    ch = _chunks(corpus, tmp_path)
    # Two different chunks classified as the same element/topic.
    classify.classify_chunk(corpus.store, chunk_id=ch[0], elements=["rulex"], agent_id="cl")
    classify.classify_chunk(corpus.store, chunk_id=ch[1], elements=["rulex"], agent_id="cl")
    sid = corpus.define_study("Q", created_by="lead")
    corpus.assess(sid, ch[0], stance="supports", weight=0.8, agent_id="a1")
    corpus.assess(sid, ch[1], stance="against", weight=0.8, agent_id="a1")
    # Same-chunk conflicts() sees nothing (opposing stances are on different chunks)…
    assert corpus.study_conflicts(sid)["total"] == 0
    # …but the cross-chunk semantic scan flags the shared-element contradiction.
    sc = corpus.study_semantic_conflicts(sid)
    assert sc["total"] == 1 and sc["conflicts"][0]["element"] == "rulex"
    assert sc["checked"] == 2 and sc["skipped_unclassified"] == 0


def test_semantic_conflicts_honest_when_unclassified(corpus: Corpus, tmp_path: Path) -> None:
    ch = _chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    corpus.assess(sid, ch[0], stance="supports", weight=0.5, agent_id="a1")
    corpus.assess(sid, ch[1], stance="against", weight=0.5, agent_id="a1")
    sc = corpus.study_semantic_conflicts(sid)
    # Nothing classified → it says "not looked", not a falsely-clean zero.
    assert sc["checked"] == 0 and sc["total"] == 0
    assert "note" in sc and "classif" in sc["note"].lower()


def test_delete_study_cascades_findings(corpus: Corpus, tmp_path: Path) -> None:
    ch = _chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    corpus.create_finding(sid, statement="p", supporting_chunk_ids=[ch[0], ch[1]],
                          stance="against", weight=0.9, agent_id="s")
    corpus.delete_study(sid)
    assert corpus.cypher("MATCH (f:Finding) RETURN count(f) AS n").to_list()[0]["n"] == 0
    # SUPPORTED_BY edges went with the DETACH DELETE.
    assert corpus.cypher(
        "MATCH (:Finding)-[r:SUPPORTED_BY]->(:Chunk) RETURN count(r) AS n"
    ).to_list()[0]["n"] == 0
