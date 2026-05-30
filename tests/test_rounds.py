"""0.0.13 Phase 5: leveled review — escalation rounds + lens registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from kglite_docs import Corpus, lenses
from kglite_docs.errors import InvalidEnumError

# Self-contained test lenses (don't depend on the legal pack's load order).
lenses.register_lens("tl_detect", prompt="hunt patterns", unit_type="chunk", description="t")
lenses.register_lens("tl_regrade", prompt="re-grade", unit_type="finding", description="t")


def _chunks(corpus: Corpus, tmp_path: Path, n: int = 6) -> list[str]:
    p = tmp_path / "d.md"
    p.write_text(
        "\n\n".join(
            f"# Sec {i}\n\nParagraph {i} with several distinct words for chunk {i} here."
            for i in range(n)
        ),
        encoding="utf-8",
    )
    corpus.ingest(p, structure_aware=True)
    return [r["id"] for r in corpus.cypher(
        "MATCH (c:Chunk:Ready) RETURN c.id AS id ORDER BY c.chunk_index"
    ).to_list()]


def test_available_lenses_lists_registered(corpus: Corpus) -> None:
    names = {x["name"] for x in corpus.available_lenses()}
    assert {"tl_detect", "tl_regrade"} <= names


def test_escalate_contested_panel_raises_confidence(corpus: Corpus, tmp_path: Path) -> None:
    ch = _chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    fid = corpus.create_finding(sid, statement="disparate", supporting_chunk_ids=[ch[0], ch[1]],
                                stance="against", weight=0.9, agent_id="synth",
                                provenance="primary_text")["finding_id"]
    # A fresh finding is contested-scope work (needs_more).
    esc = corpus.escalate_study(sid, kind="panel", created_by="lead", scope="contested", reviewers=2)
    assert esc["target_kind"] == "finding" and esc["worklist_size"] == 1
    assert esc["worklist"][0]["id"] == fid
    rid = esc["round_id"]
    # Two concurring reviewers via record_review → settled.
    corpus.record_review(rid, fid, verdict="verified", agent_id="p1", provenance="primary_text")
    r2 = corpus.record_review(rid, fid, verdict="verified", agent_id="p2", provenance="primary_text")
    assert r2["vote"]["escalation_state"] == "settled"
    assert corpus.close_round(rid)["status"] == "done"
    # Settled findings are no longer contested-scope work.
    again = corpus.escalate_study(sid, kind="panel", created_by="lead", scope="contested")
    assert again["worklist_size"] == 0


def test_detectability_round_claims_uncovered_and_links_finding(corpus: Corpus, tmp_path: Path) -> None:
    ch = _chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    for c in ch[:4]:
        corpus.assess(sid, c, stance="neutral", weight=0.1, agent_id="r1")
    esc = corpus.escalate_study(sid, kind="expert", created_by="lead",
                                scope="uncovered", lens="tl_detect", limit=10)
    assert esc["target_kind"] == "chunk" and esc["worklist_size"] == 4
    rid = esc["round_id"]
    # Claim a non-overlapping batch (punchcard keyed on the round).
    claimed = corpus.next_review(rid, agent_id="d1", limit=2)
    assert len(claimed) == 2
    # Record coverage + a finding the lens surfaced (linked to the round).
    corpus.record_review(rid, ch[0], target_kind="chunk", agent_id="d1")
    corpus.create_finding(sid, statement="ignored arg", supporting_chunk_ids=[ch[0]],
                          stance="against", weight=0.7, agent_id="d1", origin_round_id=rid)
    assert corpus.close_round(rid)["new_findings"] == 1
    assert len(corpus.list_rounds(sid)) == 1


def test_escalate_validation(corpus: Corpus, tmp_path: Path) -> None:
    _chunks(corpus, tmp_path, n=2)
    sid = corpus.define_study("Q", created_by="lead")
    with pytest.raises(InvalidEnumError):  # unknown lens is a named gap, not silent
        corpus.escalate_study(sid, kind="expert", created_by="lead", scope="uncovered", lens="nope")
    with pytest.raises(InvalidEnumError):  # bad kind
        corpus.escalate_study(sid, kind="bogus", created_by="lead")
    with pytest.raises(InvalidEnumError):  # unknown round
        corpus.record_review("round_nope", "x", agent_id="a1")
