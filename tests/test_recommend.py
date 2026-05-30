"""0.0.13 Phase 7: follow-on study recommendations (proposals → spawn)."""

from __future__ import annotations

from pathlib import Path

import pytest

from kglite_docs import Corpus, recommend
from kglite_docs.errors import InvalidEnumError

recommend.register_recommendation_trigger(
    "tr_bias", question_template="Follow-on: possible bias across the record?",
    suggested_lens="", rationale="a bias finding implies a bias study",
)


def _chunks(corpus: Corpus, tmp_path: Path, n: int = 3) -> list[str]:
    p = tmp_path / "d.md"
    p.write_text(
        "\n\n".join(f"# Sec {i}\n\nParagraph {i} distinct words for chunk {i} here." for i in range(n)),
        encoding="utf-8",
    )
    corpus.ingest(p, structure_aware=True)
    return [r["id"] for r in corpus.cypher(
        "MATCH (c:Chunk:Ready) RETURN c.id AS id ORDER BY c.chunk_index"
    ).to_list()]


def test_recommend_and_spawn(corpus: Corpus, tmp_path: Path) -> None:
    ch = _chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    assert corpus.recommend_studies(sid) == []  # clean study proposes nothing
    corpus.create_finding(sid, statement="unequal", supporting_chunk_ids=[ch[0], ch[1]],
                          stance="against", weight=0.9, agent_id="synth", finding_type="tr_bias")
    recs = corpus.recommend_studies(sid)
    assert len(recs) == 1 and recs[0]["trigger"] == "tr_bias"
    assert recs[0]["seed_finding_ids"]  # seeded with the triggering finding
    assert len(corpus.recommend_studies(sid)) == 1  # idempotent, no duplicate
    sp = corpus.spawn_study(recs[0]["recommendation_id"], approved_by="lead")
    child = sp["child_study_id"]
    assert child and child != sid
    # SPAWNED_FROM edge (child → source) records the trigger.
    edge = corpus.cypher(
        "MATCH (:Study {id: $c})-[r:SPAWNED_FROM]->(:Study {id: $s}) RETURN r.reason AS reason",
        params={"c": child, "s": sid},
    ).to_list()
    assert edge and edge[0]["reason"] == "tr_bias"
    assert corpus.list_recommendations(sid)[0]["status"] == "approved"


def test_unmapped_finding_proposes_nothing(corpus: Corpus, tmp_path: Path) -> None:
    ch = _chunks(corpus, tmp_path)
    sid = corpus.define_study("Q", created_by="lead")
    corpus.create_finding(sid, statement="x", supporting_chunk_ids=[ch[0]],
                          stance="against", weight=0.5, agent_id="synth", finding_type="no_trigger_here")
    assert corpus.recommend_studies(sid) == []


def test_spawn_unknown_recommendation_raises(corpus: Corpus) -> None:
    with pytest.raises(InvalidEnumError):
        corpus.spawn_study("rec_nope", approved_by="lead")
