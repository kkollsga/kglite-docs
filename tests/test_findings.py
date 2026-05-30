"""0.0.13 Phase 1: cross-chunk Finding unit — a pattern asserted over a SET of
chunks (what per-chunk assess structurally can't see)."""

from __future__ import annotations

from pathlib import Path

import pytest

from kglite_docs import Corpus
from kglite_docs.errors import InvalidEnumError


def _chunks(corpus: Corpus, tmp_path: Path, n: int = 4) -> list[str]:
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
