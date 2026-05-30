"""0.0.13 Phase 8: timeline/Event layer + queryable entity-value scalars."""

from __future__ import annotations

from pathlib import Path

from kglite_docs import Corpus


def _doc_id(corpus: Corpus) -> str:
    return corpus.cypher("MATCH (d:Document) RETURN d.id AS id").to_list()[0]["id"]


def test_queryable_money_and_date_scalars(corpus: Corpus, tmp_path: Path) -> None:
    p = tmp_path / "d.md"
    p.write_text(
        "# A\n\nPaid $1,250.00 on 2021-03-04 to the party as agreed.\n\n"
        "# B\n\nAwarded $3,000 in damages on 2022-01-15 by the court.",
        encoding="utf-8",
    )
    corpus.ingest(p, structure_aware=True)
    rows = corpus.cypher(
        "MATCH (c:Chunk) RETURN c.money_max AS mm, c.date_first AS df ORDER BY c.chunk_index"
    ).to_list()
    assert rows[0]["mm"] == 1250.0 and rows[0]["df"] == "2021-03-04"
    assert rows[1]["mm"] == 3000.0
    # Aggregates work in Cypher without re-parsing entities_json.
    assert corpus.cypher("MATCH (c:Chunk) RETURN sum(c.money_max) AS t").to_list()[0]["t"] == 4250.0


def test_timeline_detects_disparate_treatment_and_contradiction(corpus: Corpus, tmp_path: Path) -> None:
    (tmp_path / "d.md").write_text("# Case\n\nA tribunal record with several rulings.", encoding="utf-8")
    corpus.ingest(tmp_path / "d.md")
    doc = _doc_id(corpus)
    # Same trigger (non_appearance) → different outcome by actor = disparate treatment.
    corpus.add_event(doc, date="2021-01-01", actor="respondent", action="non_appearance",
                     outcome="default", agent_id="ex")
    corpus.add_event(doc, date="2021-02-01", actor="claimant", action="non_appearance",
                     outcome="dismissal", agent_id="ex")
    # Same (actor, action) with conflicting outcomes = contradictory operative outcomes.
    corpus.add_event(doc, date="2021-03-01", actor="court", action="disposition", outcome="extinguished", agent_id="ex")
    corpus.add_event(doc, date="2021-04-01", actor="court", action="disposition", outcome="condemned", agent_id="ex")

    assert [e["date"] for e in corpus.timeline(doc)] == [
        "2021-01-01", "2021-02-01", "2021-03-01", "2021-04-01"]
    tc = corpus.timeline_conflicts(doc)
    assert tc["events"] == 4
    assert len(tc["disparate_treatment"]) == 1
    assert tc["disparate_treatment"][0]["action"] == "non_appearance"
    assert len(tc["contradictory_outcomes"]) == 1
    assert tc["contradictory_outcomes"][0]["outcomes"] == ["condemned", "extinguished"]


def test_clean_timeline_has_no_conflicts(corpus: Corpus, tmp_path: Path) -> None:
    (tmp_path / "d.md").write_text("# Case\n\nA clean record.", encoding="utf-8")
    corpus.ingest(tmp_path / "d.md")
    doc = _doc_id(corpus)
    corpus.add_event(doc, date="2021-01-01", actor="court", action="ruling", outcome="granted", agent_id="ex")
    tc = corpus.timeline_conflicts(doc)
    assert tc["events"] == 1 and tc["total"] == 0
