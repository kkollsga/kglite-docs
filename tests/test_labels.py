"""Multi-label round-trips (kglite 0.10.5).

Asserts that the labels we add at write time survive save+reload and
are queryable via the standard Cypher predicate `MATCH (n:Label)`.
The user-facing API doesn't change — these tests bypass the
public `Corpus` methods and check the underlying graph directly to
validate the new shape.
"""

from __future__ import annotations

from pathlib import Path

import kglite

from kglite_docs import Corpus


def _ingest_md(corpus: Corpus, tmp_path: Path) -> str:
    p = tmp_path / "d.md"
    p.write_text(
        "# A\n\nFirst paragraph here.\n\n# B\n\nSecond paragraph here.\n",
        encoding="utf-8",
    )
    corpus.ingest(p)
    corpus.index()
    return corpus.search("First paragraph", top_k=1)[0]["id"]


def test_agent_role_kind_labels_set_on_upsert(corpus: Corpus) -> None:
    corpus.upsert_agent(
        "reviewer-1", role="reviewer", kind="llm", model="sonnet",
    )
    labels = corpus.store.node_labels("Agent", "reviewer-1")
    assert "Agent" in labels
    assert "Reviewer" in labels
    assert "LLM" in labels


def test_agent_label_query_matches(corpus: Corpus) -> None:
    corpus.upsert_agent("rev-a", role="reviewer", kind="llm")
    corpus.upsert_agent("rev-b", role="reviewer", kind="llm")
    corpus.upsert_agent("wri-a", role="writer", kind="llm")
    reviewers = corpus.cypher(
        "MATCH (a:Agent:Reviewer) RETURN a.id AS id ORDER BY a.id"
    ).to_list()
    assert {r["id"] for r in reviewers} == {"rev-a", "rev-b"}


def test_chunk_status_labels_on_ingest(corpus: Corpus, tmp_path: Path) -> None:
    _ingest_md(corpus, tmp_path)
    ready = corpus.cypher("MATCH (c:Chunk:Ready) RETURN count(c) AS n").to_list()
    assert ready[0]["n"] >= 1
    # And the chunk carries both labels
    cid = corpus.search("First paragraph", top_k=1)[0]["id"]
    labels = corpus.store.node_labels("Chunk", cid)
    assert {"Chunk", "Ready"}.issubset(set(labels))


def test_summary_verification_label_swap(corpus: Corpus, tmp_path: Path) -> None:
    """add_summary → :Unverified; verify_summary → :Verified.
    Old label is removed (mutual exclusion via swap_label)."""
    cid = _ingest_md(corpus, tmp_path)
    sid = corpus.add_summary(cid, "summary text", agent_id="writer")
    pre = corpus.store.node_labels("Summary", sid)
    assert "Unverified" in pre
    assert "Verified" not in pre
    corpus.verify_summary(sid, verdict="verified", verifier_agent_id="reviewer")
    post = corpus.store.node_labels("Summary", sid)
    assert "Verified" in post
    assert "Unverified" not in post


def test_review_ticket_labels_through_lifecycle(corpus: Corpus, tmp_path: Path) -> None:
    """enqueue → :New; claim → :InReview; complete → :Reviewed.
    Labels stay mutually exclusive across transitions."""
    cid = _ingest_md(corpus, tmp_path)
    tid = corpus.enqueue_review(cid)
    assert "New" in corpus.store.node_labels("ReviewTicket", tid)

    corpus.claim_review(tid, agent_id="rev")
    after_claim = corpus.store.node_labels("ReviewTicket", tid)
    assert "InReview" in after_claim
    assert "New" not in after_claim

    corpus.complete_review(tid, agent_id="rev", verdict="reviewed")
    after_complete = corpus.store.node_labels("ReviewTicket", tid)
    assert "Reviewed" in after_complete
    assert "InReview" not in after_complete


def test_labels_survive_save_and_reload(corpus: Corpus, tmp_path: Path) -> None:
    """The headline guarantee: labels persist through `.kgl` round-trip."""
    db = tmp_path / "labelled.kgl"
    # Override the corpus path so save() goes where we want
    c = Corpus.create(db, embedder=corpus.embedder)
    c.upsert_agent("agent-x", role="reviewer", kind="llm")
    p = tmp_path / "d.md"
    p.write_text("# Topic\n\nbody\n", encoding="utf-8")
    c.ingest(p)
    c.index()
    cid = c.search("body", top_k=1)[0]["id"]
    sid = c.add_summary(cid, "X", agent_id="agent-x")
    c.save()

    # Reload via raw kglite — verify labels survived
    g = kglite.load(str(db))
    reviewers = g.cypher("MATCH (a:Agent:Reviewer) RETURN a.id AS id").to_list()
    assert any(r["id"] == "agent-x" for r in reviewers)
    ready_chunks = g.cypher("MATCH (c:Chunk:Ready) RETURN count(c) AS n").to_list()
    assert ready_chunks[0]["n"] >= 1
    unverified = g.cypher("MATCH (s:Summary:Unverified) RETURN s.id AS id").to_list()
    assert any(s["id"] == sid for s in unverified)


def test_cross_type_label_predicate(corpus: Corpus, tmp_path: Path) -> None:
    """`MATCH (n:Reviewed)` returns nodes of any primary type carrying the
    Reviewed label — the headline feature the kglite team called out."""
    cid = _ingest_md(corpus, tmp_path)
    tid = corpus.enqueue_review(cid)
    corpus.claim_review(tid, agent_id="rev")
    corpus.complete_review(tid, agent_id="rev", verdict="reviewed")

    # ReviewTicket:Reviewed should match
    out = corpus.cypher(
        "MATCH (n:Reviewed) RETURN labels(n) AS labels, n.id AS id"
    ).to_list()
    assert any("ReviewTicket" in r["labels"] for r in out)


def test_study_labels_survive_save_and_reload(corpus: Corpus, tmp_path: Path) -> None:
    """Stance / study-status / assessment-verification labels round-trip
    through save+reload and are queryable as predicates."""
    db = tmp_path / "study.kgl"
    c = Corpus.create(db, embedder=corpus.embedder)
    p = tmp_path / "d.md"
    p.write_text("# A\n\nbody one\n\n# B\n\nbody two\n", encoding="utf-8")
    c.ingest(p)
    cid = c.cypher("MATCH (ch:Chunk:Ready) RETURN ch.id AS id LIMIT 1").to_list()[0]["id"]
    sid = c.define_study("Q", created_by="lead")
    a = c.assess(sid, cid, stance="supports", weight=0.9, agent_id="a1")
    c.verify_assessment(a["assessment_id"], verdict="verified", verifier_agent_id="checker")
    c.save()

    g = kglite.load(str(db))
    assert g.cypher("MATCH (s:Study:Open) RETURN s.id AS id").to_list()[0]["id"] == sid
    assert g.cypher("MATCH (a:Assessment:Supports) RETURN count(a) AS n").to_list()[0]["n"] == 1
    assert g.cypher("MATCH (a:Assessment:Verified) RETURN count(a) AS n").to_list()[0]["n"] == 1
    # cross-type: Verified now spans Summary and Assessment
    verified = g.cypher("MATCH (n:Verified) RETURN labels(n) AS labels").to_list()
    assert any("Assessment" in r["labels"] for r in verified)
