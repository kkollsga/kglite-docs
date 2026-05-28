"""Review-queue (kanban) lifecycle, state transitions, and conflicts."""

from __future__ import annotations

from pathlib import Path

import pytest

from kglite_docs import Corpus
from kglite_docs.review import ReviewConflict


def _ingest_chunks(corpus: Corpus, tmp_path: Path) -> list[str]:
    p = tmp_path / "d.md"
    p.write_text(
        "# A\n\nFirst para.\n\n# B\n\nSecond para.\n\n# C\n\nThird para.\n",
        encoding="utf-8",
    )
    corpus.ingest(p)
    return [
        r["c.id"] for r in corpus.cypher(
            "MATCH (c:Chunk) RETURN c.id ORDER BY c.chunk_index"
        ).to_list()
    ]


# ─── enqueue ─────────────────────────────────────────────────────────────


def test_enqueue_creates_new_ticket(corpus: Corpus, tmp_path: Path) -> None:
    cid = _ingest_chunks(corpus, tmp_path)[0]
    tid = corpus.enqueue_review(cid)
    queue = corpus.list_review_queue(status="new")
    assert any(t["id"] == tid for t in queue)


def test_enqueue_chunks_for_review_bulk(corpus: Corpus, tmp_path: Path) -> None:
    _ingest_chunks(corpus, tmp_path)
    r = corpus.enqueue_chunks_for_review()
    assert r["enqueued"] >= 3
    # Idempotent: second call doesn't re-enqueue
    r2 = corpus.enqueue_chunks_for_review()
    assert r2["enqueued"] == 0
    assert r2["skipped"] >= 3


# ─── claim ───────────────────────────────────────────────────────────────


def test_claim_transitions_new_to_in_review(corpus: Corpus, tmp_path: Path) -> None:
    cid = _ingest_chunks(corpus, tmp_path)[0]
    tid = corpus.enqueue_review(cid)
    result = corpus.claim_review(tid, agent_id="reviewer-a")
    assert result["status"] == "in_review"
    assert result["claimed_by"] == "reviewer-a"


def test_double_claim_is_rejected(corpus: Corpus, tmp_path: Path) -> None:
    cid = _ingest_chunks(corpus, tmp_path)[0]
    tid = corpus.enqueue_review(cid)
    corpus.claim_review(tid, agent_id="reviewer-a")
    with pytest.raises(ReviewConflict):
        corpus.claim_review(tid, agent_id="reviewer-b")


def test_claim_next_returns_hydrated_target(corpus: Corpus, tmp_path: Path) -> None:
    ids = _ingest_chunks(corpus, tmp_path)
    for cid in ids:
        corpus.enqueue_review(cid, priority=0)
    # One high-priority ticket should come out first
    corpus.enqueue_review(ids[-1], priority=10)
    ticket = corpus.claim_next_review(agent_id="reviewer-a")
    assert ticket is not None
    assert ticket["status"] == "in_review"
    assert ticket["claimed_by"] == "reviewer-a"
    # Target hydrated with the actual chunk fields
    assert "target" in ticket and "text" in ticket["target"]


def test_claim_next_empty_queue_returns_none(corpus: Corpus) -> None:
    assert corpus.claim_next_review(agent_id="reviewer-a") is None


# ─── unclaim ─────────────────────────────────────────────────────────────


def test_unclaim_returns_ticket_to_new(corpus: Corpus, tmp_path: Path) -> None:
    cid = _ingest_chunks(corpus, tmp_path)[0]
    tid = corpus.enqueue_review(cid)
    corpus.claim_review(tid, agent_id="reviewer-a")
    corpus.unclaim_review(tid, agent_id="reviewer-a", reason="changed my mind")
    queue = corpus.list_review_queue(status="new")
    assert any(t["id"] == tid for t in queue)


def test_unclaim_by_non_holder_fails(corpus: Corpus, tmp_path: Path) -> None:
    cid = _ingest_chunks(corpus, tmp_path)[0]
    tid = corpus.enqueue_review(cid)
    corpus.claim_review(tid, agent_id="reviewer-a")
    with pytest.raises(ReviewConflict):
        corpus.unclaim_review(tid, agent_id="reviewer-b")


# ─── complete ────────────────────────────────────────────────────────────


def test_complete_review_with_tags_and_metadata(corpus: Corpus, tmp_path: Path) -> None:
    cid = _ingest_chunks(corpus, tmp_path)[0]
    tid = corpus.enqueue_review(cid)
    corpus.claim_review(tid, agent_id="reviewer-a")
    result = corpus.complete_review(
        tid, agent_id="reviewer-a",
        verdict="reviewed", accuracy=0.92, authenticity="verified",
        notes="checked facts against source", tags=["q2-pass", "cite-ok"],
    )
    assert result["status"] == "reviewed"
    # Tags landed on the chunk
    rows = corpus.list_tags(chunk_id=cid, agent_id="reviewer-a")
    names = {r["name"] for r in rows}
    assert {"q2-pass", "cite-ok"}.issubset(names)
    # Ticket reports the new status
    ticket = corpus.get_review_ticket(tid)
    assert ticket["status"] == "reviewed"
    assert any(e["type"] == "reviewed" for e in ticket["events"])


def test_complete_by_non_claimer_fails(corpus: Corpus, tmp_path: Path) -> None:
    cid = _ingest_chunks(corpus, tmp_path)[0]
    tid = corpus.enqueue_review(cid)
    corpus.claim_review(tid, agent_id="reviewer-a")
    with pytest.raises(ReviewConflict):
        corpus.complete_review(tid, agent_id="reviewer-b", verdict="reviewed")


def test_complete_unclaimed_fails(corpus: Corpus, tmp_path: Path) -> None:
    cid = _ingest_chunks(corpus, tmp_path)[0]
    tid = corpus.enqueue_review(cid)
    with pytest.raises(ReviewConflict):
        corpus.complete_review(tid, agent_id="reviewer-a", verdict="reviewed")


def test_invalid_verdict_rejected(corpus: Corpus, tmp_path: Path) -> None:
    cid = _ingest_chunks(corpus, tmp_path)[0]
    tid = corpus.enqueue_review(cid)
    corpus.claim_review(tid, agent_id="reviewer-a")
    with pytest.raises(ValueError):
        corpus.complete_review(tid, agent_id="reviewer-a", verdict="ship-it")


# ─── stats + queue ───────────────────────────────────────────────────────


def test_review_stats_counts_by_status(corpus: Corpus, tmp_path: Path) -> None:
    ids = _ingest_chunks(corpus, tmp_path)
    for cid in ids[:3]:
        corpus.enqueue_review(cid)
    # claim one, complete one, leave one new
    t_claim = corpus.claim_next_review(agent_id="rev")
    t_complete = corpus.claim_next_review(agent_id="rev")
    corpus.complete_review(t_complete["ticket_id"], agent_id="rev")

    s = corpus.review_stats()
    assert s["tickets_total"] == 3
    assert s["by_status"].get("new", 0) == 1
    assert s["by_status"].get("in_review", 0) == 1
    assert s["by_status"].get("reviewed", 0) == 1
    assert s["in_review_by_agent"].get("rev", 0) == 1


def test_queue_filter_by_status(corpus: Corpus, tmp_path: Path) -> None:
    cid = _ingest_chunks(corpus, tmp_path)[0]
    tid = corpus.enqueue_review(cid)
    corpus.claim_review(tid, agent_id="rev")
    new_q = corpus.list_review_queue(status="new")
    in_review_q = corpus.list_review_queue(status="in_review")
    assert not any(t["id"] == tid for t in new_q)
    assert any(t["id"] == tid for t in in_review_q)


def test_audit_trail_is_immutable(corpus: Corpus, tmp_path: Path) -> None:
    cid = _ingest_chunks(corpus, tmp_path)[0]
    tid = corpus.enqueue_review(cid)
    corpus.claim_review(tid, agent_id="rev")
    corpus.unclaim_review(tid, agent_id="rev")
    corpus.claim_review(tid, agent_id="rev")
    corpus.complete_review(tid, agent_id="rev", verdict="needs_revision",
                           notes="missing citation")
    ticket = corpus.get_review_ticket(tid)
    types = [e["type"] for e in ticket["events"]]
    # enqueue → in_review → new → in_review → needs_revision
    assert types == ["new", "in_review", "new", "in_review", "needs_revision"]
    assert ticket["status"] == "needs_revision"


# ─── concurrent claim guard ──────────────────────────────────────────────


def test_two_agents_cant_both_claim_next(corpus: Corpus, tmp_path: Path) -> None:
    """`claim_next` uses a process-local lock — two sequential calls
    must return different tickets."""
    ids = _ingest_chunks(corpus, tmp_path)
    for cid in ids:
        corpus.enqueue_review(cid)
    a = corpus.claim_next_review(agent_id="agent-a")
    b = corpus.claim_next_review(agent_id="agent-b")
    assert a is not None and b is not None
    assert a["ticket_id"] != b["ticket_id"]
