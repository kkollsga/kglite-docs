"""Agent-as-template — upsert config, fetch, attribute activity."""

from __future__ import annotations

from pathlib import Path

from kglite_docs import Corpus


def _ingest(corpus: Corpus, tmp_path: Path) -> str:
    p = tmp_path / "d.md"
    p.write_text("# T\n\nFirst paragraph.\n\n# U\n\nSecond.\n", encoding="utf-8")
    corpus.ingest(p)
    return corpus.search("paragraph", top_k=1)[0]["id"]


def test_upsert_creates_template(corpus: Corpus) -> None:
    cfg = corpus.upsert_agent(
        "reviewer-strict",
        role="reviewer", model="claude-sonnet-4-6",
        system_prompt="You are a strict fact-checker.",
        tools=["check_grounding", "verify_claim"],
        context={"strictness": "high", "min_citations": 2},
        description="Strict reviewer.",
    )
    assert cfg["id"] == "reviewer-strict"
    assert cfg["role"] == "reviewer"
    assert cfg["model"] == "claude-sonnet-4-6"
    assert cfg["tools"] == ["check_grounding", "verify_claim"]
    assert cfg["context"] == {"strictness": "high", "min_citations": 2}
    assert cfg["created"] is True


def test_upsert_field_level_merge(corpus: Corpus) -> None:
    corpus.upsert_agent(
        "reviewer", role="reviewer",
        system_prompt="initial prompt",
        tools=["check_grounding"],
        description="initial desc",
    )
    # Update only the prompt; other fields preserved
    out = corpus.upsert_agent("reviewer", system_prompt="updated prompt")
    assert out["system_prompt"] == "updated prompt"
    assert out["role"] == "reviewer"
    assert out["tools"] == ["check_grounding"]
    assert out["description"] == "initial desc"
    assert out["created"] is False


def test_get_agent_hydrates_json(corpus: Corpus) -> None:
    corpus.upsert_agent(
        "writer",
        role="writer", model="opus-4.7",
        tools=["search", "compose_context"],
        context={"target_length": 600, "citation_style": "inline"},
    )
    cfg = corpus.get_agent("writer")
    assert isinstance(cfg["tools"], list)
    assert isinstance(cfg["context"], dict)
    assert cfg["context"]["target_length"] == 600


def test_get_unknown_agent_returns_empty(corpus: Corpus) -> None:
    assert corpus.get_agent("nobody") == {}


def test_list_agents_filter_by_role(corpus: Corpus) -> None:
    corpus.upsert_agent("reviewer-a", role="reviewer", model="sonnet")
    corpus.upsert_agent("reviewer-b", role="reviewer", model="opus")
    corpus.upsert_agent("writer-a", role="writer", model="sonnet")

    rs = corpus.list_agents(role="reviewer")
    assert {r["id"] for r in rs} == {"reviewer-a", "reviewer-b"}
    ws = corpus.list_agents(role="writer")
    assert {r["id"] for r in ws} == {"writer-a"}


def test_lazy_register_preserves_template(corpus: Corpus, tmp_path: Path) -> None:
    """register_agent (lazy on-first-use) must not clobber a
    pre-configured template."""
    corpus.upsert_agent(
        "reviewer", role="reviewer",
        system_prompt="reviewer prompt",
        tools=["check_grounding"],
    )
    # Now lazy-trigger from an activity write
    cid = _ingest(corpus, tmp_path)
    corpus.add_summary(cid, "a summary", agent_id="reviewer")
    cfg = corpus.get_agent("reviewer")
    assert cfg["role"] == "reviewer"
    assert cfg["system_prompt"] == "reviewer prompt"
    assert cfg["tools"] == ["check_grounding"]


def test_agent_activity_aggregates_buckets(corpus: Corpus, tmp_path: Path) -> None:
    """agent_activity returns the agent's writes bucketed by type."""
    cid = _ingest(corpus, tmp_path)
    corpus.upsert_agent("reviewer", role="reviewer")
    sid = corpus.add_summary(cid, "summary x", agent_id="reviewer")
    corpus.tag_chunk(cid, "topic-a", agent_id="reviewer")

    a = corpus.agent_activity("reviewer")
    assert a["agent"]["id"] == "reviewer"
    assert any(s["id"] == sid for s in a["summaries"])
    assert any(t["chunk_id"] == cid for t in a["tags"])


def test_agent_activity_scoped_to_target(corpus: Corpus, tmp_path: Path) -> None:
    """target_id filter narrows activity to one chunk."""
    cid1 = _ingest(corpus, tmp_path)
    # Second doc
    p2 = tmp_path / "d2.md"
    p2.write_text("# Q\n\nOther body.\n", encoding="utf-8")
    corpus.ingest(p2)
    cid2 = corpus.cypher(
        "MATCH (c:Chunk) WHERE c.id <> $cid RETURN c.id AS id LIMIT 1",
        params={"cid": cid1},
    ).to_list()[0]["id"]

    corpus.add_summary(cid1, "s1", agent_id="reviewer")
    corpus.add_summary(cid2, "s2", agent_id="reviewer")
    corpus.tag_chunk(cid2, "interesting", agent_id="reviewer")

    scoped = corpus.agent_activity("reviewer", target_id=cid1)
    assert len(scoped["summaries"]) == 1
    assert scoped["summaries"][0]["target_id"] == cid1
    assert not scoped["tags"]  # tags were only on cid2

    scoped2 = corpus.agent_activity("reviewer", target_id=cid2)
    assert len(scoped2["summaries"]) == 1
    assert scoped2["summaries"][0]["target_id"] == cid2
    assert len(scoped2["tags"]) == 1


def test_template_round_trip_for_real_orchestration(corpus: Corpus) -> None:
    """End-to-end: a config you can drop into an LLM client.

    Demonstrates the intended usage: store the template once,
    retrieve later, and use the fields to launch a call."""
    corpus.upsert_agent(
        "reviewer-strict",
        role="reviewer", model="claude-sonnet-4-6",
        system_prompt=(
            "You are a strict fact-checker. For each claim, verify "
            "against source chunks. Mark unsupported claims clearly."
        ),
        tools=["check_grounding", "verify_claim"],
        context={"strictness": "high"},
    )
    cfg = corpus.get_agent("reviewer-strict")
    # Pretend we're launching anthropic.messages.create(...) with these:
    assert cfg["model"]
    assert cfg["system_prompt"]
    assert "check_grounding" in cfg["tools"]
    assert cfg["context"]["strictness"] == "high"
