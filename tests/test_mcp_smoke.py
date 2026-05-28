"""MCP server tests — boot the FastMCP app and round-trip real tool calls."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from kglite_docs import Corpus


@pytest.mark.mcp
def test_mcp_tools_registered(corpus: Corpus) -> None:
    """The app should boot and expose the typed tool surface."""
    from kglite_docs.mcp_server.server import build_app

    app = build_app(corpus)
    assert app is not None
    # FastMCP keeps registered tools; the public way to enumerate is
    # `await app.list_tools()`.
    tools = asyncio.run(app.list_tools())
    names = {t.name for t in tools}
    # A handful of names we *know* we registered — sanity, not exhaustive
    expected = {
        "search", "list_documents", "get_chunk", "compose_context",
        "add_summary", "verify_summary", "tag_chunk",
        "ocr_status", "list_pending_ocr",
        "cluster_chunks", "review_stats",
        "enqueue_review", "claim_next_review", "complete_review",
        "check_grounding", "verify_claim",
    }
    missing = expected - names
    assert not missing, f"MCP missing tools: {missing}"


@pytest.mark.mcp
def test_mcp_search_round_trip(corpus: Corpus, tmp_path: Path) -> None:
    """End-to-end: ingest a doc through the Corpus, call the MCP `search`
    tool, parse the response, assert the hit shape matches what an
    agent would see."""
    p = tmp_path / "x.md"
    p.write_text(
        "# Topic\n\nDense passage retrieval uses BERT and a dual encoder.\n",
        encoding="utf-8",
    )
    corpus.ingest(p)

    from kglite_docs.mcp_server.server import build_app
    app = build_app(corpus)

    # `call_tool` returns a tuple (content_blocks, structured_output)
    result = asyncio.run(app.call_tool("search", {"query": "dense retrieval", "top_k": 3}))
    content, structured = result
    # structured is the tool's return value rendered into JSON-compatible form
    assert structured is not None
    hits = structured["result"] if isinstance(structured, dict) and "result" in structured else structured
    assert isinstance(hits, list)
    assert hits, "search returned no hits"
    h0 = hits[0]
    assert "id" in h0 and "score" in h0


@pytest.mark.mcp
def test_mcp_ocr_status_round_trip(corpus: Corpus, tmp_path: Path) -> None:
    """Confirm an MCP tool that returns a dict (not a list) round-trips
    its shape too."""
    from PIL import Image
    img = tmp_path / "scan.png"
    Image.new("RGB", (32, 32)).save(img)
    corpus.ingest(img)

    from kglite_docs.mcp_server.server import build_app
    app = build_app(corpus)

    _, structured = asyncio.run(app.call_tool("ocr_status", {}))
    payload = structured["result"] if isinstance(structured, dict) and "result" in structured else structured
    assert payload["pending_pages"] == 1
    assert payload["documents_total"] == 1


@pytest.mark.mcp
def test_mcp_review_kanban_round_trip(corpus: Corpus, tmp_path: Path) -> None:
    """Full kanban: enqueue → claim_next → complete, all via MCP."""
    p = tmp_path / "d.md"
    p.write_text("# A\n\nbody\n", encoding="utf-8")
    corpus.ingest(p)
    chunk_id = corpus.cypher(
        "MATCH (c:Chunk) RETURN c.id AS id LIMIT 1"
    ).to_list()[0]["id"]

    from kglite_docs.mcp_server.server import build_app
    app = build_app(corpus)

    # enqueue
    _, enq = asyncio.run(app.call_tool(
        "enqueue_review",
        {"target_id": chunk_id, "target_kind": "Chunk", "priority": 1},
    ))
    ticket_id = enq["result"] if isinstance(enq, dict) and "result" in enq else enq
    assert isinstance(ticket_id, str)

    # claim_next
    _, claim = asyncio.run(app.call_tool(
        "claim_next_review", {"agent_id": "agent-mcp"},
    ))
    ticket = claim["result"] if isinstance(claim, dict) and "result" in claim else claim
    assert ticket["status"] == "in_review"
    assert ticket["claimed_by"] == "agent-mcp"

    # complete
    _, done = asyncio.run(app.call_tool(
        "complete_review",
        {
            "ticket_id": ticket["ticket_id"], "agent_id": "agent-mcp",
            "verdict": "reviewed", "accuracy": 0.9,
            "tags": ["mcp-test"],
        },
    ))
    final = done["result"] if isinstance(done, dict) and "result" in done else done
    assert final["status"] == "reviewed"
