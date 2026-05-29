"""MCP server tests — boot the FastMCP app and round-trip real tool calls."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from kglite_docs import Corpus
from kglite_docs.errors import NotIndexedError


@pytest.mark.mcp
def test_mcp_tools_registered(corpus: Corpus) -> None:
    """The app should boot and expose the CLI-style noun dispatchers."""
    from kglite_docs.mcp_server.server import build_app

    app = build_app(corpus)
    assert app is not None
    tools = asyncio.run(app.list_tools())
    names = {t.name for t in tools}
    expected = {
        "document", "chunk", "search", "summary",
        "tag", "agent", "review", "ocr",
        "cluster", "translate", "study",
        "cypher_query", "graph_overview",
    }
    missing = expected - names
    assert not missing, f"MCP missing tools: {missing}"


@pytest.mark.mcp
def test_mcp_study_workflow_round_trip(corpus: Corpus, tmp_path: Path) -> None:
    """define → assess → ledger → verify → conclude via the MCP study tool."""
    p = tmp_path / "ev.md"
    body = "# Doc\n\n" + "\n\n".join(
        f"Paragraph {i}: dense single-vector retrieval versus late interaction "
        f"token matching, point {i} with filler." for i in range(250)
    )
    p.write_text(body, encoding="utf-8")
    corpus.ingest(p)

    from kglite_docs.mcp_server.server import build_app
    app = build_app(corpus, warm_embedder=False)

    sid = _call(app, "study", {"action": "define", "question": "Late interaction is necessary", "agent_id": "lead"})
    assert isinstance(sid, str)

    nxt = _call(app, "study", {"action": "next", "study_id": sid, "limit": 3}, as_list=True)
    assert isinstance(nxt, list) and nxt
    cid = nxt[0]["id"]

    _call(app, "study", {"action": "assess", "study_id": sid, "chunk_id": cid,
                         "stance": "supports", "weight": 0.8, "rationale": "x", "agent_id": "r1"})
    led = _call(app, "study", {"action": "ledger", "study_id": sid})
    assert led["rows"][0]["chunk_id"] == cid and led["rows"][0]["weight"] == 0.8
    assert led["tallies"]["supports"] == 1

    aid = led["rows"][0]["assessment_id"]
    v = _call(app, "study", {"action": "verify", "assessment_id": aid,
                            "verdict": "verified", "verifier_agent_id": "checker"})
    assert v["status"] == "verified"

    _call(app, "study", {"action": "conclude", "study_id": sid, "text": "Supported.", "agent_id": "lead"})
    got = _call(app, "study", {"action": "get", "study_id": sid})
    assert got["conclusions"][0]["text"] == "Supported."


@pytest.mark.mcp
def test_mcp_document_status_and_coverage(corpus: Corpus, tmp_path: Path) -> None:
    """document('status') + document('coverage') round-trip via the MCP app."""
    p = tmp_path / "s.md"
    p.write_text("# T\n\nsome body text here for coverage\n", encoding="utf-8")
    corpus.ingest(p)

    from kglite_docs.mcp_server.server import build_app
    app = build_app(corpus, warm_embedder=False)

    st = _call(app, "document", {"action": "status"})
    assert st["docs"] == 1 and "unembedded" in st

    cov = _call(app, "document", {"action": "coverage"})
    assert "summary" in cov and isinstance(cov["documents"], list)


@pytest.mark.mcp
def test_mcp_tag_list_exposes_confidence(corpus: Corpus, tmp_path: Path) -> None:
    """tag('list') must surface confidence so the typed surface can rank."""
    p = tmp_path / "t.md"
    p.write_text("# T\n\nbody text here\n", encoding="utf-8")
    corpus.ingest(p)
    cid = corpus.cypher("MATCH (c:Chunk:Ready) RETURN c.id AS id LIMIT 1").to_list()[0]["id"]

    from kglite_docs.mcp_server.server import build_app
    app = build_app(corpus, warm_embedder=False)
    _call(app, "tag", {"action": "add", "chunk_id": cid, "name": "important",
                       "agent_id": "me", "confidence": 0.75})
    rows = _call(app, "tag", {"action": "list", "chunk_id": cid}, as_list=True)
    assert rows and rows[0].get("confidence") == 0.75


@pytest.mark.mcp
def test_mcp_skill_prompts_registered(corpus: Corpus) -> None:
    """Skill markdown files should surface as MCP prompts."""
    from kglite_docs.mcp_server.server import build_app

    app = build_app(corpus)
    prompts = asyncio.run(app.list_prompts())
    names = {p.name for p in prompts}
    expected = {
        "00-start-here",
        "analyze-documents",
        "compare-documents",
        "cross-checked-review",
    }
    missing = expected - names
    assert not missing, f"MCP missing skill prompts: {missing}"


def _call(app, name: str, args: dict, as_list: bool = False):
    """Call an MCP tool and return its parsed value.

    FastMCP returns either ``(unstructured, structured)`` for typed
    tools or just ``unstructured`` (a list of ContentBlock) for
    Any-typed tools. For unstructured-list mode, lists get exploded
    into one block per element — pass ``as_list=True`` if the
    expected return is a list (avoids "list of 1" collapsing to dict).
    """
    import asyncio
    import json
    r = asyncio.run(app.call_tool(name, args))
    if isinstance(r, tuple) and len(r) == 2:
        _, structured = r
        if isinstance(structured, dict) and "result" in structured:
            return structured["result"]
        return structured
    if isinstance(r, list):
        decoded = []
        for block in r:
            text = getattr(block, "text", None)
            if text is None:
                decoded.append(block)
                continue
            try:
                decoded.append(json.loads(text))
            except (json.JSONDecodeError, TypeError):
                decoded.append(text)
        if as_list:
            return decoded
        return decoded[0] if len(decoded) == 1 else decoded
    return r


@pytest.mark.mcp
def test_mcp_document_ingest_index_then_search(corpus: Corpus, tmp_path: Path) -> None:
    """End-to-end via MCP: ingest (no embed) -> index -> search()."""
    p = tmp_path / "x.md"
    p.write_text(
        "# Topic\n\nDense passage retrieval uses BERT and a dual encoder.\n",
        encoding="utf-8",
    )

    from kglite_docs.mcp_server.server import build_app
    app = build_app(corpus)

    ing_result = _call(app, "document", {"action": "ingest", "path": str(p)})
    assert isinstance(ing_result, dict)
    assert ing_result["created"] is True
    assert ing_result["chunk_count"] >= 1
    # Embedding is opt-in: ingest alone embeds nothing and hints at index.
    assert ing_result["embedded"] == 0
    assert "hint" in ing_result and "index" in ing_result["hint"]

    # Search before indexing is a loud signal, not a silent [] — the corpus
    # has ready chunks but none embedded (BUG-2/FEAT-3).
    with pytest.raises(NotIndexedError):
        corpus.search("dense retrieval", top_k=3)

    idx = _call(app, "document", {"action": "index"})
    assert isinstance(idx, dict)
    assert idx["embedded"] >= 1
    assert idx["pending"] == 0

    hits = _call(app, "search", {"query": "dense retrieval", "top_k": 3}, as_list=True)
    assert isinstance(hits, list)
    assert hits, "search returned no hits after index"
    assert "id" in hits[0] and "score" in hits[0]

    ctx = _call(app, "search", {"query": "dense retrieval", "mode": "compose", "max_tokens": 500})
    assert isinstance(ctx, dict)
    assert "items" in ctx and "used_tokens" in ctx


@pytest.mark.mcp
def test_mcp_ingest_without_embedding_is_browsable(corpus: Corpus, tmp_path: Path) -> None:
    """A non-semantic workflow: ingest (no embed) then browse chunks via
    cypher — never touches the embedding model."""
    p = tmp_path / "y.md"
    p.write_text("# H\n\nSome browsable body text here.\n", encoding="utf-8")

    from kglite_docs.mcp_server.server import build_app
    app = build_app(corpus, warm_embedder=False)

    r = _call(app, "document", {"action": "ingest", "path": str(p)})
    assert r["chunk_count"] >= 1
    assert r["embedded"] == 0

    # Chunks exist and are tracked as unembedded via the c.embedded property.
    n = corpus.count_unembedded()
    assert n >= 1
    rows = corpus.cypher(
        "MATCH (c:Chunk:Ready) WHERE c.embedded = false RETURN count(c) AS n"
    ).to_list()
    assert int(rows[0]["n"]) == n


@pytest.mark.mcp
def test_index_is_bounded_and_loopable(corpus: Corpus, tmp_path: Path) -> None:
    """index() does bounded work per call and is safe to loop until
    pending hits 0 — the guard against a single index call blowing a
    per-call timeout on a large corpus."""
    p = tmp_path / "many.md"
    # Chunker targets ~512 tokens/chunk, so generate enough prose to yield
    # several chunks (each paragraph ~20 tokens × 120 ≈ 2.4k tokens).
    body = "# Doc\n\n" + "\n\n".join(
        f"Paragraph {i}: dense passage retrieval and late interaction are "
        f"distinct neural retrieval techniques discussed at length, item {i}."
        for i in range(120)
    )
    p.write_text(body, encoding="utf-8")
    corpus.ingest(p)
    total = corpus.count_unembedded()
    assert total >= 3, "need several chunks to exercise the cap"

    # One chunk at a time → each call makes bounded progress, pending shrinks.
    seen = []
    guard = 0
    while corpus.count_unembedded() and guard < 100:
        r = corpus.index(max_chunks=1)
        assert r["embedded"] == 1
        seen.append(r["pending"])
        guard += 1
    assert corpus.count_unembedded() == 0
    assert seen == sorted(seen, reverse=True), "pending should monotonically decrease"
    assert len(seen) == total
    # And search works once fully indexed.
    assert corpus.search("distinct content", top_k=2)


@pytest.mark.mcp
def test_mcp_ingest_persists_to_disk(corpus: Corpus, tmp_path: Path) -> None:
    """Tool-driven ingest must survive: reopening the .kgl shows the doc."""
    p = tmp_path / "z.md"
    p.write_text("# Persist\n\nThis must be on disk after the tool call.\n", encoding="utf-8")

    from kglite_docs.mcp_server.server import build_app
    app = build_app(corpus, warm_embedder=False)

    r = _call(app, "document", {"action": "ingest", "path": str(p)})
    doc_id = r["doc_id"]

    db_path = corpus.store.path
    assert db_path is not None and Path(db_path).exists(), "ingest did not write the .kgl"

    reopened = Corpus.open(db_path)
    found = reopened.cypher(
        "MATCH (d:Document {id: $id}) RETURN d.id AS id", params={"id": doc_id}
    ).to_list()
    assert found and found[0]["id"] == doc_id


@pytest.mark.mcp
def test_mcp_ocr_status_round_trip(corpus: Corpus, tmp_path: Path) -> None:
    """ocr('status') round-trips its dict shape."""
    from PIL import Image
    img = tmp_path / "scan.png"
    Image.new("RGB", (32, 32)).save(img)
    corpus.ingest(img)

    from kglite_docs.mcp_server.server import build_app
    app = build_app(corpus)

    payload = _call(app, "ocr", {"action": "status"})
    assert payload["pending_pages"] == 1
    assert payload["documents_total"] == 1


@pytest.mark.mcp
def test_mcp_review_kanban_round_trip(corpus: Corpus, tmp_path: Path) -> None:
    """Full kanban via MCP: review('enqueue') → review('claim_next') → review('complete')."""
    p = tmp_path / "d.md"
    p.write_text("# A\n\nbody\n", encoding="utf-8")
    corpus.ingest(p)
    chunk_id = corpus.cypher(
        "MATCH (c:Chunk) RETURN c.id AS id LIMIT 1"
    ).to_list()[0]["id"]

    from kglite_docs.mcp_server.server import build_app
    app = build_app(corpus)

    ticket_id = _call(
        app, "review",
        {"action": "enqueue", "target_id": chunk_id, "target_kind": "Chunk", "priority": 1},
    )
    assert isinstance(ticket_id, str)

    ticket = _call(app, "review", {"action": "claim_next", "agent_id": "agent-mcp"})
    assert ticket["status"] == "in_review"
    assert ticket["claimed_by"] == "agent-mcp"

    final = _call(app, "review", {
        "action": "complete",
        "ticket_id": ticket["ticket_id"], "agent_id": "agent-mcp",
        "verdict": "reviewed", "accuracy": 0.9,
        "tags": ["mcp-test"],
    })
    assert final["status"] == "reviewed"


@pytest.mark.mcp
def test_mcp_summary_write_verify_ground(corpus: Corpus, tmp_path: Path) -> None:
    """summary('add') → summary('ground') → summary('verify') via MCP."""
    p = tmp_path / "s.md"
    p.write_text("# T\n\nDense retrieval uses dual BERT encoders.\n", encoding="utf-8")
    corpus.ingest(p)
    chunk_id = corpus.cypher(
        "MATCH (c:Chunk) RETURN c.id AS id LIMIT 1"
    ).to_list()[0]["id"]

    from kglite_docs.mcp_server.server import build_app
    app = build_app(corpus)

    sid = _call(app, "summary", {
        "action": "add", "target_id": chunk_id,
        "text": "Dense retrieval uses dual BERT encoders.",
        "agent_id": "writer",
    })
    assert isinstance(sid, str)

    g = _call(app, "summary", {"action": "ground", "id": sid, "threshold": 0.4})
    assert "supported_fraction" in g

    v = _call(app, "summary", {
        "action": "verify", "id": sid,
        "verdict": "verified", "verifier_agent_id": "reviewer",
    })
    assert v["status"] == "verified"
