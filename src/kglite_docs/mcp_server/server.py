"""FastMCP server wiring.

Registers the typed kglite-docs tool surface, the bundled
`cypher_query` + `graph_overview` helpers from `mcp_methods.fastmcp`,
and the methodology skills as MCP prompts (so the agent can load
canonical workflows on demand).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("kglite_docs.mcp_server")


# Surfaced to MCP clients on `initialize` — first contact orientation.
# Hits the canonical happy path and points at the loadable skills for
# task-specific methodology.
_INSTRUCTIONS = """\
kglite-docs — agent-first knowledge base for documents.

You're connected to a typed graph backed by kglite. Documents are stored as
`(Document)-[:HAS_PAGE]->(Page)-[:HAS_CHUNK]->(Chunk)` with bge-m3 embeddings
on every chunk. Status / role / lifecycle live as secondary labels —
`(c:Chunk:Ready)`, `(s:Summary:Verified)`, `(a:Agent:Reviewer:LLM)` —
queryable as label predicates.

THE INTERFACE IS A PSEUDO-CLI: noun tools, each takes an action
verb as the first positional arg. Think `git branch list`, not 45
standalone functions.

  document(action, ...)   ingest | index | list | get | export | compare |
                          status | coverage
  chunk(action, ...)      get | similar
  search(query, mode=…)   hits (default) | compose
  study(action, ...)      define | assess | next | ledger | verify |
                          conclude | get | list | reopen | delete
  summary(action, ...)    add | verify | list | ground | claim | consensus
  tag(action, ...)        add | remove | list | chunks
  agent(action, ...)      upsert | get | list | activity
  review(action, ...)     enqueue | enqueue_chunks | claim_next | claim |
                          unclaim | complete | list | get | stats
  ocr(action, ...)        status | pending | submit
  cluster(action, ...)    run | get | list | export
  translate(action, ...)  add | list | assemble

  cypher_query("MATCH ...")  — raw Cypher escape hatch
  graph_overview()           — schema + node/edge counts

EMBEDDING IS OPTIONAL & EXPLICIT. `ingest` does NOT embed — it parses,
chunks, and stores fast (no model load). Run `document("index")` once
afterwards to embed and unlock `search`. Workflows that only browse,
run cypher, tag, review, OCR, export, or translate need no embeddings —
skip `index` entirely. (One-shot: `document("ingest", ..., embed=True)`.)

ORIENT FIRST: `document("status")` for a corpus snapshot, `document("coverage")`
to see what's image-only/low-text (unanalyzed unless OCR'd) or unembedded —
coverage is reported as data, never silently assumed.

THE HAPPY PATH (semantic search/analysis):

  1. document("ingest", path="/abs/paper.pdf")
        — or directory=..., or text=... + title=...  (fast; no embeddings)
  2. document("index")
        — embed the new chunks. Required before search. Bounded per call
        (~30s); if the result shows `pending > 0`, call it again until
        `pending == 0` (a big corpus takes a few calls).
  3. search(query, mode="compose", max_tokens=3000)
        — budgeted, ranked context bundle (preferred over plain hits
        when you're handing the result to an LLM)
  4. summary("add", target_id="...", text="...", agent_id="me")
        — persist analysis back to the graph
  5. summary("verify", id=sid, verdict="verified",
             verifier_agent_id="other") — second agent verifies.
             Self-verification is rejected by the server.

For "compare two docs": ingest + index, then
document("compare", doc_a, doc_b, queries=[...]).
For fact-checking: summary("ground", id) + summary("claim", text=...).

THE EVIDENCE-STUDY PATH (judge documents for/against a claim — needs NO
embeddings; iterate chunks via study("next"), so you can skip index):

  1. sid = study("define", question="X is necessary", agent_id="lead")
  2. for ch in study("next", study_id=sid, agent_id="reader-1", doc_id="..."):
        study("assess", study_id=sid, chunk_id=ch["id"],
              stance="supports"|"against"|"neutral"|"deferred", weight=0..1,
              rationale="...", agent_id="reader-1")
        — passing agent_id to next CLAIMS the chunks (punchcard), so a
          fan-out of analysts gets disjoint batches and never overlaps.
          One first-class, verifiable record per chunk; re-assessing supersedes.
  3. study("ledger", study_id=sid)            — weight-ranked evidence + tallies
     study("ledger", study_id=sid, stance="supports")   — just the supporting side
  4. study("verify", assessment_id=..., verdict="verified"|"disputed"|
           "duplicate", verifier_agent_id="checker")    — 2nd-agent check
  5. study("conclude", study_id=sid, text="...", agent_id="lead")
  Manage studies with study("list") / get / reopen / delete.

LOADABLE METHODOLOGY (load via MCP prompts):
  /00-start-here          — overview, this doc but expanded.
  /analyze-documents      — given N docs + a task, the canonical pipeline.
  /compare-documents      — the two-doc comparison idiom.
  /cross-checked-review   — the write/verify/ground flow.

Every mutation takes `agent_id`. Use one stable id per agent session so
your writes attribute to one Agent node, queryable via
`agent("activity", id="me")`.
"""


def build_app(corpus: Any, *, warm_embedder: bool = True) -> Any:
    """Construct and return a FastMCP app wired to `corpus`.

    `warm_embedder` (default True) kicks off a background load of the
    bge-m3 ONNX session so the first `document('index')` / `search` is
    warm. Set False for pure non-embedding deployments to avoid loading
    ~2 GB that will never be used."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:  # pragma: no cover - extras gate
        raise RuntimeError(
            "kglite-docs MCP server requires `pip install 'kglite-docs[mcp]'`"
        ) from e

    app = FastMCP("kglite-docs", instructions=_INSTRUCTIONS)

    # Register typed tools
    from kglite_docs.mcp_server.tools import register_typed_tools
    register_typed_tools(app, corpus)

    # Register the kglite-style escape hatches from mcp_methods
    try:
        from mcp_methods.fastmcp import register_cypher_query, register_overview
        register_cypher_query(app, corpus.store.g)
        register_overview(app, corpus.store.g, overview_prefix=(
            "kglite-docs knowledge base. Use the typed tools "
            "(`search`, `get_chunk`, etc.) first; reach for `cypher_query` "
            "only when the typed surface doesn't cover what you need."
        ))
    except Exception as exc:
        log.warning("could not register mcp-methods helpers: %s", exc)

    # Register methodology skills as MCP prompts. Manifest lives next to
    # this file; the project layer (manifest.skills/) is auto-detected.
    try:
        from mcp_methods import SkillRegistry
        from mcp_methods.fastmcp import register_skills_as_prompts
        manifest = Path(__file__).parent / "manifest.yaml"
        if manifest.exists():
            reg = SkillRegistry.from_manifest(str(manifest), include_bundled=False)
            registered = register_skills_as_prompts(app, reg)
            log.info("registered %d skill prompts", registered)
        else:  # pragma: no cover — manifest packaged with wheel
            log.warning("manifest.yaml missing at %s — skills not loaded", manifest)
    except Exception as exc:
        log.warning("could not register skills: %s", exc)

    # Warm the bge-m3 ONNX session off-thread so the first index/search
    # doesn't pay the ~8s session-init + HF-hub round-trip inline (which
    # blows the MCP client's per-call timeout from a blind state).
    # No-op-safe for stub embedders (load() is a no-op there).
    if warm_embedder:
        try:
            from kglite_docs.embed import prefetch_embedder
            prefetch_embedder(corpus.embedder)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("could not start embedder warm-load: %s", exc)

    return app
