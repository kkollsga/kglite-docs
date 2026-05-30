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

  document(action, ...)   ingest | index | list | get | sections | map |
                          export | compare | status | coverage
  chunk(action, ...)      get | similar
  search(query, mode=…)   hits (default) | compose
  study(action, ...)      define | assess | next | ledger | verify |
                          conclude | get | list | reopen | delete
  summary(action, ...)    add | verify | list | ground | claim | consensus
  tag(action, ...)        add | remove | list | chunks |
                          unclassified | classify | classify_many
  agent(action, ...)      upsert | get | list | activity
  review(action, ...)     enqueue | enqueue_chunks | claim_next | claim |
                          unclaim | complete | list | get | stats
  ocr(action, ...)        status | pending | request | submit
                          (no engine ships — YOU transcribe: request a
                           needs_ocr page → get its image + a verbatim prompt
                           → submit. agent_type routes to an OCR subagent.)
  cluster(action, ...)    run | get | list | export
  translate(action, ...)  add | list | assemble

  cypher_query("MATCH ...")  — raw Cypher escape hatch
  graph_overview()           — schema + node/edge counts

EMBEDDING IS OPTIONAL & EXPLICIT. `ingest` does NOT embed — it parses,
chunks, and stores fast (no model load). Run `document("index")` once
afterwards to embed and unlock `search`. Workflows that only browse,
run cypher, tag, review, OCR, export, or translate need no embeddings —
skip `index` entirely. (One-shot: `document("ingest", ..., embed=True)`.)

ORIENT FIRST: `document("status")` for a corpus snapshot, `document("map")` for a
triage overview (content_kind breakdown, boilerplate, low-quality, entity
coverage — so you route work without reading the corpus), `document("coverage")`
to see what's image-only/low-text (unanalyzed unless OCR'd) or unembedded —
all reported as data, never silently assumed. Chunks are pre-tagged with cheap
deterministic labels: `MATCH (c:Chunk:Table)` / `:HasMoney` / `:Boilerplate` etc. `document("sections",
doc_id=…)` lists a doc's sections (the grain between document and chunk); pass a
section's id as `section_id` to scope `study("next"/"ledger")`.

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
              rationale="...", agent_id="reader-1",
              provenance="primary_text"|"characterization"|"scanned_unread")
        — passing agent_id to next CLAIMS the chunks (punchcard), so a
          fan-out of analysts gets disjoint batches and never overlaps.
          One first-class, verifiable record per chunk; re-assessing supersedes.
          `provenance` (optional, default primary_text) records what you actually
          checked — the ledger surfaces it so unread/paraphrase-based calls show.
          Optional `quote`/`char_start`/`char_end` pin the exact passage (a
          pinpoint cite), validated against the chunk text.
  3. study("ledger", study_id=sid)            — weight-ranked evidence + tallies
     study("ledger", study_id=sid, stance="supports")   — just the supporting side
     — current-by-default: corrected assessments are hidden; pass
       include_superseded=True for the full history.
     study("conflicts", study_id=sid)   — chunks with opposing current
       assessments (supports vs against); review the contested evidence first.
  4. study("verify", assessment_id=..., verdict="verified"|"disputed"|
           "duplicate", verifier_agent_id="checker")    — 2nd-agent check
     study("supersede", assessment_id=old, stance=..., weight=..., agent_id=...)
       — correct another agent's row to one current winner (audit-preserving).
  5. SYNTHESIZE (mandatory — per-chunk scoring is blind to cross-chunk patterns):
     read study("synthesis_prompt"), then over the whole ledger hunt disparate
     treatment, contradictions, two-operative-outcomes, omissions, aggregations —
     record each as study("finding", supporting_chunk_ids=[…], stance=…, weight=…).
     A 2nd agent grades them: study("verify", finding_id=…, verdict=…,
     verifier_agent_id="checker") → confidence/escalation_state. Then mark the
     pass done: study("synthesize", study_id=sid, agent_id="lead").
  6. study("conclude", study_id=sid, text="...", agent_id="lead")
       — REFUSES until step 5 ran (so you can't ship a confident-incomplete
         conclusion). Genuinely nothing to synthesize? pass
         acknowledge_no_synthesis=True to record an audited skip.
  Manage studies with study("list") / get / reopen / delete.

MANY STUDIES ON ONE BIG CORPUS? CLASSIFY ONCE, ROUTE MANY. Re-reading a
1000-page case for every study is wasteful. Classify each chunk once into a
domain element schema (a legal pack is loaded by default), then scope studies
to the relevant element:
  1. for ch in tag("unclassified", agent_id="cls-1"):
        tag("classify", chunk_id=ch["id"], elements=[...], agent_id="cls-1")
        — element ids from the schema (legal: holding, judge_remark, settlement,
          testimony, statute, …). Empty elements = "no element applies".
  2. document("map")   — see the element breakdown + how many are unclassified.
  3. study("next", study_id=sid, element="judge_remark")   — reads judge-remark
        chunks FIRST (advisory: full list still returned, nothing hidden).
     study("ledger", study_id=sid, element="judge_remark") — carries a
        `scope_coverage` block (in_scope vs excluded) so an early stop is informed.

LOADABLE METHODOLOGY (load via MCP prompts):
  /00-start-here          — overview, this doc but expanded.
  /analyze-documents      — given N docs + a task, the canonical pipeline.
  /compare-documents      — the two-doc comparison idiom.
  /cross-checked-review   — the write/verify/ground flow.

Every mutation takes `agent_id`. Use one stable id per agent session so
your writes attribute to one Agent node, queryable via
`agent("activity", id="me")`.
"""


def build_app(
    corpus: Any, *, warm_embedder: bool = True,
    schema_packs: tuple[str, ...] = ("legal",),
) -> Any:
    """Construct and return a FastMCP app wired to `corpus`.

    `warm_embedder` (default True) kicks off a background load of the
    bge-m3 ONNX session so the first `document('index')` / `search` is
    warm. Set False for pure non-embedding deployments to avoid loading
    ~2 GB that will never be used.

    `schema_packs` (default the bundled legal vocabulary) are domain element
    schemas to register at startup, so `tag('classify')` / `study(element=…)`
    know the vocabulary. Pass `()` for a domain-agnostic server."""
    for pack in schema_packs:
        try:
            from kglite_docs.schemas import load_schema
            load_schema(pack)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("could not load schema pack %r: %s", pack, exc)
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
