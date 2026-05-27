# Architecture

## High-level shape

```
┌────────────────────────────────────────────────────────────────┐
│  Agents (Claude Code, Claude Desktop, custom MCP clients)      │
└────────────────────────┬───────────────────────────────────────┘
                         │ MCP / stdio
┌────────────────────────▼───────────────────────────────────────┐
│  kglite_docs.mcp_server (FastMCP)                              │
│   typed tools: search / add_summary / verify / tag / cluster …  │
│   escape hatches: cypher_query / graph_overview (mcp_methods)   │
└────────────────────────┬───────────────────────────────────────┘
                         │ Python
┌────────────────────────▼───────────────────────────────────────┐
│  Corpus façade                                                  │
│   ingest / search / enrich / cluster / ocr / translate / export │
└────────────────────────┬───────────────────────────────────────┘
                         │
       ┌─────────────────┼─────────────────┐
       ▼                 ▼                 ▼
   kglite             BgeM3Embedder      pymupdf4llm/python-docx/…
   (.kgl on disk)     (ONNX, CLS-pool)   (parsing)
```

## Why kglite is the storage layer

We considered building a bespoke Rust crate but found kglite already provides every primitive we'd otherwise reimplement:

- `.kgl` single-file persistence — no DB server, no migrations
- Multiple embedding stores per `(node_type, text_column)` — independent vector indices for chunks, summaries, document titles
- Vector search with cosine / dot / euclidean / poincaré
- Native clustering procedures (`CALL cluster`, `CALL louvain`, `CALL connected_components`)
- Full Cypher engine for arbitrary queries
- The exact `BgeM3Embedder` we want (ONNX, CLS pool, 8192-token cap, idle cooldown)
- An MCP framework binding via `mcp-methods`

Wrapping kglite means we ship a pure-Python wheel — no maturin build matrix.

## Graph model

| Node | Identity | Carries |
|---|---|---|
| `Document` | sha256(file bytes) | title, path, format, ingested_at, page_count, byte_size |
| `Page` | `{doc_id}#p{n}` | page_number, markdown, has_text, needs_ocr |
| `Chunk` | `{doc_id}#p{n}#c{k}` | text, token_count, headings_json, status, text_hash, view_count, last_viewed_at |
| `Summary` | uuid | text, depth, model, verification_status, verified_at, verifier_notes, source_text_hash |
| `Tag` | slug(name) | name, kind, description |
| `Tagging` | uuid | chunk_id, tag_id, by_agent, created_at, confidence (reified to allow multi-agent distinct applications) |
| `Translation` | uuid | chunk_id, target_lang, text, model, status, source_text_hash |
| `Cluster` | `{run_id}__c{label}` | algorithm, run_id, created_at, size, note |
| `Agent` | caller-supplied | kind, model, first_seen, last_seen, action_count |
| `View` | uuid | agent_id, target_id, target_kind, at, context |
| `Note` | uuid | target_id, target_kind, text, agent_id, created_at |

Edges: `HAS_PAGE`, `HAS_CHUNK`, `NEXT_CHUNK`, `IN_CLUSTER`, `SIMILAR_TO`, `SUMMARIZES`, `VERIFIES`, `TAGGED_AS`, `OF_TAG`, `HAS_TRANSLATION`, `AUTHORED`, `VERIFIED_BY`, `VIEWED`, `ANNOTATED`, `CITES`.

### Why reify Tagging?

kglite enforces at most one edge per `(src, dst, type)` triple. A naive `(Chunk)-[:TAGGED_AS]->(Tag)` collapses two distinct agent applications into one edge, losing provenance. We sidestep that by inserting a `Tagging` node per (chunk × tag × agent), so Alice's "important" tag and Bob's "important" tag are *separate* `Tagging` nodes pointing at the same `Tag`. The wire-level Cypher is more verbose; the typed `Corpus.tag_chunk()` API hides it.

## Identity propagation

Every state-changing operation in the library takes an `agent_id` parameter. The server lazy-registers the agent on first sight and bumps `last_seen` + `action_count` on each call. This means `list_agents()` always reflects current activity without an explicit registration step.

Read-only operations (`get_document`, `cypher_query`, `graph_overview`) don't require an agent id. `search` / `get_chunk` accept one optionally — when provided, they record a view (and a `View` node if there's worthwhile context like the query string).

## Idempotency + staleness

- **Re-ingest of the same file**: skipped (file hash already present).
- **Re-ingest of a *modified* file**: keyed off the new file hash → new `Document` node. Old document remains; old summaries are not deleted but may be flagged `stale` (via `enrich.mark_stale_for_doc`) when `source_text_hash` no longer matches.
- **OCR re-submission**: deletes the placeholder `needs_ocr` chunks on that page and replaces with fresh ones.
- **Tag re-application by the same agent**: no-op; returns `created: False`.

## Where to look in the code

| Concern | Module |
|---|---|
| Public Python API | `corpus.py` |
| Schema names + constants | `schema.py` |
| `KnowledgeGraph` wrapping | `store.py` |
| Parsing (per-format) | `ingest/parser.py`, `ingest/formats.py` |
| Chunking | `ingest/chunker.py` |
| Ingest orchestration | `ingest/pipeline.py` |
| Embeddings | `embed.py` (subclasses kglite's `BgeM3Embedder`) |
| Summaries + verification | `enrich.py` |
| Tags | `tagging.py` |
| Agents + views | `activity.py` |
| OCR loop | `ocr.py` |
| Clustering | `cluster.py` (incl. numpy k-means + DBSCAN fallbacks) |
| Quality / grounding | `quality.py` |
| Translation | `translate.py` |
| Export to MD/DOCX/PDF | `export.py` |
| MCP server | `mcp_server/{server,tools,__main__}.py` |
| Agent caller abstraction | `agents.py` |
| CLI | `cli.py` |

## Performance characteristics

| Operation | Typical cost | Bottleneck |
|---|---|---|
| Ingest one ~10-page PDF | ~5s | bge-m3 embed (~50ms × N chunks; CPU ONNX) |
| Bulk ingest 16 PDFs | ~5min | embedding-dominated |
| `search` (10k chunks) | ~5ms | vector_search is brute-force in kglite |
| `cluster_chunks(kmeans, k=8)` on 1k chunks | ~200ms | python-side numpy |
| `add_summary` | ~10ms | one Cypher MATCH + one upsert + one embed |
| `verify_summary` | ~5ms | three Cyphers |
| `compose_context` | ~10ms + embed query | vector_search + Cypher join |

See `docs/perf.md` (or the consolidated plan) for the bottleneck audit and future levers.
