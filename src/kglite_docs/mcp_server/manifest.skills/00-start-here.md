---
name: 00-start-here
description: "First contact — the CLI-style tool surface, the canonical happy path, and where to find more methodology when needed."
references_tools:
  - document
  - search
  - summary
  - agent
  - graph_overview
  - cypher_query
---

# Start here — kglite-docs in 90 seconds

You're connected to an **agent-first knowledge base for documents**. Multi-format ingest (PDF / DOCX / PPTX / MD / HTML / TXT / images), 1024-dim BAAI/bge-m3 embeddings, typed Cypher under the hood.

## The shape of the graph

```
(Document) ─HAS_PAGE→ (Page) ─HAS_CHUNK→ (Chunk) ─NEXT_CHUNK→ (Chunk)
                                            │
                                            ├─SUMMARIZES← (Summary) ─VERIFIED_BY→ (Agent)
                                            ├─TAGGED_AS→ (Tagging) ─OF_TAG→ (Tag)
                                            ├─HAS_TRANSLATION→ (Translation)
                                            ├─IN_CLUSTER→ (Cluster)
                                            └─←VIEWED─ (Agent)
```

Status / role / lifecycle live as **secondary labels** — `(c:Chunk:Ready)`, `(s:Summary:Verified)`, `(a:Agent:Reviewer:LLM)`. Cross-type label queries work: `MATCH (n:Verified)` returns every "verified" thing regardless of primary type.

## The interface is a CLI

11 noun tools, each takes a verb as the first positional arg. Think `git branch list`, not 45 standalone functions.

| Tool | Actions |
|---|---|
| `document(action, ...)` | `ingest` · `list` · `get` · `export` · `compare` |
| `chunk(action, ...)` | `get` · `similar` |
| `search(query, mode=...)` | `hits` (default) · `compose` |
| `summary(action, ...)` | `add` · `verify` · `list` · `ground` · `claim` · `consensus` |
| `tag(action, ...)` | `add` · `remove` · `list` · `chunks` |
| `agent(action, ...)` | `upsert` · `get` · `list` · `activity` |
| `review(action, ...)` | `enqueue` · `enqueue_chunks` · `claim_next` · `claim` · `unclaim` · `complete` · `list` · `get` · `stats` |
| `ocr(action, ...)` | `status` · `pending` · `submit` |
| `cluster(action, ...)` | `run` · `get` · `list` · `export` |
| `translate(action, ...)` | `add` · `list` · `assemble` |
| `cypher_query(query)` | raw Cypher escape hatch |
| `graph_overview()` | schema + counts |

## The canonical happy path

For ~80% of tasks ("read these PDFs and …"):

```python
# 1. Ingest
document("ingest", path="/abs/path/paper.pdf")
# → {"doc_id": "doc_3e67...", "created": true, "chunk_count": 43, ...}

# 2. Retrieve (compose mode returns a budgeted bundle — preferred for LLM input)
ctx = search("dense retrieval methods", mode="compose", max_tokens=3000)
# → {"query": ..., "items": [{"chunk_id", "text", "score", ...}], "used_tokens": 2847}

# 3. Persist analysis
sid = summary("add", target_id="doc_3e67...#p2#c3",
              text="DPR uses dual BERT encoders with in-batch negatives.",
              agent_id="me")

# 4. Cross-check with a different agent (server rejects self-verification)
summary("verify", id=sid, verdict="verified",
        verifier_agent_id="reviewer")
```

## Useful next stops

- **`graph_overview()`** — schema dump with node counts. Good first call if you've never seen this graph.
- **`cypher_query("MATCH (n) ...")`** — raw Cypher escape hatch. Full kglite Cypher (`text_score`, `CALL louvain()`, multi-label predicates).
- **`document("compare", doc_a, doc_b, queries=[...])`** — side-by-side cross-doc retrieval.
- **`summary("ground", id, threshold=0.5)`** + **`summary("claim", text="...", against_chunk_ids=[...])`** — hallucination guards. Use `ground` *before* `verify` if you wrote the summary yourself.
- **`review("enqueue_chunks", doc_id=...)` → `review("claim_next", agent_id=...)` → `review("complete", ticket_id=..., agent_id=..., verdict=...)`** — kanban for batched / governance-y review work.

## Companion methodology skills

Load these via MCP prompts for specific task shapes:

- `/analyze-documents` — given N docs + a task (summarise / extract / answer).
- `/compare-documents` — given two docs, the comparison pattern.
- `/cross-checked-review` — the writer / verifier / grounding flow.

## A note on identity

Every mutation takes an `agent_id` string. Use something stable per agent session (`"claude-reviewer-1"`, not a uuid you regenerate every call) so writes attribute to one `Agent` node and `agent("activity", id="me")` works.
