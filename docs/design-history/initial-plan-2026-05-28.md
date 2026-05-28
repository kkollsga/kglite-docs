# kglite-docs — agent-first PDF knowledge base (initial plan, 2026-05-28)

> **Historical snapshot.** This is the planning doc the project was built
> from. Some details drifted as we hit kglite bugs and the user added
> features mid-build — the *current* shape of the library lives in
> `docs/architecture.md`, `docs/getting-started.md`, and
> `CHANGELOG.md`. Kept here as a record of the design rationale.

## Context

We need a Python library that ingests PDFs and makes them queryable by AI agents over MCP. The agent must be able to:

- Find relevant chunks (semantic + filtered)
- Walk a document by structure (page → chunk → neighbors)
- Trigger OCR on scanned pages and patch results back in
- Write enrichments (summaries, extractions, notes) and have a *second* agent fact-check them
- Generate text efficiently from a budgeted, ranked context
- Discover related content via clusters / similarity / traversal

The user already maintains two Rust/Python libraries that cover most of the heavy lifting: **kglite** (Rust-backed knowledge graph with `.kgl` storage, native bge-m3 embeddings, `set_embeddings` / `vector_search` / `CALL cluster()` / `CALL louvain()`, Cypher engine, fluent API) and **mcp-methods** (Rust-backed MCP framework: YAML-driven `McpServer`, FastMCP helpers `register_cypher_query` / `register_overview` / `register_source_tools`, `ElementCache`, ripgrep-backed search). The local bge-m3 weights are cached at `/Volumes/EksternalHome/LLMs/hub/models--BAAI--bge-m3`.

**Packaging decision (locked in):** ships as a **separate library `kglite-docs`** under `kkollsga/kglite-docs`, depending on `kglite>=0.10.3` and `mcp-methods`. Not folded into kglite — keeps kglite's footprint lean (pymupdf is a ~30 MB wheel), keeps kglite's identity as a general-purpose graph DB, and lets `kglite-docs` evolve its document-specific schema and MCP surface on its own cadence. Aligns with the `kglite-*` family naming (`kglite-mcp-server`, etc.). Python module: `kglite_docs`. CLI: `kglite-docs-mcp` (or `python -m kglite_docs.mcp_server`).

## Backend recommendation: kglite, with typed MCP tools on top (not raw Cypher)

The user's open question was whether kglite/Cypher is the right interface for an agent working with chunked PDFs. **Yes for storage; no for the primary agent surface.**

- **Cypher is the wrong default tool for agents.** Asking an agent to write `MATCH (c:Chunk) WHERE c.doc_id = ... AND text_score(c.text, $q) > 0.7 RETURN c.text ORDER BY ...` for every lookup burns tokens and invites query mistakes. Typed tools like `search(query, filters, top_k)` are cheaper and harder to misuse.
- **Cypher is the right escape hatch.** Power-user / Claude / debugging path. Exposed as one tool (`cypher_query`) via mcp-methods' built-in helper.
- **kglite is the right *storage*.** It already gives us, in one package: `.kgl` single-file persistence; embedding stores per `(node_type, text_column)` (one for Chunks, one for Summaries, one for Document titles — searchable independently); vector search with cosine/dot/euclidean/poincaré; clustering (k-means/DBSCAN/louvain/connected_components) callable from Cypher; the proven BgeM3Embedder; the mcp-methods bindings via `kglite._mcp_internal`. Building a separate Rust crate would duplicate ~80% of this for no measured benefit. If a hot path emerges later (e.g. chunk dedup over millions of chunks), drop in a small maturin helper then.

So the architecture is **typed MCP tool layer → thin Python ingestion+enrichment service → kglite KnowledgeGraph → .kgl on disk**, with raw Cypher exposed as one escape-hatch tool.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  MCP clients (Claude Code, Claude Desktop, agent runners)      │
└────────────────────────┬───────────────────────────────────────┘
                         │ stdio / MCP
┌────────────────────────▼───────────────────────────────────────┐
│  kglite_docs.mcp_server  (FastMCP app + YAML manifest mode)    │
│    typed tools: search / get_chunk / add_summary / verify /…   │
│    escape hatch: cypher_query (reuses mcp-methods helper)      │
└────────────────────────┬───────────────────────────────────────┘
                         │ Python calls
┌────────────────────────▼───────────────────────────────────────┐
│  kglite_docs.store      KnowledgeGraph façade over kglite       │
│  kglite_docs.ingest     pymupdf4llm → chunker → embedder        │
│  kglite_docs.enrich     summaries, verifications, lineage       │
│  kglite_docs.ocr        scanned-page detection + agent handoff  │
│  kglite_docs.cluster    cluster runs + bookkeeping              │
│  kglite_docs.context    budgeted prompt composition             │
└────────────────────────┬───────────────────────────────────────┘
                         │
        ┌────────────────┼───────────────────────┐
        ▼                ▼                       ▼
   kglite                BgeM3Embedder            pymupdf4llm
   (.kgl file)           (local bge-m3 cache)     (pages → md)
```

## Graph schema

Designed to lean into the graph: agents, their views, their tags, and their summaries are all first-class nodes so the agent collaboration history is queryable end-to-end.

| Node | Key props |
|---|---|
| `Document` | `id` (hash of file bytes), `title`, `path`, `source_uri`, `ingested_at`, `page_count`, `mime`, `byte_size`, `metadata` (free-form JSON), `lang` |
| `Page` | `id` (`{doc_id}#p{n}`), `doc_id`, `page_number`, `has_text` (bool), `needs_ocr` (bool), `markdown` (raw pymupdf4llm output), `width_pt`, `height_pt` |
| `Chunk` | `id` (`{doc_id}#p{n}#c{k}`), `doc_id`, `page_number`, `chunk_index` (k on page), `text`, `token_count`, `headings` (JSON path), `status` (`ready`/`needs_ocr`/`empty`), `text_hash`, `view_count` (denormalised aggregate), `last_viewed_at` |
| `Summary` | `id` (uuid), `target_id` (chunk/page/document id), `target_kind`, `depth` (`chunk`/`section`/`document`), `text`, `model`, `created_at`, `verification_status` (`unverified`/`verified`/`disputed`/`stale`), `verified_at`, `verifier_notes`, `source_text_hash` (for staleness). Author + verifier are edges (`AUTHORED` / `VERIFIED_BY`) to `Agent`, not free-text props. |
| `Tag` | `id` (slug of name), `name`, `kind` (`topic`/`entity`/`custom`), `description` |
| `Cluster` | `id`, `algorithm`, `params`, `created_at`, `note` |
| `Agent` | `id` (caller-supplied identity, e.g. `"claude-opus-4.7"` or `"alice"`), `kind` (`llm`/`human`/`service`), `model`, `first_seen`, `last_seen`, `action_count` (denormalised) |
| `View` | `id` (uuid), `agent_id`, `target_id`, `target_kind`, `at`, `context` (free-form: query string, session id, etc.). Created when context is worth recording; pure visits also bump `Chunk.view_count`. |
| `Note` | `id` (uuid), `target_id`, `target_kind`, `text`, `agent_id`, `created_at` |

| Edge | Source → Target | Notes |
|---|---|---|
| `HAS_PAGE` | Document → Page | |
| `HAS_CHUNK` | Page → Chunk | also Document → Chunk for fast doc-scoped queries |
| `NEXT_CHUNK` | Chunk → Chunk | within doc, in reading order |
| `IN_CLUSTER` | Chunk → Cluster | written by clustering pass |
| `SIMILAR_TO` | Chunk → Chunk | optional, top-k vector neighbours; `score` prop |
| `SUMMARIZES` | Summary → Chunk/Page/Document | |
| `VERIFIES` | Summary → Summary | when verifier *writes* a verification summary rather than just a status flag |
| `TAGGED_AS` | Chunk/Document → Tag | props: `by_agent`, `created_at`, `confidence` (optional) — so the same chunk can be tagged by multiple agents distinctly |
| `CITES` | Chunk → Chunk \| Document | extracted citations (future) |
| `AUTHORED` | Agent → Summary/Note/View | the agent who wrote it |
| `VERIFIED_BY` | Summary → Agent | the agent who verified it (distinct from author; enforced) |
| `VIEWED` | Agent → Chunk | denormalised aggregate edge with `count`, `last_at`; backed by per-event `View` nodes when context is recorded |

**Identity propagation**: every MCP tool that mutates state or records activity takes an `agent_id` parameter. The server auto-creates the `Agent` node on first sight (lazy registration) and bumps `last_seen` + `action_count` on every call. Read-only tools (`get_document`, `cypher_query`, `graph_overview`) don't require it; `search` / `get_chunk` accept it optionally and record a `VIEWED` increment when supplied.

**Embedding stores** (separate, queryable independently):
- `('Chunk', 'text')` — primary semantic search
- `('Summary', 'text')` — find consensus / contradiction in enrichments
- `('Document', 'title')` — fast doc-level routing

## Chunking pipeline

1. `pymupdf4llm.to_markdown(pdf, page_chunks=True)` → list of per-page markdown with metadata.
2. Per page:
   - If `len(text.strip()) == 0` AND page has image content → mark `Page.needs_ocr = True`, create a single `Chunk` with `status='needs_ocr'` and no text. Embedding deferred until OCR submitted.
   - Else split markdown on heading boundaries (`#` levels) and paragraph breaks; pack greedy into ~512-token windows (bge-m3 tokenizer, never crossing a page) with ~64-token overlap. Inherit the current heading path into `Chunk.headings`.
3. Compute `text_hash = sha256(normalized_text)` so we can detect staleness later.
4. Batch-embed via `BgeM3Embedder.embed(texts)`; insert via `g.set_embeddings('Chunk', 'text', {chunk_id: vec})`.
5. Wire `NEXT_CHUNK` edges in reading order; wire `HAS_PAGE` / `HAS_CHUNK`.
6. Persist with `g.save(path)`.

Idempotency: ingestion keyed by `Document.id` = sha256(file bytes). Re-ingesting the same file is a no-op; re-ingesting a *changed* file creates a new Document and leaves the old summaries marked `stale` if any of their `source_text_hash` no longer matches.

## OCR for scanned pages (agent-driven)

1. Detection at ingest time: page text empty + page has at least one image → `needs_ocr=True`.
2. MCP tool `list_pending_ocr(doc_id=None, limit=20)` returns pages with `needs_ocr=True` plus a base64-encoded PNG render of the page (PyMuPDF `page.get_pixmap(dpi=200)`).
3. Agent reads the image, returns markdown via `submit_ocr(page_id, markdown, model, confidence)`. We then run the normal chunking pipeline on that markdown, flip `needs_ocr=False`, and record `ocr_metadata` (agent, model, ts) on the Page.
4. No local OCR engine in v1 — keeps the wheel pure-Python. Optional `[ocr]` extra later if there's demand.

## Enrichment + verification layer

**Write path** — `add_summary(target_id, text, depth, model, agent_id, tags=[])`:
- Creates `Summary` node with `verification_status='unverified'`, captures `source_text_hash` (= concat of underlying chunks' hashes), embeds the summary text into `('Summary','text')`.
- Returns the new `summary_id`.

**Verification path** — `verify_summary(summary_id, verdict, notes, verifier_agent_id)`:
- `verdict` is one of `verified` / `disputed` / `needs_revision`.
- **Server enforces**: `verifier_agent_id != summary.author_agent` (no self-verification).
- Updates `verification_status`, `verified_by`, `verified_at`, `verifier_notes`.
- If the verifier wants to record their *own* analysis rather than just a verdict, they call `add_summary(...)` first and then `link_verification(verifier_summary_id, target_summary_id)` which writes a `VERIFIES` edge — the model now sees a chain of analysis.

**Staleness** — on re-ingest of a document, any `Summary` whose `source_text_hash` doesn't match the regenerated chunks is marked `stale` automatically.

**Querying enrichments**:
- `get_summaries(target_id, status=None, depth=None)` → list of summaries with verification metadata.
- `find_consensus(query)` → semantic search across `Summary` embeddings, grouped by target, surfaces agreement/conflict counts.

## MCP tool surface

Bundled tools (typed; agent's primary interface):

| Tool | Purpose |
|---|---|
| `search` | Semantic + filtered chunk search. Args: `query`, `filters` (doc_id, tags, date range, page range, status), `top_k`, `with_summaries` (bool), `with_neighbors` (bool). |
| `list_documents` | Tabular doc listing with filters / sort / limit. |
| `get_document` | Doc metadata + table of contents (headings tree) + ingest status. |
| `get_chunk` | One chunk with optional neighbors and summaries inlined. |
| `similar_chunks` | Nearest neighbors of a given chunk_id. |
| `get_summaries` | List summaries on a target, filterable by status/depth. |
| `add_summary` | Write a new summary. |
| `verify_summary` | Apply a verification verdict. Self-verification rejected. |
| `link_verification` | Write `VERIFIES` edge between two summaries. |
| `list_pending_ocr` | List `needs_ocr` pages with rendered images. |
| `submit_ocr` | Patch OCR result back into a page; re-chunks + re-embeds. |
| `cluster_chunks` | Run clustering pass (k-means / louvain / dbscan); writes `Cluster` nodes + `IN_CLUSTER` edges. |
| `get_cluster` | Inspect a cluster: members, centroid label, neighbors. |
| `cluster_overview` | Tour: cluster sizes + top-tag summary. |
| `compose_context` | Returns a budgeted prompt-ready context bundle for a query (chunks + summaries within `max_tokens`). Lets agents skip the manual top-k → concat dance. |
| `tag_chunk` | Apply a tag to a chunk (creates `Tag` if new). Args: `chunk_id`, `tag_name`, `kind`, `agent_id`, optional `confidence`. |
| `untag_chunk` | Remove a tag from a chunk (only the calling agent's application). |
| `list_tags` | Tags filtered by document / chunk / kind / agent. |
| `chunks_by_tag` | All chunks carrying a given tag. |
| `record_view` | Explicit view record with context (e.g. the query that led to it). Auto-fires from `search`/`get_chunk` when `agent_id` is passed; this tool covers the manual case. |
| `add_note` | Free-form annotation tied to a chunk/doc by an agent. |
| `agent_activity` | Timeline + summary of one agent's actions (views, tags, summaries, verifications). |
| `list_agents` | Registered agents with last-seen + action counts. |
| `cypher_query` | Escape hatch. Reuses `mcp_methods.fastmcp.register_cypher_query`. |
| `graph_overview` | Schema inventory. Reuses `mcp_methods.fastmcp.register_overview`. |

Server modes (matching kglite's pattern):
- `kglite-docs-mcp --db corpus.kgl` — open an existing knowledge base.
- `kglite-docs-mcp --ingest <dir>` — ingest a directory then serve.
- `kglite-docs-mcp --manifest server.yaml` — YAML-driven config (env, trust gates, custom tool descriptions, embedder cooldown). Manifest schema lifted from kglite's pattern via `mcp_methods.server.Manifest`.

## Library API (Python)

```python
from kglite_docs import Corpus

corpus = Corpus.open("kb.kgl")           # or Corpus.create("kb.kgl")
corpus.ingest_pdf("paper.pdf")            # idempotent
corpus.ingest_dir("./pdfs", recursive=True)

hits = corpus.search("retrieval augmented generation", top_k=10,
                     filters={"doc_id": "..."})
ctx  = corpus.compose_context("RAG vs fine-tuning", max_tokens=4000)

sid = corpus.add_summary(chunk_id, "...", agent_id="claude-1", model="opus-4.7")
corpus.verify_summary(sid, verdict="verified", verifier_agent_id="claude-2", notes="...")

corpus.cluster_chunks(algorithm="louvain")
corpus.save()  # also auto-saves on close
```

## Project layout

```
kglite_docs/
├── pyproject.toml                # hatchling, deps: kglite, mcp-methods, pymupdf4llm, pymupdf
├── README.md                     # quickstart + MCP setup snippet
├── LICENSE                       # MIT
├── src/kglite_docs/
│   ├── __init__.py               # re-exports Corpus, schema constants
│   ├── corpus.py                 # Corpus façade (open/create/ingest/search/…)
│   ├── schema.py                 # node/edge type names, embedding-store keys
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── pipeline.py           # orchestrate parse → chunk → embed → write
│   │   ├── parser.py             # pymupdf4llm wrapper
│   │   ├── chunker.py            # token-aware paragraph chunker
│   │   └── hashing.py            # file_hash, text_hash
│   ├── embed.py                  # bge-m3 adapter (subclass / config of kglite's BgeM3Embedder)
│   ├── enrich.py                 # add_summary, verify_summary, link_verification, staleness
│   ├── ocr.py                    # needs_ocr detection, render page, submit_ocr
│   ├── cluster.py                # cluster_chunks, get_cluster, overview
│   ├── context.py                # compose_context budgeting
│   ├── store.py                  # KnowledgeGraph façade (load/save, transactions)
│   └── mcp_server/
│       ├── __init__.py
│       ├── __main__.py           # python -m kglite_docs.mcp_server
│       ├── server.py             # FastMCP wiring; YAML manifest mode
│       ├── tools.py              # tool registrations (typed; arg validation; help strings)
│       └── manifest.py           # thin wrapper over mcp_methods.server.Manifest
├── sample_data/pdfs/             # 16 PDFs already downloaded
├── tests/
│   ├── conftest.py               # tmp .kgl fixture; small bge-m3 stub or real if available
│   ├── test_ingest_pdf.py        # uses sample arXiv PDFs
│   ├── test_chunker.py           # boundary cases, token cap, page never crossed
│   ├── test_idempotency.py       # re-ingest is no-op; changed file marks stale
│   ├── test_ocr_flow.py          # synthesised scanned page; list → submit → re-embed
│   ├── test_enrich.py            # add/verify; self-verify rejected; staleness
│   ├── test_cluster.py           # louvain on a corpus produces stable buckets
│   ├── test_search.py            # semantic + metadata filtering
│   ├── test_compose_context.py   # budget respected; ordered by score
│   ├── test_mcp_smoke.py         # boot server, call each tool over in-memory transport
│   └── golden/                   # frozen expected outputs for chunking + search
├── docs/                         # mkdocs-material or sphinx; getting-started + reference
└── .github/workflows/
    ├── ci.yml                    # lint, mypy, pytest on 3.10–3.13 × macOS/Linux
    └── release.yml               # tag → build sdist + wheel, publish to PyPI via trusted publishing
```

## Testing strategy

- **Unit**: chunker, hashing, schema marshalling, verification rules (self-verification rejection, staleness flip). Fast, no model required.
- **Integration (real embedder)**: marker `@pytest.mark.embed` — uses local bge-m3 weights. Run in CI with weights cached in actions cache (size ≈ 2 GB → use `actions/cache` keyed on model id). For PRs from forks where the cache miss is too expensive, fall back to a `DummyEmbedder` that returns deterministic random vectors.
- **End-to-end**: ingest 3 of the sample arXiv PDFs, search across them, write & verify a summary, run a clustering pass, snapshot the resulting `.kgl` file's schema with `g.describe()` against a golden text.
- **OCR fixture**: rasterise page 1 of `bert.pdf` to an image-only PDF at test time using PyMuPDF — gives us a deterministic scanned fixture without committing binaries.
- **MCP smoke**: spin up the FastMCP app in an in-process test, list tools, call each tool with a representative input, assert response shape.
- Coverage target: 85%+ on `corpus.py`, `ingest/`, `enrich.py`. Tool registrations are smoke-tested rather than unit-tested.

## CI/CD + PyPI release

- **CI** (`ci.yml`): on push / PR — `ruff`, `mypy --strict`, `pytest -q`. Matrix Python 3.10/3.11/3.12/3.13 × macOS-14/ubuntu-latest. Cache `~/.cache/fastembed` for the bge-m3 weights to keep the embed-marked tests fast.
- **Release** (`release.yml`): triggered on `v*` tag. Builds sdist + pure-Python wheel via `python -m build`. Publishes to PyPI using **trusted publisher** (OIDC, no API token in repo). Creates a GitHub Release with the changelog excerpt.
- **Versioning**: SemVer; `__version__` in `__init__.py` driven by `hatch-vcs` from git tags.
- **Pre-release smoke**: `release.yml` first publishes to TestPyPI, installs into a clean venv, runs `python -c "from kglite_docs import Corpus; Corpus.create(':memory:')"` as a smoke test, then promotes to PyPI.
- **Distribution**: pure-Python wheel — no per-OS matrix, no maturin step (kglite provides the Rust).

## Verification (how we'll know it works end-to-end)

After implementation:

1. **Ingest** — `python -m kglite_docs.cli ingest sample_data/pdfs --db corpus.kgl` ingests all 16 PDFs in < 5 min on the user's hardware; `corpus.list_documents()` returns 16 rows.
2. **Search quality** — `corpus.search("transformer attention mechanism", top_k=5)` returns chunks where `attention_is_all_you_need.pdf` is at top-1.
3. **OCR loop** — script that rasterises `bert.pdf` to image-only, ingests, asserts `list_pending_ocr()` returns the pages, submits agent-style OCR text back, asserts chunks now `status='ready'` and embeddable.
4. **Verification rules** — `add_summary(..., agent='A')` then `verify_summary(..., verifier='A')` raises; with `verifier='B'` succeeds and flips status.
5. **Clustering** — `cluster_chunks('louvain')` on the corpus yields ≥ 3 clusters; chunks from the same paper concentrate in the same cluster (precision > 0.6 on a manual spot-check).
6. **MCP** — `claude mcp add kglite-docs -- python -m kglite_docs.mcp_server --db corpus.kgl`; in a Claude conversation, the tool list shows our typed surface; `search` + `compose_context` round-trip works against a real query.
7. **Release dry-run** — push a `v0.0.1rc1` tag → TestPyPI upload succeeds; `pip install -i https://test.pypi.org/simple/ kglite-docs==0.0.1rc1` into a clean venv; CLI runs.

## Milestones (rough)

- **M1 — Storage + ingest** (week 1): `Corpus`, schema, chunker, embed, save/load. `ingest_pdf` works end-to-end on one of the sample PDFs. Tests for chunker + hashing.
- **M2 — Search + enrichments** (week 2): `search`, `compose_context`, `add_summary`, `verify_summary`, staleness. Integration test on a 3-PDF corpus.
- **M3 — OCR + clustering** (week 3): scanned-page detection, `list_pending_ocr` / `submit_ocr`, `cluster_chunks`. Synthetic scanned fixture in tests.
- **M4 — MCP server + CLI** (week 4): all typed tools, YAML manifest support, `cypher_query` escape hatch. End-to-end MCP smoke test.
- **M5 — Docs, CI, release** (week 5): README quickstart, mkdocs site, CI matrix, TestPyPI dry-run, PyPI v0.1.0.

## All requirements (consolidated from the conversation)

The plan tracks every requirement the user surfaced, in one place:

- **pymupdf4llm as the PDF parser** ✅ wired in `ingest/parser.py`
- **Auto-chunk with `(page_number, chunk_index)` provenance** ✅ `Chunk.page_number` + `Chunk.chunk_index`
- **Embeddings per chunk** ✅ `set_embeddings('Chunk', 'text', ...)` keyed on chunk id
- **Auto-cluster chunks → connected groups** ✅ `cluster_chunks()` with k-means / louvain / DBSCAN; writes `IN_CLUSTER` edges and `Cluster` nodes
- **BAAI/bge-m3 from `/Volumes/EksternalHome/LLMs/hub/...`** ✅ `embed.make_embedder(cache_dir=...)` reuses the local cache
- **Optional Rust maturin backend** → ❌ decided against: kglite IS the Rust backend; revisit if profiling demands it
- **kkollsga/kglite as inspiration** ✅ explored via open-source MCP; we depend on its public API
- **kkollsga/mcp-methods reuse** ✅ `register_cypher_query` / `register_overview` wired as escape hatches in our FastMCP server
- **MCP server for ingestion + query** ✅ `kglite-docs-mcp` with the full typed surface
- **Quick doc list + filters** ✅ `list_documents(filters={...}, limit=...)`
- **Scanned pages support** ✅ detection at ingest; agent OCR loop via `list_pending_ocr` → `submit_ocr`
- **Agent-driven OCR (vision agent reads, returns markdown)** ✅ same path
- **Develop in one go with proper testing** ✅ pytest suite (53+ tests), stub embedder for speed, real-embed marker for slow ones
- **Publish to GitHub + PyPI** ✅ GitHub Actions workflows for CI + trusted-publisher PyPI release
- **LLM enrichment layer (summaries)** ✅ `Summary` nodes with author/model provenance; `add_summary`
- **Cross-checking / fact-checking by a *different* agent** ✅ `verify_summary` rejects self-verification server-side; statuses: unverified/verified/disputed/needs_revision/stale
- **Multiple ways to query + enrich** ✅ `search`, `similar_chunks`, `compose_context`, `cluster_overview`, `find_consensus`, `get_summaries`, `cypher_query`
- **Efficient text generation** ✅ `compose_context(query, max_tokens)` returns a budgeted, ranked context bundle ready to drop into an LLM prompt
- **Lean into graph structure** ✅ `Agent`, `View`, `Tagging`, `Note`, `Cluster`, `Summary`, `Translation` are all first-class nodes
- **Agent-view tracking** ✅ implicit on `search`/`get_chunk` when `agent_id` given; explicit `record_view`; aggregate `Chunk.view_count` + `Chunk.last_viewed_at`; `View` node for query-context recording
- **Agent-assigned tags** ✅ multi-agent distinct tagging via reified `Tagging` nodes (kglite enforces single edge per src/dst/type, so we reify); `confidence` optional
- **Multi-format ingestion** ✅ PDF / DOCX / PPTX / MD / HTML / TXT / images, all → uniform `Document → Page → Chunk`
- **Native translation support (chunk by chunk)** ✅ `Translation` nodes with author/model/status; `add_translation`, `assemble_translated_document`
- **Export to MD / DOCX / PDF** ✅ `export_document` / `export_cluster` / `export_summary` / `export_bundle` via `kglite_docs.export`
- **Real agent workflow demo (Sonnet)** ✅ `demos/workflow.py` uses `agents.call_agent` (Anthropic SDK if `ANTHROPIC_API_KEY` is set, else `claude -p` CLI fallback)
- **Anti-hallucination methods** ✅ `check_grounding(summary_id)` (per-sentence cosine to source) + `verify_claim(claim, against_chunk_ids)` (free-text claim → best-supporting chunks)
- **User-friendly API (with and without AI)** ✅ `Corpus` façade exposes everything in ~30 methods; CLI mirrors them; MCP surface mirrors them again
- **Publish to GitHub** ⏳ user-driven step (we set up the repo + push when approved)
- **GitHub Actions for PyPI** ⏳ `.github/workflows/{ci,release}.yml` shipped; trusted publisher needs one-time setup on PyPI by the user

## Performance + bottleneck considerations

What we ship in v0.1, with explicit acceptance criteria:

| Bottleneck | Current handling | Future lever |
|---|---|---|
| **bge-m3 cold start (~1s ONNX session init)** | `BgeM3Embedder` cool-down lifecycle from kglite; session stays resident for 15 min by default | `--no-cooldown` flag for batch ingest jobs that should release memory after |
| **Per-chunk embed cost** | Batch all chunks of a document in one `embedder.embed([...])` call; not per-chunk | Parallel batching across files in `ingest_dir` (thread-pool of embedder calls — needs careful lock around ORT session) |
| **`set_embeddings` replaces the store** | We added `Store.add_embeddings()` which pulls existing + merges + writes — incremental ingest doesn't lose prior embeddings | Push for an upstream `kglite.add_embeddings` API to avoid the read-merge-write round-trip |
| **Vector search at scale** | kglite's vector_search is brute-force over the selection. Fine for ~100K chunks | If we cross ~1M chunks, add a HNSW index in kglite (upstream feature) |
| **PyMuPDF parse + chunker tokenizer load** | Tokenizer is `lru_cache`-d so it's loaded once per process | For multi-process ingest, share via a sidecar daemon (v0.2+) |
| **DataFrame round-trips into kglite** | `Store.upsert_nodes` / `upsert_edges` use pandas → kglite. ~ms overhead per call; negligible vs embedding cost | Bulk-load all chunks for a document in one DataFrame (already done) |
| **JSON serialisation in cypher params** | kglite handles primitive types and lists; we stringify nested dicts as JSON in `metadata_json` / `headings_json` to be safe | Drop the JSON dance when kglite gains native nested-JSON property support |
| **`get_cluster` member fetch** | One query per cluster; OK for <100 clusters | Add a `cluster_overview_detailed()` that batches |
| **OCR page-image rendering at 200 DPI** | ~50ms per page; lazy — only on `list_pending_ocr(include_images=True)` | Stream as base64 to keep MCP payloads smaller, or write to disk + URL |

Bottleneck detection rituals we run before each release:

1. `python -c "from kglite_docs import Corpus; ..."` smoke ingest 16 PDFs; record total time + per-stage breakdown.
2. `cProfile` over a 16-doc ingest to surface dominant call sites.
3. `kglite`'s `g.profile()` on the heaviest Cypher queries.
4. Memory: `tracemalloc` peak around ingest + cluster passes.

## Open considerations

- **Concurrency**: kglite is process-local; concurrent agents on the same `.kgl` need a single-writer pattern. v1 ships a `Corpus` that holds a process-wide write lock; multi-process writers can wait for v0.2 (`fasteners` file lock or a tiny long-lived sidecar server).
- **Embedding store size**: bge-m3 is 1024-dim float32 → ~4 KB / chunk. 16 PDFs ≈ a few thousand chunks ≈ ~10 MB embedding state. `.kgl` handles this trivially.
- **Cypher-as-escape-hatch surface area**: agents will *occasionally* need to write Cypher (e.g. "summaries created last week and not yet verified"). We ship a Cypher cheat-sheet skill file via `mcp_methods.fastmcp.register_skills_as_prompts`.
- **Future-Rust path**: if/when a hot path appears (chunk dedup at 10M+ scale, GPU-side reranking), drop a small maturin helper into `kglite_docs_rs/` without disturbing the Python API.
- **PDF export polish**: ReportLab Platypus output is functional but plain. v0.2 candidate: switch to WeasyPrint (or pandoc) for richer output once we have a styling budget.
- **Bidirectional translation**: v0.1 only stores forward translations (chunk → translation in target lang). If we want round-trip review (translate back to source, diff against original), it's a v0.2 nicety.
