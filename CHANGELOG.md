# Changelog

All notable changes to **kglite-docs** are recorded here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/); versions follow
[SemVer](https://semver.org/), but pre-1.0 minor releases may include
breaking changes (called out below).

## [Unreleased]

## [0.0.1] — 2026-05-28

First public alpha. The core agent-first PDF knowledge-base API is in
place: ingest → chunk → embed → search → enrich → cluster → review,
served over Python, CLI, and MCP. Everything is exercised by 81 unit +
integration tests; a real end-to-end Sonnet workflow demo is included.

### Added

- **Multi-format ingest.** PDF, DOCX, PPTX, MD, HTML, TXT, and common
  image formats (PNG/JPG/TIFF/WebP/BMP), all flowing into the same
  Document → Page → Chunk graph.
- **Token-aware paragraph chunking** at ~512 bge-m3 tokens with 64-token
  overlap; never crosses a page boundary.
- **BAAI/bge-m3 embeddings** via ONNX (CLS-pooled, 1024-dim, 8192-token
  cap), inherits kglite's cool-down lifecycle for warm-call latency.
- **Semantic search + `compose_context(query, max_tokens=…)`** —
  budgeted, ranked, prompt-ready bundles for agents.
- **Enrichment with cross-checking.** `add_summary` / `verify_summary`
  enforce that the verifier is a different agent. Status is event-sourced
  via `VerificationEvent` so we keep a full audit trail.
- **Tagging via reified `Tagging` nodes** so multiple agents can tag the
  same chunk distinctly with `(by_agent, created_at, confidence)`.
- **Agent identity propagation.** Lazy-registered `Agent` nodes;
  `search` / `get_chunk` accept `agent_id=` to bump `Chunk.view_count`
  and record `View` nodes when the query context is worth keeping.
- **Scanned-page OCR loop.** `list_pending_ocr` → agent reads PNG →
  `submit_ocr(page_id, markdown)` re-chunks + re-embeds.
- **`ocr_status()`** — coverage summary across the corpus, per-doc detail
  with a pending-fraction. Exposed as Python / CLI / MCP.
- **`kglite-docs ocr-do`** — CLI subcommand that drives the OCR loop
  with any agent command containing `{image}` (e.g. `claude -p --image …`).
- **Clustering.** `cluster_chunks(algorithm='louvain'|'kmeans'|'dbscan')`
  with `Cluster` + `IN_CLUSTER` graph state; `most_connected_cluster`
  for synthesis use cases.
- **Quality / anti-hallucination.** `check_grounding(summary_id)`
  scores per-sentence support; `verify_claim(claim_text, …)` finds
  best-supporting chunks.
- **Translation layer.** `Translation` nodes per chunk × language;
  `assemble_translated_document` stitches back with fallback to source.
- **Export.** `export_document` / `export_cluster` / `export_summary` /
  `export_bundle` to Markdown, DOCX (`python-docx`), or PDF (ReportLab).
- **Review queue (kanban).** Event-sourced `ReviewTicket` →
  `ReviewEvent` audit trail. `claim_next_review` / `complete_review`
  with verdict + accuracy + authenticity + tags. Process-local lock
  protects against double-claim within a process.
- **MCP server.** `kglite-docs-mcp --db kb.kgl` exposes 30+ typed tools
  plus `cypher_query` / `graph_overview` escape hatches via mcp-methods.
- **CLI.** `kglite-docs ingest|search|list|cluster|show|ocr-status|ocr-do`.
- **Context manager on `Corpus`.** `with Corpus.open(path) as c:`
  auto-saves on clean exit, skips save on exception so a partial
  mutation isn't persisted.
- **Typed errors.** `KgliteDocsError` base + concrete subclasses
  (`IngestError`, `UnsupportedFormatError`, `MissingSourceError`,
  `ReviewConflict`, `SelfVerificationError`, `InvalidEnumError`,
  `GroundingError`, `ConcurrencyError`).
- **Typed args + returns.** `Literal[…]` for verdicts, statuses,
  depths, target kinds, algorithms, tag kinds; `TypedDict`
  return shapes (`SearchHit`, `OcrStatus`, `ReviewTicketDetail`, …)
  for IDE autocomplete and `mypy --strict`.
- **End-to-end Sonnet workflow demo** (`demos/workflow.py`): ingest
  → cluster → Sonnet summarises → Sonnet drafts an article with
  `[chunk_id]` back-references → second Sonnet pass fact-checks →
  verifications persist as `VerificationEvent` nodes.
- **Docs.** Getting-started, architecture, workflows, performance,
  publishing, troubleshooting, contributing — all in `docs/`.
- **CI + release.** GitHub Actions workflows for `ruff` + `mypy` +
  `pytest` on Py 3.10–3.13 × macOS/Linux, plus a trusted-publisher
  PyPI release pipeline triggered on `v*` tags.

### Dependencies

- Requires `kglite>=0.10.4`. (Earlier 0.10.3 hit two upstream bugs we
  filed; both fixed in 0.10.4 — see "Bug workarounds peeled" below.)
- Requires `mcp-methods>=0.3` (optional, via the `[mcp]` extra).
- bge-m3 ONNX weights download to `~/.cache/fastembed/` on first use
  (~2 GB, one-time). Set `HF_HUB_CACHE` to reuse an existing HF cache.

### Bug workarounds peeled

Two kglite bugs we hit and reported during this build, both fixed
upstream in 0.10.4:

- **`add_nodes(Chunk, …)` invalidating the id index used by
  `set_embeddings`.** Pre-0.10.4 this silently wiped all prior chunk
  embeddings on the second-and-later document ingested into a corpus.
  We carried a save+reload workaround in `pipeline.py` (since removed).
- **`mmap_vec` panic on the second String SET on the same node.**
  Forced us into event-sourced verifications and reviews. The kglite
  fix lets us mutate Strings normally now, but we kept the event
  sourcing — the audit trail is a better model on its own merits.

Filed in the kglite inbox at
`KGLite/inbox/read/2026-05-28-from-kglite-docs-multi-file-ingest-loses-old-chunk-embeddings.md`.

### Known limitations

- **Single-writer per `.kgl` file.** kglite is process-local; concurrent
  writers to the same file aren't supported. `Corpus` holds a Python-
  level lock for the review queue's claim path, which covers
  multi-threaded but not multi-process scenarios.
- **No OpenAI/Cohere embedder adapter shipped.** The `EmbeddingModel`
  protocol is documented (`dimension`, `embed`, optional `load`/
  `unload`), but only `BgeM3Embedder` is wired by default. Writing an
  alternative is straightforward.
- **PDF export is "good enough" ReportLab Platypus, not pixel-perfect.**
  Use the MD or DOCX path for richer output, or pipe MD through pandoc.
- **`find_consensus` and `link_verification` are experimental.** They
  work but the shape of the API isn't as well-considered as the rest.
  Subject to change in 0.x.

### Test posture

- 81 unit + integration tests, ~1s run time with the stub embedder.
- `@pytest.mark.embed` tests use the real bge-m3 model (skipped in CI
  to keep things fast; run locally with the model cached).
- End-to-end Sonnet workflow demo proves the agent path beyond unit tests.

[Unreleased]: https://github.com/kkollsga/kglite-docs/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/kkollsga/kglite-docs/releases/tag/v0.0.1
