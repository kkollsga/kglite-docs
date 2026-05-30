# Changelog

All notable changes to **kglite-docs** are recorded here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/); versions follow
[SemVer](https://semver.org/), but pre-1.0 minor releases may include
breaking changes (called out below).

## [Unreleased]

## [0.0.9] — 2026-05-30

Release theme: **document structure** — a middle grain between document and
chunk. Sections (from the PDF outline or headings), section-scoped studies,
pinpoint char-span cites on assessments, and opt-in structure-aware chunking.

### Added — document structure
- **Structure-aware chunking (FEAT-10).** `ingest(..., structure_aware=True)` (and
  `document("ingest", structure_aware=True)`) starts a fresh chunk at every
  top-level heading and never packs — or overlaps — across one, so a chunk never
  straddles two sections and its heading attribution is exact. Size-based
  splitting within a section is unchanged (the token target is still honored).
  Opt-in and additive — the default packs greedily exactly as before; pairs with
  Section nodes (cleaner section boundaries) and pinpoint cites.
- **Pinpoint spans on assessments (FEAT-6).** `assess` takes an optional
  `quote` and/or `char_start`/`char_end` — the exact passage an assessment rests
  on — validated against the chunk text (an out-of-range span or a quote not
  found in the chunk is rejected; `quote` alone is located to derive offsets).
  Surfaced per row in `study_ledger` (`quote`/`char_start`/`char_end`), so the
  evidence record carries pinpoint cites instead of just "somewhere in this
  chunk." Backward-compatible: rows without a span read back as `""`/`-1`.
- **`Section` nodes + section-scoped studies (FEAT-9).** Ingest now derives
  `Section` nodes — the grain between document and chunk — from the PDF outline
  (`doc.get_toc()`) when present, else from top-level heading boundaries (generic,
  all formats). `(:Document)-[:HAS_SECTION]->(:Section)-[:HAS_CHUNK]->(:Chunk)`;
  each chunk carries `section_id` (+ a `doc_type` slot verticals can fill).
  `document("sections", doc_id)` / `Corpus.list_sections()` list them with per-
  section `chunk_count`; `study("next"/"ledger", section_id=…)` scope the
  work-list and evidence ledger to one section. `IngestResult.section_count` and
  `get_chunk().section_id` make the structure observable. Backward-compatible —
  re-ingest documents from before this release to populate sections.

## [0.0.8] — 2026-05-30

Release theme: **evidence integrity** — make the assess/verify model legally
defensible: record *what was checked* not just how strongly, let agents park
blocked evidence, resolve cross-agent corrections to one current truth, and
surface the disagreement.

### Added — evidence integrity
- **`study("conflicts", study_id)` surfaces contested evidence (FEAT-8).** Returns
  the chunks that have *both* a current `supports` and a current `against`
  assessment, each with its opposing rows split by side (most contested first) —
  the disagreement an orchestrator should review before concluding. Computed over
  the current set (latest-per-agent, non-superseded), so a correction that
  resolves a disagreement removes it from the list.
- **`study("supersede", …)` + current-by-default ledger (FEAT-5, BUG-4).**
  Audit-preserving correction: `supersede_assessment(old_id, …)` records a
  replacement assessment linked to the one it corrects by an explicit
  `(:Assessment)-[:SUPERSEDES]->(:Assessment)` edge. `study_ledger` and its
  tallies are now **current-by-default** — a superseded assessment is hidden
  (each row carries a `superseded` flag), with `include_superseded=True` for the
  full history. This resolves the cross-agent ambiguity where a pre-filter's row
  and an analyst's correction both showed with no winner, **without** weakening
  multi-agent coexistence (only *explicitly* superseded rows drop out). The old
  assessment is never deleted — the correction trail stays legible.
- **Provenance axis on assessments (FEAT-4).** Each `assess` now records
  `provenance` — *what was actually checked* (the basis), distinct from `weight`
  (the strength): `primary_text` (read the source — default),
  `characterization` (a paraphrase/summary, not the source), or `scanned_unread`
  (a scan no one actually read — provisional). It's a secondary label on the
  Assessment, surfaced per row in `study_ledger`, so a conclusion resting on
  characterizations or unread scans is no longer indistinguishable from one
  resting on primary text. `verify` takes an optional `provenance` recording what
  the *verifier* checked (stored on the verification event). Backward-compatible:
  assessments written before this release read back as `primary_text`.
- **`deferred` stance for evidence assessments (FEAT-7).** Alongside
  `supports`/`against`/`neutral`, an agent can now assess a chunk as `deferred` —
  "read but can't judge yet" (an image-only / `needs_ocr` chunk, or a claim
  awaiting a source not yet ingested). Unlike `neutral` ("read, irrelevant"),
  `deferred` is counted distinctly in `study_ledger` tallies
  (`deferred`/`deferred_weight`) and keeps the chunk in the `study("next")`
  work-list so it resurfaces for a later pass — blocked evidence is parked, never
  silently dropped. Filter the ledger with `stance="deferred"` to list what's
  still blocked.

## [0.0.7] — 2026-05-30

Release theme: **honest coverage** — every coverage-reducing decision (image-only
pages, unembedded chunks, an unindexed corpus, a truncated ledger) is now
observable in the return value instead of silently assumed.

### Added — honest coverage
- **`document("coverage")` / `Corpus.coverage_report()` (FEAT-1):** per-document
  + corpus extraction & embedding coverage — `image_pages`, `low_text_pages`,
  `extractable_text_ratio`, `pending_ocr`, `unembedded`/`embedded`, and a
  human-readable `summary` ("N image-only & L low-text pages need OCR; U chunks
  unembedded — search blind until index()"). Coverage is reported as data, never
  silently assumed.
- **`document("status")` / `Corpus.status()` (FEAT-2):** one-call corpus snapshot
  (docs, pages, chunks, embedded/unembedded, image_pages, pending_ocr, studies)
  — the first thing to check. Pages now persist `extractable_alnum` at ingest;
  pages ingested before this release count as low-text until re-ingested.
- **`compose_context()` now reports `searched_fraction` (FEAT-3):** the share of
  the corpus actually searchable (embedded chunks ÷ ready chunks). `1.0` = full
  coverage; `< 1.0` means part of the corpus was invisible to the query.
- **`study_ledger()` reports `total` + `returned` and accepts `doc_id` scoping
  (BUG-3).** The ledger caps at `limit` (default 200) but used to give no hint
  when it clipped the evidence — an orchestrator could draw a conclusion from a
  silently-truncated record. It now returns `total` (matches before the limit)
  and `returned` (rows handed back); `total > returned` means raise `limit` to
  see the rest. `doc_id=` scopes the whole ledger (rows + counts) to one
  document, mirroring `study("next", doc_id=…)`.

### Fixed
- **Retrieval over an unindexed corpus is now a loud signal, not a silent `[]`
  (BUG-2).** `search()` / `compose_context()` used to return `[]` when *nothing*
  was embedded — indistinguishable from "this query genuinely has no matches,"
  so an agent would wrongly conclude no evidence exists (amplified by 0.0.6's
  `embed=False` default). Now: a corpus with ready chunks but **0 embedded**
  raises `NotIndexedError` (call `index()` / `ingest(embed=True)` first); a
  **partially** indexed corpus emits a `UserWarning` and exposes the gap via
  `searched_fraction`; a genuinely empty corpus still returns `[]`.
  **Behaviour change** for library callers: searching a ready-but-unembedded
  corpus now raises instead of returning `[]`. The MCP happy path
  (ingest → index → search) is unaffected.
- **`cluster_chunks(algorithm="louvain")` no longer pollutes the graph with
  phantom `Chunk` stubs.** Bare `CALL louvain()` clusters the *structural*
  graph (Pages + the Document, not just chunks), so the resulting `IN_CLUSTER`
  edges vivified non-chunk endpoints as stub `Chunk` nodes. Louvain now runs
  only when a `SIMILAR_TO` chunk-similarity graph exists (its documented
  precondition) and filters results to chunk nodes; otherwise it falls back to
  embedding k-means — which is what the corpus does today, so the behaviour is
  the same minus the stray stubs.
- Test runs are now warning-clean: the `IN_CLUSTER` stub warning is fixed at the
  source, and pymupdf's third-party SWIG `DeprecationWarning`s are filtered
  (tightly scoped in `pyproject.toml` so our own warnings stay loud).
- **Image-only PDF pages are now detected for OCR by text density, not "has any
  text at all" (BUG-1).** pymupdf4llm emits a `==> picture … intentionally
  omitted <==` placeholder for image regions, which previously made a scanned
  page count as `ready` and hid it from `ocr_status()`. A page with image
  content but fewer than `OCR_TEXT_THRESHOLD` (~120) real alphanumeric chars is
  now flagged `needs_ocr`. Pages also carry an `image_block_count` (for the
  upcoming coverage report). Honest-coverage: unreadable pages are observable.

### Documentation
- **Confidentiality posture (FEAT-13):** new
  [Confidentiality](https://kglite-docs.readthedocs.io/en/latest/privacy/) page
  (+ README section) stating plainly that all parsing, embedding, and
  analysis run locally against a local `.kgl` — the only network call is a
  one-time bge-m3 weight download from HuggingFace; no document content is ever
  transmitted. Covers the benign "unauthenticated requests to HF Hub" message,
  automatic `HF_HUB_OFFLINE` once cached, and a fully air-gapped setup recipe.

## [0.0.6] — 2026-05-29

### Added — evidence-study workflow (new `study` MCP noun)

- **First-class evidence analysis.** A validated multi-agent trial showed
  the prior workaround (stance smuggled into a tag *name*, weight overloaded
  onto tag *confidence*, rationale in a disconnected `Summary`, a 3-query
  stitch to rank) was the wrong shape. The new `study` noun owns the flow:
  `define` a question → `assess` each chunk (`supports`/`against`/`neutral`
  + 0–1 probative `weight` + `rationale`) → `ledger` (weight-ranked evidence
  + support/against tallies; `stance=` filters to one side) → `verify`
  (second agent: `verified`/`disputed`/`duplicate`) → `conclude` (a verifiable
  summary on the study) → `list`/`get`/`reopen`/`delete`. `next` is a
  resumable work-list of un-assessed chunks.
- **Reified `Assessment` nodes** (`(Chunk)-[:ASSESSED_AS]->(Assessment)-[:OF_STUDY]->(Study)`,
  `(Agent)-[:AUTHORED]->`): multiple agents can co-assess the same chunk; each
  assessment is independently verifiable. Append-only / latest-wins (no
  mutated-property hazard); stance/status/verification tracked as secondary
  labels for index-speed filtering.
- **No embeddings required.** `assess` never touches the model (rationale is a
  plain property) — a whole study runs on an un-`index`ed corpus; agents
  iterate chunks via `next`. Recording is ~10ms/chunk.
- **Punchcard claiming on `next`.** `study("next", study_id, agent_id=…)`
  atomically *checks out* the returned chunks (a `Checkout` node) and excludes
  chunks already claimed by others, so a fan-out of analysts gets disjoint
  batches and never overlaps. Claims auto-expire after 30 min and assessing
  releases implicitly; omitting `agent_id` is a read-only preview. (A blind
  two-agent trial had both analysts grab the same chunks; this closes that.)
- **Read-context + recorded spans for incoherent chunks.** `chunk("get",
  window=N)` returns the N chunks before/after in reading order *with text*
  (via the `NEXT_CHUNK` spine), so an agent can interpret a chunk that doesn't
  stand on its own. When the neighbours were actually needed, `study("assess",
  context_chunk_ids=[…])` records them as `USED_CONTEXT` edges: the `ledger`
  row surfaces the span (retrieval pulls the full relevant neighbours), and
  those context chunks are **excluded from the work-list** so no agent
  re-judges the same meaning.
- MCP surface: 12 → **13 noun tools** (revisit merges later). Server
  `_INSTRUCTIONS` gains an evidence-study happy-path; the `study` tool
  docstring is a copy-pasteable loop so agents need no prompt-side convention.

### Fixed — tag ranking exposes confidence

- `tag("list")` now returns each application's `confidence` (it was dropped,
  so the typed surface couldn't rank by it); `tag("chunks")` ranks by
  strongest confidence then tagger count. `add_summary(embed=False)` makes
  summary embedding opt-out (study conclusions default to no-embed).

### Changed — embedding is now optional and explicit

- **Ingest no longer embeds by default.** `document("ingest")` (and
  `Corpus.ingest`) now parse, chunk, and store *without* touching the
  embedding model — fast, no model load, no per-call-timeout risk. Real
  PDFs that took 48–90s to ingest now return in ~6s. Ready chunks are
  tracked as unembedded via the `c.embedded` property.
- **New `document("index")` action** (`Corpus.index`) computes embeddings
  for ready-but-unembedded chunks, in length-sorted batches to cut bge-m3
  padding waste. Run it after ingest to enable `search`; skip it entirely
  for non-semantic workflows (browse / cypher / tag / review / ocr /
  export / translate need no embeddings). One-shot still available via
  `document("ingest", ..., embed=True)`.
- **`index` is bounded per call and loop-friendly.** A blind-agent test
  showed that while ingest was now fast, indexing a whole multi-document
  corpus in a single call still ran ~100s (and a 144-chunk corpus ran long
  enough that the agent abandoned it) — the per-call-timeout risk had just
  moved from ingest to index. `index` now does at most ~30s of work
  (wall-clock budget, or `max_chunks`) per call, commits the partial
  progress, and returns `pending > 0` with a `hint` to call again. The
  agent loops `while pending: index()`; no single call exceeds the budget
  (+ at most one batch). `max_seconds=None` drains everything in one pass
  (used by CLI `--ingest` preload, where there's no timeout).
- Ingest results now carry `embedded` count + a `hint` pointing at `index`
  when chunks are unembedded.

### Fixed

- **Tool-driven ingest now persists to disk.** The long-lived MCP server
  previously only saved when launched with `--ingest`; interactive
  `document("ingest")` mutations were lost on restart. Now saved at the
  tool boundary (ingest/index) plus a save-on-shutdown backstop for all
  mutations.
- Embedding lifecycle is tracked by the `c.embedded` property (with an
  additive `:Embedded` label for browsing) rather than a removable
  `:Unembedded` label. This was originally a workaround for a kglite
  multi-label read bug — `MATCH (n:Label)` over-reported after
  `remove_label` — which we reported and kglite fixed in **0.10.6** (root
  cause was read-side fast-paths consulting only the primary-type index,
  not a stale index as we first guessed; no data corruption, no `.kgl`
  migration). The property-based design is version-independent so we kept
  it; removable labels / `swap_label` now work correctly regardless.

### Dependencies

- Bumped `kglite>=0.10.9` (multi-label read fix from 0.10.6; FxHash query
  perf from 0.10.7: `multi_where −38%`, `group_by −23%`, `where_scan −21%;
  four Cypher-dialect fixes in 0.10.8 we reported building `study` —
  `labels()`/properties on a `collect()[0]` node, parameters inside
  `EXISTS {}`, node-expression inline-map values, and `DETACH DELETE`
  skipping NULL vars; and the 0.10.9 self-healing id-index that makes
  `MATCH (n {id: X})` / MERGE lookups O(1) — a free win for the study
  workflow's many id lookups) and `mcp-methods>=0.3.40` (fixes `graph_overview` forwarding the dropped
  `describe(limit=)` kwarg, and `cypher_query` failing `-> str` validation
  on kglite's `ResultView`). Both escape-hatch tools now work on kglite
  0.10.x. The latent over-reporting in the other `swap_label` lifecycles
  (summary verification, OCR, review kanban, translations) is fixed by the
  0.10.6 upgrade.

### Performance

- Background bge-m3 warm-load at server boot (off-thread), `HF_HUB_OFFLINE`
  when weights are cached, and `cooldown=0` for the server — the model
  load (~8s) and HF-hub network checks no longer land on the first
  ingest/search call. `--no-warmup` skips it for non-embedding
  deployments.

## [0.0.5] — 2026-05-28

### Changed (MCP surface — agent-facing)

- **Collapsed 45 standalone MCP tools into 12 CLI-style noun dispatchers.**
  Each dispatcher takes an action verb as the first positional arg:

  ```
  document(action, ...)   ingest | list | get | export | compare
  chunk(action, ...)      get | similar
  search(query, mode=…)   hits (default) | compose
  summary(action, ...)    add | verify | list | ground | claim | consensus
  tag(action, ...)        add | remove | list | chunks
  agent(action, ...)      upsert | get | list | activity
  review(action, ...)     enqueue | enqueue_chunks | claim_next | claim |
                          unclaim | complete | list | get | stats
  ocr(action, ...)        status | pending | submit
  cluster(action, ...)    run | get | list | export
  translate(action, ...)  add | list | assemble
  ```

  Plus `cypher_query` and `graph_overview` from mcp-methods. An agent
  reads the methodology skill and copies verb-noun pairs straight into
  tool calls (`document("ingest", path=...)`, `summary("verify", id=...,
  verdict=...)`) — patterned like `git branch list`, not 45 standalone
  functions to scan.

  The Python `Corpus` API is **unchanged** — the dispatch only lives in
  the MCP shim.

### Added (agent onboarding)

- **`Corpus.compare_documents(doc_a, doc_b, queries=[...])`** —
  side-by-side cross-document retrieval. For each query, returns the
  top hits from each document plus a budgeted merged context bundle.
  Exposed as `document("compare", ...)` over MCP.
- **First-contact `instructions=`** on the FastMCP server.
  Connecting agents receive a ~250-word orientation on `initialize`
  covering the CLI surface and the canonical happy path.
- **4 methodology skill files** registered as MCP prompts via
  mcp-methods' `SkillRegistry`:
  - `/00-start-here` — overview + the canonical happy path
  - `/analyze-documents` — canonical pipeline for "given N docs + a task"
  - `/compare-documents` — the two-doc comparison idiom
  - `/cross-checked-review` — write/verify/ground flow for trustworthy
    summaries

  Skills are loaded by the agent on demand (`/<skill-name>`) — not
  preloaded into context.

### Removed (redundant — covered elsewhere)

- `register_agent` → `agent("upsert", ...)`
- `link_verification` → niche; use `cypher_query`
- `record_view` → implicit when passing `agent_id` to `search` / `chunk("get", ...)`
- `untag_chunk` → `tag("remove", ...)`
- `ingest_dir`, `ingest_text` → `document("ingest", directory=...)` /
  `document("ingest", text=..., title=...)`

## [0.0.4] — 2026-05-28

### Changed (internal data model — public API unchanged)

- **Adopted kglite 0.10.5 multi-label nodes.** Categorical / lifecycle
  state — `Agent.role`, `Agent.kind`, `Chunk.status`,
  `Summary.verification_status`, `Tag.kind`, `Translation.status`,
  and `ReviewTicket` lifecycle — now lives as secondary labels on the
  relevant node (`(a:Agent:Reviewer:LLM)`, `(c:Chunk:Ready)`,
  `(s:Summary:Verified)`, etc.). Property scans are replaced with
  `MATCH (n:Label)` predicates, which kglite indexes natively. Cross-
  type predicates like `MATCH (n:Reviewed)` return any reviewed thing
  in the graph regardless of primary type — useful for governance
  queries.

  The `Corpus` Python / CLI / MCP surface is **unchanged**: callers
  still pass `status="verified"` / `role="reviewer"` / `verdict="reviewed"`
  strings. A new `schema.label_for(discriminator, value)` helper maps
  user-facing strings → canonical PascalCase label names at the
  boundary. Free-text discriminators (notably `Agent.role`) get
  slug → PascalCase conversion: `"fact-checker"` → `:FactChecker`.

  Labels are maintained atomically via a new `Store.swap_label`
  primitive: every state transition (`verify_summary`, `submit_ocr`,
  `claim_review`, `complete_review`, `mark_translation_reviewed`)
  removes the old label and adds the new one in one call.

  Event-sourced models (`VerificationEvent`, `ReviewEvent`) still
  carry the immutable audit trail. The labels denormalise "current
  state" for O(label-index) reads instead of "scan events, pick latest".

  + 7 new tests covering label round-trip, swap mutual-exclusion,
  save/reload survival, and cross-type label predicates.

- **`kglite>=0.10.5`** required (was `>=0.10.4`).

### Breaking — raw Cypher only

If you wrote raw Cypher against `corpus.cypher(...)` and filtered on
`Summary.verification_status`, `Chunk.status`, `Agent.role`, `Agent.kind`,
`Tag.kind`, or `Translation.status`, switch from
`WHERE n.prop = '...'` to `MATCH (n:NodeType:Label)`. Example:

```cypher
-- before
MATCH (s:Summary) WHERE s.verification_status = 'verified' RETURN s
-- after
MATCH (s:Summary:Verified) RETURN s
```

Label names use PascalCase: `Verified`, `Disputed`, `NeedsRevision`,
`Stale`, `Unverified`, `Ready`, `NeedsOcr`, `Empty`, `LLM`, `Human`,
`Service`, `Reviewer`, `Writer`, `FactChecker`, etc. The Cypher
escape hatch tool in the MCP server still works the same way — the
agent just writes label predicates now.

`.kgl` files created by v0.0.x still load on v0.0.4 — labels won't
exist on those nodes (so label-filter queries against old corpora
return empty); re-ingest to populate labels.

## [0.0.3] — 2026-05-28

### Added

- **Agent nodes carry a reusable template.** The `Agent` node was
  previously identity + counters; it now also holds `role`,
  `system_prompt`, `model`, `tools` (list), `context` (free-form JSON
  dict), and `description`. The graph IS the registry — orchestrators
  fetch an agent's loading context with `get_agent(agent_id)`, use
  the fields to launch the actual LLM call, and every subsequent
  graph write under the same `agent_id` attributes back to the
  template.

  New methods on `Corpus`:

  - `upsert_agent(agent_id, *, kind, model, role, system_prompt,
    tools, context, description)` — field-level merge.
  - `get_agent(agent_id)` — returns full template + counters; `tools`
    and `context` come back hydrated (real list / dict, not JSON
    strings).
  - `list_agents(role=..., kind=...)` — discovery, with filters.
  - `agent_activity(agent_id, *, target_id=None)` — bucketed
    rollup of everything an agent has done (views, summaries, tags,
    translations, review/verification events). Pass `target_id` to
    scope to one chunk.

  Old `register_agent` (the lazy on-first-use path) is preserved
  and now explicitly does *not* clobber template fields when
  the agent already exists. `add_summary` / `tag_chunk` / etc.
  still lazy-register if you haven't upserted first.

  New types: `AgentConfig`, `AgentActivity`.

  New MCP tools: `upsert_agent`, `get_agent`, `agent_activity`
  (existing `register_agent` / `list_agents` keep working;
  `list_agents` gained `role=` / `kind=` filters).

  9 new tests covering create → update merge → preserves-on-lazy →
  scoped activity rollups → realistic orchestration round-trip.

  Example::

      corpus.upsert_agent(
          "reviewer-strict",
          role="reviewer", model="claude-sonnet-4-6",
          system_prompt="You are a strict fact-checker...",
          tools=["check_grounding", "verify_claim"],
          context={"strictness": "high", "min_citations": 2},
      )
      # Later — orchestrator side
      cfg = corpus.get_agent("reviewer-strict")
      anthropic.messages.create(
          model=cfg["model"], system=cfg["system_prompt"], ...
      )

## [0.0.2] — 2026-05-28

### Changed

- **`mcp` + `mcp-methods` moved into core dependencies.** They were
  previously gated behind the `[mcp]` extra, but `mcp` was already
  pulled in transitively by `kglite` anyway, and `mcp-methods` is only
  ~17 MB — rounding error against `pymupdf` (30 MB) and `onnxruntime`
  (80 MB) which are core. The MCP server is the headline feature;
  gating it behind an extra was friction with no real savings.

  Install is now just `pip install kglite-docs`. Old install commands
  with `[mcp]` keep working (pip warns, doesn't fail).

### CI / tooling

- Bumped all `actions/*` to versions supporting Node.js 24 ahead of
  GitHub's 2026-06-02 deprecation of Node 20.

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
- Requires `mcp-methods>=0.3` (was previously in the `[mcp]` extra; moved to core deps in 0.0.2).
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

[Unreleased]: https://github.com/kkollsga/kglite-docs/compare/v0.0.4...HEAD
[0.0.4]: https://github.com/kkollsga/kglite-docs/releases/tag/v0.0.4
[0.0.3]: https://github.com/kkollsga/kglite-docs/releases/tag/v0.0.3
[0.0.2]: https://github.com/kkollsga/kglite-docs/releases/tag/v0.0.2
[0.0.1]: https://github.com/kkollsga/kglite-docs/releases/tag/v0.0.1
