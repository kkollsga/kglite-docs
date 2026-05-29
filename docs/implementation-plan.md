# kglite-docs phased implementation plan

Turns `ROADMAP.md` (*what* to build) into *how*: an ordered catalog of
**commit-sized phases**, each with concrete file/function anchors (verified
against the tree) and tests. Each phase is **self-contained — pick one up later
as its own plan→execute loop.** Phases group into releases; a release ships only
after all its phases are committed (joint bump+push).

**Execution model** (per `CLAUDE.md` → "Default work procedure", applied per
loop): a phase (or small group) is run in a session; within that loop it goes to
green — `ruff check src/ tests/` + `mypy src/kglite_docs` + relevant tests —
then **commit (no version bump, no push)**. Only when a release's phases are all
committed do we do the **joint boundary step**: release pre-flight, bump
`pyproject.toml` + promote the CHANGELOG + **push together** (user gives the
in-the-moment approval; the push auto-publishes to PyPI). No phase touches the
version or pushes mid-release.

North star throughout: **honest coverage** — every coverage-reducing decision is
observable in the return value.

**Conventions reused everywhere:** feature logic as stateless module functions
(`store` first) → thin `Corpus` delegating method (`cast(T, …)` at the boundary)
→ MCP `tools.py` dispatcher branch (+ `_persist` if mutating, + docstring) →
`types.py` Literal/TypedDict → `schema.py` first for any new node/label. Reified
nodes + bulk `upsert_edges` (never cypher `CREATE`) + label state via
`label_for`/`swap_label`. See `ROADMAP.md` for the BUG-/FEAT- rationale.

---

## Release 0.0.7 — "Honest coverage"  (BUG-1/2/3, FEAT-1/2/3/13)

The trust release: no skipped page, unindexed chunk, or truncated ledger is ever
invisible. Highest ROI, mostly S/M, benefits every use case.

### Phase 7.1 — Fix OCR detection of image-only pages (BUG-1) · M
- `ingest/parser.py` `parse_pdf` (~L33–67): replace `has_text = bool(markdown)` /
  `needs_ocr = (not has_text) and has_images` with a **text-density** decision.
  Add `_extractable_alnum(markdown) -> int` that strips the `==> picture …
  intentionally omitted <==` markers + footer boilerplate and counts alnum
  chars. Set `needs_ocr = has_images and alnum < THRESHOLD (~120)`; keep
  `has_text` truthful. Carry `extractable_alnum` + `image_block_count`
  (`page.get_images(full=False)`) on `PageContent` (new fields, default 0).
- `ingest/formats.py`: keep per-format `needs_ocr` consistent (image → True,
  text formats → alnum-based where cheap; PPTX already does `not md.strip()`).
- `ingest/pipeline.py` page_rows (~L153–173): persist `extractable_alnum` +
  `image_block_count` on the `Page` node (for FEAT-1).
- Tests (`tests/test_ocr_detection.py`): image-only page → `needs_ocr=True`; real
  text page → ready; a lone "picture omitted" fragment does not pass as ready.
- Commit: `fix: detect image-only pages as needs_ocr via text-density (BUG-1)`

### Phase 7.2 — `coverage_report()` + `corpus.status()` (FEAT-1, FEAT-2) · M
- `ocr.py` (or new `coverage.py`): `coverage_report(store, *, doc_id=None)`
  extending the `ocr_status` Page-aggregation cypher: add
  `extractable_text_ratio`, `image_only_pages`, `low_text_pages`, `unembedded`
  (reuse `count_unembedded`), `pending_ocr`, + a one-line human `summary`
  ("X% of pages are image-only and unanalyzed unless OCR'd").
- `corpus.py`: `coverage_report(...)`; **`status()`** →
  `{docs, chunks, embedded, unembedded, image_pages, pending_ocr, studies}`.
  Both `cast(T, …)`.
- `types.py`: `CoverageReport`, `CorpusStatus` TypedDicts; add to `__all__`.
- `mcp_server/tools.py`: `ocr("coverage")` + `document("status")` (prefer an
  action on an existing noun over a new tool — keep ~13).
- `mcp_server/server.py` `_INSTRUCTIONS`: "call status first" hint.
- Tests: mixed corpus → ratios/lists/summary; `status()` keys.
- Commit: `feat: coverage_report() + corpus.status() for honest coverage (FEAT-1/2)`

### Phase 7.3 — Loud retrieval when unindexed (BUG-2, FEAT-3) · S
- `errors.py`: `class NotIndexedError(KgliteDocsError)` (mirror `GroundingError`).
- `corpus.py` `search` (~L482) + `context.py` `compose_context` (~L18): if
  `count_unembedded() > 0` **and** 0 embedded → raise `NotIndexedError`
  (unambiguous misuse). Partially indexed → attach
  `searched_fraction = embedded/ready` + `warnings.warn(...)`.
- `types.py`: `searched_fraction: float` on `ComposedContext` (document the
  `NotIndexedError` for the bare-list `search` path).
- Tests: ingest **without** index → `search`/`compose` raise; partial → fraction
  < 1.0 + warning. Update tests that assumed silent `[]`.
- Commit: `fix: NotIndexedError + searched_fraction instead of silent [] (BUG-2/FEAT-3)`

### Phase 7.4 — Ledger truncation + doc scoping observable (BUG-3) · S
- `study.py` `ledger` (~L394–465): add `total` (count before LIMIT) + `returned`
  (`len(rows)`); add `doc_id=None` scoping (extend `where`); keep `limit` but
  make truncation visible.
- `corpus.py` `study_ledger` + `tools.py` `study` ledger branch: thread `doc_id`.
- `types.py` `Ledger`: `total: int`, `returned: int`.
- Tests (`tests/test_study.py`): >limit → `total>returned`; `doc_id=` scopes.
- Commit: `fix: study_ledger reports total/returned + doc_id scope (BUG-3)`

### Phase 7.5 — Confidentiality docs (FEAT-13) · S
- `docs/privacy.md` (or a `getting-started.md` section) + `README.md`: state
  plainly *all parsing/embedding/assessment is local; the only network call is a
  one-time bge-m3 fetch from HF; no document content is transmitted*; explain the
  benign "unauthenticated requests to HF Hub" message + `HF_HUB_OFFLINE` when
  cached.
- Commit: `docs: confidentiality posture — everything runs local (FEAT-13)`

**→ Release boundary: pre-flight, bump `0.0.6 → 0.0.7`, promote CHANGELOG, push (joint).**

---

## Release 0.0.8 — "Evidence integrity"  (FEAT-4/5/7/8, BUG-4)

Make the assess/verify model legally defensible and generally trustworthy.

### Phase 8.1 — `deferred` stance (FEAT-7) · S
- `schema.py`: `STANCE_DEFERRED="deferred"` + `LABEL_DEFERRED`, extend
  `VALID_STANCES` + `_STUDY_STANCE_LABELS`.
- `types.py`: extend `Stance` literal with `"deferred"`.
- `study.py` `assess`: accept `deferred`; `_tallies` counts it separately (not
  folded into neutral); `next_unassessed` treats deferred as parked, not done.
- Tests: assess `deferred` → ledger/tallies show it distinctly (pairs with BUG-1:
  image/`:NeedsOcr` chunks are naturally `deferred`).
- Commit: `feat: deferred stance for blocked/needs-evidence chunks (FEAT-7)`

### Phase 8.2 — Provenance axis (FEAT-4) · M
- `schema.py`: `assessment.provenance` discriminator —
  `primary_text|characterization|scanned_unread` + labels.
- `study.py` `assess` (node dict ~L280) + `verify_assessment` (event ~L356):
  record `provenance` (what was actually checked: primary source vs paraphrase vs
  unread scan) + the provenance label; `ledger` surfaces it from labels (same
  pattern as status).
- `types.py`: `Provenance` literal; `provenance` on `AssessmentRow`.
- `corpus.py` + `tools.py`: thread `provenance` through `assess`/`verify` + docstring.
- Tests: each provenance value round-trips into the ledger.
- Commit: `feat: provenance axis (primary/characterization/scanned_unread) (FEAT-4)`

### Phase 8.3 — Supersede + current-ledger truth (FEAT-5, BUG-4) · M
- `schema.py`: `SUPERSEDES` edge constant.
- `study.py`: `supersede_assessment(store, *, old_id, …new assess args…)` → new
  Assessment + `(:Assessment)-[:SUPERSEDES]->(:Assessment)`; `ledger` hides
  superseded by default, `include_superseded=True` shows history. Resolves the
  cross-agent ambiguity (BUG-4).
- `corpus.py` `supersede_assessment`; `tools.py` `study("supersede", …)`.
- `types.py`: `superseded: bool` on `AssessmentRow` (when included).
- Tests: supersede → ledger shows new only; `include_superseded` shows both.
- Commit: `feat: supersede_assessment + current-by-default ledger (FEAT-5/BUG-4)`

### Phase 8.4 — Conflict surfacing (FEAT-8) · M
- `study.py`: `conflicts(store, *, study_id)` → objects (chunk/section) with both
  `supports` and `against` current assessments + the opposing rows.
- `corpus.py` `study_conflicts`; `tools.py` `study("conflicts", id)`.
- `types.py`: `ConflictRow` / result TypedDict.
- Tests: opposing assessments on one chunk → surfaced; agreement → not.
- Commit: `feat: study('conflicts') surfaces contested evidence (FEAT-8)`

**→ Release boundary: bump `0.0.7 → 0.0.8`, push (joint).**

---

## Release 0.0.9 — "Document structure"  (FEAT-9/6/10)

Biggest quality lever; unblocks section-scoped studies + pinpoint cites.

### Phase 9.1 — `Section` nodes + `chunk.doc_type`/`section_id` (FEAT-9) · L
- `ingest/parser.py`: capture the PDF outline via `doc.get_toc()` in `parse_pdf`;
  carry it out (sidecar `outline` of `(level, title, page)`).
- `schema.py`: `SECTION="Section"` node; `HAS_SECTION` edge.
- `ingest/pipeline.py`: build `Section` nodes from (a) PDF outline, else (b)
  top-level `headings_json` boundaries; `(:Document)-[:HAS_SECTION]->(:Section
  {doc_type, title, page_start, page_end})`, `(:Section)-[:HAS_CHUNK]->(:Chunk)`,
  set `chunk.section_id` + `chunk.doc_type`. Best-effort + **generic** (domain
  adapters like PROJUDI stay OUT of core → legal vertical).
- `corpus.py`: `list_sections(doc_id)`; thread `section_id` into
  `next_unassessed` / `study_ledger`.
- `tools.py`: `document("sections")`; `study("next"/"ledger", section_id=…)`.
- `types.py`: `SectionRow`; `section_id`/`doc_type` on `ChunkDetail`.
- Tests: PDF w/ outline → sections + linkage; section-scoped `next`/`ledger`.
- Commit: `feat: Section/SourceDoc nodes + section-scoped studies (FEAT-9)`

### Phase 9.2 — Pinpoint spans on assessments (FEAT-6) · M
- `study.py` `assess`: optional `quote`/`char_start`/`char_end`, stored on the
  `Assessment` (and/or `USED_CONTEXT` edges via `upsert_edges(..., properties=)`).
- `ledger`: surface span/quote per row for pinpoint cites.
- `types.py`: span fields on `AssessmentRow`.
- Tests: assess with span → ledger emits it; offsets validated vs chunk length.
- Commit: `feat: pinpoint char-span/quote on assessments (FEAT-6)`

### Phase 9.3 — Structure-aware chunking (FEAT-10) · M
- `ingest/chunker.py`: opt-in flag to start a new chunk at heading boundaries
  (don't pack across a top-level heading); default preserves current behavior;
  thread from `ingest`.
- `ingest/pipeline.py`: pass the flag.
- Tests: doc w/ headings → chunks respect boundaries; token target honored.
- Commit: `feat: structure-aware chunking on heading boundaries (FEAT-10)`

**→ Release boundary: bump `0.0.8 → 0.0.9`, push (joint).**

---

## Release 0.0.10 — "Scale & polish"  (FEAT-11/12/14 + surface review)

### Phase 10.1 — Concurrency: docs + `assess_many` + file-lock (FEAT-12) · M
- Docs: a plain "single-writer; fan out read-only, funnel writes through one
  process / the MCP server" section.
- `study.py`/`corpus.py`: `assess_many(study_id, rows)` — batched write + single
  `_persist`.
- `store.py`: advisory file-lock (`<db>.lock`) on mutate → a second writer raises
  `ConcurrencyError` loudly instead of corrupting.
- `tools.py`: `study("assess_many", …)`.
- Tests: batch assess; second-writer lock raises.
- Commit: `feat: single-writer guardrails — assess_many + advisory lock (FEAT-12)`

### Phase 10.2 — Summary-augmented chunking (FEAT-11, opt-in) · M
- `ingest/`: optional doc/section summary prepended to each chunk before
  embedding (LLM pass; opt-in flag). Improves global context.
- Tests: flag on → embedded form carries the summary prefix.
- Commit: `feat: optional summary-augmented chunking (FEAT-11)`

### Phase 10.3 — Ergonomics + surface review (FEAT-14) · S
- `store.py`/docs: document `ResultView`; make `for row in corpus.cypher(...)` +
  `row["col"]` work (iter/getitem); `get_chunk` → typed `ChunkDetail` with
  attr+`__getitem__`.
- Review whether the MCP surface can fold back toward 12 (relocate `summary`'s
  ephemeral `claim`/`consensus`, now superseded by `study`); decide + maybe deprecate.
- Tests: cypher iteration; get_chunk access patterns.
- Commit: `feat: ResultView/get_chunk ergonomics + surface review (FEAT-14)`

**→ Release boundary: bump `0.0.9 → 0.0.10`, push (joint).**

---

## Tracking (no code phases)

- **kglite offloads (ROADMAP Part C):** KG-1 (`label` property shadowed) + KG-2
  (`CONTAINS` reserved) reported to `../KGLite/inbox`. **Adopt via a kglite pin
  bump when fixed** — fold into whichever release is in flight; no workaround
  here. (FEAT-9 / 8.4 already follow both workarounds.)
- **Legal vertical (ROADMAP Part E):** charges / speaker ontology / citation
  resolver / PROJUDI adapters live in a separate `kglite-docs-legal` package on
  these primitives. Out of scope; revisit after 0.0.9.

## Verification (per release, before the joint bump+push)

1. `./.venv/bin/python -m ruff check src/ tests/` → clean.
2. `./.venv/bin/python -m mypy src/kglite_docs` → 0 errors (hold the line).
3. `./.venv/bin/python -m pytest` (incl. `-m "mcp or embed"`, which CI skips) → green.
4. End-to-end via `scripts/mcp_session.py` for the release's headline path
   (e.g. 0.0.7: ingest an image-heavy PDF → `coverage` shows image pages;
   `search` before `index` raises `NotIndexedError`).
5. Dry `pip wheel . --no-deps` includes `manifest.yaml` + `manifest.skills/`.
6. Promote CHANGELOG `[Unreleased]` → the new version header at the boundary.

## Scope / ordering notes
- Phases within a release are independently committable, ordered by dependency
  (8.1 deferred before 8.2 provenance; 9.1 sections before 9.2 spans /
  section-scope). 0.0.7 lands first.
- If a phase uncovers a kglite engine gap: **stop, repro on the current pin,
  offload to `../KGLite/inbox`, continue without a workaround** (per `CLAUDE.md`).
