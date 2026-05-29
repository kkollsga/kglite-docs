# kglite-docs roadmap

Status of and plan for the agent-first document knowledge base. This roadmap
was synthesized from two field reports against a **live 1,564-page legal
matter** (a real Brazilian litigation, 0.0.6), a multi-agent "draft the brief
from the graph alone" study, a 103-agent legal-KG/e-discovery literature pass,
and our own multi-agent trials.

## Vision & guiding principles

kglite-docs is the **agentic document-review spine**: ingest documents, let
agents weigh per-chunk evidence for/against a question, verify across agents,
and produce an audit-trailed conclusion. **Legal is a flagship use case** (and
the most demanding one — see Stanford's finding that commercial legal-RAG
tools hallucinate [17–33% of queries][stanford]), but every primitive here is
designed to generalize to forensics, medical record review, due-diligence,
compliance, research synthesis, and investigative journalism.

Four principles, in priority order:

1. **Honest coverage is the product.** *Every coverage-reducing decision must
   be observable in the return value.* Silent truncation/skipping is fine in a
   code-search tool; in an evidence tool it is a liability — a user who trusts
   a green light while half the record is invisible can make a wrong, confident
   decision. This is the #1 theme and the gap between "good demo" and
   "trustworthy for professional work."
2. **Generalized primitives, not domain concepts.** The core grows *primitives*
   — sections, spans, a provenance axis, conflict-surfacing — that verticals
   (legal, medical, …) compose. We do **not** bake `Charge`/`Sentença`/
   `Deposition` into the core. Keep the MCP surface ~13 thin noun-tools.
3. **Provenance & audit are first-class.** `by_agent` / `model` / `created_at`
   on every assessment, cross-agent verification, and *what was actually
   checked* (primary source vs paraphrase) — this is chain-of-custody.
4. **Single-writer truth.** One writer owns the `.kgl`; parallelism is "many
   `agent_id`s through one process," not many OS processes. Make that explicit
   and make a second writer fail loudly, never corrupt.

## Status legend

`PLANNED` · `IN PROGRESS` · `DONE` · `OFFLOADED` (waiting on a kglite fix)

Effort: `S` (≤½ day) · `M` (1–2 days) · `L` (3+ days)

---

# Part A — Bugs (kglite-docs)

Defects in our code. Each is verified with a root cause and repro.

### BUG-1 · OCR detection misses image-only pages — `CRITICAL` · `M` · PLANNED
**Symptom.** `ocr_status()` reported `pending_pages: 0` for a suit where ~48%
of chunks (583 of 1,211) are scanned-image exhibits (the *primary* evidence:
notarized WhatsApp ATA, depositions, emails). The tool signalled full coverage
while half the record was unread.
**Root cause (confirmed).** `ingest/parser.py:57`
`needs_ocr = (not has_text) and has_images`, with `has_text = bool(markdown)`
(`parser.py:49`). pymupdf4llm emits a tiny `==> picture intentionally omitted
<==` fragment for image pages, so `markdown` is non-empty → `has_text = True`
→ `needs_ocr = False` → the page is stored `:Ready`.
**Fix.** Replace the boolean with a **text-density heuristic**: strip
boilerplate/footers and the `picture … omitted` markers, then if extractable
alphanumeric chars `< threshold` (≈120) OR the page is dominated by image
blocks, mark `:NeedsOcr` (or the new `:LowText` state). The "picture
intentionally omitted" marker is itself a strong positive signal. Store the
per-page `extractable_alnum` count for the coverage features (FEAT-1).
**Tests.** Ingest a known image-only PDF → assert `needs_ocr=True`; a
text+image mixed page → still ready; regression on the sample corpus.

### BUG-2 · `search` / `compose_context` return `[]` silently when nothing is embedded — `CRITICAL` · `S` · PLANNED
**Symptom.** With unembedded chunks, `search("…")` → `[]` and
`compose_context("…")` → `{items: []}` — *indistinguishable* from "this query
genuinely has no matches." An agent concludes "no evidence on this point" when
in fact nothing is indexed: a false negative over the entire corpus.
**Amplified by us.** 0.0.6 made `ingest(embed=False)` the default, so corpora
now sit unembedded far more often — this bug is strictly more likely post-0.0.6.
**Fix.**
- Raise a typed `NotIndexedError` when **0** chunks are embedded (unambiguous
  misuse).
- When *partially* indexed, attach `searched_fraction` / `embedded_coverage`
  to the result and `warnings.warn(...)`, so a caller always knows it queried
  12% of the corpus, not 100%.
**Note.** I deliberately punted this in 0.0.5 to keep return types stable; the
evidence-tool lens makes it a P0. Prefer the typed error for the zero case.

### BUG-3 · `study_ledger` silently truncates at `limit=200` — `HIGH` · `S` · PLANNED
**Symptom.** A 1,211-assessment study returns 200 rows by default with no
signal the rest exist (same silent-incompleteness family as BUG-1/2).
**Fix.** Attach `total` and `returned` to the `Ledger` result; keep `limit` but
make truncation observable. Add `study_ledger(doc_id=…)` (and, post FEAT-4,
`section_id=…`) scoping so callers needn't filter manually. Consider an
unbounded default for ledgers specifically (they're the deliverable).

### BUG-4 · `study_ledger` per-chunk truth is ambiguous across agents — `MEDIUM` · `S` · PLANNED
**Symptom.** A pre-filter wrote assessments under one `agent_id` and an analyst
corrected them under another → the ledger shows two rows for one chunk with no
"current" winner. Our latest-wins dedup is **per `(chunk, agent)`** (by design,
for multi-agent coexistence), so cross-agent correction isn't resolved.
**Fix.** This is the read side of FEAT-5 (`supersedes` edge). Until then,
document that the ledger is per-`(chunk, agent)` and add an optional
`collapse="latest"|"by_agent"` mode. *Not* a change to the default dedup.

---

# Part B — Feature requests (kglite-docs)

Generalized capabilities. Grouped; each notes the motivating evidence and the
generalization beyond legal.

## B.1 Honest coverage & observability (the P0 theme)

### FEAT-1 · `corpus.coverage_report()` + per-page extraction metrics — `HIGH` · `M` · PLANNED
A first-class, legal-grade coverage object: per-doc and per-section
`extractable_text_ratio`, `image_only_pages`, `low_text_pages`, `unembedded`,
`pending_ocr`, and a one-line human summary ("X% of pages are image-only and
unanalyzed unless OCR'd"). Makes "we did not read these" impossible to miss.
*Generalizes:* any scanned/mixed corpus (medical charts, archives, due
diligence) has the same blind-spot risk.

### FEAT-2 · `corpus.status()` — one-call state — `MEDIUM` · `S` · PLANNED
`{docs, chunks, embedded, unembedded, image_pages, pending_ocr, studies}` —
the first thing an agent should call; answers BUG-1/2/3 at a glance. The
no-arg "where am I?" primitive.

### FEAT-3 · Coverage in `search` results — `MEDIUM` · `S` · PLANNED
(Pairs with BUG-2.) Every retrieval result carries `searched_fraction`; a
study/ledger carries `assessed_fraction`. Coverage travels with the data.

## B.2 Evidence-model primitives

### FEAT-4 · Provenance axis on assessments — `HIGH` · `M` · PLANNED
**The single highest-value generalizable idea from the field work.** `verify`
today conflates *"verified against the primary source"* with *"verified against
a party's paraphrase of it"* — a fatal equivocation in a brief (a `verified`
chunk turned out to confirm only that *the defense asserts* a witness admitted
something; the verifier never read the scanned deposition). Add a
`provenance` field to `assess`/`verify`: `primary_text | characterization |
scanned_unread`, surfaced in the ledger. Backed by the [17–33% hallucination
benchmark][stanford] and the [DRM grounding metric][drm]. *Generalizes:* every
evidence/fact-check/research workflow needs "is this grounded in the source or
in someone's summary of it."

### FEAT-5 · `supersede_assessment(old_id, …)` + edge — `MEDIUM` · `S` · PLANNED
Audit-preserving correction: an explicit `(:Assessment)-[:SUPERSEDES]->(:Assessment)`
edge; `study_ledger` returns current-by-default with `include_superseded=True`
for history. Resolves BUG-4 without weakening multi-agent coexistence.

### FEAT-6 · Pinpoint spans on assessments — `MEDIUM` · `M` · PLANNED
Let `assess`/`add_summary` record a **character span / quote offset** within a
chunk (extends the `USED_CONTEXT` span work shipped in 0.0.6), so the ledger
emits pinpoint cites ("fls. 249 ¶3") and an export can highlight the exact
passage. *Generalizes:* citations/quote-grounding in any domain. See
[pinpoint-citation pipeline][pinpoint].

### FEAT-7 · `stance="deferred"` / `blocked_on` state — `MEDIUM` · `S` · PLANNED
A first-class "needs-evidence" state so image/un-OCR'd pages aren't parked as
`neutral, weight=0` (which pollutes tallies and hides them). Keeps the worklist
honest: "26% awaiting extraction," not "neutral." Pairs with BUG-1 (deferred is
the natural stance for `:NeedsOcr` chunks).

### FEAT-8 · Cross-object conflict surfacing — `HIGH` · `M` · PLANNED
Flag when a `supports` and an `against` assessment concern the **same object**
(chunk, section, or — post FEAT-9 — claim). In the field run a critical adverse
fact (a PAD that *dismissed* the defendant, contradicting the "all PADs
archived" theme) was hidden by the flat model; agents nearly pleaded a
falsehood that invites sanctions. A `study("conflicts", id)` action returning
contested objects. *Generalizes:* contradiction detection in any analysis.

## B.3 Document structure

### FEAT-9 · `Section` / `SourceDoc` nodes + `chunk.doc_type` — `HIGH` · `L` · PLANNED
**Biggest quality lever.** A court "processo" PDF is a *container of documents*;
flattening to `Document→Page→Chunk` caused real false findings (speaker
misattribution across exhibits) and blocks the user's own ask: "what was the
main evidence in the contestação?" Model:
```
(:Document)-[:HAS_SECTION]->(:Section {doc_type, title, page_start, page_end,
                                       author?, date?, source_ref?})
(:Section)-[:HAS_CHUNK]->(:Chunk)        // + chunk.section_id / chunk.doc_type
(:Section)-[:NEXT_SECTION]->(:Section)   // docket / reading order
```
**Generalized population** (this is the key design point — the *signal* is
domain-specific, the *model* is not): build `Section`s from, in order of
availability, (a) the **PDF outline/bookmarks**, (b) `headings_json` we already
capture, (c) pluggable **format adapters** (e.g. a PROJUDI adapter that reads
the footer `Id.` + the *Índice de Documentos* — kept out of core). Unlocks
section-scoped Studies (`study("next", section_id=…)`), per-section coverage,
and "main evidence of section X" as one traversal. Validated by a working
prototype on the live corpus (84 sub-documents recovered) and by [typed-KG >
RAG retrieval results][kcap].

### FEAT-10 · Structure-aware chunking — `MEDIUM` · `M` · PLANNED
Chunk on section/heading boundaries (EMENTA, DOS FATOS, DO DIREITO,
DISPOSITIVO…) so arguments aren't split mid-claim (a `DISPOSITIVO` and its
dispositive verb landed in different chunks). Improves retrieval and per-chunk
assessment quality. *Generalizes:* any structured document (contracts,
papers, manuals).

### FEAT-11 · Summary-augmented chunking — `MEDIUM` · `M` · PLANNED
Optionally prepend a document/section-level summary to each chunk before
embedding so chunks don't lose global context (mitigates speaker/source
confusion). Opt-in (costs an LLM summary pass). Per [NLLP 2025][sac].

## B.4 Concurrency & scale

### FEAT-12 · Document the concurrency model + guardrails — `HIGH` · `S` · PLANNED
The "parallel agents" story is really **one writer (the MCP server / one
process), many `agent_id`s**. External OS-level parallel writers to one `.kgl`
race on save (our punchcard is safe for *sequential* separate processes via the
persisted checkout, but not truly concurrent ones). Confirmed independently in
the field run (agents were inverted to read-only judges + a single writer).
Ship: (a) a plain-English concurrency section in the docs; (b) a **loud
file-lock** so a second writer fails fast instead of corrupting; (c)
`corpus.assess_many(rows)` to make the single-writer funnel ergonomic.

## B.5 Ergonomics & trust docs

### FEAT-13 · Confidentiality statement — `HIGH (adoption)` · `S` · PLANNED
Document plainly: **all parsing/embedding/assessment is local; the only network
call is a one-time bge-m3 model fetch from HuggingFace; no document content is
transmitted.** The first-run `You are sending unauthenticated requests to the
HF Hub` message reads like data egress (it's only the model download) — we
already set `HF_HUB_OFFLINE` when weights are cached; document the rest. For
legal/medical adoption this sentence matters more than any feature.

### FEAT-14 · `ResultView` / `get_chunk` shape ergonomics — `LOW` · `S` · PLANNED
Document `ResultView` in the `Corpus.cypher` docstring and make it
iterable/dict-row friendly (`for row in corpus.cypher(...)`, `row["col"]`).
Make `get_chunk` return a typed `ChunkDetail` with both attribute and
`__getitem__` access, in the signature.

---

# Part C — Offloaded to the parent kglite library

Defects in kglite's Cypher engine, **re-verified on kglite 0.10.9**. Posted to
the kglite inbox; we adopt the fix on their next release (no kglite-docs change
needed). **We do not work around these in kglite-docs.**

| ID | Issue (0.10.9) | Status |
|----|----------------|--------|
| KG-1 | A node **property named `label` is shadowed** — `RETURN n.label` returns the node's *type string*, not the property. | `OFFLOADED` (reported) |
| KG-2 | **`CONTAINS` cannot be a relationship type** (reserved operator) — `CREATE (a)-[:CONTAINS]->(b)` is a syntax error. | `OFFLOADED` (reported) |

**Re-verified as ALREADY FIXED on 0.10.9** but **still passed to kglite** (with
a "did not reproduce on 0.10.9" caveat, so they can confirm regression coverage
or spot a residual edge case): shared-variable comma-join (`(s)-->(c),(a)-->(c)`),
reverse-arrow match (`(c)<-[:R]-(a)`), inline-map multi-MATCH `CREATE`, and —
most importantly — **cypher-`CREATE`d edges now survive save→load**. Our own
study edges were never exposed (we use the bulk `add_connections` API), and we
confirmed assess→save→reopen→ledger is intact on 0.10.9.

> Workarounds while KG-1/KG-2 are open: don't name a chunk/section property
> `label` (use `doc_type` etc.); don't name an edge type `CONTAINS` (use
> `HAS_SECTION`). FEAT-9 already follows both.

---

# Part D — Milestones

### 0.0.7 — "Honest coverage" (the trust release)
BUG-1, BUG-2, BUG-3, FEAT-1, FEAT-2, FEAT-3, FEAT-13. Theme: no skipped page,
unindexed chunk, or truncated ledger is ever invisible. Highest ROI; mostly
`S`/`M`; benefits every use case.

### 0.0.8 — "Evidence integrity"
FEAT-4 (provenance), FEAT-5 (supersede), FEAT-7 (deferred), FEAT-8 (conflict
surfacing), BUG-4. Makes the assessment/verify model legally defensible and
generally trustworthy.

### 0.0.9 — "Document structure"
FEAT-9 (sections), FEAT-6 (pinpoint spans), FEAT-10 (structure-aware chunking).
The biggest quality lever; unblocks section-scoped studies and pinpoint cites.

### 0.1.0 — "Scale & polish"
FEAT-11 (summary-augmented chunking), FEAT-12 (concurrency + `assess_many` +
file-lock), FEAT-14 (ergonomics). Plus the deferred review of whether the MCP
surface can fold back toward 12 tools (e.g. relocating `summary`'s ephemeral
`claim`/`consensus`, now superseded by the `study` flow).

---

# Part E — The legal vertical (separate package, builds on the primitives)

Legal is a flagship validated use case, but its domain concepts live in a
`kglite-docs-legal` extension, not the core. It composes the primitives above:

- **Charge tracker** — atomize each accusation head (humiliation, affair-rumor,
  staged-scene, insubordination…) as nodes; link each to supporting/rebutting
  evidence (FEAT-8 conflict surfacing) and disciplinary outcome, so no charge
  is unanswered and quantum/causation isolate per head. ([DISCOG TAR-as-link-
  prediction][discog].)
- **Speaker/attribution** — a `Person`/`speaker` node + edge so WhatsApp/
  deposition attributions travel *in the ledger row* (fixes the recurring
  Bruno-vs-Eduardo misattribution). Built on FEAT-9 sections + FEAT-4.
- **Citation/cross-reference resolver** — map `DOC 06, fls. 131`, `Evento 18`,
  `SINDICÂNCIA_2.pdf p.10` to chunk/section ids (built on FEAT-6 spans +
  FEAT-9). ([Pinpoint-citation pipeline][pinpoint].)
- **PROJUDI / docket adapters** — the format-specific `Section` population for
  FEAT-9 (footer `Id.` + Índice de Documentos parsing).
- **Operative-ruling / holding locator** — tag the `DISPOSITIVO` / dano-moral
  quantum so a brief can argue proportionality.

Adopt the [LexChronos][lex] extract→score→verify **rubric** (incl. a
duplicate/repetition dimension) and confidence-based auto-stop as a legal
preset over the generic `study` loop.

---

## References

- [stanford]: Magesh et al., *Hallucination-Free? Assessing Legal AI Tools* — leading legal-RAG tools hallucinate 17–33% of queries. https://onlinelibrary.wiley.com/doi/full/10.1111/jels.12413
- [kcap]: Kondo et al., typed legal KG (Fact/LegalNorm/Application/Provision) beats RAG (micro-recall 0.667 vs 0.351). K-CAP 2025. https://dl.acm.org/doi/10.1145/3731443.3771354
- [discog]: DISCOG — TAR as link-prediction over a heterogeneous graph + LLM validate layer. ACL 2025. https://arxiv.org/pdf/2405.19164
- [lex]: LexChronos — dual-agent extract→score→verify with explicit rubric + auto-stop. 2026. https://arxiv.org/pdf/2603.01651
- [sac]: Summary-Augmented Chunking. NLLP 2025. https://aclanthology.org/2025.nllp-1.3/
- [drm]: Document-Level Retrieval Mismatch (provenance metric). https://arxiv.org/html/2510.06999v1
- [pinpoint]: Pinpoint citations as a normalized KG-triplet pipeline stage. https://arxiv.org/html/2502.20364v1

*Source field reports: `inbox/read/2026-05-29-from-claude-code-legal-field-report-v0.0.6.md` and `…-followup-sectioning-and-legal-kg-roadmap.md`.*
