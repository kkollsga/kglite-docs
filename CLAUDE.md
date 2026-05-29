# kglite-docs — Claude Code Conventions

Agent-first document knowledge base: ingest documents, let agents weigh
per-chunk evidence for/against a question, verify across agents, conclude.
A **pure-Python wrapper over [kglite](https://github.com/kkollsga/kglite)**
(the graph engine + bge-m3 embedder) exposed as a library *and* an MCP server.

## Build & test

Use the repo venv for everything — `./.venv/bin/python`:

```bash
./.venv/bin/python -m pytest -m "not embed"        # fast suite (stub embedder)
./.venv/bin/python -m pytest                        # full suite (real bge-m3; slower)
./.venv/bin/python -m ruff check src/ tests/        # CI HARD gate
./.venv/bin/python -m mypy src/kglite_docs          # strict; keep it clean
```

- The venv is an **editable install pointed via `.venv/lib/.../_editable_impl_kglite_docs.pth`** at this repo's `src/`. If imports break after a move, that `.pth` is the thing to fix (the repo was relocated from `Python/Document_Ingestion`). Don't `pip install -e .` casually — the venv lacks the `editables` build dep; edit the `.pth` instead.
- **What CI runs** (`.github/workflows/ci.yml`, matrix ubuntu+macOS × py3.10–3.13): `ruff check src/ tests/` (**hard gate**), `mypy src/kglite_docs` (*advisory* — `|| true` — but we keep it at **0 errors**; don't regress it), and `pytest -m "not embed and not mcp"`.
- CI **skips `mcp` and `embed` tests** (no model in CI, server boot kept out of the matrix). **Run them locally before any release** — `pytest -m "mcp or embed"`. A bug there won't fail CI but will ship.

## Architecture

```
Corpus (corpus.py)         ← the public façade; thin delegating methods
  └─ Store (store.py)      ← wraps kglite.KnowledgeGraph (`self.g`); dict↔DataFrame adapter
       └─ kglite.KnowledgeGraph
feature modules            ← ingest/ (parse→chunk→graph), study.py, enrich.py (summaries),
                             tagging.py, review.py, cluster.py, quality.py, activity.py, …
mcp_server/                ← server.py (FastMCP wiring + _INSTRUCTIONS), tools.py (noun dispatchers)
```

- **`schema.py` is the single source of truth** for node-type / edge-type
  constants, secondary-label names, and the discriminator → label maps
  (`label_for("chunk.status", …)`). Add new node types / labels here.
- **`types.py` is the source of truth for typed returns** — `Literal`s for
  enum args, `TypedDict`s (often `total=False`) for dict-shaped returns.
- Feature modules are **stateless functions** taking `store` (and `embedder`
  where needed) first; `Corpus` is a thin delegating wrapper. Mirror an
  existing module (`tagging.py`, `study.py`) when adding one.

## The kglite relationship — the boundary principle (north star)

kglite-docs is a wrapper. The dividing line, in both directions:

> **Generic graph capability belongs in kglite. Document-ingestion,
> agent-workflow, and evidence-model logic belongs here.**

- **Don't reinvent graph primitives** — Cypher, vector search, labels,
  save/load, `describe()` live in kglite and reach us through `self.g` /
  `corpus.cypher()`. If you want a new graph capability, prefer asking kglite
  for it over building a fragile workaround here.
- **Offload kglite bugs; don't work around them.** When kglite's engine is
  wrong (a Cypher-dialect gap, a label-index issue), the protocol is:
  1. **Reproduce on the *currently pinned* kglite first** — versions move fast
     (0.10.5→0.10.9 in days); ~⅔ of reported "bugs" turn out already-fixed.
     Re-verifying before reporting has saved us from chasing ghosts repeatedly.
  2. Write a note to `../KGLite/inbox/unread/` (`YYYY-MM-DD-from-kglite-docs-*.md`)
     with a **standalone minimal repro** and the version. Lead with a thank-you;
     these get fixed fast (our last batch landed in 0.10.8 same-day).
  3. **Do not add a workaround in kglite-docs.** Bump the kglite pin once they
     ship the fix and adopt it. (Track open offloads in `ROADMAP.md` Part C.)
  - Even a bug that *doesn't* reproduce on the current version gets passed along
    (with a "didn't reproduce on X" caveat) — useful for their regression coverage.

## North star: honest coverage

*Every coverage-reducing decision must be observable in the return value.*
Silent truncation / skipping is fine in a code-search tool; in an **evidence**
tool it's a liability — a user who trusts a green light while half the record is
invisible makes a confident wrong decision. When you add anything that bounds,
samples, filters, or skips: surface it (`total`/`returned`, `searched_fraction`,
a coverage field, a typed error, or a loud warning). This is the difference
between "good demo" and "trustworthy for legal/medical/forensic work." See
`ROADMAP.md`.

## MCP surface discipline

The agent surface is **~13 thin noun tools**, each a CLI-style dispatcher taking
an `action` verb first (`document("ingest", …)`, `study("assess", …)`) — think
`git branch list`, not 45 functions. Adding a new top-level noun is a real cost;
prefer a new `action` on an existing noun. Legal/medical/etc. domain concepts do
**not** go in the core — they're a separate vertical built on the primitives.

## Key patterns (learned the hard way)

- **Reified nodes for multi-agent annotation.** kglite allows at most one edge
  per `(src, dst, type)`, so anything multiple agents annotate independently is a
  reified node (`Tagging`, `Assessment`, `ReviewEvent`), not an edge property.
- **Create edges via the bulk API (`store.upsert_edges` → `g.add_connections`),
  not cypher `CREATE`.** Bulk-API edges are the well-tested persistence path;
  cypher-`CREATE`d edges had save/load bugs in older kglite. Our study edges use
  the bulk API and survive save/reopen — keep it that way.
- **Append-only + label state, never mutate a property you'll re-read.** Status
  lives as a secondary label (`label_for(...)` + `swap_label`), set once as a
  property at creation. (Historical kglite footgun: in-place String `SET` could
  panic; the house style dodges it and gives free audit history.)
- **TypedDict returns: `cast()` at the `Corpus` boundary.** Methods return plain
  dicts from cypher but are typed as `TypedDict`s — wrap with `cast(T, …)` (see
  `corpus.py`). Keeps strict-mypy at 0 without weakening the public types.
- **Persist after MCP mutations.** Mutating tool branches call `_persist(corpus)`
  (`tools.py`); there's a save-on-shutdown backstop in `__main__.py`. The
  long-lived server holds changes in memory otherwise.
- **Embedding is opt-in / two-phase.** `ingest(embed=False)` (default) just
  parses+chunks; `index()` embeds later (bounded, loop-friendly). Many workflows
  (browse, tag, review, **study**) need no embeddings at all — don't force them.
- **Cypher quirks to avoid** (current kglite): don't name a node property
  `label` (shadowed by the type accessor) or an edge type `CONTAINS` (reserved);
  build "latest-per-group" with `WITH … collect(x)[0]`. If `labels()`/props on a
  collected node ever misbehave, re-`MATCH` it by id.

## Code health

Each pass through a file leaves it more compartmentalized than you found it.

- Factor a function past ~80 lines or when it handles 3+ unrelated concerns.
- Fixing a bug — scan for the *class* of bug (the reported symptom is rarely the
  only instance). The OCR-detection and "silent empty" bugs are both the same
  honest-coverage class; fix the class.
- A new feature is a chance to extract a wanted helper. Don't over-design; don't
  pass it up.
- Don't add a parameter/branch/flag without checking whether the existing
  structure should absorb it instead.

## When changing an agent-facing method — the checklist

A capability is usually defined in **five** places. Touch all that apply:

1. **Feature module** (`study.py`, `enrich.py`, …) — the implementation.
2. **`corpus.py`** — the thin delegating method (+ typed signature).
3. **`types.py`** — `Literal` / `TypedDict` for new args or returns.
4. **`mcp_server/tools.py`** — the noun dispatcher branch (+ `_persist` if it
   mutates), and the **docstring** (agents read it).
5. **`mcp_server/server.py` `_INSTRUCTIONS`** — the happy-path, if it changes the
   workflow an agent should follow.
6. **`CHANGELOG.md` `[Unreleased]`** + **tests** (`tests/test_*`).

`schema.py` first if it introduces a node type / label.

## Concurrency

**Single-writer.** One process owns the `.kgl`. "Parallel agents" means *many
`agent_id`s through one writer* (the MCP server), not many OS processes writing
one file — concurrent external writers race on save and can corrupt. The
`study("next", agent_id=…)` punchcard is safe for *sequential* separate
processes (persisted checkout) but not truly-concurrent ones. State this in any
"parallel agents" docs; funnel external fan-out writes through one process.

## Commits & releases

- **Version source of truth:** `pyproject.toml` (`version = "x.y.z"`, ~line 9).
  Static. Pre-1.0 0.0.x: each release bumps the patch digit even for features;
  breaking changes are allowed (call them out in the CHANGELOG).
- **Release flow:** bump the version, commit, **push to `main`** → `ci.yml` runs
  → on success `release.yml` checks PyPI and, if the version is new, builds +
  publishes (Trusted Publishing) + tags a GitHub release. No manual tag.
- Commit format `type: short description` (`feat`/`fix`/`docs`/`refactor`/
  `test`/`chore`). Update `CHANGELOG.md` `[Unreleased]` for user-visible changes.
- End commit messages with the `Co-Authored-By:` trailer.

**Pushing requires explicit, in-the-moment approval.** Default is *don't push*.
"Looks good" / "we're ready" earlier in a turn does **not** authorize a push — an
unapproved push to `main` auto-publishes to PyPI (irreversible). When in doubt,
prepare the commit, stop, and ask.

**Exception — the CI fix-and-push loop.** When an approved push fails CI and the
cause is a bug in shipped code or CI/test infra (not a feature gap), push
`fix(...)`/`ci(...)` commits for *that same loop* without re-asking until CI is
green. Stops applying once CI is green (fresh approval for the next release), if
a fix changes the release shape (new version/feature → ask), after ~3 iterations
without progress, or if the conversation pivots.

**One version bump per push.** A version isn't "released" until pushed. If a
`vx.y.z` release commit is already local/unpushed, fold follow-up work into the
same `[x.y.z]` CHANGELOG block — don't mint `x.y.z+1` on top. Mint a new version
only after a clean push to origin.

### Default work procedure (phase-by-phase)

Once a plan is approved (e.g. via plan mode), execute it **autonomously, phase
by phase** — don't pause between phases to check in.

For **each phase**:
1. Implement the phase.
2. **Make it green:** `ruff check src/ tests/`, `mypy src/kglite_docs`, and the
   relevant tests all pass.
3. **Commit the phase** (`type: description`, update `CHANGELOG.md
   [Unreleased]`). **No version bump. No push.** One commit per phase keeps
   history bisectable and each phase reversible.
4. Proceed to the next phase.

The only mid-plan stops are genuine blockers — a failing test you can't fix, or
an architectural surprise that invalidates a later phase. Surface those; don't
power through.

**At the end of the complete plan, the release is a joint step with the user.**
Run the release pre-flight (below), then **bump the version + push together**:
Claude prepares the version-bump commit and the CHANGELOG promotion; the user
gives the in-the-moment push approval, and we publish. **No earlier phase
touches `pyproject.toml`'s version or pushes** — phase commits stay local and
the single irreversible PyPI-publishing push is one deliberate, user-approved
moment at the end.

**Release pre-flight** (run locally — CI doesn't cover all of it): `ruff check
src/ tests/` clean, `mypy src/kglite_docs` clean, `pytest` (incl. `mcp`+`embed`)
green, and confirm the wheel includes data files (`manifest.yaml`,
`manifest.skills/`) via a dry `pip wheel . --no-deps`.
