"""Typed MCP tool registrations — CLI-flavored dispatchers.

Each tool is a *noun*; the first positional `action` is the *verb*.
Patterned like a CLI (`document ingest`, `summary verify`, `review claim_next`)
so an agent reading the methodology skill files can copy the verb-noun
combos straight into tool calls.

The dispatch lives here; each branch is a thin shim over the underlying
`Corpus` method, which keeps the typed Python API stable and unchanged.
"""

from __future__ import annotations

import contextlib
from typing import Any


def _require(value: Any, name: str, action: str, tool: str) -> Any:
    if value is None:
        raise ValueError(f"{tool}({action!r}): {name} is required")
    return value


def _persist(corpus: Any) -> None:
    """Flush the in-memory graph to its .kgl path after a mutation. The
    long-lived MCP server otherwise only holds changes in memory — a
    crash or hard kill would lose tool-driven ingest/index work. No-op
    for path-less (in-memory) corpora."""
    if getattr(corpus.store, "path", None) is not None:
        with contextlib.suppress(Exception):
            corpus.save()


def register_typed_tools(app: Any, corpus: Any) -> None:
    """Register the kglite-docs CLI-style noun dispatchers on a FastMCP app."""

    # ─── document ─────────────────────────────────────────────────────────

    @app.tool()
    def document(
        action: str,
        path: str | None = None,
        directory: str | None = None,
        text: str | None = None,
        title: str | None = None,
        format: str | None = None,
        source_uri: str | None = None,
        recursive: bool = True,
        embed: bool = False,
        structure_aware: bool = False,
        context_summary: str = "",
        batch_size: int = 64,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        doc_id: str | None = None,
        doc_a: str | None = None,
        doc_b: str | None = None,
        queries: list[str] | None = None,
        top_k_per_query: int = 5,
        max_tokens_per_query: int = 2000,
        out_path: str | None = None,
        include_summaries: bool = False,
        agent_id: str | None = None,
    ) -> Any:
        """Document-level operations.

        Actions:

        - **`ingest`** — load files into the corpus. Pass exactly one of:
          `path="/abs/paper.pdf"` (single file),
          `directory="/abs/papers/"` (bulk, `recursive=True` by default),
          or `text="..."` + `title="..."` (inline content).
          Pipeline: parse → token-aware chunk → store. **Embedding is
          opt-in** — by default ingest does NOT embed (fast, no model
          load); it returns a `hint` to run `index` next. Pass
          `embed=True` to embed inline in one shot. Idempotent on sha256.
          Pass `structure_aware=True` to start a fresh chunk at every
          top-level heading (cleaner section boundaries; default packs greedily).
          Pass `context_summary="..."` (single file / text mode) to prepend a
          doc-level blurb to each chunk *before embedding* — the vector carries
          global context (less cross-doc source confusion); stored text is
          unchanged. You provide the summary; none is generated here.
        - **`index`** — embed ready-but-unembedded chunks so `search`
          works. Run after `ingest` (the two-phase flow). **Bounded per
          call** (≈30s wall-clock budget) so a big corpus never blocks
          past a timeout: if the result has `pending > 0` (and a `hint`),
          call `index` again — repeat until `pending == 0`. Optional
          `doc_id` to scope. Idempotent. Skip entirely for non-semantic
          workflows (browse / cypher / tag / review / ocr / export /
          translate need no embeddings).
        - **`list`** — list ingested documents. Optional `filters`, `limit`.
        - **`get`** — fetch metadata + heading-derived TOC. Requires `doc_id`.
        - **`sections`** — the document's sections (the grain between document
          and chunk, derived from the PDF outline or top-level headings at
          ingest), in reading order with per-section `chunk_count`. Requires
          `doc_id`. Use a section's id as `section_id` to scope `study("next")`/
          `study("ledger")`.
        - **`export`** — write doc to MD / DOCX / PDF. Requires `doc_id`,
          `out_path`; optional `format`, `include_summaries`.
        - **`compare`** — side-by-side cross-doc retrieval. Requires `doc_a`,
          `doc_b`, `queries=[...]`. For each query, returns top hits from
          each doc plus a budgeted merged context bundle.
        - **`status`** — one-call corpus snapshot: docs, pages, chunks,
          embedded/unembedded, image_pages, pending_ocr, studies. Check first.
        - **`coverage`** — honest extraction + embedding coverage per doc +
          corpus, with a human `summary` (what's image-only / low-text /
          unembedded). Optional `doc_id` to scope.
        - **`map`** — one-call triage overview so you orient *without reading the
          corpus*: chunk counts, the `content_kind` breakdown (prose/table/list/
          code/sparse), boilerplate / low-quality counts, structured-entity
          coverage (chunks with dates/money/emails/urls/ids), embedding state,
          and OCR-pending pages, plus a human `summary`. Optional `doc_id`.
          Route work with the matching label predicates (`MATCH (c:Chunk:Table)`,
          `MATCH (c:Chunk:HasMoney)`).

        Examples::

            document("ingest", path="/abs/paper.pdf")   # then document("index")
            document("ingest", directory="/abs/papers/")
            document("ingest", path="/abs/paper.pdf", embed=True)  # one-shot
            document("index")                            # embed everything
            document("list")
            document("get", doc_id="doc_abc...")
            document("export", doc_id="doc_abc...", out_path="paper.md")
            document("compare", doc_a="doc_a...", doc_b="doc_b...",
                     queries=["retrieval method", "training objective"])
        """
        if action == "ingest":
            modes = sum(x is not None for x in (path, directory, text))
            if modes != 1:
                raise ValueError(
                    "document('ingest'): pass exactly one of path, directory, text",
                )
            if directory is not None:
                results = corpus.ingest_dir(
                    directory, recursive=recursive, embed=embed,
                    structure_aware=structure_aware,
                )
                _persist(corpus)
                pending = corpus.count_unembedded()
                out: dict[str, Any] = {
                    "ingested": sum(1 for r in results if r.created),
                    "skipped": sum(1 for r in results if not r.created),
                    "total_chunks": sum(r.chunk_count for r in results),
                    "embedded": sum(r.embedded for r in results),
                    "unembedded": pending,
                    "ocr_pending": sum(r.ocr_pending_pages for r in results),
                    "docs": [
                        {"doc_id": r.doc_id, "created": r.created,
                         "pages": r.page_count, "chunks": r.chunk_count,
                         "format": r.format}
                        for r in results
                    ],
                }
                if pending:
                    out["hint"] = f"{pending} chunks unembedded — run document('index') to enable search"
                return out
            if text is not None:
                if not title:
                    raise ValueError(
                        "document('ingest', text=...): title is required",
                    )
                r = corpus.ingest(
                    text=text, title=title, format=format or "md", embed=embed,
                    structure_aware=structure_aware, context_summary=context_summary,
                )
            else:
                r = corpus.ingest(
                    path, title=title, format=format, source_uri=source_uri,
                    embed=embed, structure_aware=structure_aware,
                    context_summary=context_summary,
                )
            _persist(corpus)
            res = {
                "doc_id": r.doc_id, "created": r.created,
                "page_count": r.page_count, "chunk_count": r.chunk_count,
                "embedded": r.embedded, "ocr_pending_pages": r.ocr_pending_pages,
                "format": r.format,
            }
            pending = corpus.count_unembedded(doc_id=r.doc_id)
            if pending:
                res["hint"] = f"{pending} chunks unembedded — run document('index', doc_id='{r.doc_id}') (or omit doc_id for all) to enable search"
            return res
        if action == "index":
            result = corpus.index(doc_id=doc_id, batch_size=batch_size)
            _persist(corpus)
            # index() is bounded per call (wall-clock budget) so a large
            # corpus never blocks past a per-call timeout — loop until
            # pending hits 0.
            if result.get("pending"):
                result["hint"] = (
                    f"{result['pending']} chunks still unembedded — call "
                    "document('index') again (repeat until pending == 0)"
                )
            return result
        if action == "list":
            return corpus.list_documents(filters=filters, limit=limit)
        if action == "get":
            return corpus.get_document(_require(doc_id, "doc_id", action, "document"))
        if action == "export":
            return str(corpus.export_document(
                _require(doc_id, "doc_id", action, "document"),
                _require(out_path, "out_path", action, "document"),
                format=format, include_summaries=include_summaries,
            ))
        if action == "compare":
            return corpus.compare_documents(
                _require(doc_a, "doc_a", action, "document"),
                _require(doc_b, "doc_b", action, "document"),
                queries=_require(queries, "queries", action, "document"),
                top_k_per_query=top_k_per_query,
                max_tokens_per_query=max_tokens_per_query,
                agent_id=agent_id,
            )
        if action == "sections":
            return corpus.list_sections(_require(doc_id, "doc_id", action, "document"))
        if action == "status":
            return corpus.status()
        if action == "coverage":
            return corpus.coverage_report(doc_id=doc_id)
        if action == "map":
            return corpus.triage_map(doc_id=doc_id)
        raise ValueError(
            f"document(): unknown action {action!r}. Valid: ingest, index, list, "
            "get, sections, map, export, compare, status, coverage",
        )

    # ─── chunk ────────────────────────────────────────────────────────────

    @app.tool()
    def chunk(
        action: str,
        id: str | None = None,
        with_neighbors: bool = False,
        with_summaries: bool = False,
        window: int = 0,
        top_k: int = 10,
        agent_id: str | None = None,
    ) -> Any:
        """Chunk-level reads.

        Actions:

        - **`get`** — fetch one chunk by id. Optional `with_neighbors=True`
          (prev/next ids), `with_summaries=True` (inline verified summaries),
          `window=N` (read-context: the N chunks before & after in reading
          order, *with text*, as `context_before`/`context_after` — use this
          when a chunk is hard to interpret on its own), `agent_id` (records
          a View).
        - **`similar`** — nearest-neighbor chunks by embedding cosine.
          Pass `top_k` (default 10).

        Examples::

            chunk("get", id="doc_abc#p2#c3", with_summaries=True)
            chunk("get", id="doc_abc#p2#c3", window=1)   # + neighbours' text
            chunk("similar", id="doc_abc#p2#c3", top_k=5)
        """
        if action == "get":
            return corpus.get_chunk(
                _require(id, "id", action, "chunk"),
                with_neighbors=with_neighbors,
                with_summaries=with_summaries,
                window=window,
                agent_id=agent_id,
            )
        if action == "similar":
            return corpus.similar_chunks(
                _require(id, "id", action, "chunk"), top_k=top_k,
            )
        raise ValueError(
            f"chunk(): unknown action {action!r}. Valid: get, similar",
        )

    # ─── search ───────────────────────────────────────────────────────────

    @app.tool()
    def search(
        query: str,
        mode: str = "hits",
        top_k: int = 10,
        max_tokens: int = 4000,
        per_doc_cap: int | None = None,
        filters: dict[str, Any] | None = None,
        with_summaries: bool = False,
        include_summaries: bool = True,
        agent_id: str | None = None,
    ) -> Any:
        """Semantic search over chunks. Two modes:

        - **`mode="hits"`** (default) — top-`top_k` chunks by bge-m3
          cosine similarity. `filters={"doc_id": "..."}` to scope.
          `with_summaries=True` to inline verified summaries.
          Returns a list of `{id, score, text, doc_id, page, ...}`.
        - **`mode="compose"`** — *use this when you're feeding the result
          to an LLM*. Returns a budgeted, ranked context bundle
          (`{query, budget_tokens, used_tokens, items}`) packed up to
          `max_tokens`. Optional `per_doc_cap` keeps one doc from
          crowding out the bundle. Inlines verified summaries by
          default (`include_summaries=True`).

        Pass `agent_id="me"` to record View edges on every hit (queryable
        later via `agent("activity", id="me")`).

        Examples::

            search("dense retrieval")
            search("dense retrieval", top_k=5, filters={"doc_id": "doc_a"})
            search("dense retrieval", mode="compose", max_tokens=3000)
        """
        if mode == "hits":
            return corpus.search(
                query, top_k=top_k, filters=filters,
                with_summaries=with_summaries, agent_id=agent_id,
            )
        if mode == "compose":
            return corpus.compose_context(
                query, max_tokens=max_tokens, per_doc_cap=per_doc_cap,
                include_summaries=include_summaries, agent_id=agent_id,
            )
        raise ValueError(
            f"search(): unknown mode {mode!r}. Valid: hits, compose",
        )

    # ─── summary ──────────────────────────────────────────────────────────

    @app.tool()
    def summary(
        action: str,
        target_id: str | None = None,
        id: str | None = None,
        text: str | None = None,
        agent_id: str | None = None,
        verifier_agent_id: str | None = None,
        verdict: str | None = None,
        target_kind: str = "Chunk",
        depth: str = "chunk",
        model: str = "",
        tags: list[str] | None = None,
        notes: str = "",
        status: str | None = None,
        threshold: float = 0.5,
        query: str | None = None,
        against_chunk_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> Any:
        """Summary write/verify/inspect operations.

        Actions:

        - **`add`** — write a Summary on a Chunk/Page/Document. Status
          starts `unverified`. Requires `target_id`, `text`, `agent_id`.
          Optional `target_kind` (default "Chunk"), `depth` (chunk/section/
          document), `model`, `tags`. Returns the summary id.
        - **`verify`** — apply a verdict (`verified`/`disputed`/
          `needs_revision`). Requires `id`, `verdict`, `verifier_agent_id`.
          Server rejects self-verification.
        - **`list`** — summaries on a target. Requires `target_id`;
          optional `target_kind`, `status`, `depth` filters.
        - **`ground`** — sentence-level grounding score against source
          chunks (hallucination guard). Requires `id`; optional
          `threshold` (default 0.5). Returns weak_sentences for review.
        - **`claim`** — find chunks supporting a free-text claim
          (`verify_claim`). Requires `text`; optional `against_chunk_ids`,
          `top_k`. *Deprecated* — prefer the `study` flow (define → assess →
          ledger) to evaluate a claim across chunks (richer, multi-agent,
          verifiable); this one-shot helper remains for quick checks.
        - **`consensus`** — semantic search across summaries; groups by
          target with status counts. Requires `query`; optional `top_k`.

        Examples::

            sid = summary("add", target_id="doc_a#p2#c3",
                          text="DPR uses dual BERT...", agent_id="writer")
            summary("verify", id=sid, verdict="verified",
                    verifier_agent_id="reviewer")
            summary("ground", id=sid, threshold=0.5)
            summary("claim", text="dense beats BM25 on QA",
                    against_chunk_ids=["doc_a#p2#c3"])
        """
        if action == "add":
            return corpus.add_summary(
                _require(target_id, "target_id", action, "summary"),
                _require(text, "text", action, "summary"),
                target_kind=target_kind, depth=depth,
                agent_id=_require(agent_id, "agent_id", action, "summary"),
                model=model, tags=tags or [],
            )
        if action == "verify":
            return corpus.verify_summary(
                _require(id, "id", action, "summary"),
                verdict=_require(verdict, "verdict", action, "summary"),
                verifier_agent_id=_require(
                    verifier_agent_id, "verifier_agent_id", action, "summary",
                ),
                notes=notes,
            )
        if action == "list":
            return corpus.get_summaries(
                _require(target_id, "target_id", action, "summary"),
                target_kind=target_kind if target_kind != "Chunk" else None,
                status=status, depth=depth if depth != "chunk" else None,
            )
        if action == "ground":
            return corpus.check_grounding(
                _require(id, "id", action, "summary"), threshold=threshold,
            )
        if action == "claim":
            return corpus.verify_claim(
                _require(text, "text", action, "summary"),
                against_chunk_ids=against_chunk_ids, top_k=top_k,
            )
        if action == "consensus":
            return corpus.find_consensus(
                _require(query, "query", action, "summary"), top_k=top_k,
            )
        raise ValueError(
            f"summary(): unknown action {action!r}. "
            "Valid: add, verify, list, ground, claim, consensus",
        )

    # ─── tag ──────────────────────────────────────────────────────────────

    @app.tool()
    def tag(
        action: str,
        chunk_id: str | None = None,
        name: str | None = None,
        agent_id: str | None = None,
        kind: str = "custom",
        confidence: float | None = None,
        doc_id: str | None = None,
        section_id: str | None = None,
        elements: list[str] | None = None,
        items: list[dict[str, Any]] | None = None,
        model: str = "",
        limit: int = 100,
    ) -> Any:
        """Tag + element-classification operations on chunks.

        Actions:

        - **`add`** — tag a chunk. Idempotent per (chunk, name, agent).
          Requires `chunk_id`, `name`, `agent_id`; optional `kind`
          (topic/entity/custom/review), `confidence` [0,1].
        - **`remove`** — remove the calling agent's tag application.
          Requires `chunk_id`, `name`, `agent_id`.
        - **`list`** — list tag applications. Optional `doc_id`, `chunk_id`,
          `agent_id`, `kind` filters.
        - **`chunks`** — find all chunks carrying a given tag. Requires
          `name`; optional `limit`.

        Element classification (multi-study routing — classify a corpus ONCE so
        many studies route to their chunks via `study(..., element=…)` instead of
        re-scanning; a schema pack must be loaded server-side, e.g. legal):
        - **`unclassified`** — pull the next ready-but-unclassified chunks (claim
          them with `agent_id`, like `study("next")`). Optional `doc_id`,
          `section_id`, `limit`.
        - **`classify`** — record one chunk's element types. Requires `chunk_id`,
          `elements` (a list of registered element ids, e.g. `["holding"]`),
          `agent_id`; optional `model`, `confidence`. Empty `elements` = "no
          element applies" → marks it unclassified-but-covered. Unknown element
          ids are rejected.
        - **`classify_many`** — batch: `items=[{chunk_id, elements, agent_id}, …]`.

        Examples::

            tag("add", chunk_id="doc_a#p1#c0", name="dense-retrieval",
                agent_id="me", kind="topic")
            for ch in tag("unclassified", agent_id="cls-1"):
                tag("classify", chunk_id=ch["id"], elements=["holding"], agent_id="cls-1")
        """
        if action == "unclassified":
            r = corpus.next_unclassified(
                doc_id=doc_id, section_id=section_id, agent_id=agent_id, limit=limit,
            )
            if agent_id:  # claiming mutates (writes a checkout) → persist
                _persist(corpus)
            return r
        if action == "classify":
            r = corpus.classify_chunk(
                _require(chunk_id, "chunk_id", action, "tag"),
                elements=_require(elements, "elements", action, "tag"),
                agent_id=_require(agent_id, "agent_id", action, "tag"),
                model=model, confidence=confidence,
            )
            _persist(corpus)
            return r
        if action == "classify_many":
            r = corpus.classify_many(_require(items, "items", action, "tag"))
            _persist(corpus)
            return r
        if action == "add":
            return corpus.tag_chunk(
                _require(chunk_id, "chunk_id", action, "tag"),
                _require(name, "name", action, "tag"),
                kind=kind,
                agent_id=_require(agent_id, "agent_id", action, "tag"),
                confidence=confidence,
            )
        if action == "remove":
            return corpus.untag_chunk(
                _require(chunk_id, "chunk_id", action, "tag"),
                _require(name, "name", action, "tag"),
                agent_id=_require(agent_id, "agent_id", action, "tag"),
            )
        if action == "list":
            return corpus.list_tags(
                doc_id=doc_id, chunk_id=chunk_id,
                agent_id=agent_id, kind=None if kind == "custom" else kind,
            )
        if action == "chunks":
            return corpus.chunks_by_tag(
                _require(name, "name", action, "tag"), limit=limit,
            )
        raise ValueError(
            f"tag(): unknown action {action!r}. Valid: add, remove, list, chunks, "
            "unclassified, classify, classify_many",
        )

    # ─── agent ────────────────────────────────────────────────────────────

    @app.tool()
    def agent(
        action: str,
        id: str | None = None,
        kind: str = "llm",
        model: str = "",
        role: str = "",
        system_prompt: str = "",
        tools: list[str] | None = None,
        context: dict[str, Any] | None = None,
        description: str = "",
        target_id: str | None = None,
        limit: int = 50,
    ) -> Any:
        """Agent identity + template operations.

        Actions:

        - **`upsert`** — register or update an Agent node with role,
          system_prompt, model, tools, context. Field-level merge:
          unspecified fields preserved. Requires `id`.
        - **`get`** — fetch full template + counters. Requires `id`.
          Returns `{id, kind, model, role, system_prompt, tools, context,
          description, first_seen, last_seen, action_count}` or `{}` if
          unknown.
        - **`list`** — list agents. Optional `role`, `kind` filters.
        - **`activity`** — what an agent has done (views, summaries,
          tags, translations, review/verification events). Requires `id`;
          optional `target_id` to scope.

        Example::

            agent("upsert", id="reviewer-strict",
                  role="reviewer", model="claude-sonnet-4-6",
                  system_prompt="You are a strict fact-checker...",
                  tools=["summary"], context={"strictness": "high"})
            cfg = agent("get", id="reviewer-strict")
            agent("activity", id="reviewer-strict", target_id="doc_a")
        """
        if action == "upsert":
            return corpus.upsert_agent(
                _require(id, "id", action, "agent"),
                kind=kind, model=model, role=role,
                system_prompt=system_prompt, tools=tools, context=context,
                description=description,
            )
        if action == "get":
            return corpus.get_agent(_require(id, "id", action, "agent"))
        if action == "list":
            return corpus.list_agents(
                role=role or None, kind=kind if kind != "llm" else None,
            )
        if action == "activity":
            return corpus.agent_activity(
                _require(id, "id", action, "agent"),
                target_id=target_id, limit=limit,
            )
        raise ValueError(
            f"agent(): unknown action {action!r}. Valid: upsert, get, list, activity",
        )

    # ─── review (kanban) ──────────────────────────────────────────────────

    @app.tool()
    def review(
        action: str,
        target_id: str | None = None,
        target_kind: str = "Chunk",
        ticket_id: str | None = None,
        agent_id: str | None = None,
        priority: int = 0,
        min_priority: int | None = None,
        note: str = "",
        notes: str = "",
        reason: str = "",
        enqueued_by: str = "system",
        doc_id: str | None = None,
        status_filter: str | None = "ready",
        status: str | None = None,
        verdict: str = "reviewed",
        accuracy: float | None = None,
        authenticity: str | None = None,
        tags: list[str] | None = None,
        with_target: bool = True,
        with_events: bool = True,
        limit: int = 50,
    ) -> Any:
        """Review kanban (write → review → done).

        Actions:

        - **`enqueue`** — add one target node to the queue. Requires
          `target_id`; optional `target_kind`, `priority`, `note`,
          `enqueued_by`. Returns ticket id.
        - **`enqueue_chunks`** — bulk-enqueue chunks (optionally one doc /
          one status). Skips chunks with open tickets.
        - **`claim_next`** — pull next ticket and claim it. Requires
          `agent_id`. Returns ticket with hydrated target, or `null`.
          The agent's main entry point.
        - **`claim`** — claim a specific ticket. Requires `ticket_id`,
          `agent_id`.
        - **`unclaim`** — release without verdict (ticket → `new`).
          Requires `ticket_id`, `agent_id`; optional `reason`.
        - **`complete`** — close ticket with verdict. Requires
          `ticket_id`, `agent_id`; optional `verdict` (reviewed/
          needs_revision/rejected), `accuracy` [0,1], `authenticity`,
          `notes`, `tags` (applied to target with `kind="review"`).
        - **`list`** — browse queue. Optional `status`, `target_kind`,
          `agent_id` (claimed-by), `limit`.
        - **`get`** — full ticket detail + audit trail. Requires
          `ticket_id`.
        - **`stats`** — counts by status + per-agent in-review load.

        Example agent loop::

            while True:
                t = review("claim_next", agent_id="me")
                if t is None: break
                review("complete", ticket_id=t["ticket_id"], agent_id="me",
                       verdict="reviewed", tags=["accurate"])
        """
        if action == "enqueue":
            return corpus.enqueue_review(
                _require(target_id, "target_id", action, "review"),
                target_kind=target_kind, priority=priority,
                note=note, enqueued_by=enqueued_by,
            )
        if action == "enqueue_chunks":
            return corpus.enqueue_chunks_for_review(
                doc_id=doc_id, status_filter=status_filter,
                priority=priority, enqueued_by=enqueued_by,
            )
        if action == "claim":
            return corpus.claim_review(
                _require(ticket_id, "ticket_id", action, "review"),
                agent_id=_require(agent_id, "agent_id", action, "review"),
            )
        if action == "claim_next":
            return corpus.claim_next_review(
                agent_id=_require(agent_id, "agent_id", action, "review"),
                target_kind=target_kind if target_kind != "Chunk" else None,
                min_priority=min_priority,
            )
        if action == "unclaim":
            return corpus.unclaim_review(
                _require(ticket_id, "ticket_id", action, "review"),
                agent_id=_require(agent_id, "agent_id", action, "review"),
                reason=reason,
            )
        if action == "complete":
            return corpus.complete_review(
                _require(ticket_id, "ticket_id", action, "review"),
                agent_id=_require(agent_id, "agent_id", action, "review"),
                verdict=verdict, accuracy=accuracy,
                authenticity=authenticity, notes=notes, tags=tags,
            )
        if action == "list":
            return corpus.list_review_queue(
                status=status,
                target_kind=target_kind if target_kind != "Chunk" else None,
                agent_id=agent_id, limit=limit,
            )
        if action == "get":
            return corpus.get_review_ticket(
                _require(ticket_id, "ticket_id", action, "review"),
                with_target=with_target, with_events=with_events,
            )
        if action == "stats":
            return corpus.review_stats()
        raise ValueError(
            f"review(): unknown action {action!r}. Valid: enqueue, "
            "enqueue_chunks, claim, claim_next, unclaim, complete, list, get, stats",
        )

    # ─── ocr ──────────────────────────────────────────────────────────────

    @app.tool()
    def ocr(
        action: str,
        doc_id: str | None = None,
        page_id: str | None = None,
        markdown: str | None = None,
        agent_id: str | None = None,
        limit: int = 20,
        include_images: bool = True,
        dpi: int = 200,
        model: str = "",
        confidence: float | None = None,
    ) -> Any:
        """OCR pipeline (for pages with empty text after parse).

        Actions:

        - **`status`** — per-document OCR coverage (ready vs. pending
          pages). Optional `doc_id`.
        - **`pending`** — pages flagged `needs_ocr`, each with a base64
          PNG render. Optional `doc_id`, `limit`, `include_images`, `dpi`.
        - **`submit`** — patch agent-supplied OCR markdown back into a
          page; re-chunks + re-embeds. Requires `page_id`, `markdown`,
          `agent_id`.

        Example::

            ocr("status")
            for p in ocr("pending", limit=5):
                # call vision LLM on p["image_b64"] → md
                ocr("submit", page_id=p["page_id"], markdown=md,
                    agent_id="ocr-bot")
        """
        if action == "status":
            return corpus.ocr_status(doc_id=doc_id)
        if action == "pending":
            return corpus.list_pending_ocr(
                doc_id=doc_id, limit=limit,
                include_images=include_images, dpi=dpi,
            )
        if action == "submit":
            return corpus.submit_ocr(
                _require(page_id, "page_id", action, "ocr"),
                _require(markdown, "markdown", action, "ocr"),
                agent_id=_require(agent_id, "agent_id", action, "ocr"),
                model=model, confidence=confidence,
            )
        raise ValueError(
            f"ocr(): unknown action {action!r}. Valid: status, pending, submit",
        )

    # ─── cluster ──────────────────────────────────────────────────────────

    @app.tool()
    def cluster(
        action: str,
        id: str | None = None,
        algorithm: str = "louvain",
        params: dict[str, Any] | None = None,
        note: str = "",
        top_terms: int = 10,
        out_path: str | None = None,
        format: str | None = None,
    ) -> Any:
        """Topic clustering on chunk embeddings.

        Actions:

        - **`run`** — run a clustering pass; writes Cluster nodes + edges.
          Optional `algorithm` (louvain/kmeans/dbscan), `params`, `note`.
        - **`get`** — inspect a cluster (members + top lexical terms).
          Requires `id`; optional `top_terms`.
        - **`list`** — all clusters with sizes + run ids.
        - **`export`** — write cluster (members + summaries) to MD/DOCX/PDF.
          Requires `id`, `out_path`; optional `format`.

        Example::

            cluster("run", algorithm="louvain")
            cluster("list")
            cluster("get", id="cluster_3", top_terms=15)
        """
        if action == "run":
            return corpus.cluster_chunks(
                algorithm=algorithm, params=params, note=note,
            )
        if action == "get":
            return corpus.get_cluster(
                _require(id, "id", action, "cluster"), top_terms=top_terms,
            )
        if action == "list":
            return corpus.cluster_overview()
        if action == "export":
            return str(corpus.export_cluster(
                _require(id, "id", action, "cluster"),
                _require(out_path, "out_path", action, "cluster"),
                format=format,
            ))
        raise ValueError(
            f"cluster(): unknown action {action!r}. Valid: run, get, list, export",
        )

    # ─── translate ────────────────────────────────────────────────────────

    @app.tool()
    def translate(
        action: str,
        chunk_id: str | None = None,
        doc_id: str | None = None,
        target_lang: str | None = None,
        text: str | None = None,
        agent_id: str | None = None,
        model: str = "",
        status: str = "draft",
        prefer_reviewed: bool = True,
    ) -> Any:
        """Translation operations (per-chunk).

        Actions:

        - **`add`** — store a translation. Requires `chunk_id`,
          `target_lang`, `text`, `agent_id`; optional `model`, `status`
          (draft/reviewed).
        - **`list`** — list translations on a chunk. Requires `chunk_id`;
          optional `target_lang`.
        - **`assemble`** — stitch a document's translated chunks together.
          Untranslated pages fall back to the original. Requires `doc_id`,
          `target_lang`; optional `prefer_reviewed`.

        Examples::

            translate("add", chunk_id="doc_a#p1#c0", target_lang="es",
                      text="...", agent_id="translator-1")
            translate("assemble", doc_id="doc_a", target_lang="es")
        """
        if action == "add":
            return corpus.add_translation(
                _require(chunk_id, "chunk_id", action, "translate"),
                _require(target_lang, "target_lang", action, "translate"),
                _require(text, "text", action, "translate"),
                agent_id=_require(agent_id, "agent_id", action, "translate"),
                model=model, status=status,
            )
        if action == "list":
            return corpus.get_translations(
                _require(chunk_id, "chunk_id", action, "translate"),
                target_lang=target_lang,
            )
        if action == "assemble":
            return corpus.assemble_translated_document(
                _require(doc_id, "doc_id", action, "translate"),
                target_lang=_require(target_lang, "target_lang", action, "translate"),
                prefer_reviewed=prefer_reviewed,
            )
        raise ValueError(
            f"translate(): unknown action {action!r}. Valid: add, list, assemble",
        )

    # ─── study (evidence analysis) ─────────────────────────────────────────

    @app.tool()
    def study(
        action: str,
        study_id: str | None = None,
        question: str | None = None,
        title: str | None = None,
        chunk_id: str | None = None,
        stance: str | None = None,
        weight: float | None = None,
        provenance: str | None = None,
        quote: str = "",
        char_start: int | None = None,
        char_end: int | None = None,
        rationale: str = "",
        rows: list[dict[str, Any]] | None = None,
        context_chunk_ids: list[str] | None = None,
        agent_id: str | None = None,
        model: str = "",
        assessment_id: str | None = None,
        finding_id: str | None = None,
        verdict: str | None = None,
        verifier_agent_id: str | None = None,
        notes: str = "",
        text: str | None = None,
        statement: str | None = None,
        supporting_chunk_ids: list[str] | None = None,
        finding_type: str = "",
        doc_id: str | None = None,
        section_id: str | None = None,
        element: str | None = None,
        min_weight: float | None = None,
        verified_only: bool = False,
        include_superseded: bool = False,
        status: str | None = None,
        created_by: str | None = None,
        embed: bool = False,
        acknowledge_no_synthesis: bool = False,
        limit: int = 200,
    ) -> Any:
        """Evidence study: judge chunks for/against a claim, rank, verify.

        Record per-chunk evidence toward a stored question, then retrieve it
        weight-ranked and have a second agent verify it. Each `assess` is a
        first-class, independently-verifiable record (multiple agents can
        assess the same chunk). **No embeddings needed** — agents iterate
        chunks via `next`, so you can skip `document('index')` entirely.

        Actions:

        - **`define`** — create a study. Requires `question`, `agent_id`
          (the creator); optional `title`. Returns the study id.
        - **`assess`** — record one assessment. Requires `study_id`,
          `chunk_id`, `stance` (`supports`/`against`/`neutral`/`deferred` —
          `deferred` = read but can't judge yet, e.g. an image/needs_ocr chunk;
          it's counted distinctly and stays in the work-list), `weight`
          (0..1 probative strength), `agent_id`; optional `rationale`,
          `model`, **`provenance`** (what you actually checked, surfaced per row
          in the ledger: `primary_text` = read the source [default],
          `characterization` = a paraphrase/summary, `scanned_unread` = an unread
          scan [provisional]), an optional **pinpoint span** — `quote` (the exact
          passage; located in the chunk) and/or `char_start`+`char_end`
          (validated against the chunk text; surfaced in the ledger for pinpoint
          cites), and **`context_chunk_ids`** — neighbor chunks you
          had to read to interpret this one (e.g. from `chunk("get", window=…)`).
          They're recorded so the ledger can pull the full span later and so
          they're excluded from the work-list (no one re-judges them).
          Append-only; never embeds.
        - **`assess_many`** — batch version: one call, one validated write, one
          persist. Requires `study_id` and `rows` — a list of dicts each with
          `chunk_id`/`stance`/`weight`/`agent_id` (+ the same optionals as
          `assess`). Prefer this for a fan-out of assessments. One bad row aborts
          the whole batch (nothing written).
        - **`supersede`** — audit-preserving correction: record a new assessment
          that replaces an existing one. Requires `assessment_id` (the old one)
          plus the new `stance`/`weight`/`agent_id` (and optional `rationale`/
          `model`/`provenance`); inherits the old one's study+chunk. The old
          assessment is kept but hidden from the ledger by default — use this
          (not a bare `assess`) to correct *another agent's* row to one current
          winner per chunk.
        - **`next`** — pull the next chunks to assess (in reading order).
          **Pass `agent_id` to claim them** — a punchcard checkout that stops
          parallel analysts from grabbing the same chunks; claims auto-expire
          (~30 min) and assessing releases. Without `agent_id` it's a
          read-only preview. Requires `study_id`; optional `doc_id`,
          `section_id` (scope to one section — see `document("sections")`),
          `element` (advisory: float chunks classified as that element type to
          the front — read your subset first without re-scanning; the full list
          is still returned, nothing hidden), `limit`. For a fan-out, give each
          analyst the same `study_id` and its own `agent_id` for disjoint batches.
        - **`ledger`** — weight-ranked evidence + support/against tallies.
          Requires `study_id`; optional `stance` (→ just the supporting or
          contradicting side), `min_weight`, `verified_only`, `doc_id` (scope to
          one document), `section_id` (scope to one section), `element` (advisory:
          float that element's rows first + attach a `scope_coverage` block
          showing what was deprioritized), `include_superseded` (default false —
          corrected
          assessments are hidden; pass true for the full history), `limit`.
          Reports `total`/`returned` so truncation is visible (`total > returned`
          ⇒ raise `limit` to see the rest).
        - **`conflicts`** — the contested evidence: chunks with both a current
          `supports` and `against` assessment, each with its opposing rows (most
          contested first). Requires `study_id`. Read these first — that's where
          the disagreement is.
        - **`finding`** — record a **cross-chunk** pattern (what per-chunk
          `assess` can't see: disparate treatment, conflicting rulings, …).
          Requires `study_id`, `statement`, `supporting_chunk_ids=[…]` (the real
          chunks it rests on), `stance`, `weight`, `agent_id`; optional
          `finding_type` (free-text, becomes a routing label), `provenance`,
          `rationale`. Findings are a separate collection from the per-chunk
          assessment ledger.
        - **`findings`** — list a study's cross-chunk findings (weight-ranked,
          each with its supporting chunks **and** the reviewer-agreement rollup:
          reviewer_count, vote_tally, agreement, confidence, escalation_state).
          Requires `study_id`; optional `finding_type` filter.
        - **`verify`** — a second agent grades an assessment **or a finding** —
          the independent vote confidence is built from. Pass `assessment_id`
          (per-chunk) **or** `finding_id` (cross-chunk), plus `verdict`
          (`verified`/`disputed`/`duplicate` — `duplicate` = "same as another")
          and `verifier_agent_id`; optional `provenance` (what the verifier
          checked, recorded on the event). Verifying a finding recomputes its
          escalation_state (settled / contested / needs_more). Self-verification
          is rejected.
        - **`synthesize`** — mark the cross-chunk **synthesis pass** as run (the
          second altitude above per-chunk assess: hunt disparate treatment,
          contradictions, omissions, aggregations — record each as a `finding`).
          Requires `study_id`, `agent_id`; optional `notes`. Read
          `study("synthesis_prompt")` first for what to hunt. **Clears the
          conclude gate.**
        - **`synthesis_prompt`** — the prompt to read before synthesizing (the
          domain-neutral hunt list + any registered domain addenda).
        - **`conclude`** — write the study's conclusion (stored as a
          verifiable summary on the study). Requires `study_id`, `text`,
          `agent_id`; optional `embed`. **Refuses unless the study has been
          synthesized** — pass `acknowledge_no_synthesis=true` to record an
          audited skip (the skip is never silent).
        - **`list`** — studies that have been run. Optional `status`
          (`open`/`closed`), `created_by`.
        - **`get`** — one study: metadata + tallies + conclusions. Requires
          `study_id`.
        - **`reopen`** — flip a closed study back to open. Requires
          `study_id`, `agent_id`.
        - **`delete`** — destructive cascade: removes the study + all its
          assessments, verification events, and conclusions. Requires
          `study_id`.

        Happy path (no embeddings, no convention to invent)::

            sid = study("define", question="X is necessary", agent_id="lead")
            for ch in study("next", study_id=sid, doc_id="doc_a"):
                study("assess", study_id=sid, chunk_id=ch["id"],
                      stance="supports", weight=0.8, rationale="...",
                      agent_id="reader-1")
            study("ledger", study_id=sid)                  # ranked evidence
            study("ledger", study_id=sid, stance="supports")  # just the supporting side
            study("verify", assessment_id=..., verdict="verified",
                  verifier_agent_id="checker")
            study("conclude", study_id=sid, text="...", agent_id="lead")
        """
        if action == "define":
            return corpus.define_study(
                _require(question, "question", action, "study"),
                created_by=_require(agent_id, "agent_id", action, "study"),
                title=title, status=status or "open",
            )
        if action == "assess":
            r = corpus.assess(
                _require(study_id, "study_id", action, "study"),
                _require(chunk_id, "chunk_id", action, "study"),
                stance=_require(stance, "stance", action, "study"),
                weight=_require(weight, "weight", action, "study"),
                rationale=rationale,
                agent_id=_require(agent_id, "agent_id", action, "study"),
                model=model, provenance=provenance or "primary_text",
                quote=quote, char_start=char_start, char_end=char_end,
                context_chunk_ids=context_chunk_ids,
            )
            _persist(corpus)
            return r
        if action == "assess_many":
            r = corpus.assess_many(
                _require(study_id, "study_id", action, "study"),
                _require(rows, "rows", action, "study"),
            )
            _persist(corpus)
            return r
        if action == "supersede":
            r = corpus.supersede_assessment(
                _require(assessment_id, "assessment_id", action, "study"),
                stance=_require(stance, "stance", action, "study"),
                weight=_require(weight, "weight", action, "study"),
                agent_id=_require(agent_id, "agent_id", action, "study"),
                rationale=rationale, model=model,
                provenance=provenance or "primary_text",
                context_chunk_ids=context_chunk_ids,
            )
            _persist(corpus)
            return r
        if action == "next":
            r = corpus.next_unassessed(
                _require(study_id, "study_id", action, "study"),
                doc_id=doc_id, section_id=section_id, element=element,
                agent_id=agent_id, limit=limit,
            )
            if agent_id:  # claiming next mutates (writes a checkout) → persist
                _persist(corpus)
            return r
        if action == "ledger":
            return corpus.study_ledger(
                _require(study_id, "study_id", action, "study"),
                stance=stance, min_weight=min_weight,
                verified_only=verified_only, doc_id=doc_id, section_id=section_id,
                element=element, include_superseded=include_superseded, limit=limit,
            )
        if action == "verify":
            # One verb, two targets: a per-chunk Assessment (assessment_id) or a
            # cross-chunk Finding (finding_id) — the latter is the vote confidence
            # is built from.
            if finding_id is not None:
                r = corpus.verify_finding(
                    finding_id,
                    verdict=_require(verdict, "verdict", action, "study"),
                    verifier_agent_id=_require(
                        verifier_agent_id, "verifier_agent_id", action, "study",
                    ),
                    notes=notes, provenance=provenance,
                )
            else:
                r = corpus.verify_assessment(
                    _require(assessment_id, "assessment_id", action, "study"),
                    verdict=_require(verdict, "verdict", action, "study"),
                    verifier_agent_id=_require(
                        verifier_agent_id, "verifier_agent_id", action, "study",
                    ),
                    notes=notes, provenance=provenance,
                )
            _persist(corpus)
            return r
        if action == "synthesize":
            r = corpus.synthesize_study(
                _require(study_id, "study_id", action, "study"),
                agent_id=_require(agent_id, "agent_id", action, "study"),
                note=notes,
            )
            _persist(corpus)
            return r
        if action == "synthesis_prompt":
            return {"prompt": corpus.synthesis_prompt()}
        if action == "conclude":
            r = corpus.conclude_study(
                _require(study_id, "study_id", action, "study"),
                _require(text, "text", action, "study"),
                agent_id=_require(agent_id, "agent_id", action, "study"),
                model=model, embed=embed,
                acknowledge_no_synthesis=acknowledge_no_synthesis,
            )
            _persist(corpus)
            return r
        if action == "conflicts":
            return corpus.study_conflicts(_require(study_id, "study_id", action, "study"))
        if action == "finding":
            r = corpus.create_finding(
                _require(study_id, "study_id", action, "study"),
                statement=_require(statement, "statement", action, "study"),
                supporting_chunk_ids=_require(
                    supporting_chunk_ids, "supporting_chunk_ids", action, "study"),
                stance=_require(stance, "stance", action, "study"),
                weight=_require(weight, "weight", action, "study"),
                agent_id=_require(agent_id, "agent_id", action, "study"),
                finding_type=finding_type, provenance=provenance or "primary_text",
                rationale=rationale, model=model,
            )
            _persist(corpus)
            return r
        if action == "findings":
            return corpus.list_findings(
                _require(study_id, "study_id", action, "study"),
                finding_type=finding_type or None,
            )
        if action == "list":
            return corpus.list_studies(status=status, created_by=created_by)
        if action == "get":
            return corpus.get_study(_require(study_id, "study_id", action, "study"))
        if action == "reopen":
            r = corpus.reopen_study(
                _require(study_id, "study_id", action, "study"),
                agent_id=_require(agent_id, "agent_id", action, "study"),
            )
            _persist(corpus)
            return r
        if action == "delete":
            r = corpus.delete_study(_require(study_id, "study_id", action, "study"))
            _persist(corpus)
            return r
        raise ValueError(
            f"study(): unknown action {action!r}. Valid: define, assess, assess_many, "
            "supersede, next, ledger, conflicts, finding, findings, verify, "
            "synthesize, synthesis_prompt, conclude, list, get, reopen, delete",
        )
