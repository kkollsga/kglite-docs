"""Typed MCP tool registrations.

Each tool is a thin shim over the Corpus method of the same name.
Argument names and shapes are the agent-facing contract — keep them
stable across versions.
"""

from __future__ import annotations

from typing import Any


def register_typed_tools(app: Any, corpus: Any) -> None:
    """Register the kglite-docs tools on a FastMCP app."""

    @app.tool()
    def list_documents(filters: dict[str, Any] | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """List ingested documents with metadata. Optional `filters` is an
        equality map (e.g. `{"lang": "en"}`)."""
        return corpus.list_documents(filters=filters, limit=limit)

    @app.tool()
    def get_document(doc_id: str) -> dict[str, Any] | None:
        """Fetch one document's metadata + a heading-derived TOC."""
        return corpus.get_document(doc_id)

    @app.tool()
    def search(
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        with_summaries: bool = False,
        agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic chunk search across the whole corpus.

        Embeds `query` with the same bge-m3 model the corpus was indexed
        under, then returns the top-`top_k` chunks by cosine similarity.

        - `filters`: equality predicates on chunk fields, e.g.
          `{"doc_id": "doc_abc..."}` or `{"page": 5}`. Combine multiple
          keys for AND.
        - `with_summaries=True`: inline verified summaries on each hit.
        - `agent_id`: record a `View` on each hit so attention is
          queryable later. Skip when running speculative searches.

        Returns a list of `{id, score, text, doc_id, page, status, ...}`
        ordered by score descending. Score 1.0 = identical embedding;
        ~0.7 is "strong match" for bge-m3 cosine."""
        return corpus.search(
            query, top_k=top_k, filters=filters,
            with_summaries=with_summaries, agent_id=agent_id,
        )

    @app.tool()
    def get_chunk(
        chunk_id: str,
        with_neighbors: bool = False,
        with_summaries: bool = False,
        agent_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Fetch one chunk with optional prev/next ids and inlined summaries."""
        return corpus.get_chunk(
            chunk_id, with_neighbors=with_neighbors,
            with_summaries=with_summaries, agent_id=agent_id,
        )

    @app.tool()
    def similar_chunks(chunk_id: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Nearest-neighbor chunks for a given chunk_id."""
        return corpus.similar_chunks(chunk_id, top_k=top_k)

    @app.tool()
    def compose_context(
        query: str,
        max_tokens: int = 4000,
        per_doc_cap: int | None = None,
        include_summaries: bool = True,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Build a prompt-ready, token-budgeted context bundle.

        Runs `search`, packs results greedy until `max_tokens` is hit
        (bge-m3 tokenizer count), and returns
        `{query, budget_tokens, used_tokens, items}`.

        Use this *instead of* calling `search` then concatenating
        chunks yourself — it inlines verified summaries when present
        (so the agent sees pre-checked context), respects the budget,
        and applies optional `per_doc_cap` to keep one document from
        crowding out the bundle."""
        return corpus.compose_context(
            query, max_tokens=max_tokens, per_doc_cap=per_doc_cap,
            include_summaries=include_summaries, agent_id=agent_id,
        )

    @app.tool()
    def add_summary(
        target_id: str,
        text: str,
        agent_id: str,
        target_kind: str = "Chunk",
        depth: str = "chunk",
        model: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """Write a Summary on a Chunk / Page / Document.

        Status starts as `unverified`. A *different* agent must call
        `verify_summary(summary_id, verdict, verifier_agent_id=...)` to
        flip it to `verified` / `disputed` / `needs_revision` — the
        server rejects self-verification.

        `depth` is one of `chunk`, `section`, or `document`. Optional
        `tags` get applied to the *target* node via `tag_chunk`.

        Returns the summary id (uuid). Persist it if you need to refer
        back, e.g. for a later `verify_summary` call."""
        return corpus.add_summary(
            target_id, text, target_kind=target_kind, depth=depth,
            agent_id=agent_id, model=model, tags=tags or [],
        )

    @app.tool()
    def verify_summary(
        summary_id: str,
        verdict: str,
        verifier_agent_id: str,
        notes: str = "",
    ) -> dict[str, Any]:
        """Apply a verification verdict to a summary. Verifier must
        differ from the author."""
        return corpus.verify_summary(
            summary_id, verdict=verdict, verifier_agent_id=verifier_agent_id, notes=notes,
        )

    @app.tool()
    def link_verification(verifier_summary_id: str, target_summary_id: str) -> dict[str, Any]:
        """Record that one summary verifies / disputes another."""
        return corpus.link_verification(verifier_summary_id, target_summary_id)

    @app.tool()
    def get_summaries(
        target_id: str,
        target_kind: str | None = None,
        status: str | None = None,
        depth: str | None = None,
    ) -> list[dict[str, Any]]:
        """List summaries on a target, filterable by status/depth."""
        return corpus.get_summaries(
            target_id, target_kind=target_kind, status=status, depth=depth,
        )

    @app.tool()
    def find_consensus(query: str, top_k: int = 20) -> list[dict[str, Any]]:
        """Semantic search across summaries; groups by target with status counts."""
        return corpus.find_consensus(query, top_k=top_k)

    @app.tool()
    def tag_chunk(
        chunk_id: str, tag_name: str, agent_id: str,
        kind: str = "custom", confidence: float | None = None,
    ) -> dict[str, Any]:
        """Tag a chunk. Idempotent per (chunk, tag, agent)."""
        return corpus.tag_chunk(
            chunk_id, tag_name, kind=kind, agent_id=agent_id, confidence=confidence,
        )

    @app.tool()
    def untag_chunk(chunk_id: str, tag_name: str, agent_id: str) -> dict[str, Any]:
        """Remove the calling agent's application of a tag from a chunk."""
        return corpus.untag_chunk(chunk_id, tag_name, agent_id=agent_id)

    @app.tool()
    def list_tags(
        doc_id: str | None = None, chunk_id: str | None = None,
        agent_id: str | None = None, kind: str | None = None,
    ) -> list[dict[str, Any]]:
        """List tag applications, filterable by doc/chunk/agent/kind."""
        return corpus.list_tags(
            doc_id=doc_id, chunk_id=chunk_id, agent_id=agent_id, kind=kind,
        )

    @app.tool()
    def chunks_by_tag(tag_name: str, limit: int = 100) -> list[dict[str, Any]]:
        """Find all chunks carrying a given tag."""
        return corpus.chunks_by_tag(tag_name, limit=limit)

    @app.tool()
    def register_agent(agent_id: str, kind: str = "llm", model: str = "") -> dict[str, Any]:
        """Explicit agent registration. Most write paths auto-register
        on first use; this tool exists for setting `kind`/`model`."""
        return corpus.register_agent(agent_id, kind=kind, model=model)

    @app.tool()
    def list_agents() -> list[dict[str, Any]]:
        """List registered agents with last-seen + action counts."""
        return corpus.list_agents()

    @app.tool()
    def record_view(chunk_id: str, agent_id: str, context: str = "") -> dict[str, Any]:
        """Record an agent viewing a chunk (explicit; search/get_chunk
        also do this when given `agent_id`)."""
        return corpus.record_view(chunk_id, agent_id, context=context)

    @app.tool()
    def ocr_status(doc_id: str | None = None) -> dict[str, Any]:
        """Per-document OCR coverage + corpus-wide totals (ready vs.
        pending pages). Pass `doc_id` to scope to one document."""
        return corpus.ocr_status(doc_id=doc_id)

    @app.tool()
    def list_pending_ocr(
        doc_id: str | None = None, limit: int = 20,
        include_images: bool = True, dpi: int = 200,
    ) -> list[dict[str, Any]]:
        """Pages flagged `needs_ocr=True`, each with a base64 PNG render."""
        return corpus.list_pending_ocr(
            doc_id=doc_id, limit=limit, include_images=include_images, dpi=dpi,
        )

    @app.tool()
    def submit_ocr(
        page_id: str, markdown: str, agent_id: str,
        model: str = "", confidence: float | None = None,
    ) -> dict[str, Any]:
        """Patch agent-supplied OCR markdown back into a page; re-chunks + re-embeds."""
        return corpus.submit_ocr(
            page_id, markdown, agent_id=agent_id, model=model, confidence=confidence,
        )

    @app.tool()
    def cluster_chunks(
        algorithm: str = "louvain", params: dict[str, Any] | None = None, note: str = "",
    ) -> dict[str, Any]:
        """Run a clustering pass over the chunk embeddings. Writes
        Cluster nodes + IN_CLUSTER edges."""
        return corpus.cluster_chunks(algorithm=algorithm, params=params, note=note)

    @app.tool()
    def get_cluster(cluster_id: str, top_terms: int = 10) -> dict[str, Any] | None:
        """Inspect a cluster: members + top lexical terms."""
        return corpus.get_cluster(cluster_id, top_terms=top_terms)

    @app.tool()
    def cluster_overview() -> list[dict[str, Any]]:
        """List all clusters with sizes and run ids."""
        return corpus.cluster_overview()

    @app.tool()
    def check_grounding(summary_id: str, threshold: float = 0.5) -> dict[str, Any]:
        """Score how well each sentence in a summary is supported by
        its source chunk(s) — a hallucination guard.

        Splits the summary on `.!?`, embeds each sentence, computes
        cosine similarity to the source chunk's embedding, and flags
        any sentence whose best match is below `threshold` (default 0.5)
        as `supported=False` in the returned `weak_sentences` list.

        Returns `{summary_id, sentences[], supported_fraction,
        grounding_score, weak_sentences[], threshold}`.

        Cheap baseline (cosine, not NLI) — surface weak claims for
        human/agent review rather than treating it as ground truth.
        Pair with `verify_claim(claim_text, ...)` for free-text claims."""
        return corpus.check_grounding(summary_id, threshold=threshold)

    @app.tool()
    def verify_claim(
        claim_text: str, against_chunk_ids: list[str] | None = None, top_k: int = 5,
    ) -> dict[str, Any]:
        """Find the chunks that best support a free-text claim."""
        return corpus.verify_claim(
            claim_text, against_chunk_ids=against_chunk_ids, top_k=top_k,
        )

    @app.tool()
    def add_translation(
        chunk_id: str, target_lang: str, text: str, agent_id: str,
        model: str = "", status: str = "draft",
    ) -> str:
        """Store a translation of a chunk into `target_lang`."""
        return corpus.add_translation(
            chunk_id, target_lang, text, agent_id=agent_id, model=model, status=status,
        )

    @app.tool()
    def get_translations(chunk_id: str, target_lang: str | None = None) -> list[dict[str, Any]]:
        """List translations on a chunk, optionally filtered by language."""
        return corpus.get_translations(chunk_id, target_lang=target_lang)

    @app.tool()
    def assemble_translated_document(
        doc_id: str, target_lang: str, prefer_reviewed: bool = True,
    ) -> dict[str, Any]:
        """Stitch a document's translated chunks back together. Untranslated
        pages fall back to the original."""
        return corpus.assemble_translated_document(
            doc_id, target_lang=target_lang, prefer_reviewed=prefer_reviewed,
        )

    @app.tool()
    def export_document(
        doc_id: str, out_path: str, format: str | None = None,
        include_summaries: bool = False,
    ) -> str:
        """Export a document to MD / DOCX / PDF. Returns the written path."""
        p = corpus.export_document(
            doc_id, out_path, format=format, include_summaries=include_summaries,
        )
        return str(p)

    @app.tool()
    def export_cluster(cluster_id: str, out_path: str, format: str | None = None) -> str:
        """Export a cluster (members + summaries) to MD / DOCX / PDF."""
        return str(corpus.export_cluster(cluster_id, out_path, format=format))

    # ─── review queue (kanban) ────────────────────────────────────────────

    @app.tool()
    def enqueue_review(
        target_id: str, target_kind: str = "Chunk",
        priority: int = 0, note: str = "", enqueued_by: str = "system",
    ) -> str:
        """Add a target node (Chunk/Summary/Document/Page) to the review
        queue. Returns the ticket id."""
        return corpus.enqueue_review(
            target_id, target_kind=target_kind, priority=priority,
            note=note, enqueued_by=enqueued_by,
        )

    @app.tool()
    def enqueue_chunks_for_review(
        doc_id: str | None = None, status_filter: str | None = "ready",
        priority: int = 0, enqueued_by: str = "system",
    ) -> dict[str, Any]:
        """Bulk-enqueue all chunks (optionally one doc / one status).
        Skips chunks that already have an open ticket."""
        return corpus.enqueue_chunks_for_review(
            doc_id=doc_id, status_filter=status_filter,
            priority=priority, enqueued_by=enqueued_by,
        )

    @app.tool()
    def claim_review(ticket_id: str, agent_id: str) -> dict[str, Any]:
        """Claim a specific ticket atomically. Fails if not in `new`."""
        return corpus.claim_review(ticket_id, agent_id=agent_id)

    @app.tool()
    def claim_next_review(
        agent_id: str, target_kind: str | None = None,
        min_priority: int | None = None,
    ) -> dict[str, Any] | None:
        """Pull the next ticket off the review queue and claim it for
        this agent atomically.

        Highest `priority` first, then oldest `created_at`. Returns
        `{ticket_id, status: "in_review", claimed_by, target: {...}, ...}`
        with the target node hydrated (full chunk text + page +
        headings, ready to inspect), or `null` if the queue is empty.

        This is the agent's main entry point — typical loop:

            while True:
                t = claim_next_review(agent_id="me")
                if t is None: break
                # inspect t.target, call verify_claim / check_grounding,
                # then complete_review(t.ticket_id, ..., verdict=..., tags=...)

        Within a single MCP-server process, two callers can't claim the
        same ticket (process-local lock)."""
        return corpus.claim_next_review(
            agent_id=agent_id, target_kind=target_kind, min_priority=min_priority,
        )

    @app.tool()
    def unclaim_review(ticket_id: str, agent_id: str, reason: str = "") -> dict[str, Any]:
        """Release a claim without a verdict — ticket returns to `new`."""
        return corpus.unclaim_review(ticket_id, agent_id=agent_id, reason=reason)

    @app.tool()
    def complete_review(
        ticket_id: str, agent_id: str,
        verdict: str = "reviewed",
        accuracy: float | None = None,
        authenticity: str | None = None,
        notes: str = "",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Close a review ticket with a verdict and any review metadata.

        `verdict`: one of `reviewed` (passed), `needs_revision`
        (author should revisit), or `rejected` (content is wrong).
        Only the agent currently holding the ticket can complete it —
        `ReviewConflict` otherwise.

        `accuracy` ∈ [0, 1] is your subjective confidence the content
        is correct; `authenticity` is a free-text or enum verdict
        (`verified` / `disputed` / etc.). `notes` becomes part of the
        immutable audit trail. `tags` are applied to the target chunk
        with `kind="review"` so they're distinguishable from topical
        tags later (`list_tags(kind="review")`)."""
        return corpus.complete_review(
            ticket_id, agent_id=agent_id, verdict=verdict,
            accuracy=accuracy, authenticity=authenticity, notes=notes, tags=tags,
        )

    @app.tool()
    def list_review_queue(
        status: str | None = None, target_kind: str | None = None,
        agent_id: str | None = None, limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Browse the review board with filters: status (new/in_review/
        reviewed/needs_revision/rejected), target_kind, claimed-by-agent."""
        return corpus.list_review_queue(
            status=status, target_kind=target_kind,
            agent_id=agent_id, limit=limit,
        )

    @app.tool()
    def get_review_ticket(
        ticket_id: str, with_target: bool = True, with_events: bool = True,
    ) -> dict[str, Any] | None:
        """Full ticket detail with hydrated target and full event audit trail."""
        return corpus.get_review_ticket(
            ticket_id, with_target=with_target, with_events=with_events,
        )

    @app.tool()
    def review_stats() -> dict[str, Any]:
        """Kanban summary: per-status counts + per-agent in-review load."""
        return corpus.review_stats()
