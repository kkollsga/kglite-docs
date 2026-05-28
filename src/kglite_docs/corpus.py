"""Public façade — the only object most users need.

`Corpus.create(path)` / `Corpus.open(path)` open a kglite-docs knowledge
base. Method names mirror what the MCP server exposes so library users
and agent users see the same vocabulary.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from pathlib import Path
from types import TracebackType
from typing import Any

from kglite_docs import cluster as cluster_mod
from kglite_docs import context as context_mod
from kglite_docs import enrich as enrich_mod
from kglite_docs import export as export_mod
from kglite_docs import ocr as ocr_mod
from kglite_docs import quality as quality_mod
from kglite_docs import review as review_mod
from kglite_docs import translate as translate_mod
from kglite_docs.activity import (
    list_agents as _list_agents,
)
from kglite_docs.activity import (
    record_view as _record_view,
)
from kglite_docs.activity import (
    register_agent as _register_agent,
)
from kglite_docs.embed import make_embedder
from kglite_docs.ingest.pipeline import IngestResult
from kglite_docs.ingest.pipeline import ingest_document as _ingest_doc
from kglite_docs.schema import CHUNK, CHUNK_TEXT_COL
from kglite_docs.store import Store
from kglite_docs.tagging import (
    chunks_by_tag as _chunks_by_tag,
)
from kglite_docs.tagging import (
    list_tags as _list_tags,
)
from kglite_docs.tagging import (
    tag_chunk as _tag_chunk,
)
from kglite_docs.tagging import (
    untag_chunk as _untag_chunk,
)
from kglite_docs.types import (
    AgentKind,
    AgentRow,
    ChunkDetail,
    ClusterAlgorithm,
    ComposedContext,
    DocumentDetail,
    DocumentRow,
    ExportFormat,
    GroundingReport,
    OcrStatus,
    PendingOcrRow,
    ReviewStats,
    ReviewStatus,
    ReviewTicketDetail,
    ReviewTicketRow,
    ReviewVerdict,
    SearchHit,
    SummaryDepth,
    SummaryRow,
    SummaryStatus,
    SummaryVerdict,
    TagKind,
    TagRow,
    TargetKind,
    TranslationStatus,
)


class Corpus:
    """The PDF knowledge base. Light wrapper over `Store` + an embedder."""

    def __init__(self, store: Store, embedder: Any | None = None) -> None:
        self._store = store
        self._embedder = embedder or make_embedder()

    # ─── construction ──────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        path: str | Path | None = None,
        *,
        embedder: Any | None = None,
    ) -> Corpus:
        return cls(Store.create(path), embedder=embedder)

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        embedder: Any | None = None,
    ) -> Corpus:
        return cls(Store.open(path), embedder=embedder)

    def save(self, path: str | Path | None = None) -> None:
        self._store.save(path)

    def close(self) -> None:
        """Persist + drop in-process state. Mostly useful at the tail of a
        `with` block (which calls this automatically) or to release the
        embedder's ONNX session early on long-lived processes."""
        if self._store.path is not None:
            with contextlib.suppress(Exception):
                self.save()
        unload = getattr(self._embedder, "unload", None)
        if callable(unload):
            with contextlib.suppress(Exception):
                unload()

    def __enter__(self) -> Corpus:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Save on clean exit; skip the save if an exception is propagating
        so we don't persist a partial mutation."""
        if exc_type is None and self._store.path is not None:
            with contextlib.suppress(Exception):
                self.save()

    @property
    def store(self) -> Store:
        return self._store

    @property
    def embedder(self) -> Any:
        return self._embedder

    # ─── ingestion ─────────────────────────────────────────────────────────

    def ingest(
        self,
        path: str | Path | None = None,
        *,
        text: str | None = None,
        title: str | None = None,
        source_uri: str | None = None,
        metadata: dict[str, object] | None = None,
        format: str | None = None,
    ) -> IngestResult:
        """Ingest a document. Three modes:

        - ``ingest("paper.pdf")`` — file path; format auto-detected from
          the extension. Pass ``format=`` to override.
        - ``ingest(text="# Notes\\n…", title="my-notes")`` — raw text /
          markdown. Useful for agent-generated synthesis articles.
        - ``ingest("doc.bin", format="md")`` — file path with explicit
          format hint when the extension doesn't match.

        Returns an :class:`IngestResult` with the assigned ``doc_id``
        (sha256 of file or text bytes), chunk count, and OCR-pending
        page count.

        Raises :class:`UnsupportedFormatError` for unknown formats and
        :class:`IngestError` for parse failures.
        """
        if path is None and text is None:
            raise ValueError("ingest(): pass either path= or text=")
        if path is not None and text is not None:
            raise ValueError("ingest(): pass path= or text=, not both")
        if text is not None:
            if not title:
                raise ValueError("ingest(text=…): title= is required for text mode")
            import tempfile
            fmt = (format or "md").lower()
            suffix = "." + fmt.lstrip(".")
            with tempfile.NamedTemporaryFile(
                "w", suffix=suffix, delete=False, encoding="utf-8",
            ) as f:
                f.write(text)
                tmp_path = Path(f.name)
            try:
                return _ingest_doc(
                    self._store, self._embedder, tmp_path,
                    title=title, source_uri=source_uri or "",
                    metadata=metadata, format=fmt,
                )
            finally:
                tmp_path.unlink(missing_ok=True)
        return _ingest_doc(
            self._store, self._embedder, path,  # type: ignore[arg-type]
            title=title, source_uri=source_uri, metadata=metadata, format=format,
        )

    def ingest_dir(
        self,
        directory: str | Path,
        *,
        recursive: bool = True,
        patterns: list[str] | None = None,
    ) -> list[IngestResult]:
        """Ingest every supported file under ``directory``. By default
        scans for all known formats: PDF, DOCX, PPTX, MD, HTML, TXT, and
        common image formats."""
        from kglite_docs.ingest.formats import SUPPORTED_FORMATS
        directory = Path(directory)
        if patterns is None:
            patterns = [f"*.{ext}" for ext in SUPPORTED_FORMATS]
        finder = directory.rglob if recursive else directory.glob
        seen: set[Path] = set()
        results: list[IngestResult] = []
        for pat in patterns:
            for f in sorted(finder(pat)):
                if f in seen or not f.is_file():
                    continue
                seen.add(f)
                try:
                    results.append(self.ingest(f))
                except Exception as exc:
                    import logging
                    logging.getLogger("kglite_docs").warning(
                        "ingest failed for %s: %s", f, exc,
                    )
        return results

    # ─── documents ────────────────────────────────────────────────────────

    def list_documents(
        self,
        *,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[DocumentRow]:
        where_clause = ""
        params: dict[str, Any] = {}
        if filters:
            preds = []
            for k, v in filters.items():
                preds.append(f"d.{k} = ${k}")
                params[k] = v
            where_clause = "WHERE " + " AND ".join(preds)
        df = self._store.cypher(
            f"""
            MATCH (d:Document)
            {where_clause}
            OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
            RETURN d.id AS id, d.title AS title, d.page_count AS pages,
                   d.ingested_at AS ingested_at, d.byte_size AS bytes,
                   count(c) AS chunk_count
            ORDER BY d.ingested_at DESC
            LIMIT {int(limit)}
            """,
            params=params,
        )
        return _df_to_dicts(df)

    def get_document(self, doc_id: str) -> DocumentDetail | None:
        df = self._store.cypher(
            "MATCH (d:Document {id: $id}) RETURN d.id AS id, d.title AS title, "
            "d.page_count AS pages, d.ingested_at AS ingested_at, d.byte_size AS bytes, "
            "d.path AS path, d.metadata_json AS metadata_json",
            params={"id": doc_id},
        )
        rows = _df_to_dicts(df)
        if not rows:
            return None
        doc = rows[0]
        # Attach a table of contents from page headings
        toc_df = self._store.cypher(
            "MATCH (d:Document {id: $id})-[:HAS_CHUNK]->(c:Chunk) "
            "RETURN DISTINCT c.headings_json AS headings, c.page_number AS page "
            "ORDER BY page",
            params={"id": doc_id},
        )
        doc["toc"] = _df_to_dicts(toc_df)
        return doc

    # ─── chunks ───────────────────────────────────────────────────────────

    def get_chunk(
        self,
        chunk_id: str,
        *,
        with_neighbors: bool = False,
        with_summaries: bool = False,
        agent_id: str | None = None,
    ) -> ChunkDetail | None:
        df = self._store.cypher(
            f"""
            MATCH (c:Chunk {{id: $id}})
            RETURN c.id AS id, c.title AS title, c.doc_id AS doc_id,
                   c.page_number AS page, c.chunk_index AS chunk_index,
                   c.{CHUNK_TEXT_COL} AS text, c.token_count AS token_count,
                   c.headings_json AS headings, c.status AS status,
                   c.view_count AS view_count
            """,
            params={"id": chunk_id},
        )
        rows = _df_to_dicts(df)
        if not rows:
            return None
        chunk = rows[0]
        if with_neighbors:
            n_df = self._store.cypher(
                "MATCH (c:Chunk {id: $id})-[:NEXT_CHUNK]->(n:Chunk) RETURN n.id AS id",
                params={"id": chunk_id},
            )
            p_df = self._store.cypher(
                "MATCH (p:Chunk)-[:NEXT_CHUNK]->(c:Chunk {id: $id}) RETURN p.id AS id",
                params={"id": chunk_id},
            )
            chunk["next_id"] = (_df_to_dicts(n_df) or [{}])[0].get("id")
            chunk["prev_id"] = (_df_to_dicts(p_df) or [{}])[0].get("id")
        if with_summaries:
            chunk["summaries"] = self.get_summaries(chunk_id, target_kind=CHUNK)
        if agent_id:
            _record_view(self._store, agent_id=agent_id, target_id=chunk_id, target_kind=CHUNK, context="get_chunk")
        return chunk

    # ─── search ───────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        agent_id: str | None = None,
        with_summaries: bool = False,
    ) -> list[SearchHit]:
        q_vec = self._embedder.embed([query])[0]
        hits = self._store.vector_search(
            CHUNK, CHUNK_TEXT_COL, q_vec, top_k=top_k, filters=filters
        )
        # Attach text + page + doc info — kglite's vector_search may strip
        # extra props post-reload, so we always re-join via Cypher.
        ids = [h["id"] for h in hits]
        if ids:
            df = self._store.cypher(
                f"MATCH (c:Chunk) WHERE c.id IN $ids "
                f"RETURN c.id AS id, c.{CHUNK_TEXT_COL} AS text, c.doc_id AS doc_id, "
                "c.page_number AS page, c.headings_json AS headings, c.status AS status",
                params={"ids": ids},
            )
            extras = {r["id"]: r for r in _df_to_dicts(df)}
            for h in hits:
                h.update({k: v for k, v in extras.get(h["id"], {}).items() if k != "id"})
        if agent_id and hits:
            for h in hits:
                _record_view(
                    self._store, agent_id=agent_id, target_id=h["id"],
                    target_kind=CHUNK, context=f"search:{query[:80]}",
                )
        if with_summaries and hits:
            for h in hits:
                h["summaries"] = self.get_summaries(h["id"], target_kind=CHUNK)
        return hits

    def similar_chunks(self, chunk_id: str, *, top_k: int = 10) -> list[SearchHit]:
        df = self._store.cypher(
            f"MATCH (c:Chunk {{id: $id}}) RETURN c.{CHUNK_TEXT_COL} AS text",
            params={"id": chunk_id},
        )
        rows = _df_to_dicts(df)
        if not rows or not rows[0].get("text"):
            return []
        return self.search(rows[0]["text"], top_k=top_k + 1)[1 : top_k + 1]

    def compose_context(
        self,
        query: str,
        *,
        max_tokens: int = 4000,
        per_doc_cap: int | None = None,
        include_summaries: bool = True,
        agent_id: str | None = None,
    ) -> ComposedContext:
        return context_mod.compose_context(
            self, query=query, max_tokens=max_tokens,
            per_doc_cap=per_doc_cap, include_summaries=include_summaries,
            agent_id=agent_id,
        )

    # ─── enrichments ──────────────────────────────────────────────────────

    def add_summary(
        self,
        target_id: str,
        text: str,
        *,
        target_kind: TargetKind = "Chunk",
        depth: SummaryDepth = "chunk",
        agent_id: str,
        model: str = "",
        tags: Iterable[str] = (),
    ) -> str:
        return enrich_mod.add_summary(
            self._store, self._embedder,
            target_id=target_id, target_kind=target_kind, depth=depth,
            text=text, agent_id=agent_id, model=model, tags=list(tags),
        )

    def verify_summary(
        self,
        summary_id: str,
        *,
        verdict: SummaryVerdict,
        verifier_agent_id: str,
        notes: str = "",
    ) -> dict[str, Any]:
        return enrich_mod.verify_summary(
            self._store, summary_id=summary_id, verdict=verdict,
            verifier_agent_id=verifier_agent_id, notes=notes,
        )

    def link_verification(
        self, verifier_summary_id: str, target_summary_id: str
    ) -> dict[str, Any]:
        return enrich_mod.link_verification(
            self._store, verifier_summary_id=verifier_summary_id,
            target_summary_id=target_summary_id,
        )

    def get_summaries(
        self,
        target_id: str,
        *,
        target_kind: TargetKind | None = None,
        status: SummaryStatus | None = None,
        depth: SummaryDepth | None = None,
    ) -> list[SummaryRow]:
        return enrich_mod.get_summaries(
            self._store, target_id=target_id, target_kind=target_kind,
            status=status, depth=depth,
        )

    def find_consensus(self, query: str, *, top_k: int = 20) -> list[dict[str, Any]]:
        return enrich_mod.find_consensus(self._store, self._embedder, query=query, top_k=top_k)

    # ─── tagging ──────────────────────────────────────────────────────────

    def tag_chunk(
        self,
        chunk_id: str,
        tag_name: str,
        *,
        kind: TagKind = "custom",
        agent_id: str,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        return _tag_chunk(
            self._store, chunk_id=chunk_id, tag_name=tag_name,
            kind=kind, agent_id=agent_id, confidence=confidence,
        )

    def untag_chunk(self, chunk_id: str, tag_name: str, *, agent_id: str) -> dict[str, Any]:
        return _untag_chunk(self._store, chunk_id=chunk_id, tag_name=tag_name, agent_id=agent_id)

    def list_tags(self, **filters: Any) -> list[TagRow]:
        return _list_tags(self._store, **filters)

    def chunks_by_tag(self, tag_name: str, *, limit: int = 100) -> list[dict[str, Any]]:
        return _chunks_by_tag(self._store, tag_name=tag_name, limit=limit)

    # ─── agents ───────────────────────────────────────────────────────────

    def register_agent(
        self, agent_id: str, *, kind: AgentKind = "llm", model: str = ""
    ) -> dict[str, Any]:
        return _register_agent(self._store, agent_id=agent_id, kind=kind, model=model)

    def list_agents(self) -> list[AgentRow]:
        return _list_agents(self._store)

    def record_view(
        self, chunk_id: str, agent_id: str, *, context: str = ""
    ) -> dict[str, Any]:
        return _record_view(
            self._store, agent_id=agent_id, target_id=chunk_id,
            target_kind=CHUNK, context=context,
        )

    # ─── ocr ──────────────────────────────────────────────────────────────

    def ocr_status(self, *, doc_id: str | None = None) -> OcrStatus:
        """Coverage summary: which documents have un-OCR'd pages, and
        what fraction of the corpus is still pending. Pass `doc_id` to
        narrow to one document."""
        return ocr_mod.ocr_status(self._store, doc_id=doc_id)

    def list_pending_ocr(
        self, *, doc_id: str | None = None, limit: int = 20,
        include_images: bool = True, dpi: int = 200,
    ) -> list[PendingOcrRow]:
        return ocr_mod.list_pending_ocr(
            self._store, doc_id=doc_id, limit=limit,
            include_images=include_images, dpi=dpi,
        )

    def submit_ocr(
        self,
        page_id: str,
        markdown: str,
        *,
        agent_id: str,
        model: str = "",
        confidence: float | None = None,
    ) -> dict[str, Any]:
        return ocr_mod.submit_ocr(
            self._store, self._embedder, page_id=page_id, markdown=markdown,
            agent_id=agent_id, model=model, confidence=confidence,
        )

    # ─── clustering ───────────────────────────────────────────────────────

    def cluster_chunks(
        self,
        *,
        algorithm: ClusterAlgorithm = "louvain",
        params: dict[str, Any] | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        return cluster_mod.cluster_chunks(
            self._store, algorithm=algorithm, params=params or {}, note=note,
        )

    def get_cluster(self, cluster_id: str, *, top_terms: int = 10) -> dict[str, Any] | None:
        return cluster_mod.get_cluster(self._store, cluster_id=cluster_id, top_terms=top_terms)

    def cluster_overview(self) -> list[dict[str, Any]]:
        return cluster_mod.cluster_overview(self._store)

    # ─── quality ──────────────────────────────────────────────────────────

    def check_grounding(
        self,
        summary_id: str,
        *,
        threshold: float = 0.5,
    ) -> GroundingReport:
        return quality_mod.check_grounding(
            self._store, self._embedder, summary_id=summary_id, threshold=threshold,
        )

    def verify_claim(
        self,
        claim_text: str,
        *,
        against_chunk_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        return quality_mod.verify_claim(
            self._store, self._embedder, claim_text=claim_text,
            against_chunk_ids=against_chunk_ids, top_k=top_k,
        )

    # ─── translation ──────────────────────────────────────────────────────

    def add_translation(
        self,
        chunk_id: str,
        target_lang: str,
        text: str,
        *,
        agent_id: str,
        model: str = "",
        status: TranslationStatus = "draft",
    ) -> str:
        """Store an agent-produced translation for a single chunk."""
        return translate_mod.add_translation(
            self._store, chunk_id=chunk_id, target_lang=target_lang,
            text=text, agent_id=agent_id, model=model, status=status,
        )

    def get_translations(
        self, chunk_id: str, *, target_lang: str | None = None
    ) -> list[dict[str, Any]]:
        return translate_mod.get_translations(
            self._store, chunk_id=chunk_id, target_lang=target_lang,
        )

    def mark_translation_reviewed(
        self, translation_id: str, *, reviewer_agent_id: str
    ) -> dict[str, Any]:
        return translate_mod.mark_translation_reviewed(
            self._store, translation_id=translation_id,
            reviewer_agent_id=reviewer_agent_id,
        )

    def assemble_translated_document(
        self, doc_id: str, *, target_lang: str, prefer_reviewed: bool = True
    ) -> dict[str, Any]:
        """Stitch a document's translated chunks back together. Pages
        without a translation fall back to the original text."""
        return translate_mod.assemble_translated_document(
            self._store, doc_id=doc_id, target_lang=target_lang,
            prefer_reviewed=prefer_reviewed,
        )

    # ─── export ───────────────────────────────────────────────────────────

    def export_document(
        self,
        doc_id: str,
        out_path: str | Path,
        *,
        format: ExportFormat | None = None,
        include_summaries: bool = False,
    ) -> Path:
        return export_mod.export_document(
            self, doc_id, out_path, format=format,
            include_summaries=include_summaries,
        )

    def export_cluster(
        self,
        cluster_id: str,
        out_path: str | Path,
        *,
        format: ExportFormat | None = None,
        include_member_text: bool = False,
    ) -> Path:
        return export_mod.export_cluster(
            self, cluster_id, out_path, format=format,
            include_member_text=include_member_text,
        )

    def export_summary(
        self, summary_id: str, out_path: str | Path, *, format: ExportFormat | None = None
    ) -> Path:
        return export_mod.export_summary(self, summary_id, out_path, format=format)

    def export_bundle(
        self,
        items: list[dict[str, Any]],
        out_path: str | Path,
        *,
        format: ExportFormat | None = None,
        title: str = "Synthesis bundle",
    ) -> Path:
        return export_mod.export_bundle(
            self, items, out_path, format=format, title=title,
        )

    # ─── review queue (kanban) ────────────────────────────────────────────

    def enqueue_review(
        self, target_id: str, *, target_kind: TargetKind = "Chunk",
        priority: int = 0, note: str = "", enqueued_by: str = "system",
    ) -> str:
        """Add a target node (chunk/summary/document/page) to the review
        queue. Returns the ticket id."""
        return review_mod.enqueue(
            self._store, target_id=target_id, target_kind=target_kind,
            priority=priority, note=note, enqueued_by=enqueued_by,
        )

    def enqueue_chunks_for_review(
        self, *, doc_id: str | None = None, status_filter: str | None = "ready",
        priority: int = 0, enqueued_by: str = "system",
    ) -> dict[str, Any]:
        """Bulk-enqueue every chunk (optionally scoped to one document or
        a Chunk.status filter). Skips chunks that already have a ticket."""
        return review_mod.enqueue_chunks(
            self._store, doc_id=doc_id, status_filter=status_filter,
            priority=priority, enqueued_by=enqueued_by,
        )

    def claim_review(self, ticket_id: str, *, agent_id: str) -> dict[str, Any]:
        """Atomically claim a specific ticket. Raises `ReviewConflict` if
        it's not currently in the `new` state."""
        return review_mod.claim(
            self._store, ticket_id=ticket_id, agent_id=agent_id,
        )

    def claim_next_review(
        self, *, agent_id: str, target_kind: TargetKind | None = None,
        min_priority: int | None = None,
    ) -> ReviewTicketDetail | None:
        """Atomic 'pull from the queue': finds the highest-priority `new`
        ticket and claims it for `agent_id`. Returns the ticket with the
        target hydrated, or `None` if the queue is empty."""
        return review_mod.claim_next(
            self._store, agent_id=agent_id,
            target_kind=target_kind, min_priority=min_priority,
        )

    def unclaim_review(
        self, ticket_id: str, *, agent_id: str, reason: str = "",
    ) -> dict[str, Any]:
        """Release a claim without a verdict. Only the current claimer
        can unclaim."""
        return review_mod.unclaim(
            self._store, ticket_id=ticket_id, agent_id=agent_id, reason=reason,
        )

    def complete_review(
        self, ticket_id: str, *, agent_id: str,
        verdict: ReviewVerdict = "reviewed",
        accuracy: float | None = None,
        authenticity: str | None = None,
        notes: str = "",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Mark a ticket reviewed. `verdict` is one of
        `reviewed` / `needs_revision` / `rejected`. Optional `accuracy`
        (0-1) and `authenticity` capture the agent's judgement. `tags`
        are applied to the target chunk (only when target_kind=Chunk)."""
        return review_mod.complete(
            self._store, ticket_id=ticket_id, agent_id=agent_id,
            verdict=verdict, accuracy=accuracy, authenticity=authenticity,
            notes=notes, tags=tags,
        )

    def list_review_queue(
        self, *, status: ReviewStatus | None = None,
        target_kind: TargetKind | None = None, agent_id: str | None = None,
        limit: int = 50,
    ) -> list[ReviewTicketRow]:
        """List tickets with their current event-sourced status."""
        return review_mod.list_queue(
            self._store, status=status, target_kind=target_kind,
            agent_id=agent_id, limit=limit,
        )

    def get_review_ticket(
        self, ticket_id: str, *, with_target: bool = True, with_events: bool = True,
    ) -> ReviewTicketDetail | None:
        """Full ticket detail including the target node and the
        immutable event audit trail."""
        return review_mod.get_ticket(
            self._store, ticket_id=ticket_id,
            with_target=with_target, with_events=with_events,
        )

    def review_stats(self) -> ReviewStats:
        """Kanban board summary: counts per status + per-agent in-review."""
        return review_mod.stats(self._store)

    # ─── cypher escape hatch ──────────────────────────────────────────────

    def cypher(self, query: str, params: dict[str, Any] | None = None) -> Any:
        return self._store.cypher(query, params)

    def schema(self) -> dict[str, Any]:
        return self._store.schema()


from kglite_docs.store import rows as _df_to_dicts  # noqa: E402
