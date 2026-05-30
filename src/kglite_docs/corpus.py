"""Public façade — the only object most users need.

`Corpus.create(path)` / `Corpus.open(path)` open a kglite-docs knowledge
base. Method names mirror what the MCP server exposes so library users
and agent users see the same vocabulary.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Iterable
from pathlib import Path
from types import TracebackType
from typing import Any, cast

from kglite_docs import cluster as cluster_mod
from kglite_docs import context as context_mod
from kglite_docs import coverage as coverage_mod
from kglite_docs import enrich as enrich_mod
from kglite_docs import export as export_mod
from kglite_docs import ocr as ocr_mod
from kglite_docs import quality as quality_mod
from kglite_docs import review as review_mod
from kglite_docs import study as study_mod
from kglite_docs import translate as translate_mod
from kglite_docs.activity import (
    agent_activity as _agent_activity,
)
from kglite_docs.activity import (
    get_agent as _get_agent,
)
from kglite_docs.activity import (
    list_agents as _list_agents,
)
from kglite_docs.activity import (
    record_view as _record_view,
)
from kglite_docs.activity import (
    register_agent as _register_agent,
)
from kglite_docs.activity import (
    upsert_agent as _upsert_agent,
)
from kglite_docs.embed import make_embedder
from kglite_docs.errors import NotIndexedError
from kglite_docs.ingest.pipeline import IngestResult
from kglite_docs.ingest.pipeline import ingest_document as _ingest_doc
from kglite_docs.schema import (
    CHUNK,
    CHUNK_TEXT_COL,
    LABEL_EMBEDDED,
    LABEL_READY,
)
from kglite_docs.store import AttrDict, Store
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
    AgentActivity,
    AgentConfig,
    AgentKind,
    AgentRow,
    AssessmentVerdict,
    ChunkDetail,
    ClusterAlgorithm,
    ComparisonQueryResult,
    ComparisonResult,
    ComposedContext,
    ConflictReport,
    CorpusStatus,
    CoverageReport,
    DocumentDetail,
    DocumentRow,
    ExportFormat,
    GroundingReport,
    Ledger,
    OcrStatus,
    PendingOcrRow,
    Provenance,
    ReviewStats,
    ReviewStatus,
    ReviewTicketDetail,
    ReviewTicketRow,
    ReviewVerdict,
    SearchHit,
    SectionRow,
    Stance,
    StudyRow,
    StudyStatus,
    SummaryDepth,
    SummaryRow,
    SummaryStatus,
    SummaryVerdict,
    TagKind,
    TagRow,
    TargetKind,
    TranslationStatus,
    TriageMap,
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
        embed: bool = False,
        structure_aware: bool = False,
        context_summary: str = "",
    ) -> IngestResult:
        """Ingest a document. Three modes:

        - ``ingest("paper.pdf")`` — file path; format auto-detected from
          the extension. Pass ``format=`` to override.
        - ``ingest(text="# Notes\\n…", title="my-notes")`` — raw text /
          markdown. Useful for agent-generated synthesis articles.
        - ``ingest("doc.bin", format="md")`` — file path with explicit
          format hint when the extension doesn't match.

        Embedding is **opt-in**. By default (``embed=False``) ingest does
        not touch the embedding model — it parses, chunks, and writes the
        graph, leaving ready chunks ``:Unembedded``. Call :meth:`index`
        afterwards (or pass ``embed=True`` here) to compute vectors and
        enable :meth:`search`. Non-semantic workflows (browse, cypher,
        tag, review, OCR, export, translate) need no embeddings at all.

        With ``structure_aware=True`` chunking starts a fresh chunk at every
        top-level heading (never packing or overlapping across one) — cleaner
        Section boundaries and pinpoint cites; the default packs greedily.

        ``context_summary`` (opt-in) is a document-level blurb prepended to each
        chunk *before embedding* so the vector carries global context (mitigates
        cross-document speaker/source confusion); the stored chunk text is
        unchanged. You supply the summary (e.g. from an LLM pass) — none is
        generated here.

        Returns an :class:`IngestResult` with the assigned ``doc_id``
        (sha256 of file or text bytes), chunk count, OCR-pending page
        count, and how many chunks were embedded (0 unless ``embed=True``).

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
                    metadata=metadata, format=fmt, embed=embed,
                    structure_aware=structure_aware, context_summary=context_summary,
                )
            finally:
                tmp_path.unlink(missing_ok=True)
        return _ingest_doc(
            self._store, self._embedder, path,  # type: ignore[arg-type]
            title=title, source_uri=source_uri, metadata=metadata,
            format=format, embed=embed, structure_aware=structure_aware,
            context_summary=context_summary,
        )

    def ingest_dir(
        self,
        directory: str | Path,
        *,
        recursive: bool = True,
        patterns: list[str] | None = None,
        embed: bool = False,
        structure_aware: bool = False,
    ) -> list[IngestResult]:
        """Ingest every supported file under ``directory``. By default
        scans for all known formats: PDF, DOCX, PPTX, MD, HTML, TXT, and
        common image formats.

        As with :meth:`ingest`, embedding is opt-in (``embed=False``).
        For bulk loads prefer the default and call :meth:`index` once at
        the end — one batched embedding pass beats per-file embedding."""
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
                    results.append(self.ingest(f, embed=embed, structure_aware=structure_aware))
                except Exception as exc:
                    import logging
                    logging.getLogger("kglite_docs").warning(
                        "ingest failed for %s: %s", f, exc,
                    )
        return results

    # ─── indexing (embedding) ───────────────────────────────────────────────

    def count_unembedded(self, *, doc_id: str | None = None) -> int:
        """How many ready chunks are still awaiting embedding. Scope with
        ``doc_id``. Tracked by the ``c.embedded`` boolean property (not a
        removable label — see :meth:`index`)."""
        preds = ["c.embedded = false"]
        params: dict[str, Any] = {}
        if doc_id:
            preds.append("c.doc_id = $doc_id")
            params["doc_id"] = doc_id
        q = "MATCH (c:Chunk:Ready) WHERE " + " AND ".join(preds) + " RETURN count(c) AS n"
        rows_ = _df_to_dicts(self._store.cypher(q, params=params))
        return int(rows_[0]["n"]) if rows_ else 0

    def _embedding_coverage(self) -> tuple[int, int]:
        """Corpus-wide ``(embedded, ready)`` chunk counts — the basis for the
        loud-retrieval signal in :meth:`search`. ``embedded + unembedded ==
        ready`` always holds (embedded is derived from :meth:`count_unembedded`,
        which tracks the authoritative ``c.embedded`` property)."""
        rows_ = _df_to_dicts(self._store.cypher(
            f"MATCH (c:Chunk:{LABEL_READY}) RETURN count(c) AS n"
        ))
        ready = int(rows_[0]["n"]) if rows_ else 0
        embedded = ready - self.count_unembedded()
        return embedded, ready

    def index(
        self,
        *,
        doc_id: str | None = None,
        batch_size: int = 16,
        max_chunks: int | None = None,
        max_seconds: float | None = 30.0,
    ) -> dict[str, Any]:
        """Embed ready-but-unembedded chunks — the optional second phase
        of ingestion that makes :meth:`search` work.

        **Bounded and loop-friendly.** A single call does at most
        ``max_seconds`` of work (wall-clock budget, default 30s) or
        ``max_chunks`` chunks, whichever is hit first, then commits what
        it embedded and returns ``pending > 0`` if more remain. This keeps
        any one call comfortably under an MCP client's per-call timeout
        even for a large multi-document corpus — the caller loops until
        ``pending == 0``::

            while corpus.index()["pending"]:
                ...

        Pass ``max_seconds=None`` (and ``max_chunks=None``) to drain
        everything in one call when there's no timeout to worry about
        (e.g. CLI preload). Idempotent: only touches chunks not yet
        embedded (tracked by the ``c.embedded`` property), so looping or
        re-running is safe. Scope to one document with ``doc_id``.

        Chunks are embedded in length-sorted batches so each batch pads to
        a similar sequence length (bge-m3 caps at 8192) — avoids a few long
        chunks inflating the padding for a document's worth of short ones.
        The ``:Embedded`` label is added (never removed) as chunks are
        indexed; the pending side is the property, because kglite's
        ``remove_label`` leaves the label-predicate index stale.

        Returns ``{"embedded": n, "pending": remaining, "doc_id": ...}``.
        """
        import time

        preds = ["c.embedded = false"]
        params: dict[str, Any] = {}
        if doc_id:
            preds.append("c.doc_id = $doc_id")
            params["doc_id"] = doc_id
        # Join the Document (one per chunk via HAS_CHUNK) for its optional
        # embed_context (FEAT-11) — prepended to the embed input below.
        q = (
            "MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk:Ready) WHERE " + " AND ".join(preds)
            + f" RETURN c.id AS id, c.{CHUNK_TEXT_COL} AS text, d.embed_context AS embed_context"
        )
        pending = [r for r in _df_to_dicts(self._store.cypher(q, params=params)) if r.get("text")]
        if not pending:
            return {"embedded": 0, "pending": 0, "doc_id": doc_id}

        # Group similar-length chunks together to minimise padding waste.
        pending.sort(key=lambda r: len(r["text"]))
        if max_chunks is not None and max_chunks > 0:
            pending = pending[:max_chunks]

        budget = max_seconds if (max_seconds and max_seconds > 0) else None
        start = time.monotonic()
        all_vecs: dict[str, list[float]] = {}
        for i in range(0, len(pending), batch_size):
            batch = pending[i : i + batch_size]
            ids = [r["id"] for r in batch]
            vecs = self._embedder.embed([_embed_input(r) for r in batch])
            all_vecs.update(zip(ids, vecs, strict=False))
            # Stop once the wall-clock budget is spent (checked after each
            # batch, so a call overshoots by at most one batch).
            if budget is not None and (time.monotonic() - start) > budget:
                break

        embedded_ids = list(all_vecs.keys())
        self._store.add_embeddings(CHUNK, CHUNK_TEXT_COL, all_vecs)
        self._store.cypher(
            "MATCH (c:Chunk) WHERE c.id IN $ids SET c.embedded = true",
            params={"ids": embedded_ids},
        )
        self._store.add_label(CHUNK, embedded_ids, LABEL_EMBEDDED)
        return {
            "embedded": len(embedded_ids),
            "pending": self.count_unembedded(doc_id=doc_id),
            "doc_id": doc_id,
        }

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
        return cast(list[DocumentRow], _df_to_dicts(df))

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
        return cast(DocumentDetail | None, doc)

    def list_sections(self, doc_id: str) -> list[SectionRow]:
        """Sections of a document (the grain between document and chunk),
        in reading order, each with its `chunk_count`. Sections are derived at
        ingest from the PDF outline or top-level headings; re-ingest documents
        ingested before this feature to populate them."""
        df = self._store.cypher(
            "MATCH (d:Document {id: $id})-[:HAS_SECTION]->(s:Section) "
            "OPTIONAL MATCH (s)-[:HAS_CHUNK]->(c:Chunk) "
            "WITH s, count(c) AS chunk_count "
            "RETURN s.id AS id, s.doc_id AS doc_id, s.title AS title, "
            "s.page_start AS page_start, s.page_end AS page_end, "
            "s.level AS level, s.doc_type AS doc_type, chunk_count "
            "ORDER BY s.ordinal",
            params={"id": doc_id},
        )
        return cast(list[SectionRow], _df_to_dicts(df))

    # ─── chunks ───────────────────────────────────────────────────────────

    def get_chunk(
        self,
        chunk_id: str,
        *,
        with_neighbors: bool = False,
        with_summaries: bool = False,
        window: int = 0,
        agent_id: str | None = None,
    ) -> ChunkDetail | None:
        df = self._store.cypher(
            f"""
            MATCH (c:Chunk {{id: $id}})
            RETURN c.id AS id, c.title AS title, c.doc_id AS doc_id,
                   c.page_number AS page, c.chunk_index AS chunk_index,
                   c.{CHUNK_TEXT_COL} AS text, c.token_count AS token_count,
                   c.word_count AS word_count, c.char_count AS char_count,
                   c.content_kind AS content_kind, c.quality_score AS quality_score,
                   c.boilerplate AS boilerplate, c.entities_json AS entities_json,
                   c.headings_json AS headings, c.status AS status,
                   c.section_id AS section_id, c.doc_type AS doc_type,
                   c.view_count AS view_count
            """,
            params={"id": chunk_id},
        )
        rows = _df_to_dicts(df)
        if not rows:
            return None
        chunk = AttrDict(rows[0])  # both chunk["page"] and chunk.page work
        # Parse the structured-entity hints (FEAT-11.2) into a dict for the agent.
        import json
        try:
            chunk["entities"] = json.loads(chunk.pop("entities_json", None) or "{}")
        except (TypeError, ValueError):
            chunk["entities"] = {}
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
        if window and window > 0:
            # Read-context: the `window` chunks before and after in reading
            # order (the NEXT_CHUNK spine), with text — so an agent can
            # interpret an incoherent chunk via its neighbours in one call.
            n = int(window)
            before = _df_to_dicts(self._store.cypher(
                f"MATCH (p:Chunk)-[:NEXT_CHUNK*1..{n}]->(c:Chunk {{id: $id}}) "
                f"RETURN p.id AS id, p.{CHUNK_TEXT_COL} AS text, p.page_number AS page, "
                "p.chunk_index AS chunk_index ORDER BY p.page_number, p.chunk_index",
                params={"id": chunk_id},
            ))
            after = _df_to_dicts(self._store.cypher(
                f"MATCH (c:Chunk {{id: $id}})-[:NEXT_CHUNK*1..{n}]->(nx:Chunk) "
                f"RETURN nx.id AS id, nx.{CHUNK_TEXT_COL} AS text, nx.page_number AS page, "
                "nx.chunk_index AS chunk_index ORDER BY nx.page_number, nx.chunk_index",
                params={"id": chunk_id},
            ))
            chunk["context_before"] = before
            chunk["context_after"] = after
        if with_summaries:
            chunk["summaries"] = self.get_summaries(chunk_id, target_kind=CHUNK)
        if agent_id:
            _record_view(self._store, agent_id=agent_id, target_id=chunk_id, target_kind=CHUNK, context="get_chunk")
        return cast(ChunkDetail | None, chunk)

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
        # Honest coverage: an unindexed corpus must be a loud signal, not a
        # silent []. 0 embedded (but chunks exist) → raise; partial → warn.
        embedded, ready = self._embedding_coverage()
        if ready > 0 and embedded == 0:
            raise NotIndexedError(
                f"0 of {ready} ready chunk(s) are embedded — call index() "
                "(or ingest(embed=True)) before search()."
            )
        if 0 < embedded < ready:
            warnings.warn(
                f"searching {embedded}/{ready} embedded chunk(s); "
                f"{ready - embedded} unembedded are invisible — "
                "run index() for full coverage.",
                stacklevel=2,
            )
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
        return cast(list[SearchHit], hits)

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
        return cast(ComposedContext, context_mod.compose_context(
            self, query=query, max_tokens=max_tokens,
            per_doc_cap=per_doc_cap, include_summaries=include_summaries,
            agent_id=agent_id,
        ))

    def compare_documents(
        self,
        doc_a: str,
        doc_b: str,
        *,
        queries: list[str],
        top_k_per_query: int = 5,
        max_tokens_per_query: int = 2000,
        agent_id: str | None = None,
    ) -> ComparisonResult:
        """Side-by-side cross-document retrieval.

        For each query, returns the top hits from `doc_a` and `doc_b`
        independently, plus a budgeted merged context bundle ready to
        hand to a downstream LLM that's writing a comparison.

        `queries` is a list of *axes* you want to compare on — e.g.
        ``["retrieval architecture", "training objective",
        "evaluation metrics"]`` for two papers. 3–7 queries is the
        sweet spot.

        Returns a `ComparisonResult` dict:

            {
              "doc_a_id": ..., "doc_a_title": ...,
              "doc_b_id": ..., "doc_b_title": ...,
              "queries": [
                {
                  "query": "...",
                  "doc_a_hits": [...],   # top_k_per_query
                  "doc_b_hits": [...],
                  "merged_context": {used_tokens, items}   # ComposedContext
                },
                ...
              ]
            }
        """
        # Hydrate titles once
        meta_df = self._store.cypher(
            "MATCH (d:Document) WHERE d.id IN $ids "
            "RETURN d.id AS id, d.title AS title",
            params={"ids": [doc_a, doc_b]},
        )
        titles: dict[str, str] = {
            r["id"]: r.get("title", "") for r in _df_to_dicts(meta_df)
        }
        per_query: list[dict[str, Any]] = []
        for q in queries:
            doc_a_hits = self.search(
                q, top_k=top_k_per_query,
                filters={"doc_id": doc_a}, agent_id=agent_id,
            )
            doc_b_hits = self.search(
                q, top_k=top_k_per_query,
                filters={"doc_id": doc_b}, agent_id=agent_id,
            )
            # Merged context across both docs with the per-query budget,
            # `per_doc_cap` ensures balanced representation.
            merged = self.compose_context(
                q,
                max_tokens=max_tokens_per_query,
                per_doc_cap=max(2, top_k_per_query // 2),
                agent_id=None,  # don't double-count views
            )
            per_query.append({
                "query": q,
                "doc_a_hits": doc_a_hits,
                "doc_b_hits": doc_b_hits,
                "merged_context": merged,
            })
        return {
            "doc_a_id": doc_a, "doc_a_title": titles.get(doc_a, ""),
            "doc_b_id": doc_b, "doc_b_title": titles.get(doc_b, ""),
            "queries": cast(list[ComparisonQueryResult], per_query),
        }

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
        return cast(list[SummaryRow], enrich_mod.get_summaries(
            self._store, target_id=target_id, target_kind=target_kind,
            status=status, depth=depth,
        ))

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
        return cast(list[TagRow], _list_tags(self._store, **filters))

    def chunks_by_tag(self, tag_name: str, *, limit: int = 100) -> list[dict[str, Any]]:
        return _chunks_by_tag(self._store, tag_name=tag_name, limit=limit)

    # ─── agents ───────────────────────────────────────────────────────────

    def register_agent(
        self, agent_id: str, *, kind: AgentKind = "llm", model: str = "",
    ) -> dict[str, Any]:
        """Idempotent lazy registration. Bumps `last_seen` + counters
        if the agent already exists; minimal-record creates otherwise.
        Does *not* overwrite template fields — use `upsert_agent` for
        that."""
        return _register_agent(self._store, agent_id=agent_id, kind=kind, model=model)

    def upsert_agent(
        self,
        agent_id: str,
        *,
        kind: AgentKind = "llm",
        model: str = "",
        role: str = "",
        system_prompt: str = "",
        tools: list[str] | None = None,
        context: dict[str, Any] | None = None,
        description: str = "",
    ) -> AgentConfig:
        """Write the agent's template: role, system prompt, model,
        tool list, free-form context. Field-level merge with whatever
        already exists. Returns the resulting config.

        Once defined, fetch with `get_agent(agent_id)` and use the
        config to launch your LLM call — the agent_id you then use
        for subsequent `add_summary` / `complete_review` / etc.
        will be attributed back to this template."""
        return cast(AgentConfig, _upsert_agent(
            self._store, agent_id=agent_id, kind=kind, model=model,
            role=role, system_prompt=system_prompt,
            tools=tools, context=context, description=description,
        ))

    def get_agent(self, agent_id: str) -> AgentConfig:
        """Full agent config — template + counters. Empty dict if the
        agent isn't registered yet."""
        return cast(AgentConfig, _get_agent(self._store, agent_id=agent_id))

    def list_agents(
        self, *, role: str | None = None, kind: AgentKind | None = None,
    ) -> list[AgentRow]:
        """List configured agents, optionally filtered by role or kind."""
        return cast(list[AgentRow], _list_agents(self._store, role=role, kind=kind))

    def agent_activity(
        self,
        agent_id: str,
        *,
        target_id: str | None = None,
        limit: int = 50,
    ) -> AgentActivity:
        """Everything this agent has done in the corpus — optionally
        scoped to one target node. Buckets: views, summaries, tags,
        translations, review_events, verification_events."""
        return cast(AgentActivity, _agent_activity(
            self._store, agent_id=agent_id,
            target_id=target_id, limit=limit,
        ))

    def record_view(
        self, chunk_id: str, agent_id: str, *, context: str = ""
    ) -> dict[str, Any]:
        return _record_view(
            self._store, agent_id=agent_id, target_id=chunk_id,
            target_kind=CHUNK, context=context,
        )

    # ─── coverage / status ────────────────────────────────────────────────

    def status(self) -> CorpusStatus:
        """One-call snapshot: docs, pages, chunks, embedded/unembedded,
        image_pages, pending_ocr, studies. The first thing to check."""
        return cast(CorpusStatus, coverage_mod.corpus_status(self._store))

    def coverage_report(self, *, doc_id: str | None = None) -> CoverageReport:
        """Honest extraction + embedding coverage per document + corpus-wide,
        with a human-readable `summary` — what's image-only / low-text
        (unanalyzed unless OCR'd) and how many chunks are unembedded (search
        blind until `index()`). Pass `doc_id` to scope the per-doc rows."""
        return cast(CoverageReport, coverage_mod.coverage_report(self._store, doc_id=doc_id))

    def triage_map(self, *, doc_id: str | None = None) -> TriageMap:
        """One cheap call that aggregates the deterministic content signals — the
        content_kind breakdown, boilerplate / low-quality counts, structured-
        entity coverage, embedding state, OCR-pending pages — so an agent orients
        without reading the corpus. Scope with `doc_id`."""
        return cast(TriageMap, coverage_mod.triage_map(self._store, doc_id=doc_id))

    # ─── ocr ──────────────────────────────────────────────────────────────

    def ocr_status(self, *, doc_id: str | None = None) -> OcrStatus:
        """Coverage summary: which documents have un-OCR'd pages, and
        what fraction of the corpus is still pending. Pass `doc_id` to
        narrow to one document."""
        return cast(OcrStatus, ocr_mod.ocr_status(self._store, doc_id=doc_id))

    def list_pending_ocr(
        self, *, doc_id: str | None = None, limit: int = 20,
        include_images: bool = True, dpi: int = 200,
    ) -> list[PendingOcrRow]:
        return cast(list[PendingOcrRow], ocr_mod.list_pending_ocr(
            self._store, doc_id=doc_id, limit=limit,
            include_images=include_images, dpi=dpi,
        ))

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
        return cast(GroundingReport, quality_mod.check_grounding(
            self._store, self._embedder, summary_id=summary_id, threshold=threshold,
        ))

    def verify_claim(
        self,
        claim_text: str,
        *,
        against_chunk_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Find chunks that support a free-text claim via vector search.

        Deprecated: prefer the `study` flow (`define_study` → `assess` →
        `study_ledger`) to evaluate a claim across chunks — it's richer
        (for/against + weight + provenance), multi-agent, and verifiable. This
        one-shot helper remains for quick checks."""
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
        return cast(ReviewTicketDetail | None, review_mod.claim_next(
            self._store, agent_id=agent_id,
            target_kind=target_kind, min_priority=min_priority,
        ))

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
        return cast(list[ReviewTicketRow], review_mod.list_queue(
            self._store, status=status, target_kind=target_kind,
            agent_id=agent_id, limit=limit,
        ))

    def get_review_ticket(
        self, ticket_id: str, *, with_target: bool = True, with_events: bool = True,
    ) -> ReviewTicketDetail | None:
        """Full ticket detail including the target node and the
        immutable event audit trail."""
        return cast(ReviewTicketDetail | None, review_mod.get_ticket(
            self._store, ticket_id=ticket_id,
            with_target=with_target, with_events=with_events,
        ))

    def review_stats(self) -> ReviewStats:
        """Kanban board summary: counts per status + per-agent in-review."""
        return cast(ReviewStats, review_mod.stats(self._store))

    # ─── evidence study ───────────────────────────────────────────────────

    def define_study(
        self, question: str, *, created_by: str,
        title: str | None = None, status: StudyStatus = "open",
    ) -> str:
        """Create a Study (a question/claim to gather evidence for/against).
        Returns the study id. See `assess` / `study_ledger` / `verify_assessment`."""
        return study_mod.define_study(
            self._store, question=question, title=title,
            created_by=created_by, status=status,
        )

    def assess(
        self, study_id: str, chunk_id: str, *,
        stance: Stance, weight: float, agent_id: str,
        rationale: str = "", model: str = "",
        provenance: Provenance = "primary_text",
        quote: str = "", char_start: int | None = None, char_end: int | None = None,
        context_chunk_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Record stance (supports/against/neutral/deferred) + probative weight
        [0,1] + rationale on a chunk toward a study. Append-only; never embeds.
        `deferred` = read but unjudgeable yet (blocked/needs evidence): counted
        distinctly and kept in the work-list for a later pass.

        `provenance` records *what was checked* (the basis, vs `weight` the
        strength): `primary_text` (read the source — default), `characterization`
        (a paraphrase/summary), or `scanned_unread` (an unread scan; provisional).
        Surfaced per row in `study_ledger`.

        `quote`/`char_start`/`char_end` are an optional pinpoint span — the exact
        passage the call rests on, surfaced in the ledger for pinpoint cites.
        Validated against the chunk text (out-of-range / quote-not-found rejected).

        `context_chunk_ids`: neighbor chunks read to interpret the focal one;
        recorded so retrieval pulls the span and they're excluded from the
        work-list (no double-judging)."""
        return study_mod.assess(
            self._store, study_id=study_id, chunk_id=chunk_id,
            stance=stance, weight=weight, rationale=rationale,
            agent_id=agent_id, model=model, provenance=provenance,
            quote=quote, char_start=char_start, char_end=char_end,
            context_chunk_ids=context_chunk_ids,
        )

    def assess_many(self, study_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Batch-assess many chunks in one validated, batched write (a single
        persist through the MCP layer). Each row is a dict with
        `chunk_id`/`stance`/`weight`/`agent_id` (+ the optional `assess` fields).
        One bad row aborts the whole batch — nothing is written."""
        return study_mod.assess_many(self._store, study_id=study_id, rows=rows)

    def supersede_assessment(
        self, old_id: str, *,
        stance: Stance, weight: float, agent_id: str,
        rationale: str = "", model: str = "",
        provenance: Provenance = "primary_text",
        context_chunk_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Audit-preserving correction: record a new assessment that explicitly
        supersedes `old_id` (a `SUPERSEDES` edge). The old one is kept but hidden
        from `study_ledger` by default — resolving cross-agent corrections to a
        single current row per chunk. Inherits the old assessment's study+chunk."""
        return study_mod.supersede_assessment(
            self._store, old_id=old_id, stance=stance, weight=weight,
            agent_id=agent_id, rationale=rationale, model=model,
            provenance=provenance, context_chunk_ids=context_chunk_ids,
        )

    def study_ledger(
        self, study_id: str, *,
        stance: Stance | None = None, min_weight: float | None = None,
        verified_only: bool = False, doc_id: str | None = None,
        section_id: str | None = None, element: str | None = None,
        include_superseded: bool = False, limit: int = 200,
    ) -> Ledger:
        """Weight-ranked evidence ledger for a study + support/against tallies.
        Pass `stance="supports"`/`"against"` to retrieve just that side, or
        `doc_id=`/`section_id=` to scope to one document or section.
        Current-by-default: superseded assessments are hidden unless
        `include_superseded=True` (each row carries a `superseded` flag). The
        result reports `total` (matches before `limit`) and `returned`; `total >
        returned` means it was clipped."""
        return cast(Ledger, study_mod.ledger(
            self._store, study_id=study_id, stance=stance,
            min_weight=min_weight, verified_only=verified_only,
            doc_id=doc_id, section_id=section_id, element=element,
            include_superseded=include_superseded, limit=limit,
        ))

    def verify_assessment(
        self, assessment_id: str, *,
        verdict: AssessmentVerdict, verifier_agent_id: str, notes: str = "",
        provenance: Provenance | None = None,
    ) -> dict[str, Any]:
        """Second-agent check of an assessment: verified / disputed / duplicate.
        Self-verification is rejected. `provenance` (optional) records what the
        verifier checked — stored on the verification event."""
        return study_mod.verify_assessment(
            self._store, assessment_id=assessment_id, verdict=verdict,
            verifier_agent_id=verifier_agent_id, notes=notes, provenance=provenance,
        )

    def conclude_study(
        self, study_id: str, text: str, *,
        agent_id: str, model: str = "", embed: bool = False,
    ) -> str:
        """Write a conclusion (stored as a verifiable Summary on the Study)."""
        return study_mod.conclude_study(
            self._store, self._embedder, study_id=study_id, text=text,
            agent_id=agent_id, model=model, embed=embed,
        )

    def list_studies(
        self, *, status: StudyStatus | None = None, created_by: str | None = None,
    ) -> list[StudyRow]:
        """List studies that have been run (newest first)."""
        return cast(list[StudyRow], study_mod.list_studies(self._store, status=status, created_by=created_by))

    def get_study(self, study_id: str) -> StudyRow | None:
        """Study metadata + tallies + its conclusion summaries."""
        return cast(StudyRow | None, study_mod.get_study(self._store, study_id=study_id))

    def study_conflicts(self, study_id: str) -> ConflictReport:
        """Chunks with both a current `supports` and `against` assessment — the
        contested evidence to review first. Computed over the current
        (non-superseded, latest-per-agent) set; each conflict carries its
        opposing rows split by side."""
        return cast(ConflictReport, study_mod.conflicts(self._store, study_id=study_id))

    def next_unassessed(
        self, study_id: str, *,
        doc_id: str | None = None, section_id: str | None = None,
        element: str | None = None,
        agent_id: str | None = None, limit: int = 20,
        ttl_seconds: int = 1800,
    ) -> list[dict[str, Any]]:
        """Work-list of chunks not yet assessed for this study. When
        ``agent_id`` is given, atomically *claims* (checks out) the returned
        chunks so parallel analysts don't overlap; without it, a read-only
        preview. `doc_id`/`section_id` scope the work-list (hard filters);
        `element` is an **advisory** scope — chunks classified as that registered
        element type sort first (the full list is still returned, nothing hidden),
        so a study reads its subset first without re-scanning. Claims auto-expire
        after ``ttl_seconds``."""
        return study_mod.next_unassessed(
            self._store, study_id=study_id, doc_id=doc_id, section_id=section_id,
            element=element, agent_id=agent_id, limit=limit, ttl_seconds=ttl_seconds,
        )

    def reopen_study(self, study_id: str, *, agent_id: str) -> dict[str, Any]:
        """Flip a study back to open for deeper analysis."""
        return study_mod.reopen_study(self._store, study_id=study_id, agent_id=agent_id)

    def delete_study(self, study_id: str) -> dict[str, Any]:
        """Cascade-delete a study + its assessments, verification events, and
        conclusions. Destructive."""
        return study_mod.delete_study(self._store, study_id=study_id)

    # ─── cypher escape hatch ──────────────────────────────────────────────

    def cypher(self, query: str, params: dict[str, Any] | None = None) -> Any:
        """Run raw Cypher and return kglite's `ResultView`. It's ergonomic:
        iterate it (`for row in corpus.cypher(...)` — each row is a plain dict),
        index it (`result[0]["col"]`), take its `len(result)`, list it
        (`result.to_list()`), or read `result.columns`."""
        return self._store.cypher(query, params)

    def schema(self) -> dict[str, Any]:
        return self._store.schema()


from kglite_docs.store import rows as _df_to_dicts  # noqa: E402


def _embed_input(row: dict[str, Any]) -> str:
    """Text to embed for a chunk — prepends the document's `embed_context`
    (FEAT-11 summary-augmented chunking) when present, else the chunk text."""
    text = row["text"]
    ctx = row.get("embed_context")
    return f"{ctx}\n\n{text}" if ctx else text
