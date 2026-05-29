"""Thin façade over `kglite.KnowledgeGraph` carrying the kglite-docs schema
conventions. Created/loaded through `Corpus` — usually you don't need this
module directly.

Why a façade: kglite's bulk node API takes pandas DataFrames; we want to add
nodes from lists of dicts (and one node at a time for tags / agents).
Centralising the conversion keeps the call sites readable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import kglite
import pandas as pd


class Store:
    """Wraps a `KnowledgeGraph` plus its on-disk source path."""

    def __init__(self, graph: kglite.KnowledgeGraph, path: Path | None = None) -> None:
        self.g = graph
        self.path = path

    # ─── construction ──────────────────────────────────────────────────────

    @classmethod
    def create(cls, path: str | Path | None = None) -> Store:
        g = kglite.KnowledgeGraph()
        return cls(g, Path(path) if path else None)

    @classmethod
    def open(cls, path: str | Path) -> Store:
        p = Path(path)
        g = kglite.load(str(p))
        return cls(g, p)

    def save(self, path: str | Path | None = None) -> None:
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("save(): no path set and no path supplied")
        self.g.save(str(target))
        self.path = target

    # ─── nodes ─────────────────────────────────────────────────────────────

    def upsert_nodes(
        self,
        node_type: str,
        rows: Iterable[Mapping[str, Any]],
        *,
        id_field: str = "id",
        title_field: str = "title",
    ) -> int:
        """Insert nodes from an iterable of dicts. Skips rows whose `id`
        already exists for this type (kglite-style idempotency).

        Returns the count actually inserted.
        """
        rows = list(rows)
        if not rows:
            return 0
        # Filter out already-present ids
        existing_ids = self._existing_ids(node_type, [r[id_field] for r in rows])
        fresh = [r for r in rows if r[id_field] not in existing_ids]
        if not fresh:
            return 0
        df = pd.DataFrame(fresh)
        # Ensure title column exists; if not, fall back to id
        if title_field not in df.columns:
            df[title_field] = df[id_field].astype(str)
        self.g.add_nodes(df, node_type, unique_id_field=id_field, node_title_field=title_field)
        return len(fresh)

    def _existing_ids(self, node_type: str, ids: list[Any]) -> set[Any]:
        """Return the subset of `ids` already present as nodes of `node_type`."""
        if not ids:
            return set()
        df = self.g.cypher(
            f"MATCH (n:{node_type}) WHERE n.id IN $ids RETURN n.id AS id",
            params={"ids": list(ids)},
        )
        # cypher returns a kglite ResultView / polars-like frame; index access
        # is dynamic, so treat it as Any for the column lookup.
        rv: Any = df
        try:
            return set(rv["id"].to_list())
        except Exception:
            return set()

    def upsert_edges(
        self,
        edge_type: str,
        rows: Iterable[Mapping[str, Any]],
        *,
        source_type: str,
        target_type: str,
        source_field: str = "src",
        target_field: str = "dst",
        properties: list[str] | None = None,
    ) -> int:
        """Add edges from an iterable of dicts containing source+target ids
        and any per-edge properties. By default, all non-id columns become
        edge properties; pass `properties=[...]` to restrict.

        kglite doesn't enforce edge idempotency at the bulk API level —
        avoid duplicate rows yourself.
        """
        rows = list(rows)
        if not rows:
            return 0
        df = pd.DataFrame(rows)
        kwargs: dict[str, Any] = {
            "source_type": source_type,
            "target_type": target_type,
            "source_id_field": source_field,
            "target_id_field": target_field,
        }
        if properties is not None:
            kwargs["columns"] = list(properties)
        else:
            # Auto-detect property columns: anything that's not src/dst
            prop_cols = [c for c in df.columns if c not in (source_field, target_field)]
            if prop_cols:
                kwargs["columns"] = prop_cols
        self.g.add_connections(df, edge_type, **kwargs)
        return len(rows)

    # ─── labels (kglite 0.10.5+) ──────────────────────────────────────────

    def add_label(
        self, node_type: str, ids: list[str] | str, label: str,
    ) -> None:
        """Attach a secondary label to nodes by id. No-op if `label` is
        empty or `ids` is empty."""
        if not label or not ids:
            return
        if isinstance(ids, str):
            ids = [ids]
        self.g.add_label(node_type, ids, label)

    def remove_label(
        self, node_type: str, ids: list[str] | str, label: str,
    ) -> None:
        """Remove a secondary label. Safe to call when the label isn't
        present (kglite skips the no-op)."""
        if not label or not ids:
            return
        if isinstance(ids, str):
            ids = [ids]
        self.g.remove_label(node_type, ids, label)

    def swap_label(
        self,
        node_type: str,
        ids: list[str] | str,
        *,
        remove: str = "",
        add: str = "",
        remove_any_of: tuple[str, ...] = (),
    ) -> None:
        """State-transition primitive: remove one (or any of N) labels,
        then add a new one. Used by every lifecycle move
        (Unverified → Verified, NeedsOcr → Ready, New → InReview, …).

        Pass `remove_any_of=(...)` when the current label is "one of these
        five mutually-exclusive states" and you want to drop whichever
        is currently set before adding the new one. Cheaper than a
        per-node lookup, since `remove_label` is a no-op when the label
        isn't present.
        """
        if isinstance(ids, str):
            ids = [ids]
        if not ids:
            return
        if remove:
            self.remove_label(node_type, ids, remove)
        for r in remove_any_of:
            if r and r != add:
                self.remove_label(node_type, ids, r)
        if add:
            self.add_label(node_type, ids, add)

    def node_labels(self, node_type: str, node_id: str) -> list[str]:
        """All labels on a node (primary + secondaries). Useful for
        tests + diagnostics; not used in hot paths."""
        df = self.g.cypher(
            f"MATCH (n:{node_type} {{id: $id}}) RETURN labels(n) AS labels",
            params={"id": node_id},
        )
        out = rows(df)
        return list(out[0]["labels"]) if out and out[0].get("labels") else []

    # ─── embeddings ────────────────────────────────────────────────────────

    def set_embeddings(
        self,
        node_type: str,
        text_column: str,
        embeddings: Mapping[Any, list[float]],
        *,
        metric: str = "cosine",
    ) -> dict[str, Any]:
        """Replace the entire embedding store for `(node_type, text_column)`.
        Most callers want `add_embeddings` (which merges with what's already
        there) — this method exists for explicit reset scenarios."""
        return self.g.set_embeddings(node_type, text_column, dict(embeddings), metric=metric)

    def add_embeddings(
        self,
        node_type: str,
        text_column: str,
        embeddings: Mapping[Any, list[float]],
        *,
        metric: str = "cosine",
    ) -> dict[str, Any]:
        """Merge new embeddings into the existing store for
        `(node_type, text_column)`. kglite's `set_embeddings` is a full
        replace; this wrapper pulls the current set, layers the new
        entries on top, and writes back. Adds (or overwrites) per-id.
        """
        if not embeddings:
            return {"embeddings_stored": 0, "dimension": 0, "skipped": 0}
        merged: dict[Any, list[float]] = {}
        try:
            existing = self.g.embeddings(node_type, text_column)
            if existing:
                merged.update(existing)
        except Exception:
            pass
        merged.update(embeddings)
        return self.g.set_embeddings(node_type, text_column, merged, metric=metric)

    def vector_search(
        self,
        node_type: str,
        text_column: str,
        query_vec: list[float],
        *,
        top_k: int = 10,
        metric: str = "cosine",
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run vector_search over `node_type`, optionally narrowed by simple
        equality filters via the fluent `.where({col: val})` chain."""
        sel = self.g.select(node_type)
        if filters:
            sel = sel.where(filters)
        return list(sel.vector_search(text_column, query_vec, top_k=top_k, metric=metric))

    # ─── cypher passthrough ───────────────────────────────────────────────

    def cypher(self, query: str, params: dict[str, Any] | None = None) -> Any:
        return self.g.cypher(query, params=params or {})

    def schema(self) -> dict[str, Any]:
        return self.g.schema()

    def describe(self, **kwargs: Any) -> Any:
        return self.g.describe(**kwargs)


def rows(df: Any) -> list[dict[str, Any]]:
    """Coerce a kglite ResultView (or pandas/polars DataFrame) into a list
    of dicts. kglite returns `ResultView` from `.cypher(...)`, which
    exposes `.to_list()` (not `.to_dicts()`)."""
    if df is None:
        return []
    if hasattr(df, "to_list"):
        try:
            out = df.to_list()
            if isinstance(out, list):
                return out
        except Exception:
            pass
    if hasattr(df, "to_dicts"):
        return list(df.to_dicts())
    if hasattr(df, "to_dict"):
        return list(df.to_dict(orient="records"))
    return []
