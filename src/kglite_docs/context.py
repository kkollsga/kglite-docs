"""Compose a budgeted prompt-ready context bundle for a query.

Workflow: vector_search for top chunks → optionally pull verified
summaries on each → pack greedy until `max_tokens` is hit. Returns a
dict the agent can drop straight into a system/user prompt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kglite_docs.ingest.chunker import count_tokens

if TYPE_CHECKING:
    from kglite_docs.corpus import Corpus


def compose_context(
    corpus: "Corpus",
    *,
    query: str,
    max_tokens: int = 4000,
    per_doc_cap: int | None = None,
    include_summaries: bool = True,
    agent_id: str | None = None,
) -> dict[str, Any]:
    """Return a packed context bundle. Structure::

        {
          "query": ...,
          "budget_tokens": ...,
          "used_tokens": ...,
          "items": [
            {
              "chunk_id": ..., "doc_id": ..., "page": ...,
              "text": ..., "summaries": [...], "score": ...,
              "tokens": ...
            },
            ...
          ],
        }
    """
    hits = corpus.search(query, top_k=max(20, max_tokens // 200), agent_id=agent_id,
                         with_summaries=include_summaries)
    items: list[dict[str, Any]] = []
    used = 0
    per_doc: dict[str, int] = {}
    for h in hits:
        text = h.get("text") or ""
        if not text:
            continue
        doc_id = h.get("doc_id") or ""
        if per_doc_cap is not None and per_doc.get(doc_id, 0) >= per_doc_cap:
            continue
        # Build the rendered text including summaries if present
        rendered = text
        sums = h.get("summaries") or []
        verified_sums = [s for s in sums if s.get("status") == "verified"]
        if verified_sums:
            rendered += "\n\n[verified summary: " + verified_sums[0]["text"] + "]"
        tokens = count_tokens(rendered)
        if used + tokens > max_tokens:
            continue
        items.append({
            "chunk_id": h["id"],
            "doc_id": doc_id,
            "page": h.get("page"),
            "headings": h.get("headings"),
            "score": h.get("score"),
            "text": text,
            "summaries": verified_sums,
            "tokens": tokens,
        })
        used += tokens
        per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
    return {
        "query": query,
        "budget_tokens": max_tokens,
        "used_tokens": used,
        "items": items,
    }
