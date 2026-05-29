"""Quality / grounding checks for agent-derived data.

Tools here help reduce hallucinations in two ways:

1. **Grounding check** — given a summary, measure how strongly each
   sentence of the summary is supported by the source chunks. A
   sentence whose best cosine similarity to *any* source span is below
   a threshold is flagged as potentially ungrounded.
2. **Claim verification** — given a free-text claim and a set of
   chunks, return the chunks that best support (or refute) it, with
   similarity scores. Lets a downstream agent fact-check without
   guessing where to look.

These are *baseline* tools — embedding-similarity is a proxy, not
truth. They surface weak grounding for human/agent review; they don't
replace it.
"""

from __future__ import annotations

import re
from typing import Any

from kglite_docs.schema import CHUNK, CHUNK_TEXT_COL
from kglite_docs.store import Store

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


from kglite_docs.store import rows as _df_dicts  # noqa: E402


def split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter. Avoids spaCy/NLTK for install footprint."""
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def _cosine(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def check_grounding(
    store: Store,
    embedder: Any,
    *,
    summary_id: str,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """For each sentence of the summary, find the best-supporting chunk
    (among the chunks the summary targets, plus their NEXT_CHUNK
    neighbours). Sentences with max similarity < `threshold` are flagged.
    """
    s_rows = _df_dicts(store.cypher(
        "MATCH (s:Summary {id: $sid}) "
        "RETURN s.text AS text, s.target_id AS target_id, s.target_kind AS target_kind",
        params={"sid": summary_id},
    ))
    if not s_rows:
        raise ValueError(f"summary not found: {summary_id}")
    s = s_rows[0]

    # Pull source chunks
    sources = _resolve_source_chunks(store, s["target_id"], s["target_kind"])
    if not sources:
        return {
            "summary_id": summary_id, "sentences": [], "supported_fraction": 0.0,
            "grounding_score": 0.0, "weak_sentences": [],
        }

    src_texts = [c["text"] for c in sources if c.get("text")]
    src_ids = [c["id"] for c in sources if c.get("text")]
    if not src_texts:
        return {
            "summary_id": summary_id, "sentences": [], "supported_fraction": 0.0,
            "grounding_score": 0.0, "weak_sentences": [],
        }

    sentences = split_sentences(s["text"])
    if not sentences:
        return {
            "summary_id": summary_id, "sentences": [], "supported_fraction": 1.0,
            "grounding_score": 1.0, "weak_sentences": [],
        }

    # Embed sentences + source texts (or pull pre-computed chunk embeddings)
    pre_vecs = store.g.embeddings(CHUNK, "text")
    src_vecs: list[list[float] | None] = []
    missing_idx = []
    for i, cid in enumerate(src_ids):
        if cid in pre_vecs:
            src_vecs.append(pre_vecs[cid])
        else:
            src_vecs.append(None)
            missing_idx.append(i)
    if missing_idx:
        new_vecs = embedder.embed([src_texts[i] for i in missing_idx])
        for i, v in zip(missing_idx, new_vecs, strict=False):
            src_vecs[i] = v

    sent_vecs = embedder.embed(sentences)

    out: list[dict[str, Any]] = []
    supported = 0
    total_score = 0.0
    for sent, sv in zip(sentences, sent_vecs, strict=False):
        best_score = -1.0
        best_cid = ""
        for cid, srcv in zip(src_ids, src_vecs, strict=False):
            score = _cosine(sv, srcv)  # type: ignore[arg-type]
            if score > best_score:
                best_score = score
                best_cid = cid
        if best_score >= threshold:
            supported += 1
        total_score += max(best_score, 0.0)
        out.append({
            "sentence": sent,
            "best_chunk_id": best_cid,
            "best_score": float(best_score),
            "supported": best_score >= threshold,
        })

    return {
        "summary_id": summary_id,
        "sentences": out,
        "supported_fraction": supported / len(sentences),
        "grounding_score": total_score / len(sentences),
        "weak_sentences": [o for o in out if not o["supported"]],
        "threshold": threshold,
    }


def verify_claim(
    store: Store,
    embedder: Any,
    *,
    claim_text: str,
    against_chunk_ids: list[str] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Find the chunks that best support a free-text claim.

    If `against_chunk_ids` is given, similarity is computed only over
    those chunks. Otherwise the claim is run against the whole chunk
    embedding store via vector_search.
    """
    if against_chunk_ids:
        rows = _df_dicts(store.cypher(
            f"MATCH (c:Chunk) WHERE c.id IN $ids "
            f"RETURN c.id AS id, c.{CHUNK_TEXT_COL} AS text, c.doc_id AS doc_id, c.page_number AS page",
            params={"ids": against_chunk_ids},
        ))
        if not rows:
            return {"claim": claim_text, "support": []}
        pre_vecs = store.g.embeddings(CHUNK, "text")
        claim_vec = embedder.embed([claim_text])[0]
        scored: list[dict[str, Any]] = []
        for r in rows:
            v = pre_vecs.get(r["id"])
            if v is None:
                v = embedder.embed([r["text"]])[0]
            score = _cosine(claim_vec, v)
            scored.append({**r, "score": float(score)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return {"claim": claim_text, "support": scored[:top_k]}
    # Whole-corpus vector search
    vec = embedder.embed([claim_text])[0]
    hits = store.vector_search(CHUNK, CHUNK_TEXT_COL, vec, top_k=top_k)
    # Hydrate with text
    if hits:
        ids = [h["id"] for h in hits]
        df = _df_dicts(store.cypher(
            f"MATCH (c:Chunk) WHERE c.id IN $ids "
            f"RETURN c.id AS id, c.{CHUNK_TEXT_COL} AS text, c.doc_id AS doc_id, c.page_number AS page",
            params={"ids": ids},
        ))
        text_by_id = {r["id"]: r for r in df}
        for h in hits:
            h.update({k: v for k, v in text_by_id.get(h["id"], {}).items() if k != "id"})
    return {"claim": claim_text, "support": hits}


def _resolve_source_chunks(
    store: Store, target_id: str, target_kind: str
) -> list[dict[str, Any]]:
    if target_kind == CHUNK:
        df = _df_dicts(store.cypher(
            f"MATCH (c:Chunk {{id: $id}}) RETURN c.id AS id, c.{CHUNK_TEXT_COL} AS text",
            params={"id": target_id},
        ))
        return df
    if target_kind == "Document":
        df = _df_dicts(store.cypher(
            f"MATCH (d:Document {{id: $id}})-[:HAS_CHUNK]->(c:Chunk) "
            f"RETURN c.id AS id, c.{CHUNK_TEXT_COL} AS text ORDER BY c.page_number, c.chunk_index",
            params={"id": target_id},
        ))
        return df
    if target_kind == "Page":
        df = _df_dicts(store.cypher(
            f"MATCH (p:Page {{id: $id}})-[:HAS_CHUNK]->(c:Chunk) "
            f"RETURN c.id AS id, c.{CHUNK_TEXT_COL} AS text ORDER BY c.chunk_index",
            params={"id": target_id},
        ))
        return df
    return []
