"""Cluster chunks via kglite's `CALL cluster()` / `CALL louvain()` /
`CALL connected_components()` procedures.

Each clustering run creates a `Cluster` node per community and writes
`IN_CLUSTER` edges from chunks to their cluster. Re-clustering does NOT
delete prior runs by default — they're labelled with a run id so you
can keep multiple views simultaneously.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from kglite_docs.schema import CHUNK, CLUSTER, IN_CLUSTER
from kglite_docs.store import Store


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


from kglite_docs.store import rows as _df_dicts  # noqa: E402


def cluster_chunks(
    store: Store,
    *,
    algorithm: str = "louvain",
    params: dict[str, Any] | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Run a clustering algorithm over the chunk embeddings.

    Supported algorithms:
    - ``louvain`` — community detection on a chunk-similarity graph
      derived from embeddings.
    - ``kmeans`` — k-means clustering on the chunk embedding vectors.
    - ``dbscan`` — density-based clustering on the chunk embedding vectors.

    Returns a dict with the run id, the number of clusters, and a
    member summary.
    """
    params = dict(params or {})
    run_id = f"cluster_run_{uuid.uuid4().hex[:12]}"
    created_at = _now()

    if algorithm == "louvain":
        # Use kglite's louvain over the SIMILAR_TO graph if present, else
        # fall back to an embedding-driven approach via .compare()/cluster.
        try:
            rows = _df_dicts(store.cypher("CALL louvain() YIELD node, community RETURN node.id AS chunk_id, community AS community"))
        except Exception:
            # Fall back to embedding-based clustering with k=auto
            rows = _embedding_cluster(store, algorithm="louvain", **params)
    elif algorithm in ("kmeans", "dbscan"):
        rows = _embedding_cluster(store, algorithm=algorithm, **params)
    else:
        raise ValueError(f"unsupported algorithm: {algorithm}")

    # Group by community
    by_community: dict[Any, list[str]] = {}
    for r in rows:
        cid = r.get("chunk_id") or r.get("id")
        comm = r.get("community") or r.get("cluster") or r.get("label")
        if cid is None or comm is None:
            continue
        by_community.setdefault(comm, []).append(cid)

    # Create one Cluster node per community
    cluster_rows: list[dict[str, Any]] = []
    cluster_to_chunks: list[tuple[str, str]] = []
    for comm, members in by_community.items():
        if not members:
            continue
        cluster_id = f"{run_id}__c{comm}"
        cluster_rows.append({
            "id": cluster_id,
            "title": f"{algorithm}#{comm} ({len(members)} chunks)",
            "run_id": run_id,
            "algorithm": algorithm,
            "params_json": _safe_json(params),
            "created_at": created_at,
            "note": note,
            "size": len(members),
        })
        for cid in members:
            cluster_to_chunks.append((cid, cluster_id))

    store.upsert_nodes(CLUSTER, cluster_rows)
    store.upsert_edges(
        IN_CLUSTER,
        [{"src": cid, "dst": cluster_id} for cid, cluster_id in cluster_to_chunks],
        source_type=CHUNK, target_type=CLUSTER,
    )
    return {
        "run_id": run_id,
        "algorithm": algorithm,
        "clusters": len(cluster_rows),
        "members": len(cluster_to_chunks),
    }


def _embedding_cluster(
    store: Store, *, algorithm: str, **params: Any
) -> list[dict[str, Any]]:
    """Pull chunk embeddings, run sklearn-style clustering in numpy, return rows."""
    import numpy as np

    embs = store.g.embeddings(CHUNK, "text")
    if not embs:
        return []
    ids = list(embs.keys())
    X = np.array([embs[i] for i in ids], dtype=np.float32)

    if algorithm in ("louvain",):
        # crude fallback: k-means with k = sqrt(n)
        k = max(2, int(np.sqrt(len(ids))))
        labels = _kmeans(X, k=k)
    elif algorithm == "kmeans":
        k = int(params.get("k", max(2, int(np.sqrt(len(ids))))))
        labels = _kmeans(X, k=k, max_iter=int(params.get("max_iter", 50)))
    elif algorithm == "dbscan":
        eps = float(params.get("eps", 0.5))
        min_samples = int(params.get("min_samples", 3))
        labels = _dbscan(X, eps=eps, min_samples=min_samples)
    else:
        raise ValueError(f"unsupported algorithm: {algorithm}")
    return [{"chunk_id": ids[i], "community": int(labels[i])} for i in range(len(ids))]


def _kmeans(X: Any, *, k: int, max_iter: int = 50, seed: int = 0) -> Any:
    import numpy as np
    rng = np.random.default_rng(seed)
    n, d = X.shape
    if k >= n:
        return np.arange(n)
    # k-means++ init
    idx = [int(rng.integers(0, n))]
    for _ in range(1, k):
        d2 = ((X - X[idx[-1]]) ** 2).sum(axis=1)
        for j in idx:
            d2 = np.minimum(d2, ((X - X[j]) ** 2).sum(axis=1))
        probs = d2 / d2.sum() if d2.sum() > 0 else np.ones(n) / n
        idx.append(int(rng.choice(n, p=probs)))
    centroids = X[idx].copy()
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = dists.argmin(axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                centroids[j] = members.mean(axis=0)
    return labels


def _dbscan(X: Any, *, eps: float, min_samples: int) -> Any:
    import numpy as np
    n = len(X)
    labels = -np.ones(n, dtype=np.int32)
    cluster_id = 0
    # Precompute neighbors (cosine distance — assumes vectors are
    # roughly L2-normalised; bge-m3 outputs aren't normalised by default
    # so we normalise here).
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    sim = X_norm @ X_norm.T  # cosine similarity
    dist = 1 - sim
    neighbors = [np.where(dist[i] < eps)[0] for i in range(n)]
    for i in range(n):
        if labels[i] != -1:
            continue
        if len(neighbors[i]) < min_samples:
            labels[i] = -2  # mark noise
            continue
        cluster_id += 1
        seeds = list(neighbors[i])
        labels[i] = cluster_id
        k = 0
        while k < len(seeds):
            j = seeds[k]
            if labels[j] == -2:
                labels[j] = cluster_id
            elif labels[j] == -1:
                labels[j] = cluster_id
                if len(neighbors[j]) >= min_samples:
                    for nbr in neighbors[j]:
                        if nbr not in seeds:
                            seeds.append(nbr)
            k += 1
    # Re-label noise (-2) as -1 (the standard DBSCAN noise label)
    labels[labels == -2] = -1
    return labels


def get_cluster(
    store: Store, *, cluster_id: str, top_terms: int = 10
) -> dict[str, Any] | None:
    rows = _df_dicts(store.cypher(
        "MATCH (cl:Cluster {id: $id}) "
        "RETURN cl.id AS id, cl.title AS title, cl.algorithm AS algorithm, "
        "cl.run_id AS run_id, cl.created_at AS created_at, cl.size AS size, cl.note AS note",
        params={"id": cluster_id},
    ))
    if not rows:
        return None
    cluster = rows[0]
    members = _df_dicts(store.cypher(
        "MATCH (c:Chunk)-[:IN_CLUSTER]->(cl:Cluster {id: $id}) "
        "RETURN c.id AS id, c.doc_id AS doc_id, c.page_number AS page, "
        "c.title AS title, c.text AS text "
        "ORDER BY c.doc_id, c.page_number",
        params={"id": cluster_id},
    ))
    cluster["members"] = members
    cluster["docs"] = sorted({m["doc_id"] for m in members if m.get("doc_id")})
    # Quick lexical top-terms (frequency over member text; cheap baseline)
    cluster["top_terms"] = _top_terms([m["text"] for m in members if m.get("text")], k=top_terms)
    return cluster


def cluster_overview(store: Store) -> list[dict[str, Any]]:
    return _df_dicts(store.cypher(
        "MATCH (cl:Cluster) OPTIONAL MATCH (c:Chunk)-[:IN_CLUSTER]->(cl) "
        "RETURN cl.id AS id, cl.algorithm AS algorithm, cl.run_id AS run_id, "
        "cl.created_at AS created_at, cl.size AS size, count(c) AS actual_size "
        "ORDER BY actual_size DESC"
    ))


def most_connected_cluster(store: Store) -> dict[str, Any] | None:
    """Return the cluster whose members have the highest combined
    in-degree (NEXT_CHUNK + SIMILAR_TO + TAGGED_AS + summaries). A
    pragmatic notion of "connectedness" for our schema."""
    rows = _df_dicts(store.cypher(
        """
        MATCH (cl:Cluster)<-[:IN_CLUSTER]-(c:Chunk)
        OPTIONAL MATCH (c)<-[:SUMMARIZES]-(s:Summary)
        OPTIONAL MATCH (c)-[:TAGGED_AS]->(t:Tag)
        OPTIONAL MATCH (c)-[:NEXT_CHUNK]-(o:Chunk)
        WITH cl, c, count(DISTINCT s) AS s_count, count(DISTINCT t) AS t_count, count(DISTINCT o) AS n_count
        RETURN cl.id AS id, cl.size AS size,
               sum(s_count) AS summaries,
               sum(t_count) AS tags,
               sum(n_count) AS neighbors,
               (sum(s_count) + sum(t_count) + sum(n_count)) AS connectedness
        ORDER BY connectedness DESC
        LIMIT 1
        """
    ))
    return rows[0] if rows else None


_STOP = frozenset(["the", "of", "and", "a", "to", "in", "is", "it", "that", "for", "as", "on", "with", "by", "are", "this", "be", "from", "or", "an", "at", "which", "not", "we", "can", "has", "have", "its", "but", "they", "their", "were", "also", "more", "than", "into", "other", "such", "these", "those", "will", "would", "may", "should", "could"])


def _top_terms(texts: list[str], *, k: int = 10) -> list[tuple[str, int]]:
    import re
    counts: dict[str, int] = {}
    for t in texts:
        for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", t.lower()):
            if w in _STOP:
                continue
            counts[w] = counts.get(w, 0) + 1
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:k]


def _safe_json(obj: object) -> str:
    import json
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return "{}"
