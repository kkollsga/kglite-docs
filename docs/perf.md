# Performance audit

A quick snapshot of where the time goes in kglite-docs, what's been optimised, and what's left as a lever.

## Quick numbers (local M-series Mac, CPU-only ONNX)

| Operation | Typical cost |
|---|---|
| First-call bge-m3 warm-up | ~1s |
| Embed batch of 50 chunks (cold) | ~3-5s |
| Embed batch of 50 chunks (warm) | ~1-2s |
| Ingest one ~10-page arXiv PDF | ~5-10s |
| Ingest 16 arXiv PDFs (cold) | ~3-5 min |
| `search()` over 1k chunks | ~3-5ms |
| `cluster_chunks(kmeans, k=8)` on 1k chunks | ~200ms |
| `add_summary` (with embed) | ~30-60ms |
| Cypher `MATCH n WHERE n.x = $v RETURN n.y` (1k nodes) | ~1-2ms |

## What's been optimised

- **Batch embedding per document** — `ingest_pipeline` calls `embedder.embed([...])` once per document with all its chunks, not per-chunk.
- **Embedder cool-down (inherited from kglite)** — the ONNX session stays resident for 15 min after the last call. Re-ingests + searches in a session stay warm at ~50ms / batch.
- **Tokenizer caching** — `_bge_m3_tokenizer()` is `lru_cache`-d so the bpe vocab loads once per process.
- **Add-merge for embeddings** — kglite's `set_embeddings` is a full-replace. We added `Store.add_embeddings()` which pulls the existing dict, layers new entries, and writes back. So incremental ingest doesn't lose prior embeddings.
- **Idempotent ingest** — re-ingesting an unchanged file (same sha256) short-circuits early; no re-parse, no re-embed.
- **Bulk DataFrame upserts** — every node + edge insert batches all rows in one `add_nodes` / `add_connections` call; we never call kglite with one row at a time inside a hot loop.

## Open levers (not yet pulled)

- **Parallel ingest across documents** — currently sequential. With a ProcessPoolExecutor each worker would need its own embedder (no shared GIL-released ORT session), so memory triples per worker. v0.2 candidate.
- **GPU-accelerated bge-m3** — drop in `CoreMLExecutionProvider` (macOS) or `CUDAExecutionProvider` (linux). The bottleneck is so dominantly the embed step that a 5x GPU speedup buys ~40% of the wall-clock back.
- **HNSW-style vector index** — kglite's `vector_search` is brute-force over the current selection. Fine to ~100K chunks; past that, push for an HNSW index upstream.
- **Cluster on streaming inputs** — `cluster_chunks` currently pulls all embeddings into numpy. For 1M+ chunks, use mini-batch k-means.
- **Page-render lazy URL** — `list_pending_ocr` returns base64 PNG inline. For high-volume scanned corpora, write to disk and return URLs.

## How to benchmark

```bash
.venv/bin/python -X importtime -c "import kglite_docs" 2>import.log
.venv/bin/python -m cProfile -o profile.out -m kglite_docs.cli ingest sample_data/pdfs --db bench.kgl
.venv/bin/python -c "import pstats; p = pstats.Stats('profile.out'); p.sort_stats('cumulative').print_stats(40)"
```

Inside kglite, the `g.profile()` method emits per-pass Cypher timings — use it whenever you write a non-trivial query.
