# Getting started

A 10-minute walkthrough from `pip install` to "I'm using this from an agent."

## 1. Install

```bash
pip install "kglite-docs[mcp]"
```

This pulls in everything the library needs: storage (`kglite`), multi-format parsers (`pymupdf4llm`, `python-docx`, `python-pptx`, `markdownify`), the bge-m3 inference stack (`tokenizers`, `onnxruntime`, `huggingface-hub`), and the MCP framework (`mcp`, `mcp-methods`).

On first use the bge-m3 ONNX weights (~2 GB) are pulled from HuggingFace Hub into `~/.cache/fastembed/`. If you already have them cached elsewhere, set `HF_HUB_CACHE=/path/to/cache` before running anything.

## 2. Build your first corpus

```python
from kglite_docs import Corpus

corpus = Corpus.create("kb.kgl")        # creates the file
corpus.ingest("paper.pdf")              # PDF, DOCX, PPTX, MD, HTML, TXT, or images
corpus.ingest_dir("./more-papers")      # bulk ingest
corpus.save()                            # persists to kb.kgl
```

Behind the scenes for every document we:

1. Hash the file bytes → `doc_id` (idempotent re-ingest).
2. Extract text per page with `pymupdf4llm` (PDF) or the format-specific parser.
3. Detect scanned/image-only pages and mark them `needs_ocr=True`.
4. Chunk each page into ~512-token windows on paragraph + heading boundaries; never cross page boundaries.
5. Embed each chunk with BAAI/bge-m3 (CLS-pooled, 1024-dim).
6. Insert nodes + edges in one transactional batch and persist.

## 3. Search

```python
hits = corpus.search("transformer attention mechanism", top_k=5)
for h in hits:
    print(f"[{h['score']:.2f}] p.{h['page']} — {h['text'][:120]}")
```

Pass `filters={"doc_id": "..."}` to restrict the search to one document, or `with_summaries=True` to inline any verified summaries on each hit. Passing `agent_id="me"` records a view, bumping the chunk's `view_count` so you can ask "what's been most consulted?" later.

## 4. Compose a prompt-ready context bundle

```python
ctx = corpus.compose_context(
    "compare DPR and ColBERT on TREC", max_tokens=4000
)
# ctx["items"] is the budgeted, ranked set of chunks + verified summaries
```

Drop `ctx["items"]` straight into your LLM prompt — chunk ids are included so the model can cite them in its response.

## 5. Write summaries with cross-checking

```python
sid = corpus.add_summary(
    target_id=hits[0]["id"], target_kind="Chunk",
    text="DPR uses a dual BERT encoder; ColBERT keeps token-level vectors.",
    agent_id="claude-alice", model="sonnet-4.6",
)
# A second agent verifies. Self-verification is rejected.
corpus.verify_summary(
    sid, verdict="verified",
    verifier_agent_id="claude-bob", notes="cross-checked p.2 + p.5",
)
```

Verdicts: `verified`, `disputed`, `needs_revision`. Status moves to `stale` automatically when the underlying chunk text changes (re-ingest of a modified document).

## 6. Tag and discover

```python
corpus.tag_chunk(hits[0]["id"], "dual-encoder", agent_id="claude-alice")
# Multiple agents can tag the same chunk; tags are tracked per-agent
corpus.list_tags(chunk_id=hits[0]["id"])
corpus.chunks_by_tag("dual-encoder", limit=20)
```

## 7. Cluster + back-reference

```python
corpus.cluster_chunks(algorithm="kmeans", params={"k": 8})
for cl in corpus.cluster_overview()[:3]:
    detail = corpus.get_cluster(cl["id"], top_terms=10)
    print(detail["id"], detail["top_terms"])
```

`get_cluster()` returns the members + lexical top-terms. Pair it with `compose_context` to write a synthesis article that cites the right chunks.

## 8. Quality gates against hallucination

```python
# How well does the summary's text actually align with its source chunk(s)?
report = corpus.check_grounding(sid, threshold=0.5)
print(f"supported: {report['supported_fraction']:.0%}")
for weak in report["weak_sentences"]:
    print(f"  weak: {weak['sentence']}")

# Free-text claim: where in the corpus does this come from?
v = corpus.verify_claim("ColBERT uses MaxSim scoring", top_k=5)
for s in v["support"]:
    print(s["score"], s["doc_id"], s["page"], s["text"][:80])
```

Both methods are cheap baselines (embedding similarity, not a full NLI model), but they surface obviously ungrounded claims for human or agent review.

## 9. Translate

```python
tid = corpus.add_translation(
    hits[0]["id"], "no",
    "DPR bruker en BERT-basert tokoder.",
    agent_id="claude-translator",
)
# Second pass reviews
corpus.mark_translation_reviewed(tid, reviewer_agent_id="claude-translator-2")
# Stitch a target-language document
nor = corpus.assemble_translated_document(hits[0]["doc_id"], target_lang="no")
print(f"coverage: {nor['coverage']:.0%}")
```

## 10. Export

```python
# A document — or a cluster, or a summary — as MD / DOCX / PDF
corpus.export_document(hits[0]["doc_id"], "out.docx", include_summaries=True)
corpus.export_cluster(cluster_id, "cluster.pdf")
# Bundle several into one deliverable
corpus.export_bundle(
    [
        {"kind": "markdown", "text": "## Background\n\nMy notes…"},
        {"kind": "cluster", "id": cluster_id},
        {"kind": "doc", "id": hits[0]["doc_id"]},
    ],
    "synthesis.pdf",
    title="My synthesis",
)
```

## 11. Run it for an agent over MCP

```bash
kglite-docs-mcp --db kb.kgl
```

Or register with Claude Code:

```bash
claude mcp add kglite-docs -- kglite-docs-mcp --db /abs/path/kb.kgl
```

The agent now sees typed tools (`search`, `compose_context`, `add_summary`, …) plus the `cypher_query` escape hatch for power use. Self-verification is enforced server-side; staleness is auto-flipped on re-ingest.

## Next steps

- **`docs/architecture.md`** — graph model, why kglite, where the bottlenecks are.
- **`docs/workflows.md`** — agent-driven patterns: research, comparison, fact-checking.
- **`demos/workflow.py`** — full end-to-end Sonnet workflow (ingest → cluster → summarise → article → fact-check).
- **`docs/contributing.md`** — running the test suite, releasing.
