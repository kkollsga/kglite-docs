# kglite-docs

**Agent-first knowledge base for documents.** Ingest PDFs, DOCX, PPTX, Markdown, HTML, plain text, or images; chunk + embed them with BAAI/bge-m3; cluster, tag, summarise, fact-check, and translate them; serve the whole thing to AI agents over MCP. Built on [kglite](https://github.com/kkollsga/kglite) (the storage / vector / clustering engine) and [mcp-methods](https://github.com/kkollsga/mcp-methods) (the MCP framework).

## Why another RAG library?

Most "RAG libraries" hand the agent `search(query) → list[chunk]` and call it a day. kglite-docs treats the corpus as a *living* knowledge graph that records who did what:

- **Documents → pages → chunks** keyed by file hash + page + position
- **Agents** are first-class nodes; their views, tags, summaries, and verifications are all queryable
- **Summaries with cross-checking** — one agent writes, a *different* agent verifies (self-verification is server-rejected)
- **Translations** as their own nodes, so multiple translators can co-exist with provenance
- **Clusters + back-references** so an article generated from a cluster cites the chunks it came from
- **Grounding checks** — built-in tools to score how well an agent's summary aligns with its sources, surfacing weak claims before they ship
- **Typed MCP tools** (no Cypher required for the common case) **+ Cypher escape hatch** when you need it

## Quick install

```bash
pip install "kglite-docs[mcp]"
```

This pulls in:
- `kglite` — Rust-backed graph storage + native vector search + clustering procedures (saves to a single `.kgl` file)
- `pymupdf4llm` + `pymupdf` + `python-docx` + `python-pptx` + `markdownify` — multi-format parsing
- `tokenizers` + `onnxruntime` + `huggingface-hub` — BAAI/bge-m3 inference (CLS-pooled, 1024-dim, 8192-token cap)
- `mcp` + `mcp-methods` — MCP server framework

On first use the bge-m3 ONNX weights are downloaded to `~/.cache/fastembed/` (~2 GB, one-time).

## 60-second quickstart (Python)

```python
from kglite_docs import Corpus

# Open or create a knowledge base (single .kgl file on disk)
corpus = Corpus.create("kb.kgl")

# Ingest anything in a folder — pdf, docx, pptx, md, html, txt, png/jpg
corpus.ingest_dir("./papers")
# Or one at a time
corpus.ingest("paper.pdf")
corpus.ingest("notes.md")

# Semantic search with optional metadata filters
hits = corpus.search("transformer attention mechanism", top_k=5,
                     filters={"doc_id": "doc_abc..."})
for h in hits:
    print(f"[{h['score']:.2f}] p.{h['page']} — {h['text'][:120]}")

# Compose a budgeted context bundle for a downstream LLM call
ctx = corpus.compose_context("RAG vs fine-tuning", max_tokens=4000)
# ctx["items"] is a ranked list of (chunk_id, text, page, score, …) within budget

# Agent enrichment: one agent writes, another verifies
sid = corpus.add_summary(
    target_id=hits[0]["id"], target_kind="Chunk",
    text="DPR uses a dual BERT encoder with in-batch negatives.",
    agent_id="claude-alice", model="sonnet-4.6",
)
corpus.verify_summary(sid, verdict="verified",
                      verifier_agent_id="claude-bob",
                      notes="checked against page 5")

# Tags (multiple agents can tag the same chunk distinctly)
corpus.tag_chunk(hits[0]["id"], "dual-encoder", agent_id="claude-alice")

# Grounding check — surface ungrounded claims in a summary
report = corpus.check_grounding(sid, threshold=0.5)
print(f"supported {report['supported_fraction']:.0%}")

# Cluster + inspect
corpus.cluster_chunks(algorithm="kmeans", params={"k": 8})
for c in corpus.cluster_overview()[:5]:
    print(c["id"], c["actual_size"])

# Translations
tid = corpus.add_translation(hits[0]["id"], "no",
                             "DPR bruker en BERT-basert tokoder.",
                             agent_id="claude-translator")
doc_no = corpus.assemble_translated_document(hits[0]["doc_id"], target_lang="no")
print(f"translation coverage: {doc_no['coverage']:.0%}")

# Export to MD / DOCX / PDF
corpus.export_document(hits[0]["doc_id"], "out.docx")
corpus.export_cluster(cluster_id, "cluster.pdf")

corpus.save()
```

## Run as an MCP server

```bash
kglite-docs-mcp --db kb.kgl
```

Register with Claude Code:

```bash
claude mcp add kglite-docs -- kglite-docs-mcp --db /abs/path/kb.kgl
```

The agent now sees these tools (all typed; no Cypher required for the common path):

| Category | Tools |
|---|---|
| **Discovery** | `list_documents`, `get_document`, `graph_overview` |
| **Search** | `search`, `similar_chunks`, `get_chunk`, `compose_context` |
| **Enrichment** | `add_summary`, `verify_summary`, `link_verification`, `get_summaries`, `find_consensus` |
| **Tags / activity** | `tag_chunk`, `untag_chunk`, `list_tags`, `chunks_by_tag`, `record_view`, `list_agents` |
| **OCR** | `list_pending_ocr`, `submit_ocr` |
| **Clustering** | `cluster_chunks`, `get_cluster`, `cluster_overview` |
| **Translation** | `add_translation`, `get_translations`, `assemble_translated_document` |
| **Export** | `export_document`, `export_cluster` |
| **Quality** | `check_grounding`, `verify_claim` |
| **Escape hatch** | `cypher_query`, `graph_overview` (via `mcp-methods`) |

## CLI

```bash
kglite-docs ingest ./papers --db kb.kgl --recursive
kglite-docs search "dense retrieval" --db kb.kgl --top-k 5
kglite-docs list --db kb.kgl
kglite-docs cluster --db kb.kgl --algorithm louvain
kglite-docs show doc <doc_id> --db kb.kgl
```

## Graph model in one picture

```
(Document)──HAS_PAGE──→(Page)──HAS_CHUNK──→(Chunk)──NEXT_CHUNK──→(Chunk)
                                              │
                                              ├──IN_CLUSTER──→(Cluster)
                                              ├──TAGGED_AS──→(Tagging)──OF_TAG──→(Tag)
                                              ├──SUMMARIZES←──(Summary)──VERIFIED_BY──→(Agent)
                                              ├──HAS_TRANSLATION──→(Translation)
                                              └─←─VIEWED──(Agent)
```

Each agent that interacts with the corpus is its own `Agent` node, and *every* mutation (`add_summary`, `tag_chunk`, `add_translation`, `submit_ocr`) records the agent. Views from `search` / `get_chunk` (with `agent_id`) bump per-chunk view counters and optionally create `View` nodes when context is worth recording.

Three independent embedding stores live on the graph:

- `(Chunk, text)` — primary semantic search
- `(Summary, text)` — find agreement/disagreement across enrichments
- `(Document, title)` — fast doc-level routing

## Multi-format ingestion

All formats produce the same downstream `Document → Page → Chunk` structure, so the rest of the pipeline doesn't care what the source was:

| Format | Extension | Pagination |
|---|---|---|
| PDF | `.pdf` | Real pages |
| Word | `.docx` | One page per top-level heading (or whole doc) |
| PowerPoint | `.pptx` | One page per slide |
| Markdown | `.md`, `.markdown` | One page per top-level H1 |
| HTML | `.html`, `.htm` | HTML → markdown → split on H1 |
| Text | `.txt` | One page |
| Images | `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.webp` | One page, flagged `needs_ocr=True` |

Image-only pages route through the **agent OCR loop**: ingest marks them `needs_ocr=True`, `list_pending_ocr()` returns the page with a base64 PNG render, the agent reads it and calls `submit_ocr(page_id, markdown, ...)` which re-chunks and re-embeds in place.

## Agent workflows

`kglite-docs.agents.call_agent()` is a small abstraction over Sonnet/Opus calls — uses the Anthropic SDK if `ANTHROPIC_API_KEY` is set, else falls back to the `claude -p` CLI. See `demos/workflow.py` for an end-to-end pipeline that ingests, clusters, asks Sonnet to summarise + tag the most-connected cluster, drafts an article with back-references, then runs a second Sonnet pass to fact-check it.

```bash
python demos/workflow.py --db demo.kgl --pdfs ./papers
```

## Tests

```bash
pytest                       # full suite
pytest -m "not embed"         # skip tests that need bge-m3 weights
pytest tests/test_quality.py  # one file
```

The default suite uses a deterministic stub embedder so it runs in under a second. Tests marked `@pytest.mark.embed` pull bge-m3 ONNX weights on first run and cache them.

## License

MIT.
