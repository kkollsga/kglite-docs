# kglite-docs

> **Agent-first knowledge base for documents.** Ingest PDFs, Office files, Markdown, HTML, or images; chunk + embed them with [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3); cluster, tag, summarise, fact-check, translate, and review them — and serve the whole thing to AI agents over MCP.

[![PyPI](https://img.shields.io/pypi/v/kglite-docs.svg)](https://pypi.org/project/kglite-docs/)
[![Python](https://img.shields.io/pypi/pyversions/kglite-docs.svg)](https://pypi.org/project/kglite-docs/)
[![Docs](https://readthedocs.org/projects/kglite-docs/badge/?version=latest)](https://kglite-docs.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Built on [`kglite`](https://github.com/kkollsga/kglite) (storage + vector search + clustering) and [`mcp-methods`](https://github.com/kkollsga/mcp-methods) (MCP framework).

---

## Why this and not generic RAG?

Most "RAG libraries" hand the agent `search(query) → list[chunk]` and stop. kglite-docs treats the corpus as a *living* knowledge graph that records who did what — and gives the agent typed tools to act on it.

- 📄 **Multi-format ingest** — PDF, DOCX, PPTX, MD, HTML, TXT, images. All flow into the same `Document → Page → Chunk` shape.
- 🤝 **Agents are first-class nodes** — their views, tags, summaries, verifications, and reviews are all queryable.
- ✅ **Cross-checked summaries** — one agent writes, a *different* agent verifies. Self-verification is rejected server-side.
- 📋 **Review kanban** — chunks move through `new → in_review → reviewed` with an immutable audit trail.
- 🛡️ **Grounding checks** — score how well an agent's summary aligns with its sources. Catch hallucinations before they ship.
- 🌍 **Translations** — per-chunk, multi-translator, with author/reviewer provenance.
- 🖼️ **Agent-driven OCR** — scanned pages handed back as rendered PNGs; agent transcribes and the graph absorbs the result.

## Install

```bash
pip install kglite-docs
```

## 30 seconds of Python

```python
from kglite_docs import Corpus

with Corpus.create("kb.kgl") as corpus:           # auto-saves on exit
    corpus.ingest_dir("./papers")                  # PDF / DOCX / PPTX / MD / HTML / images
    hits = corpus.search("transformer attention", top_k=5, agent_id="me")
    ctx = corpus.compose_context("transformer attention", max_tokens=3000)
    # ctx["items"] is a ranked, token-budgeted bundle ready for your LLM prompt
```

## 30 seconds of agent loop

Cross-checked enrichment in five lines:

```python
sid = corpus.add_summary(
    target_id=hits[0]["id"], text="DPR uses a dual BERT encoder…",
    agent_id="writer", model="opus-4.7",
)
# A different agent verifies — self-verification is rejected
corpus.verify_summary(sid, verdict="verified",
                      verifier_agent_id="reviewer", notes="checked p.5")
# Score how grounded the summary is in its source chunks
print(corpus.check_grounding(sid)["supported_fraction"])    # → 1.0
```

## Run it as an MCP server

```bash
kglite-docs-mcp --db kb.kgl
```

Register with Claude Code:

```bash
claude mcp add kglite-docs -- kglite-docs-mcp --db /abs/path/kb.kgl
```

The agent now sees ~30 typed tools (`search`, `compose_context`, `add_summary`, `verify_summary`, `tag_chunk`, `cluster_chunks`, `claim_next_review`, …) plus `cypher_query` as an escape hatch.

## Read the docs

📖 **Full documentation at [kglite-docs.readthedocs.io](https://kglite-docs.readthedocs.io/)**

- [Getting started](https://kglite-docs.readthedocs.io/en/latest/getting-started/) — 10 minutes from `pip install` to a running agent
- [Agent workflows](https://kglite-docs.readthedocs.io/en/latest/workflows/) — research, comparison, fact-checking, OCR loops, hallucination guards
- [Architecture](https://kglite-docs.readthedocs.io/en/latest/architecture/) — graph model, design rationale, the 30+ typed MCP tools
- [API reference](https://kglite-docs.readthedocs.io/en/latest/api/corpus/) — every method, every argument, IDE-friendly type stubs
- [Troubleshooting](https://kglite-docs.readthedocs.io/en/latest/troubleshooting/) — common failure modes
- [Changelog](https://kglite-docs.readthedocs.io/en/latest/changelog/)

## License

MIT.
