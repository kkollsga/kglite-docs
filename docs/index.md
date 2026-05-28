# kglite-docs

Agent-first PDF knowledge base. Ingest PDFs, DOCX, PPTX, Markdown, HTML, plain text, or images; chunk + embed them with BAAI/bge-m3; cluster, tag, summarise, fact-check, translate, and review them; serve the whole thing to AI agents over MCP. Built on [kglite](https://github.com/kkollsga/kglite) (storage + vector + clustering) and [mcp-methods](https://github.com/kkollsga/mcp-methods) (MCP framework).

## Why another RAG library?

Most "RAG libraries" hand the agent a `search(query) → list[chunk]` and call it a day. kglite-docs treats the corpus as a *living* knowledge graph that records who did what:

- **Documents → pages → chunks** keyed by file hash, page, and position
- **Agents** are first-class nodes; their views, tags, summaries, verifications, and reviews are all queryable
- **Cross-checking** — one agent writes a summary, a *different* agent verifies (self-verification is rejected server-side)
- **Review kanban** — chunks/summaries flow through `new → in_review → reviewed` with an immutable event audit trail
- **Translations** as their own nodes with author/reviewer provenance
- **Clusters + back-references** so an article generated from a cluster cites the chunks it came from
- **Grounding checks** — built-in tools to score how well an agent's summary aligns with its sources

## Quick install

```bash
pip install "kglite-docs[mcp]"
```

## 30-second taste

```python
from kglite_docs import Corpus

with Corpus.create("kb.kgl") as corpus:           # auto-saves on exit
    corpus.ingest_dir("./papers")                  # PDF/DOCX/PPTX/MD/HTML/TXT/images
    hits = corpus.search("transformer attention", top_k=5)
    print(hits[0]["text"][:120])
```

Then serve to agents:

```bash
kglite-docs-mcp --db kb.kgl
```

## Where to go next

- **[Getting started](getting-started.md)** — 10 minutes from `pip install` to "I'm using this from an agent."
- **[Workflows](workflows.md)** — research, comparison, fact-checking, OCR, hallucination guards.
- **[Architecture](architecture.md)** — the graph model, the design rationale, where to look in the code.
- **[Troubleshooting](troubleshooting.md)** — common failure modes.
- **[API reference](api/corpus.md)** — full method-by-method docs.
- **[Changelog](changelog.md)** — release notes.

## License

MIT.
