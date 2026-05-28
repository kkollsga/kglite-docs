# Troubleshooting

Common failure modes and how to recover.

## Install / first run

### `pip install` succeeds but `from kglite_docs import Corpus` fails

```
ImportError: cannot import name '...' from 'kglite_docs._version'
```

You installed in editable mode (`pip install -e .`) without git tags. `hatch-vcs` derives `__version__` from the most recent `v*` tag. Either tag the local commit (`git tag v0.0.1`) or run `pip install .` (non-editable, which baked in the fallback version).

### bge-m3 weights take forever to download

The ONNX weights are ~2 GB. First-call latency is dominated by this download into `~/.cache/fastembed/`. To reuse an existing HuggingFace cache:

```bash
export HF_HUB_CACHE=/path/to/your/huggingface/hub
export FASTEMBED_CACHE_PATH=/path/to/your/huggingface/hub
```

If the download keeps failing or partial-completing, delete the cache directory and retry — `hf_hub_download` is idempotent.

### `Warning: You are sending unauthenticated requests to the HF Hub`

Cosmetic — HuggingFace warns when no `HF_TOKEN` is set. The download still works. Set `HF_TOKEN` to silence it and avoid future rate-limit headaches.

## Ingest

### `UnsupportedFormatError: unsupported format: 'rtf'`

You hit a format we don't parse. Either:
- Pass `format="md"` (or another supported format) explicitly to hint a parser, or
- Convert the file with `pandoc` (`pandoc x.rtf -o x.md`) and ingest the result.

Supported extensions: `pdf`, `docx`, `pptx`, `md`, `markdown`, `html`, `htm`, `txt`, plus image formats (`png`, `jpg`, `jpeg`, `tif`, `tiff`, `webp`, `bmp`).

### Ingest is slow on long-text chunks

bge-m3 embed time scales with sequence length. Pages with very long paragraphs (close to the 8192-token cap) take seconds each on CPU. Options:
- Enable a GPU execution provider (`CoreMLExecutionProvider` on macOS, `CUDAExecutionProvider` on Linux) when constructing the embedder.
- Lower the chunker's `target_tokens` (default 512) so each chunk is shorter.
- Run ingest jobs as background tasks; the embedder cool-down keeps subsequent calls warm for 15 minutes.

### Re-ingesting a modified file is silent — old chunks remain

By design. `Document.id` is the sha256 of the file bytes — if the file changed, you'll get a *new* `Document` node alongside the old one. The old document's summaries don't auto-update; check `enrich.mark_stale_for_doc()` if you want to flip stale summaries to `verification_status='stale'` after the new ingest.

## OCR

### `list_pending_ocr` returns rows with `image_error: source file missing`

The `Document.path` recorded at ingest time no longer points at the file on disk. The OCR loop can't render without the source. Two recovery paths:

1. Restore or relocate the file to the recorded path.
2. Re-ingest the file from its current location — that creates a fresh `Document` (new path), and OCR will work against it. The old broken-path document remains but won't OCR.

### `kglite-docs ocr-do` fails on every page

Most common cause: your `--agent-cmd` doesn't actually call a vision-capable model with the image. The CLI requires `{image}` in the template, but it doesn't *check* the agent actually reads it. Debug with `--dry-run` first:

```bash
kglite-docs ocr-do --db kb.kgl --agent-cmd 'whatever {image}' --dry-run
```

Then run a single page to confirm the agent's output:

```bash
kglite-docs ocr-do --db kb.kgl --limit 1 --agent-cmd '…'
```

A passing OCR returns markdown of > 0 chars on stdout with exit code 0. Empty stdout or non-zero exit is logged as a skip.

## Storage

### Corruption / unable to open a `.kgl` file

kglite's `.kgl` is a memory-mapped binary. If the file is truncated mid-write (Ctrl-C during `save()`, disk full, kernel panic), it may not reload. Recovery:

1. **Best case** — you have a recent `.kgle` (embeddings) export. `kglite.load(other.kgl)` + `g.import_embeddings('snap.kgle')` recovers vectors against a rebuilt graph.
2. **Otherwise** — re-ingest. `Document.id` is content-keyed, so re-running `ingest_dir(src)` reproduces every doc deterministically. Summaries/tags/reviews are lost; the underlying data isn't.

### File grows much faster than expected

Each chunk holds:
- ~4 KB text (typical)
- 1024 × 4 B = 4 KB embedding
- ~1 KB schema overhead

So ~9 KB / chunk. A 16-doc corpus with ~50 chunks/doc is ~7 MB; 1k docs are ~450 MB. If you're seeing 10x that, check `embedding_diagnostics()` — every property *could* have its own store and a stale one might be lingering:

```python
for row in store.g.embedding_diagnostics():
    if row["status"] == "embedded" and row["text_column"] != "text":
        store.g.remove_embeddings(row["node_type"], row["text_column"])
```

## Concurrency

### `ReviewConflict: agent 'b' can't claim ticket held by 'a'`

Two agents tried to grab the same ticket. By design — only one agent can hold an `in_review` ticket at a time. Either `unclaim_review` (if you're holder `a`) or `claim_next_review` to pick the next free one.

### Multiple processes writing to the same `.kgl`

Don't. kglite is single-writer per file. Two processes calling `c.save()` against the same path will produce a corrupted file. Designs that need concurrent writers:

- **Read-many, write-one**: route all writes through one long-lived process (the MCP server). Other processes open the `.kgl` read-only.
- **Sharded**: split the corpus by document — one `.kgl` per source, one writer process per shard.

Multi-process write coordination via kglite-level transactions is on the upstream roadmap.

## MCP

### Claude doesn't see kglite-docs tools

Check that:

1. `kglite-docs-mcp --db kb.kgl` runs cleanly from a shell (no traceback before the MCP server boots).
2. The MCP config in Claude Code points at the right command:
   ```bash
   claude mcp add kglite-docs -- kglite-docs-mcp --db /absolute/path/to/kb.kgl
   ```
   Use an *absolute* path — Claude's working directory may not be yours.
3. `kglite-docs` is installed (`mcp` + `mcp-methods` are core deps, no extras needed): `pip install kglite-docs`.

### MCP search returns hits, but `text` is missing from one of them

`vector_search` returns ids + scores reliably; we re-join via Cypher to attach `text` / `page` / `doc_id`. If a chunk is *just* embedded but not in the node store (e.g. mid-OCR), the join misses. Usually self-correcting on the next save+reload.

## Embeddings

### "set_embeddings: 'skipped': N" warning

Means some of the ids in the dict didn't resolve to nodes in the current graph. Common causes:

- Re-ingesting an old `.kgle` against a graph with new ids.
- Race between two writers (shouldn't happen — see Concurrency).
- Pre-0.10.4 kglite (id-index bug — upgrade to `kglite>=0.10.4`).

### How do I move the bge-m3 model to a different location?

```python
from kglite_docs.embed import make_embedder
embedder = make_embedder(cache_dir="/my/path/hub")
corpus = Corpus.open("kb.kgl", embedder=embedder)
```

Or set the env var before *anything else* imports the embedder:

```bash
export HF_HUB_CACHE=/my/path/hub
```

## Still stuck?

- Check `docs/architecture.md` for what *should* happen at each stage.
- Crank logging: `logging.getLogger("kglite_docs").setLevel("DEBUG")` plus
  `logging.getLogger("kglite.mcp_server").setLevel("DEBUG")` if running the MCP server.
- File an issue at <https://github.com/kkollsga/kglite-docs/issues> with the
  minimal repro plus the output of `kglite-docs --version`, the kglite
  version (`python -c "import kglite; print(kglite.__version__)"`), and
  your OS / Python version.
