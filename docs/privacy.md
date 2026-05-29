# Confidentiality & data handling

**Everything runs on your machine.** kglite-docs is a local library and a local
MCP server — parsing, chunking, embedding, clustering, tagging, summarising,
grounding, OCR hand-off, and the evidence-study workflow all execute in your own
process against a local file. This matters for legal, medical, forensic, and
other confidential corpora: your documents do not leave the host.

## Where your data lives

- **The corpus is a single local file** — a kglite `.kgl` graph you choose the
  path for (`Corpus.create("/path/to/case.kgl")`). Document text, chunks,
  embeddings, tags, summaries, assessments, and audit history are all stored
  there. Nothing is written elsewhere.
- **The MCP server is a local process.** It talks to its client over stdio /
  a local transport and reads/writes only that one `.kgl` (single-writer — see
  [Architecture](architecture.md)). It is not a network service and does not
  open a listening socket to the outside world.
- **No telemetry.** kglite-docs sends no usage data, no analytics, and no
  document content anywhere. There is no "phone home."

## The one network call: the embedding model

The **only** outbound network request kglite-docs makes is a **one-time download
of the [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) ONNX weights** from the
HuggingFace Hub, the first time you embed anything. After that the weights are
cached on disk and reused.

- **Your documents are never uploaded.** Embeddings are computed locally by
  `onnxruntime` on the cached model. The HF request fetches *model weights*; it
  does not send your text, queries, or any corpus content.
- **It's a public model — no account or token needed.** You will see:

  ```
  Warning: You are sending unauthenticated requests to the HF Hub.
  ```

  This is cosmetic. The download works without a token. Set `HF_TOKEN` only if
  you want to silence it or avoid Hub rate limits — see
  [Troubleshooting](troubleshooting.md).
- **Once cached, kglite-docs goes offline automatically.** When the MCP server
  warm-loads the embedder and finds the weights already on disk, it sets
  `HF_HUB_OFFLINE=1` for the process so no further Hub round-trips (not even an
  ETag check) happen. You can also set it yourself.

## Air-gapped / fully offline operation

For an environment that must never touch the network:

1. On a connected machine, download the weights once (any single `index()` /
   `search()` call, or pre-seed the HF cache directory).
2. Copy the HF cache to the target host and point `HF_HUB_CACHE` at it
   (see [Troubleshooting](troubleshooting.md) for the exact path).
3. Export `HF_HUB_OFFLINE=1` (and `TRANSFORMERS_OFFLINE=1`) before starting.

With the model cached and offline mode set, kglite-docs makes **no network
requests at all** — ingest, index, search, study, and review run entirely
locally.

## What this does *not* cover

kglite-docs controls only its own behaviour. If **you** pass chunk text or a
composed-context bundle to a remote LLM (e.g. an agent calling a hosted model),
that content leaves your machine through *that* call — not through kglite-docs.
Keep confidential corpora behind a local model if the document text itself must
never transit a third party.
