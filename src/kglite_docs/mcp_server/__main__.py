"""Console entry: ``python -m kglite_docs.mcp_server`` or
``kglite-docs-mcp``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _maybe_enable_hf_offline() -> None:
    """If the bge-m3 weights are already cached, run HuggingFace Hub in
    offline mode. `huggingface_hub` reads `HF_HUB_OFFLINE` at *import*
    time, so this must run before any kglite import pulls it in — hence
    the inline cache check here (not via `kglite_docs.embed`, which would
    import kglite first). Skips the per-load ETag HEAD requests and, more
    importantly, removes the network as a failure mode on the first embed
    from a blind state (offline weights load deterministically; an
    online load can hang or 429 when HF is unreachable / rate-limiting)."""
    import os
    from pathlib import Path

    if os.environ.get("HF_HUB_OFFLINE") is not None:
        return
    cache = Path(
        os.environ.get("FASTEMBED_CACHE_PATH", Path.home() / ".cache" / "fastembed")
    )
    snaps = cache / "models--BAAI--bge-m3" / "snapshots"
    if any(snaps.glob("*/onnx/model.onnx_data")):
        os.environ["HF_HUB_OFFLINE"] = "1"


def main(argv: list[str] | None = None) -> int:
    _maybe_enable_hf_offline()
    parser = argparse.ArgumentParser(prog="kglite-docs-mcp")
    parser.add_argument("--db", required=True, help="Path to the .kgl knowledge base")
    parser.add_argument("--ingest", help="Optional directory to ingest (and index) before serving")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirs when ingesting")
    parser.add_argument(
        "--no-warmup", action="store_true",
        help="Skip the background bge-m3 warm-load (for non-embedding deployments)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level)

    from kglite_docs import Corpus
    from kglite_docs.embed import make_embedder
    from kglite_docs.mcp_server.server import build_app

    db = Path(args.db)
    # cooldown=0: keep the model resident for the life of the server. The
    # library default (900s) is tuned for read-only graph servers that
    # reclaim RAM during long idle; an interactive ingest+search session
    # would otherwise re-pay the ~8s load on the first embed after 15 min
    # idle. The session is warm-loaded off-thread in build_app().
    embedder = make_embedder(cooldown_seconds=0)
    corpus = (
        Corpus.open(db, embedder=embedder)
        if db.exists()
        else Corpus.create(db, embedder=embedder)
    )
    if args.ingest:
        # A pre-built served DB should be searchable, so embed here.
        # No per-call timeout offline → drain everything in one pass.
        corpus.ingest_dir(args.ingest, recursive=args.recursive)
        corpus.index(max_seconds=None)
        corpus.save(db)

    app = build_app(corpus, warm_embedder=not args.no_warmup)
    try:
        app.run(transport="stdio")
    finally:
        # Persist tool-driven mutations on clean shutdown (stdio close).
        # Per-call saves on ingest/index are the in-flight safety net;
        # this catches everything else (summaries, tags, reviews, …).
        import contextlib
        with contextlib.suppress(Exception):
            corpus.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
