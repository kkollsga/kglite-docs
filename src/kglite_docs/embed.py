"""BAAI/bge-m3 embedder.

Subclasses kglite's own `BgeM3Embedder` so we inherit the cool-down /
release lifecycle, ONNX session caching, CLS pooling, and 8192-token
truncation it ships with. Adds a thin `embed_query` wrapper that
returns a single vector (the common case for `search`).

The model weights are pulled from HuggingFace on first load and cached
in `~/.cache/fastembed/` by default; if the user already has them at
`/Volumes/EksternalHome/LLMs/hub/models--BAAI--bge-m3`, point
`HF_HOME` or pass `cache_dir=` to reuse them.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from kglite.mcp_server.bge_m3 import (
    DEFAULT_COOLDOWN_SECONDS,
    DIMENSION,
    MAX_LENGTH,
    BgeM3Embedder,
)

log = logging.getLogger("kglite_docs.embed")

__all__ = [
    "DEFAULT_COOLDOWN_SECONDS",
    "DIMENSION",
    "MAX_LENGTH",
    "BgeM3Embedder",
    "make_embedder",
    "weights_cached",
    "prefetch_embedder",
]


def make_embedder(
    cache_dir: str | Path | None = None,
    cooldown_seconds: int | None = None,
) -> BgeM3Embedder:
    """Construct a configured embedder. Thin wrapper kept here so callers
    don't reach into kglite's internal module path."""
    return BgeM3Embedder(
        cache_dir=Path(cache_dir) if cache_dir is not None else None,
        cooldown_seconds=cooldown_seconds,
    )


def _default_cache_dir() -> Path:
    """Where bge-m3 weights land by default — mirrors BgeM3Embedder."""
    return Path(
        os.environ.get("FASTEMBED_CACHE_PATH", Path.home() / ".cache" / "fastembed")
    )


def weights_cached(cache_dir: str | Path | None = None) -> bool:
    """True when the bge-m3 ONNX weights are already on disk, so `load()`
    can run without touching the HuggingFace Hub."""
    base = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    snaps = base / "models--BAAI--bge-m3" / "snapshots"
    return any(snaps.glob("*/onnx/model.onnx_data"))


def prefetch_embedder(
    embedder: Any, *, offline_when_cached: bool = True
) -> threading.Thread | None:
    """Warm the embedder's ONNX session in a background daemon thread.

    The first ``embed()`` (inside the first ingest/search) otherwise pays
    ~8s of ``ort.InferenceSession`` init plus an HF-hub network round-trip
    *inline* — long enough to blow an MCP client's per-call timeout when an
    agent hits it from a blind state. Loading off-thread at boot moves that
    cost off the request path: ``initialize`` returns immediately and the
    model is resident (or nearly so) by the time the first tool call lands.

    Duck-typed and failure-safe: any object with a callable ``load()``
    works (stub embedders included); errors are logged and deferred to
    first real use. Returns the started thread, or ``None`` if the
    embedder has no ``load()``.
    """
    load = getattr(embedder, "load", None)
    if not callable(load):
        return None

    cache_dir = getattr(embedder, "_cache_dir", None)
    if offline_when_cached and cache_dir is not None and weights_cached(cache_dir):
        # Skip the per-load HF-hub ETag check (and its rate-limit
        # exposure) when the weights are already local. setdefault so an
        # operator who set it explicitly wins.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    def _run() -> None:
        try:
            t0 = time.monotonic()
            load()
            log.info("embedder warm-loaded in %.1fs (background)", time.monotonic() - t0)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("embedder warm-load failed (will load on first use): %s", exc)

    thread = threading.Thread(target=_run, name="kglite-docs-warmload", daemon=True)
    thread.start()
    return thread
