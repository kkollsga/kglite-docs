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

from pathlib import Path

from kglite.mcp_server.bge_m3 import (
    DEFAULT_COOLDOWN_SECONDS,
    DIMENSION,
    MAX_LENGTH,
    BgeM3Embedder,
)

__all__ = ["DEFAULT_COOLDOWN_SECONDS", "DIMENSION", "MAX_LENGTH", "BgeM3Embedder", "make_embedder"]


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
