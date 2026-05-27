"""Shared fixtures.

By default the unit tests use a deterministic stub embedder (no model
download, no GPU, no I/O) so the suite is fast and hermetic. Tests
that need the real bge-m3 model carry the `@pytest.mark.embed` mark.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterator

import pytest

from kglite_docs import Corpus


class StubEmbedder:
    """Deterministic 8-dim embedder. Maps text → vector by hashing token
    chunks, so the vector is stable across runs and unique per text."""

    dimension = 8

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            vec = [(b - 128) / 128.0 for b in digest[: self.dimension]]
            # normalise so cosine ranges are well-behaved
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out

    def load(self) -> None: ...
    def unload(self) -> None: ...


@pytest.fixture
def stub_embedder() -> StubEmbedder:
    return StubEmbedder()


@pytest.fixture
def corpus(tmp_path: Path, stub_embedder: StubEmbedder) -> Corpus:
    """Fresh in-memory corpus with the stub embedder."""
    return Corpus.create(tmp_path / "test.kgl", embedder=stub_embedder)


@pytest.fixture
def sample_pdf_dir() -> Path:
    return Path(__file__).parent.parent / "sample_data" / "pdfs"


@pytest.fixture
def sample_mixed_dir() -> Path:
    return Path(__file__).parent.parent / "sample_data" / "mixed"


@pytest.fixture(scope="session", autouse=True)
def _use_local_hf_cache() -> None:
    """If the user's local HF cache is present, point at it so the
    embed-marked tests reuse cached weights instead of re-downloading."""
    candidate = Path("/Volumes/EksternalHome/LLMs/hub")
    if candidate.exists():
        os.environ.setdefault("HF_HUB_CACHE", str(candidate))
        os.environ.setdefault("FASTEMBED_CACHE_PATH", str(candidate))
