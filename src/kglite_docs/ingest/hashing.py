"""Deterministic hashing for documents and chunk text.

- `file_hash(path)` keys a Document by raw bytes — re-ingesting the same
  file is a no-op.
- `text_hash(text)` keys a Chunk's contents — used to detect when
  underlying text drifts and any derived summaries should be marked
  ``stale``.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Iterable

_WHITESPACE_RUN = re.compile(r"\s+")


def file_hash(path: str | Path, *, prefix: str = "doc_") -> str:
    """sha256 of a file's bytes, returned as `{prefix}<hex>`. Streamed read
    so multi-GB PDFs don't blow up memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return f"{prefix}{h.hexdigest()}"


def normalize_text(text: str) -> str:
    """Canonicalise a chunk's text so trivial reflows don't change the hash.

    Normalisation: NFC unicode; collapse runs of whitespace to single space;
    strip leading/trailing whitespace. Does *not* lowercase — case carries
    semantic meaning (acronyms, proper nouns, code).
    """
    return _WHITESPACE_RUN.sub(" ", unicodedata.normalize("NFC", text)).strip()


def text_hash(text: str) -> str:
    """sha256 of the normalised text. Stable across trivial whitespace
    re-flows from PDF re-extraction."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def combined_hash(parts: Iterable[str]) -> str:
    """sha256 over a sequence of strings; order-sensitive. Used to derive a
    `source_text_hash` for a Summary from the hashes of the chunks it
    summarises."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1e")  # ASCII record separator
    return h.hexdigest()
