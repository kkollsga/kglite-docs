"""Test helpers — small public utilities for exercising kglite-docs in tests.

Importing this module pulls in nothing heavy; it's safe to use from a test suite
or a notebook. The functions here encode the bits of tribal knowledge that are
otherwise copy-pasted across test files.
"""

from __future__ import annotations

from typing import Any


def make_chunks(corpus: Any, n: int = 4, *, words: int = 8) -> list[str]:
    """Ingest a synthetic document that produces exactly `n` ready chunks, in
    order, and return their ids.

    The lever (easy to get wrong): short paragraphs get *packed* into one chunk,
    so `n` short paragraphs do not give `n` chunks. The reliable way to force one
    chunk per unit is a heading per section plus `structure_aware=True` — which
    is what this does. `words` controls the body length per section.
    """
    body = " ".join(f"word{i}" for i in range(max(1, words)))
    md = "\n\n".join(f"# Section {i}\n\n{body} for chunk {i} here." for i in range(n))
    corpus.ingest(text=md, title="make_chunks", format="md", structure_aware=True)
    return [
        r["id"]
        for r in corpus.cypher(
            "MATCH (c:Chunk:Ready) RETURN c.id AS id ORDER BY c.doc_id, c.chunk_index"
        ).to_list()
    ]
