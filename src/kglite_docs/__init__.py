"""kglite-docs — agent-first PDF knowledge base on top of kglite."""

from __future__ import annotations

from kglite_docs.corpus import Corpus
from kglite_docs.schema import (
    AGENT,
    CHUNK,
    CHUNK_TEXT_EMB,
    CLUSTER,
    DOCUMENT,
    DOCUMENT_TITLE_EMB,
    NOTE,
    PAGE,
    SUMMARY,
    SUMMARY_TEXT_EMB,
    TAG,
    VIEW,
)

try:
    from kglite_docs._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "0.0.0+local"

__all__ = [
    "AGENT",
    "CHUNK",
    "CHUNK_TEXT_EMB",
    "CLUSTER",
    "DOCUMENT",
    "DOCUMENT_TITLE_EMB",
    "NOTE",
    "PAGE",
    "SUMMARY",
    "SUMMARY_TEXT_EMB",
    "TAG",
    "VIEW",
    "Corpus",
    "__version__",
]
