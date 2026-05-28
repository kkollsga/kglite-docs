"""kglite-docs — agent-first PDF knowledge base on top of kglite."""

from __future__ import annotations

from kglite_docs.corpus import Corpus
from kglite_docs.errors import (
    ConcurrencyError,
    GroundingError,
    IngestError,
    InvalidEnumError,
    KgliteDocsError,
    MissingSourceError,
    ReviewConflict,
    SelfVerificationError,
    UnsupportedFormatError,
)
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
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("kglite-docs")
except Exception:  # pragma: no cover - not installed (e.g. running from source)
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
    "ConcurrencyError",
    "Corpus",
    "GroundingError",
    "IngestError",
    "InvalidEnumError",
    "KgliteDocsError",
    "MissingSourceError",
    "ReviewConflict",
    "SelfVerificationError",
    "UnsupportedFormatError",
    "__version__",
]
