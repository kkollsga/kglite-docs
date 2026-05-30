"""Exception hierarchy.

Catch `KgliteDocsError` for anything raised by this library. The
concrete subclasses below are what callers reach for when they want to
distinguish *why* a call failed.
"""

from __future__ import annotations


class KgliteDocsError(Exception):
    """Base class for all kglite-docs errors."""


class IngestError(KgliteDocsError):
    """Couldn't ingest a document — unsupported format, parse failure,
    missing source file, etc."""


class UnsupportedFormatError(IngestError):
    """The file extension isn't one of our supported formats. Pass
    `format=` explicitly if you know better."""


class MissingSourceError(KgliteDocsError):
    """A Document's recorded `path` doesn't exist on disk anymore.
    Typically surfaced from the OCR loop, which needs to render the
    page from the original file."""


class ReviewConflict(KgliteDocsError):
    """Review-queue state-machine violation: trying to claim an
    in-review ticket, complete one you don't hold, etc."""


class SelfVerificationError(KgliteDocsError):
    """An agent tried to verify its own summary. Verifications must
    come from a different agent."""


class GroundingError(KgliteDocsError):
    """A check_grounding call against a target that doesn't exist or
    has no source chunks."""


class NotIndexedError(KgliteDocsError):
    """search / compose_context on a corpus that has ready chunks but
    none embedded — call `Corpus.index()` (or `ingest(embed=True)`)
    first. Raised instead of silently returning `[]`, which is
    indistinguishable from a query that genuinely has no matches."""


class InvalidEnumError(KgliteDocsError, ValueError):
    """A string value isn't one of the allowed enum members. Subclasses
    ValueError too, so old `except ValueError` paths still work."""


class ConcurrencyError(KgliteDocsError):
    """Multi-process write attempted on a single-writer `.kgl`."""


class SynthesisRequiredError(KgliteDocsError):
    """`conclude_study` was called before a ledger-wide cross-chunk synthesis
    pass ran (`synthesis_status == pending`). Run `study("synthesize", …)`
    first, or pass `acknowledge_no_synthesis=True` to record an explicit,
    audited skip. Guards against a *confident-incomplete* conclusion — the
    happy path marking a study "done" while a whole class of cross-chunk
    finding is still unreachable."""


__all__ = [
    "ConcurrencyError",
    "GroundingError",
    "IngestError",
    "InvalidEnumError",
    "KgliteDocsError",
    "MissingSourceError",
    "NotIndexedError",
    "ReviewConflict",
    "SelfVerificationError",
    "SynthesisRequiredError",
    "UnsupportedFormatError",
]
