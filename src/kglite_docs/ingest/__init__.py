"""Ingestion subpackage: multi-format parsing, chunking, hashing."""

from kglite_docs.ingest.chunker import Chunk, chunk_page
from kglite_docs.ingest.formats import SUPPORTED_FORMATS, detect_format, parse_document
from kglite_docs.ingest.hashing import file_hash, text_hash
from kglite_docs.ingest.parser import PageContent, parse_pdf

__all__ = [
    "Chunk", "PageContent", "SUPPORTED_FORMATS",
    "chunk_page", "detect_format", "file_hash", "parse_document",
    "parse_pdf", "text_hash",
]
