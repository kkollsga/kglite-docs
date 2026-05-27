"""Token-aware paragraph chunker.

Splits a page's markdown into chunks targeting `~target_tokens`
bge-m3 tokens, with `~overlap_tokens` of overlap, never crossing a
page boundary. Splits on heading + blank-line boundaries first; only
falls back to mid-paragraph splitting when a single paragraph exceeds
the cap.

The tokenizer comes from the same `tokenizers.Tokenizer` instance
bge-m3 uses, so the token count we target here matches what the model
actually sees at encode time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from kglite_docs.ingest.hashing import normalize_text, text_hash

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_BLANK_LINE_RE = re.compile(r"\n\s*\n+")

DEFAULT_TARGET_TOKENS = 512
DEFAULT_OVERLAP_TOKENS = 64
HARD_TOKEN_CAP = 8000  # bge-m3 max is 8192; leave headroom


@dataclass
class Chunk:
    """One chunk as produced by `chunk_page`. The caller assigns the
    final node id; we just carry positional + content fields."""

    chunk_index: int  # 0-based, within the page
    text: str
    token_count: int
    headings: list[str] = field(default_factory=list)
    text_hash_value: str = ""

    def __post_init__(self) -> None:
        if not self.text_hash_value:
            self.text_hash_value = text_hash(self.text)


@lru_cache(maxsize=1)
def _bge_m3_tokenizer() -> Tokenizer:
    """Download (cached) and return the bge-m3 tokenizer used by the
    real encoder. Used purely for length counting at chunk time — the
    actual embeddings come from the encoder later."""
    path = hf_hub_download(repo_id="BAAI/bge-m3", filename="onnx/tokenizer.json")
    return Tokenizer.from_file(path)


def count_tokens(text: str, tokenizer: Tokenizer | None = None) -> int:
    """Number of bge-m3 tokens in ``text`` (no special tokens added)."""
    tok = tokenizer or _bge_m3_tokenizer()
    return len(tok.encode(text, add_special_tokens=False).ids)


def chunk_page(
    markdown: str,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    initial_headings: list[str] | None = None,
    tokenizer: Tokenizer | None = None,
) -> list[Chunk]:
    """Greedy paragraph packer.

    Algorithm:

    1. Walk the markdown line-by-line. Each line is either a heading
       (updates the heading stack) or a body line. Group body lines into
       paragraphs separated by blank lines.
    2. Pack paragraphs greedily into the current chunk until adding the
       next paragraph would exceed `target_tokens`. Emit the chunk; start
       a new one seeded with the trailing `overlap_tokens` of the prior
       chunk's text (preserving heading context).
    3. A single paragraph larger than `target_tokens` is split on
       sentence boundaries, then on words if still too large.
    4. Heading lines themselves are *not* emitted as standalone chunks;
       they update the heading stack inherited by subsequent chunks.

    Returns one `Chunk` per chunk emitted; an empty input returns `[]`.
    """
    if not markdown.strip():
        return []

    tok = tokenizer or _bge_m3_tokenizer()
    heading_stack: list[str] = list(initial_headings or [])
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_tokens = 0
    chunk_idx = 0
    pending_overlap: str = ""

    def emit() -> None:
        nonlocal buf, buf_tokens, chunk_idx, pending_overlap
        if not buf:
            return
        text = normalize_text("\n\n".join(buf))
        if not text:
            buf = []
            buf_tokens = 0
            return
        chunks.append(
            Chunk(
                chunk_index=chunk_idx,
                text=text,
                token_count=count_tokens(text, tok),
                headings=list(heading_stack),
            )
        )
        chunk_idx += 1
        # Compute the trailing overlap (in tokens) for the next chunk
        # seed. We approximate by re-encoding and decoding the last N
        # tokens of the emitted chunk.
        if overlap_tokens > 0:
            ids = tok.encode(text, add_special_tokens=False).ids
            if len(ids) > overlap_tokens:
                pending_overlap = tok.decode(ids[-overlap_tokens:]).strip()
            else:
                pending_overlap = ""
        buf = []
        buf_tokens = 0

    # Iterate paragraphs while tracking heading stack
    for para in _iter_paragraphs(markdown, heading_stack):
        if not para.strip():
            continue
        para_tokens = count_tokens(para, tok)
        if para_tokens > target_tokens:
            # Oversized paragraph: split further
            for piece, piece_tokens in _split_oversized(para, target_tokens, tok):
                _accept_piece(piece, piece_tokens, buf, target_tokens)
                if _sum_tokens(buf, tok) >= target_tokens:
                    emit()
                    if pending_overlap:
                        buf.append(pending_overlap)
                        pending_overlap = ""
                        buf_tokens = count_tokens(buf[0], tok)
            continue

        if buf_tokens + para_tokens > target_tokens and buf:
            emit()
            if pending_overlap:
                buf.append(pending_overlap)
                pending_overlap = ""
                buf_tokens = count_tokens(buf[0], tok)
        buf.append(para)
        buf_tokens += para_tokens

    emit()
    return chunks


def _iter_paragraphs(markdown: str, heading_stack: list[str]) -> Iterator[str]:
    """Yield paragraph-shaped strings from markdown, mutating
    `heading_stack` in place when headings are encountered."""
    # Split into blocks separated by blank lines.
    blocks = _BLANK_LINE_RE.split(markdown)
    for block in blocks:
        block = block.strip("\n")
        if not block:
            continue
        # If the block starts with a heading line, update the stack
        # and yield the rest (if any).
        lines = block.split("\n")
        consumed = 0
        for ln in lines:
            m = _HEADING_RE.match(ln)
            if m is None:
                break
            level = len(m.group(1))
            title = m.group(2).strip()
            # Truncate stack to level-1, then append
            del heading_stack[level - 1 :]
            heading_stack.append(title)
            consumed += 1
        rest = "\n".join(lines[consumed:]).strip()
        if rest:
            yield rest


def _split_oversized(
    para: str, cap: int, tok: Tokenizer
) -> Iterator[tuple[str, int]]:
    """Split a paragraph that exceeds `cap` tokens into smaller pieces.
    Splits on sentence boundaries first, then on words if a sentence is
    *still* too large (very rare for real prose)."""
    sentences = re.split(r"(?<=[.!?])\s+", para)
    cur: list[str] = []
    cur_tokens = 0
    for s in sentences:
        s_tokens = count_tokens(s, tok)
        if s_tokens > cap:
            if cur:
                yield " ".join(cur), cur_tokens
                cur, cur_tokens = [], 0
            # word-level fallback
            words = s.split()
            wbuf: list[str] = []
            wtokens = 0
            for w in words:
                w_t = count_tokens(w, tok)
                if wtokens + w_t > cap and wbuf:
                    yield " ".join(wbuf), wtokens
                    wbuf, wtokens = [], 0
                wbuf.append(w)
                wtokens += w_t
            if wbuf:
                yield " ".join(wbuf), wtokens
            continue
        if cur_tokens + s_tokens > cap and cur:
            yield " ".join(cur), cur_tokens
            cur, cur_tokens = [], 0
        cur.append(s)
        cur_tokens += s_tokens
    if cur:
        yield " ".join(cur), cur_tokens


def _accept_piece(piece: str, piece_tokens: int, buf: list[str], cap: int) -> None:
    buf.append(piece)


def _sum_tokens(buf: list[str], tok: Tokenizer) -> int:
    if not buf:
        return 0
    return count_tokens("\n\n".join(buf), tok)


def load_tokenizer_from_local(path: str | Path) -> Tokenizer:
    """Optional helper for users who keep the bge-m3 weights at a non-
    standard path. Pass the path to `onnx/tokenizer.json`."""
    return Tokenizer.from_file(str(Path(path)))
