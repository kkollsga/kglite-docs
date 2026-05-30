"""Cheap, deterministic per-chunk content signals — computed at ingest with no
model and no extra dependency, so agents spend tokens on judgment instead of
mechanical triage. Pure functions; the pipeline applies them and the results are
stored as additive chunk properties + label predicates (never lossy: a signal
only *labels*, it never drops or hides a chunk)."""

from __future__ import annotations

import re

#: A chunk shorter than this (in words) is `sparse` rather than prose.
SPARSE_WORDS = 8
#: quality_score below this flags a (prose/sparse) chunk `:LowQuality`.
LOW_QUALITY_THRESHOLD = 0.55

_WORD = re.compile(r"\w+")
# Detection must survive the chunker's whitespace-collapse (chunk text has no
# newlines), so these are newline-agnostic.
_TABLE_SEP = re.compile(r"\|\s*:?-{2,}")           # a `|---` table separator
_LIST_MARKER = re.compile(r"(?:^|\s)(?:[-*+]|\d+[.)])\s+\S")
_LIST_START = re.compile(r"\s*(?:[-*+]\s|\d+[.)]\s)")
# Characters that are "normal" in extracted prose — the rest count as noise.
_CLEAN_CHARS = set(" .,;:!?-'\"()[]{}%/$#&@*+=\n\t")


def word_count(text: str) -> int:
    return len(_WORD.findall(text or ""))


def char_count(text: str) -> int:
    return len((text or "").strip())


def _looks_like_table(markdown: str) -> bool:
    # A markdown table has several column pipes and a `|---` separator row —
    # both survive newline-collapse. (Multi-line form: ≥2 lines with ≥2 pipes.)
    if markdown.count("|") < 4:
        return False
    if _TABLE_SEP.search(markdown):
        return True
    return sum(1 for ln in markdown.splitlines() if ln.count("|") >= 2) >= 2


def classify_content_kind(markdown: str) -> str:
    """One of ``prose | table | list | code | sparse`` (``""`` for empty).

    Priority: code-fence > markdown-table > list-markers > sparse > prose. A
    routing hint for agents — never a filter."""
    md = (markdown or "").strip()
    if not md:
        return ""
    if "```" in md or "~~~" in md:
        return "code"
    if _looks_like_table(md):
        return "table"
    # A list: starts with a marker and has ≥2 markers (newline-agnostic).
    if _LIST_START.match(md) and len(_LIST_MARKER.findall(md)) >= 2:
        return "list"
    if word_count(md) < SPARSE_WORDS:
        return "sparse"
    return "prose"


#: Cap values stored per entity type per chunk (avoid bloat on dense chunks).
MAX_ENTITIES_PER_TYPE = 50

_MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
_ENTITY_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "url": re.compile(r"\b(?:https?://|www\.)[^\s)<>\]]+", re.I),
    "money": re.compile(
        r"[$€£¥]\s?\d[\d.,]*"
        r"|\b\d[\d.,]*\s?(?:USD|EUR|GBP|NOK|SEK|DKK|kr|dollars?|euros?|pounds?)\b"
        r"|\b(?:USD|EUR|GBP|NOK|SEK|DKK|kr)\s?\d[\d.,]*",
        re.I,
    ),
    "date": re.compile(
        rf"\b\d{{4}}-\d{{2}}-\d{{2}}\b"
        rf"|\b\d{{1,2}}/\d{{1,2}}/\d{{2,4}}\b"
        rf"|\b{_MONTHS}\s+\d{{1,2}},?\s+\d{{4}}\b"
        rf"|\b\d{{1,2}}\s+{_MONTHS}\s+\d{{4}}\b",
        re.I,
    ),
    # Structured codes: letters then digits with a separator, or ≥3 digits.
    "identifier": re.compile(r"\b[A-Z]{2,}[-/]\d{2,}\b|\b[A-Z]{2,}\d{3,}\b"),
}


def extract_entities(text: str) -> dict[str, list[str]]:
    """Cheap, deterministic structured-entity extraction (regex) — dates, money,
    emails, URLs, identifiers. Generic and **recall-oriented**: an advisory
    routing hint (`MATCH (c:Chunk:HasMoney)`), not a guarantee. Domain-specific
    entity types belong in a vertical, not here. Returns `{type: [values]}`,
    each list order-preserving, de-duplicated, and capped."""
    text = text or ""
    out: dict[str, list[str]] = {}
    for etype, pat in _ENTITY_PATTERNS.items():
        seen: list[str] = []
        for m in pat.finditer(text):
            v = m.group(0).strip()
            if v and v not in seen:
                seen.append(v)
                if len(seen) >= MAX_ENTITIES_PER_TYPE:
                    break
        if seen:
            out[etype] = seen
    return out


def text_quality(markdown: str) -> float:
    """A 0..1 heuristic for how clean the extracted text looks (1.0 = clean).
    Low scores flag likely-garbled extraction (bad OCR/encoding). Advisory."""
    md = (markdown or "").strip()
    if not md:
        return 1.0  # nothing to judge — not "low quality"
    clean = sum(1 for ch in md if ch.isalnum() or ch in _CLEAN_CHARS)
    ratio = clean / len(md)
    words = _WORD.findall(md)
    if words:
        avg = sum(len(w) for w in words) / len(words)
        if avg > 18 or avg < 1.5:  # garble tends toward very long / very short tokens
            ratio *= 0.6
    return round(ratio, 3)
