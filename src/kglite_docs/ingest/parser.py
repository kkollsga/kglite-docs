"""PDF → per-page structured content via pymupdf4llm + pymupdf.

pymupdf4llm gives us markdown-formatted text per page (headings,
paragraphs, lists, tables). We additionally inspect each page with
PyMuPDF for image presence so we can detect scanned/image-only pages
that need the OCR loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf
import pymupdf4llm

# A page that carries images but fewer than this many extractable
# alphanumeric chars is treated as image-only and routed to OCR. Tunable:
# a scanned exhibit yields ~0 real chars (just a "picture omitted" marker
# and/or a short footer); a genuine text page yields hundreds+.
OCR_TEXT_THRESHOLD = 120

# pymupdf4llm emits placeholders like `==> picture … intentionally omitted <==`
# for image regions; these are not real extractable text.
_IMG_PLACEHOLDER_RE = re.compile(r"(?is)==>.*?<==")
_OMITTED_LINE_RE = re.compile(r"(?im)^.*intentionally omitted.*$")
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def _extractable_alnum(markdown: str) -> int:
    """Count real extractable alphanumeric chars, ignoring pymupdf4llm image
    placeholders / markdown image syntax. Used to tell a text page from a
    scanned-image page whose only "text" is a placeholder."""
    cleaned = _IMG_PLACEHOLDER_RE.sub(" ", markdown or "")
    cleaned = _OMITTED_LINE_RE.sub(" ", cleaned)
    cleaned = _MD_IMAGE_RE.sub(" ", cleaned)
    return sum(1 for c in cleaned if c.isalnum())


def _needs_ocr(markdown: str, has_images: bool) -> bool:
    """True when a page has image content but negligible real text — i.e. a
    scanned/image-only page that must be OCR'd to be analyzed."""
    return has_images and _extractable_alnum(markdown) < OCR_TEXT_THRESHOLD


@dataclass
class PageContent:
    """One page's extracted state, ready for chunking."""

    page_number: int  # 1-indexed
    markdown: str
    has_text: bool
    has_images: bool
    needs_ocr: bool  # image content but < OCR_TEXT_THRESHOLD extractable chars
    width_pt: float
    height_pt: float
    image_block_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_pdf(path: str | Path) -> list[PageContent]:
    """Extract per-page markdown + layout flags. Returns one entry per page,
    ordered 1..N. Safe to call on an OCR-less scanned PDF — those pages
    come back with empty markdown and `needs_ocr=True`."""
    path = Path(path)
    # page_chunks=True returns a list of dicts, one per page, each with
    # markdown text + structural metadata.
    md_pages: list[dict[str, Any]] = pymupdf4llm.to_markdown(
        str(path), page_chunks=True, show_progress=False
    )

    out: list[PageContent] = []
    with pymupdf.open(str(path)) as doc:
        # Document-level outline (bookmarks) — the most authoritative section
        # boundaries when present. Ride it out on the first page's metadata so
        # the list[PageContent] contract is unchanged.
        try:
            outline = doc.get_toc(simple=True) or []
        except Exception:  # pragma: no cover - defensive (malformed outline)
            outline = []
        for i, page_md in enumerate(md_pages):
            page = doc[i]
            markdown = (page_md.get("text") or "").strip()
            if not markdown:
                # pymupdf4llm sometimes returns empty markdown for a page it
                # can't structure even though the page *has* extractable text.
                # Fall back to raw PyMuPDF extraction so that text isn't silently
                # dropped as an :Empty chunk (and mis-classified as not-needing-
                # OCR because there are no raster images). Honest coverage.
                markdown = (page.get_text("text") or "").strip()
            has_text = bool(markdown)
            image_block_count = len(page.get_images(full=False))
            has_images = image_block_count > 0
            out.append(
                PageContent(
                    page_number=i + 1,
                    markdown=markdown,
                    has_text=has_text,
                    has_images=has_images,
                    needs_ocr=_needs_ocr(markdown, has_images),
                    width_pt=float(page.rect.width),
                    height_pt=float(page.rect.height),
                    image_block_count=image_block_count,
                    metadata={
                        k: v
                        for k, v in page_md.items()
                        if k in {"toc_items", "tables", "images", "graphics"}
                    },
                )
            )
    if out and outline:
        out[0].metadata["doc_outline"] = outline
    return out


def render_page_png(path: str | Path, page_number: int, *, dpi: int = 200) -> bytes:
    """Rasterise a single page to PNG bytes. Used by the OCR handoff:
    the MCP `list_pending_ocr` tool returns these for an agent to read."""
    with pymupdf.open(str(path)) as doc:
        page = doc[page_number - 1]
        pix = page.get_pixmap(dpi=dpi)
        return bytes(pix.tobytes("png"))
