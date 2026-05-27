"""PDF → per-page structured content via pymupdf4llm + pymupdf.

pymupdf4llm gives us markdown-formatted text per page (headings,
paragraphs, lists, tables). We additionally inspect each page with
PyMuPDF for image presence so we can detect scanned/image-only pages
that need the OCR loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf
import pymupdf4llm


@dataclass
class PageContent:
    """One page's extracted state, ready for chunking."""

    page_number: int  # 1-indexed
    markdown: str
    has_text: bool
    has_images: bool
    needs_ocr: bool  # text-empty AND has images
    width_pt: float
    height_pt: float
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
        for i, page_md in enumerate(md_pages):
            page = doc[i]
            markdown = (page_md.get("text") or "").strip()
            has_text = bool(markdown)
            has_images = bool(page.get_images(full=False))
            out.append(
                PageContent(
                    page_number=i + 1,
                    markdown=markdown,
                    has_text=has_text,
                    has_images=has_images,
                    needs_ocr=(not has_text) and has_images,
                    width_pt=float(page.rect.width),
                    height_pt=float(page.rect.height),
                    metadata={
                        k: v
                        for k, v in page_md.items()
                        if k in {"toc_items", "tables", "images", "graphics"}
                    },
                )
            )
    return out


def render_page_png(path: str | Path, page_number: int, *, dpi: int = 200) -> bytes:
    """Rasterise a single page to PNG bytes. Used by the OCR handoff:
    the MCP `list_pending_ocr` tool returns these for an agent to read."""
    with pymupdf.open(str(path)) as doc:
        page = doc[page_number - 1]
        pix = page.get_pixmap(dpi=dpi)
        return bytes(pix.tobytes("png"))
