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

# Image-area fraction of a page that counts as "image-bearing" / "image-dominated".
# Detection is recall-biased: missing a scanned exhibit (the silent miss) is far
# costlier than over-flagging a figure page (the agent just OCRs it cheaply).
COVER_MIN = 0.5       # enough image area to treat sparse text as needing OCR
COVER_DOMINANT = 0.6  # an image-dominated page (likely a full-page scan)
# A genuine text page yields hundreds+ of real alnum chars; below this on an
# image-dominated page is treated as a scan whose "text" is junk OCR-able later.
TEXT_RICH = 400

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


def _image_coverage(page: Any) -> float:
    """Fraction of the page area covered by raster images (0..1). Catches scans
    whose raster isn't reported as an embedded XObject by `get_images`, and lets
    us judge a page *image-dominated* rather than merely *image-bearing*."""
    try:
        infos = page.get_image_info()
    except Exception:  # pragma: no cover - defensive (odd page objects)
        return 0.0
    area = abs(float(page.rect.width) * float(page.rect.height)) or 1.0
    covered = 0.0
    for info in infos:
        bbox = info.get("bbox") if isinstance(info, dict) else None
        if not bbox:
            continue
        x0, y0, x1, y1 = bbox
        covered += abs((x1 - x0) * (y1 - y0))
    return min(covered / area, 1.0)


def _needs_ocr(markdown: str, *, has_images: bool, image_coverage: float) -> bool:
    """True when a page is a scan/image-only page that must be OCR'd to be
    analyzed. Density- + coverage-aware (recall-biased) so the failure modes that
    let scanned exhibits silently pass as `ready` are caught:

    - **sparse text on an image-bearing page** — the original rule, now also
      tripped by image *coverage* (a full-page raster with no detected XObject);
    - **image-dominated but not text-rich** — a full-page scan whose junk
      fragments clear the sparse floor (>120 chars) is still a scan.
    """
    alnum = _extractable_alnum(markdown)
    if alnum < OCR_TEXT_THRESHOLD and (has_images or image_coverage >= COVER_MIN):
        return True
    # image-dominated but not text-rich → a full-page scan with junk fragments
    return image_coverage >= COVER_DOMINANT and alnum < TEXT_RICH


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
    image_coverage: float = 0.0  # fraction of page area covered by raster images
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
            image_coverage = _image_coverage(page)
            out.append(
                PageContent(
                    page_number=i + 1,
                    markdown=markdown,
                    has_text=has_text,
                    has_images=has_images,
                    needs_ocr=_needs_ocr(
                        markdown, has_images=has_images, image_coverage=image_coverage,
                    ),
                    width_pt=float(page.rect.width),
                    height_pt=float(page.rect.height),
                    image_block_count=image_block_count,
                    image_coverage=image_coverage,
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


#: A vision model's effective input is ~1.15 MP (~1568 px long edge); a page
#: rendered larger is *silently* downscaled by the API — which destroys fine
#: print on dense/faded scans. So the library (the image SENDER) right-sizes what
#: it ships instead of leaving an opaque resize to the model.
MODEL_MAX_PIXELS = 1_150_000


def _fit_scale(rect: Any, dpi: int, max_pixels: int) -> float:
    """The pymupdf render scale for `dpi`, reduced so the rasterised page lands
    at or under `max_pixels`."""
    base = dpi / 72.0
    px = (rect.width * base) * (rect.height * base)
    return base * (max_pixels / px) ** 0.5 if px > max_pixels else base


def render_page_png(
    path: str | Path, page_number: int, *, dpi: int = 200,
    max_pixels: int = MODEL_MAX_PIXELS,
) -> bytes:
    """Rasterise a whole page to PNG bytes, **downscaled to fit `max_pixels`** so
    the image isn't silently shrunk by the consumer. Used for previews
    (`list_pending_ocr`). For detail-preserving OCR of dense pages use
    `render_page_images` (which tiles instead of downscaling)."""
    with pymupdf.open(str(path)) as doc:
        page = doc[page_number - 1]
        scale = _fit_scale(page.rect, dpi, max_pixels)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale))
        return bytes(pix.tobytes("png"))


def render_page_images(
    path: str | Path, page_number: int, *, dpi: int = 200,
    max_pixels: int = MODEL_MAX_PIXELS, overlap: float = 0.08,
) -> list[dict[str, Any]]:
    """Render a page as one or more base64 PNG **tiles**, each within
    `max_pixels`. If the page at `dpi` already fits, returns a single full-page
    tile. Otherwise splits it into overlapping horizontal bands rendered at full
    `dpi` (detail preserved within the cap) — the lever for dense/faded scans
    that a single downscaled image turns to blur. Each tile:
    `{tile_index, image_b64, image_mime, bbox:[x0,y0,x1,y1], px:[w,h]}`."""
    import base64
    import math

    with pymupdf.open(str(path)) as doc:
        page = doc[page_number - 1]
        rect = page.rect
        base = dpi / 72.0
        full_px = (rect.width * base) * (rect.height * base)

        def _tile(idx: int, clip: Any) -> dict[str, Any]:
            band_px = (clip.width * base) * (clip.height * base)
            scale = base if band_px <= max_pixels else base * (max_pixels / band_px) ** 0.5
            pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=clip)
            return {
                "tile_index": idx,
                "image_b64": base64.b64encode(bytes(pix.tobytes("png"))).decode("ascii"),
                "image_mime": "image/png",
                "bbox": [clip.x0, clip.y0, clip.x1, clip.y1],
                "px": [pix.width, pix.height],
            }

        if full_px <= max_pixels:
            return [_tile(0, rect)]
        n = math.ceil(full_px / max_pixels)
        band = rect.height / n
        ov = band * overlap
        tiles: list[dict[str, Any]] = []
        for i in range(n):
            y0 = rect.y0 + i * band - (ov if i > 0 else 0)
            y1 = rect.y0 + (i + 1) * band + (ov if i < n - 1 else 0)
            tiles.append(_tile(i, pymupdf.Rect(rect.x0, max(rect.y0, y0), rect.x1, min(rect.y1, y1))))
        return tiles
