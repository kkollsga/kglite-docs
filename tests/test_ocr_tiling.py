"""0.0.15 Phase 3: adaptive image prep — right-size the render, tile dense pages."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from kglite_docs import Corpus
from kglite_docs.ingest.parser import (
    MODEL_MAX_PIXELS,
    render_page_images,
    render_page_png,
)

_CAP = int(MODEL_MAX_PIXELS * 1.03)  # allow sub-1% pixmap rounding


def _scan(path: Path, w: int = 600, h: int = 800) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=w, height=h)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, w, h))
    pix.clear_with(180)
    page.insert_image(pymupdf.Rect(0, 0, w, h), pixmap=pix)
    page.insert_textbox(pymupdf.Rect(40, h - 100, w - 40, h - 10), "x", fontsize=10)
    doc.save(str(path))
    doc.close()


def test_render_png_fits_cap(tmp_path: Path) -> None:
    _scan(tmp_path / "p.pdf")
    # Even at a high dpi the whole-page preview is downscaled under the cap.
    png = render_page_png(tmp_path / "p.pdf", 1, dpi=400)
    pix = pymupdf.Pixmap(png)
    assert pix.width * pix.height <= _CAP


def test_dense_page_tiles_each_within_cap(tmp_path: Path) -> None:
    _scan(tmp_path / "p.pdf")
    tiles = render_page_images(tmp_path / "p.pdf", 1, dpi=200)  # 600x800@200 ≈ 3.7 MP
    assert len(tiles) >= 2
    for t in tiles:
        assert t["px"][0] * t["px"][1] <= _CAP
        assert t["image_b64"]
    # consecutive tiles overlap (text at a seam isn't lost)
    assert tiles[0]["bbox"][3] > tiles[1]["bbox"][1]
    # tiles cover the page top-to-bottom
    assert abs(tiles[0]["bbox"][1] - 0.0) < 1 and abs(tiles[-1]["bbox"][3] - 800.0) < 1


def test_small_page_single_tile(tmp_path: Path) -> None:
    _scan(tmp_path / "p.pdf", w=300, h=300)  # ~0.4 MP @200 → fits
    tiles = render_page_images(tmp_path / "p.pdf", 1, dpi=150)
    assert len(tiles) == 1 and tiles[0]["tile_index"] == 0


def test_request_tiles_then_submit_stitches(corpus: Corpus, tmp_path: Path) -> None:
    _scan(tmp_path / "p.pdf")
    corpus.ingest(tmp_path / "p.pdf")
    pid = corpus.list_pending_ocr(include_images=False)[0]["page_id"]
    task = corpus.request_ocr(page_id=pid, agent_id="ocr")
    assert task["tile_count"] >= 2 and len(task["tiles"]) == task["tile_count"]
    # page records the tiling for provenance
    assert corpus.cypher("MATCH (p:Page {id:$p}) RETURN p.ocr_tiles AS n",
                         params={"p": pid}).to_list()[0]["n"] == task["tile_count"]
    # transcribe each tile; submit stitches in tile_index order
    corpus.submit_ocr(pid, agent_id="ocr", tiles=[
        {"tile_index": 1, "markdown": "Second band: the appeal was denied and costs assessed. " * 3},
        {"tile_index": 0, "markdown": "# Header\n\nFirst band: the court found the defendant liable. " * 3},
    ])
    page_md = corpus.cypher("MATCH (p:Page {id:$p}) RETURN p.markdown AS m",
                            params={"p": pid}).to_list()[0]["m"]
    assert page_md.index("First band") < page_md.index("Second band")  # stitched in order
    assert corpus.ocr_status()["readable_pages"] == 1
