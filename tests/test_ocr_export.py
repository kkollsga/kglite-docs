"""0.0.15 Phase 4: OCR export/import sidecar + OCR'd chunks are first-class
(embedded + searchable, not counted as unembedded)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pymupdf
import pytest

from kglite_docs import Corpus
from kglite_docs.errors import InvalidEnumError


def _scan(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 400, 400))
    pix.clear_with(180)
    page.insert_image(pymupdf.Rect(0, 0, 400, 400), pixmap=pix)
    page.insert_textbox(pymupdf.Rect(40, 350, 360, 390), "x", fontsize=10)
    doc.save(str(path))
    doc.close()


def _ocr_a_scan(corpus: Corpus, tmp_path: Path, text: str) -> str:
    _scan(tmp_path / "0000711-82.2025.pdf")
    r = corpus.ingest(tmp_path / "0000711-82.2025.pdf")
    pid = corpus.list_pending_ocr(include_images=False)[0]["page_id"]
    corpus.submit_ocr(pid, text, agent_id="sonnet", model="claude-sonnet-4-6")
    return r.doc_id


def test_ocrd_chunk_is_embedded_and_searchable(corpus: Corpus, tmp_path: Path) -> None:
    _ocr_a_scan(corpus, tmp_path, "# Confession\n\n" + ("The defendant admits owing the amount. " * 6))
    # First-class: marked embedded, counted embedded, search finds it with no
    # spurious "unembedded/invisible" warning or NotIndexedError.
    assert corpus.cypher("MATCH (c:Chunk:Embedded) RETURN count(c) AS n").to_list()[0]["n"] == 1
    assert corpus.coverage_report()["unembedded"] == 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hits = corpus.search("defendant admits owing", top_k=3)
    assert hits and "admits" in hits[0]["text"]


def test_export_then_import_round_trips(corpus: Corpus, tmp_path: Path) -> None:
    doc_id = _ocr_a_scan(corpus, tmp_path, "# Ruling\n\n" + ("The court found the party liable. " * 6))
    exp = corpus.export_ocr(doc_id)
    assert exp["out_path"].endswith("0000711-82.2025.ocr.json")
    sidecar = json.loads(Path(exp["out_path"]).read_text())
    assert sidecar["doc_id"] == doc_id and sidecar["page_count"] == 1
    pg = sidecar["pages"][0]
    assert pg["ocr_status"] == "ocr_ok" and pg["legible_chars"] > 0 and "liable" in pg["text"]
    assert sidecar["by_model"] == {"claude-sonnet-4-6": 1}
    # Wipe the OCR, then re-import from the sidecar — done once, re-applied.
    pid = corpus.cypher("MATCH (:Document {id:$d})-[:HAS_PAGE]->(p:Page) RETURN p.id AS id",
                        params={"d": doc_id}).to_list()[0]["id"]
    corpus.cypher("MATCH (:Page {id:$p})-[:HAS_CHUNK]->(ch:Chunk) DETACH DELETE ch", params={"p": pid})
    corpus.cypher("MATCH (p:Page {id:$p}) SET p.ocr_outcome = null, p.markdown = ''", params={"p": pid})
    imp = corpus.import_ocr(exp["out_path"])
    assert imp["pages_imported"] == 1 and imp["pages_skipped"] == 0
    assert corpus.ocr_status()["readable_pages"] == 1


def test_import_requires_ingested_doc(corpus: Corpus, tmp_path: Path) -> None:
    sidecar = tmp_path / "x.ocr.json"
    sidecar.write_text(json.dumps({"doc_id": "doc_missing", "pages": [
        {"page_number": 1, "text": "hi", "ocr_agent": "a"}]}))
    with pytest.raises(InvalidEnumError, match="not in this corpus"):
        corpus.import_ocr(str(sidecar))
