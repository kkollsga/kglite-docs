"""Phase 11.1: deterministic content signals at ingest — content_kind, word/char
count, quality flag, boilerplate. Additive + lossless: a signal labels, never
drops, a chunk."""

from __future__ import annotations

from pathlib import Path

from kglite_docs import Corpus
from kglite_docs.signals import (
    char_count,
    classify_content_kind,
    text_quality,
    word_count,
)


def test_classify_content_kind_collapsed_forms() -> None:
    # Chunk text is whitespace-collapsed; detection must survive that.
    assert classify_content_kind("| a | b | |---|---| | 1 | 2 |") == "table"
    assert classify_content_kind("- one - two - three") == "list"
    assert classify_content_kind("1. first 2. second 3. third") == "list"
    assert classify_content_kind("```py print(1) ```") == "code"
    assert classify_content_kind("See above.") == "sparse"
    assert classify_content_kind(
        "This is a normal paragraph of several words describing the method here."
    ) == "prose"
    # A prose chunk that merely contains dashes is not a list.
    assert classify_content_kind(
        "The cost-benefit analysis - a key point - matters quite a lot here."
    ) == "prose"
    assert classify_content_kind("") == ""


def test_text_quality_and_counts() -> None:
    assert text_quality("This is clean readable prose with normal words.") > 0.9
    assert text_quality("x9#@ ~~~ ¤¤¤ §§§ ‹‹‹ zzqqxx ￿￿") < 0.55
    assert word_count("one two three") == 3
    assert char_count("  abc  ") == 3


def test_ingest_stamps_content_signals(corpus: Corpus, tmp_path: Path) -> None:
    md = (
        "# Intro\n\nA normal paragraph of prose with plenty of words to classify.\n\n"
        "# Data\n\n| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\n"
        "# Steps\n\n- first step here\n- second step here\n- third step here\n"
    )
    p = tmp_path / "d.md"
    p.write_text(md, encoding="utf-8")
    doc_id = corpus.ingest(p).doc_id

    kinds = {
        r["k"] for r in corpus.cypher(
            "MATCH (c:Chunk:Ready) WHERE c.doc_id = $d RETURN c.content_kind AS k",
            params={"d": doc_id},
        ).to_list()
    }
    assert {"prose", "table", "list"} <= kinds
    # Label predicates work (the point of triage routing).
    n_tables = corpus.cypher(
        "MATCH (c:Chunk:Table) WHERE c.doc_id = $d RETURN count(c) AS n",
        params={"d": doc_id},
    ).to_list()[0]["n"]
    assert n_tables >= 1

    # get_chunk surfaces the signals (both item + attr access).
    cid = corpus.cypher(
        "MATCH (c:Chunk:Prose) WHERE c.doc_id = $d RETURN c.id AS id",
        params={"d": doc_id},
    ).to_list()[0]["id"]
    d = corpus.get_chunk(cid)
    assert d is not None
    assert d["content_kind"] == "prose" and d.word_count > 0
    assert d["char_count"] > 0 and 0.0 <= d["quality_score"] <= 1.0
    assert d["boilerplate"] is False


def test_boilerplate_flagged_for_duplicate_chunks(corpus: Corpus, tmp_path: Path) -> None:
    # Verbatim-repeated content (a disclaimer, a repeated page) yields duplicate
    # chunks → flagged boilerplate, but kept (lossless).
    block = "This standard confidentiality disclaimer is repeated verbatim here. " * 80
    md = tmp_path / "rep.md"
    md.write_text(
        f"# Intro\n\nUnique opening words for the first part.\n\n{block}\n\n"
        f"# Middle\n\nDifferent unique words in the middle.\n\n{block}\n",
        encoding="utf-8",
    )
    doc_id = corpus.ingest(md).doc_id

    bp = corpus.cypher(
        "MATCH (c:Chunk:Boilerplate) WHERE c.doc_id = $d RETURN c.text AS t",
        params={"d": doc_id},
    ).to_list()
    assert len(bp) >= 2 and all("confidentiality disclaimer" in r["t"] for r in bp)
    # Lossless: the boilerplate chunks are still present as ready chunks.
    total = corpus.cypher(
        "MATCH (c:Chunk:Ready) WHERE c.doc_id = $d RETURN count(c) AS n",
        params={"d": doc_id},
    ).to_list()[0]["n"]
    assert total >= len(bp)
    # A unique chunk is not boilerplate.
    uniq = corpus.cypher(
        "MATCH (c:Chunk:Ready) WHERE c.doc_id = $d AND c.boilerplate = false RETURN count(c) AS n",
        params={"d": doc_id},
    ).to_list()[0]["n"]
    assert uniq >= 1
