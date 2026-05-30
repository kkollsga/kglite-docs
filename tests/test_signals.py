"""Phase 11.1: deterministic content signals at ingest — content_kind, word/char
count, quality flag, boilerplate. Additive + lossless: a signal labels, never
drops, a chunk."""

from __future__ import annotations

from pathlib import Path

from kglite_docs import Corpus
from kglite_docs.signals import (
    char_count,
    classify_content_kind,
    extract_entities,
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


def test_extract_entities() -> None:
    e = extract_entities(
        "Email jane@acme.com, see https://acme.com on 2021-03-04; paid $1,250,000 "
        "and NOK 50000 re PAD-003 and ABC1234."
    )
    assert e["email"] == ["jane@acme.com"]
    assert e["url"] == ["https://acme.com"]
    assert "2021-03-04" in e["date"]
    assert "$1,250,000" in e["money"] and any("NOK" in m.upper() for m in e["money"])
    assert set(e["identifier"]) >= {"PAD-003", "ABC1234"}
    # No false positives on plain prose.
    assert extract_entities("Dense retrieval uses BERT and a dual encoder here.") == {}


def test_ingest_pretags_entities(corpus: Corpus, tmp_path: Path) -> None:
    md = tmp_path / "e.md"
    md.write_text(
        "# Filing\n\nThe settlement of $1,250,000 was wired to ops@acme.com on "
        "2021-03-04 under reference PAD-003.\n\n"
        "# Other\n\nA paragraph with no structured entities at all in it here.\n",
        encoding="utf-8",
    )
    doc_id = corpus.ingest(md).doc_id

    # Label predicates route an agent straight to the high-value chunk.
    money = corpus.cypher(
        "MATCH (c:Chunk:HasMoney) WHERE c.doc_id = $d RETURN c.id AS id",
        params={"d": doc_id},
    ).to_list()
    assert len(money) == 1
    # get_chunk surfaces parsed entities.
    d = corpus.get_chunk(money[0]["id"])
    assert d is not None
    ents = d["entities"]
    assert "$1,250,000" in ents["money"]
    assert ents["email"] == ["ops@acme.com"]
    assert "2021-03-04" in ents["date"] and "PAD-003" in ents["identifier"]
    # The entity-free chunk carries no Has* label / empty entities.
    plain = corpus.cypher(
        "MATCH (c:Chunk:Ready) WHERE c.doc_id = $d AND NOT c:HasMoney "
        "RETURN c.id AS id", params={"d": doc_id},
    ).to_list()
    assert plain and corpus.get_chunk(plain[0]["id"])["entities"] == {}


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


def test_triage_map_aggregates_signals(corpus: Corpus, tmp_path: Path) -> None:
    md = tmp_path / "m.md"
    md.write_text(
        "# Prose\n\nA normal paragraph of prose with several words for the map test.\n\n"
        "# Table\n\n| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\n"
        "# Facts\n\nPaid $1,250,000 to ops@acme.com on 2021-03-04 under PAD-003.\n",
        encoding="utf-8",
    )
    doc_id = corpus.ingest(md).doc_id
    m = corpus.triage_map(doc_id=doc_id)
    assert m["chunks"] >= 3 and m["ready"] >= 3
    assert m["content_kinds"].get("prose", 0) >= 1
    assert m["content_kinds"].get("table", 0) >= 1
    assert m["entities"].get("money", 0) >= 1
    assert m["entities"].get("email", 0) >= 1
    assert m["unembedded"] == m["ready"]  # not indexed in this test
    assert isinstance(m["summary"], str) and "chunks" in m["summary"]


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
