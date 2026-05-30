"""0.0.12 Phase 3: the bundled legal schema pack (data only — the engine never
names a legal term; these flow through the generic registration seam)."""

from __future__ import annotations

import pytest

from kglite_docs import classify, schema
from kglite_docs.schemas import available_schemas, load_schema


def test_load_registers_legal_vocabulary() -> None:
    load_schema("legal")
    vals = schema.valid_element_values()
    # The scenario's named elements are all first-class.
    assert {"holding", "judge_remark", "settlement", "testimony", "statute"} <= vals
    assert schema.element_label("judge_remark") == "JudgeRemark"
    assert schema.element_label("case_citation") == "CaseCitation"
    # Family discriminators resolve.
    assert schema.label_for("chunk.legal_role", "holding") == "Holding"
    assert "Settlement" in schema.labels_for("chunk.legal_role")


def test_load_is_idempotent_and_unknown_raises() -> None:
    load_schema("legal")
    load_schema("legal")  # no error on re-load
    assert "legal" in available_schemas()
    with pytest.raises(ValueError, match="unknown schema pack"):
        load_schema("medical")


def test_rubric_text_covers_all_elements() -> None:
    from kglite_docs.schemas import legal
    text = legal.rubric_text()
    for eid in (*legal.LEGAL_ROLE, *legal.LEGAL_AUTHORITY, *legal.LEGAL_EVIDENCE):
        assert eid in legal.RUBRIC and eid in text
    assert "multi-label" in text.lower()


def test_classify_with_legal_elements_routes(corpus, tmp_path) -> None:  # type: ignore[no-untyped-def]
    load_schema("legal")
    p = tmp_path / "case.md"
    p.write_text(
        "# Order\n\nThe court holds the motion is granted; judgment is entered for plaintiff.\n\n"
        "# Aside\n\nThe judge remarked that counsel's tardiness was, frankly, exasperating.\n",
        encoding="utf-8",
    )
    corpus.ingest(p, structure_aware=True)
    ids = [r["id"] for r in corpus.cypher(
        "MATCH (c:Chunk:Ready) RETURN c.id AS id ORDER BY c.chunk_index"
    ).to_list()]
    classify.classify_chunk(corpus.store, chunk_id=ids[0], elements=["holding", "disposition_order"], agent_id="a1")
    classify.classify_chunk(corpus.store, chunk_id=ids[1], elements=["judge_remark"], agent_id="a1")
    # A "rank judge remarks" study would route here — only the remark chunk.
    remarks = corpus.cypher("MATCH (c:Chunk:JudgeRemark) RETURN c.id AS id").to_list()
    assert [r["id"] for r in remarks] == [ids[1]]
    holdings = corpus.cypher("MATCH (c:Chunk:Holding) RETURN c.id AS id").to_list()
    assert [r["id"] for r in holdings] == [ids[0]]
