"""0.0.14 Phase 3: source-party dimension — whose words a document holds."""

from __future__ import annotations

import pytest

from kglite_docs import Corpus, parties
from kglite_docs.errors import InvalidEnumError

parties.register_source_party("tp_adverse", description="the opposing party")


def test_source_party_at_ingest_labels_doc_and_chunks(corpus: Corpus) -> None:
    r = corpus.ingest(text="# Deposition\n\nI admit the debt.", title="dep",
                      source_party="tp_adverse")
    assert corpus.get_document(r.doc_id)["source_party"] == "tp_adverse"
    labels = corpus.cypher(
        "MATCH (c:Chunk {doc_id: $d}) RETURN labels(c) AS L", params={"d": r.doc_id}
    ).to_list()[0]["L"]
    assert "TpAdverse" in labels  # free-text → PascalCase label
    # queryable as a predicate
    assert corpus.cypher("MATCH (c:Chunk:TpAdverse) RETURN count(c) AS n").to_list()[0]["n"] >= 1


def test_ledger_surfaces_source_party(corpus: Corpus) -> None:
    r = corpus.ingest(text="# Dep\n\nI admit it.", title="d", source_party="tp_adverse")
    ch = corpus.cypher("MATCH (c:Chunk:Ready {doc_id:$d}) RETURN c.id AS id",
                       params={"d": r.doc_id}).to_list()[0]["id"]
    sid = corpus.define_study("Q", created_by="lead")
    corpus.assess(sid, ch, stance="supports", weight=0.9, agent_id="a1", provenance="primary_text")
    row = corpus.study_ledger(sid)["rows"][0]
    # An admission against interest: primary text authored by the adverse party.
    assert row["source_party"] == "tp_adverse" and row["provenance"] == "primary_text"


def test_retag_swaps_label_cleanly(corpus: Corpus) -> None:
    parties.register_source_party("tp_court", description="the court")
    r = corpus.ingest(text="# X\n\nbody text here please.", title="d", source_party="tp_adverse")
    corpus.set_source_party(r.doc_id, "tp_court")
    labels = corpus.cypher(
        "MATCH (c:Chunk {doc_id:$d}) RETURN labels(c) AS L", params={"d": r.doc_id}
    ).to_list()[0]["L"]
    assert "TpCourt" in labels and "TpAdverse" not in labels
    assert corpus.get_document(r.doc_id)["source_party"] == "tp_court"


def test_available_and_validation(corpus: Corpus) -> None:
    vals = {p["value"] for p in corpus.available_source_parties()}
    assert "tp_adverse" in vals
    r = corpus.ingest(text="# X\n\nbody.", title="d")
    with pytest.raises(InvalidEnumError):  # empty party
        corpus.set_source_party(r.doc_id, "")
    with pytest.raises(InvalidEnumError):  # unknown doc
        corpus.set_source_party("doc_nope", "tp_adverse")
