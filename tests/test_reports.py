"""0.0.15 Phase 7: versioned study reports on the graph (not .md litter)."""

from __future__ import annotations

from pathlib import Path

import pytest

from kglite_docs import Corpus
from kglite_docs.errors import InvalidEnumError


def test_named_append_only_versions(corpus: Corpus) -> None:
    sid = corpus.define_study("Q", created_by="lead")
    corpus.save_report(sid, name="client-brief", text="# Brief\n\nDraft one.",
                       agent_id="lead", cites=["finding_1"])
    v2 = corpus.save_report(sid, name="client-brief", text="# Brief\n\nDraft two.", agent_id="lead")
    corpus.save_report(sid, name="judge-conduct", text="# Conduct\n\nNotes.", agent_id="lead")
    assert v2["version"] == 2
    rs = {r["name"]: r for r in corpus.list_reports(sid)}
    assert rs["client-brief"]["latest_version"] == 2 and rs["client-brief"]["versions"] == 2
    assert rs["judge-conduct"]["latest_version"] == 1
    # latest-wins on read; history preserved
    assert corpus.get_report(sid, name="client-brief")["text"].endswith("Draft two.")
    assert corpus.get_report(sid, name="client-brief", version=1)["text"].endswith("Draft one.")
    assert corpus.get_report(sid, name="client-brief", version=1)["cites"] == ["finding_1"]


def test_export_report_on_demand(corpus: Corpus, tmp_path: Path) -> None:
    sid = corpus.define_study("Q", created_by="lead")
    corpus.save_report(sid, name="brief", text="# B\n\nthe brief body", agent_id="lead")
    out = tmp_path / "brief.md"
    res = corpus.export_report(sid, str(out), name="brief")
    assert out.read_text() == "# B\n\nthe brief body" and res["version"] == 1


def test_report_validation_and_cascade(corpus: Corpus) -> None:
    sid = corpus.define_study("Q", created_by="lead")
    with pytest.raises(InvalidEnumError):  # empty text
        corpus.save_report(sid, name="x", text="  ", agent_id="lead")
    with pytest.raises(InvalidEnumError):  # missing study
        corpus.save_report("study_nope", name="x", text="y", agent_id="lead")
    corpus.save_report(sid, name="x", text="content", agent_id="lead")
    corpus.delete_study(sid)
    assert corpus.cypher("MATCH (r:Report) RETURN count(r) AS n").to_list()[0]["n"] == 0
