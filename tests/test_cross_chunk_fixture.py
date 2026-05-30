"""0.0.13 Phase 9: the TR-7788 cross-chunk regression fixture, wired into CI.

The fixture (sample_data/cross_chunk_fixture/) encodes the trap: each ruling is
routine in isolation; the defects (F1 disparate treatment, F2 conflicting
dispositions) live only across sections. A per-chunk pass must recover NOTHING; a
synthesis pass must recover F1/F2 and leave the negative control alone — checked
against the fixture's own gold standard."""

from __future__ import annotations

import importlib.util
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import pytest

from kglite_docs import Corpus
from kglite_docs.errors import SynthesisRequiredError

FIX = Path(__file__).resolve().parent.parent / "sample_data" / "cross_chunk_fixture"


def _load_harness() -> Any:
    spec = importlib.util.spec_from_file_location("cc_harness", FIX / "harness.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build(corpus: Corpus) -> tuple[str, dict[str, str], dict[str, Any]]:
    raw = (FIX / "CASE_TR-7788.md").read_text(encoding="utf-8")
    gold = json.loads((FIX / "expected_findings.json").read_text(encoding="utf-8"))
    for part in re.split(r"(?m)^## ", raw)[1:]:
        corpus.ingest(text="## " + part, title=part.splitlines()[0].strip(),
                      format="md", structure_aware=True)
    chunks = corpus.cypher("MATCH (n:Chunk) RETURN n.id AS id, n.text AS t").to_list()
    anchors = {
        name: next((r["id"] for r in chunks if a["marker"].lower() in (r["t"] or "").lower()), None)
        for name, a in gold["anchors"].items()
    }
    sid = corpus.define_study(gold["per_chunk_baseline_expectation"]["question"], created_by="harness")
    return sid, anchors, gold


def test_fixture_anchors_resolve_and_baseline_misses(corpus: Corpus) -> None:
    sid, anchors, _ = _build(corpus)
    for name in ("default_ruling", "dismissal_ruling", "merits_condemnation",
                 "appeal_assertion", "omission_target", "nc_extension"):
        assert anchors[name], f"anchor {name} did not resolve"
    # Per-chunk baseline: each ruling, read alone, is routine → neutral.
    for name in ("default_ruling", "dismissal_ruling", "merits_condemnation"):
        corpus.assess(sid, anchors[name], stance="neutral", weight=0.0, agent_id="baseline")
    # The trap: same-chunk conflicts is blind to the cross-chunk defect.
    assert corpus.study_conflicts(sid)["total"] == 0
    assert corpus.list_findings(sid) == []  # nothing recovered per-chunk


def test_fixture_synthesis_recovers_gold(corpus: Corpus) -> None:
    harness = _load_harness()
    sid, anchors, _ = _build(corpus)
    # The synthesis pass records the emergent cross-chunk findings.
    corpus.create_finding(sid, statement="disparate treatment of the parties' absences",
                          supporting_chunk_ids=[anchors["default_ruling"], anchors["dismissal_ruling"]],
                          stance="against", weight=0.9, agent_id="synth",
                          finding_type="disparate_treatment", provenance="primary_text")
    corpus.create_finding(sid, statement="two conflicting operative dispositions",
                          supporting_chunk_ids=[anchors["dismissal_ruling"], anchors["merits_condemnation"]],
                          stance="against", weight=0.8, agent_id="synth",
                          finding_type="conflicting_dispositions", provenance="primary_text")
    corpus.create_finding(sid, statement="undecided jurisdiction motion",
                          supporting_chunk_ids=[anchors["omission_target"]],
                          stance="against", weight=0.7, agent_id="synth",
                          finding_type="omission", provenance="primary_text")
    # Hand our findings to the fixture's own gold-standard checker.
    candidate = [{
        "type": f["finding_type"],
        "linked_chunk_ids": [s["id"] for s in f["supporting"]],
        "provenance": f["provenance"], "weight": f["weight"],
        "escalation_state": f["escalation_state"],
    } for f in corpus.list_findings(sid)]
    assert harness.check(candidate, anchors) is True


def test_fixture_conclude_is_gated(corpus: Corpus) -> None:
    sid, _, _ = _build(corpus)
    # Can't conclude while blind to the cross-chunk class…
    with pytest.raises(SynthesisRequiredError):
        corpus.conclude_study(sid, "complete", agent_id="lead")
    # …until the synthesis pass has run.
    corpus.synthesize_study(sid, agent_id="lead")
    assert isinstance(corpus.conclude_study(sid, "complete", agent_id="lead"), str)


def test_make_chunks_helper() -> None:
    # The public test helper: N sections → N chunks (the structure_aware lever).
    from kglite_docs.testing import make_chunks
    from tests.conftest import StubEmbedder
    c = Corpus.create(Path(tempfile.mkdtemp()) / "t.kgl", embedder=StubEmbedder())
    assert len(make_chunks(c, 5)) == 5
