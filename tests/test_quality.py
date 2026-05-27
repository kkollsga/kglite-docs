"""Quality: grounding checks + claim verification.

These tests use the stub embedder, which deterministically hashes the
text. That means "identical sentence to source chunk" will produce
high cosine; "totally different sentence" produces low cosine.
"""

from __future__ import annotations

from pathlib import Path

from kglite_docs import Corpus


def _ingest(corpus: Corpus, tmp_path: Path) -> str:
    p = tmp_path / "q.md"
    p.write_text(
        "# Section\n\nDense passage retrieval uses BERT and a dual encoder.\n",
        encoding="utf-8",
    )
    corpus.ingest(p)
    return corpus.search("Dense passage retrieval", top_k=1)[0]["id"]


def test_grounded_summary_marked_supported(corpus: Corpus, tmp_path: Path) -> None:
    cid = _ingest(corpus, tmp_path)
    # Use the exact source sentence as the summary
    sid = corpus.add_summary(
        cid, "Dense passage retrieval uses BERT and a dual encoder.",
        agent_id="alice", model="m",
    )
    g = corpus.check_grounding(sid, threshold=0.95)
    assert g["supported_fraction"] == 1.0


def test_ungrounded_summary_flagged(corpus: Corpus, tmp_path: Path) -> None:
    cid = _ingest(corpus, tmp_path)
    # Patently unrelated sentence with no shared token-hash
    sid = corpus.add_summary(
        cid, "The Eiffel Tower was built in 1889 for the World's Fair.",
        agent_id="alice", model="m",
    )
    g = corpus.check_grounding(sid, threshold=0.9)
    assert g["supported_fraction"] < 1.0
    assert g["weak_sentences"]


def test_verify_claim_returns_support_list(corpus: Corpus, tmp_path: Path) -> None:
    _ingest(corpus, tmp_path)
    v = corpus.verify_claim("BERT and dual encoder", top_k=3)
    assert "support" in v and isinstance(v["support"], list)
