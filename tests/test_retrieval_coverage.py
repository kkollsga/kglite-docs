"""BUG-2/FEAT-3: retrieval over an unindexed corpus is a loud signal, not a
silent []. Fully unindexed → NotIndexedError; partially indexed → a warning
plus a `searched_fraction` < 1.0 on the composed bundle."""

from __future__ import annotations

from pathlib import Path

import pytest

from kglite_docs import Corpus
from kglite_docs.errors import NotIndexedError


def _ingest(corpus: Corpus, tmp_path: Path, name: str, body: str) -> str:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return corpus.ingest(p).doc_id  # embed=False by default


def test_search_on_unindexed_corpus_raises(corpus: Corpus, tmp_path: Path) -> None:
    _ingest(corpus, tmp_path, "a.md", "# A\n\nDense passage retrieval uses BERT.\n")
    with pytest.raises(NotIndexedError):
        corpus.search("retrieval")
    with pytest.raises(NotIndexedError):
        corpus.compose_context("retrieval")


def test_empty_corpus_returns_empty_not_raises(corpus: Corpus) -> None:
    # No ready chunks at all → nothing to search is honest; don't cry "unindexed".
    assert corpus.search("anything") == []


def test_partial_index_warns_and_reports_fraction(corpus: Corpus, tmp_path: Path) -> None:
    _ingest(corpus, tmp_path, "a.md", "# A\n\nDense passage retrieval uses BERT.\n")
    corpus.index()  # embed doc A's chunks
    # A second, unembedded document now makes the corpus partially indexed.
    _ingest(corpus, tmp_path, "b.md", "# B\n\nLSTM recurrent networks were superseded.\n")

    with pytest.warns(UserWarning, match="unembedded are invisible"):
        corpus.search("retrieval")

    with pytest.warns(UserWarning):
        bundle = corpus.compose_context("retrieval")
    assert 0.0 < bundle["searched_fraction"] < 1.0


def test_fully_indexed_no_warning_full_fraction(
    corpus: Corpus, tmp_path: Path, recwarn: pytest.WarningsRecorder
) -> None:
    _ingest(corpus, tmp_path, "a.md", "# A\n\nDense passage retrieval uses BERT.\n")
    corpus.index()
    bundle = corpus.compose_context("retrieval")
    assert bundle["searched_fraction"] == 1.0
    assert not [w for w in recwarn.list if issubclass(w.category, UserWarning)]
