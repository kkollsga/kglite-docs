"""Hashing primitives — file_hash + text_hash + normalisation."""

from __future__ import annotations

from pathlib import Path

from kglite_docs.ingest.hashing import combined_hash, file_hash, normalize_text, text_hash


def test_file_hash_is_deterministic(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_bytes(b"hello world")
    assert file_hash(f) == file_hash(f)


def test_file_hash_prefix() -> None:
    out = file_hash(Path(__file__))
    assert out.startswith("doc_")
    assert len(out) == 4 + 64  # "doc_" + sha256 hex


def test_text_hash_is_whitespace_stable() -> None:
    a = "Hello   world\n\n  foo"
    b = "Hello world\nfoo"
    assert text_hash(a) == text_hash(b)


def test_text_hash_preserves_case() -> None:
    # case is semantically meaningful (acronyms, proper nouns)
    assert text_hash("BERT") != text_hash("bert")


def test_normalize_text_basic() -> None:
    assert normalize_text("  a   b\nc  ") == "a b c"


def test_combined_hash_is_order_sensitive() -> None:
    assert combined_hash(["a", "b"]) != combined_hash(["b", "a"])


def test_combined_hash_handles_empty_sequence() -> None:
    # not a crash, returns a stable empty-input hash
    assert isinstance(combined_hash([]), str)
