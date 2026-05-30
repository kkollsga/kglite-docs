"""FEAT-12: single-writer advisory lock. A second *live* process opening the
same `.kgl` raises ConcurrencyError; same-process reopen and stale locks are
allowed."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kglite_docs import Corpus
from kglite_docs.errors import ConcurrencyError


def test_same_process_reopen_is_allowed(tmp_path: Path, stub_embedder: object) -> None:
    db = tmp_path / "shared.kgl"
    c1 = Corpus.create(db, embedder=stub_embedder)
    c1.save()
    # Same process, second handle — the PID matches, so no conflict.
    c2 = Corpus.open(db, embedder=stub_embedder)
    assert c2 is not None
    c1.store.close()
    c2.store.close()


def test_foreign_live_writer_raises(tmp_path: Path, stub_embedder: object) -> None:
    db = tmp_path / "shared.kgl"
    c1 = Corpus.create(db, embedder=stub_embedder)
    c1.save()
    # Simulate a different live process owning the lock: write a PID that is
    # alive but not us (PID 1 — init/launchd — is always running).
    lock = Path(str(db) + ".lock")
    lock.write_text("1")
    with pytest.raises(ConcurrencyError):
        Corpus.open(db, embedder=stub_embedder)
    c1.store.close()


def test_stale_lock_is_reclaimed(tmp_path: Path, stub_embedder: object) -> None:
    db = tmp_path / "shared.kgl"
    Corpus.create(db, embedder=stub_embedder).save()
    # A dead PID (very high, not running) → the lock is stale and reclaimed.
    lock = Path(str(db) + ".lock")
    dead_pid = 2_000_000_000
    assert dead_pid != os.getpid()
    lock.write_text(str(dead_pid))
    c = Corpus.open(db, embedder=stub_embedder)  # should not raise
    assert lock.read_text().strip() == str(os.getpid())
    c.store.close()


def test_in_memory_corpus_takes_no_lock(stub_embedder: object) -> None:
    c = Corpus.create(embedder=stub_embedder)  # no path → in-memory
    assert c.store._lock_path is None
