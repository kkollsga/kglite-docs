"""Source-party dimension — *whose words* a document holds.

`provenance` records what was checked (primary text vs paraphrase); it can't say
*who authored it*. But an admission against interest — primary text by the
**adverse** party — is among the most probative evidence there is, and nothing
captured it. This adds a generic **source-party** tag on a Document, inherited to
its chunks, so an analyst can ask "primary-text supporting evidence authored by
the opposing party" and have an admission surface.

Generic + domain-opaque: the party is a free-text token labelled via
`label_for("doc.source_party", …)` (PascalCase, so core works with no pack);
the legal vocabulary (claimant / respondent / defense / accuser / judge / court /
expert / third_party) is *registered as data* for discovery, never named in core.
Document-level for now (one source per exhibit — the 80% case); a per-chunk
override is a future refinement.
"""

from __future__ import annotations

from typing import Any

from kglite_docs.errors import InvalidEnumError
from kglite_docs.schema import CHUNK, DOCUMENT, label_for
from kglite_docs.store import Store
from kglite_docs.store import rows as _df_dicts

#: value → description. A discovery registry only; labeling never requires it.
_PARTIES: dict[str, str] = {}


def register_source_party(value: str, *, description: str = "") -> None:
    """Register a source-party value for discovery (`available_source_parties`).
    Idempotent; raises on a conflicting redefinition."""
    existing = _PARTIES.get(value)
    if existing is not None and existing != description:
        raise ValueError(f"source party {value!r} already registered differently")
    _PARTIES[value] = description


def available_source_parties() -> list[dict[str, str]]:
    """Registered source parties (value + canonical label + description). Empty
    until a schema pack registers them; labeling works for any value regardless."""
    return [{"value": v, "label": label_for("doc.source_party", v), "description": d}
            for v, d in sorted(_PARTIES.items())]


def party_label(value: str) -> str:
    """Canonical secondary-label for a party value (`defense` → `Defense`,
    `third_party` → `ThirdParty`). `""` for an empty value."""
    return label_for("doc.source_party", value)


def set_source_party(store: Store, *, doc_id: str, party: str) -> dict[str, Any]:
    """Tag a Document with its source party and **inherit to its chunks** (a
    `source_party` property + a secondary label on the Document and every chunk),
    so `MATCH (c:Chunk:Defense)` and ledger surfacing work. Re-tagging swaps the
    prior party label cleanly."""
    party = (party or "").strip()
    if not party:
        raise InvalidEnumError("party must be a non-empty string")
    rows = _df_dicts(store.cypher(
        "MATCH (d:Document {id: $id}) RETURN d.source_party AS old", params={"id": doc_id},
    ))
    if not rows:
        raise InvalidEnumError(f"document not found: {doc_id}")
    old = (rows[0].get("old") or "").strip()
    new_label = party_label(party)
    chunk_ids = [r["id"] for r in _df_dicts(store.cypher(
        "MATCH (:Document {id: $id})-[:HAS_CHUNK]->(c:Chunk) RETURN c.id AS id",
        params={"id": doc_id},
    ))]
    # Set the property on the document + its chunks.
    store.cypher("MATCH (d:Document {id: $id}) SET d.source_party = $p",
                 params={"id": doc_id, "p": party})
    store.cypher("MATCH (:Document {id: $id})-[:HAS_CHUNK]->(c:Chunk) SET c.source_party = $p",
                 params={"id": doc_id, "p": party})
    # Swap labels: drop the prior party label (if changing), add the new one.
    old_label = party_label(old) if old else ""
    if old_label and old_label != new_label:
        store.remove_label(DOCUMENT, [doc_id], old_label)
        if chunk_ids:
            store.remove_label(CHUNK, chunk_ids, old_label)
    if new_label:
        store.add_label(DOCUMENT, [doc_id], new_label)
        if chunk_ids:
            store.add_label(CHUNK, chunk_ids, new_label)
    return {"doc_id": doc_id, "source_party": party, "chunks_tagged": len(chunk_ids)}
