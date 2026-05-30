"""The synthesis pass — the second altitude above per-chunk `assess`.

Per-chunk scoring is excellent for atomic, citable evidence but blind to
*emergent* patterns: a defect that exists only in the JOIN of several chunks
(disparate treatment, two contradictory operative rulings, an argument the
record never answers). Each chunk reads lawful in isolation, so the happy path
can mark a study "done" while a whole class of finding is unreachable.

kglite-docs is agent-first: the library can't reason, so it provides (1) the
**prompt** that tells an agent what to hunt, and (2) the **gate** that refuses to
let a study conclude until the pass has run (see `study.conclude_study`). This
module is the prompt half — domain-neutral core text plus an extensible seam for
domain packs (e.g. the legal pack) to append their own hunt list. Core never
names a domain term.
"""

from __future__ import annotations

_CORE_PROMPT = """\
SYNTHESIS PASS — read the whole study ledger (every assessment + its rationale)
together and hunt the patterns per-chunk scoring cannot see. No single chunk
declares these; each is emergent across the record. For every pattern you find,
record a cross-chunk Finding (`study("finding", supporting_chunk_ids=[…])`)
citing the chunks it rests on, with a stance/weight/provenance.

Hunt for:
- Same-actor inconsistencies — the same party/author taking contradictory
  positions across the record.
- Disparate treatment — like situations resolved differently for different
  subjects (same trigger → different outcome by who it applies to).
- Contradiction — two statements/decisions that cannot both stand.
- Two-operative-outcomes — a later resolution that overrides or conflicts with
  an earlier one without acknowledging it.
- Omission — a claim, argument, or question raised in the record that is never
  answered or addressed.
- Aggregation — a total/trend that only appears by summing or ordering values
  scattered across many chunks.

A clean record yields no findings — that is a valid result, not a failure. Do
not invent patterns; cite primary text or mark the provenance honestly."""

#: Domain packs append their hunt list here (order = registration order).
_ADDENDA: list[str] = []


def register_synthesis_addendum(text: str) -> None:
    """Append a domain-specific synthesis hunt list to the core prompt. Idempotent
    for identical text (a pack importing twice adds nothing). Used by data packs
    (e.g. `schemas/legal.py`) so the agent's synthesis prompt grows domain cues
    without core naming any domain term."""
    text = (text or "").strip()
    if text and text not in _ADDENDA:
        _ADDENDA.append(text)


def synthesis_prompt() -> str:
    """The full synthesis prompt: the domain-neutral core plus any registered
    domain addenda. An agent reads this before running the synthesis pass."""
    if not _ADDENDA:
        return _CORE_PROMPT
    return _CORE_PROMPT + "\n\n" + "\n\n".join(_ADDENDA)
