"""The analytical-lens registry — the open seam of the leveled-review system.

A *lens* is a named reviewer strategy: a prompt that tells an agent what class of
pattern to hunt, plus the unit type it scans (`chunk` for a detectability sweep
that may surface new findings, `finding` for a re-grading panel). Leveled review
escalates *detectability* by running a lens that hasn't run yet (R3) — and the
payoff (R4/§6 of the spec) is that an *un-run* lens becomes a **named, listed
blind spot** instead of an unknown gap.

Like the element registry in `schema.py`, this is a generic seam: core ships it
**empty** and never names a lens; domain packs (e.g. `schemas/legal.py`) register
their lenses at import. Designing it as an open registry is what keeps the
control system useful as new failure modes get named over time.
"""

from __future__ import annotations

from typing import Any

#: name → {prompt, unit_type, description}. Mutable; populated by domain packs.
_LENSES: dict[str, dict[str, Any]] = {}

_VALID_UNIT_TYPES = frozenset({"chunk", "finding"})


def register_lens(
    name: str, *, prompt: str, unit_type: str = "chunk", description: str = "",
) -> None:
    """Register an analytical lens. Idempotent for an identical re-registration;
    raises on a conflicting redefinition. `unit_type` is what the lens scans:
    `chunk` (a detectability sweep that may emit new findings) or `finding`
    (a re-grading / panel pass)."""
    if unit_type not in _VALID_UNIT_TYPES:
        raise ValueError(
            f"invalid unit_type {unit_type!r} (expected one of {sorted(_VALID_UNIT_TYPES)})"
        )
    spec = {"prompt": prompt, "unit_type": unit_type, "description": description}
    existing = _LENSES.get(name)
    if existing is not None and existing != spec:
        raise ValueError(f"lens {name!r} already registered differently")
    _LENSES[name] = spec


def is_registered_lens(name: str) -> bool:
    return name in _LENSES


def available_lenses() -> tuple[str, ...]:
    """Names of all registered lenses (sorted)."""
    return tuple(sorted(_LENSES))


def lens_info(name: str) -> dict[str, Any]:
    """Full spec for a lens. Raises `KeyError` for an unknown lens."""
    return dict(_LENSES[name])


def lens_prompt(name: str) -> str:
    """The reviewer prompt for a lens. Raises `KeyError` for an unknown lens."""
    return str(_LENSES[name]["prompt"])


def lens_unit_type(name: str) -> str:
    """`chunk` or `finding` — what this lens scans. Raises for an unknown lens."""
    return str(_LENSES[name]["unit_type"])
