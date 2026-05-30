"""Bundled domain schema packs (data, not engine logic).

A schema pack registers a controlled element vocabulary through the generic
`schema.register_element_discriminator` seam, so the core engine stays
domain-opaque — it never names a legal/medical term. Packs are opt-in: load the
one(s) you need.

    from kglite_docs.schemas import load_schema
    load_schema("legal")          # registers the legal element vocabulary

The MCP server loads configured packs at startup; library users load explicitly.
"""

from __future__ import annotations

import importlib

#: Schema packs shipped with kglite-docs (each a `kglite_docs.schemas.<name>` module).
_BUNDLED: frozenset[str] = frozenset({"legal"})


def available_schemas() -> frozenset[str]:
    """Names of the bundled schema packs."""
    return _BUNDLED


def load_schema(name: str) -> None:
    """Activate a bundled schema pack by name (registers its element vocabulary).
    Idempotent — importing a pack twice re-runs nothing. Raises for an unknown
    pack name."""
    if name not in _BUNDLED:
        raise ValueError(
            f"unknown schema pack {name!r}; available: {sorted(_BUNDLED)}"
        )
    importlib.import_module(f"kglite_docs.schemas.{name}")
