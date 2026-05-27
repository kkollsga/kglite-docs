"""FastMCP server wiring.

Registers the typed kglite-docs tool surface, plus the bundled
`cypher_query` + `graph_overview` helpers from `mcp_methods.fastmcp`.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("kglite_docs.mcp_server")


def build_app(corpus: Any) -> Any:
    """Construct and return a FastMCP app wired to `corpus`."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:  # pragma: no cover - extras gate
        raise RuntimeError(
            "kglite-docs MCP server requires `pip install 'kglite-docs[mcp]'`"
        ) from e

    app = FastMCP("kglite-docs")

    # Register typed tools
    from kglite_docs.mcp_server.tools import register_typed_tools
    register_typed_tools(app, corpus)

    # Register the kglite-style escape hatches from mcp_methods
    try:
        from mcp_methods.fastmcp import register_cypher_query, register_overview
        register_cypher_query(app, corpus.store.g)
        register_overview(app, corpus.store.g, overview_prefix=(
            "kglite-docs knowledge base. Use the typed tools "
            "(`search`, `get_chunk`, etc.) first; reach for `cypher_query` "
            "only when the typed surface doesn't cover what you need."
        ))
    except Exception as exc:
        log.warning("could not register mcp-methods helpers: %s", exc)

    return app
