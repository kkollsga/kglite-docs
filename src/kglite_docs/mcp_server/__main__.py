"""Console entry: ``python -m kglite_docs.mcp_server`` or
``kglite-docs-mcp``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kglite-docs-mcp")
    parser.add_argument("--db", required=True, help="Path to the .kgl knowledge base")
    parser.add_argument("--ingest", help="Optional directory to ingest before serving")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirs when ingesting")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level)

    from kglite_docs import Corpus
    from kglite_docs.mcp_server.server import build_app

    db = Path(args.db)
    corpus = Corpus.open(db) if db.exists() else Corpus.create(db)
    if args.ingest:
        corpus.ingest_dir(args.ingest, recursive=args.recursive)
        corpus.save(db)

    app = build_app(corpus)
    app.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
