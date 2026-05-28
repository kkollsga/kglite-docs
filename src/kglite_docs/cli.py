"""kglite-docs CLI — ingest, search, list, cluster from the shell.

Examples::

    kglite-docs ingest paper.pdf --db kb.kgl
    kglite-docs ingest ./pdfs/ --db kb.kgl --recursive
    kglite-docs search "transformer attention" --db kb.kgl
    kglite-docs list --db kb.kgl
    kglite-docs cluster --db kb.kgl --algorithm louvain
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kglite_docs import Corpus


def _open_or_create(db_path: str | Path) -> Corpus:
    p = Path(db_path)
    return Corpus.open(p) if p.exists() else Corpus.create(p)


def _cmd_ingest(args: argparse.Namespace) -> int:
    corpus = _open_or_create(args.db)
    target = Path(args.target)
    if target.is_dir():
        results = corpus.ingest_dir(target, recursive=args.recursive)
        print(json.dumps({
            "ingested": sum(1 for r in results if r.created),
            "skipped": sum(1 for r in results if not r.created),
            "total_chunks": sum(r.chunk_count for r in results),
            "ocr_pending": sum(r.ocr_pending_pages for r in results),
        }, indent=2))
    else:
        r = corpus.ingest(target)
        print(json.dumps({
            "doc_id": r.doc_id, "created": r.created,
            "pages": r.page_count, "chunks": r.chunk_count,
            "ocr_pending": r.ocr_pending_pages,
        }, indent=2))
    corpus.save(args.db)
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    corpus = Corpus.open(args.db)
    hits = corpus.search(args.query, top_k=args.top_k, agent_id=args.agent or None)
    for h in hits:
        text = (h.get("text") or "")[: args.snippet]
        print(f"[{h.get('score', 0):.3f}] {h['id']}  p.{h.get('page')}  {text}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    corpus = Corpus.open(args.db)
    docs = corpus.list_documents(limit=args.limit)
    for d in docs:
        print(f"{d.get('id')}  {d.get('title')}  pages={d.get('pages')}  chunks={d.get('chunk_count')}")
    return 0


def _cmd_cluster(args: argparse.Namespace) -> int:
    corpus = _open_or_create(args.db)
    r = corpus.cluster_chunks(algorithm=args.algorithm)
    print(json.dumps(r, indent=2))
    corpus.save(args.db)
    return 0


def _cmd_ocr_status(args: argparse.Namespace) -> int:
    corpus = Corpus.open(args.db)
    status = corpus.ocr_status(doc_id=args.doc or None)
    print(
        f"{status['pending_pages']}/{status['total_pages']} pages pending OCR "
        f"({status['documents_with_pending']}/{status['documents_total']} docs)"
    )
    if args.verbose:
        for d in status["documents"]:
            marker = "!" if d["pending"] else " "
            print(
                f"  {marker} {d['pending']:>3}/{d['pages']:<3} {d['format']:<5} "
                f"{d['title']}  ({d['doc_id'][:18]}…)"
            )
    return 0 if status["pending_pages"] == 0 else 1


def _cmd_show(args: argparse.Namespace) -> int:
    corpus = Corpus.open(args.db)
    if args.kind == "doc":
        d = corpus.get_document(args.id)
    elif args.kind == "chunk":
        d = corpus.get_chunk(args.id, with_neighbors=True, with_summaries=True)
    else:
        print(f"unknown kind: {args.kind}", file=sys.stderr)
        return 2
    print(json.dumps(d, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kglite-docs")
    sp = p.add_subparsers(dest="cmd", required=True)

    pi = sp.add_parser("ingest", help="Ingest a document or directory (PDF/DOCX/PPTX/MD/HTML/TXT/images)")
    pi.add_argument("target", help="Path to a file or a directory")
    pi.add_argument("--db", required=True)
    pi.add_argument("--recursive", action="store_true")
    pi.set_defaults(func=_cmd_ingest)

    ps = sp.add_parser("search", help="Semantic search")
    ps.add_argument("query")
    ps.add_argument("--db", required=True)
    ps.add_argument("--top-k", type=int, default=10)
    ps.add_argument("--snippet", type=int, default=180)
    ps.add_argument("--agent", default="")
    ps.set_defaults(func=_cmd_search)

    pl = sp.add_parser("list", help="List documents")
    pl.add_argument("--db", required=True)
    pl.add_argument("--limit", type=int, default=100)
    pl.set_defaults(func=_cmd_list)

    pc = sp.add_parser("cluster", help="Run clustering")
    pc.add_argument("--db", required=True)
    pc.add_argument("--algorithm", default="louvain")
    pc.set_defaults(func=_cmd_cluster)

    po = sp.add_parser("ocr-status", help="OCR coverage summary across the corpus")
    po.add_argument("--db", required=True)
    po.add_argument("--doc", default="", help="Scope to one document id")
    po.add_argument("-v", "--verbose", action="store_true", help="Per-document detail")
    po.set_defaults(func=_cmd_ocr_status)

    psh = sp.add_parser("show", help="Show a document or chunk by id")
    psh.add_argument("kind", choices=["doc", "chunk"])
    psh.add_argument("id")
    psh.add_argument("--db", required=True)
    psh.set_defaults(func=_cmd_show)

    args = p.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
