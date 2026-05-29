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
from typing import Any

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


def _cmd_ocr_do(args: argparse.Namespace) -> int:
    """Run an agent command across every page flagged `needs_ocr=True`.

    The command may use these placeholders:

    - ``{image}`` — path to a freshly-rendered PNG of the page
    - ``{page}`` — 1-based page number
    - ``{doc_title}`` — document title
    - ``{doc_id}`` — document id

    The command's stdout is taken as the OCR markdown and passed to
    `submit_ocr`. Exit code != 0 (or empty stdout) on a page → that page
    is skipped and logged. Pages are processed serially in v1.
    """
    import shlex
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    corpus = Corpus.open(args.db)
    pending = corpus.list_pending_ocr(
        doc_id=args.doc or None,
        limit=args.limit,
        include_images=False,  # we render to a temp file ourselves
        dpi=args.dpi,
    )
    if not pending:
        print("nothing to do — no pages flagged needs_ocr=True")
        return 0

    print(f"{len(pending)} pages pending across {len({p['doc_id'] for p in pending})} docs")
    if args.dry_run:
        for p in pending:
            print(f"  would process p.{p['page_number']} of {p['doc_title']}  ({p['page_id']})")
        return 0

    if "{image}" not in args.agent_cmd:
        print(
            "ERROR: --agent-cmd must contain the {image} placeholder so the\n"
            "       page render can be passed to your vision agent.",
            file=sys.stderr,
        )
        return 2

    from kglite_docs.ingest.formats import render_page_image
    succeeded = failed = 0
    for p in pending:
        # Render the page to a temp PNG using whatever path was stored on
        # the Document node when it was ingested.
        try:
            doc_path = corpus.cypher(
                "MATCH (d:Document {id: $id}) RETURN d.path AS path",
                params={"id": p["doc_id"]},
            ).to_list()[0]["path"]
            png = render_page_image(doc_path, int(p["page_number"]), dpi=args.dpi)
        except Exception as exc:
            print(f"  ✗ p.{p['page_number']} {p['page_id']}: render failed — {exc}", file=sys.stderr)
            failed += 1
            continue

        with tempfile.NamedTemporaryFile("wb", suffix=".png", delete=False) as fh:
            fh.write(png)
            image_path = Path(fh.name)
        try:
            cmd = args.agent_cmd.format(
                image=str(image_path),
                page=p["page_number"],
                doc_title=p.get("doc_title", ""),
                doc_id=p["doc_id"],
            )
            try:
                proc = subprocess.run(
                    shlex.split(cmd) if args.shell != "yes" else cmd,
                    shell=(args.shell == "yes"),
                    capture_output=True, text=True,
                    timeout=args.timeout, check=False,
                )
            except subprocess.TimeoutExpired:
                print(f"  ✗ p.{p['page_number']} {p['page_id']}: agent timed out (>{args.timeout}s)", file=sys.stderr)
                failed += 1
                continue
            if proc.returncode != 0:
                stderr_preview = (proc.stderr or "")[:160].replace("\n", " ")
                print(f"  ✗ p.{p['page_number']} {p['page_id']}: agent exited {proc.returncode} — {stderr_preview}", file=sys.stderr)
                failed += 1
                continue
            md = (proc.stdout or "").strip()
            if not md:
                print(f"  ✗ p.{p['page_number']} {p['page_id']}: empty agent output", file=sys.stderr)
                failed += 1
                continue
            corpus.submit_ocr(
                p["page_id"], md,
                agent_id=args.agent_id, model=args.model,
            )
            print(f"  ✓ p.{p['page_number']} of {p.get('doc_title','')}  ({len(md)} chars)")
            succeeded += 1
        finally:
            image_path.unlink(missing_ok=True)

    corpus.save(args.db)
    total = succeeded + failed
    print(f"\nfinished: {succeeded}/{total} pages OCR'd, {failed} failures")
    return 0 if failed == 0 else 1


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
    d: Any
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

    pdo = sp.add_parser(
        "ocr-do",
        help="Run an agent command across every page flagged needs_ocr=True",
        description=(
            "Iterate over pages that need OCR. For each, render the page to a "
            "PNG, run the supplied agent command (must include the {image} "
            "placeholder), and submit the command's stdout back as the page's "
            "markdown. Pages are processed serially."
        ),
    )
    pdo.add_argument("--db", required=True)
    pdo.add_argument(
        "--agent-cmd", required=True,
        help='Command template — must contain {image}. Other placeholders: '
             '{page}, {doc_title}, {doc_id}. Example: '
             '\'claude -p --bare --image {image} "Transcribe to markdown"\'',
    )
    pdo.add_argument("--agent-id", default="cli-ocr-agent",
                     help="Agent id recorded on each submission (default: cli-ocr-agent)")
    pdo.add_argument("--model", default="",
                     help="Model name to record on each page (informational)")
    pdo.add_argument("--doc", default="", help="Scope to one document id")
    pdo.add_argument("--limit", type=int, default=100,
                     help="Max pages to process this run (default: 100)")
    pdo.add_argument("--dpi", type=int, default=200,
                     help="DPI for the page render handed to the agent")
    pdo.add_argument("--timeout", type=int, default=180,
                     help="Per-page agent timeout in seconds (default: 180)")
    pdo.add_argument("--shell", choices=["no", "yes"], default="no",
                     help='If "yes", run the command through a shell (allows '
                          'pipes/quoting). Default splits with shlex.')
    pdo.add_argument("--dry-run", action="store_true",
                     help="List what would be processed; don't invoke the agent")
    pdo.set_defaults(func=_cmd_ocr_do)

    psh = sp.add_parser("show", help="Show a document or chunk by id")
    psh.add_argument("kind", choices=["doc", "chunk"])
    psh.add_argument("id")
    psh.add_argument("--db", required=True)
    psh.set_defaults(func=_cmd_show)

    args = p.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
