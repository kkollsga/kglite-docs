"""End-to-end workflow demo:

1. Ingest all PDFs in ``sample_data/pdfs``
2. Cluster the chunks (louvain → falls back to k-means on embeddings)
3. Identify the most-connected cluster
4. For each member chunk, ask a Sonnet agent for a 1-line summary + 3 tags
5. Write the summaries + tags back to the graph
6. Ask Sonnet to compose an article about the cluster, with chunk-id
   back-references
7. Ask a different Sonnet agent to fact-check the article against the
   underlying chunks (verification edges + grounding scores)
8. Persist everything to disk + print a summary

LLM calls go through ``kglite_docs.agents.call_agent`` which picks
the best available backend (Anthropic SDK if ``ANTHROPIC_API_KEY`` is
set, else the ``claude -p`` CLI). No code changes needed to switch.

Usage::

    python demos/workflow.py --db demo.kgl --pdfs sample_data/pdfs --max-chunks 30
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from kglite_docs import Corpus
from kglite_docs.agents import call_agent, default_caller

log = logging.getLogger("kglite_docs.workflow")


CLUSTER_AGENT_SYSTEM = (
    "You are a research analyst. Read the supplied document chunks and produce "
    "concise structured analysis. Be terse and factual. Output ONLY valid JSON."
)

ARTICLE_AGENT_SYSTEM = (
    "You are a technical writer. Given a set of source-chunk summaries with ids, "
    "produce a coherent ~600-word article on the unifying topic. Every factual "
    "claim must end with a back-reference of the form `[chunk_id]`. Use 3-5 "
    "section headings. Do not invent facts not present in the source summaries."
)

FACTCHECK_AGENT_SYSTEM = (
    "You are a fact-checker. You will be shown a draft article and the source "
    "chunk summaries it references. For each claim in the article, decide if "
    "it is SUPPORTED by the source summaries, PARTIALLY_SUPPORTED, or UNSUPPORTED. "
    "Output JSON: {claims: [{claim, citations: [chunk_ids], verdict, note}], "
    "overall_verdict: verified|disputed|needs_revision}."
)


def _ingest_all(corpus: Corpus, pdf_dir: Path) -> dict[str, int]:
    """Ingest every PDF in `pdf_dir`. Returns timing + counts."""
    t0 = time.monotonic()
    results = corpus.ingest_dir(pdf_dir, recursive=False)
    elapsed = time.monotonic() - t0
    return {
        "ingested": sum(1 for r in results if r.created),
        "skipped": sum(1 for r in results if not r.created),
        "total_chunks": sum(r.chunk_count for r in results),
        "elapsed_seconds": round(elapsed, 1),
    }


def _pick_cluster(corpus: Corpus) -> dict | None:
    from kglite_docs.cluster import most_connected_cluster
    return most_connected_cluster(corpus.store)


def _strip_json(text: str) -> str:
    """Extract a JSON object/array from a model response that may be
    wrapped in fences or have surrounding chatter."""
    fence = re.search(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    # Otherwise grab the first balanced { ... } or [ ... ]
    for opener, closer in [("{", "}"), ("[", "]")]:
        first = text.find(opener)
        last = text.rfind(closer)
        if first != -1 and last != -1 and last > first:
            return text[first : last + 1]
    return text


def _summarise_chunk_batch(
    corpus: Corpus,
    chunks: list[dict],
    *,
    agent_id: str,
    model: str = "sonnet",
) -> None:
    """Ask one Sonnet call to handle a batch of chunks. Writes a Summary
    + tags for each chunk via the typed Corpus API."""
    payload = "\n\n".join(
        f"[{c['id']}]\nheadings: {c.get('headings')}\n"
        f"text: {(c.get('text') or '')[:1500]}"
        for c in chunks
    )
    prompt = (
        "For each chunk below, return one JSON object per line (JSONL) with keys "
        "`chunk_id`, `summary` (one sentence), and `tags` (list of 1-3 short "
        "lowercase topical tags). Use the exact chunk_id provided. Output "
        "ONLY JSONL, no commentary.\n\n"
        f"{payload}"
    )
    raw = call_agent(prompt, system=CLUSTER_AGENT_SYSTEM, model=model)
    # JSONL parse — tolerate fenced blocks and stray text
    raw = _strip_json(raw) if raw.strip().startswith(("{", "[")) else raw
    for line in raw.splitlines():
        line = line.strip().lstrip(",")
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        cid = obj.get("chunk_id")
        summary = obj.get("summary") or ""
        tags = obj.get("tags") or []
        if not cid or not summary:
            continue
        try:
            corpus.add_summary(cid, summary, agent_id=agent_id, model=model)
        except Exception as exc:
            log.warning("add_summary failed for %s: %s", cid, exc)
        for tag in tags[:3]:
            try:
                corpus.tag_chunk(cid, str(tag), kind="topic", agent_id=agent_id)
            except Exception as exc:
                log.warning("tag_chunk failed for %s tag=%r: %s", cid, tag, exc)


def analyse_cluster(
    corpus: Corpus,
    cluster_id: str,
    *,
    agent_id: str = "claude-sonnet-analyst",
    model: str = "sonnet",
    batch_size: int = 5,
    max_parallel: int = 4,
) -> dict:
    """Spin up Sonnet calls to summarise + tag every chunk in a cluster."""
    cl = corpus.get_cluster(cluster_id)
    if cl is None:
        raise ValueError(f"cluster not found: {cluster_id}")
    members = cl["members"]
    batches = [members[i : i + batch_size] for i in range(0, len(members), batch_size)]
    t0 = time.monotonic()
    log.info("analyse_cluster: %d chunks → %d batches × Sonnet", len(members), len(batches))
    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = [
            pool.submit(_summarise_chunk_batch, corpus, batch, agent_id=agent_id, model=model)
            for batch in batches
        ]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as exc:
                log.warning("batch failed: %s", exc)
    elapsed = time.monotonic() - t0
    return {
        "cluster_id": cluster_id, "members_analysed": len(members),
        "batches": len(batches), "elapsed_seconds": round(elapsed, 1),
    }


def write_article(
    corpus: Corpus,
    cluster_id: str,
    *,
    agent_id: str = "claude-sonnet-writer",
    model: str = "sonnet",
) -> dict:
    """Draft an article from the cluster's summaries with back-refs."""
    cl = corpus.get_cluster(cluster_id)
    members = cl["members"] if cl else []
    summaries: list[dict] = []
    for m in members:
        for s in corpus.get_summaries(m["id"]):
            summaries.append({"chunk_id": m["id"], "summary": s["text"]})
    if not summaries:
        return {"article": "", "summaries_used": 0}
    payload = "\n".join(f"[{s['chunk_id']}] {s['summary']}" for s in summaries[:50])
    prompt = (
        "Source chunk summaries (one per line, with chunk_id in brackets):\n\n"
        f"{payload}\n\n"
        "Write a coherent ~600-word article on the unifying topic. Every factual "
        "claim must end with a back-reference of the form `[chunk_id]`. Use 3-5 "
        "section headings. Output ONLY the article — no preamble."
    )
    article = call_agent(prompt, system=ARTICLE_AGENT_SYSTEM, model=model)
    # Persist into the corpus as a document so it's queryable
    title = f"Synthesis of cluster {cluster_id}"
    r = corpus.ingest_text(article, title=title, source_uri=f"synthesis:{cluster_id}", format="md")
    return {
        "article": article,
        "summaries_used": len(summaries),
        "doc_id": r.doc_id,
    }


def fact_check_article(
    corpus: Corpus,
    article_text: str,
    cluster_id: str,
    *,
    verifier_agent_id: str = "claude-sonnet-factcheck",
    model: str = "sonnet",
) -> dict:
    """Run a second Sonnet pass to fact-check claims against the source
    summaries. Persists verification verdicts on each cited summary."""
    cl = corpus.get_cluster(cluster_id)
    members = cl["members"] if cl else []
    summaries: list[dict] = []
    for m in members:
        for s in corpus.get_summaries(m["id"]):
            summaries.append({"chunk_id": m["id"], "summary_id": s["id"], "text": s["text"]})
    source_block = "\n".join(f"[{s['chunk_id']}] {s['text']}" for s in summaries[:60])
    prompt = (
        f"ARTICLE TO CHECK:\n{article_text}\n\n"
        f"SOURCE SUMMARIES:\n{source_block}\n\n"
        "Return JSON only."
    )
    raw = call_agent(prompt, system=FACTCHECK_AGENT_SYSTEM, model=model)
    raw = _strip_json(raw)
    try:
        report = json.loads(raw)
    except json.JSONDecodeError:
        report = {"raw": raw[:1000], "parse_error": True}

    # Persist verification verdicts on each cited summary
    summary_by_chunk: dict[str, list[str]] = {}
    for s in summaries:
        summary_by_chunk.setdefault(s["chunk_id"], []).append(s["summary_id"])
    verdict_map = {
        "SUPPORTED": "verified",
        "PARTIALLY_SUPPORTED": "needs_revision",
        "UNSUPPORTED": "disputed",
    }
    persisted = 0
    for claim in report.get("claims", []) if isinstance(report, dict) else []:
        verdict = verdict_map.get((claim.get("verdict") or "").upper(), None)
        if not verdict:
            continue
        for cid in claim.get("citations", []) or []:
            for sid in summary_by_chunk.get(cid, []):
                try:
                    corpus.verify_summary(
                        sid, verdict=verdict,
                        verifier_agent_id=verifier_agent_id,
                        notes=str(claim.get("note", ""))[:500],
                    )
                    persisted += 1
                except Exception as exc:
                    log.debug("verify_summary failed for %s: %s", sid, exc)
    return {"report": report, "verifications_persisted": persisted}


def run(
    pdf_dir: Path,
    db_path: Path,
    *,
    max_chunks_per_cluster: int = 60,
    model: str = "sonnet",
) -> dict:
    out_dir = db_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Verify a caller is available before doing any expensive ingest
    default_caller()

    corpus = Corpus.open(db_path) if db_path.exists() else Corpus.create(db_path)

    ingest_stats = _ingest_all(corpus, pdf_dir)
    log.info("ingest done: %s", ingest_stats)
    corpus.save()

    cluster_run = corpus.cluster_chunks(algorithm="kmeans", params={"k": 8})
    log.info("cluster done: %s", cluster_run)
    corpus.save()

    target = _pick_cluster(corpus)
    if not target:
        return {"error": "no clusters", "ingest": ingest_stats}
    log.info("most-connected cluster: %s", target)

    analyse_stats = analyse_cluster(
        corpus, target["id"], model=model,
    )
    corpus.save()

    article = write_article(corpus, target["id"], model=model)
    corpus.save()
    (out_dir / "synthesis_article.md").write_text(article["article"], encoding="utf-8")

    fact_check = fact_check_article(
        corpus, article["article"], target["id"], model=model,
    )
    corpus.save()
    (out_dir / "factcheck_report.json").write_text(
        json.dumps(fact_check["report"], indent=2), encoding="utf-8",
    )

    return {
        "ingest": ingest_stats,
        "cluster": cluster_run,
        "target_cluster": target,
        "analyse": analyse_stats,
        "article": {
            "saved_to": str(out_dir / "synthesis_article.md"),
            "doc_id": article.get("doc_id"),
            "summaries_used": article.get("summaries_used"),
        },
        "fact_check": {
            "saved_to": str(out_dir / "factcheck_report.json"),
            "verifications_persisted": fact_check.get("verifications_persisted"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="demo.kgl")
    p.add_argument("--pdfs", default="sample_data/pdfs")
    p.add_argument("--model", default="sonnet")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(message)s")
    out = run(Path(args.pdfs), Path(args.db), model=args.model)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
