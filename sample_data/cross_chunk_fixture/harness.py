#!/usr/bin/env python3
"""
Cross-chunk / emergent-pattern regression harness for kglite-docs.

What it does:
  1. Ingests the synthetic anonymized case (CASE_TR-7788.md) into a fresh corpus.
  2. Locates the four anchor chunks by marker phrase (chunker-agnostic).
  3. Runs the BASELINE: a per-chunk "find judicial impropriety" pass that mirrors
     what the documented study workflow does today, and shows it recovers ZERO of
     the gold synthesis findings (each ruling is routine in isolation).
  4. Exposes `check(candidate_findings)` — drop your synthesis-pass output in and
     it asserts against expected_findings.json (PASS/FAIL per gold finding).

The point: step 3 should MISS; your synthesis layer should make `check(...)` PASS.

Usage:
    python harness.py                      # build + show the baseline miss + gold target
    # then, in your dev code:
    from harness import build, check
    corpus, study_id, anchors = build()
    my_findings = my_synthesis_pass(corpus, study_id)   # <-- implement this
    check(my_findings, anchors)
"""
import json, os, re, tempfile
from kglite_docs import Corpus

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.join(HERE, "CASE_TR-7788.md")
GOLD = json.load(open(os.path.join(HERE, "expected_findings.json")))

def build(db_path=None):
    """Ingest the fixture; return (corpus, study_id, anchors={name: chunk_id}).
    Each section is ingested as its own unit so chunk boundaries are deterministic
    (this also mirrors a real case file = many separate documents/exhibits)."""
    db_path = db_path or os.path.join(tempfile.mkdtemp(), "tr7788.kgl")
    c = Corpus.create(db_path)
    raw = open(CASE).read()
    # split into sections on the "## §" headers; ingest each as a separate doc
    parts = re.split(r"(?m)^## ", raw)
    for p in parts[1:]:
        title = p.splitlines()[0].strip()
        c.ingest(text="## " + p, title=title, format="md", structure_aware=True)
    chunks = c.cypher("MATCH (n:Chunk) RETURN n.id AS id, n.text AS t", {}).to_list()
    anchors = {}
    for name, a in GOLD["anchors"].items():
        m = a["marker"]
        hit = next((r["id"] for r in chunks if m.lower() in (r["t"] or "").lower()), None)
        anchors[name] = hit
        if hit is None:
            print(f"  !! anchor {name!r} marker not found: {m!r}")
    sid = c.define_study(question=GOLD["per_chunk_baseline_expectation"]["question"],
                         created_by="harness", title="TR-7788 impropriety study")
    return c, sid, anchors

def baseline_per_chunk(corpus, study_id, anchors):
    """Mirror the documented per-chunk flow: score each ruling chunk in isolation.
    Honest stand-in for an LLM scorer: each ruling, read ALONE, is a routine lawful
    order -> neutral. (No single chunk states a defect.) Returns recovered gold ids."""
    # In a real run an agent scores each chunk; in isolation §2/§3/§4 are 'neutral'.
    # We encode that expected behaviour here so the harness is self-contained.
    for name in ("default_ruling", "dismissal_ruling", "merits_condemnation"):
        cid = anchors[name]
        if cid:
            corpus.assess(study_id, cid, stance="neutral", weight=0.0, agent_id="baseline",
                          rationale="Read in isolation, this is a routine lawful order.",
                          provenance="primary_text")
    # §5 is the only non-neutral hit, correctly down-graded to characterization:
    if anchors["appeal_assertion"]:
        corpus.assess(study_id, anchors["appeal_assertion"], stance="supports", weight=0.4,
                      agent_id="baseline", provenance="characterization",
                      rationale="Party brief ASSERTS unequal treatment; advocacy, not a ruling.")
    # which gold findings did per-chunk recover? -> none (F1/F2 are cross-chunk)
    return []

def check(candidate_findings, anchors):
    """Assert a synthesis output against the gold. candidate_findings: list of
    {type, linked_chunk_ids:[...], provenance, weight, escalation_state?}.
    Validates F1/F2 (cross-chunk patterns), F4 (omission, lens-gated), and the
    NC1 negative control (a lawful distinction that must NOT be promoted)."""
    name_of = {v: k for k, v in anchors.items() if v}
    ok = True
    def links_of(f):
        return {name_of.get(cid) for cid in f.get("linked_chunk_ids", [])}
    for g in GOLD["gold_synthesis_findings"]:
        gid = g["id"]
        if gid == "F3":
            continue  # provenance-handling, advisory (see README)
        if gid == "NC1":  # NEGATIVE CONTROL — must NOT be promoted as bias/disparity
            promoted = [f for f in candidate_findings
                        if "nc_extension" in links_of(f)
                        and f.get("type") in ("disparate_treatment", "bias", "asymmetric_rule")
                        and float(f.get("weight", 0)) >= 0.5
                        and f.get("escalation_state") != "contested"]
            passed = not promoted
            print(f"  [{'PASS' if passed else 'FAIL'}] NC1 (negative control) — §8 must NOT be promoted as bias"
                  + ("" if passed else "  <-- FALSE POSITIVE"))
            ok = ok and passed
            continue
        if gid == "F4":  # omission — recoverable only by an omission lens
            matched = next((f for f in candidate_findings
                            if f.get("type") == "omission"
                            and "omission_target" in links_of(f)
                            and float(f.get("weight", 0)) >= g["min_weight"]), None)
            # acceptable alternative: explicitly reported as an un-run-lens blind spot
            print(f"  [{'PASS' if matched else 'FAIL'}] F4 (omission) — §7 undecided motion"
                  + ("" if matched else "  (or report 'omission' lens NOT run as a blind spot)"))
            ok = ok and matched is not None
            continue
        want = set(g["must_link_anchors"])
        matched = None
        for f in candidate_findings:
            if f.get("type") == g["type"] and want.issubset(links_of(f)) \
               and f.get("provenance") == g["required_provenance"] \
               and float(f.get("weight", 0)) >= g["min_weight"]:
                matched = f; break
        print(f"  [{'PASS' if matched else 'FAIL'}] {gid} ({g['type']}) — links {sorted(want)}")
        ok = ok and matched is not None
    print("RESULT:", "PASS ✅ synthesis recovered the emergent findings (and avoided the false positive)" if ok
          else "FAIL ❌ — see above (missed pattern, missed omission, or promoted the negative control)")
    return ok

if __name__ == "__main__":
    print("Building fixture corpus…")
    c, sid, anchors = build()
    print("Anchors resolved:")
    for k, v in anchors.items():
        print(f"  {k:20} -> {v}")
    print("\nBASELINE (per-chunk only, as documented today):")
    recovered = baseline_per_chunk(c, sid, anchors)
    print(f"  gold findings recovered by per-chunk pass: {recovered}  (expected: [] — the miss)")
    print(f"  study_conflicts(): {c.study_conflicts(sid).get('total', '?')} (expected 0 — blind to cross-chunk)")
    print("\nGOLD the synthesis layer must recover:")
    for g in GOLD["gold_synthesis_findings"]:
        link = g.get("must_link_anchors") or [g.get("anchor")]
        print(f"  {g['id']} {g['type']:24} links={link}")
    print("\nDemo A — a CORRECT synthesis output (recovers F1/F2/F4, leaves NC1 alone) -> PASS:")
    good = [
        {"type": "disparate_treatment", "linked_chunk_ids": [anchors["default_ruling"], anchors["dismissal_ruling"]], "provenance": "primary_text", "weight": 0.9},
        {"type": "conflicting_dispositions", "linked_chunk_ids": [anchors["dismissal_ruling"], anchors["merits_condemnation"]], "provenance": "primary_text", "weight": 0.8},
        {"type": "omission", "linked_chunk_ids": [anchors["omission_target"]], "provenance": "primary_text", "weight": 0.7},
    ]
    check(good, anchors)
    print("\nDemo B — a NAIVE output that also flags §8 as bias (false positive) -> must FAIL on NC1:")
    naive = good + [
        {"type": "disparate_treatment", "linked_chunk_ids": [anchors["nc_extension"]], "provenance": "primary_text", "weight": 0.6},
    ]
    check(naive, anchors)
    c.close()
