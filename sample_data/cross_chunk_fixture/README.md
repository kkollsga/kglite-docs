# Fixture: cross-chunk / emergent-pattern detection (`TR-7788`)

A small, **fully synthetic and anonymized** regression fixture for the gap described in
`inbox/.../cross-chunk-pattern-detection.md` and `.../out-of-the-box-synthesis-options.md`:
**in legal documents the errors are never spelled out** — they're emergent from the
*relationship* between sections, so a per-chunk scorer is structurally blind to them.

## The case (synthetic)
A tribunal: §2 declares the **Respondent** in DEFAULT for her *medically-justified*
absence; §3 DISMISSES the action because the **Claimant** *unjustifiably* failed to
appear; §4 then CONDEMNS the Respondent on the merits (contradicting §3); §5 is the
Respondent's appeal brief that *names* the unfairness — but it's advocacy, not a ruling.

Read in isolation, **every ruling (§2/§3/§4) is routine and lawful** → scores neutral.
The defects exist only across sections:
- **F1 disparate treatment** — same trigger (non-appearance), harsher outcome for the
  party with the *better* excuse. (links §2 + §3)
- **F2 conflicting dispositions** — dismissal (§3) vs merits condemnation (§4) can't
  both govern. (links §3 + §4)
- **F3 provenance discipline** — §5 explicitly states the defect, but it's a party
  characterization; the *proof* is in the primary rulings (F1/F2), not §5. A system
  must not pass by simply trusting §5, nor miss F1/F2 because §5 was disputed.

## Files
- `CASE_TR-7788.md` — the synthetic case text.
- `expected_findings.json` — gold: anchors (by marker phrase), the per-chunk baseline
  expectation (recovers **zero** gold findings), the gold synthesis findings, pass
  criteria, and negative controls (§1 petition, §6 clerical note must never appear).
- `harness.py` — ingests the case, resolves anchors, runs the baseline (shows the miss),
  and exposes `check(candidate_findings, anchors)` to assert your synthesis output.

## How to use while developing
```bash
python harness.py        # shows: baseline recovers [], study_conflicts()=0, then the gold target
```
```python
from harness import build, check
corpus, study_id, anchors = build()
findings = my_synthesis_pass(corpus, study_id)   # <-- your Option-1/2 implementation
#   each finding: {"type","linked_chunk_ids":[...],"provenance","weight"}
check(findings, anchors)                          # PASS only if F1 & F2 recovered
```

## Pass bar
- `check()` PASSES iff your synthesis layer emits **F1** and **F2** as `primary_text`
  findings, each linking its two anchor rulings, `weight >= 0.7`.
- §5 must be tagged `characterization` (F3); §1 and §6 must never surface.
- Bonus (the "out-of-the-box" bar): `conclude_study` should refuse / warn while F1/F2
  are unsurfaced — i.e. a study that concludes on per-chunk neutrals alone is a FAIL.

The harness's built-in demo feeds the gold-as-findings into `check()` and prints PASS,
proving the assertions are wired correctly; your job is to make a *real* synthesis pass
produce that same output from the corpus.

## Variants added (recall + precision)
- **§7 omission (F4)** — a timely jurisdiction motion the court **never ruled on**. The
  defect is the *absence* of a ruling, invisible to contradiction/disparity lenses; only
  an **omission lens** (which models what *should* be present) catches it. If no omission
  lens runs, the system must at least report it as a coverage blind spot — never silently
  pass §7 as clean.
- **§8 negative control (NC1)** — the claimant got an extension the respondent never
  requested: different outcome, **same rule applied evenly** = a lawful distinction, NOT
  bias. A naive disparity detector flags it; a correct system (or a panel under leveled
  review) keeps it `contested` and does **not** promote it. Promoting NC1 is a precision
  FAIL. `harness.py` Demo A (correct) PASSES; Demo B (flags §8) FAILS on NC1 — so the
  fixture guards both recall and precision.
