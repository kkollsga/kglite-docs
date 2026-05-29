---
name: cross-checked-review
description: "The write/verify/ground pattern that makes agent-generated summaries trustworthy. One agent writes, a different agent verifies; summary('ground') catches hallucinations before they ship."
references_tools:
  - summary
  - review
  - agent
---

# Cross-checked review

The headline kglite-docs pattern. **Two agents (or roles)**: one writes summaries, a *different* one verifies. Self-verification is rejected server-side. Add `summary("ground", ...)` for objective signal alongside the human-in-the-loop verdict.

## The minimum viable pattern

```python
# Writer side
sid = summary(
    "add",
    target_id="doc_3e67...#p2#c3",
    text="DPR uses a dual BERT encoder with in-batch negatives.",
    agent_id="writer-1",
    model="opus-4.7",
)

# Self-check before declaring done (catches obvious hallucinations)
summary("ground", id=sid, threshold=0.5)
# → {"supported_fraction": 1.0, "grounding_score": 0.78,
#    "weak_sentences": []}
# If weak_sentences is non-empty, those are likely fabrications.

# Reviewer side (DIFFERENT agent_id — server enforces)
summary(
    "verify", id=sid,
    verdict="verified",  # or "disputed", "needs_revision"
    verifier_agent_id="reviewer-1",
    notes="checked against p.2",
)
```

The Summary node now carries a `Verified` label. `MATCH (s:Summary:Verified) RETURN s` returns all confirmed ones across the graph.

## When to scale up to the kanban

If you've got >10 summaries and want a queue (especially for human review):

```python
# Enqueue all chunks of a doc for review
review("enqueue_chunks", doc_id="doc_3e67...", priority=5)

# Reviewer agent loop
while True:
    ticket = review("claim_next", agent_id="reviewer-1")
    if ticket is None:
        break
    chunk = ticket["target"]
    # ... agent reads chunk["text"], decides verdict ...
    review(
        "complete",
        ticket_id=ticket["ticket_id"], agent_id="reviewer-1",
        verdict="reviewed",       # or needs_revision / rejected
        accuracy=0.92,             # optional confidence in [0,1]
        authenticity="verified",   # optional free-text or enum
        notes="cross-checked citations",
        tags=["accurate", "core-claim"],   # tags applied to the chunk
    )
```

Tickets move through `New → InReview → Reviewed` (or `NeedsRevision` / `Rejected`). Status lives as a label on the ticket and is queryable: `MATCH (t:ReviewTicket:InReview) RETURN t`.

## Anti-hallucination tools

- **`summary("ground", id=...)`** — per-sentence similarity of summary text against the source chunk(s) it summarises. `supported_fraction < 0.7` is a yellow flag; `weak_sentences` are the specific claims that don't map back. Cheap (cosine, not NLI) — surfaces obvious hallucinations.
- **`summary("claim", text=..., against_chunk_ids=[...])`** — for an arbitrary claim, finds the chunks that best support it. Use when reviewing a summary's *individual claims*, not the whole summary. Useful in the verifier loop.

These are *signals*, not truth oracles. Use them to surface candidates for human/agent review; don't let an agent auto-merge based purely on a high grounding score.

## Per-agent templates

If you have a recurring reviewer role with a consistent system prompt, define it once:

```python
agent(
    "upsert", id="reviewer-strict",
    role="reviewer", kind="llm", model="claude-sonnet-4-6",
    system_prompt=(
        "You are a strict fact-checker. For each claim, verify against "
        "the provided source chunks. Anything not directly supported → "
        "PARTIALLY_SUPPORTED or UNSUPPORTED."
    ),
    tools=["summary", "chunk"],
    context={"strictness": "high"},
)

# Later — load the template, use it to launch the LLM call
cfg = agent("get", id="reviewer-strict")
# anthropic.messages.create(model=cfg["model"], system=cfg["system_prompt"], ...)
```

All subsequent graph writes under `agent_id="reviewer-strict"` attribute back to the same Agent node. `agent("activity", id="reviewer-strict")` then shows everything that agent has done.

## What NOT to do

- **Don't self-verify.** The server rejects `summary("verify", ...)` where author == verifier. By design.
- **Don't skip `summary("ground", ...)`** on summaries you generated yourself. It catches the dumb cases (numbers you fabricated, attributions to wrong papers, etc.).
- **Don't write 50 summaries before any verification.** The whole point is the second-pass safety net. Stagger writes and verifies so the verifier catches a pattern early if you're systematically off.
