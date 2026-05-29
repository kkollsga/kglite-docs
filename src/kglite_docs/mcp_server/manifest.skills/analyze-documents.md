---
name: analyze-documents
description: "Given N documents + a task (summarise / answer / extract), here's the canonical pipeline — document(ingest) → search(compose) → summary(add) → optional verify/ground."
references_tools:
  - document
  - search
  - summary
---

# Analyze N documents

Use when the task is: "given these documents, do X" — summarise key findings, answer a question, extract specific facts, build a synthesis. Works for any supported format (PDF / DOCX / PPTX / MD / HTML / TXT / images).

## The pipeline

```
document("ingest", ...)              ── load files
        │
        ▼
document("list")                      ── verify what's in the graph
        │
        ▼
search(query, mode="compose")         ── budgeted, ranked context bundle
        │
        ▼
<your LLM call>                       ── pass ctx["items"] as the prompt body
        │
        ▼
summary("add", ...)                   ── persist the result back into the graph
        │
        ▼ (optional but recommended)
summary("ground", id=...)             ── score whether claims map to source
        │
        ▼ (recommended for high-trust outputs)
summary("verify", id=..., verdict=, verifier_agent_id=)
                                       ── second agent applies a verdict
```

## Concrete example

```python
# 1. Ingest
document("ingest", path="/abs/path/paper1.pdf")
document("ingest", path="/abs/path/paper2.pdf")
# Or: document("ingest", directory="/abs/path/papers/", recursive=True)

# 2. List what we've got
document("list")
# → [{"id": "doc_3e67...", "title": "paper1", ...}, ...]

# 3. Build a budgeted context for your specific question
ctx = search(
    "how do these papers approach dense retrieval?",
    mode="compose",
    max_tokens=3000,
    agent_id="analyzer-1",
)
# Returns {query, budget_tokens, used_tokens, items: [chunks + verified summaries]}.
# Drop ctx["items"] into your LLM prompt body.

# 4. ... your LLM call here, off-graph ...
# Generated text: "Both papers use dual-encoder architectures but differ in ..."

# 5. Persist back as a summary anchored to the most relevant chunk
sid = summary(
    "add",
    target_id=ctx["items"][0]["chunk_id"],
    text="<your LLM output>",
    agent_id="analyzer-1",
    model="sonnet-4.6",
)

# 6. Self-check the grounding before declaring it done
summary("ground", id=sid, threshold=0.5)
# → {"supported_fraction": 0.83, "weak_sentences": [...]}
# weak_sentences are likely hallucinations or unsupported claims.

# 7. If a second agent is available, have them verify
summary(
    "verify", id=sid,
    verdict="verified",   # or "disputed", "needs_revision"
    verifier_agent_id="reviewer-1",
    notes="checked p.5 against the source",
)
```

## When to scope to one document vs all

- **Cross-doc questions** (synthesis, comparison) — let `search(..., mode="compose")` pull from the whole corpus. Pass `per_doc_cap=3` if you want balanced representation.
- **Per-doc questions** (summarise *this* paper) — pass `filters={"doc_id": "..."}` to `search` in `hits` mode, or scope before composing.
- **Two-doc comparison specifically** — use `document("compare", doc_a, doc_b, queries=[...])` — it's purpose-built and returns side-by-side hits.

## Common mistakes

- **Rolling your own top-k → concat instead of `search(..., mode="compose")`**. The compose mode respects the token budget, inlines verified summaries, and ranks consistently with the same model the embeddings were built under. Don't reinvent it.
- **Writing summaries without a stable `agent_id`** — if you use a different id each call, your work doesn't attribute back to a single `Agent` node and `agent("activity", id=...)` shows fragments.
- **Skipping `summary("ground", ...)`** on summaries you wrote yourself. It's a sanity check before claiming "I've analyzed the documents." Takes <1s.
- **Self-verifying** — `summary("verify", ...)` rejects calls where the verifier and author are the same agent. By design. Get a second agent.
