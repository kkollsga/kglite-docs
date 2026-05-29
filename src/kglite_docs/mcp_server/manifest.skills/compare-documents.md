---
name: compare-documents
description: "Given two documents, the side-by-side retrieval pattern. document('compare', ...) gives you per-query hits from each doc plus a merged context bundle."
references_tools:
  - document
  - search
  - summary
  - tag
---

# Compare two documents

The natural ask when the user hands you two PDFs and says "compare these." Don't roll a custom pipeline — `document("compare", ...)` is purpose-built.

## The shape

```python
document("ingest", path="/abs/a.pdf")
document("ingest", path="/abs/b.pdf")

document("list")
# → grab the doc_ids for both

document(
    "compare",
    doc_a="doc_3e67...",         # full doc_id of the first document
    doc_b="doc_23e3...",         # full doc_id of the second
    queries=[
        "retrieval architecture",
        "training objective",
        "evaluation metrics",
        "computational cost",
    ],
    top_k_per_query=5,
    max_tokens_per_query=2000,
)
```

Returns:

```python
{
    "doc_a_id": "doc_3e67...", "doc_a_title": "DPR",
    "doc_b_id": "doc_23e3...", "doc_b_title": "ColBERT",
    "queries": [
        {
            "query": "retrieval architecture",
            "doc_a_hits": [{"id": ..., "score": 0.81, "text": ..., "page": 2}, ...],
            "doc_b_hits": [{"id": ..., "score": 0.79, "text": ..., "page": 3}, ...],
            "merged_context": {            # budgeted, ranked, both-docs
                "used_tokens": 1850,
                "items": [chunks from both, interleaved by score]
            }
        },
        ...
    ]
}
```

## How to use the result

For each query, you have two parallel sets of hits and one merged context.

- **For a *symmetric* comparison** ("how do they differ on retrieval architecture?") — feed `merged_context.items` to your LLM with a "compare and contrast" prompt.
- **For *per-doc* observations** ("what does paper A say about retrieval?") — feed `doc_a_hits` only.
- **For *which doc handles X better*** — both `doc_a_hits` and `doc_b_hits` carry scores; if doc A's top score is much higher than doc B's for the same query, doc A is more relevant on that axis.

## Choosing the queries list

The queries shape the comparison. Good practice:

- **3-7 queries**, each a *concept or claim*, not a literal question.
- Cover the axes the user actually cares about. For papers: architecture, training, evaluation, limitations, cost. For policy docs: scope, exemptions, enforcement, timeline. For reports: methodology, findings, recommendations.
- If the user gave you a specific axis ("how do their approaches to noise handling differ"), one query is enough.

## Don't roll your own

Two failure modes if you skip `document("compare", ...)`:

1. **Sequential per-doc searches without budget control.** You pull top-5 from doc A, top-5 from doc B, concat. No token budget, no rank harmonisation across docs, easy to drown your prompt.
2. **Search across both docs without filtering, then post-filter in your head.** You miss chunks from the less-relevant doc that are still worth showing for contrast.

`document("compare", ...)` does both correctly: per-doc filtered search to guarantee representation, then merged context with a shared budget.

## After the comparison

If you want the result to be queryable later, persist it:

```python
summary(
    "add",
    target_id=doc_a_id, target_kind="Document",
    text="<your comparison writeup>",
    depth="document",
    agent_id="comparison-agent",
)
# Tag both docs for the comparison axis
tag("add", chunk_id=doc_a_chunk_id, name="compared-with-colbert",
    agent_id="comparison-agent")
```

Then a future query like "show me documents already compared with X" works via `tag("chunks", name="compared-with-colbert")`.
