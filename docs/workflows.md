# Agent workflows

Patterns for multi-step agent work against a kglite-docs corpus. All examples assume an MCP-connected agent (or a Python script calling the `Corpus` API directly).

## Research a topic

```
agent: search("transformer attention", top_k=8, agent_id="alice")
       → 8 chunks across N documents

agent: for each interesting chunk:
         get_chunk(chunk_id, with_neighbors=True, agent_id="alice")
         tag_chunk(chunk_id, "attention", agent_id="alice")
         add_summary(chunk_id, text="...", agent_id="alice")

agent: compose_context("transformer attention", max_tokens=3000)
       → ranked, budgeted, ready for the model
```

The `agent_id` threading means a later query like
`MATCH (c:Chunk)<-[:VIEWED]-(a:Agent {id:'alice'}) RETURN c.doc_id, count(c)`
shows where alice's research focus actually went.

## Compare two papers

```
agent: list_documents(filters={"title": "DPR"})  → doc_a
agent: list_documents(filters={"title": "ColBERT"})  → doc_b

agent: get_document(doc_a)  # see the TOC
agent: get_document(doc_b)

agent: search("token-level vectors", filters={"doc_id": doc_a}, top_k=5, agent_id="me")
agent: search("token-level vectors", filters={"doc_id": doc_b}, top_k=5, agent_id="me")

agent: compose_context("how do they differ", max_tokens=4000)
       → mixed context across both docs

agent: add_summary(comparison_chunk_id, "...", target_kind="Document",
                   depth="document", agent_id="me")
```

## Cross-checked synthesis (one agent writes, another verifies)

```
WRITER:
agent_a: cluster_chunks()                  # if not yet clustered
agent_a: cluster_overview()                # find the topic of interest
agent_a: get_cluster(cluster_id)
agent_a: compose_context(prompt_about_cluster_topic, max_tokens=6000)
agent_a: <generate article with back-refs>  # off-graph LLM call
agent_a: add_summary(cluster_id, article_text, target_kind="Cluster",
                     agent_id="writer", model="opus-4.7")

VERIFIER (different agent):
agent_b: get_summaries(cluster_id)         # pulls writer's article
agent_b: for each claim in article:
           verify_claim(claim, against_chunk_ids=[ids cited in claim])
agent_b: verify_summary(summary_id, verdict="verified",
                        verifier_agent_id="verifier",
                        notes="checked 12 claims, 11 supported, 1 needs revision")
```

`verify_summary` rejects self-verification server-side. Status moves to `verified` / `disputed` / `needs_revision`.

## OCR for scanned PDFs

```
USER ingests a scanned PDF.

agent: list_pending_ocr(limit=5)
       → each entry includes a base64 PNG render of the page

agent: <reads the image, transcribes to markdown>
agent: submit_ocr(page_id, transcribed_markdown,
                  agent_id="vision-ocr", model="claude-vision")

→ Pipeline re-chunks the markdown, re-embeds, marks the page ready.
```

## Translation workflow

```
TRANSLATOR:
agent_a: list_documents()
agent_a: for each chunk in target document:
           add_translation(chunk_id, target_lang="no", text="...",
                           agent_id="translator-a", model="opus-4.7",
                           status="draft")

REVIEWER (can be same or different agent):
agent_b: get_translations(chunk_id, target_lang="no")
agent_b: for each draft translation:
           # if accurate:
           mark_translation_reviewed(translation_id, reviewer_agent_id="reviewer-b")
           # if not, write a new translation:
           add_translation(chunk_id, "no", improved_text, agent_id="reviewer-b")

CONSUMER:
agent_c: assemble_translated_document(doc_id, target_lang="no")
agent_c: export_document(doc_id, "translated.docx")
```

## Hallucination guarding

Before a summary leaves the agent's "draft" state:

```
agent: report = check_grounding(summary_id, threshold=0.55)
agent: if report["supported_fraction"] < 0.7:
         # too many sentences with weak source alignment
         <revise the summary>
agent: for weak in report["weak_sentences"]:
         <re-search for support or remove the claim>
```

The threshold (default 0.5) is a cosine-similarity cutoff between each
sentence in the summary and its best-matching source chunk. It's a
*proxy*, not a truth oracle — surface the weak claims for human or
second-agent review.

## Power-user: write your own Cypher

The typed tools cover the 80% case. When you need something specific,
`cypher_query` is registered as a tool by `mcp-methods`:

```cypher
// Find chunks alice tagged but never had a summary written about
MATCH (c:Chunk)-[:TAGGED_AS]->(tg:Tagging {by_agent: 'alice'})
WHERE NOT (c)<-[:SUMMARIZES]-(:Summary)
RETURN c.id, c.doc_id, c.page_number
```

```cypher
// Cluster with the most cross-document chunks
MATCH (c:Chunk)-[:IN_CLUSTER]->(cl:Cluster)
WITH cl, count(DISTINCT c.doc_id) AS doc_count, count(c) AS chunk_count
ORDER BY doc_count DESC LIMIT 5
RETURN cl.id, doc_count, chunk_count
```

`graph_overview()` (also registered by `mcp-methods`) prints the schema with property samples — the right first call when writing custom Cypher.
