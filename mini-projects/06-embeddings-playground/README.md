# 06 — Embeddings Playground

Twelve short sentences, three clusters (Caesar, general Roman, modern unrelated), embedded with BGE-M3 and stored in pgvector. Three probe queries. Does cosine similarity actually find the right cluster?

## What it demonstrates

- **The end-to-end loop**: embed → store → search. Same pattern Module 4 will scale to the Roman corpus.
- **Why the operator and the index opclass must match.** The HNSW index uses `vector_cosine_ops`, so queries must use `<=>` (cosine distance). Using `<->` (L2) or `<#>` (inner product) silently triggers a sequential scan — fine on 12 rows, ruinous on 50k.
- **Cosine similarity is 1 − cosine distance.** Both are useful: distance for `ORDER BY` (lower = closer matches the index), similarity for human-readable output (0.7 reads better than 0.3).
- **Vector dimension is a hard contract.** The script asserts `embedder.dimension === 1024` before any INSERT. If you accidentally load a different model in LM Studio, this fails loudly instead of producing nonsense `text-embedding-3-small`-shaped INSERTs.
- **Embedding API output order = input order.** `samples[i]` ↔ `vectors[i]` with no bookkeeping needed. The wrapper additionally sorts by `.index` defensively just in case.

## What it solves

A smoke test for the wiring. If "Who killed Julius Caesar?" doesn't surface "Brutus and a group of senators assassinated Caesar" as the top hit, something is wrong with the stack (wrong model, wrong column, wrong operator) — not the technique.

## Run

```
pnpm dev mini-projects/06-embeddings-playground/index.ts
```

Requirements:
- Postgres + pgvector running: `docker compose up -d`
- LM Studio on `localhost:1234` with `text-embedding-bge-m3` loaded

The script is idempotent — re-running it deletes any prior `source='playground'` rows first.

## What to look for

- **All three probes should top-1 MATCH.** BGE-M3 is strong enough that the right cluster wins comfortably even with only 4 sentences per cluster.
- **Similarity scores by cluster.** The top hit's similarity will be much higher than off-cluster hits — typically 0.6–0.7 for in-cluster vs 0.3–0.4 for off-cluster. The *gap* matters more than the absolute number.
- **The Caesar probe** is the most discriminating: "Brutus and a group of senators assassinated Caesar" should beat "Caesar crossed the Rubicon" because the query specifically asks about *killing*. This is semantic search beating keyword search — neither query nor result share many words, but the meaning aligns.
- **Latency.** BGE-M3 on a 5070 Ti should embed all 12 sentences in well under a second. OpenAI would take 300–500ms just for the network round-trip.
- **Cost: $0.00.** Local inference.

## Things to play with

- **Swap the embedder** to OpenAI: change `createEmbedder({ provider: 'lmstudio' })` to `createEmbedder({ provider: 'openai' })`, switch the INSERT/SELECT to the `embedding` (1536-dim) column. Re-run and compare top-1 similarities — usually within a few hundredths.
- **Add an adversarial sample**: `{ text: 'Caesar salad has anchovies and parmesan.', cluster: 'modern' }`. Re-run the Caesar probe. Watch whether BGE-M3 correctly puts this lower than the actual Caesar-assassination samples (it should — but the similarity will be uncomfortably high because "Caesar" appears in both).
- **Try a multilingual query**: `'¿Quién mató a Julio César?'` (Spanish for the same Caesar question). BGE-M3 is trained on 100+ languages, so the cross-language match should still surface the right cluster. text-embedding-3-small is much weaker at this.
- **Add a synonym-style probe**: `'Roman political assemblies'` (the corpus uses "Senate"). This is exactly the **Synonym mismatch** category we'll evaluate in Module 5 — see whether BGE-M3 bridges the vocabulary gap on its own.

## Related code

- [`lib/embeddings.ts`](../../lib/embeddings.ts) — `createEmbedder` factory
- [`db/init/02-schema.sql`](../../db/init/02-schema.sql) — the `chunks` table + HNSW indexes
- [`docker-compose.yml`](../../docker-compose.yml) — Postgres + pgvector container
