-- Module 3 schema: a single `chunks` table for the Roman corpus, with TWO
-- embedding columns so we can A/B compare OpenAI vs BGE-M3 on the same data.
--
-- Design notes (educational, per repo convention):
--
-- * Two embedding columns, both nullable:
--     embedding      VECTOR(1536)  -- OpenAI text-embedding-3-small
--     embedding_bge  VECTOR(1024)  -- BAAI BGE-M3 (via LM Studio)
--
--   pgvector columns are dimension-typed — a 1024-dim vector CANNOT live in
--   a VECTOR(1536) column. Locking down to one column would force a costly
--   re-ingest later just to try a different backend, which is exactly the
--   Module 3 pitfall. Two columns + nullability means we can backfill each
--   backend independently (e.g., embed everything with OpenAI first, then
--   add BGE-M3 only for the rows we want to compare).
--
--   In production you'd pick ONE and drop the other. For learning, side-by-
--   side is the right call.
--
-- * HNSW + opclass per backend: each embedding column gets its own HNSW
--   index. `vector_cosine_ops` tells the index to optimize for the `<=>`
--   (cosine distance) operator — the standard for text embeddings. Other
--   choices: vector_l2_ops (`<->`, Euclidean), vector_ip_ops (`<#>`, inner
--   product). You MUST query with the operator that matches the index's
--   opclass, or the planner falls back to a sequential scan.
--
-- * metadata JSONB: keeps chunk-level fields (chapter title, position in
--   document, char offsets, etc.) flexible without schema churn during
--   Module 4 ingest experiments. GIN-index it later if we start filtering
--   on metadata at query time.
--
-- * IF NOT EXISTS makes every statement idempotent — safe to re-run by hand
--   even though the postgres entrypoint only auto-runs init scripts once.

CREATE TABLE IF NOT EXISTS chunks (
  id            SERIAL PRIMARY KEY,
  text          TEXT NOT NULL,
  embedding     VECTOR(1536),   -- OpenAI text-embedding-3-small
  embedding_bge VECTOR(1024),   -- BAAI BGE-M3 via LM Studio
  source        TEXT,
  metadata      JSONB
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
  ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_embedding_bge_hnsw
  ON chunks USING hnsw (embedding_bge vector_cosine_ops);
