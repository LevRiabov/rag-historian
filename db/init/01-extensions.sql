-- Runs ONCE on first container start (postgres entrypoint convention).
-- Enables pgvector inside the rag_historian database created by POSTGRES_DB.
-- Re-run safely: IF NOT EXISTS makes it idempotent if you ever apply it by hand.
CREATE EXTENSION IF NOT EXISTS vector;

-- pg_search: BM25 full-text search with proper IDF term weighting (Module 6.2).
-- Provided by the ParadeDB image. Needed because Postgres core FTS (ts_rank)
-- has no IDF, so it can't down-weight ubiquitous terms in a single-topic corpus.
CREATE EXTENSION IF NOT EXISTS pg_search;
