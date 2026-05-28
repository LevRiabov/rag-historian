-- Runs ONCE on first container start (postgres entrypoint convention).
-- Enables pgvector inside the rag_historian database created by POSTGRES_DB.
-- Re-run safely: IF NOT EXISTS makes it idempotent if you ever apply it by hand.
CREATE EXTENSION IF NOT EXISTS vector;
