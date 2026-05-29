-- Module 4 schema for the Roman Research Agent.
--
-- Two tables: `sources` (one row per source document) and `chunks` (one row
-- per text chunk, referencing a source). Replaces the standalone Module 3
-- chunks table — the source-tier metadata (primary vs historian) and
-- chunk-position metadata (chapter, char offsets, prev/next via chunk_index)
-- are the things future modules will lean on heavily. Adding them later
-- means a partial re-ingest, so we bake them in now.
--
-- Init scripts only run on a FRESH Docker volume. To apply this after a
-- prior `docker compose up` against an older schema:
--     docker compose down -v && docker compose up -d
-- (This wipes data. We only have Module 3 playground data, which is
-- deterministic from `mini-projects/06-embeddings-playground/index.ts` —
-- regeneratable in seconds. No production data to lose.)
--
-- Design notes (educational, per repo convention):
--
-- * `sources` carries the author/tier/year metadata that source-aware
--   prompts will reference ("Plutarch wrote 150 years after the events;
--   Caesar wrote during them — treat them differently"). Splitting it
--   from `chunks` avoids repeating author/title on every chunk row and
--   makes future source-filter queries cheap.
--
-- * `tier` is plain TEXT instead of an enum or CHECK constraint. We expect
--   to start with 'primary' and 'historian' but want freedom to add
--   'commentary' / 'translation-note' later without a migration. Document
--   intent in the README; let the prompt layer enforce semantics.
--
-- * Two embedding columns (1536d OpenAI + 1024d BGE-M3), both nullable.
--   Same rationale as Module 3: BGE-M3 for cheap iteration, OpenAI for
--   one comparison run. The chunks table is the source of truth for which
--   backends are populated — `WHERE embedding_bge IS NOT NULL` filters
--   to the ones BGE-M3 has been run against.
--
-- * `chunking_version` discriminator. Module 6.1 will add structure-aware
--   chunking; we want both the naive baseline and the new chunks coexisting
--   in the same table, distinguished by a string label ('naive-v1',
--   'structure-v1', etc.). Comparison runs filter on this column. Without
--   it, we'd need a parallel table per chunking strategy or be stuck
--   overwriting the baseline every time we tried a new one.
--
-- * `chunk_index` is per-(source, chunking_version) — gives us a stable
--   ordering for the source-viewer feature ("show chunk + 2 before + 2
--   after"). UNIQUE constraint on (source_id, chunking_version, chunk_index)
--   catches double-ingest bugs at INSERT time.
--
-- * `char_start` / `char_end` are byte offsets into the original cleaned
--   text. Stored separately because we'll want to render the source-viewer
--   page from the original (with chunk highlighted), not from the chunk
--   text itself (which is post-tokenization, may have lost whitespace).
--
-- * HNSW indexes with `vector_cosine_ops` — same as Module 3. Query with
--   the `<=>` operator to use the index; other operators silently fall
--   back to a sequential scan.
--
-- * `chunks_source_idx` (btree on source_id, chunking_version, chunk_index)
--   accelerates two common access patterns: "show all chunks of this
--   source" and "show neighbors of chunk N within this source/version".

CREATE TABLE IF NOT EXISTS sources (
  id            SERIAL PRIMARY KEY,
  slug          TEXT UNIQUE NOT NULL,           -- 'caesar-gallic-war' — stable, URL-safe identifier
  title         TEXT NOT NULL,                  -- 'The Gallic War'
  author        TEXT NOT NULL,                  -- 'Julius Caesar'
  tier          TEXT NOT NULL,                  -- 'primary' | 'historian' (see note above)
  year_written  INTEGER,                        -- approximate; negative for BC (e.g. -50)
  translator    TEXT,                           -- for the English edition we ingested
  language      TEXT NOT NULL DEFAULT 'en',     -- of stored text, not of original composition
  source_url    TEXT,                           -- Project Gutenberg URL
  metadata      JSONB
);

CREATE TABLE IF NOT EXISTS chunks (
  id                SERIAL PRIMARY KEY,
  source_id         INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
  chunking_version  TEXT NOT NULL DEFAULT 'naive-v1',
  chunk_index       INTEGER NOT NULL,           -- 0-based, sequence within (source, version)
  chapter           TEXT,                       -- 'Book I, Chapter 7' or similar — for citations
  text              TEXT NOT NULL,
  char_start        INTEGER,                    -- offset into the cleaned source text
  char_end          INTEGER,
  embedding         VECTOR(1536),               -- OpenAI text-embedding-3-small
  embedding_bge     VECTOR(1024),               -- BAAI BGE-M3 via LM Studio
  metadata          JSONB,                      -- {token_count, ...}
  UNIQUE (source_id, chunking_version, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
  ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_embedding_bge_hnsw
  ON chunks USING hnsw (embedding_bge vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_source_idx
  ON chunks (source_id, chunking_version, chunk_index);
