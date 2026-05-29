/**
 * roman-research/ingest/index.ts — full ingest pipeline orchestrator.
 *
 * Per source: download → parse → chunk → embed (both backends if OpenAI
 * key is set) → upsert source row → replace prior chunks for this
 * (source, version) → bulk insert. Idempotent: re-running produces the
 * same DB state.
 *
 * Run:
 *   pnpm dev roman-research/ingest/index.ts
 *
 * Requirements:
 *   - Postgres + pgvector:  docker compose up -d
 *   - LM Studio on :1234 with text-embedding-bge-m3 loaded
 *   - OPENAI_API_KEY in .env (OPTIONAL — OpenAI column stays NULL if absent)
 */
import 'dotenv/config';
import pg from 'pg';
import pgvector from 'pgvector/pg';

import {
  addCost,
  type Cost,
  createEmbedder,
  type Embedder,
  formatCost,
} from '../../lib/index.ts';
import { type Chunk, chunkSections, DEFAULT_CHUNK_OPTIONS, totalTokens } from './chunk.ts';
import { downloadAll, readRawFile } from './download.ts';
import { parseSource } from './parse.ts';
import { gutenbergTextUrl, SOURCES, type SourceManifest } from './sources.ts';

const CHUNKING_VERSION = 'naive-v1';

const KB = (bytes: number) => `${Math.round(bytes / 1024)} KB`;
const ZERO_COST: Cost = {
  inputUSD: 0,
  outputUSD: 0,
  cacheCreationUSD: 0,
  cacheReadUSD: 0,
  totalUSD: 0,
};

// ---------------------------------------------------------------------------
// DB + embedder setup
// ---------------------------------------------------------------------------
const databaseUrl = process.env.DATABASE_URL;
if (!databaseUrl) {
  throw new Error('DATABASE_URL is not set. Check .env and `docker compose up -d`.');
}
const db = new pg.Client({ connectionString: databaseUrl });
await db.connect();
await pgvector.registerType(db);

const bgeEmbedder: Embedder = createEmbedder({ provider: 'lmstudio' });
const openaiEmbedder: Embedder | null = process.env.OPENAI_API_KEY
  ? createEmbedder({ provider: 'openai' })
  : null;

console.log(`=== Roman Research Agent — Ingest (Turn B: chunk + embed + store) ===\n`);
console.log(`Chunking version:  ${CHUNKING_VERSION}`);
console.log(`Target tokens:     ${DEFAULT_CHUNK_OPTIONS.targetTokens}`);
console.log(`Overlap tokens:    ${DEFAULT_CHUNK_OPTIONS.overlapTokens}`);
console.log(`BGE-M3 embedder:   ${bgeEmbedder.label}`);
console.log(
  `OpenAI embedder:   ${openaiEmbedder?.label ?? 'SKIPPED (set OPENAI_API_KEY to enable)'}`,
);
console.log();

// ---------------------------------------------------------------------------
// Download all (deduped by Gutenberg ID — Caesar's works share a file)
// ---------------------------------------------------------------------------
const downloads = await downloadAll(SOURCES);
console.log(`Downloaded ${downloads.size} unique Gutenberg files.\n`);

// ---------------------------------------------------------------------------
// Per-source pipeline
// ---------------------------------------------------------------------------
let totalChunks = 0;
let totalCharsCleaned = 0;
let totalBgeCost: Cost = ZERO_COST;
let totalOpenaiCost: Cost = ZERO_COST;
let totalBgeLatencyMs = 0;
let totalOpenaiLatencyMs = 0;

for (const source of SOURCES) {
  console.log(`--- ${source.slug} ---`);

  // Parse
  const raw = await readRawFile(source.gutenbergId);
  const parsed = parseSource(source.slug, raw);
  console.log(
    `  Parsed:    ${parsed.sections.length} sections, ${KB(parsed.cleanedText.length)} cleaned`,
  );
  totalCharsCleaned += parsed.cleanedText.length;

  // Chunk
  const chunks = chunkSections(parsed.sections);
  const chunkTokens = totalTokens(chunks);
  const avgTokens = chunks.length > 0 ? Math.round(chunkTokens / chunks.length) : 0;
  const maxTokens = Math.max(0, ...chunks.map((c) => c.tokenCount));
  console.log(
    `  Chunked:   ${chunks.length} chunks (avg ${avgTokens} tok, max ${maxTokens} tok, total ${chunkTokens.toLocaleString()} tok)`,
  );

  // Embed (in parallel — different network paths)
  const texts = chunks.map((c) => c.text);
  console.log(`  Embedding both backends in parallel...`);
  const [bgeResult, openaiResult] = await Promise.all([
    bgeEmbedder.embed(texts),
    openaiEmbedder ? openaiEmbedder.embed(texts) : Promise.resolve(null),
  ]);
  console.log(
    `    BGE-M3:  ${bgeResult.vectors.length} vectors in ${bgeResult.latencyMs}ms (${bgeEmbedder.dimension}d)`,
  );
  totalBgeCost = addCost(totalBgeCost, bgeResult.cost);
  totalBgeLatencyMs += bgeResult.latencyMs;
  if (openaiResult) {
    console.log(
      `    OpenAI:  ${openaiResult.vectors.length} vectors in ${openaiResult.latencyMs}ms (${openaiEmbedder?.dimension}d, ${formatCost(openaiResult.cost)})`,
    );
    totalOpenaiCost = addCost(totalOpenaiCost, openaiResult.cost);
    totalOpenaiLatencyMs += openaiResult.latencyMs;
  }

  // Upsert source → get id. ON CONFLICT keeps re-runs cheap and updates
  // any metadata changes (e.g., when we tweak the manifest).
  const sourceId = await upsertSource(source);
  console.log(`  Source:    id=${sourceId}`);

  // Replace prior chunks for this (source, version). Different versions
  // (e.g., 'structure-v1' in Module 6.1) coexist peacefully.
  const deleted = await db.query(
    `DELETE FROM chunks WHERE source_id = $1 AND chunking_version = $2`,
    [sourceId, CHUNKING_VERSION],
  );
  if ((deleted.rowCount ?? 0) > 0) {
    console.log(`  Cleared:   ${deleted.rowCount} prior chunks`);
  }

  // Bulk insert. Postgres parameter limit (~65k) easily accommodates a
  // single INSERT for our 250-ish chunks per source × 10 columns each.
  const tInsert = Date.now();
  await bulkInsertChunks(sourceId, chunks, bgeResult.vectors, openaiResult?.vectors ?? null);
  console.log(`  Inserted:  ${chunks.length} chunks in ${Date.now() - tInsert}ms`);

  totalChunks += chunks.length;
  console.log();
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------
console.log(`=== Totals ===`);
console.log(`Sources:       ${SOURCES.length}`);
console.log(`Cleaned text:  ${KB(totalCharsCleaned)} (${totalCharsCleaned.toLocaleString()} chars)`);
console.log(`Chunks:        ${totalChunks}`);
console.log(
  `BGE-M3:        ${totalBgeLatencyMs}ms wall-clock, ${formatCost(totalBgeCost)} (local = free)`,
);
console.log(
  `OpenAI:        ${openaiEmbedder ? `${totalOpenaiLatencyMs}ms wall-clock, ${formatCost(totalOpenaiCost)}` : 'SKIPPED'}`,
);

// DB sanity check: count chunks per source via SQL.
console.log(`\n=== DB verification ===`);
const verification = await db.query<{
  slug: string;
  author: string;
  chunks: string;
  with_bge: string;
  with_openai: string;
}>(`
  SELECT
    s.slug,
    s.author,
    COUNT(c.id) AS chunks,
    COUNT(c.embedding_bge) AS with_bge,
    COUNT(c.embedding) AS with_openai
  FROM sources s
  LEFT JOIN chunks c ON c.source_id = s.id AND c.chunking_version = $1
  GROUP BY s.id, s.slug, s.author
  ORDER BY s.id
`, [CHUNKING_VERSION]);
for (const row of verification.rows) {
  console.log(
    `  ${row.slug.padEnd(22)} chunks=${row.chunks.padStart(4)}  bge=${row.with_bge.padStart(4)}  openai=${row.with_openai.padStart(4)}`,
  );
}

await db.end();

// ===========================================================================
// Helpers
// ===========================================================================

async function upsertSource(source: SourceManifest): Promise<number> {
  const result = await db.query<{ id: number }>(
    `INSERT INTO sources (slug, title, author, tier, year_written, translator, language, source_url, metadata)
     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
     ON CONFLICT (slug) DO UPDATE SET
       title = EXCLUDED.title,
       author = EXCLUDED.author,
       tier = EXCLUDED.tier,
       year_written = EXCLUDED.year_written,
       translator = EXCLUDED.translator,
       language = EXCLUDED.language,
       source_url = EXCLUDED.source_url,
       metadata = EXCLUDED.metadata
     RETURNING id`,
    [
      source.slug,
      source.title,
      source.author,
      source.tier,
      source.yearWritten,
      source.translator,
      source.language,
      gutenbergTextUrl(source.gutenbergId),
      source.metadata ?? {},
    ],
  );
  const row = result.rows[0];
  if (!row) throw new Error(`Failed to upsert source ${source.slug}`);
  return row.id;
}

async function bulkInsertChunks(
  sourceId: number,
  chunks: Chunk[],
  bgeVectors: number[][],
  openaiVectors: number[][] | null,
): Promise<void> {
  if (chunks.length === 0) return;
  // 10 columns per row: source_id, chunking_version, chunk_index, chapter,
  // text, char_start, char_end, embedding, embedding_bge, metadata.
  const COLS_PER_ROW = 10;
  const values: unknown[] = [];
  const tuples: string[] = [];
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    if (!chunk) continue;
    const bge = bgeVectors[i];
    if (!bge) throw new Error(`Missing BGE vector for chunk ${i}`);
    const openai = openaiVectors?.[i] ?? null;
    const base = i * COLS_PER_ROW;
    tuples.push(
      `($${base + 1}, $${base + 2}, $${base + 3}, $${base + 4}, $${base + 5}, $${base + 6}, $${base + 7}, $${base + 8}, $${base + 9}, $${base + 10})`,
    );
    values.push(
      sourceId,
      CHUNKING_VERSION,
      chunk.chunkIndex,
      chunk.chapter,
      chunk.text,
      chunk.charStart,
      chunk.charEnd,
      openai ? pgvector.toSql(openai) : null,
      pgvector.toSql(bge),
      { tokenCount: chunk.tokenCount },
    );
  }
  await db.query(
    `INSERT INTO chunks
       (source_id, chunking_version, chunk_index, chapter, text,
        char_start, char_end, embedding, embedding_bge, metadata)
     VALUES ${tuples.join(', ')}`,
    values,
  );
}
