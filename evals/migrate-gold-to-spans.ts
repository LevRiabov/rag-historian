/**
 * evals/migrate-gold-to-spans.ts — one-time migration of the golden set from
 * version-specific `goldChunkIds` to chunking-invariant `goldSpans`.
 *
 * Why (Module 6.1): see the GoldSpan doc in evals/types.ts. We're about to
 * store multiple chunking variants side-by-side; recall must score all of
 * them from ONE golden set, which means gold can't be tied to naive-v1 chunk
 * IDs anymore.
 *
 * What it does: for each entry's legacy `goldChunkIds`, look up that naive-v1
 * chunk's (source slug, char_start, char_end) and emit it as a goldSpan. The
 * span ≈ the passage we originally hand-labeled, now expressed in source
 * coordinates that every future chunking variant shares. The legacy
 * `goldChunkIds` are PRESERVED on each entry for audit; the harness ignores
 * them going forward.
 *
 * Idempotent: re-running re-derives goldSpans from the (unchanged) legacy IDs.
 *
 * Run (DB must hold naive-v1 chunks — i.e. after a Module 4/5 ingest):
 *   pnpm dev evals/migrate-gold-to-spans.ts
 */
import 'dotenv/config';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

import pg from 'pg';

import type { GoldSpan } from './types.ts';

const NAIVE_VERSION = 'naive-v1';
const goldenPath = path.join('evals', 'golden-set.json');

// The on-disk shape still carries legacy goldChunkIds; goldSpans may already
// exist from a prior run. We re-derive goldSpans either way.
interface RawEntry {
  id: string;
  category: string;
  question: string;
  idealAnswer: string;
  goldSpans?: GoldSpan[];
  goldChunkIds?: number[];
  notes?: string;
}

const raw = await readFile(goldenPath, 'utf-8');
const entries = JSON.parse(raw) as RawEntry[];
console.log(`Loaded ${entries.length} entries from ${goldenPath}`);

const databaseUrl = process.env.DATABASE_URL;
if (!databaseUrl) throw new Error('DATABASE_URL is not set. Check .env + docker compose up -d.');
const db = new pg.Client({ connectionString: databaseUrl });
await db.connect();

// Build id -> span lookup for every naive-v1 chunk referenced by any entry.
const allIds = [...new Set(entries.flatMap((e) => e.goldChunkIds ?? []))];
const lookup = new Map<number, GoldSpan>();
if (allIds.length > 0) {
  const rows = await db.query<{
    id: number;
    slug: string;
    char_start: number;
    char_end: number;
  }>(
    `SELECT c.id, s.slug, c.char_start, c.char_end
       FROM chunks c
       JOIN sources s ON s.id = c.source_id
      WHERE c.chunking_version = $1 AND c.id = ANY($2::int[])`,
    [NAIVE_VERSION, allIds],
  );
  for (const r of rows.rows) {
    lookup.set(r.id, { sourceSlug: r.slug, charStart: r.char_start, charEnd: r.char_end });
  }
}
await db.end();

// Resolve each entry's gold IDs to spans; warn on any ID we couldn't find
// (stale label, or DB not ingested at naive-v1).
let missing = 0;
const migrated = entries.map((e) => {
  const ids = e.goldChunkIds ?? [];
  const spans: GoldSpan[] = [];
  for (const id of ids) {
    const span = lookup.get(id);
    if (!span) {
      console.warn(`  ⚠ [${e.id}] gold chunk id ${id} not found in ${NAIVE_VERSION} — skipped`);
      missing++;
      continue;
    }
    spans.push(span);
  }
  // Deterministic ordering so diffs are stable across re-runs.
  spans.sort((a, b) => a.sourceSlug.localeCompare(b.sourceSlug) || a.charStart - b.charStart);

  // Rebuild with explicit field order (goldSpans canonical, goldChunkIds kept
  // for provenance right after it).
  return {
    id: e.id,
    category: e.category,
    question: e.question,
    idealAnswer: e.idealAnswer,
    goldSpans: spans,
    goldChunkIds: ids,
    ...(e.notes !== undefined ? { notes: e.notes } : {}),
  };
});

await writeFile(goldenPath, `${JSON.stringify(migrated, null, 2)}\n`, 'utf-8');

const totalSpans = migrated.reduce((n, e) => n + e.goldSpans.length, 0);
console.log(
  `\nWrote ${migrated.length} entries (${totalSpans} gold spans${missing ? `, ${missing} ids skipped` : ''}) to ${goldenPath}`,
);
