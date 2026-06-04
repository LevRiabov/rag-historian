/**
 * evals/promptfoo/build-tests.ts — generate Promptfoo test cases from the
 * golden set + the current retrieval pipeline.
 *
 * Why this separation exists:
 *   - Our harness (`evals/run.ts`) does RETRIEVAL inside the loop, so each
 *     model swap re-retrieves from scratch — wasteful when we just want to
 *     compare GENERATION across models on the same chunks.
 *   - Promptfoo's strength is side-by-side comparison: same prompt, multiple
 *     providers, one UI showing all outputs.
 *   - This script "freezes" retrieval into a deterministic test fixture so
 *     every provider in Promptfoo gets the IDENTICAL evidence base. The
 *     comparison then isolates generator quality, not retrieval variance.
 *
 * Re-run this whenever:
 *   - The golden set changes (new questions, edited gold chunks)
 *   - The corpus is re-ingested (chunk IDs may shift)
 *   - The embedder is swapped (different vectors → different top-K)
 *
 * Output: `evals/promptfoo/tests.json` — committed to git as a snapshot.
 *
 * Run:
 *   pnpm dev evals/promptfoo/build-tests.ts
 */
import 'dotenv/config';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

import pg from 'pg';
import pgvector from 'pgvector/pg';

import { retrieve } from '../../roman-research/query/retrieve.ts';
import { type ChunkRange, covers } from '../metrics/recall.ts';
import type { GoldenEntry } from '../types.ts';

/** Matches what production passes to the LLM (mirrors roman-research/query). */
const TOP_K = 5;

const goldenPath = path.join('evals', 'golden-set.json');
const outPath = path.join('evals', 'promptfoo', 'tests.json');

const goldenRaw = await readFile(goldenPath, 'utf-8');
const golden = JSON.parse(goldenRaw) as GoldenEntry[];
console.log(`Loaded ${golden.length} questions from ${goldenPath}`);

const databaseUrl = process.env.DATABASE_URL;
if (!databaseUrl) throw new Error('DATABASE_URL is not set.');
const db = new pg.Client({ connectionString: databaseUrl });
await db.connect();
await pgvector.registerType(db);

interface PromptfooTestCase {
  description: string;
  vars: {
    question: string;
    sources_block: string;
    ideal_answer: string;
    /** "refuse" or "answer" — what the assistant SHOULD do. Used in
     *  rubrics via {{expected_behavior}} so the judge knows the correct
     *  branch. Mirrors metadata.should_refuse but in a string form
     *  that templates render cleanly. */
    expected_behavior: 'refuse' | 'answer';
  };
  metadata: {
    id: string;
    category: string;
    /** True when refusal is the expected behavior. Kept in metadata for
     *  filtering / reporting in the Promptfoo UI. */
    should_refuse: boolean;
    /** Number of retrieved chunks present in goldChunkIds (diagnostic). */
    gold_hits_in_topk: number;
  };
}

const tests: PromptfooTestCase[] = [];

for (const entry of golden) {
  process.stdout.write(`  [${entry.id}] retrieving... `);
  const retrieved = await retrieve(db, entry.question, { topK: TOP_K, provider: 'llamacpp' });

  // Same format as roman-research/query/answer.ts formatUserMessage's sources block.
  const sourcesBlock = retrieved
    .map(
      (c, i) => `[${i + 1}] ${c.source.author}, ${c.source.title}, ${c.chapter}\n${c.text.trim()}`,
    )
    .join('\n\n');

  const ranges: ChunkRange[] = retrieved.map((r) => ({
    sourceSlug: r.source.slug,
    charStart: r.charStart,
    charEnd: r.charEnd,
  }));
  const goldHits = entry.goldSpans.filter((span) => ranges.some((c) => covers(c, span))).length;

  const shouldRefuse = entry.goldSpans.length === 0 || entry.category === 'out-of-scope';

  tests.push({
    description: entry.id,
    vars: {
      question: entry.question,
      sources_block: sourcesBlock || '(no sources retrieved)',
      ideal_answer: entry.idealAnswer,
      expected_behavior: shouldRefuse ? 'refuse' : 'answer',
    },
    metadata: {
      id: entry.id,
      category: entry.category,
      should_refuse: shouldRefuse,
      gold_hits_in_topk: goldHits,
    },
  });
  console.log(`done (${retrieved.length} chunks, ${goldHits}/${entry.goldSpans.length} gold)`);
}

await db.end();

await mkdir(path.dirname(outPath), { recursive: true });
await writeFile(outPath, `${JSON.stringify(tests, null, 2)}\n`, 'utf-8');
console.log(`\nWrote ${tests.length} test cases to ${outPath}`);
console.log(`\nNext: pnpm exec promptfoo eval`);
