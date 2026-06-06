/**
 * evals/rejudge-completeness.ts — re-score COMPLETENESS on a saved run, without
 * re-running the agent or retrieval.
 *
 * Completeness needs only (question, idealAnswer, candidateAnswer) — all stored in
 * the result JSON. So when the completeness RUBRIC changes (e.g. Module 7 fix:
 * stop penalizing accurate extra content / don't judge fabrication), we can re-judge
 * the exact same answers for ~$0.10 and see the rubric's isolated effect — no model
 * calls for generation, no DB.
 *
 *   pnpm dev evals/rejudge-completeness.ts --in=evals/results/2026-06-05-agent-haiku.json
 *   pnpm dev evals/rejudge-completeness.ts --in=...json --out=...rejudged.json
 */
import 'dotenv/config';
import { readFile, writeFile } from 'node:fs/promises';

import { CLAUDE_MODELS, type ClaudeModel } from '../lib/index.ts';
import { createJudge } from './metrics/generation.ts';
import { mean } from './metrics/recall.ts';
import type { BatchResult, Category } from './types.ts';

const flags: Record<string, string> = {};
for (const a of process.argv.slice(2)) {
  const m = /^--([^=]+)(?:=(.*))?$/.exec(a);
  if (m?.[1]) flags[m[1]] = m[2] ?? '';
}
const inPath = flags.in;
if (!inPath) {
  console.error(
    'Usage: pnpm dev evals/rejudge-completeness.ts --in=<result.json> [--out=...] [--judge-model=haiku]',
  );
  process.exit(1);
}
const judgeModel: ClaudeModel =
  CLAUDE_MODELS[(flags['judge-model'] ?? 'haiku') as 'haiku' | 'sonnet' | 'opus'];

const batch = JSON.parse(await readFile(inPath, 'utf-8')) as BatchResult;
const judge = createJudge(judgeModel);
console.log(
  `Re-judging completeness on ${batch.results.length} questions from ${inPath}  (judge: ${judgeModel})\n`,
);

const CATEGORIES: Category[] = [
  'literal',
  'synonym',
  'multi-hop',
  'synthesis',
  'contradiction',
  'out-of-scope',
];

interface Row {
  id: string;
  cat: Category;
  old: number;
  next: number;
}
const rows: Row[] = [];

for (const r of batch.results) {
  if (!r.generation) continue;
  const old = r.generation.completeness.score;
  const res = await judge.completeness(
    r.entry.question,
    r.entry.idealAnswer,
    r.generation.answerText,
  );
  rows.push({ id: r.entry.id, cat: r.entry.category, old, next: res.score });
  // Update in place so an optional --out carries the corrected scores.
  r.generation.completeness = {
    score: res.score,
    reasoning: res.reasoning,
    missedFacts: res.missedFacts,
  };
  const arrow =
    res.score > old ? `↑ +${res.score - old}` : res.score < old ? `↓ ${res.score - old}` : '=';
  console.log(`  [${r.entry.id}] ${r.entry.category.padEnd(13)} ${old} → ${res.score}  ${arrow}`);
}

console.log('\n=== Completeness: old → new ===');
console.log(
  `  overall (n=${rows.length}):  ${mean(rows.map((r) => r.old)).toFixed(2)} → ${mean(rows.map((r) => r.next)).toFixed(2)}`,
);
for (const cat of CATEGORIES) {
  const rs = rows.filter((r) => r.cat === cat);
  if (rs.length === 0) continue;
  const o = mean(rs.map((r) => r.old));
  const n = mean(rs.map((r) => r.next));
  const delta = n - o;
  console.log(
    `  ${cat.padEnd(13)} n=${rs.length}  ${o.toFixed(2)} → ${n.toFixed(2)}  (${delta >= 0 ? '+' : ''}${delta.toFixed(2)})`,
  );
}

if (flags.out) {
  // Patch the stored completeness aggregates so the written file is self-consistent.
  const withGen = batch.results.filter((r) => r.generation);
  if (batch.aggregates.generation) {
    batch.aggregates.generation.completeness = mean(
      withGen.map((r) => r.generation?.completeness.score ?? Number.NaN),
    );
  }
  for (const cat of CATEGORIES) {
    const g = batch.aggregates.byCategory[cat]?.generation;
    if (g) {
      g.completeness = mean(
        batch.results
          .filter((r) => r.entry.category === cat && r.generation)
          .map((r) => r.generation?.completeness.score ?? Number.NaN),
      );
    }
  }
  await writeFile(flags.out, JSON.stringify(batch, null, 2), 'utf-8');
  console.log(`\nRe-judged batch written to ${flags.out}`);
}
