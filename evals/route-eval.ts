/**
 * evals/route-eval.ts — measure the routing CLASSIFIER against the gold categories
 * (Module 9, Slice 3, Step B/C-decoupled).
 *
 * The routing experiment has two unknowns:
 *   1. Is the POLICY worth it? (oracle routing vs all-Haiku/all-Sonnet) — estimable
 *      from existing M7 numbers, no new agent runs needed.
 *   2. Can a CHEAP classifier predict the right tier? — THIS script, ~50 Haiku
 *      structured calls (cents), no agent loop, no Sonnet spend.
 *
 * We have the gold category for every golden question, which gives the ground-truth
 * routing decision:
 *   simple    = literal | synonym | out-of-scope   (Haiku is fine / should refuse)
 *   reasoning = multi-hop | synthesis | contradiction  (the F19 Sonnet lever)
 *
 * Error asymmetry (the number that matters):
 *   - DANGEROUS misroute: reasoning gold → predicted simple  → routed to Haiku → quality loss
 *   - SAFE misroute:      simple gold    → predicted reasoning → wasted money only
 * A good router minimizes DANGEROUS misroutes even at the cost of some SAFE ones.
 *
 *   pnpm dev evals/route-eval.ts
 */
import 'dotenv/config';
import { readFile } from 'node:fs/promises';
import path from 'node:path';

import { createRateLimiter, createRouter, type RouteTier } from '../lib/index.ts';
import type { Category } from './types.ts';

interface GoldEntry {
  id: string;
  category: Category;
  question: string;
}

const REASONING_CATEGORIES = new Set<Category>(['multi-hop', 'synthesis', 'contradiction']);
const goldTier = (c: Category): RouteTier => (REASONING_CATEGORIES.has(c) ? 'reasoning' : 'simple');

const goldPath = path.join(import.meta.dirname, 'golden-set.json');
const gold: GoldEntry[] = JSON.parse(await readFile(goldPath, 'utf8'));

const router = createRouter(); // Haiku classifier → {Haiku, Sonnet}
const limiter = createRateLimiter({ maxConcurrent: 8 });

console.log(`Classifying ${gold.length} golden questions (Haiku, temp=0)…\n`);

interface Row {
  id: string;
  category: Category;
  goldTier: RouteTier;
  predTier: RouteTier;
  rationale: string;
  costUSD: number;
}

const rows: Row[] = await Promise.all(
  gold.map((e) =>
    limiter.schedule(async () => {
      const r = await router.route(e.question);
      return {
        id: e.id,
        category: e.category,
        goldTier: goldTier(e.category),
        predTier: r.tier,
        rationale: r.rationale,
        costUSD: r.cost.totalUSD,
      };
    }),
  ),
);

// ---- Tally -----------------------------------------------------------------
let correct = 0;
let dangerous = 0; // reasoning gold → simple pred (quality risk)
let safe = 0; // simple gold → reasoning pred (cost waste)
const perCat: Record<string, { n: number; correct: number; toSonnet: number }> = {};

for (const r of rows) {
  const cat = r.category;
  perCat[cat] ??= { n: 0, correct: 0, toSonnet: 0 };
  perCat[cat].n++;
  if (r.predTier === 'reasoning') perCat[cat].toSonnet++;
  if (r.predTier === r.goldTier) {
    correct++;
    perCat[cat].correct++;
  } else if (r.goldTier === 'reasoning' && r.predTier === 'simple') {
    dangerous++;
  } else {
    safe++;
  }
}

const total = rows.length;
const totalCost = rows.reduce((s, r) => s + r.costUSD, 0);
const reasoningTotal = rows.filter((r) => r.goldTier === 'reasoning').length;
const simpleTotal = total - reasoningTotal;

// ---- Report ----------------------------------------------------------------
console.log('--- Per-category (predicted → Sonnet share) ---');
console.log('category        n   acc    →Sonnet');
for (const cat of Object.keys(perCat).sort()) {
  const p = perCat[cat];
  if (!p) continue;
  const acc = ((100 * p.correct) / p.n).toFixed(0).padStart(3);
  console.log(
    `${cat.padEnd(14)} ${String(p.n).padStart(2)}  ${acc}%   ${p.toSonnet}/${p.n}${
      goldTier(cat as Category) === 'reasoning'
        ? '  (should be all Sonnet)'
        : '  (should be all Haiku)'
    }`,
  );
}

console.log('\n--- Headline ---');
console.log(
  `Overall accuracy:        ${correct}/${total} (${((100 * correct) / total).toFixed(1)}%)`,
);
console.log(
  `DANGEROUS misroutes:     ${dangerous}/${reasoningTotal} reasoning Qs sent to Haiku  ← quality risk (minimize)`,
);
console.log(
  `SAFE misroutes:          ${safe}/${simpleTotal} simple Qs sent to Sonnet     ← cost waste only`,
);
console.log(
  `Classification cost:     $${totalCost.toFixed(6)}  (~$${(totalCost / total).toFixed(6)}/q)`,
);

// Show the dangerous ones explicitly — these are what a quality regression looks like.
const dangerousRows = rows.filter((r) => r.goldTier === 'reasoning' && r.predTier === 'simple');
if (dangerousRows.length) {
  console.log('\n--- DANGEROUS misroutes (reasoning → Haiku) ---');
  for (const r of dangerousRows) {
    console.log(`  [${r.category}] ${r.id}: "${r.rationale}"`);
  }
}
