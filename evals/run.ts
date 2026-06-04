/**
 * evals/run.ts — load the golden set, run retrieval (+ optional generation
 * + judging), score, print.
 *
 * Run:
 *   pnpm dev evals/run.ts                                        # retrieval only
 *   pnpm dev evals/run.ts --embedder=openai
 *   pnpm dev evals/run.ts --generation                           # + faithfulness/completeness/refusal
 *   pnpm dev evals/run.ts --generation --llm=claude-haiku        # use Claude as generator instead of LM Studio
 *   pnpm dev evals/run.ts --out=evals/results/2026-05-29-naive-bge.json
 *
 * Two modes:
 *   - Retrieval-only (default): fast, cheap, runs in seconds. Use this
 *     for iterating on chunking / embedding / retrieval strategies.
 *   - Full (--generation): also generates an answer per question and runs
 *     three LLM-as-judge calls. Slower (~minutes), small cost (~$0.50 per
 *     run on Haiku judge). Use this when measuring final answer quality.
 *
 * Output: per-question detail + aggregate table. Optionally persists the
 * full BatchResult to `--out` as JSON for diffing runs over time.
 */
import 'dotenv/config';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

import pg from 'pg';
import pgvector from 'pgvector/pg';

import { CLAUDE_MODELS, type ClaudeModel, type EmbeddingProvider } from '../lib/index.ts';
import { answerQuestion, DEFAULT_LLM, type LLMChoice } from '../roman-research/query/answer.ts';
import { expandToParents, retrieve } from '../roman-research/query/retrieve.ts';
import { classifyRefusal, createJudge } from './metrics/generation.ts';
import {
  type ChunkRange,
  covers,
  mean,
  spanRecallAtK,
  spanReciprocalRank,
} from './metrics/recall.ts';
import {
  type BatchResult,
  type Category,
  EVAL_K_VALUES,
  type GenerationScore,
  type GoldenEntry,
  type QuestionResult,
  RETRIEVE_TOP_K,
} from './types.ts';

const CATEGORIES: Category[] = [
  'literal',
  'synonym',
  'multi-hop',
  'synthesis',
  'contradiction',
  'out-of-scope',
];

// ---------------------------------------------------------------------------
// CLI parsing — same minimal pattern as roman-research/query/index.ts.
// ---------------------------------------------------------------------------
const args = process.argv.slice(2);
const flags: Record<string, string> = {};
for (const a of args) {
  // --key=value form
  const kv = /^--([^=]+)=(.+)$/.exec(a);
  if (kv?.[1] && kv[2]) {
    flags[kv[1]] = kv[2];
    continue;
  }
  // --key boolean form (empty string indicates "set")
  const k = /^--(.+)$/.exec(a);
  if (k?.[1]) flags[k[1]] = '';
}

const embedder = (flags.embedder ?? 'llamacpp') as EmbeddingProvider;
const goldenPath = flags.golden ?? path.join('evals', 'golden-set.json');
const outPath = flags.out;
const chunkingVersion = flags['chunking-version'] ?? 'naive-v1';
// --hybrid: dense + lexical (BM25-style FTS) fused via RRF (Module 6.2).
// Default vector-only mirrors the Module 4/5 baseline.
const retrievalMode: 'vector' | 'hybrid' = flags.hybrid !== undefined ? 'hybrid' : 'vector';
// Weight on the lexical (BM25) RRF arm; <1 down-weights it so it adds signal
// on exact-term queries without overriding the vector arm on semantic ones.
const lexicalWeight = Number(flags['lexical-weight'] ?? '1');
// --rerank: second-stage cross-encoder (Module 6.3). --rerank-pool sets how
// many first-stage candidates get reranked (default 50).
const rerankEnabled = flags.rerank !== undefined;
const rerankPoolK = Number(flags['rerank-pool'] ?? '50');
const genEnabled = flags.generation !== undefined || flags.full !== undefined;
const llm = (flags.llm ?? DEFAULT_LLM) as LLMChoice;
// Optional override for the llama-swap model profile — pass a profile name
// from the llama-swap config, e.g. 'qwen-9b-32k'. Only honored when
// llm='llamacpp'; ignored for Claude routes. Default lives in
// roman-research/query/answer.ts (qwen-9b-16k).
const llamacppModel = flags['llamacpp-model'] || undefined;
// Generator K — how many chunks we PASS TO THE LLM (matches production
// query pipeline). Distinct from RETRIEVE_TOP_K (20), which is the wider
// candidate set we score recall@k against. Default 5 mirrors what a real
// user gets through `roman-research/query`.
const generatorTopK = Number(flags['generator-k'] ?? '5');
// Judge defaults to Haiku — cheap + reliable for structured grading.
// Override via --judge-model=sonnet|opus for higher-stakes calibration runs.
const judgeModelKey = (flags['judge-model'] ?? 'haiku') as 'haiku' | 'sonnet' | 'opus';
const judgeModel: ClaudeModel = CLAUDE_MODELS[judgeModelKey];
// --show-answers: print the full generated answer text per question
// (under the score line). Verbose but essential for judge calibration —
// trust-but-verify the LLM-as-judge by reading actual outputs.
const showAnswers = flags['show-answers'] !== undefined;

// ---------------------------------------------------------------------------
// Load golden set
// ---------------------------------------------------------------------------
const goldenRaw = await readFile(goldenPath, 'utf-8');
const golden = JSON.parse(goldenRaw) as GoldenEntry[];
console.log(`Loaded ${golden.length} questions from ${goldenPath}`);
const rerankDesc = rerankEnabled ? `on (pool ${rerankPoolK})` : 'off';
console.log(
  `Embedder: ${embedder}  |  Chunking: ${chunkingVersion}  |  Mode: ${retrievalMode}  |  Rerank: ${rerankDesc}  |  Top-K: ${RETRIEVE_TOP_K}`,
);
if (genEnabled) {
  const llmDesc = llm === 'llamacpp' && llamacppModel ? `llamacpp (${llamacppModel})` : llm;
  console.log(
    `Generation: ON  |  Generator: ${llmDesc} (top-${generatorTopK} chunks)  |  Judge: ${judgeModel}`,
  );
} else {
  console.log(`Generation: OFF (retrieval-only run)`);
}
console.log();

// ---------------------------------------------------------------------------
// DB
// ---------------------------------------------------------------------------
const db = new pg.Client({ connectionString: process.env.DATABASE_URL });
await db.connect();
await pgvector.registerType(db);

// ---------------------------------------------------------------------------
// Run each question, collect QuestionResult.
// ---------------------------------------------------------------------------
const judge = genEnabled ? createJudge(judgeModel) : null;
const results: QuestionResult[] = [];

function indent(text: string, prefix = '      '): string {
  return text
    .split('\n')
    .map((line) => `${prefix}${line}`)
    .join('\n');
}

function tokPerSec(outTokens: number, ms: number): string {
  if (ms <= 0 || outTokens <= 0) return '?';
  return ((outTokens * 1000) / ms).toFixed(1);
}

for (const entry of golden) {
  console.log(`\n[${entry.id}] (${entry.category})`);
  console.log(`  Q: ${entry.question}`);

  // --- Retrieve ---------------------------------------------------------
  const tR = Date.now();
  const retrieved = await retrieve(db, entry.question, {
    topK: RETRIEVE_TOP_K,
    provider: embedder,
    chunkingVersion,
    mode: retrievalMode,
    lexicalWeight,
    rerank: rerankEnabled,
    rerankPoolK,
  });
  const latencyMs = Date.now() - tR;
  const retrievedIds = retrieved.map((r) => r.chunkId);
  const similarities = retrieved.map((r) => Number(r.similarity.toFixed(3)));

  // Map to the minimal shape recall.ts needs, preserving retrieval order so
  // recall@k / MRR see the same ranking the generator would.
  const ranges: ChunkRange[] = retrieved.map((r) => ({
    sourceSlug: r.source.slug,
    charStart: r.charStart,
    charEnd: r.charEnd,
  }));

  const recallByK: Record<number, number> = {};
  for (const k of EVAL_K_VALUES) {
    recallByK[k] = spanRecallAtK(ranges, entry.goldSpans, k);
  }
  const rr = spanReciprocalRank(ranges, entry.goldSpans);
  const goldHits =
    entry.goldSpans.length === 0
      ? 'n/a (out-of-scope)'
      : `${entry.goldSpans.filter((span) => ranges.some((c) => covers(c, span))).length}/${entry.goldSpans.length}`;
  console.log(
    `  retrieve: ${latencyMs}ms  | gold hits in top-${RETRIEVE_TOP_K}: ${goldHits}  | recall@5=${fmtPct(recallByK[5] ?? Number.NaN).trim()}  MRR=${Number.isNaN(rr) ? '-' : rr.toFixed(3)}`,
  );
  if (showAnswers) {
    // Show top-3 retrieved chunks so the answer can be evaluated against
    // what the generator actually saw.
    for (let i = 0; i < Math.min(3, retrieved.length); i++) {
      const c = retrieved[i];
      if (!c) continue;
      const range: ChunkRange = {
        sourceSlug: c.source.slug,
        charStart: c.charStart,
        charEnd: c.charEnd,
      };
      const isGold = entry.goldSpans.some((span) => covers(range, span)) ? '★' : ' ';
      console.log(
        `    ${isGold} [${i + 1}] sim=${c.similarity.toFixed(3)}  ${c.source.author}, ${c.chapter}  (chunk_id=${c.chunkId})`,
      );
    }
  }

  // --- Generate + judge -------------------------------------------------
  // Wrapped in try/catch so a transient judge failure (e.g., Anthropic
  // timeout) doesn't crash the entire run mid-way through. Failed
  // questions are recorded without `generation` and the run continues.
  let generation: GenerationScore | undefined;
  if (judge) {
    try {
      // Pass only top-K chunks to the generator — matches what a real user
      // gets through `roman-research/query`. Sending all 20 retrieved chunks
      // would (a) blow past LM Studio's 8192 context, (b) not reflect the
      // production pipeline we want to score.
      //
      // Slice to K FIRST, then expand parents (parent-child-v1) — so K bounds
      // the parent count and de-duped siblings can yield <K blocks. No-op for
      // flat variants (no parent pointer).
      const generatorChunks = expandToParents(retrieved.slice(0, generatorTopK));

      const tG = Date.now();
      const answer = await answerQuestion(entry.question, generatorChunks, { llm, llamacppModel });
      const genMs = Date.now() - tG;
      console.log(
        `  generate: ${genMs}ms  (${answer.outputTokens} out tok, ${tokPerSec(answer.outputTokens, genMs)} tok/s, ${answer.llmLabel})`,
      );
      if (showAnswers) {
        console.log(`  A:\n${indent(answer.text)}`);
      }

      // Judges see the SAME chunks the generator saw — faithfulness must be
      // evaluated against the actual evidence base, not the full 20-candidate set.
      const tJ = Date.now();
      const [faith, complete, refusalJudgment] = await Promise.all([
        judge.faithfulness(entry.question, generatorChunks, answer.text),
        judge.completeness(entry.question, entry.idealAnswer, answer.text),
        judge.refusal(answer.text),
      ]);
      const judgeMs = Date.now() - tJ;
      const shouldRefuse = entry.category === 'out-of-scope' || entry.goldSpans.length === 0;
      const classification = classifyRefusal(refusalJudgment.didRefuse, shouldRefuse);
      console.log(
        `  judge:    ${judgeMs}ms  F=${faith.score}/5  C=${complete.score}/5  R=${classification}`,
      );
      if (showAnswers && faith.unsupportedClaims.length > 0) {
        console.log(
          `    unsupported: ${faith.unsupportedClaims
            .slice(0, 2)
            .map((c) => '"' + c.slice(0, 100) + '"')
            .join('; ')}`,
        );
      }
      if (showAnswers && complete.missedFacts.length > 0) {
        console.log(
          `    missed:      ${complete.missedFacts
            .slice(0, 2)
            .map((c) => '"' + c.slice(0, 100) + '"')
            .join('; ')}`,
        );
      }

      generation = {
        answerText: answer.text,
        generatorLabel: answer.llmLabel,
        generatorLatencyMs: answer.latencyMs,
        generatorCostUSD: answer.rawCostUSD,
        faithfulness: {
          score: faith.score,
          reasoning: faith.reasoning,
          unsupportedClaims: faith.unsupportedClaims,
        },
        completeness: {
          score: complete.score,
          reasoning: complete.reasoning,
          missedFacts: complete.missedFacts,
        },
        refusal: {
          didRefuse: refusalJudgment.didRefuse,
          shouldRefuse,
          correct: classification === 'correct-refused' || classification === 'correct-answered',
          classification,
          reasoning: refusalJudgment.reasoning,
        },
        judgeLabel: judge.label,
        judgeCostUSD: faith.costUSD + complete.costUSD + refusalJudgment.costUSD,
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.log(`  ✗ generation/judging failed after retries: ${msg.slice(0, 120)}`);
      console.log(`    (this question is marked as ungraded; run continues)`);
      // generation stays undefined — aggregates will exclude it.
    }
  }

  results.push({
    entry,
    retrievedChunkIds: retrievedIds,
    similarities,
    latencyMs,
    recallAtK: recallByK,
    mrr: rr,
    generation,
  });
}

await db.end();

// ---------------------------------------------------------------------------
// Aggregate
// ---------------------------------------------------------------------------
function aggregateRecall(rs: QuestionResult[]): Record<number, number> {
  const agg: Record<number, number> = {};
  for (const k of EVAL_K_VALUES) {
    agg[k] = mean(rs.map((r) => r.recallAtK[k] ?? Number.NaN));
  }
  return agg;
}
function aggregateMRR(rs: QuestionResult[]): number {
  return mean(rs.map((r) => r.mrr));
}

function aggregateGeneration(
  rs: QuestionResult[],
):
  | { answeredCount: number; faithfulness: number; completeness: number; refusalAccuracy: number }
  | undefined {
  const withGen = rs.filter((r) => r.generation !== undefined);
  if (withGen.length === 0) return undefined;
  return {
    answeredCount: withGen.length,
    faithfulness: mean(withGen.map((r) => (r.generation as GenerationScore).faithfulness.score)),
    completeness: mean(withGen.map((r) => (r.generation as GenerationScore).completeness.score)),
    refusalAccuracy:
      withGen.filter((r) => (r.generation as GenerationScore).refusal.correct).length /
      withGen.length,
  };
}

const inScope = results.filter((r) => r.entry.goldSpans.length > 0);

const byCategory = {} as BatchResult['aggregates']['byCategory'];
for (const cat of CATEGORIES) {
  const catResults = results.filter((r) => r.entry.category === cat);
  const inScopeCat = catResults.filter((r) => r.entry.goldSpans.length > 0);
  byCategory[cat] = {
    count: catResults.length,
    recallAtK: aggregateRecall(inScopeCat),
    mrr: aggregateMRR(inScopeCat),
    generation: aggregateGeneration(catResults),
  };
}

const totalGeneratorCostUSD = results
  .map((r) => r.generation?.generatorCostUSD ?? 0)
  .reduce((a, b) => a + b, 0);
const totalJudgeCostUSD = results
  .map((r) => r.generation?.judgeCostUSD ?? 0)
  .reduce((a, b) => a + b, 0);
const topGenAgg = aggregateGeneration(results);

const batch: BatchResult = {
  results,
  aggregates: {
    recallAtK: aggregateRecall(inScope),
    mrr: aggregateMRR(inScope),
    byCategory,
    generation: topGenAgg ? { ...topGenAgg, totalJudgeCostUSD, totalGeneratorCostUSD } : undefined,
  },
  config: {
    embedder,
    chunkingVersion,
    retrievalMode,
    rerank: rerankEnabled ? `bge-reranker-v2-m3 (pool ${rerankPoolK})` : 'off',
    topK: RETRIEVE_TOP_K,
    timestamp: new Date().toISOString(),
    generator: genEnabled ? llm : undefined,
    judge: genEnabled ? judgeModel : undefined,
  },
};

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------
function fmtPct(v: number): string {
  if (Number.isNaN(v)) return '   -  ';
  return `${(v * 100).toFixed(1).padStart(5)}%`;
}

// Per-question detail was printed live during the run (above) — skip the
// duplicate end-of-run dump. Keep only the aggregates.
console.log('\n=== Aggregates (in-scope only) ===');
const headers = ['k=1', 'k=3', 'k=5', 'k=10', 'k=20'];
console.log(`  Recall:  ${headers.map((h) => h.padStart(7)).join(' ')}`);
console.log(
  `           ${EVAL_K_VALUES.map((k) => fmtPct(batch.aggregates.recallAtK[k] ?? Number.NaN).padStart(7)).join(' ')}`,
);
const mrrAgg = batch.aggregates.mrr;
console.log(`  MRR:     ${Number.isNaN(mrrAgg) ? '  -' : mrrAgg.toFixed(3)}`);

if (batch.aggregates.generation) {
  const g = batch.aggregates.generation;
  console.log(`\n=== Generation aggregates (n=${g.answeredCount}) ===`);
  console.log(`  Faithfulness:      ${g.faithfulness.toFixed(2)} / 5`);
  console.log(`  Completeness:      ${g.completeness.toFixed(2)} / 5`);
  console.log(`  Refusal accuracy:  ${(g.refusalAccuracy * 100).toFixed(1)}%`);
  console.log(`  Generator cost:    $${g.totalGeneratorCostUSD.toFixed(4)}`);
  console.log(`  Judge cost:        $${g.totalJudgeCostUSD.toFixed(4)}`);
}

console.log('\n=== Per category ===');
for (const cat of CATEGORIES) {
  const b = batch.aggregates.byCategory[cat];
  if (b.count === 0) continue;
  const recall5 = fmtPct(b.recallAtK[5] ?? Number.NaN);
  const recall20 = fmtPct(b.recallAtK[20] ?? Number.NaN);
  const mrr = Number.isNaN(b.mrr) ? '  -' : b.mrr.toFixed(3);
  let line = `  ${cat.padEnd(13)} n=${b.count}  recall@5=${recall5}  recall@20=${recall20}  MRR=${mrr}`;
  if (b.generation) {
    line += `  | faith=${b.generation.faithfulness.toFixed(2)} comp=${b.generation.completeness.toFixed(2)} refusal=${(b.generation.refusalAccuracy * 100).toFixed(0)}%`;
  }
  console.log(line);
}

// (Per-question generation detail was streamed live above.)

// ---------------------------------------------------------------------------
// Optional: persist run for future diffs
// ---------------------------------------------------------------------------
if (outPath) {
  await mkdir(path.dirname(outPath), { recursive: true });
  await writeFile(outPath, JSON.stringify(batch, null, 2), 'utf-8');
  console.log(`\nFull result written to ${outPath}`);
}
