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

import {
  CLAUDE_MODELS,
  type ClaudeModel,
  createLangfuseTracer,
  type EmbeddingProvider,
  flushLangfuse,
  LLAMACPP_MODELS,
} from '../lib/index.ts';
import { type AgentLLM, runAgent } from '../roman-research/agent/index.ts';
import { answerQuestion, DEFAULT_LLM, type LLMChoice } from '../roman-research/query/answer.ts';
import { generateQueryVariations } from '../roman-research/query/expand.ts';
import { generateHydeDoc } from '../roman-research/query/hyde.ts';
import { expandToParents, retrieve, retrieveMultiQuery } from '../roman-research/query/retrieve.ts';
import { classifyRefusal, createJudge } from './metrics/generation.ts';
import {
  type ChunkRange,
  coveredCount,
  covers,
  mean,
  spanRecallAtK,
  spanReciprocalRank,
} from './metrics/recall.ts';
import {
  type AgentAggregate,
  type AgentMetrics,
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
// --hyde: query rewriting via Hypothetical Document Embeddings (Module 6.5).
// Generate a hypothetical answer with the local LLM and embed THAT for the
// first-stage vector search (reranking/BM25 still use the raw question). Aimed
// at the synthesis floor — abstract questions with no entity anchor.
// --hyde-model overrides the llama-swap profile (default qwen-9b-16k).
const hydeEnabled = flags.hyde !== undefined;
const hydeModel = flags['hyde-model'] || undefined;
// --hyde-concat: embed `question + hypothetical doc` together instead of the
// doc alone. Keeps the question's own discriminative terms in the vector (pure
// HyDE throws them away — which crushed synonym recall) while still gaining the
// answer-shaped expansion. The robustness variant of HyDE.
const hydeConcat = flags['hyde-concat'] !== undefined;
// --expand: query expansion / multi-query (Module 6.5). Generate N rephrasings
// with the local LLM, retrieve each, fuse the rankings via RRF. Additive (the
// original question is always query #0), so it can only widen coverage. Targets
// synonym / multi-hop. --expand-n sets the rephrasing count (default 3),
// --expand-model the llama-swap profile.
const expandEnabled = flags.expand !== undefined;
const expandN = Number(flags['expand-n'] ?? '3');
const expandModel = flags['expand-model'] || undefined;
const genEnabled = flags.generation !== undefined || flags.full !== undefined;
// --agent (Module 7): replace single-shot retrieve+generate with the agentic
// loop (roman-research/agent). The agent drives its OWN retrieval via tools, so
// chunking/hybrid/rerank/hyde/expand flags don't apply; --llm picks the driver
// (claude-haiku default | claude-sonnet | claude-opus | llamacpp). --max-iter
// caps iterations; --think uses qwen's thinking-on profile (llamacpp only).
const agentEnabled = flags.agent !== undefined;
const maxIterations = Number(flags['max-iter'] ?? '30');
const agentThink = flags.think !== undefined;
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
// --category=contradiction (or a comma list) restricts the run to those
// categories — cheap iteration when tuning one category (e.g. a prompt fix).
const categoryFilter = flags.category;
// --ids=q-025,q-020 restricts to specific question ids (targeted re-runs, e.g.
// re-judging just the questions a model did poorly on). Composes with --category.
const idsFilter = flags.ids ? flags.ids.split(',') : null;
const golden = (JSON.parse(goldenRaw) as GoldenEntry[]).filter(
  (e) =>
    (!categoryFilter || categoryFilter.split(',').includes(e.category)) &&
    (!idsFilter || idsFilter.includes(e.id)),
);
console.log(
  `Loaded ${golden.length} questions from ${goldenPath}${categoryFilter ? ` (category: ${categoryFilter})` : ''}`,
);
const rerankDesc = rerankEnabled ? `on (pool ${rerankPoolK})` : 'off';
const hydeDesc = hydeEnabled
  ? `on (${hydeModel ?? 'qwen-9b-16k'}${hydeConcat ? ', concat' : ''})`
  : 'off';
const expandDesc = expandEnabled ? `on (n=${expandN}, ${expandModel ?? 'qwen-9b-16k'})` : 'off';
console.log(
  `Embedder: ${embedder}  |  Chunking: ${chunkingVersion}  |  Mode: ${retrievalMode}  |  Rerank: ${rerankDesc}  |  HyDE: ${hydeDesc}  |  Expand: ${expandDesc}  |  Top-K: ${RETRIEVE_TOP_K}`,
);
if (agentEnabled) {
  const driver = llm === 'llamacpp' ? `llamacpp (${agentThink ? 'think' : 'qwen-9b-16k'})` : llm;
  console.log(
    `Agent: ON  |  Driver: ${driver}  |  max-iter: ${maxIterations}  |  Judge: ${judgeModel}`,
  );
} else if (genEnabled) {
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
const judge = genEnabled || agentEnabled ? createJudge(judgeModel) : null;
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

  // --- Agent mode (Module 7) --------------------------------------------
  // The agent drives its own multi-step retrieval; we score the final article
  // with the SAME judges, plus agent-specific metrics. goldCoverage (over the
  // union of consulted chunks) stands in for ranked recall@k.
  if (agentEnabled) {
    try {
      // One Langfuse trace per question (no-op if keys absent). The tracer nests
      // a generation per model round-trip + a span per tool call; eval scores get
      // attached below so Langfuse becomes the per-question A/B drill-down.
      const lf = createLangfuseTracer({
        name: `agent:${entry.category}:${entry.id}`,
        input: entry.question,
        metadata: { id: entry.id, category: entry.category, llm },
        tags: ['agent', llm, entry.category],
      });
      const ag = await runAgent(db, entry.question, {
        llm: llm as AgentLLM,
        maxIterations,
        llamacppModel: agentThink ? LLAMACPP_MODELS.qwen9b16kThink : llamacppModel,
        tracer: lf.tracer,
      });
      const consultedRanges: ChunkRange[] = ag.consultedChunks.map((c) => ({
        sourceSlug: c.source.slug,
        charStart: c.charStart,
        charEnd: c.charEnd,
      }));
      const goldCoverage =
        entry.goldSpans.length === 0
          ? Number.NaN
          : coveredCount(consultedRanges, entry.goldSpans) / entry.goldSpans.length;
      console.log(
        `  agent:    ${ag.latencyMs}ms  stop=${ag.stop}  tools=${ag.toolCalls} ${JSON.stringify(ag.toolCallsByName)}  consulted=${ag.consultedCount}  coverage=${Number.isNaN(goldCoverage) ? 'n/a' : fmtPct(goldCoverage).trim()}`,
      );
      if (showAnswers) console.log(`  A:\n${indent(ag.article)}`);

      let generation: GenerationScore | undefined;
      if (judge) {
        try {
          const tJ = Date.now();
          const [faith, complete, refusalJudgment] = await Promise.all([
            // labelByChunkId=true: the agent cites by chunk_id, so the judge's
            // evidence must be labeled by chunk_id too (else correct citations
            // read as fabricated — Module 7 measurement fix).
            judge.faithfulness(entry.question, ag.consultedChunks, ag.article, true),
            judge.completeness(entry.question, entry.idealAnswer, ag.article),
            judge.refusal(ag.article),
          ]);
          const judgeMs = Date.now() - tJ;
          const shouldRefuse = entry.category === 'out-of-scope' || entry.goldSpans.length === 0;
          const classification = classifyRefusal(refusalJudgment.didRefuse, shouldRefuse);
          console.log(
            `  judge:    ${judgeMs}ms  F=${faith.score}/5  C=${complete.score}/5  R=${classification}`,
          );
          generation = {
            answerText: ag.article,
            generatorLabel: ag.llmLabel,
            generatorLatencyMs: ag.latencyMs,
            generatorCostUSD: ag.rawCostUSD,
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
              correct:
                classification === 'correct-refused' || classification === 'correct-answered',
              classification,
              reasoning: refusalJudgment.reasoning,
            },
            judgeLabel: judge.label,
            judgeCostUSD: faith.costUSD + complete.costUSD + refusalJudgment.costUSD,
          };
          // Attach the judge scores to the Langfuse trace (the eval drill-down).
          lf.score('faithfulness', faith.score, faith.reasoning);
          lf.score('completeness', complete.score, complete.reasoning);
          lf.score('refusal_correct', generation.refusal.correct ? 1 : 0, classification);
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          console.log(
            `  ✗ judging failed after retries: ${msg.slice(0, 120)} (ungraded; continuing)`,
          );
        }
      }

      const agent: AgentMetrics = {
        toolCalls: ag.toolCalls,
        toolCallsByName: ag.toolCallsByName,
        stop: ag.stop,
        calledFinalize: (ag.toolCallsByName.finalize ?? 0) > 0,
        consultedCount: ag.consultedCount,
        goldCoverage,
        latencyMs: ag.latencyMs,
        costUSD: ag.rawCostUSD,
      };
      if (!Number.isNaN(goldCoverage)) lf.score('gold_coverage', goldCoverage);
      lf.score('tool_calls', ag.toolCalls);
      lf.end(ag.article, {
        stop: ag.stop,
        toolCalls: ag.toolCalls,
        consulted: ag.consultedCount,
        costUSD: ag.rawCostUSD,
        llmLabel: ag.llmLabel,
      });
      results.push({
        entry,
        retrievedChunkIds: ag.consultedChunks.map((c) => c.chunkId),
        similarities: [],
        latencyMs: ag.latencyMs,
        recallAtK: {},
        mrr: Number.NaN,
        generation,
        agent,
      });
    } catch (err) {
      // One question failing (e.g. local context overflow from over-search)
      // must not kill the batch — mirror the generation path's resilience.
      const msg = err instanceof Error ? err.message : String(err);
      console.log(`  ✗ agent run failed: ${msg.slice(0, 160)} (ungraded; continuing)`);
    }
    continue;
  }

  // --- HyDE (optional) --------------------------------------------------
  // Generate the hypothetical-answer doc BEFORE retrieval and embed it instead
  // of the question. Timed separately so the per-query LLM tax is visible.
  let embedText: string | undefined;
  if (hydeEnabled) {
    const hyde = await generateHydeDoc(entry.question, { llamacppModel: hydeModel });
    embedText = hydeConcat ? `${entry.question}\n\n${hyde.doc}` : hyde.doc;
    console.log(
      `  hyde:     ${hyde.latencyMs}ms  (${hyde.outputTokens} out tok)  "${hyde.doc.slice(0, 90).replace(/\n/g, ' ')}…"`,
    );
  }

  // --- Query expansion (optional) ---------------------------------------
  // Build the multi-query set BEFORE retrieval. The original question is always
  // query #0 so expansion is additive. Timed via the retrieve block below.
  let queries: string[] | undefined;
  if (expandEnabled) {
    const exp = await generateQueryVariations(entry.question, {
      n: expandN,
      llamacppModel: expandModel,
    });
    queries = [entry.question, ...exp.variations];
    console.log(
      `  expand:   ${exp.latencyMs}ms  (+${exp.variations.length} variations)  ${exp.variations
        .map((v) => `"${v.slice(0, 55).replace(/\n/g, ' ')}"`)
        .join('  ')}`,
    );
  }

  // --- Retrieve ---------------------------------------------------------
  const retrieveOpts = {
    topK: RETRIEVE_TOP_K,
    provider: embedder,
    chunkingVersion,
    mode: retrievalMode,
    lexicalWeight,
    rerank: rerankEnabled,
    rerankPoolK,
  };
  const tR = Date.now();
  const retrieved = queries
    ? await retrieveMultiQuery(db, entry.question, queries, retrieveOpts)
    : await retrieve(db, entry.question, { ...retrieveOpts, embedText });
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
            .map((c) => `"${c.slice(0, 100)}"`)
            .join('; ')}`,
        );
      }
      if (showAnswers && complete.missedFacts.length > 0) {
        console.log(
          `    missed:      ${complete.missedFacts
            .slice(0, 2)
            .map((c) => `"${c.slice(0, 100)}"`)
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
// Flush any queued Langfuse events (no-op when tracing is disabled).
await flushLangfuse();

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

function aggregateAgent(rs: QuestionResult[]): AgentAggregate | undefined {
  const withA = rs.filter((r): r is QuestionResult & { agent: AgentMetrics } => r.agent != null);
  if (withA.length === 0) return undefined;
  return {
    runs: withA.length,
    avgToolCalls: mean(withA.map((r) => r.agent.toolCalls)),
    // mean() skips NaN, so out-of-scope (no gold) is excluded from coverage.
    avgGoldCoverage: mean(withA.map((r) => r.agent.goldCoverage)),
    finalizeRate: withA.filter((r) => r.agent.calledFinalize).length / withA.length,
    avgConsulted: mean(withA.map((r) => r.agent.consultedCount)),
    avgLatencyMs: mean(withA.map((r) => r.agent.latencyMs)),
    avgCostUSD: mean(withA.map((r) => r.agent.costUSD)),
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
    agent: aggregateAgent(catResults),
  };
}

const totalGeneratorCostUSD = results
  .map((r) => r.generation?.generatorCostUSD ?? 0)
  .reduce((a, b) => a + b, 0);
const totalJudgeCostUSD = results
  .map((r) => r.generation?.judgeCostUSD ?? 0)
  .reduce((a, b) => a + b, 0);
const topGenAgg = aggregateGeneration(results);
const topAgentAgg = aggregateAgent(results);
const totalAgentCostUSD = results.map((r) => r.agent?.costUSD ?? 0).reduce((a, b) => a + b, 0);

const batch: BatchResult = {
  results,
  aggregates: {
    recallAtK: aggregateRecall(inScope),
    mrr: aggregateMRR(inScope),
    byCategory,
    generation: topGenAgg ? { ...topGenAgg, totalJudgeCostUSD, totalGeneratorCostUSD } : undefined,
    agent: topAgentAgg ? { ...topAgentAgg, totalCostUSD: totalAgentCostUSD } : undefined,
  },
  config: {
    embedder,
    chunkingVersion,
    retrievalMode,
    rerank: rerankEnabled ? `bge-reranker-v2-m3 (pool ${rerankPoolK})` : 'off',
    hyde: hydeDesc,
    expand: expandDesc,
    topK: RETRIEVE_TOP_K,
    timestamp: new Date().toISOString(),
    generator: genEnabled ? llm : undefined,
    judge: judge ? judgeModel : undefined,
    agent: agentEnabled
      ? `${llm}${agentThink ? ' (think)' : ''}, max-iter ${maxIterations}`
      : undefined,
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
// Ranked recall@k is a single-shot metric — the agent has no fixed top-K, so in
// agent mode we report gold coverage (over the consulted union) instead.
if (!batch.aggregates.agent) {
  const headers = ['k=1', 'k=3', 'k=5', 'k=10', 'k=20'];
  console.log(`  Recall:  ${headers.map((h) => h.padStart(7)).join(' ')}`);
  console.log(
    `           ${EVAL_K_VALUES.map((k) => fmtPct(batch.aggregates.recallAtK[k] ?? Number.NaN).padStart(7)).join(' ')}`,
  );
  const mrrAgg = batch.aggregates.mrr;
  console.log(`  MRR:     ${Number.isNaN(mrrAgg) ? '  -' : mrrAgg.toFixed(3)}`);
}

if (batch.aggregates.agent) {
  const a = batch.aggregates.agent;
  console.log(`\n=== Agent aggregates (n=${a.runs}) ===`);
  console.log(`  Avg tool calls:    ${a.avgToolCalls.toFixed(1)}`);
  console.log(`  Avg gold coverage: ${fmtPct(a.avgGoldCoverage).trim()}  (over consulted union)`);
  console.log(`  Finalize rate:     ${(a.finalizeRate * 100).toFixed(0)}%`);
  console.log(`  Avg consulted:     ${a.avgConsulted.toFixed(1)} chunks`);
  console.log(`  Avg latency:       ${Math.round(a.avgLatencyMs)}ms`);
  console.log(`  Total cost:        $${a.totalCostUSD.toFixed(4)}`);
}

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
  let line: string;
  if (b.agent) {
    const cover = Number.isNaN(b.agent.avgGoldCoverage)
      ? '  n/a '
      : fmtPct(b.agent.avgGoldCoverage);
    line = `  ${cat.padEnd(13)} n=${b.count}  tools=${b.agent.avgToolCalls.toFixed(1).padStart(4)}  coverage=${cover}`;
  } else {
    const recall5 = fmtPct(b.recallAtK[5] ?? Number.NaN);
    const recall20 = fmtPct(b.recallAtK[20] ?? Number.NaN);
    const mrr = Number.isNaN(b.mrr) ? '  -' : b.mrr.toFixed(3);
    line = `  ${cat.padEnd(13)} n=${b.count}  recall@5=${recall5}  recall@20=${recall20}  MRR=${mrr}`;
  }
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
