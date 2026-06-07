/**
 * roman-research/agent/cli.ts — run the Roman Research Agent on one question,
 * logging each tool call live (the ReAct trace) and the final article + metrics.
 *
 *   pnpm dev roman-research/agent/cli.ts "Did Caesar want to be made king?"
 *   pnpm dev roman-research/agent/cli.ts "..." --llm=llamacpp
 *   pnpm dev roman-research/agent/cli.ts "..." --llm=claude-sonnet --max-iter=20
 *
 * Flags:
 *   --llm=<name>     'claude-haiku' (default) | 'claude-sonnet' | 'claude-opus' | 'llamacpp'
 *   --max-iter=<n>   iteration cap (default 30)
 *   --think          llama.cpp only: use the thinking-on profile (qwen-9b-16k-think)
 *   --no-cache       Claude only: disable prompt caching (to A/B the cost win)
 *   --fallback       Claude only: wrap in a Claude→local-Qwen fallback chain
 *
 * Needs docker (ParadeDB) + llama-swap (embeddings + rerank, and the chat model
 * if --llm=llamacpp) up.
 */
import 'dotenv/config';
import pg from 'pg';
import pgvector from 'pgvector/pg';

import { createLangfuseTracer, flushLangfuse, LLAMACPP_MODELS } from '../../lib/index.ts';
import { type AgentLLM, runAgent } from './index.ts';

const argv = process.argv.slice(2);
const flags: Record<string, string> = {};
const positional: string[] = [];
for (const arg of argv) {
  const kv = /^--([^=]+)=(.+)$/.exec(arg);
  if (kv?.[1] && kv[2]) flags[kv[1]] = kv[2];
  else if (arg.startsWith('--')) flags[arg.slice(2)] = '';
  else positional.push(arg);
}
const question = positional.join(' ').trim();
if (!question) {
  console.error(
    'Usage: pnpm dev roman-research/agent/cli.ts "<question>" [--llm=...] [--max-iter=30] [--think]',
  );
  process.exit(1);
}

const llm = (flags.llm ?? 'claude-haiku') as AgentLLM;
const maxIterations = Number(flags['max-iter'] ?? '30');
const llamacppModel = 'think' in flags ? LLAMACPP_MODELS.qwen9b16kThink : LLAMACPP_MODELS.qwen9b64k;

const databaseUrl = process.env.DATABASE_URL;
if (!databaseUrl) throw new Error('DATABASE_URL not set. Check .env and `docker compose up -d`.');
const db = new pg.Client({ connectionString: databaseUrl });
await db.connect();
await pgvector.registerType(db);

console.log(`\n=== Roman Research Agent ===`);
console.log(`Q:   ${question}`);
console.log(`LLM: ${llm}  |  max-iter: ${maxIterations}\n`);
console.log('--- ReAct trace ---');

// One Langfuse trace for this run (no-op if keys absent). Pass its tracer so the
// loop nests a generation per round-trip + a span per tool call.
const lf = createLangfuseTracer({
  name: `agent:cli:${question.slice(0, 60)}`,
  input: question,
  metadata: { llm, maxIterations },
  tags: ['agent', 'cli', llm],
});
const result = await runAgent(db, question, {
  llm,
  maxIterations,
  llamacppModel,
  cache: !('no-cache' in flags),
  fallback: 'fallback' in flags,
  tracer: lf.tracer,
  // Live tool-call log: the multi-step behavior we actually want to watch.
  onStep: (step) => {
    const input = JSON.stringify(step.toolInput);
    const preview = step.toolOutput.replace(/\s+/g, ' ').slice(0, 80);
    console.log(`[iter ${step.iteration}] ${step.toolName}(${input})`);
    console.log(`           → ${preview}…`);
  },
});
lf.end(result.article, { stop: result.stop, toolCalls: result.toolCalls });

console.log(`\n--- Article (${result.llmLabel}) ---`);
console.log(result.article || '(empty — loop did not finalize)');

console.log(`\n--- Stats ---`);
console.log(`Stop:       ${result.stop}`);
console.log(`Tool calls: ${result.toolCalls}  ${JSON.stringify(result.toolCallsByName)}`);
console.log(
  `Tokens:     in=${result.usage.inputTokens}, out=${result.usage.outputTokens}  |  ${result.costFormatted}  |  ${result.latencyMs}ms`,
);
console.log(
  `Cache:      write=${result.usage.cacheCreationTokens ?? 0}, read=${result.usage.cacheReadTokens ?? 0} tokens`,
);

await db.end();
// Flush queued Langfuse events before exit (no-op when tracing is disabled).
await flushLangfuse();
if (lf.active) console.log('\n(Langfuse trace sent.)');
