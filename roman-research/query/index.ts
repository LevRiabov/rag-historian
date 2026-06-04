/**
 * roman-research/query/index.ts — naive RAG CLI.
 *
 * Run:
 *   pnpm dev roman-research/query/index.ts "Who assassinated Julius Caesar?"
 *   pnpm dev roman-research/query/index.ts "Did Caesar want to be king?" --k=8
 *   pnpm dev roman-research/query/index.ts "What happened at Pharsalus?" --embedder=openai
 *   pnpm dev roman-research/query/index.ts "Why did Caesar cross the Rubicon?" --llm=claude-sonnet
 *
 * Flags:
 *   --k=<n>             top-K chunks to retrieve (default 5)
 *   --embedder=<name>   'llamacpp' (BGE-M3, default) or 'openai' (text-embedding-3-small)
 *   --llm=<name>        'llamacpp' (default, free, local qwen-9b-16k),
 *                       'claude-sonnet' | 'claude-haiku' | 'claude-opus'
 *
 * This is the Module 4 deliverable: naive RAG end-to-end, no eval yet.
 * Modules 5–6 will measure and iterate; this is the baseline they iterate
 * against. The "naive" matters: structure-aware chunking, hybrid search,
 * reranking, contextual retrieval, query rewriting are all OFF here on
 * purpose so we know where the floor is.
 */
import 'dotenv/config';
import pg from 'pg';
import pgvector from 'pgvector/pg';

import type { EmbeddingProvider } from '../../lib/index.ts';
import { answerQuestion, DEFAULT_LLM, type LLMChoice } from './answer.ts';
import { expandToParents, formatCitation, type RetrievedChunk, retrieve } from './retrieve.ts';

// ---------------------------------------------------------------------------
// CLI parsing — minimal, no library. Flags are `--key=value`.
// ---------------------------------------------------------------------------
const args = process.argv.slice(2);
const flags: Record<string, string> = {};
const positional: string[] = [];
for (const arg of args) {
  const kv = /^--([^=]+)=(.+)$/.exec(arg);
  if (kv?.[1] && kv[2]) {
    flags[kv[1]] = kv[2];
    continue;
  }
  // Bare boolean flag, e.g. `--hybrid` → empty string marks it "set".
  const bare = /^--(.+)$/.exec(arg);
  if (bare?.[1]) {
    flags[bare[1]] = '';
    continue;
  }
  positional.push(arg);
}
const question = positional.join(' ').trim();
if (!question) {
  console.error(
    'Usage: pnpm dev roman-research/query/index.ts "<question>" [--k=5] [--embedder=llamacpp|openai] [--llm=llamacpp|claude-sonnet|claude-haiku|claude-opus]',
  );
  process.exit(1);
}

const VALID_LLMS: LLMChoice[] = ['llamacpp', 'claude-sonnet', 'claude-haiku', 'claude-opus'];

const topK = Number(flags.k ?? '5');
const embedder = (flags.embedder ?? 'llamacpp') as EmbeddingProvider;
const chunkingVersion = flags['chunking-version'] ?? 'naive-v1';
const retrievalMode: 'vector' | 'hybrid' = flags.hybrid !== undefined ? 'hybrid' : 'vector';
const llmFlag = flags.llm ?? DEFAULT_LLM;
if (!VALID_LLMS.includes(llmFlag as LLMChoice)) {
  console.error(`Unknown --llm value: ${llmFlag}. Valid: ${VALID_LLMS.join(', ')}`);
  process.exit(1);
}
const llm = llmFlag as LLMChoice;

// ---------------------------------------------------------------------------
// DB setup
// ---------------------------------------------------------------------------
const databaseUrl = process.env.DATABASE_URL;
if (!databaseUrl) {
  throw new Error('DATABASE_URL is not set. Check .env and `docker compose up -d`.');
}
const db = new pg.Client({ connectionString: databaseUrl });
await db.connect();
await pgvector.registerType(db);

console.log(`\n=== Roman Research Agent ===`);
console.log(`Q:        ${question}`);
console.log(
  `Embedder: ${embedder}  |  top-K: ${topK}  |  LLM: ${llm}  |  chunking: ${chunkingVersion}  |  mode: ${retrievalMode}\n`,
);

// ---------------------------------------------------------------------------
// Retrieve
// ---------------------------------------------------------------------------
const tRetrieve = Date.now();
const chunks: RetrievedChunk[] = await retrieve(db, question, {
  topK,
  provider: embedder,
  chunkingVersion,
  mode: retrievalMode,
});
console.log(`Retrieved ${chunks.length} chunks in ${Date.now() - tRetrieve}ms:\n`);

for (let i = 0; i < chunks.length; i++) {
  const c = chunks[i];
  if (!c) continue;
  const preview = c.text.replace(/\s+/g, ' ').slice(0, 100);
  console.log(`  [${i + 1}] ${c.similarity.toFixed(3)}  ${formatCitation(c)}`);
  console.log(`        "${preview}…"`);
}
console.log();

// ---------------------------------------------------------------------------
// Answer
// ---------------------------------------------------------------------------
console.log(`Generating answer...\n`);
// parent-child-v1: expand retrieved children to their parent sections before
// generation. No-op for flat variants.
const answer = await answerQuestion(question, expandToParents(chunks), { llm });
console.log(`--- Answer (${answer.llmLabel}) ---`);
console.log(answer.text);
console.log();

// ---------------------------------------------------------------------------
// Citations footer + cost summary
// ---------------------------------------------------------------------------
console.log(`--- Citations ---`);
for (let i = 0; i < chunks.length; i++) {
  const c = chunks[i];
  if (!c) continue;
  console.log(`  [${i + 1}] ${formatCitation(c)} (similarity ${c.similarity.toFixed(3)})`);
}

console.log(`\n--- Stats ---`);
console.log(
  `Tokens: in=${answer.inputTokens}, out=${answer.outputTokens}  |  ${answer.costFormatted}  |  ${answer.latencyMs}ms generation`,
);

await db.end();
