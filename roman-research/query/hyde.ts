/**
 * roman-research/query/hyde.ts — HyDE (Hypothetical Document Embeddings).
 *
 * The query-side fix for the retrieval geometry gap (Module 6.5). A bi-encoder
 * maps the QUESTION and its ANSWER to different regions of embedding space: a
 * question is short, interrogative, and names nothing concrete ("What factors
 * led to the conspiracy against Caesar?"); the answering passage is long,
 * declarative, and dense with the very entities the question omits (the
 * Lupercalia crown, the perpetual dictatorship, Brutus and Cassius). So the raw
 * question vector lands FAR from the chunks that answer it — worst on abstract
 * "synthesis" questions, which have no entity anchor at all (our retrieval floor:
 * synthesis recall@5 = 22.9%).
 *
 * HyDE closes the gap by asking the LLM to HALLUCINATE an answer first, then
 * embedding THAT instead of the question. The fake answer doesn't need to be
 * factually correct — it needs to be SHAPED like the real answer: declarative,
 * the right length, full of the right *kind* of entities. Even a wrong detail
 * lands the vector in the right neighborhood, because it now "speaks the
 * language of the corpus" instead of the language of questions.
 *
 * Important: HyDE only changes what we EMBED. The original question still drives
 * reranking (a cross-encoder must read the real question to judge relevance) and
 * the BM25 lexical arm (hallucinated terms would poison exact-match). So the
 * generated doc is passed to `retrieve()` as `embedText`, leaving `question`
 * intact for both downstream stages.
 *
 * Cost note: every query now needs an extra LLM call before retrieval. Free here
 * (local qwen, thinking off), but a real per-query latency tax in production —
 * we measure it in the eval harness.
 */
import { createLlamacpp, LLAMACPP_MODELS } from '../../lib/index.ts';

/**
 * Persona + format constraints. The crucial line is "if you are unsure, write
 * the most historically plausible answer" — we WANT a confident hallucination
 * shaped like a source passage, NOT a hedge ("the sources may say…") or a
 * refusal. Hedging produces meta-text about the question, which embeds back near
 * the question (defeating HyDE). We also cap length: the doc should resemble ONE
 * retrieved chunk (~a paragraph), not an essay — a sprawling answer dilutes the
 * vector across many topics and blurs the neighborhood we're aiming for.
 */
const HYDE_SYSTEM_PROMPT = `You are a Roman historian writing about Julius Caesar's career and death.

Given a question, write a single short paragraph (3 to 5 sentences) that answers it as if it were an excerpt from a primary historical source. Write in plain declarative prose. Name concrete people, places, dates, and events. Do NOT hedge, do NOT say "the sources say" or "it is believed" — state the facts directly as a historian would. If you are unsure of a detail, write the most historically plausible answer anyway. Output only the paragraph, no preamble.`;

export interface HydeOptions {
  /** llama-swap profile to generate with. Defaults to qwen-9b-16k (thinking
   *  off — a hypothetical paragraph needs no chain of thought, and thinking
   *  would burn the token budget). A 16k context is ample: input is one short
   *  question, output one paragraph. */
  llamacppModel?: string;
}

export interface HydeResult {
  /** The hypothetical answer paragraph — embed THIS, not the question. */
  doc: string;
  latencyMs: number;
  inputTokens: number;
  outputTokens: number;
}

/**
 * Generate a hypothetical answer document for `question`. Pass the returned
 * `doc` to `retrieve(..., { embedText: doc })` so the first-stage vector search
 * runs on the answer-shaped text while reranking/BM25 keep the real question.
 */
export async function generateHydeDoc(
  question: string,
  options: HydeOptions = {},
): Promise<HydeResult> {
  const model = options.llamacppModel ?? LLAMACPP_MODELS.qwen9b16k;
  const client = createLlamacpp({ defaultModel: model });
  const t0 = Date.now();
  const result = await client.chat({
    system: HYDE_SYSTEM_PROMPT,
    messages: [{ role: 'user', content: `Question: ${question}` }],
  });
  return {
    doc: result.text.trim(),
    latencyMs: Date.now() - t0,
    inputTokens: result.usage.inputTokens,
    outputTokens: result.usage.outputTokens,
  };
}
