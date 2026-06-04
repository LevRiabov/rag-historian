/**
 * roman-research/query/expand.ts — query expansion / multi-query (Module 6.5).
 *
 * The ADDITIVE half of query rewriting, and the structural opposite of HyDE.
 * Where HyDE *replaces* the question with a hallucinated answer (and so can lose
 * — it throws away the question's own discriminative terms; measured to crater
 * synonym recall, see notes/module-6-advanced-rag.md), expansion *keeps* the
 * original question and ADDS rephrasings alongside it. The retriever runs every
 * query, and the candidate pools are unioned. So expansion can only widen
 * coverage, never delete signal.
 *
 * Why it targets our actual weak spots:
 *   - synonym  — each variation uses different vocabulary, so at least one is
 *     likely to match the source's exact wording the question missed.
 *   - multi-hop — the answer is spread across passages; different phrasings net
 *     different facets, and the union covers more of the spread.
 *
 * The original question is ALWAYS query #0 in the multi-query set (see
 * retrieveMultiQuery), so a variation that drifts can't pull the pool away from
 * what the plain question already finds — drift only costs a slot, never the
 * baseline hit. A reranker over the wider, more diverse union is the natural
 * backstop: more true positives in the pool for it to lift to the top.
 *
 * Cost: one LLM call per query (like HyDE) — a per-query latency tax. Free here
 * (local qwen, thinking off); measured in the eval harness.
 */
import { z } from 'zod';

import { createLlamacpp, LLAMACPP_MODELS } from '../../lib/index.ts';

const VariationsSchema = z.object({
  /** The rephrasings — NOT including the original (we add that back ourselves). */
  variations: z.array(z.string()),
});

/**
 * Persona + intent. We ask for genuine REPHRASINGS (synonyms, varied
 * specificity, surfaced implied entities), not answers — the failure mode to
 * avoid is the model "helpfully" answering instead of rewriting. Constrained to
 * the corpus's domain so variations stay on-topic (a wild rephrasing retrieves
 * noise that the union then has to carry).
 */
const EXPAND_SYSTEM_PROMPT = `You rewrite a user's question into alternative search queries for a corpus of ancient histories of Julius Caesar (Caesar's own commentaries, Plutarch, Suetonius, Appian).

Produce rephrasings that mean the same thing but maximize the chance of matching the source text:
- swap in synonyms and period-appropriate vocabulary,
- vary the specificity (one broader, one narrower),
- surface entities the question only implies (names, places, offices, events).

Do NOT answer the question. Do NOT repeat the original verbatim. Return only the rephrasings.`;

export interface ExpandOptions {
  /** How many rephrasings to request (the original is added separately, so the
   *  multi-query set is n+1). Default 3 — enough vocabulary diversity without a
   *  large union pool to rerank. */
  n?: number;
  /** llama-swap profile (default qwen-9b-16k, thinking off). */
  llamacppModel?: string;
}

export interface ExpandResult {
  /** The rephrasings only (caller prepends the original question). */
  variations: string[];
  latencyMs: number;
  inputTokens: number;
  outputTokens: number;
}

/**
 * Generate `n` rephrasings of `question`. The caller builds the multi-query set
 * as `[question, ...variations]` and passes it to `retrieveMultiQuery`.
 */
export async function generateQueryVariations(
  question: string,
  options: ExpandOptions = {},
): Promise<ExpandResult> {
  const { n = 3, llamacppModel } = options;
  const model = llamacppModel ?? LLAMACPP_MODELS.qwen9b16k;
  const client = createLlamacpp({ defaultModel: model });
  const t0 = Date.now();
  const result = await client.structured({
    system: EXPAND_SYSTEM_PROMPT,
    messages: [
      {
        role: 'user',
        content: `Rewrite this question into exactly ${n} alternative search queries:\n\n${question}`,
      },
    ],
    schema: VariationsSchema,
  });
  // Defensive: trim, drop empties, and cap at n in case the model over-produces.
  const variations = result.data.variations
    .map((v) => v.trim())
    .filter((v) => v.length > 0)
    .slice(0, n);
  return {
    variations,
    latencyMs: Date.now() - t0,
    inputTokens: result.usage.inputTokens,
    outputTokens: result.usage.outputTokens,
  };
}
