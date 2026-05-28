/**
 * lib/embeddings.ts — unified embedding interface over OpenAI and LM Studio.
 *
 * Both backends speak the same `/v1/embeddings` HTTP API, so we use the same
 * `openai` SDK for both — only `baseURL`, `apiKey`, and `model` differ. This
 * file exists to centralize three concerns that are easy to get wrong:
 *
 *   1. **Dimension awareness.** pgvector columns are dimension-typed. A
 *      1024-dim vector cannot go into a VECTOR(1536) column, and the error
 *      surfaces only at INSERT time. The embedder exposes `.dimension`
 *      synchronously so callers can sanity-check before any DB round-trip.
 *
 *   2. **Batching.** OpenAI's embeddings endpoint accepts ~2048 inputs per
 *      call; the per-call overhead dominates if you embed one-at-a-time
 *      (100× the wall clock). The wrapper transparently splits a 500-text
 *      call into N sub-calls of `batchSize` each and concatenates results
 *      in order. The output vectors are in the SAME order as the inputs.
 *
 *   3. **Provider lock-in awareness.** This is the Module 3 pitfall: once
 *      you've embedded the corpus with model X, switching to Y means
 *      re-embedding everything. The factory is deliberately NOT env-driven
 *      (unlike createLocalLLM) — embedder choice is a per-stack decision
 *      that should be explicit at the call site, not flipped with a shell
 *      var. Use two embedders if you want A/B comparison; the `chunks` table
 *      has two columns precisely for this.
 *
 * Surface (same shape regardless of backend):
 *
 *   const embedder = createEmbedder({ provider: 'openai' });
 *   const result   = await embedder.embed(['hello', 'world']);
 *   // → { vectors: number[][], usage, cost, latencyMs }
 *
 * Cost: OpenAI is metered ($0.02 / 1M tokens for -small). LM Studio is free.
 * Both report usage so the math stays uniform regardless of provider.
 */
import OpenAI from 'openai';

import {
  calculateLMStudioEmbeddingCost,
  calculateOpenAIEmbeddingCost,
  EMBEDDING_DIMENSIONS,
  LM_STUDIO_EMBEDDING_MODELS,
  OPENAI_EMBEDDING_MODELS,
} from './cost.ts';
import { noopTracer, safeCall, type Tracer } from './tracer.ts';
import type { Cost } from './types.ts';

// ============================================================================
// Configuration
// ============================================================================

export type EmbeddingProvider = 'openai' | 'lmstudio';

interface BaseEmbedderConfig {
  /**
   * Default model identifier for this provider. Defaults:
   *   openai   → text-embedding-3-small
   *   lmstudio → text-embedding-bge-m3
   * Make sure the model is actually loaded in LM Studio before calling.
   */
  defaultModel?: string;
  /**
   * Override the auto-detected vector dimension. Only needed if you're using
   * a model that isn't in `EMBEDDING_DIMENSIONS` AND you want to read
   * `.dimension` synchronously before the first embed call. Otherwise the
   * dimension is discovered from the first response.
   */
  dimension?: number;
  /**
   * Maximum inputs per API call. The wrapper splits larger calls into
   * sequential sub-calls. Defaults: 100 (openai), 32 (lmstudio — local
   * inference is memory-constrained).
   */
  batchSize?: number;
  tracer?: Tracer;
}

export interface OpenAIEmbedderConfig extends BaseEmbedderConfig {
  provider: 'openai';
  /** Defaults to OPENAI_API_KEY env var. */
  apiKey?: string;
}

export interface LMStudioEmbedderConfig extends BaseEmbedderConfig {
  provider: 'lmstudio';
  /** Defaults to LM_STUDIO_BASE_URL env var or http://localhost:1234/v1. */
  baseURL?: string;
}

export type EmbedderConfig = OpenAIEmbedderConfig | LMStudioEmbedderConfig;

export interface EmbedOpts {
  /** Per-call model override. Falls back to the embedder's defaultModel. */
  model?: string;
}

// ============================================================================
// Result shapes
// ============================================================================

export interface EmbedResult {
  /** One vector per input, in input order. */
  vectors: number[][];
  /** Total tokens across all sub-batches. */
  usage: { inputTokens: number };
  /** Total USD across all sub-batches. */
  cost: Cost;
  /** Wall-clock ms across all sub-batches (sequential, not parallel). */
  latencyMs: number;
}

export interface Embedder {
  /** Pretty label like `OpenAI (text-embedding-3-small, 1536d)`. */
  label: string;
  provider: EmbeddingProvider;
  model: string;
  /**
   * Vector dimension. Resolved from `EMBEDDING_DIMENSIONS` or the `dimension`
   * config override. `undefined` only if the model is unknown AND no override
   * was given — in that case the dimension becomes known after the first call.
   */
  dimension: number | undefined;
  embed(texts: string[], opts?: EmbedOpts): Promise<EmbedResult>;
  embedOne(text: string, opts?: EmbedOpts): Promise<number[]>;
}

// ============================================================================
// Factory
// ============================================================================

const ZERO_COST: Cost = {
  inputUSD: 0,
  outputUSD: 0,
  cacheCreationUSD: 0,
  cacheReadUSD: 0,
  totalUSD: 0,
};

export function createEmbedder(config: EmbedderConfig): Embedder {
  const { provider } = config;
  const tracer = config.tracer ?? noopTracer;

  // Resolve provider-specific defaults inside one branch so the rest of the
  // factory is provider-agnostic.
  let sdk: OpenAI;
  let model: string;
  let batchSize: number;
  let calculateCost: (tokens: number, model: string) => Cost;

  if (provider === 'openai') {
    sdk = new OpenAI({
      apiKey: config.apiKey ?? process.env.OPENAI_API_KEY,
    });
    model = config.defaultModel ?? OPENAI_EMBEDDING_MODELS.small;
    batchSize = config.batchSize ?? 100;
    calculateCost = calculateOpenAIEmbeddingCost;
  } else {
    sdk = new OpenAI({
      baseURL: config.baseURL ?? process.env.LM_STUDIO_BASE_URL ?? 'http://localhost:1234/v1',
      // LM Studio ignores API keys but the SDK requires SOMETHING.
      apiKey: 'lm-studio',
    });
    model = config.defaultModel ?? LM_STUDIO_EMBEDDING_MODELS.bgeM3;
    batchSize = config.batchSize ?? 32;
    calculateCost = (tokens) => calculateLMStudioEmbeddingCost(tokens);
  }

  // Mutable so we can populate after the first response for unknown models.
  // Closed over by the embed() function below.
  let dimension: number | undefined = config.dimension ?? EMBEDDING_DIMENSIONS[model];

  // --------------------------------------------------------------------------
  // embed — batched
  // --------------------------------------------------------------------------
  /**
   * Embed N texts → N vectors, preserving input order.
   *
   * Sub-batches run sequentially (not parallel). Parallel would speed up
   * large corpora but rate-limit handling becomes nontrivial — Module 9
   * is the right place for that. Until then, sequential keeps the math
   * (latency, cost) clean and predictable.
   */
  async function embed(texts: string[], opts: EmbedOpts = {}): Promise<EmbedResult> {
    const useModel = opts.model ?? model;
    if (texts.length === 0) {
      return { vectors: [], usage: { inputTokens: 0 }, cost: ZERO_COST, latencyMs: 0 };
    }

    const t0 = Date.now();
    const allVectors: number[][] = [];
    let totalTokens = 0;

    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);

      safeCall(tracer, 'onRequest', {
        provider,
        model: useModel,
        operation: 'embed',
        messageCount: batch.length,
      });

      const tBatch = Date.now();
      try {
        const response = await sdk.embeddings.create({
          model: useModel,
          input: batch,
          // Pin to 'float' explicitly. The OpenAI SDK v6+ defaults to 'base64'
          // for wire efficiency and decodes transparently for OpenAI proper —
          // but LM Studio's base64 encoding does NOT round-trip through the
          // SDK's decoder. The visible symptom: a 1024-dim BGE-M3 vector
          // comes back as a 256-float array of zeros. Forcing 'float' returns
          // plain JSON arrays of floats from both backends, slightly more
          // verbose on the wire but provider-agnostic and correct.
          encoding_format: 'float',
        });

        // OpenAI's docs say `.data` is returned in input order, but the
        // contract also exposes `.index` per entry. Sort by index to be safe —
        // if order is already correct this is a no-op.
        const ordered = [...response.data].sort((a, b) => a.index - b.index);
        for (const item of ordered) {
          allVectors.push(item.embedding);
        }

        // Reconcile dimension against the actual response — ALWAYS, not just
        // when undefined. The EMBEDDING_DIMENSIONS lookup is an optimization
        // for sync access before the first call; the response is ground truth.
        // If they disagree, warn loudly (likely cause: wrong model loaded in
        // LM Studio, or model was updated to a different dim).
        const firstVec = ordered[0]?.embedding;
        if (firstVec) {
          const actualDim = firstVec.length;
          if (dimension !== undefined && dimension !== actualDim) {
            console.warn(
              `[embedder] ${useModel}: expected ${dimension}-dim from lookup, ` +
                `got ${actualDim}-dim from response. Trusting the response.`,
            );
          }
          dimension = actualDim;
        }

        const batchTokens = response.usage?.prompt_tokens ?? 0;
        totalTokens += batchTokens;

        safeCall(tracer, 'onResponse', {
          usage: { inputTokens: batchTokens, outputTokens: 0 },
          cost: calculateCost(batchTokens, useModel),
          latencyMs: Date.now() - tBatch,
          stopReason: 'end_turn',
        });
      } catch (err) {
        safeCall(tracer, 'onError', err, { operation: 'embed' });
        throw err;
      }
    }

    return {
      vectors: allVectors,
      usage: { inputTokens: totalTokens },
      cost: calculateCost(totalTokens, useModel),
      latencyMs: Date.now() - t0,
    };
  }

  async function embedOne(text: string, opts?: EmbedOpts): Promise<number[]> {
    const result = await embed([text], opts);
    const vec = result.vectors[0];
    if (!vec) throw new Error('Embedding API returned no vector.');
    return vec;
  }

  return {
    get label() {
      const dimStr = dimension !== undefined ? `, ${dimension}d` : '';
      const providerLabel = provider === 'openai' ? 'OpenAI' : 'LM Studio';
      return `${providerLabel} (${model}${dimStr})`;
    },
    provider,
    model,
    get dimension() {
      return dimension;
    },
    embed,
    embedOne,
  };
}
