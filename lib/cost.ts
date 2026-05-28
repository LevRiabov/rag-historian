/**
 * lib/cost.ts — pricing data + cost calculation.
 *
 * Why isolated: prices change. Putting them in one file makes "update prices"
 * a single-file diff. `PRICES_AS_OF` is your reminder to refresh.
 * Source of truth: https://docs.anthropic.com/en/docs/about-claude/models
 *
 * Why per-million-tokens: that's the unit providers publish. Storing it the
 * same way avoids off-by-1000 bugs when reading the docs page.
 *
 * Anthropic billing has FOUR rates per model:
 *   - input        — uncached input tokens
 *   - output       — generated tokens
 *   - cache write  — input tokens written into the prompt cache (~1.25× input)
 *   - cache read   — input tokens served from the cache    (~0.1×  input)
 * The cache_read rate is the lever in Module 6.4 (contextual retrieval ingest):
 * after the first write, subsequent reads of the same prefix are 10× cheaper.
 */
import type { Cost, Usage } from './types.ts';

/** Last refreshed against the Anthropic docs. Re-check periodically. */
export const PRICES_AS_OF = '2026-05-26' as const;

interface ModelPricing {
  /** USD per million uncached input tokens. */
  input: number;
  /** USD per million output tokens. */
  output: number;
  /** USD per million cache-write tokens (~1.25× input). */
  cacheWrite: number;
  /** USD per million cache-read tokens (~0.1× input). */
  cacheRead: number;
}

/**
 * Friendly names for Claude models. Use these instead of typing the raw model
 * ID — autocomplete works on both halves (`CLAUDE_MODELS.haiku` AND the
 * resulting string literal).
 *
 * Single source of truth: the `ClaudeModel` type below is derived from this
 * object, so adding a model is a one-line change here and the type updates
 * automatically. Make sure to also add its pricing to ANTHROPIC_PRICES below.
 */
export const CLAUDE_MODELS = {
  /** Frontier — best reasoning, deepest agent ability; slowest, priciest. */
  opus: 'claude-opus-4-7',
  /** Workhorse — strong general intelligence, ~5× cheaper than Opus. */
  sonnet: 'claude-sonnet-4-6',
  /** Tiny — fast and cheap; great for routing, classification, simple tool calls. */
  haiku: 'claude-haiku-4-5-20251001',
} as const;

/**
 * Union of all valid Claude model IDs. Derived from CLAUDE_MODELS so the two
 * can't drift. Use this in API signatures wherever you accept a model name —
 * the compiler will reject typos.
 */
export type ClaudeModel = (typeof CLAUDE_MODELS)[keyof typeof CLAUDE_MODELS];

/**
 * Pricing table. Keys are the raw model IDs (same strings as `ClaudeModel`).
 *
 * Typed as `Record<string, ...>` (not `Record<ClaudeModel, ...>`) on purpose —
 * lets `calculateAnthropicCost` accept a plain string at runtime and return
 * ZERO_COST for unknown models (e.g., a brand-new release not yet in the
 * table) rather than fighting the type system at the call site.
 * The `satisfies` clause still enforces that every entry IS a valid model.
 */
export const ANTHROPIC_PRICES: Record<string, ModelPricing> = {
  [CLAUDE_MODELS.opus]: { input: 15, output: 75, cacheWrite: 18.75, cacheRead: 1.5 },
  [CLAUDE_MODELS.sonnet]: { input: 3, output: 15, cacheWrite: 3.75, cacheRead: 0.3 },
  [CLAUDE_MODELS.haiku]: { input: 1, output: 5, cacheWrite: 1.25, cacheRead: 0.1 },
} satisfies Record<ClaudeModel, ModelPricing>;

/**
 * LM Studio model identifiers (matches the user's loaded models).
 *
 * LM Studio uses `org/model-name` format — passing the bare model name causes
 * it to look up a *different* copy and waste VRAM loading a duplicate. Use
 * these constants to stay aligned with what's actually in the runtime.
 *
 * Reasoning capability varies by family:
 *   - gpt-oss-20b → leveled (`'low'|'medium'|'high'`) via `reasoning_effort`
 *   - Gemma 4 / Qwen3 → boolean via `chat_template_kwargs.enable_thinking`
 *
 * Our wrapper's `reasoning` opt accepts BOTH shapes; the lmstudio client
 * translates appropriately.
 */
export const LM_STUDIO_MODELS = {
  /** OpenAI's open-source 20B; tool-trained; honors `reasoning_effort` levels. */
  gptOss20b: 'openai/gpt-oss-20b',
  /** Google Gemma 4 efficient-4B. Boolean reasoning. */
  gemma4_4b: 'google/gemma-4-e4b',
  /** Google Gemma 4 mid-tier (26B-a4b variant). Boolean reasoning. */
  gemma4_26b: 'google/gemma-4-26b-a4b',
  /** Qwen 3.6 mid-tier (35B-a3b variant). Boolean reasoning. */
  qwen3_35b: 'qwen/qwen3.6-35b-a3b',
} as const;

/** Union of known LM Studio model IDs. Not enforced as a type bound on
 *  config (LM Studio is open-ended — any GGUF works), but useful for the
 *  models we routinely test against. */
export type LMStudioModel = (typeof LM_STUDIO_MODELS)[keyof typeof LM_STUDIO_MODELS];

/** LM Studio is local — zero $ cost regardless of model. */
export const LMSTUDIO_PRICING: ModelPricing = {
  input: 0,
  output: 0,
  cacheWrite: 0,
  cacheRead: 0,
};

const ZERO_COST: Cost = {
  inputUSD: 0,
  outputUSD: 0,
  cacheCreationUSD: 0,
  cacheReadUSD: 0,
  totalUSD: 0,
};

/**
 * USD cost from usage + model name.
 *
 * Unknown models return zero rather than throwing — cost tracking is
 * observational, not load-bearing for the call to succeed. If you see $0
 * unexpectedly, it means the model isn't in the pricing table.
 */
export function calculateAnthropicCost(usage: Usage, model: string): Cost {
  // Cost tracking is observational; unknown models return ZERO_COST instead
  // of throwing — a new model release shouldn't break the call path.
  const pricing = ANTHROPIC_PRICES[model];
  if (!pricing) return ZERO_COST;
  return applyPricing(usage, pricing);
}

/** LM Studio variant — always zero, but kept symmetric for tracer plumbing. */
export function calculateLMStudioCost(usage: Usage): Cost {
  return applyPricing(usage, LMSTUDIO_PRICING);
}

function applyPricing(usage: Usage, p: ModelPricing): Cost {
  const M = 1_000_000;
  const inputUSD = (usage.inputTokens * p.input) / M;
  const outputUSD = (usage.outputTokens * p.output) / M;
  const cacheCreationUSD = ((usage.cacheCreationTokens ?? 0) * p.cacheWrite) / M;
  const cacheReadUSD = ((usage.cacheReadTokens ?? 0) * p.cacheRead) / M;
  return {
    inputUSD,
    outputUSD,
    cacheCreationUSD,
    cacheReadUSD,
    totalUSD: inputUSD + outputUSD + cacheCreationUSD + cacheReadUSD,
  };
}

/** Sum two Cost records (used in the agentic loop to accumulate across turns). */
export function addCost(a: Cost, b: Cost): Cost {
  return {
    inputUSD: a.inputUSD + b.inputUSD,
    outputUSD: a.outputUSD + b.outputUSD,
    cacheCreationUSD: a.cacheCreationUSD + b.cacheCreationUSD,
    cacheReadUSD: a.cacheReadUSD + b.cacheReadUSD,
    totalUSD: a.totalUSD + b.totalUSD,
  };
}

/** Sum two Usage records. */
export function addUsage(a: Usage, b: Usage): Usage {
  return {
    inputTokens: a.inputTokens + b.inputTokens,
    outputTokens: a.outputTokens + b.outputTokens,
    cacheCreationTokens: (a.cacheCreationTokens ?? 0) + (b.cacheCreationTokens ?? 0),
    cacheReadTokens: (a.cacheReadTokens ?? 0) + (b.cacheReadTokens ?? 0),
  };
}

/** Human-readable cost summary. 6 decimal places — enough for fractions of a tenth of a cent. */
export function formatCost(cost: Cost): string {
  return `$${cost.totalUSD.toFixed(6)} (in $${cost.inputUSD.toFixed(6)}, out $${cost.outputUSD.toFixed(6)})`;
}
