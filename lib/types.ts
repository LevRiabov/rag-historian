/**
 * lib/types.ts — shared types across the LLM wrapper.
 *
 * These shapes sit ONE LEVEL ABOVE the provider SDKs so callers (mini-projects,
 * the future agent) don't need to import @anthropic-ai/sdk or openai directly.
 * Each provider client returns these normalized shapes and exposes the raw SDK
 * response as `.raw` for advanced use.
 *
 * Why a separate types file: tools.ts, cost.ts, claude.ts, and lmstudio.ts all
 * need to reference the same shapes. Putting them here avoids circular imports.
 */
import type { z } from 'zod';

/** Chat-message roles. Anthropic puts `system` on a separate top-level param;
 *  OpenAI puts it inside the messages array. The wrapper normalizes this away. */
export type Role = 'user' | 'assistant' | 'system';

/** A message at our API layer — content is plain text. Content blocks
 *  (text + tool_use + tool_result) are an internal concern of each provider. */
export interface Message {
  role: Role;
  content: string;
}

/**
 * Token usage as reported by the provider.
 *
 * Anthropic uniquely reports cache_creation and cache_read counts; OpenAI does
 * not (LM Studio runs locally so caching isn't a server-side concept anyway).
 * Cache rates are what make Module 6.4 contextual retrieval economically
 * viable — cache reads cost ~10× less than fresh input.
 */
export interface Usage {
  inputTokens: number;
  outputTokens: number;
  /** Tokens billed at the cache-write rate (~1.25× input). Anthropic only. */
  cacheCreationTokens?: number;
  /** Tokens billed at the cache-read rate (~0.1× input). Anthropic only. */
  cacheReadTokens?: number;
}

/** Cost in USD. Broken out so the pricing source is auditable. */
export interface Cost {
  inputUSD: number;
  outputUSD: number;
  cacheCreationUSD: number;
  cacheReadUSD: number;
  totalUSD: number;
}

/**
 * Normalized stop reasons across both providers.
 *   - 'end_turn'      — model finished naturally
 *   - 'max_tokens'    — hit the max_tokens budget (response was truncated)
 *   - 'tool_use'      — model wants a tool called (you should be in runTools)
 *   - 'stop_sequence' — hit a caller-configured stop sequence
 *   - 'unknown'       — anything else (rare; check .raw)
 */
export type StopReason = 'end_turn' | 'max_tokens' | 'tool_use' | 'stop_sequence' | 'unknown';

/** Return type of a non-streaming chat call. */
export interface ChatResult {
  text: string;
  /**
   * Model's private chain-of-thought when reasoning was enabled. Undefined when
   * reasoning is off OR the model/runtime didn't return any.
   *
   * Both providers return this separately from `text`:
   *   - Anthropic: as `thinking` content blocks alongside the `text` blocks
   *   - Ollama / LM Studio (OpenAI-compat): in `choice.message.reasoning`
   * Surfacing it lets you see *what changed* between reasoning and no-reasoning
   * runs — otherwise the only visible signal is token counts.
   */
  reasoning?: string;
  usage: Usage;
  cost: Cost;
  stopReason: StopReason;
  /** The raw SDK response — escape hatch when you need provider specifics. */
  raw: unknown;
}

/**
 * A tool definition — the canonical shape callers provide.
 *
 * The Zod schema does triple duty:
 *   1. Generates JSON Schema for the API (so the model sees the contract).
 *   2. Validates the model's arguments at runtime (catches hallucinated input).
 *   3. Types the `execute` function's argument via `z.infer<Schema>`.
 *
 * The generic on Schema lets `defineTool` infer `execute`'s parameter at call
 * sites without manual annotation.
 */
/**
 * The `z.ZodObject<z.ZodRawShape>` bound is the "any object shape" upper bound —
 * it accepts a `z.object({ ... })` of any specific shape while letting us
 * compose a heterogeneous `Tool[]` without per-tool casts. Avoids `any`.
 */
export interface Tool<Schema extends z.ZodObject<z.ZodRawShape> = z.ZodObject<z.ZodRawShape>> {
  name: string;
  description: string;
  schema: Schema;
  execute: (input: z.infer<Schema>) => unknown | Promise<unknown>;
}

/** Reasons the agentic tool loop can stop. */
export type ToolLoopStop = 'final_answer' | 'max_iterations' | 'cost_cap';

/** Per-iteration info surfaced via `runTools`' `onStep` callback. */
export interface ToolStep {
  iteration: number;
  toolName: string;
  toolInput: unknown;
  toolOutput: string;
  usage: Usage;
  cost: Cost;
}

/** Return type of the agentic tool loop. */
export interface ToolLoopResult {
  /** The final assistant message text (the model's answer to the user). */
  text: string;
  /** Aggregated usage across all iterations. */
  usage: Usage;
  /** Aggregated cost. */
  cost: Cost;
  /** Why the loop ended. */
  stop: ToolLoopStop;
  /** Each tool call that happened, in order. */
  steps: ToolStep[];
  /** Full message history (provider-shaped). */
  messages: unknown[];
}

/** Return type of `structured()` — parsed data + accounting. */
export interface StructuredResult<T> {
  data: T;
  /** Chain-of-thought when reasoning was enabled. See `ChatResult.reasoning`. */
  reasoning?: string;
  usage: Usage;
  cost: Cost;
  raw: unknown;
}
