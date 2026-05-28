/**
 * lib/tracer.ts — observability hook seam.
 *
 * Why design this NOW even though Langfuse / OTel are deferred:
 * adding tracing later shouldn't require touching every call site. By routing
 * through a `Tracer` interface from day one, "wire up Langfuse" later is a
 * ~30-line drop-in: write a langfuseTracer that implements these hooks, pass
 * it to createClaude / createLMStudio. Same shape works for OTel.
 *
 * The hooks are intentionally coarse:
 *   - onRequest / onResponse — one per API round-trip
 *   - onToolCall             — one per tool execution in the agentic loop
 *   - onError                — anything we caught
 * Finer-grained tracing (per-token, per-content-block) belongs in a streaming
 * tracer we'll add only if/when needed.
 */
import type { Cost, Usage } from './types.ts';

export interface RequestInfo {
  /**
   * Which provider the request went through.
   *   - 'anthropic' — Claude via the Anthropic SDK
   *   - 'openai'    — OpenAI proper (embeddings in Module 3, frontier chat later)
   *   - 'lmstudio'  — LM Studio AND Ollama via the OpenAI-compatible endpoint
   *                   (chat / streamText / stream / runTools / embeddings)
   *   - 'ollama'    — Ollama's NATIVE /api/chat endpoint, used by
   *                   `createOllama().structured()` to get reliable grammar-
   *                   constrained output (the OpenAI-compat layer's renderers
   *                   don't always engage GBNF on every model).
   */
  provider: 'anthropic' | 'openai' | 'lmstudio' | 'ollama';
  model: string;
  /** Label like 'chat' / 'streamText' / 'runTools' / 'structured'. */
  operation: string;
  messageCount: number;
  toolCount?: number;
}

export interface ResponseInfo {
  usage: Usage;
  cost: Cost;
  /** ms from request start to response end. */
  latencyMs: number;
  /** The provider's raw stop reason string (not the normalized enum). */
  stopReason: string;
}

export interface ToolCallInfo {
  iteration: number;
  toolName: string;
  toolInput: unknown;
  toolOutput: string;
  /** ms spent inside the tool's execute(). */
  latencyMs: number;
}

/**
 * The tracer interface. All methods optional — implement only what you need.
 * The wrapper guards each call so a missing or throwing hook can't crash the
 * request path.
 */
export interface Tracer {
  onRequest?: (info: RequestInfo) => void;
  onResponse?: (info: ResponseInfo) => void;
  onToolCall?: (info: ToolCallInfo) => void;
  onError?: (err: unknown, ctx: { operation: string }) => void;
}

/** Default no-op tracer. */
export const noopTracer: Tracer = {};

/**
 * Safely invoke a tracer hook. Observability code MUST NOT take down the
 * request path — try/catch enforces that contract.
 */
export function safeCall(tracer: Tracer, hook: keyof Tracer, ...args: unknown[]): void {
  const fn = tracer[hook];
  if (typeof fn !== 'function') return;
  try {
    (fn as (...a: unknown[]) => void)(...args);
  } catch {
    // Swallowed by design — observability bugs must not affect requests.
  }
}
