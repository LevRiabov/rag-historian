/**
 * lib/fallback.ts — cross-provider fallback chain (Module 9).
 *
 * SDK-level retries (429 / 5xx with exponential backoff) already live INSIDE each
 * client — Anthropic's `maxRetries`, the OpenAI SDK's equivalent. Reimplementing
 * them here would only diverge. What a single SDK can't do is fail OVER to a
 * different provider, so that's all this adds: a chain of already-configured
 * clients where, when one still throws after its own retries (provider outage,
 * bad/blocked API key, exhausted backoff), the next client takes the call.
 *
 * The Roman agent wires it as Claude → local Qwen: if Anthropic is unreachable or
 * the key is bad, the question is still answered by the local model. This is an
 * AVAILABILITY pattern, not a quality one — falling back to Qwen changes answer
 * quality to qwen-level (Module 7: agent completeness 4.04 → 3.38). The point is
 * "still returns an answer", not "returns the same answer".
 *
 * The wrapper exposes the SAME method surface (runTools / chat / structured) as
 * the underlying clients, so it drops into `runAgent` transparently. Each tier is
 * a client ALREADY configured with its own default model — the chain never passes
 * one provider's model id to another (a ClaudeModel must never reach llama.cpp).
 */
import type { z } from 'zod';

import type { ChatCallOpts, RunToolsOpts, StructuredOpts } from './claude.ts';
import { noopTracer, safeCall, type Tracer } from './tracer.ts';
import type { ChatResult, StructuredResult, ToolLoopResult } from './types.ts';

/** The subset of a client's surface the fallback chain forwards. `createClaude`
 *  and `createLlamacpp`/`createLMStudio` both satisfy this structurally. */
export interface FallbackCapableClient {
  runTools(opts: RunToolsOpts): Promise<ToolLoopResult>;
  chat(opts: ChatCallOpts): Promise<ChatResult>;
  structured<Schema extends z.ZodObject<z.ZodRawShape>>(
    opts: StructuredOpts<Schema>,
  ): Promise<StructuredResult<z.infer<Schema>>>;
}

export interface FallbackTier {
  /** A client pre-configured with its own default model. */
  client: FallbackCapableClient;
  /** Human label for logs/traces, e.g. 'claude-haiku' or 'llamacpp-qwen'. */
  label: string;
}

export interface FallbackConfig {
  /** Ordered tiers: index 0 is primary, the rest are fallbacks. Non-empty. */
  chain: FallbackTier[];
  /** Observability hook — a fallback hop is reported via onError so it shows up
   *  in Langfuse / logs even though the overall call ultimately succeeds. */
  tracer?: Tracer;
  /** Should this error advance to the next tier? Default: any throw does. Narrow
   *  it to, say, keep client-side validation (4xx) errors from wasting a fallback. */
  shouldFallback?: (err: unknown) => boolean;
}

function errMsg(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/**
 * Build a client that tries each tier in order. Same surface as the wrapped
 * clients, so callers can't tell they're talking to a chain.
 */
export function createFallbackClient(config: FallbackConfig): FallbackCapableClient {
  const { chain } = config;
  if (chain.length === 0) throw new Error('createFallbackClient: chain must be non-empty.');
  const tracer = config.tracer ?? noopTracer;
  const shouldFallback = config.shouldFallback ?? (() => true);

  // One generic driver for all three methods: run `fn` against each tier's client,
  // falling through to the next on a fallback-eligible error.
  async function runChain<T>(
    operation: string,
    fn: (client: FallbackCapableClient) => Promise<T>,
  ): Promise<T> {
    let lastErr: unknown;
    for (let i = 0; i < chain.length; i++) {
      const tier = chain[i];
      if (!tier) continue;
      try {
        return await fn(tier.client);
      } catch (err) {
        lastErr = err;
        safeCall(tracer, 'onError', err, { operation: `fallback:${operation}:${tier.label}` });
        const next = chain[i + 1];
        if (!next || !shouldFallback(err)) throw err;
        // stderr, never stdout — stdout may be a JSON-RPC channel (MCP).
        console.error(
          `[fallback] ${operation}: tier "${tier.label}" failed (${errMsg(err)}) → trying "${next.label}"`,
        );
      }
    }
    throw lastErr; // unreachable while chain is non-empty
  }

  return {
    runTools: (opts) => runChain('runTools', (c) => c.runTools(opts)),
    chat: (opts) => runChain('chat', (c) => c.chat(opts)),
    structured: <Schema extends z.ZodObject<z.ZodRawShape>>(opts: StructuredOpts<Schema>) =>
      runChain('structured', (c) => c.structured(opts)),
  };
}
