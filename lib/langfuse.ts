/**
 * lib/langfuse.ts — Langfuse implementation of the Tracer seam (Module 7).
 *
 * Implements the lib/tracer.ts `Tracer` interface so ANY client (createClaude /
 * createLlamacpp) auto-emits a Langfuse trace TREE per run — no call-site change,
 * exactly what tracer.ts promised ("a ~30-line drop-in").
 *
 * NO-OP when keys are absent: `createLangfuseTracer` returns a noop tracer + inert
 * handle if LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY aren't set, so evals run
 * unchanged without Langfuse and light up the moment keys are added to `.env`.
 *
 * Trace shape per agent run (one createLangfuseTracer() = one trace):
 *   TRACE  agent:<category>:<id>          input=question, output=article
 *   ├─ generation  gen-1 (model, usage, cost, latency)   ← onRequest/onResponse
 *   ├─ span        tool:search_corpus (input → output)    ← onToolCall
 *   ├─ generation  gen-2 ...
 *   └─ scores: faithfulness / completeness / refusal_correct / gold_coverage / tool_calls
 *
 * Interface limitation: the Tracer hooks carry usage/cost/tool-IO but NOT prompt/
 * completion TEXT (RequestInfo/ResponseInfo are coarse by design). So generations
 * show model+usage+cost+timing; the ROOT trace carries the question→article text
 * and the tool spans carry the real research path. Richer per-generation text would
 * need extending the Tracer interface — deferred (not needed for the eval drill-down).
 */
import { Langfuse } from 'langfuse';

import { noopTracer, type Tracer } from './tracer.ts';

// Cache the client across questions: undefined = not yet resolved, null = disabled.
let cachedClient: Langfuse | null | undefined;

function getClient(): Langfuse | null {
  if (cachedClient !== undefined) return cachedClient;
  const publicKey = process.env.LANGFUSE_PUBLIC_KEY;
  const secretKey = process.env.LANGFUSE_SECRET_KEY;
  if (!publicKey || !secretKey) {
    cachedClient = null; // keys absent → tracing disabled (clean no-op)
    return null;
  }
  cachedClient = new Langfuse({ publicKey, secretKey, baseUrl: process.env.LANGFUSE_BASE_URL });
  return cachedClient;
}

export interface LangfuseTraceHandle {
  /** The Tracer to pass to createClaude / createLlamacpp / runAgent. */
  tracer: Tracer;
  /** Attach a numeric score (eval result) to the trace. */
  score(name: string, value: number, comment?: string): void;
  /** Finalize the trace: set its output + summary metadata. */
  end(output?: string, metadata?: Record<string, unknown>): void;
  /** True when a real Langfuse trace is active (keys present). */
  readonly active: boolean;
}

export interface CreateLangfuseTracerOpts {
  /** Trace name, e.g. `agent:contradiction:q-023`. */
  name: string;
  /** Trace input (the question). */
  input?: unknown;
  metadata?: Record<string, unknown>;
  tags?: string[];
}

/**
 * Create a per-run Langfuse trace + a Tracer that nests generations (one per
 * model round-trip) and spans (one per tool call) under it. Returns an inert
 * handle when keys are absent.
 */
export function createLangfuseTracer(opts: CreateLangfuseTracerOpts): LangfuseTraceHandle {
  const client = getClient();
  if (!client) {
    return { tracer: noopTracer, score: () => {}, end: () => {}, active: false };
  }

  const trace = client.trace({
    name: opts.name,
    input: opts.input,
    metadata: opts.metadata,
    tags: opts.tags,
  });

  // The agent loop is sequential, so a single "pending generation" ref pairs each
  // onRequest (open) with its onResponse (close).
  let pendingGen: ReturnType<typeof trace.generation> | null = null;
  let genCount = 0;

  const tracer: Tracer = {
    onRequest(info) {
      genCount += 1;
      pendingGen = trace.generation({
        name: `gen-${genCount} (${info.operation})`,
        model: info.model,
        metadata: {
          provider: info.provider,
          messageCount: info.messageCount,
          toolCount: info.toolCount,
        },
      });
    },
    onResponse(info) {
      if (!pendingGen) return;
      pendingGen.end({
        usage: { input: info.usage.inputTokens, output: info.usage.outputTokens, unit: 'TOKENS' },
        // costDetails (not metadata) is what Langfuse rolls up into trace cost.
        costDetails: {
          input: info.cost.inputUSD,
          output: info.cost.outputUSD,
          total: info.cost.totalUSD,
        },
        metadata: { latencyMs: info.latencyMs, stopReason: info.stopReason },
      });
      pendingGen = null;
    },
    onToolCall(info) {
      trace
        .span({
          name: `tool:${info.toolName}`,
          input: info.toolInput,
          metadata: { iteration: info.iteration, latencyMs: info.latencyMs },
        })
        .end({ output: info.toolOutput });
    },
    onError(err, ctx) {
      trace.event({
        name: 'error',
        level: 'ERROR',
        statusMessage: err instanceof Error ? err.message : String(err),
        metadata: ctx,
      });
    },
  };

  return {
    tracer,
    active: true,
    score(name, value, comment) {
      trace.score({ name, value, comment });
    },
    end(output, metadata) {
      trace.update({ output, metadata });
    },
  };
}

/** Flush queued events to Langfuse (no-op if disabled). Call once at run end. */
export async function flushLangfuse(): Promise<void> {
  const client = getClient();
  if (client) await client.flushAsync();
}

// ============================================================================
// Prompt versioning (Module 9)
// ============================================================================
//
// Langfuse "prompt management": each prompt NAME holds a stack of numbered
// VERSIONS, with movable LABELS (e.g. `production`) pointing at one version.
// Workflow: registerPrompt() tags a new version → getManagedPrompt() fetches the
// `production`-labelled one at runtime → rollback is moving the label in the UI
// (or another registerPrompt with that label). The in-code const is always the
// FALLBACK, so the agent runs unchanged when Langfuse is down or keys are absent —
// same philosophy as the no-op tracer.

export interface ManagedPrompt {
  text: string;
  /** Langfuse version number, or null when served from the in-code fallback. */
  version: number | null;
  source: 'langfuse' | 'fallback';
}

function isFallback(p: unknown): boolean {
  return Boolean((p as { isFallback?: boolean })?.isFallback);
}

/**
 * Fetch a prompt by label at runtime, falling back to `fallbackText` whenever
 * Langfuse is disabled/unreachable or the prompt isn't registered yet. Cached by
 * the SDK (cacheTtlSeconds) so a 50-question eval makes ~one network call.
 */
export async function getManagedPrompt(
  name: string,
  fallbackText: string,
  opts: { label?: string; cacheTtlSeconds?: number } = {},
): Promise<ManagedPrompt> {
  const client = getClient();
  if (!client) return { text: fallbackText, version: null, source: 'fallback' };
  try {
    const p = await client.getPrompt(name, undefined, {
      label: opts.label ?? 'production',
      cacheTtlSeconds: opts.cacheTtlSeconds ?? 60,
      type: 'text',
      fallback: fallbackText,
    });
    if (isFallback(p)) return { text: fallbackText, version: null, source: 'fallback' };
    const text = typeof p.prompt === 'string' ? p.prompt : fallbackText;
    return { text, version: p.version ?? null, source: 'langfuse' };
  } catch {
    return { text: fallbackText, version: null, source: 'fallback' };
  }
}

/**
 * Register a prompt version under the given labels (default `production`). Skips
 * creation when the currently-labelled version is byte-identical, so re-running a
 * register script doesn't spam new versions. Returns whether it created one.
 */
export async function registerPrompt(
  name: string,
  text: string,
  opts: { labels?: string[] } = {},
): Promise<{ created: boolean; version: number | null }> {
  const client = getClient();
  if (!client) return { created: false, version: null };
  const labels = opts.labels ?? ['production'];
  try {
    const current = await client.getPrompt(name, undefined, {
      label: labels[0],
      cacheTtlSeconds: 0,
      type: 'text',
    });
    if (!isFallback(current) && current.prompt === text) {
      return { created: false, version: current.version ?? null };
    }
  } catch {
    // Not found / first registration → fall through to create.
  }
  const created = await client.createPrompt({ name, type: 'text', prompt: text, labels });
  return { created: true, version: created.version ?? null };
}
