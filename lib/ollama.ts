/**
 * lib/ollama.ts — Ollama wrapper.
 *
 * Ollama exposes two HTTP APIs side by side on port 11434:
 *
 *   /v1/chat/completions  — OpenAI-compatible (we use this for chat / runTools
 *                           / streaming via the OpenAI SDK)
 *   /api/chat             — Ollama's NATIVE endpoint (we use this for
 *                           `structured()` ONLY — see below for why)
 *
 * Why two paths instead of one:
 *
 *   The OpenAI-compat layer is a thin adapter. Internally Ollama routes the
 *   request through a per-model RENDERER + PARSER pair (`ollama show <model>`
 *   reveals these). The renderer formats the prompt for the model's native
 *   chat template; the parser demuxes thinking / tool / content channels on
 *   the way back. Critically, the renderer decides whether to attach a GBNF
 *   grammar from the request's `response_format` / `tool_choice`. Some
 *   renderers do this faithfully (gpt-oss:20b via Harmony); some don't
 *   (gemma4 — observed to emit out-of-enum values + ignore tool_choice).
 *
 *   The NATIVE `/api/chat` endpoint takes a JSON schema directly via `format`
 *   and binds it to llama.cpp's grammar sampler uniformly — same code path
 *   for every model. So structured output is reliable there even when the
 *   OpenAI-compat path silently drops the constraint.
 *
 *   GOTCHA — schema property order matters. Ollama's grammar compiler enforces
 *   the order properties appear in the schema's `properties` block AND only
 *   reliably applies enum/format constraints to the FIRST property. If you put
 *   a free-form `string` field before an `enum` field, Gemma will sometimes
 *   emit out-of-enum values for the enum (e.g. copying a phrase from the user
 *   prompt). Put your most-constrained field FIRST in the schema, and match
 *   that order in any few-shot example outputs. LM Studio's GBNF doesn't have
 *   this quirk, which is why the same model is stable there with any order.
 *
 *   `chat()` / `runTools()` etc. stay on the OpenAI-compat path because the
 *   bug only bites grammar-constrained output, and reusing the OpenAI SDK
 *   keeps streaming / tool-loop code in one place.
 *
 * Quick start:
 *   ollama pull gpt-oss:20b           # download the model
 *   ollama list                        # verify it's there
 *   curl http://localhost:11434/v1/models  # confirm API is up
 */
import { z } from 'zod';

import { calculateLMStudioCost } from './cost.ts';
import { createLMStudio, type LMStudioConfig, type LMStudioStructuredOpts } from './lmstudio.ts';
import { noopTracer, safeCall } from './tracer.ts';
import type { StructuredResult, Usage } from './types.ts';

/**
 * Ollama model identifiers — reflects what's currently pulled locally.
 * Add more with `ollama pull <id>` and a new entry here.
 *
 * Capability varies sharply by model — Ollama's chat templates differ in
 * what they implement. Verify with `ollama show <model>` (the `Capabilities`
 * block is authoritative) before assuming a model supports a feature.
 *
 * | Model           | chat | tool_use | json_schema    | reasoning |
 * |-----------------|------|----------|----------------|-----------|
 * | gpt-oss:20b     |  ✅  |    ✅    |     ✅         | levels    |
 * | gemma4:e2b      |  ✅  |    ✅    |  ✅ via native | levels†   |
 * | gemma4:latest   |  ✅  |    ✅    |  ✅ via native | levels†   |
 * | gemma4:26b      |  ✅  |    ✅    |  ✅ via native | levels†   |
 *
 * † `reasoning_effort` on the OpenAI-compat endpoint and `think` on the native
 *   endpoint both work. Gemma's chat template accepts the leveled form
 *   ('low' / 'medium' / 'high') but internally it's a boolean toggle — any
 *   level just turns thinking on. Reasoning text comes back on
 *   `choice.message.reasoning` (compat) or `message.thinking` (native).
 *
 * "json_schema via native" means `createOllama().structured()` automatically
 * goes through Ollama's native `/api/chat` endpoint where grammar binding is
 * uniform across models. The OpenAI-compat path is unreliable for Gemma
 * specifically — its renderer doesn't always attach GBNF when the request
 * has long few-shot examples + thinking enabled.
 *
 * Recommended additions if you want everything through one endpoint:
 *   - `qwen3:8b` — ~5GB, full structured/tool support via OpenAI-compat
 *   - `llama3.3:70b` — heavyweight, all features, needs serious VRAM
 */
export const OLLAMA_MODELS = {
  /** OpenAI's open-source 20B — tool use, leveled reasoning, schema enforcement.
   *  Best general-purpose local model for structured tasks. */
  gptOss20b: 'gpt-oss:20b',
  /** Gemma 4 efficient 2B — tiny, fast, supports tools + thinking. Useful for
   *  capability-floor comparisons (does this task work even on a 2B model?).
   *  Structured output uses Ollama's native path (see capability note above). */
  gemma4_e2b: 'gemma4:e2b',
  /** Gemma 4 default (`latest` tag, ~4B). Tools + thinking; structured via native. */
  gemma4: 'gemma4:latest',
  /** Gemma 4 26B — heavyweight; tools + thinking. Use when you need stronger
   *  reasoning on free-form text. Same structured-via-native path. */
  gemma4_26b: 'gemma4:26b',
} as const;

/** Union of known Ollama model IDs. Not enforced as a type bound — Ollama
 *  accepts any pulled model — but useful for autocomplete on the favorites. */
export type OllamaModel = (typeof OLLAMA_MODELS)[keyof typeof OLLAMA_MODELS];

/** Public client shape: same surface as createLMStudio (callers can swap
 *  freely), but `structured()` is overridden to use the native /api/chat path. */
export type OllamaClient = ReturnType<typeof createLMStudio>;

/**
 * Build an Ollama client. Mostly identical to `createLMStudio` — same methods,
 * same opts — but `structured()` is overridden to go through Ollama's native
 * `/api/chat` endpoint for reliable grammar-constrained output.
 *
 * The base URL is normalized to always end in `/v1` for the OpenAI-compat
 * methods. The native endpoint is derived by stripping the `/v1` suffix.
 * Callers can set `OLLAMA_BASE_URL` to either `http://localhost:11434` (Ollama's
 * native URL) or `http://localhost:11434/v1` — both work.
 */
export function createOllama(config: LMStudioConfig = {}): OllamaClient {
  const rawBase = config.baseURL ?? process.env.OLLAMA_BASE_URL ?? 'http://localhost:11434';
  const v1Base = normalizeOllamaBaseURL(rawBase);
  // Native `/api/chat` lives one level up from `/v1`. Strip the suffix once
  // here so the structured() override doesn't recompute it per call.
  const nativeBase = v1Base.replace(/\/v1$/, '');
  const defaultModel = config.defaultModel ?? OLLAMA_MODELS.gptOss20b;
  const defaultMaxTokens = config.defaultMaxTokens ?? 4096;
  const tracer = config.tracer ?? noopTracer;

  const base = createLMStudio({
    ...config,
    baseURL: v1Base,
    defaultModel,
  });

  /**
   * Override `structured()` to hit Ollama's native /api/chat with `format`.
   *
   * Why not reuse the LMStudio path: the OpenAI-compat layer's renderers don't
   * always engage GBNF for `response_format` (see file header). The native API
   * binds the schema to llama.cpp's grammar sampler directly. Same model, same
   * GGUF, but the constraint actually fires.
   *
   * Stays signature-compatible with `createLMStudio().structured()` so callers
   * (and the abTest harness in 03-classification) don't notice the swap.
   */
  async function structured<Schema extends z.ZodObject<z.ZodRawShape>>(
    opts: LMStudioStructuredOpts<Schema>,
  ): Promise<StructuredResult<z.infer<Schema>>> {
    const model = opts.model ?? defaultModel;

    // Strip `$schema` — Ollama's grammar compiler rejects the JSON Schema
    // dialect marker (and it carries no semantic value for token masking).
    const jsonSchema = z.toJSONSchema(opts.schema) as Record<string, unknown>;
    delete jsonSchema.$schema;

    // Native `/api/chat` takes system as a 'system'-role message in the array,
    // not a top-level param. Convert here so callers don't have to know.
    const messages = opts.system
      ? [{ role: 'system' as const, content: opts.system }, ...opts.messages]
      : opts.messages;

    // Native API options live under `options` (mirroring llama.cpp's sampler
    // args). `num_predict` is the native equivalent of `max_tokens`.
    const samplerOptions: Record<string, unknown> = {};
    if (opts.temperature !== undefined) samplerOptions.temperature = opts.temperature;
    samplerOptions.num_predict = opts.maxTokens ?? defaultMaxTokens;

    // `think` interacts BADLY with `format` on some models (Gemma 4): setting
    // a truthy `think` value silently disables GBNF grammar enforcement AND
    // the thinking channel itself — the model emits unconstrained plain text.
    // gpt-oss:20b is unaffected, but we don't want to thread per-model logic
    // here. Workaround: only forward `think` when the caller is EXPLICITLY
    // turning thinking OFF (`reasoning: false`). For undefined / truthy values
    // we omit `think`, which leaves the model at its own default — every
    // thinking-capable model in OLLAMA_MODELS already thinks by default, so
    // a caller asking for thinking still gets it. Trade-off: callers lose
    // fine-grained level control on the native path. Worth it for stability.
    const body: Record<string, unknown> = {
      model,
      messages: messages.map((m) => ({ role: m.role, content: m.content })),
      stream: false,
      format: jsonSchema,
      options: samplerOptions,
    };
    if (opts.reasoning === false) {
      body.think = false;
    }

    safeCall(tracer, 'onRequest', {
      provider: 'ollama',
      model,
      operation: 'structured',
      messageCount: opts.messages.length,
    });

    const t0 = Date.now();
    let response: Response;
    try {
      response = await fetch(`${nativeBase}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
    } catch (err) {
      safeCall(tracer, 'onError', err, { operation: 'structured' });
      throw err;
    }

    if (!response.ok) {
      // Surface Ollama's error body — it usually identifies the cause clearly
      // (e.g. "model not found", "invalid format schema").
      const errText = await response.text();
      const err = new Error(
        `Ollama native API returned ${response.status} ${response.statusText}: ${errText}`,
      );
      safeCall(tracer, 'onError', err, { operation: 'structured' });
      throw err;
    }

    const raw = (await response.json()) as NativeChatResponse;
    const usage: Usage = {
      inputTokens: raw.prompt_eval_count ?? 0,
      outputTokens: raw.eval_count ?? 0,
    };
    // Local inference — cost is always zero. Reuse the LMStudio cost calc so
    // the shape stays consistent across providers (tracer aggregators expect
    // the full Cost object even when totals are 0).
    const cost = calculateLMStudioCost(usage);

    safeCall(tracer, 'onResponse', {
      usage,
      cost,
      latencyMs: Date.now() - t0,
      // Native API doesn't expose a finish_reason — `done_reason` exists in
      // newer versions but isn't load-bearing for structured output (the
      // grammar guarantees completion). Report a stable string.
      stopReason: raw.done_reason ?? 'stop',
    });

    const text = raw.message?.content ?? '';
    if (!text) {
      throw new Error(
        `Ollama native API returned empty content. done_reason=${raw.done_reason ?? '(none)'}`,
      );
    }

    let parsedJson: unknown;
    try {
      parsedJson = JSON.parse(text);
    } catch (err) {
      // Shouldn't happen with grammar-constrained sampling, but if Ollama's
      // grammar engine ever fails to engage, surface what we got — same
      // failure mode the OpenAI-compat path used to hit, now visibly rare.
      throw new Error(
        `Could not JSON-parse Ollama native response: ${String(err)}\nModel produced: ${text}`,
      );
    }
    const parsed = opts.schema.safeParse(parsedJson);
    if (!parsed.success) {
      throw new Error(
        `Structured output failed schema validation: ${parsed.error.message}\n` +
          `Model produced: ${JSON.stringify(parsedJson, null, 2)}`,
      );
    }

    const thinking = raw.message?.thinking;
    return {
      data: parsed.data,
      reasoning: typeof thinking === 'string' && thinking.length > 0 ? thinking : undefined,
      usage,
      cost,
      raw,
    };
  }

  return { ...base, structured };
}

/** Strip trailing slashes; ensure `/v1` suffix. */
function normalizeOllamaBaseURL(rawUrl: string): string {
  const trimmed = rawUrl.replace(/\/+$/, '');
  return trimmed.endsWith('/v1') ? trimmed : `${trimmed}/v1`;
}

/**
 * Shape of `/api/chat` response (only the fields we read).
 * See https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
 */
interface NativeChatResponse {
  message?: { role: string; content: string; thinking?: string };
  done?: boolean;
  done_reason?: string;
  prompt_eval_count?: number;
  eval_count?: number;
}
