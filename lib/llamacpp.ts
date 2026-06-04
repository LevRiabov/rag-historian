/**
 * lib/llamacpp.ts — llama.cpp (via llama-swap) wrapper.
 *
 * llama.cpp's `llama-server` exposes an OpenAI-compatible API, and llama-swap
 * proxies one stable port (default :8080) that hot-swaps models on demand. So
 * this is the SAME OpenAI-compatible surface as LM Studio — `createLlamacpp`
 * just delegates to `createLMStudio` with the llama-swap base URL and a
 * llama-swap model profile as the default. Same methods: chat / streamText /
 * stream / runTools / structured.
 *
 * Why a dedicated client instead of reusing LM Studio: it's the project's local
 * runtime now (embeddings + reranking already moved here; see lib/embeddings.ts
 * and lib/rerank.ts). Keeping it named honestly — rather than pointing a client
 * called "lmstudio" at port 8080 — keeps the stack legible.
 *
 * Two things that differ from LM Studio in practice:
 *
 *   1. **Structured output is reliable here, no override needed.** `structured()`
 *      uses `response_format: json_schema`, which llama.cpp binds to its GBNF
 *      grammar sampler — a hard token-level constraint (this is the very engine
 *      LM Studio runs internally). So unlike `createOllama` we do NOT override
 *      structured() onto a native endpoint; the inherited path is the good one.
 *
 *   2. **Thinking is selected by MODEL PROFILE, not per-request.** Each profile
 *      bakes `--reasoning on|off` in the llama-swap config: `qwen-9b-16k` is
 *      thinking-OFF, `qwen-9b-16k-think` is thinking-ON. So callers pick the
 *      profile they want and should NOT pass `reasoning` (the deprecated
 *      per-request `chat_template_kwargs.enable_thinking` is what `reasoning`
 *      maps to, and the profile flag already governs).
 *
 * Trace label caveat: the inherited chat/structured methods emit traces tagged
 * `provider: 'lmstudio'` (createLMStudio hardcodes it) — same imprecision as
 * createOllama's delegated methods. Cosmetic; the tracer aggregates fine.
 *
 * Quick start (the server lives at C:\llm; see llama-swap config.yaml):
 *   curl http://127.0.0.1:8080/v1/models     # confirm profiles are registered
 */
import { createLMStudio, type LMStudioConfig } from './lmstudio.ts';

/**
 * llama-swap model profiles (names must match the `models:` keys in the
 * llama-swap config.yaml). Reasoning is baked into each profile:
 *
 * | Profile             | context | thinking |
 * |---------------------|---------|----------|
 * | qwen-9b-16k         |  16k    |   off    |
 * | qwen-9b-32k         |  32k    |   off    |
 * | qwen-9b-64k         |  64k    |   off    |
 * | qwen-9b-100k        | 100k    |   off    |
 * | qwen-9b-16k-think   |  16k    |   on     |
 *
 * Requesting a different profile triggers llama-swap to load it (and swap out
 * the previous one in the same group). Pick the smallest context that fits the
 * task — KV-cache VRAM scales with context.
 */
export const LLAMACPP_MODELS = {
  /** Qwen3.5-9B, 16k context, thinking OFF. Default for RAG answer generation. */
  qwen9b16k: 'qwen-9b-16k',
  /** Qwen3.5-9B, 32k context, thinking OFF. */
  qwen9b32k: 'qwen-9b-32k',
  /** Qwen3.5-9B, 64k context, thinking OFF. */
  qwen9b64k: 'qwen-9b-64k',
  /** Qwen3.5-9B, 100k context, thinking OFF. Long-document tasks. */
  qwen9b100k: 'qwen-9b-100k',
  /** Qwen3.5-9B, 16k context, thinking ON (reasoning separated into reasoning_content). */
  qwen9b16kThink: 'qwen-9b-16k-think',
} as const;

/** Union of known llama-swap profile IDs (autocomplete on the favorites). */
export type LlamacppModel = (typeof LLAMACPP_MODELS)[keyof typeof LLAMACPP_MODELS];

/** Public client shape — identical to createLMStudio (callers can swap freely). */
export type LlamacppClient = ReturnType<typeof createLMStudio>;

/**
 * Build a llama.cpp client. Delegates to `createLMStudio` (same OpenAI-compat
 * surface) pointed at the llama-swap port.
 *
 * The base URL is normalized to end in `/v1`. `LLAMA_SWAP_BASE_URL` may be set
 * to either `http://127.0.0.1:8080` (root) or `.../v1` — both work.
 */
export function createLlamacpp(config: LMStudioConfig = {}): LlamacppClient {
  const rawBase = config.baseURL ?? process.env.LLAMA_SWAP_BASE_URL ?? 'http://127.0.0.1:8080';
  return createLMStudio({
    ...config,
    baseURL: normalizeLlamacppBaseURL(rawBase),
    defaultModel: config.defaultModel ?? LLAMACPP_MODELS.qwen9b16k,
  });
}

/** Strip trailing slashes; ensure `/v1` suffix. */
function normalizeLlamacppBaseURL(rawUrl: string): string {
  const trimmed = rawUrl.replace(/\/+$/, '');
  return trimmed.endsWith('/v1') ? trimmed : `${trimmed}/v1`;
}
