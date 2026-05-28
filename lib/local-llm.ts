/**
 * lib/local-llm.ts — pick a local LLM runtime by env var, return a unified client.
 *
 * Both runtimes expose the same surface (`chat` / `streamText` / `stream` /
 * `runTools` / `structured`), so the rest of the codebase doesn't care which
 * one is behind the wrapper. This helper exists to centralize the choice in
 * ONE place instead of every mini-project re-deriving it.
 *
 *   LOCAL_LLM_PROVIDER=lmstudio   → createLMStudio (DEFAULT)
 *   LOCAL_LLM_PROVIDER=ollama     → createOllama
 *
 * Why LM Studio is the default: its GBNF grammar enforcement is a HARD
 * constraint applied at the token sampler — every model, every request. Ollama
 * routes structured output through per-model renderers that engage GBNF
 * inconsistently (Gemma 4 will happily emit out-of-enum values; gpt-oss is
 * fine). Reliability beats convenience. Override via env when you specifically
 * want Ollama's `ollama pull` ergonomics or to test a model only available there.
 *
 * Per-runtime config (defaultModel, baseURL, etc.) lives under `lmstudio` and
 * `ollama` keys — the helper only forwards the section matching the chosen
 * provider. This lets callers spell models in each runtime's native format
 * (`openai/gpt-oss-20b` vs `gpt-oss:20b`) without per-call branching.
 */
import { createLMStudio, type LMStudioConfig } from './lmstudio.ts';
import { createOllama, OLLAMA_MODELS } from './ollama.ts';
import type { Tracer } from './tracer.ts';

/** Runtimes this helper knows how to launch. */
export type LocalProvider = 'lmstudio' | 'ollama';

export interface CreateLocalLLMConfig {
  /** Settings used when LOCAL_LLM_PROVIDER=lmstudio (or unset). */
  lmstudio?: LMStudioConfig;
  /** Settings used when LOCAL_LLM_PROVIDER=ollama. */
  ollama?: LMStudioConfig;
  /**
   * Tracer shared across both runtimes. Set here instead of duplicating it
   * inside `lmstudio` and `ollama` configs — saves the "I switched provider
   * and forgot to copy the tracer" trap.
   */
  tracer?: Tracer;
}

/** Return shape: the client itself plus metadata for logging / branching. */
export interface LocalLLM {
  /** The wrapper (createLMStudio or createOllama). Same method surface either way. */
  client: ReturnType<typeof createLMStudio>;
  /** Which runtime was actually selected (after env resolution). */
  provider: LocalProvider;
  /** Resolved model identifier, in the chosen runtime's native naming. */
  model: string;
  /** Pretty label like `LM Studio (openai/gpt-oss-20b)` — for `console.log`. */
  label: string;
}

const DEFAULT_LMSTUDIO_MODEL = 'openai/gpt-oss-20b';

/**
 * Resolve the env var to a provider. Anything unrecognized (typos, empty
 * string) falls back to 'lmstudio' — fail-safe to the more reliable default.
 */
function resolveProvider(): LocalProvider {
  const raw = process.env.LOCAL_LLM_PROVIDER?.toLowerCase().trim();
  if (raw === 'ollama') return 'ollama';
  return 'lmstudio';
}

/**
 * Build a local-LLM client. Env-driven choice with sensible defaults.
 *
 * Example:
 *   const local = createLocalLLM({
 *     lmstudio: { defaultModel: LM_STUDIO_MODELS.gemma4_4b },
 *     ollama:   { defaultModel: OLLAMA_MODELS.gemma4_e2b },
 *     tracer:   statsTracer,
 *   });
 *   console.log(`Provider: ${local.label}`);
 *   await local.client.structured({ ... });
 */
export function createLocalLLM(config: CreateLocalLLMConfig = {}): LocalLLM {
  const provider = resolveProvider();

  if (provider === 'ollama') {
    // Per-call override via env (back-compat with the older OLLAMA_MODEL convention).
    const defaultModel =
      process.env.OLLAMA_MODEL ?? config.ollama?.defaultModel ?? OLLAMA_MODELS.gptOss20b;
    const client = createOllama({
      ...config.ollama,
      defaultModel,
      tracer: config.tracer ?? config.ollama?.tracer,
    });
    return { client, provider, model: defaultModel, label: `Ollama (${defaultModel})` };
  }

  // lmstudio path
  const defaultModel =
    process.env.LM_STUDIO_MODEL ?? config.lmstudio?.defaultModel ?? DEFAULT_LMSTUDIO_MODEL;
  const client = createLMStudio({
    ...config.lmstudio,
    defaultModel,
    tracer: config.tracer ?? config.lmstudio?.tracer,
  });
  return { client, provider, model: defaultModel, label: `LM Studio (${defaultModel})` };
}
