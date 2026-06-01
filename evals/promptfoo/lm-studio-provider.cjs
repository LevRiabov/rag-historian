/**
 * evals/promptfoo/lm-studio-provider.cjs — custom Promptfoo provider for
 * LM Studio chat models. CJS extension is deliberate: our package.json
 * has `"type": "module"`, so a plain `.js` file is treated as ESM and
 * Promptfoo's `require()`-based provider loader cannot consume it. `.cjs`
 * forces CommonJS regardless of the package type.
 *
 * Why custom: Promptfoo's `openai:chat:` provider doesn't reliably forward
 * arbitrary body fields like `chat_template_kwargs.enable_thinking` across
 * provider versions. Without that field, qwen3 family stays in thinking
 * mode, burns max_tokens on reasoning, and `content` comes back empty
 * (every assertion then fails with "output is empty").
 *
 * Configuration (promptfooconfig.yaml):
 *   providers:
 *     - id: file://evals/promptfoo/lm-studio-provider.cjs
 *       label: 'qwen 3.5 9B'
 *       config:
 *         model: 'qwen/qwen3.5-9b'
 *         max_tokens: 1024
 *         temperature: 0
 *
 * Diagnostic logging via console.error (so it shows up in Promptfoo's
 * stderr without polluting its progress UI). Look for:
 *   [lm-studio] module loaded               ← Promptfoo could load this file
 *   [lm-studio] constructed: {...}          ← Promptfoo instantiated the class
 *   [lm-studio] callApi: question="..."     ← Promptfoo invoked the provider
 *   [lm-studio] response: N chars, M tok    ← LM Studio returned something
 *
 * If "module loaded" never prints, Promptfoo isn't using this file at all.
 */

const DEFAULT_BASE_URL = 'http://localhost:1234/v1';
const DEFAULT_MODEL = 'qwen/qwen3.5-9b';
const DEFAULT_MAX_TOKENS = 1024;
const DEFAULT_TEMPERATURE = 0;

console.error('[lm-studio] module loaded');

class LmStudioProvider {
  constructor(options = {}) {
    const config = options.config ?? {};
    this.providerId =
      options.id ?? `lmstudio:${config.model ?? DEFAULT_MODEL}`;
    this.config = {
      baseUrl:
        config.baseUrl ?? process.env.LM_STUDIO_BASE_URL ?? DEFAULT_BASE_URL,
      model: config.model ?? DEFAULT_MODEL,
      max_tokens: config.max_tokens ?? DEFAULT_MAX_TOKENS,
      temperature: config.temperature ?? DEFAULT_TEMPERATURE,
    };
    console.error('[lm-studio] constructed:', JSON.stringify(this.config));
  }

  id() {
    return this.providerId;
  }

  async callApi(prompt, _context, _options) {
    const messages = this._parseMessages(prompt);
    const lastUser = messages
      .slice()
      .reverse()
      .find((m) => m.role === 'user');
    const preview = (lastUser?.content ?? '').slice(0, 80).replace(/\s+/g, ' ');
    console.error(`[lm-studio] callApi: "${preview}..."`);

    const body = {
      model: this.config.model,
      messages,
      temperature: this.config.temperature,
      max_tokens: this.config.max_tokens,
      // ALWAYS disable thinking for qwen3 family. Non-qwen models ignore
      // this field as an unknown body parameter — safe no-op.
      chat_template_kwargs: { enable_thinking: false },
    };

    const url = `${this.config.baseUrl.replace(/\/$/, '')}/chat/completions`;
    const start = Date.now();

    let response;
    try {
      response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`[lm-studio] fetch failed: ${msg}`);
      return { error: `LM Studio fetch failed: ${msg}` };
    }

    if (!response.ok) {
      const text = await response.text().catch(() => '(no body)');
      console.error(
        `[lm-studio] HTTP ${response.status}: ${text.slice(0, 200)}`,
      );
      return {
        error: `LM Studio HTTP ${response.status}: ${text.slice(0, 300)}`,
      };
    }

    const data = await response.json();
    const content = data?.choices?.[0]?.message?.content ?? '';
    const usage = data?.usage ?? {};
    const latencyMs = Date.now() - start;

    console.error(
      `[lm-studio] response: ${content.length} chars, ` +
        `${usage.completion_tokens ?? 0} completion tokens, ${latencyMs}ms`,
    );

    if (content.length === 0) {
      // Surface the empty-content failure mode loudly so it's obvious in logs.
      console.error(
        '[lm-studio] WARNING: empty content despite chat_template_kwargs.enable_thinking=false. ' +
          'Either the model ignores the field (unlikely for qwen3) or LM Studio is configured ' +
          'to override it. Check LM Studio per-model settings.',
      );
    }

    return {
      output: content,
      tokenUsage: {
        total: usage.total_tokens ?? 0,
        prompt: usage.prompt_tokens ?? 0,
        completion: usage.completion_tokens ?? 0,
      },
      cost: 0,
      latencyMs,
    };
  }

  _parseMessages(prompt) {
    if (Array.isArray(prompt)) return prompt;
    if (typeof prompt !== 'string') {
      return [{ role: 'user', content: String(prompt) }];
    }
    const trimmed = prompt.trim();
    if (trimmed.startsWith('[')) {
      try {
        const parsed = JSON.parse(trimmed);
        if (Array.isArray(parsed)) return parsed;
      } catch {
        // not JSON — treat as plain text below
      }
    }
    return [{ role: 'user', content: prompt }];
  }
}

module.exports = LmStudioProvider;
