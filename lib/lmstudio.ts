/**
 * lib/lmstudio.ts — LM Studio wrapper via the OpenAI-compatible SDK.
 *
 * LM Studio exposes an OpenAI-compatible HTTP API on http://localhost:1234/v1
 * by default. We point the `openai` SDK at it.
 *
 * Same surface as claude.ts: chat / streamText / stream / runTools / structured.
 * Read both side by side — the design differences are the lesson:
 *
 *   System message:   Anthropic = separate param  | OpenAI = in messages array
 *   Tool calls:       block with structured input | message.tool_calls with JSON-string args
 *   Tool results:     content block in user turn  | message with role 'tool'
 *   Stop reason:      'tool_use'                  | 'tool_calls'
 *   Prompt caching:   yes (cache_control)         | no (LM Studio is local)
 *
 * Why bother with LM Studio at all: zero $ cost, full privacy, fast on the
 * user's 5070 Ti. Use for prototype-stage testing before burning Anthropic
 * credits. Module 9 will add proper routing between the two.
 */
import OpenAI from 'openai';
import { z } from 'zod';

import { addCost, addUsage, calculateLMStudioCost } from './cost.ts';
import { defineTool, executeTool, findTool, toOpenAITool } from './tools.ts';
import { noopTracer, safeCall, type Tracer } from './tracer.ts';
import type {
  ChatResult,
  Cost,
  Message,
  StopReason,
  StructuredResult,
  Tool,
  ToolLoopResult,
  ToolLoopStop,
  ToolStep,
  Usage,
} from './types.ts';

// ============================================================================
// Configuration
// ============================================================================

export interface LMStudioConfig {
  /** Base URL for the OpenAI-compatible endpoint. */
  baseURL?: string;
  /** Model name as loaded in LM Studio (use `org/model` format,
   *  e.g. `'openai/gpt-oss-20b'`). See `LM_STUDIO_MODELS` for known IDs. */
  defaultModel?: string;
  defaultMaxTokens?: number;
  tracer?: Tracer;
}

export interface LMStudioCallOpts {
  model?: string;
  maxTokens?: number;
  /** System prompt. LM Studio puts this inside the messages array. */
  system?: string;
  temperature?: number;
  /**
   * Enable reasoning. Accepts both shapes because different model families
   * use different params:
   *   - **Leveled** (`'low'|'medium'|'high'`) → maps to `reasoning_effort`
   *     (gpt-oss-20b, GPT-5, OpenAI-style models)
   *   - **Boolean** (`true`/`false`) → maps to
   *     `chat_template_kwargs.enable_thinking` (Gemma 4, Qwen3, most vLLM-served
   *     thinking models)
   * When set to a level we ALSO pass through the level form, so models that
   * speak only one of the two conventions get the right signal.
   * When set to `true` we additionally pass `reasoning_effort: 'medium'` as a
   * fallback for OpenAI-style models.
   */
  reasoning?: 'low' | 'medium' | 'high' | boolean;
}

/**
 * Build the reasoning portion of a chat completion body. Returns an object
 * suitable for spreading into the SDK call. Possibly empty.
 *
 * Why both `reasoning_effort` and `chat_template_kwargs.enable_thinking` get
 * sent at the same time: LM Studio's loaded model decides which one it honors.
 * Sending both is harmless to the other (extra fields are ignored).
 */
function buildLMStudioReasoning(
  reasoning: 'low' | 'medium' | 'high' | boolean | undefined,
): Record<string, unknown> {
  if (reasoning === undefined) return {};
  if (typeof reasoning === 'string') {
    return {
      reasoning_effort: reasoning,
      // Most boolean-style models treat any presence as "on"; this is the
      // belt-and-suspenders form so a level setting still works on them.
      chat_template_kwargs: { enable_thinking: true },
    };
  }
  return reasoning
    ? {
        reasoning_effort: 'medium',
        chat_template_kwargs: { enable_thinking: true },
      }
    : { chat_template_kwargs: { enable_thinking: false } };
}

export interface LMStudioChatOpts extends LMStudioCallOpts {
  messages: Message[];
}

export interface LMStudioRunToolsOpts extends LMStudioCallOpts {
  messages: Message[];
  tools: Tool[];
  maxIterations?: number;
  onStep?: (step: ToolStep) => void;
  /** Name of a TERMINAL tool. When the model calls it, the loop runs that tool
   *  then stops (stop='final_answer'), returning the tool's output as `text`.
   *  The Module 7 agent uses this for its `finalize` tool — an explicit "I'm
   *  done" signal, rather than relying on the model emitting a bare text turn. */
  terminalToolName?: string;
}

export interface LMStudioStructuredOpts<Schema extends z.ZodObject<z.ZodRawShape>>
  extends LMStudioCallOpts {
  messages: Message[];
  schema: Schema;
  outputToolName?: string;
}

// ============================================================================
// Factory
// ============================================================================

export function createLMStudio(config: LMStudioConfig = {}) {
  const sdk = new OpenAI({
    baseURL: config.baseURL ?? process.env.LM_STUDIO_BASE_URL ?? 'http://localhost:1234/v1',
    // LM Studio doesn't check API keys, but the SDK requires SOMETHING here.
    apiKey: 'lm-studio',
  });
  // Match LM Studio's `org/model-name` identifier exactly — passing just
  // `gpt-oss-20b` makes LM Studio load a SECOND copy (the unqualified version)
  // alongside the one already in memory. Wastes VRAM.
  const defaultModel = config.defaultModel ?? 'openai/gpt-oss-20b';
  const defaultMaxTokens = config.defaultMaxTokens ?? 16384;
  const tracer = config.tracer ?? noopTracer;

  // --------------------------------------------------------------------------
  // chat
  // --------------------------------------------------------------------------
  async function chat(opts: LMStudioChatOpts): Promise<ChatResult> {
    const model = opts.model ?? defaultModel;
    const t0 = Date.now();
    safeCall(tracer, 'onRequest', {
      provider: 'lmstudio',
      model,
      operation: 'chat',
      messageCount: opts.messages.length,
    });

    try {
      const response = await sdk.chat.completions.create({
        model,
        max_tokens: opts.maxTokens ?? defaultMaxTokens,
        ...(opts.temperature !== undefined && { temperature: opts.temperature }),
        ...buildLMStudioReasoning(opts.reasoning),
        messages: toOpenAIMessages(opts.messages, opts.system),
      });

      const choice = response.choices[0];
      if (!choice) throw new Error('LM Studio returned no choices.');
      const text = choice.message.content ?? '';
      const reasoning = extractReasoning(choice.message);
      const usage = extractUsage(response.usage);
      const cost = calculateLMStudioCost(usage);
      const stopReason = mapFinishReason(choice.finish_reason);

      safeCall(tracer, 'onResponse', {
        usage,
        cost,
        latencyMs: Date.now() - t0,
        stopReason: choice.finish_reason ?? 'unknown',
      });

      return { text, reasoning, usage, cost, stopReason, raw: response };
    } catch (err) {
      safeCall(tracer, 'onError', err, { operation: 'chat' });
      throw err;
    }
  }

  // --------------------------------------------------------------------------
  // streamText
  // --------------------------------------------------------------------------
  /**
   * Yields text deltas. OpenAI's streaming is structurally simpler than
   * Anthropic's: each chunk has `choices[0].delta.content` as the text. No
   * nested event types to filter.
   */
  async function* streamText(opts: LMStudioChatOpts): AsyncGenerator<string, void, undefined> {
    const model = opts.model ?? defaultModel;
    safeCall(tracer, 'onRequest', {
      provider: 'lmstudio',
      model,
      operation: 'streamText',
      messageCount: opts.messages.length,
    });

    try {
      const stream = await sdk.chat.completions.create({
        model,
        max_tokens: opts.maxTokens ?? defaultMaxTokens,
        ...(opts.temperature !== undefined && { temperature: opts.temperature }),
        ...buildLMStudioReasoning(opts.reasoning),
        messages: toOpenAIMessages(opts.messages, opts.system),
        stream: true,
      });
      for await (const chunk of stream) {
        const delta = chunk.choices[0]?.delta.content;
        if (delta) yield delta;
      }
    } catch (err) {
      safeCall(tracer, 'onError', err, { operation: 'streamText' });
      throw err;
    }
  }

  // --------------------------------------------------------------------------
  // stream — raw event stream
  // --------------------------------------------------------------------------
  function stream(opts: LMStudioChatOpts) {
    const model = opts.model ?? defaultModel;
    return sdk.chat.completions.create({
      model,
      max_tokens: opts.maxTokens ?? defaultMaxTokens,
      ...(opts.temperature !== undefined && { temperature: opts.temperature }),
      ...buildLMStudioReasoning(opts.reasoning),
      messages: toOpenAIMessages(opts.messages, opts.system),
      stream: true,
    });
  }

  // --------------------------------------------------------------------------
  // runTools — agentic loop, OpenAI-style
  // --------------------------------------------------------------------------
  /**
   * Same shape as Anthropic's runTools, but the tool-use protocol differs:
   *   - The assistant's tool calls live in `message.tool_calls`.
   *   - `function.arguments` is a JSON-encoded STRING — must parse before use.
   *   - Tool results are appended as messages with role 'tool', referencing
   *     `tool_call_id`. NOT as content blocks inside a user turn.
   *   - Finish reason 'tool_calls' (not 'tool_use') signals the model wants tools.
   */
  async function runTools(opts: LMStudioRunToolsOpts): Promise<ToolLoopResult> {
    const model = opts.model ?? defaultModel;
    const maxIterations = opts.maxIterations ?? 25;
    const openaiTools = opts.tools.map(toOpenAITool);

    // The message history grows with assistant turns (carrying tool_calls) and
    // tool-result turns. ChatCompletionMessageParam is the SDK's union covering
    // all of those shapes.
    const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = toOpenAIMessages(
      opts.messages,
      opts.system,
    );

    const steps: ToolStep[] = [];
    let totalUsage: Usage = { inputTokens: 0, outputTokens: 0 };
    let totalCost: Cost = zeroCost();
    let stop: ToolLoopStop = 'max_iterations';
    let finalText = '';

    for (let iteration = 1; iteration <= maxIterations; iteration++) {
      safeCall(tracer, 'onRequest', {
        provider: 'lmstudio',
        model,
        operation: 'runTools',
        messageCount: messages.length,
        toolCount: openaiTools.length,
      });

      const t0 = Date.now();
      const response = await sdk.chat.completions.create({
        model,
        max_tokens: opts.maxTokens ?? defaultMaxTokens,
        ...(opts.temperature !== undefined && { temperature: opts.temperature }),
        ...buildLMStudioReasoning(opts.reasoning),
        messages,
        tools: openaiTools,
      });

      const choice = response.choices[0];
      if (!choice) throw new Error('LM Studio returned no choices.');
      const usage = extractUsage(response.usage);
      const cost = calculateLMStudioCost(usage);
      totalUsage = addUsage(totalUsage, usage);
      totalCost = addCost(totalCost, cost);
      safeCall(tracer, 'onResponse', {
        usage,
        cost,
        latencyMs: Date.now() - t0,
        stopReason: choice.finish_reason ?? 'unknown',
      });

      // Append the assistant turn. Include tool_calls — needed on the next
      // turn to match up with our tool responses.
      messages.push({
        role: 'assistant',
        content: choice.message.content ?? null,
        tool_calls: choice.message.tool_calls,
      });

      if (choice.finish_reason !== 'tool_calls' || !choice.message.tool_calls?.length) {
        finalText = choice.message.content ?? '';
        stop = 'final_answer';
        break;
      }

      let terminalArticle: string | null = null;
      for (const call of choice.message.tool_calls) {
        if (call.type !== 'function') continue;
        const tool = findTool(opts.tools, call.function.name);

        // OpenAI's arguments come as a JSON STRING; parse defensively. A
        // malformed JSON string just becomes an empty object — Zod's safeParse
        // will surface the validation error to the model on the next turn.
        let input: unknown;
        try {
          input = JSON.parse(call.function.arguments);
        } catch {
          input = {};
        }

        const tStart = Date.now();
        const output = await executeTool(tool, input);
        const toolLatencyMs = Date.now() - tStart;
        safeCall(tracer, 'onToolCall', {
          iteration,
          toolName: call.function.name,
          toolInput: input,
          toolOutput: output,
          latencyMs: toolLatencyMs,
        });

        // Each tool result is its OWN message (with role 'tool'), referencing
        // the tool_call_id. Unlike Anthropic, results aren't grouped in a single
        // user turn — they're individual messages.
        messages.push({ role: 'tool', tool_call_id: call.id, content: output });

        const step: ToolStep = {
          iteration,
          toolName: call.function.name,
          toolInput: input,
          toolOutput: output,
          usage,
          cost,
        };
        steps.push(step);
        opts.onStep?.(step);

        if (opts.terminalToolName && call.function.name === opts.terminalToolName) {
          terminalArticle = output;
        }
      }

      // A terminal tool was called this turn → stop, returning its output as the
      // final text (no further round-trip). Tool-result messages above are kept
      // so `messages` stays a valid, inspectable transcript.
      if (terminalArticle !== null) {
        finalText = terminalArticle;
        stop = 'final_answer';
        break;
      }
    }

    // Budget exhausted without a terminal answer — one final tool-free call so an
    // agent loop that over-searches to the cap still returns an answer, not ''.
    // `stop` stays 'max_iterations'; with no tools the model must synthesize from
    // what it already gathered. (Local qwen over-searches, so this path matters.)
    if (stop === 'max_iterations' && opts.terminalToolName) {
      safeCall(tracer, 'onRequest', {
        provider: 'lmstudio',
        model,
        operation: 'runTools:synthesize',
        messageCount: messages.length,
        toolCount: 0,
      });
      const tS = Date.now();
      const response = await sdk.chat.completions.create({
        model,
        max_tokens: opts.maxTokens ?? defaultMaxTokens,
        ...(opts.temperature !== undefined && { temperature: opts.temperature }),
        ...buildLMStudioReasoning(opts.reasoning),
        messages,
        // No `tools` — force a text answer from what was already gathered.
      });
      const choice = response.choices[0];
      if (choice) {
        const usage = extractUsage(response.usage);
        const cost = calculateLMStudioCost(usage);
        totalUsage = addUsage(totalUsage, usage);
        totalCost = addCost(totalCost, cost);
        safeCall(tracer, 'onResponse', {
          usage,
          cost,
          latencyMs: Date.now() - tS,
          stopReason: choice.finish_reason ?? 'unknown',
        });
        messages.push({ role: 'assistant', content: choice.message.content ?? '' });
        finalText = choice.message.content ?? '';
      }
    }

    return { text: finalText, usage: totalUsage, cost: totalCost, stop, steps, messages };
  }

  // --------------------------------------------------------------------------
  // structured — auto-falling-back guaranteed-shape JSON
  // --------------------------------------------------------------------------
  /**
   * Tries two paths to guaranteed structured output, in order:
   *
   *   1. `response_format: { type: 'json_schema' }` (modern path) — supported
   *      by Ollama 0.5+, LM Studio 0.3+, OpenAI proper, GPT-5. The runtime's
   *      llama.cpp backend uses the schema for grammar-constrained sampling
   *      (GBNF): mechanically impossible to produce invalid output.
   *
   *   2. `tool_choice: 'required'` with a single output tool (older path) —
   *      for models whose chat template doesn't engage the response_format
   *      grammar (e.g. Gemma 4 in Ollama emits plain text "Billing" instead
   *      of JSON). Forced tool calls go through a different code path many
   *      such models DO honor.
   *
   * We auto-fall-back on parse failure rather than pre-checking the model:
   * the failure mode is observable, so runtime behavior drives the path.
   * Cost is ~2× for failing models (one wasted call + one real). Worth it
   * vs. asking the caller to know which mode each model speaks.
   *
   * `strict: false` on path 1 is intentional — strict mode requires every
   * property to be required and rejects `additionalProperties: true`. Lenient
   * mode still grammar-constrains while accepting `.nullish()` / `.optional()`
   * Zod patterns we use for cross-model robustness.
   */
  async function structured<Schema extends z.ZodObject<z.ZodRawShape>>(
    opts: LMStudioStructuredOpts<Schema>,
  ): Promise<StructuredResult<z.infer<Schema>>> {
    try {
      return await structuredViaJsonSchema(opts);
    } catch (err) {
      // Only fall back on parse / empty-content failures (model didn't honor
      // response_format). Real errors (network, schema validation) propagate.
      if (err instanceof Error && /JSON-parse|empty content/i.test(err.message)) {
        console.warn(
          '[structured] response_format produced non-JSON; falling back to tool_use. ' +
            "If you see this often, the model doesn't honor response_format — " +
            'consider using a different model for structured tasks.',
        );
        return structuredViaToolUse(opts);
      }
      throw err;
    }
  }

  /** Path 1: response_format json_schema — grammar-constrained on supporting runtimes. */
  async function structuredViaJsonSchema<Schema extends z.ZodObject<z.ZodRawShape>>(
    opts: LMStudioStructuredOpts<Schema>,
  ): Promise<StructuredResult<z.infer<Schema>>> {
    const model = opts.model ?? defaultModel;
    const schemaName = opts.outputToolName ?? 'output';

    const jsonSchema = z.toJSONSchema(opts.schema) as Record<string, unknown>;
    delete jsonSchema.$schema;

    safeCall(tracer, 'onRequest', {
      provider: 'lmstudio',
      model,
      operation: 'structured',
      messageCount: opts.messages.length,
    });

    const t0 = Date.now();
    const response = await sdk.chat.completions.create({
      model,
      max_tokens: opts.maxTokens ?? defaultMaxTokens,
      ...(opts.temperature !== undefined && { temperature: opts.temperature }),
      ...buildLMStudioReasoning(opts.reasoning),
      messages: toOpenAIMessages(opts.messages, opts.system),
      response_format: {
        type: 'json_schema',
        json_schema: {
          name: schemaName,
          schema: jsonSchema,
          strict: false,
        },
      },
    });

    const choice = response.choices[0];
    if (!choice) throw new Error('LM Studio returned no choices.');
    const usage = extractUsage(response.usage);
    const cost = calculateLMStudioCost(usage);
    safeCall(tracer, 'onResponse', {
      usage,
      cost,
      latencyMs: Date.now() - t0,
      stopReason: choice.finish_reason ?? 'unknown',
    });

    const text = choice.message.content;
    if (!text) {
      throw new Error('Expected JSON content in response, got empty content.');
    }
    let parsedJson: unknown;
    try {
      parsedJson = JSON.parse(text);
    } catch (err) {
      throw new Error(
        `Could not JSON-parse model response: ${String(err)}\nModel produced: ${text}`,
      );
    }
    const parsed = opts.schema.safeParse(parsedJson);
    if (!parsed.success) {
      // Should be rare now that the grammar enforces shape — but small models
      // occasionally produce JSON that's STRUCTURALLY valid against the schema
      // but semantically wrong. Surface the model output so we can see.
      throw new Error(
        `Structured output failed schema validation: ${parsed.error.message}\n` +
          `Model produced: ${JSON.stringify(parsedJson, null, 2)}`,
      );
    }
    return {
      data: parsed.data,
      reasoning: extractReasoning(choice.message),
      usage,
      cost,
      raw: response,
    };
  }

  /** Path 2: tool-use-as-output — for models that ignore response_format. */
  async function structuredViaToolUse<Schema extends z.ZodObject<z.ZodRawShape>>(
    opts: LMStudioStructuredOpts<Schema>,
  ): Promise<StructuredResult<z.infer<Schema>>> {
    const model = opts.model ?? defaultModel;
    const outputToolName = opts.outputToolName ?? 'output';

    const outputTool = defineTool({
      name: outputToolName,
      description: 'Return your final structured output via this tool.',
      schema: opts.schema,
      execute: () => '',
    });

    safeCall(tracer, 'onRequest', {
      provider: 'lmstudio',
      model,
      operation: 'structured.toolUseFallback',
      messageCount: opts.messages.length,
      toolCount: 1,
    });

    const t0 = Date.now();
    const response = await sdk.chat.completions.create({
      model,
      max_tokens: opts.maxTokens ?? defaultMaxTokens,
      ...(opts.temperature !== undefined && { temperature: opts.temperature }),
      ...buildLMStudioReasoning(opts.reasoning),
      messages: toOpenAIMessages(opts.messages, opts.system),
      tools: [toOpenAITool(outputTool)],
      tool_choice: 'required',
    });

    const choice = response.choices[0];
    if (!choice) throw new Error('LM Studio returned no choices.');
    const usage = extractUsage(response.usage);
    const cost = calculateLMStudioCost(usage);
    safeCall(tracer, 'onResponse', {
      usage,
      cost,
      latencyMs: Date.now() - t0,
      stopReason: choice.finish_reason ?? 'unknown',
    });

    const call = choice.message.tool_calls?.[0];
    if (!call || call.type !== 'function') {
      throw new Error(
        `Expected a function tool_call (tool-use fallback); got finish_reason=${choice.finish_reason}, content=${choice.message.content ?? '(none)'}`,
      );
    }
    let input: unknown;
    try {
      input = JSON.parse(call.function.arguments);
    } catch (err) {
      throw new Error(`Could not JSON-parse model's tool arguments: ${String(err)}`);
    }
    const parsed = opts.schema.safeParse(input);
    if (!parsed.success) {
      throw new Error(
        `Structured output failed schema validation (tool-use fallback): ${parsed.error.message}\n` +
          `Model produced: ${JSON.stringify(input, null, 2)}`,
      );
    }
    return {
      data: parsed.data,
      reasoning: extractReasoning(choice.message),
      usage,
      cost,
      raw: response,
    };
  }

  return { chat, streamText, stream, runTools, structured };
}

// ============================================================================
// Internal helpers
// ============================================================================

function toOpenAIMessages(messages: Message[], system: string | undefined) {
  const out: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [];
  if (system) out.push({ role: 'system', content: system });
  for (const m of messages) {
    if (m.role === 'system') {
      out.push({ role: 'system', content: m.content });
    } else {
      out.push({ role: m.role, content: m.content });
    }
  }
  return out;
}

/**
 * Pull the reasoning string off a chat-completion message.
 *
 * The OpenAI SDK's `ChatCompletionMessage` type doesn't include `reasoning` —
 * it's a non-standard extension that Ollama and some LM Studio backends add
 * when `reasoning_effort` / `think` triggers a chain of thought. We cast to a
 * narrowed shape rather than `as any` so the access stays type-safe.
 *
 * Returns undefined (not `''`) when there is no reasoning, so callers can
 * cleanly tell "no reasoning happened" apart from "reasoning ran but was empty".
 */
function extractReasoning(message: unknown): string | undefined {
  const r = (message as { reasoning?: unknown }).reasoning;
  return typeof r === 'string' && r.length > 0 ? r : undefined;
}

function extractUsage(u: { prompt_tokens: number; completion_tokens: number } | undefined): Usage {
  if (!u) return { inputTokens: 0, outputTokens: 0 };
  return {
    inputTokens: u.prompt_tokens,
    outputTokens: u.completion_tokens,
  };
}

/** Map OpenAI's finish_reason → our normalized StopReason. */
function mapFinishReason(reason: string | null | undefined): StopReason {
  switch (reason) {
    case 'stop':
      return 'end_turn';
    case 'length':
      return 'max_tokens';
    case 'tool_calls':
      return 'tool_use';
    default:
      return 'unknown';
  }
}

function zeroCost(): Cost {
  return {
    inputUSD: 0,
    outputUSD: 0,
    cacheCreationUSD: 0,
    cacheReadUSD: 0,
    totalUSD: 0,
  };
}
