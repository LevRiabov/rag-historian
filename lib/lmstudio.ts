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
import type { z } from 'zod';

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
  /** Model name as loaded in LM Studio (e.g., 'gpt-oss-20b', 'gemma-3-27b'). */
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
}

export interface LMStudioChatOpts extends LMStudioCallOpts {
  messages: Message[];
}

export interface LMStudioRunToolsOpts extends LMStudioCallOpts {
  messages: Message[];
  tools: Tool[];
  maxIterations?: number;
  onStep?: (step: ToolStep) => void;
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
  const defaultModel = config.defaultModel ?? 'gpt-oss-20b';
  const defaultMaxTokens = config.defaultMaxTokens ?? 4096;
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
        messages: toOpenAIMessages(opts.messages, opts.system),
      });

      const choice = response.choices[0];
      if (!choice) throw new Error('LM Studio returned no choices.');
      const text = choice.message.content ?? '';
      const usage = extractUsage(response.usage);
      const cost = calculateLMStudioCost(usage);
      const stopReason = mapFinishReason(choice.finish_reason);

      safeCall(tracer, 'onResponse', {
        usage,
        cost,
        latencyMs: Date.now() - t0,
        stopReason: choice.finish_reason ?? 'unknown',
      });

      return { text, usage, cost, stopReason, raw: response };
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
      }
    }

    return { text: finalText, usage: totalUsage, cost: totalCost, stop, steps, messages };
  }

  // --------------------------------------------------------------------------
  // structured — guaranteed-shape JSON via tool-use-as-output
  // --------------------------------------------------------------------------
  /**
   * Same trick as Anthropic's `structured` — define one tool, force the model
   * to call it, parse the call's arguments as the result. Avoids reliance on
   * OpenAI's `response_format: json_schema`, which is gated on model support
   * (some LM Studio models don't implement strict schema enforcement).
   */
  // biome-ignore lint/suspicious/noExplicitAny: see types.ts
  async function structured<Schema extends z.ZodObject<any>>(
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
      operation: 'structured',
      messageCount: opts.messages.length,
      toolCount: 1,
    });

    const t0 = Date.now();
    const response = await sdk.chat.completions.create({
      model,
      max_tokens: opts.maxTokens ?? defaultMaxTokens,
      ...(opts.temperature !== undefined && { temperature: opts.temperature }),
      messages: toOpenAIMessages(opts.messages, opts.system),
      tools: [toOpenAITool(outputTool)],
      // Forces calling this specific tool. Strongest schema guarantee available.
      tool_choice: { type: 'function', function: { name: outputToolName } },
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
      throw new Error('Expected a function tool_call for structured output; got none.');
    }
    let input: unknown;
    try {
      input = JSON.parse(call.function.arguments);
    } catch (err) {
      throw new Error(`Could not JSON-parse model's tool arguments: ${String(err)}`);
    }
    const parsed = opts.schema.safeParse(input);
    if (!parsed.success) {
      throw new Error(`Structured output failed schema validation: ${parsed.error.message}`);
    }
    return { data: parsed.data, usage, cost, raw: response };
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
