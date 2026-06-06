/**
 * lib/claude.ts — the Anthropic SDK wrapper.
 *
 * This is what Module 1 is really about. Everything here is built directly on
 * @anthropic-ai/sdk's `messages.create` and `messages.stream`. Read it top to
 * bottom and you'll understand the Messages API.
 *
 * Methods on the returned client:
 *   chat()        — single-shot, returns text + usage + cost
 *   streamText()  — async generator of text deltas (high-level, no events)
 *   stream()      — raw event stream from the SDK (advanced)
 *   runTools()    — agentic loop: keep calling tools until the model stops
 *   structured()  — tool-use-as-output for guaranteed JSON matching a Zod schema
 *
 * Why factory (not class): no `this` gotchas, no inheritance hooks we don't
 *   want, trivial to mock by replacing the returned object.
 *
 * Why we don't reimplement retries: the SDK already handles 429s + 5xx with
 *   exponential backoff (maxRetries default 2). Reimplementing risks divergent
 *   behavior. Need different retry behavior? Pass `maxRetries` here.
 *
 * Why prompt caching is opt-in (not automatic): cache-write tokens cost 1.25×
 *   normal input. Auto-applying cache_control would surprise-cost the caller.
 *   Made explicit via `opts.cacheSystem` so the cost is a conscious decision.
 */
import Anthropic from '@anthropic-ai/sdk';
import type { z } from 'zod';

import {
  addCost,
  addUsage,
  CLAUDE_MODELS,
  type ClaudeModel,
  calculateAnthropicCost,
} from './cost.ts';
import { defineTool, executeTool, findTool, toAnthropicTool } from './tools.ts';
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

export interface ClaudeConfig {
  /** Anthropic API key. Falls back to ANTHROPIC_API_KEY env var. */
  apiKey?: string;
  /** Model used when individual calls don't specify one. Defaults to Haiku
   *  (cheapest) — change to `CLAUDE_MODELS.sonnet` or `.opus` per call as needed. */
  defaultModel?: ClaudeModel;
  /** Default `max_tokens` for the response — required by the API. */
  defaultMaxTokens?: number;
  /** Observability hook. Default no-op; swap for Langfuse later. */
  tracer?: Tracer;
  /** SDK-level retry count (429s, 5xx). Default 2 = SDK default. */
  maxRetries?: number;
}

export interface CallOpts {
  /** Override the client's default model for this call. */
  model?: ClaudeModel;
  maxTokens?: number;
  /** Optional system prompt. Anthropic takes this on a top-level param. */
  system?: string;
  /** Mark the system prompt with cache_control. Use for large stable system
   *  prompts — first call writes the cache (1.25× input), later calls read it
   *  (0.1× input). Net win after just a few calls with the same prefix. */
  cacheSystem?: boolean;
  /** 0–1; default 1.0. Lower for deterministic tasks. NOTE: ignored when
   *  `reasoning` is set — Anthropic's extended thinking requires temperature=1. */
  temperature?: number;
  /**
   * Enable extended thinking. The model emits private reasoning content blocks
   * before the visible response.
   *   - 'low'    → 1k token thinking budget   (fast, mild benefit)
   *   - 'medium' → 4k token thinking budget   (most common useful setting)
   *   - 'high'   → 16k token thinking budget  (hard problems only)
   *   - `true`   → shorthand for 'medium'
   *   - `false` / undefined → no thinking
   * Boolean is supported for parity with `lib/lmstudio.ts` where some local
   * models only accept a boolean toggle. `maxTokens` is auto-bumped to make
   * room for visible output on top of the thinking budget.
   */
  reasoning?: 'low' | 'medium' | 'high' | boolean;
}

/** Token budgets per reasoning level for Anthropic's extended thinking. */
const REASONING_BUDGETS = { low: 1024, medium: 4096, high: 16384 } as const;

/**
 * Build the part of an Anthropic request that depends on reasoning. Returns
 * `max_tokens` (auto-bumped above the thinking budget) and an optional
 * `thinking` block. Centralized so every method behaves the same way.
 *
 * Boolean `true` maps to `'medium'` — a reasonable default for "enable thinking
 * without specifying a level." Boolean `false` is equivalent to undefined.
 */
function buildReasoningParams(
  reasoning: 'low' | 'medium' | 'high' | boolean | undefined,
  providedMaxTokens: number | undefined,
  defaultMaxTokens: number,
): { max_tokens: number; thinking?: { type: 'enabled'; budget_tokens: number } } {
  const baseMax = providedMaxTokens ?? defaultMaxTokens;
  if (!reasoning) return { max_tokens: baseMax }; // undefined or false
  const level = reasoning === true ? 'medium' : reasoning;
  const budget = REASONING_BUDGETS[level];
  return {
    // Ensure at least 1024 tokens of visible output on top of the thinking budget.
    max_tokens: Math.max(baseMax, budget + 1024),
    thinking: { type: 'enabled', budget_tokens: budget },
  };
}

export interface ChatCallOpts extends CallOpts {
  messages: Message[];
}

export interface StreamCallOpts extends CallOpts {
  messages: Message[];
}

export interface RunToolsOpts extends CallOpts {
  messages: Message[];
  tools: Tool[];
  /** Hard cap on iterations. Default 25. */
  maxIterations?: number;
  /** Stop the loop if total cost exceeds this (USD). Default Infinity. */
  costCapUSD?: number;
  /** Called after each tool call. Use for live logging / progress. */
  onStep?: (step: ToolStep) => void;
  /** Name of a TERMINAL tool. When the model calls it, the loop runs that tool
   *  then stops (stop='final_answer'), returning the tool's output as `text`.
   *  The Module 7 agent uses this for its `finalize` tool — an explicit "I'm
   *  done" signal, rather than relying on the model emitting a bare text turn. */
  terminalToolName?: string;
}

export interface StructuredOpts<Schema extends z.ZodObject<z.ZodRawShape>> extends CallOpts {
  messages: Message[];
  /** Zod schema the output must conform to. */
  schema: Schema;
  /** Optional override for the synthetic tool name (default 'output'). */
  outputToolName?: string;
}

// ============================================================================
// Factory
// ============================================================================

export function createClaude(config: ClaudeConfig = {}) {
  const sdk = new Anthropic({
    apiKey: config.apiKey ?? process.env.ANTHROPIC_API_KEY,
    maxRetries: config.maxRetries ?? 2,
  });
  // Haiku as default = cheapest by default. Bump to `CLAUDE_MODELS.sonnet` or
  // `.opus` when a specific call needs more brains (override via `opts.model`).
  const defaultModel = config.defaultModel ?? CLAUDE_MODELS.haiku;
  const defaultMaxTokens = config.defaultMaxTokens ?? 4096;
  const tracer = config.tracer ?? noopTracer;

  // --------------------------------------------------------------------------
  // chat — single non-streaming call
  // --------------------------------------------------------------------------
  /**
   * Non-streaming. Returns the assistant's text plus accounting. Use for short
   * answers, classification, anything where streaming is overkill.
   */
  async function chat(opts: ChatCallOpts): Promise<ChatResult> {
    const model = opts.model ?? defaultModel;
    const t0 = Date.now();

    safeCall(tracer, 'onRequest', {
      provider: 'anthropic',
      model,
      operation: 'chat',
      messageCount: opts.messages.length,
    });

    try {
      const response = await sdk.messages.create({
        model,
        ...buildReasoningParams(opts.reasoning, opts.maxTokens, defaultMaxTokens),
        ...(opts.temperature !== undefined && !opts.reasoning && { temperature: opts.temperature }),
        ...(opts.system !== undefined && {
          system: makeSystem(opts.system, opts.cacheSystem ?? false),
        }),
        messages: opts.messages.map(toAnthropicMessage),
      });

      const usage = extractUsage(response.usage);
      const cost = calculateAnthropicCost(usage, model);
      const stopReason = mapStopReason(response.stop_reason);
      const text = extractText(response.content);
      const reasoning = extractThinking(response.content);

      safeCall(tracer, 'onResponse', {
        usage,
        cost,
        latencyMs: Date.now() - t0,
        stopReason: response.stop_reason ?? 'unknown',
      });

      return { text, reasoning, usage, cost, stopReason, raw: response };
    } catch (err) {
      safeCall(tracer, 'onError', err, { operation: 'chat' });
      throw err;
    }
  }

  // --------------------------------------------------------------------------
  // streamText — high-level streaming as async iterable of strings
  // --------------------------------------------------------------------------
  /**
   * Yields text deltas as the model generates them. Use when you want to print
   * to a terminal or stream to a UI without dealing with event types.
   *
   * For tool use or detailed event handling, use `stream()` — this filters to
   * text deltas only.
   *
   * Anthropic streaming events (for reference):
   *   message_start, content_block_start, content_block_delta,
   *   content_block_stop, message_delta, message_stop.
   * Text comes via `content_block_delta` events with `delta.type === 'text_delta'`.
   */
  async function* streamText(opts: StreamCallOpts): AsyncGenerator<string, void, undefined> {
    const model = opts.model ?? defaultModel;
    const t0 = Date.now();

    safeCall(tracer, 'onRequest', {
      provider: 'anthropic',
      model,
      operation: 'streamText',
      messageCount: opts.messages.length,
    });

    try {
      const stream = sdk.messages.stream({
        model,
        ...buildReasoningParams(opts.reasoning, opts.maxTokens, defaultMaxTokens),
        ...(opts.temperature !== undefined && !opts.reasoning && { temperature: opts.temperature }),
        ...(opts.system !== undefined && {
          system: makeSystem(opts.system, opts.cacheSystem ?? false),
        }),
        messages: opts.messages.map(toAnthropicMessage),
      });

      for await (const event of stream) {
        if (event.type === 'content_block_delta' && event.delta.type === 'text_delta') {
          yield event.delta.text;
        }
      }

      const final = await stream.finalMessage();
      const usage = extractUsage(final.usage);
      const cost = calculateAnthropicCost(usage, model);
      safeCall(tracer, 'onResponse', {
        usage,
        cost,
        latencyMs: Date.now() - t0,
        stopReason: final.stop_reason ?? 'unknown',
      });
    } catch (err) {
      safeCall(tracer, 'onError', err, { operation: 'streamText' });
      throw err;
    }
  }

  // --------------------------------------------------------------------------
  // stream — raw event stream
  // --------------------------------------------------------------------------
  /**
   * Returns the SDK's stream object directly. Use when you need access to
   * non-text events (tool_use deltas, message_delta with final stop_reason,
   * usage info, etc.).
   *
   * The SDK's stream is iterable AND emits events; see @anthropic-ai/sdk docs.
   */
  function stream(opts: StreamCallOpts) {
    const model = opts.model ?? defaultModel;
    safeCall(tracer, 'onRequest', {
      provider: 'anthropic',
      model,
      operation: 'stream',
      messageCount: opts.messages.length,
    });
    return sdk.messages.stream({
      model,
      ...buildReasoningParams(opts.reasoning, opts.maxTokens, defaultMaxTokens),
      ...(opts.temperature !== undefined && !opts.reasoning && { temperature: opts.temperature }),
      ...(opts.system !== undefined && {
        system: makeSystem(opts.system, opts.cacheSystem ?? false),
      }),
      messages: opts.messages.map(toAnthropicMessage),
    });
  }

  // --------------------------------------------------------------------------
  // runTools — agentic loop
  // --------------------------------------------------------------------------
  /**
   * Iterative tool-use loop. Each iteration:
   *   1. Send the current message history + tools to the model.
   *   2. If `stop_reason === 'tool_use'`, execute each tool_use content block,
   *      append matching tool_result content blocks, loop.
   *   3. Otherwise, the model's text content IS the final answer — return it.
   *
   * Stops on:
   *   - 'final_answer'   — model returned text and is done
   *   - 'max_iterations' — safety cap (default 25)
   *   - 'cost_cap'       — total cost exceeded the caller's budget
   *
   * Why we accumulate messages here rather than the SDK: the conversation
   * state is OURS, not the SDK's. Anthropic's API is stateless on the server —
   * the client owns the full message history and resends it every turn.
   *
   * Why tool_result blocks live in a USER turn: that's how Anthropic structures
   * it — tool_use blocks go in the assistant turn, tool_result blocks in the
   * following user turn. Each tool_use needs exactly one matching tool_result
   * before the next assistant turn.
   */
  async function runTools(opts: RunToolsOpts): Promise<ToolLoopResult> {
    const model = opts.model ?? defaultModel;
    const maxIterations = opts.maxIterations ?? 25;
    const costCapUSD = opts.costCapUSD ?? Number.POSITIVE_INFINITY;
    const anthropicTools = opts.tools.map(toAnthropicTool);

    // Message history is Anthropic-shaped from the start to avoid re-translating.
    // Content can be a string (initial messages) OR a content-block array (after
    // the first assistant turn).
    type AnthMessage = { role: 'user' | 'assistant'; content: unknown };
    const messages: AnthMessage[] = opts.messages.map((m) => ({
      role: m.role === 'assistant' ? ('assistant' as const) : ('user' as const),
      content: m.content,
    }));

    const steps: ToolStep[] = [];
    let totalUsage: Usage = { inputTokens: 0, outputTokens: 0 };
    let totalCost: Cost = zeroCost();
    let stop: ToolLoopStop = 'max_iterations';
    let finalText = '';

    for (let iteration = 1; iteration <= maxIterations; iteration++) {
      if (totalCost.totalUSD > costCapUSD) {
        stop = 'cost_cap';
        break;
      }

      safeCall(tracer, 'onRequest', {
        provider: 'anthropic',
        model,
        operation: 'runTools',
        messageCount: messages.length,
        toolCount: anthropicTools.length,
      });

      const t0 = Date.now();
      const response = await sdk.messages.create({
        model,
        ...buildReasoningParams(opts.reasoning, opts.maxTokens, defaultMaxTokens),
        ...(opts.temperature !== undefined && !opts.reasoning && { temperature: opts.temperature }),
        ...(opts.system !== undefined && {
          system: makeSystem(opts.system, opts.cacheSystem ?? false),
        }),
        messages: messages as Anthropic.MessageParam[],
        tools: anthropicTools,
      });

      const usage = extractUsage(response.usage);
      const cost = calculateAnthropicCost(usage, model);
      totalUsage = addUsage(totalUsage, usage);
      totalCost = addCost(totalCost, cost);

      safeCall(tracer, 'onResponse', {
        usage,
        cost,
        latencyMs: Date.now() - t0,
        stopReason: response.stop_reason ?? 'unknown',
      });

      // Append the assistant turn verbatim — Anthropic REQUIRES the exact
      // tool_use blocks (with their IDs) when we send back tool_result blocks.
      messages.push({ role: 'assistant', content: response.content });

      if (response.stop_reason !== 'tool_use') {
        finalText = extractText(response.content);
        stop = 'final_answer';
        break;
      }

      // Pull out tool_use blocks (filtering also narrows the type).
      const toolUseBlocks = response.content.filter(
        (b): b is Extract<typeof b, { type: 'tool_use' }> => b.type === 'tool_use',
      );

      // One user turn carries ALL tool_results from this iteration's tool_uses.
      const toolResultsContent: Array<{
        type: 'tool_result';
        tool_use_id: string;
        content: string;
      }> = [];

      let terminalArticle: string | null = null;
      for (const block of toolUseBlocks) {
        const tool = findTool(opts.tools, block.name);
        const tStart = Date.now();
        const output = await executeTool(tool, block.input);
        const toolLatencyMs = Date.now() - tStart;

        safeCall(tracer, 'onToolCall', {
          iteration,
          toolName: block.name,
          toolInput: block.input,
          toolOutput: output,
          latencyMs: toolLatencyMs,
        });

        toolResultsContent.push({
          type: 'tool_result',
          tool_use_id: block.id,
          content: output,
        });

        const step: ToolStep = {
          iteration,
          toolName: block.name,
          toolInput: block.input,
          toolOutput: output,
          usage,
          cost,
        };
        steps.push(step);
        opts.onStep?.(step);

        if (opts.terminalToolName && block.name === opts.terminalToolName) {
          terminalArticle = output;
        }
      }

      // Keep the transcript valid (every tool_use gets its tool_result) even
      // when we're about to stop, so `messages` is inspectable.
      messages.push({ role: 'user', content: toolResultsContent });

      // Terminal tool called this turn → stop, returning its output as the final
      // text (no further round-trip).
      if (terminalArticle !== null) {
        finalText = terminalArticle;
        stop = 'final_answer';
        break;
      }
    }

    // Budget exhausted without a terminal answer. If the caller runs an
    // agent-style loop (terminalToolName set), make ONE final tool-free call so
    // the run still returns an article instead of '' — a loop that over-searches
    // to the cap would otherwise score as a non-answer. `stop` stays
    // 'max_iterations' (honest about WHY it ended); the model just synthesizes
    // from what it already gathered, since with no tools it cannot search again.
    if (stop === 'max_iterations' && opts.terminalToolName) {
      safeCall(tracer, 'onRequest', {
        provider: 'anthropic',
        model,
        operation: 'runTools:synthesize',
        messageCount: messages.length,
        toolCount: 0,
      });
      const tS = Date.now();
      const response = await sdk.messages.create({
        model,
        ...buildReasoningParams(opts.reasoning, opts.maxTokens, defaultMaxTokens),
        ...(opts.temperature !== undefined && !opts.reasoning && { temperature: opts.temperature }),
        ...(opts.system !== undefined && {
          system: makeSystem(opts.system, opts.cacheSystem ?? false),
        }),
        messages: messages as Anthropic.MessageParam[],
        // No `tools` — force a text answer from what was already gathered.
      });
      const usage = extractUsage(response.usage);
      const cost = calculateAnthropicCost(usage, model);
      totalUsage = addUsage(totalUsage, usage);
      totalCost = addCost(totalCost, cost);
      safeCall(tracer, 'onResponse', {
        usage,
        cost,
        latencyMs: Date.now() - tS,
        stopReason: response.stop_reason ?? 'unknown',
      });
      messages.push({ role: 'assistant', content: response.content });
      finalText = extractText(response.content);
    }

    return { text: finalText, usage: totalUsage, cost: totalCost, stop, steps, messages };
  }

  // --------------------------------------------------------------------------
  // structured — guaranteed-shape JSON via tool-use-as-output
  // --------------------------------------------------------------------------
  /**
   * Returns model output conforming to a Zod schema.
   *
   * Implementation trick: define ONE tool whose input_schema is the desired
   * output shape, force `tool_choice` to it, and read the resulting tool_use
   * block's input as the result. This beats prompt-engineered JSON mode because:
   *   - The model uses tool-use machinery, which is RL-trained for schema
   *     conformance — way more reliable than JSON-in-text.
   *   - The result is already validated against the schema with full TS types.
   *   - Failures (rare) are surfaced explicitly via parse error.
   */
  // biome-ignore lint/suspicious/noExplicitAny: see types.ts Tool definition
  async function structured<Schema extends z.ZodObject<any>>(
    opts: StructuredOpts<Schema>,
  ): Promise<StructuredResult<z.infer<Schema>>> {
    const model = opts.model ?? defaultModel;
    const outputToolName = opts.outputToolName ?? 'output';

    const outputTool = defineTool({
      name: outputToolName,
      description: 'Return your final structured output via this tool.',
      schema: opts.schema,
      execute: () => '', // never actually executed; we only need the schema
    });

    safeCall(tracer, 'onRequest', {
      provider: 'anthropic',
      model,
      operation: 'structured',
      messageCount: opts.messages.length,
      toolCount: 1,
    });

    const t0 = Date.now();
    const response = await sdk.messages.create({
      model,
      ...buildReasoningParams(opts.reasoning, opts.maxTokens, defaultMaxTokens),
      ...(opts.temperature !== undefined && !opts.reasoning && { temperature: opts.temperature }),
      ...(opts.system !== undefined && {
        system: makeSystem(opts.system, opts.cacheSystem ?? false),
      }),
      messages: opts.messages.map(toAnthropicMessage),
      tools: [toAnthropicTool(outputTool)],
      // FORCE the model to call this specific tool. Strongest schema guarantee
      // Anthropic's API offers — no free-form text response possible.
      tool_choice: { type: 'tool', name: outputToolName },
    });

    const usage = extractUsage(response.usage);
    const cost = calculateAnthropicCost(usage, model);
    safeCall(tracer, 'onResponse', {
      usage,
      cost,
      latencyMs: Date.now() - t0,
      stopReason: response.stop_reason ?? 'unknown',
    });

    const block = response.content.find(
      (b): b is Extract<typeof b, { type: 'tool_use' }> => b.type === 'tool_use',
    );
    if (!block) {
      throw new Error(
        'Expected a tool_use content block in structured output, got none. ' +
          'This indicates an API change or unexpected stop_reason.',
      );
    }

    const parsed = opts.schema.safeParse(block.input);
    if (!parsed.success) {
      // Include the raw tool input so debugging doesn't require re-running.
      throw new Error(
        `Structured output failed schema validation: ${parsed.error.message}\n` +
          `Model produced: ${JSON.stringify(block.input, null, 2)}`,
      );
    }

    return {
      data: parsed.data,
      reasoning: extractThinking(response.content),
      usage,
      cost,
      raw: response,
    };
  }

  return { chat, streamText, stream, runTools, structured };
}

// ============================================================================
// Internal helpers (not exported — implementation detail)
// ============================================================================

/**
 * Build the `system` parameter the SDK expects. When `cache` is true, we pass
 * an array containing a text block with `cache_control` — telling the API
 * server to cache this prompt prefix. Next request with the same prefix is
 * billed at the cache_read rate (~0.1× input).
 */
function makeSystem(text: string, cache: boolean) {
  if (!cache) return text;
  return [{ type: 'text' as const, text, cache_control: { type: 'ephemeral' as const } }];
}

function toAnthropicMessage(m: Message): { role: 'user' | 'assistant'; content: string } {
  // Anthropic's messages array supports only 'user' and 'assistant'. 'system'
  // goes on the top-level `system` param (caller should set opts.system instead).
  const role: 'user' | 'assistant' = m.role === 'assistant' ? 'assistant' : 'user';
  return { role, content: m.content };
}

/** Pull text out of a content block array (ignores tool_use blocks). */
function extractText(content: ReadonlyArray<{ type: string }>): string {
  return content
    .filter((b): b is { type: 'text'; text: string } => b.type === 'text')
    .map((b) => b.text)
    .join('');
}

/**
 * Pull the model's chain-of-thought out of a content block array.
 *
 * When extended thinking is enabled, Anthropic prepends one or more `thinking`
 * blocks before the visible `text` blocks. Each carries the private reasoning
 * (and a `signature` we don't need here). Returns undefined when no thinking
 * blocks are present — so callers can distinguish "thinking was off" from
 * "thinking ran but was empty."
 *
 * Note: `redacted_thinking` blocks (rare; appear when safety filters intercept
 * a chain of thought) are deliberately NOT surfaced — their `data` field is an
 * opaque ciphertext, not human-readable text.
 */
function extractThinking(content: ReadonlyArray<{ type: string }>): string | undefined {
  const thinking = content
    .filter((b): b is { type: 'thinking'; thinking: string } => b.type === 'thinking')
    .map((b) => b.thinking)
    .join('');
  return thinking.length > 0 ? thinking : undefined;
}

function extractUsage(u: {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens?: number | null;
  cache_read_input_tokens?: number | null;
}): Usage {
  return {
    inputTokens: u.input_tokens,
    outputTokens: u.output_tokens,
    cacheCreationTokens: u.cache_creation_input_tokens ?? undefined,
    cacheReadTokens: u.cache_read_input_tokens ?? undefined,
  };
}

function mapStopReason(reason: string | null | undefined): StopReason {
  switch (reason) {
    case 'end_turn':
    case 'max_tokens':
    case 'tool_use':
    case 'stop_sequence':
      return reason;
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
