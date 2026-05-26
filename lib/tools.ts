/**
 * lib/tools.ts — defining and converting tools the model can call.
 *
 * Why Zod (not raw JSON Schema): a tool has TWO halves that must agree —
 *   1. The schema sent to the model (so it knows the contract).
 *   2. The runtime validation of the model's arguments (catches hallucinated
 *      input — strong models still produce malformed calls occasionally).
 * Zod gives both from one declaration, plus `z.infer<Schema>` for typing the
 * `execute` function. One source of truth.
 *
 * Why TWO provider converters (Anthropic + OpenAI): the SDKs frame tools
 * differently.
 *   Anthropic:    { name, description, input_schema }
 *   OpenAI/compat:{ type: 'function', function: { name, description, parameters } }
 * Both want JSON Schema for the schema half. Zod 4's `z.toJSONSchema` returns
 * one natively — no `zod-to-json-schema` adapter needed since v4.
 *
 * Why `executeTool` returns errors AS STRINGS rather than throwing: the agentic
 * loop should keep going. Returning the error to the model as a tool_result
 * gives it a chance to recover (e.g., fix arguments and retry). Throwing
 * would abort the loop on any bad call.
 */
import { z } from 'zod';
import type { Tool } from './types.ts';

/**
 * Identity-in-code factory for a tool. Identity, but the generic Schema flows
 * from the call-site literal, so `execute({ ... })` has typed input without
 * manual annotation. Lighter than a class; readable as a literal.
 */
export function defineTool<Schema extends z.ZodObject<z.ZodRawShape>>(
  config: Tool<Schema>,
): Tool<Schema> {
  return config;
}

/**
 * Anthropic's SDK requires `input_schema.type` to be the literal `'object'`.
 * Zod's `z.toJSONSchema(z.object({...}))` always emits exactly that at runtime,
 * but we tell the type system explicitly so the SDK's overload resolves.
 */
export interface AnthropicTool {
  name: string;
  description: string;
  input_schema: {
    type: 'object';
    properties?: Record<string, unknown>;
    required?: string[];
    [k: string]: unknown;
  };
}

export interface OpenAITool {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

/** Convert canonical Tool → Anthropic SDK shape. */
export function toAnthropicTool(tool: Tool): AnthropicTool {
  return {
    name: tool.name,
    description: tool.description,
    input_schema: schemaToJSON(tool.schema),
  };
}

/** Convert canonical Tool → OpenAI / LM Studio shape. */
export function toOpenAITool(tool: Tool): OpenAITool {
  return {
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: schemaToJSON(tool.schema),
    },
  };
}

/**
 * Strip the `$schema` header from Zod's JSON Schema output — neither provider
 * wants it, and some validators reject unknown top-level keys. The runtime
 * value is always `{ type: 'object', ... }` for an object schema; the cast
 * tells TS that fact.
 */
function schemaToJSON(schema: z.ZodTypeAny): AnthropicTool['input_schema'] {
  const out = { ...(z.toJSONSchema(schema) as Record<string, unknown>) };
  delete out.$schema;
  return out as AnthropicTool['input_schema'];
}

/**
 * Look up a tool by name. Throws if the model invented a tool that doesn't
 * exist — surface loudly so we notice the hallucination.
 */
export function findTool(tools: Tool[], name: string): Tool {
  const tool = tools.find((t) => t.name === name);
  if (!tool) {
    throw new Error(
      `Model called unknown tool "${name}". Available: ${tools.map((t) => t.name).join(', ')}`,
    );
  }
  return tool;
}

/**
 * Validate and execute one tool call. Returns the result as a string the model
 * can read on the next turn. Errors are returned as strings (NOT thrown) so the
 * agentic loop can continue and let the model recover.
 */
export async function executeTool(tool: Tool, input: unknown): Promise<string> {
  const parsed = tool.schema.safeParse(input);
  if (!parsed.success) {
    return `ERROR: Invalid input for tool "${tool.name}". ${parsed.error.message}`;
  }
  try {
    const result = await tool.execute(parsed.data);
    return typeof result === 'string' ? result : JSON.stringify(result);
  } catch (err) {
    return `ERROR executing tool "${tool.name}": ${err instanceof Error ? err.message : String(err)}`;
  }
}
