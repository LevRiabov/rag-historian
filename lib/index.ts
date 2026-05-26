/**
 * lib/index.ts — public surface.
 *
 * Single import point for the rest of the project:
 *   import { createClaude, createLMStudio, defineTool } from '../lib/index.ts';
 *
 * Implementation details (types, helpers) live in their own files; this file
 * exists only to control what's exported.
 */

export type {
  CallOpts,
  ChatCallOpts,
  ClaudeConfig,
  RunToolsOpts,
  StreamCallOpts,
  StructuredOpts,
} from './claude.ts';
export { createClaude } from './claude.ts';
export type { ClaudeModel } from './cost.ts';
export {
  ANTHROPIC_PRICES,
  addCost,
  addUsage,
  CLAUDE_MODELS,
  calculateAnthropicCost,
  calculateLMStudioCost,
  formatCost,
  LMSTUDIO_PRICING,
  PRICES_AS_OF,
} from './cost.ts';
export type {
  LMStudioCallOpts,
  LMStudioChatOpts,
  LMStudioConfig,
  LMStudioRunToolsOpts,
  LMStudioStructuredOpts,
} from './lmstudio.ts';
export { createLMStudio } from './lmstudio.ts';
export type { AnthropicTool, OpenAITool } from './tools.ts';
export { defineTool } from './tools.ts';
export type { RequestInfo, ResponseInfo, ToolCallInfo, Tracer } from './tracer.ts';
export { noopTracer } from './tracer.ts';

export type {
  ChatResult,
  Cost,
  Message,
  Role,
  StopReason,
  StructuredResult,
  Tool,
  ToolLoopResult,
  ToolLoopStop,
  ToolStep,
  Usage,
} from './types.ts';
