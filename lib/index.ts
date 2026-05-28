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
export type {
  ClaudeModel,
  LMStudioEmbeddingModel,
  LMStudioModel,
  OpenAIEmbeddingModel,
} from './cost.ts';
export {
  ANTHROPIC_PRICES,
  addCost,
  addUsage,
  CLAUDE_MODELS,
  calculateAnthropicCost,
  calculateLMStudioCost,
  calculateLMStudioEmbeddingCost,
  calculateOpenAIEmbeddingCost,
  EMBEDDING_DIMENSIONS,
  formatCost,
  LM_STUDIO_EMBEDDING_MODELS,
  LM_STUDIO_MODELS,
  LMSTUDIO_PRICING,
  OPENAI_EMBEDDING_MODELS,
  OPENAI_EMBEDDING_PRICES,
  PRICES_AS_OF,
} from './cost.ts';
export type {
  Embedder,
  EmbedderConfig,
  EmbeddingProvider,
  EmbedOpts,
  EmbedResult,
  LMStudioEmbedderConfig,
  OpenAIEmbedderConfig,
} from './embeddings.ts';
export { createEmbedder } from './embeddings.ts';
export type {
  LMStudioCallOpts,
  LMStudioChatOpts,
  LMStudioConfig,
  LMStudioRunToolsOpts,
  LMStudioStructuredOpts,
} from './lmstudio.ts';
export { createLMStudio } from './lmstudio.ts';
export type { CreateLocalLLMConfig, LocalLLM, LocalProvider } from './local-llm.ts';
export { createLocalLLM } from './local-llm.ts';
export type { OllamaModel } from './ollama.ts';
export { createOllama, OLLAMA_MODELS } from './ollama.ts';
export type {
  AbTestCase,
  AbTestPrompt,
  AbTestRow,
  ExtractVars,
  Prompt,
  PromptDef,
  PromptMessage,
  PromptVars,
} from './prompts.ts';
export { abTest, definePrompt, formatAbTest, tag, tags } from './prompts.ts';
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
