/**
 * roman-research/agent/index.ts — the Roman Research Agent loop (Module 7,
 * Slice 2).
 *
 * Thin orchestration over the lib's `runTools` loop: assemble the five tools
 * (createAgentTools), pick a model, run the agentic loop with `finalize` as the
 * TERMINAL tool. The provider-specific protocol (Anthropic content blocks vs
 * OpenAI tool_calls), max-iterations, cost cap, tracing, and per-step records all
 * live in lib/{claude,lmstudio}.ts — the agent does NOT re-implement them.
 *
 * The Claude-vs-local A/B (Module 7's headline experiment) is just the `llm`
 * flag: both providers run the SAME tools + SAME system prompt through the SAME
 * ToolLoopResult shape; only the underlying runTools method differs.
 *
 * The system prompt is a Slice-3 iteration target — kept here for now; it teaches
 * a METHODOLOGY (match effort to the question; read each source separately for
 * contradictions; abstain as a first-class outcome) rather than a fixed recipe.
 */
import type { Client } from 'pg';

import {
  CLAUDE_MODELS,
  type ClaudeModel,
  type Cost,
  createClaude,
  createLlamacpp,
  formatCost,
  LLAMACPP_MODELS,
  noopTracer,
  type ToolLoopResult,
  type ToolLoopStop,
  type ToolStep,
  type Tracer,
  type Usage,
} from '../../lib/index.ts';
import type { RetrievedChunk } from '../query/retrieve.ts';
import { createAgentTools, FINALIZE_TOOL_NAME } from './tools.ts';

export const AGENT_SYSTEM_PROMPT = `You are a research assistant for Roman history, specifically Julius Caesar's career and death. You answer by RESEARCHING the corpus with tools, then writing a cited article. You do NOT answer from prior knowledge — only from what the tools return.

WHEN TO STOP (read first — do NOT over-research):
- The moment your searches return passages that let you cite the answer, call finalize. Do NOT search again "to be sure", and do NOT re-search a source you already searched — that wastes effort without improving the answer.
- Most questions need 1–3 searches TOTAL. A simple factual question usually needs a single search_corpus call.
- If a search returns nothing new or relevant, do not keep trying variations — finalize with what you have, or abstain.

METHODOLOGY (match effort to the question):
- Simple factual question → one search_corpus call, then finalize.
- Synthesis ("how did X develop", "what caused Y") → 2–4 searches on sub-questions, then unify the findings and finalize.
- Contradiction ("did Caesar want to be king?", "who started the war?") → call search_within_source ONCE per relevant source to read each account in isolation, then contrast them by name. NEVER blend conflicting accounts into one smoothed claim, and never search the same source twice.

RULES:
1. Cite every factual claim with [chunk_id] markers, using the ids returned by the search tools. Cite ONLY what the passages actually state — do NOT add facts, dates, original-language phrases, or causal/inferential claims beyond them. Attribute each claim to the SPECIFIC source that supports it; never mix up which source said what.
2. When sources disagree, surface it explicitly — name who says what (e.g. "Suetonius reports X, while Plutarch emphasizes Y").
3. If the corpus does not contain the answer, finalize with a plain statement that the sources do not cover it. Abstaining is a CORRECT outcome — never keep searching to manufacture an answer from thin evidence.
4. Be concise: two or three substantive paragraphs, not a survey.

Call finalize(article) as soon as you can support the answer with cited passages, or have confirmed the corpus does not contain it. finalize ENDS your turn.`;

/** Which model drives the loop. The Module 7 A/B is Claude vs local qwen. */
export type AgentLLM = 'claude-sonnet' | 'claude-haiku' | 'claude-opus' | 'llamacpp';

const CLAUDE_MODEL_BY_CHOICE: Record<Exclude<AgentLLM, 'llamacpp'>, ClaudeModel> = {
  'claude-sonnet': CLAUDE_MODELS.sonnet,
  'claude-haiku': CLAUDE_MODELS.haiku,
  'claude-opus': CLAUDE_MODELS.opus,
};

export interface RunAgentOptions {
  /** Driver model. Default 'claude-haiku' (the Module 6 generation anchor). */
  llm?: AgentLLM;
  /** Hard cap on agent iterations. Default 30. */
  maxIterations?: number;
  /** Kill the run if cumulative cost exceeds this (USD). Claude only (local is
   *  free). Default 0.50 per question. */
  costCapUSD?: number;
  /** llama-swap profile when llm='llamacpp'. Default qwen-9b-64k (thinking off):
   *  an agent loop accumulates a growing tool transcript, so it needs far more
   *  context than single-shot RAG — qwen over-searches and overflowed the 16k
   *  profile mid-run (Module 7). Try qwen-9b-16k-think to A/B CoT (but mind the
   *  smaller 16k window). */
  llamacppModel?: string;
  /** Observability hook (Langfuse in Slice 4). Default no-op. */
  tracer?: Tracer;
  /** Live per-tool-call callback (CLI logging / progress). */
  onStep?: (step: ToolStep) => void;
}

export interface AgentResult {
  /** The final article (finalize's `article`, or the model's last text). */
  article: string;
  /** Why the loop ended — 'final_answer' (incl. finalize), 'max_iterations',
   *  'cost_cap'. A non-final stop means the article may be empty/partial. */
  stop: ToolLoopStop;
  /** Total tool calls — the headline agent metric, per category in the eval. */
  toolCalls: number;
  /** Tool calls grouped by name (e.g. how often search_within_source fired). */
  toolCallsByName: Record<string, number>;
  steps: ToolStep[];
  /** Union of chunks surfaced by the agent's searches — the eval's faithfulness
   *  evidence base + gold-coverage source. */
  consultedChunks: RetrievedChunk[];
  /** Count of distinct chunks consulted (a cheap effort signal). */
  consultedCount: number;
  usage: Usage;
  cost: Cost;
  costFormatted: string;
  rawCostUSD: number;
  latencyMs: number;
  llmLabel: string;
}

/**
 * Run the agent on one question. Builds a fresh tool set (so per-run coverage
 * state in list_sources_consulted starts empty) and dispatches to the chosen
 * provider's runTools with finalize as the terminal tool.
 */
export async function runAgent(
  db: Client,
  question: string,
  options: RunAgentOptions = {},
): Promise<AgentResult> {
  const llm = options.llm ?? 'claude-haiku';
  const maxIterations = options.maxIterations ?? 30;
  const tracer = options.tracer ?? noopTracer;
  const { tools, getConsultedChunks } = createAgentTools(db);
  const messages = [{ role: 'user' as const, content: question }];
  const t0 = Date.now();

  if (llm === 'llamacpp') {
    const model = options.llamacppModel ?? LLAMACPP_MODELS.qwen9b64k;
    const client = createLlamacpp({ defaultModel: model, tracer });
    const result = await client.runTools({
      system: AGENT_SYSTEM_PROMPT,
      messages,
      tools,
      maxIterations,
      terminalToolName: FINALIZE_TOOL_NAME,
      onStep: options.onStep,
    });
    return toAgentResult(result, `llama.cpp (${model})`, t0, getConsultedChunks());
  }

  const model = CLAUDE_MODEL_BY_CHOICE[llm];
  const client = createClaude({ defaultModel: model, tracer });
  const result = await client.runTools({
    system: AGENT_SYSTEM_PROMPT,
    messages,
    tools,
    maxIterations,
    costCapUSD: options.costCapUSD ?? 0.5,
    terminalToolName: FINALIZE_TOOL_NAME,
    onStep: options.onStep,
  });
  return toAgentResult(result, `Claude (${model})`, t0, getConsultedChunks());
}

/** Map the generic loop result into the agent-facing shape + metrics. */
function toAgentResult(
  result: ToolLoopResult,
  llmLabel: string,
  t0: number,
  consultedChunks: RetrievedChunk[],
): AgentResult {
  const toolCallsByName: Record<string, number> = {};
  for (const step of result.steps) {
    toolCallsByName[step.toolName] = (toolCallsByName[step.toolName] ?? 0) + 1;
  }
  return {
    article: result.text,
    stop: result.stop,
    toolCalls: result.steps.length,
    toolCallsByName,
    steps: result.steps,
    consultedChunks,
    consultedCount: consultedChunks.length,
    usage: result.usage,
    cost: result.cost,
    costFormatted: formatCost(result.cost),
    rawCostUSD: result.cost.totalUSD,
    latencyMs: Date.now() - t0,
    llmLabel,
  };
}
