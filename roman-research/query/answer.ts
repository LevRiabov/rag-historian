/**
 * roman-research/query/answer.ts — format retrieved chunks into a prompt
 * and generate a cited answer with either llama.cpp (local, free, default)
 * or Claude (paid, higher quality, opt-in).
 *
 * Why llama.cpp by default: the user iterates on prompts and retrieval
 * configs many times during Modules 4–6. Local generation is free and
 * fast on the 5070 Ti, so cost doesn't gate experimentation. When the
 * eval set in Module 5 compares quality, we'll re-run with Claude.
 *
 * Prompt design (naive RAG baseline; Module 6.x will iterate):
 *
 *   System: persona (historian's assistant), citation rule, refuse-if-
 *           unsupported rule, contradiction-acknowledgement hint.
 *
 *   User:   the question + a numbered list of retrieved chunks, each
 *           tagged with author, work, chapter, year.
 *
 * The numbered list lets the model emit [N] markers that the rendering
 * layer turns back into citations. Numbering is by ARRIVAL order from
 * the retriever (not by similarity rank — same thing in practice, but
 * arrival-order is easier to reason about during prompt iteration).
 *
 * Both wrappers expose the same `chat({ system, messages })` interface
 * and return the same `ChatResult` shape, so dispatch is trivial.
 */
import {
  CLAUDE_MODELS,
  type ClaudeModel,
  createClaude,
  createLlamacpp,
  formatCost,
  LLAMACPP_MODELS,
} from '../../lib/index.ts';
import { formatYear, type RetrievedChunk } from './retrieve.ts';

const SYSTEM_PROMPT = `You are a research assistant for Roman history, specifically Julius Caesar's career and death.

Rules:
1. Answer ONLY from the provided source passages. Do not introduce facts not present in them.
2. Cite every factual claim with [N] markers matching the source numbers in the user's message. Multiple markers per claim are fine when supported by multiple sources, e.g., "Caesar crossed the Rubicon in 49 BC [1][3]."
3. When sources disagree on a fact, surface the disagreement explicitly. Name which source says what — do not paper over contradictions with a synthesized answer.
4. If the sources do not contain enough information to answer, say so plainly. Do not speculate.
5. Be concise. Prefer two or three substantive paragraphs over a long survey.`;

/** LLM choices: local (free, default) or Claude (paid). */
export type LLMChoice = 'llamacpp' | 'claude-sonnet' | 'claude-haiku' | 'claude-opus';

export const DEFAULT_LLM: LLMChoice = 'llamacpp';

const CLAUDE_MODEL_BY_CHOICE: Record<Exclude<LLMChoice, 'llamacpp'>, ClaudeModel> = {
  'claude-sonnet': CLAUDE_MODELS.sonnet,
  'claude-haiku': CLAUDE_MODELS.haiku,
  'claude-opus': CLAUDE_MODELS.opus,
};

export interface AnswerOptions {
  llm?: LLMChoice;
  /** Override the llama-swap model profile (only used when llm='llamacpp').
   *  Defaults to qwen-9b-16k (dense 9B, thinking off, strong at extracting
   *  from chunks). Pass any profile name in the llama-swap config, e.g.
   *  'qwen-9b-32k' for a larger context. */
  llamacppModel?: string;
}

export interface AnswerResult {
  text: string;
  /** Display string from formatCost (always populated, $0 for local). */
  costFormatted: string;
  rawCostUSD: number;
  inputTokens: number;
  outputTokens: number;
  latencyMs: number;
  /** Pretty label like "Claude (claude-sonnet-4-6)" or "llama.cpp (qwen-9b-16k)". */
  llmLabel: string;
}

/**
 * Render the user message: question + numbered source passages.
 * The model will reference these passages by [N] in its answer.
 */
export function formatUserMessage(question: string, chunks: RetrievedChunk[]): string {
  if (chunks.length === 0) {
    return `Question: ${question}\n\n(No sources were retrieved for this question.)`;
  }
  const lines: string[] = ['Sources:', ''];
  for (let i = 0; i < chunks.length; i++) {
    const c = chunks[i];
    if (!c) continue;
    const n = i + 1;
    const meta = `${c.source.author}, ${c.source.title} (${formatYear(c.source.yearWritten)}), ${c.chapter}`;
    lines.push(`[${n}] ${meta}`);
    lines.push(c.text.trim());
    lines.push('');
  }
  lines.push(`Question: ${question}`);
  return lines.join('\n');
}

export async function answerQuestion(
  question: string,
  chunks: RetrievedChunk[],
  options: AnswerOptions = {},
): Promise<AnswerResult> {
  const llm = options.llm ?? DEFAULT_LLM;
  const userMessage = formatUserMessage(question, chunks);
  const t0 = Date.now();

  if (llm === 'llamacpp') {
    // Local generation via llama.cpp (llama-swap). Default profile qwen-9b-16k:
    // dense 9B is more consistent at extracting facts from multi-chunk context
    // than a sparse MoE that tends to skim. Thinking is OFF — but that's baked
    // into the profile (`--reasoning off` in the llama-swap config), NOT a
    // per-request flag, so we don't pass `reasoning` here. (qwen3 defaults to
    // thinking on, which writes a long CoT that exhausts the token budget
    // before producing the answer; the profile suppresses it.) Override the
    // profile via options.llamacppModel.
    const model = options.llamacppModel ?? LLAMACPP_MODELS.qwen9b16k;
    const client = createLlamacpp({ defaultModel: model });
    const result = await client.chat({
      system: SYSTEM_PROMPT,
      messages: [{ role: 'user', content: userMessage }],
    });
    return {
      text: result.text,
      costFormatted: formatCost(result.cost),
      rawCostUSD: result.cost.totalUSD,
      inputTokens: result.usage.inputTokens,
      outputTokens: result.usage.outputTokens,
      latencyMs: Date.now() - t0,
      llmLabel: `llama.cpp (${model})`,
    };
  }

  const model = CLAUDE_MODEL_BY_CHOICE[llm];
  const claude = createClaude({ defaultModel: model });
  const result = await claude.chat({
    system: SYSTEM_PROMPT,
    messages: [{ role: 'user', content: userMessage }],
  });
  return {
    text: result.text,
    costFormatted: formatCost(result.cost),
    rawCostUSD: result.cost.totalUSD,
    inputTokens: result.usage.inputTokens,
    outputTokens: result.usage.outputTokens,
    latencyMs: Date.now() - t0,
    llmLabel: `Claude (${model})`,
  };
}
