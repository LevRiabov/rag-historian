/**
 * roman-research/query/answer.ts — format retrieved chunks into a prompt
 * and generate a cited answer with either LM Studio (local, free, default)
 * or Claude (paid, higher quality, opt-in).
 *
 * Why LM Studio by default: the user iterates on prompts and retrieval
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
  createLocalLLM,
  formatCost,
  LM_STUDIO_MODELS,
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
export type LLMChoice = 'lmstudio' | 'claude-sonnet' | 'claude-haiku' | 'claude-opus';

export const DEFAULT_LLM: LLMChoice = 'lmstudio';

const CLAUDE_MODEL_BY_CHOICE: Record<Exclude<LLMChoice, 'lmstudio'>, ClaudeModel> = {
  'claude-sonnet': CLAUDE_MODELS.sonnet,
  'claude-haiku': CLAUDE_MODELS.haiku,
  'claude-opus': CLAUDE_MODELS.opus,
};

export interface AnswerOptions {
  llm?: LLMChoice;
  /** Override the LM Studio model identifier (only used when llm='lmstudio').
   *  Defaults to qwen3.5-9b (dense, strong at extracting from chunks).
   *  Pass any string LM Studio recognizes (e.g. 'openai/gpt-oss-20b'). */
  lmStudioModel?: string;
}

export interface AnswerResult {
  text: string;
  /** Display string from formatCost (always populated, $0 for local). */
  costFormatted: string;
  rawCostUSD: number;
  inputTokens: number;
  outputTokens: number;
  latencyMs: number;
  /** Pretty label like "Claude (claude-sonnet-4-6)" or "LM Studio (openai/gpt-oss-20b)". */
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

  if (llm === 'lmstudio') {
    // createLocalLLM is env-driven (LOCAL_LLM_PROVIDER). Defaults to LM Studio.
    // Default model: qwen3.5-9b — dense 9B is more consistent at extracting
    // facts from multi-chunk context than gpt-oss-20b (which is MoE with
    // only 3.6B active params and tends to skim). Caller can override via
    // options.lmStudioModel to compare models.
    const local = createLocalLLM({
      lmstudio: { defaultModel: options.lmStudioModel ?? LM_STUDIO_MODELS.qwen3_5_9b },
    });
    const result = await local.client.chat({
      system: SYSTEM_PROMPT,
      messages: [{ role: 'user', content: userMessage }],
      // Disable thinking. qwen3 family defaults to on, which writes a long
      // chain of thought into `message.reasoning` and often exhausts the
      // max_tokens budget before producing any final `message.content`.
      // For RAG-answer generation we want the direct answer, not the
      // intermediate reasoning. (gpt-oss-20b uses leveled reasoning_effort
      // and ignores this flag; passing false is a no-op there.)
      reasoning: false,
    });
    return {
      text: result.text,
      costFormatted: formatCost(result.cost),
      rawCostUSD: result.cost.totalUSD,
      inputTokens: result.usage.inputTokens,
      outputTokens: result.usage.outputTokens,
      latencyMs: Date.now() - t0,
      llmLabel: local.label,
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
