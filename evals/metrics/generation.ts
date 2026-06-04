/**
 * evals/metrics/generation.ts — LLM-as-judge implementations for
 * faithfulness, completeness, and refusal.
 *
 * Each judge is one structured-output call via the Claude wrapper. We use
 * Haiku as the default judge: cheap, fast, and reliable enough for grading
 * with a clear rubric. Per Hamel Husain's guidance, the judge model should
 * be ≥ the generator model in capability — Haiku >> our local gpt-oss-20b
 * generator for structured grading tasks.
 *
 * Calibration discipline: after a first batch run, hand-score ~10 random
 * judgments per dimension. If the judge disagrees with you on more than 20%,
 * tune the rubric BEFORE trusting aggregates. Calibration is the
 * one-thing-most-teams-skip per the eval guides.
 *
 * Cost (rough, on our 50-question golden set):
 *   - 50 answers × 3 judges = 150 Haiku calls
 *   - ~300k input + 35k output tokens
 *   - ≈ $0.50 per full --generation run
 */
import { z } from 'zod';

import { CLAUDE_MODELS, type ClaudeModel, createClaude } from '../../lib/index.ts';
import type { RetrievedChunk } from '../../roman-research/query/retrieve.ts';
import {
  COMPLETENESS_SYSTEM,
  completenessUser,
  FAITHFULNESS_SYSTEM,
  faithfulnessUser,
  REFUSAL_SYSTEM,
  refusalUser,
} from '../judge-prompts.ts';

// ============================================================================
// Schemas — drive Claude's structured-output mode + give us typed return values.
//
// `claimsField` coerces stringified arrays into real arrays. Even with
// response_format / tool-use constraints, judge models occasionally output
// `"unsupportedClaims": "[\"...\"]"` — a string holding a JSON-looking array
// — instead of a real array. We accept either shape rather than fail-and-retry.
// ============================================================================

const claimsField = z.preprocess((val) => {
  if (Array.isArray(val)) return val;
  if (typeof val === 'string') {
    const trimmed = val.trim();
    if (trimmed.length === 0 || /^(none|n\/a)$/i.test(trimmed)) return [];
    // Try to parse as JSON array first
    try {
      const parsed = JSON.parse(trimmed);
      if (Array.isArray(parsed)) return parsed.map((x) => String(x));
    } catch {
      // not JSON — fall through
    }
    // Last resort: wrap the whole string as one element
    return [trimmed];
  }
  return [];
}, z.array(z.string()));

const FaithfulnessSchema = z.object({
  score: z.number().int().min(1).max(5),
  reasoning: z.string(),
  unsupportedClaims: claimsField,
});

const CompletenessSchema = z.object({
  score: z.number().int().min(1).max(5),
  reasoning: z.string(),
  missedFacts: claimsField,
});

const RefusalSchema = z.object({
  didRefuse: z.boolean(),
  reasoning: z.string(),
});

// ============================================================================
// Retry helper — judge calls go over the network, transient failures
// (connection timeout, 5xx, rate limit) shouldn't crash a 30-minute eval.
// SDK-level retries cover some of this but the wrapper's `structured` path
// doesn't always propagate them cleanly, so we add an explicit layer.
// ============================================================================

const MAX_JUDGE_RETRIES = 2;
const RETRY_BASE_DELAY_MS = 1000;

function isTransient(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  const msg = err.message.toLowerCase();
  return (
    msg.includes('timeout') ||
    msg.includes('timed out') ||
    msg.includes('econnreset') ||
    msg.includes('rate limit') ||
    msg.includes('429') ||
    msg.includes('502') ||
    msg.includes('503') ||
    msg.includes('504')
  );
}

async function withRetry<T>(fn: () => Promise<T>, label: string): Promise<T> {
  for (let attempt = 0; attempt <= MAX_JUDGE_RETRIES; attempt++) {
    try {
      return await fn();
    } catch (err) {
      if (attempt === MAX_JUDGE_RETRIES || !isTransient(err)) throw err;
      const delay = RETRY_BASE_DELAY_MS * 2 ** attempt;
      const msg = err instanceof Error ? err.message : String(err);
      console.warn(
        `    [${label}] transient error (${msg.slice(0, 80)}), retrying in ${delay}ms...`,
      );
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
  throw new Error('unreachable');
}

// ============================================================================
// Judge factory — one Claude instance reused across all calls in a run, so
// the connection / SDK overhead is paid once.
// ============================================================================

export interface Judge {
  faithfulness(
    question: string,
    chunks: RetrievedChunk[],
    answerText: string,
  ): Promise<{ score: number; reasoning: string; unsupportedClaims: string[]; costUSD: number }>;

  completeness(
    question: string,
    idealAnswer: string,
    candidateAnswer: string,
  ): Promise<{ score: number; reasoning: string; missedFacts: string[]; costUSD: number }>;

  refusal(answerText: string): Promise<{ didRefuse: boolean; reasoning: string; costUSD: number }>;

  /** Pretty label, e.g. "Claude (claude-haiku-4-5-...)". */
  label: string;
}

export function createJudge(model: ClaudeModel = CLAUDE_MODELS.haiku): Judge {
  const claude = createClaude({ defaultModel: model });
  const label = `Claude (${model})`;

  return {
    label,

    async faithfulness(question, chunks, answerText) {
      const result = await withRetry(
        () =>
          claude.structured({
            system: FAITHFULNESS_SYSTEM,
            messages: [{ role: 'user', content: faithfulnessUser(question, chunks, answerText) }],
            schema: FaithfulnessSchema,
          }),
        'faithfulness',
      );
      return {
        score: result.data.score,
        reasoning: result.data.reasoning,
        unsupportedClaims: result.data.unsupportedClaims,
        costUSD: result.cost.totalUSD,
      };
    },

    async completeness(question, idealAnswer, candidateAnswer) {
      const result = await withRetry(
        () =>
          claude.structured({
            system: COMPLETENESS_SYSTEM,
            messages: [
              { role: 'user', content: completenessUser(question, idealAnswer, candidateAnswer) },
            ],
            schema: CompletenessSchema,
          }),
        'completeness',
      );
      return {
        score: result.data.score,
        reasoning: result.data.reasoning,
        missedFacts: result.data.missedFacts,
        costUSD: result.cost.totalUSD,
      };
    },

    async refusal(answerText) {
      const result = await withRetry(
        () =>
          claude.structured({
            system: REFUSAL_SYSTEM,
            messages: [{ role: 'user', content: refusalUser(answerText) }],
            schema: RefusalSchema,
          }),
        'refusal',
      );
      return {
        didRefuse: result.data.didRefuse,
        reasoning: result.data.reasoning,
        costUSD: result.cost.totalUSD,
      };
    },
  };
}

// ============================================================================
// Pure helpers — refusal correctness from didRefuse + shouldRefuse.
// ============================================================================

export type RefusalClassification =
  | 'correct-refused'
  | 'correct-answered'
  | 'should-have-refused'
  | 'should-have-answered';

export function classifyRefusal(didRefuse: boolean, shouldRefuse: boolean): RefusalClassification {
  if (shouldRefuse && didRefuse) return 'correct-refused';
  if (!shouldRefuse && !didRefuse) return 'correct-answered';
  if (shouldRefuse && !didRefuse) return 'should-have-refused';
  return 'should-have-answered';
}
