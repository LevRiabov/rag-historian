/**
 * evals/judge-prompts.ts — system + user templates for the three LLM-as-judge
 * rubrics used in generation-stage evaluation.
 *
 * Why three separate judges (rather than one big "grade the answer" prompt):
 * each rubric measures a DIFFERENT failure mode and benefits from a focused
 * scoring scale. Bundling them into one prompt makes the model average across
 * concerns and produces mushy scores that obscure which thing is broken.
 * (Hamel Husain's eval guides reinforce this point repeatedly.)
 *
 * Each prompt is engineered to:
 *   1. State the ONE thing being judged in the first sentence.
 *   2. Provide a 1-5 rubric with concrete anchors per level (or binary for refusal).
 *   3. Show the model the inputs in a stable, parseable layout.
 *   4. Demand a JSON output via a Zod-backed schema (handled by the wrapper).
 *
 * Calibration: after the first batch run, hand-score ~10 random answers per
 * dimension. If the judge disagrees with you >20% of the time, refine the
 * rubric or the prompt before trusting aggregates.
 */
import type { RetrievedChunk } from '../roman-research/query/retrieve.ts';

// ============================================================================
// Faithfulness — "does every claim in the answer trace to the chunks?"
// ============================================================================

export const FAITHFULNESS_SYSTEM = `You are evaluating the FAITHFULNESS of an AI assistant's answer to retrieved source passages.

Faithfulness = every factual claim in the answer must be directly supported by at least one of the provided source passages. The answer must not:
- introduce facts not present in the sources,
- contradict the sources,
- extrapolate beyond what the sources say (e.g., add dates, motivations, or causal claims not stated).

Paraphrase is fine. Reasonable summarization is fine. Adding outside knowledge — even if accurate — is a faithfulness violation.

Score on a 1-5 scale:
  5 - Perfectly grounded. Every factual claim traces to one or more sources.
  4 - Well supported. Almost all claims trace back; one or two minor unsupported details.
  3 - Mixed. Roughly half of the substantive claims have source support; the rest are extrapolated or invented.
  2 - Mostly unsupported. Few claims trace back; most are invented or contradict sources.
  1 - Hallucinated or contradicted. Most or all claims are not in the sources, or contradict them.

If the answer is a REFUSAL (the assistant declined to answer), score 5 — refusal makes no claims that need support.

Return JSON with: score (1-5), reasoning (one or two sentences), unsupportedClaims (array of specific claims from the answer that you could not find in the sources; empty if score is 5).`;

export function faithfulnessUser(
  question: string,
  chunks: RetrievedChunk[],
  answerText: string,
): string {
  const lines: string[] = ['Question:', question, '', 'Source passages:'];
  for (let i = 0; i < chunks.length; i++) {
    const c = chunks[i];
    if (!c) continue;
    lines.push('');
    lines.push(`[${i + 1}] ${c.source.author}, ${c.source.title}, ${c.chapter}`);
    lines.push(c.text.trim());
  }
  lines.push('', 'Candidate answer:', answerText);
  return lines.join('\n');
}

// ============================================================================
// Completeness — "how much of the ideal answer's content is covered?"
// ============================================================================

export const COMPLETENESS_SYSTEM = `You are evaluating the COMPLETENESS of an AI assistant's answer.

Completeness = how much of the ideal answer's substantive factual content is covered by the candidate answer. Stylistic differences (length, tone, ordering) do not matter — only whether the candidate communicates the same key facts.

Score on a 1-5 scale:
  5 - Complete. Covers essentially every substantive fact in the ideal answer.
  4 - Most. Covers the majority of facts; minor gaps acceptable.
  3 - Half. Covers some key facts but misses others of equal importance.
  2 - Minor. Covers only a small fraction of the ideal answer's content.
  1 - Misses everything. Answers a different question, or omits all the key facts.

Special case: if the ideal answer says the system "should refuse" or describes a refusal as the correct behavior:
  - If candidate is a refusal: score 5.
  - If candidate is a substantive answer: score 1 (it answered when it should have refused).

Return JSON with: score (1-5), reasoning (one or two sentences), missedFacts (array of specific substantive facts from the ideal answer that the candidate did not cover; empty if score is 5).`;

export function completenessUser(
  question: string,
  idealAnswer: string,
  candidateAnswer: string,
): string {
  return [
    'Question:',
    question,
    '',
    'Ideal answer:',
    idealAnswer,
    '',
    'Candidate answer:',
    candidateAnswer,
  ].join('\n');
}

// ============================================================================
// Refusal detection — "is this answer a refusal?"
// ============================================================================
// We only ask the judge to CLASSIFY the answer as refusal vs not.
// "Correctness" is computed in code by comparing didRefuse against
// shouldRefuse (derived from the golden entry's category + goldChunkIds).
// Keeping the judge dumb here makes its job easy and the call cheap.

export const REFUSAL_SYSTEM = `You are classifying whether an AI assistant's answer is a REFUSAL.

A refusal is an answer where the assistant declines to provide substantive information — typically because the question is outside the source material's scope, or because the sources do not contain enough information to answer. Examples of refusal language: "I don't have a source that...", "the sources do not contain information about...", "I cannot answer that from the provided passages".

A non-refusal (substantive answer) attempts to answer the question with facts, even if those facts are incomplete or wrong.

A PARTIAL answer that gives some information AND notes limits at the end is NOT a refusal — it attempted to answer.

Return JSON with: didRefuse (boolean — true if the answer is primarily a refusal, false if it attempted a substantive answer), reasoning (one sentence).`;

export function refusalUser(answerText: string): string {
  return ['Candidate answer:', answerText].join('\n');
}
