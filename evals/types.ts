/**
 * evals/types.ts — shared types for the eval harness.
 *
 * The golden set is the load-bearing artifact: each entry pairs a question
 * with the chunk IDs that contain the answer. Without those `goldChunkIds`,
 * recall@k is unmeasurable and the whole "decompose to isolate the broken
 * component" workflow falls apart.
 */

/**
 * Question categories — chosen to surface DIFFERENT failure modes:
 *   - literal:       tests basic vector retrieval (the easy case)
 *   - synonym:       tests vocabulary mismatch (Module 6.2 hybrid search target)
 *   - multi-hop:     tests chunk decomposition (Module 6.5 query rewriting target)
 *   - synthesis:     tests LLM reasoning across multiple chunks
 *   - contradiction: tests cross-source retrieval where the sources DISAGREE.
 *                    The corpus's headline goal — a good answer must surface
 *                    facts from conflicting accounts (Caesar's self-account vs
 *                    Plutarch/Suetonius), so gold chunks span multiple sources.
 *                    Distinct from synthesis: the failure mode is retrieving
 *                    only one side of the disagreement.
 *   - out-of-scope:  tests refusal (no gold chunks; LLM should refuse)
 */
export type Category =
  | 'literal'
  | 'synonym'
  | 'multi-hop'
  | 'synthesis'
  | 'contradiction'
  | 'out-of-scope';

/**
 * One entry in the golden set. JSON-serialisable so we can store the set
 * in `evals/golden-set.json` and version it through code review.
 */
export interface GoldenEntry {
  /** Stable id — kebab-case slug; allows comparing runs over time. */
  id: string;
  category: Category;
  question: string;
  /** What a perfect answer looks like. Used by generation metrics (Turn 3). */
  idealAnswer: string;
  /**
   * Chunk IDs (chunks.id values) that contain enough information to answer.
   * Empty array for out-of-scope questions (where the right behavior is
   * refusal, not retrieval). For multi-hop/synthesis questions, list every
   * chunk that supplies a fact the ideal answer relies on.
   */
  goldChunkIds: number[];
  /** Optional: why this question belongs to this category, gotchas. */
  notes?: string;
}

/** Per-question retrieval result + scores. */
export interface QuestionResult {
  entry: GoldenEntry;
  /** chunks.id values in retrieval order, top to bottom. */
  retrievedChunkIds: number[];
  /** Cosine similarity in retrieval order, for diagnostic display. */
  similarities: number[];
  /** Latency of the retrieval call (ms). */
  latencyMs: number;
  /** recall@k for each k in EVAL_K_VALUES. NaN for out-of-scope. */
  recallAtK: Record<number, number>;
  /** MRR — 1/rank of first gold hit (0 if none). NaN for out-of-scope. */
  mrr: number;
  /** Generation-stage score — present only when the runner was invoked with
   *  --generation. Optional so a retrieval-only run stays light + cheap. */
  generation?: GenerationScore;
}

/**
 * Generation-stage measurement for one question. Includes the generated answer
 * itself (for spot-checking + future re-judging without re-generation) and the
 * three LLM-as-judge scores.
 *
 *   faithfulness:  Does every claim in the answer trace to retrieved chunks?
 *                  1-5. Detects hallucination.
 *   completeness:  When the chunks support an answer, did the LLM extract it
 *                  fully? 1-5 against the ideal answer.
 *   refusal:       Out-of-scope / unanswerable questions: did the model
 *                  refuse vs invent? Binary correctness.
 */
export interface GenerationScore {
  /** The model-generated answer text, for spot-check + re-judging later. */
  answerText: string;
  /** Pretty label of the generator model, e.g. "LM Studio (openai/gpt-oss-20b)". */
  generatorLabel: string;
  /** Wall-clock ms to generate. */
  generatorLatencyMs: number;
  /** USD spent generating (0 for local). */
  generatorCostUSD: number;

  faithfulness: { score: number; reasoning: string; unsupportedClaims: string[] };
  completeness: { score: number; reasoning: string; missedFacts: string[] };
  refusal: {
    /** Did the answer refuse to answer (LLM-as-judge classification)? */
    didRefuse: boolean;
    /** Should the question have been refused? Derived from entry, not judged. */
    shouldRefuse: boolean;
    /** didRefuse === shouldRefuse */
    correct: boolean;
    /** Four-way label for reporting. */
    classification:
      | 'correct-refused'
      | 'correct-answered'
      | 'should-have-refused'
      | 'should-have-answered';
    reasoning: string;
  };

  /** Pretty label of the judge model, e.g. "Claude (claude-haiku-...)". */
  judgeLabel: string;
  /** USD spent across all three judge calls. */
  judgeCostUSD: number;
}

/** Whole-run summary. */
export interface BatchResult {
  results: QuestionResult[];
  /** Averaged across in-scope questions only (out-of-scope excluded for
   *  retrieval; refusal accuracy is computed across ALL questions). */
  aggregates: {
    recallAtK: Record<number, number>;
    mrr: number;
    /** Per-category breakdown — helps spot which question type is weakest. */
    byCategory: Record<
      Category,
      {
        count: number;
        recallAtK: Record<number, number>;
        mrr: number;
        /** Per-category generation breakdown when --generation was on. */
        generation?: {
          answeredCount: number;
          faithfulness: number;
          completeness: number;
          refusalAccuracy: number;
        };
      }
    >;
    /**
     * Top-level generation aggregates, present only when --generation was on.
     * faithfulness/completeness average over questions where an answer was
     * generated. refusalAccuracy spans ALL questions including out-of-scope.
     */
    generation?: {
      answeredCount: number;
      faithfulness: number;
      completeness: number;
      refusalAccuracy: number;
      totalJudgeCostUSD: number;
      totalGeneratorCostUSD: number;
    };
  };
  /** Configuration used — re-runnable from this. */
  config: {
    embedder: string;
    chunkingVersion: string;
    topK: number;
    timestamp: string;
    /** Present when --generation was on. */
    generator?: string;
    judge?: string;
  };
}

/** Which k values we measure recall at. Single retrieve call serves all. */
export const EVAL_K_VALUES = [1, 3, 5, 10, 20] as const;

/** Max K to fetch from the retriever — must be ≥ max(EVAL_K_VALUES). */
export const RETRIEVE_TOP_K = Math.max(...EVAL_K_VALUES);
