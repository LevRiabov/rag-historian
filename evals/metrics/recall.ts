/**
 * evals/metrics/recall.ts — recall@k and Mean Reciprocal Rank, scored over
 * chunking-invariant GOLD SPANS rather than chunk IDs.
 *
 * Why span-based (Module 6.1): chunk IDs are minted per chunking variant, so
 * ID-based gold can only score the one variant it was labeled against. Gold
 * SPANS — character ranges in a source's cleanedText — are shared across all
 * variants (every variant's char_start/char_end live in the same coordinate
 * system). So one golden set scores every chunking strategy.
 *
 * recall@k: "Of the gold spans, how many were COVERED by some chunk in the
 *           top-K retrieved?" Range [0, 1]. Higher is better.
 *
 * MRR:      "How high did the FIRST chunk covering ANY gold span rank?"
 *           Range [0, 1]. 1.0 means the top result was always relevant.
 *
 * Coverage rule — a chunk "covers" a span if the chunk's char range contains
 * the span's MIDPOINT (same source). Midpoint, not any-overlap, on purpose:
 *   - naive-v1 chunks overlap neighbors by ~50 tokens, so an any-overlap rule
 *     would let a neighbor that merely clips a span's edge count as a hit,
 *     inflating recall. The midpoint of a span falls inside exactly the chunk
 *     that actually holds it — reproducing the old ID-based gold for small
 *     chunks while still working for whole-chapter chunks.
 *   - Caveat that remains: bigger chunks cover more spans for free, so recall
 *     is granularity-biased UP for coarse chunking. Always read a chunking
 *     A/B alongside the generation metrics (faithfulness/completeness/cost),
 *     never recall alone.
 *
 * Out-of-scope questions (goldSpans = []) make recall undefined — dividing by
 * zero. Callers should skip those questions before aggregating.
 */

import type { GoldSpan } from '../types.ts';

/** The minimum a retrieved chunk must expose for span matching: which source
 *  it belongs to and where it sits in that source's cleanedText. Structurally
 *  satisfied by RetrievedChunk (via its `source.slug` + char offsets); callers
 *  map to this shape so recall.ts stays decoupled from the retrieval types. */
export interface ChunkRange {
  sourceSlug: string;
  charStart: number;
  charEnd: number;
}

/** Does `chunk` cover `span`? True iff same source and the chunk's range
 *  contains the span's midpoint. */
export function covers(chunk: ChunkRange, span: GoldSpan): boolean {
  if (chunk.sourceSlug !== span.sourceSlug) return false;
  const midpoint = (span.charStart + span.charEnd) / 2;
  return chunk.charStart <= midpoint && midpoint < chunk.charEnd;
}

/** How many of `gold` spans are covered by at least one chunk in `chunks`? */
export function coveredCount(chunks: ChunkRange[], gold: GoldSpan[]): number {
  let hits = 0;
  for (const span of gold) {
    if (chunks.some((c) => covers(c, span))) hits++;
  }
  return hits;
}

/**
 * Fraction of gold spans covered by the top-K retrieved chunks.
 * Returns NaN if there are no gold spans (caller should skip — out-of-scope).
 */
export function spanRecallAtK(retrieved: ChunkRange[], gold: GoldSpan[], k: number): number {
  if (gold.length === 0) return Number.NaN;
  return coveredCount(retrieved.slice(0, k), gold) / gold.length;
}

/**
 * Reciprocal rank of the FIRST retrieved chunk that covers ANY gold span.
 * Returns 0 if none of the retrieved chunks cover a gold span.
 * Returns NaN if there are no gold spans.
 *
 * "MRR" is the mean of this across questions — computed by the aggregator,
 * not here. This function is the per-question building block.
 */
export function spanReciprocalRank(retrieved: ChunkRange[], gold: GoldSpan[]): number {
  if (gold.length === 0) return Number.NaN;
  for (let i = 0; i < retrieved.length; i++) {
    const chunk = retrieved[i];
    if (chunk && gold.some((span) => covers(chunk, span))) {
      return 1 / (i + 1);
    }
  }
  return 0;
}

/**
 * Mean of a list of numbers, skipping NaN. Returns NaN if the list is
 * empty after filtering — used for category aggregates where only some
 * categories have in-scope questions.
 */
export function mean(values: number[]): number {
  const finite = values.filter((v) => !Number.isNaN(v));
  if (finite.length === 0) return Number.NaN;
  return finite.reduce((s, v) => s + v, 0) / finite.length;
}
