/**
 * evals/metrics/recall.ts — recall@k and Mean Reciprocal Rank.
 *
 * recall@k: "Of the gold-labeled chunks, how many did the retriever
 *           return in its top-K?" Range [0, 1]. Higher is better.
 *           Formula: |retrieved[:k] ∩ gold| / |gold|
 *
 * MRR:      "How high did the FIRST relevant chunk rank?"
 *           Range [0, 1]. 1.0 means the top result was always relevant.
 *           Formula: mean over questions of 1 / rank-of-first-hit.
 *
 * Why both: recall@k measures coverage (did we get the chunks at all?);
 * MRR measures ordering (did we get them HIGH?). A retriever can score
 * well on recall@20 but poorly on MRR if relevant chunks always rank
 * 15-20 instead of 1-5. That's the case for reranking to fix.
 *
 * Out-of-scope questions (goldChunkIds = []) make recall undefined —
 * dividing by zero. Callers should skip those questions before aggregating.
 */

/**
 * Fraction of gold chunks that appear in the top-K retrieved.
 * Returns NaN if there are no gold chunks (caller should skip).
 */
export function recallAtK(retrievedIds: number[], goldIds: number[], k: number): number {
  if (goldIds.length === 0) return Number.NaN;
  const topK = new Set(retrievedIds.slice(0, k));
  let hits = 0;
  for (const g of goldIds) if (topK.has(g)) hits++;
  return hits / goldIds.length;
}

/**
 * Reciprocal rank of the FIRST gold chunk in the retrieved list.
 * Returns 0 if none of the gold chunks appears in the top `maxRank`.
 * Returns NaN if there are no gold chunks.
 *
 * "MRR" is the mean of this across questions — computed by the aggregator,
 * not here. This function is the per-question building block.
 */
export function reciprocalRank(
  retrievedIds: number[],
  goldIds: number[],
  maxRank: number = Number.POSITIVE_INFINITY,
): number {
  if (goldIds.length === 0) return Number.NaN;
  const goldSet = new Set(goldIds);
  const limit = Math.min(retrievedIds.length, maxRank);
  for (let i = 0; i < limit; i++) {
    const id = retrievedIds[i];
    if (id !== undefined && goldSet.has(id)) {
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
