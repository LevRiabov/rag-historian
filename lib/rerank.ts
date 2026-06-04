/**
 * lib/rerank.ts — cross-encoder reranker client (Module 6.3).
 *
 * Two-stage retrieval, second stage: a bi-encoder (vector search) is fast but
 * encodes query and chunk SEPARATELY, so its ordering is rough. A cross-encoder
 * reads query + chunk TOGETHER and scores their joint relevance — far more
 * accurate, but it costs one model pass per (query, chunk) pair, so it only
 * runs over the retrieval shortlist (top-N candidates), never the whole corpus.
 *
 * Runtime: `bge-reranker-v2-m3` served by llama.cpp via llama-swap over the
 * `/v1/rerank` endpoint (same llama-swap instance as the gen LLM + embedder;
 * the reranker GGUF runs with `--reranking`). Migrated off the earlier Infinity
 * docker service — one local endpoint now serves everything.
 *
 * Score note: llama.cpp returns RAW cross-encoder logits (can be negative,
 * unbounded), NOT the sigmoid-squashed [0,1] Infinity returned. We only ever
 * SORT by score and keep the top-k, so ordering is identical and absolute scale
 * is irrelevant — but don't threshold on these as if they were probabilities.
 *
 * Local + free, so there's no cost tracking here — only latency.
 */

// 127.0.0.1, not 'localhost' — on Windows localhost can resolve to IPv6 ::1
// first and add a connect hop; the IPv4 literal is deterministic.
const DEFAULT_BASE_URL = 'http://127.0.0.1:8080';
const DEFAULT_MODEL = 'bge-reranker-v2-m3';
// Documents per /v1/rerank request. The cross-encoder scores each (query, doc)
// pair INDEPENDENTLY, so splitting a big candidate pool into sub-batches is
// mathematically identical to one big call. On GPU llama.cpp this is mostly a
// memory-smoothing convenience (the Infinity-era CPU OOM is gone), but it's
// cheap insurance and keeps per-request payloads small.
const DEFAULT_BATCH_SIZE = 16;

export interface RerankConfig {
  /** llama-swap root URL. Defaults to env LLAMA_SWAP_BASE_URL or 127.0.0.1:8080. */
  baseURL?: string;
  /** Served model id — must match the llama-swap model profile name. */
  model?: string;
  /** Documents per request (default 16). Smaller = smaller payloads. */
  batchSize?: number;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** One reranked candidate: its position in the INPUT `documents` array plus
 *  the cross-encoder's relevance score. Higher score = more relevant. */
export interface RerankResult {
  index: number;
  score: number;
}

export interface RerankResponse {
  /** Candidates sorted by relevance, descending. Each `index` points back into
   *  the `documents` array the caller passed in. */
  ranking: RerankResult[];
  latencyMs: number;
}

interface RerankRequestBody {
  model: string;
  query: string;
  documents: string[];
}

interface RerankApiResponse {
  results: Array<{ index: number; relevance_score: number }>;
}

/** POST one sub-batch with retry-on-transient-error. A socket reset or a brief
 *  model reload (llama-swap swapping a sibling model in/out) shouldn't sink the
 *  whole eval, so we retry with backoff (same posture as the judge calls). */
async function postRerank(
  baseURL: string,
  body: RerankRequestBody,
  attempts = 3,
): Promise<RerankApiResponse> {
  let lastErr: unknown;
  for (let i = 0; i < attempts; i++) {
    try {
      const res = await fetch(`${baseURL}/v1/rerank`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(`HTTP ${res.status} ${text.slice(0, 200)}`);
      }
      return (await res.json()) as RerankApiResponse;
    } catch (err) {
      lastErr = err;
      if (i < attempts - 1) await sleep(750 * (i + 1));
    }
  }
  throw new Error(`llama.cpp /v1/rerank failed after ${attempts} attempts: ${String(lastErr)}`);
}

/**
 * Score `documents` against `query` with the cross-encoder and return them
 * ordered most-relevant-first. The caller maps each returned `index` back to
 * its chunk and keeps the top-k.
 *
 * Documents are sent in sub-batches (see DEFAULT_BATCH_SIZE) and merged; each
 * sub-batch's local indices are offset back to global positions. Empty
 * `documents` short-circuits. We sort the merged results ourselves rather than
 * trusting per-batch order — defensive and cheap.
 */
export async function rerank(
  query: string,
  documents: string[],
  config: RerankConfig = {},
): Promise<RerankResponse> {
  if (documents.length === 0) return { ranking: [], latencyMs: 0 };

  const baseURL = config.baseURL ?? process.env.LLAMA_SWAP_BASE_URL ?? DEFAULT_BASE_URL;
  const model = config.model ?? process.env.RERANK_MODEL ?? DEFAULT_MODEL;
  const batchSize = config.batchSize ?? DEFAULT_BATCH_SIZE;

  const t0 = Date.now();
  const ranking: RerankResult[] = [];
  for (let start = 0; start < documents.length; start += batchSize) {
    const slice = documents.slice(start, start + batchSize);
    const json = await postRerank(baseURL, { model, query, documents: slice });
    for (const r of json.results) {
      ranking.push({ index: start + r.index, score: r.relevance_score });
    }
  }

  ranking.sort((a, b) => b.score - a.score);
  return { ranking, latencyMs: Date.now() - t0 };
}
