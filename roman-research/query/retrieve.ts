/**
 * roman-research/query/retrieve.ts — embed the question, find top-K chunks.
 *
 * Defaults to BGE-M3 because that's the column we populated densely at
 * ingest and it's free. To compare against OpenAI, pass `provider: 'openai'`
 * and use `embedding` (1536d). Mixing query and stored embeddings across
 * models gives nonsense — both columns are populated, but you must pick ONE
 * per query and stay consistent (Module 3 pitfall made concrete).
 *
 * The HNSW index `chunks_embedding_bge_hnsw` uses `vector_cosine_ops`, so
 * we order by the `<=>` operator (cosine distance) — anything else silently
 * triggers a sequential scan.
 */
import type { Client } from 'pg';
import pgvector from 'pgvector/pg';

import { createEmbedder, type EmbeddingProvider, rerank } from '../../lib/index.ts';

export interface RetrievedChunk {
  /** chunks.id — for source-viewer linking (Module 10). */
  chunkId: number;
  /** chunks.chunk_index — for prev/next neighbor lookup. */
  chunkIndex: number;
  /** "Book I, §VII" / "§XXIV" — for citations. */
  chapter: string;
  text: string;
  /** Offset of chunk start in its source's cleanedText. Combined with
   *  source.slug this gives a chunking-version-independent coordinate the
   *  eval harness uses to match against gold answer spans (Module 6.1). */
  charStart: number;
  /** Offset of chunk end in its source's cleanedText. */
  charEnd: number;
  /** Cosine similarity in [0, 1]; computed as 1 - (embedding <=> query). */
  similarity: number;
  /** Parent-document retrieval (parent-child-v1): the full parent section this
   *  chunk was split from. Present only when the stored chunk carried a parent
   *  pointer in its metadata. `expandToParents` swaps the chunk text for this
   *  before generation; absent for flat variants. */
  parent?: {
    text: string;
    charStart: number;
    charEnd: number;
  };
  /** Reciprocal-Rank-Fusion score — populated only in hybrid mode (Module
   *  6.2). The retrieval ORDER reflects this; `similarity` is still the cosine
   *  value, kept for display/diagnostics. Absent in vector-only mode. */
  rrfScore?: number;
  /** Cross-encoder relevance score — populated only when reranking is on
   *  (Module 6.3). When present, the retrieval ORDER reflects THIS, not cosine
   *  or RRF. Higher = more relevant. Absent when reranking is off. */
  rerankScore?: number;
  /** The LLM-generated context note (contextual-v1, Module 6.4) from the
   *  chunk's metadata. Used to rerank on the contextualized text rather than
   *  the bare chunk. Absent for flat variants. */
  context?: string;
  /** Joined source metadata — everything the citation prompt needs. */
  source: {
    id: number;
    slug: string;
    title: string;
    author: string;
    tier: string;
    yearWritten: number | null;
    translator: string | null;
  };
}

export interface RetrieveOptions {
  topK?: number;
  /** Which embedding backend to use for the QUERY. Must match the column
   *  storing the indexed vectors (we have both populated). */
  provider?: EmbeddingProvider;
  /** Which chunking variant to search. The table holds multiple variants
   *  side-by-side (naive-v1, chapter-v1, ...), discriminated by this column.
   *  WITHOUT this filter a vector search mixes every variant's rows together
   *  and the results are meaningless — so every comparison run must set it.
   *  Defaults to 'naive-v1' to preserve the Module 4/5 production behavior. */
  chunkingVersion?: string;
  /** Retrieval strategy (Module 6.2):
   *   - 'vector' (default): dense cosine search only — the Module 4/5 baseline.
   *   - 'hybrid': dense + lexical (BM25 via pg_search) fused by Reciprocal Rank
   *     Fusion. Catches exact rare terms (names, numbers) the embedding buries. */
  mode?: 'vector' | 'hybrid';
  /** Hybrid only: weight on the LEXICAL (BM25) arm's RRF contribution; the
   *  vector arm is fixed at 1.0. <1 lets BM25 *add* signal (big win on literal
   *  / exact-term queries) without *overriding* the vector arm on semantic
   *  queries (synonym/synthesis), where equal weight (1.0) floods the fused
   *  top-K with lexical noise. Default 1.0 (plain RRF). */
  lexicalWeight?: number;
  /** Rerank (Module 6.3): pull a DEEPER candidate pool (`rerankPoolK`) from the
   *  first-stage retriever, re-score every candidate with the cross-encoder,
   *  then return the reranked top-`topK`. Fixes ordering (MRR / recall@5) AND
   *  realizes coverage from ranks topK+1..rerankPoolK. Default off. */
  rerank?: boolean;
  /** How many first-stage candidates to rerank (default 50). Larger = more
   *  coverage realized + more cross-encoder passes (slower). Only used when
   *  `rerank` is true. */
  rerankPoolK?: number;
}

/**
 * Embed the question and pull the top-K chunks ordered by cosine distance.
 * Joins on `sources` so the caller has author/title/tier ready for citation
 * formatting and source-tier-aware prompting.
 */
export async function retrieve(
  db: Client,
  question: string,
  options: RetrieveOptions = {},
): Promise<RetrievedChunk[]> {
  const {
    topK = 5,
    provider = 'llamacpp',
    chunkingVersion = 'naive-v1',
    mode = 'vector',
    lexicalWeight = 1.0,
    rerank: doRerank = false,
    rerankPoolK = 50,
  } = options;

  const embedder = createEmbedder({ provider });
  const queryVec = await embedder.embedOne(question);

  // Pick the matching column for the chosen embedder. `column` is internal,
  // never user input, so interpolating it into SQL is safe.
  const column = provider === 'openai' ? 'embedding' : 'embedding_bge';

  // When reranking, fetch a DEEPER first-stage pool so the cross-encoder can
  // pull gold up from ranks topK+1..rerankPoolK (realizing coverage). Without
  // rerank, just fetch topK.
  const fetchK = doRerank ? Math.max(rerankPoolK, topK) : topK;

  // Candidate pool per retrieval arm before fusion / before the version
  // filter trims. Bigger than fetchK so fusion has material to work with and
  // the HNSW post-filter doesn't under-fill. ef_search must be ≥ this for the
  // vector arm to actually surface that many true neighbors.
  const candidatePool = Math.max(fetchK * 5, 100);
  await db.query(`SET hnsw.ef_search = ${candidatePool}`);

  const rows =
    mode === 'hybrid'
      ? await hybridQuery(
          db,
          queryVec,
          question,
          chunkingVersion,
          column,
          candidatePool,
          fetchK,
          lexicalWeight,
        )
      : await vectorQuery(db, queryVec, chunkingVersion, column, fetchK);

  const candidates = rows.map(mapRow);
  if (!doRerank) return candidates;

  // Second stage: cross-encoder re-scores the deep pool jointly (query+chunk),
  // then we keep the reranked top-K. No-op safe on an empty pool.
  //
  // Rerank on the CONTEXTUALIZED text (note + chunk) when a context note is
  // present (contextual-v1). Reranking the bare chunk text makes the
  // cross-encoder blind to the disambiguation the contextual EMBEDDING used to
  // surface the chunk — so it demotes ambiguous-but-correct chunks and partially
  // undoes contextual retrieval (measured in 6.4). Feeding it the same context
  // the embedding saw keeps the two stages aligned. Falls back to bare text for
  // flat variants (no note).
  const { ranking } = await rerank(
    question,
    candidates.map((c) => (c.context ? `${c.context}\n\n${c.text}` : c.text)),
  );
  return ranking
    .slice(0, topK)
    .map(({ index, score }) => ({ ...candidates[index], rerankScore: score }) as RetrievedChunk);
}

/** Shape returned by both retrieval queries (hybrid adds rrf_score). */
interface RetrieveRow {
  chunk_id: number;
  chunk_index: number;
  chapter: string;
  text: string;
  char_start: number;
  char_end: number;
  metadata: {
    parent?: { text: string; charStart: number; charEnd: number };
    context?: string;
  } | null;
  similarity: number;
  rrf_score?: number;
  source_id: number;
  slug: string;
  title: string;
  author: string;
  tier: string;
  year_written: number | null;
  translator: string | null;
}

/** Columns selected by both queries — keeps the two SQL statements in sync. */
const PROJECTION = (column: string) => `
  c.id              AS chunk_id,
  c.chunk_index     AS chunk_index,
  c.chapter         AS chapter,
  c.text            AS text,
  c.char_start      AS char_start,
  c.char_end        AS char_end,
  c.metadata        AS metadata,
  1 - (c.${column} <=> $1) AS similarity,
  s.id              AS source_id,
  s.slug            AS slug,
  s.title           AS title,
  s.author          AS author,
  s.tier            AS tier,
  s.year_written    AS year_written,
  s.translator      AS translator`;

/** Dense-only retrieval (the Module 4/5 baseline). */
async function vectorQuery(
  db: Client,
  queryVec: number[],
  chunkingVersion: string,
  column: string,
  topK: number,
): Promise<RetrieveRow[]> {
  const result = await db.query<RetrieveRow>(
    `SELECT ${PROJECTION(column)}
       FROM chunks c
       JOIN sources s ON s.id = c.source_id
      WHERE c.${column} IS NOT NULL
        AND c.chunking_version = $3
      ORDER BY c.${column} <=> $1
      LIMIT $2`,
    [pgvector.toSql(queryVec), topK, chunkingVersion],
  );
  return result.rows;
}

/**
 * Hybrid retrieval: dense + lexical, fused by Reciprocal Rank Fusion.
 *
 * Two independent rankings — vector (cosine) and lexical (BM25 via pg_search)
 * — each capped at `candidatePool`. RRF combines them by RANK, not score:
 * `Σ 1/(K + rank_i)` with K=60 (the standard constant). Rank-based fusion is
 * the trick that lets us merge two scales (cosine distance vs BM25 score) that
 * aren't remotely comparable, with no normalization. A FULL OUTER JOIN keeps
 * chunks found by only one arm, so a pure-lexical hit (exact rare name the
 * vector missed) still surfaces — the entire point of going hybrid.
 *
 * Lexical arm = BM25 (`@@@ paradedb.match`), NOT Postgres core FTS. BM25's IDF
 * is the whole reason: it down-weights high-document-frequency terms ("Caesar",
 * "sent", "war" — ubiquitous in a single-topic corpus) and up-weights rare
 * discriminative ones ("Pharnaces", "Zela"), so the chunk that actually holds
 * the answer ranks top. Core FTS `ts_rank` has no IDF and buried those under
 * common-verb noise — measured in 6.2, the reason we moved to pg_search.
 * `paradedb.match` tokenizes the raw question safely (handles punctuation /
 * hyphens that the Tantivy query-string parser would choke on).
 */
async function hybridQuery(
  db: Client,
  queryVec: number[],
  question: string,
  chunkingVersion: string,
  column: string,
  candidatePool: number,
  topK: number,
  lexicalWeight: number,
): Promise<RetrieveRow[]> {
  // BM25 emits a perf WARNING when an extra equality filter (chunking_version)
  // blocks its Top-K index scan. Correct either way; silence the noise at our
  // scale (≈1k rows). Session-scoped, harmless to the vector arm.
  await db.query('SET paradedb.check_topk_scan = false');

  const result = await db.query<RetrieveRow>(
    `WITH vector_search AS (
        SELECT id, ROW_NUMBER() OVER (ORDER BY ${column} <=> $1) AS rank
          FROM chunks
         WHERE ${column} IS NOT NULL AND chunking_version = $2
         ORDER BY ${column} <=> $1
         LIMIT $4
      ),
      lexical_search AS (
        SELECT c.id, ROW_NUMBER() OVER (ORDER BY paradedb.score(c.id) DESC) AS rank
          FROM chunks c
         WHERE c.id @@@ paradedb.match('text', $3) AND c.chunking_version = $2
         ORDER BY paradedb.score(c.id) DESC
         LIMIT $4
      ),
      fused AS (
        SELECT COALESCE(v.id, l.id) AS id,
               COALESCE(1.0 / (60 + v.rank), 0.0)
                 + $6 * COALESCE(1.0 / (60 + l.rank), 0.0) AS rrf_score
          FROM vector_search v
          FULL OUTER JOIN lexical_search l ON v.id = l.id
      )
      SELECT ${PROJECTION(column)}, f.rrf_score AS rrf_score
        FROM fused f
        JOIN chunks c ON c.id = f.id
        JOIN sources s ON s.id = c.source_id
       ORDER BY f.rrf_score DESC
       LIMIT $5`,
    [pgvector.toSql(queryVec), chunkingVersion, question, candidatePool, topK, lexicalWeight],
  );
  return result.rows;
}

function mapRow(row: RetrieveRow): RetrievedChunk {
  return {
    chunkId: row.chunk_id,
    chunkIndex: row.chunk_index,
    chapter: row.chapter,
    text: row.text,
    charStart: row.char_start,
    charEnd: row.char_end,
    parent: row.metadata?.parent,
    context: row.metadata?.context,
    similarity: Number(row.similarity),
    ...(row.rrf_score !== undefined ? { rrfScore: Number(row.rrf_score) } : {}),
    source: {
      id: row.source_id,
      slug: row.slug,
      title: row.title,
      author: row.author,
      tier: row.tier,
      yearWritten: row.year_written,
      translator: row.translator,
    },
  };
}

/**
 * Parent-document expansion (parent-child-v1). For each retrieved CHILD chunk,
 * swap its text for its full parent SECTION so the generator sees coherent
 * section-level context instead of a 500-token slice. Distinct parents are
 * de-duplicated (two children of the same section collapse to one block),
 * preserving first-seen retrieval order. Chunks without a parent pointer (all
 * flat variants) pass through unchanged — so this is a safe no-op to call on
 * any retrieval result.
 *
 * Note: dropping to one block per parent means the post-expansion list can be
 * SHORTER than the input. Callers that slice to a generator-K should slice
 * BEFORE expanding (expand the K children), so K bounds the parent count.
 */
export function expandToParents(chunks: RetrievedChunk[]): RetrievedChunk[] {
  const seen = new Set<string>();
  const out: RetrievedChunk[] = [];
  for (const chunk of chunks) {
    if (!chunk.parent) {
      out.push(chunk);
      continue;
    }
    const key = `${chunk.source.slug}:${chunk.parent.charStart}-${chunk.parent.charEnd}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({
      ...chunk,
      text: chunk.parent.text,
      charStart: chunk.parent.charStart,
      charEnd: chunk.parent.charEnd,
    });
  }
  return out;
}

/** Format a chunk's source citation: "Caesar, The Gallic War, Book I, §VII". */
export function formatCitation(chunk: RetrievedChunk): string {
  return `${chunk.source.author}, ${chunk.source.title}, ${chunk.chapter}`;
}

/** Format the year-written for prompts: "-50" → "~50 BC", "121" → "~121 AD". */
export function formatYear(year: number | null): string {
  if (year === null) return 'date unknown';
  return year < 0 ? `~${-year} BC` : `~${year} AD`;
}
