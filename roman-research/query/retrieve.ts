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
  /** HyDE (Module 6.5): the text to EMBED for first-stage vector search, when
   *  it should differ from `question`. Pass a hypothetical answer document
   *  (`generateHydeDoc`) here to search on answer-shaped text that lands nearer
   *  the corpus than the bare question does. Deliberately scoped to the vector
   *  arm ONLY: the BM25 lexical arm and the cross-encoder reranker still use
   *  `question`, because hallucinated terms poison exact-match and a reranker
   *  must read the REAL question to judge relevance. Defaults to `question`. */
  embedText?: string;
  /** Restrict retrieval to a SINGLE source (sources.slug). The mechanism behind
   *  the Module 7 agent's `search_within_source` tool: read each account in
   *  isolation before comparing them, instead of letting one fused top-K blend
   *  conflicting sources (the contradiction-handling fix). Vector mode only —
   *  see the guard in `retrieve`. Default: search all sources. */
  sourceSlug?: string;
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
    embedText,
    sourceSlug,
  } = options;

  // Source-scoped retrieval (the agent's search_within_source) is wired through
  // the vector arm only — the one the agent's final stack uses. Guard rather
  // than silently ignore the filter in hybrid mode, which the agent never uses.
  if (sourceSlug && mode === 'hybrid') {
    throw new Error(
      'retrieve: sourceSlug is supported in vector mode only (the agent stack is vector+rerank).',
    );
  }

  const embedder = createEmbedder({ provider });
  // HyDE: embed the hypothetical-answer doc when provided, else the question
  // itself. Only the dense vector is affected — `question` still feeds BM25 and
  // the reranker below (see embedText doc on RetrieveOptions).
  const queryVec = await embedder.embedOne(embedText ?? question);

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
      : await vectorQuery(db, queryVec, chunkingVersion, column, fetchK, sourceSlug);

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

/**
 * Multi-query retrieval (query expansion, Module 6.5). Run first-stage
 * retrieval for EACH query in `queries` (the caller builds this as
 * `[originalQuestion, ...variations]` — see `expand.ts`), then fuse the per-query
 * rankings with Reciprocal Rank Fusion — the SAME rank-based fusion hybrid uses
 * across its vector/lexical arms, here applied across queries. A chunk surfaced
 * by SEVERAL variations accumulates RRF mass and rises: agreement across
 * rephrasings is a robustness signal (the chunk is relevant under multiple
 * readings of the question), exactly what we want to reward.
 *
 * Because the original question is query #0, expansion is purely additive — a
 * variation that drifts can only fail to contribute, never displace what the
 * plain question already finds. Reranking (when on) re-scores the fused pool
 * with the ORIGINAL question on context-text, same as single-query retrieve():
 * the union just gives the cross-encoder a wider, more diverse pool to lift gold
 * from.
 *
 * Note: each query gets its own embedding + SQL round-trip (sequential — they
 * share one pg connection), so wall-time scales with |queries|. Fine at our
 * scale (≈4 queries); a connection pool would parallelize it in production.
 */
export async function retrieveMultiQuery(
  db: Client,
  question: string,
  queries: string[],
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
  const column = provider === 'openai' ? 'embedding' : 'embedding_bge';

  // Pull a WIDE pool per query so the union has material and RRF can reward
  // cross-query agreement. ef_search must cover it for the vector arm.
  const fetchK = doRerank ? Math.max(rerankPoolK, topK) : topK;
  const candidatePool = Math.max(fetchK * 5, 100);
  await db.query(`SET hnsw.ef_search = ${candidatePool}`);

  // First-stage retrieval per query (sequential — single pg connection). Each
  // variation drives BOTH arms with its OWN text: its embedding for the vector
  // arm, its words for the BM25 arm (in hybrid mode) — the variation's distinct
  // vocabulary is the entire point.
  const perQueryRows: RetrieveRow[][] = [];
  for (const q of queries) {
    const vec = await embedder.embedOne(q);
    const rows =
      mode === 'hybrid'
        ? await hybridQuery(
            db,
            vec,
            q,
            chunkingVersion,
            column,
            candidatePool,
            candidatePool,
            lexicalWeight,
          )
        : await vectorQuery(db, vec, chunkingVersion, column, candidatePool);
    perQueryRows.push(rows);
  }

  // RRF across queries: Σ 1/(60 + rank_q) over every query that surfaced the
  // chunk. Rank-based, so it merges pools with no score normalization (cosine
  // vs BM25 vs across-query are all incomparable as raw scores).
  const RRF_K = 60;
  const fused = new Map<number, { row: RetrieveRow; score: number }>();
  for (const rows of perQueryRows) {
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      if (!row) continue;
      const inc = 1 / (RRF_K + (i + 1));
      const prev = fused.get(row.chunk_id);
      if (prev) prev.score += inc;
      else fused.set(row.chunk_id, { row, score: inc });
    }
  }

  // Order by fused score; rrfScore now means the CROSS-QUERY fusion score.
  const candidates = [...fused.values()]
    .sort((a, b) => b.score - a.score)
    .map(({ row, score }) => ({ ...mapRow(row), rrfScore: score }));

  if (!doRerank) return candidates.slice(0, topK);

  // Second stage: rerank the fused pool with the ORIGINAL question (NOT a
  // variation — the cross-encoder judges relevance to what the user actually
  // asked), on context-text when present (same alignment rule as retrieve()).
  const pool = candidates.slice(0, rerankPoolK);
  const { ranking } = await rerank(
    question,
    pool.map((c) => (c.context ? `${c.context}\n\n${c.text}` : c.text)),
  );
  return ranking
    .slice(0, topK)
    .map(({ index, score }) => ({ ...pool[index], rerankScore: score }) as RetrievedChunk);
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

/** Dense-only retrieval (the Module 4/5 baseline). When `sourceSlug` is set,
 *  restrict to that one source — the agent's search_within_source path. The
 *  `$4::text IS NULL OR ...` form keeps a single SQL statement for both cases. */
async function vectorQuery(
  db: Client,
  queryVec: number[],
  chunkingVersion: string,
  column: string,
  topK: number,
  sourceSlug?: string,
): Promise<RetrieveRow[]> {
  const result = await db.query<RetrieveRow>(
    `SELECT ${PROJECTION(column)}
       FROM chunks c
       JOIN sources s ON s.id = c.source_id
      WHERE c.${column} IS NOT NULL
        AND c.chunking_version = $3
        AND ($4::text IS NULL OR s.slug = $4)
      ORDER BY c.${column} <=> $1
      LIMIT $2`,
    [pgvector.toSql(queryVec), topK, chunkingVersion, sourceSlug ?? null],
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
