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

import { createEmbedder, type EmbeddingProvider } from '../../lib/index.ts';

export interface RetrievedChunk {
  /** chunks.id — for source-viewer linking (Module 10). */
  chunkId: number;
  /** chunks.chunk_index — for prev/next neighbor lookup. */
  chunkIndex: number;
  /** "Book I, §VII" / "§XXIV" — for citations. */
  chapter: string;
  text: string;
  /** Cosine similarity in [0, 1]; computed as 1 - (embedding <=> query). */
  similarity: number;
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
  const { topK = 5, provider = 'lmstudio' } = options;

  const embedder = createEmbedder({ provider });
  const queryVec = await embedder.embedOne(question);

  // Pick the matching column for the chosen embedder.
  const column = provider === 'openai' ? 'embedding' : 'embedding_bge';

  const result = await db.query<{
    chunk_id: number;
    chunk_index: number;
    chapter: string;
    text: string;
    similarity: number;
    source_id: number;
    slug: string;
    title: string;
    author: string;
    tier: string;
    year_written: number | null;
    translator: string | null;
  }>(
    `SELECT
       c.id              AS chunk_id,
       c.chunk_index     AS chunk_index,
       c.chapter         AS chapter,
       c.text            AS text,
       1 - (c.${column} <=> $1) AS similarity,
       s.id              AS source_id,
       s.slug            AS slug,
       s.title           AS title,
       s.author          AS author,
       s.tier            AS tier,
       s.year_written    AS year_written,
       s.translator      AS translator
     FROM chunks c
     JOIN sources s ON s.id = c.source_id
     WHERE c.${column} IS NOT NULL
     ORDER BY c.${column} <=> $1
     LIMIT $2`,
    [pgvector.toSql(queryVec), topK],
  );

  return result.rows.map((row) => ({
    chunkId: row.chunk_id,
    chunkIndex: row.chunk_index,
    chapter: row.chapter,
    text: row.text,
    similarity: Number(row.similarity),
    source: {
      id: row.source_id,
      slug: row.slug,
      title: row.title,
      author: row.author,
      tier: row.tier,
      yearWritten: row.year_written,
      translator: row.translator,
    },
  }));
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
