/**
 * roman-research/agent/tools.ts — the Roman Research Agent's tool set (Module 7,
 * Slice 1).
 *
 * Each tool wraps a piece of the Module 6 retrieval stack as a *model-facing
 * capability*. The agent is a reasoning layer ON TOP of the best single-shot
 * retrieval — not a different retriever — so every search here uses the proven
 * final stack (contextual-v1 + rerank-on-context; see
 * notes/module-6-comparison.md). The agent's value is the SEQUENCE of calls
 * (decompose → search each → read each source separately → compare), which
 * single-shot RAG structurally cannot do.
 *
 * Design notes:
 *  - The tool `description` strings are PROMPTS — the model picks tools from
 *    them. They carry methodology nudges ("start here", "the key tool for
 *    contradiction questions") that we'll verify actually fire in the traces.
 *  - State lives in the factory closure (`consulted`), so calling
 *    `createAgentTools(db)` once per question gives each run a fresh,
 *    self-contained coverage view for `list_sources_consulted`.
 *  - `finalize` is defined here (so its `article` arg is Zod-validated) but the
 *    LOOP (Slice 2) owns termination: it breaks when it sees a finalize call and
 *    returns the validated article. Its execute is identity so the tool is also
 *    harmless under a vanilla runTools loop.
 */
import type { Client } from 'pg';
import { z } from 'zod';

import { defineTool, type Tool } from '../../lib/index.ts';
import { SOURCES } from '../ingest/sources.ts';
import { formatCitation, type RetrievedChunk, retrieve } from '../query/retrieve.ts';

/** The terminal tool — the loop (Slice 2) watches for this name to stop. */
export const FINALIZE_TOOL_NAME = 'finalize';

/**
 * The Module 6 final stack, frozen as the agent's retrieval config. contextual-v1
 * chunks + rerank-on-context (the rerank-on-context behavior is automatic in
 * `retrieve` when a context note is present, which contextual-v1 chunks carry).
 */
const RETRIEVAL = {
  provider: 'llamacpp',
  chunkingVersion: 'contextual-v1',
  mode: 'vector',
  rerank: true,
  rerankPoolK: 50,
} as const;

/** Source slugs as a Zod enum tuple — rejects hallucinated source names and
 *  shows the model the valid set in the JSON schema. */
const SOURCE_SLUGS = SOURCES.map((s) => s.slug) as [string, ...string[]];

/** How many characters of a chunk to show as a search snippet. Enough to judge
 *  relevance; the model calls read_chunk for the full text before citing. */
const SNIPPET_CHARS = 200;

/**
 * The tools plus an accessor for the chunks consulted this run. The eval needs
 * the union of surfaced chunks as (a) the faithfulness judge's evidence base and
 * (b) a gold-coverage metric — the agent's analog of recall@k, since a dynamic
 * multi-search loop has no single ranked top-K.
 */
export interface AgentTools {
  tools: Tool[];
  /** Every chunk surfaced by a search this run (full text + spans), deduped. */
  getConsultedChunks(): RetrievedChunk[];
}

/**
 * Build the agent's five tools, closing over a pg client and per-run state.
 * Call once per question so the consulted set starts empty.
 */
export function createAgentTools(db: Client): AgentTools {
  // Full chunks surfaced by searches (deduped by id). read_chunk never adds NEW
  // evidence — the ids it reads always came from a prior search result — so
  // accumulating search results IS the complete, correct evidence base.
  const searched = new Map<number, RetrievedChunk>();
  // Ids the agent pulled in full via read_chunk (drives the coverage display).
  const readIds = new Set<number>();

  const recordSearch = (chunks: RetrievedChunk[]): void => {
    for (const c of chunks) if (!searched.has(c.chunkId)) searched.set(c.chunkId, c);
  };

  /** Render a search result list the model can act on (ids → read_chunk/cite). */
  const formatResults = (chunks: RetrievedChunk[]): string => {
    if (chunks.length === 0) {
      return 'No chunks matched. Try a different query, or a different source.';
    }
    const lines = [`Found ${chunks.length} chunk(s):`, ''];
    for (const c of chunks) {
      const snippet = c.text.replace(/\s+/g, ' ').trim().slice(0, SNIPPET_CHARS);
      lines.push(`[${c.chunkId}] ${formatCitation(c)}`);
      lines.push(`    "${snippet}…"`);
    }
    lines.push('', 'Call read_chunk(chunk_id) for the full text before citing.');
    return lines.join('\n');
  };

  const searchCorpus = defineTool({
    name: 'search_corpus',
    description:
      'Semantic search across the ENTIRE Caesar corpus (all four sources). ' +
      'Returns the best-matching chunks as `[chunk_id] citation` + snippet, best ' +
      'first. Your primary discovery tool — start here. Use focused queries, and ' +
      'issue several searches for multi-part or synthesis questions.',
    schema: z.object({
      query: z.string().describe('A focused natural-language search query.'),
      top_k: z
        .number()
        .int()
        .min(1)
        .max(20)
        .optional()
        .describe('How many chunks to return (default 5).'),
    }),
    execute: async ({ query, top_k }) => {
      const chunks = await retrieve(db, query, { ...RETRIEVAL, topK: top_k ?? 5 });
      recordSearch(chunks);
      return formatResults(chunks);
    },
  });

  const searchWithinSource = defineTool({
    name: 'search_within_source',
    description:
      'Search WITHIN ONE source only. Use when sources may disagree and you must ' +
      'read each account in isolation before comparing them — the key tool for ' +
      `contradiction questions. Valid sources: ${SOURCE_SLUGS.join(', ')}.`,
    schema: z.object({
      source: z.enum(SOURCE_SLUGS).describe('Which source to restrict the search to.'),
      query: z.string().describe('A focused natural-language search query.'),
      top_k: z.number().int().min(1).max(20).optional().describe('How many chunks (default 5).'),
    }),
    execute: async ({ source, query, top_k }) => {
      const chunks = await retrieve(db, query, {
        ...RETRIEVAL,
        topK: top_k ?? 5,
        sourceSlug: source,
      });
      recordSearch(chunks);
      return formatResults(chunks);
    },
  });

  const readChunk = defineTool({
    name: 'read_chunk',
    description:
      'Return the FULL text of one chunk by its id (from a prior search result). ' +
      'Use when a snippet looks promising but you need the complete passage to ' +
      'quote or cite it accurately.',
    schema: z.object({
      chunk_id: z.number().int().describe('A chunk id returned by a search tool.'),
    }),
    execute: async ({ chunk_id }) => {
      const row = await fetchChunk(db, chunk_id);
      if (!row) {
        return `ERROR: no chunk with id ${chunk_id}. Use ids returned by a search tool.`;
      }
      readIds.add(chunk_id);
      const citation = `${row.author}, ${row.title}, ${row.chapter}`;
      return `[${chunk_id}] ${citation}\n\n${row.text.trim()}`;
    },
  });

  const listSourcesConsulted = defineTool({
    name: 'list_sources_consulted',
    description:
      'List every chunk you have searched or read so far this session, grouped by ' +
      'source. Use to check coverage before finalizing — have you consulted every ' +
      'relevant source, especially for contradiction or synthesis questions?',
    schema: z.object({}),
    execute: async () => {
      if (searched.size === 0) return 'Nothing consulted yet. Start with search_corpus.';
      const bySource = new Map<string, string[]>();
      for (const c of searched.values()) {
        const flag = readIds.has(c.chunkId) ? ' [read in full]' : ' [snippet only]';
        const list = bySource.get(c.source.slug) ?? [];
        list.push(`  - ${formatCitation(c)}${flag}`);
        bySource.set(c.source.slug, list);
      }
      const lines: string[] = [];
      for (const [slug, items] of bySource) {
        lines.push(`${slug} (${items.length} chunk(s)):`);
        lines.push(...items);
      }
      return lines.join('\n');
    },
  });

  const finalize = defineTool({
    name: FINALIZE_TOOL_NAME,
    description:
      'Submit your final cited article and END the research. Call this when you ' +
      'can support the answer with cited passages, OR to state plainly that the ' +
      'corpus does not cover the question. Abstaining is a correct outcome — do ' +
      'not keep searching to manufacture an answer.',
    schema: z.object({
      article: z
        .string()
        .describe('The final article, with [chunk_id] citation markers for every claim.'),
    }),
    // Identity: the loop (Slice 2) reads `article` and terminates. Harmless if a
    // vanilla runTools loop echoes it back instead.
    execute: async ({ article }) => article,
  });

  return {
    tools: [searchCorpus, searchWithinSource, readChunk, listSourcesConsulted, finalize],
    getConsultedChunks: () => [...searched.values()],
  };
}

/** Row shape for a single-chunk fetch (read_chunk). */
interface ChunkRow {
  text: string;
  chapter: string;
  author: string;
  title: string;
  slug: string;
}

/** Fetch one chunk's full text + citation fields by id. Ids are unique across
 *  chunking versions, so no version filter is needed. */
async function fetchChunk(db: Client, chunkId: number): Promise<ChunkRow | null> {
  const result = await db.query<ChunkRow>(
    `SELECT c.text, c.chapter, s.author, s.title, s.slug
       FROM chunks c
       JOIN sources s ON s.id = c.source_id
      WHERE c.id = $1`,
    [chunkId],
  );
  return result.rows[0] ?? null;
}
