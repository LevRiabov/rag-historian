/**
 * roman-research/mcp-server/index.ts — the Roman corpus as an MCP server
 * (Module 8).
 *
 * The SAME retrieval stack as the Module 7 agent (contextual-v1 + rerank-on-
 * context), re-exposed over the Model Context Protocol so ANY MCP host — Claude
 * Code, Claude Desktop, Cursor — can research the Caesar corpus without our
 * code. This is a second FRONT DOOR to `retrieve`, not a new retriever; the
 * comparison-canonical chunking is untouched (repo rule: no parallel RAG
 * variants).
 *
 * The point of the module is the THREE MCP primitives, each on a different
 * control axis — the distinction a tools-only server (e.g. a GraphQL passthrough)
 * collapses:
 *   - tools     — MODEL-controlled actions   (search / read / cite)
 *   - resources — APP-controlled context     (a book, addressable by URI)
 *   - prompts   — USER-controlled templates   (a research workflow to invoke)
 *
 * Transport is stdio: the host SPAWNS this process and speaks JSON-RPC over
 * stdin/stdout. Hard rule that follows: stdout is the protocol channel —
 * NOTHING may print there or the framing corrupts. All diagnostics go to stderr
 * (console.error), never console.log. Wiring + run instructions: ./README.md.
 */
import 'dotenv/config';
import { McpServer, ResourceTemplate } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import pg from 'pg';
import pgvector from 'pgvector/pg';
import { z } from 'zod';

import { SOURCES } from '../ingest/sources.ts';
import { formatCitation, formatYear, type RetrievedChunk, retrieve } from '../query/retrieve.ts';

// ---------------------------------------------------------------------------
// Canonical retrieval config — frozen identical to the agent's (tools.ts). The
// agent and this server must search the SAME stack, or "the MCP server" and
// "the agent" would silently be two different products.
// ---------------------------------------------------------------------------
const CANONICAL_VERSION = 'contextual-v1';
const RETRIEVAL = {
  provider: 'llamacpp',
  chunkingVersion: CANONICAL_VERSION,
  mode: 'vector',
  rerank: true,
  rerankPoolK: 50,
} as const;

/** Source slugs as a Zod enum tuple — rejects hallucinated source names AND
 *  shows the host the valid set in the generated JSON schema. */
const SOURCE_SLUGS = SOURCES.map((s) => s.slug) as [string, ...string[]];

/** Snippet length in a search result. Enough to judge relevance; the caller
 *  pulls the full passage via read_roman_chunk before quoting. */
const SNIPPET_CHARS = 240;

// ---------------------------------------------------------------------------
// DB — one long-lived connection for the process lifetime. The server is a
// single-tenant local subprocess; a pool would be over-engineering here.
// ---------------------------------------------------------------------------
const databaseUrl = process.env.DATABASE_URL;
if (!databaseUrl) {
  throw new Error('DATABASE_URL not set. Check .env and `docker compose up -d`.');
}
const db = new pg.Client({ connectionString: databaseUrl });
await db.connect();
await pgvector.registerType(db);

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/** A CallToolResult carrying one text block (the only shape every tool uses). */
const text = (body: string, isError = false) => ({
  content: [{ type: 'text' as const, text: body }],
  ...(isError ? { isError: true } : {}),
});

/** Render a search result list the caller can act on: `[id] citation` + snippet,
 *  best first, with a nudge to read_roman_chunk before citing. */
function formatResults(chunks: RetrievedChunk[]): string {
  if (chunks.length === 0) {
    return 'No chunks matched. Try a different query, or a different source.';
  }
  const lines = [`Found ${chunks.length} chunk(s):`, ''];
  for (const c of chunks) {
    const snippet = c.text.replace(/\s+/g, ' ').trim().slice(0, SNIPPET_CHARS);
    lines.push(`[${c.chunkId}] ${formatCitation(c)}`);
    lines.push(`    "${snippet}…"`);
  }
  lines.push('', 'Call read_roman_chunk(chunk_id) for the full text before quoting or citing.');
  return lines.join('\n');
}

/** Full row for a single chunk — read_roman_chunk + cite_passage both need it. */
interface ChunkRow {
  text: string;
  chapter: string;
  author: string;
  title: string;
  slug: string;
  translator: string | null;
  year_written: number | null;
}

async function fetchChunk(chunkId: number): Promise<ChunkRow | null> {
  const result = await db.query<ChunkRow>(
    `SELECT c.text, c.chapter, s.author, s.title, s.slug, s.translator, s.year_written
       FROM chunks c
       JOIN sources s ON s.id = c.source_id
      WHERE c.id = $1`,
    [chunkId],
  );
  return result.rows[0] ?? null;
}

// ===========================================================================
// Server
// ===========================================================================
const server = new McpServer(
  { name: 'roman-research', version: '0.1.0' },
  { capabilities: { tools: {}, resources: {}, prompts: {} } },
);

// ---------------------------------------------------------------------------
// TOOLS — model-controlled actions. The `description` strings are PROMPTS: the
// host model picks tools from them, so they carry the same methodology nudges
// the agent's tools do ("start here", "the tool for contradiction questions").
// ---------------------------------------------------------------------------

server.registerTool(
  'search_roman_corpus',
  {
    title: 'Search the Caesar corpus',
    description:
      'Semantic search across the ENTIRE Caesar corpus (all four primary sources: ' +
      "Caesar's Gallic War & Civil War, Plutarch, Suetonius). Returns the best-" +
      'matching chunks as `[chunk_id] citation` + snippet, best first. Your primary ' +
      'discovery tool — start here. Use focused queries; issue several searches for ' +
      'multi-part or synthesis questions.',
    inputSchema: {
      query: z.string().describe('A focused natural-language search query.'),
      top_k: z
        .number()
        .int()
        .min(1)
        .max(20)
        .optional()
        .describe('How many chunks to return (default 5).'),
    },
  },
  async ({ query, top_k }) => {
    const chunks = await retrieve(db, query, { ...RETRIEVAL, topK: top_k ?? 5 });
    return text(formatResults(chunks));
  },
);

server.registerTool(
  'search_roman_source',
  {
    title: 'Search within one source',
    description:
      'Search WITHIN ONE source only. Use when sources may disagree and you must read ' +
      'each account in isolation before comparing them — the tool for contradiction ' +
      `questions. Valid sources: ${SOURCE_SLUGS.join(', ')}.`,
    inputSchema: {
      source: z.enum(SOURCE_SLUGS).describe('Which source to restrict the search to.'),
      query: z.string().describe('A focused natural-language search query.'),
      top_k: z.number().int().min(1).max(20).optional().describe('How many chunks (default 5).'),
    },
  },
  async ({ source, query, top_k }) => {
    const chunks = await retrieve(db, query, {
      ...RETRIEVAL,
      topK: top_k ?? 5,
      sourceSlug: source,
    });
    return text(formatResults(chunks));
  },
);

server.registerTool(
  'read_roman_chunk',
  {
    title: 'Read a chunk in full',
    description:
      'Return the FULL text of one chunk by its id (from a prior search result). Use ' +
      'when a snippet looks promising but you need the complete passage to quote or ' +
      'cite it accurately.',
    inputSchema: {
      chunk_id: z.number().int().describe('A chunk id returned by a search tool.'),
    },
  },
  async ({ chunk_id }) => {
    const row = await fetchChunk(chunk_id);
    if (!row) {
      return text(`No chunk with id ${chunk_id}. Use ids returned by a search tool.`, true);
    }
    const citation = `${row.author}, ${row.title}, ${row.chapter}`;
    return text(`[${chunk_id}] ${citation}\n\n${row.text.trim()}`);
  },
);

server.registerTool(
  'list_roman_sources',
  {
    title: 'List corpus sources',
    description:
      'List every book in the corpus with author, date written, and chapter/chunk ' +
      'counts. Use to see what is available before searching, or to check coverage.',
    inputSchema: {},
  },
  async () => {
    const result = await db.query<{
      slug: string;
      title: string;
      author: string;
      year_written: number | null;
      chapters: string;
      chunks: string;
    }>(
      `SELECT s.slug, s.title, s.author, s.year_written,
              COUNT(DISTINCT c.chapter) AS chapters,
              COUNT(c.id)               AS chunks
         FROM sources s
         LEFT JOIN chunks c ON c.source_id = s.id AND c.chunking_version = $1
        GROUP BY s.id
        ORDER BY s.year_written`,
      [CANONICAL_VERSION],
    );
    const lines = [`${result.rows.length} sources in the Caesar corpus:`, ''];
    for (const r of result.rows) {
      lines.push(
        `- ${r.slug} — ${r.author}, ${r.title} (${formatYear(r.year_written)}); ` +
          `${r.chapters} chapters, ${r.chunks} chunks`,
      );
    }
    return text(lines.join('\n'));
  },
);

server.registerTool(
  'cite_passage',
  {
    title: 'Format a citation',
    description:
      'Return a formatted citation string for one chunk id — author, work, chapter, ' +
      'translator, and date written. Use to attach a clean reference to a passage.',
    inputSchema: {
      chunk_id: z.number().int().describe('A chunk id returned by a search tool.'),
    },
  },
  async ({ chunk_id }) => {
    const row = await fetchChunk(chunk_id);
    if (!row) {
      return text(`No chunk with id ${chunk_id}. Use ids returned by a search tool.`, true);
    }
    const trans = row.translator ? `, trans. ${row.translator}` : '';
    return text(
      `${row.author}, ${row.title}, ${row.chapter} (${formatYear(row.year_written)}${trans})`,
    );
  },
);

// ---------------------------------------------------------------------------
// RESOURCES — app-controlled context. A book is a READABLE THING addressed by
// URI (roman://source/<slug>), not an action the model decides to invoke. The
// host attaches it to context; the model never "calls" it. This is the
// primitive a tools-only server has no way to express.
// ---------------------------------------------------------------------------

// Top-level overview at a fixed URI — the directory of the corpus.
server.registerResource(
  'sources',
  'roman://sources',
  {
    title: 'Caesar corpus — source index',
    description: 'Overview of all four primary sources, with URIs to drill into each.',
    mimeType: 'text/markdown',
  },
  async (uri) => {
    const lines = ['# Caesar corpus', '', 'Four primary sources on Julius Caesar:', ''];
    for (const s of SOURCES) {
      const note = typeof s.metadata?.note === 'string' ? ` — ${s.metadata.note}` : '';
      lines.push(`- **${s.author}, ${s.title}** (${formatYear(s.yearWritten)})${note}`);
      lines.push(`  \`roman://source/${s.slug}\``);
    }
    return { contents: [{ uri: uri.href, mimeType: 'text/markdown', text: lines.join('\n') }] };
  },
);

// One resource per book, via a URI template. The `list` callback makes all four
// enumerable by the host (so they show up as attachable context); reading one
// returns its metadata + chapter listing (the "chapter listings as nested
// resources" the module asks for).
server.registerResource(
  'source',
  new ResourceTemplate('roman://source/{slug}', {
    list: async () => ({
      resources: SOURCES.map((s) => ({
        uri: `roman://source/${s.slug}`,
        name: `${s.author}, ${s.title}`,
        description: typeof s.metadata?.note === 'string' ? s.metadata.note : undefined,
        mimeType: 'text/markdown',
      })),
    }),
    // Autocomplete the {slug} variable to the four valid slugs.
    complete: {
      slug: (value) => SOURCE_SLUGS.filter((s) => s.startsWith(value)),
    },
  }),
  {
    title: 'A corpus source',
    description: 'One book: metadata plus its chapter listing.',
    mimeType: 'text/markdown',
  },
  async (uri, { slug }) => {
    const source = SOURCES.find((s) => s.slug === slug);
    if (!source) {
      return {
        contents: [{ uri: uri.href, text: `Unknown source '${String(slug)}'.` }],
      };
    }
    const chapters = await db.query<{ chapter: string; chunks: string }>(
      `SELECT c.chapter, COUNT(*) AS chunks
         FROM chunks c
         JOIN sources s ON s.id = c.source_id
        WHERE s.slug = $1 AND c.chunking_version = $2
        GROUP BY c.chapter
        ORDER BY MIN(c.char_start)`,
      [slug, CANONICAL_VERSION],
    );
    const note = typeof source.metadata?.note === 'string' ? source.metadata.note : '';
    const lines = [
      `# ${source.author}, ${source.title}`,
      '',
      `- Written: ${formatYear(source.yearWritten)}`,
      `- Translator: ${source.translator}`,
      `- Tier: ${source.tier}`,
      ...(note ? ['', `> ${note}`] : []),
      '',
      `## Chapters (${chapters.rows.length})`,
      '',
      ...chapters.rows.map((r) => `- ${r.chapter} (${r.chunks} chunks)`),
    ];
    return { contents: [{ uri: uri.href, mimeType: 'text/markdown', text: lines.join('\n') }] };
  },
);

// ---------------------------------------------------------------------------
// PROMPTS — user-controlled templates. These surface in the host UI as
// pickable workflows (a slash command in Claude Code). The model does not
// choose them; the USER invokes one, and it expands into a starter message that
// drives the tools above with the right methodology baked in.
// ---------------------------------------------------------------------------

server.registerPrompt(
  'research_topic',
  {
    title: 'Research a topic in the corpus',
    description: 'Thorough, multi-source, cited research on a Roman-history topic.',
    argsSchema: { topic: z.string().describe('The topic or question to research.') },
  },
  ({ topic }) => ({
    messages: [
      {
        role: 'user',
        content: {
          type: 'text',
          text:
            `Research this topic using ONLY the Caesar corpus tools: "${topic}".\n\n` +
            'Method:\n' +
            '1. search_roman_corpus to discover relevant passages; issue several focused ' +
            'queries for a multi-part topic.\n' +
            '2. read_roman_chunk on the promising hits before quoting.\n' +
            '3. If the sources may disagree, use search_roman_source to read EACH account ' +
            'in isolation, then contrast them by author.\n' +
            '4. Write a cited answer — every claim backed by a [chunk_id] and a ' +
            'cite_passage reference. If the corpus does not cover it, say so plainly ' +
            'rather than guessing.',
        },
      },
    ],
  }),
);

server.registerPrompt(
  'summarize_event',
  {
    title: 'Summarize an event across sources',
    description: 'Summarize one event, noting where the sources agree and differ.',
    argsSchema: { event: z.string().describe('The event to summarize.') },
  },
  ({ event }) => ({
    messages: [
      {
        role: 'user',
        content: {
          type: 'text',
          text:
            `Summarize this event from the Caesar corpus: "${event}".\n\n` +
            "Use search_roman_source to pull each source's account separately, then give " +
            'a summary that (a) states what the sources AGREE on and (b) flags any point ' +
            'where they DIFFER, attributing each version to its author by name. Cite every ' +
            'claim with a [chunk_id]. Do not blend conflicting accounts into one.',
        },
      },
    ],
  }),
);

// ---------------------------------------------------------------------------
// Wire up stdio and run. Logs go to STDERR only (stdout is the JSON-RPC channel).
// ---------------------------------------------------------------------------
const transport = new StdioServerTransport();
await server.connect(transport);
console.error('[roman-research mcp] connected over stdio; corpus ready.');

const shutdown = async () => {
  await db.end();
  await server.close();
  process.exit(0);
};
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
