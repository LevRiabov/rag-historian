/**
 * roman-research/ingest/contextualize.ts — Contextual Retrieval ingest (Module 6.4).
 *
 * Anthropic's technique: a chunk loses its context when split from its document
 * ("he crossed at dawn" — who? which river?), so its embedding doesn't match a
 * query like "Caesar crossing the Rubicon". Fix: an LLM reads the WHOLE parent
 * document + the chunk and writes a 1–2 sentence context note naming the
 * people/places/events the chunk only implies. We embed `citation + note +
 * chunk` instead of the bare chunk — same retrieval machinery, smarter vectors.
 *
 * Produces a new chunking_version `contextual-v1`:
 *   - chunks are IDENTICAL to naive-v1 (same splits, same char offsets, so the
 *     span-based gold still matches; the original chunk text is stored for
 *     display/generation),
 *   - only the stored EMBEDDING differs (of the contextualized text),
 *   - the generated note is kept in metadata for inspection.
 *
 * Design notes that matter:
 *   - **No-think, picked by document size.** Context generation runs on
 *     llama.cpp (llama-swap), thinking OFF. The parent document can be ~55k
 *     tokens (whole Plutarch), so the profile (16k/32k/64k/100k) is chosen per
 *     document. Thinking is fatal here: it both hallucinates dates AND its
 *     reasoning tokens overflow the context budget the document needs.
 *   - **Document-as-prefix → KV-cache reuse.** The document goes FIRST and
 *     identical across a book's chunks, so llama-server reuses its KV cache for
 *     the document across consecutive chunks — Anthropic's "prompt caching",
 *     for free, locally. So we process chunks in document order.
 *   - **Phased to avoid model thrashing.** llama-swap serves ONE model at a
 *     time. So we generate ALL notes first (qwen loaded), THEN embed everything
 *     (bge-m3 loaded once). Interleaving would swap models every call.
 *   - **Footnotes kept on purpose** (Module 6.4 decision): Plutarch's edition
 *     carries scholarly footnotes (~44% of its chunks). Real corpora are messy;
 *     keeping them tests robustness AND keeps the corpus constant so contextual
 *     is a clean single-variable A/B vs vector+rerank.
 *
 * Run:
 *   pnpm dev roman-research/ingest/contextualize.ts                 # all sources
 *   pnpm dev roman-research/ingest/contextualize.ts --source=plutarch-caesar
 *   pnpm dev roman-research/ingest/contextualize.ts --limit=5       # first 5 chunks/source (smoke test)
 *   pnpm dev roman-research/ingest/contextualize.ts --dry           # generate + print, don't store
 */
import 'dotenv/config';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';

import { getEncoding } from 'js-tiktoken';
import pg from 'pg';
import pgvector from 'pgvector/pg';

import { createEmbedder, createLlamacpp, type Embedder, LLAMACPP_MODELS } from '../../lib/index.ts';
import { type Chunk, chunkSections } from './chunk.ts';
import { readRawFile } from './download.ts';
import { parseSource, type ParsedSection } from './parse.ts';
import { type SourceManifest, SOURCES } from './sources.ts';

const CHUNKING_VERSION = 'contextual-v1';
const NAIVE_VERSION = 'naive-v1';

const ENC = getEncoding('cl100k_base');
const tok = (s: string) => ENC.encode(s).length;

// bge-m3 / text-embedding-3 cap inputs at ~8192 tokens. The rare oversize chunk
// (an unbroken section, or Plutarch's footnote blob) exceeds that. naive-v1's
// LM Studio embedder silently TRUNCATED such inputs; llama.cpp ERRORS instead.
// Clamp the embedding input to match (cl100k count, conservative vs bge's hotter
// tokenizer). Only affects oversize chunks; the STORED text is left full.
const MAX_EMBED_TOKENS = 6500;
function clampForEmbedding(text: string): string {
  const ids = ENC.encode(text);
  return ids.length <= MAX_EMBED_TOKENS ? text : ENC.decode(ids.slice(0, MAX_EMBED_TOKENS));
}

// Generation is the expensive phase; cache notes to disk so a later-phase
// failure (or a re-run) doesn't re-generate. Keyed by source slug + chunk index
// (both deterministic).
const CACHE_PATH = path.join('evals', 'results', '.contextual-notes-cache.json');

// ---------------------------------------------------------------------------
// The validated prompt (manual-test tuned): citation is GIVEN (no book-number
// guessing), facts GROUNDED in the document (no hallucinated dates). Document
// goes first in the user message so it's a stable, cache-reusable prefix.
// ---------------------------------------------------------------------------
const SYSTEM_PROMPT = `You write a one- or two-sentence context note that situates a passage within a larger historical work, so the passage can be found by search. You are TOLD the work and the exact book/section the passage belongs to — treat that as fact, never change or guess it. Your job is to add the disambiguating CONTENT the passage itself leaves out: the people, peoples, places, and events it refers to, especially anything it names only by pronoun ("he", "their", "the enemy"). Ground every claim in the provided document — do NOT add dates, numbers, or facts the document does not state. Output ONLY the note: no preamble, no "Here is", no quotation marks, no markdown.`;

function buildUserMessage(documentText: string, citation: string, chunkText: string): string {
  return `<document>
${documentText}
</document>

The passage below is from ${citation} (this is given — treat as fact, do not change it).
<passage>
${chunkText.trim()}
</passage>

Write the context note that situates this passage within the document, to improve search retrieval. Name the people, peoples, places, and events it alludes to but does not itself state. Ground everything in the document; add no dates or facts the document omits. Answer with only the note.`;
}

/** Pick the smallest llama-swap profile whose context window fits the document
 *  plus headroom for the chunk, prompt, and the short answer. */
function pickProfile(docTokens: number): { model: string; ctx: number } {
  const budget = docTokens + 3000;
  if (budget <= 16384) return { model: LLAMACPP_MODELS.qwen9b16k, ctx: 16384 };
  if (budget <= 32768) return { model: LLAMACPP_MODELS.qwen9b32k, ctx: 32768 };
  if (budget <= 65536) return { model: LLAMACPP_MODELS.qwen9b64k, ctx: 65536 };
  return { model: LLAMACPP_MODELS.qwen9b100k, ctx: 102400 };
}

/** "Book III, §XIII" → "Book III"; "§II" → null. Matches formatChapter(). */
function bookOf(chapter: string): string | null {
  return chapter.includes(', §') ? (chapter.split(', §')[0] ?? null) : null;
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
const argv = process.argv.slice(2);
const flag = (name: string) =>
  argv.find((a) => a === `--${name}` || a.startsWith(`--${name}=`));
const flagVal = (name: string) => flag(name)?.split('=')[1];
const onlySource = flagVal('source');
const limit = flagVal('limit') ? Number(flagVal('limit')) : undefined;
const dryRun = flag('dry') !== undefined;

// ---------------------------------------------------------------------------
// DB + clients
// ---------------------------------------------------------------------------
const db = new pg.Client({ connectionString: process.env.DATABASE_URL });
await db.connect();
await pgvector.registerType(db);

const llm = createLlamacpp();
// Local embeddings only (bge-m3 via llama-swap) — this project does not use
// OpenAI embeddings. The `embedding` (1536d OpenAI) column stays NULL.
const bge: Embedder = createEmbedder({ provider: 'llamacpp' });

const sourcesToRun = SOURCES.filter((s) => !onlySource || s.slug === onlySource);

console.log(`=== Contextual Retrieval ingest → ${CHUNKING_VERSION} ===`);
console.log(`Sources: ${sourcesToRun.map((s) => s.slug).join(', ')}`);
console.log(`Generator: llama.cpp (qwen3.5-9b, no-think)  |  Embedder: ${bge.label}`);
if (limit) console.log(`LIMIT: first ${limit} chunks per source (smoke test)`);
if (dryRun) console.log(`DRY RUN: generating + printing, not storing`);
console.log();

// ---------------------------------------------------------------------------
// A unit of work: one chunk + the parent document it should be situated in.
// ---------------------------------------------------------------------------
interface Unit {
  source: SourceManifest;
  sourceId: number;
  chunk: Chunk;
  citation: string;
  documentText: string;
  profile: string;
  /** Filled in phase A. */
  note?: string;
  /** Filled in phase A: citation + note + chunk text — the text we embed. */
  contextualText?: string;
}

// Gold spans (for the footnote/coverage preflight).
interface GoldSpan {
  sourceSlug: string;
  charStart: number;
  charEnd: number;
}
const goldenRaw = await readFile(path.join('evals', 'golden-set.json'), 'utf-8');
const goldSpans: GoldSpan[] = (JSON.parse(goldenRaw) as Array<{ goldSpans?: GoldSpan[] }>).flatMap(
  (e) => e.goldSpans ?? [],
);

// ---------------------------------------------------------------------------
// Build the work list. Sort sources by required profile context so llama-swap
// swaps as little as possible (all 16k books, then the 64k whole-works).
// ---------------------------------------------------------------------------
const units: Unit[] = [];
let footnoteGoldHits = 0;

for (const source of sourcesToRun) {
  const raw = await readRawFile(source.gutenbergId);
  const parsed = parseSource(source.slug, raw);
  const chunks = chunkSections(parsed.sections); // identical to naive-v1

  const sourceId = await sourceIdOf(source.slug);

  // One document per book (Caesar) or one per whole work (flat sources).
  const byBook = new Map<string | null, ParsedSection[]>();
  for (const s of parsed.sections) {
    const arr = byBook.get(s.bookLabel) ?? [];
    arr.push(s);
    byBook.set(s.bookLabel, arr);
  }
  const docByBook = new Map<string | null, { text: string; profile: string; ctx: number }>();
  for (const [book, secs] of byBook) {
    const text = secs.map((s) => s.text).join('\n\n');
    const { model, ctx } = pickProfile(tok(text));
    docByBook.set(book, { text, profile: model, ctx });
  }

  const selected = limit ? chunks.slice(0, limit) : chunks;
  for (const chunk of selected) {
    const book = bookOf(chunk.chapter);
    const doc = docByBook.get(book);
    if (!doc) throw new Error(`No document for ${source.slug} ${chunk.chapter} (book=${book})`);

    // Preflight: does this chunk (a gold answer region) happen to be footnote
    // apparatus? Gold should be narrative — warn if not.
    const overlapsGold = goldSpans.some(
      (g) =>
        g.sourceSlug === source.slug &&
        chunk.charStart < g.charEnd &&
        chunk.charEnd > g.charStart,
    );
    if (overlapsGold && /\[Footnote|FOOTNOTES:/.test(chunk.text)) footnoteGoldHits++;

    units.push({
      source,
      sourceId,
      chunk,
      citation: `${source.author}, ${source.title}, ${chunk.chapter}`,
      documentText: doc.text,
      profile: doc.profile,
    });
  }

  const profiles = [...new Set([...docByBook.values()].map((d) => `${d.ctx / 1024}k`))].join(',');
  console.log(
    `  ${source.slug.padEnd(20)} ${selected.length} chunks, ${docByBook.size} document(s), profile(s) ${profiles}`,
  );
}

console.log(
  `\nGold-span preflight: ${footnoteGoldHits === 0 ? 'OK — no gold answer falls in a footnote chunk' : `⚠ ${footnoteGoldHits} gold chunks are footnotes (recall may be affected)`}`,
);

// Process order: group by profile so llama-swap loads each profile once.
units.sort((a, b) => a.profile.localeCompare(b.profile));

// ---------------------------------------------------------------------------
// Phase A — generate context notes (qwen loaded; document prefix is KV-cached
// across consecutive chunks of the same document).
// ---------------------------------------------------------------------------
let noteCache: Record<string, string> = {};
try {
  noteCache = JSON.parse(await readFile(CACHE_PATH, 'utf-8'));
  console.log(`Loaded ${Object.keys(noteCache).length} cached notes from ${CACHE_PATH}`);
} catch {
  /* no cache yet */
}
const saveCache = async () => {
  await mkdir(path.dirname(CACHE_PATH), { recursive: true });
  await writeFile(CACHE_PATH, JSON.stringify(noteCache), 'utf-8');
};

console.log(`\n--- Phase A: generating ${units.length} context notes (no-think) ---`);
const tA = Date.now();
let genFails = 0;
let cacheHits = 0;
let lastProfile = '';
for (let i = 0; i < units.length; i++) {
  const u = units[i];
  if (!u) continue;
  const key = `${u.source.slug}:${u.chunk.chunkIndex}`;

  if (noteCache[key] !== undefined) {
    u.note = noteCache[key];
    cacheHits++;
  } else {
    if (u.profile !== lastProfile) {
      console.log(`  [profile ${u.profile}]`);
      lastProfile = u.profile;
    }
    try {
      u.note = await chatWithRetry(u.profile, u.documentText, u.citation, u.chunk.text);
    } catch (err) {
      genFails++;
      u.note = ''; // degrade: embed citation + chunk without a note
      console.warn(`  ✗ [${u.source.slug} ${u.chunk.chapter}] gen failed: ${String(err).slice(0, 100)}`);
    }
    noteCache[key] = u.note;
    if (i % 50 === 0) await saveCache(); // crash-safe incremental persist
  }

  u.contextualText = u.note
    ? `${u.citation}\n${u.note}\n\n${u.chunk.text.trim()}`
    : `${u.citation}\n\n${u.chunk.text.trim()}`;

  if (dryRun && i < 8) {
    console.log(`\n  [${u.source.slug} ${u.chunk.chapter}]`);
    console.log(`  passage: ${u.chunk.text.replace(/\s+/g, ' ').slice(0, 90)}…`);
    console.log(`  note:    ${u.note}`);
  }
  if ((i + 1) % 25 === 0) {
    const rate = ((i + 1) / ((Date.now() - tA) / 1000)).toFixed(2);
    console.log(`    ${i + 1}/${units.length} (${rate} notes/s)`);
  }
}
await saveCache();
console.log(
  `Phase A done in ${Math.round((Date.now() - tA) / 1000)}s — ${cacheHits} from cache${genFails ? `, ${genFails} failures` : ''}.`,
);

if (!dryRun) {
// ---------------------------------------------------------------------------
// Phase B — embed the contextualized texts (bge-m3, local, loaded once).
// ---------------------------------------------------------------------------
console.log(`\n--- Phase B: embedding ${units.length} contextualized texts (bge-m3, local) ---`);
const texts = units.map((u) => clampForEmbedding(u.contextualText ?? ''));
const clamped = units.filter((u) => (u.contextualText?.length ?? 0) > 0 && clampForEmbedding(u.contextualText ?? '') !== u.contextualText).length;
if (clamped > 0) console.log(`  (clamped ${clamped} oversize input(s) to ${MAX_EMBED_TOKENS} tok for the embedder)`);
const tB = Date.now();
const bgeRes = await bge.embed(texts);
console.log(`  bge-m3:  ${bgeRes.vectors.length} vectors in ${bgeRes.latencyMs}ms (${bge.dimension}d)`);
console.log(`Phase B done in ${Math.round((Date.now() - tB) / 1000)}s.`);

// ---------------------------------------------------------------------------
// Phase C — store. Replace any prior contextual-v1 for these sources.
// ---------------------------------------------------------------------------
console.log(`\n--- Phase C: storing ${units.length} contextual-v1 chunks ---`);
for (const source of sourcesToRun) {
  const sid = await sourceIdOf(source.slug);
  const del = await db.query(
    `DELETE FROM chunks WHERE source_id=$1 AND chunking_version=$2`,
    [sid, CHUNKING_VERSION],
  );
  if ((del.rowCount ?? 0) > 0) console.log(`  ${source.slug}: cleared ${del.rowCount} prior`);
}

const COLS = 10;
const values: unknown[] = [];
const tuples: string[] = [];
for (let i = 0; i < units.length; i++) {
  const u = units[i];
  if (!u) continue;
  const bgeVec = bgeRes.vectors[i];
  if (!bgeVec) throw new Error(`Missing bge vector at ${i}`);
  const base = i * COLS;
  tuples.push(
    `($${base + 1},$${base + 2},$${base + 3},$${base + 4},$${base + 5},$${base + 6},$${base + 7},$${base + 8},$${base + 9},$${base + 10})`,
  );
  values.push(
    u.sourceId,
    CHUNKING_VERSION,
    u.chunk.chunkIndex,
    u.chunk.chapter,
    u.chunk.text, // ORIGINAL text — citations/display/span-offsets unchanged
    u.chunk.charStart,
    u.chunk.charEnd,
    null, // OpenAI `embedding` column — unused; this project is local-only
    pgvector.toSql(bgeVec), // CONTEXTUAL embedding (bge-m3, local)
    { tokenCount: u.chunk.tokenCount, context: u.note ?? '' },
  );
}
await db.query(
  `INSERT INTO chunks
     (source_id, chunking_version, chunk_index, chapter, text,
      char_start, char_end, embedding, embedding_bge, metadata)
   VALUES ${tuples.join(', ')}`,
  values,
);
console.log(`Stored ${units.length} chunks.`);

// ---------------------------------------------------------------------------
// Verify
// ---------------------------------------------------------------------------
const ver = await db.query<{ slug: string; n: string; with_bge: string }>(
  `SELECT s.slug, count(c.id) n, count(c.embedding_bge) with_bge
     FROM sources s LEFT JOIN chunks c ON c.source_id=s.id AND c.chunking_version=$1
    GROUP BY s.id, s.slug ORDER BY s.id`,
  [CHUNKING_VERSION],
);
console.log(`\n=== DB verification (${CHUNKING_VERSION}) ===`);
for (const r of ver.rows) console.log(`  ${r.slug.padEnd(20)} chunks=${r.n.padStart(4)} bge=${r.with_bge.padStart(4)}`);
} else {
  console.log(`\nDRY RUN — nothing stored.`);
}

await db.end();

// ===========================================================================
// Helpers
// ===========================================================================
async function sourceIdOf(slug: string): Promise<number> {
  const r = await db.query<{ id: number }>(`SELECT id FROM sources WHERE slug=$1`, [slug]);
  const id = r.rows[0]?.id;
  if (id === undefined) {
    throw new Error(`Source '${slug}' not found — run the naive-v1 ingest first.`);
  }
  return id;
}

/** One context-note call with retry; returns the trimmed note. */
async function chatWithRetry(
  model: string,
  documentText: string,
  citation: string,
  chunkText: string,
  attempts = 3,
): Promise<string> {
  let lastErr: unknown;
  for (let i = 0; i < attempts; i++) {
    try {
      const res = await llm.chat({
        model,
        system: SYSTEM_PROMPT,
        messages: [{ role: 'user', content: buildUserMessage(documentText, citation, chunkText) }],
      });
      const text = res.text.trim();
      if (text.length > 0) return text;
      throw new Error('empty note');
    } catch (err) {
      lastErr = err;
      if (i < attempts - 1) await new Promise((r) => setTimeout(r, 1000 * (i + 1)));
    }
  }
  throw new Error(`chat failed after ${attempts} attempts: ${String(lastErr)}`);
}
