/**
 * roman-research/ingest/chunk.ts — section-aware token chunker.
 *
 * Strategy:
 *   1. One section → one chunk if it fits in `targetTokens`.
 *   2. Long section → split at paragraph boundaries, pack greedily,
 *      include trailing-paragraph overlap so a fact split across the
 *      seam still appears in both chunks.
 *   3. NEVER cross section boundaries. A chunk belongs to exactly one
 *      (book, section) pair — keeps citations precise and avoids the
 *      "averaged-meaning vector" problem from mixing topics.
 *
 * Token counting uses `cl100k_base` (OpenAI text-embedding-3 tokenizer).
 * BGE-M3 uses a different tokenizer (XLM-RoBERTa) that produces ~10-20%
 * more tokens for English; both still fit comfortably under their 8k
 * context limits when our target is 500. Counting once against
 * cl100k_base is accurate for OpenAI billing and good enough for BGE.
 *
 * Why paragraph boundaries (not sentence boundaries) for splits: Gutenberg
 * texts have well-defined paragraph breaks; sentence detection in Latin
 * translations is messy (lots of "M. Cato" / "Cn. Pompey" abbreviations
 * that look like sentence-enders). Paragraphs are coarser but always
 * meaningful units. If a single paragraph exceeds `targetTokens` (rare),
 * we emit it as one oversize chunk rather than trying to split mid-prose.
 */

import { getEncoding, type Tiktoken } from 'js-tiktoken';

import type { ParsedSection } from './parse.ts';

// cl100k_base: the tokenizer for text-embedding-3-small/large, GPT-3.5/4.
// Eagerly loaded; encoding data ships in the package, no async init needed.
const ENCODER: Tiktoken = getEncoding('cl100k_base');

function countTokens(text: string): number {
  return ENCODER.encode(text).length;
}

export interface ChunkOptions {
  /** Target token budget per chunk. ~500 is the conventional naive default. */
  targetTokens: number;
  /** Tokens of overlap between consecutive chunks of the SAME section.
   *  Helps when a key fact straddles the split. Set to 0 for no overlap. */
  overlapTokens: number;
}

export const DEFAULT_CHUNK_OPTIONS: ChunkOptions = {
  targetTokens: 500,
  overlapTokens: 50,
};

export interface ChunkVariant {
  /** The `chunking_version` label stored on every row this variant produces.
   *  Variants coexist in the `chunks` table, discriminated by this string. */
  version: string;
  /** One-line rationale — surfaced in ingest logs + the Module 6 notes. */
  description: string;
  /** Build this variant's chunks from a source's parsed sections. Most
   *  variants are `chunkSections()` with different options; parent-child uses
   *  its own builder (small children + parent-section pointers). */
  chunk: (sections: ParsedSection[]) => Chunk[];
}

/**
 * Module 6.1 granularity sweep.
 *
 * The flat variants are just `chunkSections()` with different options — the
 * splitter already does "one section → one chunk if it fits, else
 * paragraph-pack to target":
 *   - a SMALL target (window-300) yields finer chunks with tighter vectors;
 *   - a LARGE target with 0 overlap (chapter) yields one-chunk-per-section,
 *     splitting only when a section exceeds the embedder-safe ceiling. The
 *     6000-token target sits under BGE-M3's 8192 context (BGE's XLM-RoBERTa
 *     tokenizer runs ~10-20% hotter than the cl100k count we track, so 6000
 *     cl100k ≈ ~7000 BGE tokens — headroom intact, no silent truncation).
 *
 * `parent-child-v1` is the interesting one: it stores naive-sized (500/50)
 * CHILDREN — so retrieval recall is identical to naive-v1 by construction
 * (same vectors, a clean A/B) — but each child carries a pointer to its full
 * parent SECTION. At generation time the retrieved children expand to their
 * parents (see retrieve.ts `expandToParents`), giving the LLM coherent
 * section-level context without the recall-inflating granularity bias of
 * embedding whole chapters. Isolates ONE variable vs naive: parent context.
 */
export const CHUNKING_VARIANTS: Record<string, ChunkVariant> = {
  'naive-v1': {
    version: 'naive-v1',
    description: 'Module 4/5 baseline — 500-token windows, 50 overlap.',
    chunk: (s) => chunkSections(s, { targetTokens: 500, overlapTokens: 50 }),
  },
  'window-300-v1': {
    version: 'window-300-v1',
    description: 'Finer windows — tighter topical vectors, ~more chunks.',
    chunk: (s) => chunkSections(s, { targetTokens: 300, overlapTokens: 30 }),
  },
  'chapter-v1': {
    version: 'chapter-v1',
    description: 'Whole section as one chunk; split only past embedder ceiling.',
    chunk: (s) => chunkSections(s, { targetTokens: 6000, overlapTokens: 0 }),
  },
  'parent-child-v1': {
    version: 'parent-child-v1',
    description: 'Naive-sized children (retrieve precise) + parent-section expansion at gen time.',
    chunk: (s) => chunkSectionsParentChild(s, { targetTokens: 500, overlapTokens: 50 }),
  },
};

export interface Chunk {
  /** 0-based, sequential within (source, chunking_version). */
  chunkIndex: number;
  /** Human-readable citation label: "Book I, §VII" or "§XXIV". */
  chapter: string;
  text: string;
  /** Offset of chunk start in ParsedSource.cleanedText. */
  charStart: number;
  /** Offset of chunk end in ParsedSource.cleanedText. */
  charEnd: number;
  /** Accurate cl100k_base token count for billing + sanity. */
  tokenCount: number;
  /** Parent-document retrieval (parent-child-v1 only): the full parent
   *  SECTION this child belongs to. Embedded + retrieved on the child text,
   *  but expanded to `parent.text` before being handed to the generator.
   *  Persisted into the chunk's metadata JSONB; absent for flat variants. */
  parent?: {
    text: string;
    charStart: number;
    charEnd: number;
  };
}

function formatChapter(section: ParsedSection): string {
  return section.bookLabel
    ? `${section.bookLabel}, §${section.chapterLabel}`
    : `§${section.chapterLabel}`;
}

interface Paragraph {
  text: string;
  /** Offset within the parent section's text. */
  localStart: number;
  localEnd: number;
  tokens: number;
}

/** Split a section's text into paragraphs with offsets + token counts. */
function paragraphsOf(sectionText: string): Paragraph[] {
  const paras: Paragraph[] = [];
  // Match content runs separated by 2+ newlines. Using a regex with capture
  // for separators keeps us in sync with the original char offsets.
  const splitRe = /\n{2,}/g;
  let cursor = 0;
  for (const match of sectionText.matchAll(splitRe)) {
    const end = match.index;
    if (end === undefined) continue;
    const para = sectionText.slice(cursor, end);
    if (para.trim().length > 0) {
      paras.push({
        text: para,
        localStart: cursor,
        localEnd: end,
        tokens: countTokens(para),
      });
    }
    cursor = end + match[0].length;
  }
  // Trailing paragraph after the last separator (or the whole text if no
  // separators were found).
  if (cursor < sectionText.length) {
    const tail = sectionText.slice(cursor);
    if (tail.trim().length > 0) {
      paras.push({
        text: tail,
        localStart: cursor,
        localEnd: sectionText.length,
        tokens: countTokens(tail),
      });
    }
  }
  return paras;
}

/**
 * Split a single section into chunks. Returns chunks without final
 * chunkIndex / chapter fields — caller assigns those when assembling
 * the full source-level chunk list.
 */
function splitSection(
  section: ParsedSection,
  opts: ChunkOptions,
): Array<Omit<Chunk, 'chunkIndex' | 'chapter'>> {
  const wholeTokens = countTokens(section.text);
  if (wholeTokens <= opts.targetTokens) {
    return [
      {
        text: section.text,
        charStart: section.charStart,
        charEnd: section.charEnd,
        tokenCount: wholeTokens,
      },
    ];
  }

  const paras = paragraphsOf(section.text);
  if (paras.length === 0) {
    // Pathological: section had no paragraph breaks AND exceeded target.
    // Emit it whole; an oversize chunk is better than a missing one.
    return [
      {
        text: section.text,
        charStart: section.charStart,
        charEnd: section.charEnd,
        tokenCount: wholeTokens,
      },
    ];
  }

  const chunks: Array<Omit<Chunk, 'chunkIndex' | 'chapter'>> = [];
  let i = 0;

  while (i < paras.length) {
    // Greedy pack: take paragraphs starting at i until adding the next
    // would exceed targetTokens. Always include at least one (even if it
    // overshoots) to guarantee forward progress.
    let j = i + 1;
    let tokens = paras[i]?.tokens ?? 0;
    while (j < paras.length) {
      const next = paras[j];
      if (!next) break;
      if (tokens + next.tokens > opts.targetTokens) break;
      tokens += next.tokens;
      j++;
    }

    const first = paras[i];
    const last = paras[j - 1];
    if (!first || !last) break;

    const localStart = first.localStart;
    const localEnd = last.localEnd;
    chunks.push({
      text: section.text.slice(localStart, localEnd),
      charStart: section.charStart + localStart,
      charEnd: section.charStart + localEnd,
      tokenCount: tokens,
    });

    // Slide window forward, leaving overlap. Walk back from j while the
    // accumulated trailing paragraphs fit in overlapTokens.
    let overlapStart = j;
    let overlapTokens = 0;
    while (overlapStart > i) {
      const para = paras[overlapStart - 1];
      if (!para) break;
      if (overlapTokens + para.tokens > opts.overlapTokens) break;
      overlapStart--;
      overlapTokens += para.tokens;
    }
    // Forward-progress guarantee: never re-emit the same starting index.
    if (overlapStart <= i) overlapStart = i + 1;
    i = overlapStart;
  }

  return chunks;
}

/**
 * Chunk all sections of a parsed source. Returns chunks with sequential
 * `chunkIndex` (0-based across the entire source) and `chapter` labels
 * derived from each section's book + chapter.
 */
export function chunkSections(
  sections: ParsedSection[],
  opts: ChunkOptions = DEFAULT_CHUNK_OPTIONS,
): Chunk[] {
  const chunks: Chunk[] = [];
  for (const section of sections) {
    const chapter = formatChapter(section);
    for (const partial of splitSection(section, opts)) {
      chunks.push({
        chunkIndex: chunks.length,
        chapter,
        text: partial.text,
        charStart: partial.charStart,
        charEnd: partial.charEnd,
        tokenCount: partial.tokenCount,
      });
    }
  }
  return chunks;
}

/**
 * Parent-document chunker. Splits each section into naive-sized CHILDREN (the
 * embedded/retrieved unit) but tags every child with its full parent SECTION.
 * Children are identical to what `chunkSections` would produce for the same
 * options — so retrieval recall matches that flat variant exactly — while the
 * parent pointer lets the query layer expand to section context for the LLM.
 */
export function chunkSectionsParentChild(
  sections: ParsedSection[],
  childOpts: ChunkOptions = DEFAULT_CHUNK_OPTIONS,
): Chunk[] {
  const chunks: Chunk[] = [];
  for (const section of sections) {
    const chapter = formatChapter(section);
    const parent = {
      text: section.text,
      charStart: section.charStart,
      charEnd: section.charEnd,
    };
    for (const partial of splitSection(section, childOpts)) {
      chunks.push({
        chunkIndex: chunks.length,
        chapter,
        text: partial.text,
        charStart: partial.charStart,
        charEnd: partial.charEnd,
        tokenCount: partial.tokenCount,
        parent,
      });
    }
  }
  return chunks;
}

/** Sum token counts across an array of chunks (analytics helper). */
export function totalTokens(chunks: Chunk[]): number {
  return chunks.reduce((sum, c) => sum + c.tokenCount, 0);
}
