/**
 * roman-research/ingest/parse.ts — extract a source's text from a raw
 * Gutenberg file and split it into structured sections.
 *
 * Per-source parsers because each text is structured differently:
 *
 *   caesar-gallic-war / caesar-civil-war:
 *     Hierarchy: Book (I-VIII for Gallic, I-III for Civil) → Section
 *     Section marker: "I.--", "II.--", ... (Roman numeral, period, double dash)
 *     Book marker:    "BOOK I", "BOOK II", ... on its own line
 *
 *   plutarch-caesar:
 *     Flat (no books).
 *     Section marker: "I.", "II.", "III.", ... (optionally followed by a
 *     footnote bracket like "I.[435]")
 *
 *   suetonius-caesar:
 *     Flat (no books).
 *     Section marker: "I.", "II.", ... (followed by whitespace + content)
 *
 * Section detection runs on the EXTRACTED source text (Gutenberg-stripped,
 * source-portion-extracted), so `charStart`/`charEnd` are offsets into the
 * `cleanedText` field of the returned `ParsedSource` — what the source
 * viewer will render later. Not offsets into the original raw file.
 */

import { findSource, type SourceManifest } from './sources.ts';

// ============================================================================
// Types
// ============================================================================

export interface ParsedSection {
  /** 'Book I' / 'Book II' / null if the source has no book hierarchy. */
  bookLabel: string | null;
  /** Roman numeral as written: 'I', 'XLII', 'LXXIX'. */
  chapterLabel: string;
  /** The section text, trimmed. */
  text: string;
  /** Offset of section start in the parent ParsedSource.cleanedText. */
  charStart: number;
  /** Offset of section end in the parent ParsedSource.cleanedText. */
  charEnd: number;
}

export interface ParsedSource {
  /** Stable slug — matches sources.ts. */
  slug: string;
  /**
   * The source's text after Gutenberg-header strip + extraction of the
   * relevant portion (e.g., just Caesar's Life out of Plutarch Vol III).
   * `charStart`/`charEnd` on sections are offsets into THIS string.
   */
  cleanedText: string;
  sections: ParsedSection[];
}

// ============================================================================
// Shared helpers
// ============================================================================

/**
 * Remove Project Gutenberg header (preamble) and footer (license).
 *
 * Standard markers since ~2018:
 *   *** START OF THE PROJECT GUTENBERG EBOOK <title> ***
 *   *** END OF THE PROJECT GUTENBERG EBOOK <title> ***
 *
 * If either marker is missing we throw — better to fail loudly than silently
 * include hundreds of lines of license text in our embeddings.
 */
export function stripGutenberg(raw: string): string {
  const startRe = /^\*+\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*+$/m;
  const endRe = /^\*+\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*+$/m;

  const startMatch = startRe.exec(raw);
  if (!startMatch) throw new Error('Project Gutenberg START marker not found.');
  const endMatch = endRe.exec(raw);
  if (!endMatch) throw new Error('Project Gutenberg END marker not found.');

  const startIdx = startMatch.index + startMatch[0].length;
  const endIdx = endMatch.index;
  return raw.slice(startIdx, endIdx).trim();
}

/**
 * Extract the portion of the file that belongs to this source, using the
 * `extract` markers from the manifest. For files that bundle multiple works,
 * this slices out the relevant work.
 */
export function extractPortion(stripped: string, source: SourceManifest): string {
  const { startMarker, startOccurrence = 1, endMarker } = source.extract;

  // Find the Nth occurrence of startMarker. We search for it after each prior
  // match, so "second occurrence" really means second standalone match.
  let startIdx = -1;
  let searchFrom = 0;
  for (let n = 0; n < startOccurrence; n++) {
    startIdx = stripped.indexOf(startMarker, searchFrom);
    if (startIdx === -1) {
      throw new Error(
        `[${source.slug}] startMarker '${startMarker}' (occurrence ${startOccurrence}) not found in stripped text.`,
      );
    }
    searchFrom = startIdx + startMarker.length;
  }

  // Start text AFTER the marker line itself (so the marker isn't part of body).
  const bodyStart = startIdx + startMarker.length;

  let bodyEnd: number;
  if (endMarker === null) {
    bodyEnd = stripped.length;
  } else {
    bodyEnd = stripped.indexOf(endMarker, bodyStart);
    if (bodyEnd === -1) {
      throw new Error(`[${source.slug}] endMarker '${endMarker}' not found after start.`);
    }
  }

  return stripped.slice(bodyStart, bodyEnd).trim();
}

/**
 * Normalize whitespace without destroying paragraph structure.
 *   - Convert CRLF/CR → LF
 *   - Collapse 3+ consecutive blank lines into 2 (one blank line = paragraph break)
 *   - Strip trailing whitespace from each line
 *
 * Doesn't touch inside-paragraph line wrapping (Gutenberg wraps at ~70 chars);
 * the chunker in Turn B can re-flow if it cares.
 */
function normalizeWhitespace(text: string): string {
  return text
    .replace(/\r\n?/g, '\n')
    .split('\n')
    .map((line) => line.replace(/\s+$/, ''))
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

// ============================================================================
// Section detection
// ============================================================================

const ROMAN_VALUES: Record<string, number> = {
  I: 1,
  V: 5,
  X: 10,
  L: 50,
  C: 100,
  D: 500,
  M: 1000,
};

/** Convert a Roman numeral string to its integer value. Handles subtractive
 *  notation (IV=4, IX=9, XL=40, etc.). Returns NaN for invalid characters. */
function romanToInt(roman: string): number {
  let total = 0;
  for (let i = 0; i < roman.length; i++) {
    const cur = ROMAN_VALUES[roman[i] as string];
    if (cur === undefined) return Number.NaN;
    const next = ROMAN_VALUES[roman[i + 1] as string] ?? 0;
    total += cur < next ? -cur : cur;
  }
  return total;
}

/** Maximum gap we'll tolerate between consecutive detected sections.
 *  Higher means more forgiveness for textual lacunae; lower means stricter
 *  rejection of false positives. 3 handles Suetonius's known gaps without
 *  letting a name like "C. Marius" (value 100) sneak in mid-document. */
const MAX_SECTION_SKIP = 3;

/**
 * Filter candidate sections to a monotonically-increasing sequence starting
 * near 1. Rejects matches whose Roman-numeral value isn't plausibly the
 * "next" section — most commonly single-letter abbreviations in proper
 * names ("C. Marius", "L. Sulla", "M. Cato") that the regex catches by
 * mistake. Tolerates `MAX_SECTION_SKIP` consecutive missing sections to
 * survive lacunae in the source manuscripts.
 */
function filterToMonotonicSequence(candidates: ParsedSection[]): ParsedSection[] {
  const accepted: ParsedSection[] = [];
  let expected = 1;
  for (const c of candidates) {
    const value = romanToInt(c.chapterLabel);
    if (!Number.isNaN(value) && value >= expected && value <= expected + MAX_SECTION_SKIP) {
      accepted.push(c);
      expected = value + 1;
    }
    // else: silently drop (false positive)
  }
  return accepted;
}

/**
 * After filtering rejected some candidates, the surviving sections have
 * stale `charEnd` (they pointed at the next candidate, which may have been
 * rejected). Re-compute each section's body to span from its start to the
 * NEXT ACCEPTED section's start, instead of to the next raw match.
 */
function rewireSectionBoundaries(
  accepted: ParsedSection[],
  text: string,
  baseOffset: number,
): ParsedSection[] {
  return accepted.map((s, i) => {
    const next = accepted[i + 1];
    const localStart = s.charStart - baseOffset;
    const localEnd = next ? next.charStart - baseOffset : text.length;
    return {
      ...s,
      text: text.slice(localStart, localEnd).trim(),
      charEnd: baseOffset + localEnd,
    };
  });
}

/**
 * Given a text and a regex that matches section headers (with the chapter
 * label in group 1), return ordered sections passing sequence validation.
 * Each section's `text` includes the header line itself (kept for citation
 * context); `charStart`/`charEnd` are offsets into the parent cleanedText.
 *
 * If `bookLabel` is provided, every returned section gets that label —
 * caller uses this when iterating book-by-book.
 */
function splitBySectionRegex(
  text: string,
  sectionRegex: RegExp,
  bookLabel: string | null,
  baseOffset: number,
): ParsedSection[] {
  const matches = [...text.matchAll(sectionRegex)];
  if (matches.length === 0) return [];

  const candidates: ParsedSection[] = [];
  for (let i = 0; i < matches.length; i++) {
    const m = matches[i];
    if (!m || m.index === undefined || !m[1]) continue;
    const sectionStart = m.index;
    const nextMatch = matches[i + 1];
    const sectionEnd = nextMatch?.index ?? text.length;
    candidates.push({
      bookLabel,
      chapterLabel: m[1],
      text: text.slice(sectionStart, sectionEnd).trim(),
      charStart: baseOffset + sectionStart,
      charEnd: baseOffset + sectionEnd,
    });
  }

  const accepted = filterToMonotonicSequence(candidates);
  return rewireSectionBoundaries(accepted, text, baseOffset);
}

// ============================================================================
// Per-source parsers
// ============================================================================

/**
 * Caesar (Gallic War + Civil War): Book → Section hierarchy.
 *
 * Books are marked by lines like "BOOK I" on their own line. Within each
 * book, sections are "I.--text", "II.--text" (Roman numeral + period +
 * double-dash + content, all in one paragraph).
 */
function parseCaesarPortion(cleanedText: string): ParsedSection[] {
  // Match BOOK markers on their own lines: "BOOK I", "BOOK II", ..., "BOOK VIII".
  const bookRe = /^BOOK ([IVX]+)$/gm;
  const bookMatches = [...cleanedText.matchAll(bookRe)];
  if (bookMatches.length === 0) {
    // Fall back: treat the whole thing as one book if no BOOK markers found.
    return splitBySectionRegex(cleanedText, /^([IVXLC]+)\.--/gm, null, 0);
  }

  const sections: ParsedSection[] = [];
  for (let i = 0; i < bookMatches.length; i++) {
    const m = bookMatches[i];
    if (!m || m.index === undefined || !m[1]) continue;
    const bookLabel = `Book ${m[1]}`;
    const bookStart = m.index + m[0].length;
    const nextBook = bookMatches[i + 1];
    const bookEnd = nextBook?.index ?? cleanedText.length;
    const bookBody = cleanedText.slice(bookStart, bookEnd);
    // Section regex: "I.--", "II.--", ... at start of line. Caesar's
    // distinctive double-dash separator avoids false positives.
    const bookSections = splitBySectionRegex(
      bookBody,
      /^([IVXLC]+)\.--/gm,
      bookLabel,
      bookStart,
    );
    sections.push(...bookSections);
  }
  return sections;
}

/**
 * Plutarch's Life of Caesar: flat sections, Roman numerals with optional
 * inline footnote markers.
 *
 * Examples from the text:
 *   I.[435] When Sulla got possession...
 *   II. The pirates asked Cæsar twenty talents...
 */
function parsePlutarchPortion(cleanedText: string): ParsedSection[] {
  return splitBySectionRegex(
    cleanedText,
    /^([IVXLC]+)\.(?:\[\d+\])?\s+/gm,
    null,
    0,
  );
}

/**
 * Suetonius's Life of Julius Caesar: flat sections, Roman numerals followed
 * by period + (usually double) space + content.
 *
 * Example: "VIII.  Quitting therefore the province..."
 */
function parseSuetoniusPortion(cleanedText: string): ParsedSection[] {
  return splitBySectionRegex(
    cleanedText,
    /^([IVXLC]+)\.\s+/gm,
    null,
    0,
  );
}

// ============================================================================
// Dispatch
// ============================================================================

type SectionParser = (cleanedText: string) => ParsedSection[];

const SECTION_PARSERS: Record<string, SectionParser> = {
  'caesar-gallic-war': parseCaesarPortion,
  'caesar-civil-war': parseCaesarPortion,
  'plutarch-caesar': parsePlutarchPortion,
  'suetonius-caesar': parseSuetoniusPortion,
};

/**
 * Parse a raw Gutenberg file for one source: strip → extract portion →
 * normalize → split into sections.
 *
 * Returns `{ cleanedText, sections }` — the source viewer in Module 10 will
 * render `cleanedText` (or a slice of it via char_start/char_end on a chunk)
 * to show the user the original passage.
 */
export function parseSource(slug: string, rawFile: string): ParsedSource {
  const source = findSource(slug);
  const stripped = stripGutenberg(rawFile);
  const portion = extractPortion(stripped, source);
  const cleanedText = normalizeWhitespace(portion);

  const sectionParser = SECTION_PARSERS[slug];
  if (!sectionParser) {
    throw new Error(`No section parser registered for '${slug}'.`);
  }
  const sections = sectionParser(cleanedText);

  return { slug, cleanedText, sections };
}
