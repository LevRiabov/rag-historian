/**
 * roman-research/ingest/sources.ts — the canonical corpus manifest.
 *
 * Four sources, all primary, all focused on Julius Caesar's career and death.
 * Pinned to specific Project Gutenberg editions so the ingest is reproducible:
 * anyone cloning the repo gets byte-identical text. (Gutenberg occasionally
 * updates files; if checksums ever diverge we'll add SHA hashes here.)
 *
 * Note on file-sharing: `caesar-gallic-war` and `caesar-civil-war` share
 * Gutenberg ID 10657 — the "De Bello Gallico and Other Commentaries"
 * edition bundles both works in one file. The downloader dedupes by URL;
 * the parser extracts each work's portion using `extractMarkers` below.
 *
 * Each source is parsed independently because:
 *   - Citations differ: "Caesar, Gallic War, Book IV, §32" is more useful
 *     than "Caesar's Commentaries, somewhere"
 *   - `year_written` differs (Gallic ~50 BC, Civil ~48 BC)
 *   - Contradiction-detection across sources requires source-level identity
 *
 * Tier field: all 4 are 'primary'. A historian-tier source (Mommsen) gets
 * added in Module 6 only if the eval set shows we need that voice.
 */

export type SourceTier = 'primary' | 'historian';

/**
 * Where in the raw file this source's text lives. The parser uses these
 * markers to find the start/end of the relevant portion within a Gutenberg
 * file that may contain other works.
 *
 *   - `startMarker` — string that immediately PRECEDES the source text
 *     (the text after this string is what we want)
 *   - `endMarker`   — string that comes AFTER the source text (text before
 *     this string is what we want). `null` means "until end of file".
 *
 * The parser searches for the SECOND occurrence of `startMarker` when
 * `startOccurrence: 2` is set — used for Caesar's works where the
 * first match is in a table-of-contents listing at the top.
 */
export interface ExtractMarkers {
  startMarker: string;
  startOccurrence?: number; // default 1
  endMarker: string | null;
}

export interface SourceManifest {
  /** URL-safe stable identifier. Used as DB sources.slug AND parser dispatch key. */
  slug: string;
  /** Human-readable title with original-language hint. */
  title: string;
  /** Author name as we want it to appear in citations. */
  author: string;
  tier: SourceTier;
  /** Approximate year written. Negative for BC. */
  yearWritten: number;
  /** Translator + edition info for the English text we ingest. */
  translator: string;
  language: string;
  /** Project Gutenberg ID; URL is derived from this. */
  gutenbergId: number;
  /** Where in the raw file this source's text starts and ends. */
  extract: ExtractMarkers;
  /** Free-form metadata for prompts / eval / display. */
  metadata?: Record<string, unknown>;
}

/**
 * Compute the canonical plain-text UTF-8 URL for a Gutenberg ID.
 * Both `/cache/epub/<id>/pg<id>.txt` and `/ebooks/<id>.txt.utf-8` work;
 * the cache pattern is more stable across Gutenberg upgrades.
 */
export function gutenbergTextUrl(id: number): string {
  return `https://www.gutenberg.org/cache/epub/${id}/pg${id}.txt`;
}

export const SOURCES: SourceManifest[] = [
  {
    slug: 'caesar-gallic-war',
    title: 'The Gallic War (De Bello Gallico)',
    author: 'Julius Caesar',
    tier: 'primary',
    yearWritten: -50,
    translator: 'W. A. McDevitte & W. S. Bohn (1869)',
    language: 'en',
    gutenbergId: 10657,
    extract: {
      // The TOC lists "THE WAR IN GAUL" first, then the actual section header
      // is the second occurrence. The actual text starts with the BOOK I marker.
      startMarker: 'THE WAR IN GAUL',
      startOccurrence: 2,
      endMarker: 'THE CIVIL WAR',
    },
    metadata: {
      workType: 'self-account',
      topic: 'gallic-war',
      note: 'First-person, written during the campaigns. Highly self-serving — Caesar wrote it as political dispatches to Rome.',
    },
  },
  {
    slug: 'caesar-civil-war',
    title: 'The Civil War (De Bello Civili)',
    author: 'Julius Caesar',
    tier: 'primary',
    yearWritten: -48,
    translator: 'W. A. McDevitte & W. S. Bohn (1869)',
    language: 'en',
    gutenbergId: 10657, // SAME file as gallic-war — downloader dedupes
    extract: {
      // Same TOC issue — first occurrence is in the contents listing.
      startMarker: 'THE CIVIL WAR',
      startOccurrence: 2,
      endMarker: null, // runs to end of file (before Gutenberg footer)
    },
    metadata: {
      workType: 'self-account',
      topic: 'civil-war',
      note: 'First-person, fighting Pompey. Same biases as Gallic War, different war.',
    },
  },
  {
    slug: 'plutarch-caesar',
    title: 'Life of Caesar (Parallel Lives, Vol III)',
    author: 'Plutarch',
    tier: 'primary',
    yearWritten: 100, // approximate; Plutarch lived 46-120 AD
    translator: 'Aubrey Stewart & George Long (1892)',
    language: 'en',
    gutenbergId: 14140,
    extract: {
      // Note the Æ ligature — that's how it appears in the source text.
      startMarker: 'LIFE OF C. CÆSAR.',
      endMarker: 'LIFE OF PHOKION.',
    },
    metadata: {
      workType: 'biography',
      writtenYearsLater: 150,
      perspective: 'Greek',
      topic: 'caesar',
      note: 'Plutarch paired Caesar with Alexander as parallel biographies. Moralizing, anecdotal.',
    },
  },
  {
    slug: 'suetonius-caesar',
    title: 'Life of Julius Caesar (The Twelve Caesars)',
    author: 'Suetonius',
    tier: 'primary',
    yearWritten: 121,
    translator: 'Alexander Thomson (1796)',
    language: 'en',
    gutenbergId: 6400,
    extract: {
      // Gutenberg's edition omits the Æ here — uses 'CASAR' instead of 'CÆSAR'.
      // OCR / transcription artifact; we match what's actually in the file.
      startMarker: 'CAIUS JULIUS CASAR.',
      endMarker: 'D. OCTAVIUS CAESAR AUGUSTUS.',
    },
    metadata: {
      workType: 'biography',
      writtenYearsLater: 170,
      perspective: 'Roman',
      topic: 'caesar',
      note: 'Suetonius was an imperial gossip historian. Vivid, salacious, full of anecdotes Tacitus would never print.',
    },
  },
];

export function findSource(slug: string): SourceManifest {
  const source = SOURCES.find((s) => s.slug === slug);
  if (!source) {
    throw new Error(
      `Unknown source slug: '${slug}'. Known: ${SOURCES.map((s) => s.slug).join(', ')}`,
    );
  }
  return source;
}
