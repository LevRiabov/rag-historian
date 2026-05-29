/**
 * roman-research/ingest/download.ts — fetch + cache Project Gutenberg files.
 *
 * Why a cache at all: Gutenberg occasionally throttles or rate-limits; more
 * importantly, ingest is something we'll run many times across Modules 4–6
 * as chunking + embedding strategies evolve. Downloading the same ~4 MB of
 * text once and reusing it is just hygiene.
 *
 * Cache layout:
 *   corpus/raw/pg<id>.txt    — raw downloaded file, byte-identical to Gutenberg
 *
 * (We add corpus/cleaned/<slug>.txt in Turn B once chunking + DB insert run —
 * post-Gutenberg-strip, post-extraction text the source viewer can render.)
 *
 * Idempotent by design: re-running this function is a no-op if the file
 * exists. Use `force: true` to re-download (e.g., when verifying that
 * Gutenberg hasn't changed the file).
 */
import { mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';

import { gutenbergTextUrl, type SourceManifest } from './sources.ts';

const CORPUS_RAW_DIR = path.join(process.cwd(), 'corpus', 'raw');

export interface DownloadResult {
  /** Absolute path to the cached file on disk. */
  path: string;
  /** Bytes on disk (matches Gutenberg's reported size). */
  bytes: number;
  /** Whether this call hit the network (false = cache hit). */
  fetched: boolean;
}

/**
 * Download a Gutenberg file if not cached. Returns the cached path.
 *
 * Multiple sources can share the same gutenbergId (Caesar's works are
 * bundled in one file). This is keyed by ID, so a second call for the
 * second source is a cache hit even on a cold start.
 */
export async function downloadGutenbergFile(id: number, force = false): Promise<DownloadResult> {
  await mkdir(CORPUS_RAW_DIR, { recursive: true });
  const filename = `pg${id}.txt`;
  const filepath = path.join(CORPUS_RAW_DIR, filename);

  if (!force) {
    try {
      const s = await stat(filepath);
      return { path: filepath, bytes: s.size, fetched: false };
    } catch {
      // ENOENT — file doesn't exist, fall through to fetch
    }
  }

  const url = gutenbergTextUrl(id);
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }
  const text = await response.text();
  await writeFile(filepath, text, 'utf-8');
  return { path: filepath, bytes: Buffer.byteLength(text, 'utf-8'), fetched: true };
}

/**
 * Convenience: download all unique Gutenberg files referenced by a list of
 * sources. Deduplicates by ID so a shared file is only fetched once.
 */
export async function downloadAll(
  sources: SourceManifest[],
  force = false,
): Promise<Map<number, DownloadResult>> {
  const ids = new Set(sources.map((s) => s.gutenbergId));
  const results = new Map<number, DownloadResult>();
  for (const id of ids) {
    results.set(id, await downloadGutenbergFile(id, force));
  }
  return results;
}

/** Read a previously-downloaded raw file by Gutenberg ID. */
export async function readRawFile(id: number): Promise<string> {
  const filepath = path.join(CORPUS_RAW_DIR, `pg${id}.txt`);
  return readFile(filepath, 'utf-8');
}
