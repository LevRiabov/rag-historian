# Roman Research Agent

A historian's-assistant RAG system over a curated corpus of Julius Caesar primary sources, built as the case-study artifact for Phase 1 of an AI-engineering foundation. **Headline feature:** surfaces contradictions across sources rather than presenting a single synthesized answer.

> This file documents project scope + architecture. For module sequencing and the AI-engineering plan, see [`../phase-1-foundation.md`](../phase-1-foundation.md). For Claude Code working notes, see [`../CLAUDE.md`](../CLAUDE.md).

## Vision

A tool a historian (or curious reader) would actually use:

- **Answers questions** about Caesar's career, the Gallic and Civil Wars, his dictatorship, and his death — from primary sources, with citations.
- **Surfaces contradictions** when sources disagree. Caesar wrote about his own wars; Plutarch and Suetonius wrote about them 150 years later from different cultural angles. When they disagree on what happened or why, the system shows both — it doesn't pretend they agree.
- **Distinguishes source tiers** (primary vs historian). Caesar's own narrative carries different epistemic weight than a later biographer's interpretation. Prompts treat them accordingly.
- **Cites everything**: author, work, chapter, and a clickable link to the original passage in context (chunk + neighbors).
- **Suggests follow-up questions** after each answer so users can extend their research naturally.
- **Optionally**: a multi-step research agent that produces longer structured articles (Module 7+).

## Corpus

Tight topical focus on Caesar (~400–500 pages total). All Project Gutenberg English translations, public domain. Overlap on key events is the point — that's what makes contradiction detection meaningful.

| Source | Author | Tier | Written | Vantage point |
|---|---|---|---|---|
| The Gallic War | Caesar | primary | ~50 BC | First-person, written during the campaigns. Self-serving. |
| The Civil War | Caesar | primary | ~48 BC | First-person, fighting Pompey. Same biases, different war. |
| Life of Caesar (Parallel Lives) | Plutarch | primary | ~75–100 AD | Greek biographer, ~150 yrs later. Moralizing, anecdotal. |
| Life of Julius Caesar (Twelve Caesars) | Suetonius | primary | ~120 AD | Roman gossip historian, ~170 yrs later. Salacious, vivid. |

Natural contradiction tests:
- **Crossing the Rubicon** — Civil War (Caesar's framing) vs Plutarch vs Suetonius
- **Battle of Pharsalus** — Civil War vs Plutarch
- **Caesar's death** — Plutarch vs Suetonius (Caesar obviously can't narrate it)
- **Gallic War events** — Caesar's commentaries vs Plutarch's biography

If the eval set (Module 5) shows we need a *historian-tier* voice for contrast, **Mommsen's *History of Rome*** (also Project Gutenberg, 19th C — the standard modern-ish source on Caesar) is the natural addition. Hold for Module 6.

## Architecture

### Data flow

```
Project Gutenberg
       │
       ▼
   ingest/  ──► sources + chunks tables (pgvector)
                          │
                          ▼
                       query/  ──► retrieved chunks + citations ──► Claude ──► answer
                          │
                          ├──► (Module 7) agent/      — multi-step research, structured articles
                          ├──► (Module 8) mcp-server/ — expose corpus as MCP tools
                          └──► (Module 10) web/       — minimal React SPA for demo
```

### Database schema

Two tables in pgvector, defined in [`../db/init/02-schema.sql`](../db/init/02-schema.sql):

- **sources** — one row per source document. Carries the metadata that prompts will reference: `author`, `title`, `tier`, `year_written`, `translator`, `source_url`. Source-aware prompts ("Plutarch wrote this 150 years after the events; Caesar wrote this *during* them — treat them differently") read from here.
- **chunks** — one row per text chunk. Carries `source_id`, `chunk_index` (order within source for prev/next neighbor lookups), `chapter`, `text`, `char_start`/`char_end` (offsets in the original for the source viewer), and TWO embedding columns (`embedding` = OpenAI 1536d, `embedding_bge` = BGE-M3 1024d) for A/B comparison without re-ingest.
- **`chunking_version` discriminator** on `chunks` lets structure-aware chunking (Module 6.1) coexist with the naive baseline in the same table, distinguished by a string label.

### Why this schema shape

Most RAG techniques in Module 6 are **query-time, not storage-time**, so the database doesn't bloat across the comparison matrix:

| Module 6 technique | Storage change |
|---|---|
| 6.1 Structure-aware chunking | New rows with different `chunking_version` |
| 6.2 Hybrid search (BM25 + vector) | One generated `tsvector` column |
| 6.3 Reranking (Cohere / BGE-reranker) | **None** — pure query-time |
| 6.4 Contextual retrieval | New `augmented_text` column or join table (TBD) |
| 6.5 Query rewriting / HyDE | **None** — pure query-time |

Comparison runs live in the **eval harness** (Module 5), not in the database. Each strategy is a different function against the same stored chunks.

## Folder structure

```
roman-research/
├── README.md          # this file — vision + architecture + decisions log
├── ingest/            # Module 4: download → parse → chunk → embed → store
├── query/             # Module 4: retrieve top-K → format with citations → answer
├── agent/             # Module 7: multi-step research agent (later)
├── mcp-server/        # Module 8: expose corpus as MCP tools (later)
└── web/               # Module 10: minimal React SPA for case-study demo (later)
```

Folders are created when their module ships — no premature empty directories.

Related, at the repo root:
- [`../corpus/`](../corpus/) — downloaded Project Gutenberg texts. **Gitignored** (large, downloadable).
- [`../db/init/`](../db/init/) — Postgres + pgvector init scripts (run on fresh Docker volume).
- [`../lib/`](../lib/) — reusable utilities (LLM wrappers, embeddings, prompts) shared with mini-projects.
- `../evals/` (future, Module 5) — golden set + Promptfoo configs + results.

## Running it

```sh
# 1. Start the database (Postgres 16 + pgvector)
docker compose up -d

# Commands below appear as each module ships.

# 2. Module 4 — ingest the corpus (coming next)
# pnpm dev roman-research/ingest/index.ts

# 3. Module 4 — query with citations
# pnpm dev roman-research/query/index.ts "Who killed Caesar?"
```

## Design decisions (running log)

Decisions that have already shaped the project — captured here so we don't re-derive them every session.

- **Caesar-focused corpus, not broad Roman history.** Overlap on a focused topic gives real multi-source contradiction tests; breadth dilutes that. Mommsen / Tacitus / Cassius Dio added later only if eval coverage demands it. *(Module 4)*
- **Single canonical chunking in the DB; comparison in the eval harness.** Most RAG techniques are query-time. Only chunking variants and contextual retrieval need additive storage. *(Module 4)*
- **Source-tier metadata (primary vs historian) baked into `sources` from day one.** Schema is sticky; adding the column later means a partial re-ingest. *(Module 4)*
- **Chunk-position metadata (`chunk_index`, `chapter`, `char_start`/`char_end`) from day one.** Powers the source-viewer feature without forcing a re-ingest later. *(Module 4)*
- **BGE-M3 as the primary embedder; OpenAI for one comparison run.** BGE-M3 is free, fast (5070 Ti), multilingual. Iteration during Modules 4–6 will re-embed the corpus many times — cost matters. *(Module 3)*
- **Web UI deferred to Module 10.** A simple React SPA is the demo deliverable, not parallel work. Building it during Modules 4–9 distracts from the AI-engineering capability that makes the case study credible. *(Module 4)*
- **No frameworks (LangChain, LangGraph, CrewAI, Vercel AI SDK).** The whole point is to build it yourself once. Frameworks later by choice, not avoidance. *(Phase 1 plan)*
