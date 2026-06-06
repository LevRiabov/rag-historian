# CLAUDE.md

Working notes for Claude Code in this repo. Keep tight — every line is context-window cost on every turn.

## Project

`rag-historian` — the Roman Research Agent. Phase 1 of an eval-driven AI-engineering foundation, building an agentic research assistant over a **Caesar-focused** Project Gutenberg corpus (4 primary sources) with **multi-source contradiction handling** as a stated goal. **Sources of truth:** module order in [phase-1-foundation.md](phase-1-foundation.md); project scope + architecture in [roman-research/README.md](roman-research/README.md). Defer to both before suggesting work.

## Current state

**Module: 7 — Agent Fundamentals COMPLETE. `agent/{tools,index,cli}.ts` (`runAgent`, 5 tools incl. `search_within_source`), lib `runTools` gained additive `terminalToolName` (finalize) + forced-finalize on max-iter, `evals/run.ts --agent` mode (same judges + agent metrics; goldCoverage replaces recall@k; chunk_id-labeled faithfulness judge), `lib/langfuse.ts` (`langfuseTracer` over the Tracer seam — trace tree + scores, no-op without keys; add LANGFUSE keys to .env to activate). Clean 50q×2-model A/B: agent beats single-shot for BOTH models (Claude comp 3.40→4.04, faith 4.64→4.72, refusal 92→98%; qwen comp 3.02→3.38). Open→Module 10: contradiction completeness (generation/reading-bound + noisy; prompt-tuning proven below noise floor at n=9 — needs stronger model / variance reduction), qwen over-search + weaker refusal (90%). NEXT MODULE: 8 — MCP Server Design. Full findings + tables: [notes/module-7-agent.md](notes/module-7-agent.md).**

Update the line above whenever a module finishes or the next one starts. The current-module marker is how I keep my bearings between sessions.

Baseline + eval methodology: [notes/module-5-evals.md](notes/module-5-evals.md). **Module 6 capstone comparison table (what worked/didn't + why): [notes/module-6-comparison.md](notes/module-6-comparison.md). Running log + every A/B delta + footguns: [notes/module-6-advanced-rag.md](notes/module-6-advanced-rag.md).** Short version: final retrieval stack is `contextual-v1` + rerank-on-context (recall@5 35.1→51.6, recall@20 55.4→70.5, MRR 0.475→0.568); **query expansion `--expand --expand-n=5` is an opt-in +2.1 r@5 on top (synonym/multi-hop only, ~1s/query cost)**. Gold is chunking-invariant SPANS; `evals/run.ts` flags: `--chunking-version`, `--hybrid`/`--lexical-weight`, `--rerank`/`--rerank-pool`, `--hyde`/`--hyde-concat`, `--expand`/`--expand-n`, `--category`, `--generation`/`--llm`/`--generator-k`. **6.5 result: HyDE REJECTED (net −9.1 r@5 — wrong tool for a small single-topic corpus); expansion modest-accept at n=5.** Open/settled: **synthesis is coverage-flat under every retrieval lever → it's reranker/generation-bound, NOT a retrieval floor** (joins contradiction as a Module 7 agent / stronger-generator problem).

## Stack

- **Runtime:** Node 20+, TypeScript strict, ESM (`"type": "module"`)
- **Package manager:** pnpm
- **Runner:** `tsx` (runs `.ts` directly, no build step)
- **Lint + format:** Biome (single tool, replaces ESLint + Prettier)
- **Env:** `dotenv`, single `.env` at repo root
- **LLM SDKs:** `@anthropic-ai/sdk` for Claude; OpenAI SDK reused for OpenAI proper + LM Studio (OpenAI-compatible) when modules need them
- **Local inference:** roman-research runs on **llama.cpp via llama-swap** (native, `http://127.0.0.1:8080`, server lives at `C:\llm`) for all three: chat (`createLlamacpp`, Qwen3.5-9B profiles — thinking on/off is per-profile via `--reasoning`), embeddings (bge-m3), and reranking (bge-reranker-v2-m3). `LLAMA_SWAP_BASE_URL` points at it. Early mini-projects (01–05) still use **LM Studio** (primary, `:1234`) / Ollama via `createLocalLLM({ lmstudio, ollama })`, flipped by `LOCAL_LLM_PROVIDER`. GBNF is a hard token-level constraint on both LM Studio and llama.cpp; Ollama's per-model renderers engage it inconsistently (Gemma 4 leaks out-of-enum values).
- **DB:** ParadeDB image (Postgres 18 + pgvector + `pg_search` BM25), `paradedb/paradedb` in docker-compose — migrated from `pgvector/pg16` in Module 6.2 for real BM25 (core FTS has no IDF). Volume mounts at `/var/lib/postgresql` (PG18). Reranking served by a separate **Infinity** container (`:7997`, bge-reranker-v2-m3; client in `lib/rerank.ts`). Embeddings are **local bge-m3 only** — never OpenAI.
- **Tracing:** Langfuse — wired in Module 1, not Module 0
- **Evals:** Promptfoo — Module 5

## Commands

- `pnpm install` — install deps
- `pnpm typecheck` — TS check, no emit
- `pnpm lint` / `pnpm lint:fix` — Biome
- `pnpm dev <file.ts>` — run a TS file directly via tsx

## Layout

- `lib/` — reusable utilities (llm wrapper, prompts, embeddings) — built across Modules 1–3
- `roman-research/` — the main project (ingest, query, agent, mcp-server)
- `mini-projects/` — per-module experiments (e.g., `01-hello-llm.ts`)
- `evals/` — golden set, metrics, Promptfoo configs, results — Module 5+
- `notes/` — accumulated learnings (prompting patterns, eval philosophy, etc.)
- `corpus/` — Project Gutenberg texts, **gitignored** (large, downloadable)

## Project-specific don'ts

- **Don't reach for LangChain / LangGraph / CrewAI / Vercel AI SDK.** Building wrappers by hand is the explicit point (Module 1 pitfall). Use libraries later by choice, never to skip fundamentals.
- **Don't mix embedding models** across stored chunks and queries — costly rebuild (Module 3 pitfall).
- **Don't change two things at once** during eval comparisons. Isolated A/B against the previous best is the only attribution method (Module 6 pitfall).
- **Don't skip evals** from Module 5 onward. They're the foundation's differentiator.
- **Don't multi-agent the Roman agent** — single agent + good tool design wins (Module 7 pitfall).
- **Don't over-engineer scaffolding.** Naive first, then improve based on eval failures.
- **Don't add error handling, retries, or guardrails until they're needed.** Module 9 is for production patterns. Adding them earlier obscures actual failure modes.
- **Don't build the web UI before Module 10.** A simple React SPA is the *demo deliverable*, not parallel work — building it during Modules 4–9 distracts from the AI-engineering capability that makes the case study credible.
- **Don't store multiple RAG variants in parallel DB columns/tables.** Most Module 6 techniques (hybrid, reranking, query-rewriting) are query-time. Comparison happens in the eval harness against ONE canonical chunking, not via parallel storage. Only chunking variants (6.1) and contextual retrieval (6.4) get additive storage.

## When in doubt

Read the current module's section in [phase-1-foundation.md](phase-1-foundation.md) — each has Objectives / Resources / Hands-on / Deliverable / Pitfalls. The Pitfalls block is usually the most load-bearing.
