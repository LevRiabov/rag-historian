# CLAUDE.md

Working notes for Claude Code in this repo. Keep tight — every line is context-window cost on every turn.

## Project

`rag-historian` — the Roman Research Agent. Phase 1 of an eval-driven AI-engineering foundation, building an agentic research assistant over a Project Gutenberg Roman history corpus. **Source of truth for scope and module order:** [phase-1-foundation.md](phase-1-foundation.md). Always defer to it before suggesting work.

## Current state

**Module: 0 — Environment setup** (in progress)

Update the line above whenever a module finishes or the next one starts. The current-module marker is how I keep my bearings between sessions.

## Stack

- **Runtime:** Node 20+, TypeScript strict, ESM (`"type": "module"`)
- **Package manager:** pnpm
- **Runner:** `tsx` (runs `.ts` directly, no build step)
- **Lint + format:** Biome (single tool, replaces ESLint + Prettier)
- **Env:** `dotenv`, single `.env` at repo root
- **LLM SDKs:** `@anthropic-ai/sdk` for Claude; OpenAI SDK reused for OpenAI proper + LM Studio (OpenAI-compatible) when modules need them
- **Local inference:** LM Studio is primary (`http://localhost:1234/v1`); Ollama secondary (e.g., BGE-M3 in Module 3)
- **DB:** Postgres 16 + pgvector — landed in Module 3, not before
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

## When in doubt

Read the current module's section in [phase-1-foundation.md](phase-1-foundation.md) — each has Objectives / Resources / Hands-on / Deliverable / Pitfalls. The Pitfalls block is usually the most load-bearing.
