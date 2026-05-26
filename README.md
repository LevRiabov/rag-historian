# rag-historian

Roman Research Agent — an agentic research assistant over a Roman history corpus from Project Gutenberg (Gibbon, Plutarch, Suetonius, Tacitus, Caesar).

Built as the Phase 1 case study of an eval-driven AI engineering foundation. The full plan lives in [phase-1-foundation.md](phase-1-foundation.md).

## Status

**Module 0: Environment setup.** Scaffold only — no LLM code yet.

## Quick start

Prerequisites: Node 20+, pnpm. Postgres 16+ with `pgvector`, LM Studio / Ollama, and various API keys come in later modules.

```bash
pnpm install
cp .env.example .env
# Fill in only the keys you currently need — others can wait.
```

## Layout

- `lib/` — reusable utilities (llm wrapper, prompts, embeddings)
- `roman-research/` — main project: ingest, query, agent, MCP server
- `mini-projects/` — per-module experiments
- `evals/` — golden set, metrics, Promptfoo configs
- `notes/` — learnings captured along the way
- `corpus/` — Project Gutenberg texts (gitignored)

## License

TBD — will land before the public case study in Module 10.
