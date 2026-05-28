# mini-projects

Per-module experiments that build intuition for the wrapper / agent / eval machinery as it grows. Each project lives in its own folder with an `index.ts` entry point and a `README.md` describing what it teaches.

## Run any project

```
pnpm dev mini-projects/<folder>/index.ts
```

Many projects accept `USE_CLAUDE=1` to switch from the default Ollama backend to Claude (requires `ANTHROPIC_API_KEY` in `.env`). Shell-specific syntax:

```powershell
# PowerShell (Windows)
$env:USE_CLAUDE='1'; pnpm dev mini-projects/<folder>/index.ts
```

```bash
# bash / zsh
USE_CLAUDE=1 pnpm dev mini-projects/<folder>/index.ts
```

## Catalogue

| # | Project | Module | What it teaches |
|---|---|---|---|
| 01 | [hello-llm](01-hello-llm/) | 1 — LLM API Foundations | Tool-loop end-to-end against Claude AND Ollama side by side |
| 02 | [extraction](02-extraction/) | 2 — Prompting & Structured Outputs | `structured()` in practice; zero-shot vs few-shot on the same inputs |
| 03 | [classification](03-classification/) | 2 — Prompting & Structured Outputs | Answer-first vs CoT vs few-shot for multi-class labeling |
| 04 | [xml-context](04-xml-context/) | 2 — Prompting & Structured Outputs | XML-delimited context vs naive concatenation, injection resistance |
| 05 | [version-compare](05-version-compare/) | 2 — Prompting & Structured Outputs | One change per version; the discipline made concrete via `abTest` |

New rows land here as each module ships.
