# Promptfoo — side-by-side LLM comparison

A second eval surface on top of the same golden set, focused on **side-by-side model comparison** with a browseable web UI. Complements `evals/run.ts` rather than replacing it.

## When to use which harness

| Need | Use |
|---|---|
| Compare 2+ models on the same questions, see outputs side-by-side | Promptfoo |
| Browse results in a shareable web UI | Promptfoo |
| Measure retrieval-stage metrics (recall@k, MRR) | `evals/run.ts` |
| Measure generation metrics across a single pipeline (faithfulness + completeness + refusal) | `evals/run.ts` |
| Test the impact of a chunking or retrieval change | `evals/run.ts` (chunks are baked into Promptfoo's test data) |

Both read `evals/golden-set.json`. There's no duplication of the test set itself.

## Files

- [`promptfooconfig.yaml`](../../promptfooconfig.yaml) (repo root) — providers, prompt, assertions
- [`prompt.json`](prompt.json) — system + user message templates (mirrors `roman-research/query/answer.ts`)
- [`build-tests.ts`](build-tests.ts) — regenerates `tests.json` from `golden-set.json` + current retrieval
- `tests.json` — generated test cases with pre-retrieved chunks baked in (committed; regenerate with `build-tests.ts`)

## Setup (one-time)

`promptfoo` is already a dev dependency.

## Run

```sh
# 1. Generate test cases from golden-set + current retrieval. Re-run whenever:
#    - golden-set.json changes
#    - the corpus is re-ingested (chunk IDs may shift)
#    - the embedder changes
pnpm dev evals/promptfoo/build-tests.ts

# 2. Run all providers × all tests, with Haiku judging faithfulness.
pnpm exec promptfoo eval

# 3. Open the web UI to browse the comparison matrix.
pnpm exec promptfoo view
# Opens http://localhost:15500 — comparison table with per-cell answers.
```

## Cost & runtime

- Generation: free (local LM Studio for the two default providers)
- Judging: ~$0.20 per full run (50 tests × 2 providers × ~3000 input tokens × Haiku rates)
- Wall clock: ~15-25 minutes depending on local model speed (qwen 9B is slower than gpt-oss MoE)

## Adding providers

Open [`promptfooconfig.yaml`](../../promptfooconfig.yaml) and copy a provider block. For LM Studio models, the `id` after `openai:chat:` must be the exact model name LM Studio exposes — check at `http://localhost:1234/v1/models` if unsure.

Frontier model providers (Claude, OpenAI proper) are commented out in the config — uncomment when you want a paid frontier comparison.

## What's intentionally NOT here

- Retrieval metrics (recall@k, MRR) — those live in `evals/run.ts` because they need our DB and embedder, and Promptfoo's job here is generation comparison.
- Completeness and refusal-correctness judges — `evals/run.ts` has them. We could add as additional `llm-rubric` assertions in `promptfooconfig.yaml` later, but the MVP keeps to one assertion to keep the matrix readable.

## Reading the comparison matrix

The web UI shows:
- Rows = test cases (50 questions from your golden set)
- Columns = providers (qwen 9B, gpt-oss 20B, etc.)
- Cells = pass/fail + full answer text + judge reasoning

Look for:
- **Cells one model passes and another fails** — the most diagnostic comparison. Read both answers, decide if the judge was right.
- **Per-category patterns** — filter by metadata.category to see if one model is systematically better on multi-hop vs literal, etc.
- **Latency column** — local-model speed differences are stark.
