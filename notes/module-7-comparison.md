# Module 7 — The Agent (capstone summary)

Short version of what we built in Module 7 and how much it moved the numbers. Full
running log + every finding (F1–F20): [module-7-agent.md](module-7-agent.md).
Baseline it's measured against: the Module 6 single-shot stack
([module-6-comparison.md](module-6-comparison.md)).

## What we built

The **Roman Research Agent v1** — a single-agent tool-use loop (no LangGraph):
- **5 tools** (`agent/tools.ts`): `search_corpus`, `search_within_source` (the
  contradiction lever), `read_chunk`, `list_sources_consulted`, `finalize`. All wrap
  the Module 6 final retrieval stack.
- **Provider-agnostic loop** (`agent/index.ts` → lib `runTools`): max-iter, cost cap,
  per-step tracing, `finalize` as a terminal tool, graceful **forced-finalize** on
  max-iter (a budget-exhausted run still answers instead of returning empty).
- **`evals/run.ts --agent`**: same 3 judges + agent metrics (tool calls, gold
  coverage, finalize rate, cost) per category; one `--llm` flag swaps Claude ↔ local.
- **Langfuse tracing** (`lib/langfuse.ts` + self-hosted `infra/langfuse/`): a trace
  tree per question with eval scores attached — the click-through drill-down.

## How it helped — agent vs single-shot RAG

| metric | M6 single-shot (Haiku) | **agent (Haiku)** | M6 single-shot (qwen) | **agent (qwen)** |
|---|---|---|---|---|
| Completeness | 3.40 | **4.04** | 3.02 | **3.38** |
| Faithfulness | 4.64 | **4.72** | 3.98 | **4.32** |
| Refusal acc. | 92% | **98%** | — | 90% |

**The agent beats single-shot RAG for BOTH models** — completeness +0.64 (Claude) /
+0.36 (qwen), faithfulness up for both, refusal +6pts on Claude. The multi-step
"search → read each source → synthesize" loop delivers what single retrieval can't.

## What we learned (the findings that travel)

1. **The strong model leverages agency more — the gap WIDENS.** Single-shot
   Claude–qwen completeness gap was 0.38; with the agent it's 0.66. qwen's tool *use*
   and coverage are competitive, but it converts them to weaker answers, and it
   **over-searches** (10.5 tools/Q vs Claude's 6.9, which self-regulates by
   difficulty). Tools don't level the field; better reasoning compounds.
2. **The eval is part of the system under test.** Three measurement bugs each moved
   the numbers more than most "real" changes:
   - **Citation scheme** (chunk_id vs [N]) made the faithfulness judge read correct
     citations as fabricated → fixing it recovered **~+0.38 faithfulness**.
   - **Forced-finalize** turned max-iter non-answers into answers (completeness
     +0.89, refusal +11 on the literal slice).
   - **Judge calibration (the big one):** a Haiku judge *systematically under-scored
     thorough multi-source answers* and couldn't be prompted out of it. Swapping to a
     **Sonnet judge** corrected contradiction **3.00 → 3.89** and revealed it had been
     **too lenient** on synthesis (4.00 → 3.50).
3. **Prompt-tuning has a noise floor; model swaps clear it.** Two data-motivated
   contradiction prompt fixes both landed *within* per-question variance (±1–2 at
   n=9). A Sonnet **model swap** moved the same hard questions **+1.5 to +3**. Spend
   the next dollar on the model/retrieval, not more prompt text.

## Corrected final picture (calibrated judge)

- **Faithfulness 4.72, refusal 98%** (Haiku agent) — production-grade *safety* for a
  cited, human-in-the-loop assistant.
- **Completeness 4.24** (Haiku agent, Sonnet-judged). Sonnet *agent* extrapolates
  higher (hit 5/5 on the hardest spot-check questions).
- **The real weak spot is SYNTHESIS (3.50), not contradiction (3.89)** — and synthesis
  is *retrieval/coverage-bound* (≈50% gold coverage; the answer is distributed),
  exactly as Module 6 predicted. Contradiction was largely a judge artifact.

## Open → Module 10
Stronger model for reasoning categories (proven lever); **RAPTOR / decomposition /
generator-k** for the synthesis coverage floor; default the completeness judge to
Sonnet/Opus (with calibration); a clean-context "finalize from read chunks" variant
worth one A/B. **Pitfall honored:** single agent + good tools — no multi-agent.
