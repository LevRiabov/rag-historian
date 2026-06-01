# Eval methodology + Module 5 baseline

> **Captured:** 2026-06-01 (end of Module 5, before Module 6 retrieval improvements).
> **Purpose:** lock in the eval methodology, baseline numbers, and known footguns so every Module 6 technique gets A/B'd against this snapshot. Complements [naive-rag-baseline.md](naive-rag-baseline.md), which is the older narrative-style snapshot from Module 4 end.

## The big idea — decomposition

A single "is the answer good?" score is useless for diagnosis. We split the pipeline at three measurement points:

```
question ─► [embed query] ─► [pgvector top-K] ──┐
                                                ├─► RETRIEVAL METRICS (recall@k, MRR)
retrieved chunks (top-20) ─────────────────────┘
                              │
                              ▼ (top-5 to generator)
                       [LLM generate]
                              │
                              ▼
                       answer with citations
                              │
                              ├─► FAITHFULNESS    (does every claim trace to chunks?)
                              ├─► COMPLETENESS   (vs ideal answer; covers all facts?)
                              └─► REFUSAL CORR.  (refuse vs answer matched expectation?)
```

This gives five independent dials. When `faithfulness=4.3` but `completeness=2.6`, we know **the LLM isn't hallucinating; retrieval isn't surfacing enough material**. That's an attribution claim no single score can make.

## The golden set

`evals/golden-set.json` — 50 hand-labeled question/ideal-answer/gold-chunk-IDs triples across 6 categories:

| Category | n | What it tests |
|---|---:|---|
| literal | 9 | Basic vector retrieval — should be the easy case |
| synonym | 8 | Vocabulary mismatch (corpus says "falling sickness", question says "epilepsy") |
| multi-hop | 9 | Combining facts from different chunks |
| synthesis | 8 | LLM reasoning across multiple sources |
| **contradiction** | 9 | Cross-source retrieval where sources disagree (project headline feature) |
| out-of-scope | 7 | Refusal — empty `goldChunkIds`, should refuse |

Each entry has `goldChunkIds: number[]` referencing rows in `chunks.id` — the load-bearing artifact for recall@k.

## Two harnesses, complementary

| Need | Tool |
|---|---|
| Retrieval metrics (recall@k, MRR), end-to-end pipeline tests, technique iteration | **Custom: [evals/run.ts](../evals/run.ts)** |
| Side-by-side model comparison, web UI, sharing with non-engineers | **Promptfoo: [promptfooconfig.yaml](../promptfooconfig.yaml)** |

Both consume the same `golden-set.json` — single source of truth. Both use Claude Haiku as the LLM-as-judge for the three generation rubrics. Promptfoo's chunks are **frozen** into `evals/promptfoo/tests.json` by `build-tests.ts` so model comparisons aren't contaminated by retrieval variance.

## Configuration locked in for the Module 5 baseline

| | Value |
|---|---|
| Corpus | 4 Caesar primary sources, ~1.25 MB cleaned text, 950 chunks |
| Chunking | `naive-v1` — token-window 500/50, never crosses section boundaries |
| Embedder for retrieval | BGE-M3 (1024d, LM Studio) |
| Retrieval K (for recall metrics) | top-20 |
| Generator K (chunks passed to LLM) | top-5 (matches production query pipeline) |
| Generator models tested | gpt-oss-20b (MoE, 3.6B active) and qwen3.5-9b (dense) |
| Judge | Claude Haiku 4.5 |
| Retrieval-only run cost | $0 (all local) |
| Full run cost (generation + 3 judges per Q) | ~$0.45-0.50 Haiku |

## Baseline numbers — what we got

### Retrieval (identical across both generator runs — retrieval is independent)

| Category | n | recall@5 | recall@10 | recall@20 | MRR |
|---|---:|---:|---:|---:|---:|
| literal | 9 | 28.7% | – | 61.1% | 0.532 |
| synonym | 8 | 31.3% | – | 50.0% | 0.219 |
| multi-hop | 9 | 38.9% | – | 53.7% | 0.458 |
| synthesis | 8 | **18.7%** | – | **34.4%** | 0.280 |
| **contradiction** | 9 | **55.6%** | – | **75.0%** | **0.833** |
| out-of-scope | 7 | – | – | – | – |
| **All in-scope** | 43 | **35.1%** | 46.3% | **55.4%** | **0.475** |

**Reads as:**
- Top-1 is rarely right (recall@1=15.9%). Reranking should fix.
- Coverage at K=20 is workable (55.4%) — chunks ARE being found, just buried.
- Contradiction category has the **best** retrieval (the project's headline feature performs well at the floor).
- Synthesis is the weakest — meta-questions about the corpus have no topical anchor for vectors.

### Generation comparison: gpt-oss-20b (MoE) vs qwen3.5-9b (dense)

| Metric | gpt-oss-20b | qwen3.5-9b | Δ |
|---|---:|---:|---|
| Faithfulness | 4.16 / 5 | **4.32 / 5** | +0.16 |
| **Completeness** | 2.64 / 5 | **3.00 / 5** | **+0.36 (+14%)** |
| Refusal accuracy | 88.0% | **92.0%** | +4 pts |
| Judge cost / run | $0.44 | $0.47 | similar |

**Reads as:** the **dense 9B model beats the sparse 20B model on extraction**. The completeness lift (+14%) is the headline finding — for synthesis and multi-hop questions, full-attention over 5 chunks beats MoE skimming. Faithfulness improved marginally; refusal accuracy improved by 4pts.

### Per-category breakdown, qwen3.5-9b (the better generator)

| Category | recall@5 | MRR | Faithfulness | Completeness | Refusal |
|---|---:|---:|---:|---:|---:|
| literal | 28.7% | 0.53 | 4.22 | 2.56 | **100%** |
| synonym | 31.3% | 0.22 | 4.13 | 2.88 | 88% |
| **multi-hop** | 38.9% | 0.46 | **3.78** | **3.11** | **100%** |
| synthesis | 18.7% | 0.28 | **4.50** | 2.63 | 75% |
| **contradiction** | **55.6%** | **0.83** | **4.22** | 2.22 | 89% |
| out-of-scope | – | – | **5.00** | **5.00** | **100%** |

**Reads as:**
- Multi-hop got the biggest gen-side win switching to qwen (comp 2.33 → 3.11, +33%) — dense attention pays off most when combining facts.
- Synthesis has the lowest retrieval (recall@5=18.7%) BUT the highest faithfulness (4.50/5) — qwen refuses to make stuff up when it doesn't have the chunks.
- Contradiction's high retrieval (55.6% recall@5) with mediocre completeness (2.22) suggests the model finds the right chunks but doesn't always surface BOTH sides of the contradiction in the answer. Module 6.5 query rewriting target.
- Out-of-scope: perfect refusal (100%). The refusal rule in the system prompt is rock-solid.

### Promptfoo cross-check, qwen3.5-9b, 50 questions × 3 assertions

```
30 passed (60.00%)
16 failed (32.00%)
 4 errors  (8.00%)
```

The 4 errors were all `LM Studio HTTP 400: Context size has been exceeded` — qwen3's thinking mode burned the context on long retrievals.

**Independent validation:** Promptfoo's Haiku judge is meaningfully stricter than our harness's. Our harness reports 4.32 faithfulness average (interpretable as ~85% pass rate at threshold 3). Promptfoo's faithfulness-only pass rate was lower. Same model, same Haiku, different rubric wrapping → different strictness. **Both are valid signals**; the absolute scores depend on the harness, but rank-ordering across models or techniques is stable.

## Known footguns (each cost ≥30 min to debug)

| Footgun | Symptom | Fix |
|---|---|---|
| **OpenAI SDK v6 base64 default** | BGE-M3 embeddings came back as 256 floats of zeros instead of 1024 real values | Pin `encoding_format: 'float'` in [lib/embeddings.ts](../lib/embeddings.ts) |
| **qwen3 thinking mode** | Empty `content` field, 4096 completion tokens, ~37s/question | Pass `reasoning: false` (→ `chat_template_kwargs.enable_thinking=false`). Partial — qwen still thinks under the hood (visible as ~5500 completion tokens for a 150-token answer), but content channel populates. LM Studio UI override doesn't fully disable either. |
| **Haiku stringified array output** | Schema validation crash: `expected array, received string` for `unsupportedClaims` | Add `z.preprocess` coercion to schemas in [evals/metrics/generation.ts](../evals/metrics/generation.ts) |
| **Anthropic SDK timeout mid-eval** | One transient timeout crashed the entire 50-question run | Add explicit retry-with-backoff inside judges + try/catch at run loop (a transient blip costs one question, not all 50) |
| **Eval generator using top-20 chunks** | LM Studio HTTP 400: context exceeded | Separate `generatorTopK=5` from retrieval `RETRIEVE_TOP_K=20` |
| **Promptfoo can't load `.js` ESM** | Silent fallback to wrong provider; same empty-output failure mode | Rename custom provider to `.cjs` + `module.exports = ...` |
| **Promptfoo `llm-rubric` doesn't pass user prompt to judge** | Faithfulness judge said "no source passages provided" | Inline `{{sources_block}}` directly into the rubric value |
| **Promptfoo CLI gated by pnpm `allowBuilds`** | `pnpm typecheck` fails before tsc even runs | Create `pnpm-workspace.yaml` with explicit `allowBuilds` decisions |

These are the kind of cross-stack issues only an actual run surfaces. Worth keeping the list — most of them will repeat in any future LLM-eval project.

## What we deliberately DIDN'T do (defer to later)

- **Judge calibration spot-check** (Hamel's "hand-grade 10, compute agreement") — would lift confidence in absolute numbers. Skip-cost: we trust rank-ordering, not absolute scores. ROI on this is highest if we ever need to defend the numbers to a stakeholder.
- **CI integration** (Promptfoo to GitHub Actions, fail builds on regression) — defer to Module 9 production-patterns when there's real CI need.
- **Custom JS retrieval assertions in Promptfoo** — recall@k as a Promptfoo assertion. Possible but redundant with our harness.
- **Adversarial / red-team tests** (Promptfoo has `redteam` mode for prompt injection / jailbreak) — relevant for a deployed product (Module 10) not for evaluating retrieval quality.
- **Cost/latency assertions** — `--type cost --threshold 0.01` to fail tests exceeding a cost budget. Nice for production; not blocking Module 6.

## Module 6 — what the eval methodology now lets us measure

Each Module 6 technique gets re-run with the same harness. The expected delta column is replaced with measured numbers after each technique.

| Technique | Predicted biggest win on | Predicted recall@5 lift | Predicted completeness lift |
|---|---|---:|---:|
| 6.1 Structure-aware chunking | Section-anchored questions | +5-10% | +0-5% |
| 6.2 Hybrid search (BM25 + vector) | Named-entity / rare-word queries | +10-15% | +5-10% |
| 6.3 Reranking | Almost everything — fixes "in top-20 but buried" pattern | +15-25% | +10-15% |
| 6.4 Contextual retrieval | Questions where chunks don't surface-match the question | +20-35% (per Anthropic) | +10-20% |
| 6.5 Query rewriting / HyDE | Multi-hop, synonym-mismatch, abstract synthesis | +20%+ on those types | +10-15% |
| 6.6 Final stack | All combined | end-state recall@5 target: 65-75% | end-state completeness target: 4.0+/5 |

After Module 5 we have **measured** numbers replacing the "expected" column wherever a technique gets implemented.

## Reproducibility

```sh
# 1. Ingest the corpus (idempotent; re-run safely)
docker compose up -d
pnpm dev roman-research/ingest/index.ts

# 2. Retrieval-only baseline (fast, free)
pnpm dev evals/run.ts --out=evals/results/$(date +%Y-%m-%d)-naive-bge.json

# 3. Full baseline with generation + judging (qwen3.5-9b, ~15 min, ~$0.50)
pnpm dev evals/run.ts --generation --show-answers --out=evals/results/$(date +%Y-%m-%d)-naive-bge-gen.json

# 4. Promptfoo cross-check
pnpm dev evals/promptfoo/build-tests.ts
pnpm exec promptfoo eval --no-cache
pnpm exec promptfoo view
```

Same golden set, same chunking_version, same embedder → numbers should be within ±5% of the table above. Anything outside that means either retrieval or judging changed and we should investigate.

## Result artifacts in repo

| File | What |
|---|---|
| `evals/results/2026-05-29-naive-bge-pilot.json` | First 5-question pilot, retrieval-only |
| `evals/results/2026-05-29-naive-bge-v2.json` | 50q retrieval-only baseline |
| `evals/results/2026-05-29-naive-bge-gen.json` | 50q full (gpt-oss-20b) — first generator baseline |
| `evals/results/2026-05-31-qwen9b-bge-gen.json` | 50q full (qwen3.5-9b) — the current baseline |
| `evals/results/2026-05-31-qwen9b-bge-gen.log` | Streaming run log with answers (for spot-check / calibration) |
| `evals/results/promptfoo-latest.json` | Promptfoo cross-check on qwen3.5-9b |

Keep these in git; they're the "before" picture every Module 6 technique compares against.
