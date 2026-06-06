# Module 10 — Retrieval Backlog (stretch experiments)

Forward-looking queue of retrieval/embedding experiments to consider during
**Module 10 (Final Integration — Roman Research Agent v2)**. None of this is on
the critical path; it's the menu of "could we push retrieval further" ideas
surfaced while finishing Module 6, parked here with rationale so the thread can
close cleanly. Revisit AFTER Modules 7–9 (agent, MCP, production patterns).

**Source of the analysis:** the Module 6 comparison
([module-6-comparison.md](module-6-comparison.md)) and the discussion after it.
Read those for the *why* behind each priority.

## Guiding principles (unchanged from Module 6)

- **Eval-driven.** Every item below ships only if it beats the Module 6 final
  stack (`contextual-v1` + rerank-on-context: recall@5 51.6 / recall@20 70.5 /
  MRR 0.568) on an isolated A/B. One change at a time.
- **The 50 golden questions are the HELD-OUT eval. Never train on them.**
  Fine-tuning uses *synthetic* pairs generated from the corpus; the golden set
  only ever scores.
- **Local embeddings only** (bge-m3 / llamacpp). Never OpenAI. (Re-embedding the
  whole corpus is the cost of any embedder change — Module 3 pitfall.)
- **Single agent, good tools** — not a swarm (Module 7 pitfall). The agent, not
  retrieval, is the real fix for the two open gaps.

## The two open problems these target

Carried from Module 6, cleanly separated:
- **Synthesis** (recall@5 22.9%) — *coverage-flat under every retrieval lever
  tried.* Not single-shot retrievable; the answer is distributed. Retrieval-side
  hope = **RAPTOR**; real fix = **Module 7 agent** (decompose → retrieve each
  sub-question).
- **Contradiction** (retrieves at 76.9%, but the generator blends the two
  accounts instead of contrasting them) — a *generation* gap. Fix = **Module 7
  agent** (`search_within_source` → read each source separately → compare), or a
  stronger generator. NOT a retrieval lever (a prompt fix was measured + reverted
  in 6.x).

> Reframe to keep in mind: recall@5 ≠ answerability. Completeness 3.40/5 (≈68%)
> already exceeds recall@5 (52%) because the generator answers from partial
> evidence. The product target is correct-or-abstain, not 100% recall.

---

## Priority table

| # | Experiment | Targets | Infra cost | Effort | Priority |
|---|---|---|---|---|---|
| 0 | **Scale corpus to a dozen+ books** | realism / stress-test | re-ingest | low | **do first** (user goal) |
| 1 | **Qwen3-0.6B embedder + reranker swap** | synonym, literal | none (same size) | low | **high** — cheap win |
| 2 | **RAPTOR** (summary-tree retrieval) | **synthesis**, multi-hop | additive rows (pgvector) | medium | **high** |
| 3 | **Generator-k sweep** (top-5 → top-10) | feeds rank 6–20 gold to LLM | none | trivial | **high** — untested knob |
| 4 | **Reranker fine-tuning** | precision@5 all cats | none (no re-embed) | medium | medium — fine-tune on-ramp |
| 5 | **bge-m3 sparse mode** (SPLADE-like) | exact terms (better BM25) | sparse index | medium | medium |
| 6 | **Embedder fine-tuning** | synonym, domain vocab | full re-embed | high | low — capstone stretch |
| 7 | **Multi-vector / ColBERT** | term matching | NEW store (non-pgvector) | high | **low** — rerank overlaps it |
| 8 | Query routing by category | per-query best retriever | classifier | medium | low |
| 9 | MMR / diversity rerank | diverse top-k for synthesis | none | low | low — quick test |

---

## Detail cards (the ones worth thinking through now)

### 0 — Scale the corpus to a dozen+ books
Your stated Module 10 goal: more sources to "fully test with the AI researcher."
Foundational — it stress-tests every technique (a bigger haystack makes the easy
categories harder and is where RAPTOR/multi-vector differences actually show).
Re-ingest with the existing pipeline; rebuild any derived index (RAPTOR tree,
fine-tuned embeddings) afterward. Do this BEFORE the others so their A/Bs run on
the real target corpus.

### 1 — Qwen3-0.6B embedder + reranker swap (cheap win)
Current: bge-m3 (~568M) + bge-reranker-v2-m3 (~568M). **Qwen3-Embedding-0.6B /
Qwen3-Reranker-0.6B are ≈ the same size** (~0.6B) but newer and currently top of
MTEB → likely a free quality bump at ~same speed/VRAM. Just A/B it. (4B = ~7×,
~8 GB, ~3–5× slower — test for ingest-time only; 8B = ~14×, ~16 GB, won't
co-reside with the 9B chat model on 16 GB → swapping.) Requires full re-embed
(embedder change). Bonus: Qwen3 supports Matryoshka dims (truncate for
storage/speed).

### 2 — RAPTOR (the synthesis lever, fits our stack)
Recursive Abstractive Processing for Tree-Organized Retrieval. **Enhanced
contextual retrieval, but it adds new ROWS, not fields:** an LLM summarizes
*clusters* of chunks into higher-level summary nodes, recursively, forming a tree
above the leaves. A synthesis query matches a summary node that already
*is* the cross-chunk answer.

```
L2 (themes)     [ Caesar's rise, dictatorship, assassination ]
                       /                        \
L1 (cluster sums) [ politics before    [ the conspiracy &
                    the Ides ]            the Ides ]
                    /    |   \            /        \
L0 (leaf chunks)  ch   ch   ch         ch         ch   (our existing 500-tok chunks)
```

- **Storage:** summary nodes are extra rows in `chunks` (`text`=summary,
  `embedding_bge`=its vector, a `level` col, `chunking_version='raptor-v1'`).
  **Additive on pgvector — no new index type** (unlike multi-vector). This is the
  reason it's preferred.
- **Retrieve ("collapsed tree"):** flatten ALL levels into one pool, normal
  vector search; the retriever auto-picks the right abstraction (leaf for
  literal, summary for synthesis).
- **Build:** embed leaves → cluster (UMAP + Gaussian Mixture, *soft* — needs a
  small Python script; that ecosystem isn't in our TS stack) → LLM-summarize each
  cluster (reuse [contextualize.ts](../roman-research/ingest/contextualize.ts):
  local qwen, doc-as-prefix) → embed summaries → recurse until few nodes.
- **Risk:** a summary can smooth over a nuance (faithfulness) — mitigated because
  collapsed-tree retrieves leaves alongside summaries, so the generator sees both.
  Rebuild on corpus update (staleness).
- **A/B:** raptor-v1 + rerank-on-context vs contextual-v1 + rerank-on-context,
  watch synthesis recall@5/@20 specifically.

### 3 — Generator-k sweep (trivial, untested)
recall@20 = 70.5% but the generator sees top-**5** (51.6%) → ~19 pts of gold sit
at ranks 6–20, unread. A/B generator-k = 10 (and 8) vs 5: does feeding more gold
raise completeness, or does "lost in the middle" dilution bite (as it did for
parent-child in 6.1)? Costs nothing but a few generation runs. Do early.

### 4 — Reranker fine-tuning (the fine-tune on-ramp)
~1,000–5,000 `(query, positive, hard-negative)` synthetic pairs (LLM writes
questions per chunk = positives; hard negatives = top non-gold from BM25/vector).
**Key advantage: changing the reranker needs NO corpus re-embedding** (it scores
at query time) → tight retrain→deploy→A/B loop. Lower-risk first taste of
fine-tuning than the embedder. Teaches the skill; modest expected gain on top of
a good off-the-shelf reranker.

### 6 — Embedder fine-tuning (highest ceiling, highest effort — last)
Same synthetic data approach, low-thousands of pairs (a dozen books → 5k–20k
pairs, free via local qwen). Highest ceiling (often +10–20 in-domain; learns
archaic vocab like "falling sickness" = epilepsy) but: real training pipeline,
synthetic-data quality is make-or-break, and **every retrain forces a full corpus
re-embed.** Fine-tune occasionally (on domain shift or model upgrade), NOT per
corpus update — a fine-tuned model generalizes to more same-domain books without
retraining. Sequence LAST; high case-study value (demonstrates domain adaptation)
even if marginal recall gain over contextual+rerank+RAPTOR is modest.

### 7 — Multi-vector / ColBERT (deprioritized — here's why)
Accuracy hierarchy: single-vector < ColBERT MaxSim < cross-encoder cross-attention.
Multi-vector is essentially **a learned reranker baked into the index** — it
*overlaps the cross-encoder we already run*, at far higher infra cost (100–500×
storage, a non-pgvector MaxSim store). Its headline benefit ("remove the costly
rerank step") is weak for us: our reranking is a local GPU pass over ~50 docs —
fast and free. Wins only at huge QPS where rerank latency dominates *and* you've
built the multi-vector infra. If term-matching is the goal, **#5 sparse mode is
the cheaper route**. Skip unless time + curiosity.

---

## Suggested sequencing

1. **#0 scale corpus** → then re-baseline the Module 6 final stack on it.
2. Cheap/high-confidence wins: **#1 Qwen3-0.6B swap**, **#3 generator-k sweep**.
3. **#2 RAPTOR** — the synthesis bet.
4. **Module 7 agent integration** — the actual fix for synthesis + contradiction
   (decompose / read-each-source). Bigger lever than anything retrieval-side for
   the two open gaps; it just lands chronologically at Module 7.
5. Stretch, if time: **#4 reranker fine-tune** → **#6 embedder fine-tune**.
6. Probably-skip: **#7 multi-vector**, **#8 routing**, **#9 MMR** (quick test only).

## The throughline

Module 6 proved the remaining wall is **reasoning, not retrieval**. RAPTOR is the
one retrieval-side technique that genuinely attacks synthesis; everything else
here (#1, #4, #5, #6) lifts the already-decent easy/medium categories a few
points. The **Module 7 agent** is the highest-leverage single move for the two
open gaps regardless of corpus size — retrieval improvements raise its floor;
they don't replace it.
