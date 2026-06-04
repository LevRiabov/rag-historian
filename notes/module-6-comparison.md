# Module 6 — The Comparison Table (case-study deliverable)

The Module 6 capstone: every advanced-RAG technique tested, the measured delta,
and the verdict with its reason. Full per-technique running log (footguns,
infra, per-category breakdowns):
[module-6-advanced-rag.md](module-6-advanced-rag.md). Baseline + methodology:
[module-5-evals.md](module-5-evals.md).

**Discipline:** every row is A/B'd against the *previous best* on the Module 5
golden set (50 questions, in-scope n=43, chunking-invariant span gold), changing
ONE thing at a time. Retrieval metrics are span-recall + MRR; generation metrics
are LLM-as-judge (Haiku) faithfulness/completeness on the top-5 chunks.

---

## The headline

**Recall@5 35.1% → 51.6%** (+16.5 pts) and **recall@20 55.4% → 70.5%** (+15.1)
on the retrieval side; **Haiku answer completeness 3.12 → 3.40** on the
generation side. Achieved with exactly **two** of the six techniques tried —
**contextual retrieval** (the lever) and **rerank-on-context** (the multiplier).
The other four were ruled out *with evidence*, which is the point of an
eval-driven foundation: knowing what *not* to ship is as valuable as the wins.

---

## Master table — retrieval (cumulative stack, each row = previous best + one change)

| Stack | recall@5 | recall@20 | MRR | Verdict |
|---|---:|---:|---:|---|
| Naive (Module 4/5 baseline) | 35.1% | 55.4% | 0.475 | — |
| + structure-aware chunking sweep | 35.1% | 55.4% | 0.475 | ❌ no change (naive already optimal) |
| + hybrid BM25 (lexical w=0.5), no rerank | 35.3% | 59.3% | 0.395 | ⚠️ coverage↑ ordering↓ — subsumed below |
| + reranking (vector → cross-encoder) | 45.0% | 60.7% | 0.515 | ✅ **adopt** (biggest lever before contextual) |
| + hybrid **and** rerank | 45.0% | 64.0% | 0.513 | ❌ ties vector+rerank @k=5 → **drop hybrid** |
| + contextual retrieval (vector only) | 51.0% | 68.4% | 0.552 | ✅ **adopt** (the single biggest lever) |
| + rerank on **bare** text | 47.9% | 67.2% | 0.524 | ❌ stages disagree → *undoes* contextual |
| **+ rerank-on-context  ← FINAL STACK** | **51.6%** | **70.5%** | **0.568** | ✅ **best on every metric** |
| + query expansion (n=5, opt-in) | 53.7% | 74.0% | 0.570 | 🔶 modest accept — opt-in flag, not default |
| ~~+ HyDE~~ (replaces query) | 41.9% | 66.9% | 0.499 | ❌ **reject** (−9.7 vs contextual) |

> Reading note: rows compound top-to-bottom *along the kept path* (naive →
> rerank → contextual → rerank-on-context). The ❌ rows branch off that path to
> show what was tried and rejected at that point.

## Master table — generation (Haiku generator, top-5, Haiku judge, n=50)

Measured at the two anchor points (full generation runs are ~minutes + ~$0.50):

| Stack | Faithfulness | Completeness | Refusal acc. | Gen cost / 50q |
|---|---:|---:|---:|---:|
| Naive baseline | 4.86 | 3.12 | 84% | $0.21 |
| **Final stack** (contextual + rerank-on-context) | 4.64 | **3.40** | **92%** | $0.21 |

Local qwen3.5-9b on the final stack: faithfulness 3.98, completeness 3.02 ($0).
The retrieval win converted to an **answer** win for the strong model
(completeness +0.28) but barely for the 9B local model (+0.02) — *a small model
is its own bottleneck; better context can't fix it.*

> The faithfulness dip (4.86 → 4.64) is **not** a regression — it's the
> completeness/faithfulness tension: better retrieval → more complete answers →
> more claims → marginally more surface for the judge to flag. Completeness +0.28
> and refusal +8 pts are the real movement; faithfulness stayed high (4.6/5).

---

## What worked, what didn't, and why

### ✅ Contextual retrieval — the lever (recall@5 +16)
An LLM reads the whole parent document + each chunk and writes a 1–2 sentence
note naming the people/places/events the chunk only *implies*; we embed
`citation + note + chunk`. **Why it won:** our chunks lost their referents —
"he crossed at dawn" never says *Rubicon*, so it embedded far from a Rubicon
query. The note injects the missing entities, so fact-bearing chunks finally
embed near their questions (literal recall@5 28.7→76.9, contradiction
55.6→77.8). It also raised the coverage *ceiling* that reranking structurally
cannot (synthesis recall@20 34→42). **Cost lives at ingest (one-time), not per
query.**

### ✅ Rerank-on-context — the multiplier (and the deepest lesson)
A cross-encoder re-scores a deep pool (top-50) reading query+chunk *jointly*.
On its own (over naive) it was the biggest pre-contextual lever: recall@5
35→45. **The lesson:** reranking on the *bare* chunk text *hurt* contextual
(51.0→47.9) — the cross-encoder saw un-contextualized text, couldn't tell what
an ambiguous chunk was about, and demoted exactly the chunks the contextual
embedding had surfaced. Feeding the reranker the *same* contextualized text
flipped it from −3.1 to best-on-every-metric. **A two-stage pipeline must share
representation, or the stages fight.**

### ❌ Structure-aware chunking sweep — naive was already optimal
Swept 300 / 500 / 6000-token and parent-child. Finer (300) lost (thin vectors
lose bridging context); coarser (chapter) inflated recall via a size artifact
but didn't improve *answers* (completeness flat, faithfulness down, over-answers
out-of-scope); parent-child needs *tiny* children to pay off — at 500-tok
children it's pure dilution (literal completeness 3.22→2.67). **Verdict:** keep
naive-v1. Value delivered: the span-gold harness + three plausible-but-wrong
ideas killed with evidence.

### ❌ Hybrid search (BM25 + vector) — real, but subsumed by the reranker
BM25 with IDF genuinely works — literal recall@5 nearly doubled (exact rare
terms like *Pharnaces*/*Zela* that dense vectors smooth over). But once a
cross-encoder is in place, **vector+rerank ≡ hybrid+rerank at k=5** (identical
recall@5/MRR): the reranker re-scores by true relevance regardless of which arm
surfaced a candidate, and the vector top-50 already holds the gold. Hybrid's
extra coverage only shows *deeper* (recall@20), which the generator never sees
at k=5. **Verdict:** drop from the final stack; bank ParadeDB/BM25 as a learning,
not a dependency.

### 🔶 Query expansion (multi-query) — modest, opt-in
LLM writes N rephrasings, retrieve each, RRF-fuse, rerank with the original
question. **Additive** (keeps the question's terms), so the right shape — but on
*our* stack the gain is small: **+2.1 recall@5, +3.5 recall@20 at the n=5 sweet
spot**, concentrated in synonym (+6.2) and multi-hop (+7.4) exactly as predicted.
n=3 too few to clear the rerank pool's noise floor; n=7 floods the fixed 50-slot
pool and *regresses*. Requires a reranker — bare fusion *degrades* precision
(consensus rewards confidently-wrong chunks). **Verdict:** opt-in flag, not
default — small gain for a per-query LLM call (~1s) + 6× embed/SQL.

### ❌ HyDE — wrong tool for this corpus (recall@5 −9.7)
Generate a hypothetical answer, embed *that* instead of the question.
**Replaces** the question → throws away its discriminative terms → synonym
craters 31→12. Concat (`question + doc`) rescued synonym but tanked the strong
categories (bge-m3 is mean-pooled, so a 200-tok hallucination drowns a 15-tok
question). **Why it fails here:** HyDE rescues queries that land in the *wrong
region* of a large, diverse corpus. Ours is tiny + single-topic — the question
already lands in the right region, and contextual retrieval already fixed the
document side, so HyDE's expansion is redundant noise. **Verdict:** reject.

---

## The meta-lessons (the part that travels to the next project)

1. **Invest in the document side before the query side.** Contextual retrieval
   (ingest-time, paid *once*) delivered +16 recall@5. The two query-side
   rewriting techniques (HyDE, expansion — per-query, paid *forever*) delivered
   −9.7 and +2.1 respectively. Both query-side techniques are mostly
   *compensation for a weak document side* — and ours wasn't weak. **A one-time
   ingest cost beat a perpetual per-query tax, decisively.**

2. **A two-stage pipeline must share representation.** Rerank-on-bare-text
   *undid* contextual retrieval because the stages disagreed about what each
   chunk was "about." Align them and the reranker *multiplies* the embedding's
   win instead of fighting it. The single most transferable finding.

3. **A strong reranker subsumes weaker retrieval tricks at small k.** Hybrid and
   (partly) expansion both add coverage the reranker was already realizing from
   the vector pool. At generator-k=5 the LLM sees the same top-5 either way.
   Coverage tricks earn their place only if you *raise* k or *lack* a reranker.

4. **Some gaps are not retrieval problems.** **Synthesis stayed 22.9% recall@5
   under every retrieval lever** (chunking, hybrid, rerank, contextual, HyDE,
   expansion). Even when expansion got synthesis chunks into the pool, the
   reranker wouldn't promote them — synthesis gold is relevant only *in
   aggregate*, invisible to a (question, single-chunk) cross-encoder. **Synthesis
   and contradiction are generation/agent problems (→ Module 7), and proving that
   at the retrieval layer is itself a 6.5 result.**

5. **Eval-driven means killing your darlings with data.** Four of six techniques
   were rejected or de-scoped — chunking variants, hybrid, HyDE, and (as default)
   expansion. None of those decisions was a hunch; each has a per-category table
   behind it. The harness (span gold + version filter + per-category +
   generation A/B) was the real Module 6 deliverable — it's what made every
   verdict defensible.

---

## Final recommended stack

**`contextual-v1` + rerank-on-context**, vector-only first stage, deep pool 50,
generator-k 5. Recall@5 **51.6%**, recall@20 **70.5%**, MRR **0.568**; Haiku
completeness **3.40**, faithfulness **4.64**, refusal **92%**.

- **Optional:** `--expand --expand-n=5` when synonym/multi-hop recall matters and
  the ~1s/query latency is affordable (+2.1 recall@5).
- **Carried as learnings, not dependencies:** ParadeDB/BM25 (hybrid), HyDE,
  chunking variants — code kept behind flags, off by default.
- **Open for Module 7:** synthesis + contradiction are generation-bound — a
  read-each-source-then-compare agent or a stronger generator, not a retrieval
  lever.
