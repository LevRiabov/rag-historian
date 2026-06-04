# Module 6 — Advanced RAG Techniques (working log)

> Running log of each technique, the change, and the measured delta. Every
> row is A/B'd against the **previous best** using the eval harness from
> Module 5. Baseline + methodology: [module-5-evals.md](module-5-evals.md).
> Rule (CLAUDE.md): change ONE thing, re-run, attribute, keep or discard.

## Baseline to beat (Module 5, naive-v1)

| | recall@5 | recall@20 | MRR | Faithfulness | Completeness | Refusal |
|---|---:|---:|---:|---:|---:|---:|
| naive-v1 / BGE-M3 / qwen3.5-9b | 35.1% | 55.4% | 0.475 | 4.32 | 3.00 | 92% |

In-scope n=43. The load-bearing diagnostic: recall@1=15.9%, recall@20=55.4% —
*the chunks are found but buried* (reranking territory). Synthesis is the
retrieval floor (recall@5=18.7%); contradiction is the ceiling (55.6%).

---

## 6.0 — Harness prerequisites (before any variant could be measured)

Two gaps had to be closed before a chunking A/B was even meaningful.

### Gap 1: `retrieve()` ignored `chunking_version`

[retrieve.ts](../roman-research/query/retrieve.ts) searched `FROM chunks`
with no version filter. Fine with one variant in the table; the moment a
second variant lands, every vector search mixes both variants' rows and the
numbers are garbage. **Fix:** added a `chunkingVersion` option (default
`naive-v1`) + `AND c.chunking_version = $3`, and bumped `hnsw.ef_search` to
100 so the post-filter doesn't under-fill top-K when variants co-reside.

### Gap 2: gold labeled by chunk ID can't survive re-chunking

`goldChunkIds` referenced `chunks.id` (SERIAL). Re-chunking mints new rows →
new IDs → the gold is meaningless for the new variant. Naively this forces a
separate hand-labeled golden set per chunking strategy (toil + breaks clean
A/B).

**Fix — chunking-invariant gold spans.** Gold is now a list of
`{ sourceSlug, charStart, charEnd }` spans in the source's `cleanedText`.
Every chunking variant derives its `char_start`/`char_end` from the *same*
coordinate system, so one golden set scores all variants.

- Recall rule = **midpoint coverage**: a retrieved chunk "covers" a gold span
  iff (same source) the chunk's char range contains the span's midpoint.
  Chosen over any-overlap because naive chunks overlap neighbors by ~50
  tokens — any-overlap would let an edge-clipping neighbor count as a hit and
  inflate recall. Midpoint lands the credit on the chunk that actually holds
  the passage.
- **Known bias:** coarser chunks cover more spans for free (a whole-chapter
  chunk trivially contains many midpoints). So recall@k is biased UP for big
  chunks — a chunking A/B must be read alongside the generation metrics
  (faithfulness/completeness) and cost/latency, never recall alone.
- Migration: [migrate-gold-to-spans.ts](../evals/migrate-gold-to-spans.ts)
  derived 105 spans from the 50 entries' legacy gold IDs (0 unresolved).
  Legacy `goldChunkIds` retained on each entry for audit only.

**Validation:** span-scored `naive-v1` reproduces the Module 5 baseline to
the decimal (recall@5 35.1%, recall@20 55.4%, MRR 0.475, every category
identical) → [results](../evals/results/2026-06-02-naive-bge-spans.json).
The measurement method changed; the measured reality didn't. Good.

---

## 6.1 — Chunking granularity (in progress)

Module 4 already shipped *section-aware* chunking (chunks never cross a
`(book, §chapter)` boundary), so 6.1 isn't "structure vs none" — it's a
**granularity sweep** over the same structure-aware base. All three variants
are the existing `chunkSections()` with different options; no new algorithm.

| Variant | targetTokens | overlap | Idea |
|---|---:|---:|---|
| `naive-v1` (baseline) | 500 | 50 | the Module 4/5 default |
| `window-300-v1` | 300 | 30 | finer — tighter topical vectors, more chunks |
| `chapter-v1` | 6000 | 0 | whole section = one chunk (splits only if it exceeds the embedder-safe ceiling) |

Storage is additive: each variant is a distinct `chunking_version`, all
co-resident in `chunks` (the only Module 6 step besides 6.4 that adds storage).

> Note for `chapter-v1` generation runs: whole-chapter chunks are large, so
> top-5 will blow the 8192-token generator context. Use `--generator-k=2`
> for its `--generation` runs; retrieval-only recall is unaffected.

### Results (retrieval-only, BGE-M3, span gold)

| Variant | chunks | recall@5 | recall@20 | MRR | vs naive (recall@5) |
|---|---:|---:|---:|---:|---:|
| **naive-v1** | 950 | 35.1% | 55.4% | 0.475 | — |
| window-300-v1 | 1054 | 34.3% | 53.9% | 0.476 | **−0.8 pt** |
| **chapter-v1** | 777 | **37.0%** | **58.9%** | **0.505** | **+1.9 pt** (size-biased) |
| **parent-child-v1** | 950 | 35.1% | 55.4% | 0.475 | **±0** (identical by design) |

`parent-child-v1` stores naive-sized (500/50) **children** — same vectors as
naive-v1, so retrieval recall is identical *by construction* (verified to the
decimal). Its whole thesis lives in **generation**: each retrieved child
expands to its full parent **section** before the LLM sees it
(retrieve.ts `expandToParents`), giving section-level context without the
recall-inflating size bias of `chapter-v1`. So it can't win the recall table —
it's a generation-only play, measured below.

Per-category recall@5 (→ vs naive; parent-child ≡ naive, omitted):

| Category | naive | window-300 | chapter |
|---|---:|---:|---:|
| literal | 28.7% | 28.7% | 28.7% |
| synonym | 31.3% | **25.0%** ↓ | 31.3% |
| multi-hop | 38.9% | 38.9% | **44.4%** ↑ |
| synthesis | 18.7% | **14.6%** ↓ | **22.9%** ↑ |
| contradiction | 55.6% | **61.1%** ↑ | 55.6% (MRR ↓ 0.83→0.78) |

### Analysis

- **window-300 (finer) lost.** Smaller chunks → thinner per-vector context →
  worse vocabulary bridging (synonym 31→25, MRR 0.22→0.18) and worse
  synthesis (18.7→14.6). The one win (contradiction 55.6→61.1) doesn't pay for
  the rest. Discard. Confirms the Module 3/4 intuition: below ~500 tokens the
  vector loses the surrounding context that lets a query match.

- **chapter (coarser) won on paper** — recall@5 +1.9, recall@20 +3.5, MRR
  +0.03 — concentrated exactly where keeping the full narrative together
  should help: multi-hop (38.9→44.4) and synthesis (18.7→22.9, recall@20
  34→49). Literal is identical (Caesar's sections are already <500 tok, so
  chapter ≡ naive there; gains come from the long Plutarch/Suetonius sections).
  Contradiction MRR dipped (0.83→0.78) — plausibly coarse chunks merge the two
  sides of a disagreement so the *first* hit ranks slightly lower.

- **⚠ Granularity-bias warning applies.** Part of chapter-v1's recall lift is
  the documented artifact: a bigger chunk covers more span midpoints for free,
  and when several gold spans cluster in one chapter, a single retrieval scores
  them all. So recall@5/+@20 OVERSTATE the real retrieval improvement. The
  decisive test is **generation**: does feeding whole chapters actually raise
  completeness, or just bloat the context? Pending A/B below.

### Generation A/B (the decisive test for 6.1)

Recall can't separate these three — naive ≡ parent-child, and chapter's edge is
size-biased. Only generation completeness can. **Confound to control: context
budget.** chapter chunks (max ~5,980 tok) and expanded parent sections blow the
local qwen 8192 window at K=5, and a context-exceeded error drops the question
(different questions graded per variant → unfair). Fix: **raise LM Studio's
loaded context to 32k** (a model load-time setting), keep the Module 5 generator
(**qwen3.5-9b, thinking OFF**) so these numbers stay comparable to the qwen
baseline (completeness=3.00), and fix K=5 so chunk *content* is the only
variable. Judge stays Haiku — independent of the generator, so no self-grading
bias on faithfulness.

```sh
# All three: same generator (qwen3.5-9b @ 32k ctx, no-think) + K, Haiku judge.
pnpm dev evals/run.ts --chunking-version=naive-v1 --generation \
  --llm=lmstudio --lmstudio-model=qwen/qwen3.5-9b --generator-k=5 --show-answers \
  --out=evals/results/2026-06-02-naive-gen-qwen.json

pnpm dev evals/run.ts --chunking-version=parent-child-v1 --generation \
  --llm=lmstudio --lmstudio-model=qwen/qwen3.5-9b --generator-k=5 --show-answers \
  --out=evals/results/2026-06-02-parentchild-gen-qwen.json

pnpm dev evals/run.ts --chunking-version=chapter-v1 --generation \
  --llm=lmstudio --lmstudio-model=qwen/qwen3.5-9b --generator-k=5 --show-answers \
  --out=evals/results/2026-06-02-chapter-gen-qwen.json
```

Then compare the `Completeness` / `Faithfulness` / `Refusal` aggregates across
the three runs (per-category too — expect parent-child's gains, if any, on
multi-hop + synthesis where section context matters most).

Decision rule: adopt parent-child-v1 as the 6.1 winner if completeness beats
naive at equal K (its recall already ties naive — strictly dominant if so).
Adopt chapter-v1 only if it *also* beats parent-child on completeness enough to
justify its context cost + contradiction-MRR regression. Otherwise keep naive
and bank the finding.

### Results (generation A/B, Claude Haiku generator @ K=5, n=50, Haiku judge)

Switched the generator to Claude Haiku: qwen3.5-9b's thinking can't be disabled
(`enable_thinking=false` only populates the content channel — it still reasons
~2–5k tokens under the hood), which both made runs ~30 min AND overflowed the
32k window on parent/chapter at K=5, dropping ~9 questions (unfair n<50). Haiku:
no thinking, 200k context (zero overflow → clean n=50), fast, ~$0.50/run.

| Variant | completeness | faithfulness | refusal | recall@5 |
|---|---:|---:|---:|---:|
| **naive-v1** | **3.12** | **4.86** | 84% | 35.1% |
| parent-child-v1 | 2.98 | 4.50 | 88% | 35.1% |
| chapter-v1 | 3.08 | 4.56 | 76% | 37.0% |

Per-category completeness (naive → parent-child → chapter):

| Category | naive | parent-child | chapter |
|---|---:|---:|---:|
| literal | **3.22** | 2.67 ↓↓ | 2.89 |
| synonym | 3.13 | 3.13 | **3.38** |
| multi-hop | 2.67 | 2.56 | **2.78** |
| synthesis | 2.50 | 2.25 ↓ | **2.75** |
| contradiction | 2.56 | **2.67** | 2.11 ↓ |

### Analysis — why bigger context lost

Completeness deltas (3.12/2.98/3.08) are within judge noise on a 1–5 scale, but
the direction is consistent and the faithfulness drop (4.86→4.50 for
parent-child) is real signal. Avg answer length: naive 1087 / parent-child 1203
(+11%) / chapter 1136 chars.

**Why parent-child underperformed despite IDENTICAL retrieval** (recall ≡ naive
by construction, so the only variable is 500-tok child slices → full sections):

1. **No precision/context gap to fill.** Parent-document retrieval pays off
   when children are *tiny* (sentence / ~150–250 tok) — too small to answer
   from, so the parent rescues context. Our 500-tok children are already
   context-rich; expanding to 3–6k tok adds redundancy, not signal. Technique
   misconfigured: children too large.
2. **Needle dilution.** The answer-bearing 500 tok now sit inside a fat
   section ("lost in the middle"). Fingerprint: **literal dropped most**
   (3.22→2.67) — a local fact's surrounding section is pure noise for it.
3. **Dedup collapsed source diversity.** `expandToParents` de-dupes parents;
   top-5 children usually cluster in 1–2 sections → 2–3 fat blocks instead of 5
   distinct passages. Fewer citable sources → hurts multi-hop / synthesis.
4. **Longer answers → more claims → lower faithfulness** (4.86→4.50).

**chapter-v1**: its recall edge did NOT convert — completeness ≈ naive,
faithfulness lower, refusal 76% (a big chapter usually contains *something*
grabbable, so it over-answers out-of-scope). Confirms the pre-registered
granularity-bias warning: recall@k rewarded size for free; answers didn't
improve.

### Verdict — 6.1

**Keep `naive-v1` (500/50) as the canonical chunking.** The sweep found the
Module 4 baseline already near the sweet spot. Bankable findings:
- finer (300) hurts — thin vectors lose bridging context;
- coarser (chapter) inflates recall but dilutes answers + over-answers;
- parent-document retrieval needs *small* children — at 500-tok children it's
  pure dilution. (Worth a future retry with ~200-tok children IF a later
  technique still leaves a context-gap to exploit; not now.)

Net: no chunking change. The value of 6.1 was the harness (span gold + version
filter) and ruling out three plausible-but-wrong ideas with evidence. On to
6.2 (hybrid search) and 6.3 (reranking) — both A/B against naive-v1.

---

## 6.2 — Hybrid search (BM25 + vector, RRF)

Goal: add lexical retrieval to catch exact rare terms (Pharnaces, Zela, "20
talents") that dense vectors smooth over. All retrieval-only, naive-v1, span
gold. Baseline to beat: recall@5 35.1% / recall@20 55.4% / MRR 0.475.

### Attempt 1 — Postgres core FTS (tsvector). Abandoned.

Two failure modes, both measured:
- `websearch_to_tsquery` is **AND** semantics → a natural-language question
  requires every word (incl. framing words "message"/"send" absent from the
  prose) → **0 lexical matches** for most questions → hybrid silently = vector.
- Rewriting to **OR** matched plenty but surfaced the WRONG chunks: top hits
  were "Diviatiacus"/"Vibullius", not the Pharnaces/Zela chunk. Root cause:
  **`ts_rank_cd` has no IDF** — it can't down-weight ubiquitous terms, and in a
  single-topic corpus "Caesar"/common verbs dominate the score.

Lesson: **Postgres core FTS ≠ BM25.** No IDF → useless lexical ranking here.

### Infra change — ParadeDB / pg_search

Swapped the DB image `pgvector/pgvector:pg16` → `paradedb/paradedb:latest`
(bundles pgvector + pg_search; real BM25 with IDF). PG18 → mount moved to
`/var/lib/postgresql`. Fresh volume + re-ingest naive-v1 (reproducible). BM25
index: `CREATE INDEX ... USING bm25 (id, text) WITH (key_field='id')`; query
`id @@@ paradedb.match('text', $q)` ranked by `paradedb.score(id)`. Verified:
BM25 correctly surfaced the §L/§XXXV/§LXVIII Pharnaces/Zela chunks core FTS
buried. (The unused `text_tsv` GIN column is left in place for rollback.)

### Results (retrieval-only, BM25 hybrid via RRF)

| Run | recall@5 | recall@20 | MRR |
|---|---:|---:|---:|
| vector baseline | 35.1% | 55.4% | **0.475** |
| hybrid, lexical weight 1.0 (equal RRF) | 32.0% | 48.6% | 0.336 |
| **hybrid, lexical weight 0.5** | 35.3% | **59.3%** | 0.395 |

Per-category recall@5 / recall@20 (vector → hybrid w=0.5):

| Category | r@5 | r@20 |
|---|---|---|
| literal | 28.7 → **50.0** 🟢 | 61.1 → **68.5** 🟢 |
| synonym | 31.3 → 12.5 🔴 | 50.0 → 43.8 🔴 |
| multi-hop | 38.9 → 33.3 | 53.7 → **61.1** 🟢 |
| synthesis | 18.7 → 14.6 | 34.4 → **38.5** 🟢 |
| contradiction | 55.6 → **61.1** 🟢 | 75.0 → **80.6** 🟢 |

### Analysis

- **BM25 works** — literal recall@5 nearly doubled (28.7→53.7 at w=1.0; +21 at
  w=0.5). IDF surfaces exact-term chunks vectors bury. That was the whole bet.
- **Equal-weight RRF (1.0) is wrong for mixed queries.** On synonym/synthesis
  (vocabulary mismatch — lexical has no signal) BM25 floods the fused top-K
  with noise and *evicts* good vector hits (synonym recall@20 50→12). A noise
  arm overriding a signal arm → everything regresses.
- **Down-weighting the lexical arm (0.5)** lets BM25 *add* where it has signal
  without *overriding* where it doesn't: recall@20 **+3.9** overall, literal +21,
  contradiction/multi-hop/synthesis coverage all up. Synonym recovers most of
  the way but stays hybrid's structural weak spot (lexical can't bridge
  vocabulary — that's 6.5 query-rewriting/HyDE territory).
- **Coverage up, ordering not (MRR 0.475→0.395).** This is the textbook hybrid
  signature: it widens the candidate net (more gold in top-20) but RRF's
  blended top is softer than pure vector. **That's exactly what reranking (6.3)
  fixes** — rerank the wider net to sharpen the top. So hybrid's real payoff is
  as a *candidate generator* feeding a reranker, not standalone.

We deliberately did NOT sweep the weight to maximize the 50-question score
(overfitting the eval). 0.5 is a principled midpoint.

### Verdict — 6.2

**Tentatively carry hybrid (lexical weight 0.5) forward as the candidate
generator**, on the strength of recall@20 +3.9 (the metric a reranker consumes)
and the large exact-term win. Its weak ordering (MRR) and synonym cost are
deferred to the techniques built to fix them: **6.3 reranking** (ordering) and
**6.5 query rewriting** (synonym). Definitive keep/drop after 6.3 measures
vector+rerank vs hybrid+rerank.

---

## 6.3 — Reranking (cross-encoder, bge-reranker-v2-m3)

Two-stage retrieval: fetch a DEEP pool (top-50) from stage 1, then re-score
every candidate with a cross-encoder (reads query+chunk JOINTLY, unlike the
bi-encoder's separate-then-compare) and keep the reranked top-20. Fixes
ordering (MRR/recall@5) AND realizes coverage from ranks 21–50.

Runtime: `bge-reranker-v2-m3` via the Infinity service (docker-compose
`infinity`, Cohere-compatible `/rerank`) — LM Studio/Ollama can't host
cross-encoders. Client: [lib/rerank.ts](../lib/rerank.ts). Local + free.

> Bring-up footgun: one /rerank request with all 50 docs OOM-restarted the
> CPU-bound Infinity container mid-run → socket reset → whole eval crashed
> (retrieval isn't wrapped in try/catch). Fix: cross-encoder scores each
> (query,doc) pair INDEPENDENTLY, so the client sub-batches (16/req) + retries.
> Mathematically identical, bounded memory.

### Results (retrieval-only, deep pool = 50 → reranked top-20)

| Stack | recall@5 | recall@10 | recall@20 | MRR |
|---|---:|---:|---:|---:|
| naive vector (baseline) | 35.1% | 46.3% | 55.4% | 0.475 |
| + hybrid (w=0.5), no rerank | 35.3% | 45.3% | 59.3% | 0.395 |
| **vector + rerank** | **45.0%** | 51.9% | 60.7% | **0.515** |
| **hybrid + rerank** | **45.0%** | 51.9% | **64.0%** | 0.513 |

Per-category recall@5 / recall@20 (baseline → vector+rerank → hybrid+rerank):

| Category | baseline r5 | vec+rr r5 | hyb+rr r5 | baseline r20 | hyb+rr r20 |
|---|---|---|---|---|---|
| literal | 28.7 | 47.2 | 50.9 | 61.1 | **83.3** 🟢 |
| synonym | 31.3 | **43.8** | 43.8 | 50.0 | 56.3 |
| multi-hop | 38.9 | 37.0 | 37.0 | 53.7 | 55.6 |
| synthesis | 18.7 | 18.7 | 18.7 | 34.4 | 34.4 |
| contradiction | 55.6 | **75.0** | 71.3 | 75.0 | 86.1 |

### Analysis

- **Reranking is the single biggest lever so far.** vector+rerank: recall@5
  **+9.9** (35.1→45.0), recall@20 **+5.3** (deep-pool coverage realized), MRR
  +0.04. Broad gains — literal +18, synonym +12 (the cross-encoder reads
  meaning better than the bi-encoder, so it even helps vocabulary mismatch),
  contradiction +19. This is the "found but buried" fix the Module 5 diagnosis
  predicted.
- **recall@1 barely moves (~16%).** The reranker reliably gets gold into the
  top-5 but the exact #1 stays hard (many questions have several gold chunks /
  near-ties). Fine — the generator sees the top-5.
- **The twist: a strong reranker largely SUBSUMES hybrid at the top.**
  hybrid+rerank == vector+rerank on recall@5 / recall@10 / MRR (45.0 / 51.9 /
  ~0.51, identical). Why: the reranker re-scores by true relevance regardless of
  which arm surfaced a candidate, and the vector top-50 pool already contains
  the gold the reranker promotes into the top-5. Hybrid's extra coverage only
  shows DEEPER — recall@20 60.7→**64.0** (+3.3), driven by literal r20
  64.8→**83.3** (BM25's exact-term chunks live at ranks 6–20 of the reranked
  list). At generator-k=5 the LLM sees the SAME top-5 either way.

### Verdict — 6.3

**Adopt reranking — it's the clear winner.** For the stack that feeds the
generator at k=5, **vector + rerank ≈ hybrid + rerank** (identical recall@5/MRR),
so the ParadeDB/BM25 complexity is NOT pulling its weight *at k=5* once a good
reranker is in place — its value is purely deeper coverage (recall@20, literal).

Open decision (settle with a generation A/B, not recall): does hybrid+rerank's
deeper coverage produce better ANSWERS than vector+rerank? At k=5 likely a tie;
it could matter if we raise generator-k (now safe — rerank cleans noise + 32k
ctx). If generation ties, **drop hybrid from the final stack** (keep
vector+rerank) and bank ParadeDB as a learning, not a dependency.

Running stack recall@5: 35.1 (naive) → 45.0 (+rerank). Still short of the 65–75
end-state target → the remaining coverage levers (6.4 contextual retrieval,
6.5 query rewriting) target the gold that never reaches the top-50 pool at all
(synthesis recall@20 stuck at 34%, the reranker's hard floor).

---

## 6.4 — Contextual Retrieval (the biggest lever)

Anthropic's technique: an LLM reads the WHOLE parent document + each chunk and
writes a 1–2 sentence note naming the people/places/events the chunk only
implies; we embed `citation + note + chunk` instead of the bare chunk. New
chunking_version `contextual-v1`: chunks identical to naive-v1 (same splits,
char offsets, ORIGINAL text stored — span gold + display unchanged), only the
stored EMBEDDING differs; the note lives in metadata.

Build: [contextualize.ts](../roman-research/ingest/contextualize.ts) — context
notes via **llama.cpp (qwen3.5-9b, no-think)**, profile picked per document
size (16k/32k/64k/100k); document placed FIRST in the prompt so llama-server's
KV cache reuses it across a document's chunks (free local prompt-caching);
phased (generate all notes, THEN embed) so llama-swap loads each model once;
notes cached to disk. Embeddings **local bge-m3 only**.

Footnotes kept on purpose (corpus realism + constant-corpus A/B). The gold-span
preflight found **8 gold spans touch footnote chunks** — so stripping footnotes
would have destroyed 8 answer regions. Keeping them was not just realistic but
necessary for recall integrity.

### Results (retrieval-only, span gold)

| Stack | recall@5 | recall@10 | recall@20 | MRR |
|---|---:|---:|---:|---:|
| naive vector (baseline) | 35.1% | 46.3% | 55.4% | 0.475 |
| vector + rerank | 45.0% | 51.9% | 60.7% | 0.515 |
| hybrid + rerank | 45.0% | — | 64.0% | 0.513 |
| contextual vector | 51.0% | 56.6% | 68.4% | 0.552 |
| contextual + rerank (bare text) | 47.9% | 55.2% | 67.2% | 0.524 |
| **contextual + rerank-on-context** | **51.6%** | **62.4%** | **70.5%** | **0.568** |

Per-category recall@5 / recall@20 (naive vector → contextual vector):

| Category | r@5 | r@20 |
|---|---|---|
| literal | 28.7 → **76.9** 🟢🟢 | 61.1 → 79.6 |
| synonym | 31.3 → 31.3 | 50.0 → **68.8** 🟢 |
| multi-hop | 38.9 → 40.7 | 53.7 → **63.0** 🟢 |
| synthesis | 18.7 → **22.9** | 34.4 → **41.7** 🟢 |
| contradiction | 55.6 → **77.8** 🟢🟢 | 75.0 → 86.1 |

### Analysis

- **Biggest single lever in Module 6.** Contextual embeddings ALONE beat every
  prior stack including hybrid+rerank: recall@5 +16, recall@20 +13, MRR +0.077
  over naive — and it lifted the coverage CEILING that reranking structurally
  cannot (synthesis @20 34→42, multi-hop @20 54→63). The notes inject the entity
  names a chunk omits, so fact-bearing chunks finally embed near their queries
  (literal @5 28.7→76.9, contradiction @5 55.6→77.8).
- **Reranking HURTS contextual — and the mechanism is the lesson.** contextual
  alone (51.0 / 0.552) beats contextual+rerank (47.9 / 0.524); rerank gutted
  literal @5 76.9→56.5. Cause: `retrieve()` reranks on the BARE `c.text` (we
  store original, un-contextualized text), so the cross-encoder sees "he crossed
  at dawn", can't tell it's the Rubicon, scores it low, and DEMOTES the very
  chunk the contextual embedding correctly surfaced. The two stages disagree
  about what each chunk is "about", so rerank partially undoes contextualization.

### rerank-on-context — the follow-up that won

Reranking on the BARE chunk text hurt contextual (the stages disagreed). Fix:
[retrieve.ts](../roman-research/query/retrieve.ts) now reranks on the
CONTEXTUALIZED text (`metadata.context` note + chunk) when a note is present.
Result: rerank flips from −3.1 to **best on every metric** — recall@10
56.6→**62.4** (+5.8), recall@20 **70.5**, MRR **0.568**. Per category, literal
recall@20 79.6→**88.9** and synonym recall@5 31.3→**43.8**.

**The lesson (case-study-worthy): a two-stage retrieval pipeline must share
representation.** The cross-encoder has to see the same disambiguating context
the bi-encoder embedding used, or the two stages fight — the reranker demotes
exactly the ambiguous-but-correct chunks contextual embeddings surfaced. Align
them and reranking *adds* to contextual instead of undoing it.

### Verdict — 6.4 (and the final retrieval stack so far)

**Final retrieval stack: `contextual-v1` + rerank-on-context.** Best on every
metric: recall@5 **51.6** (35.1 → +16.5 over naive), recall@10 **62.4**,
recall@20 **70.5** (55.4 → +15.1), MRR **0.568**. Contextual retrieval is the
single biggest lever in Module 6; rerank-on-context is the multiplier on top.

Remaining weak spot → 6.5: **synthesis** (still the floor: recall@5 22.9%,
recall@20 42.7%). Abstract meta-questions have no entity anchor for a context
note to surface, so HyDE / query rewriting (generate a hypothetical answer or
query variants to embed) is the lever aimed straight at it.

### Generation on the final stack — did retrieval become answers? (yes)

`contextual + rerank-on-context`, top-5 to generator, Haiku judge:

| Generator | faithfulness | completeness | refusal | gen cost |
|---|---:|---:|---:|---:|
| Claude Haiku | **4.64** | **3.40** | 92% | $0.21 |
| local qwen3.5-9b (no-think) | 3.98 | 3.02 | 92% | $0 |

vs the naive-retrieval baselines (same generators):

| | naive comp | final comp | Δ |
|---|---:|---:|---:|
| Claude Haiku | 3.12 | 3.40 | **+0.28** |
| local qwen | 3.00 | 3.02 | +0.02 |

- **The retrieval win became an answer win — for the strong model.** Haiku
  completeness +0.28 (recall@5 35→51.6 paid off). qwen stayed flat: a 9B model
  is its OWN bottleneck, so better context doesn't help much. Better retrieval
  helps frontier models more than local ones.
- **Frontier > local on the SAME chunks:** +0.38 completeness, +0.66
  faithfulness for $0.21/50q. The faithfulness gap (qwen hallucinates more) is
  the one that matters for a history corpus.
- **llama.cpp `--reasoning off` genuinely works** (≈API speed, no token
  explosion) — the thing LM Studio's partial disable couldn't do.

Per-category exposes two DIFFERENT remaining gaps:
- **synthesis** — low recall AND low completeness (2.25–2.63) → *retrieval*-bound
  → 6.5 HyDE.
- **contradiction** — recall@5 **76.9%** (gold IS retrieved) but completeness
  **2.22–2.44** (lowest) → *generation*-bound: the model retrieves both
  conflicting accounts but doesn't surface both sides. A system-prompt fix
  (name each source's claim on disagreement), NOT a retrieval lever.

---

## 6.5 — Query rewriting

### HyDE (Hypothetical Document Embeddings) — tested, REJECTED

Mechanism: instead of embedding the question, ask local qwen (no-think) to
hallucinate an answer paragraph and embed THAT. The fake answer doesn't need to
be *correct*, only *shaped like* the answer (declarative, entity-dense) so its
vector lands nearer the corpus than the interrogative question does. Code:
`roman-research/query/hyde.ts`; harness flag `--hyde` (+ `--hyde-concat`).
Scoped to the vector arm ONLY — `question` still drives BM25 + reranker
(`embedText` option on `retrieve()`).

A/B on contextual-v1, vector-only, all categories:

| Category | Baseline r@5 | pure HyDE | concat HyDE |
|---|---:|---:|---:|
| literal | 76.9 | 69.4 | 59.3 |
| synonym | 31.3 | **12.5** | 25.0 |
| multi-hop | 40.7 | 27.8 | 27.8 |
| synthesis | 22.9 | 25.0 | 26.0 |
| contradiction | 77.8 | 69.4 | **46.3** |
| **r@5 agg** | **51.0** | **41.9** | **37.4** |
| **r@20 agg** | **68.4** | 66.9 | 63.4 |
| **MRR** | **0.552** | 0.499 | 0.449 |

**Both variants LOSE. Rejected.** This is a structural mismatch, not a tuning
miss:
- **Pure HyDE** *replaces* the question → throws away its discriminative terms →
  **synonym craters 31→12** (the category that needs those terms most). Even on
  the target (synthesis) it's noise at r@5 and worse at r@20.
- **Concat** (`question + doc`) rescues synonym (12→25) — confirming "keep the
  question's terms" — but **tanks the strong cats** (contradiction 78→46,
  literal 77→59). Why: bge-m3 is **mean-pooled**, so a ~200-tok hallucination
  *outweighs* a ~15-tok question and drags the combined vector off the spot the
  clean question already nailed. No good middle: pure loses the terms, concat
  drowns them.
- **Root cause — wrong tool for this corpus.** HyDE rescues queries that land in
  the WRONG REGION of a large, diverse corpus. Ours is tiny + single-topic
  (everything is Caesar): the question already lands in the right region; the
  failure is fine-grained *ranking within a tight cluster*, which HyDE can't fix
  and actively disturbs by injecting hallucinated entities (drift = wrong chunk
  *within* the cluster). Contextual retrieval already fixed the document side, so
  the expansion is redundant noise. Averaging N docs wouldn't save it — concat
  shows the problem is hallucination *dilution*, not single-doc variance.
- **Takeaway:** HyDE is a large-corpus, weak-document-side technique. We have the
  opposite. Code kept (flag, off by default) for the writeup; never enabled.

### Query expansion / multi-query — tested, MODEST ACCEPT at n=5

The additive half of query rewriting: generate N rephrasings with local qwen
(structured output, ~700ms–1.2s), retrieve EACH (its own embedding + BM25 text),
fuse the per-query rankings via RRF, rerank the fused pool with the ORIGINAL
question. Code: `roman-research/query/expand.ts` +
`retrieveMultiQuery()` in `retrieve.ts`; harness flags `--expand --expand-n`.
The original question is always query #0, so it's purely additive — a drifting
variation costs a pool slot, never the baseline hit.

**Variations look great** — genuine vocabulary diversity ("falling sickness" ↔
"epileptic fits", "pecuniary distress" ↔ "monetary hardship"). qwen structured
output is reliable here.

**Needs rerank — and enough variations.** Two A/B axes:

*Vector-only (no rerank), synonym category:* HyDE-like FAILURE — r@5 31→25,
r@20 69→56. RRF rewards cross-query CONSENSUS, so a wrong-but-central chunk in 3
variations (RRF ≈ 3/65) beats original-only gold (RRF ≈ 1/75) and ejects it.
**Expansion without a reranker degrades precision.** Don't ship it bare.

*On the best stack (contextual + rerank), n-sweep:*

| n | r@5 | r@20 | MRR | synonym r@20 |
|---|---:|---:|---:|---:|
| 0 (baseline) | 51.6 | 70.5 | 0.568 | 68.8 |
| 3 | 51.6 | 72.3 | 0.569 | 68.8 |
| **5** | **53.7** | **74.0** | **0.570** | **81.3** |
| 7 | 52.3 | 71.7 | 0.566 | 62.5 |

**Inverted-U; n=5 is the peak** (at our fixed rerankPoolK=50). n=3 lacks the
vocabulary diversity to push synonym/multi-hop gold into the pool; n=7 floods the
50-slot pool with consensus-noise the reranker can't filter (synonym r@20
collapses 81→62). Per-category at n=5: **synonym r@5 +6.2 / r@20 +12.5,
multi-hop r@5 +7.4** — EXACTLY the categories theory predicts. Contradiction
−2.8 (noise). Net **+2.1 r@5, +3.5 r@20**.

- **Verdict:** modest, real win at n=5 — the opposite of HyDE (additive keeps the
  question's terms; replacement throws them away). Kept as an OPT-IN flag, not
  default: the gain is small and it adds a per-query LLM call (~1s) + 6× the
  embed/SQL work. A larger rerank pool might shift the optimum past n=5 (its own
  A/B; not chased). Worth enabling when synonym/multi-hop recall matters and
  latency budget allows.
- **Synthesis is coverage-flat under BOTH HyDE and expansion** (22.9 → 22.9,
  every run). 6.5 set out to fix the synthesis floor via retrieval; the hard
  result is **it can't be fixed at retrieval.** Even when expansion lifts
  synthesis r@20 (→45.8/49.0), the reranker won't promote those chunks to top-5
  — synthesis gold is relevant only in AGGREGATE, invisible to a (question,
  single-chunk) cross-encoder. **Synthesis joins contradiction as a Module 7
  generation/agent problem, not a retrieval lever.** This is the load-bearing
  conclusion of 6.5.
