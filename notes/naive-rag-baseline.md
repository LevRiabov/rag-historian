# Naive RAG baseline (Module 4)

> **Captured:** 2026-05-29 (end of Module 4, before any retrieval improvements).
> **Purpose:** lock in the baseline so each Module 6.x technique can A/B against the same questions and configuration. Re-run the same questions after each change; record the deltas in companion files (`module-6-1-structure-aware.md`, etc.).

## Configuration

| | |
|---|---|
| **Corpus** | 4 Caesar primary sources (Gallic War, Civil War, Plutarch's *Life of Caesar*, Suetonius's *Life of Julius Caesar*) |
| **Cleaned text** | ~1.25 MB / ~314k tokens |
| **Chunks** | 950 total — naive token-window chunker, target 500 tokens, 50 overlap, never crosses section boundaries |
| **Chunking version** | `naive-v1` |
| **Embedders** | BGE-M3 (1024d, LM Studio) AND OpenAI text-embedding-3-small (1536d). Both columns populated. |
| **Retrieval** | Cosine HNSW, top-K = 5, BGE-M3 column |
| **Generator (sample default)** | LM Studio gpt-oss-20b. Claude Sonnet 4.6 for two samples where noted. |
| **Re-rank / hybrid / contextual / query rewrite** | OFF (Module 6.x territory) |

## Cost & latency profile

| Pipeline stage | Cost (per query) | Latency |
|---|---|---|
| Embed query (BGE-M3) | $0 | ~10-50 ms (warmed) |
| pgvector top-5 search | $0 | ~50-100 ms |
| Generate answer (gpt-oss-20b) | $0 | ~1-2 s |
| Generate answer (Sonnet 4.6) | ~$0.013 | ~10 s |

Ingest one-time cost: $0.006 for OpenAI embeddings; BGE-M3 free.

## Sample question results

Each entry: question → retrieved chunks (top-5 similarity scores + source labels) → answer summary → quality rating → identified failure mode(s).

Rating scale: ✅ correct & well-cited · 🟡 correct but partial · ❌ wrong or fabricated · 🚫 correctly refused.

---

### Q1 — "Who assassinated Julius Caesar and why?"
*LLM: Claude Sonnet 4.6 · embedder: BGE-M3 · K=5*

**Retrieved (similarity / source / chapter):**
1. 0.541 — Suetonius §I (Caesar's birth, NOT relevant)
2. 0.531 — Plutarch §LXVII (immediately after the assassination)
3. 0.522 — Suetonius §LXXXIX (analysis of Caesar's ambition)
4. 0.521 — Suetonius §LXXXII (**the actual assassination scene**)
5. 0.517 — Caesar's Gallic War, Book I §XII (the Saône river, NOT relevant)

**Answer summary:** Named Cimber, "one of the Cassii", and Brutus as assassins (per Suetonius §LXXXII). Correctly stated the sources don't explain *why* and refused to speculate.

**Rating:** 🟡 — facts cited were accurate; refusal on motivations was honest; but only 2/5 retrieved chunks were directly relevant. The "why" answer would have been better if §LXXX (motivations) had ranked instead of §I (birth).

**Failure modes:**
- **Token-frequency bias:** §I ranked #1 because "Julius Caesar" appears prominently in its opening line, dominating the vector signal.
- **Off-topic in top-5:** Gallic War Saône-river chunk pollutes the context.
- **Missing key chunks:** §LXXX (motivations) and Plutarch §LXVI (assassination scene) didn't surface.

**Module 6 fixes that should help:**
- 6.2 hybrid search: BM25 would boost passages containing "assassinated" / "conspirators"
- 6.3 reranking: a cross-encoder would prune §I and the Saône chunk
- 6.4 contextual retrieval: a context line like "this passage discusses the conspirators' motivations" would lift §LXXX

---

### Q2 — "Did Caesar want to be king?"
*LLM: Claude Sonnet 4.6 · embedder: BGE-M3 · K=5*

**Retrieved:**
1. 0.589 — Suetonius §XXX (Caesar's anxiety for supreme power, Cicero quote)
2. 0.583 — Plutarch §VI (Catulus Lutatius's accusation)
3. 0.579 — Plutarch §XIII (triumph rules — tangential)
4. 0.578 — Suetonius §LXXXIX ("outrageous ambition" analysis)
5. 0.578 — Suetonius §IX (early conspiracy with Crassus)

**Answer summary:** Synthesized evidence of Caesar's broader ambitions for supreme power across Suetonius and Plutarch. Quoted Catulus Lutatius directly. Sharply distinguished the literal question ("king") from the broader one — noted *"king" never appears in the sources*.

**Rating:** ✅ — strong retrieval (4/5 directly relevant), accurate synthesis, honest about literal-vs-broad question framing.

**No major failure modes.** The literal-vs-semantic distinction the model drew is something Module 6.4 contextual retrieval would have made even cleaner.

---

### Q3 — "How many years passed between Sulla's death and Caesar's first consulship?"
*LLM: gpt-oss-20b · embedder: BGE-M3 · K=5*

**Retrieved:**
1-3. 0.564 / 0.559 / 0.537 — three chunks from Plutarch §LXVIII (Caesar's Gallic War campaign — wrong era)
4. 0.522 — Suetonius §I (Caesar's youth — wrong era)
5. 0.511 — Suetonius §XX (mentions Caesar's consulship but no date)

**Answer summary:** Correctly refused. *"I don't have a source that gives both the date of Sulla's death and the date of Caesar's first consulship, so I can't calculate."*

**Rating:** 🚫 (correct refusal) — the right answer for the data. Sulla's death (78 BC) and Caesar's first consulship (59 BC) are mentioned in the corpus but not in chunks the naive retrieval surfaced.

**Failure modes:**
- **Multi-hop with sparse anchors:** the question needs TWO separate facts (Sulla death date + Caesar consulship date), each in different chunks. Vector retrieval finds chunks similar to the WHOLE question, not chunks containing either sub-fact.
- **Synonym mismatch / temporal anchors:** the corpus uses years implicitly ("Caesar was sixteen when his father died") not explicit dates that the question phrasing reaches for.

**Module 6 fixes that should help:**
- 6.5 query rewriting / HyDE: decompose into *"When did Sulla die?"* + *"When was Caesar first consul?"*, retrieve for each, then synthesize
- 6.2 hybrid search: BM25 catches the literal names "Sulla", "consulship" better than dense vectors

---

### Q4 — "What happened when Caesar crossed the Rubicon?"
*LLM: gpt-oss-20b · embedder: BGE-M3 · K=5*

**Retrieved:**
1. 0.569 — **Plutarch §XXXII (the actual Rubicon crossing scene)**
2. 0.564 — Gallic War Book I §XII (Saône river — irrelevant)
3. 0.562 — Gallic War Book I §XIII (Helvetii battle — irrelevant)
4. 0.555 — Plutarch §LXVIII footnote 517 (mentions Caesar at Ravenna — relevant context)
5. 0.548 — Plutarch §XX (winter quarters in Sequani — irrelevant)

**Answer summary:** Synthesized Plutarch §XXXII correctly — 300 horse + 5000 legionaries, surprise rather than overwhelm, hurried to Ariminum before daybreak. Acknowledged limits of source.

**Rating:** ✅ — top-1 was the right chunk and the synthesis was accurate, citation discipline strong. Local 20B model handled it competently.

**Failure modes:**
- **Off-topic noise:** 3 of 5 retrieved chunks were unrelated (Gallic War river/battle, Plutarch winter quarters). Model handled it gracefully by ignoring them, but the context tokens are wasted.

**Module 6 fixes that should help:**
- 6.3 reranking: would drop the river/battle/winter-quarters chunks
- 6.1 structure-aware chunking: section IV.XII was a single river-description paragraph; better chunking would still have given it a separate embedding but its irrelevance is fundamentally a semantic-similarity issue reranking handles

---

### Q5 — "What did Caesar think of TikTok?"
*LLM: gpt-oss-20b · embedder: BGE-M3 · K=5*

**Retrieved:** five low-similarity Roman-history chunks (0.50-0.51, the floor of similarity in our corpus).

**Answer summary:** *"The provided passages do not contain any information about Julius Caesar's thoughts on TikTok or any modern social media platform."* (46 tokens, 800 ms.)

**Rating:** 🚫 (correct refusal) — clean one-sentence refusal. No fabrication, no apology padding.

**Failure modes:** none — out-of-scope behavior is exactly what we want.

**Note:** the 0.50-0.51 similarity range is the *floor* for our corpus. Any retrieval scoring below ~0.51 against any question can probably be treated as "not really about anything in the corpus" — useful threshold for future refusal logic.

---

## Headline numbers

- **5 probes:** 2 ✅ correct & well-cited, 1 🟡 partial-but-correct, 2 🚫 honest refusals (1 ideal, 1 should have answered if better retrieval had surfaced the facts)
- **0 ❌ wrong-or-fabricated answers** — the citation + refusal prompt is doing its job
- **Subjective overall quality: ~3/5** (user assessment) — works but visibly retrieval-bottlenecked

## Identified bottlenecks (user-named + observed)

1. **Corpus completeness.** Caesar-only focus means questions outside Caesar's narrow biography refuse correctly but feel limited. Adding Mommsen (`tier='historian'`) and/or Plutarch's *Antony* / *Brutus* would broaden coverage. **Hold for Module 6 evidence** before expanding — premature corpus growth muddies the eval signal.
2. **Top-K = 5 is often too small.** Multi-source synthesis questions want 8-15 chunks; the naive K=5 forces the retriever to be highly precise to win. K=20 + reranking down to 5 (Module 6.3) is the conventional fix — bigger candidate pool, cross-encoder picks the best 5.
3. **Token-frequency bias on common entities.** "Julius Caesar" appears so often that chunks just containing the name (e.g., his birth in Suetonius §I) outrank topically relevant chunks for almost any question. Module 6.2 hybrid search and 6.4 contextual retrieval both target this.
4. **Synonym / vocabulary mismatch.** Modern question phrasings ("king", "absolute power", "first consulship") don't match the corpus's "dictator perpetuo", "Cæsar was raised to the consulship". Module 6.5 query rewriting / HyDE targets this directly.
5. **Multi-hop questions fail by design.** Vector retrieval finds chunks similar to the WHOLE question. Questions that need to combine facts from different chunks need decomposition (HyDE, sub-query rewriting — Module 6.5).

## Hypotheses for Module 6 (to be tested with the eval set in Module 5)

These are *predictions*; the eval harness in Module 5 will measure whether they're right.

| Technique | Expected biggest win on | Predicted recall@5 lift |
|---|---|---|
| 6.1 Structure-aware chunking | Questions about specific sections (Plutarch §XIV, Suetonius LXXX) — chapter-aware retrieval | +5-10% |
| 6.2 Hybrid search (BM25 + vector) | Named-entity questions (Pompey, Pharsalus, Catiline) and rare-word queries | +10-15% |
| 6.3 Reranking | Almost everything — by pruning Q1's §I bias and Q4's noise | +15-25% |
| 6.4 Contextual retrieval | Questions where the answer is buried in a chunk whose surface text doesn't match (Q1's missing §LXXX) | +20-35% (matches Anthropic's published result on legal/code corpora) |
| 6.5 Query rewriting / HyDE | Multi-hop and synonym-mismatch questions (Q3) | +20%+ on those specific question types |

After Module 5 we'll have hard numbers replacing the "expected" column.

## How to re-run this baseline

```sh
# Make sure the database is loaded with the naive-v1 chunks
docker compose up -d
pnpm dev roman-research/ingest/index.ts

# Re-run each probe
pnpm dev roman-research/query/index.ts "Who assassinated Julius Caesar and why?" --llm=claude-sonnet
pnpm dev roman-research/query/index.ts "Did Caesar want to be king?" --llm=claude-sonnet
pnpm dev roman-research/query/index.ts "How many years passed between Sulla's death and Caesar's first consulship?"
pnpm dev roman-research/query/index.ts "What happened when Caesar crossed the Rubicon?"
pnpm dev roman-research/query/index.ts "What did Caesar think of TikTok?"
```

Same questions, same K, same chunking_version, same embedder. The only thing that changes across Module 6.x runs is the retrieval strategy.
