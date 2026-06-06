# Module 7 — Agent Fundamentals (running log)

Roman Research Agent v1: a single-agent loop that beats single-shot RAG on the
categories Module 6 *proved* are reasoning-bound, without regressing the rest.
Final table lands in a `module-7-comparison.md` (mirrors the Module 6 pattern);
this file is the as-it-happens log of every build step + A/B delta + footgun.

Baseline/methodology: [module-5-evals.md](module-5-evals.md). What RAG could and
couldn't do, and why these two gaps are agent-shaped:
[module-6-comparison.md](module-6-comparison.md).

## The thesis (what the agent must prove)

Module 6 left two gaps that are **reasoning, not retrieval**:
- **synthesis** (recall@5 22.9%, coverage-flat under every retrieval lever) — the
  answer is distributed; no single (question, chunk) match exists. Agent fix:
  decompose → search each sub-question → unify.
- **contradiction** (retrieves at 76.9% but the generator *blends* the two
  accounts) — a generation gap. Agent fix: `search_within_source` → read each
  source separately → contrast by name.

**A/B contract:** agent vs the Module 6 single-shot final stack
(`contextual-v1` + rerank-on-context, k=5), same 50 golden questions, same
judges. Win on synthesis/contradiction/multi-hop; **hold literal/synonym and the
92% out-of-scope refusal** (the metric most at risk — agents over-search and
rationalize). New agent metrics: tool-calls/q, tokens/q, cost/q, success rate,
per category. Decisions:
- **Model:** A/B **Claude** vs **local qwen3.5-9b** (tests "small model is its
  own bottleneck" at the *behavioral* level, via traces — does qwen actually
  call the multi-step tools, or shortcut + blend?).
- **Tools:** the phase-doc set **+ `search_within_source`** (the contradiction
  mechanism).

## Plan (slices)

0. Control — A/B contract (above). No code.
1. **Tools** — `agent/tools.ts`. ✅ done (below).
2. Loop — provider-agnostic; reuse `runTools` (already built in lib/claude.ts &
   llamacpp.ts: max-iter, cost cap, tracing, step records). Add finalize-as-
   terminal handling.
3. System prompt — methodology + abstain-as-first-class; iterate on evals.
4. Evals — `run.ts --agent --llm=<claude|local>`: same judges + agent metrics,
   per-category A/B + the two-model gap. Langfuse trace tree per question +
   scores attached.

---

## Slice 1 — Tools (DONE)

`roman-research/agent/tools.ts` — `createAgentTools(db): Tool[]`, five tools, all
searching the **Module 6 final stack** (contextual-v1 + rerank-on-context; the
agent is a reasoning layer on top of best single-shot retrieval, not a new
retriever).

| Tool | Wraps | Note |
|---|---|---|
| `search_corpus(query, top_k=5)` | `retrieve` (all sources) | primary discovery |
| `search_within_source(source, query, top_k=5)` | `retrieve` + new `sourceSlug` filter | **the contradiction lever**; `source` is a Zod enum over the 4 slugs (rejects hallucinated names) |
| `read_chunk(chunk_id)` | direct SQL by id | full text before citing |
| `list_sources_consulted()` | closure state | per-run coverage map; marks snippet-only vs read-in-full |
| `finalize(article)` | identity | Zod-validates `article`; **loop owns termination** (Slice 2) |

Supporting change: added optional `sourceSlug` to `RetrieveOptions` →
`vectorQuery` (`$4::text IS NULL OR s.slug = $4`, single statement for both
cases). Guarded to **throw in hybrid mode** rather than silently ignore the
filter (agent uses vector+rerank only). Optional + default-undefined → zero
behavior change for the eval harness.

Design decisions logged for later:
- **Per-run state:** `consulted` lives in the factory closure, so Slice 2 calls
  `createAgentTools(db)` once per question → fresh coverage view, no reset logic.
- **finalize is a tool, not a natural-text stop.** Keeps the article Zod-
  validated and makes `stoppedBy: finalize` explicit in traces. `runTools`
  terminates on natural text end, so Slice 2 adds a thin wrapper that breaks on a
  finalize tool_use and returns its `article`. (Open: decide in Slice 2 whether
  to special-case it in the loop or post-process.)
- **Tool descriptions carry methodology nudges** ("start here", "the key tool for
  contradiction questions", "abstaining is a correct outcome"). Whether the model
  — especially qwen — actually heeds them is a *trace-level* result to check, not
  an assumption.

Verified: `pnpm typecheck` clean; Biome clean on the new file.

**Smoke test (live, docker + llama-swap up) — `agent/smoke-tools.ts`, all 5 tools
PASS:**
- `search_corpus("Did Caesar want to be made king?")` → Plutarch §LX (*desire of
  kingly power*) + Suetonius §LXXIX (Lupercalia) — the right contradiction gold.
- `search_within_source("suetonius-caesar", ...)` → 5 **Suetonius-only** chunks →
  `sourceSlug` filter confirmed working.
- `read_chunk` → full passage; `list_sources_consulted` → grouped by source,
  `[read in full]` vs `[snippet only]` correct.
- Error paths graceful (no throw): bad source enum → Zod rejection listing valid
  slugs; bad chunk id → error string. Both recoverable by the loop.

Observations (not bugs, parked):
- **Within-source rerank quality:** `search_within_source("suetonius","diadem
  crown king")` ranked the relevant §LXXIX *3rd* (behind aedileship/Alexandria
  chapters). Single-source pool is small; rerank still ordered loosely. Agent
  would read §LXXIX anyway — but worth watching whether weak within-source
  ranking hurts contradiction answers in the eval.
- **Zod error verbosity:** the enum-rejection string is the full Zod issue array
  (noisy but parseable). If models trip on it, simplify `executeTool`'s error
  formatting — lib-wide, out of scope for Slice 1.

`smoke-tools.ts` kept as a manual harness for now; remove once the Slice 2
end-to-end loop smoke test supersedes it.

---

## Slice 2 — The loop (DONE)

Built `agent/index.ts` (`runAgent`) + `agent/cli.ts` (manual runner with a live
ReAct trace). **No new loop code** — reused the lib's `runTools` (Anthropic in
claude.ts, OpenAI-compat in lmstudio.ts, which llama.cpp delegates to). The
agent module is pure orchestration: build tools, pick model, run loop.

**Lib change — `terminalToolName` (additive) on both `runTools`.** When the
model calls the named tool, the loop runs it then stops (`final_answer`),
returning that tool's output as `text`. The agent passes `finalize`. ~6 lines
each in claude.ts + lmstudio.ts; tool-result messages are still appended so the
transcript stays valid/inspectable. Default behavior unchanged when unset.

**A/B smoke: "Did Caesar want to be made king?" (contradiction) — both PASS the
multi-step behavior single-shot RAG can't do.**

| | Claude Haiku | local qwen-9b (16k, think off) |
|---|---|---:|
| Tool calls | 11 (3 search, 4 read, 2 within-source, 1 list, 1 finalize) | 7 (2 search, 3 read, 2 within-source) |
| Read each source separately? | yes (Plutarch, Suetonius, + Caesar's own works) | yes (Plutarch, Suetonius) |
| Contrasts sources by name? | yes | yes (bold per-source sections) |
| **Called `finalize`?** | **yes** (explicit terminal) | **NO — emitted article as a text turn** |
| Stop | final_answer (finalize) | final_answer (natural text end) |
| Cost / latency / in-tok | $0.04 / 30s / 34k | $0 / 14s / 15k |

**Findings (Slice 2):**
1. **Loop is provider-agnostic and works on both** — same tools, same prompt,
   same ToolLoopResult. The headline A/B is one flag.
2. **finalize is NOT honored universally.** qwen ignored the finalize tool and
   wrote the answer as plain text; Claude called it. Both end at `final_answer`
   because natural-text termination is the backstop → **don't rely on finalize
   being called.** Making `terminalToolName` additive (not required) was correct.
   Open: do we *want* to push qwen to use finalize (stronger prompt / tool_choice),
   or accept text-termination as equivalent? Leaning accept-both (simpler); the
   article is captured correctly either way. Revisit if the eval needs a clean
   "abstained vs answered" signal.
3. **qwen did the multi-step fine here** — early dent in the "small model is its
   own bottleneck" prior, AT THE TOOL-USE LEVEL. Whether its *answer quality*
   (faithfulness/completeness) keeps up is the Slice 4 eval question. (qwen drew a
   firmer "did not want to be king" conclusion; Haiku hedged more — judge will
   score which is better supported.)
4. **Cost signal for the eval budget:** Haiku ≈ $0.04/question (context grows with
   each tool result: 34k in-tok) → ~$1.7 for 43 in-scope questions; Sonnet/Opus
   more. qwen is free. Claude `costCapUSD` defaults to $0.50/question.
5. Claude batches parallel tool calls per turn (iter 2 = 4 calls); loop handles
   it. Claude alone called `list_sources_consulted` (coverage self-check).

Verified: `pnpm typecheck` clean.

---

## Slice 4 — Eval harness (`--agent` mode) (DONE; Langfuse still pending)

Extended `evals/run.ts` with `--agent`: replaces single-shot retrieve+generate
with `runAgent`, scores the article through the SAME three judges, adds agent
metrics. Flags: `--agent`, `--llm=<driver>`, `--max-iter` (30), `--think`
(qwen thinking-on). Design choices:
- **Faithfulness evidence base = union of consulted chunks** (`getConsultedChunks`,
  added to `agent/tools.ts` — searched chunks deduped by id; read_chunk adds no
  new evidence since its ids always come from a prior search).
- **`goldCoverage`** (fraction of gold spans covered by ANY consulted chunk) is
  the agent's retrieval analog — it has no ranked top-K, so recall@k/MRR are NaN
  in agent mode and the recall table is suppressed.
- New types: `AgentMetrics` (per-q) + `AgentAggregate` (overall/per-cat) in
  `evals/types.ts`. Tracks toolCalls, toolCallsByName, stop, calledFinalize,
  consultedCount, goldCoverage, latency, cost.

**Validation run — `--agent --llm=llamacpp --category=literal` (9 q, free):
harness works end-to-end, and immediately exposed a real problem.**

| metric | value |
|---|---|
| avg tool calls | **11.7** (literal should be ~1–2) |
| avg gold coverage | 94.4% |
| finalize rate | 67% |
| faithfulness / completeness | 4.22 / 3.67 |
| refusal accuracy | 88.9% |

**Finding F6 — local qwen does NOT self-regulate effort; it over-searches and
can spiral.** The methodology prompt ("one search is usually enough for literal")
is ignored by qwen. Two pathological cases:
- **q-007 (veni vidi vici): 30 calls → `max_iterations`, never finalized** → C=1/5,
  scored `should-have-answered`. *Hitting the iteration cap yields a non-answer* —
  a real failure mode, not just inefficiency.
- q-013 (Pompey): 23 calls (11 `search_within_source`) but eventually a good
  answer (F5/C4) — just 36s of thrashing.
- q-012 (Cleopatra): coverage 50%, F=2/5 — under-covered + unsupported claims.

**Implications (drive Slice 3):**
1. The over-search risk we flagged is **real and measured**, and it's worst on the
   EASY category — the opposite of the intended effort curve. qwen's loop control
   is the bottleneck, not its retrieval (coverage 94%) or faithfulness (4.22).
2. `max_iterations` needs a **graceful "forced finalize"** — when the cap hits,
   prompt the model once more for a finalize instead of returning empty/partial
   text. (Currently a capped run can score as a non-answer.)
3. Slice 3 prompt work should target qwen's stop discipline: stronger "STOP once
   you can cite" guidance, possibly a tool-call budget hint, and/or a lower
   default `--max-iter` for the local model.

This is the eval-driven loop working as intended: the harness turned a vague
"agents might over-search" worry into three concrete, addressable findings.

Not yet done: **Langfuse** (the `langfuseTracer` impl + per-run trace tree +
score attachment) — Slice 4's second half. Full 50-q × 2-model A/B not yet run
(cost/time — pending user go-ahead).

Next: Slice 3 (prompt + forced-finalize) → re-run literal to confirm tool-calls
drop → full A/B → Langfuse.

---

## Slice 3 — Loop fix + prompt tuning (PARTIAL)

Two changes aimed at F6 (qwen over-search / max-iter non-answer):

**(a) Graceful forced-finalize on max-iter (lib, both runTools).** When the loop
exhausts iterations AND `terminalToolName` is set, make ONE final tool-free call
so the model synthesizes an answer from what it gathered (instead of returning
''). `stop` stays `max_iterations` for honesty. ~30 lines each in claude.ts +
lmstudio.ts.

**(b) Prompt rewrite** — moved "WHEN TO STOP" to the top; "most questions need
1–3 searches", "search each source ONCE", "finalize the moment you can cite".

**Re-validation — `--agent --llm=llamacpp --category=literal` (same 9 q):**

| metric | before (F6) | after Slice 3 |
|---|---|---|
| avg tool calls | 11.7 | **11.6** (≈unchanged) |
| completeness | 3.67 | **4.56** |
| refusal accuracy | 88.9% | **100%** |
| avg gold coverage | 94.4% | 74.1% (variance — see below) |
| q-007 | 30 calls → max_iter → C=1/5 | **final_answer → C=4/5** |

**Findings:**
- **F7 — the forced-finalize fix works and is the important one.** No question now
  scores as a non-answer at the cap; completeness +0.89, refusal +11pts. This was
  the real bug.
- **F8 — prompt-based stop discipline does NOT curb qwen's over-searching.** Still
  11.6 tools/literal-Q, still 16–17 `search_within_source` calls on some questions
  despite "search each source ONCE". qwen-9b doesn't honor meta-instructions about
  its own tool budget. **This is a model trait, not a correctness bug** — faith
  (4.22) and completeness (4.56) are fine; the cost is latency, not quality.
- **F9 — qwen agentic runs are high-variance.** Coverage 94%→74% across identical
  runs; individual Qs flip 100%↔50%. The multi-step path is non-deterministic, so
  qwen's per-question numbers are noisy (aggregate over the full set, don't trust
  single-Q deltas).

**Open decision:** curbing qwen over-search needs a CODE-level cap (reject the
Nth search_within_source with an error observation) or a lower local `--max-iter`
— prompt won't do it. Deferred: over-search is latency-only, and the A/B should
quantify it vs Claude (which may self-regulate) before deciding it's worth a hard
guardrail (cf. CLAUDE.md "don't add guardrails until needed").

---

## First full A/B run — Claude Haiku (n=50) + crash on qwen

Claude Haiku agent completed all 50; qwen **crashed at q-018** — its over-search
grew the transcript past the 16k profile (`16764 > 16384 tokens`). Two fixes:
- **Harness resilience:** wrapped the agent block in try/catch (was unprotected,
  unlike the generation path) — one question's failure no longer kills the batch.
- **Context:** agent local default raised qwen-9b-16k → **qwen-9b-64k** (an agent
  loop accumulates a transcript; it needs far more context than single-shot).

### Claude Haiku agent vs Module 6 single-shot (the headline)

| metric | M6 single-shot (Haiku) | agent (Haiku) | Δ |
|---|---|---|---|
| Completeness | 3.40 | **3.96** | **+0.56** |
| Faithfulness | 4.64 | 4.34 | −0.30 |
| Refusal acc. | 92% | **98%** | +6 |
| cost / 50q | $0.21 | $1.35 | 6× |

Per-category (Claude agent): tools self-regulate by difficulty — literal 4.8 →
synthesis/multi-hop 9.8/9.7 (the CORRECT effort curve, opposite of qwen's flat
~11). **F10 — Claude self-regulates effort; the over-search is a qwen trait, not
systemic** → the "no hard cap" call holds for the strong model.
- contradiction comp **3.00** (lowest) despite coverage 91.7% — retrieves both
  sides but judged incomplete. Watch.
- synthesis coverage 47.9% (still the retrieval-hard category) but comp 3.75 (up
  from the M6 synthesis floor).
- out-of-scope: 1 leak (q-045 "gunpowder weapons in Gaul" — searched + answered,
  should-have-refused). 98% overall.

### Faithfulness dip analysis (−0.30) — is it the model's fault?

Read every F≤3 judge rationale. Verdict: **genuine but benign — the expected
completeness↔faithfulness tension, NOT fabrication. Zero F=1.** Four buckets:
1. **Over-interpretation / inferential glue (dominant)** — the agent connects dots
   the sources don't: q-032 invented an "imminent kingship decree" causing the
   murder (F2 but **C5** — completeness BY over-claiming); q-043 "258,000 = deaths"
   arithmetic. *This is the direct cost of +0.56 completeness.*
2. **Source misattribution — correlates with over-search.** q-035 attributed
   Plutarch→Suetonius (31 chunks); q-028 conflated two events (37 chunks); q-043
   (60 chunks). **F11 — piling up 30–60 chunks degrades attribution faithfulness.
   So over-search is NOT purely latency-neutral — it can hurt quality.** (Direct
   test for the qwen run.)
3. **Injected outside knowledge (rare)** — q-021 "Alea iacta est / Latin / native
   language" (true, not in corpus).
4. **Measurement artifact (harness bug, FIXED)** — q-023: judge read the agent's
   `[1722]` citations as "fabricated" because `faithfulnessUser` labeled evidence
   `[1]..[N]` (arrival order) while the agent cites by chunk_id. **Fix:**
   `labelByChunkId` flag → label evidence by chunk_id in agent mode (single-shot
   keeps arrival-order). One genuine error total: q-001 reversed the
   Rubicon/Ariminum sequence (under-researched, 5 chunks).

### Fixes applied before the clean re-run
- Citation-artifact fix (above) — pure correctness.
- Faithfulness prompt line (rule 1): "Cite ONLY what the passages state — no added
  facts/dates/original-language phrases/inferences; attribute each claim to the
  specific source." Targets buckets 1–3.
- **Decision: re-run BOTH models fresh** (current Claude numbers carry the artifact
  + pre-date the prompt fix) for clean, comparable, defensible case-study numbers.

---

## Clean full A/B (n=50 each) — THE MODULE 7 RESULT

Both models, post-fix (citation labels + faithfulness prompt + qwen 64k). qwen ran
all 50, no crash.

### The headline table (agent vs single-shot, both models)

| metric | M6 single-shot Haiku | **agent Haiku** | M6 single-shot qwen | **agent qwen** |
|---|---|---|---|---|
| Faithfulness | 4.64 | **4.72** | 3.98 | **4.32** |
| Completeness | 3.40 | **4.04** | 3.02 | **3.38** |
| Refusal acc. | 92% | **98%** | — | 90% |
| avg tools | (1) | 6.9 | (1) | 10.5 |
| gold coverage | (recall@5 52%) | 76.4% | — | 69.6% |
| cost / 50q | $0.21 | $1.28 | $0 | $0 |

**F12 — the agent BEATS single-shot for BOTH models.** Claude: completeness +0.64,
refusal +6, faithfulness +0.08 (the earlier "dip" was the citation artifact —
gone). qwen: completeness +0.36, faithfulness +0.34. The thesis holds: multi-step
+ read-each-source lifts answers over single-shot, on a strong AND a weak model.

**F13 — the citation-artifact fix recovered ~+0.38 faithfulness** (Claude 4.34 →
4.72; literal/synonym → F=5.00). Bigger than expected — the [N]-vs-[chunk_id]
mismatch was silently docking faithfulness across many questions, not just q-023.
(Confounded with the "cite only what passages state" prompt line, but the
literal/synonym jump — categories with little over-claiming — is the citation fix.)

### Per-category (agent)

| category | Claude tools | Claude F/C | qwen tools | qwen F/C |
|---|---|---|---|---|
| literal | 5.4 | 5.00 / 4.11 | 9.9 | 4.33 / 3.44 |
| synonym | 6.6 | 5.00 / 4.63 | 10.0 | 4.38 / 3.50 |
| multi-hop | 9.0 | 4.33 / 4.22 | 16.1 | 4.44 / 3.67 |
| synthesis | 9.6 | 4.63 / 4.00 | 12.3 | 4.00 / 3.25 |
| contradiction | 8.1 | 4.44 / **3.00** | 9.4 | 3.89 / **2.22** |
| out-of-scope | 1.7 | R=86% | 4.0 | R=71% |

### Findings

- **F10 confirmed — Claude self-regulates effort (literal 5.4 → synthesis 9.6);
  qwen does NOT (literal 9.9, multi-hop 16.1).** Over-search is a qwen trait, not
  systemic → no hard tool-cap needed for the strong model.
- **F14 — "small model is its own bottleneck" confirmed at the ANSWER level, and the
  agent WIDENS the gap.** qwen's tool use + coverage (69.6%) are competitive, but it
  converts them to weaker answers: completeness 3.38 vs 4.04. Single-shot Claude–qwen
  completeness gap was 0.38; agent gap is 0.66. The strong model leverages agency
  more — better reasoning compounds when you give it tools.
- **F15 — contradiction completeness is the standout weakness for BOTH (Claude 3.00,
  qwen 2.22) despite high coverage (92%/80%).** The project's HEADLINE goal. It's a
  generation gap, not retrieval: the agents retrieve both sides but don't fully lay
  out each source's specific claims + the contrast to the ideal answer's depth. This
  is the #1 thing to improve next (contradiction-specific prompt / stronger model /
  draft_section tool) — and it's where retrieval already proved it can't help.
- **F16 — qwen refusal regresses (90% vs Claude 98%, M6 92%):** out-of-scope R=71%
  (over-searches → answers), contradiction R=78%. qwen's abstention discipline is
  weak. Claude holds the line (98%, one leak: q-045 gunpowder).
- qwen faster (14s vs 27s — local, no network) and free; finalize rate 68% vs 92%
  (qwen text-terminates / leans on forced-finalize more).

### Verdict
Module 7 delivers: an agent that beats single-shot RAG on both models, with the
strong model showing a clean win on every metric. The two Module 6 open gaps split:
**synthesis improved** (comp 3.75–4.00, up from the floor) but **contradiction
completeness is now the sharpest remaining gap** — generation-bound, not retrieval,
exactly as Module 6 predicted. Agent-specific metrics (tools/coverage/finalize/cost)
all reported per category. Remaining for the module: **Langfuse** (trace tree +
score attachment).

---

## Contradiction prompt fix — TRIED, REJECTED (Haiku, 9 contradiction Qs)

Hypothesis: contradiction completeness (3.00) is "blend-not-contrast" → force an
explicit source-by-source comparison in the prompt (state each source's claim by
name, the divergence, and silences).

Result: **comp 3.00 → 2.89 (flat/noise), faith 4.44 flat, but tools 8.1 → 8.7 and
q-044 ballooned 24s/4-tools → 158s/21-tools.** No gain, more over-search → reverted.

**F17 — the bottleneck is NOT answer structure; it's source-READING discipline.**
Reading the actual answers (all at coverage=100%) exposed the true mechanism:
- q-020 (C=3): wrote a complete, well-cited **Suetonius-only** answer, **omitted
  Plutarch entirely** — though Plutarch's passage was in the searched pool.
- q-044 (C=2): contrasted Plutarch vs **Suetonius**, but the gold contrast is
  Plutarch vs **Caesar's own Civil War** — contrasted the WRONG pair, never read
  Caesar's account.
- q-025 (C=1): read **adjacent** passages (Casca exchange, "Violence is meant!"),
  conflated "what was said during the attack" with "last words".

**Key metric subtlety this exposed:** `goldCoverage` measures what *search*
surfaces (100% here), but completeness depends on what the agent `read_chunk`'d +
cited. The agent surfaces every source's gold but doesn't reliably READ each
source's specific passage — so it answers from one source, or the wrong pair.
A structure prompt can't fix "you never read Plutarch's chunk."
→ Idea parked: add a `read-coverage` metric (gold covered by READ chunks, not just
searched) to separate retrieval from reading cleanly.

**Implication:** contradiction is squarely generation/reading-bound (retrieval is
already at 100% coverage), so the motivated next lever is a **stronger model
(Sonnet)** — more thorough multi-source reading — NOT more prompt engineering of
output structure. A read-discipline prompt ("read + cite ≥2 named sources before
finalizing") is a cheaper thing to try first, but the eval just showed prompt
directives mostly buy over-search here.

### Read-discipline prompt — TRIED, INCONCLUSIVE (the real lesson is about NOISE)

Second hypothesis (data-motivated by F17): force "read_chunk each relevant
source's passage; cite ≥2 named sources; prefer read_chunk over more searches".
Per-question completeness across THREE runs (baseline / structure-fix / read-disc):

| Q | base | struct | read |
|---|---|---|---|
| q-023 | 4 | **2** | 4 |
| q-024 | 2 | **4** | **1** |
| q-025 | 1 | 1 | 2 |
| q-026 | 4 | 3 | 5 |
| q-044 | 3 | 2 | 3 |
| **mean C** | **3.00** | **2.89** | **3.11** |

**F18 — run-to-run VARIANCE dominates the prompt effect at n=9.** The same question
swings ±2 across runs, and the swings are INCONSISTENT (struct-fix took q-024 2→4;
read-disc took it 2→1; q-023 went 4→2→4). The non-deterministic tool path (different
searches → different chunks read → different answer) is louder than any prompt
tweak. read-disc's +0.11 is noise. **At n=9 with ±1–2 per-Q variance we cannot
detect an effect this small** — prompt-tuning is the WRONG tool for the
contradiction gap. Both prompt fixes reverted; baseline contradiction bullet kept
(matches the headline A/B numbers).

**Methodological takeaway (travels):** to move contradiction we need either a
LARGE-enough effect to clear the noise (model change → Sonnet) or VARIANCE
REDUCTION (average k runs, and/or a bigger contradiction set than 9). A single 9-Q
prompt A/B can't resolve sub-0.3 effects. This is itself a Module 7 result: it
bounds what prompt iteration can buy and points the next spend at a model swap.

---

## Langfuse tracing (DONE — module complete)

`lib/langfuse.ts` — `createLangfuseTracer()` implements the existing `lib/tracer.ts`
`Tracer` seam, so ANY client (createClaude/createLlamacpp) auto-emits a trace TREE
with no call-site change (the "~30-line drop-in" tracer.ts promised). Installed
`langfuse@3.38`.

- **Trace shape (one per question):** root `agent:<cat>:<id>` (input=question,
  output=article) → a `generation` per model round-trip (model/usage/cost/latency,
  via onRequest→onResponse pairing) + a `span` per tool call (input/output, via
  onToolCall). Eval scores attached to the trace: faithfulness, completeness,
  refusal_correct, gold_coverage, tool_calls → **Langfuse is now the per-question
  A/B drill-down** (filter by llm/category, click a question to see its research
  path + why it scored what it did).
- **Wired into** `evals/run.ts --agent` (per-question trace + scores + flush) and
  `agent/cli.ts` (one trace per manual run).
- **No-op without keys:** returns `noopTracer` + inert handle when
  LANGFUSE_PUBLIC_KEY/SECRET_KEY are empty → evals run unchanged; lights up the
  moment keys are added. Verified: free local CLI run clean; active path is
  type-checked (langfuse ships TS types).
- **Interface limitation (parked):** the Tracer hooks carry usage/cost/tool-IO but
  NOT prompt/completion TEXT (RequestInfo/ResponseInfo are coarse). So generations
  show model+usage+cost+timing; the ROOT trace has the Q→article text and the tool
  spans have the real research path. Rich per-generation text → extend the Tracer
  interface (deferred; not needed for the drill-down).

**TO ACTIVATE:** put real `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` in `.env`,
then run any `--agent` eval or the CLI → traces + scores appear in the project.

### Local self-hosted Langfuse (done — `infra/langfuse/`)
Shared, machine-global Langfuse stack (reusable across projects), adapted from the
official v3 compose: only the web UI is published (host **3100** — 3000 was taken
by open-webui), Postgres/ClickHouse/Redis/MinIO stay internal (no clash with
ParadeDB:5432). `LANGFUSE_INIT_*` auto-provisions org/project/user + fixed keys on
first boot → zero UI clicking. Login `admin@local.test` / `langfuse-local`; project
`rag-historian`. Run: `docker compose -p langfuse -f infra/langfuse/docker-compose.yml
--env-file infra/langfuse/.env up -d`. Verified: health 200, keys valid, 4 traces
with 5 scores each landed. Project `.env` points at `http://localhost:3100`.

### Sonnet spot-check — "can a smarter model help?" (n=4, generation-bound Qs)
Ran Claude Sonnet on the 4 questions where Haiku failed WITH the facts in context
(coverage ≥75%), via the new `--ids` filter. **F19 — yes, decisively, and ABOVE the
F18 noise floor:**

| Q | Haiku C/F (cov) | Sonnet C/F (cov) |
|---|---|---|
| alexander-lament | 2/5 (100%) | **5/5** (100%) |
| marius-kinship | 2/**2** (100%) | **5/5** (100%) |
| last-words | 1/4 (100%) | 2/5 (100%) |
| pompey-head-reaction | 3/5 (50%) | 2/4 (50%) |
| **mean C** | **2.00** | **3.50 (+1.50)** |

Same tools, same retrieval — only the reasoner changed. Where facts were in context
+ reasoning was the task (q-020, q-028), Sonnet hit 5/5 (q-028 also fixed F2→F5).
Where reading/coverage was the bottleneck (q-044 at 50% cov; q-025 needs enumerating
3 variants), Sonnet didn't rescue it. **This validates the F15/F18 diagnosis: the
contradiction/multi-hop completeness gap is generation-bound, and a MODEL SWAP is the
lever with enough signal — prompt-tuning wasn't.** Sonnet ≈ $0.05/q (~$2.5 for 50).
→ Module 10: full Sonnet A/B, and the q-044/q-025 residual is reading-bound (a
read-discipline mechanism or stronger retrieval, not the writer).

---

## Judge calibration — the contradiction "gap" was partly a weak judge (F20)

Triggered by inspecting Sonnet's q-044 (C=2): the agent gave a correct 3-source
comparison, but the **completeness judge (Haiku) scored it 2 with EMPTY missedFacts**
— penalizing it for including a valid 3rd source (Suetonius, verified real in chunk
1824) and calling it "fabricated", which the completeness judge *cannot* assess (it
never sees chunks). So neither retrieval (100% coverage) nor a wrong ideal — a
**judge bug**.

Fixes applied:
1. **Completeness rubric** (`judge-prompts.ts`): explicit "extra accurate content is
   NEVER penalized; you are not shown sources so do NOT judge fabrication; score only
   coverage of the ideal's facts."
2. **`evals/rejudge-completeness.ts`**: re-score completeness from STORED answers
   (no agent/retrieval re-run, ~$0.1) — isolates rubric/judge changes.
3. **Tracer cost** (`lib/langfuse.ts`): pass `costDetails` (was metadata) so Langfuse
   rolls up trace cost.

**F20 — the Haiku judge has a deviation-penalty bias it WON'T follow instructions out
of; the real fix is a stronger judge.** Re-judging the SAME answers:
- Haiku-judge + fixed rubric: q-044 stayed 2 (missedFacts empty, still penalized the
  3rd source) → **rubric text alone didn't fix it; the judge ignored it.**
- **Sonnet-judge** (Hamel's rule: judge ≥ generator): Sonnet-agent contradiction
  3.00→**4.67** (q-044 2→5, q-025 2→4). Haiku-agent full run, Sonnet-judged:

| category | Haiku-judge | Sonnet-judge | Δ |
|---|---|---|---|
| contradiction | 3.00 | **3.89** | +0.89 (artifact corrected) |
| out-of-scope | 4.43 | 5.00 | +0.57 |
| literal | 4.11 | 4.56 | +0.44 |
| synthesis | 4.00 | **3.50** | −0.50 (Haiku judge too lenient) |
| synonym | 4.63 | 4.38 | −0.25 |
| overall | 4.04 | 4.24 | +0.20 |

A better judge is more DISCRIMINATING both ways: stops over-penalizing thorough
contradiction/OOS answers AND catches real synthesis gaps the weak judge missed.

**Reframed conclusion: SYNTHESIS (3.50, retrieval-bound) is the true weak spot, not
contradiction (3.89).** Matches Module 6's prediction; the contradiction headline was
inflated by an under-powered judge. **Action → Module 8+: make Sonnet (or Opus) the
default judge; re-judge qwen for a fair corrected A/B; consider re-judging faithfulness
similarly. Calibration (hand-score ~10) should have caught this — bank the lesson.**

## Module 7 — DONE. Summary
Roman Research Agent v1: provider-agnostic single-agent loop (5 tools incl.
`search_within_source`), `--agent` eval mode with agent metrics + same judges,
Langfuse trace tree + scores. **Result: agent beats single-shot RAG on both models**
(Claude comp 3.40→4.04 / faith 4.64→4.72 / refusal 92→98%; qwen comp 3.02→3.38).
Open gaps handed to Module 10: contradiction completeness (generation/reading-bound
+ noisy — needs a stronger model / variance reduction), qwen over-search + weaker
refusal. Pitfall honored: single agent + good tools, no multi-agent.
