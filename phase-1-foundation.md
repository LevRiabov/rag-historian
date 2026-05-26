# AI Engineering Foundation — Phase 1 Plan

> **Goal of Phase 1:** Build a real, eval-driven AI engineering foundation in TypeScript, with one shippable case study (the Roman Research Agent) and your loroplanner work properly packaged. By the end you can confidently take freelance LLM/agent work.

---

## How to use this document

- Modules are **sequential** — each builds on prior ones, don't skip ahead
- Hours are estimates; your actual pace will vary (you said 2–8 hrs/day is normal)
- Every module has: **Objectives → Resources → Hands-on → Deliverable → Pitfalls**
- The Roman Research Agent is the through-line across modules — you build it incrementally
- **Evals are the differentiator** — they appear from Module 5 onward and never go away
- Track your progress at the bottom; tick off deliverables as you go

---

## The through-line

Across all modules, you're building one project: an **agentic research assistant over a Roman history corpus from Project Gutenberg**, with progressive technique stacking and rigorous eval comparison. The final case study is the marketable artifact. Each module either adds a capability to this project or builds a skill that goes into it.

---

## Phase 1 at a glance

| # | Module | Hours | What you'll have after |
|---|--------|-------|------------------------|
| 0 | Environment setup | 4 | Working dev env, API keys, Langfuse, Ollama, repo |
| 1 | LLM API foundations | 15 | "Hello LLM" with streaming, tools, caching, traced |
| 2 | Prompting & structured outputs | 10 | Prompting pattern library, JSON-mode mastery |
| 3 | Embeddings & vector storage | 8 | pgvector running, BGE-M3 local, OpenAI embeddings |
| 4 | Naive RAG build | 15 | Roman corpus ingested, baseline RAG working |
| 5 | **Evals — critical module** | 20 | Golden eval set, Promptfoo running, LLM-as-judge |
| 6 | Advanced RAG techniques | 25 | Hybrid + reranking + contextual retrieval, measured |
| 7 | Agent fundamentals | 20 | Roman research agent v1 (single-agent) |
| 8 | MCP server design | 15 | Roman corpus exposed as MCP server |
| 9 | Production patterns | 12 | Caching, guardrails, fallbacks, prompt versioning |
| 10 | Final integration | 25 | Roman research agent shipped, case study written |

**Total: ~185 hours.** At 20 hrs/week: ~9–10 weeks. At 30 hrs/week: ~6 weeks. No deadlines — work the pace that fits.

**Parallel ongoing work (~25h spread throughout):**
- Loroplanner case study writeup (10h)
- Personal site stub (6h)
- Daily reading: 1 blog post or cookbook example per session (~9h cumulative)

---

## Module 0: Environment & Account Setup (~4h)

### Objectives
- All accounts, tools, and dev environment ready
- Repo structure established for the Roman project
- Observability hooked up from day one

### Setup checklist
- Node 20+, pnpm or npm
- Postgres 16+ with pgvector extension installed (`CREATE EXTENSION vector;`)
- Anthropic API key
- OpenAI API key (for embeddings + comparison)
- Cohere API key (free tier — for reranking later)
- Langfuse account (cloud free tier — self-host later in Phase 3)
- Ollama installed with GPU drivers verified on your 5070 Ti
- Pull initial models: `ollama pull qwen2.5:14b`, `ollama pull llama3.3:8b`, `ollama pull bge-m3`
- New git repo: `roman-research` with TS + Anthropic SDK starter

### Deliverable
A `hello.ts` that calls Claude, returns a response, and shows up as a trace in Langfuse. Commit + push.

### Pitfall
Don't spend 4 hours optimizing your repo structure. Get something working, refactor later. The skills are the point, not the boilerplate.

---

## Module 1: LLM API Foundations (~15h)

### Objectives
- Master the Anthropic SDK and the request/response model
- Understand streaming, tool use, structured outputs, prompt caching at the API level
- Have a reusable "client wrapper" you'll use in every subsequent module
- Instrument everything with Langfuse from the start

### Resources
- **Finish Chip Huyen's *AI Engineering* book** (you're partway)
- **Anthropic Cookbook** — work through these in order:
  - `multimodal/` (skim)
  - `tool_use/` — read all examples carefully
  - `extended_thinking/` — understand when to use
  - `prompt_caching/` — critical for cost
  - `misc/` — structured outputs patterns
- **Anthropic API docs:** "Build with Claude" sections on Messages API, Tool Use, Streaming
- **Anthropic "Building Effective Agents"** post — short, read twice
- **OpenAI Cookbook** — selectively: function calling, structured outputs, batch API
- **Simon Willison's blog** — subscribe, read one post/day from now on
- **Vercel AI SDK docs** — useful TS abstraction patterns

### Hands-on
Build `lib/llm.ts` — a thin wrapper around the Anthropic SDK with:
- Streaming support
- Tool call execution loop
- Structured output helper
- Automatic Langfuse tracing
- Cost tracking (tokens × rates)
- Retry logic with exponential backoff

Build `mini-projects/01-hello-llm.ts`:
- Takes a question
- Calls Claude with a simple tool (e.g., calculator)
- Streams response
- Logs trace to Langfuse
- Prints cost summary

### Deliverable
`lib/llm.ts` (reusable across all later modules) + one working mini-project. Trace visible in Langfuse with input, output, tokens, cost.

### Pitfall
Don't reach for LangChain.js or Vercel AI SDK to skip building this wrapper. Building it yourself in ~100 lines is what teaches you what's actually happening. Use libraries later by choice, not because you skipped the fundamentals.

---

## Module 2: Prompting & Structured Outputs (~10h)

### Objectives
- Treat prompting as engineering, not magic
- Know the standard patterns: zero-shot, few-shot, chain-of-thought, role prompting, output formatting
- Master structured outputs (JSON mode, tool-use-as-output, XML tags)
- Build a small library of prompt templates with versioning

### Resources
- **Anthropic's prompt engineering docs** — full read
- **"The Prompt Report"** (Schulhoff et al., 2024 — arxiv) — skim, reference
- **Anthropic's prompt library** — read 5–10 examples
- **Eugene Yan: "Patterns for Building LLM-based Systems & Products"** — read fully
- **Anthropic Cookbook: structured output techniques**

### Hands-on
Build `lib/prompts.ts`:
- Prompt template system with variable substitution
- Prompt versioning (each prompt has a version string, change log)
- Test runner that compares two prompt versions on the same inputs

Practice exercises:
- Convert messy text into structured JSON (5 input variations, measure success rate manually)
- Build a classifier prompt for ticket categories with few-shot examples
- Build a chain-of-thought reasoner for math/logic problems

### Deliverable
`lib/prompts.ts` + 3 prompt templates you've iterated on with version history. Notes in `notes/prompting-patterns.md` capturing what you learned.

### Pitfall
Don't write prompts as one giant blob. Structure them: role / context / task / format / examples / constraints. This makes them maintainable and version-able.

---

## Module 3: Embeddings & Vector Storage (~8h)

### Objectives
- Conceptual understanding of embeddings as an API user (no math, just intuition)
- pgvector mastery for production use
- Know how to swap embedding models without rewriting your stack
- Run BGE-M3 locally for multilingual capability

### Resources
- **pgvector README** (full)
- **HuggingFace MTEB leaderboard** — understand how to pick models
- **OpenAI embeddings docs** — text-embedding-3-small/large tradeoffs
- **BGE-M3 model card** on HuggingFace
- **Eugene Yan: "Real-world Recommendation Systems"** — embeddings beyond text

### Hands-on
Build `lib/embeddings.ts`:
- Unified interface for OpenAI + Ollama (BGE-M3) embeddings
- Batch embedding (100+ chunks per request) for efficiency
- Database schema: `chunks(id, text, embedding, source, metadata)` with HNSW index

Mini-exercise:
- Embed 100 short sentences with both OpenAI and BGE-M3
- Verify cosine similarity makes intuitive sense (similar sentences → high similarity)
- Measure latency and cost difference

### Deliverable
Working pgvector setup, `lib/embeddings.ts` with two backends, notes on which to use when.

### Pitfall
You cannot mix embedding models across queries and stored chunks — must use the same model for both. Locking your stack into the wrong embedding model is a costly rebuild later.

---

## Module 4: Naive RAG Build (~15h)

### Objectives
- Build the first end-to-end RAG over the Roman history corpus
- Establish a baseline you'll improve against
- Feel the failure modes firsthand

### Resources
- **Anthropic Cookbook: RAG section** — read the whole thing
- **Pinecone Learn** (despite the name, vendor-neutral content) — chunking strategies
- **Eugene Yan: "Patterns for Building LLM-based Systems"** — RAG section

### Roman corpus selection
Download from Project Gutenberg (all plain .txt):
- Gibbon's *Decline and Fall of the Roman Empire* (Volumes 1–3 sufficient initially)
- Plutarch's *Lives* (Dryden translation) — focus on Roman lives
- Suetonius's *The Twelve Caesars*
- Tacitus's *Annals* and *Histories*
- Caesar's *Gallic War* and *Civil War*

**Don't ingest everything at once.** Start with Gibbon Vol 1 + Suetonius (~300 pages). Add more later as quality permits.

### Hands-on
Build `roman-research/ingest.ts`:
1. Load txt files from `corpus/`
2. Split each by chapter (regex on chapter markers)
3. Chunk each chapter into ~500-token pieces with 50-token overlap
4. Embed each chunk (start with OpenAI text-embedding-3-small)
5. Store in pgvector with metadata: source book, chapter, position

Build `roman-research/query.ts`:
1. Embed the user's question
2. Cosine similarity search, top 5 chunks
3. Stuff chunks into prompt with citations
4. Generate answer with Claude
5. Return answer + source citations

Test with 5 hand-written questions across difficulty levels.

### Deliverable
Working naive RAG. Roman corpus chunks in pgvector. ~5 example questions answered with citations. Output committed to repo.

### Pitfall
Don't over-engineer the chunker on the first pass. Get something working end-to-end first, then improve the parts that fail your eval set in Module 5/6.

---

## Module 5: Evals — Critical Module (~20h)

> **This is the most important module in Phase 1.** Most "AI engineers" skip evals. Doing them well is your competitive edge.

### Objectives
- Build a golden eval set with question taxonomy (5 categories)
- Implement retrieval metrics (recall@k) and generation metrics (faithfulness, completeness)
- Set up Promptfoo for regression testing
- Master LLM-as-judge methodology — pitfalls, calibration, agreement with humans
- Establish the eval-driven dev workflow you'll use for every project from now on

### Resources
- **Hamel Husain: "Your AI Product Needs Evals"** — read 3 times, it's that important
- **Hamel Husain: "Creating a LLM-as-a-Judge That Drives Business Results"** — practical workflow
- **Eugene Yan: "Evals are all you need"** — core philosophy
- **Eugene Yan: "Practical Evaluation Framework for LLM Apps"**
- **Anthropic's eval docs** (in the docs site)
- **Promptfoo docs** — full read, run the tutorials
- **Optional paid course:** Hamel + Shreya's "AI Evals for Engineers & PMs" on Maven if you want to triple down (worth it for the differentiation, but not required)

### Hands-on

**Step 1 — Build the golden eval set (8h)**
30–50 question/answer pairs across 5 categories:
- **Literal lookup** (e.g., *"When did Caesar cross the Rubicon?"*) — 10 questions
- **Synonym mismatch** (e.g., *"How did Caesar gain absolute power?"* where docs say "dictator perpetuo") — 10 questions
- **Multi-hop** (e.g., *"How many years passed between the death of Sulla and Caesar's first consulship?"*) — 10 questions
- **Synthesis** (e.g., *"What were the political tensions between Pompey and Cicero after the civil war?"*) — 5 questions
- **Out-of-scope** (e.g., *"What did Caesar think of TikTok?"* — should refuse) — 5 questions

For each Q/A pair, record:
- The question
- The ideal answer
- Which chunk(s) contain the answer (chunk IDs) — this enables retrieval scoring

Generate with Claude, then human-review every one. Don't skip the review.

**Step 2 — Build retrieval metrics (4h)**
- **Recall@k** — given a question, did the top-k retrieved chunks contain the gold chunk(s)?
- **MRR (Mean Reciprocal Rank)** — at what position did the first relevant chunk appear?
- Score your Module 4 naive RAG. Establish baseline numbers.

**Step 3 — Build generation metrics (4h)**
- **Faithfulness** — is the answer grounded in the retrieved chunks? (LLM-as-judge with strict rubric)
- **Completeness** — does the answer cover what's in the gold answer? (LLM-as-judge)
- **Refusal correctness** — for out-of-scope questions, did the system refuse?

LLM-as-judge prompt design:
- Give the judge: question, ideal answer, candidate answer, retrieved chunks
- Ask for scores (1–5) per criterion + brief justification
- Calibrate by spot-checking ~10 judgments manually

**Step 4 — Promptfoo regression suite (4h)**
- Convert golden set into Promptfoo YAML
- Set up CI: every prompt or retrieval change runs the full eval
- Track results in `evals/results/` with timestamps

### Deliverable
- `evals/golden-set.json` with 30+ questions across 5 categories
- `evals/metrics.ts` with all four scoring functions
- Promptfoo config running the suite
- Baseline scores recorded for naive RAG from Module 4
- `notes/eval-philosophy.md` capturing what you learned

### Pitfall
LLM-as-judge can be biased and miscalibrated. **Always spot-check judgments manually until you trust them.** If the judge disagrees with you frequently, fix the judge prompt before trusting its scores.

---

## Module 6: Advanced RAG Techniques (~25h)

### Objectives
- Stack retrieval techniques one at a time, measuring the delta of each with your eval set
- Master hybrid search, reranking, contextual retrieval, query rewriting
- Produce the comparison table that becomes the case study centerpiece

### Resources
- **Anthropic: "Introducing Contextual Retrieval"** (2024 blog post) — the technique you'll implement
- **Cohere Rerank docs** — when and how to use
- **HyDE paper** (Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels") — read abstract + intro
- **BGE-reranker docs** for local reranking option

### Hands-on
For each technique below: implement, score with your eval set, record results, write a one-paragraph analysis.

**6.1 Structure-aware chunking (3h)**
- Replace fixed-token chunks with chapter/section-aware splitting
- Measure: did recall@5 improve? On which question types?

**6.2 Hybrid search (5h)**
- Add Postgres full-text search (BM25-style) alongside vector search
- Implement Reciprocal Rank Fusion (RRF) to combine results
- Measure: improvement on questions with named entities (e.g., "the Battle of Pharsalus")

**6.3 Reranking (4h)**
- Cohere Rerank v3 OR BGE-reranker-v2-m3 (local)
- Pull top 20 from hybrid search, rerank to top 5
- Measure: faithfulness improvement (less noise in context)

**6.4 Contextual Retrieval (8h)**
- For each chunk, generate a short context line using Claude with **prompt caching on the full document**
- Prepend the context line to the chunk before re-embedding
- Re-ingest the corpus with contextual embeddings
- Measure: recall improvement (expect 20–35% based on Anthropic's results)

**6.5 Query rewriting (3h)**
- Implement HyDE: LLM generates hypothetical answer, embed that for retrieval
- Implement query expansion: LLM generates 3 query variations, search all
- Measure on synonym-mismatch questions specifically

**6.6 The comparison table (2h)**
Build the final matrix that goes in your case study:

| Technique stack | Recall@5 | Faithfulness | Latency | Cost/query |
|---|---|---|---|---|
| Naive (Module 4 baseline) | | | | |
| + structure-aware chunking | | | | |
| + hybrid search | | | | |
| + reranking | | | | |
| + contextual retrieval | | | | |
| Final stack with local generation (Qwen 14B) | | | | |
| Final stack with frontier (Claude Opus) | | | | |

### Deliverable
Comparison table fully populated with real numbers. Each row reproducible from your repo. The numbers are the case study.

### Pitfall
Don't change two things at once. Each technique gets isolated A/B measurement against the previous best. Otherwise you can't attribute the gains.

---

## Module 7: Agent Fundamentals (~20h)

### Objectives
- Build the Roman Research Agent v1 (single-agent loop)
- Master tool design, error handling, circuit breakers
- Understand ReAct, plan-execute, reflection patterns
- Self-implement; no LangGraph

### Resources
- **Anthropic: "Building Effective Agents"** — re-read in this context
- **Lilian Weng: "LLM Powered Autonomous Agents"** — landscape overview
- **Anthropic Cookbook: tool_use examples** — re-read with agent loops in mind
- **Cognition: "Don't Build Multi-Agents"** (2024) — read for the warning

### Hands-on
Build `roman-research/agent.ts`:

**Tool design:**
- `search_corpus(query, top_k)` → returns chunk IDs + titles + snippets
- `read_chunk(chunk_id)` → returns full chunk text
- `list_sources_consulted()` → state-aware list of what's been read
- `draft_section(topic, points)` → returns a draft of one article section
- `finalize(article)` → returns the final article, ends the loop

**Agent loop:**
- Max iterations (start at 30, tune from there)
- Streaming tool results back into the conversation
- Cost cap (kill if cost exceeds threshold)
- Graceful error recovery (tool failures become observations the model can react to)
- Full Langfuse tracing of every step

**System prompt design:**
Structure: role / task / tools available / methodology hint / output format / constraints. Iterate on it with your eval set.

**Eval extension:**
Add agent-specific metrics:
- Average tool calls per question
- Average tokens per question
- Cost per article
- Success rate (LLM-as-judge on final article quality)

### Deliverable
Roman Research Agent v1 working end-to-end. Can answer a research question with a multi-paragraph article + citations. Full traces in Langfuse. Eval suite reports agent-specific metrics.

### Pitfall
**Don't reach for multi-agent.** Your loroplanner experience already taught you this. Single agent + good tool design beats multi-agent for this task.

---

## Module 8: MCP Server Design (~15h)

### Objectives
- Build a real MCP server (you've shipped MCP client work; this is the gap)
- Understand resources vs tools vs prompts in MCP
- Expose the Roman corpus as an MCP server for use with Claude/Cursor/IDEs

### Resources
- **MCP official documentation** (modelcontextprotocol.io) — full read
- **MCP TypeScript SDK** docs and examples
- **Anthropic's MCP example servers** repo on GitHub
- **MCP Inspector tool** for debugging

### Hands-on
Build `roman-research/mcp-server/`:

**Tools exposed:**
- `search_roman_corpus(query, top_k)` — same as your agent tool
- `read_roman_chunk(chunk_id)` — same
- `list_roman_sources()` — list all books in corpus with chapter counts
- `cite_passage(chunk_id)` — get formatted citation string

**Resources exposed:**
- Each book as a resource (URI: `roman://gibbon/vol1`, etc.)
- Chapter listings as nested resources

**Prompts exposed:**
- "research_topic" — pre-built prompt for using the corpus
- "summarize_event" — pre-built prompt for event summarization

Test by connecting Claude Desktop / Cursor to your local MCP server and querying it interactively.

### Deliverable
Working MCP server. Documented README with installation instructions. Demo: screenshot or video of Claude Desktop using your server to answer a Roman history question.

### Pitfall
MCP debugging is rough without the Inspector. Install it day one and use it constantly.

---

## Module 9: Production Patterns (~12h)

### Objectives
- Internalize the production patterns that separate seniors from juniors
- Apply them to the Roman agent
- Build the operational layer (caching, fallbacks, guardrails, versioning)

### Topics & resources

**Prompt caching (3h)** — Anthropic docs on cache_control. Use it on system prompts and long context (Module 6.4 contextual retrieval ingest is where it matters most).

**Model routing (2h)** — Cheap model (Haiku) for simple tool calls, smart model (Opus) for synthesis. Implement a router in `lib/llm.ts`.

**Retries & fallback chains (2h)** — If Anthropic fails, fall back to GPT-5 then to local Qwen. Use exponential backoff. Anthropic SDK has built-in retry; extend for cross-provider fallback.

**Rate limit handling (1h)** — Token bucket pattern, queueing, exponential backoff on 429s.

**Guardrails (2h)**
- Input validation (length, prompt injection patterns)
- Output validation (format, content filters, fact-check via separate Claude call)
- Anthropic's `safety` content blocks
- Open source: NeMo Guardrails, Llama Guard — read overviews

**Prompt versioning workflow (2h)** — Tag every prompt in Langfuse, attach eval results to versions, rollback is one click.

### Hands-on
Apply each pattern to the Roman agent. Verify in Langfuse:
- Cache hits visible on system prompt
- Fallback triggers when you intentionally break Anthropic key
- Cost drops visibly from caching

### Deliverable
Roman agent now production-grade: cached, with fallbacks, versioned prompts, basic guardrails. Document in `notes/production-patterns.md`.

### Pitfall
Don't add guardrails until you can measure their impact on real outputs. Guardrails that block 5% of legitimate queries are worse than no guardrails. Eval before and after every guardrail you add.

---

## Module 10: Final Integration — Roman Research Agent v2 (~25h)

### Objectives
- Integrate every technique from Modules 1–9 into the final agent
- Run the full comparison matrix one more time on the integrated stack
- Write the case study

### Hands-on

**10.1 Stack integration (10h)**
Final agent stack:
- Hybrid search + reranking + contextual retrieval (Module 6)
- Single-agent loop with full tool set (Module 7)
- MCP server option (Module 8)
- Prompt caching, fallbacks, versioning, guardrails (Module 9)
- Full Langfuse tracing throughout

**10.2 Multi-model comparison runs (8h)**
Run the full eval set against:
- Claude Opus 4.7 (frontier)
- GPT-5 via OpenAI API (frontier alternative)
- Qwen 2.5 14B local (your 5070 Ti — best mid-size local)
- Llama 3.3 8B local (fast baseline)
- Llama 3.3 70B rented on RunPod for one eval run (~$3 total)

Record all numbers in the comparison table.

**10.3 Case study writeup (5h)**
Structure (2000–3000 words):
- The problem you solved
- Why retrieval over a historical corpus is harder than typical RAG
- Architecture diagram
- Each technique with measured impact
- Final comparison table with all models
- Cost and latency analysis
- Lessons learned and tradeoffs
- Live demo link (deploy somewhere — Vercel + Postgres on Neon, or Fly.io)

**10.4 Public repo polish (2h)**
- Clean README with architecture diagram, quick start, demo gif
- Eval results published as part of repo
- Permissive license

### Deliverable
- **Roman Research Agent live demo URL**
- **Public GitHub repo** with full source, evals, results
- **Case study writeup** ready for your personal site
- **Comparison table** with real numbers across 5 models × 6 technique stacks

This is your strongest portfolio piece. It alone is worth more than most "AI engineer" bootcamp certificates.

---

## Parallel ongoing work (~25h spread across modules)

### Loroplanner case study writeup (10h, ideal during Modules 1–4)
2500–3500 words:
- Problem context (travel content needing multilingual generation)
- Architecture (MCP-driven scraping + multilingual content pipeline)
- Why MCP was the right choice
- Eval methodology you used (or would use in retrospect)
- Cost per article, scaling considerations
- Lessons learned

Don't expose code if you don't want to. Architecture diagrams + decisions + numbers are enough.

### Personal site stub (6h, ideal during Modules 5–7)
Next.js, deploy to Vercel. Five pages:
- Home — who you are, what you build, headline
- Case Studies — list (will have 1, then 2 by end of Phase 1)
- About — background, current focus
- Contact — email, LinkedIn, GitHub
- Notes / blog (optional)

Plain, clean, fast. No design ambition needed for the stub — improve later.

### Daily reading (~9h cumulative across Phase 1)
- Subscribe to: Hamel Husain, Eugene Yan, Simon Willison, Latent Space
- Aim: one blog post or cookbook example per working session
- Capture interesting patterns in `notes/reading.md`

---

## End-of-Phase 1 deliverables checklist

By the end of Phase 1, you should have:

- [ ] `lib/` utility modules (llm, prompts, embeddings) reusable across projects
- [ ] Roman Research Agent v2 deployed live with a public URL
- [ ] Public GitHub repo with full source and evals
- [ ] Eval golden set + Promptfoo regression suite
- [ ] MCP server for the Roman corpus
- [ ] Case study #1: Roman Research Agent (2000–3000 words)
- [ ] Case study #2: Loroplanner (2500–3500 words)
- [ ] Personal site stubbed and deployed with both case studies linked
- [ ] Comparison matrix with real numbers across techniques + models
- [ ] Production patterns demonstrated in code (caching, fallbacks, guardrails, versioning)
- [ ] Notes folder capturing what you learned (prompting patterns, eval philosophy, production patterns, reading)

That's the foundation. Phase 2 will be 1–2 more projects (potentially pivoting visual depending on your call) + platform applications.

---

## Common pitfalls across all modules

1. **Skipping evals because they feel like overhead.** They're the differentiator. Without them, every claim in your case study is unverifiable.
2. **Reaching for frameworks (LangChain, LangGraph, CrewAI) to skip understanding.** Build it once yourself. Use frameworks later by choice, not avoidance.
3. **Changing two things at once.** Isolated A/B with the eval set is the only way to know what helped.
4. **Over-engineering the first version.** Naive version first, then improve based on eval failures. Premature optimization is especially costly in AI work.
5. **Polishing forever.** Ship at "good enough for measurable improvement," not at perfection. Case study quality > demo polish.
6. **Treating Claude/GPT as oracles.** Spot-check LLM-as-judge outputs constantly. Both can be confidently wrong.
7. **Ignoring cost.** Track $ per query from Module 1 onward. Production work cares about cost; case studies that include cost analysis are far more credible.

---

## Resources hub (consolidated)

### Must-read blogs (subscribe today)
- Hamel Husain (evals, practical workflows)
- Eugene Yan (patterns, evals, retrieval)
- Simon Willison (daily updates, practical tips)
- Anthropic blog (techniques, model updates)

### Books
- Chip Huyen, *AI Engineering* (finish it)
- *Designing Machine Learning Systems* (Chip Huyen, optional)

### Cookbooks
- Anthropic Cookbook (github.com/anthropics/anthropic-cookbook)
- OpenAI Cookbook (selectively)

### Tools
- Langfuse (observability)
- Promptfoo (evals)
- Cursor / Claude Code (dev environment)
- MCP Inspector (debugging)
- Ollama (local models)

### Optional paid
- Hamel + Shreya's evals course on Maven (~$2K) — only if you want to triple down on evals as your selling point

---

## Where to next (Phase 2 preview)

Once Phase 1 is solid (or even before final polish), we'll write Phase 2. Likely scope:
- Project A or visual-AI equivalent — second case study
- Phase 2 will also revisit direction: by then you'll have real signal on whether agent work or visual AI fits you better
- Phase 3 plans the platform applications (Toptal, Pangea, A.Team)
- Phase 4 is first clients, rate evolution, niche refinement

Don't write Phase 2 in your head yet. Get through Module 5 (evals) and Module 6 (advanced RAG with measured comparison) — that's when the real direction becomes clearer.

---

## Progress tracking

Tick as you complete:

- [ ] Module 0: Setup
- [ ] Module 1: LLM API Foundations
- [ ] Module 2: Prompting & Structured Outputs
- [ ] Module 3: Embeddings & Vector Storage
- [ ] Module 4: Naive RAG Build
- [ ] Module 5: Evals
- [ ] Module 6: Advanced RAG Techniques
- [ ] Module 7: Agent Fundamentals
- [ ] Module 8: MCP Server Design
- [ ] Module 9: Production Patterns
- [ ] Module 10: Final Integration

Parallel:
- [ ] Loroplanner case study
- [ ] Personal site stub
- [ ] Daily reading habit established

End of Phase 1 check: all boxes above ticked, all deliverables in the checklist done, comparison matrix populated. When you're at that point, we'll plan Phase 2.
