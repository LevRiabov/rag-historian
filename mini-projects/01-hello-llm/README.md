# 01 — Hello LLM

Your first end-to-end LLM call. **Module 1 deliverable.**

## What it demonstrates

- Defining a tool with Zod — one schema, three uses (API contract, runtime validation, typed executor)
- The agentic tool loop (`runTools`) — model calls a tool, sees the result, decides next move, until it has the final answer
- Running the *same* code against **two providers** — Claude (cloud) and LM Studio (local) — to see the SDK differences side by side
- Token-level cost accounting (Anthropic) and zero-cost local inference (LM Studio)

## What it solves

A two-step math question: *"What is 47 × 219, and then what is that result minus 1337?"*. The question forces the model into at least two tool calls (each arithmetic op is one call), so we see the loop run more than once and pair `tool_use` blocks with `tool_result` blocks correctly.

## Run

From the repo root:

```bash
pnpm dev mini-projects/01-hello-llm/index.ts
```

Each path is optional — the script skips whichever isn't configured:

- **Claude** runs if `ANTHROPIC_API_KEY` is set in `.env`. Defaults to `CLAUDE_MODELS.haiku` (cheapest). Change the `model` constant in [index.ts](index.ts) to test Sonnet or Opus.
- **LM Studio** runs if a server is reachable at `http://localhost:1234`. The default model name is `gpt-oss-20b`; override via `LM_STUDIO_MODEL` env var to match the model you've loaded.

## Expected output

```
=== Claude (claude-haiku-4-5-20251001) ===
Q: What is 47 × 219, and then what is that result minus 1337?

  [iter 1] calculate({"operation":"multiply","a":47,"b":219}) → 10293
  [iter 2] calculate({"operation":"subtract","a":10293,"b":1337}) → 8956

A: 47 × 219 = 10,293. Subtracting 1,337 gives **8,956**.
Stop: final_answer
Tokens: in=812, out=63
Cost: $0.001127 (in $0.000812, out $0.000315)
```

Exact numbers vary by model, prompt cache state, and the model's text style — but the shape (two tool calls, then a final answer) should match. The right answer is **8,956**.

## Things to play with

- **Swap the model.** `CLAUDE_MODELS.sonnet` or `.opus` in the Claude block. Watch tokens and cost scale.
- **Add a second tool.** A trivial async one like `get_current_time()` returning `new Date().toISOString()` shows the async path actually traverse. The wrapper already supports async (`execute` can return `Promise<unknown>`).
- **Make the calculator fail on some inputs** — throw, or return an `"ERROR: ..."` string. Watch the model adapt on the next turn. This is the "errors as strings" pattern.
- **Try a tool-shy model on LM Studio.** Gemma 3 27B sometimes ignores tools and answers from memory; gpt-oss-20b is tool-trained and follows the schema. Comparing the two is the lesson.
- **Tighten the description.** Change `description: 'Perform a basic arithmetic operation. Use this for any math.'` to something vague like `'Compute things'` — see whether the model still calls it reliably.

## Related code

- [`lib/claude.ts`](../../lib/claude.ts) — Anthropic-side tool loop
- [`lib/lmstudio.ts`](../../lib/lmstudio.ts) — OpenAI-compatible tool loop
- [`lib/tools.ts`](../../lib/tools.ts) — `defineTool`, Zod → SDK conversion, runtime validation
- [`lib/cost.ts`](../../lib/cost.ts) — pricing table, cost calculation
