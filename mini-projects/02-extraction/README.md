# 02 — Extraction

Messy text → typed JSON via `structured()`. Compare zero-shot vs few-shot on the same inputs.

## A note on local-model robustness

Two schema choices in this project are about surviving small local models:

- **`.default([])` on the array fields** — small models (gpt-oss-20b, Gemma 3, Llama 3) sometimes OMIT empty required arrays rather than emitting `[]`. The default makes the field optional in the generated JSON Schema and fills `[]` on parse if the model skipped it.
- **`.nullish()` (not `.optional()`) on `due`** — `.optional()` accepts `string | undefined` only. Small models often emit explicit `null` for "not specified." `.nullish()` accepts both `null` and `undefined`, so either spelling parses cleanly.

The trade-off: permissive schemas hide the signal "is this empty because there's nothing, or because the model missed it?" For higher-stakes pipelines, keep fields required/strict and treat validation failures as evidence you need a stronger model. This is exactly the robustness gap Module 6's comparison matrix will quantify, and Module 9's fallback chains will exploit (retry on Claude when local validation fails). See [`notes/prompting-patterns.md`](../../notes/prompting-patterns.md#schema-design-for-cross-model-robustness) for the general pattern.

## What it demonstrates

- **`structured()` in practice** — define a Zod schema, get back fully typed output. The schema's `.describe()` text flows into the JSON Schema sent to the model and DOES affect output quality.
- **Zero-shot vs few-shot, isolated.** Both prompts use the exact same template; v2 only differs by having two example input/output pairs. This is the A/B discipline — change one thing.
- **Why few-shot works mechanically.** The model is an autoregressive next-token predictor; once it sees `<note>...</note> → {JSON}` twice, the third time it continues the pattern. Few-shot isn't "teaching the model" — it's biasing the continuation.
- **`abTest` + `formatAbTest`** in real use — the Module-5 eval harness in miniature.

## What it solves

Four meeting-note inputs of varying messiness — clean structured notes, casual sync recaps, email-style updates, low-signal standups. Goal: extract `{ attendees, decisions, action_items[] }` reliably across the range.

## Run

Default (Ollama):
```
pnpm dev mini-projects/02-extraction/index.ts
```

With Claude — PowerShell:
```powershell
$env:USE_CLAUDE='1'; pnpm dev mini-projects/02-extraction/index.ts
```

With Claude — bash/zsh:
```bash
USE_CLAUDE=1 pnpm dev mini-projects/02-extraction/index.ts
```

Requires Ollama running at `http://localhost:11434` with a tool-capable model pulled (`ollama pull gpt-oss:20b`), OR `ANTHROPIC_API_KEY` in `.env` if using Claude.

## What to look for

Compare v1 vs v2 outputs row by row:

- **Naming conventions.** v1 might use "Unknown" or "speaker" or omit when the speaker is implicit; v2 should converge on "the speaker" because that's what the examples did. **The model copies the examples' style choices.**
- **Action item granularity.** v1 might lump "Sarah will check with legal" into a single decision; v2 will more reliably extract it as an action item with `owner: "Sarah"`.
- **Empty-array handling.** On the low-signal note, v1 may invent decisions to fill the array; v2 should leave it empty (one example shows that being acceptable).
- **`due` field.** v1 inconsistently fills it; v2 follows the example's convention of inferring relative dates like "tomorrow" / "Thursday".

The takeaway: **the examples don't just teach the SHAPE (the schema already does that). They teach STYLE and EDGE CASES.**

## Things to play with

- **Drop one example from v2.** See if quality regresses noticeably or if one example was carrying most of the weight.
- **Swap an example** for a deliberately wrong / inconsistent one. Watch the model copy the mistake — examples are programs.
- **Loosen the `.describe()` text** in the schema (replace the detailed descriptions with `"x"`). Run again. The descriptions matter more than people expect.
- **Force a tighter schema** — make `due` required (not optional). Watch what the model fabricates when the note doesn't actually specify a deadline.
- **Compare Ollama vs Claude side by side.** Run with and without `USE_CLAUDE=1`. Local models often miss subtle context that frontier models catch — exactly the gap Module 6 measures.

## Related code

- [`lib/prompts.ts`](../../lib/prompts.ts) — `definePrompt`, `abTest`, `formatAbTest`
- [`lib/claude.ts`](../../lib/claude.ts) (`structured` method) — tool-use-as-output pattern
- [`lib/lmstudio.ts`](../../lib/lmstudio.ts) (`structured` method) — OpenAI-shape equivalent
