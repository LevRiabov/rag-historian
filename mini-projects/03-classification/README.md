# 03 — Classification

Multi-class labeling with three prompt strategies: answer-first vs CoT vs few-shot. Same schema, same inputs.

## What it demonstrates

- **CoT vs answer-first is a real trade-off**, not a free win:
  - Answer-first: faster, cheaper, can be more accurate on clear cases (no chance for the model to talk itself out of the right answer).
  - CoT: helps on ambiguous / multi-issue tickets where the model needs to weigh evidence. Costs more output tokens.
- **Few-shot calibrates labels AND reasoning style.** The model copies the *shape* of the example reasonings, not just the category choices.
- **Schemas with optional fields** support multiple prompt strategies against a single schema — `reasoning` is optional, so v1's "no reasoning" instruction is allowed.
- **Where prompts diverge tells you what to fix.** Two prompts disagreeing on the same case = the case is ambiguous in some way you should address (better examples, or better category definitions).

## What it solves

Eight test tickets — four clear, four ambiguous — classified into `{ billing, technical, account, feature_request, other }`.

## Run

Default (Ollama):
```
pnpm dev mini-projects/03-classification/index.ts
```

With Claude — PowerShell:
```powershell
$env:USE_CLAUDE='1'; pnpm dev mini-projects/03-classification/index.ts
```

With Claude — bash/zsh:
```bash
USE_CLAUDE=1 pnpm dev mini-projects/03-classification/index.ts
```

## What to look for

For each ticket, three labels appear side by side:

- **On clear cases**, all three should agree. If they don't, your category definitions are ambiguous (not the prompt's fault).
- **On `ambiguous billing/feature`** ("Why am I being charged for the trial? Can you add a way to extend trials?"):
  - v1 (answer-first) commits to one label. Often `billing` because it's the first issue mentioned.
  - v2 (CoT) sometimes catches the dual issue and still picks one — its reasoning will mention the tension.
  - v3 (few-shot) tends to be the most deliberate; the reasoning style is calibrated by the examples.
- **On the `multi-issue` case**, no prompt can be "right" — they have to pick one. This is a CATEGORY DESIGN problem (should you have a `multi-issue` category? Or allow multi-label?). Prompt iteration won't fix it.
- **The `noise` case ("Hi")** — v1 might still pick a real category; v2/v3 should land on `other`. Watch the reasoning of v2 vs v3: does v3's "no clear support request" justification leak into v2's reasoning style?

## Things to play with

- **Lower temperature** on the run function (`createClaude({ ... })` and `createOllama({ ... })` accept it via call opts). At T=0 you should see more consistency. Reset T=1 and run twice — see the noise floor.
- **Remove `other` from the enum.** Watch what happens to the "Hi" ticket — model is forced to invent a fit. Demonstrates why having an escape-hatch category is important.
- **Replace one few-shot example** with a deliberately wrong label. Watch v3 propagate the mistake — examples are programs, errors and all.
- **Add `confidence: number`** to the schema (0–1). See how v1/v2/v3 differ in calibration. Most models systematically overconfident — this is why log probs (where exposed) often beat self-reported confidence.

## Related code

- [`lib/prompts.ts`](../../lib/prompts.ts) — A/B harness
- [`lib/claude.ts`](../../lib/claude.ts) / [`lib/lmstudio.ts`](../../lib/lmstudio.ts) — `structured()` implementations
