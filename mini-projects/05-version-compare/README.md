# 05 — Version Compare

Three versions of the same summarization prompt, run against the same three inputs. The discipline of prompt iteration made concrete.

## What it demonstrates

- **One change per version.** v1→v2 adds a format constraint (3 bullets). v2→v3 adds a length limit + focus instruction + negative-space rule. Never two changes per version — this is the *only* way to attribute output differences to wording.
- **`changelog` is data, not commentary.** Each version's `changelog` field describes the WHY of the edit. The version log printed at the top of the run is the audit trail.
- **Concrete > vague.** "Summarize this" produces wildly variable lengths and styles. "3 bullet points, max 15 words each, no hedging" produces tight, comparable output every time. Models hold to measurable constraints better than aesthetic ones.
- **Negative-space rules earn their keep.** "Do not include disclaimers or hedging language" eliminates whole categories of wasted tokens that no positive instruction would catch.

## What it solves

Summarize three short texts (a news article paragraph, software release notes, an internal email) with three prompt versions. See how output diverges as constraints tighten.

## Run

Default (Ollama):
```
pnpm dev mini-projects/05-version-compare/index.ts
```

With Claude — PowerShell:
```powershell
$env:USE_CLAUDE='1'; pnpm dev mini-projects/05-version-compare/index.ts
```

With Claude — bash/zsh:
```bash
USE_CLAUDE=1 pnpm dev mini-projects/05-version-compare/index.ts
```

## What to look for

Reading the output left to right (v1 → v2 → v3):

- **Length and shape variance.** v1's outputs differ wildly in length and format across the three inputs. v2 enforces shape (3 bullets) but bullet length varies. v3 looks similar across all three inputs — measurable constraints flatten the variance.
- **Disclaimers and meta-commentary in v1/v2.** Watch for phrases like "Here is a summary of the text:" or "It's worth noting that...". v3 explicitly forbids these and the model complies (mostly — count any leaks).
- **Information density.** v3 is forced to be selective by the word limit. The information density per token goes up. This is the same trade-off you'll see in Module 6 — tighter constraints, more reliable output, occasionally at the cost of nuance.
- **The internal email** is the hardest case. Short input → not much to summarize. v1 will likely just paraphrase; v2 will pad to 3 bullets; v3 will look the most natural because the constraints discourage padding. Sometimes shorter prompts work better on shorter inputs — but that's a *category-specific* finding, not a universal rule.

## Things to play with

- **Add a v4** that demands first-person voice ("you should..."). One change from v3. Run the comparison again.
- **Reduce the word limit** in v3 from 15 to 8 per bullet. Watch information loss.
- **Remove the negative-space rule** from v3. Watch hedging language reappear.
- **Drop the `<text>` wrapper** in v3. Compare output cleanliness — XML wrapping isn't always strictly necessary on short, structured input, but it's cheap insurance.
- **Add a malicious test input** with "ignore previous instructions" embedded. Watch whether v3's explicit constraints help vs v1's openness — connects this back to mini-project 04.

## Related code

- [`lib/prompts.ts`](../../lib/prompts.ts) — `definePrompt`, `abTest`, `formatAbTest`
- [`notes/prompting-patterns.md`](../../notes/prompting-patterns.md) — the patterns this exercises
