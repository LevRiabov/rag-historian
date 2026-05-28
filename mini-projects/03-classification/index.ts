/**
 * mini-projects/03-classification — multi-class labeling with CoT vs answer-first.
 *
 * What this teaches:
 *   - Three prompt strategies on the same classification task:
 *       v1 answer-first   — forbid reasoning; force a single label
 *       v2 cot            — reason briefly, then answer
 *       v3 few-shot       — five examples, one per category
 *   - When CoT helps (hard / ambiguous cases) vs hurts (clear cases, latency, cost)
 *   - How few-shot calibrates BOTH the labels chosen AND the reasoning style
 *
 * Run:
 *   pnpm dev mini-projects/03-classification/index.ts
 *   USE_CLAUDE=1 pnpm dev mini-projects/03-classification/index.ts
 */
import 'dotenv/config';
import { z } from 'zod';

import {
  abTest,
  addCost,
  addUsage,
  CLAUDE_MODELS,
  type Cost,
  createClaude,
  createLocalLLM,
  definePrompt,
  formatAbTest,
  formatCost,
  LM_STUDIO_MODELS,
  OLLAMA_MODELS,
  type Tracer,
  type Usage,
} from '../../lib/index.ts';

// ---------------------------------------------------------------------------
// Schema. Reasoning is optional — v1's prompt forbids it; v2/v3 ask for it.
// Schema accommodating both lets us A/B without changing the contract.
// ---------------------------------------------------------------------------
const CATEGORIES = ['billing', 'technical', 'account', 'feature_request', 'other'] as const;

// Property ORDER matters here. `category` is listed FIRST on purpose:
//   - Ollama's native /api/chat compiles the schema into a GBNF grammar that
//     enforces property emission order. The enum constraint is only reliably
//     applied to the FIRST property — when a long-string field precedes the
//     enum, the constraint gets lost mid-stream and Gemma will happily emit
//     `"category": "login_and_access"` (copied from the user prompt).
//   - LM Studio's GBNF doesn't have this quirk, but matching the order is
//     harmless on its end.
// Lesson: when constraining a small model on a hard field, put that field
// FIRST in the schema AND emit it first in any few-shot examples.
const TicketLabel = z.object({
  // `z.preprocess` lowercases strings BEFORE enum validation. Necessary because:
  //   - LM Studio's llama.cpp uses grammar-constrained sampling (GBNF) — model
  //     literally can't emit "Billing" because the grammar only permits lowercase.
  //   - Ollama doesn't grammar-constrain by default — model freely emits
  //     "Billing" (title-cased nouns from training) and the schema rejects it.
  // Same model, different runtime strictness. Schema-side normalization survives both.
  category: z.preprocess((v) => (typeof v === 'string' ? v.toLowerCase() : v), z.enum(CATEGORIES)),
  // `.nullish()` rather than `.optional()` — small local models often emit
  // `"reasoning": null` when the prompt says not to provide reasoning. Strict
  // `.optional()` would reject that; nullish accepts both null and undefined.
  reasoning: z
    .string()
    .nullish()
    .describe('One short sentence justifying the category. Null or omitted if not asked.'),
});

type Label = z.infer<typeof TicketLabel>;

// ---------------------------------------------------------------------------
// Three prompt variants — same template variable, different instructions.
// ---------------------------------------------------------------------------
const v1AnswerFirst = definePrompt({
  name: 'v1-answer-first',
  version: '1.2',
  changelog:
    'Replace "output ONLY the category" with explicit JSON priming. Gemma on Ollama treats prompts ' +
    'that fight the schema as a license to emit plain text — soft grammar. Explicitly mentioning ' +
    'JSON keeps the schema engaged while still suppressing reasoning (the answer-first behaviour ' +
    'we want to test).',
  template:
    'Classify the support ticket into one category. Return JSON with the category field; omit reasoning.\n\n' +
    '<ticket>{{ticket}}</ticket>',
});

const v2Cot = definePrompt({
  name: 'v2-cot',
  version: '1.0',
  changelog: 'Ask for one-sentence reasoning before the category.',
  template:
    'Classify the support ticket into one category. First, briefly state your reasoning, then the category.\n\n' +
    '<ticket>{{ticket}}</ticket>',
});

const v3FewShot = definePrompt({
  name: 'v3-few-shot',
  version: '1.0',
  changelog: 'Five examples, one per category, with the CoT instruction.',
  template:
    'Classify the support ticket into one category. State a brief reasoning, then the category.\n\n' +
    '<ticket>{{ticket}}</ticket>',
  // Output property order is `category` first, then `reasoning` — matches the
  // schema order (see TicketLabel above) so Gemma's GBNF grammar keeps the
  // enum constraint engaged through to the end of the object.
  examples: [
    {
      input: { ticket: 'My credit card was double-charged for last month.' },
      output: JSON.stringify({
        category: 'billing',
        reasoning: 'Mentions a duplicate payment / charge issue.',
      }),
    },
    {
      input: { ticket: 'The app crashes every time I open settings.' },
      output: JSON.stringify({
        category: 'technical',
        reasoning: 'Reports a software malfunction.',
      }),
    },
    {
      input: { ticket: 'I need to change my email address on file.' },
      output: JSON.stringify({
        category: 'account',
        reasoning: 'Account profile change request.',
      }),
    },
    {
      input: { ticket: 'Please add a dark mode option.' },
      output: JSON.stringify({
        category: 'feature_request',
        reasoning: 'Requests a new capability.',
      }),
    },
    {
      input: { ticket: 'Hi there.' },
      output: JSON.stringify({
        category: 'other',
        reasoning: 'No clear support request expressed.',
      }),
    },
  ],
});

// ---------------------------------------------------------------------------
// Test cases — mix clear and ambiguous so we can see where prompts diverge.
// ---------------------------------------------------------------------------
const cases = [
  {
    label: 'clear billing',
    vars: { ticket: 'I was charged twice for my subscription this month.' },
  },
  { label: 'clear technical', vars: { ticket: 'App crashes when uploading a CSV over 100MB.' } },
  {
    label: 'clear account',
    vars: { ticket: 'I forgot my password and the reset link is broken.' },
  },
  {
    label: 'clear feature request',
    vars: { ticket: 'Could you add export-to-PDF? My team needs it for compliance reports.' },
  },
  { label: 'noise', vars: { ticket: 'Hi' } },
  {
    label: 'ambiguous billing/feature',
    vars: {
      ticket: 'Why am I being charged for the trial period? Can you add a way to extend trials?',
    },
  },
  {
    label: 'ambiguous technical/account',
    vars: { ticket: "Login button doesn't work on Safari 18 with my new account." },
  },
  {
    label: 'multi-issue',
    vars: { ticket: 'Need to update my billing info AND report that the new dashboard is buggy.' },
  },
];

// ---------------------------------------------------------------------------
// Analytics — same tracer pattern as 02-extraction.
// ---------------------------------------------------------------------------
let totalUsage: Usage = { inputTokens: 0, outputTokens: 0 };
let totalCost: Cost = {
  inputUSD: 0,
  outputUSD: 0,
  cacheCreationUSD: 0,
  cacheReadUSD: 0,
  totalUSD: 0,
};
let callCount = 0;
let totalLatencyMs = 0;

const statsTracer: Tracer = {
  onResponse: ({ usage, cost, latencyMs }) => {
    totalUsage = addUsage(totalUsage, usage);
    totalCost = addCost(totalCost, cost);
    totalLatencyMs += latencyMs;
    callCount += 1;
  },
};

// ---------------------------------------------------------------------------
// Client selection. Add `reasoning: 'medium'` (or 'low' / 'high') to the call
// opts inside `run()` to enable thinking — gpt-oss-20b honors all 3 levels;
// other models may ignore. Try it on the ambiguous cases and see if reasoning
// changes the verdict (often it doesn't for classification — anchoring effect).
// ---------------------------------------------------------------------------
const useClaude = process.env.USE_CLAUDE === '1' && Boolean(process.env.ANTHROPIC_API_KEY);

// Gemma 4 here on purpose — small model, useful capability-floor probe. Each
// runtime needs the model spelled in its native naming. LOCAL_LLM_PROVIDER
// picks which one runs (default: lmstudio).
const claude = useClaude
  ? createClaude({ defaultModel: CLAUDE_MODELS.haiku, tracer: statsTracer })
  : (null as never);
const local = useClaude
  ? null
  : createLocalLLM({
      lmstudio: { defaultModel: LM_STUDIO_MODELS.gemma4_4b },
      ollama: { defaultModel: OLLAMA_MODELS.gemma4_e2b },
      tracer: statsTracer,
    });
const lms = local?.client ?? (null as never);
const localModel = local?.model ?? '';

// Toggle this to experiment with reasoning. NOTE: behavior varies by runtime
// AND model — gpt-oss honors levels, Gemma collapses to a boolean toggle, and
// Ollama's native path silently disables `think` to keep grammar engaged (see
// lib/ollama.ts comments). Watch output token counts in the Summary block to
// confirm thinking actually fired.
const REASONING: 'low' | 'medium' | 'high' | boolean | undefined = 'medium';

async function run(messages: { role: 'user' | 'assistant'; content: string }[]): Promise<Label> {
  if (useClaude) {
    const result = await claude.structured({ messages, schema: TicketLabel, reasoning: REASONING });
    return result.data;
  }
  // temperature: 0 is load-bearing for Gemma 4 on Ollama. Its model defaults
  // (temperature: 1, top_p: 0.95) are loose enough that the grammar mask gets
  // sampled past — the model emits out-of-enum values like `account_access`.
  // At T=0 the grammar holds. Also: classification should be deterministic
  // anyway — variation between runs is noise, not signal.
  const result = await lms.structured({
    messages,
    schema: TicketLabel,
    reasoning: REASONING,
    temperature: 0,
  });
  // Log what the runtime echoed back — if `raw.model` differs from what we
  // requested, the runtime silently routed your call. Also handy for confirming
  // reasoning kicked in (you'd see output tokens differ vs no-reasoning runs).
  const raw = result.raw as { model?: string };
  if (raw.model && raw.model !== localModel) {
    console.log(`[ROUTED] requested=${localModel} actual=${raw.model}`);
  }
  return result.data;
}

const providerLabel = useClaude ? `Claude (${CLAUDE_MODELS.haiku})` : (local?.label ?? 'local');
console.log(`Provider: ${providerLabel}\n`);

const rows = await abTest([v1AnswerFirst, v2Cot, v3FewShot], cases, run);
console.log(formatAbTest(rows));

console.log(`\n=== Summary across ${callCount} calls ===`);
console.log(
  `Tokens: in=${totalUsage.inputTokens}, out=${totalUsage.outputTokens}, total=${totalUsage.inputTokens + totalUsage.outputTokens}`,
);
console.log(`Cost:   ${formatCost(totalCost)}`);
if (callCount > 0) {
  console.log(
    `Latency: total=${totalLatencyMs}ms, avg=${Math.round(totalLatencyMs / callCount)}ms per call`,
  );
}
