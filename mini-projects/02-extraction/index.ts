/**
 * mini-projects/02-extraction — messy text → typed JSON via structured().
 *
 * What this teaches:
 *   - structured() in practice: define a Zod schema, get typed output
 *   - Zero-shot vs few-shot on the SAME inputs (the A/B discipline)
 *   - Why few-shot's example pairs work mechanically (the model continues a pattern)
 *
 * Run:
 *   pnpm dev mini-projects/02-extraction/index.ts
 *   USE_CLAUDE=1 pnpm dev mini-projects/02-extraction/index.ts   # if you want Claude
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
// Schema: the contract the model's output must satisfy.
// `.describe()` text flows into the JSON Schema and DOES affect model behavior —
// keep descriptions accurate, not aspirational.
//
// `.default([])` on the array fields is a deliberate choice for local-model
// robustness. Small models (gpt-oss-20b, Gemma, etc.) often OMIT empty
// required arrays rather than emitting `[]`, which would otherwise blow up
// Zod validation. With `.default([])`:
//   - the field is OPTIONAL in the JSON Schema sent to the model
//   - missing values parse as `[]` automatically
//   - Claude Haiku rarely needs this, but the schema works for both.
// Trade-off: we lose the "force the model to consider all categories" pressure.
// For a higher-stakes pipeline, keep them required and accept the failures
// as a signal that you need a stronger model.
// ---------------------------------------------------------------------------
const MeetingNoteSchema = z.object({
  attendees: z
    .array(z.string())
    .default([])
    .describe('Names of people who attended. Use "the speaker" if a participant is unnamed.'),
  decisions: z
    .array(z.string())
    .default([])
    .describe('Concrete decisions made, one per array entry.'),
  action_items: z
    .array(
      z.object({
        owner: z.string().describe('Who is responsible.'),
        task: z.string().describe('What they will do.'),
        // `.nullish()` = string | null | undefined. `.optional()` alone would
        // reject the explicit `null` that some small local models emit for
        // "not specified" — Gemma 3, Llama 3 do this routinely. nullish is
        // the more cross-model-robust shape for any "may be absent" field.
        due: z.string().nullish().describe('When, if specified. Null or absent if unclear.'),
      }),
    )
    .default([])
    .describe('Concrete tasks assigned, one per array entry. Empty if none.'),
});

type MeetingNote = z.infer<typeof MeetingNoteSchema>;

// ---------------------------------------------------------------------------
// Prompts: v1 zero-shot, v2 few-shot.
// Both use the SAME template variable ({{note}}); the only difference is that
// v2 carries example pairs. That isolation is the whole point of A/B testing.
// ---------------------------------------------------------------------------
const TEMPLATE =
  'Extract the structured meeting information from this note.\n\n' + '<note>{{note}}</note>';

const v1 = definePrompt({
  name: 'v1-zero-shot',
  version: '1.0',
  changelog: 'Initial — instruction + schema only, no examples.',
  template: TEMPLATE,
});

const v2 = definePrompt({
  name: 'v2-few-shot',
  version: '1.0',
  changelog: 'Adds 2 example pairs (varied formality).',
  template: TEMPLATE,
  examples: [
    {
      input: {
        note: '1:1 with Mia. She raised concerns about onboarding. We agreed I would draft a redesign proposal by Thursday.',
      },
      output: JSON.stringify(
        {
          attendees: ['Mia', 'the speaker'],
          decisions: ['Address onboarding concerns via a redesign proposal.'],
          action_items: [
            {
              owner: 'the speaker',
              task: 'Draft an onboarding redesign proposal',
              due: 'Thursday',
            },
          ],
        },
        null,
        2,
      ),
    },
    {
      input: {
        note: 'Eng all-hands. Decided on hiring freeze through Q3. Carlos will draft the comms by tomorrow.',
      },
      output: JSON.stringify(
        {
          attendees: ['Carlos', 'engineering team'],
          decisions: ['Implement hiring freeze through Q3.'],
          action_items: [
            { owner: 'Carlos', task: 'Draft the hiring-freeze comms', due: 'tomorrow' },
          ],
        },
        null,
        2,
      ),
    },
  ],
});

// ---------------------------------------------------------------------------
// Test cases: varied messiness on purpose.
// ---------------------------------------------------------------------------
const cases = [
  {
    label: 'clean meeting',
    vars: {
      note: 'Meeting 2026-05-20: Lev, Maria, and Tom attended. Decided to ship v2 next Friday. Tom will write release notes by 5/22, Maria handles QA pass.',
    },
  },
  {
    label: 'informal',
    vars: {
      note: "Team sync today. Alex pushed back on the timeline. We agreed to defer the API rewrite. Sarah's going to check with legal on the data retention thing — hopefully by EOW.",
    },
  },
  {
    label: 'email-style',
    vars: {
      note: "FYI all — quick update from yesterday's call. Bob and I aligned on the migration approach. Plan: I'll prototype the new schema this week, Bob reviews next Mon. Need sign-off from Dana before we proceed.",
    },
  },
  {
    label: 'low-signal',
    vars: {
      note: "so we just had standup. didn't really decide anything new. james is blocked on the cert issue still. i'll bug ops about it. ravi might pair with him tomorrow.",
    },
  },
];

// ---------------------------------------------------------------------------
// Per-call analytics via the tracer seam from Module 1 (lib/tracer.ts).
// Every call into the wrapper's structured() method fires `onResponse` once;
// we accumulate usage, cost, and latency here. Same pattern would let us plug
// in Langfuse / OTel later by replacing this with a real backend.
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
// Pick a client. Defaults to LM Studio (free); set USE_CLAUDE=1 for Claude.
// The client is constructed ONCE so the tracer accumulates across every call.
// ---------------------------------------------------------------------------
const useClaude = process.env.USE_CLAUDE === '1' && Boolean(process.env.ANTHROPIC_API_KEY);

async function runStructured(
  messages: { role: 'user' | 'assistant'; content: string }[],
): Promise<MeetingNote> {
  if (useClaude) {
    const result = await claude.structured({ messages, schema: MeetingNoteSchema });
    return result.data;
  }
  const result = await lms.structured({ messages, schema: MeetingNoteSchema });
  return result.data;
}

const claude = useClaude
  ? createClaude({ defaultModel: CLAUDE_MODELS.haiku, tracer: statsTracer })
  : (null as never);
// Local runtime is LM Studio by default; flip LOCAL_LLM_PROVIDER=ollama to swap.
const local = useClaude
  ? null
  : createLocalLLM({
      lmstudio: { defaultModel: LM_STUDIO_MODELS.gptOss20b },
      ollama: { defaultModel: OLLAMA_MODELS.gptOss20b },
      tracer: statsTracer,
    });
const lms = local?.client ?? (null as never);

const providerLabel = useClaude ? `Claude (${CLAUDE_MODELS.haiku})` : (local?.label ?? 'local');
console.log(`Provider: ${providerLabel}\n`);

const rows = await abTest([v1, v2], cases, runStructured);
console.log(formatAbTest(rows));

// ---------------------------------------------------------------------------
// Summary across all calls. With 2 prompts × 4 cases, expect 8 calls.
// LM Studio reports zero cost (local inference); the token counts still tell
// you how expensive these prompts would be on a paid provider.
// ---------------------------------------------------------------------------
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
