/**
 * mini-projects/05-version-compare — three versions of the same prompt, diffed.
 *
 * What this teaches:
 *   - The discipline of "never change two things at once" made concrete.
 *   - How small wording changes shift output style dramatically (concrete vs vague,
 *     constraints, negative-space instructions).
 *   - The pattern Module 5 will scale: same inputs through multiple prompts,
 *     scored / compared, drive iteration.
 *
 * Run:
 *   pnpm dev mini-projects/05-version-compare/index.ts
 *   USE_CLAUDE=1 pnpm dev mini-projects/05-version-compare/index.ts
 */
import 'dotenv/config';

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
// Three prompt versions on the same task (summarization). The version field
// is bumped each time the template changes — this is the version log Module 5
// will lean on when prompts evolve from evals failures.
// ---------------------------------------------------------------------------
const v1Vague = definePrompt({
  name: 'v1-vague',
  version: '1.0',
  changelog: 'Initial — minimal instruction.',
  template: 'Summarize this:\n\n{{text}}',
});

const v2Structured = definePrompt({
  name: 'v2-structured',
  version: '2.0',
  changelog: 'Specify format (3 bullets). Concrete > vague.',
  template: 'Summarize this text in exactly 3 bullet points:\n\n{{text}}',
});

const v3Constrained = definePrompt({
  name: 'v3-constrained',
  version: '3.0',
  changelog:
    'Add hard length limit, focus instruction, negative-space rule (no disclaimers / hedging). Wrap input in <text> for parsing clarity.',
  template:
    'Summarize the text below in exactly 3 bullet points (max 15 words each). ' +
    'Focus on actionable / time-sensitive facts. ' +
    'Do not include disclaimers, hedging language, or restated context.\n\n' +
    '<text>\n{{text}}\n</text>',
});

// ---------------------------------------------------------------------------
// Three test inputs — different domains so we can see whether each prompt
// transfers, or whether a wording works for one shape and fails for another.
// ---------------------------------------------------------------------------
const cases = [
  {
    label: 'news article',
    vars: {
      text:
        'OpenAI announced GPT-5 today, claiming improved reasoning at lower latency. ' +
        'The model is available via API immediately, with pricing roughly 30% below GPT-4 levels. ' +
        'CEO Sam Altman emphasized that the model represents a step-change in capabilities, ' +
        'particularly for agentic workflows. Critics note that the headline benchmarks have ' +
        'not been independently verified.',
    },
  },
  {
    label: 'release notes',
    vars: {
      text:
        'The new bundler can produce ESM-only output and tree-shake unused exports. ' +
        "Build times are about 40% faster on the team's monorepo. " +
        'Migration requires renaming entry.js to entry.mjs and adjusting the build script. ' +
        "Known issue: source maps don't work in dev mode yet.",
    },
  },
  {
    label: 'internal email',
    vars: {
      text:
        'Hi team — quick heads up that the Q3 planning offsite has been moved from June 15 to June 22 ' +
        'due to a venue conflict. Same location, same agenda. Please update your calendars and let ' +
        "Sarah know if the new date doesn't work.",
    },
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
// Client + chat runner.
// ---------------------------------------------------------------------------
const useClaude = process.env.USE_CLAUDE === '1' && Boolean(process.env.ANTHROPIC_API_KEY);

const claude = useClaude
  ? createClaude({ defaultModel: CLAUDE_MODELS.haiku, tracer: statsTracer })
  : (null as never);
// Local runtime defaults to LM Studio; LOCAL_LLM_PROVIDER=ollama flips it.
const local = useClaude
  ? null
  : createLocalLLM({
      lmstudio: { defaultModel: LM_STUDIO_MODELS.gptOss20b },
      ollama: { defaultModel: OLLAMA_MODELS.gptOss20b },
      tracer: statsTracer,
    });
const lms = local?.client ?? (null as never);

async function runChat(
  messages: { role: 'user' | 'assistant'; content: string }[],
): Promise<string> {
  if (useClaude) {
    const result = await claude.chat({ messages });
    return result.text;
  }
  const result = await lms.chat({ messages });
  return result.text;
}

const providerLabel = useClaude ? `Claude (${CLAUDE_MODELS.haiku})` : (local?.label ?? 'local');
console.log(`Provider: ${providerLabel}\n`);

// Print the version log so the diff has context.
console.log('=== Version log ===');
for (const p of [v1Vague, v2Structured, v3Constrained]) {
  console.log(`[${p.name} v${p.version}] ${p.changelog ?? '(no changelog)'}`);
}
console.log();

const rows = await abTest([v1Vague, v2Structured, v3Constrained], cases, runChat);
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
