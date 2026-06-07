/**
 * roman-research/agent/register-prompt.ts — tag the agent system prompt in
 * Langfuse (Module 9, Slice 5). Run after editing AGENT_SYSTEM_PROMPT to push a
 * new version under the `production` label; rollback is moving that label in the
 * Langfuse UI (or re-running with --label).
 *
 *   pnpm dev roman-research/agent/register-prompt.ts                 # → production
 *   pnpm dev roman-research/agent/register-prompt.ts --label=staging # → staging
 *
 * No-op (clean) when Langfuse keys are absent.
 */
import 'dotenv/config';

import { flushLangfuse, getManagedPrompt, registerPrompt } from '../../lib/index.ts';
import { AGENT_PROMPT_NAME, AGENT_SYSTEM_PROMPT } from './index.ts';

const labelArg = process.argv.find((a) => a.startsWith('--label='))?.split('=')[1];
const label = labelArg ?? 'production';

const { created, version } = await registerPrompt(AGENT_PROMPT_NAME, AGENT_SYSTEM_PROMPT, {
  labels: [label],
});

if (version === null) {
  console.log('Langfuse disabled (no keys) — nothing registered. The in-code const stays in use.');
} else if (created) {
  console.log(`Registered "${AGENT_PROMPT_NAME}" v${version} under label "${label}".`);
} else {
  console.log(
    `"${AGENT_PROMPT_NAME}" v${version} already current under "${label}" — no new version.`,
  );
}

// Read it back exactly as runAgent will, to confirm the round-trip.
const fetched = await getManagedPrompt(AGENT_PROMPT_NAME, AGENT_SYSTEM_PROMPT, { label });
console.log(
  `Fetch-back: source=${fetched.source} version=${fetched.version ?? '(fallback)'} ` +
    `matches=${fetched.text === AGENT_SYSTEM_PROMPT}`,
);

await flushLangfuse();
