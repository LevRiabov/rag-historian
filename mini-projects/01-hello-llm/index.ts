/**
 * mini-projects/01-hello-llm — your first end-to-end LLM call.
 *
 * Module 1 deliverable. Demonstrates the wrapper:
 *   - Defines a calculator tool with Zod (one schema, three uses: API contract,
 *     runtime validation, typed `execute` argument).
 *   - Runs the same multi-step math question through Claude AND LM Studio.
 *   - Logs each tool call as it happens.
 *   - Prints a cost summary at the end.
 *
 * Run:
 *   pnpm dev mini-projects/01-hello-llm/index.ts
 *
 * Requirements (each path is optional — the script skips what's not configured):
 *   - ANTHROPIC_API_KEY in .env  → enables Claude
 *   - LM Studio running locally  → enables LM Studio (default localhost:1234)
 *
 * See ./README.md for what to expect, things to play with, and where to look next.
 */
import 'dotenv/config';
import { z } from 'zod';

import {
  CLAUDE_MODELS,
  createClaude,
  createLMStudio,
  defineTool,
  formatCost,
  type ToolStep,
} from '../../lib/index.ts';

// ---------------------------------------------------------------------------
// Tool definition
// ---------------------------------------------------------------------------
// The Zod schema is the contract: the model sees the JSON Schema derived from
// it, our `execute` receives the parsed and validated input. Change the schema
// in ONE place and both halves stay in sync.
const calculator = defineTool({
  name: 'calculate',
  description: 'Perform a basic arithmetic operation. Use this for any math.',
  schema: z.object({
    operation: z.enum(['add', 'subtract', 'multiply', 'divide']),
    a: z.number(),
    b: z.number(),
  }),
  execute: ({ operation, a, b }) => {
    switch (operation) {
      case 'add':
        return a + b;
      case 'subtract':
        return a - b;
      case 'multiply':
        return a * b;
      case 'divide':
        if (b === 0) throw new Error('Division by zero');
        return a / b;
    }
  },
});

const question = 'What is 47 × 219, and then what is that result minus 1337?';

// ---------------------------------------------------------------------------
// Shared logging helper
// ---------------------------------------------------------------------------
function logStep(step: ToolStep): void {
  const input = JSON.stringify(step.toolInput);
  console.log(`  [iter ${step.iteration}] ${step.toolName}(${input}) → ${step.toolOutput}`);
}

// ---------------------------------------------------------------------------
// Run against Claude
// ---------------------------------------------------------------------------
async function runClaude(): Promise<void> {
  if (!process.env.ANTHROPIC_API_KEY) {
    console.log('— Skipping Claude (ANTHROPIC_API_KEY not set in .env) —\n');
    return;
  }
  // Haiku for this demo — cheapest model, plenty smart for a calculator loop.
  // Bump to CLAUDE_MODELS.sonnet or .opus if you want to see how reasoning scales.
  const model = CLAUDE_MODELS.haiku;
  console.log(`=== Claude (${model}) ===`);
  console.log(`Q: ${question}\n`);

  const claude = createClaude({ defaultModel: model });
  const result = await claude.runTools({
    messages: [{ role: 'user', content: question }],
    tools: [calculator],
    onStep: logStep,
  });

  console.log(`\nA: ${result.text}`);
  console.log(`Stop: ${result.stop}`);
  console.log(
    `Tokens: in=${result.usage.inputTokens}, out=${result.usage.outputTokens}` +
      (result.usage.cacheReadTokens ? `, cache_read=${result.usage.cacheReadTokens}` : ''),
  );
  console.log(`Cost: ${formatCost(result.cost)}\n`);
}

// ---------------------------------------------------------------------------
// Run against LM Studio
// ---------------------------------------------------------------------------
async function runLMStudio(): Promise<void> {
  console.log('=== LM Studio (local) ===');
  console.log(`Q: ${question}\n`);

  // Defaults to gpt-oss-20b; override with LM_STUDIO_MODEL env var if you've
  // loaded a different model (e.g., 'gemma-3-27b').
  const lms = createLMStudio({
    defaultModel: process.env.LM_STUDIO_MODEL ?? 'gpt-oss-20b',
  });

  try {
    const result = await lms.runTools({
      messages: [{ role: 'user', content: question }],
      tools: [calculator],
      onStep: logStep,
    });

    console.log(`\nA: ${result.text}`);
    console.log(`Stop: ${result.stop}`);
    console.log(`Tokens: in=${result.usage.inputTokens}, out=${result.usage.outputTokens}`);
    console.log(`Cost: ${formatCost(result.cost)} (LM Studio = local = free)\n`);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.log(`Skipping LM Studio — could not connect or model unavailable (${msg})\n`);
  }
}

await runClaude();
await runLMStudio();
