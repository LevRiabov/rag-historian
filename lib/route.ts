/**
 * lib/route.ts — model routing classifier (Module 9).
 *
 * The lever: Module 7's F19 showed Sonnet decisively rescues the
 * generation-bound categories (contradiction/multi-hop/synthesis) where Haiku
 * fails WITH the facts in context, while Haiku is fine on simple lookups. So a
 * router that sends only the hard questions to Sonnet buys most of Sonnet's
 * quality at a fraction of the cost.
 *
 * The RISK (the whole reason this is eval-gated): a misrouting classifier could
 * degrade quality while chasing cost. The asymmetry is what makes a sane policy
 * safe — and shapes the prompt:
 *   - misroute UP   (simple → Sonnet): only wastes money. Sonnet answers simple
 *     questions at least as well, and still refuses out-of-scope ones.
 *   - misroute DOWN (reasoning → Haiku): LOSES quality — the exact failure F19
 *     diagnosed.
 * Therefore the classifier is told: WHEN IN DOUBT, choose 'reasoning' (escalate).
 * The only error we truly care about minimizing is reasoning→simple.
 *
 * This module is the classifier + the tier→model mapping. Whether to actually
 * ROUTE in production is decided by the eval (evals/route-eval.ts measures the
 * classifier against the gold categories; the agent A/B measures the quality/cost
 * delta) — never shipped blind.
 */
import { z } from 'zod';
import { createClaude } from './claude.ts';
import { CLAUDE_MODELS, type ClaudeModel } from './cost.ts';
import { noopTracer, type Tracer } from './tracer.ts';
import type { Cost } from './types.ts';

/** The two routing tiers. 'simple' → cheap model; 'reasoning' → smart model. */
export type RouteTier = 'simple' | 'reasoning';

/** Classifier output, schema-forced via structured(). */
export const RouteDecisionSchema = z.object({
  complexity: z
    .enum(['simple', 'reasoning'])
    .describe(
      "'simple' = one fact answerable from a single passage, or clearly outside a " +
        "Caesar-history corpus; 'reasoning' = needs combining/contrasting multiple " +
        'passages, tracing a chain of events, or synthesizing a development.',
    ),
  rationale: z.string().describe('One short clause justifying the choice.'),
});
export type RouteDecision = z.infer<typeof RouteDecisionSchema>;

export interface RouteResult {
  tier: RouteTier;
  /** The model the tier maps to — pass to runAgent / createClaude. */
  model: ClaudeModel;
  rationale: string;
  /** What the classification call itself cost (cheap-model tokens). */
  cost: Cost;
}

export interface RouterConfig {
  /** Model for 'simple' questions. Default Haiku. */
  cheapModel?: ClaudeModel;
  /** Model for 'reasoning' questions. Default Sonnet (the F19 lever). */
  smartModel?: ClaudeModel;
  /** Model that DOES the classification. Default Haiku (cheap + fast). */
  classifierModel?: ClaudeModel;
  apiKey?: string;
  tracer?: Tracer;
}

const CLASSIFIER_SYSTEM = `You are a routing classifier for a Roman-history research agent whose corpus is four primary sources on Julius Caesar (Caesar's own Gallic War & Civil War, Plutarch, Suetonius). Given a user question, decide whether answering it well needs a "reasoning" model or a cheaper "simple" model.

simple — a single concrete fact retrievable from ONE passage (who/what/when/where, a definition, one event detail), OR a question clearly OUTSIDE this corpus (it should be refused).
reasoning — requires COMBINING facts across multiple passages, CONTRASTING sources that disagree, tracing a multi-step chain ("how did X lead to Y"), or SYNTHESIZING a development over time.

Bias: when genuinely unsure, choose "reasoning". Routing a hard question to the cheap model loses answer quality; routing an easy one to the strong model only costs a little money. Err toward "reasoning".`;

/**
 * Build a router. `route(question)` runs ONE cheap classification call and
 * returns the tier, the model to use, the rationale, and the classification cost.
 */
export function createRouter(config: RouterConfig = {}) {
  const cheapModel = config.cheapModel ?? CLAUDE_MODELS.haiku;
  const smartModel = config.smartModel ?? CLAUDE_MODELS.sonnet;
  const classifierModel = config.classifierModel ?? CLAUDE_MODELS.haiku;
  const tracer = config.tracer ?? noopTracer;
  const client = createClaude({ apiKey: config.apiKey, defaultModel: classifierModel, tracer });

  async function route(question: string): Promise<RouteResult> {
    const { data, cost } = await client.structured({
      schema: RouteDecisionSchema,
      system: CLASSIFIER_SYSTEM,
      messages: [{ role: 'user', content: question }],
      // Deterministic routing — same question should route the same way.
      temperature: 0,
      maxTokens: 256,
    });
    return {
      tier: data.complexity,
      model: data.complexity === 'reasoning' ? smartModel : cheapModel,
      rationale: data.rationale,
      cost,
    };
  }

  return { route, cheapModel, smartModel };
}
