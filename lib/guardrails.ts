/**
 * lib/guardrails.ts — input/output guardrails (Module 9).
 *
 * The module pitfall is the design constraint: "Don't add guardrails until you can
 * measure their impact. Guardrails that block 5% of legitimate queries are worse
 * than no guardrails." So these are CHEAP, PURE functions (no LLM call, no latency)
 * whose false-positive rate is measured directly against the 50 golden questions +
 * the stored agent answers (evals/guardrails-eval.ts). A guardrail that flags any
 * legitimate query without a commensurate safety gain does not ship.
 *
 * Why no runtime fact-check guardrail here (a deliberate omission): the eval's
 * FAITHFULNESS JUDGE already measures whether an answer is grounded in the
 * consulted chunks. A second runtime "fact-check via Claude" would double cost +
 * latency on every answer to re-measure something we already track offline — and we
 * can't show it changes outputs. Per the pitfall, it stays out until needed.
 */

// ============================================================================
// Input guardrails
// ============================================================================

export interface InputGuardConfig {
  /** Reject questions longer than this (chars). Default 2000 — a research
   *  question, not a pasted document. */
  maxChars?: number;
  /** Extra injection patterns appended to the built-ins. */
  extraPatterns?: Array<{ re: RegExp; label: string }>;
}

export interface InputGuardResult {
  ok: boolean;
  /** Which class of rule fired (only when !ok). */
  rule?: 'length' | 'injection';
  /** Machine label for the specific rule (e.g. 'ignore-instructions'). */
  reason?: string;
}

/**
 * Prompt-injection patterns. Kept SPECIFIC (the full "ignore … instructions"
 * shape, not the bare word "ignore") so ordinary history questions don't trip
 * them — the eval confirms 0 false positives on the golden set.
 */
const INJECTION_PATTERNS: Array<{ re: RegExp; label: string }> = [
  {
    re: /\b(ignore|disregard|forget)\b[^.?!]*\b(previous|prior|above|all|the|your)\b[^.?!]*\b(instructions?|prompts?|rules?|context)\b/i,
    label: 'override-instructions',
  },
  { re: /\b(system|developer)\s+prompt\b/i, label: 'system-prompt-probe' },
  {
    re: /\b(reveal|print|show|repeat|output)\b[^.?!]*\b(your\s+)?(system\s+prompt|instructions|prompt|rules)\b/i,
    label: 'prompt-leak',
  },
  { re: /\byou\s+are\s+now\s+(a|an|the)\b/i, label: 'role-override' },
  {
    // "act as DAN", "respond like an unfiltered model", "with no rules" — the
    // jailbreak keyword carries it; don't require an article (a/an/if) in between.
    re: /\b(act|behave|respond|answer)\s+(as|like)\b[^.?!]*\b(dan|jailbreak|unfiltered|no\s+rules?)\b/i,
    label: 'jailbreak-persona',
  },
  { re: /\b(with\s+)?no\s+(rules?|restrictions?|filters?|limits?)\b/i, label: 'no-rules' },
];

/**
 * Validate a user question BEFORE spending any tokens on it. Returns `{ok:true}`
 * for anything that looks like a genuine research question; otherwise the first
 * rule that fired.
 */
export function validateInput(question: string, config: InputGuardConfig = {}): InputGuardResult {
  const maxChars = config.maxChars ?? 2000;
  const q = question.trim();
  if (q.length === 0) return { ok: false, rule: 'length', reason: 'empty' };
  if (q.length > maxChars)
    return { ok: false, rule: 'length', reason: `exceeds-${maxChars}-chars` };
  for (const p of [...INJECTION_PATTERNS, ...(config.extraPatterns ?? [])]) {
    if (p.re.test(q)) return { ok: false, rule: 'injection', reason: p.label };
  }
  return { ok: true };
}

// ============================================================================
// Output guardrails
// ============================================================================

export interface OutputGuardConfig {
  /** If provided, every cited [id] must be in this set (catches hallucinated
   *  citations). Pass the consulted-chunk ids from the agent run. */
  allowedChunkIds?: Set<number>;
}

export interface OutputGuardResult {
  ok: boolean;
  /** The article abstained (said the corpus doesn't cover it) — a VALID outcome,
   *  exempt from the citation requirement. */
  abstained: boolean;
  /** The article contains ≥1 [chunk_id] citation. */
  cited: boolean;
  /** Specific problems found (empty when ok). */
  issues: string[];
}

/**
 * Abstention phrasings. An abstaining article legitimately has no citations, so
 * the citation check must EXEMPT it — else we'd flag every correct refusal (the
 * exact "blocks legitimate output" failure the pitfall warns about). Tuned against
 * the 7 out-of-scope abstentions in the stored agent run.
 */
const ABSTENTION_PATTERNS: RegExp[] = [
  /\b(do(es)?\s+not|did\s+not|don't|doesn't)\b[^.?!]*\b(contain|cover|include|provide|address|mention|find)\b/i,
  /\bcorpus\b[^.?!]*\b(do(es)?\s+not|don't|doesn't|lacks?|has\s+no)\b/i,
  /\b(outside|beyond|not\s+within)\b[^.?!]*\bscope\b/i,
  /\b(specialized|scoped|limited|focus(ed)?)\b[^.?!]*\b(for|to|on)\b[^.?!]*\b(Caesar|corpus|sources?)\b/i,
  /\bno\s+(information|passages?|sources?|evidence|mention)\b/i,
  /\bclarify\b[^.?!]*\bscope\b/i,
  /\b(cannot|can't|unable\s+to)\b[^.?!]*\b(answer|find|locate|provide)\b/i,
];

const CITATION_RE = /\[(\d+)\]/g;

/**
 * Validate a generated article. An article is OK if it either cites at least one
 * passage OR is a clean abstention; and (when `allowedChunkIds` is supplied) cites
 * only ids it actually consulted.
 */
export function validateOutput(article: string, config: OutputGuardConfig = {}): OutputGuardResult {
  const text = (article ?? '').trim();
  const issues: string[] = [];
  if (text.length === 0) {
    return { ok: false, abstained: false, cited: false, issues: ['empty-article'] };
  }

  const citedIds = [...text.matchAll(CITATION_RE)].map((m) => Number(m[1]));
  const cited = citedIds.length > 0;
  const abstained = ABSTENTION_PATTERNS.some((re) => re.test(text));

  if (!cited && !abstained) issues.push('answered-without-citation');

  if (config.allowedChunkIds && cited) {
    const bad = [...new Set(citedIds)].filter((id) => !config.allowedChunkIds?.has(id));
    if (bad.length > 0) issues.push(`cites-unconsulted-chunks:${bad.join(',')}`);
  }

  return { ok: issues.length === 0, abstained, cited, issues };
}
