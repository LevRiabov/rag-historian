/**
 * lib/prompts.ts — prompt template + versioning + A/B comparison.
 *
 * Why a typed prompt template (not just string interpolation):
 *   1. `{{var}}` placeholders are extracted into a TypeScript literal-union at
 *      compile time — forgetting to pass one is a type error, not a runtime
 *      "undefined" silently injected into the prompt.
 *   2. Prompts get versioning + changelog as DATA, not as comments. Lets us
 *      compare versions deliberately (Module 5 will subsume with full evals).
 *   3. Few-shot examples are first-class — `examples` is a typed field, not
 *      something stringified into the template.
 *
 * Why XML tag helpers: Anthropic explicitly trains Claude to attend to XML
 * tag structures. Wrapping user input / retrieved context in tags:
 *   - Makes "where does the user's input end?" unambiguous (anti-injection)
 *   - Lets the system prompt instruct the model about specific tagged regions
 *   - Improves consistency on retrieval / extraction tasks
 * `escapeXml` defeats the canonical injection — putting a literal `</tag>` in
 * user content to break out of the parent tag.
 *
 * Why abTest is here (not waiting for Module 5): the discipline of "never
 * change two things at once" needs a concrete tool. Module 5 builds a full
 * eval harness (scoring, regression, LLM-as-judge); abTest is the minimum
 * seed — same inputs, multiple prompts, side-by-side outputs, eyeball.
 */

// ============================================================================
// Type-level placeholder extraction
// ============================================================================

/**
 * Recursively pull `{{name}}` tokens out of a template string LITERAL TYPE.
 *   ExtractVars<'Hi {{name}}, age {{age}}'> = 'name' | 'age'
 *
 * For this to work, `S` must be a string literal at the call site (TypeScript
 * preserves it when inferred through `<S extends string>`). If you assign a
 * template to a plain `string` variable first, the union collapses to `never`
 * and you lose the type safety — use the template inline or `as const`.
 */
export type ExtractVars<S extends string> = S extends `${string}{{${infer V}}}${infer Rest}`
  ? V | ExtractVars<Rest>
  : never;

/**
 * Variable record for a prompt. Three cases:
 *   1. Template widened to `string` (e.g., assigned via `'a' + 'b'` concatenation)
 *      → permissive Record<string, primitive>. We lose name-level safety; runtime
 *      validation in `substitute()` still catches missing values.
 *   2. Template is a narrower literal with NO `{{...}}` placeholders
 *      → empty record. Caller must pass `{}`.
 *   3. Template is a narrower literal WITH placeholders
 *      → typed record keyed by each extracted name. Forgetting / misspelling
 *      a variable is a compile-time error.
 *
 * `string extends S` is the standard idiom for "S is the unconstrained `string`
 * type" vs "S is a narrower literal."
 */
export type PromptVars<S extends string> = string extends S
  ? Record<string, string | number | boolean>
  : [ExtractVars<S>] extends [never]
    ? Record<string, never>
    : Record<ExtractVars<S>, string | number | boolean>;

// ============================================================================
// Prompt definition + rendering
// ============================================================================

export interface PromptDef<S extends string> {
  /** Stable identifier. Used in abTest output and version logs. */
  name: string;
  /** Semver-ish; bump on every edit so the changelog reads chronologically. */
  version: string;
  /** Brief reason for this version. Becomes load-bearing in Module 5. */
  changelog?: string;
  /** The user-message template. Use `{{name}}` for variables. */
  template: S;
  /** Optional system prompt — passed via `opts.system` on Anthropic;
   *  inserted as a system message on OpenAI/LM Studio (the wrapper normalizes). */
  system?: string;
  /** Few-shot input/output pairs. Each pair becomes a user/assistant turn
   *  in `renderMessages`. Diversity beats quantity — 3-5 well-chosen examples
   *  outperform 20 mediocre ones. */
  examples?: Array<{ input: PromptVars<S>; output: string }>;
}

/** Matches `Message` from lib/types.ts so renders drop into chat()/runTools() directly. */
export interface PromptMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface Prompt<S extends string> extends PromptDef<S> {
  /** Render the template into the bare user-message string. */
  render: (vars: PromptVars<S>) => string;
  /** Render into a full messages array with few-shot examples as user/assistant
   *  turns, then the final user query. Use this when feeding `chat()` or
   *  `structured()` directly. */
  renderMessages: (vars: PromptVars<S>) => PromptMessage[];
}

/**
 * Factory. Identity on the data fields; adds `render` and `renderMessages`.
 * The generic flows from the inline literal at the call site, so:
 *   const p = definePrompt({ template: 'Hi {{name}}', ... });
 *   p.render({ name: 'Lev' });        // OK
 *   p.render({});                      // ❌ type error: missing 'name'
 *   p.render({ name: 'Lev', x: '?' }); // ❌ type error: 'x' not in template
 *
 * The `const` modifier on the generic (TS 5+) preserves the template's literal
 * type without forcing callers to write `as const`. Without it, TS widens the
 * template to `string` and ExtractVars collapses to `never`.
 */
export function definePrompt<const S extends string>(def: PromptDef<S>): Prompt<S> {
  return {
    ...def,
    render: (vars) => substitute(def.template, vars as Record<string, unknown>),
    renderMessages: (vars) => {
      const messages: PromptMessage[] = [];
      for (const ex of def.examples ?? []) {
        messages.push({
          role: 'user',
          content: substitute(def.template, ex.input as Record<string, unknown>),
        });
        messages.push({ role: 'assistant', content: ex.output });
      }
      messages.push({
        role: 'user',
        content: substitute(def.template, vars as Record<string, unknown>),
      });
      return messages;
    },
  };
}

/**
 * Replace `{{var}}` placeholders. Throws on missing — silently inserting
 * "undefined" would produce bad model output AND waste tokens. Better to fail
 * loudly during dev.
 */
function substitute(template: string, vars: Record<string, unknown>): string {
  return template.replace(/\{\{(\w+)\}\}/g, (_, name) => {
    if (!(name in vars)) {
      throw new Error(`Prompt template references {{${name}}} but no value was provided.`);
    }
    return String(vars[name]);
  });
}

// ============================================================================
// XML tag composition
// ============================================================================

/**
 * Wrap content in an XML tag, escaping `<`, `>`, and `&` so embedded text
 * cannot break out of the tag. The model still sees the literal characters
 * (as `&lt;`, etc.) — defeats the canonical injection attack of putting
 * `</tag>` in user content.
 */
export function tag(name: string, content: string | number | boolean): string {
  return `<${name}>${escapeXml(String(content))}</${name}>`;
}

/**
 * Render multiple tag blocks separated by blank lines. Convenience for the
 * common pattern of building a context-then-question prompt:
 *   tags({ context: ctx, question: q, instructions: 'Be concise.' })
 */
export function tags(record: Record<string, string | number | boolean>): string {
  return Object.entries(record)
    .map(([name, content]) => tag(name, content))
    .join('\n\n');
}

function escapeXml(s: string): string {
  // Order matters: & must be replaced first or we'd double-encode the others.
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ============================================================================
// A/B comparison runner
// ============================================================================

/**
 * Minimal contract for an abTest entrant — anything with a name and a way to
 * produce a messages array from inputs. Real Prompt<S> objects satisfy this;
 * you can also hand-roll one inline for ad-hoc comparisons.
 */
export interface AbTestPrompt<V> {
  name: string;
  renderMessages: (vars: V) => PromptMessage[];
}

export interface AbTestCase<V> {
  /** Short identifier for this input — shown in the formatted output. */
  label: string;
  vars: V;
}

export interface AbTestRow<V, O> {
  label: string;
  vars: V;
  /** Keyed by prompt name so the same input shows each prompt's output. */
  outputs: Record<string, O>;
}

/**
 * Run every prompt against every case, collecting results in a table.
 * The `run` callback is generic in its output type — pass strings (for chat),
 * parsed data (for structured), or any shape you want — abTest just stores them.
 *
 * Cases run sequentially per row to keep output deterministic and ordered
 * (useful for terminal output). For parallel speedup over hundreds of cases,
 * Module 5's eval harness will do that properly with rate limits and retries.
 */
export async function abTest<V, O>(
  prompts: AbTestPrompt<V>[],
  cases: AbTestCase<V>[],
  run: (messages: PromptMessage[]) => Promise<O>,
): Promise<AbTestRow<V, O>[]> {
  const rows: AbTestRow<V, O>[] = [];
  for (const c of cases) {
    const outputs: Record<string, O> = {};
    for (const p of prompts) {
      outputs[p.name] = await run(p.renderMessages(c.vars));
    }
    rows.push({ label: c.label, vars: c.vars, outputs });
  }
  return rows;
}

/**
 * Pretty-print A/B results. Defaults to JSON.stringify for non-string outputs
 * (so structured-output comparisons render readably without caller boilerplate).
 * Pass a custom `stringify` for domain-specific formatting.
 *
 * Accepts only the fields we use (`label` + `outputs`) so hand-rolled rows
 * (e.g., merging two abTest runs in mini-project 04) don't have to carry a
 * synthetic `vars` field.
 */
export function formatAbTest<O>(
  rows: Array<{ label: string; outputs: Record<string, O> }>,
  options: { stringify?: (o: O) => string; maxLen?: number } = {},
): string {
  const stringify = options.stringify ?? defaultStringify;
  const maxLen = options.maxLen ?? 800;
  const out: string[] = [];
  for (const row of rows) {
    out.push(`\n━━━ ${row.label} ━━━`);
    for (const [name, output] of Object.entries(row.outputs)) {
      out.push(`\n[${name}]`);
      const rendered = stringify(output);
      const truncated =
        rendered.length > maxLen
          ? `${rendered.slice(0, maxLen)}…(+${rendered.length - maxLen} chars)`
          : rendered;
      out.push(
        truncated
          .split('\n')
          .map((line) => `  ${line}`)
          .join('\n'),
      );
    }
  }
  return out.join('\n');
}

function defaultStringify(value: unknown): string {
  if (typeof value === 'string') return value;
  return JSON.stringify(value, null, 2);
}
