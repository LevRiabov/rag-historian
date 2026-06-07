/**
 * lib/rate-limit.ts — client-side pacing (Module 9).
 *
 * Scope note (deliberately small): the provider SDKs ALREADY handle 429s with
 * exponential backoff + Retry-After. So this is NOT a backoff reimplementation —
 * it's PROACTIVE pacing for when WE fan out a batch (the 50-q eval, an ingest
 * sweep) and would otherwise burst past the limit and eat avoidable 429 round
 * trips. Two independent knobs:
 *   - `requestsPerMinute` — a token bucket (smooths the rate over time)
 *   - `maxConcurrent`     — a semaphore (caps simultaneous in-flight calls)
 * Either can be omitted for "unlimited". Calls past the limit WAIT (they're not
 * dropped). Order is preserved (FIFO).
 *
 *   const limit = createRateLimiter({ requestsPerMinute: 50, maxConcurrent: 4 });
 *   await Promise.all(questions.map((q) => limit.schedule(() => runAgent(db, q))));
 */

export interface RateLimiterConfig {
  /** Max requests STARTED per minute (token bucket). 0 / undefined = unlimited. */
  requestsPerMinute?: number;
  /** Max requests in flight at once (semaphore). 0 / undefined = unlimited. */
  maxConcurrent?: number;
}

export interface RateLimiter {
  /** Run `fn` once a token AND a concurrency slot are free. Resolves/rejects
   *  with fn's result; a rejection still releases the slot. */
  schedule<T>(fn: () => Promise<T>): Promise<T>;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export function createRateLimiter(config: RateLimiterConfig = {}): RateLimiter {
  const rpm = config.requestsPerMinute ?? 0;
  const maxConcurrent = config.maxConcurrent ?? 0;
  const intervalMs = rpm > 0 ? 60_000 / rpm : 0; // min gap between two starts
  let lastStart = 0; // ms timestamp of the previous token grant
  let active = 0;
  // FIFO of waiters released as concurrency slots free up.
  const waiters: Array<() => void> = [];

  async function acquireToken(): Promise<void> {
    if (intervalMs === 0) return;
    // Serialize the gap computation so concurrent callers don't all read the same
    // lastStart and start together.
    const now = Date.now();
    const earliest = lastStart + intervalMs;
    const wait = Math.max(0, earliest - now);
    lastStart = Math.max(now, earliest);
    if (wait > 0) await sleep(wait);
  }

  async function acquireSlot(): Promise<void> {
    if (maxConcurrent === 0) return;
    if (active < maxConcurrent) {
      active++;
      return;
    }
    // At capacity → wait. The releasing task hands its slot DIRECTLY to us (it
    // does not decrement `active`), so we must NOT increment on wake — otherwise
    // a fresh caller racing in during the gap would oversubscribe.
    await new Promise<void>((resolve) => waiters.push(resolve));
  }

  function releaseSlot(): void {
    if (maxConcurrent === 0) return;
    const next = waiters.shift();
    if (next)
      next(); // transfer the slot; `active` unchanged
    else active--; // no one waiting; free the slot
  }

  async function schedule<T>(fn: () => Promise<T>): Promise<T> {
    await acquireSlot();
    try {
      await acquireToken();
      return await fn();
    } finally {
      releaseSlot();
    }
  }

  return { schedule };
}
