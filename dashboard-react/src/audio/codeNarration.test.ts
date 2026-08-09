import { describe, expect, it } from 'vitest';
import { CodeNarrator, FenceProbe } from './codeNarration';

const PHRASES = {
  openers: ['Opener A', 'Opener B'],
  fillers: ['Filler A', 'Filler B', 'Filler C'],
  closers: ['Closer A', 'Closer B'],
};

function narrator(overrides: { now: () => number; random?: () => number; phrases?: typeof PHRASES }) {
  return new CodeNarrator(overrides.phrases ?? PHRASES, {
    minimumGapMs: 7000,
    reopenDebounceMs: 2500,
    maximumFillersPerFence: 3,
    starvedBufferSeconds: 0.75,
    random: overrides.random ?? (() => 0),
    now: overrides.now,
  });
}

/** Mirror the streaming loop: settle before the delta's prose, observe after. */
function delta(
  n: CodeNarrator,
  state: { fenceOpen: boolean; prose?: boolean; starved?: boolean; buffered?: number },
): string[] {
  const utterances: string[] = [];
  const closer = n.settlePendingClose({
    fenceNowOpen: state.fenceOpen,
    proseFollowing: state.prose ?? false,
  });
  if (closer) utterances.push(closer);
  const observed = n.observe({
    fenceOpen: state.fenceOpen,
    queueStarved: state.starved ?? true,
    bufferedSeconds: state.buffered ?? 0,
  });
  if (observed) utterances.push(observed);
  return utterances;
}

describe('CodeNarrator', () => {
  it('speaks an opener when a fence opens and a closer after it settles shut', () => {
    let clock = 0;
    const n = narrator({ now: () => clock });
    expect(delta(n, { fenceOpen: true })).toEqual([expect.stringMatching(/Opener/)]);
    clock += 100;
    expect(delta(n, { fenceOpen: false })).toEqual([]);
    clock += 2600;
    expect(delta(n, { fenceOpen: false })).toEqual([expect.stringMatching(/Closer/)]);
  });

  it('queues the opener after introductory prose in the same delta', () => {
    const clock = 0;
    const n = narrator({ now: () => clock });
    // "Here's the function:\n```js" arrives as one delta: the loop enqueues
    // the intro sentence between settle and observe, so the opener follows.
    const utterances = delta(n, { fenceOpen: true, prose: true });
    expect(utterances).toEqual([expect.stringMatching(/Opener/)]);
  });

  it('closes immediately and ahead of prose when the closing delta carries it', () => {
    let clock = 0;
    const n = narrator({ now: () => clock });
    delta(n, { fenceOpen: true });
    clock += 100;
    expect(delta(n, { fenceOpen: false, prose: true })).toEqual([
      expect.stringMatching(/Closer/),
    ]);
  });

  it('closes ahead of prose arriving shortly after the fence', () => {
    let clock = 0;
    const n = narrator({ now: () => clock });
    delta(n, { fenceOpen: true });
    clock += 100;
    delta(n, { fenceOpen: false });
    clock += 500;
    expect(delta(n, { fenceOpen: false, prose: true })).toEqual([
      expect.stringMatching(/Closer/),
    ]);
  });

  it('does not let an unspoken completed fence defeat the reopen debounce', () => {
    let clock = 0;
    const n = narrator({ now: () => clock });
    delta(n, { fenceOpen: true });
    clock += 100;
    // Closing fence emits only the fence segment (no speech): prose=false.
    delta(n, { fenceOpen: false, prose: false });
    clock += 500;
    // Quick reopen stays a silent continuation.
    expect(delta(n, { fenceOpen: true, prose: false })).toEqual([]);
  });

  it('plays the finished closer when a new fence opens after a quiet gap', () => {
    let clock = 0;
    const n = narrator({ now: () => clock });
    delta(n, { fenceOpen: true });
    clock += 100;
    delta(n, { fenceOpen: false });
    clock += 5000;
    expect(delta(n, { fenceOpen: true })).toEqual([
      expect.stringMatching(/Closer/),
      expect.stringMatching(/Opener/),
    ]);
  });

  it('never stacks fillers: silence unless starved, dry, spaced, and capped', () => {
    let clock = 0;
    const n = narrator({ now: () => clock });
    delta(n, { fenceOpen: true });
    clock += 7100;
    expect(delta(n, { fenceOpen: true, starved: false, buffered: 3 })).toEqual([]);
    expect(delta(n, { fenceOpen: true, buffered: 2 })).toEqual([]);
    expect(delta(n, { fenceOpen: true })).toEqual([expect.stringMatching(/Filler/)]);
    expect(delta(n, { fenceOpen: true })).toEqual([]);
    clock += 7100;
    expect(delta(n, { fenceOpen: true })).toEqual([expect.stringMatching(/Filler/)]);
    clock += 7100;
    expect(delta(n, { fenceOpen: true })).toEqual([expect.stringMatching(/Filler/)]);
    clock += 7100;
    expect(delta(n, { fenceOpen: true })).toEqual([]);
  });

  it('plays no orphan closer when no opener ever played', () => {
    const clock = 0;
    const n = narrator({ now: () => clock });
    expect(delta(n, { fenceOpen: false })).toEqual([]);
    expect(n.finish()).toBeNull();
  });

  it('flushes the closer at stream end, including an unterminated fence', () => {
    const clock = 0;
    const n = narrator({ now: () => clock });
    delta(n, { fenceOpen: true });
    expect(n.finish()).toMatch(/Closer/);
    expect(n.finish()).toBeNull();
  });

  it('does not repeat the same phrase twice in a row', () => {
    let clock = 0;
    const n = narrator({ now: () => clock, random: () => 0 });
    const first = delta(n, { fenceOpen: true })[0];
    clock += 7100;
    const second = delta(n, { fenceOpen: true })[0];
    clock += 7100;
    const third = delta(n, { fenceOpen: true })[0];
    expect(first).toBe('Opener A');
    expect(second).not.toBe(first);
    expect(third).not.toBe(second);
  });

  it('still speaks when every phrase collapses to one string', () => {
    let clock = 0;
    const n = narrator({
      now: () => clock,
      phrases: { openers: ['Same', 'Same'], fillers: ['Same'], closers: ['Same'] },
    });
    expect(delta(n, { fenceOpen: true })).toEqual(['Same']);
    clock += 7100;
    expect(delta(n, { fenceOpen: true })).toEqual(['Same']);
  });
});

describe('FenceProbe', () => {
  it('tracks fence state incrementally across arbitrary chunk boundaries', () => {
    const probe = new FenceProbe();
    probe.feed('Before.\n``');
    expect(probe.isOpen()).toBe(false);
    probe.feed('`js\nconst x');
    expect(probe.isOpen()).toBe(true);
    probe.feed(' = 1;\n~~~\n# still code\n');
    expect(probe.isOpen()).toBe(true);
    probe.feed('```\nAfter');
    expect(probe.isOpen()).toBe(false);
  });

  it('resets for a reasoning rewrite and re-feeds cleanly', () => {
    const probe = new FenceProbe();
    probe.feed('```py\n');
    expect(probe.isOpen()).toBe(true);
    probe.reset();
    expect(probe.isOpen()).toBe(false);
    probe.feed('Plain prose only.\n');
    expect(probe.isOpen()).toBe(false);
  });
});
