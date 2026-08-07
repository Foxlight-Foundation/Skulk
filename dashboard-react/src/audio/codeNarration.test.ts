import { describe, expect, it } from 'vitest';
import { CodeNarrator, FenceProbe } from './codeNarration';

const PHRASES = {
  openers: ['Opener A', 'Opener B'],
  fillers: ['Filler A', 'Filler B', 'Filler C'],
  closers: ['Closer A', 'Closer B'],
};

function narrator(overrides: { now: () => number; random?: () => number }) {
  return new CodeNarrator(PHRASES, {
    minimumGapMs: 7000,
    reopenDebounceMs: 2500,
    maximumFillersPerFence: 3,
    starvedBufferSeconds: 0.75,
    random: overrides.random ?? (() => 0),
    now: overrides.now,
  });
}

const open = { fenceOpen: true, queueStarved: true, bufferedSeconds: 0, proseFollowing: false };
const openBusy = { fenceOpen: true, queueStarved: false, bufferedSeconds: 3, proseFollowing: false };
const closed = { fenceOpen: false, queueStarved: true, bufferedSeconds: 0, proseFollowing: false };
const closedWithProse = { fenceOpen: false, queueStarved: true, bufferedSeconds: 0, proseFollowing: true };

describe('CodeNarrator', () => {
  it('speaks an opener when a fence opens and a closer after it settles shut', () => {
    let clock = 0;
    const n = narrator({ now: () => clock });
    expect(n.update(open)).toMatch(/Opener/);
    clock += 100;
    expect(n.update(closed)).toBeNull();
    clock += 2600;
    expect(n.update(closed)).toMatch(/Closer/);
  });

  it('never stacks fillers: silence unless starved, dry, spaced, and capped', () => {
    let clock = 0;
    const n = narrator({ now: () => clock });
    n.update(open);
    clock += 7100;
    expect(n.update(openBusy)).toBeNull();
    expect(n.update({ ...open, bufferedSeconds: 2 })).toBeNull();
    expect(n.update(open)).toMatch(/Filler/);
    expect(n.update(open)).toBeNull();
    clock += 7100;
    expect(n.update(open)).toMatch(/Filler/);
    clock += 7100;
    expect(n.update(open)).toMatch(/Filler/);
    clock += 7100;
    expect(n.update(open)).toBeNull();
  });

  it('treats a quick reopen as a continuation: no closer, no second opener', () => {
    let clock = 0;
    const n = narrator({ now: () => clock });
    n.update(open);
    clock += 100;
    n.update(closed);
    clock += 500;
    expect(n.update(open)).toBeNull();
    clock += 2600;
    expect(n.update(open)).toBeNull();
  });

  it('plays no orphan closer when no opener ever played', () => {
    const clock = 0;
    const n = narrator({ now: () => clock });
    expect(n.update(closed)).toBeNull();
    expect(n.finish()).toBeNull();
  });

  it('flushes the closer at stream end, including an unterminated fence', () => {
    const clock = 0;
    const n = narrator({ now: () => clock });
    n.update(open);
    expect(n.finish()).toMatch(/Closer/);
    expect(n.finish()).toBeNull();
  });

  it('does not repeat the same phrase twice in a row', () => {
    let clock = 0;
    const n = narrator({ now: () => clock, random: () => 0 });
    const first = n.update(open);
    clock += 7100;
    const second = n.update(open);
    clock += 7100;
    const third = n.update(open);
    expect(first).toBe('Opener A');
    expect(second).not.toBe(first);
    expect(third).not.toBe(second);
  });
});

describe('CodeNarrator debounce expiry ordering', () => {
  it('plays the finished closer when a new fence opens after a quiet gap', () => {
    let clock = 0;
    const n = narrator({ now: () => clock });
    n.update(open);
    clock += 100;
    n.update(closed);
    clock += 5000;
    expect(n.update(open)).toMatch(/Closer/);
    expect(n.update(open)).toMatch(/Opener/);
  });
});

describe('CodeNarrator collapsed translations', () => {
  it('still speaks when every phrase collapses to one string', () => {
    let clock = 0;
    const n = new CodeNarrator(
      { openers: ['Same', 'Same'], fillers: ['Same'], closers: ['Same'] },
      { now: () => clock, random: () => 0 },
    );
    expect(n.update(open)).toBe('Same');
    clock += 7100;
    expect(n.update(open)).toBe('Same');
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

describe('CodeNarrator closer ordering with trailing prose', () => {
  it('closes immediately when the closing delta carries follow-on prose', () => {
    let clock = 0;
    const n = narrator({ now: () => clock });
    n.update(open);
    clock += 100;
    expect(n.update(closedWithProse)).toMatch(/Closer/);
  });

  it('closes ahead of prose arriving shortly after the fence', () => {
    let clock = 0;
    const n = narrator({ now: () => clock });
    n.update(open);
    clock += 100;
    n.update(closed);
    clock += 500;
    expect(n.update(closedWithProse)).toMatch(/Closer/);
  });

  it('still treats a silent quick reopen as a continuation', () => {
    let clock = 0;
    const n = narrator({ now: () => clock });
    n.update(open);
    clock += 100;
    n.update(closed);
    clock += 500;
    expect(n.update(open)).toBeNull();
  });
});
