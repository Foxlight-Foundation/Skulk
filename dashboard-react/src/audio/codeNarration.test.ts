import { describe, expect, it } from 'vitest';
import { CodeNarrator } from './codeNarration';

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

const open = { fenceOpen: true, queueStarved: true, bufferedSeconds: 0 };
const openBusy = { fenceOpen: true, queueStarved: false, bufferedSeconds: 3 };
const closed = { fenceOpen: false, queueStarved: true, bufferedSeconds: 0 };

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
