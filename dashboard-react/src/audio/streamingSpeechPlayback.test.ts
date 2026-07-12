import { describe, expect, it, vi } from 'vitest';

import {
  pcm16LeToFloat32,
  completePcm16Samples,
  canUseStreamingSpeechPlayback,
  SpeechSentenceQueue,
  splitPlaybackSamples,
  splitCompleteSpeechSentences,
} from './streamingSpeechPlayback';

describe('canUseStreamingSpeechPlayback', () => {
  it('requires a secure origin and both Web Audio worklet constructors', () => {
    const constructors = { AudioContext: class {}, AudioWorkletNode: class {} };
    expect(canUseStreamingSpeechPlayback({ isSecureContext: true, ...constructors })).toBe(true);
    expect(canUseStreamingSpeechPlayback({ isSecureContext: false, ...constructors })).toBe(false);
    expect(canUseStreamingSpeechPlayback({ isSecureContext: true })).toBe(false);
  });
});

describe('pcm16LeToFloat32', () => {
  it('decodes signed little-endian samples', () => {
    const bytes = new Uint8Array([0x00, 0x80, 0x00, 0x00, 0xff, 0x7f]);
    expect(Array.from(pcm16LeToFloat32(bytes))).toEqual([-1, 0, 1]);
  });

  it('rejects partial samples', () => {
    expect(() => pcm16LeToFloat32(new Uint8Array([1]))).toThrow('incomplete sample');
  });
});

describe('completePcm16Samples', () => {
  it('carries an odd trailing byte into the next network chunk', () => {
    const first = completePcm16Samples(new Uint8Array([0x00, 0x80, 0x34]), null);
    expect(Array.from(first.complete)).toEqual([0x00, 0x80]);
    expect(first.pendingByte).toBe(0x34);

    const second = completePcm16Samples(new Uint8Array([0x12, 0xff, 0x7f]), first.pendingByte);
    expect(Array.from(second.complete)).toEqual([0x34, 0x12, 0xff, 0x7f]);
    expect(second.pendingByte).toBeNull();
  });
});

describe('splitPlaybackSamples', () => {
  it('splits a resampled backend chunk at the bounded queue capacity', () => {
    const frames = splitPlaybackSamples(new Float32Array(9), 4);
    expect(frames.map((frame) => frame.length)).toEqual([4, 4, 1]);
  });
});

describe('splitCompleteSpeechSentences', () => {
  it('returns complete sentences while retaining a partial tail', () => {
    expect(splitCompleteSpeechSentences('First sentence. Second one! Still going')).toEqual({
      sentences: ['First sentence.', 'Second one!'],
      remainder: 'Still going',
    });
  });

  it('does not split punctuation without a following boundary', () => {
    expect(splitCompleteSpeechSentences('Version 1.2 is ready')).toEqual({
      sentences: [],
      remainder: 'Version 1.2 is ready',
    });
  });
});

describe('SpeechSentenceQueue', () => {
  it('plays sentences serially in insertion order', async () => {
    const calls: string[] = [];
    let idle = false;
    let releaseFirst: (() => void) | undefined;
    const queue = new SpeechSentenceQueue(async (text) => {
      calls.push(text);
      if (text === 'First.') await new Promise<void>((resolve) => { releaseFirst = resolve; });
    }, () => undefined, () => { idle = true; });

    queue.enqueue(['First.', 'Second.']);
    await vi.waitFor(() => expect(calls).toEqual(['First.']));
    releaseFirst?.();
    await vi.waitFor(() => expect(calls).toEqual(['First.', 'Second.']));
    await vi.waitFor(() => expect(idle).toBe(true));
  });

  it('aborts active playback and drops pending sentences', async () => {
    const calls: string[] = [];
    let aborted = false;
    const queue = new SpeechSentenceQueue(async (text, signal) => {
      calls.push(text);
      await new Promise<void>((_resolve, reject) => {
        signal.addEventListener('abort', () => {
          aborted = true;
          reject(new DOMException('cancelled', 'AbortError'));
        }, { once: true });
      });
    }, () => undefined);

    queue.enqueue(['First.', 'Second.']);
    await vi.waitFor(() => expect(calls).toEqual(['First.']));
    queue.stop();
    await vi.waitFor(() => expect(aborted).toBe(true));
    expect(calls).toEqual(['First.']);
  });
});
