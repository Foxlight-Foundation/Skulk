import { describe, expect, it, vi } from 'vitest';

import {
  pcm16LeToFloat32,
  completePcm16Samples,
  canUseStreamingSpeechPlayback,
  SpeechSentenceQueue,
  resyncVisibleSpeech,
  speechTextFromMarkdown,
  SPEECH_PAUSE_MARKER,
  splitPlaybackSamples,
  splitCompleteSpeechSentences,
  streamingSpeechPlaybackMode,
  StreamingSpeechPlayback,
  validatePcmResponseHeaders,
} from './streamingSpeechPlayback';

describe('canUseStreamingSpeechPlayback', () => {
  it('uses AudioWorklet on secure origins and scheduled buffers otherwise', () => {
    class SupportedAudioContext {}
    Object.defineProperty(SupportedAudioContext.prototype, 'audioWorklet', { value: {} });
    const constructors = { AudioContext: SupportedAudioContext, AudioWorkletNode: class {} };
    expect(canUseStreamingSpeechPlayback({ isSecureContext: true, ...constructors })).toBe(true);
    expect(canUseStreamingSpeechPlayback({ isSecureContext: false, ...constructors })).toBe(true);
    expect(canUseStreamingSpeechPlayback({ isSecureContext: true })).toBe(false);
    expect(streamingSpeechPlaybackMode({ isSecureContext: true, ...constructors }))
      .toBe('audio-worklet');
    expect(streamingSpeechPlaybackMode({ isSecureContext: false, ...constructors }))
      .toBe('scheduled-buffer');
    expect(streamingSpeechPlaybackMode({
      isSecureContext: true,
      AudioContext: class {},
      AudioWorkletNode: class {},
    })).toBe('scheduled-buffer');
    expect(streamingSpeechPlaybackMode({ isSecureContext: false }))
      .toBe('unavailable');
  });
});

describe('StreamingSpeechPlayback scheduled-buffer fallback', () => {
  it('plays contiguous PCM frames without a secure AudioWorklet context', async () => {
    const scheduledStarts: number[] = [];
    const copiedFrames: number[][] = [];

    class FakeAudioBuffer {
      copyToChannel(samples: Float32Array): void {
        copiedFrames.push(Array.from(samples));
      }
    }

    class FakeAudioBufferSourceNode {
      buffer: FakeAudioBuffer | null = null;
      onended: (() => void) | null = null;

      connect(): void {}
      disconnect(): void {}
      stop(): void {}
      start(when: number): void {
        scheduledStarts.push(when);
        window.setTimeout(() => this.onended?.(), 0);
      }
    }

    class FakeAudioContext {
      readonly sampleRate = 1000;
      readonly currentTime = 1;
      readonly destination = {};
      state: AudioContextState = 'suspended';

      createBuffer(): FakeAudioBuffer {
        return new FakeAudioBuffer();
      }
      createBufferSource(): FakeAudioBufferSourceNode {
        return new FakeAudioBufferSourceNode();
      }
      async resume(): Promise<void> {
        this.state = 'running';
      }
      async close(): Promise<void> {
        this.state = 'closed';
      }
    }

    const previousAudioContext = window.AudioContext;
    Object.defineProperty(window, 'AudioContext', {
      configurable: true,
      value: FakeAudioContext,
    });
    try {
      const samples = new Int16Array(250);
      samples.fill(16384);
      const response = new Response(samples.buffer, {
        headers: {
          'X-Audio-Sample-Rate': '1000',
          'X-Audio-Channels': '1',
          'X-Audio-Sample-Format': 's16le',
        },
      });
      const playback = new StreamingSpeechPlayback(8, 4, 'scheduled-buffer');
      await playback.play(response);

      expect(copiedFrames.map((frame) => frame.length)).toEqual([100, 100, 50]);
      expect(copiedFrames[0]?.[0]).toBeCloseTo(0.5);
      expect(scheduledStarts).toHaveLength(3);
      expect(scheduledStarts[0]).toBeCloseTo(1.05);
      expect(scheduledStarts[1]).toBeCloseTo(1.15);
      expect(scheduledStarts[2]).toBeCloseTo(1.25);
    } finally {
      Object.defineProperty(window, 'AudioContext', {
        configurable: true,
        value: previousAudioContext,
      });
    }
  });

  it('appends sentence responses to one timeline before playback drains', async () => {
    const scheduledStarts: number[] = [];
    const scheduledNodes: FakeAudioBufferSourceNode[] = [];
    let contextConstructions = 0;
    let contextCloses = 0;

    class FakeAudioBuffer {
      copyToChannel(): void {}
    }

    class FakeAudioBufferSourceNode {
      buffer: FakeAudioBuffer | null = null;
      onended: (() => void) | null = null;

      connect(): void {}
      disconnect(): void {}
      stop(): void {}
      start(when: number): void {
        scheduledStarts.push(when);
        scheduledNodes.push(this);
      }
    }

    class FakeAudioContext {
      readonly sampleRate = 1000;
      readonly currentTime = 1;
      readonly destination = {};
      state: AudioContextState = 'suspended';

      constructor() {
        contextConstructions += 1;
      }

      createBuffer(): FakeAudioBuffer {
        return new FakeAudioBuffer();
      }
      createBufferSource(): FakeAudioBufferSourceNode {
        return new FakeAudioBufferSourceNode();
      }
      async resume(): Promise<void> {
        this.state = 'running';
      }
      async close(): Promise<void> {
        contextCloses += 1;
        this.state = 'closed';
      }
    }

    const response = () => new Response(new Int16Array(100).buffer, {
      headers: {
        'X-Audio-Sample-Rate': '1000',
        'X-Audio-Channels': '1',
        'X-Audio-Sample-Format': 's16le',
      },
    });
    const previousAudioContext = window.AudioContext;
    Object.defineProperty(window, 'AudioContext', {
      configurable: true,
      value: FakeAudioContext,
    });
    try {
      const playback = new StreamingSpeechPlayback(8, 4, 'scheduled-buffer');
      await playback.append(response());
      await playback.append(response());

      expect(contextConstructions).toBe(1);
      expect(scheduledStarts).toHaveLength(2);
      expect(scheduledStarts[0]).toBeCloseTo(1.05);
      expect(scheduledStarts[1]).toBeCloseTo(1.15);

      let finished = false;
      const finishing = playback.finish().then(() => { finished = true; });
      await Promise.resolve();
      expect(finished).toBe(false);

      for (const node of scheduledNodes) node.onended?.();
      await finishing;
      expect(finished).toBe(true);
      expect(contextCloses).toBe(1);
    } finally {
      Object.defineProperty(window, 'AudioContext', {
        configurable: true,
        value: previousAudioContext,
      });
    }
  });
});

describe('StreamingSpeechPlayback AudioWorklet cancellation', () => {
  it('does not construct a worklet node after stop closes the context', async () => {
    let releaseModule: (() => void) | undefined;
    let workletNodeConstructions = 0;
    const closeContext = vi.fn();
    const cancelResponseBody = vi.fn();

    class FakeAudioContext {
      readonly sampleRate = 24000;
      readonly destination = {};
      state: AudioContextState = 'suspended';
      readonly audioWorklet = {
        addModule: vi.fn(() => new Promise<void>((resolve) => { releaseModule = resolve; })),
      };

      async resume(): Promise<void> {
        this.state = 'running';
      }
      async close(): Promise<void> {
        this.state = 'closed';
        closeContext();
      }
    }

    class FakeAudioWorkletNode {
      constructor() {
        workletNodeConstructions += 1;
      }
    }

    const previousAudioContext = window.AudioContext;
    const previousAudioWorkletNode = window.AudioWorkletNode;
    Object.defineProperties(window, {
      AudioContext: { configurable: true, value: FakeAudioContext },
      AudioWorkletNode: { configurable: true, value: FakeAudioWorkletNode },
    });
    try {
      const response = new Response(new ReadableStream<Uint8Array>({
        cancel: cancelResponseBody,
      }), {
        headers: {
          'X-Audio-Sample-Rate': '24000',
          'X-Audio-Channels': '1',
          'X-Audio-Sample-Format': 's16le',
        },
      });
      const playback = new StreamingSpeechPlayback(8, 4, 'audio-worklet');
      const playing = playback.play(response);
      await vi.waitFor(() => expect(releaseModule).toBeTypeOf('function'));

      playback.stop();
      releaseModule?.();
      await playing;

      expect(workletNodeConstructions).toBe(0);
      expect(closeContext).toHaveBeenCalledOnce();
      expect(cancelResponseBody).toHaveBeenCalledOnce();
    } finally {
      Object.defineProperties(window, {
        AudioContext: { configurable: true, value: previousAudioContext },
        AudioWorkletNode: { configurable: true, value: previousAudioWorkletNode },
      });
    }
  });
});

describe('validatePcmResponseHeaders', () => {
  it('requires sample rate, mono channels, and signed little-endian PCM16', () => {
    const headers = new Headers({
      'X-Audio-Sample-Rate': '24000',
      'X-Audio-Channels': '1',
      'X-Audio-Sample-Format': 's16le',
    });
    expect(validatePcmResponseHeaders(headers)).toBe(24000);
    headers.set('X-Audio-Channels', '2');
    expect(() => validatePcmResponseHeaders(headers)).toThrow('mono');
    headers.set('X-Audio-Channels', '1');
    headers.set('X-Audio-Sample-Format', 'f32le');
    expect(() => validatePcmResponseHeaders(headers)).toThrow('PCM16');
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

  it('does not pronounce an ordered-list marker as its own sentence', () => {
    expect(splitCompleteSpeechSentences('1. **First item.**\n2. Second item.')).toEqual({
      sentences: ['1. **First item.**'],
      remainder: '2. Second item.',
    });
  });
});

describe('splitCompleteSpeechSentences structural boundaries', () => {
  it('finalizes an unpunctuated title once its blank line completes', () => {
    expect(splitCompleteSpeechSentences('**The Fox and the Ember**\n\nOnce upon a time. More')).toEqual({
      sentences: ['**The Fox and the Ember**', 'Once upon a time.'],
      remainder: 'More',
    });
  });

  it('finalizes a heading at its own newline without waiting for a blank line', () => {
    expect(splitCompleteSpeechSentences('### Chapter One\nThe valley slept. Next')).toEqual({
      sentences: ['### Chapter One', 'The valley slept.'],
      remainder: 'Next',
    });
  });

  it('does not split a soft-wrapped prose line', () => {
    expect(splitCompleteSpeechSentences('the quick brown fox\njumps over the lazy dog')).toEqual({
      sentences: [],
      remainder: 'the quick brown fox\njumps over the lazy dog',
    });
  });

  it('does not split inside a fenced code block', () => {
    const streamed = 'Look at this.\n```python\n# not a heading\n\nvalue = 1\n```\nDone. Tail';
    expect(splitCompleteSpeechSentences(streamed)).toEqual({
      sentences: ['Look at this.', '```python\n# not a heading\n\nvalue = 1\n```', 'Done.'],
      remainder: 'Tail',
    });
  });

  it('retains an unterminated fence in the tail', () => {
    const streamed = 'Intro.\n```python\nprint(1)\n\n';
    expect(splitCompleteSpeechSentences(streamed)).toEqual({
      sentences: ['Intro.'],
      remainder: '```python\nprint(1)\n\n',
    });
  });

  it('does not close a backtick fence on a tilde delimiter inside it', () => {
    const streamed = 'Start.\n```text\n~~~\n# still code\n```\nEnd. Tail';
    expect(splitCompleteSpeechSentences(streamed)).toEqual({
      sentences: ['Start.', '```text\n~~~\n# still code\n```', 'End.'],
      remainder: 'Tail',
    });
  });

  it('keeps a longer-run fence open across an inner three-backtick line', () => {
    const streamed = 'Start.\n````markdown\n```\n# code\n```\n````\nEnd. Tail';
    expect(splitCompleteSpeechSentences(streamed)).toEqual({
      sentences: ['Start.', '````markdown\n```\n# code\n```\n````', 'End.'],
      remainder: 'Tail',
    });
  });

  it('does not close a fence on a delimiter line carrying an info string', () => {
    const streamed = 'Start.\n```text\n```not-a-closer\n# code\n```\nEnd. Tail';
    expect(splitCompleteSpeechSentences(streamed)).toEqual({
      sentences: ['Start.', '```text\n```not-a-closer\n# code\n```', 'End.'],
      remainder: 'Tail',
    });
  });

  it('drops a nested-fence block whole from spoken text', () => {
    expect(speechTextFromMarkdown('Before.\n\n````markdown\n```\n# inner\n```\n````\n\nAfter.'))
      .toBe('Before. After.');
  });

  it('handles Windows newlines around a title block', () => {
    expect(speechTextFromMarkdown('**A Title**\r\n\r\nBody text.')).toBe('A Title. Body text.');
  });

  it('strips nested HTML tag fragments to a fixed point', () => {
    expect(speechTextFromMarkdown('safe <scr<script>ipt>alert(1)</scr</script>ipt> text')).toBe(
      'safe alert(1) text.',
    );
  });

  it('emits a horizontal rule as its own segment', () => {
    expect(splitCompleteSpeechSentences('Before the break.\n\n---\n\nAfter')).toEqual({
      sentences: ['Before the break.', '---'],
      remainder: 'After',
    });
  });
});

describe('speechTextFromMarkdown', () => {
  it('appends terminal punctuation to an unpunctuated title block', () => {
    expect(speechTextFromMarkdown('**The Fox and the Ember**')).toBe('The Fox and the Ember.');
  });

  it('does not double-punctuate blocks that already end a thought', () => {
    expect(speechTextFromMarkdown('A story about foxes:')).toBe('A story about foxes:');
    expect(speechTextFromMarkdown('It ended well!')).toBe('It ended well!');
  });

  it('strips emoji instead of letting the engine read them', () => {
    expect(speechTextFromMarkdown('Great work! \u{1F389}\u{1F525}')).toBe('Great work!');
    expect(speechTextFromMarkdown('Family: \u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467} together.')).toBe('Family: together.');
  });

  it('drops horizontal rules rather than reading dashes aloud', () => {
    expect(speechTextFromMarkdown('Before.\n\n---\n\nAfter.')).toBe('Before. After.');
  });
});

describe('speechTextFromMarkdown', () => {
  it('removes presentation markup while preserving readable prose and pauses', () => {
    expect(speechTextFromMarkdown([
      '## **Summary**',
      '1. Visit [the dashboard](https://example.test).',
      '2. Run `skulk` and ~~ignore~~ inspect the result',
    ].join('\n'))).toBe(
      'Summary. Visit the dashboard. Run skulk and ignore inspect the result.',
    );
  });

  it('drops fenced code rather than reading implementation punctuation aloud', () => {
    expect(speechTextFromMarkdown('Before.\n```ts\nconst value = 1;\n```\nAfter.'))
      .toBe('Before. After.');
  });
});

describe('resyncVisibleSpeech', () => {
  it('reprocesses answer text revealed when a split reasoning marker closes', () => {
    expect(resyncVisibleSpeech('</think', '</think', 'Hello. Next')).toEqual({
      sentences: ['Hello.'],
      remainder: 'Next',
    });
  });

  it('does not replay a complete sentence that was processed before a rewrite', () => {
    expect(resyncVisibleSpeech('Done. </think', '</think', 'Done. Answer. Tail')).toEqual({
      sentences: ['Answer.'],
      remainder: 'Tail',
    });
  });
});

describe('SpeechSentenceQueue', () => {
  it('maps a horizontal-rule segment to the pause marker', async () => {
    const calls: string[] = [];
    let idle = false;
    const queue = new SpeechSentenceQueue(async (text) => { calls.push(text); }, () => undefined, () => { idle = true; });
    queue.enqueue(['Before the break.', '---', 'After the break.']);
    queue.finish();
    await vi.waitFor(() => { if (!idle) throw new Error('queue not idle'); });
    expect(calls).toEqual(['Before the break.', SPEECH_PAUSE_MARKER, 'After the break.']);
  });

  it('plays sentences serially in insertion order', async () => {
    const calls: string[] = [];
    let idle = false;
    let releaseFirst: (() => void) | undefined;
    const queue = new SpeechSentenceQueue(async (text) => {
      calls.push(text);
      if (text === 'First.') await new Promise<void>((resolve) => { releaseFirst = resolve; });
    }, () => undefined, () => { idle = true; });

    queue.enqueue(['First.', 'Second.']);
    queue.finish();
    await vi.waitFor(() => expect(calls).toEqual(['First.']));
    releaseFirst?.();
    await vi.waitFor(() => expect(calls).toEqual(['First.', 'Second.']));
    await vi.waitFor(() => expect(idle).toBe(true));
  });

  it('normalizes Markdown before selecting and synthesizing queued speech', async () => {
    const calls: string[] = [];
    const queue = new SpeechSentenceQueue(
      async (text) => { calls.push(text); },
      () => undefined,
    );

    queue.enqueue(['1. **First item.**', '- [Second item](https://example.test)']);
    queue.finish();

    await vi.waitFor(() => expect(calls).toEqual(['First item.', 'Second item.']));
  });

  it('starts the next synthesis before the shared playback session drains', async () => {
    const calls: string[] = [];
    let releaseFirst: (() => void) | undefined;
    let releasePlayback: (() => void) | undefined;
    let idle = false;
    const finishPlayback = vi.fn(() => new Promise<void>((resolve) => {
      releasePlayback = resolve;
    }));
    const queue = new SpeechSentenceQueue(async (text) => {
      calls.push(text);
      if (text === 'First.') await new Promise<void>((resolve) => { releaseFirst = resolve; });
    }, () => undefined, () => { idle = true; }, finishPlayback);

    queue.enqueue(['First.', 'Second.']);
    queue.finish();
    await vi.waitFor(() => expect(calls).toEqual(['First.']));
    releaseFirst?.();
    await vi.waitFor(() => expect(calls).toEqual(['First.', 'Second.']));
    await vi.waitFor(() => expect(finishPlayback).toHaveBeenCalledOnce());
    expect(idle).toBe(false);

    releasePlayback?.();
    await vi.waitFor(() => expect(idle).toBe(true));
  });

  it('aborts active playback and drops pending sentences', async () => {
    const calls: string[] = [];
    let aborted = false;
    const stopPlayback = vi.fn();
    const queue = new SpeechSentenceQueue(async (text, signal) => {
      calls.push(text);
      await new Promise<void>((_resolve, reject) => {
        signal.addEventListener('abort', () => {
          aborted = true;
          reject(new DOMException('cancelled', 'AbortError'));
        }, { once: true });
      });
    }, () => undefined, () => undefined, async () => undefined, stopPlayback);

    queue.enqueue(['First.', 'Second.']);
    await vi.waitFor(() => expect(calls).toEqual(['First.']));
    queue.stop();
    await vi.waitFor(() => expect(aborted).toBe(true));
    expect(calls).toEqual(['First.']);
    expect(stopPlayback).toHaveBeenCalledOnce();
  });

  it('returns to idle after a playback error', async () => {
    const errors: unknown[] = [];
    let idle = false;
    const queue = new SpeechSentenceQueue(
      async () => { throw new Error('playback failed'); },
      (error) => errors.push(error),
      () => { idle = true; },
    );

    queue.enqueue(['First.']);
    await vi.waitFor(() => expect(errors).toHaveLength(1));
    await vi.waitFor(() => expect(idle).toBe(true));
  });
});
