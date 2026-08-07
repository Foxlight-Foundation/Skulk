import { StreamingLinearResampler } from './realtimeTranscription';

const DEFAULT_MAX_BUFFERED_SECONDS = 8;
const DEFAULT_RESUME_BUFFERED_SECONDS = 4;
const DEFAULT_SCHEDULED_FRAME_SECONDS = 0.1;
const DEFAULT_SCHEDULED_START_LEAD_SECONDS = 0.05;

/** Browser audio sink selected for one streaming speech response. */
export type StreamingSpeechPlaybackMode = 'audio-worklet' | 'scheduled-buffer' | 'unavailable';

type StreamingSpeechEnvironment = {
  isSecureContext: boolean;
  AudioContext?: unknown;
  AudioWorkletNode?: unknown;
};

/** Select the best raw-PCM playback implementation available in this browser. */
export function streamingSpeechPlaybackMode(
  environment: StreamingSpeechEnvironment = window,
): StreamingSpeechPlaybackMode {
  if (typeof environment.AudioContext !== 'function') return 'unavailable';
  if (
    environment.isSecureContext
    && 'audioWorklet' in (environment.AudioContext as { prototype: object }).prototype
    && typeof environment.AudioWorkletNode === 'function'
  ) {
    return 'audio-worklet';
  }
  return 'scheduled-buffer';
}

/** Return whether the browser can play streaming raw PCM on this origin. */
export function canUseStreamingSpeechPlayback(
  environment: StreamingSpeechEnvironment = window,
): boolean {
  return streamingSpeechPlaybackMode(environment) !== 'unavailable';
}

/** Validate raw PCM response framing and return its sample rate. */
export function validatePcmResponseHeaders(headers: Headers): number {
  const sampleRate = Number(headers.get('X-Audio-Sample-Rate'));
  if (!Number.isInteger(sampleRate) || sampleRate <= 0) {
    throw new Error('Streaming PCM response did not include a valid sample rate.');
  }
  if (headers.get('X-Audio-Channels') !== '1') {
    throw new Error('Streaming PCM response must contain mono audio.');
  }
  if (headers.get('X-Audio-Sample-Format') !== 's16le') {
    throw new Error('Streaming PCM response must contain little-endian PCM16 audio.');
  }
  return sampleRate;
}

const WORKLET_SOURCE = `
class SkulkPcmQueueProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.offset = 0;
    this.ended = false;
    this.port.onmessage = (event) => {
      if (event.data.type === 'audio') {
        this.queue.push(new Float32Array(event.data.samples));
      } else if (event.data.type === 'end') {
        this.ended = true;
      } else if (event.data.type === 'clear') {
        this.queue = [];
        this.offset = 0;
        this.ended = true;
      }
    };
  }

  process(_inputs, outputs) {
    const output = outputs[0][0];
    output.fill(0);
    let written = 0;
    while (written < output.length && this.queue.length > 0) {
      const current = this.queue[0];
      const count = Math.min(output.length - written, current.length - this.offset);
      output.set(current.subarray(this.offset, this.offset + count), written);
      written += count;
      this.offset += count;
      if (this.offset === current.length) {
        this.queue.shift();
        this.offset = 0;
      }
    }
    if (written > 0) this.port.postMessage({ type: 'consumed', samples: written });
    if (this.ended && this.queue.length === 0) {
      this.port.postMessage({ type: 'drained' });
      return false;
    }
    return true;
  }
}
registerProcessor('skulk-pcm-queue', SkulkPcmQueueProcessor);
`;

/** Convert little-endian signed 16-bit PCM bytes into browser-native floats. */
export function pcm16LeToFloat32(bytes: Uint8Array): Float32Array {
  if (bytes.byteLength % 2 !== 0) {
    throw new Error('Streaming PCM response ended on an incomplete sample.');
  }
  const samples = new Float32Array(bytes.byteLength / 2);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < samples.length; index += 1) {
    const sample = view.getInt16(index * 2, true);
    samples[index] = sample < 0 ? sample / 32768 : sample / 32767;
  }
  return samples;
}

/** Reassemble arbitrary network chunks into complete PCM16 samples. */
export function completePcm16Samples(
  bytes: Uint8Array,
  pendingByte: number | null,
): { complete: Uint8Array; pendingByte: number | null } {
  let joined = bytes;
  if (pendingByte !== null) {
    joined = new Uint8Array(bytes.byteLength + 1);
    joined[0] = pendingByte;
    joined.set(bytes, 1);
  }
  const completeLength = joined.byteLength - (joined.byteLength % 2);
  return {
    complete: joined.subarray(0, completeLength),
    pendingByte: completeLength === joined.byteLength ? null : joined[completeLength],
  };
}

/** Split one decoded network frame into independently transferable queue frames. */
export function splitPlaybackSamples(
  samples: Float32Array,
  maximumSamples: number,
): Float32Array[] {
  if (!Number.isInteger(maximumSamples) || maximumSamples <= 0) {
    throw new Error('Playback frame limit must be a positive integer.');
  }
  const frames: Float32Array[] = [];
  for (let offset = 0; offset < samples.length; offset += maximumSamples) {
    frames.push(samples.slice(offset, Math.min(offset + maximumSamples, samples.length)));
  }
  return frames;
}

class PlaybackFrameAccumulator {
  private readonly chunks: Float32Array[] = [];
  private sampleCount = 0;

  constructor(private readonly frameSamples: number) {
    if (!Number.isInteger(frameSamples) || frameSamples <= 0) {
      throw new Error('Scheduled playback frame size must be a positive integer.');
    }
  }

  /** Add decoded samples and return every complete fixed-size playback frame. */
  push(samples: Float32Array): Float32Array[] {
    if (samples.length > 0) {
      this.chunks.push(samples);
      this.sampleCount += samples.length;
    }
    const frames: Float32Array[] = [];
    while (this.sampleCount >= this.frameSamples) {
      frames.push(this.take(this.frameSamples));
    }
    return frames;
  }

  /** Return the final partial frame after the network stream ends. */
  flush(): Float32Array | null {
    return this.sampleCount > 0 ? this.take(this.sampleCount) : null;
  }

  private take(length: number): Float32Array {
    const frame = new Float32Array(length);
    let written = 0;
    while (written < length) {
      const chunk = this.chunks[0];
      if (!chunk) throw new Error('Scheduled playback accumulator underflow.');
      const count = Math.min(length - written, chunk.length);
      frame.set(chunk.subarray(0, count), written);
      written += count;
      if (count === chunk.length) {
        this.chunks.shift();
      } else {
        this.chunks[0] = chunk.slice(count);
      }
    }
    this.sampleCount -= length;
    return frame;
  }
}

/**
 * Marker segment understood by the speech pipeline as "insert an audible
 * pause here" rather than text to synthesize. Uses invisible-separator
 * characters so it can never collide with model prose.
 */
export const SPEECH_PAUSE_MARKER = '\u2063skulk-pause\u2063';

/** Length of the audible pause inserted for structural breaks. */
export const SPEECH_PAUSE_SECONDS = 0.6;

/**
 * A fence delimiter line, allowing Markdown container prefixes (blockquote
 * markers) before the run: a fenced block inside a blockquote is still a
 * code block, and both fence scanners must see it or its contents get read
 * aloud. Group 1 is the delimiter run, group 2 the rest of the line.
 */
const FENCE_DELIMITER_LINE = /^\s{0,3}(?:>\s?)*\s{0,3}(`{3,}|~{3,})(.*)$/;

/** A Markdown thematic break: three or more -, * or _ alone on a line. */
const HORIZONTAL_RULE_LINE = /^\s*(?:(?:-\s*){3,}|(?:\*\s*){3,}|(?:_\s*){3,})$/;

/** A line whose Markdown role ends at the newline (heading, quote, list, table). */
const STRUCTURAL_SPEECH_LINE = /^\s{0,3}(?:#{1,6}\s+|>\s?|[-+*]\s+|\d+[.)]\s+|\|)/;

/** Text already carrying an end-of-thought mark that gives TTS its pause. */
const TERMINAL_SPEECH_PUNCTUATION = /[.!?\u2026:;,]["')\]]*$/;

/* Pictographs, variation selectors, joiners, skin tones, flags, keycaps:
 * none of them have a spoken reading, and engines that try produce noise. */
const EMOJI_CHARACTERS = /[0-9#*]\u{FE0F}?\u{20E3}|\p{Extended_Pictographic}|\u{FE0F}|\u{200D}|\u{20E3}|[\u{1F3FB}-\u{1F3FF}]|[\u{1F1E6}-\u{1F1FF}]|[\u{E0020}-\u{E007F}]/gu;

/**
 * Convert rendered Markdown prose into stable text suitable for speech
 * synthesis. Works block-by-block (blank lines separate blocks) so that any
 * block ending without terminal punctuation, such as a bold story title,
 * gains a period: without one the synthesized title bleeds straight into
 * the following sentence. Emoji are stripped rather than read, and
 * horizontal rules contribute nothing here (the sentence pipeline turns
 * them into an audible pause).
 */
/**
 * Remove fenced code blocks with the same CommonMark delimiter matching the
 * sentence splitter uses (same character, equal-or-longer run, bare closer),
 * so a four-backtick fence containing a three-backtick line drops whole. An
 * unclosed fence drops through the end of the text.
 */
function stripFencedCode(text: string): string {
  const kept: string[] = [];
  let fenceCharacter = '';
  let fenceLength = 0;
  let insideFence = false;
  for (const line of text.split('\n')) {
    const delimiterMatch = FENCE_DELIMITER_LINE.exec(line);
    if (delimiterMatch) {
      const run = delimiterMatch[1];
      if (!insideFence) {
        insideFence = true;
        fenceCharacter = run[0];
        fenceLength = run.length;
        continue;
      }
      if (
        run[0] === fenceCharacter
        && run.length >= fenceLength
        && delimiterMatch[2].trim() === ''
      ) {
        insideFence = false;
        continue;
      }
    }
    if (!insideFence) kept.push(line);
  }
  return kept.join('\n');
}

/** Remove HTML tags to a fixed point so nested fragments cannot survive. */
function stripHtmlTags(value: string): string {
  let previous = value;
  for (let pass = 0; pass < 5; pass += 1) {
    const next = previous.replace(/<[^<>]*>/g, '');
    if (next === previous) return next;
    previous = next;
  }
  // Give up on pathological nesting rather than looping: angle brackets have
  // no spoken reading anyway.
  return previous.replace(/[<>]/g, ' ');
}

export function speechTextFromMarkdown(text: string): string {
  const spokenBlocks = stripFencedCode(text.replace(/\r\n/g, '\n'))
    .split(/\n\s*\n/)
    .map((block) => {
      const lines = block
        .split(/\r?\n/)
        .map((line) => {
          if (HORIZONTAL_RULE_LINE.test(line)) return '';
          const structuralLine = /^\s*(?:#{1,6}\s+|>\s*|[-+*]\s+|\d+[.)]\s+)/.test(line);
          let spoken = line
            .replace(/^\s{0,3}(?:#{1,6}\s+|>\s*)/, '')
            .replace(/^\s*(?:[-+*]|\d+[.)])\s+/, '')
            .replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1')
            .replace(/\[([^\]]+)]\([^)]*\)/g, '$1')
            .replace(/<https?:\/\/[^>]+>/g, '')
            .replace(/`([^`]+)`/g, '$1')
            .replace(/(\*\*|__)(.*?)\1/g, '$2')
            .replace(/(\*|_)(.*?)\1/g, '$2')
            .replace(/~~(.*?)~~/g, '$1');
          spoken = stripHtmlTags(spoken)
            .replace(/[\\*_~]/g, '')
            .replace(EMOJI_CHARACTERS, '')
            .replace(/\s+/g, ' ')
            .trim();
          if (structuralLine && spoken && !TERMINAL_SPEECH_PUNCTUATION.test(spoken)) spoken += '.';
          return spoken;
        })
        .filter(Boolean);
      let spoken = lines.join(' ').replace(/\s+/g, ' ').trim();
      if (spoken && !TERMINAL_SPEECH_PUNCTUATION.test(spoken)) spoken += '.';
      return spoken;
    })
    .filter(Boolean);
  return spokenBlocks.join(' ').replace(/\s+/g, ' ').trim();
}

/**
 * Whether the text currently ends inside an unclosed fenced code block,
 * honoring the same container-prefix and delimiter rules as the sentence
 * splitter. The chat streaming loop uses this on the retained speech tail
 * to drive code narration.
 */
export function hasUnterminatedFence(text: string): boolean {
  let fenceCharacter = '';
  let fenceLength = 0;
  let insideFence = false;
  for (const line of text.split('\n')) {
    const delimiterMatch = FENCE_DELIMITER_LINE.exec(line);
    if (!delimiterMatch) continue;
    const run = delimiterMatch[1];
    if (!insideFence) {
      insideFence = true;
      fenceCharacter = run[0];
      fenceLength = run.length;
    } else if (
      run[0] === fenceCharacter
      && run.length >= fenceLength
      && delimiterMatch[2].trim() === ''
    ) {
      insideFence = false;
    }
  }
  return insideFence;
}

/**
 * Split visible text into complete synthesis sentences and one retained tail.
 *
 * Boundaries come from three signals, merged in order: terminal punctuation
 * followed by whitespace (the classic case), a completed blank line (the end
 * of a Markdown block, which is how an unpunctuated bold title stops
 * bleeding into the story's first sentence), and a completed structural line
 * (heading, quote, list item, table row, or horizontal rule), whose Markdown
 * role ends at its newline.
 */
export function splitCompleteSpeechSentences(text: string): {
  sentences: string[];
  remainder: string;
} {
  // Fenced code suppresses every boundary inside it: a blank line or a
  // list-looking line of code is not a speech boundary, and splitting there
  // would hand the queue an unterminated fence whose contents get read
  // aloud. An unclosed fence suppresses to the end of the text, keeping the
  // fence in the retained tail until it completes.
  const fenceRanges: Array<{ start: number; end: number }> = [];
  {
    let fenceStart = -1;
    let fenceCharacter = '';
    let fenceLength = 0;
    let scanFrom = 0;
    for (;;) {
      const newline = text.indexOf('\n', scanFrom);
      const lineEnd = newline === -1 ? text.length : newline;
      const line = text.slice(scanFrom, lineEnd);
      const delimiterMatch = FENCE_DELIMITER_LINE.exec(line);
      if (delimiterMatch) {
        const run = delimiterMatch[1];
        if (fenceStart === -1) {
          fenceStart = scanFrom;
          fenceCharacter = run[0];
          fenceLength = run.length;
        } else if (
          // CommonMark close rule: same character, a run at least as long as
          // the opener, and nothing after it but whitespace (info strings
          // belong to openers, so a suffixed line inside the fence is code).
          run[0] === fenceCharacter
          && run.length >= fenceLength
          && delimiterMatch[2].trim() === ''
        ) {
          fenceRanges.push({ start: fenceStart, end: lineEnd + 1 });
          fenceStart = -1;
        }
      }
      if (newline === -1) break;
      scanFrom = newline + 1;
    }
    if (fenceStart !== -1) fenceRanges.push({ start: fenceStart, end: text.length + 1 });
  }
  const insideFence = (position: number) => fenceRanges.some(
    (range) => position > range.start && position <= range.end,
  );

  const boundaries: number[] = [];
  // A completed fence is itself a segment: emit a boundary right after its
  // closing line so the fence travels to the queue whole (where the Markdown
  // conversion drops it) instead of gluing onto the following prose.
  for (const range of fenceRanges) {
    if (range.end <= text.length) boundaries.push(range.end);
  }
  const matcher = /[.!?](?:["')\]*_~]*)\s+/g;
  for (const match of text.matchAll(matcher)) {
    const punctuation = match.index ?? 0;
    if (insideFence(punctuation)) continue;
    const lineStart = text.lastIndexOf('\n', punctuation) + 1;
    const linePrefix = text.slice(lineStart, punctuation + 1).trim();
    if (/^(?:\d+|[A-Za-z])[.)]$/.test(linePrefix)) continue;
    boundaries.push((match.index ?? 0) + match[0].length);
  }
  let searchFrom = 0;
  for (;;) {
    const newline = text.indexOf('\n', searchFrom);
    if (newline === -1) break;
    const lineStart = text.lastIndexOf('\n', newline - 1) + 1;
    const line = text.slice(lineStart, newline);
    if (!insideFence(newline)) {
      const structural = STRUCTURAL_SPEECH_LINE.test(line) || HORIZONTAL_RULE_LINE.test(line);
      if (!line.trim() || structural) {
        boundaries.push(newline + 1);
      }
      // A structural line also ends whatever preceded it: without this,
      // unpunctuated prose directly above a thematic break glues to the
      // rule and the queue never sees an exact horizontal-rule segment to
      // turn into its audible pause.
      if (structural && lineStart > 0) {
        boundaries.push(lineStart);
      }
    }
    searchFrom = newline + 1;
  }
  boundaries.sort((a, b) => a - b);
  const sentences: string[] = [];
  let previous = 0;
  for (const boundary of boundaries) {
    if (boundary <= previous) continue;
    const sentence = text.slice(previous, boundary).trim();
    if (sentence) sentences.push(sentence);
    previous = boundary;
  }
  return { sentences, remainder: text.slice(previous) };
}

/** Re-split visible speech after a reasoning parser rewrites earlier text. */
export function resyncVisibleSpeech(
  previousVisible: string,
  pendingTail: string,
  nextVisible: string,
): { sentences: string[]; remainder: string } {
  const processedLength = Math.max(0, previousVisible.length - pendingTail.length);
  const processedPrefix = previousVisible.slice(0, processedLength);
  const unprocessed = nextVisible.startsWith(processedPrefix)
    ? nextVisible.slice(processedPrefix.length)
    : nextVisible;
  return splitCompleteSpeechSentences(unprocessed);
}

/** One bounded browser playback session spanning one or more raw mono PCM responses. */
export class StreamingSpeechPlayback {
  private context: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;
  private workletDrain: Promise<void> | null = null;
  private workletDrainResolver: (() => void) | null = null;
  private readonly scheduledNodes = new Map<AudioBufferSourceNode, number>();
  private readonly drainWaiters = new Set<() => void>();
  private scheduledThroughSeconds = 0;
  private stopped = false;
  private finished = false;
  private bufferedSamples = 0;
  private readonly waiters = new Set<() => void>();

  constructor(
    private readonly maximumBufferedSeconds = DEFAULT_MAX_BUFFERED_SECONDS,
    private readonly resumeBufferedSeconds = DEFAULT_RESUME_BUFFERED_SECONDS,
    private readonly playbackMode = streamingSpeechPlaybackMode(),
  ) {}

  /** Play one response and close its browser audio session after playback drains. */
  async play(response: Response, signal?: AbortSignal): Promise<void> {
    try {
      await this.append(response, signal);
      await this.finish();
    } catch (error) {
      this.stop();
      throw error;
    }
  }

  /** Append one complete synthesis response without waiting for queued audio to play. */
  async append(response: Response, signal?: AbortSignal): Promise<void> {
    const sampleRate = validatePcmResponseHeaders(response.headers);
    if (!response.body) throw new Error('Streaming speech response has no body.');
    if (this.playbackMode === 'unavailable') {
      throw new Error('This browser does not expose Web Audio playback.');
    }
    if (this.finished) throw new Error('Speech playback has already finished.');
    if (this.stopped) {
      await response.body.cancel().catch(() => undefined);
      return;
    }

    const context = await this.ensureAudioContext();
    if (!context || this.stopped) {
      await response.body.cancel().catch(() => undefined);
      return;
    }
    let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
    let stopOnAbort: (() => void) | null = null;
    try {
      const playbackSampleRate = context.sampleRate;
      const resampler = playbackSampleRate === sampleRate
        ? null
        : new StreamingLinearResampler(sampleRate, playbackSampleRate);

      stopOnAbort = () => this.stop();
      signal?.addEventListener('abort', stopOnAbort, { once: true });
      reader = response.body.getReader();
      const scheduledAccumulator = this.playbackMode === 'scheduled-buffer'
        ? new PlaybackFrameAccumulator(Math.max(
            1,
            Math.floor(playbackSampleRate * DEFAULT_SCHEDULED_FRAME_SECONDS),
          ))
        : null;
      let pendingPcmByte: number | null = null;
      while (!this.stopped) {
        const { done, value } = await reader.read();
        if (done) break;
        const framed = completePcm16Samples(value, pendingPcmByte);
        pendingPcmByte = framed.pendingByte;
        if (framed.complete.byteLength === 0) continue;
        const decodedSamples = pcm16LeToFloat32(framed.complete);
        const samples = resampler?.process(decodedSamples) ?? decodedSamples;
        const maximumFrameSamples = Math.floor(
          playbackSampleRate * this.maximumBufferedSeconds,
        );
        for (const frame of splitPlaybackSamples(samples, maximumFrameSamples)) {
          if (scheduledAccumulator) {
            for (const scheduledFrame of scheduledAccumulator.push(frame)) {
              await this.enqueueScheduledFrame(
                context,
                scheduledFrame,
                playbackSampleRate,
                signal,
              );
            }
          } else {
            await this.enqueueWorkletFrame(frame, playbackSampleRate, signal);
          }
        }
      }
      if (!this.stopped) {
        if (pendingPcmByte !== null) {
          throw new Error('Streaming PCM response ended on an incomplete sample.');
        }
        const finalScheduledFrame = scheduledAccumulator?.flush();
        if (finalScheduledFrame) {
          await this.enqueueScheduledFrame(
            context,
            finalScheduledFrame,
            playbackSampleRate,
            signal,
          );
        }
      }
    } catch (error) {
      this.stop();
      throw error;
    } finally {
      if (stopOnAbort) signal?.removeEventListener('abort', stopOnAbort);
      await reader?.cancel().catch(() => undefined);
    }
  }

  /**
   * Schedule a bounded run of silence between appended responses: the
   * audible pause for structural breaks such as horizontal rules. A no-op
   * until the first appended response has created the audio session, since
   * a pause before any audio is indistinguishable from latency.
   */
  async appendSilence(seconds: number, signal?: AbortSignal): Promise<void> {
    if (this.stopped || this.finished) return;
    const context = this.context;
    if (!context) return;
    const total = Math.round(Math.max(0, Math.min(seconds, 2)) * context.sampleRate);
    if (total === 0) return;
    const maximumFrameSamples = Math.max(
      1,
      Math.floor(context.sampleRate * this.maximumBufferedSeconds),
    );
    for (const frame of splitPlaybackSamples(new Float32Array(total), maximumFrameSamples)) {
      if (this.node) {
        await this.enqueueWorkletFrame(frame, context.sampleRate, signal);
      } else {
        await this.enqueueScheduledFrame(context, frame, context.sampleRate, signal);
      }
    }
  }

  /** Seconds of decoded audio still queued for playback (narration gate). */
  bufferedSeconds(): number {
    const context = this.context;
    if (!context || this.stopped || this.finished) return 0;
    return this.bufferedSamples / context.sampleRate;
  }

  /** Mark the session complete, wait for queued audio, and close the audio context. */
  async finish(): Promise<void> {
    if (this.finished || this.stopped) return;
    this.finished = true;
    try {
      if (this.node) {
        this.node.port.postMessage({ type: 'end' });
        await this.workletDrain;
      } else {
        await this.waitForScheduledDrain();
      }
    } finally {
      await this.closeAudio();
    }
  }

  /** Stop network backpressure waits, discard queued audio, and close the context. */
  stop(): void {
    this.stopped = true;
    this.node?.port.postMessage({ type: 'clear' });
    this.clearScheduledNodes();
    this.releaseWaiters();
    this.releaseDrainWaiters();
    this.releaseWorkletDrain();
    void this.closeAudio();
  }

  private async ensureAudioContext(): Promise<AudioContext | null> {
    if (this.context) return this.context;
    const context = new AudioContext();
    this.context = context;
    if (this.playbackMode === 'audio-worklet') {
      const preparedWorklet = await this.prepareAudioWorklet(context, context.sampleRate);
      this.workletDrain = preparedWorklet?.drained ?? null;
    }
    if (this.stopped || context.state === 'closed') return null;
    await context.resume();
    return context;
  }

  private async prepareAudioWorklet(
    context: AudioContext,
    playbackSampleRate: number,
  ): Promise<{ drained: Promise<void> } | null> {
    const moduleUrl = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: 'text/javascript' }));
    try {
      await context.audioWorklet.addModule(moduleUrl);
    } finally {
      URL.revokeObjectURL(moduleUrl);
    }
    if (this.stopped || context.state === 'closed') return null;
    const node = new AudioWorkletNode(context, 'skulk-pcm-queue', {
      numberOfInputs: 0,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    this.node = node;
    node.connect(context.destination);
    const drained = new Promise<void>((resolve) => {
      this.workletDrainResolver = resolve;
      node.port.onmessage = (event: MessageEvent<{ type: string; samples?: number }>) => {
        if (event.data.type === 'consumed') {
          this.bufferedSamples = Math.max(0, this.bufferedSamples - (event.data.samples ?? 0));
          if (this.bufferedSamples <= playbackSampleRate * this.resumeBufferedSeconds) {
            this.releaseWaiters();
          }
        } else if (event.data.type === 'drained') {
          this.releaseWorkletDrain();
        }
      };
    });
    return { drained };
  }

  private async enqueueWorkletFrame(
    frame: Float32Array,
    playbackSampleRate: number,
    signal?: AbortSignal,
  ): Promise<void> {
    await this.waitForCapacity(playbackSampleRate, frame.length, signal);
    if (this.stopped || signal?.aborted) return;
    this.bufferedSamples += frame.length;
    this.node?.port.postMessage({ type: 'audio', samples: frame.buffer }, [frame.buffer]);
  }

  private async enqueueScheduledFrame(
    context: AudioContext,
    frame: Float32Array,
    playbackSampleRate: number,
    signal?: AbortSignal,
  ): Promise<void> {
    await this.waitForCapacity(playbackSampleRate, frame.length, signal);
    if (this.stopped || signal?.aborted) return;

    const audioBuffer = context.createBuffer(1, frame.length, playbackSampleRate);
    audioBuffer.copyToChannel(frame, 0);
    const source = context.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(context.destination);
    const startAt = Math.max(
      this.scheduledThroughSeconds,
      context.currentTime + DEFAULT_SCHEDULED_START_LEAD_SECONDS,
    );
    this.scheduledThroughSeconds = startAt + frame.length / playbackSampleRate;
    this.bufferedSamples += frame.length;
    this.scheduledNodes.set(source, frame.length);
    source.onended = () => {
      const scheduledSamples = this.scheduledNodes.get(source);
      if (scheduledSamples === undefined) return;
      this.scheduledNodes.delete(source);
      source.disconnect();
      this.bufferedSamples = Math.max(0, this.bufferedSamples - scheduledSamples);
      if (this.bufferedSamples <= playbackSampleRate * this.resumeBufferedSeconds) {
        this.releaseWaiters();
      }
      if (this.scheduledNodes.size === 0) this.releaseDrainWaiters();
    };
    try {
      source.start(startAt);
    } catch (error) {
      source.onended = null;
      this.scheduledNodes.delete(source);
      source.disconnect();
      this.bufferedSamples = Math.max(0, this.bufferedSamples - frame.length);
      throw error;
    }
  }

  private async waitForScheduledDrain(): Promise<void> {
    if (this.scheduledNodes.size === 0) return;
    await new Promise<void>((resolve) => this.drainWaiters.add(resolve));
  }

  private async waitForCapacity(
    sampleRate: number,
    additionalSamples: number,
    signal?: AbortSignal,
  ): Promise<void> {
    while (
      !this.stopped
      && this.bufferedSamples + additionalSamples
        > sampleRate * this.maximumBufferedSeconds
    ) {
      await new Promise<void>((resolve, reject) => {
        const release = () => {
          signal?.removeEventListener('abort', onAbort);
          resolve();
        };
        const onAbort = () => {
          this.waiters.delete(release);
          reject(new DOMException('Speech playback aborted.', 'AbortError'));
        };
        signal?.addEventListener('abort', onAbort, { once: true });
        this.waiters.add(release);
      });
    }
  }

  private releaseWaiters(): void {
    for (const resolve of this.waiters) resolve();
    this.waiters.clear();
  }

  private releaseDrainWaiters(): void {
    for (const resolve of this.drainWaiters) resolve();
    this.drainWaiters.clear();
  }

  private releaseWorkletDrain(): void {
    this.workletDrainResolver?.();
    this.workletDrainResolver = null;
  }

  private clearScheduledNodes(): void {
    for (const source of this.scheduledNodes.keys()) {
      source.onended = null;
      try {
        source.stop();
      } catch {
        // A node that has already ended does not need another stop signal.
      }
      source.disconnect();
    }
    this.scheduledNodes.clear();
    this.scheduledThroughSeconds = 0;
  }

  private async closeAudio(): Promise<void> {
    this.node?.disconnect();
    this.node = null;
    this.clearScheduledNodes();
    this.releaseDrainWaiters();
    this.releaseWorkletDrain();
    const context = this.context;
    this.context = null;
    this.workletDrain = null;
    this.bufferedSamples = 0;
    if (context && context.state !== 'closed') await context.close();
  }
}

/** Serialize sentence synthesis, then drain its shared browser playback session. */
export class SpeechSentenceQueue {
  private readonly pending: string[] = [];
  private activeController: AbortController | null = null;
  private running = false;
  private stopped = false;
  private inputFinished = false;

  constructor(
    private readonly playSentence: (text: string, signal: AbortSignal) => Promise<void>,
    private readonly onError: (error: unknown) => void,
    private readonly onIdle: () => void = () => undefined,
    private readonly finishPlayback: () => Promise<void> = async () => undefined,
    private readonly stopPlayback: () => void = () => undefined,
  ) {}

  /** Add complete visible sentences without starting overlapping synthesis calls. */
  enqueue(sentences: readonly string[]): void {
    if (this.stopped || this.inputFinished) return;
    this.pending.push(...sentences
      .map((raw) => (
        HORIZONTAL_RULE_LINE.test(raw.trim()) ? SPEECH_PAUSE_MARKER : speechTextFromMarkdown(raw)
      ))
      .filter((sentence) => sentence.length > 0));
    void this.drain();
  }

  /** Whether nothing is pending or synthesizing (narration filler gate). */
  isStarved(): boolean {
    return this.pending.length === 0 && this.activeController === null;
  }

  /** Declare that no more sentences will arrive and drain queued browser audio. */
  finish(): void {
    if (this.stopped || this.inputFinished) return;
    this.inputFinished = true;
    void this.drain();
  }

  /** Cancel the active HTTP/audio call and discard sentences not yet synthesized. */
  stop(): void {
    if (this.stopped) return;
    this.stopped = true;
    this.pending.length = 0;
    this.activeController?.abort();
    this.stopPlayback();
  }

  private async drain(): Promise<void> {
    if (this.running || this.stopped) return;
    this.running = true;
    try {
      while (!this.stopped && this.pending.length > 0) {
        const sentence = this.pending.shift();
        if (!sentence) continue;
        const controller = new AbortController();
        this.activeController = controller;
        try {
          await this.playSentence(sentence, controller.signal);
        } catch (error) {
          if (!(error instanceof DOMException && error.name === 'AbortError')) {
            this.onError(error);
            this.stop();
            return;
          }
          this.stop();
        } finally {
          if (this.activeController === controller) this.activeController = null;
        }
      }
      if (!this.stopped && this.inputFinished) await this.finishPlayback();
    } catch (error) {
      if (!(error instanceof DOMException && error.name === 'AbortError')) {
        this.onError(error);
      }
      this.stop();
    } finally {
      this.running = false;
      if (this.stopped || (this.inputFinished && this.pending.length === 0)) {
        this.onIdle();
      }
    }
  }
}
