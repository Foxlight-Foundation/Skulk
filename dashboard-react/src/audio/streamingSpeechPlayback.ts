import { StreamingLinearResampler } from './realtimeTranscription';

const DEFAULT_MAX_BUFFERED_SECONDS = 8;
const DEFAULT_RESUME_BUFFERED_SECONDS = 4;

/** Return whether the current origin exposes the secure AudioWorklet surface. */
export function canUseStreamingSpeechPlayback(
  environment: {
    isSecureContext: boolean;
    AudioContext?: unknown;
    AudioWorkletNode?: unknown;
  } = window,
): boolean {
  return Boolean(
    environment.isSecureContext
    && environment.AudioContext
    && environment.AudioWorkletNode
  );
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

/** Split visible text into complete synthesis sentences and one retained tail. */
export function splitCompleteSpeechSentences(text: string): {
  sentences: string[];
  remainder: string;
} {
  const sentences: string[] = [];
  let boundary = 0;
  const matcher = /[.!?](?:["')\]]*)\s+/g;
  for (const match of text.matchAll(matcher)) {
    const end = (match.index ?? 0) + match[0].length;
    const sentence = text.slice(boundary, end).trim();
    if (sentence) sentences.push(sentence);
    boundary = end;
  }
  return { sentences, remainder: text.slice(boundary) };
}

/** One bounded browser playback session for a raw mono PCM HTTP response. */
export class StreamingSpeechPlayback {
  private context: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;
  private stopped = false;
  private bufferedSamples = 0;
  private readonly waiters = new Set<() => void>();

  constructor(
    private readonly maximumBufferedSeconds = DEFAULT_MAX_BUFFERED_SECONDS,
    private readonly resumeBufferedSeconds = DEFAULT_RESUME_BUFFERED_SECONDS,
  ) {}

  /** Stream a validated `audio/L16` response through a bounded AudioWorklet queue. */
  async play(response: Response, signal?: AbortSignal): Promise<void> {
    const sampleRate = Number(response.headers.get('X-Audio-Sample-Rate'));
    if (!Number.isInteger(sampleRate) || sampleRate <= 0) {
      throw new Error('Streaming PCM response did not include a valid sample rate.');
    }
    if (!response.body) throw new Error('Streaming speech response has no body.');

    this.stopped = false;
    const context = new AudioContext();
    this.context = context;
    const playbackSampleRate = context.sampleRate;
    const resampler = playbackSampleRate === sampleRate
      ? null
      : new StreamingLinearResampler(sampleRate, playbackSampleRate);
    const moduleUrl = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: 'text/javascript' }));
    try {
      await context.audioWorklet.addModule(moduleUrl);
    } finally {
      URL.revokeObjectURL(moduleUrl);
    }
    if (this.stopped) return;
    const node = new AudioWorkletNode(context, 'skulk-pcm-queue', {
      numberOfInputs: 0,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    this.node = node;
    node.connect(context.destination);
    await context.resume();

    const drained = new Promise<void>((resolve) => {
      node.port.onmessage = (event: MessageEvent<{ type: string; samples?: number }>) => {
        if (event.data.type === 'consumed') {
          this.bufferedSamples = Math.max(0, this.bufferedSamples - (event.data.samples ?? 0));
          if (this.bufferedSamples <= playbackSampleRate * this.resumeBufferedSeconds) {
            this.releaseWaiters();
          }
        } else if (event.data.type === 'drained') {
          resolve();
        }
      };
    });
    const stopOnAbort = () => this.stop();
    signal?.addEventListener('abort', stopOnAbort, { once: true });
    const reader = response.body.getReader();
    let pendingPcmByte: number | null = null;
    try {
      while (!this.stopped) {
        const { done, value } = await reader.read();
        if (done) break;
        const framed = completePcm16Samples(value, pendingPcmByte);
        pendingPcmByte = framed.pendingByte;
        if (framed.complete.byteLength === 0) continue;
        const decodedSamples = pcm16LeToFloat32(framed.complete);
        const samples = resampler?.process(decodedSamples) ?? decodedSamples;
        if (samples.length > playbackSampleRate * this.maximumBufferedSeconds) {
          throw new Error('One streaming speech frame exceeds the playback buffer limit.');
        }
        await this.waitForCapacity(playbackSampleRate, samples.length, signal);
        this.bufferedSamples += samples.length;
        node.port.postMessage({ type: 'audio', samples: samples.buffer }, [samples.buffer]);
      }
      if (!this.stopped) {
        if (pendingPcmByte !== null) {
          throw new Error('Streaming PCM response ended on an incomplete sample.');
        }
        node.port.postMessage({ type: 'end' });
        await drained;
      }
    } finally {
      signal?.removeEventListener('abort', stopOnAbort);
      await reader.cancel().catch(() => undefined);
      await this.closeAudio();
    }
  }

  /** Stop network backpressure waits, discard queued audio, and close the context. */
  stop(): void {
    this.stopped = true;
    this.node?.port.postMessage({ type: 'clear' });
    this.releaseWaiters();
    void this.closeAudio();
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

  private async closeAudio(): Promise<void> {
    this.node?.disconnect();
    this.node = null;
    const context = this.context;
    this.context = null;
    this.bufferedSamples = 0;
    if (context && context.state !== 'closed') await context.close();
  }
}

/** Serialize sentence-sized synthesis calls and cancel the active call as one unit. */
export class SpeechSentenceQueue {
  private readonly pending: string[] = [];
  private activeController: AbortController | null = null;
  private running = false;
  private stopped = false;

  constructor(
    private readonly playSentence: (text: string, signal: AbortSignal) => Promise<void>,
    private readonly onError: (error: unknown) => void,
    private readonly onIdle: () => void = () => undefined,
  ) {}

  /** Add complete visible sentences without starting overlapping synthesis calls. */
  enqueue(sentences: readonly string[]): void {
    if (this.stopped) return;
    this.pending.push(...sentences.filter((sentence) => sentence.trim().length > 0));
    void this.drain();
  }

  /** Cancel the active HTTP/audio call and discard sentences not yet synthesized. */
  stop(): void {
    this.stopped = true;
    this.pending.length = 0;
    this.activeController?.abort();
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
          }
          this.stop();
        } finally {
          if (this.activeController === controller) this.activeController = null;
        }
      }
    } finally {
      this.running = false;
      if (!this.stopped && this.pending.length === 0) this.onIdle();
    }
  }
}
