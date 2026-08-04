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

/** One bounded browser playback session for a raw mono PCM HTTP response. */
export class StreamingSpeechPlayback {
  private context: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;
  private readonly scheduledNodes = new Map<AudioBufferSourceNode, number>();
  private readonly drainWaiters = new Set<() => void>();
  private scheduledThroughSeconds = 0;
  private stopped = false;
  private bufferedSamples = 0;
  private readonly waiters = new Set<() => void>();

  constructor(
    private readonly maximumBufferedSeconds = DEFAULT_MAX_BUFFERED_SECONDS,
    private readonly resumeBufferedSeconds = DEFAULT_RESUME_BUFFERED_SECONDS,
    private readonly playbackMode = streamingSpeechPlaybackMode(),
  ) {}

  /** Stream a validated `audio/pcm` response through the best available audio sink. */
  async play(response: Response, signal?: AbortSignal): Promise<void> {
    const sampleRate = validatePcmResponseHeaders(response.headers);
    if (!response.body) throw new Error('Streaming speech response has no body.');
    if (this.playbackMode === 'unavailable') {
      throw new Error('This browser does not expose Web Audio playback.');
    }

    this.stopped = false;
    const context = new AudioContext();
    this.context = context;
    let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
    let stopOnAbort: (() => void) | null = null;
    try {
      const playbackSampleRate = context.sampleRate;
      const resampler = playbackSampleRate === sampleRate
        ? null
        : new StreamingLinearResampler(sampleRate, playbackSampleRate);
      const preparedWorklet = this.playbackMode === 'audio-worklet'
        ? await this.prepareAudioWorklet(context, playbackSampleRate)
        : null;
      if (this.stopped) return;
      await context.resume();

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
        if (preparedWorklet) {
          this.node?.port.postMessage({ type: 'end' });
          await preparedWorklet.drained;
        } else {
          await this.waitForScheduledDrain();
        }
      }
    } finally {
      if (stopOnAbort) signal?.removeEventListener('abort', stopOnAbort);
      await reader?.cancel().catch(() => undefined);
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
    void this.closeAudio();
  }

  private async prepareAudioWorklet(
    context: AudioContext,
    playbackSampleRate: number,
  ): Promise<{ drained: Promise<void> }> {
    const moduleUrl = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: 'text/javascript' }));
    try {
      await context.audioWorklet.addModule(moduleUrl);
    } finally {
      URL.revokeObjectURL(moduleUrl);
    }
    const node = new AudioWorkletNode(context, 'skulk-pcm-queue', {
      numberOfInputs: 0,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    });
    this.node = node;
    node.connect(context.destination);
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
            this.stop();
            this.onIdle();
            return;
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
