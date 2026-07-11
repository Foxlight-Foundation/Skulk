const TARGET_SAMPLE_RATE = 24_000;
const TARGET_FRAME_SAMPLES = TARGET_SAMPLE_RATE / 10;
const MAX_SOCKET_BUFFERED_BYTES = 4 * 1024 * 1024;
const CONNECT_TIMEOUT_MS = 10_000;
const TRANSCRIPTION_TIMEOUT_MS = 300_000;

type WebSocketFactory = (url: string) => WebSocket;
export type TranscriptionCaptureMode = 'realtime' | 'batch' | null;

interface RealtimeServerError {
  message?: unknown;
}

interface RealtimeServerEvent {
  type?: unknown;
  transcript?: unknown;
  delta?: unknown;
  error?: RealtimeServerError;
}

/** Select the best available browser capture path for the mounted STT model. */
export function selectTranscriptionCaptureMode(
  realtimeRequested: boolean,
  realtimeCaptureAvailable: boolean,
  batchCaptureAvailable: boolean,
): TranscriptionCaptureMode {
  if (realtimeRequested && realtimeCaptureAvailable) return 'realtime';
  if (batchCaptureAvailable) return 'batch';
  return null;
}

/** Build the same-origin realtime endpoint URL for a mounted STT model. */
export function realtimeTranscriptionUrl(
  modelId: string,
  location: Pick<Location, 'protocol' | 'host'> = window.location,
): string {
  const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${scheme}//${location.host}/v1/realtime?model=${encodeURIComponent(modelId)}`;
}

/** Stateful linear resampler that preserves interpolation across input chunks. */
export class StreamingLinearResampler {
  private readonly step: number;
  private carry = new Float32Array(0);
  private position = 0;

  constructor(inputSampleRate: number, outputSampleRate: number = TARGET_SAMPLE_RATE) {
    if (!Number.isFinite(inputSampleRate) || inputSampleRate <= 0) {
      throw new Error('input sample rate must be positive');
    }
    if (!Number.isFinite(outputSampleRate) || outputSampleRate <= 0) {
      throw new Error('output sample rate must be positive');
    }
    this.step = inputSampleRate / outputSampleRate;
  }

  /** Resample one ordered mono Float32 input chunk. */
  process(chunk: Float32Array): Float32Array {
    if (chunk.length === 0) return new Float32Array(0);
    const combined = new Float32Array(this.carry.length + chunk.length);
    combined.set(this.carry);
    combined.set(chunk, this.carry.length);

    const output: number[] = [];
    while (this.position + 1 < combined.length) {
      const leftIndex = Math.floor(this.position);
      const fraction = this.position - leftIndex;
      const left = combined[leftIndex];
      const right = combined[leftIndex + 1];
      output.push(left + ((right - left) * fraction));
      this.position += this.step;
    }

    const consumed = Math.min(Math.floor(this.position), combined.length);
    this.carry = combined.slice(consumed);
    this.position -= consumed;
    return Float32Array.from(output);
  }
}

function pcm16Base64(samples: Float32Array): string {
  const bytes = new Uint8Array(samples.length * 2);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, samples[index]));
    const value = sample < 0 ? Math.round(sample * 0x8000) : Math.round(sample * 0x7fff);
    view.setInt16(index * 2, value, true);
  }
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

/** Options for one OpenAI-compatible realtime transcription socket. */
export interface RealtimeTranscriptionSocketOptions {
  modelId: string;
  onError?: (error: Error) => void;
  socketFactory?: WebSocketFactory;
  location?: Pick<Location, 'protocol' | 'host'>;
}

/** Own one bounded, single-utterance realtime transcription WebSocket. */
export class RealtimeTranscriptionSocket {
  private readonly modelId: string;
  private readonly onError?: (error: Error) => void;
  private readonly socketFactory: WebSocketFactory;
  private readonly location?: Pick<Location, 'protocol' | 'host'>;
  private socket: WebSocket | null = null;
  private resampler: StreamingLinearResampler | null = null;
  private readonly pendingSamples: number[] = [];
  private inputSampleRate: number | null = null;
  private connected = false;
  private committed = false;
  private terminal = false;
  private connectResolve: (() => void) | null = null;
  private connectReject: ((error: Error) => void) | null = null;
  private resultResolve: ((transcript: string) => void) | null = null;
  private resultReject: ((error: Error) => void) | null = null;
  private connectTimer: number | null = null;
  private resultTimer: number | null = null;

  constructor(options: RealtimeTranscriptionSocketOptions) {
    this.modelId = options.modelId;
    this.onError = options.onError;
    this.socketFactory = options.socketFactory ?? ((url) => new WebSocket(url));
    this.location = options.location;
  }

  /** Open the socket and wait for the server's effective session contract. */
  connect(): Promise<void> {
    if (this.socket) throw new Error('realtime transcription socket is already opened');
    const url = realtimeTranscriptionUrl(
      this.modelId,
      this.location ?? window.location,
    );
    const socket = this.socketFactory(url);
    this.socket = socket;
    socket.addEventListener('message', (event) => this.handleMessage(event));
    socket.addEventListener('error', () => {
      this.fail(new Error('Realtime transcription connection failed.'));
    });
    socket.addEventListener('close', () => {
      if (!this.terminal) {
        this.fail(new Error('Realtime transcription connection closed before completion.'));
      }
    });

    return new Promise<void>((resolve, reject) => {
      this.connectResolve = resolve;
      this.connectReject = reject;
      this.connectTimer = window.setTimeout(() => {
        this.fail(new Error('Realtime transcription connection timed out.'));
      }, CONNECT_TIMEOUT_MS);
    });
  }

  /** Resample and append one ordered microphone frame as 24 kHz PCM16. */
  append(samples: Float32Array, inputSampleRate: number): void {
    if (!this.connected || !this.socket || this.socket.readyState !== 1) {
      throw new Error('realtime transcription socket is not connected');
    }
    if (this.committed) throw new Error('realtime transcription input is already committed');
    if (this.socket.bufferedAmount > MAX_SOCKET_BUFFERED_BYTES) {
      const error = new Error('Realtime transcription cannot keep up with microphone audio.');
      this.fail(error);
      throw error;
    }
    if (this.inputSampleRate === null) {
      this.inputSampleRate = inputSampleRate;
      this.resampler = new StreamingLinearResampler(inputSampleRate);
    } else if (this.inputSampleRate !== inputSampleRate) {
      throw new Error('microphone sample rate changed during realtime transcription');
    }
    const resampled = this.resampler?.process(samples) ?? new Float32Array(0);
    for (const sample of resampled) this.pendingSamples.push(sample);
    while (this.pendingSamples.length >= TARGET_FRAME_SAMPLES) {
      this.sendSamples(Float32Array.from(
        this.pendingSamples.splice(0, TARGET_FRAME_SAMPLES),
      ));
    }
  }

  private sendSamples(samples: Float32Array): void {
    const socket = this.socket;
    if (!socket || socket.readyState !== 1) {
      throw new Error('realtime transcription socket is not connected');
    }
    socket.send(JSON.stringify({
      type: 'input_audio_buffer.append',
      audio: pcm16Base64(samples),
    }));
  }

  /** Half-close microphone input and resolve with the final transcript. */
  commit(): Promise<string> {
    if (!this.connected || !this.socket || this.socket.readyState !== 1) {
      return Promise.reject(new Error('realtime transcription socket is not connected'));
    }
    if (this.committed) {
      return Promise.reject(new Error('realtime transcription input is already committed'));
    }
    if (this.pendingSamples.length > 0) {
      this.sendSamples(Float32Array.from(this.pendingSamples.splice(0)));
    }
    this.committed = true;
    const result = new Promise<string>((resolve, reject) => {
      this.resultResolve = resolve;
      this.resultReject = reject;
      this.resultTimer = window.setTimeout(() => {
        this.fail(new Error('Realtime transcription result timed out.'));
      }, TRANSCRIPTION_TIMEOUT_MS);
    });
    this.socket.send(JSON.stringify({ type: 'input_audio_buffer.commit' }));
    return result;
  }

  /** Cancel the provider call by closing the owning WebSocket. */
  cancel(): void {
    if (this.terminal) return;
    this.terminal = true;
    this.clearTimers();
    this.connectReject?.(new Error('Realtime transcription was cancelled.'));
    this.resultReject?.(new Error('Realtime transcription was cancelled.'));
    this.socket?.close(1000, 'client cancelled transcription');
  }

  private handleMessage(event: MessageEvent): void {
    if (typeof event.data !== 'string') {
      this.fail(new Error('Realtime transcription returned a non-JSON event.'));
      return;
    }
    let payload: RealtimeServerEvent;
    try {
      payload = JSON.parse(event.data) as RealtimeServerEvent;
    } catch {
      this.fail(new Error('Realtime transcription returned invalid JSON.'));
      return;
    }
    if (payload.type === 'session.created') {
      this.connected = true;
      if (this.connectTimer !== null) window.clearTimeout(this.connectTimer);
      this.connectTimer = null;
      this.socket?.send(JSON.stringify({
        type: 'session.update',
        session: {
          type: 'transcription',
          audio: {
            input: {
              format: { type: 'audio/pcm', rate: TARGET_SAMPLE_RATE },
              transcription: { model: this.modelId },
              turn_detection: null,
              noise_reduction: null,
            },
          },
          include: [],
        },
      }));
      this.connectResolve?.();
      this.connectResolve = null;
      this.connectReject = null;
      return;
    }
    if (payload.type === 'conversation.item.input_audio_transcription.completed') {
      if (typeof payload.transcript !== 'string') {
        this.fail(new Error('Realtime transcription completed without transcript text.'));
        return;
      }
      this.terminal = true;
      this.clearTimers();
      this.resultResolve?.(payload.transcript);
      this.resultResolve = null;
      this.resultReject = null;
      return;
    }
    if (
      payload.type === 'conversation.item.input_audio_transcription.failed'
      || payload.type === 'error'
    ) {
      const message = typeof payload.error?.message === 'string'
        ? payload.error.message
        : 'Realtime transcription failed.';
      this.fail(new Error(message));
    }
  }

  private fail(error: Error): void {
    if (this.terminal) return;
    this.terminal = true;
    this.clearTimers();
    this.connectReject?.(error);
    this.resultReject?.(error);
    this.connectResolve = null;
    this.connectReject = null;
    this.resultResolve = null;
    this.resultReject = null;
    this.onError?.(error);
    if (this.socket?.readyState === 0 || this.socket?.readyState === 1) {
      this.socket.close(1011, 'realtime transcription failed');
    }
  }

  private clearTimers(): void {
    if (this.connectTimer !== null) window.clearTimeout(this.connectTimer);
    if (this.resultTimer !== null) window.clearTimeout(this.resultTimer);
    this.connectTimer = null;
    this.resultTimer = null;
  }
}

/** Own the browser Web Audio graph that forwards mono microphone samples. */
export class RealtimePcmCapture {
  private readonly stream: MediaStream;
  private readonly context: AudioContext;
  private readonly source: MediaStreamAudioSourceNode;
  private readonly capture: AudioWorkletNode;
  private readonly mute: GainNode;

  private constructor(
    stream: MediaStream,
    context: AudioContext,
    source: MediaStreamAudioSourceNode,
    capture: AudioWorkletNode,
    mute: GainNode,
  ) {
    this.stream = stream;
    this.context = context;
    this.source = source;
    this.capture = capture;
    this.mute = mute;
  }

  /** Open a 128-frame AudioWorklet capture graph for an existing microphone. */
  static async start(
    stream: MediaStream,
    onSamples: (samples: Float32Array, sampleRate: number) => void,
  ): Promise<RealtimePcmCapture> {
    const context = new AudioContext({ latencyHint: 'interactive' });
    try {
      await context.audioWorklet.addModule('/realtime-pcm-worklet.js');
      const source = context.createMediaStreamSource(stream);
      const capture = new AudioWorkletNode(context, 'skulk-realtime-pcm-capture');
      const mute = context.createGain();
      mute.gain.value = 0;
      capture.port.onmessage = (event: MessageEvent<unknown>) => {
        if (event.data instanceof Float32Array) {
          onSamples(event.data, context.sampleRate);
        }
      };
      source.connect(capture);
      capture.connect(mute);
      mute.connect(context.destination);
      await context.resume();
      return new RealtimePcmCapture(stream, context, source, capture, mute);
    } catch (error) {
      stream.getTracks().forEach((track) => track.stop());
      await context.close().catch(() => undefined);
      throw error;
    }
  }

  /** Stop microphone tracks and close the browser audio graph. */
  async stop(): Promise<void> {
    this.capture.port.onmessage = null;
    this.source.disconnect();
    this.capture.disconnect();
    this.mute.disconnect();
    this.stream.getTracks().forEach((track) => track.stop());
    await this.context.close().catch(() => undefined);
  }
}
