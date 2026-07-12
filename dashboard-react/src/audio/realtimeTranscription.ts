const TARGET_SAMPLE_RATE = 24_000;
const TARGET_FRAME_SAMPLES = TARGET_SAMPLE_RATE / 10;
const MAX_SOCKET_BUFFERED_BYTES = 4 * 1024 * 1024;
const CONNECT_TIMEOUT_MS = 10_000;
const TRANSCRIPTION_TIMEOUT_MS = 300_000;

type WebSocketFactory = (url: string) => WebSocket;
export type TranscriptionCaptureMode = 'realtime' | 'batch' | null;

interface RealtimeServerError {
  code?: unknown;
  message?: unknown;
}

interface RealtimeServerEvent {
  type?: unknown;
  item_id?: unknown;
  transcript?: unknown;
  delta?: unknown;
  text?: unknown;
  response?: { status?: unknown };
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
      this.socket?.close(1000, 'client completed transcription');
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

/** Callbacks emitted by one multi-turn realtime voice conversation. */
export interface RealtimeConversationCallbacks {
  onTranscript?: (text: string, final: boolean, itemId: string | null) => void;
  onAssistantText?: (text: string, final: boolean, itemId: string | null) => void;
  onSpeechStarted?: () => void;
  onSpeechStopped?: () => void;
  onResponseDone?: (status: string) => void;
  onError?: (error: Error) => void;
}

/** Options for a server-VAD realtime voice conversation. */
export interface RealtimeConversationSocketOptions extends RealtimeConversationCallbacks {
  transcriptionModelId: string;
  responseModelId?: string | null;
  socketFactory?: WebSocketFactory;
  location?: Pick<Location, 'protocol' | 'host'>;
}

/** Own a bounded multi-turn server-VAD socket used by dashboard voice chat. */
export class RealtimeConversationSocket {
  private readonly options: RealtimeConversationSocketOptions;
  private readonly socketFactory: WebSocketFactory;
  private socket: WebSocket | null = null;
  private resampler: StreamingLinearResampler | null = null;
  private inputSampleRate: number | null = null;
  private readonly pendingSamples: number[] = [];
  private readonly transcripts = new Map<string, string>();
  private readonly responses = new Map<string, string>();
  private connected = false;
  private acceptingAudio = true;
  private turnHasAudio = false;
  private responseActive = false;
  private terminal = false;
  private connectResolve: (() => void) | null = null;
  private connectReject: ((error: Error) => void) | null = null;
  private connectTimer: number | null = null;

  constructor(options: RealtimeConversationSocketOptions) {
    this.options = options;
    this.socketFactory = options.socketFactory ?? ((url) => new WebSocket(url));
  }

  /** Open the socket and install server VAD plus optional chat response routing. */
  connect(): Promise<void> {
    if (this.socket) throw new Error('realtime conversation socket is already opened');
    const socket = this.socketFactory(realtimeTranscriptionUrl(
      this.options.transcriptionModelId,
      this.options.location ?? window.location,
    ));
    this.socket = socket;
    socket.addEventListener('message', (event) => this.handleMessage(event));
    socket.addEventListener('error', () => this.fail(
      new Error('Realtime conversation connection failed.'),
    ));
    socket.addEventListener('close', () => {
      if (!this.terminal) this.fail(
        new Error('Realtime conversation connection closed unexpectedly.'),
      );
    });
    return new Promise<void>((resolve, reject) => {
      this.connectResolve = resolve;
      this.connectReject = reject;
      this.connectTimer = window.setTimeout(() => this.fail(
        new Error('Realtime conversation connection timed out.'),
      ), CONNECT_TIMEOUT_MS);
    });
  }

  /** Resample and append one ordered microphone frame to the active turn. */
  append(samples: Float32Array, inputSampleRate: number): void {
    const socket = this.socket;
    if (!this.connected || !socket || socket.readyState !== 1) {
      throw new Error('realtime conversation socket is not connected');
    }
    if (!this.acceptingAudio) return;
    if (socket.bufferedAmount > MAX_SOCKET_BUFFERED_BYTES) {
      const error = new Error('Realtime conversation cannot keep up with microphone audio.');
      this.fail(error);
      throw error;
    }
    if (this.inputSampleRate === null) {
      this.inputSampleRate = inputSampleRate;
      this.resampler = new StreamingLinearResampler(inputSampleRate);
    } else if (this.inputSampleRate !== inputSampleRate) {
      throw new Error('microphone sample rate changed during realtime conversation');
    }
    const resampled = this.resampler?.process(samples) ?? new Float32Array(0);
    if (resampled.length > 0) this.turnHasAudio = true;
    for (const sample of resampled) this.pendingSamples.push(sample);
    while (this.pendingSamples.length >= TARGET_FRAME_SAMPLES) {
      this.sendSamples(Float32Array.from(
        this.pendingSamples.splice(0, TARGET_FRAME_SAMPLES),
      ));
    }
  }

  /** Flush microphone tail and ask the server to close the current turn. */
  commitTurn(): boolean {
    const socket = this.socket;
    if (!this.connected || !socket || socket.readyState !== 1 || !this.turnHasAudio) {
      return false;
    }
    if (this.pendingSamples.length > 0) {
      this.sendSamples(Float32Array.from(this.pendingSamples.splice(0)));
    }
    socket.send(JSON.stringify({ type: 'input_audio_buffer.commit' }));
    this.acceptingAudio = false;
    return true;
  }

  /** Return whether assistant model or speech output is still draining. */
  hasActiveResponse(): boolean {
    return this.responseActive;
  }

  /** Cancel active assistant output while preserving the conversation socket. */
  cancelResponse(): void {
    if (this.connected && this.socket?.readyState === 1) {
      this.socket.send(JSON.stringify({ type: 'response.cancel' }));
    }
  }

  /** Close the socket and release all pending callback state. */
  close(): void {
    if (this.terminal) return;
    this.terminal = true;
    this.clearConnectTimer();
    this.connectReject?.(new Error('Realtime conversation was cancelled.'));
    this.connectResolve = null;
    this.connectReject = null;
    this.socket?.close(1000, 'client closed realtime conversation');
  }

  private sendSamples(samples: Float32Array): void {
    const socket = this.socket;
    if (!socket || socket.readyState !== 1) return;
    socket.send(JSON.stringify({
      type: 'input_audio_buffer.append',
      audio: pcm16Base64(samples),
    }));
  }

  private handleMessage(event: MessageEvent): void {
    if (typeof event.data !== 'string') {
      this.fail(new Error('Realtime conversation returned a non-JSON event.'));
      return;
    }
    let payload: RealtimeServerEvent;
    try {
      payload = JSON.parse(event.data) as RealtimeServerEvent;
    } catch {
      this.fail(new Error('Realtime conversation returned invalid JSON.'));
      return;
    }
    if (payload.type === 'session.created') {
      this.connected = true;
      this.clearConnectTimer();
      this.socket?.send(JSON.stringify({
        type: 'session.update',
        session: {
          type: 'transcription',
          audio: {
            input: {
              format: { type: 'audio/pcm', rate: TARGET_SAMPLE_RATE },
              transcription: { model: this.options.transcriptionModelId },
              turn_detection: { type: 'server_vad' },
              noise_reduction: null,
            },
          },
          include: [],
          ...(this.options.responseModelId
            ? { response: { model: this.options.responseModelId } }
            : {}),
        },
      }));
      this.connectResolve?.();
      this.connectResolve = null;
      this.connectReject = null;
      return;
    }
    const itemId = typeof payload.item_id === 'string' ? payload.item_id : null;
    if (payload.type === 'input_audio_buffer.speech_started') {
      this.options.onSpeechStarted?.();
      return;
    }
    if (payload.type === 'input_audio_buffer.speech_stopped') {
      this.acceptingAudio = false;
      this.options.onSpeechStopped?.();
      return;
    }
    if (
      payload.type === 'conversation.item.input_audio_transcription.delta'
      && typeof payload.delta === 'string'
    ) {
      const key = itemId ?? 'current';
      const text = (this.transcripts.get(key) ?? '') + payload.delta;
      this.transcripts.set(key, text);
      this.options.onTranscript?.(text, false, itemId);
      return;
    }
    if (
      payload.type === 'conversation.item.input_audio_transcription.completed'
      && typeof payload.transcript === 'string'
    ) {
      this.transcripts.delete(itemId ?? 'current');
      this.acceptingAudio = true;
      this.turnHasAudio = false;
      this.options.onTranscript?.(payload.transcript, true, itemId);
      return;
    }
    if (payload.type === 'response.output_text.delta' && typeof payload.delta === 'string') {
      const key = itemId ?? 'current';
      const text = (this.responses.get(key) ?? '') + payload.delta;
      this.responses.set(key, text);
      this.options.onAssistantText?.(text, false, itemId);
      return;
    }
    if (payload.type === 'response.created') {
      this.responseActive = true;
      return;
    }
    if (payload.type === 'response.output_text.done' && typeof payload.text === 'string') {
      this.responses.delete(itemId ?? 'current');
      this.options.onAssistantText?.(payload.text, true, itemId);
      return;
    }
    if (payload.type === 'response.done') {
      this.responseActive = false;
      const status = typeof payload.response?.status === 'string'
        ? payload.response.status
        : 'unknown';
      this.options.onResponseDone?.(status);
      return;
    }
    if (payload.type === 'error') {
      if (payload.error?.code === 'turn_in_progress') return;
      const message = typeof payload.error?.message === 'string'
        ? payload.error.message
        : 'Realtime conversation failed.';
      this.options.onError?.(new Error(message));
    }
  }

  private fail(error: Error): void {
    if (this.terminal) return;
    this.terminal = true;
    this.clearConnectTimer();
    this.connectReject?.(error);
    this.connectResolve = null;
    this.connectReject = null;
    this.options.onError?.(error);
    if (this.socket?.readyState === 0 || this.socket?.readyState === 1) {
      this.socket.close(1011, 'realtime conversation failed');
    }
  }

  private clearConnectTimer(): void {
    if (this.connectTimer !== null) window.clearTimeout(this.connectTimer);
    this.connectTimer = null;
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

/** Keep a newly started capture only while its realtime session still owns it. */
export async function finalizeRealtimeCaptureStartup(
  capture: Pick<RealtimePcmCapture, 'stop'>,
  componentIsMounted: boolean,
  sessionIsCurrent: boolean,
): Promise<boolean> {
  if (componentIsMounted && sessionIsCurrent) return true;
  await capture.stop();
  return false;
}
