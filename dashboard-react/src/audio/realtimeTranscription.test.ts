import { describe, expect, it, vi } from 'vitest';
import {
  finalizeRealtimeCaptureStartup,
  RealtimeTranscriptionSocket,
  StreamingLinearResampler,
  realtimeTranscriptionUrl,
  selectTranscriptionCaptureMode,
} from './realtimeTranscription';

describe('finalizeRealtimeCaptureStartup', () => {
  it('stops a capture that finishes after its session lost ownership', async () => {
    const stop = vi.fn().mockResolvedValue(undefined);

    await expect(finalizeRealtimeCaptureStartup({ stop }, false)).resolves.toBe(false);
    expect(stop).toHaveBeenCalledOnce();
  });

  it('keeps a capture owned by the current session', async () => {
    const stop = vi.fn().mockResolvedValue(undefined);

    await expect(finalizeRealtimeCaptureStartup({ stop }, true)).resolves.toBe(true);
    expect(stop).not.toHaveBeenCalled();
  });
});

class FakeWebSocket extends EventTarget {
  readyState = 1;
  bufferedAmount = 0;
  readonly sent: string[] = [];
  closeCode: number | null = null;

  send(data: string): void {
    this.sent.push(data);
  }

  close(code = 1000): void {
    this.closeCode = code;
    this.readyState = 3;
    this.dispatchEvent(new CloseEvent('close', { code }));
  }

  serverEvent(payload: object): void {
    this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(payload) }));
  }
}

describe('realtimeTranscriptionUrl', () => {
  it('uses same-origin secure WebSockets and escapes the model id', () => {
    expect(realtimeTranscriptionUrl('org/model alpha', {
      protocol: 'https:',
      host: 'skulk.example:52415',
    })).toBe('wss://skulk.example:52415/v1/realtime?model=org%2Fmodel%20alpha');
  });
});

describe('selectTranscriptionCaptureMode', () => {
  it('prefers realtime capture when the service and browser support it', () => {
    expect(selectTranscriptionCaptureMode(true, true, true)).toBe('realtime');
  });

  it('falls back to batch capture when realtime browser APIs are unavailable', () => {
    expect(selectTranscriptionCaptureMode(true, false, true)).toBe('batch');
  });

  it('reports no capture path when neither implementation is available', () => {
    expect(selectTranscriptionCaptureMode(true, false, false)).toBeNull();
  });
});

describe('StreamingLinearResampler', () => {
  it('downsamples ordered chunks without resetting at chunk boundaries', () => {
    const resampler = new StreamingLinearResampler(48_000, 24_000);
    expect([...resampler.process(Float32Array.from([0, 0.25, 0.5]))]).toEqual([0]);
    expect([...resampler.process(Float32Array.from([0.75, 1]))]).toEqual([0.5]);
  });

  it('preserves interpolation state while upsampling', () => {
    const resampler = new StreamingLinearResampler(12_000, 24_000);
    expect([...resampler.process(Float32Array.from([0, 1]))]).toEqual([0, 0.5]);
    expect([...resampler.process(Float32Array.from([0]))]).toEqual([1, 0.5]);
  });
});

describe('RealtimeTranscriptionSocket', () => {
  it('maps browser PCM and commit onto the compatibility protocol', async () => {
    const socket = new FakeWebSocket();
    const client = new RealtimeTranscriptionSocket({
      modelId: 'org/realtime-stt',
      location: { protocol: 'http:', host: 'localhost:52415' },
      socketFactory: (url) => {
        expect(url).toBe('ws://localhost:52415/v1/realtime?model=org%2Frealtime-stt');
        return socket as unknown as WebSocket;
      },
    });

    const connected = client.connect();
    socket.serverEvent({ type: 'session.created' });
    await connected;
    expect(JSON.parse(socket.sent[0])).toMatchObject({
      type: 'session.update',
      session: {
        type: 'transcription',
        audio: {
          input: {
            format: { type: 'audio/pcm', rate: 24_000 },
            transcription: { model: 'org/realtime-stt' },
          },
        },
      },
    });

    client.append(Float32Array.from([0, 0.25, 0.5, 0.75, 1]), 48_000);
    const result = client.commit();
    const append = JSON.parse(socket.sent[1]) as { type: string; audio: string };
    expect(append.type).toBe('input_audio_buffer.append');
    expect(atob(append.audio)).toHaveLength(4);

    expect(JSON.parse(socket.sent[2])).toEqual({ type: 'input_audio_buffer.commit' });
    socket.serverEvent({
      type: 'conversation.item.input_audio_transcription.delta',
      delta: 'hello',
    });
    socket.serverEvent({
      type: 'conversation.item.input_audio_transcription.completed',
      transcript: 'hello world',
    });
    await expect(result).resolves.toBe('hello world');
  });

  it('surfaces typed server errors and closes the socket', async () => {
    const socket = new FakeWebSocket();
    const onError = vi.fn();
    const client = new RealtimeTranscriptionSocket({
      modelId: 'org/realtime-stt',
      location: { protocol: 'http:', host: 'localhost' },
      socketFactory: () => socket as unknown as WebSocket,
      onError,
    });

    const connected = client.connect();
    socket.serverEvent({ type: 'session.created' });
    await connected;
    socket.serverEvent({
      type: 'error',
      error: { message: 'realtime capacity is unavailable' },
    });

    expect(onError).toHaveBeenCalledOnce();
    expect(onError.mock.calls[0][0]).toEqual(
      new Error('realtime capacity is unavailable'),
    );
    expect(socket.closeCode).toBe(1011);
  });

  it('fails instead of buffering microphone audio without bound', async () => {
    const socket = new FakeWebSocket();
    const onError = vi.fn();
    const client = new RealtimeTranscriptionSocket({
      modelId: 'org/realtime-stt',
      location: { protocol: 'http:', host: 'localhost' },
      socketFactory: () => socket as unknown as WebSocket,
      onError,
    });

    const connected = client.connect();
    socket.serverEvent({ type: 'session.created' });
    await connected;
    socket.bufferedAmount = (4 * 1024 * 1024) + 1;

    expect(() => client.append(Float32Array.from([0, 1]), 48_000)).toThrow(
      'Realtime transcription cannot keep up with microphone audio.',
    );
    expect(onError).toHaveBeenCalledOnce();
    expect(socket.closeCode).toBe(1011);
  });
});
