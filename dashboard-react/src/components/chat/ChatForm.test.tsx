import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { userEvent } from 'vitest/browser';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { darkTheme } from '../../theme/theme';
import type { ChatSpeechModelOption } from '../../types/chat';
import { ChatForm } from './ChatForm';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({
    t: (_key: string, fallback: string) => fallback,
  }),
}));

let root: Root | null = null;
let container: HTMLDivElement | null = null;

async function renderChatForm(props: React.ComponentProps<typeof ChatForm>): Promise<void> {
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  await act(async () => {
    root?.render(
      <ThemeProvider theme={darkTheme}>
        <ChatForm {...props} />
      </ThemeProvider>,
    );
  });
}

async function rerenderChatForm(props: React.ComponentProps<typeof ChatForm>): Promise<void> {
  await act(async () => {
    root?.render(
      <ThemeProvider theme={darkTheme}>
        <ChatForm {...props} />
      </ThemeProvider>,
    );
  });
}

function installRealtimeBrowserFakes(): {
  audioTrack: { enabled: boolean; stop: ReturnType<typeof vi.fn> };
  sockets: Array<EventTarget & { sent: string[]; serverEvent: (payload: object) => void }>;
} {
  const sockets: Array<EventTarget & {
    sent: string[];
    serverEvent: (payload: object) => void;
  }> = [];
  class FakeWebSocket extends EventTarget {
    readyState = 1;
    bufferedAmount = 0;
    readonly sent: string[] = [];

    constructor(url: string) {
      super();
      void url;
      sockets.push(this);
    }

    send(data: string): void {
      this.sent.push(data);
    }

    close(code = 1000): void {
      this.readyState = 3;
      this.dispatchEvent(new CloseEvent('close', { code }));
    }

    serverEvent(payload: object): void {
      this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(payload) }));
    }
  }

  class FakeAudioContext {
    readonly sampleRate = 48_000;
    readonly audioWorklet = { addModule: vi.fn().mockResolvedValue(undefined) };
    readonly destination = {};

    createMediaStreamSource(): MediaStreamAudioSourceNode {
      return { connect: vi.fn(), disconnect: vi.fn() } as unknown as MediaStreamAudioSourceNode;
    }

    createGain(): GainNode {
      return {
        gain: { value: 1 },
        connect: vi.fn(),
        disconnect: vi.fn(),
      } as unknown as GainNode;
    }

    resume = vi.fn().mockResolvedValue(undefined);
    close = vi.fn().mockResolvedValue(undefined);
  }

  class FakeAudioWorkletNode {
    readonly port = { onmessage: null as ((event: MessageEvent<unknown>) => void) | null };
    connect = vi.fn();
    disconnect = vi.fn();
  }

  const audioTrack = { enabled: true, stop: vi.fn() };
  const stream = {
    getTracks: () => [audioTrack],
    getAudioTracks: () => [audioTrack],
  } as unknown as MediaStream;
  vi.stubGlobal('WebSocket', FakeWebSocket);
  vi.stubGlobal('AudioContext', FakeAudioContext);
  vi.stubGlobal('AudioWorkletNode', FakeAudioWorkletNode);
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia: vi.fn().mockResolvedValue(stream) },
  });
  return { audioTrack, sockets };
}

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  root = null;
  container = null;
  Reflect.deleteProperty(navigator, 'mediaDevices');
  vi.unstubAllGlobals();
});

describe('ChatForm speech controls', () => {
  it('hides the speech toolbar when no mounted speech capability exists', async () => {
    await renderChatForm({ onSend: vi.fn() });

    expect(container?.querySelector('[aria-label="Select transcription model"]')).toBeNull();
    expect(container?.querySelector('[aria-label="Select speech model"]')).toBeNull();
  });

  it('gates realtime controls and speaks a typed draft through mounted models', async () => {
    vi.stubGlobal('AudioContext', class {});
    vi.stubGlobal('AudioWorkletNode', class {});
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn() },
    });
    const transcriptionModel: ChatSpeechModelOption = {
      modelId: 'org/realtime-stt',
      label: 'Realtime STT',
      supportsRealtime: true,
    };
    const speechModel: ChatSpeechModelOption = {
      modelId: 'org/streaming-tts',
      label: 'Streaming TTS',
      defaultResponseFormat: 'pcm',
      responseFormats: ['pcm', 'mp3'],
      supportsStreaming: true,
    };
    const onRealtimeVoiceEnabledChange = vi.fn();
    const onAutoSubmitVoiceChange = vi.fn();
    const onAutoSpeakAssistantChange = vi.fn();
    const onSpeakText = vi.fn();

    await renderChatForm({
      onSend: vi.fn(),
      transcriptionModels: [transcriptionModel],
      selectedTranscriptionModelId: transcriptionModel.modelId,
      realtimeTranscriptionAvailable: true,
      realtimeVoiceEnabled: true,
      autoSubmitVoice: false,
      speechModels: [speechModel],
      selectedSpeechModelId: speechModel.modelId,
      onRealtimeVoiceEnabledChange,
      onAutoSubmitVoiceChange,
      onAutoSpeakAssistantChange,
      onSpeakText,
    });

    const realtime = container?.querySelector<HTMLButtonElement>('button[aria-pressed="true"]');
    const toggles = [...(container?.querySelectorAll<HTMLButtonElement>('button') ?? [])];
    const autoSend = toggles.find((button) => button.textContent === 'Auto-send');
    const autoSpeak = toggles.find((button) => button.textContent === 'Auto');
    expect(realtime?.textContent).toBe('Realtime');
    expect(autoSend).toBeDefined();
    expect(autoSpeak).toBeDefined();

    await act(async () => {
      await userEvent.click(realtime!);
      await userEvent.click(autoSend!);
      await userEvent.click(autoSpeak!);
    });
    expect(onRealtimeVoiceEnabledChange).toHaveBeenCalledWith(false);
    expect(onAutoSubmitVoiceChange).toHaveBeenCalledWith(true);
    expect(onAutoSpeakAssistantChange).toHaveBeenCalledWith(true);

    const textarea = container?.querySelector<HTMLTextAreaElement>('textarea');
    const speakDraft = container?.querySelector<HTMLButtonElement>('[aria-label="Speak draft"]');
    expect(textarea).not.toBeNull();
    expect(speakDraft).not.toBeNull();
    await act(async () => {
      await userEvent.fill(textarea!, 'Speak this response');
    });
    expect(speakDraft?.disabled).toBe(false);
    await act(async () => {
      await userEvent.click(speakDraft!);
    });
    expect(onSpeakText).toHaveBeenCalledWith('Speak this response');
  });

  it('preserves a completed realtime draft and disables capture before manual send', async () => {
    const { audioTrack, sockets } = installRealtimeBrowserFakes();
    const onSend = vi.fn();
    const transcriptionModel: ChatSpeechModelOption = {
      modelId: 'org/realtime-stt',
      label: 'Realtime STT',
      supportsRealtime: true,
    };
    await renderChatForm({
      onSend,
      transcriptionModels: [transcriptionModel],
      selectedTranscriptionModelId: transcriptionModel.modelId,
      realtimeTranscriptionAvailable: true,
      realtimeVoiceEnabled: true,
      autoSubmitVoice: false,
    });

    const microphone = container?.querySelector<HTMLButtonElement>(
      '[aria-label="Start realtime transcription"]',
    );
    expect(microphone).not.toBeNull();
    await act(async () => { microphone?.click(); });
    await vi.waitFor(() => expect(sockets).toHaveLength(1));
    await act(async () => {
      await Promise.resolve();
      sockets[0].serverEvent({ type: 'session.created' });
    });
    const sessionUpdate = JSON.parse(sockets[0].sent[0]) as {
      session: { response?: unknown };
    };
    expect(sessionUpdate.session.response).toBeUndefined();
    await vi.waitFor(() => expect(
      container?.querySelector('[aria-label="Stop recording"]'),
    ).not.toBeNull());

    await act(async () => {
      sockets[0].serverEvent({ type: 'input_audio_buffer.speech_started' });
      sockets[0].serverEvent({ type: 'input_audio_buffer.speech_stopped' });
      sockets[0].serverEvent({
        type: 'conversation.item.input_audio_transcription.completed',
        transcript: 'Keep this transcript',
      });
      sockets[0].serverEvent({
        type: 'conversation.item.input_audio_transcription.completed',
        transcript: '',
      });
    });

    const textarea = container?.querySelector<HTMLTextAreaElement>('[aria-label="Chat message"]');
    expect(textarea?.value).toBe('Keep this transcript');
    const send = container?.querySelector<HTMLButtonElement>('[aria-label="Send message"]');
    await act(async () => { send?.click(); });

    expect(onSend).toHaveBeenCalledWith('Keep this transcript', []);
    expect(audioTrack.enabled).toBe(false);
  });

  it('finishes a stopped realtime turn when its final transcript is empty', async () => {
    const { sockets } = installRealtimeBrowserFakes();
    const transcriptionModel: ChatSpeechModelOption = {
      modelId: 'org/realtime-stt',
      label: 'Realtime STT',
      supportsRealtime: true,
    };
    await renderChatForm({
      onSend: vi.fn(),
      transcriptionModels: [transcriptionModel],
      selectedTranscriptionModelId: transcriptionModel.modelId,
      realtimeTranscriptionAvailable: true,
      realtimeVoiceEnabled: true,
      autoSubmitVoice: false,
    });

    const microphone = container?.querySelector<HTMLButtonElement>(
      '[aria-label="Start realtime transcription"]',
    );
    await act(async () => { microphone?.click(); });
    await vi.waitFor(() => expect(sockets).toHaveLength(1));
    await act(async () => {
      await Promise.resolve();
      sockets[0].serverEvent({ type: 'session.created' });
    });
    const sessionUpdate = JSON.parse(sockets[0].sent[0]) as {
      session: { response?: unknown };
    };
    expect(sessionUpdate.session.response).toBeUndefined();
    await vi.waitFor(() => expect(
      container?.querySelector('[aria-label="Stop recording"]'),
    ).not.toBeNull());
    await act(async () => {
      sockets[0].serverEvent({ type: 'input_audio_buffer.speech_started' });
      sockets[0].serverEvent({ type: 'input_audio_buffer.speech_stopped' });
    });

    const stop = container?.querySelector<HTMLButtonElement>('[aria-label="Stop recording"]');
    await act(async () => { stop?.click(); });
    await act(async () => {
      sockets[0].serverEvent({
        type: 'conversation.item.input_audio_transcription.completed',
        transcript: '   ',
      });
    });

    expect(container?.querySelector('[aria-label="Start realtime transcription"]')).not.toBeNull();
    expect(container?.textContent).not.toContain('Transcribing');
  });

  it('disables capture as soon as Auto-send submits a completed utterance', async () => {
    const { audioTrack, sockets } = installRealtimeBrowserFakes();
    const onRealtimeTranscript = vi.fn();
    const transcriptionModel: ChatSpeechModelOption = {
      modelId: 'org/realtime-stt',
      label: 'Realtime STT',
      supportsRealtime: true,
    };
    await renderChatForm({
      onSend: vi.fn(),
      transcriptionModels: [transcriptionModel],
      selectedTranscriptionModelId: transcriptionModel.modelId,
      realtimeTranscriptionAvailable: true,
      realtimeVoiceEnabled: true,
      autoSubmitVoice: true,
      onRealtimeTranscript,
    });

    const microphone = container?.querySelector<HTMLButtonElement>(
      '[aria-label="Start realtime transcription"]',
    );
    await act(async () => { microphone?.click(); });
    await vi.waitFor(() => expect(sockets).toHaveLength(1));
    await act(async () => {
      await Promise.resolve();
      sockets[0].serverEvent({ type: 'session.created' });
    });
    const sessionUpdate = JSON.parse(sockets[0].sent[0]) as {
      session: { response?: unknown };
    };
    expect(sessionUpdate.session.response).toBeUndefined();
    await vi.waitFor(() => expect(
      container?.querySelector('[aria-label="Stop recording"]'),
    ).not.toBeNull());

    await act(async () => {
      sockets[0].serverEvent({ type: 'input_audio_buffer.speech_started' });
      sockets[0].serverEvent({ type: 'input_audio_buffer.speech_stopped' });
      sockets[0].serverEvent({
        type: 'conversation.item.input_audio_transcription.completed',
        transcript: 'Send this now',
      });
    });

    expect(onRealtimeTranscript).toHaveBeenCalledWith('Send this now', true);
    expect(audioTrack.enabled).toBe(false);
    expect(container?.querySelector<HTMLTextAreaElement>('[aria-label="Chat message"]')?.value)
      .toBe('');
  });

  it('retains a buffered final transcript that arrives after chat starts loading', async () => {
    const { audioTrack, sockets } = installRealtimeBrowserFakes();
    const onRealtimeTranscript = vi.fn();
    const onLateRealtimeTranscript = vi.fn();
    const transcriptionModel: ChatSpeechModelOption = {
      modelId: 'org/realtime-stt',
      label: 'Realtime STT',
      supportsRealtime: true,
    };
    const props: React.ComponentProps<typeof ChatForm> = {
      onSend: vi.fn(),
      transcriptionModels: [transcriptionModel],
      selectedTranscriptionModelId: transcriptionModel.modelId,
      realtimeTranscriptionAvailable: true,
      realtimeVoiceEnabled: true,
      autoSubmitVoice: true,
      onRealtimeTranscript,
    };
    await renderChatForm(props);

    const microphone = container?.querySelector<HTMLButtonElement>(
      '[aria-label="Start realtime transcription"]',
    );
    await act(async () => { microphone?.click(); });
    await vi.waitFor(() => expect(sockets).toHaveLength(1));
    await act(async () => {
      await Promise.resolve();
      sockets[0].serverEvent({ type: 'session.created' });
    });
    await vi.waitFor(() => expect(
      container?.querySelector('[aria-label="Stop recording"]'),
    ).not.toBeNull());

    await rerenderChatForm({
      ...props,
      isLoading: true,
      onRealtimeTranscript: onLateRealtimeTranscript,
    });
    await act(async () => {
      sockets[0].serverEvent({
        type: 'conversation.item.input_audio_transcription.completed',
        transcript: 'Keep this draft',
      });
    });

    expect(onRealtimeTranscript).not.toHaveBeenCalled();
    expect(onLateRealtimeTranscript).toHaveBeenCalledWith('Keep this draft', true);
    expect(audioTrack.enabled).toBe(false);
    expect(container?.querySelector<HTMLTextAreaElement>('[aria-label="Chat message"]')?.value)
      .toBe('Keep this draft');
  });

  it('renders discovered voices and preserves automatic language matching', async () => {
    const onSelectedVoiceChange = vi.fn();
    const speechModel: ChatSpeechModelOption = {
      modelId: 'org/catalog-tts',
      label: 'Catalog TTS',
      responseFormats: ['mp3'],
      supportsVoiceListing: true,
    };

    await renderChatForm({
      onSend: vi.fn(),
      speechModels: [speechModel],
      selectedSpeechModelId: speechModel.modelId,
      selectedVoice: null,
      voiceOptions: [
        { id: 'serena', name: 'Serena', preferredLanguages: ['zh'] },
        { id: 'ryan', name: 'Ryan', preferredLanguages: ['en'] },
      ],
      onSelectedVoiceChange,
    });

    const voiceSelect = container?.querySelector<HTMLSelectElement>('[aria-label="Voice"]');
    expect(voiceSelect?.value).toBe('');
    expect([...voiceSelect!.options].map((option) => option.textContent)).toEqual([
      'Auto (match language)',
      'Serena (zh)',
      'Ryan (en)',
    ]);

    await act(async () => {
      await userEvent.selectOptions(voiceSelect!, 'ryan');
    });
    expect(onSelectedVoiceChange).toHaveBeenCalledWith('ryan');
  });

  it('does not offer a free-form voice for a model without built-in voices', async () => {
    const speechModel: ChatSpeechModelOption = {
      modelId: 'org/reference-only-tts',
      label: 'Reference-only TTS',
      responseFormats: ['wav'],
      supportsVoiceListing: false,
      supportsReferenceAudio: true,
    };

    await renderChatForm({
      onSend: vi.fn(),
      speechModels: [speechModel],
      selectedSpeechModelId: speechModel.modelId,
      selectedVoice: 'ryan',
    });

    expect(container?.querySelector('[aria-label="Voice"]')).toBeNull();
    expect(container?.querySelector('[aria-label="Reference audio file"]')).not.toBeNull();
  });

  it('exposes request-scoped reference audio only for a supporting TTS model', async () => {
    const onReferenceAudioChange = vi.fn();
    const onReferenceAudioTextChange = vi.fn();
    const referenceAudio = new File(['RIFF-reference'], 'speaker.wav', {
      type: 'audio/wav',
    });
    const speechModel: ChatSpeechModelOption = {
      modelId: 'org/reference-tts',
      label: 'Reference TTS',
      responseFormats: ['mp3'],
      supportsReferenceAudio: true,
    };

    await renderChatForm({
      onSend: vi.fn(),
      speechModels: [speechModel],
      selectedSpeechModelId: speechModel.modelId,
      referenceAudioFile: referenceAudio,
      referenceAudioText: 'Known transcript',
      onReferenceAudioChange,
      onReferenceAudioTextChange,
    });

    expect(container?.textContent).toContain('speaker.wav');
    const transcript = container?.querySelector<HTMLInputElement>(
      '[aria-label="Reference transcript"]',
    );
    expect(transcript?.value).toBe('Known transcript');
    await act(async () => {
      await userEvent.fill(transcript!, 'Updated transcript');
    });
    expect(onReferenceAudioTextChange).toHaveBeenLastCalledWith('Updated transcript');

    const remove = container?.querySelector<HTMLButtonElement>(
      '[aria-label="Remove reference audio"]',
    );
    await act(async () => {
      await userEvent.click(remove!);
    });
    expect(onReferenceAudioChange).toHaveBeenCalledWith(null);
    expect(onReferenceAudioTextChange).toHaveBeenCalledWith('');

    const input = container?.querySelector<HTMLInputElement>(
      '[aria-label="Reference audio file"]',
    );
    const replacement = new File(['replacement'], 'replacement.wav', {
      type: 'audio/wav',
    });
    await act(async () => {
      await userEvent.upload(input!, replacement);
    });
    expect(onReferenceAudioChange).toHaveBeenLastCalledWith(replacement);
    expect(onReferenceAudioTextChange).toHaveBeenLastCalledWith('');
  });

  it('hides reference-audio controls for TTS models without the capability', async () => {
    await renderChatForm({
      onSend: vi.fn(),
      speechModels: [{
        modelId: 'org/fixed-voice-tts',
        label: 'Fixed Voice TTS',
        responseFormats: ['mp3'],
        supportsReferenceAudio: false,
      }],
      selectedSpeechModelId: 'org/fixed-voice-tts',
    });

    expect(container?.querySelector('[aria-label="Choose reference audio"]')).toBeNull();
    expect(container?.querySelector('[aria-label="Reference audio file"]')).toBeNull();
  });
});
