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
      realtimeResponseModelId: 'org/chat',
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
});
