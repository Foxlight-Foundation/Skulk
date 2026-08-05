// Copyright 2026 Foxlight Foundation

import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { StreamingSpeechPlayback } from '../../audio/streamingSpeechPlayback';
import { store } from '../../store';
import { chatActions } from '../../store/slices/chatSlice';
import { darkTheme } from '../../theme/theme';
import { ChatView } from './ChatView';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

type RealtimeAssistantTextHandler = (text: string, final: boolean) => void;

const chatFormCapture = vi.hoisted(() => ({
  onRealtimeAssistantText: null as RealtimeAssistantTextHandler | null,
  speechModelCount: 0,
  selectedSpeechModelId: null as string | null,
  autoSpeakAssistant: false,
}));

vi.mock('../chat/ChatForm', () => ({
  ChatForm: (props: {
    onRealtimeAssistantText?: RealtimeAssistantTextHandler;
    speechModels?: readonly unknown[];
    selectedSpeechModelId?: string | null;
    autoSpeakAssistant?: boolean;
  }) => {
    chatFormCapture.onRealtimeAssistantText = props.onRealtimeAssistantText ?? null;
    chatFormCapture.speechModelCount = props.speechModels?.length ?? 0;
    chatFormCapture.selectedSpeechModelId = props.selectedSpeechModelId ?? null;
    chatFormCapture.autoSpeakAssistant = props.autoSpeakAssistant ?? false;
    return null;
  },
}));

vi.mock('../../i18n/tolgee', () => {
  const translate = (_key: string, fallback: string) => fallback;
  return {
    tolgee: { getLanguage: () => 'en' },
    useSkulkTranslation: () => ({ t: translate }),
  };
});

let root: Root | null = null;
let container: HTMLDivElement | null = null;

async function waitFor(predicate: () => boolean, message: string): Promise<void> {
  const deadline = performance.now() + 5000;
  while (performance.now() < deadline) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  throw new Error(message);
}

afterEach(async () => {
  await act(async () => root?.unmount());
  for (const conversationId of Object.keys(store.getState().chat.conversations)) {
    store.dispatch(chatActions.deleteConversation(conversationId));
  }
  store.dispatch(chatActions.selectSpeechModel(null));
  store.dispatch(chatActions.setAutoSpeakAssistant(false));
  container?.remove();
  root = null;
  container = null;
  chatFormCapture.onRealtimeAssistantText = null;
  chatFormCapture.speechModelCount = 0;
  chatFormCapture.selectedSpeechModelId = null;
  chatFormCapture.autoSpeakAssistant = false;
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('ChatView realtime speech playback', () => {
  it('creates a fresh sentence queue for every completed realtime response', async () => {
    let speechRequests = 0;
    const requestedUrls: string[] = [];
    vi.stubGlobal('AudioContext', class FakeAudioContext {});
    vi.spyOn(StreamingSpeechPlayback.prototype, 'append').mockResolvedValue();
    const finishPlayback = vi
      .spyOn(StreamingSpeechPlayback.prototype, 'finish')
      .mockResolvedValue();
    vi.spyOn(StreamingSpeechPlayback.prototype, 'stop').mockImplementation(() => undefined);
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      requestedUrls.push(url);
      if (url === '/models') {
        return new Response(JSON.stringify({
          data: [
            {
              id: 'org/chat-model',
              tasks: ['TextGeneration'],
              audio: {
                default_response_format: 'pcm',
                response_formats: ['pcm'],
                supports_streaming: true,
              },
              resolved_capabilities: {
                family: 'generic',
                supports_thinking: false,
                supports_thinking_toggle: false,
                supports_thinking_budget: false,
                default_reasoning_effort: 'none',
                disabled_reasoning_effort: 'none',
                thinking_format: 'none',
                supports_image_input: false,
                supports_audio_input: false,
                supports_speech_synthesis: true,
                supports_transcription: false,
                supports_speech_translation: false,
                supports_audio_output: true,
                supports_realtime_audio: false,
                default_audio_response_format: 'pcm',
                audio_response_formats: ['pcm'],
                supports_tool_calling: false,
                builtin_tools: [],
                tool_call_format: 'none',
                prompt_renderer: 'generic',
                output_parser: 'generic',
                supports_native_multimodal: false,
              },
            },
          ],
        }), { status: 200 });
      }
      if (url === '/v1/audio/speech') {
        speechRequests += 1;
        return new Response(new Uint8Array([0, 0]), { status: 200 });
      }
      throw new Error(`unexpected request: ${url}`);
    }));

    store.dispatch(chatActions.selectModel('org/chat-model'));
    store.dispatch(chatActions.selectSpeechModel('org/chat-model'));
    store.dispatch(chatActions.setAutoSpeakAssistant(true));
    container = document.createElement('div');
    document.body.append(container);
    root = createRoot(container);
    await act(async () => {
      root?.render(
        <Provider store={store}>
          <ThemeProvider theme={darkTheme}>
            <ChatView readyInstances={[
              {
                instanceId: 'chat-instance',
                modelId: 'org/chat-model',
                sharding: 'Pipeline',
                instanceType: 'MlxRing',
                engine: 'mlx',
                nodeStatuses: [],
                status: 'ready',
              },
            ]} />
          </ThemeProvider>
        </Provider>,
      );
    });
    await act(async () => { await Promise.resolve(); });
    await waitFor(
      () => chatFormCapture.speechModelCount === 1
        && chatFormCapture.selectedSpeechModelId === 'org/chat-model'
        && chatFormCapture.autoSpeakAssistant
        && chatFormCapture.onRealtimeAssistantText !== null,
      `realtime speech callback did not become ready (speech models: ${chatFormCapture.speechModelCount}, selected speech model: ${chatFormCapture.selectedSpeechModelId}, auto speak: ${chatFormCapture.autoSpeakAssistant}, callback: ${chatFormCapture.onRealtimeAssistantText !== null}, requests: ${requestedUrls.join(', ')})`,
    );

    await act(async () => {
      chatFormCapture.onRealtimeAssistantText?.('First sentence. ', false);
      await Promise.resolve();
    });
    await waitFor(() => speechRequests === 1, 'first sentence was not synthesized');
    await act(async () => {
      chatFormCapture.onRealtimeAssistantText?.('First sentence.', true);
      await Promise.resolve();
    });
    await waitFor(() => finishPlayback.mock.calls.length === 1, 'first response did not finish');

    await act(async () => {
      chatFormCapture.onRealtimeAssistantText?.('Second sentence. ', false);
      await Promise.resolve();
    });
    await waitFor(() => speechRequests === 2, 'second sentence was not synthesized');
    await act(async () => {
      chatFormCapture.onRealtimeAssistantText?.('Second sentence.', true);
      await Promise.resolve();
    });
    await waitFor(() => finishPlayback.mock.calls.length === 2, 'second response did not finish');

    expect(speechRequests).toBe(2);
    expect(finishPlayback).toHaveBeenCalledTimes(2);
  });
});
