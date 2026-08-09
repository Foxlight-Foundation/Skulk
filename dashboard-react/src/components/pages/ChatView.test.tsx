// Copyright 2026 Foxlight Foundation

import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { userEvent } from 'vitest/browser';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { store } from '../../store';
import { chatActions } from '../../store/slices/chatSlice';
import {
  batchSpeechMaxTokens,
  DASHBOARD_SPEECH_SEED,
} from '../../audio/speechSynthesisRequest';
import { StreamingSpeechPlayback } from '../../audio/streamingSpeechPlayback';
import { darkTheme } from '../../theme/theme';
import { ChatView } from './ChatView';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

vi.mock('../../i18n/tolgee', () => {
  const translate = (_key: string, fallback: string) => fallback;
  return {
    tolgee: { getLanguage: () => 'en' },
    useSkulkTranslation: () => ({ t: translate }),
  };
});

let root: Root | null = null;
let container: HTMLDivElement | null = null;

async function renderVisionChat(): Promise<void> {
  store.dispatch(chatActions.selectModel('org/vision-model'));
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  await act(async () => {
    root?.render(
      <Provider store={store}>
        <ThemeProvider theme={darkTheme}>
          <ChatView readyInstances={[{
            instanceId: 'instance-1',
            modelId: 'org/vision-model',
            sharding: 'Pipeline',
            instanceType: 'MlxRing',
            engine: 'mlx',
            nodeStatuses: [],
            status: 'ready',
          }]} />
        </ThemeProvider>
      </Provider>,
    );
  });
}

async function waitFor(
  predicate: () => boolean,
  message: string,
): Promise<void> {
  const deadline = performance.now() + 5000;
  while (performance.now() < deadline) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  throw new Error(message);
}

async function waitForReact(
  predicate: () => boolean,
  message: string,
): Promise<void> {
  const deadline = performance.now() + 5000;
  while (performance.now() < deadline) {
    if (predicate()) return;
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 10));
    });
  }
  throw new Error(message);
}

afterEach(async () => {
  await act(async () => root?.unmount());
  for (const conversationId of Object.keys(store.getState().chat.conversations)) {
    store.dispatch(chatActions.deleteConversation(conversationId));
  }
  container?.remove();
  root = null;
  container = null;
  vi.unstubAllGlobals();
});

describe('ChatView multimodal requests', () => {
  it('sends the uploaded image data URL and retains it in the user message', async () => {
    let capturedBody: Record<string, unknown> | null = null;
    vi.stubGlobal('fetch', vi.fn(async (
      input: RequestInfo | URL,
      init?: RequestInit,
    ) => {
      const url = String(input);
      if (url === '/models') {
        return new Response(JSON.stringify({
          data: [{
            id: 'org/vision-model',
            tasks: ['TextGeneration'],
            resolved_capabilities: {
              supports_image_input: true,
              supports_thinking_toggle: false,
            },
          }],
        }), { status: 200 });
      }
      if (url === '/v1/chat/completions') {
        capturedBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
        return new Response(
          'data: {"choices":[{"delta":{"content":"seen"}}]}\n\ndata: [DONE]\n\n',
          { status: 200, headers: { 'Content-Type': 'text/event-stream' } },
        );
      }
      throw new Error(`unexpected request: ${url}`);
    }));

    await renderVisionChat();
    const fileInput = container?.querySelector<HTMLInputElement>(
      '[aria-label="Image attachment file"]',
    );
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    await waitFor(
      () => !container?.querySelector<HTMLButtonElement>(
        '[aria-label="Attach file"]',
      )?.disabled,
      'vision attachment button did not become enabled',
    );
    const fixture = new File(
      [new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10])],
      'qualification.png',
      { type: 'image/png' },
    );
    const messageInput = container?.querySelector<HTMLTextAreaElement>(
      '[aria-label="Chat message"]',
    );
    const sendButton = container?.querySelector<HTMLButtonElement>(
      '[aria-label="Send message"]',
    );
    if (fileInput === null || fileInput === undefined) {
      throw new Error('image attachment input was not rendered');
    }
    if (messageInput === null || messageInput === undefined) {
      throw new Error('chat message input was not rendered');
    }
    if (sendButton === null || sendButton === undefined) {
      throw new Error('chat send button was not rendered');
    }
    await act(async () => {
      await userEvent.upload(fileInput, fixture);
      await userEvent.fill(messageInput, 'Read the card.');
      await userEvent.click(sendButton);
    });
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    await waitFor(() => capturedBody !== null, 'chat request was not sent');

    const messages = capturedBody?.messages as Array<Record<string, unknown>>;
    const content = messages.at(-1)?.content as Array<Record<string, unknown>>;
    expect(content[0]).toMatchObject({
      type: 'image_url',
      image_url: {
        url: expect.stringMatching(/^data:image\/png;base64,/),
      },
    });
    expect(content[1]).toEqual({ type: 'text', text: 'Read the card.' });
    expect(
      container?.querySelector('[aria-label="User message"] img[alt="qualification.png"]'),
    ).not.toBeNull();
  });
});

describe('ChatView completed-message speech', () => {
  it('replays a completed assistant turn as serial sentence requests', async () => {
    const requestedInputs: string[] = [];
    const requestedSeeds: number[] = [];
    vi.stubGlobal('AudioContext', class FakeAudioContext {});
    vi.spyOn(StreamingSpeechPlayback.prototype, 'append').mockResolvedValue();
    const finishPlayback = vi
      .spyOn(StreamingSpeechPlayback.prototype, 'finish')
      .mockResolvedValue();
    vi.spyOn(StreamingSpeechPlayback.prototype, 'stop').mockImplementation(() => undefined);
    vi.stubGlobal('fetch', vi.fn(async (
      input: RequestInfo | URL,
      init?: RequestInit,
    ) => {
      const url = String(input);
      if (url === '/models') {
        return new Response(JSON.stringify({
          data: [{
            id: 'org/speech-model',
            tasks: ['TextGeneration'],
            audio: {
              default_response_format: 'pcm',
              response_formats: ['pcm'],
              supports_streaming: true,
            },
            resolved_capabilities: {
              supports_speech_synthesis: true,
              default_audio_response_format: 'pcm',
              audio_response_formats: ['pcm'],
            },
          }],
        }), { status: 200 });
      }
      if (url === '/v1/audio/speech') {
        const body = JSON.parse(String(init?.body)) as { input: string; seed: number };
        requestedInputs.push(body.input);
        requestedSeeds.push(body.seed);
        return new Response(new Uint8Array([0, 0]), { status: 200 });
      }
      throw new Error(`unexpected request: ${url}`);
    }));

    store.dispatch(chatActions.selectModel('org/speech-model'));
    store.dispatch(chatActions.selectSpeechModel('org/speech-model'));
    store.dispatch(chatActions.addMessage({
      id: 'assistant-message',
      role: 'assistant',
      content: 'First sentence. Second sentence! Final tail',
      timestamp: Date.now(),
    }));
    container = document.createElement('div');
    document.body.append(container);
    root = createRoot(container);
    await act(async () => {
      root?.render(
        <Provider store={store}>
          <ThemeProvider theme={darkTheme}>
            <ChatView readyInstances={[{
              instanceId: 'speech-instance',
              modelId: 'org/speech-model',
              sharding: 'Pipeline',
              instanceType: 'MlxRing',
              engine: 'mlx',
              nodeStatuses: [],
              status: 'ready',
            }]} />
          </ThemeProvider>
        </Provider>,
      );
    });

    await waitForReact(
      () => container?.querySelector<HTMLButtonElement>(
        '[aria-label="Speak message"]',
      ) !== null,
      'assistant replay button did not become ready',
    );
    const replay = container?.querySelector<HTMLButtonElement>(
      '[aria-label="Speak message"]',
    );
    if (!replay) throw new Error('assistant replay button was not rendered');
    await act(async () => {
      await userEvent.click(replay);
    });

    await waitForReact(
      () => requestedInputs.length === 3,
      'completed message was not synthesized sentence by sentence',
    );
    await waitForReact(
      () => finishPlayback.mock.calls.length === 1,
      'completed message playback did not drain its shared timeline',
    );
    expect(requestedInputs).toEqual([
      'First sentence.',
      'Second sentence!',
      'Final tail.',
    ]);
    expect(requestedSeeds).toEqual([
      DASHBOARD_SPEECH_SEED,
      DASHBOARD_SPEECH_SEED,
      DASHBOARD_SPEECH_SEED,
    ]);
  });

  it('replays a completed turn in one request for a batch-only speech model', async () => {
    const requestedInputs: string[] = [];
    const requestedSeeds: number[] = [];
    const requestedMaxTokens: number[] = [];
    class FakeAudio {
      onended: (() => void) | null = null;
      onerror: (() => void) | null = null;

      constructor() {}

      play(): Promise<void> {
        window.setTimeout(() => this.onended?.(), 0);
        return Promise.resolve();
      }

      pause(): void {}
    }
    vi.stubGlobal('Audio', FakeAudio);
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:batch-speech');
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined);
    vi.stubGlobal('fetch', vi.fn(async (
      input: RequestInfo | URL,
      init?: RequestInit,
    ) => {
      const url = String(input);
      if (url === '/models') {
        return new Response(JSON.stringify({
          data: [{
            id: 'org/batch-speech-model',
            tasks: ['TextGeneration'],
            audio: {
              default_response_format: 'mp3',
              response_formats: ['mp3'],
              supports_streaming: false,
            },
            resolved_capabilities: {
              supports_speech_synthesis: true,
              default_audio_response_format: 'mp3',
              audio_response_formats: ['mp3'],
            },
          }],
        }), { status: 200 });
      }
      if (url === '/v1/audio/speech') {
        const body = JSON.parse(String(init?.body)) as {
          input: string;
          max_tokens: number;
          seed: number;
        };
        requestedInputs.push(body.input);
        requestedSeeds.push(body.seed);
        requestedMaxTokens.push(body.max_tokens);
        return new Response(new Uint8Array([73, 68, 51]), {
          status: 200,
          headers: { 'Content-Type': 'audio/mpeg' },
        });
      }
      throw new Error(`unexpected request: ${url}`);
    }));

    store.dispatch(chatActions.selectModel('org/batch-speech-model'));
    store.dispatch(chatActions.selectSpeechModel('org/batch-speech-model'));
    const completeTurn = `First sentence. ${'Carefully paced words continue. '.repeat(20)}Final tail`;
    store.dispatch(chatActions.addMessage({
      id: 'batch-assistant-message',
      role: 'assistant',
      content: completeTurn,
      timestamp: Date.now(),
    }));
    container = document.createElement('div');
    document.body.append(container);
    root = createRoot(container);
    await act(async () => {
      root?.render(
        <Provider store={store}>
          <ThemeProvider theme={darkTheme}>
            <ChatView readyInstances={[{
              instanceId: 'batch-speech-instance',
              modelId: 'org/batch-speech-model',
              sharding: 'Pipeline',
              instanceType: 'MlxRing',
              engine: 'mlx_audio',
              nodeStatuses: [],
              status: 'ready',
            }]} />
          </ThemeProvider>
        </Provider>,
      );
    });

    await waitForReact(
      () => container?.querySelector<HTMLButtonElement>(
        '[aria-label="Speak message"]',
      ) !== null,
      'batch-only assistant replay button did not become ready',
    );
    const replay = container?.querySelector<HTMLButtonElement>(
      '[aria-label="Speak message"]',
    );
    if (!replay) throw new Error('batch-only assistant replay button was not rendered');
    await act(async () => {
      await userEvent.click(replay);
    });

    await waitForReact(
      () => requestedInputs.length === 1,
      'batch-only completed message was not synthesized once',
    );
    // Speech text preparation appends terminal punctuation to the
    // unpunctuated final block before synthesis.
    const spokenTurn = `${completeTurn}.`;
    expect(requestedInputs).toEqual([spokenTurn]);
    expect(requestedSeeds).toEqual([DASHBOARD_SPEECH_SEED]);
    expect(requestedMaxTokens).toEqual([batchSpeechMaxTokens(spokenTurn)]);
    expect(requestedMaxTokens[0]).toBeGreaterThan(4096);
  });
});
