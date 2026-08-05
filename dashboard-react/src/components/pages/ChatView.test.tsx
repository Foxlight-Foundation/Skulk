// Copyright 2026 Foxlight Foundation

import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { userEvent } from 'vitest/browser';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { store } from '../../store';
import { chatActions } from '../../store/slices/chatSlice';
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
