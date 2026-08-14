// Copyright 2026 Foxlight Foundation

import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { store } from '../../store';
import { chatActions } from '../../store/slices/chatSlice';
import { darkTheme } from '../../theme/theme';
import { ChatView } from './ChatView';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

type RealtimeTranscriptHandler = (text: string, final: boolean) => void;

const chatFormCapture = vi.hoisted(() => ({
  onRealtimeTranscript: null as RealtimeTranscriptHandler | null,
  autoSubmitVoice: false,
  propNames: [] as string[],
}));

vi.mock('../chat/ChatForm', () => ({
  ChatForm: (props: {
    onRealtimeTranscript?: RealtimeTranscriptHandler;
    autoSubmitVoice?: boolean;
  }) => {
    chatFormCapture.onRealtimeTranscript = props.onRealtimeTranscript ?? null;
    chatFormCapture.autoSubmitVoice = props.autoSubmitVoice ?? false;
    chatFormCapture.propNames = Object.keys(props);
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
  store.dispatch(chatActions.setAutoSubmitVoice(false));
  container?.remove();
  root = null;
  container = null;
  chatFormCapture.onRealtimeTranscript = null;
  chatFormCapture.autoSubmitVoice = false;
  chatFormCapture.propNames = [];
  vi.unstubAllGlobals();
});

describe('ChatView realtime voice auto-send', () => {
  it('submits the final transcript through normal chat with dashboard history', async () => {
    let capturedBody: Record<string, unknown> | null = null;
    vi.stubGlobal('fetch', vi.fn(async (
      input: RequestInfo | URL,
      init?: RequestInit,
    ) => {
      const url = String(input);
      if (url === '/models') {
        return new Response(JSON.stringify({
          data: [{
            id: 'org/chat-model',
            tasks: ['TextGeneration'],
            resolved_capabilities: {
              supports_thinking_toggle: false,
              supports_image_input: false,
            },
          }],
        }), { status: 200 });
      }
      if (url === '/v1/chat/completions') {
        capturedBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
        return new Response(
          'data: {"choices":[{"delta":{"content":"Complete voice reply."}}]}\n\ndata: [DONE]\n\n',
          { status: 200, headers: { 'Content-Type': 'text/event-stream' } },
        );
      }
      throw new Error(`unexpected request: ${url}`);
    }));

    store.dispatch(chatActions.selectModel('org/chat-model'));
    store.dispatch(chatActions.setAutoSubmitVoice(true));
    store.dispatch(chatActions.addMessage({
      id: 'earlier-user',
      role: 'user',
      content: 'Earlier context.',
      timestamp: 1,
    }));
    store.dispatch(chatActions.addMessage({
      id: 'earlier-assistant',
      role: 'assistant',
      content: 'Earlier answer.',
      timestamp: 2,
    }));

    container = document.createElement('div');
    document.body.append(container);
    root = createRoot(container);
    await act(async () => {
      root?.render(
        <Provider store={store}>
          <ThemeProvider theme={darkTheme}>
            <ChatView readyInstances={[{
              instanceId: 'chat-instance',
              modelId: 'org/chat-model',
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

    await waitFor(
      () => chatFormCapture.autoSubmitVoice
        && chatFormCapture.onRealtimeTranscript !== null,
      'realtime transcript callback did not become ready',
    );
    await act(async () => {
      chatFormCapture.onRealtimeTranscript?.('Continue that', false);
      await Promise.resolve();
    });
    expect(capturedBody).toBeNull();

    await act(async () => {
      chatFormCapture.onRealtimeTranscript?.('Continue that thought.', true);
      await Promise.resolve();
    });
    await waitFor(() => capturedBody !== null, 'voice transcript was not sent to chat');

    const request = capturedBody as Record<string, unknown>;
    expect(request).not.toHaveProperty('max_tokens');
    expect(request.messages).toEqual([
      { role: 'user', content: 'Earlier context.' },
      { role: 'assistant', content: 'Earlier answer.' },
      { role: 'user', content: 'Continue that thought.' },
    ]);
    expect(chatFormCapture.propNames).not.toContain('realtimeResponseModelId');
    expect(chatFormCapture.propNames).not.toContain('onRealtimeAssistantText');
    expect(chatFormCapture.propNames).not.toContain('onRealtimeResponseDone');
    await waitFor(
      () => store.getState().chat.conversations[
        store.getState().chat.activeConversationId ?? ''
      ]?.messages.at(-1)?.content === 'Complete voice reply.',
      'normal chat response was not retained in dashboard history',
    );
  });
});
