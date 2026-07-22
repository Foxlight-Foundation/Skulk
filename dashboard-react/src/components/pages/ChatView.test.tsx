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
import type { InstanceCardData } from '../layout/InstancePanel';
import { ChatView } from './ChatView';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const translate = (_key: string, fallback: string) => fallback;

vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({
    t: translate,
  }),
}));

const MODEL_ID = 'mlx-community/Qwen3.6-27B-4bit';
const READY_INSTANCE: InstanceCardData = {
  instanceId: 'vision-instance',
  modelId: MODEL_ID,
  sharding: 'Pipeline',
  instanceType: 'MlxRing',
  engine: 'mlx',
  nodeStatuses: [],
  status: 'ready',
};

let root: Root | null = null;
let container: HTMLDivElement | null = null;

async function renderChatView(): Promise<void> {
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  await act(async () => {
    root?.render(
      <Provider store={store}>
        <ThemeProvider theme={darkTheme}>
          <ChatView readyInstances={[READY_INSTANCE]} />
        </ThemeProvider>
      </Provider>,
    );
  });
}

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  root = null;
  container = null;
  vi.unstubAllGlobals();
});

describe('ChatView vision requests', () => {
  it('uploads exact image bytes and renders the streamed vision answer', async () => {
    let chatRequest: Record<string, unknown> | null = null;
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url === '/models') {
        return Response.json({
          data: [{
            id: MODEL_ID,
            tasks: ['TextGeneration'],
            capabilities: ['text', 'vision'],
            resolved_capabilities: {
              supports_image_input: true,
              supports_thinking_toggle: false,
            },
          }],
        });
      }
      if (url === '/v1/chat/completions') {
        chatRequest = JSON.parse(String(init?.body)) as Record<string, unknown>;
        const events = [
          'data: {"choices":[{"delta":{"content":"Elon Musk is wearing a dark suit jacket and white shirt against a black background."}}]}',
          'data: [DONE]',
          '',
        ].join('\n\n');
        return new Response(events, {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        });
      }
      throw new Error(`Unexpected dashboard request: ${url}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    store.dispatch(chatActions.selectModel(MODEL_ID));

    await renderChatView();
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    await vi.waitFor(() => {
      expect(container?.querySelector<HTMLButtonElement>('[aria-label="Attach file"]')?.disabled)
        .toBe(false);
    });

    const input = container?.querySelector<HTMLInputElement>('input[type="file"][accept="image/*"]');
    const portrait = new File(['portrait-bytes'], 'portrait.png', { type: 'image/png' });
    await act(async () => {
      await userEvent.upload(input!, portrait);
    });
    expect(container?.textContent).toContain('portrait.png');

    const textarea = container?.querySelector<HTMLTextAreaElement>('textarea');
    await act(async () => {
      await userEvent.fill(textarea!, 'Identify this person and describe the clothing and background.');
      await userEvent.click(
        container!.querySelector<HTMLButtonElement>('[aria-label="Send message"]')!,
      );
    });

    await vi.waitFor(() => {
      expect(container?.textContent).toContain('Elon Musk is wearing a dark suit jacket');
      expect(chatRequest).not.toBeNull();
    });
    const messages = chatRequest?.messages as Array<{ role: string; content: unknown }>;
    const userContent = messages.at(-1)?.content as Array<{
      type: string;
      image_url?: { url: string };
      text?: string;
    }>;
    expect(userContent).toEqual([
      {
        type: 'image_url',
        image_url: { url: 'data:image/png;base64,cG9ydHJhaXQtYnl0ZXM=' },
      },
      {
        type: 'text',
        text: 'Identify this person and describe the clothing and background.',
      },
    ]);
  });
});
