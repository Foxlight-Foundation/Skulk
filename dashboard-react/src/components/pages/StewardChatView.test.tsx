// Copyright 2026 Foxlight Foundation

import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { userEvent } from 'vitest/browser';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { store } from '../../store';
import { apiSlice } from '../../store/api';
import { darkTheme } from '../../theme/theme';
import { StewardChatView } from './StewardChatView';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

vi.mock('../../i18n/tolgee', () => {
  const translate = (
    _key: string,
    fallback: string,
    params?: Record<string, string>,
  ) => {
    if (!params) return fallback;
    return Object.entries(params).reduce(
      (text, [name, value]) => text.replace(`{${name}}`, value),
      fallback,
    );
  };
  return {
    useSkulkTranslation: () => ({ t: translate }),
  };
});

let root: Root | null = null;
let container: HTMLDivElement | null = null;

interface StewardStatusFixture {
  enabled: boolean;
  present: boolean;
  ready: boolean;
  steward_model: string | null;
  instance_id: string | null;
}

function sseBody(events: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const event of events) {
        controller.enqueue(encoder.encode(`data: ${event}\n\n`));
      }
      controller.enqueue(encoder.encode('data: [DONE]\n\n'));
      controller.close();
    },
  });
}

function delta(delta: Record<string, string>): string {
  return JSON.stringify({ choices: [{ index: 0, delta }] });
}

function stubFetch(options: {
  status: StewardStatusFixture;
  sseEvents?: string[];
  onChat?: (init: RequestInit | undefined) => void;
}): void {
  const fetchStub = async (
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<Response> => {
    const url =
      typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    if (url.includes('/v1/chat/completions')) {
      options.onChat?.(init);
      return new Response(sseBody(options.sseEvents ?? []), {
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
      });
    }
    if (url.includes('/v1/steward')) {
      return new Response(JSON.stringify(options.status), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }
    return new Response('{}', { status: 404 });
  };
  vi.stubGlobal('fetch', fetchStub);
}

async function renderPage(): Promise<void> {
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  await act(async () => {
    root?.render(
      <Provider store={store}>
        <ThemeProvider theme={darkTheme}>
          <StewardChatView />
        </ThemeProvider>
      </Provider>,
    );
  });
}

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
  container?.remove();
  root = null;
  container = null;
  store.dispatch(apiSlice.util.resetApiState());
  vi.unstubAllGlobals();
});

const READY: StewardStatusFixture = {
  enabled: true,
  present: true,
  ready: true,
  steward_model: 'org/steward-4b',
  instance_id: 'inst-1',
};

describe('StewardChatView', () => {
  it('shows the disabled state when intelligent fabric is off', async () => {
    stubFetch({ status: { ...READY, enabled: false, present: false, ready: false } });
    await renderPage();
    await waitFor(
      () => container?.textContent?.includes('Intelligent Fabric is off') ?? false,
      'disabled state never rendered',
    );
  });

  it('holds the placing state until the steward is ready, not merely present', async () => {
    stubFetch({ status: { ...READY, ready: false } });
    await renderPage();
    await waitFor(
      () => container?.textContent?.includes('Steward is being placed') ?? false,
      'placing state never rendered',
    );
    expect(container?.querySelector('textarea')).toBeNull();
  });

  it('streams a turn over chat-completions with the reserved model id', async () => {
    let chatInit: RequestInit | undefined;
    stubFetch({
      status: READY,
      sseEvents: [
        delta({ reasoning_content: 'get_cluster_state\n' }),
        delta({ content: 'All three nodes are healthy.' }),
      ],
      onChat: (init) => {
        chatInit = init;
      },
    });
    await renderPage();
    await waitFor(
      () => container?.textContent?.includes('Ask the cluster') ?? false,
      'empty chat state never rendered',
    );

    const textarea = container?.querySelector('textarea');
    expect(textarea).not.toBeNull();
    await userEvent.fill(textarea as HTMLTextAreaElement, 'Is the cluster healthy?');
    await userEvent.keyboard('{Enter}');

    await waitFor(
      () => container?.textContent?.includes('All three nodes are healthy.') ?? false,
      'steward reply never rendered',
    );
    expect(chatInit?.method).toBe('POST');
    const body: unknown = JSON.parse(String(chatInit?.body));
    expect((body as { model?: string }).model).toBe('skulk/steward');
    expect((body as { stream?: boolean }).stream).toBe(true);
    expect(container?.textContent).toContain('Is the cluster healthy?');
  });
});
