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

type FetchStub = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function stubStewardFetch(options: {
  status: { enabled: boolean; present: boolean; steward_model: string | null; instance_id: string | null };
  chatReply?: { reply: string; steps: { tool: string; arguments: Record<string, unknown> }[]; steward_model: string; instance_id: string };
}): void {
  const fetchStub: FetchStub = async (input, init) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    if (url.includes('/v1/steward/chat')) {
      expect(init?.method ?? 'POST').toBe('POST');
      return jsonResponse(options.chatReply ?? { reply: '', steps: [], steward_model: '', instance_id: '' });
    }
    if (url.includes('/v1/steward')) {
      return jsonResponse(options.status);
    }
    return jsonResponse({}, 404);
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
  // Drop cached steward queries so each test's stubbed status is fetched fresh.
  store.dispatch(apiSlice.util.resetApiState());
  vi.unstubAllGlobals();
});

describe('StewardChatView', () => {
  it('shows the disabled state when intelligent fabric is off', async () => {
    stubStewardFetch({
      status: { enabled: false, present: false, steward_model: null, instance_id: null },
    });
    await renderPage();
    await waitFor(
      () => container?.textContent?.includes('Intelligent Fabric is off') ?? false,
      'disabled state never rendered',
    );
  });

  it('shows the placing state while the steward is not yet present', async () => {
    stubStewardFetch({
      status: { enabled: true, present: false, steward_model: null, instance_id: null },
    });
    await renderPage();
    await waitFor(
      () => container?.textContent?.includes('Steward is being placed') ?? false,
      'placing state never rendered',
    );
  });

  it('sends a message and renders the reply with its tool trace', async () => {
    stubStewardFetch({
      status: { enabled: true, present: true, steward_model: 'org/steward-4b', instance_id: 'inst-1' },
      chatReply: {
        reply: 'All three nodes are healthy.',
        steps: [{ tool: 'get_cluster_state', arguments: {} }],
        steward_model: 'org/steward-4b',
        instance_id: 'inst-1',
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
    expect(container?.textContent).toContain('Is the cluster healthy?');
  });
});
