import { configureStore } from '@reduxjs/toolkit';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { afterEach, expect, it, vi } from 'vitest';
import { apiSlice } from '../../store/api';
import type { PluginServiceStatus } from '../../store/endpoints/plugins';
import { darkTheme } from '../../theme/theme';
import { PluginServiceSetup } from './PluginServiceSetup';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });
vi.mock('../../i18n/tolgee', () => ({ useSkulkTranslation: () => ({ t: (_key: string, fallback: string) => fallback }) }));

let root: Root | null = null;
let host: HTMLDivElement | null = null;
let posts = 0;

function status(state: PluginServiceStatus['state'], extra: Partial<PluginServiceStatus> = {}): PluginServiceStatus {
  return { state, scope: null, progress: null, error: null, ...extra };
}

function serve(answer: (posts: number) => PluginServiceStatus | null) {
  posts = 0;
  vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = new Request(input, init);
    expect(request.headers.get('X-Skulk-Dashboard')).toBe('pairing-v1');
    if (request.method === 'POST') posts += 1;
    const body = answer(posts);
    return body === null
      ? new Response('{}', { status: 404 })
      : new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } });
  });
}

async function render(direct: boolean) {
  const store = configureStore({ reducer: { [apiSlice.reducerPath]: apiSlice.reducer }, middleware: (defaults) => defaults().concat(apiSlice.middleware) });
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  await act(async () => {
    root?.render(<Provider store={store}><ThemeProvider theme={darkTheme}>
      <PluginServiceSetup direct={direct} header={<h1>Plugins</h1>}><p>plugin content</p></PluginServiceSetup>
    </ThemeProvider></Provider>);
  });
}

afterEach(async () => {
  await act(async () => root?.unmount());
  host?.remove();
  root = null;
  host = null;
  vi.unstubAllGlobals();
});

it('sets up the plugin service once on the first owner visit and shows progress', async () => {
  serve((count) => count === 0 ? status('absent') : status('setting_up', { scope: 'user', progress: 'Preparing verified independent manager runtime...' }));
  await render(true);
  await act(async () => { await vi.waitFor(() => expect(host?.textContent).toContain('Preparing verified independent manager runtime...')); });
  expect(posts).toBe(1);
  expect(host?.textContent).toContain('Setting up plugins on this host');
  expect(host?.textContent).toContain('Plugins');
  expect(host?.textContent).not.toContain('plugin content');
});

it('never starts setup from a paired browser', async () => {
  serve(() => status('absent'));
  await render(false);
  await act(async () => { await vi.waitFor(() => expect(host?.textContent).toContain('Plugins are not set up on this host yet')); });
  expect(posts).toBe(0);
});

it('shows the plugins once the service is ready', async () => {
  serve(() => status('ready', { scope: 'user' }));
  await render(true);
  await act(async () => { await vi.waitFor(() => expect(host?.textContent).toContain('plugin content')); });
  expect(posts).toBe(0);
});

it('shows why setup failed and retries on request', async () => {
  serve(() => status('failed', { error: 'service interpreter and Skulk configuration must be outside Git checkouts' }));
  await render(true);
  await act(async () => { await vi.waitFor(() => expect(host?.textContent).toContain('outside Git checkouts')); });
  expect(posts).toBe(0);
  const retry = [...(host?.querySelectorAll('button') ?? [])].find((button) => button.textContent === 'Try again');
  await act(async () => { retry?.click(); });
  await act(async () => { await vi.waitFor(() => expect(posts).toBe(1)); });
});

it('keeps an older host without the status route working', async () => {
  serve(() => null);
  await render(true);
  await act(async () => { await vi.waitFor(() => expect(host?.textContent).toContain('plugin content')); });
});

it('offers setup again when the host comes back without it instead of spinning', async () => {
  // The first answer starts setup; the host then restarts and reports absent.
  serve(() => status('absent'));
  await render(true);
  await act(async () => { await vi.waitFor(() => expect(posts).toBe(1)); });
  await act(async () => { await vi.waitFor(() => expect(host?.textContent).toContain('Set up plugins')); });
  expect(host?.textContent).not.toContain('Setting up plugins on this host');
  const setUp = [...(host?.querySelectorAll('button') ?? [])].find((button) => button.textContent === 'Set up plugins');
  await act(async () => { setUp?.click(); });
  await act(async () => { await vi.waitFor(() => expect(posts).toBe(2)); });
});
