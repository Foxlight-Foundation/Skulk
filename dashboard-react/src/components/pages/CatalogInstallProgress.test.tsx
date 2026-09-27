import { configureStore } from '@reduxjs/toolkit';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { apiSlice } from '../../store/api';
import { darkTheme } from '../../theme/theme';
import { readJourneys, saveJourney, type BoundJourney } from './catalogJourney';
import { CatalogInstallProgress } from './CatalogInstallProgress';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });
vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({ t: (_key: string, fallback: string, params?: Record<string, unknown>) => fallback.replace(/\{(\w+)\}/g, (_match, name: string) => String(params?.[name] ?? '')) }),
}));

const pluginId = 'managed.' + '4'.repeat(32);
const journey: BoundJourney = {
  pluginId, title: 'Example Studio', bundleId: 'example.studio', sequence: 51, publisher: 'example', runtimeDigest: '9'.repeat(64),
  transferBytes: 12_000_000, installOperationId: '5'.repeat(32), activationOperationId: null, startedAt: 1,
};
let root: Root;
let host: HTMLDivElement;
let store: ReturnType<typeof makeStore>;
let installRead: () => Response;

function makeStore() { return configureStore({ reducer: { [apiSlice.reducerPath]: apiSlice.reducer }, middleware: (defaults) => defaults().concat(apiSlice.middleware) }); }
function json(body: unknown, status = 200) { return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }); }
async function contains(text: string) { await act(async () => { await vi.waitFor(() => expect(host.textContent).toContain(text), { timeout: 5000 }); }); }

beforeEach(() => {
  localStorage.removeItem('skulk-plugin-install-journeys');
  saveJourney(journey);
  vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = input instanceof Request ? input : new Request(new URL(String(input), location.href), init);
    if (new URL(request.url).pathname.endsWith('/install')) return installRead();
    return json({ detail: 'not found' }, 404);
  });
  store = makeStore();
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
});
afterEach(async () => {
  await act(async () => { root.unmount(); store.dispatch(apiSlice.util.resetApiState()); });
  host.remove();
  vi.unstubAllGlobals();
  localStorage.removeItem('skulk-plugin-install-journeys');
});

async function render() {
  await act(async () => {
    root.render(<Provider store={store}><ThemeProvider theme={darkTheme}>
      <CatalogInstallProgress title="Example Studio" publisher="example" sequence={51} transferBytes={12_000_000} updating={false}
        journey={journey} refusal={null} onDone={() => undefined} onBack={() => undefined} pollMs={1} />
    </ThemeProvider></Provider>);
  });
}

it('keeps the install to resume when the host cannot be read', async () => {
  installRead = () => { throw new TypeError('network'); };
  await render();
  await contains('This page lost contact with the host and stopped following.');
  expect(readJourneys().map((item) => item.pluginId)).toEqual([pluginId]);
});

it('forgets the install only when the host answers that its download is not there', async () => {
  installRead = () => json({ operation: null });
  await render();
  await contains('The host never confirmed the download.');
  expect(readJourneys()).toEqual([]);
});
