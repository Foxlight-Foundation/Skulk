import { configureStore } from '@reduxjs/toolkit';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { apiSlice } from '../../store/api';
import type { PluginServiceStatus } from '../../store/endpoints/plugins';
import { darkTheme } from '../../theme/theme';
import { PluginsPage } from './PluginsPage';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });
vi.mock('../../i18n/tolgee', () => ({ useSkulkTranslation: () => ({ t: (_key: string, fallback: string, params?: Record<string, unknown>) => fallback.replace(/\{(\w+)\}/g, (_match, name: string) => String(params?.[name] ?? '')) }) }));

const updating: PluginServiceStatus = {
  state: 'setting_up', scope: 'user', purpose: 'update', error: null,
  progress: 'Skulk was updated. Updating the plugin service to match; plugins come back in a minute or two.',
};
const ready: PluginServiceStatus = { state: 'ready', scope: 'user', purpose: null, progress: null, error: null };

let root: Root;
let host: HTMLDivElement;
let store: ReturnType<typeof makeStore>;
let service: PluginServiceStatus;
// How the inventory read answers: null serves it, a number fails it with that status.
let inventoryFailure: number | null;
let inventoryReads: number;

function makeStore() {
  return configureStore({ reducer: { [apiSlice.reducerPath]: apiSlice.reducer }, middleware: (defaults) => defaults().concat(apiSlice.middleware) });
}
function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}
async function contains(text: string) {
  await act(async () => { await vi.waitFor(() => expect(host.textContent).toContain(text), { timeout: 10_000 }); });
}
function buttons(label: string) {
  return [...host.querySelectorAll('button')].filter((item) => item.textContent?.trim() === label);
}

beforeEach(() => {
  service = updating;
  inventoryFailure = 409;
  inventoryReads = 0;
  vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = new Request(input, init);
    const path = new URL(request.url).pathname;
    if (path === '/state') return json({});
    if (path === '/node_id') return json('host-node');
    if (path === '/node/identity') return json({ nodeId: 'host-node', friendlyName: 'host' });
    if (path === '/v1/plugins/managed/service') return json(service);
    if (path === '/v1/plugins/managed') {
      inventoryReads += 1;
      // While the manager runs the previous build the host refuses the read.
      if (inventoryFailure === 403) return json({ detail: 'plugin management requires protected transport' }, 403);
      if (inventoryFailure !== null) return json({ detail: 'local plugin operation refused; refresh selection and operation status' }, inventoryFailure);
      return json({ installations: [] });
    }
    if (path === '/v1/plugins') {
      if (inventoryFailure !== null) return json({ detail: 'local plugin manager is unavailable' }, 503);
      return json([{ pluginId: 'bridge', available: true, nodes: [{ nodeId: 'node-1', bundleId: 'test.bundle', version: '1.0.0', status: 'ready', configurable: true }] }]);
    }
    if (path === '/v1/plugins/managed/catalog/source') return json({ revision: 1, configured: false, credential_reference: null, credential_ready: false, trust_revision: 0, builtin_store: false, builtin_store_available: false });
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
  history.replaceState(null, '', '/');
});

async function renderPage() {
  await act(async () => { root.render(<Provider store={store}><ThemeProvider theme={darkTheme}><PluginsPage /></ThemeProvider></Provider>); });
}

/** What the page must not say or offer while the host updates its plugin service. */
function expectNoFailureShown() {
  const text = host.textContent ?? '';
  expect(text).not.toContain('Plugin inventory unavailable');
  expect(text).not.toContain('The plugin service is not answering');
  expect(text).not.toContain('Inventory unavailable.');
  expect(text).not.toContain('permissions');
  expect(text).not.toContain('Plugin management is unavailable');
  expect(buttons('Retry inventory')).toEqual([]);
  expect(buttons('Refresh')).toEqual([]);
  expect(buttons('Add plugin')).toEqual([]);
  // Nothing the owner can reach is left disabled: the tabs and access stay usable.
  expect([...host.querySelectorAll('button')].filter((item) => item.disabled)).toEqual([]);
}

it('shows the update on Installed instead of an unavailable inventory, then the plugins', async () => {
  await renderPage();
  await contains('Setting up plugins on this host');
  await contains('Skulk was updated. Updating the plugin service to match');
  expect(host.textContent).toContain('Updating the plugin service…');
  expect(host.textContent).toContain('Installed plugins and their settings are kept.');
  expect(host.textContent).not.toContain('This happens once');
  expect(host.querySelector('[role=tab][aria-selected=true]')?.textContent).toBe('Installed');
  expectNoFailureShown();

  // The page keeps polling and continues by itself once the service is back.
  const readsBefore = inventoryReads;
  service = ready;
  inventoryFailure = null;
  await contains('test.bundle');
  expect(host.querySelector('[data-testid=plugin-service-setting-up]')).toBeNull();
  expect(host.textContent).not.toContain('Updating the plugin service…');
  expect(host.textContent).toContain('1 installed · 0 healthy.');
  expect(inventoryReads).toBeGreaterThan(readsBefore);
  expect(buttons('Add plugin')[0]?.disabled).toBe(false);
});

it('shows the update on Browse too, and Browse again once ready', async () => {
  history.replaceState(null, '', '/?view=browse');
  await renderPage();
  await contains('Setting up plugins on this host');
  expect(host.querySelector('[role=tab][aria-selected=true]')?.textContent).toBe('Browse');
  expect(host.textContent).toContain('Updating the plugin service…');
  expectNoFailureShown();

  // Switching tabs keeps the same notice rather than an error on the other tab.
  await act(async () => { buttons('Installed')[0].click(); });
  await contains('Setting up plugins on this host');
  expectNoFailureShown();
  await act(async () => { buttons('Browse')[0].click(); });

  service = ready;
  inventoryFailure = null;
  await act(async () => { await vi.waitFor(() => expect(host.querySelector('[data-testid=plugin-service-setting-up]')).toBeNull(), { timeout: 10_000 }); });
  expect(host.querySelector('[role=tab][aria-selected=true]')?.textContent).toBe('Browse');
  expect(host.textContent).not.toContain('Updating the plugin service…');
});

it('says a first setup is setting up, not updating', async () => {
  service = { state: 'setting_up', scope: 'user', purpose: 'setup', progress: 'Preparing verified independent manager runtime...', error: null };
  await renderPage();
  await contains('Preparing verified independent manager runtime...');
  expect(host.textContent).toContain('Setting up the plugin service…');
  expect(host.textContent).toContain('This happens once');
  expectNoFailureShown();
});

it('blames access only when the host refused it', async () => {
  service = ready;
  inventoryFailure = 403;
  await renderPage();
  await contains('Plugin inventory unavailable');
  expect(host.textContent).toContain('This browser does not have access to this host’s plugins.');
  expect(host.textContent).toContain('The host said: plugin management requires protected transport');
  expect(host.textContent).not.toContain('did not answer');
});

it('says the plugin service did not answer when the host failed for another reason', async () => {
  service = ready;
  inventoryFailure = 503;
  await renderPage();
  await contains('Plugin inventory unavailable');
  expect(host.textContent).toContain('The host’s plugin service did not answer.');
  expect(host.textContent).toContain('Plugin nodes could not be read from this host just now.');
  expect(host.textContent).not.toContain('does not have access');
  expect(host.textContent).not.toContain('Plugin management is unavailable');
  expect(host.textContent).not.toContain('permissions');
  expect(host.textContent).not.toContain('The host said');
});
