import { configureStore } from '@reduxjs/toolkit';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { apiSlice } from '../../store/api';
import { uiSliceReducer } from '../../store/slices/uiSlice';
import { darkTheme } from '../../theme/theme';
import type { CapabilityNodeSummary } from '../../types/capabilityNodes';
import { CapabilitySetupPanel } from './CapabilitySetupPanel';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });
vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({ t: (_key: string, fallback: string, params?: Record<string, unknown>) => fallback.replace(/\{(\w+)\}/g, (_match, name: string) => String(params?.[name] ?? '')) }),
}));

const pluginId = 'managed.' + '2'.repeat(32);
const screenUrl = 'https://studio.example/s/token/';
let root: Root;
let host: HTMLDivElement;
let store: ReturnType<typeof makeStore>;
let summaries: CapabilityNodeSummary[];
let hostNode: string;
let preflightAvailable: boolean;

function makeStore() {
  return configureStore({ reducer: { [apiSlice.reducerPath]: apiSlice.reducer, ui: uiSliceReducer }, middleware: (defaults) => defaults().concat(apiSlice.middleware) });
}
function json(body: unknown, status = 200) { return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }); }
async function contains(text: string) { await act(async () => { await vi.waitFor(() => expect(host.textContent).toContain(text), { timeout: 5000 }); }); }
function button(label: string) { return [...host.querySelectorAll('button')].find((item) => item.textContent === label) ?? null; }
function summary(overrides: Partial<CapabilityNodeSummary> = {}): CapabilityNodeSummary {
  return {
    pluginId, nodeId: 'studio', bundleId: 'example.studio', version: '0.1.0', title: 'Example Studio', status: 'ready', ownerAvailable: true,
    surfaces: [{ surfaceId: 'studio', title: 'Example Studio', kind: 'link', url: screenUrl, ready: true }], actions: [], operationsActive: 0,
    observedAt: '2026-09-27T12:00:00Z', ...overrides,
  };
}

beforeEach(async () => {
  summaries = [summary()];
  hostNode = 'host-node';
  preflightAvailable = true;
  vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = input instanceof Request ? input : new Request(new URL(String(input), location.href), init);
    const path = new URL(request.url).pathname;
    if (path === '/node_id') return json('host-node');
    if (path === '/node/identity') return json({ nodeId: 'host-node', friendlyName: 'host' });
    if (path === '/state') return json({ instances: {}, runners: {}, capabilityNodes: { [hostNode]: summaries } });
    if (path === '/v1/plugins') {
      return json([{ pluginId, available: true, nodes: [{ nodeId: 'studio', bundleId: 'example.studio', version: '0.1.0', status: 'configuration_invalid', configurable: true, preflightAvailable }] }]);
    }
    if (path === `/v1/plugins/${pluginId}/nodes/studio/preflight`) {
      return json({ nodeId: 'studio', revision: 2, schemaDigest: 'a'.repeat(64), valuesDigest: 'b'.repeat(64), credentialRevision: null, observedAt: 1_790_000_000,
        checks: [{ code: 'storage', passed: true, correctiveAction: null }, { code: 'engine', passed: false, correctiveAction: 'Place a model on a node that can serve it.' }] });
    }
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
});

async function render(onManage?: (pluginId: string) => void) {
  await act(async () => {
    root.render(<Provider store={store}><ThemeProvider theme={darkTheme}>
      <CapabilitySetupPanel target={{ pluginId, title: 'Example Studio' }} onManage={onManage} onBack={() => undefined} />
    </ThemeProvider></Provider>);
  });
}

it('reads a plugin that reports ready and opens its screen', async () => {
  await render(() => undefined);
  await contains('Example Studio is ready');
  await contains('It reports ready.');
  const open = host.querySelector('a[href]') as HTMLAnchorElement;
  expect(open.href).toBe(screenUrl);
  expect(open.textContent).toContain('Open Example Studio');
  // One node named like the plugin is its status line.
  await contains('StatusReady.');
  expect(button('Manage plugin')).not.toBeNull();
  // Nothing is needed, so no node asks for its settings and no checks run.
  expect(button('Open settings')).toBeNull();
  expect(button('Check setup')).toBeNull();
});

it('says what a plugin waiting for setup needs, with its own checks and its settings', async () => {
  summaries = [summary({ status: 'configuration_invalid', surfaces: [] })];
  const onManage = vi.fn();
  await render(onManage);
  await contains('Set up Example Studio');
  await contains('It needs you before it can run.');
  await contains('Waiting for setup: its settings, credentials or setup checks are not complete yet.');
  await contains('Its screens appear here once it is ready.');
  expect(host.querySelector('a[href]')).toBeNull();
  await act(async () => { await vi.waitFor(() => expect(button('Check setup')).not.toBeNull(), { timeout: 5000 }); });
  await act(async () => { button('Check setup')!.click(); });
  await contains('Place a model on a node that can serve it.');
  await act(async () => { button('Open settings')!.click(); });
  expect(onManage).toHaveBeenCalledWith(pluginId);
});

it('waits for a plugin the host has not reported, and shows a starting one as waiting', async () => {
  summaries = [];
  await render();
  await contains('Waiting for it to report…');
  expect(button('Manage plugin')).toBeNull();
  summaries = [summary({ status: 'starting', surfaces: [] })];
  preflightAvailable = false;
  await act(async () => { root.unmount(); store.dispatch(apiSlice.util.resetApiState()); });
  root = createRoot(host);
  await render();
  await contains('It is starting.');
  await contains('Starting…');
  expect(button('Open settings')).toBeNull();
});

it('keeps a loopback screen closed to a browser on another machine', async () => {
  // The plugin runs on another node and publishes its screen on loopback.
  hostNode = 'other-node';
  summaries = [summary({ surfaces: [{ surfaceId: 'studio', title: 'Example Studio', kind: 'link', url: 'http://127.0.0.1:54905/s/token/', ready: true }] })];
  await render(() => undefined);
  await contains('Example Studio is ready');
  expect(host.querySelector('a[href]')).toBeNull();
  expect(button('Open Example Studio')?.disabled).toBe(true);
  await contains('Example Studio opens only from a browser running on the host that runs it.');
});
