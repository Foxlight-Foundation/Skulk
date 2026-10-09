import { configureStore } from '@reduxjs/toolkit';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { apiSlice } from '../../store/api';
import { clusterApi } from '../../store/endpoints/cluster';
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
let enableStatus: number;
let posts: { path: string; body: Record<string, unknown> }[];
// The node's own configuration, which a status report can lag.
let configEnabled: boolean;
// What the host reports once the node is turned on; a test can make it race ahead.
let afterEnable: () => void;
// How long the reply to turning it on takes after the host has acted.
let enableReplyMs: number;

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
  enableStatus = 200;
  configEnabled = false;
  afterEnable = () => { summaries = [summary({ status: 'starting', surfaces: [] })]; };
  enableReplyMs = 0;
  posts = [];
  vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = input instanceof Request ? input : new Request(new URL(String(input), location.href), init);
    const path = new URL(request.url).pathname;
    if (path === `/v1/plugins/${pluginId}/nodes/studio/configuration`) {
      if (request.method === 'GET') return json({ nodeId: 'studio', revision: 2, schemaDigest: 'a'.repeat(64), configurationSchema: { type: 'object', properties: { tags: { type: 'array' } } }, values: {}, enabled: configEnabled });
      const body = await request.json() as Record<string, unknown>;
      posts.push({ path, body });
      if (enableStatus !== 200) return json({ detail: 'configuration refused; reload settings and validation' }, enableStatus);
      configEnabled = true;
      afterEnable();
      if (enableReplyMs > 0) await new Promise((resolve) => setTimeout(resolve, enableReplyMs));
      return json({ configuration: { nodeId: 'studio', revision: 3, schemaDigest: 'a'.repeat(64), configurationSchema: {}, values: {}, enabled: true }, validated: true });
    }
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

async function render(onManage?: (pluginId: string, nodeId?: string) => void, onDone: () => void = () => undefined, spendsMoney?: boolean, turnedOn?: string[]) {
  await act(async () => {
    root.render(<Provider store={store}><ThemeProvider theme={darkTheme}>
      <CapabilitySetupPanel target={{ pluginId, title: 'Example Studio', spendsMoney, turnedOn }} onManage={onManage} onDone={onDone} />
    </ThemeProvider></Provider>);
  });
}

it('reads a plugin that reports ready and opens its screen in a new tab', async () => {
  const onDone = vi.fn();
  await render(() => undefined, onDone);
  await contains('Example Studio is ready');
  await contains('It is running.');
  const open = host.querySelector('a[href]') as HTMLAnchorElement;
  expect(open.href).toBe(screenUrl);
  expect(open.target).toBe('_blank');
  expect(open.textContent).toContain('Open Example Studio');
  await contains('Opens in a new tab. You can also open it from its node in the Cluster view.');
  // One node named like the plugin is its status line.
  await contains('StatusRunning.');
  expect(button('Settings')).not.toBeNull();
  // Nothing is needed, so no node asks for its settings and no checks run.
  expect(button('Open settings')).toBeNull();
  expect(button('Check setup')).toBeNull();
  expect(button('Turn on')).toBeNull();
  // The page is left for Installed, never "back" to Browse.
  expect(host.textContent).not.toContain('Back to Browse');
  await act(async () => { button('Done')!.click(); });
  expect(onDone).toHaveBeenCalledTimes(1);
});

it('says a plugin service that is not answering is not a settings problem', async () => {
  // The restart after a Skulk update: the last report may even say it needs setup.
  summaries = [summary({ ownerAvailable: false, status: 'configuration_invalid', surfaces: [] })];
  await render(vi.fn());
  await contains('Its plugin service is not answering. After a Skulk update that takes a few minutes; this page updates by itself.');
  await contains('The plugin service on this host is not answering. It reports again once it is back.');
  await contains('Not answering');
  expect(host.textContent).not.toContain('Needs settings');
  expect(host.textContent).not.toContain('It needs its settings before it can run.');
  expect(button('Open settings')).toBeNull();
  expect(button('Turn on')).toBeNull();
});

it('says what a plugin waiting for setup needs, with its own checks and the one setting to open', async () => {
  summaries = [summary({ status: 'configuration_invalid', surfaces: [] })];
  const onManage = vi.fn();
  await render(onManage);
  await contains('Example Studio is installed');
  await contains('It needs its settings before it can run.');
  await contains('Waiting for setup: its settings, credentials or setup checks are not complete yet.');
  await contains('Once it is running, Open Example Studio appears here and opens it in a new tab.');
  expect(host.textContent).not.toContain('screens appear here');
  expect(host.querySelector('a[href]')).toBeNull();
  // Turning on is not the next step, so it is not offered.
  expect(button('Turn on')).toBeNull();
  await act(async () => { await vi.waitFor(() => expect(button('Check setup')).not.toBeNull(), { timeout: 5000 }); });
  await act(async () => { button('Check setup')!.click(); });
  await contains('Place a model on a node that can serve it.');
  // Check codes read in plain words.
  expect(host.textContent).toContain('Storage');
  await act(async () => { button('Open settings')!.click(); });
  expect(onManage).toHaveBeenCalledWith(pluginId, 'studio');
});

it('turns a plugin that is only off on with one action, whatever its settings form holds', async () => {
  summaries = [summary({ status: 'disabled', surfaces: [] })];
  await render(() => undefined, () => undefined, false);
  await contains('It is turned off. Turn it on to start it.');
  await contains('Turning it on runs its setup checks again. It cannot spend money.');
  // No checks until something more than turning on is needed.
  expect(button('Check setup')).toBeNull();
  await act(async () => { await vi.waitFor(() => expect(button('Turn on')).not.toBeNull(), { timeout: 5000 }); });
  await act(async () => { button('Turn on')!.click(); });
  await act(async () => { await vi.waitFor(() => expect(posts).toHaveLength(1), { timeout: 5000 }); });
  expect(posts[0].body).toEqual({ operation: 'enable', expectedRevision: 2, expectedSchemaDigest: 'a'.repeat(64) });
  await contains('Starting…');
  expect(button('Turn on')).toBeNull();
});

it('says nothing about money when the release is not known to be unable to spend it', async () => {
  summaries = [summary({ status: 'disabled', surfaces: [] })];
  await render(() => undefined, () => undefined, true);
  await contains('Turning it on runs its setup checks again.');
  expect(host.textContent).not.toContain('cannot spend money');
});

it('stays on Turn on with the host\'s reason and its checks when turning on is refused', async () => {
  summaries = [summary({ status: 'disabled', surfaces: [] })];
  enableStatus = 409;
  const onManage = vi.fn();
  await render(onManage);
  await act(async () => { await vi.waitFor(() => expect(button('Turn on')).not.toBeNull(), { timeout: 5000 }); });
  await act(async () => { button('Turn on')!.click(); });
  await contains('It was not turned on. The host said: configuration refused; reload settings and validation');
  // Its own checks run at once and say what is missing.
  await contains('Place a model on a node that can serve it.');
  expect(button('Turn on')).not.toBeNull();
  await act(async () => { button('Open settings')!.click(); });
  expect(onManage).toHaveBeenCalledWith(pluginId, 'studio');
});

it('waits for a plugin the host has not reported, and shows a starting one as waiting', async () => {
  summaries = [];
  await render();
  await contains('Waiting for it to report…');
  expect(button('Settings')).toBeNull();
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

// The owner turning it off from its settings, as the next report and its configuration show.
function turnedOffElsewhere(observedAt: string) {
  configEnabled = false;
  summaries = [summary({ status: 'disabled', surfaces: [], observedAt })];
}
async function offerTurnOn() {
  await contains('Turned off.');
  await act(async () => { await vi.waitFor(() => expect(button('Turn on')).not.toBeNull(), { timeout: 5000 }); });
  expect(host.textContent).not.toContain('Turning on…');
}

it('shows a node turned on and then off from its settings in one visit as off, with Turn on, even when the report raced ahead', async () => {
  summaries = [summary({ status: 'disabled', surfaces: [], observedAt: '2026-10-08T12:00:00Z' })];
  // The host reports it running before the reply to turning it on arrives.
  afterEnable = () => { summaries = [summary({ observedAt: '2026-10-08T12:00:05Z' })]; };
  enableReplyMs = 2500;
  await render(() => undefined);
  await act(async () => { await vi.waitFor(() => expect(button('Turn on')).not.toBeNull(), { timeout: 5000 }); });
  await act(async () => { button('Turn on')!.click(); });
  await contains('Example Studio is ready');
  // The reply arrives after that report, then the owner turns it off.
  await act(async () => { await vi.waitFor(() => expect(posts).toHaveLength(1)); await new Promise((resolve) => setTimeout(resolve, 3000)); });
  turnedOffElsewhere('2026-10-08T12:01:00Z');
  await offerTurnOn();
});

it('does not keep showing a node the install turned on as turning on once a later report says it is off', async () => {
  // The page opens on a node the install turned on, already reported running
  // in the cluster state this browser holds.
  summaries = [summary({ observedAt: '2026-10-08T12:00:05Z' })];
  configEnabled = true;
  await act(async () => { await store.dispatch(clusterApi.endpoints.getRawState.initiate(undefined, { subscribe: false })); });
  await render(() => undefined, () => undefined, undefined, ['studio']);
  await contains('Example Studio is ready');
  turnedOffElsewhere('2026-10-08T12:01:00Z');
  await offerTurnOn();
});

it('shows a node whose report lags its configuration as turning on, until its configuration says it is off', async () => {
  summaries = [summary({ status: 'disabled', surfaces: [], observedAt: '2026-10-08T12:00:00Z' })];
  configEnabled = true;
  await render(() => undefined);
  await contains('Turning on…');
  expect(button('Turn on')).toBeNull();
  turnedOffElsewhere('2026-10-08T12:00:30Z');
  await offerTurnOn();
});
