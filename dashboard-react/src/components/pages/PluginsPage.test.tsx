import { configureStore } from '@reduxjs/toolkit';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { apiSlice } from '../../store/api';
import type { ManagedRuntime, NodeConfiguration, RuntimeInstallation } from '../../store/endpoints/plugins';
import { darkTheme } from '../../theme/theme';
import { dashboardUrlOn } from '../../utils/hostDashboard';
import { saveJourney } from './catalogJourney';
import { PluginsPage } from './PluginsPage';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });
vi.mock('../../i18n/tolgee', () => ({ useSkulkTranslation: () => ({ t: (_key: string, fallback: string, params?: Record<string, unknown>) => fallback.replace(/\{(\w+)\}/g, (_match, name: string) => String(params?.[name] ?? '')) }) }));

let root: Root;
let host: HTMLDivElement;
let configuration: NodeConfiguration;
let failRead: boolean;
let reads: number;
let mutations: Record<string, unknown>[];
let store: ReturnType<typeof makeStore>;
let clusterState: Record<string, unknown>;
let installations: ManagedRuntime[];
let installOperation: RuntimeInstallation | null;
let recovers: { path: string; body: unknown }[];

function makeStore() {
  return configureStore({ reducer: { [apiSlice.reducerPath]: apiSlice.reducer }, middleware: (defaults) => defaults().concat(apiSlice.middleware) });
}
function response(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}
function button(label: string) {
  const scope = host.querySelector('[role=dialog]') ?? host;
  const element = [...scope.querySelectorAll('button')].find((candidate) => candidate.textContent === label);
  if (!element) throw new Error(`Missing button: ${label}`);
  return element;
}
async function click(label: string) {
  await act(async () => { button(label).click(); });
}
async function choose(value: string) {
  await act(async () => { host.querySelector<HTMLButtonElement>('[aria-haspopup="listbox"]')!.click(); });
  await act(async () => {
    const option = [...document.querySelectorAll<HTMLButtonElement>('[role="option"]')].find(option => option.textContent === value);
    if (!option) throw new Error('Missing region option');
    option.click();
  });
}
async function ready() {
  await act(async () => { await vi.waitFor(() => expect(host.textContent).toContain('test.bundle')); });
  await click('Configure');
  await click('Configure');
  await act(async () => { await vi.waitFor(() => expect(host.querySelector('[aria-haspopup="listbox"]')).not.toBeNull()); });
}

beforeEach(async () => {
  configuration = { nodeId: 'node-1', revision: 0, schemaDigest: 'a'.repeat(64), enabled: false,
    configurationSchema: { type: 'object', required: ['region'], properties: { region: { type: 'string', title: 'Region', enum: ['east', 'west', 'north'] } } },
    values: { region: 'east' } };
  failRead = false;
  reads = 0;
  mutations = [];
  clusterState = {};
  installations = [];
  installOperation = null;
  recovers = [];
  vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = new Request(input, init);
    // Cluster state names the other nodes; only plugin requests carry the dashboard header.
    const path = new URL(request.url).pathname;
    if (path === '/state') return response(clusterState);
    if (path === '/node_id') return response('host-node');
    if (path === '/node/identity') return response({ nodeId: 'host-node', friendlyName: 'host' });
    expect(request.headers.get('X-Skulk-Dashboard')).toBe('pairing-v1');
    if (path === '/v1/plugins/managed/service') return response({ state: 'ready', scope: 'user', progress: null, error: null });
    if (new URL(request.url).pathname === '/v1/plugins/managed') return response({ installations });
    if (path.endsWith('/recover')) {
      recovers.push({ path, body: await request.json() });
      installOperation = { ...installOperation!, attempt: 1, state: 'downloading', downloaded_bytes: 0, error_code: null };
      return response({ ...installOperation, state: 'accepted' });
    }
    if (path.endsWith('/install')) return response({ operation: installOperation });
    if (/\/installations\/[^/]+\/source$/.test(path)) return response({ revision: 4, configured: true, credential_reference: null, credential_ready: true, trust_revision: 1 });
    if (new URL(request.url).pathname === '/v1/plugins') return response([{ pluginId: 'bridge', available: true, nodes: [
      { nodeId: 'node-1', bundleId: 'test.bundle', version: '1.0.0', status: 'disabled', configurable: true },
    ] }]);
    if (request.method === 'GET') {
      reads += 1;
      return failRead ? response({}, 503) : response(configuration);
    }
    const mutation = await request.json() as Record<string, unknown>;
    mutations.push(mutation);
    if (mutation.expectedRevision !== configuration.revision) return response({}, 409);
    if (mutation.operation !== 'validate') configuration = { ...configuration, revision: configuration.revision + 1, values: mutation.values as Record<string, unknown> ?? configuration.values };
    return response({ configuration, validated: true });
  });
  store = makeStore();
  host = document.createElement('div');
  document.body.appendChild(host);
  root = createRoot(host);
  await renderPage();
});
async function renderPage() {
  await act(async () => { root.render(<Provider store={store}><ThemeProvider theme={darkTheme}><PluginsPage /></ThemeProvider></Provider>); });
}
afterEach(async () => {
  await act(async () => { root.unmount(); store.dispatch(apiSlice.util.resetApiState()); });
  host.remove();
  vi.unstubAllGlobals();
  history.replaceState(null, '', '/');
  localStorage.removeItem('skulk-plugin-install-journeys');
});

it('loads disabled-node settings on demand and fences writes by revision and schema', async () => {
  await act(async () => { await vi.waitFor(() => expect(host.textContent).toContain('test.bundle')); });
  expect(reads).toBe(0);
  await ready();
  await choose('west');
  expect(button('Enable').disabled).toBe(true);
  await click('Save settings');
  await act(async () => { await vi.waitFor(() => expect(configuration.values.region).toBe('west')); });
  expect(mutations).toEqual([{ operation: 'edit', values: { region: 'west' }, expectedRevision: 0, expectedSchemaDigest: 'a'.repeat(64) }]);
});

it('preserves a rejected draft and fetches a fresh revision before subsequent edits', async () => {
  await ready();
  await choose('west');
  configuration = { ...configuration, revision: 1, values: { region: 'north' } };
  await click('Save settings');
  await act(async () => { await vi.waitFor(() => expect(host.textContent).toContain('The change was refused')); });
  expect(host.querySelector('[aria-haspopup="listbox"]')?.textContent).toBe('west');
  const previousReads = reads;
  await click('Reload settings');
  await act(async () => { await vi.waitFor(() => expect(host.querySelector('[aria-haspopup="listbox"]')?.textContent).toBe('north')); });
  expect(reads).toBeGreaterThan(previousReads);
  await choose('east');
  await click('Save settings');
  await act(async () => { await vi.waitFor(() => expect(mutations.at(-1)?.expectedRevision).toBe(1)); });
});

it('preserves unsaved values when reloading fails', async () => {
  await ready();
  await choose('west');
  failRead = true;
  await click('Reload settings');
  await act(async () => { await vi.waitFor(() => expect(host.textContent).toContain('Your draft is preserved')); });
  expect(host.querySelector('[aria-haspopup="listbox"]')?.textContent).toBe('west');
  expect(mutations).toEqual([]);
});

it('opens the plugin named in the address, then forgets the link when closed', async () => {
  await act(async () => { root.unmount(); });
  history.replaceState(null, '', '/plugins?plugin=bridge');
  root = createRoot(host);
  await renderPage();
  await act(async () => { await vi.waitFor(() => expect(host.querySelector('[role=dialog]')?.textContent).toContain('test.bundle')); });
  await act(async () => { (host.querySelector('[role=dialog] button[aria-label="Close"]') as HTMLButtonElement).click(); });
  expect(window.location.search).toBe('');
});

it('links to the other nodes\' Plugins pages at their tailnet addresses', async () => {
  const summary = { pluginId: 'managed.' + '3'.repeat(32), nodeId: 'studio', bundleId: 'example.studio', version: '0.1.0', title: 'Example Studio', status: 'ready', ownerAvailable: true, surfaces: [], actions: [], operationsActive: 0, observedAt: '2026-09-27T12:00:00Z' };
  clusterState = {
    topology: { nodes: ['host-node', 'render-node', 'lan-node'], connections: {} },
    nodeIdentities: { 'host-node': { friendlyName: 'host' }, 'render-node': { friendlyName: 'render' }, 'lan-node': { friendlyName: 'lan' } },
    nodeNetwork: { 'render-node': { interfaces: [{ name: 'tailscale0', ipAddress: '100.70.1.2' }] }, 'lan-node': { interfaces: [{ name: 'en0', ipAddress: '192.168.1.9' }] } },
    capabilityNodes: { 'render-node': [summary] },
  };
  await act(async () => { root.unmount(); store.dispatch(apiSlice.util.resetApiState()); });
  root = createRoot(host);
  await renderPage();
  await act(async () => { await vi.waitFor(() => expect(host.querySelector('nav[aria-label="Plugins on other nodes"]')?.textContent).toContain('render · 1 plugin'), { timeout: 5000 }); });
  const row = host.querySelector('nav[aria-label="Plugins on other nodes"]')!;
  const links = [...row.querySelectorAll('a')];
  expect(links.map((link) => [link.textContent, link.getAttribute('href')])).toEqual([['render · 1 plugin', dashboardUrlOn('100.70.1.2', '/plugins')]]);
  // A node without a Tailscale address is named, not linked.
  expect(row.textContent).toContain('lan');
  expect(row.textContent).not.toContain('host');
});

it('retries a stopped install from its Installed card on the same installation, following it in Browse', async () => {
  const pluginId = 'managed.' + '7'.repeat(32);
  installations = [{ plugin_id: pluginId, release: null, selected_digest: null, selection_revision: 0, enabled: false, stale: true, error_code: null, operation_id: null, operation_state: null, service: null }];
  installOperation = {
    attempt: 0, request: { operation_id: '5'.repeat(32), runtime_digest: '9'.repeat(64), expected_source_revision: 3 },
    review: { runtime_digest: '9'.repeat(64), source_revision: 3, publisher: 'example', bundle_id: 'example.studio', version: '0.1.0', sequence: 51, platform: 'macos-arm64', python_requires: '>=3.13', skulk_build_sha256: 'b'.repeat(64), permissions: [], artifact_bytes: 12_000_000, expires_at: 1_900_000_000 },
    state: 'recovery_required', downloaded_bytes: 0, error_code: 'download_failed',
  };
  saveJourney({ pluginId, title: 'Example Studio', bundleId: 'example.studio', sequence: 51, publisher: 'example', runtimeDigest: '9'.repeat(64),
    transferBytes: 12_000_000, installOperationId: '5'.repeat(32), activationOperationId: null, startedAt: 1, interrupted: true });
  await act(async () => { root.unmount(); store.dispatch(apiSlice.util.resetApiState()); });
  root = createRoot(host);
  await renderPage();
  await act(async () => { await vi.waitFor(() => expect(host.textContent).toContain('Install needs a retry'), { timeout: 5000 }); });
  // A stopped install counts as needing attention.
  expect(host.textContent).toContain('Needs attention · 1');
  await click('Retry');
  await act(async () => { await vi.waitFor(() => expect(recovers).toHaveLength(1), { timeout: 5000 }); });
  expect(recovers[0]).toEqual({ path: `/v1/plugins/managed/installations/${pluginId}/install/${'5'.repeat(32)}/recover`, body: { expected_source_revision: 4 } });
  await act(async () => { await vi.waitFor(() => expect(host.textContent).toContain('Installing Example Studio'), { timeout: 5000 }); });
  expect(window.location.search).toBe('?view=browse');
  expect(mutations).toEqual([]);
});
