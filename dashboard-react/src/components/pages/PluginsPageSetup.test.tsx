import { configureStore } from '@reduxjs/toolkit';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { apiSlice } from '../../store/api';
import type { CatalogEntry, ManagedRuntime, RuntimeInstallation } from '../../store/endpoints/plugins';
import { darkTheme } from '../../theme/theme';
import { readJourneys, saveJourney } from './catalogJourney';
import { exampleStudioSettingsSchema } from './exampleStudioSettings.fixture';
import { PluginsPage } from './PluginsPage';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });
vi.mock('../../i18n/tolgee', () => ({ useSkulkTranslation: () => ({ t: (_key: string, fallback: string, params?: Record<string, unknown>) => fallback.replace(/\{(\w+)\}/g, (_match, name: string) => String(params?.[name] ?? '')) }) }));

const pluginId = 'managed.' + '2'.repeat(32);
const newDigest = '9'.repeat(64);
const oldDigest = 'e'.repeat(64);
const screenUrl = 'https://studio.example/s/token/';
const schemaDigest = 'a'.repeat(64);
// One release, listed for two platforms; this host is the Mac one.
const macEntry: CatalogEntry = {
  bundle_id: 'example.studio', bundle_version: '0.1.0', title: 'Example Studio', publisher: 'example', sequence: 51,
  release_digest: 'd'.repeat(64), runtime_platform: 'macos-arm64', artifact_sha256: 'a'.repeat(64), artifact_bytes: 4096, transfer_bytes: 12_000_000,
  platforms: ['darwin'], skulk_build_sha256: 'b'.repeat(64), skulk_requires: '>=2.0.0,<3', permissions: ['Use models on the fabric through the host API'],
  descriptors: ['studio.render@1.0.0'], surfaces: ['Example Studio'], operations: true, steward_risks: ['observation'], expires_at: 1_900_000_000, matches_host: true,
};
const linuxEntry: CatalogEntry = { ...macEntry, release_digest: 'f'.repeat(64), runtime_platform: 'linux-glibc-aarch64', platforms: ['linux'], matches_host: false };

let root: Root;
let host: HTMLDivElement;
let store: ReturnType<typeof makeStore>;
let runtime: ManagedRuntime | null;
let installOperation: RuntimeInstallation | null;
let nodeStatus: string;
let nodeEnabled: boolean;
let revision: number;
let refuseEnable: string | null;
let listed: CatalogEntry[];
let posts: { path: string; body: Record<string, unknown> }[];
let preflightReads: number;

function makeStore() { return configureStore({ reducer: { [apiSlice.reducerPath]: apiSlice.reducer }, middleware: (defaults) => defaults().concat(apiSlice.middleware) }); }
function json(body: unknown, status = 200) { return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }); }
async function contains(text: string, timeout = 10_000) { await act(async () => { await vi.waitFor(() => expect(host.textContent).toContain(text), { timeout }); }); }
function button(label: string) { return [...(host.querySelector('[role=dialog]') ?? host).querySelectorAll('button')].find((item) => item.textContent === label) ?? null; }
async function click(label: string) { await act(async () => { button(label)!.click(); }); }
function setupTitle() { return host.querySelector('#capability-setup-title')?.textContent ?? null; }
async function setupTitleIs(text: string) { await act(async () => { await vi.waitFor(() => expect(setupTitle()).toBe(text), { timeout: 15_000 }); }); }
const enables = () => posts.filter((post) => post.path.endsWith('/nodes/studio/configuration') && post.body.operation === 'enable');
function running(digest: string, sequence: number): ManagedRuntime {
  return {
    plugin_id: pluginId, release: { bundle_id: 'example.studio', title: 'Example Studio', bundle_version: '0.1.0', publisher: 'example', sequence },
    selected_digest: digest, selection_revision: 3, enabled: true, stale: false, error_code: null, operation_id: null, operation_state: null,
    service: { state: 'running', active_digest: digest, observed_at: 1 },
  };
}
function summary() {
  return {
    pluginId, nodeId: 'studio', bundleId: 'example.studio', version: '0.1.0', title: 'Example Studio', status: nodeStatus, ownerAvailable: true,
    surfaces: nodeStatus === 'ready' ? [{ surfaceId: 'studio', title: 'Example Studio', kind: 'link', url: screenUrl, ready: true }] : [],
    actions: [], operationsActive: 0, observedAt: new Date().toISOString(),
  };
}

beforeEach(() => {
  runtime = null;
  installOperation = null;
  nodeStatus = 'disabled';
  nodeEnabled = false;
  revision = 0;
  refuseEnable = null;
  listed = [linuxEntry, macEntry];
  posts = [];
  preflightReads = 0;
  localStorage.removeItem('skulk-plugin-install-journeys');
  let activationId = '';
  vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = input instanceof Request ? input : new Request(new URL(String(input), location.href), init);
    const path = new URL(request.url).pathname;
    const live = runtime?.service?.state === 'running';
    if (request.method === 'POST') {
      const body = await request.json() as Record<string, unknown>;
      posts.push({ path, body });
      if (path === '/v1/plugins/managed/catalog/install') {
        // The host keeps one installation id for this test, whatever the dashboard proposed.
        runtime ??= { plugin_id: pluginId, release: null, selected_digest: null, selection_revision: 0, enabled: false, stale: true, error_code: null, operation_id: null, operation_state: null, service: null };
        return json({ plugin_id: runtime.plugin_id, listing: macEntry, source: { revision: 7, configured: true, credential_reference: null, credential_ready: true, trust_revision: 1 },
          review: { runtime_digest: newDigest, source_revision: 7, publisher: 'example', bundle_id: 'example.studio', version: '0.1.0', sequence: 51, platform: 'macos-arm64', python_requires: '>=3.13', skulk_build_sha256: 'b'.repeat(64), permissions: macEntry.permissions, artifact_bytes: 12_000_000, expires_at: 1_900_000_000 } });
      }
      if (path.endsWith('/install')) {
        installOperation = { request: { operation_id: String(body.operation_id), runtime_digest: newDigest, expected_source_revision: 7 },
          review: { runtime_digest: newDigest, source_revision: 7, publisher: 'example', bundle_id: 'example.studio', version: '0.1.0', sequence: 51, platform: 'macos-arm64', python_requires: '>=3.13', skulk_build_sha256: 'b'.repeat(64), permissions: [], artifact_bytes: 12_000_000, expires_at: 1_900_000_000 },
          state: 'staged', downloaded_bytes: 12_000_000, error_code: null };
        return json({ ...installOperation, state: 'accepted' });
      }
      if (path.endsWith('/operations')) {
        activationId = String(body.operation_id);
        runtime = running(newDigest, 51);
        return json({ request: body, state: 'accepted', error_code: null });
      }
      if (path.endsWith('/nodes/studio/configuration')) {
        if (body.expectedRevision !== revision) return json({ detail: 'configuration refused; reload settings and validation' }, 409);
        if (body.operation === 'enable' && refuseEnable) return json({ detail: refuseEnable }, 409);
        if (body.operation === 'enable') { nodeEnabled = true; nodeStatus = 'ready'; revision += 1; }
        if (body.operation === 'disable') { nodeEnabled = false; nodeStatus = 'disabled'; revision += 1; }
        return json({ configuration: { nodeId: 'studio', revision, schemaDigest, configurationSchema: exampleStudioSettingsSchema, values: {}, enabled: nodeEnabled }, validated: true });
      }
      return json({ detail: 'unexpected' }, 404);
    }
    if (path === '/state') return json({ instances: {}, runners: {}, capabilityNodes: live ? { 'host-node': [summary()] } : {} });
    if (path === '/node_id') return json('host-node');
    if (path === '/node/identity') return json({ nodeId: 'host-node', friendlyName: 'host' });
    if (path === '/v1/plugins/managed/service') return json({ state: 'ready', scope: 'user', progress: null, error: null });
    if (path === '/v1/plugins/managed/catalog/source') return json({ revision: 1, configured: true, credential_reference: null, credential_ready: true, trust_revision: 1, builtin_store: true, builtin_store_available: true });
    if (path === '/v1/plugins/managed/catalog') return json({ publisher: 'example', revision: 34, created_at: 1_790_000_000, expires_at: 1_800_000_000, catalog_sha256: 'c'.repeat(64), entries: listed });
    if (path === '/v1/plugins/managed') return json({ installations: runtime ? [runtime] : [] });
    if (path.endsWith('/install')) return json({ operation: installOperation });
    if (path.endsWith('/source')) return json({ revision: 7, configured: true, credential_reference: null, credential_ready: true, trust_revision: 1 });
    if (path.includes('/operations/')) return json({ request: { operation_id: activationId, action: 'activate' }, state: 'complete', error_code: null });
    if (path === '/v1/plugins') {
      return json(live ? [{ pluginId, available: true, nodes: [{ nodeId: 'studio', bundleId: 'example.studio', version: '0.1.0', status: nodeStatus, configurable: true, preflightAvailable: true }] }] : []);
    }
    if (path.endsWith('/nodes/studio/configuration')) return json({ nodeId: 'studio', revision, schemaDigest, configurationSchema: exampleStudioSettingsSchema, values: {}, enabled: nodeEnabled });
    if (path.endsWith('/nodes/studio/preflight')) {
      preflightReads += 1;
      return json({ nodeId: 'studio', revision, schemaDigest, valuesDigest: 'b'.repeat(64), credentialRevision: null, observedAt: 1_790_000_000,
        checks: [{ code: 'durable_storage', passed: true, correctiveAction: null }, { code: 'placed_video_model', passed: false, correctiveAction: 'Place a video model on a node that can serve it.' }] });
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
  history.replaceState(null, '', '/');
  localStorage.removeItem('skulk-plugin-install-journeys');
});

async function renderAt(address: string) {
  history.replaceState(null, '', address);
  await act(async () => { root.render(<Provider store={store}><ThemeProvider theme={darkTheme}><PluginsPage /></ThemeProvider></Provider>); });
}
async function install(primary: 'Review and install' | 'Review update', confirm: 'Install' | 'Update') {
  await act(async () => { await vi.waitFor(() => expect(button(primary)).not.toBeNull(), { timeout: 10_000 }); });
  await click(primary);
  await act(async () => { (host.querySelector('#catalog-consent') as HTMLInputElement).click(); });
  await click(confirm);
}
function installedTabSelected() {
  return [...host.querySelectorAll('[role=tab]')].find((tab) => tab.textContent === 'Installed')?.getAttribute('aria-selected');
}

it('turns a capability that needs no settings on by itself after install, once, and lands on its page ready to open', async () => {
  await renderAt('/plugins?view=browse');
  await install('Review and install', 'Install');
  await setupTitleIs('Example Studio is ready');
  // The setup page belongs to the installed plugin, under Installed.
  expect(installedTabSelected()).toBe('true');
  expect(window.location.search).toBe('');
  const open = host.querySelector('a[href]') as HTMLAnchorElement;
  expect(open.href).toBe(screenUrl);
  expect(open.target).toBe('_blank');
  expect(open.textContent).toContain('Open Example Studio');
  expect(host.textContent).toContain('Opens in a new tab. You can also open it from its node in the Cluster view.');
  expect(host.textContent).not.toContain('Back to Browse');
  // One enable, fenced by the revision just read and carrying no values.
  expect(enables().map((post) => post.body)).toEqual([{ operation: 'enable', expectedRevision: 0, expectedSchemaDigest: schemaDigest }]);
  expect(readJourneys()).toEqual([]);
  // Nothing is turned on again while the page keeps polling.
  await act(async () => { await new Promise((resolve) => setTimeout(resolve, 2500)); });
  expect(enables()).toHaveLength(1);
  await click('Done');
  expect(setupTitle()).toBeNull();
  expect(host.querySelector('[aria-label="Filter plugins"]')).not.toBeNull();
  expect(installedTabSelected()).toBe('true');
});

it('leaves a node that needs its settings for its setup page, which opens exactly those settings', async () => {
  nodeStatus = 'configuration_invalid';
  await renderAt('/plugins?view=browse');
  await install('Review and install', 'Install');
  await setupTitleIs('Example Studio is installed');
  await contains('It needs its settings before it can run.');
  expect(enables()).toEqual([]);
  expect(button('Turn on')).toBeNull();
  await click('Open settings');
  // The drawer opens on the node's settings, with every optional field rendered.
  await act(async () => { await vi.waitFor(() => expect(host.querySelector('[role=dialog]')?.textContent).toContain('Skulk API URL'), { timeout: 5000 }); });
  expect(host.querySelector('[role=dialog]')?.textContent).toContain('Comfy URL');
  expect(enables()).toEqual([]);
});

it('falls back to Turn on with the host\'s reason when turning on is refused', async () => {
  refuseEnable = 'A video model must be placed before the studio can start.';
  await renderAt('/plugins?view=browse');
  await install('Review and install', 'Install');
  await setupTitleIs('Example Studio is installed');
  await contains('It was not turned on. The host said: A video model must be placed before the studio can start.');
  await act(async () => { await vi.waitFor(() => expect(button('Turn on')).not.toBeNull(), { timeout: 5000 }); });
  expect(button('Open settings')).not.toBeNull();
  // Its own checks run at once and say what is missing.
  await contains('Place a video model on a node that can serve it.');
  expect(preflightReads).toBeGreaterThan(0);
  expect(enables()).toHaveLength(1);
});

it('does not turn a node on again when the page is reloaded after the install tried', async () => {
  runtime = running(newDigest, 51);
  installOperation = { request: { operation_id: '5'.repeat(32), runtime_digest: newDigest, expected_source_revision: 7 },
    review: { runtime_digest: newDigest, source_revision: 7, publisher: 'example', bundle_id: 'example.studio', version: '0.1.0', sequence: 51, platform: 'macos-arm64', python_requires: '>=3.13', skulk_build_sha256: 'b'.repeat(64), permissions: [], artifact_bytes: 12_000_000, expires_at: 1_900_000_000 },
    state: 'staged', downloaded_bytes: 12_000_000, error_code: null };
  // The install already tried to turn it on before the page was reloaded; the owner has since turned it off.
  saveJourney({ pluginId, title: 'Example Studio', bundleId: 'example.studio', sequence: 51, publisher: 'example', runtimeDigest: newDigest, transferBytes: 12_000_000,
    installOperationId: '5'.repeat(32), activationOperationId: '6'.repeat(32), startedAt: 1, autoTurnOn: { nodes: 'all', ran: true } });
  await renderAt('/plugins?view=browse');
  await setupTitleIs('Example Studio is installed');
  await contains('It is turned off. Turn it on to start it.');
  await contains('Turning it on runs its setup checks again.');
  expect(enables()).toEqual([]);
  expect(readJourneys()).toEqual([]);
});

it('never turns on a node the owner turned off, from Browse or across an update', async () => {
  // Installed and running, with its node turned off by the owner.
  runtime = running(newDigest, 51);
  await renderAt('/plugins?view=browse');
  await act(async () => { await vi.waitFor(() => expect(button('Set up')).not.toBeNull(), { timeout: 10_000 }); });
  // The installed card names this host's listing of the release, not another platform's.
  expect(host.textContent).toContain('macOS, Apple Silicon');
  expect(host.textContent).not.toContain('Linux, ARM64');
  await click('Set up');
  await setupTitleIs('Example Studio is installed');
  await contains('It is turned off. Turn it on to start it.');
  await act(async () => { await new Promise((resolve) => setTimeout(resolve, 1500)); });
  expect(enables()).toEqual([]);
  // An update keeps the owner's choice too.
  await click('Done');
  runtime = running(oldDigest, 50);
  await act(async () => { root.unmount(); store.dispatch(apiSlice.util.resetApiState()); });
  root = createRoot(host);
  await renderAt('/plugins?view=browse');
  await install('Review update', 'Update');
  await setupTitleIs('Example Studio is installed');
  await contains('It is turned off. Turn it on to start it.');
  expect(posts.some((post) => post.path.endsWith('/operations'))).toBe(true);
  expect(enables()).toEqual([]);
});

it('turns a node on from its setup page with one action and ends ready to open', async () => {
  runtime = running(newDigest, 51);
  await renderAt('/plugins?view=browse');
  await act(async () => { await vi.waitFor(() => expect(button('Set up')).not.toBeNull(), { timeout: 10_000 }); });
  await click('Set up');
  await setupTitleIs('Example Studio is installed');
  await act(async () => { await vi.waitFor(() => expect(button('Turn on')).not.toBeNull(), { timeout: 5000 }); });
  expect(host.textContent).not.toContain('Its screens appear here');
  await contains('Once it is running, Open Example Studio appears here and opens it in a new tab.');
  // The release lists no action that can spend money.
  await contains('Turning it on runs its setup checks again. It cannot spend money.');
  await click('Turn on');
  await setupTitleIs('Example Studio is ready');
  expect(enables().map((post) => post.body)).toEqual([{ operation: 'enable', expectedRevision: 0, expectedSchemaDigest: schemaDigest }]);
  expect((host.querySelector('a[href]') as HTMLAnchorElement).href).toBe(screenUrl);
});

it('labels an installed card by what its nodes report: Open when running, Set up when it needs the owner, Manage otherwise', async () => {
  runtime = running(newDigest, 51);
  nodeStatus = 'ready';
  await renderAt('/plugins?view=browse');
  // Running with a screen: the card opens it in a new tab, resolved as the plugin's own page resolves it.
  await act(async () => { await vi.waitFor(() => expect(host.querySelector('a[href]')).not.toBeNull(), { timeout: 10_000 }); });
  const open = host.querySelector('a[href]') as HTMLAnchorElement;
  expect(open.href).toBe(screenUrl);
  expect(open.target).toBe('_blank');
  expect(open.textContent).toBe('Open Example Studio');
  expect(open.title).toBe('Opens in a new tab');
  expect(button('Set up')).toBeNull();
  // Turned off: it needs the owner.
  nodeStatus = 'disabled';
  await act(async () => { await vi.waitFor(() => expect(button('Set up')).not.toBeNull(), { timeout: 5000 }); });
  expect(host.querySelector('a[href]')).toBeNull();
  // Starting: nothing to do but follow it on its page.
  nodeStatus = 'starting';
  await act(async () => { await vi.waitFor(() => expect(button('Manage')).not.toBeNull(), { timeout: 5000 }); });
  expect(button('Set up')).toBeNull();
  await click('Manage');
  await setupTitleIs('Example Studio is installed');
  await contains('It is starting. This page updates by itself.');
  expect(enables()).toEqual([]);
});

it('closes an install\'s progress back to the capability list with a bordered Close', async () => {
  // The install is followed but has not finished, so its progress stays open.
  installOperation = null;
  saveJourney({ pluginId, title: 'Example Studio', bundleId: 'example.studio', sequence: 51, publisher: 'example', runtimeDigest: newDigest, transferBytes: 12_000_000,
    installOperationId: '5'.repeat(32), activationOperationId: null, startedAt: 1 });
  runtime = { plugin_id: pluginId, release: null, selected_digest: null, selection_revision: 0, enabled: false, stale: true, error_code: null, operation_id: null, operation_state: null, service: null };
  await renderAt('/plugins?view=browse');
  await contains('Installing Example Studio');
  expect(button('Back to Browse')).toBeNull();
  await click('Close');
  await contains('Browse capabilities');
  expect(setupTitle()).toBeNull();
});
