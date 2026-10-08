import { configureStore } from '@reduxjs/toolkit';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { apiSlice } from '../../store/api';
import type { CatalogEntry, ManagedRuntime, RuntimeInstallation, RuntimeRelease } from '../../store/endpoints/plugins';
import { darkTheme } from '../../theme/theme';
import { encodeInvitation, readJourneys, saveJourney, type InstallJourney } from './catalogJourney';
import { PluginCatalogBrowse, type BrowseRetryHandoff } from './PluginCatalogBrowse';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });
vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({ t: (_key: string, fallback: string, params?: Record<string, unknown>) => fallback.replace(/\{(\w+)\}/g, (_match, name: string) => String(params?.[name] ?? '')) }),
}));

const pluginId = 'managed.' + '2'.repeat(32);
const oldDigest = 'e'.repeat(64);
const newDigest = '9'.repeat(64);
let root: Root;
let host: HTMLDivElement;
let store: ReturnType<typeof makeStore>;
let configured: boolean;
let installed: ManagedRuntime;
let posts: { path: string; body: Record<string, unknown> }[];
let freshHost: boolean;
let loseNextBind: boolean;
let onStore: boolean;
let storeAvailable: boolean;
let refuseStoreOnce: boolean;
let assignsRevisions: boolean;
// An installation whose install stopped: registered, never selected.
let stray: ManagedRuntime | null;
let strayOperation: RuntimeInstallation | null;
let stopFreshInstall: boolean;
let failStrayRead: boolean;
let recoverStaysRunning: boolean;
let listed: CatalogEntry[];

function makeStore() { return configureStore({ reducer: { [apiSlice.reducerPath]: apiSlice.reducer }, middleware: (defaults) => defaults().concat(apiSlice.middleware) }); }
function json(body: unknown, status = 200) { return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }); }
async function contains(text: string) { await act(async () => { await vi.waitFor(() => expect(host.textContent).toContain(text), { timeout: 5000 }); }); }
function button(label: string) { return [...host.querySelectorAll('button')].find((item) => item.textContent === label) ?? null; }
async function click(label: string) { await act(async () => { button(label)?.click(); }); }
const entry: CatalogEntry = {
  bundle_id: 'example.studio', bundle_version: '0.1.0', title: 'Example Studio', publisher: 'example', sequence: 51,
  release_digest: 'd'.repeat(64), runtime_platform: 'macos-arm64', artifact_sha256: 'a'.repeat(64), artifact_bytes: 4096, transfer_bytes: 12_086_479,
  platforms: ['darwin'], skulk_build_sha256: 'b'.repeat(64), skulk_requires: '>=2.0.0,<3', permissions: ['Use models on the fabric through the host API'],
  descriptors: ['studio.render@1.0.0'], surfaces: ['Example Studio'], operations: true, steward_risks: ['observation'],
  expires_at: 1_900_000_000, matches_host: true,
};
const strayId = 'managed.' + '7'.repeat(32);
const review: RuntimeRelease = { runtime_digest: newDigest, source_revision: 7, publisher: 'example', bundle_id: entry.bundle_id, version: '0.1.0', sequence: 51, platform: 'macos-arm64', python_requires: '>=3.13', skulk_build_sha256: 'b'.repeat(64), permissions: entry.permissions, artifact_bytes: 12_000_000, expires_at: 1_900_000_000 };
function registered(pluginId: string): ManagedRuntime {
  return { plugin_id: pluginId, release: null, selected_digest: null, selection_revision: 0, enabled: false, stale: true, error_code: null, operation_id: null, operation_state: null, service: null };
}
function stoppedAt(operationId: string): RuntimeInstallation {
  return { attempt: 0, request: { operation_id: operationId, runtime_digest: newDigest, expected_source_revision: 7 }, review, state: 'recovery_required', downloaded_bytes: 12_000_000, error_code: 'installation_failed' };
}
// What this browser saved when the owner accepted the release.
const consent: InstallJourney = {
  pluginId: strayId, title: 'Example Studio', bundleId: entry.bundle_id, sequence: 51, publisher: 'example', runtimeDigest: newDigest,
  transferBytes: 12_000_000, installOperationId: '5'.repeat(32), activationOperationId: null, startedAt: 1, interrupted: true,
};

beforeEach(async () => {
  posts = [];
  configured = true;
  freshHost = false;
  loseNextBind = false;
  onStore = false;
  storeAvailable = false;
  refuseStoreOnce = false;
  assignsRevisions = false;
  stray = null;
  strayOperation = null;
  stopFreshInstall = false;
  failStrayRead = false;
  recoverStaysRunning = false;
  listed = [entry];
  localStorage.removeItem('skulk-plugin-install-journeys');
  installed = {
    plugin_id: pluginId, release: { bundle_id: 'example.studio', title: 'Example Studio', bundle_version: '0.1.0', publisher: 'example', sequence: 50 },
    selected_digest: oldDigest, selection_revision: 3, enabled: true, stale: false, error_code: null, operation_id: null, operation_state: null,
    service: { state: 'running', active_digest: oldDigest, observed_at: 1 },
  };
  let installOperationId = '';
  let activationOperationId = '';
  vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = input instanceof Request ? input : new Request(new URL(String(input), location.href), init);
    const path = new URL(request.url).pathname;
    const strayPath = stray ? `/v1/plugins/managed/installations/${stray.plugin_id}` : null;
    // Each installation's node already runs, so nothing is left to turn on.
    if (path === '/v1/plugins') {
      return json([...(freshHost ? [] : [installed]), ...(stray ? [stray] : [])].map((runtime) => ({
        pluginId: runtime.plugin_id, available: true, nodes: [{ nodeId: 'studio', bundleId: entry.bundle_id, version: '0.1.0', status: 'ready', configurable: true }],
      })));
    }
    if (strayPath && path.startsWith(strayPath + '/')) {
      const rest = path.slice(strayPath.length);
      if (request.method === 'POST') {
        const body = await request.json() as Record<string, unknown>;
        posts.push({ path, body });
        if (rest === '/install') { strayOperation = stoppedAt(String(body.operation_id)); return json({ ...strayOperation, state: 'accepted' }); }
        if (rest.endsWith('/recover') && strayOperation) {
          strayOperation = { ...strayOperation, attempt: 1, state: recoverStaysRunning ? 'downloading' : 'staged', downloaded_bytes: recoverStaysRunning ? 0 : 12_000_000, error_code: null };
          return json({ ...strayOperation, state: 'accepted' });
        }
        if (rest === '/operations') {
          activationOperationId = String(body.operation_id);
          stray = { ...stray!, enabled: true, selected_digest: newDigest, selection_revision: 1, release: { bundle_id: entry.bundle_id, title: 'Example Studio', bundle_version: '0.1.0', publisher: 'example', sequence: 51 }, service: { state: 'running', active_digest: newDigest, observed_at: 2 } };
          return json({ request: body, state: 'accepted', error_code: null });
        }
        return json({ detail: 'unexpected' }, 404);
      }
      if (rest === '/install') return failStrayRead ? json({ detail: 'unavailable' }, 503) : json({ operation: strayOperation });
      if (rest === '/source') return json({ revision: 8, configured: true, credential_reference: null, credential_ready: true, trust_revision: 1 });
      if (rest.startsWith('/operations/')) return json({ request: { operation_id: activationOperationId, action: 'activate' }, state: 'complete', error_code: null });
      return json({ detail: 'not found' }, 404);
    }
    if (request.method === 'POST') {
      const body = await request.json() as Record<string, unknown>;
      posts.push({ path, body });
      if (path === '/v1/plugins/managed/catalog/source') { configured = true; onStore = false; return json({ revision: 1, configured: true, credential_reference: null, credential_ready: true, trust_revision: 1 }); }
      if (path === '/v1/plugins/managed/catalog/source/builtin') {
        if (refuseStoreOnce) {
          refuseStoreOnce = false;
          return json({ detail: 'The catalog source changed since its revision was read. Read the catalog source status and apply the change at its current revision.' }, 409);
        }
        onStore = true;
        return json({ revision: 2, configured: true, credential_reference: null, credential_ready: true, trust_revision: 1, builtin_store: true, builtin_store_available: true });
      }
      if (path === '/v1/plugins/managed/catalog/install') {
        // The host acts on the request, but the reply never arrives.
        if (loseNextBind) { loseNextBind = false; throw new TypeError('network'); }
        // The host registers the new installation; its install will stop.
        if (stopFreshInstall) stray = registered(String(body.plugin_id));
        return json({ plugin_id: String(body.plugin_id), listing: entry, source: { revision: 7, configured: true, credential_reference: null, credential_ready: true, trust_revision: 1 },
          review: { runtime_digest: newDigest, source_revision: 7, publisher: 'example', bundle_id: entry.bundle_id, version: '0.1.0', sequence: 51, platform: 'macos-arm64', python_requires: '>=3.13', skulk_build_sha256: 'b'.repeat(64), permissions: entry.permissions, artifact_bytes: 12_000_000, expires_at: 1_900_000_000 } });
      }
      if (path.endsWith('/install')) { installOperationId = String(body.operation_id); return json({ request: body, state: 'accepted', downloaded_bytes: 0, error_code: null }); }
      if (path.endsWith('/operations')) {
        activationOperationId = String(body.operation_id);
        installed = { ...installed, selected_digest: newDigest, release: { ...installed.release!, sequence: 51 }, service: { state: 'running', active_digest: newDigest, observed_at: 2 } };
        return json({ request: body, state: 'accepted', error_code: null });
      }
      return json({ detail: 'unexpected' }, 404);
    }
    if (path === '/v1/plugins/managed/catalog/source') {
      return json({ revision: configured ? 1 : 0, configured, credential_reference: null, credential_ready: configured, trust_revision: configured ? 1 : null, builtin_store: onStore, builtin_store_available: storeAvailable, ...(assignsRevisions ? { assigns_trust_revisions: true } : {}) });
    }
    if (path === '/v1/plugins/managed/catalog') return json({ publisher: 'example', revision: 34, created_at: 1_790_000_000, expires_at: 1_800_000_000, catalog_sha256: 'c'.repeat(64), entries: listed });
    if (path === '/v1/plugins/managed') return json({ installations: [...(freshHost ? [] : [installed]), ...(stray ? [stray] : [])] });
    if (path.endsWith('/install')) return json({ operation: installOperationId ? { request: { operation_id: installOperationId, runtime_digest: newDigest, expected_source_revision: 7 }, review, state: 'staged', downloaded_bytes: 12_000_000, error_code: null } : null });
    if (path.includes('/operations/')) return json({ request: { operation_id: activationOperationId, action: 'activate' }, state: 'complete', error_code: null });
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

async function render(retry: BrowseRetryHandoff | null = null, onRetryTaken?: () => void) {
  await act(async () => { root.render(<Provider store={store}><ThemeProvider theme={darkTheme}><PluginCatalogBrowse retry={retry} onRetryTaken={onRetryTaken} /></ThemeProvider></Provider>); });
}
// A finished install continues on the installed plugin's own page.
async function running() {
  await act(async () => { await vi.waitFor(() => expect(host.querySelector('#capability-setup-title')).not.toBeNull(), { timeout: 8000 }); });
  await contains('Example Studio is installed');
}

it('updates an installed capability from its catalog listing after one consent', async () => {
  await render();
  await contains('Browse capabilities');
  await contains('Update available');
  await contains('Release 50 is installed; release 51 is built for this host.');
  await click('Review update');
  await contains('Review the update to Example Studio');
  await contains('It cannot spend money.');
  expect(button('Update')?.disabled).toBe(true);
  await act(async () => { (host.querySelector('#catalog-consent') as HTMLInputElement).click(); });
  expect(button('Update')?.disabled).toBe(false);
  await click('Update');
  await contains('Updating Example Studio');
  await running();
  expect(posts.map((post) => post.path)).toEqual([
    '/v1/plugins/managed/catalog/install',
    `/v1/plugins/managed/installations/${pluginId}/install`,
    `/v1/plugins/managed/installations/${pluginId}/operations`,
  ]);
  expect(posts[0].body).toMatchObject({ catalog_sha256: 'c'.repeat(64), bundle_id: 'example.studio', sequence: 51, plugin_id: pluginId });
  expect(posts[1].body).toMatchObject({ runtime_digest: newDigest, expected_source_revision: 7 });
  expect(posts[2].body).toMatchObject({ action: 'activate', expected_revision: 3, runtime_digest: newDigest, accept_permissions: true });
  expect(readJourneys()).toEqual([]);
});

it('lets a host that assigns trust revisions pick the next one itself', async () => {
  configured = false;
  assignsRevisions = true;
  await render();
  await contains('Connect a capability catalog');
  const code = encodeInvitation({ baseUrl: 'https://catalog.example.ts.net/', publisher: 'example', publicKey: 'f'.repeat(64), trustExpiresAt: 4_000_000_000 });
  const field = host.querySelector('#catalog-invitation') as HTMLTextAreaElement;
  await act(async () => {
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')!.set!;
    setter.call(field, code);
    field.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await click('Connect');
  await contains('Browse capabilities');
  expect(posts[0]).toMatchObject({ path: '/v1/plugins/managed/catalog/source', body: { expected_revision: 0, assign_trust_revision: true, trust: { revision: 1, expires_at: 4_000_000_000 } } });
});

it('lists the Foxlight store with nothing to paste and keeps a private catalog behind its own control', async () => {
  onStore = true;
  storeAvailable = true;
  await render();
  await contains('From the Foxlight capability store.');
  await contains('Update available');
  expect(host.querySelector('#catalog-invitation')).toBeNull();
  expect(button('Change catalog')).toBeNull();
  expect(button('Use the Foxlight store')).toBeNull();
  await click('Add a private catalog');
  await contains('This host then reads that catalog instead of the Foxlight store');
  expect(host.querySelector('#catalog-invitation')).not.toBeNull();
  await click('Cancel');
  await contains('From the Foxlight capability store.');
  expect(posts).toEqual([]);
});

it('returns a host on a private catalog to the Foxlight store', async () => {
  storeAvailable = true;
  refuseStoreOnce = true;
  await render();
  await contains('From the example catalog.');
  // A refusal is named in the host's own words and changes nothing.
  await click('Use the Foxlight store');
  await contains('The catalog source changed since its revision was read.');
  await click('Use the Foxlight store');
  await contains('From the Foxlight capability store.');
  expect(posts.map((post) => post.path)).toEqual(['/v1/plugins/managed/catalog/source/builtin', '/v1/plugins/managed/catalog/source/builtin']);
  expect(posts[1].body).toEqual({ expected_revision: 1 });
  expect(button('Add a private catalog')).not.toBeNull();
});

it('offers no way to the store on a build without one', async () => {
  await render();
  await contains('From the example catalog.');
  expect(button('Use the Foxlight store')).toBeNull();
  expect(button('Change catalog')).not.toBeNull();
});

it('connects a catalog from an invitation code before browsing it', async () => {
  configured = false;
  await render();
  await contains('Connect a capability catalog');
  const code = encodeInvitation({ baseUrl: 'https://catalog.example.ts.net/', publisher: 'example', publicKey: 'f'.repeat(64), trustExpiresAt: 4_000_000_000 });
  const field = host.querySelector('#catalog-invitation') as HTMLTextAreaElement;
  await act(async () => {
    const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')!.set!;
    setter.call(field, code);
    field.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await contains('catalog.example.ts.net');
  await click('Connect');
  await contains('Browse capabilities');
  expect(posts[0]).toMatchObject({ path: '/v1/plugins/managed/catalog/source', body: { expected_revision: 0, base_url: 'https://catalog.example.ts.net/', trust: { expires_at: 4_000_000_000, publishers: { example: 'f'.repeat(64) } } } });
  // Above anything an earlier panel created, so a returning catalog is never refused as a rollback.
  const trust = posts[0].body.trust as { revision: number };
  expect(trust.revision).toBeGreaterThanOrEqual(Math.floor(Date.now() / 1000) - 60);
});

it('continues a new installation whose binding reply was lost instead of registering another', async () => {
  freshHost = true;
  loseNextBind = true;
  await render();
  await contains('Review and install');
  await click('Review and install');
  await act(async () => { (host.querySelector('#catalog-consent') as HTMLInputElement).click(); });
  await click('Install');
  await contains('The host did not confirm this release, so it may already be bound.');
  await click('Close');
  await contains('Review and install');
  await click('Review and install');
  await act(async () => { (host.querySelector('#catalog-consent') as HTMLInputElement).click(); });
  await click('Install');
  await act(async () => { await vi.waitFor(() => expect(posts.filter((post) => post.path === '/v1/plugins/managed/catalog/install')).toHaveLength(2), { timeout: 5000 }); });
  const binds = posts.filter((post) => post.path === '/v1/plugins/managed/catalog/install').map((post) => post.body.plugin_id);
  expect(binds[0]).toMatch(/^managed\.[0-9a-f]{32}$/);
  expect(binds[1]).toBe(binds[0]);
});

it('returns to an install in progress instead of offering the release again', async () => {
  freshHost = true;
  const pending = `managed.${'7'.repeat(32)}`;
  saveJourney({ pluginId: pending, title: 'Example Studio', bundleId: 'example.studio', sequence: 51, publisher: 'example', runtimeDigest: newDigest,
    transferBytes: 12_000_000, installOperationId: '8'.repeat(32), activationOperationId: null, startedAt: 1 });
  await render();
  // Opening Browse picks the install up where it was.
  await contains('Installing Example Studio');
  await click('Close');
  await contains('Show progress');
  expect(host.textContent).toContain('Installing');
  expect(button('Review and install')).toBeNull();
  await click('Show progress');
  await contains('Installing Example Studio');
  expect(posts.filter((post) => post.path === '/v1/plugins/managed/catalog/install')).toEqual([]);
});

it('resumes a stopped install on its own installation, with the consent this browser saved, through to running', async () => {
  freshHost = true;
  stray = registered(strayId);
  strayOperation = stoppedAt('5'.repeat(32));
  saveJourney(consent);
  await render();
  // Browse opens on the list: a stopped install is not followed.
  await contains('Install needs a retry');
  await contains('Installing release 51 stopped before it finished.');
  expect(button('Review and install')).toBeNull();
  await click('Resume install');
  await contains('Installing Example Studio');
  await running();
  // No second installation: the stopped one is recovered, then activated with the accepted permissions.
  expect(posts.map((post) => post.path)).toEqual([
    `/v1/plugins/managed/installations/${strayId}/install/${'5'.repeat(32)}/recover`,
    `/v1/plugins/managed/installations/${strayId}/operations`,
  ]);
  expect(posts[0].body).toEqual({ expected_source_revision: 8 });
  expect(posts[1].body).toMatchObject({ action: 'activate', expected_revision: 0, runtime_digest: newDigest, accept_permissions: true });
  expect(readJourneys()).toEqual([]);
});

it('shows the release for review again before a retry this browser holds no consent for', async () => {
  freshHost = true;
  stray = registered(strayId);
  strayOperation = stoppedAt('5'.repeat(32));
  await render();
  await contains('Resume install');
  await click('Resume install');
  await contains('Retry installing Example Studio');
  await contains('This browser has no record that you accepted them');
  await contains('Use models on the fabric through the host API');
  expect(button('Retry install')?.disabled).toBe(true);
  expect(posts).toEqual([]);
  await act(async () => { (host.querySelector('#catalog-retry-consent') as HTMLInputElement).click(); });
  await click('Retry install');
  await running();
  expect(posts.map((post) => post.path)).toEqual([
    `/v1/plugins/managed/installations/${strayId}/install/${'5'.repeat(32)}/recover`,
    `/v1/plugins/managed/installations/${strayId}/operations`,
  ]);
  expect(posts[1].body).toMatchObject({ runtime_digest: newDigest, accept_permissions: true });
});

it('retries a fresh install whose preparation stopped from its own progress, without asking again', async () => {
  freshHost = true;
  stopFreshInstall = true;
  await render();
  await contains('Review and install');
  await click('Review and install');
  await act(async () => { (host.querySelector('#catalog-consent') as HTMLInputElement).click(); });
  await click('Install');
  await contains('The download finished, but preparing its runtime failed. Choose Retry');
  await click('Retry');
  await running();
  const pluginId = stray!.plugin_id;
  expect(posts.map((post) => post.path)).toEqual([
    '/v1/plugins/managed/catalog/install',
    `/v1/plugins/managed/installations/${pluginId}/install`,
    `/v1/plugins/managed/installations/${pluginId}/install/${String(posts[1].body.operation_id)}/recover`,
    `/v1/plugins/managed/installations/${pluginId}/operations`,
  ]);
});

it('starts a retry handed over from Installed once', async () => {
  freshHost = true;
  stray = registered(strayId);
  strayOperation = stoppedAt('5'.repeat(32));
  saveJourney(consent);
  let taken = 0;
  const handoff: BrowseRetryHandoff = { id: 1, request: { pluginId: strayId, title: 'example.studio', publisher: 'example', sequence: 51, transferBytes: 12_000_000, updating: false } };
  await render(handoff, () => { taken += 1; });
  await running();
  // Rendering again with the same handoff starts nothing more.
  await render(handoff, () => { taken += 1; });
  expect(taken).toBe(1);
  expect(posts.filter((post) => post.path.endsWith('/recover'))).toHaveLength(1);
  // The title this browser saved at consent names it, not the bundle id the card passed.
  expect(host.querySelector('#capability-setup-title')?.textContent).toBe('Example Studio is installed');
});

it('pauses new installs while an installation that names no bundle could not be read', async () => {
  freshHost = true;
  stray = registered(strayId);
  strayOperation = stoppedAt('5'.repeat(32));
  failStrayRead = true;
  await render();
  await contains('could not be read, so new installs are paused');
  // Without its install, the stopped one looks like nothing: a new install would sit beside it.
  expect(button('Review and install')?.disabled).toBe(true);
  failStrayRead = false;
  await click('Try again');
  await contains('Resume install');
  expect(host.textContent).not.toContain('new installs are paused');
  expect(posts).toEqual([]);
});

it('keeps a retry of an older release on its card while it runs, though the catalog lists a newer one', async () => {
  freshHost = true;
  stray = registered(strayId);
  strayOperation = stoppedAt('5'.repeat(32));
  recoverStaysRunning = true;
  listed = [entry, { ...entry, sequence: 52, release_digest: '1'.repeat(64) }];
  saveJourney(consent);
  await render();
  await contains('Installing release 51 stopped before it finished.');
  await click('Resume install');
  await act(async () => { await vi.waitFor(() => expect(posts.filter((post) => post.path.endsWith('/recover'))).toHaveLength(1), { timeout: 5000 }); });
  await click('Close');
  // The retry is still the bundle's only installation: release 52 is not offered beside it.
  await contains('Show progress');
  expect(button('Review and install')).toBeNull();
  await click('Show progress');
  await contains('Installing Example Studio');
  expect(posts.filter((post) => post.path === '/v1/plugins/managed/catalog/install')).toEqual([]);
});

it('follows a first install under way from another browser instead of installing beside it, after review', async () => {
  freshHost = true;
  stray = registered(strayId);
  strayOperation = { ...stoppedAt('5'.repeat(32)), state: 'staged', error_code: null };
  await render();
  await contains('Release 51 is being installed on this host.');
  expect(button('Review and install')).toBeNull();
  await click('Show progress');
  await contains('Finish installing Example Studio');
  await contains('This release is already being installed on this host.');
  expect(posts).toEqual([]);
  await act(async () => { (host.querySelector('#catalog-retry-consent') as HTMLInputElement).click(); });
  await click('Finish install');
  await running();
  // Nothing is recovered or bound: the staged release is activated with the accepted permissions.
  expect(posts.map((post) => post.path)).toEqual([`/v1/plugins/managed/installations/${strayId}/operations`]);
  expect(posts[0].body).toMatchObject({ runtime_digest: newDigest, accept_permissions: true });
});
