import { configureStore } from '@reduxjs/toolkit';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { apiSlice } from '../../store/api';
import type { CatalogEntry, ManagedOperation, ManagedRuntime, RuntimeInstallation } from '../../store/endpoints/plugins';
import { darkTheme } from '../../theme/theme';
import { saveJourney, type RetryRequest } from './catalogJourney';
import { ManagedRuntimesPanel } from './ManagedRuntimesPanel';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });
vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({ t: (_key: string, fallback: string, params?: Record<string, unknown>) => fallback.replace(/\{(\w+)\}/g, (_match, name: string) => String(params?.[name] ?? '')) }),
}));

let root: Root;
let host: HTMLDivElement;
let runtime: ManagedRuntime;
let operation: ManagedOperation | null;
let posts: { path: string; body: Record<string, unknown> | null }[];
let deletes: string[];
let operationReads: string[];
let store: ReturnType<typeof makeStore>;
let purged: boolean;
let refusePurge: boolean;
let installOperation: RuntimeInstallation | null;
let retried: RetryRequest[];
// The catalog this host reads; null when it has none.
let listing: CatalogEntry[] | null;
let updates: string[];
// Whether the host refuses to start the release.
let refuseStart: boolean;

function makeStore() {
  return configureStore({ reducer: { [apiSlice.reducerPath]: apiSlice.reducer }, middleware: (defaults) => defaults().concat(apiSlice.middleware) });
}
function response(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}
async function mount(offerUpdates = false) {
  root = createRoot(host);
  await act(async () => { root.render(<Provider store={store}><ThemeProvider theme={darkTheme}><ManagedRuntimesPanel onRetryInstall={(request) => { retried.push(request); }}
    onUpdate={offerUpdates ? (bundleId) => { updates.push(bundleId); } : undefined} /></ThemeProvider></Provider>); });
  await contains('Configure');
  await click('Configure');
}
async function remount(offerUpdates = false) {
  await act(async () => { root.unmount(); store.dispatch(apiSlice.util.resetApiState()); });
  await mount(offerUpdates);
}
function entry(sequence: number, overrides: Partial<CatalogEntry> = {}): CatalogEntry {
  return {
    bundle_id: 'example.plugin', bundle_version: `1.${sequence - 1}.0`, title: 'Example Plugin', publisher: 'fixture', sequence,
    release_digest: 'd'.repeat(64), runtime_platform: 'macos-arm64', artifact_sha256: 'a'.repeat(64), artifact_bytes: 4096, transfer_bytes: 4096,
    platforms: ['darwin'], skulk_build_sha256: 'b'.repeat(64), skulk_requires: '>=2.0.0,<3', permissions: [], descriptors: [], surfaces: [], operations: false,
    steward_risks: [], expires_at: 1_900_000_000, matches_host: true, ...overrides,
  };
}
const releaseOne = { bundle_id: 'example.plugin', title: 'Example Plugin', bundle_version: '1.0.0', publisher: 'fixture', sequence: 1 };
async function click(label: string) {
  await act(async () => {
    const button = [...host.querySelectorAll('button')].find((item) => item.textContent === label);
    if (!button) throw new Error(`Missing button: ${label}`);
    button.click();
  });
}
async function contains(text: string) {
  await act(async () => { await vi.waitFor(() => expect(host.textContent).toContain(text)); });
}
// A button in the drawer; the card's menu repeats some of its labels.
function find(label: string) { return [...(host.querySelector('[role=dialog]') ?? host).querySelectorAll<HTMLButtonElement>('button')].find((item) => item.textContent === label) ?? null; }
// The runtime card's header: its release line and status pill.
function runtimeHeader() { return host.querySelector('[role=dialog] article header')?.textContent ?? ''; }

beforeEach(async () => {
  runtime = { plugin_id: 'managed.fixture', selected_digest: 'a'.repeat(64), selection_revision: 7, enabled: true, stale: false, error_code: null, operation_id: null, operation_state: null,
    service: { state: 'running', active_digest: 'a'.repeat(64), observed_at: 1 } };
  operation = null;
  posts = [];
  deletes = [];
  operationReads = [];
  purged = false;
  refusePurge = false;
  installOperation = null;
  retried = [];
  listing = null;
  updates = [];
  refuseStart = false;
  localStorage.removeItem('skulk-plugin-install-journeys');
  vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = new Request(input, init);
    const path = new URL(request.url).pathname;
    expect(request.headers.get('X-Skulk-Dashboard')).toBe('pairing-v1');
    if (request.method === 'POST') {
      const body = path.endsWith('/recover') ? null : await request.json() as Record<string, unknown>;
      posts.push({ path, body });
      if (path.endsWith('/recover')) {
        if (!operation) throw new Error('Missing retained operation');
        operation = { ...operation, state: 'complete' };
        runtime = { ...runtime, operation_state: 'complete', enabled: false };
        return response(operation);
      }
      if (body?.action === 'activate' && refuseStart) return response({ detail: 'The installation changed since it was read. Refresh it, then try again.' }, 409);
      const id = String(body?.operation_id);
      operation = { request: { operation_id: id, action: body?.action === 'uninstall' ? 'uninstall' : 'disable' }, state: 'applying', error_code: null };
      runtime = { ...runtime, operation_id: id, operation_state: 'applying' };
      // The manager accepted the original request but the browser lost its response.
      return response({}, 503);
    }
    if (request.method === 'DELETE') {
      deletes.push(path);
      // The host removes an uninstalled installation, or one that never selected a release.
      if ((!runtime.uninstalled && runtime.selected_digest !== null) || refusePurge) return response({ detail: 'refused' }, 409);
      purged = true;
      return response({ plugin_id: runtime.plugin_id, purged: true });
    }
    if (path === '/v1/plugins/managed') return response({ installations: purged ? [] : [runtime] });
    if (path === '/v1/plugins/managed/catalog/source') {
      return response({ revision: 1, configured: listing !== null, credential_reference: null, credential_ready: true, trust_revision: 1, builtin_store: true, builtin_store_available: true });
    }
    if (path === '/v1/plugins/managed/catalog') {
      return listing ? response({ publisher: 'fixture', revision: 1, created_at: 1_790_000_000, expires_at: 1_900_000_000, catalog_sha256: 'c'.repeat(64), entries: listing }) : response({}, 404);
    }
    if (path.endsWith('/source')) return response({ revision: 1, configured: true, credential_reference: null, credential_ready: true, trust_revision: 1, follows_store: true });
    if (path.endsWith('/install')) return response({ operation: installOperation });
    operationReads.push(path);
    return operation ? response(operation) : response({}, 404);
  });
  store = makeStore();
  host = document.createElement('div');
  document.body.appendChild(host);
  await mount();
});
afterEach(async () => {
  await act(async () => { root.unmount(); store.dispatch(apiSlice.util.resetApiState()); });
  host.remove();
  vi.unstubAllGlobals();
  localStorage.removeItem('skulk-plugin-install-journeys');
});

it('recovers a server-retained operation reference after reconnect without another submission', async () => {
  await contains('managed.fixture');
  await click('Stop plugin');
  await contains('Updating');
  expect(posts).toHaveLength(1);
  expect(posts[0].body).toMatchObject({ action: 'disable', expected_revision: 7 });
  const id = String(posts[0].body?.operation_id);
  expect(id).toMatch(/^[a-f0-9]{32}$/);
  await act(async () => { root.unmount(); store.dispatch(apiSlice.util.resetApiState()); });
  await mount();
  await click('Details');
  await contains('Applying');
  await click('Refresh status');
  expect(posts).toHaveLength(1);
  expect(operationReads.some((path) => path.endsWith('/' + id))).toBe(true);
});

it('requires a separate recovery action for the original local operation', async () => {
  operation = { request: { operation_id: 'b'.repeat(32), action: 'disable' }, state: 'recovery_required', error_code: 'local_io_failed' };
  runtime = { ...runtime, operation_id: operation.request.operation_id, operation_state: operation.state };
  await act(async () => { store.dispatch(apiSlice.util.invalidateTags(['Plugins'])); });
  await contains('Recover local operation');
  expect(posts).toHaveLength(0);
  await click('Stop plugin');
  expect(posts).toHaveLength(0);
  await click('Recover local operation');
  await click('Details');
  await contains('Complete');
  expect(posts).toEqual([{ path: '/v1/plugins/managed/installations/managed.fixture/operations/' + 'b'.repeat(32) + '/recover', body: null }]);
});

it('offers explicit withdrawal of a failed initial activation without replaying it', async () => {
  operation = { request: { operation_id: 'c'.repeat(32), action: 'activate' }, state: 'recovery_required', error_code: 'validation_failed' };
  runtime = { ...runtime, enabled: false, selected_digest: null, selection_revision: 0, operation_id: operation.request.operation_id, operation_state: operation.state };
  await act(async () => { store.dispatch(apiSlice.util.invalidateTags(['Plugins'])); });
  await contains('Stop plugin or Uninstall withdraws this pending change');
  expect(posts).toHaveLength(0);
  await click('Stop plugin');
  await contains('Updating');
  expect(posts).toHaveLength(1);
  expect(posts[0].body).toMatchObject({ action: 'disable', expected_revision: 0 });
  expect(posts[0].body?.operation_id).not.toBe('c'.repeat(32));
  await click('Stop plugin');
  expect(posts).toHaveLength(1);
});


it('retains an uncertain uninstall across reconnect without another submission', async () => {
  await contains('managed.fixture');
  await click('Uninstall');
  await contains('Updating');
  expect(posts).toHaveLength(1);
  expect(posts[0].body).toMatchObject({ action: 'uninstall', expected_revision: 7 });
  await click('Uninstall');
  expect(posts).toHaveLength(1);
  const originalOperationId = posts[0].body?.operation_id;
  await act(async () => { host.querySelector<HTMLButtonElement>('button[aria-label="Close"]')!.click(); });
  expect(host.querySelector('[role="dialog"]')).toBeNull();
  await click('Configure');
  await click('Details');
  await contains('Applying');
  await click('Refresh status');
  expect(posts).toHaveLength(1);
  expect(operationReads.some(path => path.endsWith(`/${originalOperationId}`))).toBe(true);
  await act(async () => { root.unmount(); store.dispatch(apiSlice.util.resetApiState()); });
  await mount();
  await click('Details');
  await contains('Applying');
  await click('Refresh status');
  expect(posts).toHaveLength(1);
});

it('names the signed release and says why a stopped plugin cannot run', async () => {
  runtime = { ...runtime, stale: true,
    release: { bundle_id: 'foxlight.video-studio', title: 'Skulk Video Studio', bundle_version: '0.1.0', publisher: 'foxlight', sequence: 49 },
    service: { state: 'failed', active_digest: null, observed_at: 1, error_code: 'verification_failed' } };
  await act(async () => { store.dispatch(apiSlice.util.invalidateTags(['Plugins'])); });
  await contains('Skulk Video Studio');
  await contains('built for a different Skulk build');
  await contains('Needs attention');
  await contains('Not running');
  expect(host.textContent).not.toContain('Status unavailable');
});

it('falls back to the bundle identifier when the manifest declares no title', async () => {
  runtime = { ...runtime, release: { bundle_id: 'example.plugin', title: null, bundle_version: '1.0.0', publisher: 'fixture', sequence: 1 } };
  await act(async () => { store.dispatch(apiSlice.util.invalidateTags(['Plugins'])); });
  await contains('example.plugin');
});

it('leads an uninstalled plugin with what can be done now: reinstall it, or remove everything', async () => {
  runtime = { ...runtime, uninstalled: true, enabled: false, service: null };
  await act(async () => { store.dispatch(apiSlice.util.invalidateTags(['Plugins'])); });
  await contains('It is uninstalled. Its settings, credentials and records are kept, so it can come back as it was.');
  expect(runtimeHeader()).toContain('Uninstalled');
  // Stop, Start and Uninstall have nothing left to do.
  expect(find('Stop plugin')).toBeNull();
  expect(find('Start plugin')).toBeNull();
  expect(find('Uninstall')).toBeNull();
  expect(find('Remove everything')?.disabled).toBe(false);
  await contains('Remove everything deletes what it kept; after that you can install it fresh from Browse.');
  await click('Details');
  expect(find('Choose another release')?.disabled).toBe(false);
  expect(posts).toHaveLength(0);
  // Reinstalling starts the release it kept on this installation, with the permissions it already holds.
  await click('Reinstall');
  await act(async () => { await vi.waitFor(() => expect(posts).toHaveLength(1)); });
  expect(posts[0]).toMatchObject({ path: '/v1/plugins/managed/installations/managed.fixture/operations',
    body: { action: 'activate', expected_revision: 7, runtime_digest: 'a'.repeat(64), rollback: false, accept_permissions: false } });
});


it('removes an uninstalled plugin only after a second, explicit confirmation', async () => {
  runtime = { ...runtime, uninstalled: true, enabled: false, stale: true, service: null };
  await act(async () => { store.dispatch(apiSlice.util.invalidateTags(['Plugins'])); });
  // An uninstalled installation reads as uninstalled even though nothing runs to be observed.
  await contains('Uninstalled');
  await contains('Cleanup state retained');
  await click('Remove everything');
  await contains('Removing deletes everything this uninstalled plugin retained');
  expect(deletes).toHaveLength(0);
  await click('Remove now');
  await act(async () => { await vi.waitFor(() => expect(deletes).toHaveLength(1)); });
  expect(deletes[0]).toBe('/v1/plugins/managed/installations/managed.fixture');
  await act(async () => { await vi.waitFor(() => expect(host.textContent).not.toContain('managed.fixture')); });
});

it('disarms removal when the drawer is dismissed, so reopening asks again', async () => {
  runtime = { ...runtime, uninstalled: true, enabled: false, stale: true, service: null };
  await act(async () => { store.dispatch(apiSlice.util.invalidateTags(['Plugins'])); });
  await contains('Uninstalled');
  await click('Remove everything');
  await contains('Remove now');
  await act(async () => { host.querySelector<HTMLButtonElement>('button[aria-label="Close"]')!.click(); });
  await click('Configure');
  await contains('Remove everything');
  expect([...host.querySelectorAll('button')].some((item) => item.textContent === 'Remove now')).toBe(false);
  expect(deletes).toHaveLength(0);
});

it('says a refused removal even though the uninstall before it is complete', async () => {
  operation = { request: { operation_id: 'e'.repeat(32), action: 'uninstall' }, state: 'complete', error_code: null };
  runtime = { ...runtime, uninstalled: true, enabled: false, stale: true, service: null, operation_id: operation.request.operation_id, operation_state: 'complete' };
  refusePurge = true;
  await act(async () => { store.dispatch(apiSlice.util.invalidateTags(['Plugins'])); });
  await contains('Uninstalled');
  await click('Remove everything');
  await click('Remove now');
  await contains('Removal was refused');
  expect(deletes).toHaveLength(1);
  expect(host.textContent).toContain('managed.fixture');
});

it('allows uninstall to withdraw a stalled reinstallation while retaining uninstalled status', async () => {
  operation = { request: { operation_id: 'd'.repeat(32), action: 'select' }, state: 'recovery_required', error_code: 'validation_failed' };
  runtime = { ...runtime, uninstalled: true, enabled: false, operation_id: operation.request.operation_id, operation_state: operation.state };
  await act(async () => { store.dispatch(apiSlice.util.invalidateTags(['Plugins'])); });
  await contains('Stop plugin or Uninstall withdraws this pending change');
  expect(find('Stop plugin')).toBeNull();
  expect(posts).toHaveLength(0);
  await click('Uninstall');
  await contains('Updating');
  expect(posts).toHaveLength(1);
  expect(posts[0].body).toMatchObject({ action: 'uninstall', expected_revision: 7 });
});

it('routes menu withdrawals through the existing operation fence and refuses duplicate submissions', async () => {
  await act(async () => {
    [...host.querySelectorAll('button')].find(item => item.getAttribute('aria-label') === 'Close')?.click();
  });
  const menu = host.querySelector('details')!;
  await act(async () => { menu.open = true; });
  await click('Uninstall plugin…');
  await contains('The request did not return a confirmed result.');
  expect(posts).toHaveLength(1);
  expect(posts[0].body).toMatchObject({ expected_revision: 7 });
  const action = [...host.querySelectorAll<HTMLButtonElement>('button')].find(item => item.textContent === 'Uninstall plugin…')!;
  expect(action.disabled).toBe(true);
  await act(async () => action.click());
  expect(posts).toHaveLength(1);
});

const strayId = 'managed.9598f7f0' + '0'.repeat(24);
function strayInstall(): void {
  runtime = { plugin_id: strayId, release: null, selected_digest: null, selection_revision: 0, enabled: false, stale: true, error_code: null, operation_id: null, operation_state: null, service: null };
  installOperation = {
    attempt: 0, request: { operation_id: '5'.repeat(32), runtime_digest: 'f'.repeat(64), expected_source_revision: 3 },
    review: { runtime_digest: 'f'.repeat(64), source_revision: 3, publisher: 'foxlight', bundle_id: 'foxlight.video-studio', version: '0.1.0', sequence: 49, platform: 'linux-glibc-x86_64', python_requires: '>=3.13', skulk_build_sha256: 'b'.repeat(64), permissions: [], artifact_bytes: 900_000_000, expires_at: 1_900_000_000 },
    state: 'recovery_required', downloaded_bytes: 900_000_000, error_code: 'installation_failed',
  };
}
function headings() { return [...host.querySelectorAll('h2')].map((heading) => heading.textContent); }

it('names a stopped first install by its catalog title and offers its retry on the card', async () => {
  strayInstall();
  saveJourney({ pluginId: strayId, title: 'Skulk Video Studio', bundleId: 'foxlight.video-studio', sequence: 49, publisher: 'foxlight', runtimeDigest: 'f'.repeat(64),
    transferBytes: 900_000_000, installOperationId: '5'.repeat(32), activationOperationId: null, startedAt: 1, interrupted: true });
  await act(async () => { store.dispatch(apiSlice.util.invalidateTags(['Plugins'])); });
  await contains('Install needs a retry');
  expect(headings()).toContain('Skulk Video Studio');
  expect(headings()).not.toContain(strayId);
  await contains('Release 49 downloaded, but preparing its runtime failed. Retry prepares it again and finishes the install.');
  await contains('0.1.0 (49) from foxlight');
  await contains('Not installed yet');
  expect(host.textContent).not.toContain('Release status unavailable');
  await click('Retry');
  expect(retried).toEqual([{ pluginId: strayId, title: 'Skulk Video Studio', publisher: 'foxlight', sequence: 49, transferBytes: 900_000_000, updating: false }]);
  // A retry is a choice the owner makes: nothing was sent to the host from here.
  expect(posts).toEqual([]);
});

it('names a stopped first install by its bundle id when this browser never saw its catalog title', async () => {
  strayInstall();
  await act(async () => { store.dispatch(apiSlice.util.invalidateTags(['Plugins'])); });
  await contains('Install needs a retry');
  expect(headings()).toContain('foxlight.video-studio');
  expect(headings()).not.toContain(strayId);
  // The same retry is in the details drawer and its menu.
  await act(async () => { host.querySelector('details')!.open = true; });
  await click('Retry install');
  expect(retried).toHaveLength(1);
});

it('offers no retry once the stopped release is staged', async () => {
  strayInstall();
  installOperation = { ...installOperation!, state: 'staged', error_code: null };
  await act(async () => { store.dispatch(apiSlice.util.invalidateTags(['Plugins'])); });
  await contains('foxlight.video-studio');
  expect(host.textContent).not.toContain('Install needs a retry');
  expect([...host.querySelectorAll('button')].some((item) => item.textContent === 'Retry')).toBe(false);
});

it('removes a stray stopped install that never selected a release, after the same explicit confirmation', async () => {
  strayInstall();
  saveJourney({ pluginId: strayId, title: 'Skulk Video Studio', bundleId: 'foxlight.video-studio', sequence: 49, publisher: 'foxlight', runtimeDigest: 'f'.repeat(64),
    transferBytes: 900_000_000, installOperationId: '5'.repeat(32), activationOperationId: null, startedAt: 1, interrupted: true });
  await act(async () => { store.dispatch(apiSlice.util.invalidateTags(['Plugins'])); });
  await contains('Install needs a retry');
  await act(async () => { host.querySelector('details')!.open = true; });
  await click('Remove this installation…');
  await contains('Removing deletes this installation and everything its stopped install kept');
  expect(deletes).toHaveLength(0);
  await click('Remove now');
  await act(async () => { await vi.waitFor(() => expect(deletes).toEqual([`/v1/plugins/managed/installations/${strayId}`])); });
  // Nothing is left to retry, so this browser's consent record goes once the host confirms.
  await act(async () => { await vi.waitFor(() => expect(localStorage.getItem('skulk-plugin-install-journeys')).toBe('[]')); });
});

it('labels the runtime by its state, under its release and where it came from', async () => {
  const cases: [Partial<ManagedRuntime>, string][] = [
    [{}, 'Running'],
    [{ enabled: false, service: null }, 'Stopped'],
    [{ service: { state: 'failed', active_digest: null, observed_at: 1, error_code: 'owner_exited' } }, 'Failed'],
    [{ service: { state: 'starting', active_digest: null, observed_at: 1 } }, 'Starting'],
    [{ operation_id: 'f'.repeat(32), operation_state: 'applying' }, 'Updating'],
    [{ uninstalled: true, enabled: false, service: null }, 'Uninstalled'],
  ];
  for (const [overrides, label] of cases) {
    runtime = { ...runtime, enabled: true, uninstalled: false, operation_id: null, operation_state: null, service: { state: 'running', active_digest: 'a'.repeat(64), observed_at: 1 }, release: releaseOne, ...overrides };
    await act(async () => { store.dispatch(apiSlice.util.invalidateTags(['Plugins'])); });
    await act(async () => { await vi.waitFor(() => expect(runtimeHeader()).toBe(`Release 1.0.0 from fixtureSequence 1 · From the Foxlight capability store${label}`)); });
  }
  await contains('Sequence 1 · From the Foxlight capability store');
  // The raw installation id and fingerprints wait under Details.
  expect(host.querySelector('[role=dialog] article')?.textContent).not.toContain('managed.fixture');
});

it('keeps the internals under a closed Details disclosure', async () => {
  const toggle = find('Details')!;
  expect(toggle.getAttribute('aria-expanded')).toBe('false');
  expect(host.textContent).not.toContain('Local operation');
  await click('Details');
  expect(find('Details')!.getAttribute('aria-expanded')).toBe('true');
  const details = host.querySelector('[role=dialog] dl')!;
  expect(details.textContent).toContain('managed.fixture');
  expect(details.textContent).toContain('Selected release');
  // A fingerprint is short, with the whole value on hover.
  const selected = [...details.querySelectorAll('code')].find((code) => code.textContent === 'a'.repeat(12))!;
  expect(selected.getAttribute('title')).toBe('a'.repeat(64));
  expect(details.textContent).toContain('Local operation');
  expect(details.textContent).toContain('None');
  await contains('Uninstalling is not a data purge: to delete everything it kept, uninstall it, then choose Remove everything.');
  expect(find('Choose another release')).not.toBeNull();
  expect([...host.querySelectorAll('[role=dialog] button')].some((item) => item.textContent === 'Install a release')).toBe(false);
});

it('stops a running plugin and starts a stopped one with the release it already selected', async () => {
  await contains('Stop plugin stops all of its capabilities until you start it again. Its settings and data stay.');
  // The node switch below says Turn on and Turn off; the runtime never does.
  expect(find('Turn off')).toBeNull();
  await click('Stop plugin');
  await act(async () => { await vi.waitFor(() => expect(posts).toHaveLength(1)); });
  expect(posts[0]).toMatchObject({ path: '/v1/plugins/managed/installations/managed.fixture/operations', body: { action: 'disable', expected_revision: 7 } });
  runtime = { ...runtime, enabled: false, service: null, release: releaseOne, operation_id: null, operation_state: null };
  await remount();
  await contains('Start plugin runs release 1.0.0 again with its settings.');
  expect(find('Stop plugin')).toBeNull();
  await click('Start plugin');
  await act(async () => { await vi.waitFor(() => expect(posts).toHaveLength(2)); });
  expect(posts[1].body).toMatchObject({ action: 'activate', expected_revision: 7, runtime_digest: 'a'.repeat(64), rollback: false, accept_permissions: false });
});

it('offers an update only when a newer release fits this host, as Browse does', async () => {
  runtime = { ...runtime, release: releaseOne };
  // The same release: nothing newer.
  listing = [entry(1)];
  await remount(true);
  await act(async () => { await vi.waitFor(() => expect(runtimeHeader()).toContain('Running')); });
  await act(async () => { await new Promise((resolve) => setTimeout(resolve, 300)); });
  expect([...host.querySelectorAll('button')].some((item) => item.textContent?.startsWith('Update to'))).toBe(false);
  // A newer release built for another host only.
  listing = [entry(1), entry(2, { matches_host: false, release_digest: 'e'.repeat(64) })];
  await remount(true);
  await act(async () => { await new Promise((resolve) => setTimeout(resolve, 300)); });
  expect([...host.querySelectorAll('button')].some((item) => item.textContent?.startsWith('Update to'))).toBe(false);
  // A newer release that fits.
  listing = [entry(1), entry(2, { release_digest: 'e'.repeat(64) })];
  await remount(true);
  await act(async () => { await vi.waitFor(() => expect(find('Update to 1.1.0')).not.toBeNull()); });
  await click('Update to 1.1.0');
  expect(updates).toEqual(['example.plugin']);
  expect(posts).toEqual([]);
});

it('shows a refused start even after a completed stop', async () => {
  // Stopped by a disable that completed; starting it is then refused.
  operation = { request: { operation_id: 'e'.repeat(32), action: 'disable' }, state: 'complete', error_code: null };
  runtime = { ...runtime, enabled: false, service: null, release: releaseOne, operation_id: operation.request.operation_id, operation_state: 'complete' };
  refuseStart = true;
  await remount();
  await click('Start plugin');
  await contains('It was not started. The host said: The installation changed since it was read. Refresh it, then try again.');
  expect(posts).toHaveLength(1);
  // Nothing was created, so starting is not held behind an operation that never existed.
  expect(find('Start plugin')?.disabled).toBe(false);
});
