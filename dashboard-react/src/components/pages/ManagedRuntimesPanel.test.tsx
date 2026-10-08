import { configureStore } from '@reduxjs/toolkit';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { apiSlice } from '../../store/api';
import type { ManagedOperation, ManagedRuntime, RuntimeInstallation } from '../../store/endpoints/plugins';
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

function makeStore() {
  return configureStore({ reducer: { [apiSlice.reducerPath]: apiSlice.reducer }, middleware: (defaults) => defaults().concat(apiSlice.middleware) });
}
function response(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}
async function mount() {
  root = createRoot(host);
  await act(async () => { root.render(<Provider store={store}><ThemeProvider theme={darkTheme}><ManagedRuntimesPanel onRetryInstall={(request) => { retried.push(request); }} /></ThemeProvider></Provider>); });
  await contains('Configure');
  await click('Configure');
}
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
  await click('Disable runtime');
  await contains('Applying');
  expect(posts).toHaveLength(1);
  expect(posts[0].body).toMatchObject({ action: 'disable', expected_revision: 7 });
  const id = String(posts[0].body?.operation_id);
  expect(id).toMatch(/^[a-f0-9]{32}$/);
  await act(async () => { root.unmount(); store.dispatch(apiSlice.util.resetApiState()); });
  await mount();
  await contains('Applying');
  await click('Refresh operation status');
  expect(posts).toHaveLength(1);
  expect(operationReads.some((path) => path.endsWith('/' + id))).toBe(true);
});

it('requires a separate recovery action for the original local operation', async () => {
  operation = { request: { operation_id: 'b'.repeat(32), action: 'disable' }, state: 'recovery_required', error_code: 'local_io_failed' };
  runtime = { ...runtime, operation_id: operation.request.operation_id, operation_state: operation.state };
  await act(async () => { store.dispatch(apiSlice.util.invalidateTags(['Plugins'])); });
  await contains('Recover local operation');
  expect(posts).toHaveLength(0);
  await click('Disable runtime');
  expect(posts).toHaveLength(0);
  await click('Recover local operation');
  await contains('Complete');
  expect(posts).toEqual([{ path: '/v1/plugins/managed/installations/managed.fixture/operations/' + 'b'.repeat(32) + '/recover', body: null }]);
});

it('offers explicit withdrawal of a failed initial activation without replaying it', async () => {
  operation = { request: { operation_id: 'c'.repeat(32), action: 'activate' }, state: 'recovery_required', error_code: 'validation_failed' };
  runtime = { ...runtime, enabled: false, selected_digest: null, selection_revision: 0, operation_id: operation.request.operation_id, operation_state: operation.state };
  await act(async () => { store.dispatch(apiSlice.util.invalidateTags(['Plugins'])); });
  await contains('Disable or uninstall withdraws this pending local change');
  expect(posts).toHaveLength(0);
  await click('Disable runtime');
  await contains('Applying');
  expect(posts).toHaveLength(1);
  expect(posts[0].body).toMatchObject({ action: 'disable', expected_revision: 0 });
  expect(posts[0].body?.operation_id).not.toBe('c'.repeat(32));
  await click('Disable runtime');
  expect(posts).toHaveLength(1);
});


it('retains an uncertain uninstall across reconnect without another submission', async () => {
  await contains('managed.fixture');
  await click('Uninstall plugin');
  await contains('Applying');
  expect(posts).toHaveLength(1);
  expect(posts[0].body).toMatchObject({ action: 'uninstall', expected_revision: 7 });
  await click('Uninstall plugin');
  expect(posts).toHaveLength(1);
  const originalOperationId = posts[0].body?.operation_id;
  await act(async () => { host.querySelector<HTMLButtonElement>('button[aria-label="Close"]')!.click(); });
  expect(host.querySelector('[role="dialog"]')).toBeNull();
  await click('Configure');
  await contains('Applying');
  await click('Refresh operation status');
  expect(posts).toHaveLength(1);
  expect(operationReads.some(path => path.endsWith(`/${originalOperationId}`))).toBe(true);
  await act(async () => { root.unmount(); store.dispatch(apiSlice.util.resetApiState()); });
  await mount();
  await contains('Applying');
  await click('Refresh operation status');
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

it('shows retained uninstall status and offers explicit release reinstallation', async () => {
  runtime = { ...runtime, uninstalled: true, enabled: false, service: null };
  await act(async () => { store.dispatch(apiSlice.util.invalidateTags(['Plugins'])); });
  await contains('Plugin uninstalled; cleanup state retained');
  await click('Uninstall plugin');
  await click('Disable runtime');
  expect(posts).toHaveLength(0);
  expect([...host.querySelectorAll('button')].find((item) => item.textContent === 'Install a release')?.disabled).toBe(false);
});


it('removes an uninstalled plugin only after a second, explicit confirmation', async () => {
  runtime = { ...runtime, uninstalled: true, enabled: false, stale: true, service: null };
  await act(async () => { store.dispatch(apiSlice.util.invalidateTags(['Plugins'])); });
  // An uninstalled installation reads as uninstalled even though nothing runs to be observed.
  await contains('Uninstalled');
  await contains('Cleanup state retained');
  await click('Remove uninstalled plugin');
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
  await click('Remove uninstalled plugin');
  await contains('Remove now');
  await act(async () => { host.querySelector<HTMLButtonElement>('button[aria-label="Close"]')!.click(); });
  await click('Configure');
  await contains('Remove uninstalled plugin');
  expect([...host.querySelectorAll('button')].some((item) => item.textContent === 'Remove now')).toBe(false);
  expect(deletes).toHaveLength(0);
});

it('says a refused removal even though the uninstall before it is complete', async () => {
  operation = { request: { operation_id: 'e'.repeat(32), action: 'uninstall' }, state: 'complete', error_code: null };
  runtime = { ...runtime, uninstalled: true, enabled: false, stale: true, service: null, operation_id: operation.request.operation_id, operation_state: 'complete' };
  refusePurge = true;
  await act(async () => { store.dispatch(apiSlice.util.invalidateTags(['Plugins'])); });
  await contains('Uninstalled');
  await click('Remove uninstalled plugin');
  await click('Remove now');
  await contains('Removal was refused');
  expect(deletes).toHaveLength(1);
  expect(host.textContent).toContain('managed.fixture');
});

it('allows uninstall to withdraw a stalled reinstallation while retaining uninstalled status', async () => {
  operation = { request: { operation_id: 'd'.repeat(32), action: 'select' }, state: 'recovery_required', error_code: 'validation_failed' };
  runtime = { ...runtime, uninstalled: true, enabled: false, operation_id: operation.request.operation_id, operation_state: operation.state };
  await act(async () => { store.dispatch(apiSlice.util.invalidateTags(['Plugins'])); });
  await contains('Disable or uninstall withdraws this pending local change');
  await click('Disable runtime');
  expect(posts).toHaveLength(0);
  await click('Uninstall plugin');
  await contains('Applying');
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
