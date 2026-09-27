import { configureStore } from '@reduxjs/toolkit';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { apiSlice } from '../../store/api';
import type { CatalogEntry, ManagedRuntime } from '../../store/endpoints/plugins';
import { darkTheme } from '../../theme/theme';
import { encodeInvitation, readJourneys } from './catalogJourney';
import { PluginCatalogBrowse } from './PluginCatalogBrowse';

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

function makeStore() { return configureStore({ reducer: { [apiSlice.reducerPath]: apiSlice.reducer }, middleware: (defaults) => defaults().concat(apiSlice.middleware) }); }
function json(body: unknown, status = 200) { return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }); }
async function contains(text: string) { await act(async () => { await vi.waitFor(() => expect(host.textContent).toContain(text), { timeout: 5000 }); }); }
function button(label: string) { return [...host.querySelectorAll('button')].find((item) => item.textContent === label) ?? null; }
async function click(label: string) { await act(async () => { button(label)?.click(); }); }
const entry: CatalogEntry = {
  bundle_id: 'example.studio', bundle_version: '0.1.0', title: 'Example Studio', publisher: 'example', sequence: 51,
  release_digest: 'd'.repeat(64), runtime_platform: 'macos-arm64', artifact_sha256: 'a'.repeat(64), artifact_bytes: 4096, transfer_bytes: 12_086_479,
  platforms: ['darwin'], skulk_build_sha256: 'b'.repeat(64), permissions: ['Use models on the fabric through the host API'],
  descriptors: ['studio.render@1.0.0'], surfaces: ['Example Studio'], operations: true, steward_risks: ['observation'],
  expires_at: 1_900_000_000, matches_host: true,
};

beforeEach(async () => {
  posts = [];
  configured = true;
  freshHost = false;
  loseNextBind = false;
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
    if (request.method === 'POST') {
      const body = await request.json() as Record<string, unknown>;
      posts.push({ path, body });
      if (path === '/v1/plugins/managed/catalog/source') { configured = true; return json({ revision: 1, configured: true, credential_reference: null, credential_ready: true, trust_revision: 1 }); }
      if (path === '/v1/plugins/managed/catalog/install') {
        // The host acts on the request, but the reply never arrives.
        if (loseNextBind) { loseNextBind = false; throw new TypeError('network'); }
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
    if (path === '/v1/plugins/managed/catalog/source') return json({ revision: configured ? 1 : 0, configured, credential_reference: null, credential_ready: configured, trust_revision: configured ? 1 : null });
    if (path === '/v1/plugins/managed/catalog') return json({ publisher: 'example', revision: 34, created_at: 1_790_000_000, expires_at: 1_800_000_000, catalog_sha256: 'c'.repeat(64), entries: [entry] });
    if (path === '/v1/plugins/managed') return json({ installations: freshHost ? [] : [installed] });
    if (path.endsWith('/install')) return json({ operation: installOperationId ? { request: { operation_id: installOperationId, runtime_digest: newDigest, expected_source_revision: 7 }, review: {}, state: 'staged', downloaded_bytes: 12_000_000, error_code: null } : null });
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

async function render() {
  await act(async () => { root.render(<Provider store={store}><ThemeProvider theme={darkTheme}><PluginCatalogBrowse /></ThemeProvider></Provider>); });
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
  await act(async () => { await vi.waitFor(() => expect(button('Set it up')).not.toBeNull(), { timeout: 8000 }); });
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
  expect(posts[0]).toMatchObject({ path: '/v1/plugins/managed/catalog/source', body: { expected_revision: 0, base_url: 'https://catalog.example.ts.net/', trust: { revision: 1, expires_at: 4_000_000_000, publishers: { example: 'f'.repeat(64) } } } });
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
  await click('Back to Browse');
  await contains('Review and install');
  await click('Review and install');
  await act(async () => { (host.querySelector('#catalog-consent') as HTMLInputElement).click(); });
  await click('Install');
  await act(async () => { await vi.waitFor(() => expect(posts.filter((post) => post.path === '/v1/plugins/managed/catalog/install')).toHaveLength(2), { timeout: 5000 }); });
  const binds = posts.filter((post) => post.path === '/v1/plugins/managed/catalog/install').map((post) => post.body.plugin_id);
  expect(binds[0]).toMatch(/^managed\.[0-9a-f]{32}$/);
  expect(binds[1]).toBe(binds[0]);
});
