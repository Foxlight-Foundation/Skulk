import { configureStore } from '@reduxjs/toolkit';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { apiSlice } from '../../store/api';
import { darkTheme } from '../../theme/theme';
import { NodePreflightPanel } from './NodePreflightPanel';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });
vi.mock('../../i18n/tolgee', () => ({ useSkulkTranslation: () => ({ t: (_key: string, fallback: string, params?: Record<string, unknown>) => fallback.replace(/\{(\w+)\}/g, (_match, name: string) => String(params?.[name] ?? '')) }) }));
let root: Root;
let host: HTMLDivElement;
let calls: string[];
let failure: boolean;
let allPass: boolean;
let refuseEnable: boolean;
let nodeEnabled: boolean;
let mutations: unknown[];
let store: ReturnType<typeof makeStore>;
function makeStore() { return configureStore({ reducer: { [apiSlice.reducerPath]: apiSlice.reducer }, middleware: (defaults) => defaults().concat(apiSlice.middleware) }); }
async function contains(text: string) { await act(async () => { await vi.waitFor(() => expect(host.textContent).toContain(text)); }); }
async function check() { await act(async () => { host.querySelector('button')?.click(); }); }
beforeEach(async () => {
  calls = [];
  failure = false;
  allPass = false;
  refuseEnable = false;
  nodeEnabled = true;
  mutations = [];
  vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = input instanceof Request ? input : new Request(new URL(String(input), location.href), init);
    calls.push(request.method);
    const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
    if (new URL(request.url).pathname.endsWith('/configuration')) {
      const configuration = { nodeId: 'node', revision: 7, schemaDigest: 'd'.repeat(64), enabled: nodeEnabled, values: {}, configurationSchema: { type: 'object', properties: {} } };
      if (request.method === 'GET') return json(configuration);
      mutations.push(await request.json());
      return refuseEnable ? json({ detail: 'node preflight failed' }, 409) : json({ configuration: { ...configuration, revision: 8 }, validated: true });
    }
    return new Response(JSON.stringify(failure ? {} : { nodeId: 'node', revision: 4, observedAt: 1800000000, checks: [
      { code: 'runtime', passed: true, correctiveAction: null },
      { code: 'credentials', passed: allPass, correctiveAction: allPass ? null : 'Supply the required credential.' },
      { code: 'configuration', passed: true, correctiveAction: null },
      { code: 'durable_storage', passed: true, correctiveAction: null },
      { code: 'observation_current', passed: true, correctiveAction: null },
      { code: 'placed_video_model', passed: true, correctiveAction: null },
    ] }), { status: failure ? 503 : 200, headers: { 'Content-Type': 'application/json' } });
  });
  store = makeStore();
  host = document.createElement('div'); document.body.appendChild(host); root = createRoot(host);
  await render(0);
});
async function render(runRequest: number, offerStartAgain = false) {
  await act(async () => { root.render(<Provider store={store}><ThemeProvider theme={darkTheme}><NodePreflightPanel pluginId="plugin" nodeId="node" runRequest={runRequest} offerStartAgain={offerStartAgain} /></ThemeProvider></Provider>); });
}
function button(label: string) { return [...host.querySelectorAll('button')].find((item) => item.textContent === label) ?? null; }
afterEach(async () => { await act(async () => { root.unmount(); store.dispatch(apiSlice.util.resetApiState()); }); host.remove(); vi.unstubAllGlobals(); });
it('runs explicit read checks and renders corrective actions without enabling', async () => {
  expect(calls).toHaveLength(0);
  await check();
  await contains('Supply the required credential.');
  expect(host.textContent).toContain('against settings revision 4');
  expect(host.textContent).toContain('1 of 6 checks need attention.');
  expect(host.textContent).toContain('Turning it on runs fresh checks');
  expect(calls).toEqual(['GET']);
});

it('names each check in plain words, never by its raw code', async () => {
  await check();
  await contains('Supply the required credential.');
  const titles = [...host.querySelectorAll('li strong')].map((item) => item.textContent);
  expect(titles).toEqual(['Runtime', 'Credentials', 'Settings', 'Storage', 'Status report', 'Placed video model']);
  expect(host.textContent).not.toContain('durable_storage');
  expect(host.textContent).not.toContain('observation_current');
  // The one failing check is marked as such, the rest as passed.
  expect([...host.querySelectorAll('li')].map((item) => item.textContent?.includes('Needs attention'))).toEqual([false, true, false, false, false, false]);
});

it('runs the checks when its caller asks, as after a refused turn-on', async () => {
  expect(calls).toHaveLength(0);
  await render(1);
  await contains('Supply the required credential.');
  expect(calls).toEqual(['GET']);
});
it('does not present an earlier report as current after a failed refresh', async () => {
  await check(); await contains('Supply the required credential.');
  failure = true;
  await check(); await contains('Setup checks are unavailable.');
  expect(host.textContent).not.toContain('against settings revision');
  expect(calls).toEqual(['GET', 'GET']);
});

it('offers to start a stopped node again once its checks pass, with a fenced enable', async () => {
  allPass = true;
  await render(0, true);
  await check();
  await contains('All 6 checks passed.');
  await contains('it stopped when a check failed while it was starting');
  await act(async () => { button('Start it again')!.click(); });
  await contains('Starting it again. Its status updates here in a moment.');
  expect(mutations).toEqual([{ operation: 'enable', expectedRevision: 7, expectedSchemaDigest: 'd'.repeat(64) }]);
});

it('does not offer it while a check fails, or for a node that is not stopped', async () => {
  await render(0, true);
  await check();
  await contains('1 of 6 checks need attention.');
  expect(button('Start it again')).toBeNull();
  allPass = true;
  await render(0, false);
  await check();
  await contains('All 6 checks passed.');
  expect(button('Start it again')).toBeNull();
  expect(mutations).toEqual([]);
});

it('shows the host\'s reason when it still does not start', async () => {
  allPass = true;
  refuseEnable = true;
  await render(0, true);
  await check();
  await contains('All 6 checks passed.');
  await act(async () => { button('Start it again')!.click(); });
  await contains('It did not start. The host said: node preflight failed');
});

it('leaves a node that is turned off by the time it is started again off', async () => {
  // Another operator turned it off after this page saw it stopped.
  allPass = true;
  nodeEnabled = false;
  await render(0, true);
  await check();
  await contains('All 6 checks passed.');
  await act(async () => { button('Start it again')!.click(); });
  await contains('It is turned off now, so it was left off. Turn it on from its settings to start it.');
  expect(mutations).toEqual([]);
});

