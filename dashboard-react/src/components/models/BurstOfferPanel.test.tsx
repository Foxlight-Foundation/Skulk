import { configureStore } from '@reduxjs/toolkit';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { apiSlice } from '../../store/api';
import type { BurstOperation, BurstProvider, BurstRecommendation } from '../../store/endpoints/burst';
import { darkTheme } from '../../theme/theme';
import { BurstOfferPanel, formatUsd } from './BurstOfferPanel';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });
vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({
    t: (_key: string, fallback: string, params?: Record<string, unknown>) =>
      fallback.replace(/\{(\w+)\}/g, (_match, name: string) => String(params?.[name] ?? `{${name}}`)),
  }),
}));

let root: Root;
let host: HTMLDivElement;
let calls: { method: string; path: string; body: string }[];
let providers: BurstProvider[];
let phase: BurstOperation['phase'];
let store: ReturnType<typeof makeStore>;
const storageKey = 'skulk-burst-operation-v1:example%2Flarge';
const reference = { pluginId: 'managed.runpod', nodeId: 'gpu', proposalId: 'a'.repeat(64), proposalDigest: 'a'.repeat(64) };

function makeStore() { return configureStore({ reducer: { [apiSlice.reducerPath]: apiSlice.reducer }, middleware: (defaults) => defaults().concat(apiSlice.middleware) }); }
async function contains(text: string) { await act(async () => { await vi.waitFor(() => expect(host.textContent).toContain(text)); }); }
function button(text: string) { const result = [...host.querySelectorAll('button')].find((item) => item.textContent === text); if (!result) throw new Error(`Missing button ${text}`); return result; }
async function click(text: string) { await act(async () => { button(text).click(); }); }
async function render() {
  await act(async () => {
    root.render(<Provider store={store}><ThemeProvider theme={darkTheme}><BurstOfferPanel modelId="example/large" contextTokens={8192} /></ThemeProvider></Provider>);
  });
}

function provider(state: BurstProvider['state'], message: string): BurstProvider {
  return { pluginId: 'managed.runpod', nodeId: 'gpu', bundleId: 'foxlight.runpod', version: '0.5.0', title: 'Remote GPUs', memoryPools: ['vram'], state, message, maximumTotalUsd: '10.000000', maximumLifetimeSeconds: 3600 };
}

const recommendation: BurstRecommendation = {
  state: 'offer',
  message: 'Remote capacity is available. Nothing is rented until this offer is approved.',
  modelId: 'example/large',
  contextTokens: 8192,
  assessmentStatus: 'external_required',
  assessmentReasons: [],
  provider: null,
  reference,
  offer: { gpu: 'NVIDIA A100 80GB', region: 'US', hourlyPriceUsd: '1.190000', maximumHourlyPriceUsd: '2.000000', maximumTotalUsd: '4.000000', planExpiresAt: 9_999_999_999, resourceExpiresAt: 9_999_999_999, modelId: 'example/large', contextTokens: 8192, backend: 'llama_server-cuda', skulkBuild: '2.0.2' },
  approvalRevision: 'b'.repeat(64),
  review: [{ label: 'Cleanup terms', value: 'Released at expiry without a new approval.' }],
};

function operation(operationId: string): BurstOperation {
  return {
    ...reference, operationId, phase, message: phase === 'ready' ? 'The model is ready on the remote machine.' : 'The approval is being signed and sent to the provider.',
    instanceId: phase === 'ready' ? 'instance-1' : null, modelId: 'example/large', resourceExpiresAt: 9_999_999_999,
    acquisition: { operationId, reference, phase: 'acknowledged', updatedAt: 1, code: null, correctiveAction: null }, release: null,
  };
}

beforeEach(async () => {
  localStorage.removeItem(storageKey);
  calls = []; phase = 'approving';
  providers = [provider('available', 'Ready to prepare offers.')];
  vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = input instanceof Request ? input : new Request(new URL(String(input), location.href), init);
    const body = await request.text(); const url = new URL(request.url);
    calls.push({ method: request.method, path: url.pathname, body });
    let result: unknown;
    if (url.pathname === '/v1/burst/providers') result = { scope: 'local_api_host', observedAt: 1, providers };
    else if (url.pathname === '/v1/burst/recommendations') result = recommendation;
    else if (request.method === 'POST') result = operation(JSON.parse(body).operationId);
    else {
      if (request.method === 'DELETE') phase = 'releasing';
      result = operation(url.pathname.split('/').pop()!);
    }
    return new Response(JSON.stringify(result), { status: 200, headers: { 'Content-Type': 'application/json' } });
  });
  store = makeStore(); host = document.createElement('div'); document.body.appendChild(host); root = createRoot(host);
});
afterEach(async () => { await act(async () => { root.unmount(); store.dispatch(apiSlice.util.resetApiState()); }); host.remove(); localStorage.removeItem(storageKey); vi.unstubAllGlobals(); });

it('shows money in dollars with cents', () => {
  expect(formatUsd('4.000000')).toBe('$4.00');
  expect(formatUsd('1.190000')).toBe('$1.19');
  expect(formatUsd('0.004500')).toBe('$0.0045');
});

it('offers, approves once with a saved operation, follows to ready and releases', async () => {
  await render();
  await contains('Get a remote offer');
  await click('Get a remote offer');
  await contains('NVIDIA A100 80GB in US');
  await contains('$1.19 per hour');
  await contains('Cleanup terms');
  expect(calls.filter((call) => call.method === 'POST')).toHaveLength(1);
  await click('Approve and rent for at most $4.00');
  await contains('Approving');
  const approval = calls.find((call) => call.path === '/v1/burst/operations' && call.method === 'POST')!;
  const sent = JSON.parse(approval.body);
  expect(sent).toEqual({ operationId: expect.stringMatching(/^[a-f0-9]{32}$/), reference, approvalRevision: 'b'.repeat(64) });
  expect(JSON.parse(localStorage.getItem(storageKey)!)).toEqual({ pluginId: 'managed.runpod', nodeId: 'gpu', operationId: sent.operationId });
  phase = 'ready';
  await act(async () => { root.unmount(); store.dispatch(apiSlice.util.resetApiState()); });
  // A reload follows the saved rental instead of offering again.
  store = makeStore(); root = createRoot(host); await render();
  await contains('The model is ready on the remote machine.');
  await contains('Release the machine');
  await click('Release the machine');
  await contains('Releasing');
  expect(calls.filter((call) => call.method === 'DELETE')).toHaveLength(1);
  expect(calls.filter((call) => call.path === '/v1/burst/operations' && call.method === 'POST')).toHaveLength(1);
});

it('says what is missing when no provider can offer', async () => {
  providers = [];
  await render();
  await contains('No burst provider is installed');
  providers = [provider('configuration_required', 'Turn on proposals in this burst provider’s settings.')];
  await act(async () => { root.unmount(); store.dispatch(apiSlice.util.resetApiState()); });
  store = makeStore(); root = createRoot(host); await render();
  await contains('Turn on proposals');
  expect([...host.querySelectorAll('button')].some((item) => item.textContent === 'Get a remote offer')).toBe(false);
  expect(calls.every((call) => call.method === 'GET')).toBe(true);
});

it('asks for an offer without the nodes the operator excluded', async () => {
  await act(async () => {
    root.render(<Provider store={store}><ThemeProvider theme={darkTheme}><BurstOfferPanel modelId="example/large" contextTokens={8192} excludedNodeIds={['node-a', 'node-b']} /></ThemeProvider></Provider>);
  });
  await contains('Assessed without the 2 node(s) excluded above.');
  await contains('Get a remote offer');
  await click('Get a remote offer');
  await contains('NVIDIA A100 80GB in US');
  const asked = calls.find((call) => call.path === '/v1/burst/recommendations')!;
  expect(JSON.parse(asked.body)).toEqual({ modelId: 'example/large', contextTokens: 8192, excludedNodeIds: ['node-a', 'node-b'] });
});
