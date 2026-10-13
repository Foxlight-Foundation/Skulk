import { configureStore } from '@reduxjs/toolkit';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { apiSlice } from '../../store/api';
import { setupRunsSliceReducer } from '../../store/slices/setupRunsSlice';
import { uiActions, uiSliceReducer } from '../../store/slices/uiSlice';
import type { CapabilityNodeSummary } from '../../types/capabilityNodes';
import { darkTheme } from '../../theme/theme';
import { CapabilityPanel } from './CapabilityPanel';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });
vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({
    t: (_key: string, fallback: string, params?: Record<string, unknown>) =>
      fallback.replace(/\{(\w+)\}/g, (_match, name: string) => String(params?.[name] ?? `{${name}}`)),
  }),
}));

const cluster = vi.hoisted(() => ({ localNodeId: 'node-host' as string | null, nodes: {} as Record<string, unknown[]> }));
vi.mock('../../hooks/useClusterState', () => ({
  useClusterState: () => ({
    capabilityNodes: cluster.nodes,
    topology: { nodes: { 'node-host': { friendly_name: 'local' } }, edges: [] },
    localNodeId: cluster.localNodeId,
  }),
}));

function summary(pluginId: string, nodeId: string, title: string): CapabilityNodeSummary {
  return {
    pluginId,
    nodeId,
    bundleId: pluginId,
    version: '1.0.0',
    title,
    status: 'ready',
    ownerAvailable: true,
    surfaces: [],
    actions: [],
    operationsActive: 0,
    observedAt: new Date().toISOString(),
  };
}

const runpod = summary('managed.runpod', 'gpu', 'RunPod GPU capacity');
const studio = summary('managed.studio', 'studio', 'Video Studio');

let root: Root;
let host: HTMLDivElement;

beforeEach(() => {
  cluster.localNodeId = 'node-host';
  cluster.nodes = { 'node-host': [runpod, studio] };
  vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = input instanceof Request ? input : new Request(new URL(String(input), location.href), init);
    const url = new URL(request.url);
    const reply = (body: unknown) => new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } });
    if (url.pathname === '/v1/burst/providers') {
      return reply({
        scope: 'local_api_host',
        observedAt: 1,
        providers: [{ pluginId: 'managed.runpod', nodeId: 'gpu', bundleId: 'foxlight.runpod', version: '0.6.0', title: null, memoryPools: ['vram'], state: 'available', message: 'Ready to prepare offers.', maximumTotalUsd: null, maximumLifetimeSeconds: null }],
      });
    }
    if (url.pathname === '/v1/burst/budget') {
      return reply({
        pluginId: 'managed.runpod', nodeId: 'gpu', state: 'available', message: null,
        limitUsd: '10.000000', spentUsd: '2.500000', heldUsd: '0.000000', leftUsd: '7.500000',
        rentals: 1, countingSince: null, changeLimitAction: null, startOverAction: null,
      });
    }
    if (url.pathname === '/v1/burst/operations') return reply({ scope: 'local_api_host', observedAt: 1, rentals: [] });
    if (url.pathname === '/v1/plugins') return reply([]);
    return new Response('{}', { status: 404 });
  });
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
  vi.unstubAllGlobals();
});

async function render(key: string) {
  const store = configureStore({
    reducer: { ui: uiSliceReducer, setupRuns: setupRunsSliceReducer, [apiSlice.reducerPath]: apiSlice.reducer },
    middleware: (defaults) => defaults().concat(apiSlice.middleware),
  });
  store.dispatch(uiActions.openCapabilityPanel({ target: { hostNodeId: 'node-host', key }, tab: 'rentals' }));
  await act(async () => {
    root.render(
      <Provider store={store}>
        <ThemeProvider theme={darkTheme}>
          <CapabilityPanel />
        </ThemeProvider>
      </Provider>,
    );
  });
}

const tabs = () => [...host.querySelectorAll('[role="tab"]')].map((item) => item.textContent ?? '');

it('opens a local burst provider on its Rentals tab', async () => {
  await render('managed.runpod/gpu');
  await act(async () => {
    await vi.waitFor(() => expect(tabs()).toContain('Rentals'));
  });
  await act(async () => {
    await vi.waitFor(() => expect(host.textContent).toContain('$2.50 of $10.00 spent'));
  });
  expect(host.textContent).toContain('No rented machines are running.');
});

it('shows no Rentals tab for another node, falling back to its overview', async () => {
  await render('managed.studio/studio');
  await act(async () => {
    await vi.waitFor(() => expect(host.textContent).toContain('Status'));
  });
  expect(tabs()).not.toContain('Rentals');
});

it('shows no Rentals tab for a provider on another host', async () => {
  cluster.localNodeId = 'node-elsewhere';
  await render('managed.runpod/gpu');
  await act(async () => {
    await vi.waitFor(() => expect(host.textContent).toContain('Status'));
  });
  expect(tabs()).not.toContain('Rentals');
});
