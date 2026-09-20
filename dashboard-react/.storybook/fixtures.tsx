import { observabilityFixtures } from './observabilityFixtures';
import { screenFixtures } from './screenFixtures';
import type { NavRoute } from '../src/components/layout/HeaderNav';
import { useMemo, type ReactNode } from 'react';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import { apiSlice } from '../src/store/api';
import { uiActions, uiSliceReducer } from '../src/store/slices/uiSlice';
import { chatSliceReducer } from '../src/store/slices/chatSlice';

const configFixture = { config: { model_store: { enabled: false }, inference: { kv_cache_backend: 'default' }, logging: { enabled: false, ingest_url: '' }, intelligent_fabric: { enabled: false }, telemetry: { consent: 'disabled', diagnostics_consent: 'disabled', install_id: '', consented_at: '', consented_version: '', ingest_url: '' } }, configPath: 'skulk.yaml', effective: { kv_cache_backend: 'default', hf_token_set: false } };

const fixtures: Record<string, unknown> = {
  ...observabilityFixtures,
  '/config': configFixture,
  '/state': { topology: { nodes: [], connections: {} }, instances: {}, runners: {}, downloads: {}, tasks: {} },
  '/models': { data: [{ id: 'example/Chat-8B', name: 'Example Chat 8B', capabilities: ['text', 'code'], storage_size_megabytes: 5000 }] },
  '/v1/models': { object: 'list', data: [] },
  '/v1/steward': { enabled: false, ready: false, transition: 'idle', model_id: null, instance_id: null, node_id: null },
  '/v1/steward/proposals': [],
  '/v1/plugins': [{ pluginId: 'managed.example-video', available: true, nodes: [{ nodeId: 'example-node', bundleId: 'Example video capability', version: '1.0.0', status: 'ready', configurable: false }] }],
  '/v1/plugins/managed': { installations: [{ plugin_id: 'managed.example-video', selected_digest: 'abcdef0123456789', selection_revision: 1, enabled: true, stale: false, error_code: null, operation_id: null, operation_state: null, service: { state: 'running', active_digest: 'abcdef0123456789', observed_at: 1 } }, { plugin_id: 'managed.example-stale', selected_digest: null, selection_revision: 1, enabled: false, stale: true, error_code: null, operation_id: null, operation_state: null, service: null }] },
  '/v1/auth/pairing-invitations': [{ invitationId: '00000000-0000-4000-8000-000000000001', createdAt: '2026-09-18T12:00:00Z', expiresAt: '2026-09-19T12:00:00Z', successfulPairings: 1, maxPairings: 3, activeAttempts: 0, totalAttempts: 1, state: 'active' }],
  '/v1/auth/devices': { devices: [{ deviceId: 'fictional-tablet', name: 'Example tablet', pairedAt: '2026-09-01T12:00:00Z', refreshExpiresAt: null, state: 'active', current: false }, { deviceId: 'fictional-laptop', name: 'Example laptop with a long translated name', pairedAt: '2026-09-01T12:00:00Z', refreshExpiresAt: null, state: 'revoked', current: false }] },
  '/v1/tracing': { enabled: false },
};

let pluginInventoryUnavailable = false;
let activeScreen = false;
let showTelemetryConsent = false;

/** Select the offline response set before rendering a story. */
// eslint-disable-next-line react-refresh/only-export-components -- Storybook loader configuration.
export function configureFixtureScreen(enabled: boolean, telemetryConsent = false, unavailablePlugins = false) { activeScreen = enabled; showTelemetryConsent = telemetryConsent; pluginInventoryUnavailable = unavailablePlugins; }

const originalFetch = window.fetch.bind(window);
// Storybook is an offline component gallery. Every API request is intercepted,
// including unknown routes and mutations, so stories cannot operate a cluster.
window.fetch = async (input, init) => {
  const request = new Request(input, init);
  const url = new URL(request.url);
  const isStorybookAsset = /\.(?:[cm]?js|tsx?|jsx|css|map)$/.test(url.pathname) || /^\/(@|src\/|node_modules\/|sb-|__vitest)/.test(url.pathname) || ['/index.json', '/project.json'].includes(url.pathname);
  // A deterministic synthetic conversation; no mutation leaves the gallery.
  if (activeScreen && request.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const content = 'The example cluster has three nodes. Workstation has a ready chat model; GPU server is loading a model; Compact has a failed runner. Inspect that runner before deciding whether to retry.';
    return new Response(`data: ${JSON.stringify({ choices: [{ delta: { content }, finish_reason: null }] })}\n\ndata: [DONE]\n\n`, { headers: { 'Content-Type': 'text/event-stream' } });
  }
  if (request.method !== 'GET') return Response.json({ detail: 'This gallery does not execute operations.' }, { status: 403 });
  if (url.origin === location.origin && isStorybookAsset) return originalFetch(input, init);
  if (showTelemetryConsent && url.pathname === '/config') return Response.json({ ...configFixture, config: { ...configFixture.config, telemetry: { ...configFixture.config.telemetry, consent: 'unasked' } } });
  if (pluginInventoryUnavailable && url.pathname === '/v1/plugins/managed') return Response.json({ detail: 'Runtime manager unavailable.' }, { status: 503 });
  if (pluginInventoryUnavailable && url.pathname === '/v1/plugins') return Response.json([]);
  if (activeScreen && Object.hasOwn(screenFixtures, url.pathname)) return Response.json(screenFixtures[url.pathname]);
  if (Object.hasOwn(fixtures, url.pathname)) return Response.json(fixtures[url.pathname]);
  return Response.json({ detail: 'No fixture for this observation.' }, { status: 503 });
};

/** Isolated store without browser persistence or production subscriptions. */
export function FixtureProvider({ children, storyId, theme, screenRoute }: { children: ReactNode; storyId: string; screenRoute?: NavRoute; theme: 'dark' | 'light' }) {
  const store = useMemo(() => { const fixtureStore = configureStore({
    reducer: { ui: uiSliceReducer, chat: chatSliceReducer, [apiSlice.reducerPath]: apiSlice.reducer },
    middleware: getDefault => getDefault().concat(apiSlice.middleware),
  }); fixtureStore.dispatch(uiActions.setTheme(theme));
  if (screenRoute) { fixtureStore.dispatch(uiActions.setActiveRoute(screenRoute)); if (!fixtureStore.getState().ui.panelOpen) fixtureStore.dispatch(uiActions.togglePanel()); }
  return fixtureStore; }, [storyId, theme, screenRoute]);
  return <Provider store={store}>{children}</Provider>;
}
