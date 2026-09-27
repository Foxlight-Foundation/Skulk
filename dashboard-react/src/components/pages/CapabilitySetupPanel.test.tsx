import { configureStore } from '@reduxjs/toolkit';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { apiSlice } from '../../store/api';
import { uiSliceReducer } from '../../store/slices/uiSlice';
import { darkTheme } from '../../theme/theme';
import { CapabilitySetupPanel } from './CapabilitySetupPanel';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });
vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({ t: (_key: string, fallback: string, params?: Record<string, unknown>) => fallback.replace(/\{(\w+)\}/g, (_match, name: string) => String(params?.[name] ?? '')) }),
}));

const pluginId = 'managed.' + '2'.repeat(32);
const studioUrl = 'http://100.64.0.2:54905/s/token/';
let root: Root;
let host: HTMLDivElement;
let store: ReturnType<typeof makeStore>;
let readiness: Record<string, unknown>;

function makeStore() {
  return configureStore({ reducer: { [apiSlice.reducerPath]: apiSlice.reducer, ui: uiSliceReducer }, middleware: (defaults) => defaults().concat(apiSlice.middleware) });
}
function json(body: unknown, status = 200) { return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } }); }
async function contains(text: string) { await act(async () => { await vi.waitFor(() => expect(host.textContent).toContain(text), { timeout: 5000 }); }); }

beforeEach(async () => {
  readiness = {
    ready: true, api_reachable: true, video_models: ['Comfy-Org/MiniMax-H3-FL2VA-comfy-int8'],
    lanes: [{ node: 'kite4', backends: ['comfy-rocm'], placed_models: ['Comfy-Org/MiniMax-H3-FL2VA-comfy-int8'] }],
    default_model: 'Comfy-Org/MiniMax-H3-FL2VA-comfy-int8', default_host: 'kite4', reasons: [],
  };
  vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = input instanceof Request ? input : new Request(new URL(String(input), location.href), init);
    const path = new URL(request.url).pathname;
    if (path === '/node_id') return json('host-node');
    if (path === '/node/identity') return json({ nodeId: 'host-node', friendlyName: 'kite2' });
    if (path === '/state') {
      return json({
        instances: {
          video: { MlxRingInstance: { instanceId: 'video', shardAssignments: { modelId: 'Comfy-Org/MiniMax-H3-FL2VA-comfy-int8', nodeToRunner: { renderer: 'r1' } } } },
          chat: { MlxRingInstance: { instanceId: 'chat', shardAssignments: { modelId: 'google/gemma-4-31B-it-qat-q4_0-gguf', nodeToRunner: { renderer: 'r2' } } } },
        },
        runners: { r1: { RunnerReady: {} }, r2: { RunnerReady: {} } },
        capabilityNodes: { 'host-node': [{ pluginId, nodeId: 'studio', bundleId: 'foxlight.video-studio', version: '0.1.0', title: 'Skulk Video Studio', status: 'ready', ownerAvailable: true, surfaces: [{ surfaceId: 'studio', title: 'Skulk Video Studio (MiniMax H3)', kind: 'link', url: studioUrl, ready: true }], actions: [], operationsActive: 0, observedAt: new Date().toISOString() }] },
      });
    }
    if (path === '/v1/models') {
      return json({ data: [
        { id: 'Comfy-Org/MiniMax-H3-FL2VA-comfy-int8', tasks: ['TextToVideo', 'ImageToVideo'] },
        { id: 'google/gemma-4-31B-it-qat-q4_0-gguf', tasks: ['TextGeneration'] },
      ] });
    }
    if (path === '/v1/capabilities') return json({ capabilities: [{ id: 'video.readiness', version: '1.0.0' }], revisions: { 'video.readiness@1.0.0': 'f'.repeat(64) } });
    if (path === '/v1/capabilities/call') return json({ ok: true, result: readiness });
    return json({}, 404);
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
});

async function render() {
  await act(async () => {
    root.render(<Provider store={store}><ThemeProvider theme={darkTheme}>
      <CapabilitySetupPanel target={{ pluginId, title: 'Skulk Video Studio', descriptors: ['video.readiness@1.0.0'] }} onBack={() => undefined} />
    </ThemeProvider></Provider>);
  });
}

it('reads the fleet as ready and opens the capability’s own screen', async () => {
  await render();
  await contains('Skulk Video Studio is ready');
  await contains('kite4 · ComfyUI · ROCm');
  await contains('Placed on kite4 and ready.');
  // A video card is never offered as the chat model that refines prompts.
  await contains('gemma-4-31B-it-qat-q4_0-gguf is ready.');
  expect(host.textContent).not.toContain('MiniMax-H3-FL2VA-comfy-int8 is ready.');
  const open = host.querySelector('a[href]') as HTMLAnchorElement;
  expect(open.href).toBe(studioUrl);
  expect(open.getAttribute('aria-disabled')).toBe('false');
});

it('names what is missing when video models are off and nothing is placed', async () => {
  readiness = { ready: false, api_reachable: true, video_models: [], lanes: [], default_model: null, default_host: null, reasons: ['no video card is in the catalog'] };
  await render();
  await contains('Set up Skulk Video Studio');
  await contains('Off for this cluster. Start Skulk with SKULK_ENABLE_VIDEO_MODELS=1 on every node.');
  await contains('No node runs a video engine.');
  expect((host.querySelector('a[href]') as HTMLAnchorElement).getAttribute('aria-disabled')).toBe('true');
});
