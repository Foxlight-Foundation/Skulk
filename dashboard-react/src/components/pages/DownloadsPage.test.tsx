import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { darkTheme } from '../../theme/theme';
import { ModelStorePage } from './DownloadsPage';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({
    t: (_key: string, fallback: string) => fallback,
  }),
}));

vi.mock('../../hooks/useToast', () => ({
  addToast: vi.fn(),
}));

vi.mock('../layout/StoreRegistryTable', () => ({
  StoreRegistryTable: ({ entries }: { entries: Array<{ model_id: string }> }) => (
    <div data-testid="store-registry">
      {entries.map((entry) => entry.model_id).join(',')}
    </div>
  ),
}));

vi.mock('./ModelSearchModal', () => ({
  ModelSearchModal: () => null,
}));

vi.mock('../cluster/PlacementManager', () => ({
  PlacementManager: () => null,
}));

let root: Root | null = null;
let container: HTMLDivElement | null = null;

function jsonResponse(payload: object): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
}

async function renderModelStore(): Promise<void> {
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  await act(async () => {
    root?.render(
      <ThemeProvider theme={darkTheme}>
        <ModelStorePage
          topology={null}
          downloads={{}}
          nodeDisk={{}}
          instances={{}}
          runners={{}}
        />
      </ThemeProvider>,
    );
  });
}

async function flushEffects(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  root = null;
  container = null;
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe('ModelStorePage registry convergence', () => {
  it('retries an initial registry failure and renders the recovered model', async () => {
    vi.useFakeTimers();
    let registryRequests = 0;
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path === '/models') return jsonResponse({ data: [] });
      if (path === '/store/downloads') return jsonResponse({ downloads: [] });
      if (path === '/store/registry') {
        registryRequests += 1;
        if (registryRequests === 1) throw new TypeError('connection reset');
        return jsonResponse({
          entries: [{
            model_id: 'org/recovered-model',
            total_bytes: 1024,
            files: ['model.gguf'],
            downloaded_at: '2026-07-31T00:00:00Z',
          }],
        });
      }
      throw new Error(`unexpected fetch: ${path}`);
    }));

    await renderModelStore();
    await flushEffects();
    expect(container?.textContent).not.toContain('org/recovered-model');
    expect(registryRequests).toBe(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    await flushEffects();

    expect(container?.textContent).toContain('org/recovered-model');
    expect(registryRequests).toBe(2);
  });

  it('keeps polling when the final download check succeeds but registry refresh fails', async () => {
    vi.useFakeTimers();
    let registryRequests = 0;
    let downloadRequests = 0;
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path === '/models') return jsonResponse({ data: [] });
      if (path === '/store/downloads') {
        downloadRequests += 1;
        return jsonResponse({
          downloads: downloadRequests === 1 ? [{ model_id: 'org/new-model' }] : [],
        });
      }
      if (path === '/store/registry') {
        registryRequests += 1;
        if (registryRequests === 2) throw new TypeError('empty response');
        return jsonResponse({
          entries: registryRequests >= 3 ? [{
            model_id: 'org/new-model',
            total_bytes: 2048,
            files: ['model.gguf'],
            downloaded_at: '2026-07-31T00:00:00Z',
          }] : [],
        });
      }
      throw new Error(`unexpected fetch: ${path}`);
    }));

    await renderModelStore();
    await flushEffects();
    expect(registryRequests).toBe(1);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    await flushEffects();
    expect(registryRequests).toBe(2);
    expect(container?.textContent).not.toContain('org/new-model');

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    await flushEffects();

    expect(container?.textContent).toContain('org/new-model');
    expect(registryRequests).toBe(3);
  });

  it('does not restart polling when an in-flight refresh finishes after unmount', async () => {
    vi.useFakeTimers();
    let registryRequests = 0;
    let resolveRegistry: ((response: Response) => void) | undefined;
    const pendingRegistry = new Promise<Response>((resolve) => {
      resolveRegistry = resolve;
    });
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path === '/models') return jsonResponse({ data: [] });
      if (path === '/store/downloads') return jsonResponse({ downloads: [] });
      if (path === '/store/registry') {
        registryRequests += 1;
        return pendingRegistry;
      }
      throw new Error(`unexpected fetch: ${path}`);
    }));

    await renderModelStore();
    expect(registryRequests).toBe(1);

    await act(async () => root?.unmount());
    root = null;
    resolveRegistry?.(new Response(null, { status: 503 }));
    await flushEffects();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10_000);
    });

    expect(registryRequests).toBe(1);
  });
});
