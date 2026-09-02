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

function reconciliationResponse(state = 'complete'): Response {
  return jsonResponse({
    state,
    inventory_only: false,
    scanned_nodes: 2,
    discovered_artifacts: 1,
    pending_imports: 0,
    imported_artifacts: 1,
    failures: [],
    last_verified_at: '2026-08-10T12:00:00Z',
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
      if (path === '/store/reconciliation') return reconciliationResponse();
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
      if (path === '/store/reconciliation') return reconciliationResponse();
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
      if (path === '/store/reconciliation') return reconciliationResponse();
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

  it('keeps polling while reconciliation is still importing', async () => {
    vi.useFakeTimers();
    let registryRequests = 0;
    let reconciliationRequests = 0;
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path === '/models') return jsonResponse({ data: [] });
      if (path === '/store/downloads') return jsonResponse({ downloads: [] });
      if (path === '/store/reconciliation') {
        reconciliationRequests += 1;
        return reconciliationResponse(
          reconciliationRequests === 1 ? 'importing' : 'complete',
        );
      }
      if (path === '/store/registry') {
        registryRequests += 1;
        return jsonResponse({
          entries: registryRequests >= 2 ? [{
            model_id: 'org/imported-model',
            total_bytes: 4096,
            files: ['model.gguf'],
            downloaded_at: '2026-08-10T12:00:00Z',
          }] : [],
        });
      }
      throw new Error(`unexpected fetch: ${path}`);
    }));

    await renderModelStore();
    await flushEffects();
    expect(container?.textContent).not.toContain('org/imported-model');
    expect(registryRequests).toBe(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    await flushEffects();

    expect(container?.textContent).toContain('org/imported-model');
    expect(registryRequests).toBe(2);
  });
});

describe('ModelStorePage failed-download surfacing', () => {
  const GATED_REASON =
    "Access to 'meta-llama/gated' is restricted and this node sent no Hugging Face token.";

  it('toasts the store reason when a download transitions to failed, then converges', async () => {
    vi.useFakeTimers();
    const { addToast } = await import('../../hooks/useToast');
    let downloadRequests = 0;
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path === '/models') return jsonResponse({ data: [] });
      if (path === '/store/downloads') {
        downloadRequests += 1;
        return jsonResponse({
          downloads: downloadRequests === 1
            ? [{ modelId: 'meta-llama/gated', progress: 0.1, status: 'downloading' }]
            : [{ modelId: 'meta-llama/gated', progress: 0.1, status: 'failed', error: GATED_REASON }],
        });
      }
      if (path === '/store/reconciliation') return reconciliationResponse();
      if (path === '/store/registry') return jsonResponse({ entries: [] });
      throw new Error(`unexpected fetch: ${path}`);
    }));

    await renderModelStore();
    await flushEffects();
    expect(addToast).not.toHaveBeenCalled();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    await flushEffects();

    expect(addToast).toHaveBeenCalledWith(expect.objectContaining({
      type: 'error',
      message: expect.stringContaining(GATED_REASON),
    }));

    // The failed entry is terminal: the 2s poll must stop instead of spinning
    // on a listing that now permanently includes it.
    const requestsAfterToast = downloadRequests;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(6000);
    });
    expect(downloadRequests).toBe(requestsAfterToast);
  });

  it('does not toast a failure already listed on the first fetch', async () => {
    vi.useFakeTimers();
    const { addToast } = await import('../../hooks/useToast');
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      if (path === '/models') return jsonResponse({ data: [] });
      if (path === '/store/downloads') {
        return jsonResponse({
          downloads: [{ modelId: 'meta-llama/gated', progress: 0, status: 'failed', error: GATED_REASON }],
        });
      }
      if (path === '/store/reconciliation') return reconciliationResponse();
      if (path === '/store/registry') return jsonResponse({ entries: [] });
      throw new Error(`unexpected fetch: ${path}`);
    }));

    await renderModelStore();
    await flushEffects();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(4000);
    });
    await flushEffects();

    expect(addToast).not.toHaveBeenCalled();
  });
});
