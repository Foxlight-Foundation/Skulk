import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { afterEach, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';
import { darkTheme } from '../../theme/theme';
import { InferenceAssessmentPanel } from './InferenceAssessmentPanel';
import { operatorSession } from '../../auth/operatorSession';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });
vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({
    t: (_key: string, fallback: string, params?: Record<string, unknown>) =>
      fallback.replace(/\{(\w+)\}/g, (_, name: string) => String(params?.[name] ?? '')),
  }),
}));
let root: Root | null = null;
let container: HTMLDivElement;
afterEach(async () => {
  await act(async () => root?.unmount());
  container.remove(); root = null; vi.unstubAllGlobals(); vi.restoreAllMocks();
});

async function render(payload: object, modelId = 'org/model') {
  const fetcher = vi.fn(async (url: string) => {
    const params = new URL(url, 'http://localhost').searchParams;
    return new Response(JSON.stringify({ ...payload, workload: { contextTokens: Number(params.get('context_tokens')), concurrentRequests: Number(params.get('concurrent_requests')) } }), { status: 200 });
  });
  vi.stubGlobal('fetch', fetcher);
  container = document.createElement('div'); document.body.append(container);
  root = createRoot(container);
  await act(async () => root?.render(<ThemeProvider theme={darkTheme}>
    <InferenceAssessmentPanel modelId={modelId} contextTokens={null} excludedNodeIds={['excluded']} />
  </ThemeProvider>));
  await vi.waitFor(() => expect(fetcher).toHaveBeenCalledOnce());
  return fetcher;
}

const external = {
  status: 'external_required', readiness: 'not_observed',
  reasons: ['The selected artifact and context do not fit locally.'],
  assumptions: ['The full model runs remotely.'], uncertainty: ['Price and availability are unknown.'],
  externalConfigurations: [{ backend: 'llama_server-cuda', replicaCount: 1,
    minimumMemoryBytesPerNode: 48 * 1024 ** 3, estimatedMemoryBytesPerReplica: 40 * 1024 ** 3, memoryPool: 'vram' }],
};

it('shows the full remote estimate and performs only a read', async () => {
  const fetcher = await render(external);
  await vi.waitFor(() => expect(container.textContent).toContain('48.0 GiB VRAM'));
  expect(container.textContent).toContain('Estimated external capacity required');
  expect(container.textContent).toContain('No external capacity is acquired');
  const [url, options] = fetcher.mock.calls[0] as unknown as [string, RequestInit];
  expect(url).toContain('context_tokens=8192');
  expect(url).toContain('excluded_node_ids=excluded');
  expect(options.method).toBeUndefined();
});

it('binds changed concurrency to a new assessment', async () => {
  const fetcher = await render(external);
  const input = container.querySelector<HTMLInputElement>('input[aria-label="Simultaneous requests"]')!;
  await userEvent.clear(input); await userEvent.type(input, '4');
  await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(2));
  expect(String(fetcher.mock.calls[1]?.[0])).toContain('concurrent_requests=4');
});

it('shows unknown fit on unavailable assessment, not a local-fit claim', async () => {
  await render({});
  await vi.waitFor(() => expect(container.textContent).toContain('Local fit is unknown'));
  expect(container.textContent).not.toContain('Estimated local fit');
});

it('shows missing evidence and keeps acquisition inactive', async () => {
  await render({ ...external, status: 'insufficient_evidence', externalConfigurations: [], reasons: ['Memory observation is stale.'] });
  await vi.waitFor(() => expect(container.textContent).toContain('More evidence needed'));
  expect(container.textContent).toContain('Memory observation is stale');
  expect(container.querySelector('button')).toBeNull();
});

it('routes reads through the operator session so paired credentials propagate', async () => {
  vi.spyOn(operatorSession, 'fetch').mockImplementation((url, options) => fetch(url, {
    ...options, headers: { Authorization: 'Bearer fixture-paired-access' },
  }));
  const fetcher = await render(external);
  expect(operatorSession.fetch).toHaveBeenCalledOnce();
  const [, options] = fetcher.mock.calls[0] as unknown as [string, RequestInit];
  expect(new Headers(options.headers).get('Authorization')).toBe('Bearer fixture-paired-access');
});
