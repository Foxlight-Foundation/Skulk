import { configureStore } from '@reduxjs/toolkit';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from 'styled-components';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { apiSlice } from '../../store/api';
import type { CapabilityNodeSummary } from '../../types/capabilityNodes';
import { darkTheme } from '../../theme/theme';
import { CapabilityFlyout } from './CapabilityFlyout';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });
vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({
    t: (_key: string, fallback: string, params?: Record<string, string | number>) =>
      fallback.replace(/\{(\w+)\}/g, (_match, name: string) => String(params?.[name] ?? '')),
  }),
}));

const runpod: CapabilityNodeSummary = {
  pluginId: 'managed.runpod',
  nodeId: 'gpu',
  bundleId: 'foxlight.runpod',
  version: '0.6.0',
  title: 'RunPod GPU capacity',
  status: 'ready',
  ownerAvailable: true,
  surfaces: [],
  actions: [],
  operationsActive: 0,
  observedAt: new Date().toISOString(),
};
const KEY = 'managed.runpod/gpu';

let root: Root;
let host: HTMLDivElement;
let budgetReads: number;

beforeEach(() => {
  budgetReads = 0;
  vi.stubGlobal('fetch', async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = input instanceof Request ? input : new Request(new URL(String(input), location.href), init);
    const url = new URL(request.url);
    if (url.pathname === '/v1/burst/budget') {
      budgetReads += 1;
      return new Response(
        JSON.stringify({
          pluginId: 'managed.runpod', nodeId: 'gpu', state: 'available', message: null,
          limitUsd: '10.000000', spentUsd: '2.500000', heldUsd: '1.400000', leftUsd: '6.100000',
          rentals: 1, countingSince: null, changeLimitAction: 'campaign-limits', startOverAction: 'start-over',
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      );
    }
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

async function render(props: Partial<Parameters<typeof CapabilityFlyout>[0]>) {
  const onOpenPanel = vi.fn();
  await act(async () => {
    root.render(
      <Provider store={configureStore({ reducer: { [apiSlice.reducerPath]: apiSlice.reducer }, middleware: (defaults) => defaults().concat(apiSlice.middleware) })}>
        <ThemeProvider theme={darkTheme}>
          <CapabilityFlyout
            anchor={{ x: 200, y: 120 }}
            canvasHeight={600}
            canvasWidth={800}
            hostName="local"
            hostNodeId="node-host"
            localNodeId="node-host"
            onClose={() => undefined}
            onOpenPanel={onOpenPanel}
            onSelectKey={() => undefined}
            selectedKey={KEY}
            summaries={[runpod]}
            {...props}
          />
        </ThemeProvider>
      </Provider>,
    );
  });
  return { onOpenPanel };
}

const buttons = () => [...host.querySelectorAll('button')].map((item) => item.textContent ?? '');

describe('CapabilityFlyout for a burst provider', () => {
  it('shows spending and opens the Rentals view on this dashboard’s host', async () => {
    const { onOpenPanel } = await render({ burstProviderKeys: new Set([KEY]) });
    await act(async () => {
      await vi.waitFor(() => expect(host.textContent).toContain('$2.50 of $10.00 spent'));
    });
    expect(host.textContent).toContain('$1.40 held for machines running now');
    const rentals = [...host.querySelectorAll('button')].find((item) => item.textContent === 'Rentals');
    await act(async () => rentals?.click());
    expect(onOpenPanel).toHaveBeenCalledWith('node-host', KEY, 'rentals');
  });

  it('shows neither for a provider on another host', async () => {
    await render({ burstProviderKeys: new Set([KEY]), localNodeId: 'node-elsewhere' });
    expect(buttons()).not.toContain('Rentals');
    expect(host.textContent).not.toContain('spent');
    expect(budgetReads).toBe(0);
  });

  it('shows neither for a node that is not a burst provider', async () => {
    await render({ burstProviderKeys: new Set() });
    expect(buttons()).not.toContain('Rentals');
    expect(budgetReads).toBe(0);
  });
});
