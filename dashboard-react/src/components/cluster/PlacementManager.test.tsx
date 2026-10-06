import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { userEvent } from 'vitest/browser';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { darkTheme } from '../../theme/theme';
import type { NodeInfo, TopologyData } from '../../types/topology';
import { PlacementManager, type PlacementManagerProps } from './PlacementManager';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });

vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({
    t: (_key: string, fallback: string) => fallback,
  }),
}));

vi.mock('../models/ModelCard', () => ({ ModelCard: () => <div>Model card fixture</div> }));

let root: Root | null = null;
let container: HTMLDivElement | null = null;

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  root = null;
  container = null;
  vi.unstubAllGlobals();
});

const topology: TopologyData = { nodes: { 'node-1': {} as NodeInfo }, edges: [] };

function preview(overrides: Record<string, unknown>) {
  return {
    model_id: 'org/served-GGUF',
    sharding: 'Pipeline',
    instance_meta: 'MlxRing',
    instance: {
      MlxRingInstance: { shardAssignments: { nodeToRunner: { 'node-1': 'runner-1' } } },
    },
    memory_delta_by_node: null,
    error: null,
    max_context_tokens: 262144,
    default_context_tokens: 32768,
    reserves_context_at_load: true,
    kv_bytes_per_token: 20480,
    ...overrides,
  };
}

async function renderDialog(onLaunch: PlacementManagerProps['onLaunch']) {
  vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ previews: [preview({})] }), { status: 200 })));
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  await act(async () => {
    root?.render(
      <ThemeProvider theme={darkTheme}>
        <PlacementManager modelId="org/served-GGUF" topology={topology} open onClose={vi.fn()} onLaunch={onLaunch} />
      </ThemeProvider>,
    );
  });
  await vi.waitFor(() => {
    expect(container?.querySelector('input[aria-label="Context window"]')).not.toBeNull();
  });
  return container.querySelector<HTMLInputElement>('input[aria-label="Context window"]')!;
}

function launchButton(): HTMLButtonElement {
  return [...container!.querySelectorAll<HTMLButtonElement>('button')].find((button) =>
    button.textContent?.includes('Launch Model'),
  )!;
}

describe('PlacementManager context window', () => {
  it('shows the default and launches without a window when it is untouched', async () => {
    const onLaunch = vi.fn();
    const field = await renderDialog(onLaunch);
    expect(field.value).toBe('32768');
    expect(container?.textContent).toContain('This engine reserves the whole window when the model loads');
    await act(async () => launchButton().click());
    expect(onLaunch).toHaveBeenCalledOnce();
    expect(onLaunch.mock.calls[0]?.[0]).not.toHaveProperty('contextTokens');
  });

  it('launches with the window the operator chose', async () => {
    const onLaunch = vi.fn();
    const field = await renderDialog(onLaunch);
    await userEvent.clear(field);
    await userEvent.type(field, '65536');
    await act(async () => launchButton().click());
    expect(onLaunch.mock.calls[0]?.[0]).toMatchObject({ contextTokens: 65536 });
  });

  it('blocks a window above what the placement holds', async () => {
    const onLaunch = vi.fn();
    const field = await renderDialog(onLaunch);
    await userEvent.clear(field);
    await userEvent.type(field, '300000');
    expect(launchButton().disabled).toBe(true);
    expect(container?.textContent).toContain('Choose between {min} and {max} tokens for this placement.');
  });
});
