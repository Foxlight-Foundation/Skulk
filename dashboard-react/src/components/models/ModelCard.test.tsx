import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { darkTheme } from '../../theme/theme';
import type { PlacementPreview } from '../../types/models';
import { ModelCard } from './ModelCard';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });

vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({
    t: (_key: string, fallback: string, params?: Record<string, unknown>) =>
      params
        ? fallback.replace(/\{(\w+)\}/g, (_, name: string) => String(params[name] ?? ''))
        : fallback,
  }),
}));

const BASE_PREVIEW: PlacementPreview = {
  model_id: 'Comfy-Org/MiniMax-H3-FL2VA-comfy-int8',
  sharding: 'Pipeline',
  instance_meta: 'MlxRing',
  instance: null,
  memory_delta_by_node: null,
  error: null,
};

let root: Root | null = null;
let container: HTMLDivElement | null = null;

function render(preview: PlacementPreview): HTMLDivElement {
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
  act(() => {
    root?.render(
      <ThemeProvider theme={darkTheme}>
        <ModelCard model={{ id: preview.model_id }} apiPreview={preview} hideActions />
      </ThemeProvider>,
    );
  });
  return container;
}

afterEach(() => {
  act(() => root?.unmount());
  container?.remove();
  root = null;
  container = null;
});

describe('ModelCard engine install notice', () => {
  it('tells the operator the video engine installs with the model', () => {
    const view = render({
      ...BASE_PREVIEW,
      engine_install: {
        engine: 'comfy',
        node_ids: ['node-a'],
        approximate_download_bytes: 7 * 1024 ** 3,
        detail: 'unused for comfy',
      },
    });

    const notice = view.querySelector('[data-testid="engine-install-notice"]');
    expect(notice?.textContent).toBe(
      'The video engine (about 7 GB) will be installed with this model, so placement will take longer.',
    );
    // A warning callout with its icon, not a line of fine print.
    expect(notice?.getAttribute('role')).toBe('note');
    expect(notice?.querySelector('svg')).not.toBeNull();
  });

  it('shows the server notice for an engine it has no wording for', () => {
    const view = render({
      ...BASE_PREVIEW,
      engine_install: {
        engine: 'future',
        node_ids: ['node-a'],
        approximate_download_bytes: 0,
        detail: 'The future engine will be installed with this model, so placement will take longer.',
      },
    });

    expect(view.querySelector('[data-testid="engine-install-notice"]')?.textContent).toBe(
      'The future engine will be installed with this model, so placement will take longer.',
    );
  });

  it('shows no notice when the engine is already installed', () => {
    const view = render({ ...BASE_PREVIEW, engine_install: null });
    expect(view.querySelector('[data-testid="engine-install-notice"]')).toBeNull();
  });
});
