import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { darkTheme } from '../../theme/theme';
import type { HuggingFaceModel, ModelInfo } from '../../types/models';
import { ModelBrowser } from './ModelBrowser';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({
    t: (_key: string, fallback: string) => fallback,
  }),
}));

const MODELS: ModelInfo[] = [
  {
    id: 'mlx-community/Qwen3-4B-4bit',
    name: 'Qwen3 4B',
    base_model: 'qwen3-4b',
    family: 'qwen',
    storage_size_megabytes: 2100,
  },
  {
    id: 'mlx-community/LongCat-AudioDiT-1B-4bit',
    name: 'LongCat AudioDiT',
    base_model: 'longcat-audiodit',
    family: 'longcat_audiodit',
    storage_size_megabytes: 1300,
  },
  {
    id: 'CogniSoftOrg/canary-1b-v2-mlx-bf16',
    name: 'Canary 1B',
    base_model: 'canary-1b',
    family: 'canary',
    storage_size_megabytes: 3100,
  },
];

const HUB_MODELS: HuggingFaceModel[] = [{
  id: 'org/hub-model',
  author: 'org',
  downloads: 100,
  likes: 10,
  last_modified: '2026-08-01',
  tags: [],
}];

let root: Root | null = null;
let container: HTMLDivElement | null = null;

async function renderBrowser(): Promise<void> {
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  await act(async () => {
    root?.render(
      <ThemeProvider theme={darkTheme}>
        <ModelBrowser
          models={MODELS}
          selectedModelId={null}
          favorites={new Set()}
          canModelFit={() => true}
          getModelFitStatus={() => 'fits_now'}
          onSelect={vi.fn()}
          onToggleFavorite={vi.fn()}
          hfTrendingModels={HUB_MODELS}
          mode="store-download"
        />
      </ThemeProvider>,
    );
  });
}

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  root = null;
  container = null;
});

describe('ModelBrowser store discovery taxonomy', () => {
  it('uses readable source and family controls without icon-only navigation', async () => {
    await renderBrowser();

    expect(container?.querySelector('nav')).toBeNull();
    expect(container?.textContent).toContain('Supported models');
    expect(container?.textContent).not.toContain('Recommended');

    const sourceButtons = container?.querySelectorAll('[role="group"][aria-label="Model source"] button');
    expect(sourceButtons?.length).toBe(2);
    expect(sourceButtons?.[0]?.textContent).toBe('Supported models');
    expect(sourceButtons?.[1]?.textContent).toBe('Search Hugging Face');

    const familySelect = container?.querySelector<HTMLSelectElement>(
      'select[aria-label="Filter supported models by family"]',
    );
    expect(familySelect).not.toBeNull();
    expect(Array.from(familySelect?.options ?? []).map((option) => option.text)).toEqual([
      'All supported models',
      'Canary',
      'LongCat AudioDiT',
      'Qwen',
    ]);
  });

  it('filters catalog families and switches separately to Hugging Face discovery', async () => {
    await renderBrowser();

    const familySelect = container?.querySelector<HTMLSelectElement>(
      'select[aria-label="Filter supported models by family"]',
    );
    expect(familySelect).not.toBeNull();
    await act(async () => {
      if (!familySelect) return;
      familySelect.value = 'canary';
      familySelect.dispatchEvent(new Event('change', { bubbles: true }));
    });

    expect(container?.textContent).toContain('Canary 1B');
    expect(container?.textContent).not.toContain('Qwen3 4B');

    const hubButton = Array.from(container?.querySelectorAll('button') ?? [])
      .find((button) => button.textContent === 'Search Hugging Face');
    expect(hubButton).not.toBeUndefined();
    await act(async () => hubButton?.click());

    expect(container?.querySelector('select[aria-label="Filter supported models by family"]')).toBeNull();
    expect(container?.querySelector<HTMLInputElement>('input')?.placeholder)
      .toBe('Search all of Hugging Face...');
    expect(container?.textContent).toContain('org/hub-model');
    expect(container?.textContent).not.toContain('Canary 1B');
  });
});
