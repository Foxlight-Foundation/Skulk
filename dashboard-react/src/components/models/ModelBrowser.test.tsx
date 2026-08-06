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
    t: (_key: string, fallback: string, params?: Record<string, unknown>) =>
      params
        ? fallback.replace(/\{(\w+)\}/g, (_, name: string) => String(params[name] ?? ''))
        : fallback,
  }),
}));

const MODELS: ModelInfo[] = [
  {
    id: 'mlx-community/Qwen3-4B-4bit',
    name: 'Qwen3-4B-4bit',
    base_model: 'Qwen3 4B',
    family: 'qwen',
    quantization: '4bit',
    storage_size_megabytes: 2100,
  },
  {
    id: 'mlx-community/LongCat-AudioDiT-1B-4bit',
    name: 'LongCat AudioDiT',
    base_model: 'LongCat AudioDiT 1B',
    family: 'longcat_audiodit',
    storage_size_megabytes: 1300,
  },
  {
    id: 'CogniSoftOrg/canary-1b-v2-mlx-bf16',
    name: 'Canary 1B',
    base_model: 'Canary 1B',
    family: 'canary',
    storage_size_megabytes: 3100,
  },
];

const HUB_MODELS: HuggingFaceModel[] = [{
  id: 'org/hub-model-8bit-GGUF',
  author: 'org',
  downloads: 100,
  likes: 10,
  last_modified: '2026-08-01',
  tags: [],
}];

let root: Root | null = null;
let container: HTMLDivElement | null = null;

async function renderBrowser(onSelect = vi.fn()): Promise<ReturnType<typeof vi.fn>> {
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
          onSelect={onSelect}
          onToggleFavorite={vi.fn()}
          hfTrendingModels={HUB_MODELS}
          mode="store-download"
        />
      </ThemeProvider>,
    );
  });
  return onSelect;
}

function familyChips(): HTMLButtonElement[] {
  return Array.from(
    container?.querySelectorAll<HTMLButtonElement>(
      '[role="group"][aria-label="Filter supported models by family"] button',
    ) ?? [],
  );
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

    expect(familyChips().map((chip) => chip.textContent)).toEqual([
      'All',
      'Canary',
      'LongCat AudioDiT',
      'Qwen',
    ]);
  });

  it('filters catalog families and switches separately to Hugging Face discovery', async () => {
    await renderBrowser();

    const canaryChip = familyChips().find((chip) => chip.textContent === 'Canary');
    expect(canaryChip).not.toBeUndefined();
    await act(async () => canaryChip?.click());

    expect(container?.textContent).toContain('Canary 1B');
    expect(container?.textContent).not.toContain('Qwen3 4B');

    const hubButton = Array.from(container?.querySelectorAll('button') ?? [])
      .find((button) => button.textContent === 'Search Hugging Face');
    expect(hubButton).not.toBeUndefined();
    await act(async () => hubButton?.click());

    expect(familyChips().length).toBe(0);
    expect(container?.querySelector<HTMLInputElement>('input')?.placeholder)
      .toBe('Search all of Hugging Face...');
    expect(container?.textContent).toContain('hub-model');
    // Quantization and artifact format derived from the repo name are
    // surfaced on the row.
    expect(container?.textContent).toContain('8bit');
    expect(container?.textContent).toContain('GGUF');
    expect(container?.textContent).not.toContain('Canary 1B');
  });

  it('shows readable card titles and starts downloads only from the explicit button', async () => {
    const onSelect = await renderBrowser();

    // Titles come from the card's human-readable base model, not the repo tail.
    expect(container?.textContent).toContain('Qwen3 4B');
    // A row that represents exactly one artifact surfaces its quantization
    // and artifact format (the fixture id is an mlx-community repo).
    expect(container?.textContent).toContain('4bit');
    expect(container?.textContent).toContain('MLX');

    // Clicking the row body of a single-variant group must NOT start a download.
    const rowTitle = Array.from(container?.querySelectorAll('span') ?? [])
      .find((el) => el.textContent === 'Qwen3 4B');
    expect(rowTitle).not.toBeUndefined();
    await act(async () => rowTitle?.click());
    expect(onSelect).not.toHaveBeenCalled();

    // The explicit Download button starts it.
    const downloadButton = container?.querySelector<HTMLButtonElement>(
      'button[aria-label="Download mlx-community/Qwen3-4B-4bit"]',
    );
    expect(downloadButton).not.toBeNull();
    await act(async () => downloadButton?.click());
    expect(onSelect).toHaveBeenCalledWith('mlx-community/Qwen3-4B-4bit');
  });
});
