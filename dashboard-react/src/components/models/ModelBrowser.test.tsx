import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { darkTheme } from '../../theme/theme';
import type { HuggingFaceModel, ModelInfo, PickerMode, InstanceStatus } from '../../types/models';
import { ModelBrowser } from './ModelBrowser';
import type { BurstInfo } from './burst';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });

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
    catalog_source: 'registry',
    registry_provenance: 'foxlight',
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

async function renderBrowser(
  onSelect = vi.fn(),
  getBurstInfo?: (variantId: string) => BurstInfo | null,
  mode: PickerMode = 'store-download',
  models: ModelInfo[] = MODELS,
  getModelFitStatus: () => 'fits_now' | 'fits_cluster_capacity' = () => 'fits_now',
  instanceStatuses?: Record<string, InstanceStatus>,
  canModelFit = () => true,
  recentModelIds?: string[],
): Promise<ReturnType<typeof vi.fn>> {
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  await act(async () => {
    root?.render(
      <ThemeProvider theme={darkTheme}>
        <ModelBrowser
          models={models}
          selectedModelId={null}
          favorites={new Set()}
          canModelFit={canModelFit}
          instanceStatuses={instanceStatuses}
          recentModelIds={recentModelIds}
          getModelFitStatus={getModelFitStatus}
          onSelect={onSelect}
          onToggleFavorite={vi.fn()}
          hfTrendingModels={HUB_MODELS}
          mode={mode}
          getBurstInfo={getBurstInfo}
          fleetMemoryBytes={64 * 2 ** 30}
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
    expect(container?.textContent).toContain('Supported');
    expect(container?.textContent).not.toContain('Recommended');

    const sourceButtons = container?.querySelectorAll('[role="group"][aria-label="Model source"] button');
    expect(sourceButtons?.length).toBe(2);
    expect(sourceButtons?.[0]?.textContent).toBe('Supported');
    expect(sourceButtons?.[1]?.textContent).toBe('Hugging Face');

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
      .find((button) => button.textContent === 'Hugging Face');
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
    expect(container?.textContent).toContain('Foxlight');

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

  it('describes current fit without claiming a recommendation', async () => {
    await renderBrowser(vi.fn(), undefined, 'launch');

    expect(container?.textContent).toContain('Fits this cluster');
    expect(container?.textContent).not.toContain('Recommended');
  });

  it('includes models that fit total cluster capacity under the fit heading', async () => {
    await renderBrowser(
      vi.fn(),
      undefined,
      'launch',
      MODELS,
      () => 'fits_cluster_capacity',
    );

    expect(container?.textContent).toContain('Fits this cluster');
    expect(container?.textContent).not.toContain('Other');
  });

  it('offers grouped downloads without silently choosing a variant', async () => {
    const onSelect = vi.fn();
    await renderBrowser(onSelect, undefined, 'store-download', [
      MODELS[0],
      { ...MODELS[0], id: 'local/Qwen3-4B-8bit', quantization: '8bit' },
    ]);
    const download = Array.from(container?.querySelectorAll('button') ?? [])
      .find(button => button.textContent === 'Download');
    expect(download).toBeDefined();
    await act(async () => download?.click());
    expect(container?.textContent).toContain('local/Qwen3-4B-8bit'.split('/').pop());
    expect(onSelect).not.toHaveBeenCalled();
    expect(container?.querySelector('[aria-label="Download local/Qwen3-4B-8bit"]')).not.toBeNull();
  });

  it('does not apply registry provenance to an unprovenanced grouped variant', async () => {
    const mixedModels = [
      MODELS[0],
      {
        ...MODELS[0],
        id: 'local/Qwen3-4B-8bit',
        name: 'Qwen3-4B-8bit',
        quantization: '8bit',
        catalog_source: 'bundled' as const,
        registry_provenance: null,
      },
    ];
    await renderBrowser(vi.fn(), undefined, 'launch', mixedModels);

    expect(container?.textContent).not.toContain('Foxlight');
    const rowTitle = Array.from(container?.querySelectorAll('span') ?? [])
      .find((element) => element.textContent === 'Qwen3 4B');
    await act(async () => rowTitle?.click());
    expect(container?.textContent).toContain('Foxlight');
  });

  it('partitions burst models after placeable ones and keeps them interactive', async () => {
    // Qwen3 4B exceeds the (fictional) fleet; the others are placeable.
    const onSelect = await renderBrowser(vi.fn(), (variantId) =>
      variantId.includes('Qwen3-4B')
        ? { reason: 'size', neededBytes: 128 * 2 ** 30 }
        : null);

    expect(container?.textContent).toContain('Needs burst capacity');
    expect(container?.querySelector('[aria-label="Burst"]')).not.toBeNull();

    // The burst section renders after the placeable card.
    const text = container?.textContent ?? '';
    expect(text.indexOf('Canary 1B')).toBeLessThan(text.indexOf('Qwen3 4B'));

    // Burst rows still download from their explicit button.
    const downloadButton = container?.querySelector<HTMLButtonElement>(
      'button[aria-label="Download mlx-community/Qwen3-4B-4bit"]',
    );
    expect(downloadButton).not.toBeNull();
    await act(async () => downloadButton?.click());
    expect(onSelect).toHaveBeenCalledWith('mlx-community/Qwen3-4B-4bit');
  });
});


describe('discovery evidence and download independence', () => {
  it('applies readiness and search filters to recent models', async () => {
    await renderBrowser(undefined, undefined, 'store-download', MODELS, undefined, {
      [MODELS[0].id]: { status: 'Ready', statusClass: 'ready' },
      [MODELS[1].id]: { status: 'Loading', statusClass: 'loading' },
    }, undefined, MODELS.map(model => model.id));
    const recent = familyChips().find(chip => chip.textContent === 'Recent')!;
    await act(async () => recent.click());
    expect(container!.textContent).toContain('3 model groups');

    const ready = [...container!.querySelectorAll('label')].find(label => label.textContent === 'Ready now')!.querySelector('input')!;
    await act(async () => ready.click());
    expect(container!.textContent).toContain('1 model group');
    expect(container!.textContent).toContain('Qwen3 4B');
    expect(container!.textContent).not.toContain('LongCat AudioDiT 1B');

    const search = container!.querySelector<HTMLInputElement>('input[aria-label="Search models"]')!;
    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!.call(search, 'Canary');
      search.dispatchEvent(new Event('input', { bubbles: true }));
    });
    expect(container!.textContent).toContain('0 model groups');
    await act(async () => ready.click());
    expect(container!.textContent).toContain('Canary 1B');
    expect(container!.textContent).not.toContain('Qwen3 4B');
  });

  it('shows no ready models when readiness evidence is unavailable', async () => {
    await renderBrowser();
    const ready = [...container!.querySelectorAll('label')].find(label => label.textContent === 'Ready now')!.querySelector('input')!;
    await act(async () => ready.click());
    expect(container!.textContent).toContain('0 model groups');
    expect(container!.textContent).not.toContain('Qwen3 4B');
  });

  it('filters readiness independently of catalog and storage status', async () => {
    await renderBrowser(undefined, undefined, 'store-download', MODELS, undefined, {
      [MODELS[0].id]: { status: 'Ready', statusClass: 'ready' },
      [MODELS[1].id]: { status: 'Loading', statusClass: 'loading' },
    });
    const ready = [...container!.querySelectorAll('label')].find(label => label.textContent === 'Ready now')!.querySelector('input')!;
    await act(async () => ready.click());
    expect(container!.textContent).toContain('Qwen3 4B');
    expect(container!.textContent).not.toContain('LongCat AudioDiT 1B');
    expect(container!.textContent).not.toContain('Canary 1B');
  });

  it('allows a store download when no current placement capacity exists', async () => {
    const select = vi.fn();
    await renderBrowser(select, undefined, 'store-download', MODELS, undefined, undefined, () => false);
    const download = container!.querySelector<HTMLButtonElement>(`button[aria-label="Download ${MODELS[0].id}"]`)!;
    expect(download.disabled).toBe(false);
    await act(async () => download.click());
    expect(select).toHaveBeenCalledWith(MODELS[0].id);
  });
});
