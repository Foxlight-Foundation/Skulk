import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { darkTheme } from '../../theme/theme';
import { StoreRegistryTable, type ModelCardInfo } from './StoreRegistryTable';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({
    t: (_key: string, fallback: string) => fallback,
  }),
}));

let root: Root | null = null;
let container: HTMLDivElement | null = null;

async function renderRegistry(modelId: string, card: ModelCardInfo): Promise<void> {
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  await act(async () => {
    root?.render(
      <ThemeProvider theme={darkTheme}>
        <StoreRegistryTable
          entries={[{
            model_id: modelId,
            total_bytes: 1024,
            files: ['model.safetensors'],
            downloaded_at: new Date().toISOString(),
          }]}
          activeModelIds={[modelId]}
          modelCards={{ [modelId]: card }}
          clusterCards={{
            [modelId]: {
              modelId,
              sharding: 'Pipeline',
              instanceType: 'MlxRing',
              nodes: [],
              isReady: true,
            },
          }}
          onRefresh={vi.fn()}
          onDelete={vi.fn()}
          onChat={vi.fn()}
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

describe('StoreRegistryTable chat action', () => {
  it('shows chat for a ready text-generation model', async () => {
    await renderRegistry('org/chat', { tasks: ['TextGeneration'], capabilities: ['text'] });

    expect(container?.querySelector('button[title="Chat with model"]')).not.toBeNull();
  });

  it('hides chat for a ready TTS model', async () => {
    await renderRegistry('org/tts', { tasks: ['TextToSpeech'], capabilities: ['tts'], tags: ['tts'] });

    expect(container?.querySelector('button[title="Chat with model"]')).toBeNull();
  });
});

describe('StoreRegistryTable failed downloads', () => {
  const GATED_REASON = "Access to 'org/gated' is restricted and this node sent no Hugging Face token.";

  async function renderWithDownload(status: string, error?: string): Promise<void> {
    container = document.createElement('div');
    document.body.append(container);
    root = createRoot(container);
    await act(async () => {
      root?.render(
        <ThemeProvider theme={darkTheme}>
          <StoreRegistryTable
            entries={[{
              model_id: 'org/gated',
              total_bytes: 1024,
              files: ['model.safetensors'],
              downloaded_at: new Date().toISOString(),
            }]}
            activeDownloads={[{ modelId: 'org/gated', progress: 0.2, status, error }]}
            activeModelIds={[]}
            onRefresh={vi.fn()}
            onDelete={vi.fn()}
            onLaunch={vi.fn()}
          />
        </ThemeProvider>,
      );
    });
  }

  it('badges a failed registered download instead of showing a stuck progress bar', async () => {
    await renderWithDownload('failed', GATED_REASON);

    expect(container?.textContent).toContain('Download failed');
    expect(container?.textContent).not.toContain('20%');
    // A failed download must not lock the row: the operator fixes the cause
    // and retries, so Launch stays available for the installed generation.
    expect(container?.querySelector('button[title="Launch model"]')).not.toBeNull();
  });

  it('keeps the progress bar and suppresses launch while a download is live', async () => {
    await renderWithDownload('downloading');

    expect(container?.textContent).toContain('20%');
    expect(container?.textContent).not.toContain('Download failed');
    expect(container?.querySelector('button[title="Launch model"]')).toBeNull();
  });

  it('badges a failed unregistered (pending row) download', async () => {
    container = document.createElement('div');
    document.body.append(container);
    root = createRoot(container);
    await act(async () => {
      root?.render(
        <ThemeProvider theme={darkTheme}>
          <StoreRegistryTable
            entries={[]}
            activeDownloads={[{ modelId: 'org/unregistered', progress: 0, status: 'failed', error: GATED_REASON }]}
            activeModelIds={[]}
            onRefresh={vi.fn()}
            onDelete={vi.fn()}
          />
        </ThemeProvider>,
      );
    });

    expect(container?.textContent).toContain('org/unregistered');
    expect(container?.textContent).toContain('Download failed');
  });
});
