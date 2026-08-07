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
