import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { darkTheme } from '../../theme/theme';
import { RunningInstanceCard, type RunningInstanceCardProps } from './RunningInstanceCard';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({
    t: (_key: string, fallback: string) => fallback,
  }),
}));

let root: Root | null = null;
let container: HTMLDivElement | null = null;

async function renderCard(props: RunningInstanceCardProps): Promise<void> {
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  await act(async () => {
    root?.render(
      <ThemeProvider theme={darkTheme}>
        <RunningInstanceCard {...props} />
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

const readyCard: RunningInstanceCardProps = {
  instanceId: 'instance-1',
  modelId: 'org/model',
  sharding: 'Pipeline',
  instanceType: 'MlxRing',
  engine: 'mlx',
  nodeStatuses: [{ name: 'node', state: 'ready' }],
  status: 'ready',
  onChat: vi.fn(),
};

describe('RunningInstanceCard chat action', () => {
  it('shows chat for a ready text-generation model', async () => {
    await renderCard({ ...readyCard, supportsTextChat: true });

    expect([...container!.querySelectorAll('button')].some((button) => button.textContent?.includes('Chat'))).toBe(true);
  });

  it('hides chat for a ready speech-only model', async () => {
    await renderCard({ ...readyCard, modelId: 'org/tts', supportsTextChat: false });

    expect([...container!.querySelectorAll('button')].some((button) => button.textContent?.includes('Chat'))).toBe(false);
  });
});
