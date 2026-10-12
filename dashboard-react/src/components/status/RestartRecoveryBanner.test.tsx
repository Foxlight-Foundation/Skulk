import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { darkTheme } from '../../theme/theme';
import { RestartRecoveryBanner } from './RestartRecoveryBanner';

vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({
    t: (_key: string, fallback: string, params?: Record<string, unknown>) =>
      fallback.replace(/\{(\w+)\}/g, (_match, name: string) => String(params?.[name] ?? '')),
  }),
}));

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });
let root: Root;
let host: HTMLDivElement;

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  await act(async () => root.unmount());
  host.remove();
});

async function render(pending: boolean, placements: number | null) {
  await act(async () =>
    root.render(
      <ThemeProvider theme={darkTheme}>
        <RestartRecoveryBanner pending={pending} placements={placements} />
      </ThemeProvider>,
    ),
  );
  return host.querySelector('[role="status"]')?.textContent ?? null;
}

it('says how many models are coming back while a restore runs', async () => {
  expect(await render(true, 3)).toBe('Bringing back 3 models after the restart');
  expect(await render(true, 1)).toBe('Bringing back 1 model after the restart');
});

it('says it is waiting before the master has chosen what to bring back', async () => {
  expect(await render(true, null)).toBe(
    'Restarting: waiting for the machines to rejoin before bringing models back',
  );
});

it('stays hidden when nothing is coming back or no restore is running', async () => {
  expect(await render(true, 0)).toBeNull();
  expect(await render(false, 2)).toBeNull();
});
