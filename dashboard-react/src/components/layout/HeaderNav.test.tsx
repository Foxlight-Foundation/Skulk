import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { darkTheme } from '../../theme/theme';
import { HeaderNav } from './HeaderNav';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });

interface HeaderState {
  ui: { theme: string; observabilityPanelOpen: boolean };
}

vi.mock('../../store/hooks', () => ({
  useAppDispatch: () => vi.fn(),
  useAppSelector: (selector: (state: HeaderState) => unknown) =>
    selector({ ui: { theme: 'dark', observabilityPanelOpen: false } }),
}));

vi.mock('../../store/endpoints/steward', () => ({
  useGetStewardStatusQuery: () => ({ data: undefined }),
}));

vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({
    t: (_key: string, fallback: string) => fallback,
  }),
}));

let root: Root | null = null;
let container: HTMLDivElement | null = null;

afterEach(() => {
  act(() => root?.unmount());
  container?.remove();
  root = null;
  container = null;
});

describe('HeaderNav brand', () => {
  it('shows the trademark notice beside the wordmark and the version after a gap', async () => {
    container = document.createElement('div');
    document.body.append(container);
    root = createRoot(container);
    await act(async () => {
      root?.render(
        <ThemeProvider theme={darkTheme}>
          <HeaderNav compact={false} />
        </ThemeProvider>,
      );
    });

    const trademark = Array.from(container.querySelectorAll('span')).find(
      (element) => element.textContent === '™',
    );
    expect(trademark).toBeDefined();
    const wordmark = trademark?.parentElement;
    expect(wordmark?.firstChild?.textContent).toBe('Skulk');
    const version = trademark?.nextElementSibling;
    expect(version?.textContent).toBe(__APP_VERSION__);

    // The notice touches the wordmark; the version sits clearly apart from it.
    const trademarkBox = trademark?.getBoundingClientRect();
    const versionBox = version?.getBoundingClientRect();
    expect(trademarkBox && versionBox && versionBox.left - trademarkBox.right).toBeGreaterThanOrEqual(4);
  });
});
