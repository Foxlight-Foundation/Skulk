import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { darkTheme } from '../../theme/theme';
import { RightDrawer } from './RightDrawer';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({
    t: (_key: string, fallback: string) => fallback,
  }),
}));

let root: Root | null = null;
let container: HTMLDivElement | null = null;

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  root = null;
  container = null;
});

async function render(open: boolean) {
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  const onClose = vi.fn();
  await act(async () => {
    root?.render(
      <ThemeProvider theme={darkTheme}>
        <RightDrawer
          ariaLabel="Test drawer"
          closeLabel="Close test drawer"
          id="test-drawer"
          maxWidth={900}
          minWidth={300}
          onClose={onClose}
          onWidthChange={() => undefined}
          open={open}
          resizeLabel="Resize test drawer"
          title="Drawer"
          width={480}
        >
          <p>drawer body</p>
        </RightDrawer>
      </ThemeProvider>,
    );
  });
  return { onClose };
}

describe('RightDrawer', () => {
  it('renders nothing while closed', async () => {
    await render(false);
    expect(container?.querySelector('#test-drawer')).toBeNull();
  });

  it('closes from the backdrop, the close button, and Escape', async () => {
    const { onClose } = await render(true);
    expect(container?.querySelector('#test-drawer')?.textContent).toContain('drawer body');
    await act(async () => {
      container?.querySelector<HTMLElement>('[data-testid="right-drawer-backdrop"]')?.click();
    });
    await act(async () => {
      container?.querySelector<HTMLButtonElement>('button[aria-label="Close test drawer"]')?.click();
    });
    await act(async () => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    });
    expect(onClose).toHaveBeenCalledTimes(3);
  });
});
