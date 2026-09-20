import { act } from 'react';
import { userEvent } from 'vitest/browser';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { darkTheme } from '../../theme/theme';
import { RightDrawer } from './RightDrawer';

Object.defineProperty(globalThis, 'IS_REACT_ACT_ENVIRONMENT', { value: true, configurable: true });

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

async function render(open: boolean, title: string = 'Drawer') {
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
          title={title}
          width={480}
        >
          <p>drawer body</p><button>First action</button><button>Last action</button>
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

describe('RightDrawer header', () => {
  it('keeps the close button inside the drawer under a long unbroken title', async () => {
    await render(true, 'x'.repeat(200));
    const aside = container?.querySelector<HTMLElement>('#test-drawer');
    const close = container?.querySelector<HTMLButtonElement>('button[aria-label="Close test drawer"]');
    expect(aside).not.toBeNull();
    expect(close).not.toBeNull();
    const asideRect = (aside as HTMLElement).getBoundingClientRect();
    const closeRect = (close as HTMLElement).getBoundingClientRect();
    expect(closeRect.right).toBeLessThanOrEqual(asideRect.right + 1);
    expect(closeRect.width).toBeGreaterThan(0);
  });
});

it('contains focus and restores the original opener when another drawer replaces it', async () => {
  const opener = document.createElement('button'); opener.textContent = 'Open'; document.body.append(opener); opener.focus();
  container = document.createElement('div'); document.body.append(container); root = createRoot(container);
  const closeFirst = vi.fn();
  const drawer = (id: string, close: () => void) => <RightDrawer key={id} open onClose={close} title={id} ariaLabel={id} width={440} minWidth={320} maxWidth={720} onWidthChange={() => {}} closeLabel={`Close ${id}`} resizeLabel="Resize"><button>Inside {id}</button></RightDrawer>;
  await act(async () => root?.render(<ThemeProvider theme={darkTheme}>{drawer('first', closeFirst)}</ThemeProvider>));
  expect(document.activeElement?.getAttribute('aria-label')).toBe('first');
  opener.focus();
  expect(document.activeElement?.getAttribute('aria-label')).toBe('first');
  await act(async () => root?.render(<ThemeProvider theme={darkTheme}>{drawer('first', closeFirst)}{drawer('second', () => {})}</ThemeProvider>));
  expect(closeFirst).toHaveBeenCalledOnce();
  expect(document.activeElement?.getAttribute('aria-label')).toBe('second');
  await act(async () => root?.render(<ThemeProvider theme={darkTheme}>{drawer('second', () => {})}</ThemeProvider>));
  await act(async () => root?.unmount()); root = null;
  expect(document.activeElement).toBe(opener);
  expect(document.body.style.overflow).not.toBe('hidden');
  opener.remove();
});

it('loops keyboard focus within the drawer', async () => {
  await render(true);
  const controls = [...container!.querySelectorAll<HTMLElement>('button, [role="separator"]')].filter(element => element.getClientRects().length > 0);
  await userEvent.tab();
  expect(document.activeElement).toBe(controls[0]);
  for (const control of controls.slice(1)) {
    await userEvent.tab();
    expect(document.activeElement).toBe(control);
  }
  await userEvent.tab();
  expect(document.activeElement).toBe(controls[0]);
  await userEvent.tab({ shift: true });
  expect(document.activeElement).toBe(controls.at(-1));
});
