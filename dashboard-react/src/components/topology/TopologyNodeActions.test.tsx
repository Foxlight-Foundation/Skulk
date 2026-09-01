import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { darkTheme } from '../../theme/theme';
import { TopologyNodeActions } from './TopologyNodeActions';

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

describe('TopologyNodeActions tooltips', () => {
  it('uses InfoTooltip for restart and diagnostics actions', async () => {
    container = document.createElement('div');
    document.body.append(container);
    root = createRoot(container);

    await act(async () => {
      root?.render(
        <ThemeProvider theme={darkTheme}>
          <TopologyNodeActions
            infoContent="Node information"
            onInspect={vi.fn()}
            onRestart={vi.fn()}
          />
        </ThemeProvider>,
      );
    });

    const tooltipTriggers = container.querySelectorAll<HTMLElement>('.topology-action-tooltip');
    const actionButtons = container.querySelectorAll<HTMLButtonElement>('button');
    expect(tooltipTriggers).toHaveLength(2);
    expect(actionButtons[0]?.hasAttribute('title')).toBe(false);
    expect(actionButtons[1]?.hasAttribute('title')).toBe(false);

    await act(async () => tooltipTriggers[0]?.focus());
    expect(document.querySelector('[role="tooltip"]')?.textContent).toContain(
      'Restart this node - releases GPU memory and rejoins the cluster',
    );

    await act(async () => tooltipTriggers[1]?.focus());
    expect(document.querySelector('[role="tooltip"]')?.textContent).toContain(
      'Inspect live node diagnostics',
    );
  });
});
