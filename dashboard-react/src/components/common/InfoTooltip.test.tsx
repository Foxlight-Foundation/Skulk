import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { afterEach, describe, expect, it } from 'vitest';

import { darkTheme } from '../../theme/theme';
import { InfoTooltip } from './InfoTooltip';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

let root: Root | null = null;
let container: HTMLDivElement | null = null;

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  root = null;
  container = null;
});

describe('InfoTooltip containment', () => {
  it('wraps long immutable identifiers within the viewport', async () => {
    container = document.createElement('div');
    document.body.append(container);
    root = createRoot(container);
    const identity = `local_${'abcdefghijklmnopqrstuvwxyz234567'.repeat(4)}`;

    await act(async () => {
      root?.render(
        <ThemeProvider theme={darkTheme}>
          <InfoTooltip content={<span>{identity}</span>} delay={0} />
        </ThemeProvider>,
      );
    });

    const trigger = container.querySelector<HTMLElement>('[tabindex="0"]');
    expect(trigger).not.toBeNull();
    await act(async () => trigger?.focus());

    const tooltip = document.querySelector<HTMLElement>('[role="tooltip"]');
    expect(tooltip).not.toBeNull();
    const style = getComputedStyle(tooltip!);
    const content = tooltip!.querySelector<HTMLElement>('div');
    expect(content).not.toBeNull();
    const contentStyle = getComputedStyle(content!);
    expect(style.boxSizing).toBe('border-box');
    expect(style.overflowWrap).toBe('anywhere');
    expect(contentStyle.overflowX).toBe('hidden');
    expect(contentStyle.overflowY).toBe('auto');
    expect(tooltip!.getBoundingClientRect().right).toBeLessThanOrEqual(
      window.innerWidth,
    );
    expect(content!.scrollWidth).toBeLessThanOrEqual(content!.clientWidth);
  });
});
