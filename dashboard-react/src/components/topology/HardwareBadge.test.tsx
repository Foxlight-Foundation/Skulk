import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { afterEach, describe, expect, it } from 'vitest';
import { darkTheme, lightTheme } from '../../theme/theme';
import type { DeviceModel } from '../../types/topology';
import { HardwareBadge } from './HardwareBadge';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

let root: Root | null = null;
let container: HTMLDivElement | null = null;

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  root = null;
  container = null;
});

async function renderBadge(model: DeviceModel, light = false): Promise<SVGGElement> {
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  await act(async () => {
    root?.render(
      <ThemeProvider theme={light ? lightTheme : darkTheme}>
        <svg>
          <HardwareBadge model={model} />
        </svg>
      </ThemeProvider>,
    );
  });
  return container.querySelector<SVGGElement>('[data-hardware-badge]')!;
}

describe('HardwareBadge', () => {
  it.each([
    ['amd-strix', 'AMD'],
    ['nvidia-gpu', 'NVIDIA'],
  ] as const)('centers the %s wordmark in its container', async (model, mark) => {
    const badge = await renderBadge(model);
    const text = badge.querySelector('text');
    expect(text?.textContent).toBe(mark);
    expect(text?.getAttribute('x')).toBe('24');
    expect(text?.getAttribute('y')).toBe('17');
    expect(text?.getAttribute('text-anchor')).toBe('middle');
    expect(text?.getAttribute('dominant-baseline')).toBe('middle');
  });

  it.each(['macbook-pro', 'mac-studio', 'mac-mini'] as const)(
    'uses only an Apple mark for %s',
    async (model) => {
      const badge = await renderBadge(model, true);
      expect(badge.querySelector('path')).not.toBeNull();
      expect(badge.querySelector('text')).toBeNull();
      expect(badge.querySelectorAll('rect')).toHaveLength(1);
    },
  );
});
