import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { NodeInfo } from '../../types/topology';
import { darkTheme } from '../../theme/theme';
import { ClusterNode } from './ClusterNode';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({
    t: (_key: string, fallback: string) => fallback,
  }),
}));

const nodeInfo: NodeInfo = {
  friendly_name: 'kite3',
  last_mactop_update: Date.now(),
  mactop_info: {
    gpu_usage: [0, 0.25],
    memory: { ram_total: 24, ram_usage: 12 },
    sys_power: 10,
    temp: { gpu_temp_avg: 35 },
  },
  system_info: { chip: 'Apple M4', model_id: 'Mac mini' },
};

let root: Root | null = null;
let container: HTMLDivElement | null = null;

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  root = null;
  container = null;
});

describe('ClusterNode interaction surface', () => {
  it('covers the node, metadata, whitespace, and action rail as one hover target', async () => {
    const onInteractionChange = vi.fn();
    container = document.createElement('div');
    document.body.append(container);
    root = createRoot(container);

    await act(async () => {
      root?.render(
        <ThemeProvider theme={darkTheme}>
          <svg>
            <ClusterNode
              nodeId="kite3"
              nodeInfo={nodeInfo}
              onInteractionChange={onInteractionChange}
              x={0}
              y={0}
            />
          </svg>
        </ThemeProvider>,
      );
    });

    const interactionSurface = container.querySelector('[data-node-interaction-surface="true"]');
    expect(interactionSurface).not.toBeNull();
    expect(interactionSurface).toHaveAttribute('x', '-112');
    expect(interactionSurface).toHaveAttribute('y', '-52');
    expect(interactionSurface).toHaveAttribute('width', '224');
    expect(interactionSurface).toHaveAttribute('height', '206');
    expect(interactionSurface).toHaveAttribute('pointer-events', 'all');

    const nodeText = container.querySelectorAll('.topology-node > text');
    expect(nodeText).toHaveLength(4);
    for (const text of nodeText) {
      expect(text).toHaveAttribute('font-family', darkTheme.fonts.body);
      expect(text).not.toHaveAttribute('font-family', darkTheme.fonts.mono);
    }

    await act(async () => {
      interactionSurface?.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
    });
    expect(onInteractionChange).toHaveBeenLastCalledWith(true);
    expect(container.querySelector('[role="toolbar"]')).not.toBeNull();

    await act(async () => {
      interactionSurface?.dispatchEvent(new MouseEvent('mouseout', { bubbles: true }));
    });
    expect(onInteractionChange).toHaveBeenLastCalledWith(false);
  });
});
