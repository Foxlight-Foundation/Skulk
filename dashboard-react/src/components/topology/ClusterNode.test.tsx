import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { NodeInfo } from '../../types/topology';
import { darkTheme } from '../../theme/theme';
import { ClusterNode } from './ClusterNode';
import type { CapabilityNodeSummary } from '../../types/capabilityNodes';

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
  it('bridges metadata and actions without extending beyond the node selector width', async () => {
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
    expect(interactionSurface).toHaveAttribute('x', '-46');
    expect(interactionSurface).toHaveAttribute('y', '-52');
    expect(interactionSurface).toHaveAttribute('width', '92');
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

  it('places the hardware badge symmetrically on the requested side', async () => {
    container = document.createElement('div');
    document.body.append(container);
    root = createRoot(container);

    await act(async () => {
      root?.render(
        <ThemeProvider theme={darkTheme}>
          <svg>
            <ClusterNode
              hardwareBadgeSide="right"
              nodeId="kite3"
              nodeInfo={nodeInfo}
              x={0}
              y={0}
            />
          </svg>
        </ThemeProvider>,
      );
    });

    const badge = container.querySelector('[data-hardware-badge-side="right"]');
    expect(badge).toHaveAttribute('transform', 'translate(56, -17)');
  });
});

const studioNode: CapabilityNodeSummary = {
  pluginId: 'foxlight.video-studio',
  nodeId: 'studio',
  bundleId: 'foxlight.video-studio',
  version: '1.0.0',
  title: 'Video Studio',
  status: 'ready',
  ownerAvailable: true,
  surfaces: [],
  actions: [],
  operationsActive: 0,
  observedAt: new Date().toISOString(),
};

describe('ClusterNode capability satellites', () => {
  it('draws a satellite per capability node and reports its canvas anchor', async () => {
    const onSatelliteSelect = vi.fn();
    container = document.createElement('div');
    document.body.append(container);
    root = createRoot(container);

    await act(async () => {
      root?.render(
        <ThemeProvider theme={darkTheme}>
          <svg>
            <ClusterNode
              nodeId="kite6"
              nodeInfo={nodeInfo}
              onSatelliteSelect={onSatelliteSelect}
              satellites={[studioNode, { ...studioNode, nodeId: 'comfy', title: 'ComfyUI', status: 'failed' }]}
              scale={0.5}
              x={100}
              y={200}
            />
          </svg>
        </ThemeProvider>,
      );
    });

    const satellites = container.querySelectorAll('.topology-capability-satellite');
    expect(satellites).toHaveLength(2);
    expect(satellites[1]).toHaveAttribute('data-capability-health', 'error');
    const button = container.querySelector<HTMLButtonElement>(
      '[data-capability-key="foxlight.video-studio/studio"] button',
    );
    expect(button?.title).toBe('Video Studio');
    await act(async () => button?.click());
    expect(onSatelliteSelect).toHaveBeenCalledTimes(1);
    const [key, anchor] = onSatelliteSelect.mock.calls[0] as [string, { x: number; y: number }];
    expect(key).toBe('foxlight.video-studio/studio');
    // The anchor is the satellite center scaled into canvas space: above the
    // node (y < 200) and within half the orbit radius of its center.
    expect(anchor.y).toBeLessThan(200);
    expect(Math.hypot(anchor.x - 100, anchor.y - 200)).toBeCloseTo(36);
  });
});
