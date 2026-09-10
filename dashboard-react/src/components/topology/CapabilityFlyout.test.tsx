import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { ThemeProvider } from 'styled-components';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { CapabilityNodeSummary } from '../../types/capabilityNodes';
import { darkTheme } from '../../theme/theme';
import { CapabilityFlyout } from './CapabilityFlyout';

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

vi.mock('../../i18n/tolgee', () => ({
  useSkulkTranslation: () => ({
    t: (_key: string, fallback: string, params?: Record<string, string | number>) =>
      fallback.replace(/\{(\w+)\}/g, (_match, name: string) => String(params?.[name] ?? '')),
  }),
}));

const studio: CapabilityNodeSummary = {
  pluginId: 'foxlight.video-studio',
  nodeId: 'studio',
  bundleId: 'foxlight.video-studio',
  version: '1.0.0',
  title: 'Video Studio',
  status: 'ready',
  ownerAvailable: true,
  surfaces: [{ surfaceId: 'studio', title: 'Open Studio', kind: 'link', url: 'http://127.0.0.1:8188/', ready: true }],
  actions: [],
  operationsActive: 0,
  observedAt: new Date().toISOString(),
};

let root: Root | null = null;
let container: HTMLDivElement | null = null;

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  root = null;
  container = null;
});

async function render(props: Partial<Parameters<typeof CapabilityFlyout>[0]> = {}) {
  container = document.createElement('div');
  document.body.append(container);
  root = createRoot(container);
  const onClose = vi.fn();
  const onOpenPanel = vi.fn();
  await act(async () => {
    root?.render(
      <ThemeProvider theme={darkTheme}>
        <CapabilityFlyout
          anchor={{ x: 200, y: 120 }}
          canvasHeight={600}
          canvasWidth={800}
          hostName="kite6"
          hostNodeId="node-host"
          localNodeId="node-host"
          onClose={onClose}
          onOpenPanel={onOpenPanel}
          onSelectKey={() => undefined}
          selectedKey="foxlight.video-studio/studio"
          summaries={[studio]}
          {...props}
        />
      </ThemeProvider>,
    );
  });
  return { onClose, onOpenPanel };
}

describe('CapabilityFlyout', () => {
  it('opens link surfaces in a new tab without a referrer or opener', async () => {
    await render();
    const link = container?.querySelector<HTMLAnchorElement>('a[href="http://127.0.0.1:8188/"]');
    expect(link).not.toBeNull();
    expect(link?.target).toBe('_blank');
    expect(link?.rel).toBe('noopener noreferrer');
    expect(link?.textContent).toContain('Open Studio');
  });

  it('closes on Escape and opens the panel from the details entry', async () => {
    const { onClose, onOpenPanel } = await render();
    const details = [...(container?.querySelectorAll('button') ?? [])].find((button) =>
      button.textContent?.includes('Details'),
    );
    await act(async () => details?.click());
    expect(onOpenPanel).toHaveBeenCalledWith('node-host', 'foxlight.video-studio/studio');
    await act(async () => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    });
    expect(onClose).toHaveBeenCalled();
  });

  it('closes on an outside press but leaves satellite presses to the graph toggle', async () => {
    const { onClose } = await render();
    const satellite = document.createElement('div');
    satellite.className = 'topology-capability-satellite';
    const inner = document.createElement('button');
    satellite.append(inner);
    document.body.append(satellite);
    try {
      await act(async () => {
        inner.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
      });
      expect(onClose).not.toHaveBeenCalled();
      await act(async () => {
        document.body.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
      });
      expect(onClose).toHaveBeenCalledTimes(1);
    } finally {
      satellite.remove();
    }
  });

  it('disables a loopback surface when the dashboard is served by another host', async () => {
    await render({ localNodeId: 'node-elsewhere' });
    expect(container?.querySelector('a[href="http://127.0.0.1:8188/"]')).toBeNull();
    const disabled = [...(container?.querySelectorAll('button:disabled') ?? [])].find((button) =>
      button.textContent?.includes('Open Studio'),
    );
    expect(disabled?.title).toContain('Reachable only from a browser on kite6');
  });

  it('explains remote management when the dashboard is on another host', async () => {
    await render({ localNodeId: 'node-elsewhere' });
    expect(container?.textContent).toContain('managed from the dashboard on kite6');
  });
});
