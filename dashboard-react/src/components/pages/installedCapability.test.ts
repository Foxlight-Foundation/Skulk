import { describe, expect, it } from 'vitest';
import type { CapabilityNodeSummary } from '../../types/capabilityNodes';
import { installedCardAction, type HostedCapabilityNode } from './installedCapability';

const screen = { surfaceId: 'studio', title: 'Example Studio', kind: 'link' as const, url: 'https://studio.example/s/token/', ready: true };
function node(overrides: Partial<CapabilityNodeSummary> = {}, hostNodeId = 'host-node'): HostedCapabilityNode {
  return {
    hostNodeId,
    summary: {
      pluginId: 'managed.' + '2'.repeat(32), nodeId: 'studio', bundleId: 'example.studio', version: '0.1.0', title: 'Example Studio', status: 'ready',
      ownerAvailable: true, surfaces: [screen], actions: [], operationsActive: 0, observedAt: '2026-10-08T12:00:00Z', ...overrides,
    },
  };
}
const action = (hosted: HostedCapabilityNode[]) => installedCardAction(hosted, 'host-node', 'dashboard.example');

describe('installedCardAction', () => {
  it('opens a running plugin\'s screen', () => {
    expect(action([node()])).toEqual({ kind: 'open', surface: screen });
  });

  it('sets up a plugin a node of which is off, needs settings, stopped or not answering', () => {
    expect(action([node({ status: 'disabled', surfaces: [] })])).toEqual({ kind: 'set-up' });
    expect(action([node({ status: 'configuration_invalid', surfaces: [] })])).toEqual({ kind: 'set-up' });
    expect(action([node({ status: 'failed', surfaces: [] })])).toEqual({ kind: 'set-up' });
    expect(action([node({ ownerAvailable: false })])).toEqual({ kind: 'set-up' });
    // Another node that needs the owner wins over one that runs.
    expect(action([node(), node({ nodeId: 'worker', status: 'disabled', surfaces: [] })])).toEqual({ kind: 'set-up' });
  });

  it('manages a plugin that is starting, not reported yet, or has no screen this browser can open', () => {
    expect(action([node({ status: 'starting', surfaces: [] })])).toEqual({ kind: 'manage' });
    expect(action([])).toEqual({ kind: 'manage' });
    expect(action([node({ surfaces: [] })])).toEqual({ kind: 'manage' });
    // A loopback screen on another host opens only from a browser on that host.
    expect(action([node({ surfaces: [{ ...screen, url: 'http://127.0.0.1:54905/s/token/' }] }, 'other-node')])).toEqual({ kind: 'manage' });
  });
});
