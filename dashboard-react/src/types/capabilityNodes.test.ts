import { describe, expect, it } from 'vitest';
import {
  CAPABILITY_NODE_STALE_AFTER_MS,
  capabilityNodeHealth,
  capabilityNodeKey,
  capabilityNodeTitle,
  isCapabilityNodeVisible,
  type CapabilityNodeSummary,
} from './capabilityNodes';

const NOW = Date.parse('2026-09-10T12:00:00Z');

function summary(overrides: Partial<CapabilityNodeSummary> = {}): CapabilityNodeSummary {
  return {
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
    observedAt: new Date(NOW - 1000).toISOString(),
    ...overrides,
  };
}

describe('capability node health', () => {
  it('maps owner status onto the satellite levels', () => {
    expect(capabilityNodeHealth(summary(), NOW)).toBe('ok');
    expect(capabilityNodeHealth(summary({ status: 'starting' }), NOW)).toBe('warn');
    expect(capabilityNodeHealth(summary({ status: 'degraded' }), NOW)).toBe('warn');
    expect(capabilityNodeHealth(summary({ status: 'configuration_invalid' }), NOW)).toBe('warn');
    expect(capabilityNodeHealth(summary({ status: 'failed' }), NOW)).toBe('error');
    expect(capabilityNodeHealth(summary({ ownerAvailable: false }), NOW)).toBe('error');
    expect(capabilityNodeHealth(summary({ status: 'installed' }), NOW)).toBe('muted');
    expect(capabilityNodeHealth(summary({ status: 'disabled' }), NOW)).toBe('muted');
  });

  it('mutes a node whose host stopped reporting', () => {
    const stale = summary({
      observedAt: new Date(NOW - CAPABILITY_NODE_STALE_AFTER_MS - 1).toISOString(),
    });
    expect(capabilityNodeHealth(stale, NOW)).toBe('muted');
    expect(capabilityNodeHealth(summary({ observedAt: 'not a date' }), NOW)).toBe('ok');
  });

  it('keys and titles nodes predictably', () => {
    expect(capabilityNodeKey(summary())).toBe('foxlight.video-studio/studio');
    expect(capabilityNodeTitle(summary({ title: '  ' }))).toBe('studio');
    expect(capabilityNodeTitle(summary({ title: null }))).toBe('studio');
    expect(capabilityNodeTitle(summary())).toBe('Video Studio');
    expect(isCapabilityNodeVisible(summary({ status: 'disabled' }))).toBe(false);
    expect(isCapabilityNodeVisible(summary({ status: 'failed' }))).toBe(true);
  });
});
